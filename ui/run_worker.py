"""ui/run_worker.py — Background-run execution loop.

``execute_run`` is the function that actually walks a :class:`RunJob`'s
files through the pipeline. It is invoked by ``RunManager.start()``, either
on a plain ``threading.Thread`` (survives tab refresh/close — the thread is
owned by the server process, not any Streamlit script run) or, under
``CHORUS_SYNC_RUN=1``, inline on the calling thread for deterministic tests.

This module must never call ``st.*`` — it has no ScriptRunContext when run
on a background thread, and Streamlit calls from a foreign thread raise.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ui.run_state import RUNS_DIR, append_history, new_state, write_state_atomic

if TYPE_CHECKING:
    from ui.run_manager import RunManager
    from ui.run_state import RunJob

logger = logging.getLogger(__name__)

# Throttle state-file writes from the progress callback to ~2/second so a
# 197-segment file doesn't hammer disk I/O; stage transitions (the history
# ring buffer) always flush immediately regardless of the throttle.
_WRITE_INTERVAL_SECONDS = 0.5

_OUTPUT_PATH_KEYS = (
    "consensus_path",
    "ai_context_path",
    "bundle_path",
    "best_guess_path",
    "diarised_path",
)


def execute_run(job: RunJob, manager: RunManager) -> None:
    """Run every file in *job* sequentially, updating state as it goes.

    On success, the pipeline's results dict is stored in
    ``manager``'s in-memory registry (keyed by run_id, then filename). On a
    per-file exception, the error is captured onto that file's state entry
    and the loop continues to the next file — mirroring the try/except in
    ``ui/pipeline_invocation.py``'s sequential run loop. Wrapped in an outer
    try/finally so any unexpected crash still lands the run ``finished``
    with whatever per-file errors were captured, never leaving the state
    file stuck on ``running`` with a silently dead thread.
    """
    # Deferred import: preserves ``ui.pipeline_invocation.run_pipeline`` as
    # the existing test patch target (that module binds the name at import
    # time via ``from pipeline_runner import run_pipeline``).
    from ui import pipeline_invocation

    run_dir = RUNS_DIR / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s", "%H:%M:%S"
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    # Guarantee INFO records reach this handler for the run's duration even
    # if something else (test runners, host app config) has raised the root
    # logger's effective level above INFO; restored in finally.
    previous_root_level = root_logger.level
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)

    state = new_state(job.run_id, manager.boot_id, job.config, job.files, time.time())
    state["log_path"] = str(log_path)
    write_state_atomic(state)
    manager._results[job.run_id] = {}

    logger.info(
        "Starting background run %s (%d file%s)",
        job.run_id,
        len(state["files"]),
        "" if len(state["files"]) == 1 else "s",
    )

    last_write = 0.0

    def throttled_write(*, force: bool = False) -> None:
        nonlocal last_write
        now = time.monotonic()
        if force or (now - last_write) >= _WRITE_INTERVAL_SECONDS:
            last_write = now
            write_state_atomic(state)

    try:
        for file_state in state["files"]:
            spool_path = Path(file_state["spool_path"])
            file_state["status"] = "running"
            throttled_write(force=True)
            logger.info("Processing %s", file_state["name"])
            t0 = time.time()

            def _progress_cb(
                label: str, frac: float, _fs: dict[str, Any] = file_state
            ) -> None:
                _fs["label"] = label
                _fs["progress"] = frac
                throttled_write()

            def _event_cb(
                event: dict[str, Any], _fs: dict[str, Any] = file_state
            ) -> None:
                prev = (_fs.get("stage"), _fs.get("detail"))
                _fs["stage"] = event.get("stage")
                _fs["detail"] = event.get("detail")
                _fs["stage_index"] = event.get("stage_index")
                _fs["stage_total"] = event.get("stage_total")
                _fs["passes_done"] = event.get("passes_done")
                _fs["passes_total"] = event.get("passes_total")
                _fs["segment"] = event.get("segment")
                _fs["segments_total"] = event.get("segments_total")
                _fs["last_event_at"] = time.time()
                if (_fs["stage"], _fs["detail"]) != prev:
                    # Stage transition: append to history and flush immediately.
                    append_history(_fs, _fs["stage"], _fs["detail"])
                    throttled_write(force=True)
                else:
                    # Segment tick within the same stage/detail: never
                    # grows history, just the normal throttled write.
                    throttled_write()

            try:
                results = pipeline_invocation.run_pipeline(
                    audio_path=spool_path,
                    progress_callback=_progress_cb,
                    event_callback=_event_cb,
                    **job.config,
                )
                manager._results[job.run_id][file_state["name"]] = results
                file_state["status"] = "done"
                file_state["progress"] = 1.0
                file_state["output_paths"] = {
                    key: str(results[key])
                    for key in _OUTPUT_PATH_KEYS
                    if results.get(key) is not None
                }
            except (
                Exception
            ) as exc:  # noqa: BLE001 - mirrors pipeline_invocation.py run loop
                file_state["status"] = "error"
                file_state["error"] = str(exc)
                logger.exception("Pipeline error for %s", file_state["name"])
            finally:
                file_state["elapsed"] = round(time.time() - t0, 2)
                throttled_write(force=True)
                with contextlib.suppress(OSError):
                    spool_path.unlink(missing_ok=True)
    finally:
        state["status"] = "finished"
        state["finished_at"] = time.time()
        write_state_atomic(state)
        logger.info("Background run %s finished", job.run_id)
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_root_level)
        handler.close()
