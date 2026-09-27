"""ui/run_manager.py — Process-wide singleton owning background pipeline runs.

``RunManager`` enforces a single active run at a time and is the only place
in this feature that touches Streamlit (via ``get_run_manager``'s
``@st.cache_resource``). The class itself has no Streamlit dependency and
is safe to construct directly under plain pytest.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
import time
import uuid
from typing import Any

import streamlit as st

from ui.run_state import (
    ACTIVE_RUN_FILE,
    RUNS_DIR,
    RunJob,
    load_state,
    write_state_atomic,
)
from ui.run_worker import execute_run


class RunManager:
    """Owns the single background run thread and its in-memory results."""

    _process_boot_id = uuid.uuid4().hex
    _process_lock = threading.Lock()
    _active_thread: threading.Thread | None = None

    def __init__(self) -> None:
        self.boot_id = self._process_boot_id
        self._lock = self._process_lock
        self._results_lock = threading.Lock()
        self._results: dict[str, dict[str, dict]] = {}
        self.mark_interrupted_if_stale()

    def is_running(self) -> bool:
        """Return True if a run is actually executing (live-thread check).

        Checking the state file's ``status`` alone is not enough: a state
        file can say ``running`` while the process that would ever update
        it is gone (server restart) — that is exactly the "stale" case
        ``mark_interrupted_if_stale`` handles separately.
        """
        with self._results_lock:
            thread = type(self)._active_thread
            return thread is not None and thread.is_alive()

    def start(self, job: RunJob) -> bool:
        """Start executing *job*. Returns False if a run is already active."""
        with self._lock:
            if self.is_running():
                return False

            if os.environ.get("CHORUS_SYNC_RUN") == "1":
                execute_run(job, self)
            else:
                thread = threading.Thread(
                    target=execute_run,
                    args=(job, self),
                    daemon=True,
                    name=f"chorus-run-{job.run_id}",
                )
                type(self)._active_thread = thread
                thread.start()
            return True

    def get_state(self) -> dict[str, Any] | None:
        """Return the current state file contents, or None if absent."""
        return load_state()

    def get_results(self, run_id: str) -> dict[str, dict]:
        """Return a snapshot of the in-memory results registry for *run_id* (per filename).

        Returns a shallow copy to ensure mutations by the caller do not affect
        the manager's internal state.
        """
        with self._results_lock:
            return dict(self._results.get(run_id, {}))

    def set_file_result(self, run_id: str, name: str, results: dict[str, Any]) -> None:
        """Store pipeline results for a file, thread-safely.

        Called by the background worker thread to stash results after
        processing a file successfully.
        """
        with self._results_lock:
            if run_id not in self._results:
                self._results[run_id] = {}
            self._results[run_id][name] = results

    def mark_interrupted_if_stale(self) -> None:
        """Rewrite a ``running`` state left over from a prior process as ``interrupted``.

        Called from ``__init__``: a state file still claiming ``running``
        whose ``boot_id`` doesn't match this freshly minted process means
        the server that owned that run is gone. Partial outputs remain on
        disk; Past Jobs is the recovery path.

        Also cleans up spool directories from dead runs. Only directories
        from active runs (status == "running" with this process's boot_id)
        are preserved; all other directories are deleted.
        """
        state = load_state()
        # A run is currently active if it's running and owned by this process.
        active_run_id = None
        if (
            state
            and state.get("status") == "running"
            and state.get("boot_id") == self.boot_id
        ):
            active_run_id = state.get("run_id")

        if (
            state
            and state.get("status") == "running"
            and state.get("boot_id") != self.boot_id
        ):
            state["status"] = "interrupted"
            state["finished_at"] = time.time()
            write_state_atomic(state)

        # Clean up orphaned spool directories from dead runs.
        if RUNS_DIR.exists():
            for run_dir in RUNS_DIR.iterdir():
                if run_dir.is_dir() and run_dir.name != active_run_id:
                    with contextlib.suppress(OSError):
                        shutil.rmtree(run_dir)

    def clear_finished(self) -> None:
        """Clear a finished/interrupted run's state so a new one can start clean."""
        state = load_state()
        if state and state.get("status") != "running":
            with self._results_lock:
                self._results.pop(state.get("run_id"), None)
            with contextlib.suppress(OSError):
                ACTIVE_RUN_FILE.unlink()


@st.cache_resource
def get_run_manager() -> RunManager:
    """Return the process-wide RunManager singleton (the only st.* touch-point)."""
    return RunManager()
