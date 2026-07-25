"""ui/run_manager.py — Process-wide singleton owning background pipeline runs.

``RunManager`` enforces a single active run at a time and is the only place
in this feature that touches Streamlit (via ``get_run_manager``'s
``@st.cache_resource``). The class itself has no Streamlit dependency and
is safe to construct directly under plain pytest.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from typing import Any

import streamlit as st

from ui.run_state import ACTIVE_RUN_FILE, RunJob, load_state, write_state_atomic
from ui.run_worker import execute_run


class RunManager:
    """Owns the single background run thread and its in-memory results."""

    def __init__(self) -> None:
        self.boot_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._results: dict[str, dict[str, dict]] = {}
        self.mark_interrupted_if_stale()

    def is_running(self) -> bool:
        """Return True if a run is actually executing (live-thread check).

        Checking the state file's ``status`` alone is not enough: a state
        file can say ``running`` while the process that would ever update
        it is gone (server restart) — that is exactly the "stale" case
        ``mark_interrupted_if_stale`` handles separately.
        """
        return self._thread is not None and self._thread.is_alive()

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
                self._thread = thread
                thread.start()
            return True

    def get_state(self) -> dict[str, Any] | None:
        """Return the current state file contents, or None if absent."""
        return load_state()

    def get_results(self, run_id: str) -> dict[str, dict]:
        """Return the in-memory results registry for *run_id* (per filename)."""
        return self._results.get(run_id, {})

    def mark_interrupted_if_stale(self) -> None:
        """Rewrite a ``running`` state left over from a prior process as ``interrupted``.

        Called from ``__init__``: a state file still claiming ``running``
        whose ``boot_id`` doesn't match this freshly minted process means
        the server that owned that run is gone. Partial outputs remain on
        disk; Past Jobs is the recovery path.
        """
        state = load_state()
        if (
            state
            and state.get("status") == "running"
            and state.get("boot_id") != self.boot_id
        ):
            state["status"] = "interrupted"
            state["finished_at"] = time.time()
            write_state_atomic(state)

    def clear_finished(self) -> None:
        """Clear a finished/interrupted run's state so a new one can start clean."""
        state = load_state()
        if state and state.get("status") != "running":
            self._results.pop(state.get("run_id"), None)
            with contextlib.suppress(OSError):
                ACTIVE_RUN_FILE.unlink()


@st.cache_resource
def get_run_manager() -> RunManager:
    """Return the process-wide RunManager singleton (the only st.* touch-point)."""
    return RunManager()
