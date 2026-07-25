"""tests/conftest.py — Shared fixtures for the background-run UI tests (WP2).

``CHORUS_SYNC_RUN=1`` makes ``RunManager.start()`` execute inline (see
``ui/run_manager.py``), so an ``AppTest`` script that clicks Start lands
directly in the Finished view within a single ``at.run()`` call instead of
needing to poll a background thread. Without also clearing the active-run
state file and the ``get_run_manager`` singleton between tests, one test's
finished/running state would leak into the next test's fresh ``AppTest``
run (both read the same on-disk ``outputs/active_run.json`` and the same
process-wide cached ``RunManager``).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _chorus_sync_run(monkeypatch):
    monkeypatch.setenv("CHORUS_SYNC_RUN", "1")

    from ui.run_manager import get_run_manager
    from ui.run_state import ACTIVE_RUN_FILE

    def _clear_state() -> None:
        if ACTIVE_RUN_FILE.exists():
            ACTIVE_RUN_FILE.unlink()
        get_run_manager.clear()

    _clear_state()
    yield
    _clear_state()
