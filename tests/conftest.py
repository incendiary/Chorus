"""tests/conftest.py — Shared fixtures for the background-run UI tests (WP2).

``CHORUS_SYNC_RUN=1`` makes ``RunManager.start()`` execute inline (see
``ui/run_manager.py``), so an ``AppTest`` script that clicks Start lands
directly in the Finished view within a single ``at.run()`` call instead of
needing to poll a background thread. This fixture isolates all state file
I/O to a temporary per-test location using ``monkeypatch`` and ``tmp_path``,
so each test gets its own isolated ``active_run.json`` and the real
production file is never touched. The ``get_run_manager`` singleton is
cleared before and after each test to prevent state leakage between runs.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _chorus_sync_run(monkeypatch, tmp_path):
    monkeypatch.setenv("CHORUS_SYNC_RUN", "1")

    from ui import run_manager as run_manager_module
    from ui import run_state as run_state_module
    from ui.run_manager import get_run_manager

    # Redirect ACTIVE_RUN_FILE to a temp location in all modules that imported it.
    # This prevents any test from touching the real outputs/active_run.json.
    active_run_file = tmp_path / "active_run.json"
    monkeypatch.setattr(run_state_module, "ACTIVE_RUN_FILE", active_run_file)
    monkeypatch.setattr(run_manager_module, "ACTIVE_RUN_FILE", active_run_file)

    def _clear_state() -> None:
        get_run_manager.clear()

    _clear_state()
    yield
    _clear_state()
