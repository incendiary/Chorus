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


@pytest.fixture(autouse=True)
def _chorus_output_roots(monkeypatch, tmp_path):
    """Redirect every output root to *tmp_path* so the test suite never
    writes into the real project ``outputs/`` directory.

    ``from config import X`` binds its own module-level name at import time,
    so patching ``config.X`` alone does not reach a module that already
    imported ``X`` by name — each such module is patched individually here.
    A test that needs a specific layout (e.g. ``patch_consensus_dir`` in
    tests/test_integration.py) still overrides these with its own
    monkeypatch calls, which simply take precedence for that test.
    """
    import config as config_module

    variants_dir = tmp_path / "variants"
    transcripts_dir = tmp_path / "transcripts"
    consensus_dir = tmp_path / "consensus"
    jobs_dir = tmp_path / "jobs"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(config_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(config_module, "VARIANTS_DIR", variants_dir)
    monkeypatch.setattr(config_module, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(config_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(config_module, "JOBS_DIR", jobs_dir)

    import audio_processor.pipeline as audio_pipeline_module
    import consensus_merger.renderer as renderer_module
    import diarisation.diariser as diariser_module
    import export_engine.ai_context as ai_context_module
    import export_engine.exporter as exporter_module
    import pipeline_runner as pipeline_runner_module
    import transcription_engine.orchestrator as orchestrator_module
    import transcription_engine.whisper_engine as whisper_engine_module
    import ui.run_state as run_state_module

    monkeypatch.setattr(audio_pipeline_module, "VARIANTS_DIR", variants_dir)
    monkeypatch.setattr(orchestrator_module, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(whisper_engine_module, "TRANSCRIPTS_DIR", transcripts_dir)
    monkeypatch.setattr(renderer_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(ai_context_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(exporter_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(diariser_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(pipeline_runner_module, "CONSENSUS_DIR", consensus_dir)
    monkeypatch.setattr(run_state_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(run_state_module, "OUTPUTS_DIR", tmp_path)

    import ui.run_worker as run_worker_module

    monkeypatch.setattr(run_worker_module, "RUNS_DIR", runs_dir)
