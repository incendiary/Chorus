"""
tests/test_run_manager.py — WP1 background-run core tests.

Covers ``ui/run_state.py`` (atomic I/O), ``ui/run_manager.py`` (``RunManager``
lifecycle and single-run enforcement), and ``ui/run_worker.py``
(``execute_run``'s per-file loop). Pure pytest: ``ui.pipeline_invocation.run_pipeline``
is mocked (the same patch target ``tests/test_ui_run_loop.py`` uses, since
``ui/run_worker.py`` calls it via ``from ui import pipeline_invocation`` then
attribute access), and ``CHORUS_SYNC_RUN=1`` forces ``RunManager.start()`` to
run inline for deterministic, single-threaded tests except where a test
needs real concurrency (single-run enforcement).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ui import run_manager as run_manager_module
from ui import run_state as run_state_module
from ui import run_worker as run_worker_module
from ui.run_manager import RunManager
from ui.run_state import FileEntry, RunJob, load_state, write_state_atomic

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    """Redirect the active-run file and runs directory into tmp_path.

    Patched on every module holding its own imported reference so none of
    these tests can ever touch the real outputs/ directory.
    """
    active_run_file = tmp_path / "active_run.json"
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(run_state_module, "ACTIVE_RUN_FILE", active_run_file)
    monkeypatch.setattr(run_state_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(run_manager_module, "ACTIVE_RUN_FILE", active_run_file)
    monkeypatch.setattr(run_manager_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(run_worker_module, "RUNS_DIR", runs_dir)
    return active_run_file, runs_dir


@pytest.fixture
def _sync_mode(monkeypatch):
    monkeypatch.setenv("CHORUS_SYNC_RUN", "1")


def _make_job(tmp_path, run_id="run-1", names=("a.wav", "b.wav")) -> RunJob:
    files = []
    for name in names:
        spool_path = tmp_path / f"spool_{name}"
        spool_path.write_bytes(b"fake-audio")
        files.append(
            FileEntry(name=name, stem=Path(name).stem, spool_path=str(spool_path))
        )
    return RunJob(run_id=run_id, config={"language": "en"}, files=files)


def _fake_run_pipeline_factory(fail_names: set[str] | None = None):
    fail_names = fail_names or set()

    def _fake(audio_path, progress_callback=None, event_callback=None, **kwargs):
        name = Path(audio_path).name.replace("spool_", "")
        if progress_callback:
            progress_callback("Applying audio cleaning filters…", 0.05)
        if event_callback:
            event_callback(
                {
                    "stage": "cleaning",
                    "detail": None,
                    "frac": 0.05,
                    "passes_done": None,
                    "passes_total": None,
                    "segment": None,
                    "segments_total": None,
                    "stage_index": 1,
                    "stage_total": 6,
                }
            )
        if name in fail_names:
            raise RuntimeError(f"boom-{name}")
        return {"consensus_path": Path("dummy_consensus.md"), "elapsed_seconds": 0.01}

    return _fake


# ─────────────────────────────────────────────────────────────────────────────
# ui/run_state.py — atomic write
# ─────────────────────────────────────────────────────────────────────────────


def test_write_state_atomic_survives_concurrent_read(tmp_path):
    stop = threading.Event()
    errors: list[Exception] = []

    def writer():
        i = 0
        while not stop.is_set():
            state = {
                "schema_version": 1,
                "run_id": "run-x",
                "status": "running",
                "files": [{"name": f"f{n}.wav"} for n in range(20)],
                "counter": i,
            }
            write_state_atomic(state)
            i += 1

    def reader():
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            try:
                state = load_state()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                continue
            if state is not None:
                # A successful read must always be a complete, valid document
                # — never a torn/partial write.
                assert state["schema_version"] == 1
                assert state["run_id"] == "run-x"
                assert len(state["files"]) == 20

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    reader()
    stop.set()
    writer_thread.join(timeout=5)

    assert not errors


# ─────────────────────────────────────────────────────────────────────────────
# RunManager — single-run enforcement
# ─────────────────────────────────────────────────────────────────────────────


def test_second_start_while_running_returns_false(tmp_path, monkeypatch):
    monkeypatch.delenv("CHORUS_SYNC_RUN", raising=False)

    started = threading.Event()
    release = threading.Event()

    def blocking_fake(
        audio_path, progress_callback=None, event_callback=None, **kwargs
    ):
        started.set()
        release.wait(timeout=5)
        return {"consensus_path": Path("dummy.md"), "elapsed_seconds": 0.01}

    monkeypatch.setattr("ui.pipeline_invocation.run_pipeline", blocking_fake)

    manager = RunManager()
    job1 = _make_job(tmp_path, run_id="run-1", names=("a.wav",))
    job2 = _make_job(tmp_path, run_id="run-2", names=("b.wav",))

    assert manager.start(job1) is True
    assert started.wait(timeout=5)
    assert manager.is_running() is True
    assert manager.start(job2) is False

    release.set()
    type(manager)._active_thread.join(timeout=5)
    assert manager.is_running() is False


def test_second_manager_cannot_overlap_live_run(tmp_path, monkeypatch):
    """Direct construction must not bypass process-wide single-run ownership."""
    monkeypatch.delenv("CHORUS_SYNC_RUN", raising=False)

    started = threading.Event()
    release = threading.Event()

    def blocking_fake(
        audio_path, progress_callback=None, event_callback=None, **kwargs
    ):
        started.set()
        release.wait(timeout=5)
        return {"consensus_path": Path("dummy.md"), "elapsed_seconds": 0.01}

    monkeypatch.setattr("ui.pipeline_invocation.run_pipeline", blocking_fake)

    first = RunManager()
    assert first.start(_make_job(tmp_path, run_id="run-1", names=("a.wav",)))
    assert started.wait(timeout=5)

    second = RunManager()
    assert second.get_state()["status"] == "running"
    assert second.start(_make_job(tmp_path, run_id="run-2", names=("b.wav",))) is False

    release.set()
    type(first)._active_thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# RunManager / run_worker — per-file exception handling
# ─────────────────────────────────────────────────────────────────────────────


def test_per_file_exception_captured_and_batch_continues(
    tmp_path, _sync_mode, monkeypatch
):
    monkeypatch.setattr(
        "ui.pipeline_invocation.run_pipeline",
        _fake_run_pipeline_factory(fail_names={"a.wav"}),
    )

    manager = RunManager()
    job = _make_job(tmp_path, names=("a.wav", "b.wav"))
    assert manager.start(job) is True

    state = manager.get_state()
    assert state["status"] == "finished"
    files_by_name = {f["name"]: f for f in state["files"]}
    assert files_by_name["a.wav"]["status"] == "error"
    assert "boom-a.wav" in files_by_name["a.wav"]["error"]
    assert files_by_name["b.wav"]["status"] == "done"

    results = manager.get_results(job.run_id)
    assert "b.wav" in results
    assert "a.wav" not in results


def test_source_filename_forwarded_is_the_original_name_not_the_spool_name(
    tmp_path, _sync_mode, monkeypatch
):
    """execute_run must pass the original FileEntry.name as source_filename,
    not let run_pipeline default to the mangled spool_path filename — the
    per-job output folder must be named after the recording, not the
    random-suffixed temp file."""
    captured: dict[str, str | None] = {}

    def _fake(audio_path, progress_callback=None, event_callback=None, **kwargs):
        captured["source_filename"] = kwargs.get("source_filename")
        return {"consensus_path": Path("dummy_consensus.md"), "elapsed_seconds": 0.01}

    monkeypatch.setattr("ui.pipeline_invocation.run_pipeline", _fake)

    manager = RunManager()
    job = _make_job(tmp_path, names=("Interview Recording.wav",))
    assert manager.start(job) is True

    assert captured["source_filename"] == "Interview Recording.wav"


# ─────────────────────────────────────────────────────────────────────────────
# RunManager — stale state marked interrupted
# ─────────────────────────────────────────────────────────────────────────────


def test_stale_running_state_marked_interrupted_on_construction(tmp_path):
    write_state_atomic(
        {
            "schema_version": 1,
            "run_id": "old-run",
            "status": "running",
            "boot_id": "some-other-process-boot-id",
            "started_at": time.time(),
            "finished_at": None,
            "config": {},
            "log_path": None,
            "files": [],
        }
    )

    manager = RunManager()  # fresh boot_id, must not match the stale one

    state = manager.get_state()
    assert state["status"] == "interrupted"
    assert state["finished_at"] is not None


def test_running_state_from_this_process_not_marked_interrupted(tmp_path):
    manager = RunManager()
    write_state_atomic(
        {
            "schema_version": 1,
            "run_id": "current-run",
            "status": "running",
            "boot_id": manager.boot_id,
            "started_at": time.time(),
            "finished_at": None,
            "config": {},
            "log_path": None,
            "files": [],
        }
    )

    manager.mark_interrupted_if_stale()

    state = manager.get_state()
    assert state["status"] == "running"


# ─────────────────────────────────────────────────────────────────────────────
# run_worker — spool cleanup, run.log
# ─────────────────────────────────────────────────────────────────────────────


def test_spool_files_deleted_after_run(tmp_path, _sync_mode, monkeypatch):
    monkeypatch.setattr(
        "ui.pipeline_invocation.run_pipeline", _fake_run_pipeline_factory()
    )

    manager = RunManager()
    job = _make_job(tmp_path, names=("a.wav", "b.wav"))
    spool_paths = [Path(f.spool_path) for f in job.files]
    assert all(p.exists() for p in spool_paths)

    manager.start(job)

    assert all(not p.exists() for p in spool_paths)


def test_run_log_written(tmp_path, _sync_mode, monkeypatch, isolated_paths):
    monkeypatch.setattr(
        "ui.pipeline_invocation.run_pipeline", _fake_run_pipeline_factory()
    )
    _, runs_dir = isolated_paths

    manager = RunManager()
    job = _make_job(tmp_path, run_id="log-run")
    manager.start(job)

    state = manager.get_state()
    assert state["log_path"]
    log_path = Path(state["log_path"])
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() != ""
    assert log_path.parent == runs_dir / "log-run"


# ─────────────────────────────────────────────────────────────────────────────
# run_worker — history ring buffer capped at 50, segment ticks don't grow it
# ─────────────────────────────────────────────────────────────────────────────


def test_history_capped_and_segment_ticks_dont_grow_it(
    tmp_path, _sync_mode, monkeypatch
):
    def fake(audio_path, progress_callback=None, event_callback=None, **kwargs):
        if event_callback:
            for i in range(60):
                for tick in range(3):
                    event_callback(
                        {
                            "stage": "transcribing",
                            "detail": f"pass {i}",
                            "frac": 0.3,
                            "passes_done": i,
                            "passes_total": 60,
                            "segment": tick + 1,
                            "segments_total": 3,
                            "stage_index": 3,
                            "stage_total": 6,
                        }
                    )
        return {"consensus_path": Path("dummy.md"), "elapsed_seconds": 0.01}

    monkeypatch.setattr("ui.pipeline_invocation.run_pipeline", fake)

    manager = RunManager()
    job = _make_job(tmp_path, names=("a.wav",))
    manager.start(job)

    state = manager.get_state()
    history = state["files"][0]["history"]
    assert len(history) == 50
    # Oldest 10 distinct transitions ("pass 0".."pass 9") were evicted; the
    # buffer holds the most recent 50 (["transcribing", "pass 10"] .. "pass 59"),
    # and each of the 3 same-stage/detail segment ticks per pass did not add
    # a duplicate entry.
    assert history[0] == ["transcribing", "pass 10"]
    assert history[-1] == ["transcribing", "pass 59"]
    assert len(history) == len({tuple(h) for h in history})


# ─────────────────────────────────────────────────────────────────────────────
# ui/run_state.py — dataclass round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_file_entry_round_trip():
    entry = FileEntry(name="a.wav", stem="a", spool_path="/tmp/a.wav")
    data = entry.to_dict()
    restored = FileEntry.from_dict(data)
    assert restored == entry


def test_load_state_returns_none_when_absent(tmp_path):
    assert load_state() is None


def test_load_state_returns_none_on_corrupt_file(tmp_path, isolated_paths):
    active_run_file, _ = isolated_paths
    active_run_file.parent.mkdir(parents=True, exist_ok=True)
    active_run_file.write_text("{not valid json", encoding="utf-8")
    assert load_state() is None


class TestScriptRunContextNoiseSuppression:
    """Streamlit logs 'missing ScriptRunContext!' for every record emitted off
    the main thread. The worker runs without a script context by design, so
    each real log line arrived with ~10 lines of noise behind it, making the
    console unusable for watching a run (RC-3, 28 July production batch).
    """

    _CTX_LOGGER = "streamlit.runtime.scriptrunner_utils.script_run_context"

    def test_context_logger_is_quietened_during_a_run(
        self, tmp_path, monkeypatch, _sync_mode
    ):
        import logging

        ctx_logger = logging.getLogger(self._CTX_LOGGER)
        monkeypatch.setattr(ctx_logger, "level", logging.WARNING)

        observed: dict[str, int] = {}

        def _fake_pipeline(**kwargs):
            observed["level"] = ctx_logger.level
            return {"elapsed_seconds": 0.1}

        monkeypatch.setattr("ui.pipeline_invocation.run_pipeline", _fake_pipeline)

        manager = RunManager()
        manager.start(_make_job(tmp_path, names=("a.wav",)))

        # Suppressed while the run executes…
        assert observed["level"] == logging.ERROR
        # …and restored afterwards.
        assert ctx_logger.level == logging.WARNING


# ─────────────────────────────────────────────────────────────────────────────
# RD-24: Thread safety for _results and _active_thread
# ─────────────────────────────────────────────────────────────────────────────


def test_get_results_returns_a_copy(tmp_path, _sync_mode, monkeypatch):
    """Mutating the returned dict must not affect the manager's internal state."""
    monkeypatch.setattr(
        "ui.pipeline_invocation.run_pipeline", _fake_run_pipeline_factory()
    )

    manager = RunManager()
    job = _make_job(tmp_path, names=("a.wav",))
    manager.start(job)

    results = manager.get_results(job.run_id)
    original_len = len(results)
    results["fake_key"] = {"fake": "data"}

    # Verify the mutation didn't affect the manager's internal state.
    fresh_results = manager.get_results(job.run_id)
    assert len(fresh_results) == original_len
    assert "fake_key" not in fresh_results


def test_set_file_result_stores_value(tmp_path, monkeypatch):
    """set_file_result must store results in the manager."""
    monkeypatch.setattr(
        "ui.pipeline_invocation.run_pipeline", _fake_run_pipeline_factory()
    )

    manager = RunManager()
    run_id = "test-run"
    filename = "test.wav"
    results = {"consensus_path": Path("test.md")}

    manager.set_file_result(run_id, filename, results)

    stored = manager.get_results(run_id)
    assert filename in stored
    assert stored[filename] == results


def test_concurrent_writes_and_reads_no_exception(tmp_path, monkeypatch):
    """Multiple threads calling set_file_result while get_results is called must finish with no exception."""
    import time

    manager = RunManager()
    run_id = "concurrent-test"
    errors: list[Exception] = []

    def writer(index):
        try:
            for i in range(10):
                manager.set_file_result(run_id, f"file{index}_{i}.wav", {"index": i})
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        end_time = time.monotonic() + 1.0
        while time.monotonic() < end_time:
            try:
                results = manager.get_results(run_id)
                # Just access the dict to trigger any lock-related errors.
                _ = len(results)
                time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    writers = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    reader_thread = threading.Thread(target=reader)

    for w in writers:
        w.start()
    reader_thread.start()

    for w in writers:
        w.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not errors
    # Verify all values are present.
    results = manager.get_results(run_id)
    assert len(results) == 30  # 3 writers * 10 files


# ─────────────────────────────────────────────────────────────────────────────
# RD-25: Leftover spool directory cleanup
# ─────────────────────────────────────────────────────────────────────────────


def test_leftover_spool_directory_from_dead_run_is_removed(
    tmp_path, monkeypatch, isolated_paths
):
    """A spool directory from a dead run should be cleaned up when mark_interrupted_if_stale is called."""
    _, runs_dir = isolated_paths

    # Create a leftover directory from a dead run.
    dead_run_dir = runs_dir / "dead-run-id"
    dead_run_dir.mkdir(parents=True, exist_ok=True)
    (dead_run_dir / "spool_file.wav").write_text("fake audio")

    # Create a stale state file claiming that dead run is still running.
    write_state_atomic(
        {
            "schema_version": 1,
            "run_id": "dead-run-id",
            "status": "running",
            "boot_id": "some-other-process-boot-id",
            "started_at": time.time(),
            "finished_at": None,
            "config": {},
            "log_path": None,
            "files": [],
        }
    )

    # mark_interrupted_if_stale should delete the leftover directory.
    RunManager()
    assert not dead_run_dir.exists()


def test_active_run_directory_is_kept(tmp_path, monkeypatch, isolated_paths):
    """When a state file indicates an active run, its directory is not deleted."""
    _, runs_dir = isolated_paths

    # Create a manager to get this process's boot_id.
    manager = RunManager()

    # Create directories after the manager init (so they won't be deleted by init).
    active_run_dir = runs_dir / "active-run-id"
    active_run_dir.mkdir(parents=True, exist_ok=True)
    (active_run_dir / "spool_file.wav").write_text("fake audio")

    dead_run_dir = runs_dir / "dead-run-id"
    dead_run_dir.mkdir(parents=True, exist_ok=True)
    (dead_run_dir / "spool_file.wav").write_text("fake audio")

    # Write a state file saying the active run is running with this boot_id.
    write_state_atomic(
        {
            "schema_version": 1,
            "run_id": "active-run-id",
            "status": "running",
            "boot_id": manager.boot_id,
            "started_at": time.time(),
            "finished_at": None,
            "config": {},
            "log_path": None,
            "files": [],
        }
    )

    # Call mark_interrupted_if_stale(). It should keep the active run's
    # directory (because status=="running" and boot_id matches) and delete all others.
    manager.mark_interrupted_if_stale()

    assert active_run_dir.exists(), "Active run's directory should be kept"
    assert not dead_run_dir.exists(), "Dead run's directory should be deleted"


def test_missing_runs_dir_does_not_raise(tmp_path, monkeypatch, isolated_paths):
    """If RUNS_DIR doesn't exist yet, mark_interrupted_if_stale should not raise."""
    _, runs_dir = isolated_paths

    # Ensure RUNS_DIR doesn't exist.
    if runs_dir.exists():
        import shutil

        shutil.rmtree(runs_dir)

    # This should not raise.
    RunManager()
    # mark_interrupted_if_stale is called in __init__, so if we get here, it worked.
