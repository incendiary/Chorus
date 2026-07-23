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
    manager._thread.join(timeout=5)
    assert manager.is_running() is False


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
