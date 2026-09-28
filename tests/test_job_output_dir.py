"""tests/test_job_output_dir.py — Regression tests for utils.job_output_dir.

Covers the per-job output folder layout introduced to stop re-runs of the
same recording overwriting each other's outputs (see CLAUDE.md/ROADMAP
work item: per-job output folders).
"""

from __future__ import annotations

import time

from utils import job_output_dir


def _write(path, data: bytes) -> None:
    path.write_bytes(data)


def test_same_bytes_different_names_share_project_dir(tmp_path):
    """Two files with identical audio bytes but different names map to the
    same ``<stem>-<sha8>`` project directory (the sha8, not the name, is
    what identifies the recording)."""
    root = tmp_path / "jobs"
    audio_a = tmp_path / "interview.wav"
    audio_b = tmp_path / "interview_renamed.wav"
    _write(audio_a, b"identical audio bytes")
    _write(audio_b, b"identical audio bytes")

    run_a = job_output_dir(audio_a, root=root)
    time.sleep(1.1)  # force a distinct timestamp folder
    run_b = job_output_dir(audio_b, source_name="interview.wav", root=root)

    assert run_a.parent == run_b.parent
    assert run_a != run_b


def test_same_name_different_bytes_get_different_project_dirs(tmp_path):
    """Two different recordings that happen to share a filename never share
    a project directory, because the sha8 differs."""
    root = tmp_path / "jobs"
    audio_a = tmp_path / "a" / "recording.wav"
    audio_b = tmp_path / "b" / "recording.wav"
    audio_a.parent.mkdir()
    audio_b.parent.mkdir()
    _write(audio_a, b"content one")
    _write(audio_b, b"content two, totally different")

    run_a = job_output_dir(audio_a, root=root)
    run_b = job_output_dir(audio_b, root=root)

    assert run_a.parent != run_b.parent


def test_two_calls_produce_distinct_run_dirs(tmp_path):
    """Calling job_output_dir twice for the same file never returns the same
    run directory, even within the same second."""
    root = tmp_path / "jobs"
    audio = tmp_path / "recording.wav"
    _write(audio, b"some audio bytes")

    run_1 = job_output_dir(audio, root=root)
    run_2 = job_output_dir(audio, root=root)

    assert run_1 != run_2
    assert run_1.parent == run_2.parent
    assert run_1.is_dir()
    assert run_2.is_dir()


def test_default_root_is_config_jobs_dir(tmp_path, monkeypatch):
    """With no explicit root, the helper falls back to config.JOBS_DIR."""
    import config as config_module

    monkeypatch.setattr(config_module, "JOBS_DIR", tmp_path / "jobs")
    audio = tmp_path / "recording.wav"
    _write(audio, b"some bytes")

    run_dir = job_output_dir(audio)

    assert run_dir.is_relative_to(tmp_path / "jobs")
