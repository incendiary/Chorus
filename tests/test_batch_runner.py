"""
tests/test_batch_runner.py — Unit tests for the batch processor.

Covers:
  - Multi-file directory processing (each file processed in turn)
  - Per-file output isolation under ``<output_dir>/<stem>/``
  - Partial failure: one file failing does not abort the batch
  - Empty directory / no matching files handled gracefully
  - Progress callback invocation
  - Batch report written to the expected location
  - CLI argument parsing

No real transcription, model loading, or network calls occur.  The
``run_pipeline`` function imported by ``batch_runner`` is patched in each
test, following the same mocking style used in ``tests/test_integration.py``.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from batch_processor.batch_runner import (
    BatchResult,
    _build_parser,
    _write_batch_report,
    discover_audio_files,
    run_batch,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_wav(path: Path, duration_s: float = 0.1) -> Path:
    """Write a minimal valid WAV file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return path


def _fake_pipeline_result(audio_path: Path, output_dir: Path | None = None) -> dict:
    """Return a minimal pipeline result dict for a given audio path."""
    base = output_dir or Path("outputs/consensus")
    consensus_dir = base / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    md_path = consensus_dir / f"{audio_path.stem}_consensus.md"
    md_path.write_text("# Consensus\n", encoding="utf-8")
    return {
        "consensus_path": md_path,
        "export_paths": {},
        "transcripts": {
            "original": {"text": "hello world", "language": "en", "segments": []}
        },
        "stem": audio_path.stem,
        "elapsed_seconds": 0.1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# discover_audio_files
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverAudioFiles:
    """Tests for the file-discovery helper."""

    def test_single_wav_file(self, tmp_path: Path) -> None:
        """A single WAV path should be returned as a one-element list."""
        wav = _make_wav(tmp_path / "a.wav")
        result = discover_audio_files([wav])
        assert result == [wav.resolve()]

    def test_non_audio_file_excluded(self, tmp_path: Path) -> None:
        """Non-audio files should be skipped with a warning."""
        txt = tmp_path / "notes.txt"
        txt.write_text("hello", encoding="utf-8")
        result = discover_audio_files([txt])
        assert result == []

    def test_directory_scan(self, tmp_path: Path) -> None:
        """All audio files in a directory should be discovered."""
        _make_wav(tmp_path / "a.wav")
        _make_wav(tmp_path / "b.mp3")
        result = discover_audio_files([tmp_path])
        assert len(result) == 2

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        """An empty directory should yield an empty list without raising."""
        result = discover_audio_files([tmp_path])
        assert result == []

    def test_recursive_flag(self, tmp_path: Path) -> None:
        """Recursive scanning should find files in subdirectories."""
        sub = tmp_path / "sub"
        _make_wav(sub / "deep.wav")
        result = discover_audio_files([tmp_path], recursive=True)
        assert len(result) == 1
        assert result[0].name == "deep.wav"

    def test_non_recursive_excludes_subdirs(self, tmp_path: Path) -> None:
        """Non-recursive scan must not descend into subdirectories."""
        sub = tmp_path / "sub"
        _make_wav(sub / "deep.wav")
        result = discover_audio_files([tmp_path], recursive=False)
        assert result == []

    def test_deduplication(self, tmp_path: Path) -> None:
        """The same file passed twice should appear only once."""
        wav = _make_wav(tmp_path / "a.wav")
        result = discover_audio_files([wav, wav])
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# run_batch — core behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBatch:
    """Tests for the main run_batch orchestration function."""

    def test_empty_inputs_returns_empty_list(self, tmp_path: Path) -> None:
        """run_batch with no matching audio files should return [] without crashing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("batch_processor.batch_runner._write_batch_report"):
            results = run_batch(inputs=[empty_dir])
        assert results == []

    def test_processes_multiple_files(self, tmp_path: Path) -> None:
        """Each audio file in the input directory should produce a BatchResult."""
        _make_wav(tmp_path / "a.wav")
        _make_wav(tmp_path / "b.wav")
        _make_wav(tmp_path / "c.wav")

        call_log: list[Path] = []

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            call_log.append(audio_path)
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            results = run_batch(inputs=[tmp_path])

        assert len(results) == 3
        assert len(call_log) == 3
        assert all(r.success for r in results)

    def test_per_file_output_isolation(self, tmp_path: Path) -> None:
        """When output_dir is given, each file must write into its own <stem>/ subdir."""
        _make_wav(tmp_path / "interview.wav")
        _make_wav(tmp_path / "lecture.wav")
        out_root = tmp_path / "outputs"

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            file_out = kwargs.get("output_dir")
            # The pipeline is passed an isolated subdir per file.
            assert file_out is not None
            assert file_out.parent == out_root
            assert file_out.name == audio_path.stem
            return _fake_pipeline_result(audio_path, file_out)

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            run_batch(inputs=[tmp_path], output_dir=out_root)

    def test_subdirs_are_stem_named(self, tmp_path: Path) -> None:
        """output_dir/<stem>/ paths should match audio file stems."""
        _make_wav(tmp_path / "my_file.wav")
        out_root = tmp_path / "out"
        captured: list[Path] = []

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            captured.append(kwargs.get("output_dir"))
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            run_batch(inputs=[tmp_path], output_dir=out_root)

        assert len(captured) == 1
        assert captured[0] == out_root / "my_file"

    def test_partial_failure_does_not_abort(self, tmp_path: Path) -> None:
        """A pipeline error on one file must not prevent the remaining files from running."""
        _make_wav(tmp_path / "good.wav")
        _make_wav(tmp_path / "bad.wav")
        _make_wav(tmp_path / "also_good.wav")

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            if "bad" in audio_path.name:
                raise RuntimeError("Simulated decode failure")
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            results = run_batch(inputs=[tmp_path])

        assert len(results) == 3
        success_count = sum(r.success for r in results)
        failure_count = sum(not r.success for r in results)
        assert success_count == 2
        assert failure_count == 1

    def test_failed_result_carries_error_message(self, tmp_path: Path) -> None:
        """A failed BatchResult should record the exception message."""
        _make_wav(tmp_path / "broken.wav")

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            raise RuntimeError("disk full")

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            results = run_batch(inputs=[tmp_path])

        assert len(results) == 1
        assert not results[0].success
        assert "disk full" in results[0].error

    def test_progress_callback_invoked_per_file(self, tmp_path: Path) -> None:
        """The progress callback should be called once per completed (or failed) file."""
        _make_wav(tmp_path / "a.wav")
        _make_wav(tmp_path / "b.wav")
        calls: list[tuple[int, int, str]] = []

        def _cb(current: int, total: int, filename: str) -> None:
            calls.append((current, total, filename))

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            run_batch(inputs=[tmp_path], progress_callback=_cb)

        assert len(calls) == 2
        # Indices should count up from 1
        assert calls[0][0] == 1
        assert calls[1][0] == 2
        # Total should always be 2
        assert all(c[1] == 2 for c in calls)

    def test_elapsed_seconds_recorded(self, tmp_path: Path) -> None:
        """Each BatchResult should have a non-negative elapsed_seconds field."""
        _make_wav(tmp_path / "a.wav")

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            results = run_batch(inputs=[tmp_path])

        assert results[0].elapsed_seconds >= 0.0

    def test_batch_report_written_on_success(self, tmp_path: Path) -> None:
        """_write_batch_report should be called after processing completes."""
        _make_wav(tmp_path / "a.wav")

        mock_report = MagicMock()

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            return _fake_pipeline_result(audio_path, kwargs.get("output_dir"))

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report", mock_report),
        ):
            run_batch(inputs=[tmp_path])

        mock_report.assert_called_once()

    def test_no_output_dir_uses_default_consensus(self, tmp_path: Path) -> None:
        """When no output_dir is supplied, pipeline is called with output_dir=None."""
        _make_wav(tmp_path / "x.wav")
        captured: list[Path | None] = []

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            captured.append(kwargs.get("output_dir"))
            return _fake_pipeline_result(audio_path)

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            run_batch(inputs=[tmp_path])

        assert captured == [None]

    def test_language_forwarded_to_pipeline(self, tmp_path: Path) -> None:
        """The language parameter should be forwarded to run_pipeline."""
        _make_wav(tmp_path / "x.wav")
        captured: list[str | None] = []

        def _mock_pipeline(audio_path: Path, **kwargs) -> dict:
            captured.append(kwargs.get("language"))
            return _fake_pipeline_result(audio_path)

        with (
            patch("pipeline_runner.run_pipeline", side_effect=_mock_pipeline),
            patch("batch_processor.batch_runner._write_batch_report"),
        ):
            run_batch(inputs=[tmp_path], language="fr")

        assert captured == ["fr"]


# ─────────────────────────────────────────────────────────────────────────────
# _write_batch_report
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteBatchReport:
    """Tests for the Markdown report writer."""

    def test_report_written_to_consensus_dir(self, tmp_path: Path, monkeypatch) -> None:
        """The report should appear at CONSENSUS_DIR/batch_report.md."""
        out_dir = tmp_path / "consensus"
        out_dir.mkdir(parents=True)
        monkeypatch.setattr("config.CONSENSUS_DIR", out_dir)
        monkeypatch.setattr("batch_processor.batch_runner.CONSENSUS_DIR", out_dir)

        r = BatchResult(Path("a.wav"))
        r.success = True
        r.elapsed_seconds = 1.5

        path = _write_batch_report([r])
        assert path == out_dir / "batch_report.md"
        assert path.exists()

    def test_report_counts_successes_and_failures(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The report Markdown should record the correct success/failure tallies."""
        out_dir = tmp_path / "consensus"
        out_dir.mkdir(parents=True)
        monkeypatch.setattr("config.CONSENSUS_DIR", out_dir)
        monkeypatch.setattr("batch_processor.batch_runner.CONSENSUS_DIR", out_dir)

        r_ok = BatchResult(Path("good.wav"))
        r_ok.success = True
        r_ok.elapsed_seconds = 0.5

        r_bad = BatchResult(Path("bad.wav"))
        r_bad.success = False
        r_bad.error = "decode error"
        r_bad.elapsed_seconds = 0.1

        path = _write_batch_report([r_ok, r_bad])
        text = path.read_text(encoding="utf-8")

        assert "Succeeded:** 1" in text
        assert "Failed:** 1" in text
        assert "good.wav" in text
        assert "bad.wav" in text


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildParser:
    """Tests for the argparse CLI definition."""

    def test_default_output_dir_is_none(self) -> None:
        """--output-dir should default to None when not supplied."""
        parser = _build_parser()
        args = parser.parse_args(["a.mp3"])
        assert args.output_dir is None

    def test_output_dir_parsed(self) -> None:
        """--output-dir should capture the supplied path string."""
        parser = _build_parser()
        args = parser.parse_args(["a.mp3", "--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_recursive_flag(self) -> None:
        """--recursive flag should set args.recursive to True."""
        parser = _build_parser()
        args = parser.parse_args(["a.mp3", "--recursive"])
        assert args.recursive is True

    def test_export_formats(self) -> None:
        """--export should accept known format names."""
        parser = _build_parser()
        args = parser.parse_args(["a.mp3", "--export", "pdf", "srt"])
        assert args.export == ["pdf", "srt"]

    def test_language_short_flag(self) -> None:
        """-l should set the language parameter."""
        parser = _build_parser()
        args = parser.parse_args(["a.mp3", "-l", "de"])
        assert args.language == "de"
