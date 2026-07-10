"""
tests/test_exporter.py — Unit tests for export_engine.exporter.

Covers:
  - Timestamp formatting helpers (SRT and VTT)
  - SRT export: file created, correct structure
  - VTT export: file created, starts with WEBVTT header
  - ZIP export: honours output_dir for sidecars and does not contaminate global dir
"""

from __future__ import annotations

import io
import re
import zipfile

from export_engine.exporter import (
    _seconds_to_srt_ts,
    _seconds_to_vtt_ts,
    export_srt,
    export_vtt,
    export_zip,
)

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestampFormatting:
    def test_srt_zero_seconds(self):
        assert _seconds_to_srt_ts(0.0) == "00:00:00,000"

    def test_srt_one_hour_one_minute_one_second_half(self):
        assert _seconds_to_srt_ts(3661.5) == "01:01:01,500"

    def test_srt_sub_second_precision(self):
        assert _seconds_to_srt_ts(0.123) == "00:00:00,123"

    def test_vtt_zero_seconds(self):
        assert _seconds_to_vtt_ts(0.0) == "00:00:00.000"

    def test_vtt_uses_period_not_comma(self):
        """VTT format uses '.' as millisecond separator, not ','."""
        ts = _seconds_to_vtt_ts(3661.5)
        assert "." in ts
        assert "," not in ts

    def test_srt_uses_comma_not_period(self):
        ts = _seconds_to_srt_ts(3661.5)
        assert "," in ts
        assert ts.count(".") == 0


# ─────────────────────────────────────────────────────────────────────────────
# SRT export
# ─────────────────────────────────────────────────────────────────────────────


def _mock_whisper_result():
    return {
        "text": "Hello world. This is a test.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello world."},
            {"start": 2.5, "end": 5.0, "text": " This is a test."},
        ],
        "language": "en",
    }


class TestSRTExport:
    def test_srt_file_is_created(self):
        result = export_srt(_mock_whisper_result(), stem="test_srt")
        assert result.exists()

    def test_srt_file_has_correct_extension(self):
        result = export_srt(_mock_whisper_result(), stem="test_srt_ext")
        assert result.suffix == ".srt"

    def test_srt_has_valid_cue_structure(self):
        result = export_srt(_mock_whisper_result(), stem="test_srt_seq")
        content = result.read_text(encoding="utf-8").strip()

        cues = [block.splitlines() for block in content.split("\n\n") if block.strip()]
        assert len(cues) == 2

        ts_pattern = re.compile(
            r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
        )
        for idx, cue in enumerate(cues, start=1):
            assert cue[0] == str(idx)
            assert ts_pattern.match(cue[1])
            assert cue[2].strip()

    def test_srt_empty_segments_produces_empty_file(self):
        result = export_srt({"segments": []}, stem="test_srt_empty")
        content = result.read_text(encoding="utf-8")
        assert content.strip() == ""


# ─────────────────────────────────────────────────────────────────────────────
# VTT export
# ─────────────────────────────────────────────────────────────────────────────


class TestVTTExport:
    def test_vtt_file_is_created(self):
        result = export_vtt(_mock_whisper_result(), stem="test_vtt")
        assert result.exists()

    def test_vtt_starts_with_webvtt_header(self):
        result = export_vtt(_mock_whisper_result(), stem="test_vtt_header")
        content = result.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")

    def test_vtt_has_valid_header_and_cue_structure(self):
        result = export_vtt(_mock_whisper_result(), stem="test_vtt_period")
        content = result.read_text(encoding="utf-8")

        lines = content.splitlines()
        assert lines[0] == "WEBVTT"

        ts_pattern = re.compile(
            r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$"
        )
        ts_lines = [ln for ln in lines if "-->" in ln]
        assert len(ts_lines) == 2
        for line in ts_lines:
            assert ts_pattern.match(line)
            assert "," not in line

        assert "\n\n" in content, "VTT cues should be separated by blank lines"

    def test_vtt_contains_transcript_text(self):
        result = export_vtt(_mock_whisper_result(), stem="test_vtt_text")
        content = result.read_text(encoding="utf-8")
        assert "Hello world" in content or "hello world" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# JSON transcript bundle
# ─────────────────────────────────────────────────────────────────────────────


class TestExportTranscriptBundle:
    def _make_votes(self):
        from consensus_merger.alignment import WordVote

        return [
            WordVote(
                word="hello",
                count=4,
                total=4,
                confidence=1.0,
                tier="HIGH",
                variants=["hello"],
            ),
            WordVote(
                word="world",
                count=2,
                total=4,
                confidence=0.5,
                tier="MEDIUM",
                variants=["world", "word"],
            ),
            WordVote(
                word="garbl",
                count=1,
                total=4,
                confidence=0.25,
                tier="LOW",
                variants=["garbl"],
            ),
        ]

    def _make_transcripts(self):
        return {
            "original": {
                "text": "hello world garbl",
                "language": "en",
                "model": "base",
                "device": "cpu",
            },
            "highpass": {
                "text": "hello word garbl",
                "language": "en",
                "model": "base",
                "device": "cpu",
            },
        }

    def test_bundle_file_created(self, tmp_path):
        from export_engine.exporter import export_transcript_bundle

        path = export_transcript_bundle(
            self._make_transcripts(), self._make_votes(), "test", output_dir=tmp_path
        )
        assert path.exists()
        assert path.name == "test_bundle.json"

    def test_bundle_structure(self, tmp_path):
        import json

        from export_engine.exporter import export_transcript_bundle

        path = export_transcript_bundle(
            self._make_transcripts(), self._make_votes(), "test", output_dir=tmp_path
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "meta" in data
        assert data["meta"]["stem"] == "test"
        assert "variants" in data
        assert "original" in data["variants"]
        assert data["variants"]["original"]["text"] == "hello world garbl"
        assert "consensus" in data
        assert len(data["consensus"]) == 3
        assert data["consensus"][0] == {
            "word": "hello",
            "tier": "HIGH",
            "confidence": 1.0,
            "count": 4,
            "total": 4,
            "variants": ["hello"],
        }
        assert "statistics" in data
        assert data["statistics"]["high"] == 1
        assert data["statistics"]["medium"] == 1
        assert data["statistics"]["low"] == 1
        assert data["statistics"]["total_words"] == 3

    def test_bundle_in_pipeline_output(self, tmp_path):
        """run_pipeline should return bundle_path in its result dict."""
        from unittest.mock import patch

        from tests.test_integration import (
            _generate_sine_wav,
            _mock_run_transcription_pass,
        )

        audio = _generate_sine_wav(tmp_path / "audio.wav")
        with patch(
            "pipeline_runner.run_transcription_pass",
            side_effect=_mock_run_transcription_pass,
        ):
            from pipeline_runner import run_pipeline

            result = run_pipeline(
                audio_path=audio, language="en", output_dir=tmp_path / "out"
            )
        assert "bundle_path" in result
        assert result["bundle_path"].exists()
        assert result["bundle_path"].name.endswith("_bundle.json")


# ─────────────────────────────────────────────────────────────────────────────
# ZIP export with output_dir isolation
# ─────────────────────────────────────────────────────────────────────────────


class TestZipExportOutputDirIsolation:
    """Test that export_zip respects output_dir and does not leak to global dir."""

    def _make_mock_whisper_result(self) -> dict:
        """Minimal Whisper result for testing."""
        return {
            "text": "test recording",
            "segments": [{"start": 0.0, "end": 1.0, "text": " test recording"}],
            "language": "en",
        }

    def test_zip_reads_sidecars_from_output_dir(self, tmp_path):
        """ZIP should include sidecars written to the specified output_dir."""
        # Create a consensus markdown in output_dir
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        consensus_path = output_dir / "test_consensus.md"
        consensus_path.write_text("# Test Consensus\n\nHello world.\n")

        # Create the sidecars in output_dir
        speakers_path = output_dir / "test_speakers.json"
        speakers_path.write_text('{"SPEAKER_00": "Alice"}')

        ai_context_path = output_dir / "test_ai_context.md"
        ai_context_path.write_text("# AI Context\n\nGood recording.\n")

        diarised_path = output_dir / "test_diarised.md"
        diarised_path.write_text("# Diarised\n\nAlice: Hello world.\n")

        # Build ZIP
        zip_bytes = export_zip(
            consensus_md_path=consensus_path,
            whisper_result=self._make_mock_whisper_result(),
            stem="test",
            output_dir=output_dir,
        )

        # Verify ZIP contains all sidecars
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names = zf.namelist()
        assert any(
            "_speakers.json" in n for n in names
        ), "Missing speakers sidecar in ZIP"
        assert any("_ai_context.md" in n for n in names), "Missing AI context in ZIP"
        assert any(
            "_diarised.md" in n for n in names
        ), "Missing diarised sidecar in ZIP"

    def test_zip_does_not_pick_up_stale_sidecars_from_global_dir(
        self, tmp_path, monkeypatch
    ):
        """ZIP from isolated output_dir should not include stale files from global dir."""
        # Mock global CONSENSUS_DIR to tmp_path / "global"
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        monkeypatch.setattr("export_engine.exporter.CONSENSUS_DIR", global_dir)

        # Create stale files in global dir
        stale_speakers = global_dir / "test_speakers.json"
        stale_speakers.write_text('{"SPEAKER_00": "OldName"}')

        stale_ai_context = global_dir / "test_ai_context.md"
        stale_ai_context.write_text("# Old AI Context\n\nStale data.\n")

        # Create fresh consensus in isolated output_dir
        isolated_dir = tmp_path / "isolated"
        isolated_dir.mkdir()
        consensus_path = isolated_dir / "test_consensus.md"
        consensus_path.write_text("# Test Consensus\n\nFresh recording.\n")

        # Build ZIP from isolated_dir (no sidecars there)
        zip_bytes = export_zip(
            consensus_md_path=consensus_path,
            whisper_result=self._make_mock_whisper_result(),
            stem="test",
            output_dir=isolated_dir,
        )

        # Verify ZIP does NOT contain the stale sidecars from global dir
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names = zf.namelist()
        assert not any(
            "_speakers.json" in n for n in names
        ), "ZIP should not include stale speakers sidecar from global dir"
        assert not any(
            "_ai_context.md" in n for n in names
        ), "ZIP should not include stale AI context from global dir"

    def test_zip_contains_consensus_and_plaintext(self, tmp_path):
        """ZIP should always include consensus markdown and plaintext variants."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        consensus_path = output_dir / "test_consensus.md"
        consensus_path.write_text(
            "# Consensus\n\n## Consensus Transcript\n\n**~~guess~~**[^1]\n\n"
        )

        zip_bytes = export_zip(
            consensus_md_path=consensus_path,
            whisper_result=self._make_mock_whisper_result(),
            stem="test",
            output_dir=output_dir,
        )

        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names = zf.namelist()
        # Should have consensus markdown
        assert any("consensus.md" in n for n in names)
        # Should have plaintext variants
        assert any("most_likely" in n for n in names)
