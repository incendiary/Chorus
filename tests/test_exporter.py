"""
tests/test_exporter.py — Unit tests for export_engine.exporter.

Covers:
  - Timestamp formatting helpers (SRT and VTT)
  - SRT export: file created, correct structure
  - VTT export: file created, starts with WEBVTT header
"""

from __future__ import annotations

import re

from export_engine.exporter import (
    _seconds_to_srt_ts,
    _seconds_to_vtt_ts,
    export_srt,
    export_vtt,
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
