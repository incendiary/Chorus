"""
tests/test_merger.py — Unit tests for consensus_merger.merger.

Covers:
  - ValueError on empty or all-empty transcript inputs
  - Successful merge returns a valid Path
  - Output file is written to disk
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consensus_merger.merger import merge_transcripts


def _make_result(text: str) -> dict:
    """Minimal Whisper result dict with just a text body."""
    return {
        "text": text,
        "segments": [],
        "language": "en",
        "variant": "test",
        "model": "base",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Error cases
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeTranscriptsErrors:
    def test_empty_dict_raises_value_error(self):
        with pytest.raises(ValueError, match="No transcripts"):
            merge_transcripts({}, stem="test")

    def test_all_empty_texts_raise_value_error(self):
        transcripts = {
            "original": _make_result(""),
            "highpass": _make_result("   "),
            "normalised": _make_result("\n"),
            "denoised": _make_result(""),
        }
        with pytest.raises(ValueError, match="empty"):
            merge_transcripts(transcripts, stem="test")


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeTranscriptsSuccess:
    def test_single_variant_returns_path(self):
        transcripts = {"original": _make_result("hello world")}
        out = merge_transcripts(transcripts, stem="test_single")
        assert isinstance(out, Path)
        assert out.exists()

    def test_four_variants_returns_path(self):
        transcripts = {
            "original": _make_result("the quick brown fox"),
            "highpass": _make_result("the quick brown fox"),
            "normalised": _make_result("the quick brown dog"),
            "denoised": _make_result("the slow brown fox"),
        }
        out = merge_transcripts(transcripts, stem="test_four")
        assert isinstance(out, Path)
        assert out.exists()
        assert out.suffix == ".md"

    def test_output_contains_consensus_content(self):
        transcripts = {
            "original": _make_result("coventry building society mortgage"),
            "highpass": _make_result("coventry building society mortgage"),
        }
        out = merge_transcripts(transcripts, stem="test_content")
        content = out.read_text(encoding="utf-8")
        assert "coventry" in content.lower()
