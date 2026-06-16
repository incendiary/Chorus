"""
tests/test_speaker_names.py — Tests for speaker name persistence.

Verifies:
  - save_speaker_names writes valid JSON sidecar
  - load_speaker_names reads it back correctly
  - Identity mappings and empty names are filtered out on save
  - Missing/corrupt files return empty dict gracefully
  - get_unique_speakers extracts labels in order of first appearance
  - render_diarised_md respects speaker_map naming
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diarisation.diariser import (
    LabelledSegment,
    SpeakerSegment,
    get_unique_speakers,
    load_speaker_names,
    render_diarised_md,
    save_speaker_names,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_consensus_dir(tmp_path, monkeypatch):
    """Override CONSENSUS_DIR to a temp directory."""
    monkeypatch.setattr("diarisation.diariser.CONSENSUS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_labelled() -> list[LabelledSegment]:
    """Sample labelled segments with two speakers."""
    return [
        LabelledSegment(speaker="SPEAKER_00", start=0.0, end=2.5, text="Hello there."),
        LabelledSegment(
            speaker="SPEAKER_01", start=2.5, end=5.0, text="Hi, how are you?"
        ),
        LabelledSegment(
            speaker="SPEAKER_00", start=5.0, end=8.0, text="I'm fine thanks."
        ),
        LabelledSegment(speaker="SPEAKER_02", start=8.0, end=10.0, text="Me too."),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# save_speaker_names
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveSpeakerNames:
    def test_basic_save(self, tmp_consensus_dir):
        """Should write a valid JSON file with the given mapping."""
        mapping = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        path = save_speaker_names("test_audio", mapping)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}

    def test_filters_identity_mappings(self, tmp_consensus_dir):
        """Entries where name == label should be excluded."""
        mapping = {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "SPEAKER_01",  # identity — should be filtered
        }
        path = save_speaker_names("test_audio", mapping)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"SPEAKER_00": "Alice"}

    def test_filters_empty_names(self, tmp_consensus_dir):
        """Empty or whitespace-only names should be excluded."""
        mapping = {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "",
            "SPEAKER_02": "   ",
        }
        path = save_speaker_names("test_audio", mapping)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"SPEAKER_00": "Alice"}

    def test_strips_whitespace(self, tmp_consensus_dir):
        """Names should have leading/trailing whitespace stripped."""
        mapping = {"SPEAKER_00": "  Alice  "}
        path = save_speaker_names("test_audio", mapping)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"SPEAKER_00": "Alice"}

    def test_overwrites_existing(self, tmp_consensus_dir):
        """Saving again should overwrite the previous file."""
        save_speaker_names("test_audio", {"SPEAKER_00": "Alice"})
        save_speaker_names("test_audio", {"SPEAKER_00": "Bob"})
        data = json.loads(
            (tmp_consensus_dir / "test_audio_speakers.json").read_text(encoding="utf-8")
        )
        assert data == {"SPEAKER_00": "Bob"}


# ─────────────────────────────────────────────────────────────────────────────
# load_speaker_names
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadSpeakerNames:
    def test_load_existing(self, tmp_consensus_dir):
        """Should load a previously saved mapping."""
        save_speaker_names("test_audio", {"SPEAKER_00": "Alice"})
        result = load_speaker_names("test_audio")
        assert result == {"SPEAKER_00": "Alice"}

    def test_missing_file_returns_empty(self, tmp_consensus_dir):
        """Should return {} if no sidecar file exists."""
        result = load_speaker_names("nonexistent")
        assert result == {}

    def test_corrupt_json_returns_empty(self, tmp_consensus_dir):
        """Should return {} gracefully if JSON is invalid."""
        path = tmp_consensus_dir / "corrupt_speakers.json"
        path.write_text("{invalid json!!!", encoding="utf-8")
        result = load_speaker_names("corrupt")
        assert result == {}

    def test_non_dict_json_returns_empty(self, tmp_consensus_dir):
        """Should return {} if JSON is valid but not a dict."""
        path = tmp_consensus_dir / "list_speakers.json"
        path.write_text('["not", "a", "dict"]', encoding="utf-8")
        result = load_speaker_names("list")
        assert result == {}

    def test_coerces_values_to_strings(self, tmp_consensus_dir):
        """Should coerce non-string values to strings."""
        path = tmp_consensus_dir / "typed_speakers.json"
        path.write_text('{"SPEAKER_00": 123, "SPEAKER_01": true}', encoding="utf-8")
        result = load_speaker_names("typed")
        assert result == {"SPEAKER_00": "123", "SPEAKER_01": "True"}


# ─────────────────────────────────────────────────────────────────────────────
# get_unique_speakers
# ─────────────────────────────────────────────────────────────────────────────


class TestGetUniqueSpeakers:
    def test_order_of_first_appearance(self, sample_labelled):
        """Should return speakers in order of first appearance."""
        result = get_unique_speakers(sample_labelled)
        assert result == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def test_no_duplicates(self, sample_labelled):
        """Should not have duplicate entries."""
        result = get_unique_speakers(sample_labelled)
        assert len(result) == len(set(result))

    def test_empty_input(self):
        """Should return [] for empty input."""
        assert get_unique_speakers([]) == []

    def test_single_speaker(self):
        """Single speaker returns a one-element list."""
        segs = [LabelledSegment(speaker="SPEAKER_00", start=0, end=1, text="hi")]
        assert get_unique_speakers(segs) == ["SPEAKER_00"]


# ─────────────────────────────────────────────────────────────────────────────
# render_diarised_md with speaker names
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderDiarisedMdWithNames:
    def test_uses_speaker_map(self, tmp_consensus_dir, sample_labelled):
        """Speaker map names should appear in the rendered Markdown."""
        speaker_map = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        path = render_diarised_md(sample_labelled, "test", speaker_map=speaker_map)
        text = path.read_text(encoding="utf-8")

        assert "Alice" in text
        assert "Bob" in text
        # SPEAKER_02 has no mapping — should appear as-is
        assert "SPEAKER_02" in text

    def test_no_speaker_map(self, tmp_consensus_dir, sample_labelled):
        """Without a map, raw labels should appear."""
        path = render_diarised_md(sample_labelled, "test", speaker_map=None)
        text = path.read_text(encoding="utf-8")
        assert "SPEAKER_00" in text
        assert "SPEAKER_01" in text

    def test_empty_speaker_map(self, tmp_consensus_dir, sample_labelled):
        """Empty map should work the same as None."""
        path = render_diarised_md(sample_labelled, "test", speaker_map={})
        text = path.read_text(encoding="utf-8")
        assert "SPEAKER_00" in text
