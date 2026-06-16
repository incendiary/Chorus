"""
tests/test_ai_context.py — Tests for the AI context pack generator.

Verifies:
  - Basic generation produces a valid Markdown file
  - All required sections are present
  - Confidence statistics are calculated correctly
  - Uncertainty annotations list the correct words
  - Speaker information is included when provided
  - Clean transcript contains all words
  - Empty votes list is handled gracefully
"""

from __future__ import annotations

import pytest

from consensus_merger.alignment import WordVote
from export_engine.ai_context import generate_ai_context_pack

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_consensus_dir(tmp_path, monkeypatch):
    """Override CONSENSUS_DIR to a temp directory."""
    monkeypatch.setattr("export_engine.ai_context.CONSENSUS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_votes() -> list[WordVote]:
    """Sample votes with mixed confidence tiers."""
    return [
        WordVote(
            word="the",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["the", "the", "the", "the"],
        ),
        WordVote(
            word="quick",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["quick", "quick", "quick", "quick"],
        ),
        WordVote(
            word="brown",
            count=2,
            total=4,
            confidence=0.5,
            tier="MEDIUM",
            variants=["brown", "brown", "down", "drown"],
        ),
        WordVote(
            word="fox",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["fox", "fox", "fox", "fox"],
        ),
        WordVote(
            word="jumps",
            count=1,
            total=4,
            confidence=0.25,
            tier="LOW",
            variants=["jumps", "dumps", "bumps", "lumps"],
        ),
        WordVote(
            word="over",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["over", "over", "over", "over"],
        ),
        WordVote(
            word="the",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["the", "the", "the", "the"],
        ),
        WordVote(
            word="lazy",
            count=3,
            total=4,
            confidence=0.75,
            tier="HIGH",
            variants=["lazy", "lazy", "lazy", "hazy"],
        ),
        WordVote(
            word="dog",
            count=4,
            total=4,
            confidence=1.0,
            tier="HIGH",
            variants=["dog", "dog", "dog", "dog"],
        ),
    ]


@pytest.fixture
def sample_transcripts_meta() -> dict[str, dict]:
    """Sample transcripts metadata."""
    return {
        "original": {
            "text": "the quick brown fox jumps over the lazy dog",
            "model": "base",
            "language": "en",
        },
        "cleaned_hp": {
            "text": "the quick down fox dumps over the hazy dog",
            "model": "base",
            "language": "en",
        },
        "normalised": {
            "text": "the quick drown fox bumps over the lazy dog",
            "model": "base",
            "language": "en",
        },
        "denoised": {
            "text": "the quick brown fox lumps over the lazy dog",
            "model": "base",
            "language": "en",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Basic generation
# ─────────────────────────────────────────────────────────────────────────────


class TestBasicGeneration:
    def test_creates_file(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should create a .md file at the expected path."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test_audio",
            transcripts_meta=sample_transcripts_meta,
        )
        assert path.exists()
        assert path.name == "test_audio_ai_context.md"

    def test_output_is_markdown(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Output should start with a level-1 heading."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test_audio",
            transcripts_meta=sample_transcripts_meta,
        )
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# AI Context Pack")

    def test_non_empty_output(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Output should be non-trivial (> 500 chars given all the sections)."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test_audio",
            transcripts_meta=sample_transcripts_meta,
        )
        text = path.read_text(encoding="utf-8")
        assert len(text) > 500


# ─────────────────────────────────────────────────────────────────────────────
# Required sections
# ─────────────────────────────────────────────────────────────────────────────


class TestRequiredSections:
    def test_methodology_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain the methodology explanation."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Methodology" in text
        assert "multi-pass consensus" in text

    def test_processing_config_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain processing configuration."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Processing Configuration" in text
        assert "Whisper model" in text

    def test_confidence_stats_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain confidence statistics."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Confidence Statistics" in text
        assert "HIGH" in text
        assert "MEDIUM" in text
        assert "LOW" in text

    def test_clean_transcript_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain the clean transcript."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Clean Transcript" in text

    def test_uncertainty_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain uncertainty annotations."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Uncertainty Annotations" in text

    def test_usage_guidance_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should contain usage guidance for AI systems."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Usage Guidance" in text


# ─────────────────────────────────────────────────────────────────────────────
# Confidence statistics
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceStats:
    def test_correct_counts(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should report correct HIGH/MEDIUM/LOW counts."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        # 7 HIGH, 1 MEDIUM, 1 LOW in sample_votes
        assert "| HIGH | 7 |" in text
        assert "| MEDIUM | 1 |" in text
        assert "| LOW | 1 |" in text

    def test_reliability_score(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Should include overall reliability score."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        # 7/9 HIGH = 77.8%
        assert "77.8%" in text


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty annotations
# ─────────────────────────────────────────────────────────────────────────────


class TestUncertaintyAnnotations:
    def test_lists_uncertain_words(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """MEDIUM and LOW words should appear in the uncertainty table."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "brown" in text  # MEDIUM word
        assert "jumps" in text  # LOW word

    def test_variants_listed(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Variant forms should be listed for uncertain words."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        # "jumps" LOW word has variants: bumps, dumps, jumps, lumps
        assert "bumps" in text
        assert "dumps" in text
        assert "lumps" in text

    def test_all_high_no_table(self, tmp_consensus_dir, sample_transcripts_meta):
        """When all words are HIGH, should say 'No uncertain words'."""
        all_high = [
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
                count=4,
                total=4,
                confidence=1.0,
                tier="HIGH",
                variants=["world"],
            ),
        ]
        path = generate_ai_context_pack(
            votes=all_high, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "No uncertain words" in text


# ─────────────────────────────────────────────────────────────────────────────
# Clean transcript
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanTranscript:
    def test_contains_all_words(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Clean transcript should contain every consensus word."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        for v in sample_votes:
            assert v.word in text


# ─────────────────────────────────────────────────────────────────────────────
# Speaker information
# ─────────────────────────────────────────────────────────────────────────────


class TestSpeakerInfo:
    def test_no_speakers_no_section(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Without speaker labels, no Speaker Information section."""
        path = generate_ai_context_pack(
            votes=sample_votes, stem="test", transcripts_meta=sample_transcripts_meta
        )
        text = path.read_text(encoding="utf-8")
        assert "## Speaker Information" not in text

    def test_with_speakers(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """With speaker labels, should include speaker table."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test",
            transcripts_meta=sample_transcripts_meta,
            speaker_labels=["SPEAKER_00", "SPEAKER_01"],
            speaker_names={"SPEAKER_00": "Alice"},
        )
        text = path.read_text(encoding="utf-8")
        assert "## Speaker Information" in text
        assert "SPEAKER_00" in text
        assert "Alice" in text
        assert "SPEAKER_01" in text


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_votes(self, tmp_consensus_dir, sample_transcripts_meta):
        """Empty votes list should not crash."""
        path = generate_ai_context_pack(
            votes=[], stem="empty", transcripts_meta=sample_transcripts_meta
        )
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "AI Context Pack" in text

    def test_elapsed_seconds_displayed(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Elapsed time should appear in the output."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test",
            transcripts_meta=sample_transcripts_meta,
            elapsed_seconds=12.5,
        )
        text = path.read_text(encoding="utf-8")
        assert "12.5" in text

    def test_alignment_strategy_displayed(
        self, tmp_consensus_dir, sample_votes, sample_transcripts_meta
    ):
        """Custom alignment strategy should appear."""
        path = generate_ai_context_pack(
            votes=sample_votes,
            stem="test",
            transcripts_meta=sample_transcripts_meta,
            alignment_strategy="positional",
        )
        text = path.read_text(encoding="utf-8")
        assert "positional" in text
