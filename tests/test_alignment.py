"""
tests/test_alignment.py — Unit tests for consensus_merger.alignment.

Covers:
  - Empty inputs (no ZeroDivisionError, no index errors)
  - Tier assignment correctness
  - Confidence values in valid range
  - Performance: 10,000-word transcript completes in under 2 seconds
"""

from __future__ import annotations

import time

from consensus_merger.alignment import WordVote, align_transcripts

# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — empty and trivial inputs
# ─────────────────────────────────────────────────────────────────────────────


class TestAlignTranscriptsEdgeCases:
    def test_empty_dict_returns_empty_list(self):
        result = align_transcripts({})
        assert result == []

    def test_single_empty_transcript_returns_empty_list(self):
        result = align_transcripts({"original": ""})
        assert result == []

    def test_all_empty_transcripts_return_empty_list(self):
        result = align_transcripts({"a": "", "b": "", "c": "   ", "d": "\n"})
        assert result == []

    def test_single_word_transcript(self):
        result = align_transcripts({"original": "hello"})
        assert len(result) == 1
        assert result[0].word == "hello"

    def test_single_variant_all_high_confidence(self):
        """With only one transcript, every word has confidence 1.0 → HIGH tier."""
        result = align_transcripts({"original": "the quick brown fox"})
        assert all(v.tier == "HIGH" for v in result)
        assert all(v.confidence == 1.0 for v in result)


# ─────────────────────────────────────────────────────────────────────────────
# Tier assignment
# ─────────────────────────────────────────────────────────────────────────────


class TestTierAssignment:
    def test_unanimous_agreement_is_high(self):
        """All four variants agree → HIGH."""
        text = "coventry building society"
        variants = {"a": text, "b": text, "c": text, "d": text}
        result = align_transcripts(variants)
        assert all(v.tier == "HIGH" for v in result)

    def test_single_occurrence_is_low(self):
        """A word present in only one of four transcripts → LOW."""
        variants = {
            "a": "hello world unique",
            "b": "hello world",
            "c": "hello world",
            "d": "hello world",
        }
        result = align_transcripts(variants)
        # "unique" appears only in 'a' — should be LOW
        unique_votes = [v for v in result if v.word == "unique"]
        assert unique_votes, "Expected 'unique' in vote sequence"
        assert unique_votes[0].tier == "LOW"

    def test_confidence_always_in_unit_interval(self):
        variants = {
            "a": "the quick brown fox",
            "b": "the quick brown dog",
            "c": "a quick brown fox",
            "d": "the slow brown fox",
        }
        result = align_transcripts(variants)
        for vote in result:
            assert 0.0 <= vote.confidence <= 1.0

    def test_word_vote_fields_populated(self):
        result = align_transcripts({"a": "test phrase", "b": "test phrase"})
        for vote in result:
            assert isinstance(vote, WordVote)
            assert vote.word
            assert vote.total == 2
            assert vote.count >= 1
            assert vote.tier in ("HIGH", "MEDIUM", "LOW")


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy matching
# ─────────────────────────────────────────────────────────────────────────────


class TestFuzzyMatching:
    def test_near_identical_words_grouped(self):
        """'colour' and 'color' should be grouped as the same word."""
        variants = {
            "a": "colour",
            "b": "color",
            "c": "colour",
            "d": "colour",
        }
        result = align_transcripts(variants)
        assert len(result) == 1
        assert result[0].tier in ("HIGH", "MEDIUM")


# ─────────────────────────────────────────────────────────────────────────────
# Performance benchmark
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_ten_thousand_words_under_two_seconds(self):
        """Positional alignment must complete in under 2 s for 10k words."""
        words = " ".join(f"word{i}" for i in range(10_000))
        variants = {"a": words, "b": words, "c": words, "d": words}

        start = time.perf_counter()
        result = align_transcripts(variants, strategy="positional")
        elapsed = time.perf_counter() - start

        assert len(result) == 10_000
        assert elapsed < 2.0, f"Alignment took {elapsed:.2f}s — exceeds 2s limit"


class TestConfigurableThresholds:
    """Per-run threshold overrides must change tier and grouping behaviour."""

    VARIANTS = {
        "a": "hello world",
        "b": "hello world",
        "c": "hello world",
        "d": "hello there",
    }

    def _tier_of(self, votes, word):
        return next(v.tier for v in votes if v.word == word)

    def test_consensus_threshold_override_changes_tier_both_strategies(self):
        from consensus_merger.alignment import align_transcripts

        for strategy in ("sequence", "positional"):
            # 3-of-4 agreement on "world": HIGH at the 0.75 default…
            default_votes = align_transcripts(self.VARIANTS, strategy=strategy)
            assert self._tier_of(default_votes, "world") == "HIGH"

            # …but MEDIUM when the bar is raised above 0.75.
            strict_votes = align_transcripts(
                self.VARIANTS, strategy=strategy, consensus_threshold=0.9
            )
            assert self._tier_of(strict_votes, "world") == "MEDIUM"

    def test_similarity_threshold_override_changes_grouping(self):
        from consensus_merger.alignment import align_transcripts

        # "colour" vs "color" — similar but not identical spellings.
        variants = {"a": "colour", "b": "color", "c": "colour", "d": "colour"}

        # Permissive similarity groups them into one 4-of-4 HIGH vote.
        merged = align_transcripts(
            variants, strategy="positional", similarity_threshold=0.6
        )
        assert len(merged) == 1
        assert merged[0].count == 4

        # A 1.0 threshold demands exact matches, splitting the group.
        split = align_transcripts(
            variants, strategy="positional", similarity_threshold=1.0
        )
        assert split[0].count == 3

    def test_default_behaviour_unchanged_when_omitted(self):
        from config import CONSENSUS_THRESHOLD
        from consensus_merger.alignment import align_transcripts

        explicit = align_transcripts(
            self.VARIANTS, consensus_threshold=CONSENSUS_THRESHOLD
        )
        implicit = align_transcripts(self.VARIANTS)
        assert [v.tier for v in explicit] == [v.tier for v in implicit]
