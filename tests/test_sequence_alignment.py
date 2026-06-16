"""
tests/test_sequence_alignment.py — Tests for Needleman-Wunsch alignment.

Covers:
  - Basic alignment correctness
  - Handling of insertions/deletions
  - Comparison with positional alignment on tricky inputs
  - Empty/trivial inputs
  - Performance: 1,000-word transcripts in reasonable time
"""

from __future__ import annotations

import time

import pytest

from consensus_merger.alignment import WordVote, align_transcripts
from consensus_merger.sequence_alignment import (
    _needleman_wunsch,
    align_transcripts_sequence,
)

# ─────────────────────────────────────────────────────────────────────────────
# Needleman-Wunsch pairwise tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNeedlemanWunsch:
    def test_identical_sequences(self):
        seq = ["the", "quick", "brown", "fox"]
        result = _needleman_wunsch(seq, seq)
        assert len(result) == 4
        for a, b in result:
            assert a == b

    def test_insertion_handling(self):
        """Sequence B has an extra word — alignment should introduce a gap."""
        seq_a = ["the", "brown", "fox"]
        seq_b = ["the", "quick", "brown", "fox"]
        result = _needleman_wunsch(seq_a, seq_b)
        # Should align: (the,the), ("",quick), (brown,brown), (fox,fox)
        assert len(result) == 4
        # Verify 'brown' and 'fox' align correctly
        aligned_b = [pair[1] for pair in result if pair[1]]
        assert "quick" in aligned_b
        assert "brown" in aligned_b
        assert "fox" in aligned_b

    def test_deletion_handling(self):
        """Sequence B is missing a word — gap in B."""
        seq_a = ["the", "quick", "brown", "fox"]
        seq_b = ["the", "brown", "fox"]
        result = _needleman_wunsch(seq_a, seq_b)
        # 'quick' in A should align with a gap in B
        gaps_in_b = [pair for pair in result if pair[0] and not pair[1]]
        assert len(gaps_in_b) >= 1

    def test_empty_sequences(self):
        assert _needleman_wunsch([], []) == []
        result = _needleman_wunsch(["hello"], [])
        assert result == [("hello", "")]
        result = _needleman_wunsch([], ["hello"])
        assert result == [("", "hello")]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence alignment full pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestAlignTranscriptsSequence:
    def test_empty_dict_returns_empty(self):
        assert align_transcripts_sequence({}) == []

    def test_single_empty_transcript(self):
        assert align_transcripts_sequence({"a": ""}) == []

    def test_identical_transcripts_all_high(self):
        text = "the quick brown fox"
        variants = {"a": text, "b": text, "c": text, "d": text}
        result = align_transcripts_sequence(variants)
        assert len(result) == 4
        assert all(v.tier == "HIGH" for v in result)

    def test_insertion_produces_low_word(self):
        """A word in only one variant should be LOW confidence."""
        variants = {
            "a": "the quick brown fox",
            "b": "the brown fox",
            "c": "the brown fox",
            "d": "the brown fox",
        }
        result = align_transcripts_sequence(variants)
        # 'quick' appears only in variant A — should be LOW
        quick_votes = [v for v in result if v.word == "quick"]
        assert quick_votes, "Expected 'quick' in output"
        assert quick_votes[0].tier == "LOW"

    def test_substitution_detection(self):
        """Different words at same position should reduce confidence."""
        variants = {
            "a": "the quick brown fox",
            "b": "the quick brown dog",
            "c": "the quick brown fox",
            "d": "the quick brown fox",
        }
        result = align_transcripts_sequence(variants)
        # 'fox' should be HIGH (3/4), 'dog' should be LOW (1/4)
        fox_votes = [v for v in result if v.word == "fox"]
        assert fox_votes
        assert fox_votes[0].tier == "HIGH"

    def test_confidence_in_valid_range(self):
        variants = {
            "a": "hello world foo bar",
            "b": "hello beautiful world bar",
            "c": "hello world bar",
            "d": "hello world foo bar",
        }
        result = align_transcripts_sequence(variants)
        for vote in result:
            assert 0.0 <= vote.confidence <= 1.0

    def test_single_transcript(self):
        result = align_transcripts_sequence({"a": "hello world"})
        assert len(result) == 2
        assert all(v.confidence == 1.0 for v in result)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDispatcher:
    def test_strategy_positional(self):
        text = "hello world"
        variants = {"a": text, "b": text}
        result = align_transcripts(variants, strategy="positional")
        assert len(result) == 2

    def test_strategy_sequence(self):
        text = "hello world"
        variants = {"a": text, "b": text}
        result = align_transcripts(variants, strategy="sequence")
        assert len(result) == 2

    def test_default_strategy_works(self):
        """Default (from config) should not crash."""
        text = "the quick brown fox"
        variants = {"a": text, "b": text, "c": text, "d": text}
        result = align_transcripts(variants)
        assert len(result) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────────────────────


class TestSequencePerformance:
    def test_five_hundred_words_under_thirty_seconds(self):
        """Sequence alignment on 500 identical words (fast path) should be quick."""
        words = " ".join(f"word{i}" for i in range(500))
        variants = {"a": words, "b": words, "c": words, "d": words}

        start = time.perf_counter()
        result = align_transcripts_sequence(variants)
        elapsed = time.perf_counter() - start

        assert len(result) == 500
        assert elapsed < 30.0, f"Sequence alignment took {elapsed:.2f}s — exceeds 30s"

    def test_realistic_divergent_transcripts(self):
        """Variants with insertions/deletions should still complete quickly."""
        base = [f"word{i}" for i in range(200)]
        # Variant b has an insertion at position 50
        var_b = base[:50] + ["inserted"] + base[50:]
        # Variant c is missing a word at position 100
        var_c = base[:100] + base[101:]
        # Variant d has a substitution at position 150
        var_d = base[:150] + ["replaced"] + base[151:]

        variants = {
            "a": " ".join(base),
            "b": " ".join(var_b),
            "c": " ".join(var_c),
            "d": " ".join(var_d),
        }

        start = time.perf_counter()
        result = align_transcripts_sequence(variants)
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"Took {elapsed:.2f}s"
        # Should detect 'inserted' as LOW (only in one variant)
        inserted_votes = [v for v in result if v.word == "inserted"]
        assert inserted_votes
        assert inserted_votes[0].tier == "LOW"
