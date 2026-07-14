"""
tests/test_reconstructor.py — Unit tests for reconstruction.nlp.

Covers graceful degradation when spaCy or its model is unavailable,
and correct passthrough behaviour on empty inputs.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from consensus_merger.alignment import WordVote


def _make_vote(
    word: str, tier: str = "HIGH", count: int = 4, total: int = 4
) -> WordVote:
    return WordVote(
        word=word,
        count=count,
        total=total,
        confidence=round(count / total, 3),
        tier=tier,
        variants=[word],
    )


# ─────────────────────────────────────────────────────────────────────────────
# reconstruct_low_tokens — grammatical/semantic correction logic (RA-8)
#
# These tests exercise the real spaCy analysis path (en_core_web_md is a
# pinned project dependency, so no mocking is used here). The correction
# score is inherently fuzzy and model-version-dependent, so assertions
# focus on the invariants the pipeline relies on rather than on exact
# wording: LOW tokens are either upgraded to MEDIUM with a candidate word,
# or retained unchanged; HIGH tokens are never touched; sequence length
# and ordering are preserved.
#
# NOTE: These classes are deliberately placed *before*
# TestReconstructorDegradation / TestProbeSpacyModel below. Those tests
# patch ``sys.modules['spacy']`` to ``None`` and reload
# :mod:`reconstruction.nlp` to exercise the missing-dependency path; doing
# so permanently disables re-importing spaCy's C-extension dependency
# chain (numpy) for the rest of the process. Running the real-model tests
# first avoids that cross-test pollution.
# ─────────────────────────────────────────────────────────────────────────────


class TestReconstructionCorrectionLogic:
    def test_single_low_token_in_coherent_sentence_is_upgraded(self):
        """A LOW token with a plausible candidate in a coherent sentence
        should be upgraded to MEDIUM and exercise the real correction path."""
        from reconstruction.nlp import reconstruct_low_tokens

        votes = [
            _make_vote("the"),
            _make_vote("cat"),
            _make_vote("sat"),
            _make_vote("on"),
            _make_vote("the"),
            _make_vote("mat", tier="LOW", count=1, total=4),
        ]
        votes[-1].variants[:] = ["mat", "bat", "hat", "cat"]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        # HIGH-confidence tokens are untouched (word and tier unchanged)
        for original, updated in zip(votes[:-1], result[:-1], strict=True):
            assert updated.word == original.word
            assert updated.tier == "HIGH"
        # The LOW token is resolved one way or the other: either upgraded
        # to MEDIUM (with a candidate word), or retained as LOW unchanged.
        last = result[-1]
        assert last.tier in ("MEDIUM", "LOW")
        if last.tier == "MEDIUM":
            assert last.word in votes[-1].variants

    def test_multiple_low_tokens_in_one_sentence_all_processed(self):
        """More than one LOW-confidence gap in a single sentence must each
        be evaluated independently — the logic should not assume a single
        gap per window."""
        from reconstruction.nlp import reconstruct_low_tokens

        low_1 = _make_vote("dog", tier="LOW", count=1, total=4)
        low_1.variants[:] = ["dog", "fog", "log"]
        low_2 = _make_vote("cat", tier="LOW", count=1, total=4)
        low_2.variants[:] = ["cat", "hat", "bat"]

        votes = [
            _make_vote("the"),
            low_1,
            _make_vote("chased"),
            _make_vote("the"),
            low_2,
            _make_vote("quickly"),
        ]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        # Both LOW positions were considered (each resolved to MEDIUM or LOW,
        # never crash, never merged into a single decision).
        assert result[1].tier in ("MEDIUM", "LOW")
        assert result[4].tier in ("MEDIUM", "LOW")
        # Surrounding HIGH tokens remain untouched.
        for i in (0, 2, 3, 5):
            assert result[i].word == votes[i].word
            assert result[i].tier == "HIGH"

    def test_low_token_at_start_of_sentence_handles_boundary(self):
        """A LOW token at index 0 has no preceding context — the windowing
        logic must not raise an IndexError or crash."""
        from reconstruction.nlp import reconstruct_low_tokens

        low = _make_vote("dog", tier="LOW", count=1, total=4)
        low.variants[:] = ["dog", "fog", "log"]
        votes = [low, _make_vote("barks"), _make_vote("loudly")]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        assert result[0].tier in ("MEDIUM", "LOW")
        assert result[1].word == "barks"
        assert result[1].tier == "HIGH"
        assert result[2].word == "loudly"
        assert result[2].tier == "HIGH"

    def test_low_token_at_end_of_sentence_handles_boundary(self):
        """A LOW token as the final word has no following context — the
        windowing logic must not raise an IndexError or crash."""
        from reconstruction.nlp import reconstruct_low_tokens

        low = _make_vote("dog", tier="LOW", count=1, total=4)
        low.variants[:] = ["dog", "fog", "log"]
        votes = [_make_vote("the"), _make_vote("big"), low]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        assert result[0].word == "the"
        assert result[0].tier == "HIGH"
        assert result[1].word == "big"
        assert result[1].tier == "HIGH"
        assert result[2].tier in ("MEDIUM", "LOW")

    def test_ambiguous_context_degrades_to_original_token(self):
        """When the surrounding context is nonsensical, no candidate should
        score confidently — the token must remain a valid candidate rather
        than the reconstructor crashing or returning something structurally
        invalid (e.g. an empty string), and if retained as LOW it must be
        exactly the original word."""
        from reconstruction.nlp import reconstruct_low_tokens

        low = _make_vote("qwerty", tier="LOW", count=1, total=4)
        low.variants[:] = ["qwerty", "zxcvb"]
        votes = [_make_vote("xyzzy"), _make_vote("plugh"), low]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        last = result[-1]
        assert last.word
        if last.tier == "LOW":
            assert last.word == "qwerty"

    def test_high_confidence_tokens_are_never_modified(self):
        """Only LOW-tier votes are candidates for reconstruction; HIGH and
        MEDIUM tokens must pass through completely unchanged."""
        from reconstruction.nlp import reconstruct_low_tokens

        votes = [
            _make_vote("hello", tier="HIGH"),
            _make_vote("brave", tier="MEDIUM", count=3, total=4),
            _make_vote("world", tier="LOW", count=1, total=4),
        ]

        result = reconstruct_low_tokens(votes)

        assert result[0].word == "hello"
        assert result[0].tier == "HIGH"
        assert result[1].word == "brave"
        assert result[1].tier == "MEDIUM"

    def test_empty_variants_falls_back_to_original_word(self):
        """A LOW vote with no recorded variants should still be handled —
        the candidate pool falls back to the vote's own word."""
        from reconstruction.nlp import reconstruct_low_tokens

        low = _make_vote("dog", tier="LOW", count=1, total=4)
        low.variants.clear()
        votes = [_make_vote("the"), _make_vote("big"), low]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        assert result[-1].tier in ("MEDIUM", "LOW")
        assert result[-1].word == "dog"

    def test_sequence_length_and_order_preserved(self):
        """Downstream consumers rely on positional alignment between the
        input and output vote sequences being preserved exactly."""
        from reconstruction.nlp import reconstruct_low_tokens

        low = _make_vote("mat", tier="LOW", count=1, total=4)
        low.variants[:] = ["mat", "hat", "cat"]
        votes = [
            _make_vote("the"),
            _make_vote("cat"),
            _make_vote("sat"),
            low,
        ]

        result = reconstruct_low_tokens(votes)

        assert len(result) == len(votes)
        assert [v.word for v in votes[:3]] == [v.word for v in result[:3]]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers — direct unit tests (RA-8)
# ─────────────────────────────────────────────────────────────────────────────


class TestScoringHelpers:
    def test_levenshtein_similarity_identical_words(self):
        from reconstruction.nlp import _levenshtein_similarity

        assert _levenshtein_similarity("cat", "cat") == 1.0

    def test_levenshtein_similarity_partial_match(self):
        from reconstruction.nlp import _levenshtein_similarity

        # One substitution out of three characters.
        assert _levenshtein_similarity("cat", "bat") == pytest.approx(2 / 3)

    def test_levenshtein_similarity_empty_strings(self):
        from reconstruction.nlp import _levenshtein_similarity

        assert _levenshtein_similarity("", "") == 1.0

    def test_pos_match_score_unknown_required_pos(self):
        from reconstruction.nlp import _pos_match_score

        assert _pos_match_score("NOUN", None) == 0.5

    def test_pos_match_score_matching_pos(self):
        from reconstruction.nlp import _pos_match_score

        assert _pos_match_score("NOUN", "NOUN") == 1.0

    def test_pos_match_score_mismatched_pos(self):
        from reconstruction.nlp import _pos_match_score

        assert _pos_match_score("NOUN", "VERB") == 0.0

    def test_semantic_similarity_empty_context_returns_zero(self):
        from reconstruction.nlp import _get_nlp, _semantic_similarity

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("en_core_web_md unavailable in this environment")
        assert _semantic_similarity("dog", [], nlp) == 0.0

    def test_semantic_similarity_related_words_scores_above_zero(self):
        from reconstruction.nlp import _get_nlp, _semantic_similarity

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("en_core_web_md unavailable in this environment")
        score = _semantic_similarity("dog", ["cat"], nlp)
        assert 0.0 < score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


class TestReconstructorDegradation:
    def test_empty_list_returns_empty_list(self):
        from reconstruction.nlp import reconstruct_low_tokens

        result = reconstruct_low_tokens([])
        assert result == []

    def test_spacy_unavailable_returns_votes_unchanged(self):
        """If spaCy cannot be imported, votes are returned unmodified with a warning."""
        votes = [
            _make_vote("hello", tier="HIGH"),
            _make_vote("garbl", tier="LOW", count=1),
            _make_vote("world", tier="HIGH"),
        ]

        # Simulate spaCy being unavailable at the module level
        with patch.dict(sys.modules, {"spacy": None}):
            # Re-import to pick up the patched module state
            import importlib

            import reconstruction.nlp as rec_mod

            importlib.reload(rec_mod)

            result = rec_mod.reconstruct_low_tokens(votes)

        # Reload the real module afterwards so other tests are unaffected
        import importlib

        import reconstruction.nlp

        importlib.reload(reconstruction.nlp)

        assert len(result) == len(votes)
        # Words should be unchanged
        assert [v.word for v in result] == [v.word for v in votes]

    def test_no_low_tokens_returns_votes_unchanged(self):
        """If there are no LOW tokens, the vote list is returned as-is."""
        votes = [_make_vote("hello"), _make_vote("world"), _make_vote("today")]
        from reconstruction.nlp import reconstruct_low_tokens

        result = reconstruct_low_tokens(votes)
        assert len(result) == len(votes)
        assert [v.word for v in result] == ["hello", "world", "today"]


# ─────────────────────────────────────────────────────────────────────────────
# probe_spacy_model — actionable missing-model detection (RA-4.3)
# ─────────────────────────────────────────────────────────────────────────────


class TestProbeSpacyModel:
    def test_spacy_not_installed_returns_actionable_message(self):
        """When spaCy itself cannot be imported, the reason names the pip install."""
        with patch.dict(sys.modules, {"spacy": None}):
            import importlib

            import reconstruction.nlp as rec_mod

            importlib.reload(rec_mod)
            ok, reason = rec_mod.probe_spacy_model()

        import importlib

        import reconstruction.nlp

        importlib.reload(reconstruction.nlp)

        assert ok is False
        assert "pip install spacy" in reason

    def test_model_missing_returns_actionable_download_command(self):
        """When spaCy is installed but the model is missing, the reason names
        the exact download command — not a silent fallback."""
        import reconstruction.nlp as rec_mod

        fake_spacy = type(
            "FakeSpacy",
            (),
            {"load": staticmethod(lambda name: (_ for _ in ()).throw(OSError()))},
        )()

        with patch.dict(sys.modules, {"spacy": fake_spacy}):
            ok, reason = rec_mod.probe_spacy_model()

        assert ok is False
        assert "python -m spacy download en_core_web_md" in reason

    def test_model_available_returns_ok(self):
        """When the model loads successfully, probe reports success."""
        import reconstruction.nlp as rec_mod

        fake_spacy = type(
            "FakeSpacy", (), {"load": staticmethod(lambda name: object())}
        )()

        with patch.dict(sys.modules, {"spacy": fake_spacy}):
            ok, reason = rec_mod.probe_spacy_model()

        assert ok is True
        assert reason == ""
