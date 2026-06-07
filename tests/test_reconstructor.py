"""
tests/test_reconstructor.py — Unit tests for nlp_reconstructor.reconstructor.

Covers graceful degradation when spaCy or its model is unavailable,
and correct passthrough behaviour on empty inputs.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

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
# Graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


class TestReconstructorDegradation:
    def test_empty_list_returns_empty_list(self):
        from nlp_reconstructor.reconstructor import reconstruct_low_tokens

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

            import nlp_reconstructor.reconstructor as rec_mod

            importlib.reload(rec_mod)

            result = rec_mod.reconstruct_low_tokens(votes)

        # Reload the real module afterwards so other tests are unaffected
        import importlib

        import nlp_reconstructor.reconstructor

        importlib.reload(nlp_reconstructor.reconstructor)

        assert len(result) == len(votes)
        # Words should be unchanged
        assert [v.word for v in result] == [v.word for v in votes]

    def test_no_low_tokens_returns_votes_unchanged(self):
        """If there are no LOW tokens, the vote list is returned as-is."""
        votes = [_make_vote("hello"), _make_vote("world"), _make_vote("today")]
        from nlp_reconstructor.reconstructor import reconstruct_low_tokens

        result = reconstruct_low_tokens(votes)
        assert len(result) == len(votes)
        assert [v.word for v in result] == ["hello", "world", "today"]
