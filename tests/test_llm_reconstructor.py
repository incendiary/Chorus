"""tests/test_llm_reconstructor.py — unit tests for llm_reconstructor module."""

from __future__ import annotations

from consensus_merger.alignment import WordVote
from llm_reconstructor.reconstructor import reconstruct_low_tokens_llm


def _vote(word: str, tier: str = "HIGH", variants: list[str] | None = None) -> WordVote:
    return WordVote(
        word=word,
        count=1 if tier == "LOW" else 3,
        total=4,
        confidence=0.25 if tier == "LOW" else 0.75,
        tier=tier,
        variants=variants or [word],
    )


def test_llm_reconstructor_empty_input():
    assert reconstruct_low_tokens_llm([]) == []


def test_llm_reconstructor_no_low_tokens():
    votes = [_vote("hello", tier="HIGH"), _vote("world", tier="MEDIUM")]
    result = reconstruct_low_tokens_llm(votes)
    assert [v.word for v in result] == ["hello", "world"]


def test_llm_reconstructor_upgrades_low_token(monkeypatch):
    votes = [
        _vote("the", tier="HIGH"),
        _vote("garbl", tier="LOW", variants=["garbl", "garble"]),
        _vote("word", tier="HIGH"),
    ]

    monkeypatch.setattr(
        "llm_reconstructor.reconstructor.suggest_token",
        lambda **kwargs: "garble",
    )

    result = reconstruct_low_tokens_llm(votes)
    assert result[1].word == "garble"
    assert result[1].tier == "MEDIUM"


def test_llm_reconstructor_ignores_unknown_suggestion(monkeypatch):
    votes = [
        _vote("the", tier="HIGH"),
        _vote("garbl", tier="LOW", variants=["garbl", "garble"]),
        _vote("word", tier="HIGH"),
    ]

    monkeypatch.setattr(
        "llm_reconstructor.reconstructor.suggest_token",
        lambda **kwargs: "not-in-candidates",
    )

    result = reconstruct_low_tokens_llm(votes)
    assert result[1].word == "garbl"
    assert result[1].tier == "LOW"
