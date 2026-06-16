"""tests/test_llm_reconstructor.py — unit tests for llm_reconstructor module."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib import error

from consensus_merger.alignment import WordVote
from llm_reconstructor import ollama_client
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


# ─────────────────────────────────────────────────────────────────────────────
# ollama_client failure modes
# ─────────────────────────────────────────────────────────────────────────────


def _make_urlopen_response(body: str):
    """Return a mock context-manager response object."""
    response = MagicMock()
    response.read.return_value = body.encode("utf-8")
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


def test_suggest_token_timeout_returns_none():
    with patch("llm_reconstructor.ollama_client.request.urlopen", side_effect=TimeoutError("timed out")):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result is None


def test_suggest_token_url_error_returns_none():
    with patch(
        "llm_reconstructor.ollama_client.request.urlopen",
        side_effect=error.URLError("connection refused"),
    ):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result is None


def test_suggest_token_http_error_returns_none():
    http_err = error.HTTPError(url="http://localhost", code=503, msg="Service Unavailable", hdrs=None, fp=None)
    with patch("llm_reconstructor.ollama_client.request.urlopen", side_effect=http_err):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result is None


def test_suggest_token_malformed_json_returns_none():
    with patch(
        "llm_reconstructor.ollama_client.request.urlopen",
        return_value=_make_urlopen_response("not valid { json"),
    ):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result is None


def test_suggest_token_empty_response_field_returns_none():
    with patch(
        "llm_reconstructor.ollama_client.request.urlopen",
        return_value=_make_urlopen_response('{"response": "  "}'),
    ):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result is None


def test_suggest_token_success():
    with patch(
        "llm_reconstructor.ollama_client.request.urlopen",
        return_value=_make_urlopen_response('{"response": "bar"}'),
    ):
        result = ollama_client.suggest_token(context="hello world", candidates=["foo", "bar"])
    assert result == "bar"


def test_reconstruct_votes_unchanged_on_all_failures(monkeypatch):
    """votes should be returned unchanged when every Ollama call fails."""
    votes = [
        _vote("the", tier="HIGH"),
        _vote("garbl", tier="LOW", variants=["garbl", "garble"]),
        _vote("world", tier="HIGH"),
    ]

    with patch("llm_reconstructor.ollama_client.request.urlopen", side_effect=TimeoutError("timed out")):
        result = reconstruct_low_tokens_llm(votes)

    assert result[1].word == "garbl"
    assert result[1].tier == "LOW"
