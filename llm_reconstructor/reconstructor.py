"""LLM-assisted reconstruction of LOW-confidence consensus tokens."""

from __future__ import annotations

import logging

from consensus_merger.alignment import WordVote
from llm_reconstructor.ollama_client import suggest_token

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 4


def reconstruct_low_tokens_llm(votes: list[WordVote]) -> list[WordVote]:
    """Attempt to upgrade LOW-confidence words using a local Ollama model."""
    if not votes:
        return []

    low_indices = [idx for idx, vote in enumerate(votes) if vote.tier == "LOW"]
    if not low_indices:
        return votes

    updated = list(votes)

    for idx in low_indices:
        vote = votes[idx]
        candidates = sorted({v.strip().lower() for v in vote.variants if v.strip()})
        if not candidates:
            continue

        ctx_start = max(0, idx - CONTEXT_WINDOW)
        ctx_end = min(len(votes), idx + CONTEXT_WINDOW + 1)
        context_words = [votes[i].word for i in range(ctx_start, ctx_end) if i != idx]
        context = " ".join(context_words)

        suggestion = suggest_token(context=context, candidates=candidates)
        if suggestion is None:
            continue

        if suggestion not in candidates:
            logger.debug("Ignoring out-of-candidate suggestion '%s'", suggestion)
            continue

        if suggestion == vote.word.lower():
            continue

        updated[idx] = WordVote(
            word=suggestion,
            count=vote.count,
            total=vote.total,
            confidence=vote.confidence,
            tier="MEDIUM",
            variants=vote.variants,
        )

    return updated
