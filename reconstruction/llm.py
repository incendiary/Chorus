"""LLM-assisted reconstruction of LOW-confidence consensus tokens."""

from __future__ import annotations

import logging

from consensus_merger.alignment import WordVote
from reconstruction.ollama_client import probe_model, suggest_token

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 4


def get_reconstruction_status(model: str | None = None) -> dict[str, str]:
    """Check the status of LLM reconstruction readiness.

    Parameters
    ----------
    model : str, optional
        Ollama model name to probe. If None, uses the default from config.

    Returns
    -------
    dict[str, str]
        Status dictionary with keys:
          - "status": one of "complete" or "unavailable".
          - "reason": human-readable explanation (empty string when complete).
    """
    ok, reason = probe_model(model)
    if not ok:
        return {"status": "unavailable", "reason": reason}
    return {"status": "complete", "reason": ""}


def reconstruct_low_tokens_llm(
    votes: list[WordVote], model: str | None = None
) -> list[WordVote]:
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

        # Build per-candidate agreement fractions from vote counts.
        # For LOW tokens count == 1; variants list may contain forms from
        # other transcripts.  We estimate weight by variant occurrence.
        total = max(vote.total, 1)
        variant_counts: dict[str, int] = {}
        for form in vote.variants:
            key = form.strip().lower()
            if key:
                variant_counts[key] = variant_counts.get(key, 0) + 1
        candidate_weights = {
            c: round(variant_counts.get(c, 1) / total, 3) for c in candidates
        }

        suggestion = suggest_token(
            context=context,
            candidates=candidates,
            candidate_weights=candidate_weights,
            model=model,
        )
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
