"""
reconstruction — unified low-confidence token reconstruction.

Chorus reconstructs LOW-confidence consensus tokens through two interchangeable
strategies:

  - ``"nlp"`` — spaCy grammatical and semantic analysis (see :mod:`reconstruction.nlp`).
  - ``"llm"`` — a local Ollama model (see :mod:`reconstruction.llm`).

Callers select a *strategy*, not a *module*, through the single
:func:`reconstruct` entry point.  The underlying strategy functions
(``reconstruct_low_tokens`` and ``reconstruct_low_tokens_llm``) and the Ollama
helpers (``list_models``, ``probe_model``) remain importable for callers that
need them directly.
"""

from __future__ import annotations

from typing import Any

from consensus_merger.alignment import WordVote
from reconstruction import llm, nlp
from reconstruction.llm import reconstruct_low_tokens_llm
from reconstruction.nlp import probe_spacy_model, reconstruct_low_tokens
from reconstruction.ollama_client import list_models, probe_model

__all__ = [
    "reconstruct",
    "reconstruct_low_tokens",
    "reconstruct_low_tokens_llm",
    "list_models",
    "probe_model",
    "probe_spacy_model",
]


def reconstruct(votes: list[WordVote], *, strategy: str, **opts: Any) -> list[WordVote]:
    """Reconstruct LOW-confidence tokens using the chosen *strategy*.

    Parameters
    ----------
    votes : list[WordVote]
        The consensus vote sequence from ``alignment.align_transcripts``.
    strategy : str
        Either ``"nlp"`` (spaCy) or ``"llm"`` (Ollama).
    **opts
        Strategy-specific options.  The ``"llm"`` strategy accepts a
        ``model`` keyword naming the Ollama model to use.

    Returns
    -------
    list[WordVote]
        The updated vote list.

    Raises
    ------
    ValueError
        If *strategy* is not a recognised reconstruction strategy.
    """
    if strategy == "nlp":
        return nlp.reconstruct_low_tokens(votes)
    if strategy == "llm":
        return llm.reconstruct_low_tokens_llm(votes, model=opts.get("model"))
    raise ValueError(
        f"Unknown reconstruction strategy: {strategy!r}. " "Expected 'nlp' or 'llm'."
    )
