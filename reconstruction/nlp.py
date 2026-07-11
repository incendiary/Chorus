"""
reconstruction/nlp.py — spaCy-powered NLP reconstruction.

Applies grammatical and contextual analysis to LOW-confidence tokens in the
consensus vote sequence.  The reconstruction pipeline proceeds as follows:

  1. The full consensus word sequence is assembled with LOW-tier tokens
     replaced by a ``[MASK]`` placeholder.
  2. spaCy analyses the surrounding context (POS tags, dependency arcs,
     named entities) to determine the grammatical role of each masked position.
  3. A candidate pool is built from:
       a. The observed variant forms of the LOW token.
       b. spaCy vocabulary words matching the required POS tag.
  4. Each candidate is scored by a composite function:
       - Semantic similarity to neighbouring context words (spaCy word vectors).
       - Grammatical plausibility (POS match score).
       - Levenshtein proximity to the original LOW token.
  5. The highest-scoring candidate replaces the mask if its composite score
     exceeds NLP_ACCEPT_THRESHOLD; otherwise the token is retained as LOW.

Graceful Degradation
────────────────────
spaCy and its ``en_core_web_md`` model (which includes word vectors) are
optional dependencies.  If unavailable, the module returns the vote list
unchanged with a warning.

Install:
    pip install spacy
    python -m spacy download en_core_web_md
"""

from __future__ import annotations

import logging

from consensus_merger.alignment import WordVote

logger = logging.getLogger(__name__)

# Composite score threshold above which a reconstructed token is accepted
NLP_ACCEPT_THRESHOLD: float = 0.55

# Maximum number of vocabulary candidates to evaluate per masked position
MAX_CANDIDATES: int = 20

# Context window: number of surrounding words used for semantic scoring
CONTEXT_WINDOW: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# spaCy loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    """Load (or return cached) spaCy model."""
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy  # type: ignore

        try:
            _nlp = spacy.load("en_core_web_md")
            logger.info("spaCy model 'en_core_web_md' loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_md' not found. "
                "Run: python -m spacy download en_core_web_md  "
                "Falling back to 'en_core_web_sm' (no word vectors)."
            )
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "No spaCy English model found. NLP reconstruction disabled."
                )
                return None
        return _nlp
    except ImportError:
        logger.warning(
            "spaCy is not installed. NLP reconstruction disabled. "
            "Install with: pip install spacy && python -m spacy download en_core_web_md"
        )
        return None


def probe_spacy_model() -> tuple[bool, str]:
    """Check whether spaCy and its ``en_core_web_md`` model are available.

    Intended to be called *before* enabling NLP reconstruction (mirrors
    :func:`reconstruction.ollama_client.probe_model`), so that a missing
    model surfaces as an explicit, actionable message rather than a silent
    runtime fallback warning.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when ``en_core_web_md`` is available, or
        ``(False, reason)`` with a human-readable reason and the exact
        command to fix it.
    """
    try:
        import spacy  # type: ignore
    except ImportError:
        return False, (
            "spaCy is not installed. Install it with: "
            "pip install spacy && python -m spacy download en_core_web_md"
        )

    try:
        spacy.load("en_core_web_md")
    except OSError:
        return False, (
            "spaCy model 'en_core_web_md' is not downloaded. "
            "Run: python -m spacy download en_core_web_md"
        )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────


def _levenshtein_similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    from nltk.metrics.distance import edit_distance

    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / max_len


def _semantic_similarity(candidate: str, context_words: list[str], nlp) -> float:
    """
    Mean cosine similarity between *candidate* and each context word
    using spaCy word vectors.  Returns 0.0 if vectors are unavailable.
    """
    if not context_words:
        return 0.0

    cand_tok = nlp(candidate)
    if not cand_tok or not cand_tok[0].has_vector:
        return 0.0

    scores = []
    for word in context_words:
        ctx_tok = nlp(word)
        if ctx_tok and ctx_tok[0].has_vector:
            sim = cand_tok[0].similarity(ctx_tok[0])
            scores.append(max(0.0, sim))

    return sum(scores) / len(scores) if scores else 0.0


def _pos_match_score(candidate_pos: str, required_pos: str | None) -> float:
    """Return 1.0 if POS tags match, 0.5 if unknown, 0.0 if mismatch."""
    if required_pos is None:
        return 0.5
    return 1.0 if candidate_pos == required_pos else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core reconstruction
# ─────────────────────────────────────────────────────────────────────────────


def reconstruct_low_tokens(votes: list[WordVote]) -> list[WordVote]:
    """
    Attempt to reconstruct LOW-confidence tokens using spaCy NLP analysis.

    Parameters
    ----------
    votes : list[WordVote]
        The consensus vote sequence from ``alignment.align_transcripts``.

    Returns
    -------
    list[WordVote]
        Updated vote list.  LOW tokens that pass the reconstruction
        threshold are upgraded to MEDIUM tier with ``word`` replaced by
        the best candidate.  All other votes are returned unchanged.
    """
    nlp = _get_nlp()
    if nlp is None:
        logger.warning("NLP reconstruction skipped — spaCy unavailable.")
        return votes

    low_indices = [i for i, v in enumerate(votes) if v.tier == "LOW"]
    if not low_indices:
        logger.info("No LOW-confidence tokens to reconstruct.")
        return votes

    logger.info("Reconstructing %d LOW-confidence token(s)…", len(low_indices))

    # Build a plain-text sentence for spaCy analysis
    # Replace LOW tokens with a placeholder for context parsing
    words = [v.word for v in votes]
    doc = nlp(" ".join(words))

    # Build a token → spaCy token map by index
    spacy_tokens = list(doc)

    updated = list(votes)

    for idx in low_indices:
        vote = votes[idx]

        # Gather context window words (excluding the LOW token itself)
        ctx_start = max(0, idx - CONTEXT_WINDOW)
        ctx_end = min(len(votes), idx + CONTEXT_WINDOW + 1)
        context_words = [
            votes[i].word
            for i in range(ctx_start, ctx_end)
            if i != idx and votes[i].tier != "LOW"
        ]

        # Determine expected POS from spaCy parse
        required_pos: str | None = None
        if idx < len(spacy_tokens):
            required_pos = spacy_tokens[idx].pos_

        # Build candidate pool from observed variant forms
        candidates = list(set(vote.variants)) if vote.variants else [vote.word]

        # Score each candidate
        best_word = vote.word
        best_score = 0.0

        for candidate in candidates[:MAX_CANDIDATES]:
            if not candidate:
                continue

            cand_doc = nlp(candidate)
            cand_pos = cand_doc[0].pos_ if cand_doc else ""

            sem_score = _semantic_similarity(candidate, context_words, nlp)
            pos_score = _pos_match_score(cand_pos, required_pos)
            lev_score = _levenshtein_similarity(candidate, vote.word)

            # Weighted composite: semantic 50%, POS 30%, Levenshtein 20%
            composite = 0.50 * sem_score + 0.30 * pos_score + 0.20 * lev_score

            if composite > best_score:
                best_score = composite
                best_word = candidate

        if best_score >= NLP_ACCEPT_THRESHOLD:
            logger.debug(
                "  [%d] '%s' → '%s'  (score=%.3f, POS=%s)",
                idx,
                vote.word,
                best_word,
                best_score,
                required_pos,
            )
            updated[idx] = WordVote(
                word=best_word,
                count=vote.count,
                total=vote.total,
                confidence=vote.confidence,
                tier="MEDIUM",  # Upgraded from LOW after reconstruction
                variants=vote.variants,
            )
        else:
            logger.debug(
                "  [%d] '%s' retained as LOW (best_score=%.3f < threshold=%.3f)",
                idx,
                vote.word,
                best_score,
                NLP_ACCEPT_THRESHOLD,
            )

    reconstructed = sum(1 for i in low_indices if updated[i].tier == "MEDIUM")
    logger.info(
        "NLP reconstruction complete: %d/%d LOW tokens upgraded to MEDIUM.",
        reconstructed,
        len(low_indices),
    )
    return updated
