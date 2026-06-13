"""
consensus_merger/alignment.py — Word-level alignment and confidence scoring.

This module implements the core voting logic for the Chorus consensus engine.
Given N transcript strings, it:

  1. Tokenises each transcript into a normalised word sequence.
  2. Aligns the sequences using a majority-vote sliding window.
  3. Assigns a confidence weight to every word position based on how many
     transcripts agree on that token.
  4. Applies NLTK-based fuzzy similarity for near-matches that differ only
     by minor spelling or recognition artefacts.

Confidence Tiers
────────────────
  HIGH    — word present (exact or fuzzy match) in ≥ CONSENSUS_THRESHOLD
            fraction of transcripts.  Rendered as plain text.
  MEDIUM  — word present in exactly 2 transcripts but below threshold.
            Rendered with a single-underline highlight in the output.
  LOW     — word present in only 1 transcript.
            Rendered with a double-asterisk (bold) warning highlight.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import nltk
from nltk.metrics.distance import edit_distance

from config import ALIGNMENT_STRATEGY, CONSENSUS_THRESHOLD, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Ensure required NLTK data is available
for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WordVote:
    """Represents a single word position in the consensus sequence."""

    word: str  # Canonical (most-voted) word form
    count: int  # Number of transcripts containing this word
    total: int  # Total number of transcripts compared
    confidence: float  # count / total
    tier: str  # "HIGH", "MEDIUM", or "LOW"
    variants: list[str] = field(default_factory=list)  # All observed forms


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s'-]")


def _tokenise(text: str) -> list[str]:
    """
    Normalise and tokenise a transcript string into a word list.

    Lowercases the text, strips punctuation (preserving apostrophes and
    hyphens for contractions and compound words), and splits on whitespace.
    """
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    return [t for t in tokens if t]


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy similarity
# ─────────────────────────────────────────────────────────────────────────────


def _normalised_similarity(a: str, b: str) -> float:
    """
    Return a normalised similarity score in [0, 1] between two strings.

    Uses Levenshtein edit distance normalised by the length of the longer
    string.  A score of 1.0 indicates an exact match.
    """
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = edit_distance(a, b)
    return 1.0 - dist / max_len


def _best_fuzzy_match(
    word: str,
    candidates: Sequence[str],
) -> tuple[str, float]:
    """
    Return the best fuzzy match for *word* from *candidates* and its score.
    """
    best_word = word
    best_score = 0.0
    for candidate in candidates:
        score = _normalised_similarity(word, candidate)
        if score > best_score:
            best_score = score
            best_word = candidate
    return best_word, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Core alignment algorithm
# ─────────────────────────────────────────────────────────────────────────────


def _align_positional(transcripts: dict[str, str]) -> list[WordVote]:
    """
    Align multiple transcript strings using positional (index-based) comparison.

    This is the legacy algorithm: fast but assumes all variants produce
    similar word counts. Insertions/deletions in one variant will shift
    all subsequent positions out of alignment.

    Parameters
    ----------
    transcripts : dict[str, str]
        Mapping of variant key → plain-text transcript string.

    Returns
    -------
    list[WordVote]
        Ordered list of WordVote objects representing the consensus sequence.
    """
    if not transcripts:
        return []

    token_lists = {key: _tokenise(text) for key, text in transcripts.items()}
    n_transcripts = len(token_lists)

    # Pad all token lists to the same length
    max_len = max((len(tl) for tl in token_lists.values()), default=0)
    padded = {key: tl + [""] * (max_len - len(tl)) for key, tl in token_lists.items()}

    votes: list[WordVote] = []

    for pos in range(max_len):
        # Collect all non-empty tokens at this position
        position_tokens = [padded[key][pos] for key in padded if padded[key][pos]]

        if not position_tokens:
            continue

        # Group tokens by fuzzy similarity
        groups: dict[str, list[str]] = {}
        for token in position_tokens:
            placed = False
            for canonical in list(groups.keys()):
                score = _normalised_similarity(token, canonical)
                if score >= SIMILARITY_THRESHOLD:
                    groups[canonical].append(token)
                    placed = True
                    break
            if not placed:
                groups[token] = [token]

        # Find the largest group (most agreed-upon token)
        canonical, members = max(groups.items(), key=lambda kv: len(kv[1]))
        count = len(members)
        confidence = count / n_transcripts

        # Assign confidence tier
        if confidence >= CONSENSUS_THRESHOLD:
            tier = "HIGH"
        elif count >= 2:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        votes.append(
            WordVote(
                word=canonical,
                count=count,
                total=n_transcripts,
                confidence=round(confidence, 3),
                tier=tier,
                variants=list(set(members)),
            )
        )

    logger.info(
        "Alignment complete: %d words | HIGH=%d MEDIUM=%d LOW=%d",
        len(votes),
        sum(1 for v in votes if v.tier == "HIGH"),
        sum(1 for v in votes if v.tier == "MEDIUM"),
        sum(1 for v in votes if v.tier == "LOW"),
    )
    return votes


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def align_transcripts(
    transcripts: dict[str, str],
    strategy: str | None = None,
) -> list[WordVote]:
    """
    Align multiple transcript strings and produce a word-level vote sequence.

    Dispatches to the appropriate alignment algorithm based on the *strategy*
    parameter or the global ``ALIGNMENT_STRATEGY`` config value.

    Parameters
    ----------
    transcripts : dict[str, str]
        Mapping of variant key → plain-text transcript string.
    strategy : str, optional
        Override the alignment strategy: ``"sequence"`` (Needleman-Wunsch) or
        ``"positional"`` (legacy index-based). Defaults to config value.

    Returns
    -------
    list[WordVote]
        Ordered list of WordVote objects representing the consensus sequence.
    """
    strategy = (strategy or ALIGNMENT_STRATEGY).strip().lower()

    if strategy == "sequence":
        from consensus_merger.sequence_alignment import align_transcripts_sequence

        return align_transcripts_sequence(transcripts)

    return _align_positional(transcripts)
