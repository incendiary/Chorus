"""
consensus_merger/alignment.py — Word-level alignment and confidence scoring.

This module implements the core voting logic for the Chorus consensus engine.
Given N transcript strings, it:

  1. Tokenises each transcript into a normalised word sequence.
  2. Aligns the sequences using a majority-vote sliding window.
  3. Assigns a confidence weight to every word position based on how many
     transcripts agree on that token.
  4. Applies Levenshtein-based fuzzy similarity for near-matches that differ only
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
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from config import ALIGNMENT_STRATEGY, CONSENSUS_THRESHOLD, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

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
    # The engine needs only Levenshtein distance, not NLTK's tokenisers or
    # downloadable corpora. Keeping the small dynamic-programming routine here
    # makes imports deterministic and fully offline.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for row, left_char in enumerate(a, start=1):
        current = [row]
        for column, right_char in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    dist = previous[-1]
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


def _group_fuzzy_tokens(
    tokens: Sequence[str], similarity_threshold: float
) -> list[tuple[str, list[str]]]:
    """Return deterministic, order-invariant fuzzy-match components.

    Fuzzy similarity is not transitive. Greedily attaching a token to the
    first matching canonical therefore made the vote depend on transcript
    completion and dictionary insertion order. Treating matching forms as an
    undirected graph and returning its connected components gives the same
    result for every input ordering. The canonical form is the most frequently
    observed spelling, with lexical order as a stable tie-breaker.
    """
    counts = Counter(tokens)
    forms = sorted(counts)
    neighbours = {form: set() for form in forms}
    for index, left in enumerate(forms):
        for right in forms[index + 1 :]:
            if _normalised_similarity(left, right) >= similarity_threshold:
                neighbours[left].add(right)
                neighbours[right].add(left)

    groups: list[tuple[str, list[str]]] = []
    unseen = set(forms)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[str] = set()
        while stack:
            form = stack.pop()
            if form in component:
                continue
            component.add(form)
            unseen.discard(form)
            stack.extend(sorted(neighbours[form] - component, reverse=True))

        canonical = min(component, key=lambda form: (-counts[form], form))
        members = [form for form in sorted(component) for _ in range(counts[form])]
        groups.append((canonical, members))

    return sorted(groups, key=lambda group: group[0])


def _vote_for_tokens(
    position_tokens: Sequence[str],
    *,
    n_transcripts: int,
    consensus_threshold: float,
    similarity_threshold: float,
) -> WordVote:
    """Build one vote while retaining every observed form at the position."""
    groups = _group_fuzzy_tokens(position_tokens, similarity_threshold)
    canonical, members = min(
        groups,
        key=lambda group: (-len(group[1]), group[0]),
    )
    count = len(members)
    confidence = count / n_transcripts

    if confidence >= consensus_threshold:
        tier = "HIGH"
    elif count >= 2:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return WordVote(
        word=canonical,
        count=count,
        total=n_transcripts,
        confidence=round(confidence, 3),
        tier=tier,
        variants=sorted(set(position_tokens)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core alignment algorithm
# ─────────────────────────────────────────────────────────────────────────────


def _align_positional(
    transcripts: dict[str, str],
    *,
    consensus_threshold: float | None = None,
    similarity_threshold: float | None = None,
) -> list[WordVote]:
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
    if consensus_threshold is None:
        consensus_threshold = CONSENSUS_THRESHOLD
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD

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

        votes.append(
            _vote_for_tokens(
                position_tokens,
                n_transcripts=n_transcripts,
                consensus_threshold=consensus_threshold,
                similarity_threshold=similarity_threshold,
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
    *,
    consensus_threshold: float | None = None,
    similarity_threshold: float | None = None,
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
    consensus_threshold : float, optional
        Agreement fraction at or above which a word is tier HIGH.
        Defaults to ``config.CONSENSUS_THRESHOLD``.
    similarity_threshold : float, optional
        Normalised similarity at or above which two word forms count as the
        same word. Defaults to ``config.SIMILARITY_THRESHOLD``.

    Returns
    -------
    list[WordVote]
        Ordered list of WordVote objects representing the consensus sequence.
    """
    strategy = (strategy or ALIGNMENT_STRATEGY).strip().lower()

    if strategy == "sequence":
        from consensus_merger.sequence_alignment import align_transcripts_sequence

        return align_transcripts_sequence(
            transcripts,
            consensus_threshold=consensus_threshold,
            similarity_threshold=similarity_threshold,
        )

    return _align_positional(
        transcripts,
        consensus_threshold=consensus_threshold,
        similarity_threshold=similarity_threshold,
    )
