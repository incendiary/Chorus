"""
consensus_merger/sequence_alignment.py — Needleman-Wunsch word-level alignment.

Implements a proper sequence alignment algorithm that handles insertions,
deletions, and substitutions between transcript variants.  Unlike the
positional (index-based) approach, this correctly aligns transcripts of
different lengths — essential when one Whisper variant hallucinates or
omits words relative to the others.

Algorithm
─────────
1. Choose the longest transcript as the reference sequence.
2. Align each other transcript to the reference using Needleman-Wunsch
   with a custom scoring function (Levenshtein-based similarity).
3. Build a multi-alignment matrix from the pairwise alignments.
4. At each aligned position, vote on the canonical word using the same
   fuzzy-grouping and confidence-tier logic as the positional approach.

Scoring
───────
- Match bonus:     +2 (exact match or similarity ≥ SIMILARITY_THRESHOLD)
- Mismatch penalty: -1
- Gap penalty:      -1 (insertion/deletion in one transcript)
"""

from __future__ import annotations

import logging

import numpy as np

from config import CONSENSUS_THRESHOLD, SIMILARITY_THRESHOLD
from consensus_merger.alignment import (
    WordVote,
    _normalised_similarity,
    _tokenise,
    _vote_for_tokens,
)

logger = logging.getLogger(__name__)

# Scoring parameters for the Needleman-Wunsch matrix
_MATCH_SCORE = 2
_MISMATCH_PENALTY = -1
_GAP_PENALTY = -1


# ─────────────────────────────────────────────────────────────────────────────
# Needleman-Wunsch pairwise alignment
# ─────────────────────────────────────────────────────────────────────────────


def _score_pair(a: str, b: str) -> int:
    """Score a pair of tokens: match, fuzzy match, or mismatch."""
    if a == b:
        return _MATCH_SCORE
    if _normalised_similarity(a, b) >= SIMILARITY_THRESHOLD:
        return _MATCH_SCORE
    return _MISMATCH_PENALTY


def _needleman_wunsch(
    seq_a: list[str],
    seq_b: list[str],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[str, str]]:
    """
    Align two token sequences using Needleman-Wunsch global alignment.

    Uses a banded approach for long sequences (band width = 2 * max expected
    indel length) to keep runtime manageable on transcripts > 500 words.

    Returns a list of aligned pairs: (token_a, token_b) where either may be
    "" (representing a gap/insertion).
    """
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 and m == 0:
        return []
    if n == 0:
        return [("", tok) for tok in seq_b]
    if m == 0:
        return [(tok, "") for tok in seq_a]

    # Fast path: identical sequences
    if seq_a == seq_b:
        return list(zip(seq_a, seq_b, strict=False))

    # Pre-compute score lookup: build set of exact matches for O(1) checks
    # For fuzzy matching, only call _normalised_similarity when no exact match
    def score_at(i: int, j: int) -> int:
        a_tok = seq_a[i]
        b_tok = seq_b[j]
        if a_tok == b_tok:
            return _MATCH_SCORE
        if _normalised_similarity(a_tok, b_tok) >= similarity_threshold:
            return _MATCH_SCORE
        return _MISMATCH_PENALTY

    # Use banded NW for long sequences to avoid O(n*m) blowup
    # Band width: allow up to 10% length difference + 50 words of drift
    band_width = max(50, int(0.1 * max(n, m)))
    use_band = n > 200 and m > 200

    # Build scoring matrix
    score_matrix = np.full((n + 1, m + 1), -99999, dtype=np.int32)
    score_matrix[0, 0] = 0
    for i in range(1, n + 1):
        score_matrix[i, 0] = i * _GAP_PENALTY
    for j in range(1, m + 1):
        score_matrix[0, j] = j * _GAP_PENALTY

    for i in range(1, n + 1):
        if use_band:
            # Diagonal band: j should be near i * m / n
            center_j = int(i * m / n)
            j_start = max(1, center_j - band_width)
            j_end = min(m + 1, center_j + band_width + 1)
        else:
            j_start = 1
            j_end = m + 1

        for j in range(j_start, j_end):
            match = score_matrix[i - 1, j - 1] + score_at(i - 1, j - 1)
            delete = score_matrix[i - 1, j] + _GAP_PENALTY
            insert = score_matrix[i, j - 1] + _GAP_PENALTY
            score_matrix[i, j] = max(match, delete, insert)

    # Traceback
    aligned_pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = score_at(i - 1, j - 1)
            if score_matrix[i, j] == score_matrix[i - 1, j - 1] + s:
                aligned_pairs.append((seq_a[i - 1], seq_b[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + _GAP_PENALTY:
            aligned_pairs.append((seq_a[i - 1], ""))
            i -= 1
        else:
            aligned_pairs.append(("", seq_b[j - 1]))
            j -= 1

    aligned_pairs.reverse()
    return aligned_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sequence alignment via reference
# ─────────────────────────────────────────────────────────────────────────────


def _build_multi_alignment(
    token_lists: dict[str, list[str]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, str]]:
    """
    Align all token sequences against the longest (reference) sequence.

    Returns a list of column dicts, where each column maps variant key →
    aligned token (or "" for gaps).
    """
    if not token_lists:
        return []

    # Choose the longest transcript with a stable key tie-breaker. A completion-
    # order-dependent reference can otherwise change the alignment on multi-GPU
    # runs where result insertion order is nondeterministic.
    ref_key = min(token_lists, key=lambda key: (-len(token_lists[key]), key))
    ref_tokens = token_lists[ref_key]

    if not ref_tokens:
        return []

    # Align each non-reference sequence to the reference
    pairwise_alignments: dict[str, list[tuple[str, str]]] = {}
    for key, tokens in token_lists.items():
        if key == ref_key:
            continue
        pairwise_alignments[key] = _needleman_wunsch(
            ref_tokens, tokens, similarity_threshold
        )

    if not pairwise_alignments:
        # Only one transcript — each word is its own column.
        return [{ref_key: token} for token in ref_tokens]

    # Columns are keyed by reference position rather than by list index, so
    # merging one variant can never shift the positions another variant has
    # already been written to.
    #
    # An earlier implementation inserted gap columns directly into a flat list
    # while walking each variant in turn. Every insertion shifted the columns
    # to its right, but the next variant's counter still tracked raw reference
    # positions, so each successive variant was written progressively further
    # out of place. On real four-variant output that scattered agreeing words
    # across separate columns and drove the HIGH-confidence rate down to 4.6 %
    # where the same data supports roughly 40 %.
    ref_columns: list[dict[str, str]] = [{ref_key: token} for token in ref_tokens]

    # Collect each complete insertion run by reference anchor. Zipping runs by
    # depth is incorrect when variants insert unequal prefixes, for example
    # ["um", "hello"] versus ["hello"]. Aligning each run separately keeps the
    # shared suffix in one voting column.
    insertion_runs: dict[int, dict[str, list[str]]] = {}

    for key, alignment in pairwise_alignments.items():
        ref_idx = 0
        for ref_tok, other_tok in alignment:
            if ref_tok:
                if ref_idx < len(ref_columns):
                    ref_columns[ref_idx][key] = other_tok
                ref_idx += 1
                continue
            # Insertion in this variant with no reference counterpart.
            insertion_runs.setdefault(ref_idx, {}).setdefault(key, []).append(other_tok)

    gap_columns = {
        anchor: _build_multi_alignment(runs, similarity_threshold)
        for anchor, runs in insertion_runs.items()
    }

    columns: list[dict[str, str]] = []
    for ref_idx, column in enumerate(ref_columns):
        columns.extend(gap_columns.get(ref_idx, ()))
        columns.append(column)
    # Insertions running past the final reference token.
    columns.extend(gap_columns.get(len(ref_columns), ()))

    return columns


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def align_transcripts_sequence(
    transcripts: dict[str, str],
    *,
    consensus_threshold: float | None = None,
    similarity_threshold: float | None = None,
) -> list[WordVote]:
    """
    Align multiple transcript strings using Needleman-Wunsch sequence alignment.

    This produces the same output format (list[WordVote]) as the positional
    algorithm, but correctly handles insertions and deletions between variants.

    Parameters
    ----------
    transcripts : dict[str, str]
        Mapping of variant key → plain-text transcript string.
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
    if consensus_threshold is None:
        consensus_threshold = CONSENSUS_THRESHOLD
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD

    if not transcripts:
        return []

    token_lists = {key: _tokenise(text) for key, text in transcripts.items()}
    n_transcripts = len(token_lists)

    # Filter out empty token lists
    token_lists = {k: v for k, v in token_lists.items() if v}
    if not token_lists:
        return []

    # Build multi-alignment
    columns = _build_multi_alignment(token_lists, similarity_threshold)

    votes: list[WordVote] = []

    for col in columns:
        # Collect all non-empty tokens at this aligned position
        position_tokens = [tok for tok in col.values() if tok]

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
        "Sequence alignment complete: %d words | HIGH=%d MEDIUM=%d LOW=%d",
        len(votes),
        sum(1 for v in votes if v.tier == "HIGH"),
        sum(1 for v in votes if v.tier == "MEDIUM"),
        sum(1 for v in votes if v.tier == "LOW"),
    )
    return votes
