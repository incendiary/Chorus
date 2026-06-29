"""
consensus_merger/merger.py — Top-level consensus merge entry point.

Provides the single public function ``merge_transcripts`` which orchestrates
the full consensus pipeline:

  1. Extracts plain-text bodies from each transcript result dict.
  2. Calls ``alignment.align_transcripts`` to produce word-level votes.
  3. Calls ``renderer.render_consensus`` to produce the annotated Markdown.
  4. Returns the path to the final consensus document.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from consensus_merger.alignment import WordVote, align_transcripts
from consensus_merger.renderer import render_consensus

logger = logging.getLogger(__name__)


def _extract_non_empty_text_map(
    transcripts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Extract non-empty text values from Whisper result payloads."""
    if not transcripts:
        raise ValueError("No transcripts provided to merge.")

    text_map: dict[str, str] = {
        key: result.get("text", "").strip() for key, result in transcripts.items()
    }

    non_empty = {k: v for k, v in text_map.items() if v}
    if not non_empty:
        raise ValueError("All transcripts are empty — nothing to merge.")

    return non_empty


def merge_transcripts_with_votes(
    transcripts: dict[str, dict[str, Any]],
    stem: str,
    strategy: str | None = None,
    enable_nlp: bool = False,
    enable_llm: bool = False,
    ollama_model: str | None = None,
    consensus_dir: Path | None = None,
    source_filename: str | None = None,
) -> tuple[Path, list[WordVote]]:
    """Run consensus alignment/render and return both output path and votes."""
    non_empty = _extract_non_empty_text_map(transcripts)

    logger.info("Merging %d transcript variants for stem '%s'.", len(non_empty), stem)

    votes = align_transcripts(non_empty, strategy=strategy)

    if enable_nlp:
        from reconstruction import reconstruct

        votes = reconstruct(votes, strategy="nlp")

    if enable_llm:
        from reconstruction import reconstruct

        votes = reconstruct(votes, strategy="llm", model=ollama_model)

    out_path = render_consensus(
        votes,
        stem,
        transcripts,
        consensus_dir=consensus_dir,
        source_filename=source_filename,
    )
    return out_path, votes


def merge_transcripts(
    transcripts: dict[str, dict[str, Any]],
    stem: str,
    strategy: str | None = None,
) -> Path:
    """
    Merge multiple Whisper transcript results into a single consensus document.

    Parameters
    ----------
    transcripts : dict[str, dict]
        Mapping of variant key → Whisper result dict.  Each dict must contain
        at minimum a ``"text"`` key with the plain-text transcript string.
    stem : str
        Base filename stem used for output naming.

    Returns
    -------
    Path
        Absolute path of the generated consensus ``.md`` file.

    Raises
    ------
    ValueError
        If *transcripts* is empty or contains no usable text.
    """
    out_path, _votes = merge_transcripts_with_votes(
        transcripts=transcripts,
        stem=stem,
        strategy=strategy,
        enable_nlp=False,
    )
    return out_path
