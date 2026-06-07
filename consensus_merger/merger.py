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

from consensus_merger.alignment import align_transcripts
from consensus_merger.renderer import render_consensus

logger = logging.getLogger(__name__)


def merge_transcripts(
    transcripts: dict[str, dict[str, Any]],
    stem: str,
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
    if not transcripts:
        raise ValueError("No transcripts provided to merge.")

    # Extract plain-text bodies
    text_map: dict[str, str] = {
        key: result.get("text", "").strip() for key, result in transcripts.items()
    }

    non_empty = {k: v for k, v in text_map.items() if v}
    if not non_empty:
        raise ValueError("All transcripts are empty — nothing to merge.")

    logger.info("Merging %d transcript variants for stem '%s'.", len(non_empty), stem)

    # Align and vote
    votes = align_transcripts(non_empty)

    # Render to Markdown
    out_path = render_consensus(votes, stem, transcripts)

    return out_path
