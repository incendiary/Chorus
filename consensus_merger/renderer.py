"""
consensus_merger/renderer.py — Consensus Markdown renderer.

Converts the list of WordVote objects produced by the alignment module into
a richly annotated Markdown document.  The rendering conventions are:

  HIGH confidence   → plain text (no decoration)
  MEDIUM confidence → `==highlighted==` (rendered as yellow highlight in most
                      Markdown viewers that support extended syntax; falls back
                      to plain text in standard renderers)
  LOW confidence    → **~~struck-through bold~~** with an inline annotation
                      showing the observed variants and confidence score

A summary statistics table is prepended to the document, and a legend is
appended to assist human reviewers.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from config import CONSENSUS_DIR, VARIANT_LABELS
from consensus_merger.alignment import WordVote

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────


def _render_word(vote: WordVote) -> str:
    """Return the Markdown-decorated string for a single WordVote."""
    if vote.tier == "HIGH":
        return vote.word

    if vote.tier == "MEDIUM":
        # Extended Markdown highlight syntax (supported by Obsidian, Typora, etc.)
        return f"=={vote.word}=="

    # LOW tier — bold + strikethrough with annotation
    variants_str = (
        " / ".join(sorted(set(vote.variants))) if vote.variants else vote.word
    )
    pct = int(vote.confidence * 100)
    return f"**~~{vote.word}~~**[^{pct}%: {variants_str}]"


def _build_stats(votes: list[WordVote]) -> dict[str, int]:
    """Compute summary statistics from the vote list."""
    return {
        "total_words": len(votes),
        "high": sum(1 for v in votes if v.tier == "HIGH"),
        "medium": sum(1 for v in votes if v.tier == "MEDIUM"),
        "low": sum(1 for v in votes if v.tier == "LOW"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def render_consensus(
    votes: list[WordVote],
    stem: str,
    transcripts_meta: dict[str, dict],
    consensus_dir: Path | None = None,
    source_filename: str | None = None,
) -> Path:
    """
    Render the consensus Markdown document and write it to consensus_dir.

    Parameters
    ----------
    votes : list[WordVote]
        Ordered word-vote sequence from ``alignment.align_transcripts``.
    stem : str
        Base filename stem for the output file.
    transcripts_meta : dict[str, dict]
        Mapping of variant key → transcript result dict (used for metadata
        such as detected language and model name).
    consensus_dir : Path, optional
        Directory to write the consensus ``.md`` file into.  Defaults to
        ``config.CONSENSUS_DIR`` when *None*.

    Returns
    -------
    Path
        Absolute path of the written ``.md`` file.
    """
    stats = _build_stats(votes)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ── Header ──────────────────────────────────────────────────────────────
    lines: list[str] = [
        "# Chorus — Consensus Transcript",
        "",
        f"> **Generated:** {now}  ",
    ]
    if source_filename:
        lines.append(f"> **Source file:** `{source_filename}`  ")
    lines.extend([
        f"> **Source stem:** `{stem}`  ",
        f"> **Whisper model:** `{next(iter(transcripts_meta.values())).get('model', 'unknown')}`  ",  # noqa: E501
        f"> **Language detected:** `{next(iter(transcripts_meta.values())).get('language', 'unknown')}`",  # noqa: E501
        "",
    ])

    # ── Variant summary table ────────────────────────────────────────────────
    lines += [
        "## Transcription Variants",
        "",
        "| Variant Key | Label | Words |",
        "|-------------|-------|------:|",
    ]
    for key, meta in transcripts_meta.items():
        label = VARIANT_LABELS.get(key, key)
        word_count = len(meta.get("text", "").split())
        lines.append(f"| `{key}` | {label} | {word_count} |")
    lines.append("")

    # ── Confidence statistics ────────────────────────────────────────────────
    total = stats["total_words"] or 1
    lines += [
        "## Confidence Statistics",
        "",
        "| Tier | Count | Percentage | Meaning |",
        "|------|------:|-----------:|---------|",
        f"| HIGH   | {stats['high']}   | {stats['high']/total*100:.1f}% | Present in ≥ 75 % of transcripts — kept as-is |",  # noqa: E501
        f"| MEDIUM | {stats['medium']} | {stats['medium']/total*100:.1f}% | Present in 2 transcripts — highlighted for review |",  # noqa: E501
        f"| LOW    | {stats['low']}    | {stats['low']/total*100:.1f}% | Present in only 1 transcript — flagged for removal |",  # noqa: E501
        "",
    ]

    # ── Consensus body ───────────────────────────────────────────────────────
    lines += [
        "## Consensus Transcript",
        "",
    ]

    # Group words into paragraphs of ~80 tokens for readability
    para_size = 80
    word_chunks = [votes[i : i + para_size] for i in range(0, len(votes), para_size)]
    for chunk in word_chunks:
        rendered_words = [_render_word(v) for v in chunk]
        lines.append(" ".join(rendered_words))
        lines.append("")

    # ── Legend ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Highlighting Legend",
        "",
        "| Rendering | Confidence Tier | Recommended Action |",
        "|-----------|----------------|--------------------|",
        "| Plain text | **HIGH** (≥ 75 %) | Accept — high agreement across all variants |",  # noqa: E501
        "| `==highlighted==` | **MEDIUM** (50 %) | Review — present in 2 of 4 transcripts |",  # noqa: E501
        "| **~~struck bold~~** | **LOW** (25 %) | Flag — single-transcript word; likely artefact |",  # noqa: E501
        "",
        "> **Note:** Percentages above are relative to the total number of transcription variants "  # noqa: E501
        "(default: 4).  Thresholds are configurable via `config.py`.",
        "",
    ]

    # ── Write file ───────────────────────────────────────────────────────────
    out_dir = consensus_dir if consensus_dir is not None else CONSENSUS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_consensus.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Consensus document written → %s", out_path)
    return out_path
