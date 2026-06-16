"""
export_engine/ai_context.py — AI Context Pack generator.

Produces a structured Markdown document designed to be fed directly to an LLM
alongside a question about the transcription.  It contains:

  1. **Methodology overview** — how Chorus generated the transcript
  2. **Processing metadata** — model, language, device, alignment strategy
  3. **Confidence statistics** — HIGH/MEDIUM/LOW word distribution
  4. **Clean transcript** — the "most likely" text without markup
  5. **Uncertainty annotations** — every uncertain word with its variants,
     position, and confidence score
  6. **Speaker information** — if diarisation was enabled

This gives an LLM all the context it needs to understand the provenance and
reliability of each word in the transcript.

Usage
─────
    from export_engine.ai_context import generate_ai_context_pack

    path = generate_ai_context_pack(
        votes=votes,
        stem="recording",
        transcripts_meta=transcripts,
        processing_meta={...},
    )
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import (
    ALIGNMENT_STRATEGY,
    CONSENSUS_DIR,
    NOISE_FLOOR_MODE,
    VARIANT_LABELS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _methodology_section() -> str:
    """Return the standard methodology explanation block."""
    return """## Methodology

Chorus is a **multi-pass consensus transcription engine** that improves
transcription accuracy through redundancy and voting:

1. **Audio Variant Generation** — The original audio is processed through
   multiple cleaning filters (high-pass, normalisation, spectral denoising)
   to produce several variants of the same recording.

2. **Independent Transcription** — Each audio variant is independently
   transcribed by OpenAI Whisper, producing multiple transcripts of the
   same source material with different processing characteristics.

3. **Word-Level Consensus Voting** — All transcripts are aligned
   word-by-word and a confidence vote is computed for each position:
   - **HIGH (≥ 75% agreement):** The word appears in most or all variants.
     Very likely correct.
   - **MEDIUM (50% agreement):** The word appears in roughly half the
     variants. Probably correct but worth a second look.
   - **LOW (25% agreement):** The word appears in only one variant.
     Likely an artefact, mishearing, or hallucination.

4. **Consensus Rendering** — The final transcript uses the most-voted word
   at each position, annotated with confidence tiers.

This approach significantly reduces single-pass transcription errors,
particularly for noisy audio, accented speech, or domain-specific terminology.
"""


def _build_uncertainty_table(votes: list) -> str:
    """Build a Markdown table of all uncertain (non-HIGH) words."""
    uncertain = [(idx, v) for idx, v in enumerate(votes) if v.tier != "HIGH"]

    if not uncertain:
        return (
            "## Uncertainty Annotations\n\n"
            "✅ **No uncertain words** — all words achieved HIGH confidence "
            "(≥ 75% agreement across variants).\n"
        )

    lines = [
        "## Uncertainty Annotations",
        "",
        "Words below did NOT achieve full consensus. The `Variants` column shows",
        "all word forms observed across the different transcription passes.",
        "",
        "| Position | Word (chosen) | Confidence | Tier | Variants |",
        "|----------|---------------|-----------|------|----------|",
    ]

    for idx, v in uncertain:
        variants_str = " / ".join(sorted(set(v.variants))) if v.variants else "—"
        pct = f"{v.confidence * 100:.0f}%"
        lines.append(f"| {idx + 1} | {v.word} | {pct} | {v.tier} | {variants_str} |")

    lines.append("")
    lines.append(
        f"> **Total uncertain words:** {len(uncertain)} / {len(votes)} "
        f"({len(uncertain) / max(len(votes), 1) * 100:.1f}%)"
    )
    return "\n".join(lines)


def _build_clean_transcript(votes: list) -> str:
    """Build the clean transcript (most-likely words only)."""
    words = [v.word for v in votes]
    # Break into ~80-word paragraphs for readability
    para_size = 80
    paragraphs = []
    for i in range(0, len(words), para_size):
        paragraphs.append(" ".join(words[i : i + para_size]))
    return "\n\n".join(paragraphs)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_ai_context_pack(
    votes: list,
    stem: str,
    transcripts_meta: dict[str, dict],
    elapsed_seconds: float = 0.0,
    alignment_strategy: str | None = None,
    speaker_labels: list[str] | None = None,
    speaker_names: dict[str, str] | None = None,
) -> Path:
    """
    Generate an AI-ready context pack for the given transcription.

    The output is a structured Markdown file that an LLM can consume to
    understand the provenance, reliability, and content of the transcript.

    Parameters
    ----------
    votes : list[WordVote]
        Ordered word-vote sequence from alignment.
    stem : str
        Base filename stem.
    transcripts_meta : dict[str, dict]
        Mapping of variant key → transcript result dict.
    elapsed_seconds : float
        Pipeline processing time.
    alignment_strategy : str, optional
        Which alignment algorithm was used.
    speaker_labels : list[str], optional
        Ordered list of detected speaker labels.
    speaker_names : dict[str, str], optional
        Mapping of speaker label → human-readable name.

    Returns
    -------
    Path
        Path to the written ``{stem}_ai_context.md`` file.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    strategy = alignment_strategy or ALIGNMENT_STRATEGY

    # Stats
    total = len(votes) or 1
    n_high = sum(1 for v in votes if v.tier == "HIGH")
    n_med = sum(1 for v in votes if v.tier == "MEDIUM")
    n_low = sum(1 for v in votes if v.tier == "LOW")

    # First transcript metadata for model/language
    first_meta = next(iter(transcripts_meta.values()), {})
    model = first_meta.get("model", WHISPER_MODEL)
    language = first_meta.get("language", "auto-detected")

    sections: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    sections.append(f"""# AI Context Pack — `{stem}`

> This document is machine-generated by **Chorus Engine** and designed to be
> consumed by an AI/LLM alongside questions about this transcription.
> It provides full provenance, confidence data, and uncertainty annotations.

**Generated:** {now}
**Processing time:** {elapsed_seconds:.1f} s
""")

    # ── Methodology ──────────────────────────────────────────────────────────
    sections.append(_methodology_section())

    # ── Processing Metadata ──────────────────────────────────────────────────
    sections.append(f"""## Processing Configuration

| Parameter | Value |
|-----------|-------|
| Whisper model | `{model}` |
| Detected language | `{language}` |
| Compute device | `{WHISPER_DEVICE}` |
| Alignment strategy | `{strategy}` |
| Noise floor mode | `{NOISE_FLOOR_MODE}` |
| Transcription variants | {len(transcripts_meta)} |
| Total consensus words | {len(votes)} |
""")

    # ── Variant details ──────────────────────────────────────────────────────
    variant_lines = [
        "## Variant Details",
        "",
        "| Key | Label | Word Count |",
        "|-----|-------|-----------|",
    ]
    for key, meta in transcripts_meta.items():
        label = VARIANT_LABELS.get(key, key)
        wc = len(meta.get("text", "").split())
        variant_lines.append(f"| `{key}` | {label} | {wc} |")
    variant_lines.append("")
    sections.append("\n".join(variant_lines))

    # ── Confidence Statistics ────────────────────────────────────────────────
    sections.append(f"""## Confidence Statistics

| Tier | Count | Percentage | Interpretation |
|------|------:|----------:|----------------|
| HIGH | {n_high} | {n_high / total * 100:.1f}% | ≥ 75% variant agreement — very likely correct |
| MEDIUM | {n_med} | {n_med / total * 100:.1f}% | 50% agreement — probably correct, worth reviewing |
| LOW | {n_low} | {n_low / total * 100:.1f}% | 25% agreement — single variant only, possibly an error |

**Overall reliability score:** {n_high / total * 100:.1f}% of words are HIGH confidence.
""")

    # ── Speaker Information ──────────────────────────────────────────────────
    if speaker_labels:
        speaker_names = speaker_names or {}
        speaker_lines = [
            "## Speaker Information",
            "",
            "Speaker diarisation was enabled for this transcription.",
            "",
            "| Label | Name |",
            "|-------|------|",
        ]
        for spk in speaker_labels:
            name = speaker_names.get(spk, "_(unnamed)_")
            speaker_lines.append(f"| `{spk}` | {name} |")
        speaker_lines.append("")
        sections.append("\n".join(speaker_lines))

    # ── Clean Transcript ─────────────────────────────────────────────────────
    sections.append("## Clean Transcript (Most Likely)\n")
    sections.append(
        "The following is the plain-text transcript using the most-voted word "
        "at each position. No confidence markup is applied.\n"
    )
    sections.append(_build_clean_transcript(votes))
    sections.append("")

    # ── Uncertainty Annotations ──────────────────────────────────────────────
    sections.append(_build_uncertainty_table(votes))

    # ── Usage Guidance ───────────────────────────────────────────────────────
    sections.append("""
## Usage Guidance for AI Systems

When using this transcript:

1. **Trust HIGH-confidence words** — they have strong multi-variant agreement.
2. **Verify MEDIUM-confidence words** — check surrounding context for sense.
3. **Treat LOW-confidence words with scepticism** — they may be hallucinations,
   filler words, or misheard audio. Consider the variants listed above.
4. **If asked about accuracy**, reference the confidence statistics above.
5. **If quoting the transcript**, prefer HIGH-confidence passages.
6. **For critical applications** (legal, medical), flag that this is an
   automated transcription and recommend human verification of uncertain sections.

---

*Generated by Chorus Engine — AI Context Pack Module*
""")

    # ── Write file ───────────────────────────────────────────────────────────
    out_path = CONSENSUS_DIR / f"{stem}_ai_context.md"
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("AI context pack written → %s", out_path)
    return out_path
