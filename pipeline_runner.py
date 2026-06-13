"""
pipeline_runner.py — End-to-end Chorus pipeline coordinator.

This module exposes ``run_pipeline``, the single entry point that wires the
three processing stages together:

  Stage 1 — Audio Processing   : clean and export audio variants
  Stage 2 — Transcription      : run Whisper over each variant
  Stage 3 — Consensus Merge    : vote, weight, and render the final document

It can be invoked directly from the command line for headless operation or
called programmatically by the Streamlit UI.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from audio_processor.pipeline import process_audio
from transcription_engine.orchestrator import run_transcription_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chorus.pipeline")

# Safe filename pattern — only alphanumeric, hyphens, and underscores retained
_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitise_stem(raw: str) -> str:
    """Sanitise a filename stem to safe filesystem characters."""
    sanitised = _SAFE_STEM_RE.sub("_", raw).strip("_")
    return sanitised or "audio"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    audio_path: str | Path,
    language: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Path]:
    """
    Execute the full Chorus pipeline on a single audio file.

    Parameters
    ----------
    audio_path : str | Path
        Path to the raw input audio file.
    language : str, optional
        BCP-47 language code hint for Whisper (e.g., ``"en"``).
        If ``None``, Whisper auto-detects the language.
    progress_callback : callable, optional
        Called as ``progress_callback(stage_label, fraction_complete)``
        at key milestones.  Fraction is in [0.0, 1.0].

    Returns
    -------
    dict
        Keys:
          ``"variant_paths"``  — dict of variant key → WAV path
          ``"transcripts"``    — dict of variant key → Whisper result dict
          ``"consensus_path"`` — Path to the final consensus ``.md`` file
          ``"elapsed_seconds"``— total wall-clock time

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    """
    audio_path = Path(audio_path)
    stem = _sanitise_stem(audio_path.stem)
    t_start = time.perf_counter()

    def _progress(label: str, frac: float) -> None:
        logger.info("Progress [%.0f%%] — %s", frac * 100, label)
        if progress_callback:
            progress_callback(label, frac)

    # ── Stage 1: Audio Processing ────────────────────────────────────────────
    _progress("Applying audio cleaning filters…", 0.05)
    variant_paths = process_audio(audio_path)
    _progress("Audio variants ready.", 0.25)

    # ── Stage 2: Transcription ───────────────────────────────────────────────
    _progress("Loading Whisper model…", 0.30)

    def _transcription_progress(step: int, total: int, label: str) -> None:
        frac = 0.30 + (step / total) * 0.50
        _progress(f"Transcribing: {label}", frac)

    transcripts = run_transcription_pass(
        variant_paths=variant_paths,
        stem=stem,
        language=language,
        progress_callback=_transcription_progress,
    )
    _progress("All transcription variants complete.", 0.80)

    # ── Stage 3: Consensus Merge ─────────────────────────────────────────────
    _progress("Running consensus analysis…", 0.85)

    # Extract plain-text bodies
    text_map = {
        k: v.get("text", "").strip()
        for k, v in transcripts.items()
        if v.get("text", "").strip()
    }
    if not text_map:
        raise ValueError("All transcripts are empty.")

    from consensus_merger.alignment import align_transcripts
    from consensus_merger.renderer import render_consensus

    votes = align_transcripts(text_map)

    # ── Optional: NLP Reconstruction ─────────────────────────────────────────
    # If the user enables NLP reconstruction via environment variable
    import os

    if os.environ.get("ENABLE_NLP_RECONSTRUCTION", "false").lower() == "true":
        _progress("Running spaCy NLP reconstruction…", 0.90)
        from nlp_reconstructor.reconstructor import reconstruct_low_tokens

        votes = reconstruct_low_tokens(votes)

    consensus_path = render_consensus(votes, stem, transcripts)
    _progress("Consensus document generated.", 0.95)

    # ── Optional: Speaker Diarisation ────────────────────────────────────────
    diarised_path = None
    if os.environ.get("ENABLE_DIARISATION", "false").lower() == "true":
        _progress("Running speaker diarisation…", 0.97)
        try:
            from diarisation.diariser import (
                diarise,
                label_transcript,
                render_diarised_md,
            )

            speaker_segs = diarise(variant_paths["original"])
            labelled = label_transcript(speaker_segs, transcripts["original"])
            diarised_path = render_diarised_md(labelled, stem)
        except Exception as e:
            logger.warning(f"Diarisation failed: {e}")

    _progress("Pipeline complete.", 1.00)

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info("Pipeline complete in %.2f s → %s", elapsed, consensus_path)

    return {
        "variant_paths": variant_paths,
        "transcripts": transcripts,
        "consensus_path": consensus_path,
        "diarised_path": diarised_path,
        "elapsed_seconds": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="chorus",
        description="Chorus — Multi-pass consensus audio transcription engine.",
    )
    parser.add_argument("audio", help="Path to the input audio file.")
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="BCP-47 language code hint for Whisper (e.g., 'en', 'fr'). "
        "Omit for auto-detection.",
    )
    args = parser.parse_args()

    results = run_pipeline(args.audio, language=args.language)
    print(f"\n✓ Consensus transcript: {results['consensus_path']}")
    print(f"  Completed in {results['elapsed_seconds']} s")
    sys.exit(0)
