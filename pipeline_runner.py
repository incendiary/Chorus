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
import sys
import time
from collections.abc import Callable
from pathlib import Path

from audio_processor.pipeline import process_audio
from transcription_engine.orchestrator import run_transcription_pass
from utils import sanitise_stem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chorus.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    audio_path: str | Path,
    language: str | None = None,
    enable_nlp: bool = False,
    enable_diarisation: bool = False,
    alignment_strategy: str | None = None,
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
    enable_nlp : bool
        If True, run spaCy NLP reconstruction on LOW-confidence tokens.
    enable_diarisation : bool
        If True, run pyannote speaker diarisation.
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
    stem = sanitise_stem(audio_path.stem, fallback="audio")
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

    from consensus_merger.merger import merge_transcripts_with_votes

    if enable_nlp:
        _progress("Running spaCy NLP reconstruction…", 0.90)

    consensus_path, votes = merge_transcripts_with_votes(
        transcripts=transcripts,
        stem=stem,
        strategy=alignment_strategy,
        enable_nlp=enable_nlp,
    )
    _progress("Consensus document generated.", 0.95)

    # ── AI Context Pack (always generated) ───────────────────────────────────
    _progress("Generating AI context pack…", 0.96)
    from export_engine.ai_context import generate_ai_context_pack

    ai_context_path = generate_ai_context_pack(
        votes=votes,
        stem=stem,
        transcripts_meta=transcripts,
        elapsed_seconds=round(time.perf_counter() - t_start, 2),
        alignment_strategy=alignment_strategy,
    )

    # ── Optional: Speaker Diarisation ────────────────────────────────────────
    diarised_path = None
    speaker_labels: list[str] = []
    if enable_diarisation:
        _progress("Running speaker diarisation…", 0.97)
        try:
            from diarisation.diariser import (
                diarise,
                get_unique_speakers,
                label_transcript,
                load_speaker_names,
                render_diarised_md,
            )

            speaker_segs = diarise(variant_paths["original"])
            labelled = label_transcript(speaker_segs, transcripts["original"])
            speaker_labels = get_unique_speakers(labelled)

            # Load any previously saved speaker names for this stem
            speaker_map = load_speaker_names(stem)

            diarised_path = render_diarised_md(labelled, stem, speaker_map=speaker_map)

            # Update AI context pack with speaker information
            ai_context_path = generate_ai_context_pack(
                votes=votes,
                stem=stem,
                transcripts_meta=transcripts,
                elapsed_seconds=round(time.perf_counter() - t_start, 2),
                alignment_strategy=alignment_strategy,
                speaker_labels=speaker_labels,
                speaker_names=speaker_map,
            )
        except Exception as exc:
            logger.warning("Diarisation failed: %s", exc)

    _progress("Pipeline complete.", 1.00)

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info("Pipeline complete in %.2f s → %s", elapsed, consensus_path)

    return {
        "variant_paths": variant_paths,
        "transcripts": transcripts,
        "consensus_path": consensus_path,
        "ai_context_path": ai_context_path,
        "diarised_path": diarised_path,
        "speaker_labels": speaker_labels,
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
