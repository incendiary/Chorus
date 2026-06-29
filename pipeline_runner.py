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
from config import ensure_output_dirs
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
    consensus_models: tuple[str, ...] | None = None,
    enable_nlp: bool = False,
    enable_llm: bool = False,
    ollama_model: str | None = None,
    enable_diarisation: bool = False,
    alignment_strategy: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    output_dir: Path | None = None,
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
    consensus_models : tuple[str, ...], optional
        Ordered Whisper model names to include in consensus transcription.
        If ``None``, configured defaults are used.
    enable_nlp : bool
        If True, run spaCy NLP reconstruction on LOW-confidence tokens.
    enable_llm : bool
        If True, run local LLM reconstruction (Ollama) on LOW-confidence tokens.
    enable_diarisation : bool
        If True, run pyannote speaker diarisation.
    progress_callback : callable, optional
        Called as ``progress_callback(stage_label, fraction_complete)``
        at key milestones.  Fraction is in [0.0, 1.0].
    output_dir : Path, optional
        Root directory for all pipeline outputs.  When provided, sub-dirs
        ``variants/``, ``transcripts/``, and ``consensus/`` are created
        inside it.  Defaults to the global ``config.OUTPUTS_DIR`` layout.

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
    ensure_output_dirs()
    stem = sanitise_stem(audio_path.stem, fallback="audio")
    source_filename = (
        audio_path.name
    )  # Original filename with extension for traceability
    t_start = time.perf_counter()

    # Derive per-stage output dirs from optional override
    variants_dir: Path | None = None
    transcripts_dir: Path | None = None
    consensus_dir: Path | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        variants_dir = output_dir / "variants"
        transcripts_dir = output_dir / "transcripts"
        consensus_dir = output_dir / "consensus"

    def _progress(label: str, frac: float) -> None:
        logger.info("Progress [%.0f%%] — %s", frac * 100, label)
        if progress_callback:
            progress_callback(label, frac)

    # ── Stage 1: Audio Processing ────────────────────────────────────────────
    _progress("Applying audio cleaning filters…", 0.05)
    variant_paths = process_audio(audio_path, output_dir=variants_dir)
    _progress("Audio variants ready.", 0.25)

    # ── Stage 2: Transcription ───────────────────────────────────────────────
    _progress("Loading Whisper model…", 0.30)

    def _transcription_progress(step: int, total: int, label: str) -> None:
        frac = 0.30 + (step / total) * 0.50
        _progress(f"Transcribing: {label}", frac)

    # Segment-level callback: fires once per decoded segment, providing
    # finer-grained progress updates between variant-level steps.
    def _segment_progress(seg_idx: int, seg_total: int, _text: str) -> None:
        if seg_total < 1:
            return
        # We don't know total variants at this point; interpolate within current step range
        seg_frac = (seg_idx + 1) / seg_total
        _progress(
            f"Decoding segments… ({seg_idx + 1}/{seg_total})", seg_frac * 0.02 + 0.30
        )

    transcripts = run_transcription_pass(
        variant_paths=variant_paths,
        stem=stem,
        language=language,
        progress_callback=_transcription_progress,
        transcripts_dir=transcripts_dir,
        model_names=consensus_models,
        segment_callback=_segment_progress,
    )
    _progress("All transcription variants complete.", 0.80)

    # Release Whisper models from memory before LLM reconstruction so that
    # Ollama (a separate process) can claim the freed RAM/unified memory.
    # On 32 GB Apple Silicon, large (~3 GB) + neural-chat:13b (~28 GB) = 31 GB;
    # clearing first keeps peak to whichever is larger, not the sum.
    if enable_llm:
        from transcription_engine.whisper_engine import clear_model_cache

        clear_model_cache()
        _progress("Whisper model released; handing off to LLM…", 0.82)

    # ── Stage 3: Consensus Merge ─────────────────────────────────────────────
    _progress("Running consensus analysis…", 0.85)

    from consensus_merger.merger import merge_transcripts_with_votes

    if enable_nlp:
        _progress("Running spaCy NLP reconstruction…", 0.90)
    if enable_llm:
        _progress("Running LLM reconstruction…", 0.92)

    consensus_path, votes = merge_transcripts_with_votes(
        transcripts=transcripts,
        stem=stem,
        strategy=alignment_strategy,
        enable_nlp=enable_nlp,
        enable_llm=enable_llm,
        ollama_model=ollama_model,
        consensus_dir=consensus_dir,
        source_filename=source_filename,
    )
    _progress("Consensus document generated.", 0.95)

    # ── AI Context Pack (always generated) ───────────────────────────────────
    _progress("Generating AI context pack…", 0.96)
    from export_engine.ai_context import generate_ai_context_pack
    from export_engine.exporter import export_transcript_bundle

    ai_context_path = generate_ai_context_pack(
        votes=votes,
        stem=stem,
        transcripts_meta=transcripts,
        elapsed_seconds=round(time.perf_counter() - t_start, 2),
        alignment_strategy=alignment_strategy,
        source_filename=source_filename,
        output_dir=consensus_dir,
    )

    bundle_path = export_transcript_bundle(
        transcripts=transcripts,
        votes=votes,
        stem=stem,
        source_filename=source_filename,
        output_dir=consensus_dir,
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
            speaker_map = load_speaker_names(stem, output_dir=consensus_dir)

            diarised_path = render_diarised_md(
                labelled, stem, speaker_map=speaker_map, output_dir=consensus_dir
            )

            # Update AI context pack with speaker information
            ai_context_path = generate_ai_context_pack(
                votes=votes,
                stem=stem,
                transcripts_meta=transcripts,
                elapsed_seconds=round(time.perf_counter() - t_start, 2),
                alignment_strategy=alignment_strategy,
                speaker_labels=speaker_labels,
                source_filename=source_filename,
                output_dir=consensus_dir,
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
        "bundle_path": bundle_path,
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
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Root directory for pipeline outputs (variants/, transcripts/, consensus/). "
        "Defaults to the global outputs/ directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_pipeline(args.audio, language=args.language, output_dir=output_dir)
    print(f"\n✓ Consensus transcript: {results['consensus_path']}")
    print(f"  Completed in {results['elapsed_seconds']} s")
    sys.exit(0)
