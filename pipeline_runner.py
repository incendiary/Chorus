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
from config import CONSENSUS_DIR, ensure_output_dirs
from transcription_engine.orchestrator import run_transcription_pass
from utils import sanitise_stem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chorus.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline event stages
# ─────────────────────────────────────────────────────────────────────────────

# Canonical stage order as they actually fire from the ``_progress`` call
# sites below. ``reconstruction`` and ``diarisation`` are optional and are
# only included for a given run when the corresponding feature is enabled.
_ALWAYS_ON_STAGES = ("cleaning", "loading_model", "transcribing", "consensus")


def active_stages(
    enable_nlp: bool = False,
    enable_llm: bool = False,
    enable_diarisation: bool = False,
) -> list[str]:
    """Return the ordered list of stages this run will pass through.

    Shared with ``ui/run_worker.py`` so stage numbering (``stage_index`` /
    ``stage_total``) is computed identically wherever it is needed, without
    duplicating the enable-flag logic.
    """
    stages = list(_ALWAYS_ON_STAGES)
    if enable_nlp or enable_llm:
        stages.append("reconstruction")
    stages.append("export")
    if enable_diarisation:
        stages.append("diarisation")
    stages.append("done")
    return stages


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
    consensus_threshold: float | None = None,
    similarity_threshold: float | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    event_callback: Callable[[dict], None] | None = None,
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
    event_callback : callable, optional
        Called as ``event_callback(event)`` at the same milestones as
        ``progress_callback``, with a structured dict:
        ``{"stage", "detail", "frac", "passes_done", "passes_total",
        "segment", "segments_total", "stage_index", "stage_total"}``.
        ``stage`` is one of the values returned by :func:`active_stages`
        for this run's enabled options. ``progress_callback``'s label text
        is unaffected by this kwarg — both are derived from the same
        underlying milestones.
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
          ``"best_guess_path"``— Path to the clean, markup-free best-guess ``.txt`` file
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

    stages = active_stages(
        enable_nlp=enable_nlp,
        enable_llm=enable_llm,
        enable_diarisation=enable_diarisation,
    )
    stage_total = len(stages)

    def _stage_index(stage: str) -> int:
        return stages.index(stage) + 1

    def _emit_event(
        frac: float,
        *,
        stage: str,
        detail: str | None = None,
        passes_done: int | None = None,
        passes_total: int | None = None,
        segment: int | None = None,
        segments_total: int | None = None,
    ) -> None:
        if event_callback:
            event_callback(
                {
                    "stage": stage,
                    "detail": detail,
                    "frac": frac,
                    "passes_done": passes_done,
                    "passes_total": passes_total,
                    "segment": segment,
                    "segments_total": segments_total,
                    "stage_index": _stage_index(stage),
                    "stage_total": stage_total,
                }
            )

    def _progress(
        label: str,
        frac: float,
        *,
        stage: str,
        detail: str | None = None,
        passes_done: int | None = None,
        passes_total: int | None = None,
        segment: int | None = None,
        segments_total: int | None = None,
    ) -> None:
        # Keeps progress_callback's (label, frac) call sequence byte-identical
        # to before event plumbing was added; event_callback is additive.
        logger.info("Progress [%.0f%%] — %s", frac * 100, label)
        if progress_callback:
            progress_callback(label, frac)
        _emit_event(
            frac,
            stage=stage,
            detail=detail,
            passes_done=passes_done,
            passes_total=passes_total,
            segment=segment,
            segments_total=segments_total,
        )

    # ── Stage 1: Audio Processing ────────────────────────────────────────────
    _progress("Applying audio cleaning filters…", 0.05, stage="cleaning")
    variant_paths = process_audio(audio_path, output_dir=variants_dir)
    _progress("Audio variants ready.", 0.25, stage="cleaning")

    # ── Stage 2: Transcription ───────────────────────────────────────────────
    _progress("Loading Whisper model…", 0.30, stage="loading_model")

    def _on_pass_start(step: int, total: int, label: str) -> None:
        # Event-only (no progress_callback call, to keep its label sequence
        # byte-identical to before event plumbing existed): fires before the
        # first segment of each pass, so the current-pass detail is available
        # even if a pass yields no segment ticks.
        _emit_event(
            0.30 + ((step - 1) / total) * 0.50,
            stage="transcribing",
            detail=label,
            passes_done=step - 1,
            passes_total=total,
        )

    def _transcription_progress(step: int, total: int, label: str) -> None:
        frac = 0.30 + (step / total) * 0.50
        _progress(
            f"Transcribing: {label}",
            frac,
            stage="transcribing",
            detail=label,
            passes_done=step,
            passes_total=total,
        )

    # Segment-level callback: fires once per decoded segment, providing
    # finer-grained progress updates between variant-level steps. Sequential
    # branch only — the orchestrator never invokes this from the parallel
    # branch, where a single segment counter would be incoherent.
    def _segment_progress(seg_idx: int, seg_total: int, _text: str, label: str) -> None:
        if seg_total < 1:
            return
        # We don't know total variants at this point; interpolate within current step range
        seg_frac = (seg_idx + 1) / seg_total
        _progress(
            f"Decoding segments… ({seg_idx + 1}/{seg_total})",
            seg_frac * 0.02 + 0.30,
            stage="transcribing",
            detail=label,
            segment=seg_idx + 1,
            segments_total=seg_total,
        )

    transcripts = run_transcription_pass(
        variant_paths=variant_paths,
        stem=stem,
        language=language,
        progress_callback=_transcription_progress,
        transcripts_dir=transcripts_dir,
        model_names=consensus_models,
        segment_callback=_segment_progress,
        on_pass_start=_on_pass_start,
    )
    _progress("All transcription variants complete.", 0.80, stage="transcribing")

    # Release Whisper models from memory before LLM reconstruction so that
    # Ollama (a separate process) can claim the freed RAM/unified memory.
    # On 32 GB Apple Silicon, large (~3 GB) + neural-chat:13b (~28 GB) = 31 GB;
    # clearing first keeps peak to whichever is larger, not the sum.
    if enable_llm:
        from transcription_engine.whisper_engine import unload_model

        unload_model()
        _progress(
            "Whisper model released; handing off to LLM…",
            0.82,
            stage="transcribing",
        )

    # ── Stage 3: Consensus Merge ─────────────────────────────────────────────
    _progress("Running consensus analysis…", 0.85, stage="consensus")

    from consensus_merger.merger import merge_transcripts_with_votes

    if enable_nlp:
        from reconstruction import probe_spacy_model

        _nlp_ok, _nlp_reason = probe_spacy_model()
        if not _nlp_ok:
            logger.warning(
                "NLP reconstruction requested but unavailable: %s "
                "LOW-confidence tokens will be left unreconstructed for this run.",
                _nlp_reason,
            )
        _progress(
            "Running spaCy NLP reconstruction…",
            0.90,
            stage="reconstruction",
            detail="spaCy NLP",
        )
    if enable_llm:
        _progress(
            "Running LLM reconstruction…",
            0.92,
            stage="reconstruction",
            detail=ollama_model or "Ollama LLM",
        )

    consensus_path, votes = merge_transcripts_with_votes(
        transcripts=transcripts,
        stem=stem,
        strategy=alignment_strategy,
        enable_nlp=enable_nlp,
        enable_llm=enable_llm,
        ollama_model=ollama_model,
        consensus_dir=consensus_dir,
        source_filename=source_filename,
        consensus_threshold=consensus_threshold,
        similarity_threshold=similarity_threshold,
    )
    _progress("Consensus document generated.", 0.95, stage="consensus")

    # ── AI Context Pack (always generated) ───────────────────────────────────
    _progress("Generating AI context pack…", 0.96, stage="export")
    from export_engine.ai_context import generate_ai_context_pack
    from export_engine.exporter import (
        BUNDLE_SCHEMA_VERSION,
        export_best_guess,
        export_transcript_bundle,
    )

    ai_context_path = generate_ai_context_pack(
        votes=votes,
        stem=stem,
        transcripts_meta=transcripts,
        elapsed_seconds=round(time.perf_counter() - t_start, 2),
        alignment_strategy=alignment_strategy,
        source_filename=source_filename,
        output_dir=consensus_dir,
        consensus_threshold=consensus_threshold,
        similarity_threshold=similarity_threshold,
        schema_version=BUNDLE_SCHEMA_VERSION,
    )

    bundle_path = export_transcript_bundle(
        transcripts=transcripts,
        votes=votes,
        stem=stem,
        source_filename=source_filename,
        output_dir=consensus_dir,
    )

    best_guess_path = export_best_guess(
        consensus_path,
        stem,
        output_dir=consensus_dir,
    )

    # ── Copy parsing guide to output directory ───────────────────────────────
    # Write it wherever the consensus outputs land: the per-run directory when
    # output_dir was given, else the global CONSENSUS_DIR — so every run
    # (UI, CLI, and batch) yields the parsing guide alongside its outputs.
    parsing_guide_src = Path(__file__).resolve().parent / "docs" / "CHORUS_FOR_LLMS.md"
    parsing_guide_dir = consensus_dir if consensus_dir is not None else CONSENSUS_DIR
    if parsing_guide_src.exists():
        parsing_guide_dir.mkdir(parents=True, exist_ok=True)
        parsing_guide_dst = parsing_guide_dir / "HOW_TO_PARSE_CHORUS_OUTPUT.md"
        parsing_guide_dst.write_text(
            parsing_guide_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
        logger.info("Parsing guide written → %s", parsing_guide_dst)

    # ── Optional: Speaker Diarisation ────────────────────────────────────────
    diarised_path = None
    speaker_labels: list[str] = []
    if enable_diarisation:
        _progress("Running speaker diarisation…", 0.97, stage="diarisation")
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
                consensus_threshold=consensus_threshold,
                similarity_threshold=similarity_threshold,
                schema_version=BUNDLE_SCHEMA_VERSION,
            )
        except Exception as exc:
            logger.warning("Diarisation failed: %s", exc)

    _progress("Pipeline complete.", 1.00, stage="done")

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info("Pipeline complete in %.2f s → %s", elapsed, consensus_path)

    return {
        "variant_paths": variant_paths,
        "transcripts": transcripts,
        "consensus_path": consensus_path,
        "ai_context_path": ai_context_path,
        "bundle_path": bundle_path,
        "best_guess_path": best_guess_path,
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
