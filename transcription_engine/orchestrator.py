"""
transcription_engine/orchestrator.py — Multi-variant transcription orchestrator.

Accepts the dictionary of audio variant paths produced by the audio processing
pipeline and runs Whisper transcription over each one.  Returns a mapping of
variant key → transcript dict for consumption by the consensus merger.

The orchestrator also writes a plain-text (.txt) summary alongside each JSON
transcript so that the outputs directory remains human-browsable without
requiring JSON parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import TRANSCRIPTS_DIR, VARIANT_LABELS
from transcription_engine.whisper_engine import transcribe

logger = logging.getLogger(__name__)


def run_transcription_pass(
    variant_paths: dict[str, Path],
    stem: str,
    language: str | None = None,
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """
    Transcribe every audio variant and return all results.

    Parameters
    ----------
    variant_paths : dict[str, Path]
        Mapping of variant key → WAV file path, as returned by
        ``audio_processor.pipeline.process_audio``.
    stem : str
        Base filename stem used for output naming.
    language : str, optional
        BCP-47 language code hint passed to Whisper.
    progress_callback : callable, optional
        If provided, called as ``progress_callback(step, total, label)``
        after each variant completes — useful for Streamlit progress bars.

    Returns
    -------
    dict[str, dict]
        Mapping of variant key → Whisper result dict (includes ``text``,
        ``segments``, ``language``, ``variant``, ``model``).
    """
    transcripts: dict[str, dict[str, Any]] = {}
    total = len(variant_paths)

    for step, (key, audio_path) in enumerate(variant_paths.items(), start=1):
        label = VARIANT_LABELS.get(key, key)
        logger.info("[%d/%d] Transcribing: %s", step, total, label)

        result = transcribe(
            audio_path=audio_path,
            variant_key=key,
            stem=stem,
            language=language,
        )
        transcripts[key] = result

        # Write a companion plain-text file for human inspection
        txt_path = TRANSCRIPTS_DIR / f"{stem}_{key}.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Chorus Transcript — {label}\n")
            fh.write(f"# Model : {result.get('model', 'unknown')}\n")
            fh.write(f"# Language detected: {result.get('language', 'unknown')}\n\n")
            fh.write(result.get("text", "").strip())
            fh.write("\n")

        if progress_callback:
            progress_callback(step, total, label)

    logger.info("All %d transcription variants complete.", total)
    return transcripts


def load_transcripts_from_disk(stem: str) -> dict[str, dict[str, Any]]:
    """
    Re-load previously generated transcript JSON files from TRANSCRIPTS_DIR.

    Useful for resuming a pipeline run without re-running Whisper.

    Parameters
    ----------
    stem : str
        Base filename stem.

    Returns
    -------
    dict[str, dict]
        Mapping of variant key → transcript dict.
    """
    import json

    transcripts: dict[str, dict[str, Any]] = {}
    for key in VARIANT_LABELS:
        path = TRANSCRIPTS_DIR / f"{stem}_{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                transcripts[key] = json.load(fh)
            logger.info("Loaded cached transcript: %s", path.name)
        else:
            logger.warning("Transcript not found on disk: %s", path)

    return transcripts
