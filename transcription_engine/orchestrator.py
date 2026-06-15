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

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import (
    TRANSCRIPTS_DIR,
    TRANSCRIPTION_PARALLELISM,
    VARIANT_LABELS,
    WHISPER_DEVICE,
)
from transcription_engine.whisper_engine import transcribe

logger = logging.getLogger(__name__)


def _get_cuda_device_count() -> int:
    """Return available CUDA device count (0 if unavailable)."""
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def _resolve_parallelism(total_variants: int) -> int:
    """Resolve effective parallel worker count from config and environment."""
    if total_variants <= 1:
        return 1

    raw = str(TRANSCRIPTION_PARALLELISM).strip().lower()
    if raw and raw != "auto":
        try:
            configured = int(raw)
        except ValueError:
            logger.warning(
                "Invalid TRANSCRIPTION_PARALLELISM='%s'; falling back to auto.",
                TRANSCRIPTION_PARALLELISM,
            )
        else:
            return max(1, min(total_variants, configured))

    if WHISPER_DEVICE == "mps":
        return 1

    if WHISPER_DEVICE.startswith("cuda"):
        gpu_count = _get_cuda_device_count()
        if gpu_count > 1:
            return min(total_variants, gpu_count)
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(total_variants, min(4, cpu_count)))


def _build_device_pool(parallelism: int) -> list[str]:
    """Return device assignments used by workers."""
    if parallelism <= 1:
        return [WHISPER_DEVICE]

    if WHISPER_DEVICE.startswith("cuda"):
        gpu_count = _get_cuda_device_count()
        if gpu_count > 1:
            return [f"cuda:{idx}" for idx in range(min(parallelism, gpu_count))]

    return [WHISPER_DEVICE]


def _write_txt_companion(
    stem: str,
    key: str,
    label: str,
    result: dict[str, Any],
    transcripts_dir: Path | None = None,
) -> None:
    """Write human-readable transcript companion file."""
    out_dir = transcripts_dir if transcripts_dir is not None else TRANSCRIPTS_DIR
    txt_path = out_dir / f"{stem}_{key}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Chorus Transcript — {label}\n")
        fh.write(f"# Model : {result.get('model', 'unknown')}\n")
        fh.write(f"# Language detected: {result.get('language', 'unknown')}\n")
        fh.write(f"# Device: {result.get('device', 'unknown')}\n\n")
        fh.write(result.get("text", "").strip())
        fh.write("\n")


def _transcribe_one(
    key: str,
    audio_path: Path,
    stem: str,
    language: str | None,
    device: str,
    transcripts_dir: Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Run one transcription unit and return key/label/result."""
    label = VARIANT_LABELS.get(key, key)
    result = transcribe(
        audio_path=audio_path,
        variant_key=key,
        stem=stem,
        language=language,
        device=device,
        transcripts_dir=transcripts_dir,
    )
    _write_txt_companion(
        stem=stem,
        key=key,
        label=label,
        result=result,
        transcripts_dir=transcripts_dir,
    )
    return key, label, result


def run_transcription_pass(
    variant_paths: dict[str, Path],
    stem: str,
    language: str | None = None,
    progress_callback=None,
    transcripts_dir: Path | None = None,
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
    workers = _resolve_parallelism(total)
    device_pool = _build_device_pool(workers)

    if workers <= 1:
        for step, (key, audio_path) in enumerate(variant_paths.items(), start=1):
            label = VARIANT_LABELS.get(key, key)
            logger.info("[%d/%d] Transcribing: %s", step, total, label)
            _, label, result = _transcribe_one(
                key=key,
                audio_path=audio_path,
                stem=stem,
                language=language,
                device=device_pool[0],
                transcripts_dir=transcripts_dir,
            )
            transcripts[key] = result
            if progress_callback:
                progress_callback(step, total, label)
    else:
        logger.info(
            "Running transcription in parallel with %d workers on %s",
            workers,
            ", ".join(device_pool),
        )

        items = list(variant_paths.items())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, (key, audio_path) in enumerate(items):
                label = VARIANT_LABELS.get(key, key)
                logger.info("[queued %d/%d] Transcribing: %s", idx + 1, total, label)
                device = device_pool[idx % len(device_pool)]
                future = executor.submit(
                    _transcribe_one,
                    key,
                    audio_path,
                    stem,
                    language,
                    device,
                    transcripts_dir,
                )
                futures[future] = key

            for step, future in enumerate(as_completed(futures), start=1):
                key, label, result = future.result()
                transcripts[key] = result
                logger.info("[%d/%d] Completed: %s", step, total, label)
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
