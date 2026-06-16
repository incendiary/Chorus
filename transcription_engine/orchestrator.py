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
    CONSENSUS_MODEL_LABELS,
    CONSENSUS_MODELS,
    TRANSCRIPTS_DIR,
    TRANSCRIPTION_PARALLELISM,
    VARIANT_LABELS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
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


def _configured_models(model_names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return the configured model list, always with at least one entry."""
    models = tuple(model_names) if model_names is not None else tuple(CONSENSUS_MODELS)
    if models:
        return models
    return (WHISPER_MODEL,)


def _build_result_key(
    model_name: str,
    variant_key: str,
    primary_model: str,
) -> str:
    """Build transcript key, preserving legacy keys for the primary model."""
    if model_name == primary_model:
        return variant_key
    return f"{model_name}__{variant_key}"


def _build_transcription_jobs(
    variant_paths: dict[str, Path],
    model_names: tuple[str, ...] | None = None,
) -> list[tuple[str, str, Path, str, str]]:
    """Expand configured models × variants into concrete transcription jobs."""
    jobs: list[tuple[str, str, Path, str, str]] = []
    models = _configured_models(model_names=model_names)
    primary_model = models[0]

    for model_name in models:
        model_label = CONSENSUS_MODEL_LABELS.get(model_name, f"Whisper {model_name}")
        for variant_key, audio_path in variant_paths.items():
            variant_label = VARIANT_LABELS.get(variant_key, variant_key)
            result_key = _build_result_key(model_name, variant_key, primary_model)
            if model_name == primary_model:
                label = variant_label
            else:
                label = f"{model_label} — {variant_label}"
            jobs.append((result_key, variant_key, audio_path, model_name, label))

    return jobs


def _transcribe_one(
    result_key: str,
    variant_key: str,
    audio_path: Path,
    stem: str,
    language: str | None,
    device: str,
    model_name: str,
    label: str,
    transcripts_dir: Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Run one transcription unit and return key/label/result."""
    result = transcribe(
        audio_path=audio_path,
        variant_key=variant_key,
        stem=stem,
        language=language,
        device=device,
        model_name=model_name,
        transcripts_dir=transcripts_dir,
    )
    _write_txt_companion(
        stem=stem,
        key=result_key,
        label=label,
        result=result,
        transcripts_dir=transcripts_dir,
    )
    return result_key, label, result


def run_transcription_pass(
    variant_paths: dict[str, Path],
    stem: str,
    language: str | None = None,
    progress_callback=None,
    transcripts_dir: Path | None = None,
    model_names: tuple[str, ...] | None = None,
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
    model_names : tuple[str, ...], optional
        Ordered list of Whisper model names to run. When omitted, uses
        ``config.CONSENSUS_MODELS``.

    Returns
    -------
    dict[str, dict]
        Mapping of transcript key → Whisper result dict (includes ``text``,
        ``segments``, ``language``, ``variant``, ``model``). Primary model
        keys keep legacy names (for example ``original``); secondary model
        keys are namespaced as ``<model>__<variant>``.
    """
    transcripts: dict[str, dict[str, Any]] = {}
    jobs = _build_transcription_jobs(variant_paths, model_names=model_names)
    total = len(jobs)
    workers = _resolve_parallelism(total)
    device_pool = _build_device_pool(workers)

    if workers <= 1:
        for step, (result_key, variant_key, audio_path, model_name, label) in enumerate(
            jobs,
            start=1,
        ):
            logger.info("[%d/%d] Transcribing: %s", step, total, label)
            result_key, label, result = _transcribe_one(
                result_key=result_key,
                variant_key=variant_key,
                audio_path=audio_path,
                stem=stem,
                language=language,
                device=device_pool[0],
                model_name=model_name,
                label=label,
                transcripts_dir=transcripts_dir,
            )
            transcripts[result_key] = result
            if progress_callback:
                progress_callback(step, total, label)
    else:
        logger.info(
            "Running transcription in parallel with %d workers on %s",
            workers,
            ", ".join(device_pool),
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, (
                result_key,
                variant_key,
                audio_path,
                model_name,
                label,
            ) in enumerate(jobs):
                logger.info("[queued %d/%d] Transcribing: %s", idx + 1, total, label)
                device = device_pool[idx % len(device_pool)]
                future = executor.submit(
                    _transcribe_one,
                    result_key,
                    variant_key,
                    audio_path,
                    stem,
                    language,
                    device,
                    model_name,
                    label,
                    transcripts_dir,
                )
                futures[future] = result_key

            for step, future in enumerate(as_completed(futures), start=1):
                result_key, label, result = future.result()
                transcripts[result_key] = result
                logger.info("[%d/%d] Completed: %s", step, total, label)
                if progress_callback:
                    progress_callback(step, total, label)

    logger.info("All %d transcription variants complete.", total)
    return transcripts


def load_transcripts_from_disk(
    stem: str,
    transcripts_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Re-load previously generated transcript JSON files from disk.

    Useful for resuming a pipeline run without re-running Whisper.

    Parameters
    ----------
    stem : str
        Base filename stem.
    transcripts_dir : Path, optional
        Directory to read transcripts from.  When omitted, falls back to
        ``config.TRANSCRIPTS_DIR`` (global default).

    Returns
    -------
    dict[str, dict]
        Mapping of variant key → transcript dict.
    """
    import json

    read_dir = transcripts_dir if transcripts_dir is not None else TRANSCRIPTS_DIR
    transcripts: dict[str, dict[str, Any]] = {}
    models = _configured_models()
    primary_model = models[0]
    for model_name in models:
        for variant_key in VARIANT_LABELS:
            key = _build_result_key(model_name, variant_key, primary_model)
            path = read_dir / f"{stem}_{key}.json"
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    transcripts[key] = json.load(fh)
                logger.info("Loaded cached transcript: %s", path.name)
            else:
                logger.warning("Transcript not found on disk: %s", path)

    return transcripts
