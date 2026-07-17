"""
transcription_engine/whisper_engine.py — Whisper model wrapper.

Provides a thin, reusable wrapper around OpenAI's `whisper` Python package
for offline, local transcription. Models are cached per (model size, device)
so parallel transcription can target different compute backends (for example
cuda:0, cuda:1) without repeated load overhead.

Model selection is controlled via config.WHISPER_MODEL (default: "base").
The "base" model (~145 MB) provides an excellent trade-off between speed
and word-error rate for English speech on CPU hardware.

Segment-level progress
──────────────────────
``transcribe()`` accepts an optional *segment_callback* that is invoked after
each segment is decoded:

    def on_segment(segment_index: int, total_segments: int, text: str) -> None: ...

Because OpenAI Whisper's Python API is synchronous and does not stream, the
callback fires during a post-transcription pass over ``result["segments"]``.
This allows callers to update progress bars incrementally once transcription
completes, without requiring lower-level model patching.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import whisper

from config import TRANSCRIPTS_DIR, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

logger = logging.getLogger(__name__)

# Module-level model cache, keyed by (model size, device).
_models: dict[tuple[str, str], whisper.Whisper] = {}

# The MPS float64 fallback fires once per pass on Apple Silicon; warn the
# user once per process and keep subsequent occurrences at INFO level.
_mps_float64_warned = False
_model_lock = threading.Lock()


def _get_model(
    device: str | None = None,
    model_name: str | None = None,
) -> tuple[whisper.Whisper, str, str]:
    """Load (or return cached) Whisper model on the requested device."""
    requested_device = device or WHISPER_DEVICE
    selected_model = model_name or WHISPER_MODEL
    cache_key = (selected_model, requested_device)

    with _model_lock:
        if cache_key in _models:
            return _models[cache_key], requested_device, selected_model

        logger.info(
            "Loading Whisper model '%s' on device '%s'…",
            selected_model,
            requested_device,
        )
        try:
            model = whisper.load_model(selected_model, device=requested_device)
            _models[cache_key] = model
            logger.info(
                "Whisper model '%s' loaded on %s.",
                selected_model,
                requested_device,
            )
            return model, requested_device, selected_model
        except (RuntimeError, Exception) as exc:  # noqa: BLE001
            if requested_device != "cpu":
                logger.warning(
                    "Failed to load model on '%s' (%s) — falling back to CPU.",
                    requested_device,
                    exc,
                )
                cpu_key = (selected_model, "cpu")
                if cpu_key not in _models:
                    _models[cpu_key] = whisper.load_model(selected_model, device="cpu")
                return _models[cpu_key], "cpu", selected_model
            raise


def transcribe(
    audio_path: str | Path,
    variant_key: str,
    stem: str,
    language: str | None = None,
    device: str | None = None,
    model_name: str | None = None,
    transcripts_dir: Path | None = None,
    segment_callback: Any | None = None,
) -> dict[str, Any]:
    """
    Transcribe a single audio file and persist the result as JSON.

    The JSON transcript contains:
      - ``text``     : full plain-text transcript
      - ``segments`` : list of timed segment dicts (start, end, text)
      - ``language`` : detected or specified language code
      - ``variant``  : the cleaning variant label
      - ``model``    : Whisper model name used

    Parameters
    ----------
    audio_path : str | Path
        Path to the WAV (or any ffmpeg-supported) audio file.
    variant_key : str
        Short key identifying the cleaning variant (e.g., "highpass").
    stem : str
        Base filename stem used for output naming.
    language : str, optional
        BCP-47 language code hint.  If None, Whisper auto-detects.
    device : str, optional
        Device override (e.g. ``"cuda:0"``). If None, ``config.WHISPER_DEVICE``
        is used.
    model_name : str, optional
        Whisper model override (e.g. ``"small"``). If None,
        ``config.WHISPER_MODEL`` is used.
    segment_callback : callable, optional
        If provided, called as
        ``segment_callback(segment_index, total_segments, text)``
        for each segment after transcription completes.  Because the
        OpenAI Whisper API is synchronous, this fires in a post-transcription
        pass rather than during decoding.

    Returns
    -------
    dict
        The full Whisper result dict, augmented with ``variant`` and ``model``.

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio variant not found: {audio_path}")

    model, active_device, active_model = _get_model(
        device=device,
        model_name=model_name,
    )
    lang = language or WHISPER_LANGUAGE

    logger.info("Transcribing variant '%s': %s", variant_key, audio_path.name)

    decode_options: dict[str, Any] = {}
    if lang:
        decode_options["language"] = lang

    # Always enable word-level timestamps for richer export options
    decode_options["word_timestamps"] = True

    try:
        result = model.transcribe(str(audio_path), **decode_options)
    except TypeError as exc:
        # Whisper's word-timestamp DTW alignment calls .double() (float64), which
        # MPS does not support. Fall back to CPU and retry with the same options.
        if "float64" not in str(exc) and "MPS" not in str(exc):
            raise
        global _mps_float64_warned
        if not _mps_float64_warned:
            _mps_float64_warned = True
            logger.warning(
                "MPS does not support float64 word-timestamp alignment; affected "
                "passes will retry on CPU. This is expected on Apple Silicon and "
                "only logged once per run — later occurrences log at INFO level."
            )
        else:
            logger.info(
                "MPS float64 fallback — retrying pass on CPU (expected on Apple "
                "Silicon)."
            )
        cpu_model, _, _ = _get_model(model_name=active_model, device="cpu")
        result = cpu_model.transcribe(str(audio_path), **decode_options)

    # Augment result with metadata
    result["variant"] = variant_key
    result["model"] = active_model
    result["device"] = active_device

    # Persist to transcripts_dir/<stem>_<variant>.json
    out_dir = transcripts_dir if transcripts_dir is not None else TRANSCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{variant_key}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    logger.info("Transcript saved → %s", out_path)

    # Post-transcription segment progress callback
    if segment_callback is not None:
        segments = result.get("segments", [])
        total = len(segments)
        for idx, seg in enumerate(segments):
            segment_callback(idx, total, seg.get("text", "").strip())

    return result


def unload_model() -> None:
    """Release all cached models from memory (useful for testing)."""
    with _model_lock:
        _models.clear()
    logger.info("Whisper model cache cleared.")
