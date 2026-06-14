"""
transcription_engine/whisper_engine.py — Whisper model wrapper.

Provides a thin, reusable wrapper around OpenAI's `whisper` Python package
for offline, local transcription.  The model is loaded once and cached as a
module-level singleton to avoid repeated disk I/O across multiple calls.

Model selection is controlled via config.WHISPER_MODEL (default: "base").
The "base" model (~145 MB) provides an excellent trade-off between speed
and word-error rate for English speech on CPU hardware.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import whisper

from config import TRANSCRIPTS_DIR, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

logger = logging.getLogger(__name__)

# Module-level model cache — loaded on first call to transcribe()
_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    """Load (or return cached) Whisper model, falling back to CPU on device errors."""
    global _model
    if _model is None:
        logger.info(
            "Loading Whisper model '%s' on device '%s'…", WHISPER_MODEL, WHISPER_DEVICE
        )
        try:
            _model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
        except (RuntimeError, Exception) as exc:  # noqa: BLE001
            if WHISPER_DEVICE != "cpu":
                logger.warning(
                    "Failed to load model on '%s' (%s) — falling back to CPU.",
                    WHISPER_DEVICE,
                    exc,
                )
                _model = whisper.load_model(WHISPER_MODEL, device="cpu")
            else:
                raise
        logger.info("Whisper model loaded.")
    return _model


def transcribe(
    audio_path: str | Path,
    variant_key: str,
    stem: str,
    language: str | None = None,
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

    model = _get_model()
    lang = language or WHISPER_LANGUAGE

    logger.info("Transcribing variant '%s': %s", variant_key, audio_path.name)

    decode_options: dict[str, Any] = {}
    if lang:
        decode_options["language"] = lang

    # Always enable word-level timestamps for richer export options
    decode_options["word_timestamps"] = True

    result = model.transcribe(str(audio_path), **decode_options)

    # Augment result with metadata
    result["variant"] = variant_key
    result["model"] = WHISPER_MODEL

    # Persist to TRANSCRIPTS_DIR/<stem>_<variant>.json
    out_path = TRANSCRIPTS_DIR / f"{stem}_{variant_key}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    logger.info("Transcript saved → %s", out_path)
    return result


def unload_model() -> None:
    """Release the cached model from memory (useful for testing)."""
    global _model
    _model = None
    logger.info("Whisper model unloaded from cache.")
