"""
audio_processor/pipeline.py — Audio cleaning pipeline orchestrator.

Loads a raw audio file, resamples it to the target sample rate, and
produces four output WAV files:
  - original   : unprocessed, resampled copy
  - highpass   : High-Pass Focus variant
  - normalised : Dynamic Range Normalisation variant
  - denoised   : Denoise Filter variant

Each variant is written to VARIANTS_DIR and the function returns a
mapping of variant label → output file path for downstream consumption
by the transcription engine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import soundfile as sf

from audio_processor.filters import denoise_filter, dynamic_range_norm, high_pass_focus
from config import TARGET_SAMPLE_RATE, VARIANT_LABELS, VARIANTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def process_audio(input_path: str | Path) -> dict[str, Path]:
    """
    Run the full cleaning pipeline on *input_path*.

    Parameters
    ----------
    input_path : str | Path
        Path to the raw audio file (any format supported by librosa/ffmpeg).

    Returns
    -------
    dict[str, Path]
        Mapping of variant key → absolute path of the exported WAV file.
        Keys: "original", "highpass", "normalised", "denoised".

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    RuntimeError
        If audio loading fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    logger.info("Loading audio: %s", input_path)
    audio, sr = librosa.load(str(input_path), sr=TARGET_SAMPLE_RATE, mono=True)
    logger.info("Loaded %.2f s @ %d Hz", len(audio) / sr, sr)

    stem = input_path.stem

    # Define the pipeline: label → (filter_fn | None)
    pipeline: dict[str, object] = {
        "original": None,  # no processing
        "highpass": high_pass_focus,
        "normalised": dynamic_range_norm,
        "denoised": denoise_filter,
    }

    output_paths: dict[str, Path] = {}

    for key, filter_fn in pipeline.items():
        if filter_fn is None:
            processed = audio.copy()
        else:
            logger.info("Applying filter: %s", VARIANT_LABELS[key])
            processed = filter_fn(audio, sr)

        out_path = VARIANTS_DIR / f"{stem}_{key}.wav"
        sf.write(str(out_path), processed, sr, subtype="PCM_16")
        output_paths[key] = out_path
        logger.info("Saved variant '%s' → %s", key, out_path)

    return output_paths


def get_audio_info(input_path: str | Path) -> dict:
    """
    Return basic metadata about an audio file without full processing.

    Parameters
    ----------
    input_path : str | Path
        Path to the audio file.

    Returns
    -------
    dict
        Keys: duration_seconds, sample_rate, channels, format.
    """
    input_path = Path(input_path)
    info = sf.info(str(input_path))
    return {
        "duration_seconds": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
    }
