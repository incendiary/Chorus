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
import numpy as np
import soundfile as sf

from audio_processor.filters import denoise_filter, dynamic_range_norm, high_pass_focus
from config import TARGET_SAMPLE_RATE, VARIANT_LABELS, VARIANTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────


def _decode_with_ffmpeg(input_path: Path) -> tuple[np.ndarray, int]:
    """Decode *input_path* through ffmpeg via ``pydub``.

    Covers the compressed containers that libsndfile cannot open — most
    importantly MP4/AAC (``.m4a``, the Apple Voice Memos format) — using the
    ffmpeg binary that is already a documented installation prerequisite.
    Returns a mono float32 signal and its native sample rate.
    """
    from pydub import AudioSegment

    segment = AudioSegment.from_file(str(input_path))
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels).mean(axis=1)
    samples /= float(1 << (8 * segment.sample_width - 1))
    return samples, segment.frame_rate


def _load_audio(input_path: Path) -> tuple[np.ndarray, int]:
    """Decode *input_path* to a mono, float32 signal at ``TARGET_SAMPLE_RATE``.

    Audio is decoded through ``soundfile`` (PySoundFile) first — the fast,
    native path for WAV/FLAC/MP3 — falling back to ffmpeg via ``pydub`` for
    formats libsndfile does not support (``.m4a`` and other MP4/AAC
    containers).  The signal is mixed down to mono and resampled to the
    target rate.

    Raises
    ------
    RuntimeError
        If neither decoder can read the file (for example, a corrupt or
        genuinely unsupported file).  The message names the offending file
        and suggests a remedy.
    """
    try:
        audio, native_sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        # libsndfile cannot open MP4/AAC containers (.m4a, .mp4, .aac);
        # retry through ffmpeg before declaring the file unreadable.
        try:
            audio, native_sr = _decode_with_ffmpeg(input_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to decode audio file: {input_path}. The file may be "
                "corrupt, or in a format neither libsndfile nor ffmpeg could "
                "read; please supply a valid audio file (WAV, FLAC, MP3, M4A, "
                "and other ffmpeg-supported formats are accepted)."
            ) from exc

    # Mix down to mono by averaging channels when the source is multi-channel.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to the target rate using librosa's non-deprecated resampler.
    if native_sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(audio, dtype="float32"), TARGET_SAMPLE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def process_audio(
    input_path: str | Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Run the full cleaning pipeline on *input_path*.

    Parameters
    ----------
    input_path : str | Path
        Path to the raw audio file (any format supported by librosa/ffmpeg).
    output_dir : Path, optional
        Directory to write variant WAV files into.  Defaults to
        ``config.VARIANTS_DIR`` when *None*.

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
    audio, sr = _load_audio(input_path)

    logger.info(
        "Loaded %.2f s @ %d Hz (%.1f MB)", len(audio) / sr, sr, audio.nbytes / 1e6
    )

    stem = input_path.stem
    variants_dir = output_dir if output_dir is not None else VARIANTS_DIR
    variants_dir.mkdir(parents=True, exist_ok=True)

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

        out_path = variants_dir / f"{stem}_{key}.wav"
        sf.write(str(out_path), processed, sr, subtype="PCM_16")
        output_paths[key] = out_path
        logger.info("Saved variant '%s' → %s", key, out_path)

        # Release processed array immediately after writing to disk
        del processed

    # Release source audio — all variants are now on disk
    del audio
    logger.info("Audio arrays released — variants persisted to disk.")

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
