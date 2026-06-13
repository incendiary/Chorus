"""
audio_processor/filters.py — Three distinct audio-cleaning configurations.

Each filter function accepts a NumPy audio array (float32, mono) and the
sample rate, and returns a cleaned array of the same shape.  All processing
is performed in the time/frequency domain using librosa and scipy.

Cleaning Configurations
───────────────────────
1. high_pass_focus      — Emphasises vocal frequency ranges (≥ 80 Hz) and
                          strips low-end rumble via a Butterworth high-pass filter.
2. dynamic_range_norm   — Flattens audio spikes and boosts quiet segments through
                          peak normalisation followed by RMS-based loudness levelling.
3. denoise_filter       — Estimates the noise floor from a silent reference window
                          and subtracts it from the full spectrogram
                          (spectral subtraction).
"""

from __future__ import annotations

import librosa
import numpy as np
import scipy.signal as signal

from config import (
    HIGH_PASS_CUTOFF_HZ,
    NOISE_FLOOR_MODE,
    NOISE_REDUCTION_PROP,
    NORMALISATION_TARGET_DBFS,
    TARGET_SAMPLE_RATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Ensure the array is float32 and clipped to [-1, 1]."""
    audio = audio.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert a dBFS value to a linear amplitude scalar."""
    return 10 ** (dbfs / 20.0)


# ─────────────────────────────────────────────────────────────────────────────
# Filter 1 — High-Pass Focus
# ─────────────────────────────────────────────────────────────────────────────


def high_pass_focus(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Apply a 5th-order Butterworth high-pass filter at HIGH_PASS_CUTOFF_HZ.

    This configuration strips sub-vocal low-frequency rumble (HVAC noise,
    microphone handling noise, etc.) while preserving the fundamental vocal
    range (80 Hz – 8 kHz) in full fidelity.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal, float32, normalised to [-1, 1].
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        High-pass filtered audio, float32.
    """
    audio = _to_float32(audio)
    nyquist = sr / 2.0
    cutoff_norm = HIGH_PASS_CUTOFF_HZ / nyquist

    # Clamp to a valid range for the filter design
    cutoff_norm = np.clip(cutoff_norm, 1e-6, 1.0 - 1e-6)

    b, a = signal.butter(5, cutoff_norm, btype="high", analog=False)
    filtered = signal.filtfilt(b, a, audio).astype(np.float32)
    return _to_float32(filtered)


# ─────────────────────────────────────────────────────────────────────────────
# Filter 2 — Dynamic Range Normalisation
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_range_norm(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Flatten audio spikes and boost quiet segments.

    The algorithm proceeds in two stages:
      1. Peak normalisation — scales the entire signal so the loudest sample
         reaches 0 dBFS, eliminating clipping artefacts.
      2. RMS levelling — computes the RMS energy of the peak-normalised signal
         and applies a gain correction to bring it to NORMALISATION_TARGET_DBFS,
         ensuring consistent perceived loudness across variants.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal, float32, normalised to [-1, 1].
    sr : int
        Sample rate in Hz (unused here but kept for API consistency).

    Returns
    -------
    np.ndarray
        Dynamically normalised audio, float32.
    """
    audio = _to_float32(audio)

    # Stage 1: Peak normalisation
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak

    # Stage 2: RMS levelling
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        target_linear = _dbfs_to_linear(NORMALISATION_TARGET_DBFS)
        gain = target_linear / rms
        audio = audio * gain

    return _to_float32(audio)


# ─────────────────────────────────────────────────────────────────────────────
# Filter 3 — Denoise Filter (Spectral Subtraction)
# ─────────────────────────────────────────────────────────────────────────────


def _find_silence_window_vad(
    audio: np.ndarray, sr: int, frame_ms: int = 30, min_silence_ms: int = 200
) -> np.ndarray:
    """
    Find the quietest segment of audio using energy-based VAD.

    Splits the audio into short frames and identifies the longest contiguous
    run of low-energy frames. Falls back to the first 0.5 s if no silence is
    detected (e.g., the entire recording is speech).

    Parameters
    ----------
    audio : np.ndarray
        Full audio signal.
    sr : int
        Sample rate.
    frame_ms : int
        Frame length in milliseconds for energy analysis.
    min_silence_ms : int
        Minimum silence duration to consider (ms).

    Returns
    -------
    np.ndarray
        The audio segment identified as noise/silence.
    """
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len

    if n_frames < 2:
        # Audio too short for VAD — use entire signal as noise estimate
        return audio

    # Compute RMS energy per frame
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))

    # Threshold: frames below 20% of median RMS are considered silent
    median_rms = np.median(rms_per_frame)
    if median_rms < 1e-9:
        return audio[: int(0.5 * sr)]  # near-silent file

    silence_threshold = 0.20 * median_rms
    is_silent = rms_per_frame < silence_threshold

    # Find the longest contiguous silence run
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i, silent in enumerate(is_silent):
        if silent:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_len = 0

    # Check final run
    if current_len > best_len:
        best_start = current_start
        best_len = current_len

    min_frames = int(min_silence_ms / frame_ms)
    if best_len >= min_frames:
        start_sample = best_start * frame_len
        end_sample = (best_start + best_len) * frame_len
        return audio[start_sample:end_sample]

    # No sufficiently long silence found — fall back to first 0.5 s
    return audio[: int(0.5 * sr)]


def denoise_filter(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Reduce background noise via power-spectrum subtraction.

    The noise profile is estimated from either:
      - A VAD-detected silence segment (NOISE_FLOOR_MODE="vad", default), or
      - The first 0.5 seconds (NOISE_FLOOR_MODE="fixed", legacy behaviour).

    The mean power spectrum of the reference window is then subtracted —
    scaled by NOISE_REDUCTION_PROP — from every STFT frame of the full signal.
    Phase is preserved and the inverse STFT reconstructs the cleaned waveform.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal, float32, normalised to [-1, 1].
    sr : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Noise-reduced audio, float32.
    """
    audio = _to_float32(audio)

    n_fft = 2048
    hop_len = 512

    # Compute STFT of the full signal
    stft_full = librosa.stft(audio, n_fft=n_fft, hop_length=hop_len)
    magnitude, phase = np.abs(stft_full), np.angle(stft_full)

    # Estimate noise floor
    if NOISE_FLOOR_MODE == "vad":
        noise_ref = _find_silence_window_vad(audio, sr)
    else:
        # Fixed mode: first 0.5 s
        ref_samples = int(0.5 * sr)
        ref_samples = max(ref_samples, n_fft)
        noise_ref = audio[:ref_samples]

    # Ensure noise reference is long enough for at least one STFT frame
    if len(noise_ref) < n_fft:
        noise_ref = np.pad(noise_ref, (0, n_fft - len(noise_ref)))

    stft_noise = librosa.stft(noise_ref, n_fft=n_fft, hop_length=hop_len)
    noise_profile = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

    # Spectral subtraction with half-wave rectification (no negative magnitudes)
    cleaned_magnitude = magnitude - NOISE_REDUCTION_PROP * noise_profile
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

    # Reconstruct waveform from cleaned magnitude and original phase
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_len, length=len(audio))

    return _to_float32(cleaned_audio)
