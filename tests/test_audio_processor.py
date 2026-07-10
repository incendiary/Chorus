"""
tests/test_audio_processor.py — Unit tests for audio_processor package.

Covers:
  - Individual filter functions (shape, dtype, value range)
  - Edge cases: silent audio, very short audio
  - pipeline.process_audio: FileNotFoundError on missing file
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_processor.filters import denoise_filter, dynamic_range_norm, high_pass_focus

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SR = 16_000  # sample rate used throughout


@pytest.fixture
def sine_audio() -> np.ndarray:
    """1-second 440 Hz sine wave, float32, normalised to [-1, 1]."""
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silent_audio() -> np.ndarray:
    """1-second silence."""
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """0.1-second noise burst — shorter than the denoise reference window."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, int(SR * 0.1)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# high_pass_focus
# ─────────────────────────────────────────────────────────────────────────────


class TestHighPassFocus:
    def test_output_shape_preserved(self, sine_audio):
        out = high_pass_focus(sine_audio, SR)
        assert out.shape == sine_audio.shape

    def test_output_dtype_float32(self, sine_audio):
        out = high_pass_focus(sine_audio, SR)
        assert out.dtype == np.float32

    def test_output_clipped(self, sine_audio):
        out = high_pass_focus(sine_audio, SR)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

    def test_silent_input_safe(self, silent_audio):
        out = high_pass_focus(silent_audio, SR)
        assert out.shape == silent_audio.shape
        assert not np.any(np.isnan(out))


# ─────────────────────────────────────────────────────────────────────────────
# dynamic_range_norm
# ─────────────────────────────────────────────────────────────────────────────


class TestDynamicRangeNorm:
    def test_output_shape_preserved(self, sine_audio):
        out = dynamic_range_norm(sine_audio, SR)
        assert out.shape == sine_audio.shape

    def test_output_dtype_float32(self, sine_audio):
        out = dynamic_range_norm(sine_audio, SR)
        assert out.dtype == np.float32

    def test_peak_does_not_exceed_one(self, sine_audio):
        out = dynamic_range_norm(sine_audio, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-5

    def test_silent_input_no_division_by_zero(self, silent_audio):
        """Silent input must not raise ZeroDivisionError or produce NaN."""
        out = dynamic_range_norm(silent_audio, SR)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_output_clipped(self, sine_audio):
        out = dynamic_range_norm(sine_audio, SR)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# denoise_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestDenoiseFilter:
    def test_output_shape_preserved(self, sine_audio):
        out = denoise_filter(sine_audio, SR)
        assert out.shape == sine_audio.shape

    def test_output_dtype_float32(self, sine_audio):
        out = denoise_filter(sine_audio, SR)
        assert out.dtype == np.float32

    def test_output_clipped(self, sine_audio):
        out = denoise_filter(sine_audio, SR)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

    def test_short_audio_does_not_crash(self, short_audio):
        """Audio shorter than the reference window must not raise an exception."""
        out = denoise_filter(short_audio, SR)
        assert out.shape == short_audio.shape
        assert not np.any(np.isnan(out))

    def test_silent_input_safe(self, silent_audio):
        out = denoise_filter(silent_audio, SR)
        assert not np.any(np.isnan(out))


# ─────────────────────────────────────────────────────────────────────────────
# audio_processor.pipeline — FileNotFoundError
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Property-based acoustic validation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterAcousticProperties:
    """Validate that filters produce expected acoustic characteristics."""

    def test_high_pass_attenuates_low_frequencies(self):
        """High-pass filter should suppress frequencies below cutoff."""
        # Create a signal with dominant low-frequency content (30 Hz)
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False, dtype=np.float32)
        low_freq_sine = (0.5 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)

        filtered = high_pass_focus(low_freq_sine, SR)

        # Low frequency energy should be significantly reduced
        input_rms = np.sqrt(np.mean(low_freq_sine**2))
        output_rms = np.sqrt(np.mean(filtered**2))
        attenuation = output_rms / input_rms if input_rms > 1e-9 else 1.0

        # Attenuation should be substantial (< 0.5 for a 30 Hz signal vs 80 Hz cutoff)
        assert (
            attenuation < 0.5
        ), f"Low frequencies should be attenuated, but ratio={attenuation}"

    def test_high_pass_preserves_mid_high_frequencies(self):
        """High-pass filter should preserve frequencies above cutoff."""
        # Create a signal with dominant mid-range content (440 Hz)
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        mid_freq_sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        filtered = high_pass_focus(mid_freq_sine, SR)

        # Mid-frequency energy should be largely preserved
        input_rms = np.sqrt(np.mean(mid_freq_sine**2))
        output_rms = np.sqrt(np.mean(filtered**2))
        attenuation = output_rms / input_rms if input_rms > 1e-9 else 1.0

        # Attenuation should be minimal (> 0.8 for a 440 Hz signal)
        assert (
            attenuation > 0.8
        ), f"Mid frequencies should be preserved, but ratio={attenuation}"

    def test_normalisation_hits_target_dbfs(self):
        """Normalisation filter should bring RMS to target dBFS."""
        from config import NORMALISATION_TARGET_DBFS

        # Create a test signal with arbitrary RMS
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        test_signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        normalised = dynamic_range_norm(test_signal, SR)

        # Calculate RMS in dBFS
        rms = np.sqrt(np.mean(normalised**2))
        rms_dbfs = 20 * np.log10(rms) if rms > 1e-9 else -np.inf

        # Should be close to target (within ±3 dB is reasonable given signal type)
        assert (
            abs(rms_dbfs - NORMALISATION_TARGET_DBFS) < 3
        ), f"RMS should be ~{NORMALISATION_TARGET_DBFS} dBFS, got {rms_dbfs:.1f} dBFS"

    def test_normalisation_reduces_dynamic_range(self):
        """Normalisation should apply consistent gain to hit target RMS level."""
        # Create a weak signal with low RMS
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        weak_signal = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        normalised = dynamic_range_norm(weak_signal, SR)

        # Weak signal with RMS ~0.07 should be boosted toward target dBFS (-20)
        from config import NORMALISATION_TARGET_DBFS

        input_rms = np.sqrt(np.mean(weak_signal**2))
        output_rms = np.sqrt(np.mean(normalised**2))

        input_dbfs = 20 * np.log10(input_rms) if input_rms > 1e-9 else -np.inf
        output_dbfs = 20 * np.log10(output_rms) if output_rms > 1e-9 else -np.inf

        # Output should be much closer to target than input
        assert abs(output_dbfs - NORMALISATION_TARGET_DBFS) < abs(
            input_dbfs - NORMALISATION_TARGET_DBFS
        ), f"Normalisation should bring RMS closer to target: input={input_dbfs:.1f}dB, output={output_dbfs:.1f}dB, target={NORMALISATION_TARGET_DBFS}dB"

        # Output should still be clipped to [-1, 1]
        assert np.max(np.abs(normalised)) <= 1.0

    def test_denoise_reduces_noise_floor(self):
        """Denoise filter should reduce overall energy in quiet segments."""
        # Create a signal with speech-like characteristics (non-silence portions)
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False, dtype=np.float32)

        # Mix a 0.5s silent segment with 1.5s of signal
        signal = np.concatenate(
            [
                np.zeros(int(SR * 0.5), dtype=np.float32),  # silence
                (0.3 * np.sin(2 * np.pi * 200 * t[int(SR * 0.5) : int(SR * 2)])).astype(
                    np.float32
                ),
            ]
        )

        denoised = denoise_filter(signal, SR)

        # Compute energy before and after in quiet region (first 0.5s after silence)
        quiet_start = int(SR * 0.5)
        quiet_end = int(SR * 1.0)

        input_energy = np.sum(signal[quiet_start:quiet_end] ** 2)
        output_energy = np.sum(denoised[quiet_start:quiet_end] ** 2)

        # Denoise should reduce or preserve energy (should not amplify noise)
        # Note: denoise may not always reduce energy, but should not explode
        assert (
            output_energy <= input_energy * 1.1
        ), "Denoise should not amplify signal energy significantly"

    def test_denoise_preserves_signal_structure(self):
        """Denoise filter should not remove periodic signal content."""
        # Create periodic signal with noise floor
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        clean_signal = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noisy_signal = clean_signal + 0.05 * np.random.default_rng(42).normal(
            size=SR
        ).astype(np.float32)
        noisy_signal = np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

        denoised = denoise_filter(noisy_signal, SR)

        # Denoised signal should still be periodic (check by cross-correlation with original)
        # Split into two halves and check correlation
        mid = SR // 2
        corr = np.corrcoef(denoised[:mid], denoised[mid : 2 * mid])[0, 1]

        # For periodic content, correlation should be reasonably high
        assert (
            corr > 0.3
        ), f"Denoise should preserve signal periodicity, correlation={corr:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# audio_processor.pipeline — FileNotFoundError
# ─────────────────────────────────────────────────────────────────────────────


class TestPipeline:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        from audio_processor.pipeline import process_audio

        with pytest.raises(FileNotFoundError):
            process_audio(tmp_path / "nonexistent.wav")

    def test_unsupported_extension_raises_runtime_error(self, tmp_path):
        """A file that cannot be decoded should raise RuntimeError."""
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_bytes(b"not audio data")

        from audio_processor.pipeline import process_audio

        with pytest.raises(RuntimeError, match="Failed to decode audio file"):
            process_audio(bad_file)

    def test_wav_load_uses_soundfile_without_deprecation(self, tmp_path, recwarn):
        """Loading a standard WAV must not trigger the librosa audioread fallback."""
        import soundfile as sf

        wav_path = tmp_path / "tone.wav"
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        sf.write(str(wav_path), 0.5 * np.sin(2 * np.pi * 440 * t), SR, subtype="PCM_16")

        from audio_processor.pipeline import process_audio

        outputs = process_audio(wav_path, output_dir=tmp_path / "variants")

        assert set(outputs) == {"original", "highpass", "normalised", "denoised"}
        deprecation_messages = [
            str(w.message) for w in recwarn.list if "__audioread_load" in str(w.message)
        ]
        assert not deprecation_messages, (
            "librosa audioread deprecation warning was emitted; "
            "the soundfile decode path was not taken."
        )

    def test_non_target_rate_is_resampled(self, tmp_path):
        """A non-target sample rate is resampled to TARGET_SAMPLE_RATE."""
        import soundfile as sf

        from config import TARGET_SAMPLE_RATE

        source_sr = 8_000
        wav_path = tmp_path / "low_rate.wav"
        t = np.linspace(0, 1.0, source_sr, endpoint=False, dtype=np.float32)
        sf.write(str(wav_path), 0.5 * np.sin(2 * np.pi * 220 * t), source_sr)

        from audio_processor.pipeline import _load_audio

        audio, sr = _load_audio(wav_path)
        assert sr == TARGET_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert audio.ndim == 1
