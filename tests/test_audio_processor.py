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


@pytest.fixture()
def sine_audio() -> np.ndarray:
    """1-second 440 Hz sine wave, float32, normalised to [-1, 1]."""
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def silent_audio() -> np.ndarray:
    """1-second silence."""
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture()
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


class TestPipeline:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        from audio_processor.pipeline import process_audio

        with pytest.raises(FileNotFoundError):
            process_audio(tmp_path / "nonexistent.wav")

    def test_unsupported_extension_raises(self, tmp_path):
        """A file with an unrecognised extension that cannot be decoded should raise."""
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_bytes(b"not audio data")

        from audio_processor.pipeline import process_audio

        with pytest.raises(Exception):  # noqa: B017, PT011
            process_audio(bad_file)
