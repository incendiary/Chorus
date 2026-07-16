"""
tests/test_benchmark.py — sanity gates for the RB-2 WER/calibration benchmark.

These protect against a benchmark that is "wrong but green": they exercise
only the pure helper functions in ``benchmarks.run_benchmark`` (normalisation,
WER scoring, SNR mixing, calibration counting) and never download
LibriSpeech or load Whisper.
"""

from __future__ import annotations

import numpy as np

from benchmarks.run_benchmark import calibration_table, snr_mix, wer

REFERENCE = "The quick brown fox jumps over the lazy dog."


def test_wer_is_zero_for_normalised_reference_as_hypothesis():
    """Gate 1: feeding the normalised reference text as the hypothesis gives WER 0.0."""
    assert wer(REFERENCE, REFERENCE) == 0.0
    # Also true when hypothesis differs only in case/punctuation.
    assert wer(REFERENCE, "the QUICK brown fox jumps over the lazy dog") == 0.0


def test_wer_is_high_for_unrelated_text():
    """Gate 2: feeding a completely different text gives WER >= 0.9."""
    hypothesis = "purple elephants dance quietly beneath the moonlit garden wall"
    assert wer(REFERENCE, hypothesis) >= 0.9


def test_snr_mix_hits_target_within_half_a_db():
    """Gate 3: the SNR helper's output measures within +/-0.5 dB of the target."""
    signal = np.sin(np.linspace(0, 100 * np.pi, 16_000)).astype(np.float32)
    target_snr_db = 5.0

    noisy = snr_mix(signal, target_snr_db, np.random.default_rng(42))

    noise = noisy.astype(np.float64) - signal.astype(np.float64)
    signal_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean(noise**2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    assert abs(measured_snr_db - target_snr_db) <= 0.5


def test_calibration_table_hand_built_five_word_case():
    """
    Gate 4: a hand-built 5-word case with known alignment produces the
    expected per-tier precision numbers.

    Reference: "the quick brown fox jumps"
    Hypothesis words (with tiers):
        the    (HIGH)   -> correct   (equal)
        quick  (HIGH)   -> correct   (equal)
        slow   (MEDIUM) -> incorrect (substitute for "brown")
        fox    (LOW)    -> correct   (equal)
        leaps  (LOW)    -> incorrect (substitute for "jumps")

    Expected: HIGH 2/2 = 1.0, MEDIUM 0/1 = 0.0, LOW 1/2 = 0.5.
    """
    reference_text = "the quick brown fox jumps"
    hyp_words = [
        {"word": "the", "tier": "HIGH"},
        {"word": "quick", "tier": "HIGH"},
        {"word": "slow", "tier": "MEDIUM"},
        {"word": "fox", "tier": "LOW"},
        {"word": "leaps", "tier": "LOW"},
    ]

    table = calibration_table(hyp_words, reference_text)

    assert table["HIGH"] == {"count": 2, "precision": 1.0}
    assert table["MEDIUM"] == {"count": 1, "precision": 0.0}
    assert table["LOW"] == {"count": 2, "precision": 0.5}


def test_calibration_table_ignores_punctuation_only_words():
    """A hypothesis word that normalises to nothing (e.g. bare punctuation) is dropped
    rather than desynchronising the tier list from the alignment."""
    reference_text = "the quick brown fox"
    hyp_words = [
        {"word": "the", "tier": "HIGH"},
        {"word": "--", "tier": "LOW"},
        {"word": "quick", "tier": "HIGH"},
        {"word": "brown", "tier": "HIGH"},
        {"word": "fox", "tier": "HIGH"},
    ]

    table = calibration_table(hyp_words, reference_text)

    assert table["HIGH"] == {"count": 4, "precision": 1.0}
    assert "LOW" not in table
