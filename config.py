"""
config.py — Central configuration for the Chorus Engine.

All tunable parameters, paths, and model settings are defined here
to provide a single source of truth across all modules.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# Output Directories
# ─────────────────────────────────────────────
OUTPUTS_DIR = BASE_DIR / "outputs"
VARIANTS_DIR = OUTPUTS_DIR / "variants"
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"
CONSENSUS_DIR = OUTPUTS_DIR / "consensus"

for _dir in (VARIANTS_DIR, TRANSCRIPTS_DIR, CONSENSUS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Whisper Model Configuration
# ─────────────────────────────────────────────
# Supported values: "tiny", "base", "small", "medium", "large"
# "base" offers a strong balance between speed and accuracy for local use.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

# Device: "cpu" or "cuda" (auto-detected if not set)
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

# Language hint (None = auto-detect)
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", None)

# ─────────────────────────────────────────────
# Audio Processing Configuration
# ─────────────────────────────────────────────
# Sample rate used throughout the pipeline (Hz)
TARGET_SAMPLE_RATE = 16_000

# High-Pass Filter cutoff frequency (Hz) — strips sub-vocal rumble
HIGH_PASS_CUTOFF_HZ = 80

# Dynamic range normalisation target loudness (dBFS)
NORMALISATION_TARGET_DBFS = -20.0

# Noise reduction: proportion of noise floor to subtract (0.0 – 1.0)
NOISE_REDUCTION_PROP = 0.75

# ─────────────────────────────────────────────
# Consensus Merger Configuration
# ─────────────────────────────────────────────
# Minimum fraction of transcripts a word must appear in to be kept without review
CONSENSUS_THRESHOLD = 0.75  # i.e., present in ≥ 3 of 4 transcripts

# NLTK similarity threshold for fuzzy-match acceptance (0.0 – 1.0)
SIMILARITY_THRESHOLD = 0.80

# ─────────────────────────────────────────────
# Audio Cleaning Variant Labels
# ─────────────────────────────────────────────
VARIANT_LABELS = {
    "original": "Original (unprocessed)",
    "highpass": "High-Pass Focus",
    "normalised": "Dynamic Range Normalisation",
    "denoised": "Denoise Filter",
}
