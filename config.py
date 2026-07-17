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


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    for out_dir in (VARIANTS_DIR, TRANSCRIPTS_DIR, CONSENSUS_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Whisper Model Configuration
# ─────────────────────────────────────────────
# Supported values: "tiny", "base", "small", "medium", "large"
# "base" offers a strong balance between speed and accuracy for local use.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")


def _parse_consensus_models(
    raw_value: str | None,
    default_model: str,
) -> tuple[str, ...]:
    """Parse comma-separated model names into a stable, de-duplicated tuple."""
    if raw_value is None:
        return (default_model,)

    parsed: list[str] = []
    seen: set[str] = set()
    for chunk in raw_value.split(","):
        model_name = chunk.strip().lower()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        parsed.append(model_name)

    if not parsed:
        return (default_model,)
    return tuple(parsed)


# Model list for upcoming multi-model consensus passes.
# Example: CONSENSUS_MODELS=base,small,medium
CONSENSUS_MODELS = _parse_consensus_models(
    os.environ.get("CONSENSUS_MODELS"),
    WHISPER_MODEL,
)

# Human-readable labels for configured consensus model variants.
CONSENSUS_MODEL_LABELS = {model: f"Whisper {model}" for model in CONSENSUS_MODELS}

# Compute device for Whisper inference.
# Explicit override: set WHISPER_DEVICE=cpu | cuda | mps in your environment.
# If unset, the best available device is probed automatically:
#   NVIDIA GPU (CUDA) → Apple Silicon GPU (MPS) → CPU


def _detect_device() -> str:
    """Return the best available compute device for PyTorch/Whisper."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001, S110
        pass  # torch unavailable — fall back silently to cpu
    return "cpu"


_env_device = os.environ.get("WHISPER_DEVICE", "").strip().lower()
WHISPER_DEVICE: str = _env_device if _env_device else _detect_device()

# Language hint (None = auto-detect)
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", None)

# Transcription concurrency for variant passes.
# Accepted values:
#   - "auto" (default): choose sensible parallelism per device/capacity
#   - integer string (e.g. "1", "2", "4")
TRANSCRIPTION_PARALLELISM = os.environ.get("TRANSCRIPTION_PARALLELISM", "auto")

# Local Ollama settings for optional LLM token reconstruction.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "20"))

# ─────────────────────────────────────────────
# Audio Processing Configuration
# ─────────────────────────────────────────────
# Accepted input extensions, shared by the UI uploader and the batch scanner.
# The loader decodes natively via libsndfile where possible and falls back to
# ffmpeg for everything else, so this list covers common audio formats plus
# video containers whose audio track ffmpeg can extract. An unreadable file
# still fails per-file with a clear error rather than being silently skipped.
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".m4b",
        ".mp4",
        ".aac",
        ".ogg",
        ".oga",
        ".opus",
        ".flac",
        ".webm",
        ".wma",
        ".amr",
        ".3gp",
        ".aif",
        ".aiff",
        ".caf",
        ".mka",
        ".mkv",
        ".mov",
    }
)

# Sample rate used throughout the pipeline (Hz)
TARGET_SAMPLE_RATE = 16_000

# High-Pass Filter cutoff frequency (Hz) — strips sub-vocal rumble
HIGH_PASS_CUTOFF_HZ = 80

# Dynamic range normalisation target loudness (dBFS)
NORMALISATION_TARGET_DBFS = -20.0

# Noise reduction: proportion of noise floor to subtract (0.0 – 1.0)
NOISE_REDUCTION_PROP = 0.75

# Noise floor detection: "vad" (auto-detect silence via energy) or "fixed" (first 0.5 s)
NOISE_FLOOR_MODE = os.environ.get("NOISE_FLOOR_MODE", "vad")

# ─────────────────────────────────────────────
# Consensus Merger Configuration
# ─────────────────────────────────────────────
# Minimum fraction of transcripts a word must appear in to be kept without review
CONSENSUS_THRESHOLD = 0.75  # i.e., present in ≥ 3 of 4 transcripts

# NLTK similarity threshold for fuzzy-match acceptance (0.0 – 1.0)
SIMILARITY_THRESHOLD = 0.80

# Alignment strategy: "sequence" (Needleman-Wunsch, accurate) or "positional" (fast, legacy)
# Sequence alignment handles word insertions/deletions across variants.
# Positional alignment is faster but assumes variants have similar word counts.
ALIGNMENT_STRATEGY = os.environ.get("ALIGNMENT_STRATEGY", "sequence")

# ─────────────────────────────────────────────
# Audio Cleaning Variant Labels
# ─────────────────────────────────────────────
VARIANT_LABELS = {
    "original": "Original (unprocessed)",
    "highpass": "High-Pass Focus",
    "normalised": "Dynamic Range Normalisation",
    "denoised": "Denoise Filter",
}
