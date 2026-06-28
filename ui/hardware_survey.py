"""ui/hardware_survey.py — Hardware detection and Chorus setting recommendations.

Mirrors the logic in devops-practices/survey-ollama-env.sh but runs in-process,
returning a structured dict rather than printing to a terminal.
"""

from __future__ import annotations

import platform
import subprocess


def detect_hardware() -> dict:
    """Return a dict describing the current machine's hardware.

    Keys: ram_gb (int), gpu_type (str), gpu_vram_gb (int), cpu_cores (int).
    gpu_type is one of: "nvidia_cuda", "apple_mps", "none".
    gpu_vram_gb is 0 for non-NVIDIA hardware (Apple uses unified memory).
    """
    ram_gb = _get_ram_gb()
    cpu_cores = _get_cpu_cores()
    gpu_type, gpu_vram_gb = _detect_gpu()
    return {
        "ram_gb": ram_gb,
        "cpu_cores": cpu_cores,
        "gpu_type": gpu_type,
        "gpu_vram_gb": gpu_vram_gb,
    }


def recommend_settings(hw: dict) -> dict:
    """Derive recommended Chorus settings from hardware dict.

    Returns a dict with keys:
      whisper_model  (str)  — "tiny" | "base" | "small" | "medium" | "large"
      device         (str)  — "auto" | "cpu" | "cuda" | "mps"
      parallelism    (str)  — "auto" | integer string
    """
    ram_gb = hw["ram_gb"]
    gpu_type = hw["gpu_type"]
    gpu_vram_gb = hw["gpu_vram_gb"]

    whisper_model = _recommend_whisper_model(ram_gb, gpu_type, gpu_vram_gb)
    device = _recommend_device(gpu_type)
    parallelism = _recommend_parallelism(ram_gb, gpu_type)

    return {
        "whisper_model": whisper_model,
        "device": device,
        "parallelism": parallelism,
    }


def summarise(hw: dict, rec: dict) -> str:
    """Return a short human-readable summary of detection results."""
    gpu_label = {
        "nvidia_cuda": f"NVIDIA CUDA ({hw['gpu_vram_gb']} GB VRAM)",
        "apple_mps": "Apple Silicon (MPS)",
        "none": "None (CPU only)",
    }.get(hw["gpu_type"], hw["gpu_type"])

    return (
        f"**RAM:** {hw['ram_gb']} GB  |  "
        f"**Cores:** {hw['cpu_cores']}  |  "
        f"**GPU:** {gpu_label}\n\n"
        f"**Recommended:** model `{rec['whisper_model']}`, "
        f"device `{rec['device']}`, "
        f"parallelism `{rec['parallelism']}`"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_ram_gb() -> int:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) // (1024**3)
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024**2)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _get_cpu_cores() -> int:
    try:
        import os

        return os.cpu_count() or 1
    except Exception:  # noqa: BLE001
        return 1


def _detect_gpu() -> tuple[str, int]:
    """Return (gpu_type, vram_gb). Uses torch if available, falls back to nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return "nvidia_cuda", vram_bytes // (1024**3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "apple_mps", 0
    except Exception:  # noqa: BLE001
        pass

    # torch unavailable — fall back to nvidia-smi subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vram_mb = int(out.strip().splitlines()[0])
        return "nvidia_cuda", vram_mb // 1024
    except Exception:  # noqa: BLE001
        pass

    return "none", 0


def _recommend_whisper_model(ram_gb: int, gpu_type: str, gpu_vram_gb: int) -> str:
    if gpu_type == "nvidia_cuda":
        if gpu_vram_gb >= 10:
            return "large"
        if gpu_vram_gb >= 5:
            return "medium"
        if gpu_vram_gb >= 2:
            return "small"
        return "base"

    if gpu_type == "apple_mps":
        if ram_gb >= 32:
            return "large"
        if ram_gb >= 16:
            return "medium"
        if ram_gb >= 8:
            return "small"
        return "base"

    # CPU only
    if ram_gb >= 16:
        return "medium"
    if ram_gb >= 8:
        return "small"
    if ram_gb >= 4:
        return "base"
    return "tiny"


def _recommend_device(gpu_type: str) -> str:
    if gpu_type == "nvidia_cuda":
        return "cuda"
    if gpu_type == "apple_mps":
        return "mps"
    return "cpu"


def _recommend_parallelism(ram_gb: int, gpu_type: str) -> str:
    # auto is almost always correct; only pin to 1 on very constrained CPU machines
    if gpu_type == "none" and ram_gb < 8:
        return "1"
    return "auto"
