"""tests/test_hardware_survey.py — unit tests for ui/hardware_survey.py.

Covers the hardware detection helpers (RAM, CPU cores, GPU/MPS/CUDA) and the
recommendation logic that backs the one-click hardware preset button in the
Streamlit UI. Subprocess calls (sysctl, nvidia-smi) and the optional torch
import are mocked so tests run identically regardless of the host machine.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ui import hardware_survey
from ui.hardware_survey import (
    _detect_gpu,
    _get_cpu_cores,
    _get_ram_gb,
    _recommend_device,
    _recommend_parallelism,
    _recommend_whisper_model,
    detect_hardware,
    recommend_settings,
    recommend_settings_background,
    summarise,
)

# ── detect_hardware() ─────────────────────────────────────────────────────────


class TestDetectHardware:
    def test_assembles_dict_from_helpers(self, monkeypatch):
        monkeypatch.setattr(hardware_survey, "_get_ram_gb", lambda: 16)
        monkeypatch.setattr(hardware_survey, "_get_cpu_cores", lambda: 8)
        monkeypatch.setattr(hardware_survey, "_detect_gpu", lambda: ("apple_mps", 0))

        hw = detect_hardware()

        assert hw == {
            "ram_gb": 16,
            "cpu_cores": 8,
            "gpu_type": "apple_mps",
            "gpu_vram_gb": 0,
        }


# ── _get_ram_gb() ──────────────────────────────────────────────────────────────


class TestGetRamGb:
    def test_macos_reads_sysctl(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Darwin")
        with patch.object(subprocess, "check_output", return_value=f"{17179869184}\n"):
            assert _get_ram_gb() == 16

    def test_linux_reads_proc_meminfo(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Linux")
        meminfo_content = "MemFree:  1000 kB\nMemTotal:  16777216 kB\nOther: 1\n"

        with patch("builtins.open", mock_open(read_data=meminfo_content)):
            assert _get_ram_gb() == 16

    def test_macos_sysctl_failure_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Darwin")
        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError("sysctl not found"),
        ):
            assert _get_ram_gb() == 0

    def test_linux_missing_proc_meminfo_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Linux")
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _get_ram_gb() == 0

    def test_unknown_platform_returns_zero(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Windows")
        assert _get_ram_gb() == 0

    def test_unparseable_sysctl_output_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setattr(hardware_survey.platform, "system", lambda: "Darwin")
        with patch.object(subprocess, "check_output", return_value="not-a-number"):
            assert _get_ram_gb() == 0


# ── _get_cpu_cores() ───────────────────────────────────────────────────────────


class TestGetCpuCores:
    def test_returns_os_cpu_count(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 12)
        assert _get_cpu_cores() == 12

    def test_falls_back_to_one_when_cpu_count_is_none(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert _get_cpu_cores() == 1

    def test_falls_back_to_one_on_exception(self, monkeypatch):
        def _raise():
            raise RuntimeError("boom")

        monkeypatch.setattr("os.cpu_count", _raise)
        assert _get_cpu_cores() == 1


# ── _detect_gpu() ──────────────────────────────────────────────────────────────


class TestDetectGpu:
    def test_torch_reports_cuda_available(self, monkeypatch):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=12 * 1024**3
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _detect_gpu() == ("nvidia_cuda", 12)

    def test_torch_reports_mps_available(self, monkeypatch):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _detect_gpu() == ("apple_mps", 0)

    def test_torch_present_but_no_gpu_falls_back_to_nvidia_smi(self, monkeypatch):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            assert _detect_gpu() == ("none", 0)

    def test_torch_missing_falls_back_to_nvidia_smi_success(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        with patch.object(subprocess, "check_output", return_value="8192\n"):
            assert _detect_gpu() == ("nvidia_cuda", 8)

    def test_torch_missing_and_nvidia_smi_missing_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            assert _detect_gpu() == ("none", 0)

    def test_torch_missing_and_nvidia_smi_returns_garbage(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        with patch.object(subprocess, "check_output", return_value="not-a-number"):
            assert _detect_gpu() == ("none", 0)


# ── _recommend_whisper_model() ────────────────────────────────────────────────


class TestRecommendWhisperModel:
    @pytest.mark.parametrize(
        ("vram_gb", "expected"),
        [
            (12, "large"),
            (10, "large"),
            (6, "medium"),
            (5, "medium"),
            (3, "small"),
            (2, "small"),
            (1, "base"),
            (0, "base"),
        ],
    )
    def test_nvidia_tiers_by_vram(self, vram_gb, expected):
        assert _recommend_whisper_model(0, "nvidia_cuda", vram_gb) == expected

    @pytest.mark.parametrize(
        ("ram_gb", "expected"),
        [
            (64, "large"),
            (32, "large"),
            (24, "medium"),
            (16, "medium"),
            (12, "small"),
            (8, "small"),
            (4, "base"),
            (0, "base"),
        ],
    )
    def test_apple_mps_tiers_by_ram(self, ram_gb, expected):
        assert _recommend_whisper_model(ram_gb, "apple_mps", 0) == expected

    @pytest.mark.parametrize(
        ("ram_gb", "expected"),
        [
            (32, "medium"),
            (16, "medium"),
            (12, "small"),
            (8, "small"),
            (6, "base"),
            (4, "base"),
            (2, "tiny"),
            (0, "tiny"),
        ],
    )
    def test_cpu_only_tiers_by_ram(self, ram_gb, expected):
        assert _recommend_whisper_model(ram_gb, "none", 0) == expected


# ── _recommend_device() and _recommend_parallelism() ─────────────────────────


class TestRecommendDevice:
    @pytest.mark.parametrize(
        ("gpu_type", "expected"),
        [("nvidia_cuda", "cuda"), ("apple_mps", "mps"), ("none", "cpu")],
    )
    def test_maps_gpu_type_to_device(self, gpu_type, expected):
        assert _recommend_device(gpu_type) == expected


class TestRecommendParallelism:
    def test_pins_to_one_on_constrained_cpu_only_machine(self):
        assert _recommend_parallelism(4, "none") == "1"

    def test_auto_on_cpu_with_enough_ram(self):
        assert _recommend_parallelism(8, "none") == "auto"

    def test_auto_when_gpu_present_even_with_low_ram(self):
        assert _recommend_parallelism(4, "nvidia_cuda") == "auto"
        assert _recommend_parallelism(4, "apple_mps") == "auto"


# ── recommend_settings() / recommend_settings_background() ───────────────────


class TestRecommendSettings:
    def test_m_series_mac_16gb_ram(self):
        hw = {"ram_gb": 16, "cpu_cores": 8, "gpu_type": "apple_mps", "gpu_vram_gb": 0}
        assert recommend_settings(hw) == {
            "whisper_model": "medium",
            "device": "mps",
            "parallelism": "auto",
        }

    def test_no_gpu_8gb_ram(self):
        hw = {"ram_gb": 8, "cpu_cores": 4, "gpu_type": "none", "gpu_vram_gb": 0}
        assert recommend_settings(hw) == {
            "whisper_model": "small",
            "device": "cpu",
            "parallelism": "auto",
        }

    def test_no_gpu_low_ram_pins_parallelism(self):
        hw = {"ram_gb": 4, "cpu_cores": 2, "gpu_type": "none", "gpu_vram_gb": 0}
        assert recommend_settings(hw) == {
            "whisper_model": "base",
            "device": "cpu",
            "parallelism": "1",
        }

    def test_nvidia_gpu_with_vram(self):
        hw = {
            "ram_gb": 32,
            "cpu_cores": 16,
            "gpu_type": "nvidia_cuda",
            "gpu_vram_gb": 11,
        }
        assert recommend_settings(hw) == {
            "whisper_model": "large",
            "device": "cuda",
            "parallelism": "auto",
        }


class TestRecommendSettingsBackground:
    def test_drops_one_model_tier_and_pins_parallelism(self):
        hw = {"ram_gb": 16, "cpu_cores": 8, "gpu_type": "apple_mps", "gpu_vram_gb": 0}
        # Foreground recommendation for this profile is "medium" (index 3).
        assert recommend_settings(hw)["whisper_model"] == "medium"

        bg = recommend_settings_background(hw)
        assert bg == {"whisper_model": "small", "device": "mps", "parallelism": "1"}

    def test_does_not_go_below_tiny(self):
        hw = {"ram_gb": 0, "cpu_cores": 1, "gpu_type": "none", "gpu_vram_gb": 0}
        # Foreground recommendation is already the lowest tier, "tiny".
        assert recommend_settings(hw)["whisper_model"] == "tiny"

        bg = recommend_settings_background(hw)
        assert bg["whisper_model"] == "tiny"
        assert bg["parallelism"] == "1"


# ── summarise() ────────────────────────────────────────────────────────────────


class TestSummarise:
    def test_nvidia_summary_includes_vram(self):
        hw = {
            "ram_gb": 32,
            "cpu_cores": 16,
            "gpu_type": "nvidia_cuda",
            "gpu_vram_gb": 11,
        }
        rec = recommend_settings(hw)
        text = summarise(hw, rec)

        assert "NVIDIA CUDA (11 GB VRAM)" in text
        assert "**RAM:** 32 GB" in text
        assert "**Cores:** 16" in text
        assert "model `large`" in text
        assert "device `cuda`" in text

    def test_apple_summary_label(self):
        hw = {"ram_gb": 16, "cpu_cores": 8, "gpu_type": "apple_mps", "gpu_vram_gb": 0}
        rec = recommend_settings(hw)
        text = summarise(hw, rec)

        assert "Apple Silicon (MPS)" in text

    def test_cpu_only_summary_label(self):
        hw = {"ram_gb": 8, "cpu_cores": 4, "gpu_type": "none", "gpu_vram_gb": 0}
        rec = recommend_settings(hw)
        text = summarise(hw, rec)

        assert "None (CPU only)" in text

    def test_unknown_gpu_type_falls_back_to_raw_value(self):
        hw = {
            "ram_gb": 8,
            "cpu_cores": 4,
            "gpu_type": "some_future_gpu",
            "gpu_vram_gb": 0,
        }
        rec = recommend_settings(hw)
        text = summarise(hw, rec)

        assert "some_future_gpu" in text
