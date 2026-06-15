"""tests/test_orchestrator.py — unit tests for transcription orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription_engine import orchestrator


@pytest.fixture
def variant_paths(tmp_path) -> dict[str, Path]:
    """Create fake variant audio files."""
    keys = ["original", "highpass", "normalised", "denoised"]
    paths: dict[str, Path] = {}
    for key in keys:
        path = tmp_path / f"{key}.wav"
        path.write_bytes(b"fake")
        paths[key] = path
    return paths


def test_resolve_parallelism_from_config_integer(monkeypatch):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTION_PARALLELISM", "3")
    assert orchestrator._resolve_parallelism(4) == 3
    assert orchestrator._resolve_parallelism(2) == 2


def test_resolve_parallelism_auto_cuda_multi_gpu(monkeypatch):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTION_PARALLELISM", "auto")
    monkeypatch.setattr(orchestrator, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 2)
    assert orchestrator._resolve_parallelism(4) == 2


def test_build_device_pool_round_robin_cuda(monkeypatch):
    monkeypatch.setattr(orchestrator, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 3)
    assert orchestrator._build_device_pool(3) == ["cuda:0", "cuda:1", "cuda:2"]


def test_run_transcription_pass_parallel(monkeypatch, tmp_path, variant_paths):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 2)
    monkeypatch.setattr(orchestrator, "_build_device_pool", lambda workers: ["cuda:0", "cuda:1"])

    seen_devices: list[str] = []

    def fake_transcribe(
        audio_path, variant_key, stem, language=None, device=None, **kwargs
    ):
        seen_devices.append(device or "")
        return {
            "text": f"text for {variant_key}",
            "language": language or "en",
            "model": "base",
            "device": device,
        }

    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe)

    progress_calls: list[tuple[int, int, str]] = []

    def progress(step: int, total: int, label: str) -> None:
        progress_calls.append((step, total, label))

    transcripts = orchestrator.run_transcription_pass(
        variant_paths=variant_paths,
        stem="sample",
        language="en",
        progress_callback=progress,
    )

    assert set(transcripts.keys()) == set(variant_paths.keys())
    assert len(progress_calls) == len(variant_paths)
    assert set(seen_devices).issubset({"cuda:0", "cuda:1"})

    # Companion text files should be written for each variant.
    for key in variant_paths:
        txt_path = tmp_path / f"sample_{key}.txt"
        assert txt_path.exists()
        text = txt_path.read_text(encoding="utf-8")
        assert "# Chorus Transcript" in text


def test_run_transcription_pass_multimodel_keys(monkeypatch, tmp_path, variant_paths):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "CONSENSUS_MODELS", ("base", "small"))
    monkeypatch.setattr(
        orchestrator,
        "CONSENSUS_MODEL_LABELS",
        {"base": "Whisper base", "small": "Whisper small"},
    )
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 1)

    seen_calls: list[tuple[str | None, str]] = []

    def fake_transcribe(
        audio_path, variant_key, stem, language=None, device=None, model_name=None, **kwargs
    ):
        seen_calls.append((model_name, variant_key))
        return {
            "text": f"{model_name}:{variant_key}",
            "language": language or "en",
            "model": model_name,
            "device": device,
        }

    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe)

    transcripts = orchestrator.run_transcription_pass(
        variant_paths=variant_paths,
        stem="sample",
        language="en",
    )

    primary_keys = set(variant_paths.keys())
    secondary_keys = {f"small__{k}" for k in variant_paths}
    assert set(transcripts.keys()) == primary_keys | secondary_keys

    expected_calls = {
        ("base", k) for k in variant_paths
    } | {
        ("small", k) for k in variant_paths
    }
    assert set(seen_calls) == expected_calls


def test_load_transcripts_from_disk_multimodel(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "CONSENSUS_MODELS", ("base", "small"))

    payload = {"text": "hello", "model": "small"}
    target = tmp_path / "sample_small__original.json"
    target.write_text('{"text": "hello", "model": "small"}', encoding="utf-8")

    transcripts = orchestrator.load_transcripts_from_disk("sample")

    assert "small__original" in transcripts
    assert transcripts["small__original"]["text"] == payload["text"]
