"""tests/test_orchestrator.py — unit tests for transcription orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
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


# These patch ``config`` rather than the orchestrator's own namespace, which is
# how the application actually changes these settings. Patching the module
# copy (as these tests did originally) asserted the import-time binding and so
# could never catch that real callers had no effect — see RC-2.


def test_resolve_parallelism_from_config_integer(monkeypatch):
    # A configured count is honoured only up to what is thread-safe; multiple
    # CUDA devices give each worker its own cached model.
    monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "3")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 3)
    assert orchestrator._resolve_parallelism(4) == 3
    assert orchestrator._resolve_parallelism(2) == 2


def test_resolve_parallelism_auto_cuda_multi_gpu(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "auto")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 2)
    assert orchestrator._resolve_parallelism(4) == 2


def test_build_device_pool_round_robin_cuda(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 3)
    assert orchestrator._build_device_pool(3) == ["cuda:0", "cuda:1", "cuda:2"]


def test_run_transcription_pass_parallel(monkeypatch, tmp_path, variant_paths):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 2)
    monkeypatch.setattr(
        orchestrator, "_build_device_pool", lambda workers: ["cuda:0", "cuda:1"]
    )

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
        audio_path,
        variant_key,
        stem,
        language=None,
        device=None,
        model_name=None,
        **kwargs,
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

    expected_calls = {("base", k) for k in variant_paths} | {
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


def test_load_transcripts_from_disk_custom_dir(monkeypatch, tmp_path):
    """transcripts_dir override should take precedence over TRANSCRIPTS_DIR."""
    global_dir = tmp_path / "global"
    custom_dir = tmp_path / "isolated_run"
    global_dir.mkdir()
    custom_dir.mkdir()
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", global_dir)
    monkeypatch.setattr(orchestrator, "CONSENSUS_MODELS", ("base",))

    # Write a file only in the custom dir
    (custom_dir / "sample_original.json").write_text(
        '{"text": "isolated", "model": "base"}', encoding="utf-8"
    )

    transcripts = orchestrator.load_transcripts_from_disk(
        "sample", transcripts_dir=custom_dir
    )

    assert "original" in transcripts
    assert transcripts["original"]["text"] == "isolated"

    # Nothing should have been read from global dir
    transcripts_global = orchestrator.load_transcripts_from_disk("sample")
    assert "original" not in transcripts_global


def test_run_transcription_pass_model_names_override(
    monkeypatch, tmp_path, variant_paths
):
    monkeypatch.setattr(orchestrator, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "CONSENSUS_MODELS", ("base", "small"))
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 1)

    seen_models: list[str | None] = []

    def fake_transcribe(
        audio_path,
        variant_key,
        stem,
        language=None,
        device=None,
        model_name=None,
        **kwargs,
    ):
        seen_models.append(model_name)
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
        model_names=("medium",),
    )

    assert set(transcripts.keys()) == set(variant_paths.keys())
    assert set(seen_models) == {"medium"}


class TestThreadSafeParallelism:
    """Whisper is not thread-safe: workers sharing one cached model instance
    corrupt each other's KV cache (KeyError: Linear(...)). Concurrency is only
    safe across distinct devices, since models cache per (model, device).

    Regression test for the 28 July production failure, where CPU + 4 workers
    destroyed six files' transcription after ~45 minutes each.
    """

    def test_cpu_parallelism_is_capped_to_one(self, monkeypatch):
        import config
        from transcription_engine import orchestrator

        monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
        monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "4")

        assert orchestrator._resolve_parallelism(4) == 1
        assert orchestrator._build_device_pool(1) == ["cpu"]

    def test_mps_parallelism_is_capped_to_one(self, monkeypatch):
        import config
        from transcription_engine import orchestrator

        monkeypatch.setattr(config, "WHISPER_DEVICE", "mps")
        monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "4")

        assert orchestrator._resolve_parallelism(4) == 1

    def test_auto_mode_on_cpu_is_also_capped(self, monkeypatch):
        """The auto path previously returned min(4, cpu_count) on CPU."""
        import config
        from transcription_engine import orchestrator

        monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
        monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "auto")

        assert orchestrator._resolve_parallelism(4) == 1

    def test_multi_gpu_cuda_keeps_real_concurrency(self, monkeypatch):
        """Separate CUDA devices mean separate cached models — safe."""
        import config
        from transcription_engine import orchestrator

        monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda:0")
        monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "auto")
        monkeypatch.setattr(orchestrator, "_get_cuda_device_count", lambda: 4)

        assert orchestrator._resolve_parallelism(4) == 4
        assert orchestrator._build_device_pool(4) == [
            "cuda:0",
            "cuda:1",
            "cuda:2",
            "cuda:3",
        ]

    def test_runtime_config_change_is_honoured(self, monkeypatch):
        """RC-2: settings are read at call time, not bound at import, so the
        sidebar's device and parallelism controls actually take effect."""
        import config
        from transcription_engine import orchestrator

        monkeypatch.setattr(config, "WHISPER_DEVICE", "mps")
        monkeypatch.setattr(config, "TRANSCRIPTION_PARALLELISM", "auto")
        assert orchestrator._build_device_pool(1) == ["mps"]

        # Simulate the UI switching device after import.
        monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
        assert orchestrator._build_device_pool(1) == ["cpu"]
