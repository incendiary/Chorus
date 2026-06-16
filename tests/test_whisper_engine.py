"""tests/test_whisper_engine.py — unit tests for Whisper model caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from transcription_engine import whisper_engine


class _DummyModel:
    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(self, audio_path: str, **kwargs) -> dict[str, Any]:
        self.calls.append((audio_path, kwargs))
        return {"text": self.text, "segments": [], "language": "en"}


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    whisper_engine.unload_model()
    yield
    whisper_engine.unload_model()


def test_get_model_cache_key_includes_model_and_device(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_load_model(model_name: str, device: str):
        calls.append((model_name, device))
        return object()

    monkeypatch.setattr(whisper_engine.whisper, "load_model", fake_load_model)

    model_a, device_a, model_name_a = whisper_engine._get_model(
        model_name="base", device="cpu"
    )
    model_b, _, _ = whisper_engine._get_model(model_name="small", device="cpu")
    model_c, _, _ = whisper_engine._get_model(model_name="base", device="cpu")

    assert device_a == "cpu"
    assert model_name_a == "base"
    assert model_a is model_c
    assert model_a is not model_b
    assert calls == [("base", "cpu"), ("small", "cpu")]


def test_get_model_fallback_uses_cpu_cache_per_model(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_load_model(model_name: str, device: str):
        calls.append((model_name, device))
        if device.startswith("cuda"):
            raise RuntimeError("GPU unavailable")
        return object()

    monkeypatch.setattr(whisper_engine.whisper, "load_model", fake_load_model)

    _, device_1, model_1 = whisper_engine._get_model(model_name="small", device="cuda:0")
    _, device_2, model_2 = whisper_engine._get_model(model_name="small", device="cuda:1")

    assert (device_1, model_1) == ("cpu", "small")
    assert (device_2, model_2) == ("cpu", "small")
    assert calls.count(("small", "cpu")) == 1


def test_transcribe_respects_explicit_model_name(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    model = _DummyModel(text="transcribed")

    def fake_get_model(device=None, model_name=None):
        return model, "cpu", model_name or "base"

    monkeypatch.setattr(whisper_engine, "_get_model", fake_get_model)

    result = whisper_engine.transcribe(
        audio_path=audio_path,
        variant_key="original",
        stem="sample",
        language="en",
        model_name="tiny",
        transcripts_dir=tmp_path,
    )

    assert result["text"] == "transcribed"
    assert result["model"] == "tiny"
    assert result["device"] == "cpu"

    saved = tmp_path / "sample_original.json"
    assert saved.exists()
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["model"] == "tiny"
    assert payload["variant"] == "original"

    assert model.calls
    _, kwargs = model.calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["word_timestamps"] is True


def test_transcribe_segment_callback_fires(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    class SegmentModel(_DummyModel):
        def transcribe(self, audio_path: str, **kwargs) -> dict:
            base = super().transcribe(audio_path, **kwargs)
            base["segments"] = [
                {"start": 0.0, "end": 0.5, "text": "hello"},
                {"start": 0.5, "end": 1.0, "text": "world"},
            ]
            return base

    def fake_get_model(device=None, model_name=None):
        return SegmentModel(), "cpu", model_name or "base"

    monkeypatch.setattr(whisper_engine, "_get_model", fake_get_model)

    fired: list[tuple[int, int, str]] = []

    def cb(idx, total, text):
        fired.append((idx, total, text))

    whisper_engine.transcribe(
        audio_path=audio_path,
        variant_key="original",
        stem="sample",
        transcripts_dir=tmp_path,
        segment_callback=cb,
    )

    assert len(fired) == 2
    assert fired[0] == (0, 2, "hello")
    assert fired[1] == (1, 2, "world")
