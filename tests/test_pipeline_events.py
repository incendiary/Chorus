"""
tests/test_pipeline_events.py — WP1a pipeline event-plumbing tests.

Covers ``pipeline_runner.run_pipeline``'s new ``event_callback`` kwarg and
the orchestrator changes that feed it current-pass detail
(``transcription_engine/orchestrator.py::run_transcription_pass``'s
``on_pass_start`` hook and per-job ``segment_callback`` wrapping).

Only ``transcribe`` (the whisper_engine boundary) and the consensus/export
stage functions are mocked — the real orchestrator sequential/parallel
branches run, so the segment-callback wrapping and parallel-mode null
behaviour are exercised for real. No real Whisper model loading or network
calls occur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import pipeline_runner
from transcription_engine import orchestrator

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


def _fake_transcribe(
    audio_path,
    variant_key,
    stem,
    language=None,
    device=None,
    model_name=None,
    transcripts_dir=None,
    segment_callback=None,
):
    """Stand-in for transcription_engine.whisper_engine.transcribe."""
    total_segments = 3
    if segment_callback:
        for idx in range(total_segments):
            segment_callback(idx, total_segments, f"segment {idx}")
    return {
        "text": "hello world",
        "language": language or "en",
        "segments": [],
        "model": model_name or "base",
        "variant": variant_key,
        "device": device,
    }


def _fake_merge(transcripts, stem, *, consensus_dir=None, **_kwargs):
    consensus_dir = consensus_dir or Path("outputs/consensus")
    path = consensus_dir / f"{stem}_consensus.md"
    path.write_text("# Consensus\n", encoding="utf-8")
    return path, {}


def _fake_ai_context(*, output_dir=None, stem="sample", **_kwargs):
    output_dir = output_dir or Path("outputs/consensus")
    path = output_dir / f"{stem}_ai_context.md"
    path.write_text("# AI Context\n", encoding="utf-8")
    return path


def _fake_bundle(*, output_dir=None, stem="sample", **_kwargs):
    output_dir = output_dir or Path("outputs/consensus")
    path = output_dir / f"{stem}_bundle.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _fake_best_guess(consensus_path, stem, *, output_dir=None, **_kwargs):
    output_dir = output_dir or Path("outputs/consensus")
    path = output_dir / f"{stem}_best_guess.txt"
    path.write_text("hello world", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _mock_pipeline_stages(monkeypatch, tmp_path):
    variants_dir = tmp_path / "variants"
    transcripts_dir = tmp_path / "transcripts"
    consensus_dir = tmp_path / "consensus"
    for d in (variants_dir, transcripts_dir, consensus_dir):
        d.mkdir(parents=True, exist_ok=True)

    def fake_process_audio(audio_path, output_dir=None):
        return {
            "original": variants_dir / "original.wav",
            "highpass": variants_dir / "highpass.wav",
        }

    monkeypatch.setattr(pipeline_runner, "process_audio", fake_process_audio)
    monkeypatch.setattr(orchestrator, "transcribe", _fake_transcribe)
    monkeypatch.setattr(
        orchestrator,
        "_write_txt_companion",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "consensus_merger.merger.merge_transcripts_with_votes", _fake_merge
    )
    monkeypatch.setattr(
        "export_engine.ai_context.generate_ai_context_pack", _fake_ai_context
    )
    monkeypatch.setattr("export_engine.exporter.export_transcript_bundle", _fake_bundle)
    monkeypatch.setattr("export_engine.exporter.export_best_guess", _fake_best_guess)
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 1)


@pytest.fixture
def audio_file(tmp_path) -> Path:
    path = tmp_path / "input.wav"
    path.write_bytes(b"fake-audio")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Event sequence / stage indices
# ─────────────────────────────────────────────────────────────────────────────


def test_event_sequence_and_stage_indices(tmp_path, audio_file):
    events: list[dict[str, Any]] = []

    pipeline_runner.run_pipeline(
        audio_path=audio_file,
        consensus_models=("base",),
        output_dir=tmp_path,
        event_callback=events.append,
    )

    stages_seen = [e["stage"] for e in events]
    # Canonical order for this config (no nlp/llm/diarisation).
    assert stages_seen[0] == "cleaning"
    assert "loading_model" in stages_seen
    assert "transcribing" in stages_seen
    assert "consensus" in stages_seen
    assert "export" in stages_seen
    assert stages_seen[-1] == "done"
    assert "reconstruction" not in stages_seen
    assert "diarisation" not in stages_seen

    # stage_total is constant across all events for a given run and matches
    # the active-stage count for this config.
    stage_totals = {e["stage_total"] for e in events}
    assert stage_totals == {len(pipeline_runner.active_stages())}

    # stage_index is always a valid 1-based position within stage_total.
    for e in events:
        assert 1 <= e["stage_index"] <= e["stage_total"]

    # Every event carries a frac in [0, 1].
    for e in events:
        assert 0.0 <= e["frac"] <= 1.0


def test_stage_total_grows_with_optional_stages(tmp_path, audio_file):
    events: list[dict[str, Any]] = []

    pipeline_runner.run_pipeline(
        audio_path=audio_file,
        consensus_models=("base",),
        enable_nlp=True,
        output_dir=tmp_path,
        event_callback=events.append,
    )

    stages_seen = {e["stage"] for e in events}
    assert "reconstruction" in stages_seen
    assert events[0]["stage_total"] == len(
        pipeline_runner.active_stages(enable_nlp=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# progress_callback stays byte-identical
# ─────────────────────────────────────────────────────────────────────────────


def test_progress_callback_labels_unaffected_by_event_callback(tmp_path, audio_file):
    """The exact (label, frac) sequence must be identical whether or not
    event_callback is supplied — event plumbing is strictly additive."""
    labels_without_events: list[tuple[str, float]] = []
    pipeline_runner.run_pipeline(
        audio_path=audio_file,
        consensus_models=("base",),
        output_dir=tmp_path,
        progress_callback=lambda label, frac: labels_without_events.append(
            (label, frac)
        ),
    )

    labels_with_events: list[tuple[str, float]] = []
    pipeline_runner.run_pipeline(
        audio_path=audio_file,
        consensus_models=("base",),
        output_dir=tmp_path,
        progress_callback=lambda label, frac: labels_with_events.append((label, frac)),
        event_callback=lambda event: None,
    )

    assert labels_with_events == labels_without_events
    assert any(label.startswith("Transcribing: ") for label, _ in labels_with_events)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel branch: segment fields stay null
# ─────────────────────────────────────────────────────────────────────────────


def test_parallel_mode_nulls_segment_fields(tmp_path, audio_file, monkeypatch):
    # Force the parallel branch (workers > 1); it never wires segment_callback
    # or on_pass_start through to _transcribe_one.
    monkeypatch.setattr(orchestrator, "_resolve_parallelism", lambda total: 2)
    monkeypatch.setattr(
        orchestrator, "_build_device_pool", lambda workers: ["cpu", "cpu"]
    )

    events: list[dict[str, Any]] = []
    pipeline_runner.run_pipeline(
        audio_path=audio_file,
        consensus_models=("base",),
        output_dir=tmp_path,
        event_callback=events.append,
    )

    transcribing_events = [e for e in events if e["stage"] == "transcribing"]
    assert transcribing_events, "expected at least one transcribing-stage event"
    for e in transcribing_events:
        assert e["segment"] is None
        assert e["segments_total"] is None
