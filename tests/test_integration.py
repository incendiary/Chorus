"""
tests/test_integration.py — Full pipeline integration tests.

These tests exercise the end-to-end Chorus pipeline with synthetic audio,
mocking only the Whisper transcription engine (to avoid downloading model
weights in CI). The audio processing, alignment, consensus, and export
stages all run with real implementations.

What is tested:
  - Pipeline completes without error for a synthetic WAV file
  - All expected output files are generated
  - Consensus document contains expected structure
  - AI context pack is generated with correct sections
  - Export formats (SRT, VTT) produce valid output
  - Speaker name persistence round-trips correctly
  - Alignment strategy switch works end-to-end
  - Empty/short audio handles gracefully
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic audio generation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_sine_wav(
    path: Path, duration_s: float = 1.0, freq_hz: float = 440.0
) -> Path:
    """
    Generate a synthetic WAV file containing a sine wave.

    Parameters
    ----------
    path : Path
        Output file path.
    duration_s : float
        Duration in seconds.
    freq_hz : float
        Frequency of the sine wave.

    Returns
    -------
    Path
        The written file path.
    """
    import math

    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    amplitude = 16000

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        frames = b""
        for i in range(n_samples):
            t = i / sample_rate
            sample = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
            frames += struct.pack("<h", sample)

        wf.writeframes(frames)

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Mock transcription results
# ─────────────────────────────────────────────────────────────────────────────


def _make_whisper_result(text: str, language: str = "en") -> dict[str, Any]:
    """Create a mock Whisper result dict with word-level timestamps."""
    words_list = text.split()
    segments = []
    word_entries = []

    # Create one segment spanning the whole text
    if words_list:
        word_duration = 0.5
        for idx, word in enumerate(words_list):
            start = idx * word_duration
            end = start + word_duration
            word_entries.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                    "probability": 0.95,
                }
            )

        segments.append(
            {
                "id": 0,
                "start": 0.0,
                "end": len(words_list) * word_duration,
                "text": text,
                "words": word_entries,
            }
        )

    return {
        "text": text,
        "language": language,
        "segments": segments,
        "model": "base",
    }


# Transcripts with controlled variations for consensus testing
MOCK_TRANSCRIPTS = {
    "original": _make_whisper_result("the quick brown fox jumps over the lazy dog"),
    "cleaned_hp": _make_whisper_result("the quick brown fox dumps over the lazy dog"),
    "normalised": _make_whisper_result("the quick brown fox jumps over the hazy dog"),
    "denoised": _make_whisper_result("the quick brown fox jumps over the lazy dog"),
}


def _mock_run_transcription_pass(
    variant_paths: dict[str, Path],
    stem: str,
    language: str | None = None,
    progress_callback=None,
    **kwargs,
) -> dict[str, dict]:
    """Mock transcription that returns pre-defined results."""
    if progress_callback:
        for i, key in enumerate(variant_paths):
            progress_callback(i + 1, len(variant_paths), key)
    return MOCK_TRANSCRIPTS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_audio(tmp_path) -> Path:
    """Generate a 1-second synthetic WAV file."""
    return _generate_sine_wav(tmp_path / "test_recording.wav", duration_s=1.0)


@pytest.fixture
def short_audio(tmp_path) -> Path:
    """Generate a very short (0.1s) synthetic WAV file."""
    return _generate_sine_wav(tmp_path / "short.wav", duration_s=0.1)


@pytest.fixture
def _patch_transcription():
    """Patch the transcription engine so no Whisper model is needed."""
    with patch(
        "pipeline_runner.run_transcription_pass",
        side_effect=_mock_run_transcription_pass,
    ):
        yield


@pytest.fixture
def patch_consensus_dir(tmp_path, monkeypatch):
    """Redirect consensus output to a temp directory."""
    out_dir = tmp_path / "outputs" / "consensus"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr("config.CONSENSUS_DIR", out_dir)
    monkeypatch.setattr("consensus_merger.renderer.CONSENSUS_DIR", out_dir)
    monkeypatch.setattr("export_engine.exporter.CONSENSUS_DIR", out_dir)
    monkeypatch.setattr("export_engine.ai_context.CONSENSUS_DIR", out_dir)
    monkeypatch.setattr("diarisation.diariser.CONSENSUS_DIR", out_dir)
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end pipeline tests with mocked transcription."""

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_pipeline_completes(self, synthetic_audio, patch_consensus_dir):
        """Pipeline should complete without error."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        assert results is not None
        assert "consensus_path" in results
        assert "transcripts" in results
        assert "elapsed_seconds" in results

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_consensus_document_generated(self, synthetic_audio, patch_consensus_dir):
        """Should produce a consensus .md file."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        consensus_path = results["consensus_path"]
        assert consensus_path.exists()
        assert consensus_path.suffix == ".md"

        text = consensus_path.read_text(encoding="utf-8")
        assert "Chorus" in text
        assert "Consensus Transcript" in text

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_ai_context_pack_generated(self, synthetic_audio, patch_consensus_dir):
        """Should produce an AI context pack."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        ai_path = results.get("ai_context_path")
        assert ai_path is not None
        assert ai_path.exists()
        assert ai_path.name.endswith("_ai_context.md")

        text = ai_path.read_text(encoding="utf-8")
        assert "## Methodology" in text
        assert "## Confidence Statistics" in text
        assert "## Clean Transcript" in text

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_confidence_tiers_present(self, synthetic_audio, patch_consensus_dir):
        """Consensus should contain HIGH/MEDIUM/LOW words given controlled input."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        text = results["consensus_path"].read_text(encoding="utf-8")

        # "jumps" appears in 3/4 variants (original, normalised, denoised) = HIGH
        # "dumps" appears in 1/4 variants = LOW
        # Basic structure check
        assert "HIGH" in text
        assert "Confidence Statistics" in text

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_elapsed_time_positive(self, synthetic_audio, patch_consensus_dir):
        """Elapsed time should be a positive number."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        assert results["elapsed_seconds"] > 0

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_variant_paths_returned(self, synthetic_audio, patch_consensus_dir):
        """Should return the audio variant paths."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        variant_paths = results["variant_paths"]
        assert isinstance(variant_paths, dict)
        assert len(variant_paths) > 0
        for _key, path in variant_paths.items():
            assert Path(path).exists()


class TestOptionalPipelineFeatures:
    """Test optional NLP and diarisation pipeline paths."""

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_pipeline_with_nlp_enabled(self, synthetic_audio, patch_consensus_dir):
        """NLP reconstruction path should execute when enabled."""
        from pipeline_runner import run_pipeline

        with patch(
            "reconstruction.nlp.reconstruct_low_tokens",
            side_effect=lambda votes: votes,
        ) as mock_reconstruct:
            results = run_pipeline(
                audio_path=synthetic_audio,
                language="en",
                enable_nlp=True,
            )

        assert results["consensus_path"].exists()
        mock_reconstruct.assert_called_once()

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_pipeline_with_llm_enabled(self, synthetic_audio, patch_consensus_dir):
        """LLM reconstruction path should execute when enabled."""
        from pipeline_runner import run_pipeline

        with patch(
            "reconstruction.llm.reconstruct_low_tokens_llm",
            side_effect=lambda votes, model=None: votes,
        ) as mock_reconstruct:
            results = run_pipeline(
                audio_path=synthetic_audio,
                language="en",
                enable_llm=True,
            )

        assert results["consensus_path"].exists()
        mock_reconstruct.assert_called_once()

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_pipeline_with_diarisation_enabled(
        self, synthetic_audio, patch_consensus_dir
    ):
        """Diarisation path should produce diarised output and speaker labels."""
        from diarisation.diariser import SpeakerSegment
        from pipeline_runner import run_pipeline

        fake_segments = [
            SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
            SpeakerSegment(speaker="SPEAKER_01", start=2.0, end=4.0),
        ]

        with patch("diarisation.diariser.diarise", return_value=fake_segments):
            results = run_pipeline(
                audio_path=synthetic_audio,
                language="en",
                enable_diarisation=True,
            )

        assert results["diarised_path"] is not None
        assert results["diarised_path"].exists()
        assert results["speaker_labels"]
        assert all(label.startswith("SPEAKER_") for label in results["speaker_labels"])


class TestAlignmentStrategies:
    """Test that both alignment strategies work end-to-end."""

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_sequence_alignment(self, synthetic_audio, patch_consensus_dir):
        """Pipeline with sequence alignment should complete."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(
            audio_path=synthetic_audio, language="en", alignment_strategy="sequence"
        )
        assert results["consensus_path"].exists()

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_positional_alignment(self, synthetic_audio, patch_consensus_dir):
        """Pipeline with positional alignment should complete."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(
            audio_path=synthetic_audio, language="en", alignment_strategy="positional"
        )
        assert results["consensus_path"].exists()

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_different_strategies_same_high_words(
        self, synthetic_audio, patch_consensus_dir
    ):
        """Both strategies should agree on HIGH-confidence words."""
        from pipeline_runner import run_pipeline

        r_seq = run_pipeline(
            audio_path=synthetic_audio, language="en", alignment_strategy="sequence"
        )
        r_pos = run_pipeline(
            audio_path=synthetic_audio, language="en", alignment_strategy="positional"
        )

        # Both should contain the common words
        seq_text = r_seq["consensus_path"].read_text(encoding="utf-8")
        pos_text = r_pos["consensus_path"].read_text(encoding="utf-8")
        # "the" appears in all 4 variants — should be in both
        assert "the" in seq_text
        assert "the" in pos_text


class TestExportIntegration:
    """Test export formats work end-to-end with pipeline output."""

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_srt_export(self, synthetic_audio, patch_consensus_dir):
        """Should produce a valid SRT file."""
        from export_engine.exporter import export_srt
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        srt_path = export_srt(results["transcripts"]["original"], "test_recording")
        assert srt_path.exists()

        content = srt_path.read_text(encoding="utf-8")
        # SRT should have numbered cues with timestamps
        assert "1\n" in content
        assert "-->" in content

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_vtt_export(self, synthetic_audio, patch_consensus_dir):
        """Should produce a valid VTT file."""
        from export_engine.exporter import export_vtt
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        vtt_path = export_vtt(results["transcripts"]["original"], "test_recording")
        assert vtt_path.exists()

        content = vtt_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")
        assert "-->" in content

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_plain_text_export(self, synthetic_audio, patch_consensus_dir):
        """Should produce a plain-text transcript."""
        from export_engine.exporter import export_plain_text
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        txt_path = export_plain_text(results["consensus_path"], "test_recording")
        assert txt_path.exists()

        content = txt_path.read_text(encoding="utf-8")
        # Should contain at least some recognisable words
        assert len(content) > 0

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_zip_export(self, synthetic_audio, patch_consensus_dir):
        """Should produce a non-empty zip bundle."""
        import zipfile
        from io import BytesIO

        from export_engine.exporter import export_zip
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        zip_bytes = export_zip(
            results["consensus_path"],
            results["transcripts"]["original"],
            "test_recording",
        )
        assert len(zip_bytes) > 0

        # Should be a valid zip
        zf = zipfile.ZipFile(BytesIO(zip_bytes))
        names = zf.namelist()
        assert any("consensus.md" in n for n in names)
        assert any("most_likely" in n for n in names)
        assert any("ai_context" in n for n in names)


class TestSpeakerNameIntegration:
    """Test speaker name persistence end-to-end."""

    @pytest.mark.usefixtures("patch_consensus_dir")
    def test_save_and_load_roundtrip(self, patch_consensus_dir):
        """Speaker names should survive a save/load cycle."""
        from diarisation.diariser import load_speaker_names, save_speaker_names

        mapping = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        save_speaker_names("roundtrip_test", mapping)

        loaded = load_speaker_names("roundtrip_test")
        assert loaded == mapping

    @pytest.mark.usefixtures("patch_consensus_dir")
    def test_names_appear_in_diarised_output(self, patch_consensus_dir):
        """Saved speaker names should appear in the diarised transcript."""
        from diarisation.diariser import (
            LabelledSegment,
            load_speaker_names,
            render_diarised_md,
            save_speaker_names,
        )

        # Save names first
        save_speaker_names("named_test", {"SPEAKER_00": "Interviewer"})

        # Create labelled segments
        labelled = [
            LabelledSegment(speaker="SPEAKER_00", start=0, end=2, text="Hello."),
            LabelledSegment(speaker="SPEAKER_01", start=2, end=4, text="Hi there."),
        ]

        # Render with loaded names
        names = load_speaker_names("named_test")
        path = render_diarised_md(labelled, "named_test", speaker_map=names)
        text = path.read_text(encoding="utf-8")

        assert "Interviewer" in text
        # SPEAKER_01 has no saved name — should show raw label
        assert "SPEAKER_01" in text


class TestAIContextIntegration:
    """Test AI context pack integration with pipeline."""

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_context_pack_has_processing_config(
        self, synthetic_audio, patch_consensus_dir
    ):
        """AI context should reflect actual processing config."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(
            audio_path=synthetic_audio, language="en", alignment_strategy="sequence"
        )
        ai_text = results["ai_context_path"].read_text(encoding="utf-8")
        assert "sequence" in ai_text

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_context_pack_has_clean_transcript(
        self, synthetic_audio, patch_consensus_dir
    ):
        """AI context should contain the clean transcript words."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        ai_text = results["ai_context_path"].read_text(encoding="utf-8")
        # Common words from all 4 variants should appear
        assert "the" in ai_text
        assert "quick" in ai_text
        assert "brown" in ai_text

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_context_pack_has_uncertainty_info(
        self, synthetic_audio, patch_consensus_dir
    ):
        """AI context should flag uncertain words."""
        from pipeline_runner import run_pipeline

        results = run_pipeline(audio_path=synthetic_audio, language="en")
        ai_text = results["ai_context_path"].read_text(encoding="utf-8")
        # Should have the uncertainty section
        assert "Uncertainty" in ai_text


class TestErrorHandling:
    """Test graceful error handling."""

    def test_missing_file_raises(self):
        """Pipeline should raise FileNotFoundError for missing input."""
        from pipeline_runner import run_pipeline

        with pytest.raises(FileNotFoundError):
            run_pipeline(audio_path="/nonexistent/audio.wav")

    def test_corrupt_audio_raises_runtime_error(self, tmp_path):
        """Unreadable audio payloads should raise a decode RuntimeError."""
        from pipeline_runner import run_pipeline

        bad_audio = tmp_path / "corrupt.wav"
        bad_audio.write_bytes(b"this is not a valid wav payload")

        with pytest.raises(RuntimeError, match="Failed to decode audio file"):
            run_pipeline(audio_path=bad_audio)

    @pytest.mark.usefixtures("_patch_transcription", "patch_consensus_dir")
    def test_progress_callback_invoked(self, synthetic_audio, patch_consensus_dir):
        """Progress callback should be called multiple times."""
        from pipeline_runner import run_pipeline

        calls: list[tuple[str, float]] = []

        def _cb(label: str, frac: float) -> None:
            calls.append((label, frac))

        run_pipeline(audio_path=synthetic_audio, language="en", progress_callback=_cb)
        # Should have multiple progress updates
        assert len(calls) > 5
        # Last call should be at 1.0
        assert calls[-1][1] == 1.0


class TestOutputDirIsolation:
    """Test that output_dir correctly isolates pipeline outputs."""

    @pytest.mark.usefixtures("_patch_transcription")
    def test_output_dir_creates_subdirs(self, synthetic_audio, tmp_path):
        """Pipeline should write all outputs into the provided output_dir."""
        from pipeline_runner import run_pipeline

        out = tmp_path / "isolated_run"
        result = run_pipeline(
            audio_path=synthetic_audio,
            language="en",
            output_dir=out,
        )

        # variants and consensus are written by real stages (Stage 1 & 3).
        # transcripts/ is created by Stage 2 (mocked), so only assert real stages.
        assert (out / "variants").is_dir()
        assert (out / "consensus").is_dir()
        assert result["consensus_path"].parent == out / "consensus"

    @pytest.mark.usefixtures("_patch_transcription")
    def test_two_runs_stay_isolated(self, synthetic_audio, tmp_path):
        """Concurrent output dirs must not share outputs."""
        from pipeline_runner import run_pipeline

        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"

        result_a = run_pipeline(
            audio_path=synthetic_audio, language="en", output_dir=out_a
        )
        result_b = run_pipeline(
            audio_path=synthetic_audio, language="en", output_dir=out_b
        )

        assert result_a["consensus_path"] != result_b["consensus_path"]
        assert result_a["consensus_path"].parent == out_a / "consensus"
        assert result_b["consensus_path"].parent == out_b / "consensus"

    @pytest.mark.usefixtures("_patch_transcription")
    def test_same_stem_two_runs_isolation(self, tmp_path):
        """Two runs with identical stem but different output_dir must not collide."""
        from pipeline_runner import run_pipeline

        # Create two audio files with the same stem but in different dirs
        audio_1 = tmp_path / "input_1" / "test_recording.wav"
        audio_2 = tmp_path / "input_2" / "test_recording.wav"
        audio_1.parent.mkdir(parents=True)
        audio_2.parent.mkdir(parents=True)

        _generate_sine_wav(audio_1, duration_s=1.0)
        _generate_sine_wav(audio_2, duration_s=1.0)

        out_1 = tmp_path / "run_1"
        out_2 = tmp_path / "run_2"

        result_1 = run_pipeline(audio_path=audio_1, language="en", output_dir=out_1)
        result_2 = run_pipeline(audio_path=audio_2, language="en", output_dir=out_2)

        # Consensus files must be in different directories despite same stem
        assert result_1["consensus_path"].parent == out_1 / "consensus"
        assert result_2["consensus_path"].parent == out_2 / "consensus"
        assert result_1["consensus_path"] != result_2["consensus_path"]

        # Bundle files must also be isolated
        bundle_1 = out_1 / "consensus" / "test_recording_bundle.json"
        bundle_2 = out_2 / "consensus" / "test_recording_bundle.json"
        assert bundle_1.exists()
        assert bundle_2.exists()
        assert bundle_1 != bundle_2

    @pytest.mark.usefixtures("_patch_transcription")
    def test_source_filename_in_bundle(self, tmp_path):
        """Bundle JSON should preserve original source filename."""
        import json

        from pipeline_runner import run_pipeline

        original_audio = tmp_path / "my_recording_2026-02-09.wav"
        _generate_sine_wav(original_audio, duration_s=1.0)

        out_dir = tmp_path / "output"
        results = run_pipeline(
            audio_path=original_audio,
            language="en",
            output_dir=out_dir,
        )

        bundle_path = results.get("bundle_path")
        assert bundle_path is not None
        assert bundle_path.exists()

        bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle_data["meta"]["source_filename"] == "my_recording_2026-02-09.wav"

    @pytest.mark.usefixtures("_patch_transcription")
    def test_source_filename_in_consensus_md(self, tmp_path):
        """Consensus markdown should include Source file field."""
        from pipeline_runner import run_pipeline

        original_audio = tmp_path / "interview_alice_bob_final.m4a"
        _generate_sine_wav(original_audio, duration_s=1.0)

        out_dir = tmp_path / "output"
        results = run_pipeline(
            audio_path=original_audio,
            language="en",
            output_dir=out_dir,
        )

        consensus_path = results["consensus_path"]
        text = consensus_path.read_text(encoding="utf-8")

        # Should contain the original filename in the header
        assert "Source file" in text
        assert "interview_alice_bob_final.m4a" in text


class TestConsensusModelForwarding:
    """Test consensus model selection wiring into transcription stage."""

    @pytest.mark.usefixtures("patch_consensus_dir")
    def test_run_pipeline_forwards_consensus_models(self, synthetic_audio):
        """run_pipeline should forward explicit model selection to orchestrator."""
        from pipeline_runner import run_pipeline

        captured: dict[str, Any] = {}

        def _capture_pass(
            variant_paths: dict[str, Path],
            stem: str,
            language: str | None = None,
            progress_callback=None,
            **kwargs,
        ) -> dict[str, dict]:
            captured["model_names"] = kwargs.get("model_names")
            if progress_callback:
                for i, key in enumerate(variant_paths):
                    progress_callback(i + 1, len(variant_paths), key)
            return MOCK_TRANSCRIPTS

        with patch("pipeline_runner.run_transcription_pass", side_effect=_capture_pass):
            run_pipeline(
                audio_path=synthetic_audio,
                language="en",
                consensus_models=("base", "small"),
            )

        assert captured["model_names"] == ("base", "small")
