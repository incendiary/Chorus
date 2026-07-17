"""
tests/test_ui_run_loop.py — Execution tests for the UI run loop and results rendering.

Covers ui/pipeline_invocation.py (run_one_file, render_run_section) and
ui/results.py (render_file_results and its helpers), which previously had
13 %/12 % coverage and had never been executed under test (REVIEW.md PF-4).

Three groups:
  A. run_one_file unit tests — plain function, mocked run_pipeline.
  B. render_run_section via AppTest — mocked pipeline, real Streamlit run loop.
  C. render_file_results via AppTest.from_function — canned results, real files.

No real transcription, model loading, or network calls occur. Patches target
``ui.pipeline_invocation.run_pipeline`` because that module binds the name at
import time (``from pipeline_runner import run_pipeline``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from ui.sidebar import SidebarConfig

APP_PATH = "ui/app.py"
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and canned data
# ─────────────────────────────────────────────────────────────────────────────

CONSENSUS_MD = """\
# Chorus — Consensus Transcript

> **Generated:** 2026-07-16 00:00 UTC
> **Source stem:** `sample`

## Transcription Variants

| Variant Key | Label | Words |
|-------------|-------|------:|
| `original` | Original (unprocessed) | 8 |

## Confidence Statistics

| Tier | Count | Percentage | Meaning |
|------|------:|-----------:|---------|
| HIGH   | 5   | 62.5% | Present in ≥ 75 % of transcripts — kept as-is |
| MEDIUM | 2 | 25.0% | Present in 2 transcripts — highlighted for review |
| LOW    | 1    | 12.5% | Present in only 1 transcript — flagged for removal |

## Consensus Transcript

the quick brown fox jumps ==over== ==the== **~~lazy~~**[^25%: lazy / hazy]

---

## Highlighting Legend

- Plain — HIGH confidence
"""


def _make_config(**overrides) -> SidebarConfig:
    """A SidebarConfig matching the sidebar defaults, with overrides."""
    defaults = {
        "model_choice": "base",
        "consensus_models": ("base", "small"),
        "device_choice": "auto",
        "parallelism_choice": "auto",
        "language": "en",
        "alignment_choice": "sequence",
        "consensus_threshold": 0.75,
        "similarity_threshold": 0.80,
        "noise_mode_choice": "vad",
        "enable_nlp": False,
        "enable_llm": True,
        "ollama_model": "qwen2.5:3b",
        "enable_diarisation": False,
        "export_pdf": False,
        "export_docx": False,
        "export_srt": False,
    }
    defaults.update(overrides)
    return SidebarConfig(**defaults)


def _make_transcripts() -> dict:
    """Four variant transcript dicts shaped like Whisper results."""
    return {
        key: {
            "text": "the quick brown fox jumps over the lazy",
            "language": "en",
            "model": "base",
            "segments": [],
        }
        for key in ("original", "highpass", "normalised", "denoised")
    }


@pytest.fixture()
def canned_results(tmp_path, monkeypatch) -> dict:
    """A run_pipeline result dict backed by real files, with export
    destinations redirected into tmp_path."""
    out_dir = tmp_path / "consensus"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("export_engine.exporter.CONSENSUS_DIR", out_dir)
    monkeypatch.setattr("diarisation.diariser.CONSENSUS_DIR", out_dir)

    consensus_path = out_dir / "sample_consensus.md"
    consensus_path.write_text(CONSENSUS_MD, encoding="utf-8")
    ai_context_path = out_dir / "sample_ai_context.md"
    ai_context_path.write_text("# AI Context Pack\n\nMethodology…", encoding="utf-8")
    bundle_path = out_dir / "sample_bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")

    return {
        "variant_paths": {},
        "transcripts": _make_transcripts(),
        "consensus_path": consensus_path,
        "ai_context_path": ai_context_path,
        "bundle_path": bundle_path,
        "best_guess_path": out_dir / "sample_best_guess.txt",
        "diarised_path": None,
        "speaker_labels": [],
        "elapsed_seconds": 1.23,
    }


class _FakeUpload:
    """Stand-in for a Streamlit UploadedFile (Group A only)."""

    def __init__(self, name: str, content: bytes = b"RIFF-fake-audio"):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


def _download_button_ids_and_labels(at: AppTest) -> list[tuple[str, str]]:
    """Return (element id, label) for every download button in the app."""
    return [(el.proto.id, el.proto.label) for el in at.get("download_button")]


# ─────────────────────────────────────────────────────────────────────────────
# Group A — run_one_file unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunOneFile:
    def test_success_forwards_config_kwargs(self, canned_results):
        """run_one_file must forward every SidebarConfig field that
        run_pipeline accepts, with the exact expected values."""
        from ui.pipeline_invocation import run_one_file

        upload = _FakeUpload("My Interview.wav", b"fake-bytes")
        config_obj = _make_config()
        seen_audio_bytes = {}

        def _fake_pipeline(**kwargs):
            # The temp file must exist and contain the upload bytes at call time.
            seen_audio_bytes["data"] = Path(kwargs["audio_path"]).read_bytes()
            return canned_results

        mock_pipeline = MagicMock(side_effect=_fake_pipeline)
        with patch("ui.pipeline_invocation.run_pipeline", mock_pipeline):
            results, tmp_path, stem = run_one_file(
                upload, MagicMock(), MagicMock(), [], MagicMock(), config_obj
            )

        assert results is canned_results
        assert stem == "My_Interview"
        assert seen_audio_bytes["data"] == b"fake-bytes"

        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["language"] == "en"
        assert kwargs["consensus_models"] == ("base", "small")
        assert kwargs["enable_nlp"] is False
        assert kwargs["enable_llm"] is True
        assert kwargs["ollama_model"] == "qwen2.5:3b"
        assert kwargs["enable_diarisation"] is False
        assert kwargs["alignment_strategy"] == "sequence"
        assert kwargs["consensus_threshold"] == 0.75
        assert kwargs["similarity_threshold"] == 0.80
        assert callable(kwargs["progress_callback"])
        assert Path(kwargs["audio_path"]).suffix == ".wav"
        assert Path(kwargs["audio_path"]).name.startswith("My_Interview_")

        tmp_path.unlink(missing_ok=True)

    def test_pipeline_error_propagates_to_caller(self):
        """run_one_file itself does not swallow pipeline errors — per-file
        capture is the caller's job (render_run_section)."""
        from ui.pipeline_invocation import run_one_file

        upload = _FakeUpload("bad.wav")
        with patch(
            "ui.pipeline_invocation.run_pipeline",
            side_effect=RuntimeError("decoder exploded"),
        ):
            with pytest.raises(RuntimeError, match="decoder exploded"):
                run_one_file(
                    upload, MagicMock(), MagicMock(), [], MagicMock(), _make_config()
                )

    def test_progress_callback_drives_slots_and_log(self, canned_results):
        """Calling the forwarded progress callback must update the progress
        slot, the status slot, and the log-line buffer without crashing."""
        from ui.pipeline_invocation import run_one_file

        progress_slot = MagicMock()
        status_slot = MagicMock()
        log_expander = MagicMock()
        log_lines: list[str] = []

        def _fake_pipeline(**kwargs):
            kwargs["progress_callback"]("Transcribing…", 0.5)
            kwargs["progress_callback"]("Finalising…", 2.0)  # clamped to 1.0
            return canned_results

        with patch("ui.pipeline_invocation.run_pipeline", side_effect=_fake_pipeline):
            _, tmp_path, _ = run_one_file(
                _FakeUpload("clip.mp3"),
                progress_slot,
                status_slot,
                log_lines,
                log_expander,
                _make_config(),
            )

        progress_slot.progress.assert_any_call(0.5, text="Transcribing…")
        progress_slot.progress.assert_any_call(1.0, text="Finalising…")
        status_slot.markdown.assert_any_call("**Status:** Transcribing…")
        assert log_lines == ["`50%` — Transcribing…", "`200%` — Finalising…"]

        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Group B — render_run_section via AppTest
# ─────────────────────────────────────────────────────────────────────────────

_SEQUENTIAL = "Sequential — results appear per file"
_ALL_AT_ONCE = "All at once — results shown at end"


def _upload_and_run(files: list[tuple[str, bytes, str]], mode: str | None) -> AppTest:
    """Upload files, optionally pick a processing mode, click Start Chorus."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.file_uploader[0].set_value(files)
    at.run()
    if mode is not None:
        radio = next(r for r in at.radio if r.label == "Processing mode")
        radio.set_value(mode)
    run_btn = next(b for b in at.button if "Start Chorus" in b.label)
    run_btn.set_value(True)
    at.run()
    return at


@pytest.fixture()
def _deterministic_hw():
    """Pin the hardware recommendation so the mode radio default is stable."""
    with patch(
        "ui.pipeline_invocation.hw_recommendation",
        return_value=(_SEQUENTIAL, "pinned for tests"),
    ):
        yield


@pytest.fixture(autouse=True)
def _restore_run_env(monkeypatch):
    """render_run_section mutates os.environ and the live config module;
    register the current values so monkeypatch restores them afterwards."""
    import os

    import config

    for var in (
        "WHISPER_MODEL",
        "CONSENSUS_MODELS",
        "NOISE_FLOOR_MODE",
        "WHISPER_DEVICE",
        "TRANSCRIPTION_PARALLELISM",
    ):
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "WHISPER_DEVICE", config.WHISPER_DEVICE)
    monkeypatch.setattr(
        config, "TRANSCRIPTION_PARALLELISM", config.TRANSCRIPTION_PARALLELISM
    )


class TestRenderRunSection:
    def test_sequential_two_files_renders_both_status_panels(
        self, canned_results, _deterministic_hw
    ):
        """Two files, sequential mode: pipeline runs twice, a per-file panel
        renders for each, and the forwarded progress callback works live."""

        def _fake_pipeline(**kwargs):
            kwargs["progress_callback"]("Transcribing…", 0.4)
            return canned_results

        mock_pipeline = MagicMock(side_effect=_fake_pipeline)
        with patch("ui.pipeline_invocation.run_pipeline", mock_pipeline):
            at = _upload_and_run(
                [
                    ("alpha.wav", b"a" * 16, "audio/wav"),
                    ("beta.wav", b"b" * 16, "audio/wav"),
                ],
                mode=_SEQUENTIAL,
            )

        assert not at.exception
        assert mock_pipeline.call_count == 2

        expander_labels = [e.label for e in at.expander]
        assert any("alpha.wav" in lbl for lbl in expander_labels)
        assert any("beta.wav" in lbl for lbl in expander_labels)

        # Both files completed: the run summary metrics show 2/0.
        completed = [m.value for m in at.metric if m.label == "Completed"]
        failed = [m.value for m in at.metric if m.label == "Failed"]
        assert "2" in {str(v) for v in completed}
        assert {str(v) for v in failed} == {"0"}

    def test_partial_failure_renders_error_and_other_results(
        self, canned_results, _deterministic_hw
    ):
        """One file failing must not abort the batch: the failure renders the
        error panel with the exception text, and the good file still renders
        its full results."""

        def _fake_pipeline(**kwargs):
            if Path(kwargs["audio_path"]).name.startswith("bad_"):
                raise RuntimeError("decoder exploded on bad.wav")
            return canned_results

        mock_pipeline = MagicMock(side_effect=_fake_pipeline)
        with patch("ui.pipeline_invocation.run_pipeline", mock_pipeline):
            at = _upload_and_run(
                [
                    ("bad.wav", b"x" * 16, "audio/wav"),
                    ("good.wav", b"y" * 16, "audio/wav"),
                ],
                mode=_SEQUENTIAL,
            )

        assert not at.exception
        assert mock_pipeline.call_count == 2

        # Failure surfaced via render_processing_error.
        error_values = [e.value for e in at.error]
        assert any("Processing failed for bad.wav" in v for v in error_values)
        code_values = [c.value for c in at.code]
        assert any("decoder exploded on bad.wav" in v for v in code_values)

        # The successful file still rendered full results (download buttons).
        labels = [lbl for _, lbl in _download_button_ids_and_labels(at)]
        assert any("Consensus Markdown" in lbl for lbl in labels)

        # Summary reflects the split outcome.
        failed = {str(m.value) for m in at.metric if m.label == "Failed"}
        assert "1" in failed
        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "ATTENTION REQUIRED" in markdown_blob

    def test_all_at_once_three_files_runs_pipeline_three_times(
        self, canned_results, _deterministic_hw
    ):
        """Three files auto-switch to batch (all-at-once) mode: the pipeline
        runs once per file and results render after processing, with the
        quick-navigation bar and results filter present."""
        mock_pipeline = MagicMock(return_value=canned_results)
        with patch("ui.pipeline_invocation.run_pipeline", mock_pipeline):
            at = _upload_and_run(
                [
                    ("one.wav", b"1" * 16, "audio/wav"),
                    ("two.wav", b"2" * 16, "audio/wav"),
                    ("three.wav", b"3" * 16, "audio/wav"),
                ],
                mode=None,  # 3+ files: mode radio is not shown, batch is forced
            )

        assert not at.exception
        assert mock_pipeline.call_count == 3

        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "Quick Navigation" in markdown_blob
        assert "ALL FILES COMPLETED" in markdown_blob

        # Results filter radio appears for 3+ files.
        filter_radio = next(r for r in at.radio if r.label == "Results filter")
        assert filter_radio.options == ["All", "Completed", "Failed"]

        completed = {str(m.value) for m in at.metric if m.label == "Completed"}
        assert "3" in completed

    def test_two_files_all_at_once_mode_toggle(self, canned_results, _deterministic_hw):
        """With two files the mode radio is shown; choosing all-at-once still
        executes the pipeline exactly twice and renders both results."""
        mock_pipeline = MagicMock(return_value=canned_results)
        with patch("ui.pipeline_invocation.run_pipeline", mock_pipeline):
            at = _upload_and_run(
                [
                    ("alpha.wav", b"a" * 16, "audio/wav"),
                    ("beta.wav", b"b" * 16, "audio/wav"),
                ],
                mode=_ALL_AT_ONCE,
            )

        assert not at.exception
        assert mock_pipeline.call_count == 2
        completed = {str(m.value) for m in at.metric if m.label == "Completed"}
        assert "2" in completed


# ─────────────────────────────────────────────────────────────────────────────
# Group C — render_file_results rendering
# ─────────────────────────────────────────────────────────────────────────────


def _results_script(project_root, results, show_low, formats_to_export):
    import sys

    sys.path.insert(0, project_root)

    from ui.results import render_file_results

    render_file_results(
        "sample.wav",
        results,
        None,
        "sample",
        show_low=show_low,
        formats_to_export=formats_to_export,
    )


def _render_results(results: dict, *, show_low: bool = True, formats=None) -> AppTest:
    at = AppTest.from_function(
        _results_script,
        kwargs={
            "project_root": PROJECT_ROOT,
            "results": results,
            "show_low": show_low,
            "formats_to_export": formats or [],
        },
        default_timeout=30,
    )
    at.run()
    return at


class TestRenderFileResults:
    def test_download_buttons_render_for_each_artefact(self, canned_results):
        """Consensus, most-likely, best-guess, zip archive, AI context, and
        per-variant JSON download buttons must all render with real bytes."""
        at = _render_results(canned_results)
        assert not at.exception

        ids_labels = _download_button_ids_and_labels(at)
        labels = [lbl for _, lbl in ids_labels]
        ids = [el_id for el_id, _ in ids_labels]

        assert any("Consensus Markdown" in lbl for lbl in labels)
        assert any("Most Likely Transcript" in lbl for lbl in labels)
        assert any("Best-Guess Transcript" in lbl for lbl in labels)
        assert any("Full Output Archive" in lbl for lbl in labels)
        assert any("AI Context Pack" in lbl for lbl in labels)

        for key in (
            "dl_md_sample",
            "dl_txt_sample",
            "dl_best_guess_sample",
            "dl_zip_sample",
            "dl_ai_ctx_sample",
            "dl_json_original_sample",
        ):
            assert any(el_id.endswith(f"-{key}") for el_id in ids), key

    def test_tier_statistics_show_correct_counts(self, canned_results):
        """The confidence overview must parse the stats table and show the
        HIGH/MEDIUM/LOW counts from the consensus document (5/2/1)."""
        at = _render_results(canned_results)
        assert not at.exception

        metrics = {m.label: str(m.value) for m in at.metric}
        assert metrics["High confidence words"] == "5"
        assert metrics["Medium confidence words"] == "2"
        assert metrics["Low confidence words"] == "1"

        # Percentage bars derive from the same counts (8 total words).
        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "5 (62%)" in markdown_blob
        assert "2 (25%)" in markdown_blob
        assert "1 (12%)" in markdown_blob

    def test_no_diarisation_degrades_gracefully(self, canned_results):
        """diarised_path=None: no speaker section, no exception."""
        assert canned_results["diarised_path"] is None
        at = _render_results(canned_results)
        assert not at.exception

        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "Speaker Diarisation" not in markdown_blob

    def test_diarisation_section_renders_speaker_table(self, canned_results, tmp_path):
        """With diarisation data present, the speaker-name table, transcript
        preview, and diarised download button all render."""
        diarised_path = canned_results["consensus_path"].parent / "sample_diarised.md"
        diarised_path.write_text(
            "# Diarised\n\n**SPEAKER_00:** hello there\n", encoding="utf-8"
        )
        results = {
            **canned_results,
            "diarised_path": diarised_path,
            "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
        }

        at = _render_results(results)
        assert not at.exception

        # One name input per speaker, plus the save button.
        name_inputs = [t for t in at.text_input if t.label.startswith("Name for ")]
        assert {t.label for t in name_inputs} == {
            "Name for SPEAKER_00",
            "Name for SPEAKER_01",
        }
        assert any("Save Speaker Names" in b.label for b in at.button)

        ids_labels = _download_button_ids_and_labels(at)
        assert any(el_id.endswith("-dl_diar_sample") for el_id, _ in ids_labels)
