"""
tests/test_run_status_panel.py — WP2 Amendment B status-panel tests.

Covers ``ui/run_status_panel.py::render_file_status_panel`` (AppTest wrapper
feeding crafted ``file_state`` dicts, mirroring the on-disk state schema in
``ui/run_state.py``) and the O(1) log-line regression in
``ui/pipeline_invocation.py::run_one_file`` that this amendment replaces.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = None  # set in fixture below


def _panel_script(file_state: dict) -> None:
    from ui.run_status_panel import render_file_status_panel

    render_file_status_panel(file_state)


def _render_panel(file_state: dict) -> AppTest:
    at = AppTest.from_function(
        _panel_script, kwargs={"file_state": file_state}, default_timeout=30
    )
    at.run()
    return at


def _base_state(**overrides) -> dict:
    state = {
        "name": "sample.wav",
        "stem": "sample",
        "status": "running",
        "progress": 0.3,
        "stage": "transcribing",
        "stage_index": 3,
        "stage_total": 6,
        "detail": "Whisper base · Original (unprocessed)",
        "passes_done": 3,
        "passes_total": 8,
        "segment": 45,
        "segments_total": 197,
        "parallel_workers": None,
        "last_event_at": time.time(),
        "started_at": time.time() - 12,
        "history": [],
        "_config": {},
    }
    state.update(overrides)
    return state


class TestStatusLine:
    def test_progress_text_includes_segment_when_present(self):
        at = _render_panel(_base_state())
        assert not at.exception

        progress_els = at.get("progress")
        assert len(progress_els) == 1
        text = progress_els[0].proto.text
        assert "Transcribing" in text
        assert "Whisper base · Original (unprocessed)" in text
        assert "segment 45/197" in text

    def test_progress_text_omits_segment_when_null(self):
        at = _render_panel(_base_state(segment=None, segments_total=None))
        assert not at.exception

        text = at.get("progress")[0].proto.text
        assert "segment" not in text
        # Falls back to the pass counter when no segment info is available.
        assert "pass 3/8" in text

    def test_caption_shows_stage_elapsed_and_pass(self):
        at = _render_panel(_base_state())
        assert not at.exception

        caption_blob = "\n".join(c.value for c in at.caption)
        assert "Stage 3/6" in caption_blob
        assert "elapsed" in caption_blob
        assert "pass 3/8" in caption_blob


class TestChecklist:
    def test_checklist_glyphs_reflect_current_stage(self):
        at = _render_panel(_base_state())
        assert not at.exception

        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "✅ Audio cleaning" in markdown_blob
        assert "🔵 Transcribing (3/8)" in markdown_blob
        assert "⚪ Consensus" in markdown_blob
        assert "⚪ Export" in markdown_blob

    def test_optional_stages_omitted_when_disabled(self):
        at = _render_panel(_base_state(_config={}))
        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "Reconstruction" not in markdown_blob
        assert "Diarisation" not in markdown_blob

    def test_optional_stages_shown_when_enabled(self):
        at = _render_panel(
            _base_state(_config={"enable_llm": True, "enable_diarisation": True})
        )
        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "Reconstruction" in markdown_blob
        assert "Diarisation" in markdown_blob

    def test_all_complete_when_done(self):
        at = _render_panel(_base_state(stage="done", stage_index=6, stage_total=6))
        markdown_blob = "\n".join(m.value for m in at.markdown)
        assert "⚪" not in markdown_blob


class TestHistory:
    def test_history_renders_newest_first(self):
        history = [
            ["cleaning", None],
            ["transcribing", "Whisper base · Original (unprocessed)"],
            ["transcribing", "Whisper base · Denoised"],
        ]
        at = _render_panel(_base_state(history=history))
        assert not at.exception

        markdown_blob = [m.value for m in at.markdown]
        denoised_idx = next(i for i, v in enumerate(markdown_blob) if "Denoised" in v)
        cleaning_idx = next(
            i
            for i, v in enumerate(markdown_blob)
            if v.strip("- ").startswith("Audio cleaning")
        )
        # Newest transition (Denoised) must render before the oldest (cleaning).
        assert denoised_idx < cleaning_idx

    def test_empty_history_shows_placeholder(self):
        at = _render_panel(_base_state(history=[]))
        assert not at.exception
        caption_blob = "\n".join(c.value for c in at.caption)
        assert "No stage transitions recorded yet." in caption_blob


class TestStaleWarning:
    def test_stale_event_triggers_warning(self):
        at = _render_panel(_base_state(last_event_at=time.time() - 90))
        assert not at.exception
        warning_blob = "\n".join(w.value for w in at.warning)
        assert "No progress events for" in warning_blob

    def test_recent_event_no_warning(self):
        at = _render_panel(_base_state(last_event_at=time.time() - 5))
        assert not at.exception
        assert len(at.warning) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Regression: run_one_file's log callback is O(1) per call, not a growing join
# ─────────────────────────────────────────────────────────────────────────────


class TestRunOneFileLogRegression:
    def test_200_progress_calls_leave_log_expander_receiving_only_last_line(self):
        """Amendment B replaced the O(n²) ``"\\n\\n".join(log_lines)`` render
        with ``log_expander.markdown(log_lines[-1])``: each of 200 progress
        callbacks must pass exactly one line — never the accumulated history."""
        from ui.pipeline_invocation import run_one_file

        log_expander = MagicMock()
        log_lines: list[str] = []

        def _fake_pipeline(**kwargs):
            for i in range(200):
                kwargs["progress_callback"](f"Step {i}", i / 200)
            return {"consensus_path": None, "elapsed_seconds": 0.01}

        class _FakeUpload:
            name = "clip.wav"

            def read(self) -> bytes:
                return b"fake"

        from ui.sidebar import SidebarConfig

        config_obj = SidebarConfig(
            model_choice="base",
            consensus_models=("base",),
            device_choice="auto",
            parallelism_choice="auto",
            language="en",
            alignment_choice="sequence",
            consensus_threshold=0.75,
            similarity_threshold=0.80,
            noise_mode_choice="vad",
            enable_nlp=False,
            enable_llm=False,
            ollama_model=None,
            enable_diarisation=False,
            export_pdf=False,
            export_docx=False,
            export_srt=False,
        )

        with patch("ui.pipeline_invocation.run_pipeline", side_effect=_fake_pipeline):
            _, tmp_path, _ = run_one_file(
                _FakeUpload(),
                MagicMock(),
                MagicMock(),
                log_lines,
                log_expander,
                config_obj,
            )

        assert log_expander.markdown.call_count == 200
        # Every call received a single line, never a joined multi-line blob.
        for call in log_expander.markdown.call_args_list:
            (arg,) = call.args
            assert "\n" not in arg
        # The final call received only the last line, not the full history.
        last_call_arg = log_expander.markdown.call_args_list[-1].args[0]
        assert last_call_arg == log_lines[-1]
        assert len(log_lines) == 200

        tmp_path.unlink(missing_ok=True)
