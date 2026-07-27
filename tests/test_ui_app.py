"""
tests/test_ui_app.py — Smoke and behaviour tests for the Streamlit UI.

Uses streamlit.testing.v1.AppTest for headless testing without a browser.
No real transcription, model loading, or network calls occur — the pipeline
entry point is never invoked (these tests never click "Start Chorus"), and
the Ollama/spaCy availability probes are mocked where determinism requires it.
"""

from __future__ import annotations

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = "ui/app.py"


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def _checkbox(at: AppTest, label_contains: str):
    return next(cb for cb in at.checkbox if label_contains in cb.label)


def _selectbox(at: AppTest, label: str):
    return next(sb for sb in at.selectbox if sb.label == label)


# ─────────────────────────────────────────────────────────────────────────────
# App renders
# ─────────────────────────────────────────────────────────────────────────────


class TestAppRenders:
    def test_app_renders_without_exception(self):
        """The app should load and run cleanly with no interaction."""
        at = _run_app()
        assert not at.exception

    def test_no_upload_does_not_crash(self):
        """With no files uploaded, the app renders the upload prompt and stops
        there — no pipeline stage should run and no exception should occur."""
        at = _run_app()
        assert not at.exception
        uploader = at.get("file_uploader")
        assert len(uploader) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────


class TestSidebarControls:
    def test_model_size_selector_includes_large(self):
        at = _run_app()
        model_sb = _selectbox(at, "Model size")
        assert model_sb.options == ["tiny", "base", "small", "medium", "large"]

    def test_device_selector_options(self):
        at = _run_app()
        device_sb = _selectbox(at, "Compute device")
        assert device_sb.options == [
            "Auto-detect (recommended)",
            "CPU",
            "NVIDIA CUDA (GPU)",
            "Apple MPS (Apple Silicon)",
        ]

    def test_hardware_preset_selector_offers_max_and_background(self):
        at = _run_app()
        preset_sb = _selectbox(at, "Settings preset")
        assert preset_sb.options == ["Max", "Background"]

    def test_parallelism_worker_count_appears_when_auto_disabled(self):
        """Auto parallelism is on by default (no worker-count input shown).
        Disabling it should reveal a number_input for the worker count."""
        at = _run_app()
        assert len(at.number_input) == 0

        auto_cb = _checkbox(at, "Auto parallelism")
        auto_cb.set_value(False)
        at.run()

        assert not at.exception
        assert len(at.number_input) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Ollama unreachable path
# ─────────────────────────────────────────────────────────────────────────────


class TestOllamaUnreachable:
    def test_llm_checkbox_triggers_setup_dialog_when_unreachable(self):
        """Enabling LLM reconstruction when Ollama is unreachable should not
        crash, and should flag the setup dialog with the failure reason."""
        with patch(
            "reconstruction.ollama_client.probe_model",
            return_value=(False, "Connection refused"),
        ):
            at = _run_app()
            llm_cb = _checkbox(at, "LLM Reconstruction")
            llm_cb.set_value(True)
            at.run()

        assert not at.exception
        assert at.session_state["show_ollama_dialog"] is True
        assert at.session_state["ollama_fail_reason"] == "Connection refused"

    def test_llm_checkbox_succeeds_when_ollama_reachable(self):
        """When Ollama is reachable, no setup dialog should be flagged."""
        with (
            patch(
                "reconstruction.ollama_client.probe_model",
                return_value=(True, ""),
            ),
            patch(
                "reconstruction.ollama_client.list_models",
                return_value=["llama3.1:8b"],
            ),
        ):
            at = _run_app()
            llm_cb = _checkbox(at, "LLM Reconstruction")
            llm_cb.set_value(True)
            at.run()

        assert not at.exception
        assert "show_ollama_dialog" not in at.session_state or not at.session_state.get(
            "show_ollama_dialog"
        )


# ─────────────────────────────────────────────────────────────────────────────
# spaCy unavailable path
# ─────────────────────────────────────────────────────────────────────────────


class TestSpacyUnavailable:
    def test_nlp_checkbox_triggers_setup_dialog_when_model_missing(self):
        """Enabling NLP reconstruction when the spaCy model is unavailable
        should not crash, and should flag the setup dialog with the reason."""
        with patch(
            "reconstruction.probe_spacy_model",
            return_value=(False, "Model en_core_web_md not found"),
        ):
            at = _run_app()
            nlp_cb = _checkbox(at, "NLP Reconstruction")
            nlp_cb.set_value(True)
            at.run()

        assert not at.exception
        assert at.session_state["show_spacy_dialog"] is True
        assert at.session_state["spacy_fail_reason"] == "Model en_core_web_md not found"


class TestLanguageSelector:
    """The language control offers valid Whisper codes and confirms the choice."""

    def test_defaults_to_auto_detect_with_feedback(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("ui/app.py", default_timeout=60)
        at.run()

        selector = [s for s in at.selectbox if s.label == "Language"][0]
        assert selector.value is None
        captions = [c.value for c in at.caption]
        assert any("Auto-detect" in c for c in captions)

    def test_selecting_a_language_is_confirmed_on_screen(self):
        """Choosing a language must give visible feedback — the old free-text
        box silently accepted input, so it looked like nothing happened."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("ui/app.py", default_timeout=60)
        at.run()
        [s for s in at.selectbox if s.label == "Language"][0].set_value("fr")
        at.run()

        assert not at.exception
        selector = [s for s in at.selectbox if s.label == "Language"][0]
        assert selector.value == "fr"
        assert any("French" in c.value for c in at.caption)

    def test_only_whisper_supported_codes_are_offered(self):
        """Every offered code must be one Whisper actually accepts."""
        from whisper.tokenizer import LANGUAGES

        from ui.sidebar import _LANGUAGE_CODES

        assert set(_LANGUAGE_CODES) <= set(LANGUAGES)
        assert _LANGUAGE_CODES[0] == "en"  # most common first


class TestBuildLabel:
    """The header states which build is running.

    Python imports at process start, so a running app keeps serving the code
    it launched with even after a pull — the label makes that visible instead
    of leaving it to be inferred from process timestamps.
    """

    def test_label_reports_version_branch_and_commit(self):
        from ui.build_info import BUILD_LABEL

        assert BUILD_LABEL.startswith("v")
        # In a git checkout the label carries branch and short commit too.
        assert "@" in BUILD_LABEL

    def test_label_is_rendered_in_the_header(self):
        from streamlit.testing.v1 import AppTest

        from ui.build_info import BUILD_LABEL

        at = AppTest.from_file("ui/app.py", default_timeout=60)
        at.run()

        assert not at.exception
        assert any(BUILD_LABEL in m.value for m in at.markdown)

    def test_falls_back_to_version_when_git_unavailable(self, monkeypatch):
        """Docker images ship VERSION but no git metadata."""
        from ui import build_info

        monkeypatch.setattr(build_info, "_git", lambda *args: None)
        assert build_info._describe_build() == f"v{build_info._read_version()}"
