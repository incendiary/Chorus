"""tests/test_run_indicator.py — Tests for the cross-page run-progress indicator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _state(status: str, files: list[dict] | None = None) -> dict:
    """A minimal active-run state document in the given *status*."""
    return {
        "schema_version": 1,
        "run_id": f"test_run_{status}",
        "status": status,
        "boot_id": "boot_123",
        "started_at": 1234567890,
        "finished_at": None if status == "running" else 1234567900,
        "log_path": None,
        "config": {},
        "files": files or [],
    }


def _all_markdown(at: AppTest) -> str:
    """Every markdown value rendered by the app, joined for substring checks."""
    return "\n".join(el.value for el in at.markdown)


class TestRunIndicator:
    """render_run_indicator() renders only for active/interrupted runs."""

    def test_no_state_renders_nothing(self) -> None:
        """With no active-run state, no pill is rendered."""
        with patch("ui.run_indicator.load_state", return_value=None):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()

        assert not at.exception
        assert "Transcribing" not in _all_markdown(at)

    def test_running_state_renders_pill(self) -> None:
        """A running state renders the pill with file counts and percentage."""
        files = [
            {"name": "file1.wav", "status": "done", "progress": 1.0},
            {"name": "file2.wav", "status": "running", "progress": 0.5},
        ]
        with patch(
            "ui.run_indicator.load_state", return_value=_state("running", files)
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()

        assert not at.exception
        markdown = _all_markdown(at)
        # 1 of 2 files complete; mean progress (1.0 + 0.5) / 2 = 75%.
        assert "Transcribing 1/2" in markdown
        assert "75%" in markdown

    def test_interrupted_state_shows_warning(self) -> None:
        """An interrupted state surfaces a warning rather than a progress pill."""
        with patch("ui.run_indicator.load_state", return_value=_state("interrupted")):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()

        assert not at.exception
        surfaced = "\n".join(el.value for el in at.warning) + _all_markdown(at)
        assert "interrupt" in surfaced.lower()
        assert "Transcribing" not in surfaced

    def test_finished_state_renders_nothing(self) -> None:
        """A finished run is not an active run — the indicator stays silent."""
        with patch("ui.run_indicator.load_state", return_value=_state("finished")):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()

        assert not at.exception
        assert "Transcribing" not in _all_markdown(at)
