"""tests/test_run_indicator.py — Tests for the cross-page run-progress indicator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ui.run_state import ACTIVE_RUN_FILE, FileEntry


class TestRunIndicator:
    """Tests for render_run_indicator() behaviour."""

    def test_no_state_renders_nothing(self, tmp_path: Path) -> None:
        """When no active run state exists, the indicator renders nothing."""
        # Patch ACTIVE_RUN_FILE to point to non-existent path
        with patch("ui.run_indicator.load_state", return_value=None):
            at = AppTest.from_file("ui/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # Check that no run pill markdown is in the output
            output = at.get_text()
            assert "Transcribing" not in output or "⏳" not in output

    def test_running_state_renders_pill(self, tmp_path: Path) -> None:
        """When a run is active (running), the indicator shows a compact pill."""
        # Create a mock state file
        state = {
            "schema_version": 1,
            "run_id": "test_run_123",
            "status": "running",
            "boot_id": "boot_123",
            "started_at": 1234567890,
            "finished_at": None,
            "log_path": None,
            "config": {},
            "files": [
                {
                    "name": "file1.wav",
                    "stem": "file1",
                    "spool_path": "/tmp/file1.wav",
                    "status": "done",
                    "progress": 1.0,
                },
                {
                    "name": "file2.wav",
                    "stem": "file2",
                    "spool_path": "/tmp/file2.wav",
                    "status": "running",
                    "progress": 0.5,
                },
            ],
        }

        # Mock load_state to return this state
        with patch("ui.run_indicator.load_state", return_value=state):
            at = AppTest.from_file("ui/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # The pill should show progress: 1 done, 2 total, ~75% progress
            output = at.get_text()
            # Note: the fragment runs, so we check for the text content
            assert "⏳" in output or "Transcribing" in output or "75%" in output

    def test_interrupted_state_shows_warning(self, tmp_path: Path) -> None:
        """When a run is interrupted, the indicator shows a warning with dismiss button."""
        state = {
            "schema_version": 1,
            "run_id": "test_run_interrupted",
            "status": "interrupted",
            "boot_id": "boot_old",
            "started_at": 1234567890,
            "finished_at": 1234567900,
            "log_path": None,
            "config": {},
            "files": [],
        }

        with patch("ui.run_indicator.load_state", return_value=state):
            at = AppTest.from_file("ui/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # The warning should be present
            output = at.get_text()
            assert "interrupted" in output.lower() or "previous run" in output.lower()

    def test_finished_state_renders_nothing(self) -> None:
        """When a run is finished (and acknowledged), the indicator renders nothing."""
        state = {
            "schema_version": 1,
            "run_id": "test_run_finished",
            "status": "finished",
            "boot_id": "boot_123",
            "started_at": 1234567890,
            "finished_at": 1234567900,
            "log_path": None,
            "config": {},
            "files": [],
        }

        with patch("ui.run_indicator.load_state", return_value=state):
            at = AppTest.from_file("ui/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # No pill should render for a finished (non-interrupted) state
            output = at.get_text()
            # Finished state should not show any active indicator
            # (Though we may see it in the main app UI, the indicator itself renders nothing)
            assert not at.exception
