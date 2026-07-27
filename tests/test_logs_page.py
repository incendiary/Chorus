"""tests/test_logs_page.py — Tests for the disk-backed Logs page viewer.

The page reads run logs from ``outputs/runs/<run_id>/run.log`` rather than the
session-state buffer, so an overnight run is still inspectable the next
morning. ``RUNS_DIR`` is patched at its source (``ui.run_state``) *before* the
page module is loaded by ``AppTest``, because the page does
``from ui.run_state import RUNS_DIR`` — patching the page's own namespace is
impossible by name, as ``2_Logs`` is not a valid Python identifier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

LOGS_PAGE = "ui/pages/2_Logs.py"


def _write_log(runs_dir: Path, run_id: str, lines: list[str]) -> Path:
    """Create ``runs_dir/run_id/run.log`` containing *lines*."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_file


def _run_page(runs_dir: Path) -> AppTest:
    """Run the Logs page with ``RUNS_DIR`` pointed at *runs_dir*."""
    with patch("ui.run_state.RUNS_DIR", runs_dir):
        at = AppTest.from_file(LOGS_PAGE, default_timeout=30)
        at.run()
    return at


def _rendered_text(at: AppTest) -> str:
    """All text the page rendered, joined for substring assertions."""
    parts: list[str] = [el.value for el in at.markdown]
    parts += [el.value for el in at.info]
    parts += [el.value for el in at.code]
    return "\n".join(str(p) for p in parts)


class TestLogsPageDiskBacked:
    def test_empty_state_no_logs(self, tmp_path: Path) -> None:
        """With no run logs on disk, a clear empty-state message is shown."""
        at = _run_page(tmp_path / "nonexistent_runs")

        assert not at.exception
        assert "No run logs found" in _rendered_text(at)

    def test_crafted_log_renders_newest_first(self, tmp_path: Path) -> None:
        """Entries render newest-first, not in file order."""
        runs_dir = tmp_path / "runs"
        _write_log(
            runs_dir,
            "test_run_001",
            [
                "10:00:00  INFO     whisper_engine — Starting transcription",
                "10:00:01  INFO     consensus_merger — Merging variants",
                "10:00:02  INFO     export_engine — Exporting to PDF",
            ],
        )

        at = _run_page(runs_dir)
        assert not at.exception
        text = _rendered_text(at)

        assert "Exporting to PDF" in text
        assert "Starting transcription" in text
        # Newest-first: the last log line must appear above the first one.
        assert text.index("Exporting to PDF") < text.index("Starting transcription")

    def test_timestamp_stripped_dedup_collapses_repeats(self, tmp_path: Path) -> None:
        """Consecutive identical lines (ignoring timestamps) collapse to one row."""
        runs_dir = tmp_path / "runs"
        _write_log(
            runs_dir,
            "test_run_002",
            [
                "10:00:00  INFO     whisper_engine — Processing segment 1",
                "10:00:01  INFO     whisper_engine — Processing segment 1",
                "10:00:02  INFO     whisper_engine — Processing segment 1",
                "10:00:03  INFO     export_engine — Exporting to PDF",
            ],
        )

        at = _run_page(runs_dir)
        assert not at.exception
        text = _rendered_text(at)

        # Three messages differing only by timestamp collapse to a single row
        # carrying a repeat badge.
        assert text.count("Processing segment 1") == 1
        assert "3" in text

    def test_tail_n_respected(self, tmp_path: Path) -> None:
        """Only the newest N entries render, per the tail-N control."""
        runs_dir = tmp_path / "runs"
        _write_log(
            runs_dir,
            "test_run_003",
            [f"10:00:{i % 60:02d}  INFO     test — Line {i}" for i in range(100)],
        )

        at = _run_page(runs_dir)
        assert not at.exception
        text = _rendered_text(at)

        # Default tail is 50: the newest entry is present, the oldest is not.
        assert "Line 99" in text
        assert "Line 5\n" not in text
        assert "Line 5 " not in text
