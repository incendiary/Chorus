"""tests/test_past_jobs_page.py — Tests for the Past Jobs page's dual-layout listing.

The page must list runs from the new per-job layout
(``JOBS_DIR/<stem>-<sha8>/<timestamp>/consensus/``) and, separately, legacy
flat-layout runs from ``CONSENSUS_DIR`` under a "Legacy (before v5.0.0)"
section — see CLAUDE.md's per-job output folders work item. ``CONSENSUS_DIR``
and ``JOBS_DIR`` are patched at their source (``config``) *before* the page
module is loaded by ``AppTest``, because the page does
``from config import CONSENSUS_DIR, JOBS_DIR`` — patching the page's own
namespace is impossible by name, as ``3_Past_Jobs`` is not a valid Python
identifier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

PAST_JOBS_PAGE = str(
    Path(__file__).resolve().parent.parent / "ui" / "pages" / "3_Past_Jobs.py"
)


def _run_page(jobs_dir: Path, consensus_dir: Path) -> AppTest:
    with (
        patch("config.JOBS_DIR", jobs_dir),
        patch("config.CONSENSUS_DIR", consensus_dir),
    ):
        at = AppTest.from_file(PAST_JOBS_PAGE, default_timeout=30)
        at.run()
    return at


def _rendered_text(at: AppTest) -> str:
    parts: list[str] = [el.value for el in at.markdown]
    parts += [el.value for el in at.info]
    parts += [str(el.label) for el in at.expander]
    parts += [el.value for el in at.subheader]
    return "\n".join(str(p) for p in parts)


def _make_current_layout_run(jobs_dir: Path, stem: str) -> Path:
    """Create a minimal new-layout run: <jobs_dir>/<stem>-abcd1234/<ts>/consensus/."""
    run_dir = jobs_dir / f"{stem}-abcd1234" / "20260101-120000"
    consensus_subdir = run_dir / "consensus"
    consensus_subdir.mkdir(parents=True)
    (consensus_subdir / f"{stem}_consensus.md").write_text(
        "# Consensus\nHello world.", encoding="utf-8"
    )
    return consensus_subdir


def _make_legacy_run(consensus_dir: Path, stem: str) -> Path:
    consensus_dir.mkdir(parents=True, exist_ok=True)
    anchor = consensus_dir / f"{stem}_consensus.md"
    anchor.write_text("# Legacy consensus\nOld run.", encoding="utf-8")
    return anchor


class TestPastJobsEmptyState:
    def test_no_runs_shows_empty_state(self, tmp_path: Path) -> None:
        at = _run_page(tmp_path / "jobs", tmp_path / "consensus")
        assert not at.exception
        assert "No completed runs found" in _rendered_text(at)


class TestPastJobsCurrentLayout:
    def test_new_layout_run_is_listed(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        _make_current_layout_run(jobs_dir, "interview")

        at = _run_page(jobs_dir, tmp_path / "consensus_empty")

        assert not at.exception
        text = _rendered_text(at)
        assert "interview" in text
        assert "Legacy (before v5.0.0)" not in text


class TestPastJobsLegacyLayout:
    def test_legacy_flat_file_is_listed_under_legacy_section(
        self, tmp_path: Path
    ) -> None:
        consensus_dir = tmp_path / "consensus"
        _make_legacy_run(consensus_dir, "2024-01-01_10-00-00_old_recording")

        at = _run_page(tmp_path / "jobs_empty", consensus_dir)

        assert not at.exception
        text = _rendered_text(at)
        assert "Legacy (before v5.0.0)" in text
        assert "old recording" in text


class TestPastJobsBothLayouts:
    def test_new_and_legacy_runs_both_listed(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        consensus_dir = tmp_path / "consensus"
        _make_current_layout_run(jobs_dir, "interview")
        _make_legacy_run(consensus_dir, "2024-01-01_10-00-00_old_recording")

        at = _run_page(jobs_dir, consensus_dir)

        assert not at.exception
        text = _rendered_text(at)
        assert "interview" in text
        assert "Legacy (before v5.0.0)" in text
        assert "old recording" in text
