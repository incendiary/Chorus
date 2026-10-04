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


def _delete_via_ui(at: AppTest, key_suffix: str) -> AppTest:
    at.button(key=f"del_btn_{key_suffix}").click().run()
    at.button(key=f"confirm_yes_{key_suffix}").click().run()
    return at


class TestPastJobsDelete:
    def test_current_layout_delete_removes_whole_run_dir_and_empty_project(
        self, tmp_path: Path
    ) -> None:
        jobs_dir = tmp_path / "jobs"
        consensus_subdir = _make_current_layout_run(jobs_dir, "interview")
        run_dir = consensus_subdir.parent
        (run_dir / "variants").mkdir()
        (run_dir / "variants" / "original.wav").write_bytes(b"x")
        (run_dir / "transcripts").mkdir()
        (run_dir / "transcripts" / "original.json").write_text("{}")
        project_dir = run_dir.parent
        (project_dir / "interview_speakers.json").write_text("{}")

        at = _run_page(jobs_dir, tmp_path / "consensus_empty")
        _delete_via_ui(at, "interview-abcd1234_20260101-120000")

        assert not at.exception
        assert not run_dir.exists()
        assert not project_dir.exists()

    def test_current_layout_delete_keeps_project_sidecar_while_other_run_remains(
        self, tmp_path: Path
    ) -> None:
        jobs_dir = tmp_path / "jobs"
        consensus_subdir = _make_current_layout_run(jobs_dir, "interview")
        run_dir = consensus_subdir.parent
        project_dir = run_dir.parent
        other = project_dir / "20260102-120000" / "consensus"
        other.mkdir(parents=True)
        (other / "interview_consensus.md").write_text("# Other")
        sidecar = project_dir / "interview_speakers.json"
        sidecar.write_text("{}")

        at = _run_page(jobs_dir, tmp_path / "consensus_empty")
        _delete_via_ui(at, "interview-abcd1234_20260101-120000")

        assert not at.exception
        assert not run_dir.exists()
        assert (other / "interview_consensus.md").exists()
        assert sidecar.exists()

    def test_legacy_delete_removes_only_its_own_files(self, tmp_path: Path) -> None:
        consensus_dir = tmp_path / "consensus"
        stem = "2024-01-01_10-00-00_old_recording"
        anchor = _make_legacy_run(consensus_dir, stem)
        bystander = consensus_dir / "2024-02-02_10-00-00_other_consensus.md"
        bystander.write_text("# Other")

        at = _run_page(tmp_path / "jobs_empty", consensus_dir)
        _delete_via_ui(at, f"legacy_{stem}")

        assert not at.exception
        assert not anchor.exists()
        assert bystander.exists()
        assert consensus_dir.exists()
