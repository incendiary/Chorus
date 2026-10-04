"""ui/pages/3_Past_Jobs.py — Browse and re-download completed transcription runs."""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONSENSUS_DIR, JOBS_DIR  # noqa: E402
from ui.run_indicator import render_run_indicator  # noqa: E402

st.set_page_config(
    page_title="Past Jobs — Chorus",
    page_icon="🗂",
    layout="wide",
)

render_run_indicator(is_subpage=True)

st.title("🗂 Past Jobs")
st.caption("Browse completed transcription runs and re-download their outputs.")

# ── File suffix metadata ──────────────────────────────────────────────────────
# Each entry: (filename suffix, display label, MIME type). Under the current
# per-job layout, every file for a run shares the same stem and lives in
# that run's own consensus/ directory, so no base-stem guessing is needed.
_SUFFIX_META: list[tuple[str, str, str]] = [
    ("_consensus.md", "Consensus (Markdown)", "text/markdown"),
    ("_consensus.pdf", "PDF", "application/pdf"),
    (
        "_consensus.docx",
        "DOCX",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ("_consensus.srt", "SRT Subtitles", "text/plain"),
    ("_consensus.vtt", "VTT Subtitles", "text/plain"),
    ("_most_likely.txt", "Most Likely (Plain Text)", "text/plain"),
    ("_most_likely_clean.txt", "Most Likely (Clean)", "text/plain"),
    ("_best_guess.txt", "Best Guess (Plain Text)", "text/plain"),
    ("_ai_context.md", "AI Context Pack", "text/markdown"),
    ("_bundle.json", "JSON Bundle", "application/json"),
    ("_diarised.md", "Diarised Transcript", "text/markdown"),
]

# Suffixes whose files use the base stem (no run ID in the filename) — kept
# for the legacy flat-layout section only.
_BASE_STEM_SUFFIXES: frozenset[str] = frozenset(
    {
        "_consensus.pdf",
        "_consensus.docx",
        "_consensus.srt",
        "_consensus.vtt",
        "_most_likely.txt",
        "_most_likely_clean.txt",
    }
)

_SUFFIX_MIME: dict[str, str] = {s: m for s, _, m in _SUFFIX_META}


def _make_zip(run_files: dict[str, Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in run_files.values():
            zf.write(path, arcname=path.name)
    return buf.getvalue()


def _format_mtime(p: Path) -> str:
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%-d %B %Y at %H:%M")


def _delete_run(run_files: dict[str, Path], run_dir: Path | None = None) -> None:
    """Delete a run.

    A current-layout run (*run_dir* given) loses its whole directory, including
    variants/ and transcripts/; its project directory goes too once no other
    run remains, taking the project-level speaker names sidecar with it. A
    legacy flat-layout run (no *run_dir*) loses only its own files.
    """
    if run_dir is None:
        for path in run_files.values():
            path.unlink(missing_ok=True)
        return
    shutil.rmtree(run_dir, ignore_errors=True)
    project_dir = run_dir.parent
    if not any(p.is_dir() for p in project_dir.iterdir()):
        shutil.rmtree(project_dir, ignore_errors=True)


def _render_run_expander(
    *,
    expander_label: str,
    source_name: str,
    mtime_display: str,
    run_files: dict[str, Path],
    zip_filename: str,
    dedupe_key: str,
    run_dir: Path | None = None,
) -> None:
    """Render one run's expander: metadata, delete, and download controls.

    Shared between the current-layout and legacy sections so both look and
    behave identically to the user.
    """
    confirm_key = f"confirm_delete_{dedupe_key}"

    with st.expander(expander_label, expanded=False):
        meta_cols = st.columns([2, 2, 1])
        with meta_cols[0]:
            st.markdown(f"**Source:** {source_name or '—'}")
        with meta_cols[1]:
            st.markdown(f"**Completed:** {mtime_display}")
        with meta_cols[2]:
            if st.button(
                "🗑 Delete", key=f"del_btn_{dedupe_key}", use_container_width=True
            ):
                st.session_state[confirm_key] = True

        if st.session_state.get(confirm_key):
            st.warning(
                f"Delete **{len(run_files)} file{'s' if len(run_files) != 1 else ''}** "
                "for this run? This cannot be undone."
            )
            col_yes, col_no, _ = st.columns([1, 1, 2])
            with col_yes:
                if st.button(
                    "Yes, delete",
                    key=f"confirm_yes_{dedupe_key}",
                    type="primary",
                    use_container_width=True,
                ):
                    _delete_run(run_files, run_dir)
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            with col_no:
                if st.button(
                    "Cancel",
                    key=f"confirm_no_{dedupe_key}",
                    use_container_width=True,
                ):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            return

        col_zip, _spacer = st.columns([1, 3])
        with col_zip:
            st.download_button(
                "⬇ Download All (ZIP)",
                data=_make_zip(run_files),
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{dedupe_key}",
            )

        st.markdown("---")

        items = list(run_files.items())
        for row_start in range(0, len(items), 3):
            cols = st.columns(3)
            for col_idx, (file_label, file_path) in enumerate(
                items[row_start : row_start + 3]
            ):
                mime = next(
                    (m for s, m in _SUFFIX_MIME.items() if file_path.name.endswith(s)),
                    "application/octet-stream",
                )
                with cols[col_idx]:
                    st.download_button(
                        f"⬇ {file_label}",
                        data=file_path.read_bytes(),
                        file_name=file_path.name,
                        mime=mime,
                        use_container_width=True,
                        key=f"dl_{dedupe_key}_{file_path.name}",
                    )


# ── Current layout: JOBS_DIR/<stem>-<sha8>/<timestamp>/consensus/ ─────────────
# Each run's own consensus/ directory holds every file for that run under one
# shared stem, so there is no base-stem guessing to do: just glob it.


@dataclass(slots=True)
class _Run:
    project_name: str
    run_stamp: str
    anchor: Path
    files: dict[str, Path] = field(default_factory=dict)
    mtime: float = field(init=False)

    def __post_init__(self) -> None:
        self.mtime = self.anchor.stat().st_mtime


def _collect_current_layout_runs() -> list[_Run]:
    runs: list[_Run] = []
    if not JOBS_DIR.exists():
        return runs

    for project_dir in JOBS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            consensus_subdir = run_dir / "consensus"
            anchors = sorted(consensus_subdir.glob("*_consensus.md"))
            if not anchors:
                continue
            anchor = anchors[0]
            full_stem = anchor.name[: -len("_consensus.md")]
            run = _Run(project_dir.name, run_dir.name, anchor)
            run.files["Consensus (Markdown)"] = anchor
            for suffix, label, _ in _SUFFIX_META:
                if suffix == "_consensus.md":
                    continue
                candidate = consensus_subdir / f"{full_stem}{suffix}"
                if candidate.exists():
                    run.files[label] = candidate
            runs.append(run)

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def _run_source_name(run: _Run) -> str:
    """Derive a human-readable source name from the project directory name
    (``<stem>-<sha8>``), stripping the trailing ``-<sha8>`` suffix."""
    return run.project_name.rsplit("-", 1)[0].replace("_", " ")


current_runs = _collect_current_layout_runs()

# ── Legacy layout: flat CONSENSUS_DIR/*_consensus.md ───────────────────────


def _find_base_stem(anchor: Path, full_stem: str) -> str | None:
    """Return the base stem (without run ID) by checking for known export files."""
    for suffix in _BASE_STEM_SUFFIXES:
        for candidate in anchor.parent.glob(f"*{suffix}"):
            base = candidate.name[: -len(suffix)]
            if base and full_stem.startswith(base):
                return base
    return None


def _collect_legacy_run_files(anchor: Path) -> dict[str, Path]:
    """Return all on-disk files for this legacy flat-layout run."""
    full_stem = anchor.name[: -len("_consensus.md")]
    base_stem = _find_base_stem(anchor, full_stem)

    found: dict[str, Path] = {"Consensus (Markdown)": anchor}

    for suffix, label, _ in _SUFFIX_META:
        if suffix == "_consensus.md":
            continue
        stem = base_stem if suffix in _BASE_STEM_SUFFIXES else full_stem
        if stem is None:
            continue
        candidate = anchor.parent / f"{stem}{suffix}"
        if candidate.exists():
            found[label] = candidate

    return found


def _parse_legacy_run_meta(
    full_stem: str, base_stem: str | None
) -> tuple[str, str, str]:
    """Return (source_name, date_str, time_str) parsed from the stem."""
    parts = full_stem.split("_", 2)
    if (
        len(parts) >= 2
        and len(parts[0]) == 10
        and parts[0][4] == "-"
        and parts[0][7] == "-"
    ):
        date_str = parts[0]
        time_str = parts[1].replace("-", ":")
        if base_stem:
            prefix = f"{date_str}_{parts[1]}_"
            source = base_stem[len(prefix) :].replace("_", " ").strip()
        elif len(parts) > 2:
            source = parts[2].replace("_", " ").strip()
        else:
            source = ""
        return source, date_str, time_str
    return full_stem.replace("_", " "), "", ""


legacy_anchors = (
    sorted(
        CONSENSUS_DIR.glob("*_consensus.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if CONSENSUS_DIR.exists()
    else []
)

# ── Main render ───────────────────────────────────────────────────────────────
if not current_runs and not legacy_anchors:
    st.info(
        "No completed runs found yet. Run a transcription from the main page first."
    )
    st.stop()

total_count = len(current_runs) + len(legacy_anchors)
st.markdown(f"**{total_count} completed run{'s' if total_count != 1 else ''} found.**")
st.divider()

# Show in-progress banner if a run is active
from ui.run_state import load_state  # noqa: E402

_state = load_state()
if _state and _state.get("status") == "running":
    st.info("🔄 1 run in progress — updates appear below as it completes.", icon="🔄")

# ── Group current-layout runs by date ───────────────────────────────────────
by_date: dict[date, list[_Run]] = defaultdict(list)
for run in current_runs:
    by_date[datetime.fromtimestamp(run.mtime).date()].append(run)

for date_key in sorted(by_date, reverse=True):
    st.subheader(date_key.strftime("%-d %B %Y"))

    for run in by_date[date_key]:
        source_name = _run_source_name(run)
        time_str = datetime.fromtimestamp(run.mtime).strftime("%H:%M")
        expander_label = f"**{source_name}** — {time_str}"
        dedupe_key = f"{run.project_name}_{run.run_stamp}"

        _render_run_expander(
            expander_label=expander_label,
            source_name=source_name,
            mtime_display=_format_mtime(run.anchor),
            run_files=run.files,
            zip_filename=f"{run.project_name}_{run.run_stamp}.zip",
            dedupe_key=dedupe_key,
            run_dir=run.anchor.parent.parent,
        )

    st.divider()

# ── Legacy section (flat outputs/consensus/, pre-v5.0.0) ───────────────────
if legacy_anchors:
    st.subheader("Legacy (before v5.0.0)")
    st.caption(
        "Runs from before per-job output folders. Files live in the shared "
        "outputs/consensus/ directory and are listed here so existing "
        "casework stays reachable."
    )

    for anchor in legacy_anchors:
        full_stem = anchor.name[: -len("_consensus.md")]
        base_stem = _find_base_stem(anchor, full_stem)
        run_files = _collect_legacy_run_files(anchor)
        source_name, _, time_str = _parse_legacy_run_meta(full_stem, base_stem)
        mtime = _format_mtime(anchor)

        if source_name and time_str:
            expander_label = f"**{source_name}** — {time_str}"
        else:
            expander_label = f"**{full_stem}**"

        _render_run_expander(
            expander_label=expander_label,
            source_name=source_name,
            mtime_display=mtime,
            run_files=run_files,
            zip_filename=f"{full_stem}.zip",
            dedupe_key=f"legacy_{full_stem}",
        )

    st.divider()
