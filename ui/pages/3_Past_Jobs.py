"""ui/pages/3_Past_Jobs.py — Browse and re-download completed transcription runs."""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONSENSUS_DIR  # noqa: E402

st.set_page_config(
    page_title="Past Jobs — Chorus",
    page_icon="🗂",
    layout="wide",
)

st.title("🗂 Past Jobs")
st.caption("Browse completed transcription runs and re-download their outputs.")

# ── File suffix metadata ──────────────────────────────────────────────────────
# Each entry: (filename suffix, display label, MIME type)
# Ordered as they should appear in the per-file grid.
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
    ("_ai_context.md", "AI Context Pack", "text/markdown"),
    ("_bundle.json", "JSON Bundle", "application/json"),
    ("_diarised.md", "Diarised Transcript", "text/markdown"),
]

# Suffixes whose files use the base stem (no run ID in the filename).
# The exporter writes these at the top-level export step before attaching the
# run ID used by the per-run JSON/Markdown artefacts.
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


def _find_base_stem(anchor: Path, full_stem: str) -> str | None:
    """Return the base stem (without run ID) by checking for known export files."""
    for suffix in _BASE_STEM_SUFFIXES:
        for candidate in anchor.parent.glob(f"*{suffix}"):
            base = candidate.name[: -len(suffix)]
            if base and full_stem.startswith(base):
                return base
    return None


def _collect_run_files(anchor: Path) -> dict[str, Path]:
    """Return all on-disk files for this run, keyed by display label."""
    full_stem = anchor.name[: -len("_consensus.md")]
    base_stem = _find_base_stem(anchor, full_stem)

    found: dict[str, Path] = {"Consensus (Markdown)": anchor}

    for suffix, label, _ in _SUFFIX_META:
        if suffix == "_consensus.md":
            continue
        if suffix in _BASE_STEM_SUFFIXES:
            stem = base_stem
        else:
            stem = full_stem
        if stem is None:
            continue
        candidate = anchor.parent / f"{stem}{suffix}"
        if candidate.exists():
            found[label] = candidate

    return found


def _make_zip(run_files: dict[str, Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in run_files.values():
            zf.write(path, arcname=path.name)
    return buf.getvalue()


def _format_mtime(p: Path) -> str:
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%-d %B %Y at %H:%M")


def _parse_run_meta(full_stem: str, base_stem: str | None) -> tuple[str, str, str]:
    """Return (source_name, date_str, time_str) parsed from the stem.

    source_name is the sanitised source filename without extension or run ID.
    Falls back gracefully when the stem does not follow the expected pattern.
    """
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


# ── Main render ───────────────────────────────────────────────────────────────
if not CONSENSUS_DIR.exists():
    st.info(
        "No completed runs found yet. Run a transcription from the main page first."
    )
    st.stop()

anchors = sorted(
    CONSENSUS_DIR.glob("*_consensus.md"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not anchors:
    st.info(
        "No completed runs found yet. Run a transcription from the main page first."
    )
    st.stop()

run_count = len(anchors)
st.markdown(f"**{run_count} completed run{'s' if run_count != 1 else ''} found.**")
st.divider()

for anchor in anchors:
    full_stem = anchor.name[: -len("_consensus.md")]
    base_stem = _find_base_stem(anchor, full_stem)
    run_files = _collect_run_files(anchor)
    source_name, date_str, time_str = _parse_run_meta(full_stem, base_stem)
    mtime = _format_mtime(anchor)

    # Source name leads; date/time is secondary context.
    if source_name and date_str:
        expander_label = f"**{source_name}** — {date_str} {time_str}"
    else:
        expander_label = f"**{full_stem}**"

    with st.expander(expander_label, expanded=False):
        # Source + completion timestamp — visible immediately on expand.
        meta_cols = st.columns([2, 2])
        with meta_cols[0]:
            st.markdown(f"**Source:** {source_name or '—'}")
        with meta_cols[1]:
            st.markdown(f"**Completed:** {mtime}")

        st.caption(
            "Source filename is the sanitised name from the stem. "
            "Exact original filename with extension will be stored in a future release."
        )

        col_zip, _spacer = st.columns([1, 3])
        with col_zip:
            st.download_button(
                "⬇ Download All (ZIP)",
                data=_make_zip(run_files),
                file_name=f"{full_stem}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{full_stem}",
            )

        st.markdown("---")

        items = list(run_files.items())
        for row_start in range(0, len(items), 3):
            cols = st.columns(3)
            for col_idx, (file_label, file_path) in enumerate(
                items[row_start : row_start + 3]
            ):
                mime = "application/octet-stream"
                for suffix, m in _SUFFIX_MIME.items():
                    if file_path.name.endswith(suffix):
                        mime = m
                        break
                with cols[col_idx]:
                    st.download_button(
                        f"⬇ {file_label}",
                        data=file_path.read_bytes(),
                        file_name=file_path.name,
                        mime=mime,
                        use_container_width=True,
                        key=f"dl_{file_path.name}",
                    )
