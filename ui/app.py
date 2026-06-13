"""
ui/app.py — Chorus Streamlit Interface.

Provides a clean, single-page web UI for:
  - Uploading one or more audio files (any format supported by ffmpeg)
  - Configuring the Whisper model, language hint, and processing mode
  - Triggering the full Chorus pipeline with live progress feedback
  - Previewing and downloading per-file results
  - Downloading all outputs as a single zip archive
  - Exporting a clean "most likely" plain-text transcript

Run with:
    streamlit run ui/app.py
or via Docker Compose (see README.md).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path when running from the ui/ subdirectory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VARIANT_LABELS, WHISPER_MODEL  # noqa: E402
from export_engine.exporter import (  # noqa: E402
    export_all,
    export_plain_text,
    export_zip,
)
from pipeline_runner import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

# Safe filename pattern — only alphanumeric, hyphens, and underscores retained
_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitise_stem(raw_name: str) -> str:
    """Sanitise a user-supplied filename stem to safe filesystem characters."""
    stem = Path(raw_name).stem
    sanitised = _SAFE_STEM_RE.sub("_", stem).strip("_")
    return sanitised or "upload"


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chorus — Consensus Transcription Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .chorus-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .chorus-header h1 { margin: 0; font-size: 2.4rem; letter-spacing: -0.5px; }
    .chorus-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 1rem; }

    .tier-badge-high   { background:#d4edda; color:#155724; padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }
    .tier-badge-medium { background:#fff3cd; color:#856404; padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }
    .tier-badge-low    { background:#f8d7da; color:#721c24; padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }

    .stProgress > div > div > div { background-color: #0f3460; }
    .metric-card { background:#f8f9fa; border-radius:8px; padding:1rem;
                   text-align:center; border:1px solid #dee2e6; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="chorus-header">
    <h1>🎙️ Chorus</h1>
    <p>Multi-pass consensus audio transcription engine — powered by OpenAI Whisper</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def _hw_recommendation() -> tuple[str, str]:
    """Return (recommended_mode_label, reason_caption) based on available RAM."""
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        cores = psutil.cpu_count(logical=False) or 1
    except Exception:  # noqa: BLE001
        return (
            "Sequential — results appear per file",
            "ℹ️ Could not detect hardware — Sequential is the safer default.",
        )

    if ram_gb < 8:
        return (
            "Sequential — results appear per file",
            f"⚠️ {ram_gb:.1f} GB RAM detected ({cores} cores). "
            "**Sequential recommended** — All-at-once loads all recordings into memory "
            "simultaneously and may cause out-of-memory errors on this machine.",
        )
    if ram_gb < 16:
        return (
            "Sequential — results appear per file",
            f"ℹ️ {ram_gb:.1f} GB RAM / {cores} cores detected. "
            "Sequential is the safer choice for longer recordings. "
            "All-at-once is fine for short clips.",
        )
    return (
        "All at once — results shown at end",
        f"✅ {ram_gb:.1f} GB RAM / {cores} cores detected. Either mode is fine.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Whisper Model")
    model_choice = st.selectbox(
        "Model size",
        options=["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(WHISPER_MODEL),
        help="Larger models are more accurate but slower. 'base' is recommended for local CPU use.",  # noqa: E501
    )

    st.subheader("Language")
    lang_input = st.text_input(
        "Language code (optional)",
        value="",
        placeholder="e.g. en, fr, de — leave blank for auto-detect",
        help="BCP-47 language code. Leave empty for automatic detection.",
    )
    language = lang_input.strip() or None

    st.subheader("Variants")
    st.info(
        f"Chorus always produces **{len(VARIANT_LABELS)} variants**:\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in VARIANT_LABELS.items()),
        icon="ℹ️",
    )

    st.subheader("Advanced Features")
    enable_nlp = st.checkbox(
        "🧠 NLP Reconstruction",
        help="Use spaCy to grammatically reconstruct LOW-confidence tokens.",
    )
    enable_diarisation = st.checkbox(
        "🗣️ Speaker Diarisation",
        help="Identify multiple speakers (requires HUGGINGFACE_TOKEN).",
    )

    st.subheader("Export Formats")
    export_pdf = st.checkbox("PDF Document", value=False)
    export_docx = st.checkbox("Word Document (.docx)", value=False)
    export_srt = st.checkbox("Subtitles (.srt)", value=False)

    st.divider()
    st.markdown("**Confidence Thresholds**")
    st.caption("Configurable in `config.py`")
    st.markdown(
        """
| Tier | Threshold |
|------|-----------|
| 🟢 HIGH   | ≥ 75 % agreement |
| 🟡 MEDIUM | 50 % agreement   |
| 🔴 LOW    | 25 % agreement   |
"""
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area — Upload & Run
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("1 · Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Drag and drop or click to browse — multiple files supported",
        type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac", "webm"],
        accept_multiple_files=True,
        help="Any audio format supported by FFmpeg. Upload multiple files to process them in one session.",  # noqa: E501
    )

with col_info:
    st.subheader("Pipeline Stages")
    st.markdown(
        """
1. 🎛️ **Audio Processing** — 3 cleaning filters applied
2. 🤖 **Transcription** — Whisper runs on each variant
3. 🗳️ **Consensus Merge** — Word-level voting & confidence scoring
4. 📄 **Output** — Annotated Markdown + plain-text transcript
"""
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_files:
    st.divider()
    st.subheader("2 · Run Pipeline")

    # ── Processing mode (only shown for multiple files) ───────────────────────
    rec_mode, rec_reason = _hw_recommendation()

    if len(uploaded_files) > 1:
        mode_choice = st.radio(
            "Processing mode",
            options=[
                "Sequential — results appear per file",
                "All at once — results shown at end",
            ],
            index=0 if rec_mode.startswith("Sequential") else 1,
            horizontal=True,
            help=(
                "**Sequential:** each file is fully processed and its results shown "
                "before the next file starts. Lower peak memory — best for longer "
                "recordings or machines with less RAM.\n\n"
                "**All at once:** all files are processed back-to-back before any "
                "results are displayed. Processing is still single-threaded; the only "
                "difference is when results appear."
            ),
        )
        st.caption(rec_reason)
        sequential = mode_choice.startswith("Sequential")
    else:
        sequential = True

    # ── LOW-word display toggle ───────────────────────────────────────────────
    show_low = st.toggle(
        "Include uncertain words in plain transcript",
        value=True,
        help=(
            "Controls the **Most Likely Transcript** download only — "
            "the annotated consensus document is unaffected.\n\n"
            "**On:** LOW-confidence words appear as `[word?]`.\n"
            "**Off:** LOW-confidence words are omitted entirely.\n\n"
            "Both variants are always included in the Download All zip."
        ),
    )

    # ── Run button ────────────────────────────────────────────────────────────
    n = len(uploaded_files)
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button(
            f"▶ Start Chorus ({n} file{'s' if n > 1 else ''})",
            type="primary",
            use_container_width=True,
        )

    if run_btn:
        os.environ["WHISPER_MODEL"] = model_choice
        os.environ["ENABLE_NLP_RECONSTRUCTION"] = str(enable_nlp).lower()
        os.environ["ENABLE_DIARISATION"] = str(enable_diarisation).lower()

        formats_to_export = [
            fmt
            for fmt, checked in [
                ("pdf", export_pdf),
                ("docx", export_docx),
                ("srt", export_srt),
            ]
            if checked
        ]

        # ── Per-file processing ───────────────────────────────────────────────

        def _run_one_file(
            uf: object,
            progress_slot: object,
            status_slot: object,
            log_lines: list[str],
            log_expander: object,
        ) -> tuple[dict, Path, str]:
            """Process a single uploaded file.

            Returns (results, tmp_path, original_stem).
            """
            original_stem = _sanitise_stem(uf.name)
            suffix = Path(uf.name).suffix.lower()
            # Secure temp file: unique path, exclusive creation, no race condition
            tmp_fd = tempfile.NamedTemporaryFile(
                suffix=suffix, prefix=f"{original_stem}_", delete=False
            )
            tmp_path = Path(tmp_fd.name)
            tmp_fd.write(uf.read())
            tmp_fd.close()

            def _progress(label: str, frac: float) -> None:
                progress_slot.progress(min(frac, 1.0), text=label)
                status_slot.markdown(f"**Status:** {label}")
                log_lines.append(f"`{frac * 100:.0f}%` — {label}")
                with log_expander:
                    st.markdown("\n\n".join(log_lines))

            results = run_pipeline(
                audio_path=tmp_path,
                language=language,
                progress_callback=_progress,
            )
            return results, tmp_path, original_stem

        def _render_file_results(
            filename: str,
            results: dict,
            tmp_path: Path,
            original_stem: str,
        ) -> None:
            """Render the results section for one processed file."""
            transcripts = results["transcripts"]
            consensus_path = results["consensus_path"]
            consensus_text = consensus_path.read_text(encoding="utf-8")

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            for col, (key, meta) in zip(
                [m1, m2, m3, m4], transcripts.items(), strict=False
            ):
                wc = len(meta.get("text", "").split())
                col.metric(VARIANT_LABELS.get(key, key), f"{wc} words")

            # Consensus document preview
            st.markdown("#### 📄 Consensus Transcript")
            st.markdown(consensus_text, unsafe_allow_html=False)

            # ── Download buttons ──────────────────────────────────────────────
            dl_cols = st.columns(3)

            with dl_cols[0]:
                st.download_button(
                    label="⬇️ Consensus (.md)",
                    data=consensus_text,
                    file_name=consensus_path.name,
                    mime="text/markdown",
                    type="primary",
                    key=f"dl_md_{original_stem}",
                )

            with dl_cols[1]:
                plain_path = export_plain_text(
                    consensus_path, original_stem, include_low=show_low
                )
                st.download_button(
                    label="⬇️ Most Likely (.txt)",
                    data=plain_path.read_text(encoding="utf-8"),
                    file_name=plain_path.name,
                    mime="text/plain",
                    key=f"dl_txt_{original_stem}",
                )

            with dl_cols[2]:
                with st.spinner("Building zip…"):
                    zip_bytes = export_zip(
                        consensus_path,
                        transcripts["original"],
                        original_stem,
                        include_formats=formats_to_export or None,
                    )
                st.download_button(
                    label="⬇️ Download All (.zip)",
                    data=zip_bytes,
                    file_name=f"{original_stem}_chorus_all.zip",
                    mime="application/zip",
                    key=f"dl_zip_{original_stem}",
                )

            # Additional format exports
            if formats_to_export:
                st.markdown("##### 📥 Additional Exports")
                with st.spinner("Generating exports…"):
                    export_paths = export_all(
                        consensus_path,
                        transcripts["original"],
                        original_stem,
                        formats_to_export,
                    )
                fmt_cols = st.columns(len(formats_to_export))
                for col, fmt in zip(fmt_cols, formats_to_export, strict=False):
                    epath = export_paths.get(fmt)
                    if epath and epath.exists():
                        with col:
                            with open(epath, "rb") as f:
                                st.download_button(
                                    label=f"⬇️ .{fmt.upper()}",
                                    data=f,
                                    file_name=epath.name,
                                    mime="application/octet-stream",
                                    key=f"dl_{fmt}_{original_stem}",
                                )

            # Speaker diarisation
            if results.get("diarised_path") and results["diarised_path"].exists():
                st.markdown("##### 🗣️ Speaker Diarisation")
                diar_text = results["diarised_path"].read_text(encoding="utf-8")
                with st.expander("Preview Diarised Transcript"):
                    st.markdown(diar_text)
                st.download_button(
                    label="⬇️ Diarised Transcript (.md)",
                    data=diar_text,
                    file_name=results["diarised_path"].name,
                    mime="text/markdown",
                    key=f"dl_diar_{original_stem}",
                )

            # Individual variant transcripts
            st.markdown("##### 🔍 Individual Variant Transcripts")
            tabs = st.tabs([VARIANT_LABELS.get(k, k) for k in transcripts])
            for tab, (key, meta) in zip(tabs, transcripts.items(), strict=False):
                with tab:
                    st.markdown(
                        f"**Detected language:** `{meta.get('language', 'unknown')}`"
                    )
                    st.markdown(f"**Model:** `{meta.get('model', 'unknown')}`")
                    st.text_area(
                        "Transcript text",
                        value=meta.get("text", "").strip(),
                        height=200,
                        key=f"transcript_{key}_{original_stem}",
                    )
                    st.download_button(
                        label=f"⬇️ {original_stem}_{key}.json",
                        data=json.dumps(meta, ensure_ascii=False, indent=2),
                        file_name=f"{original_stem}_{key}.json",
                        mime="application/json",
                        key=f"dl_json_{key}_{original_stem}",
                    )

        # ── Main run loop ─────────────────────────────────────────────────────

        st.divider()
        st.subheader("3 · Results")

        if sequential:
            # Process and render each file as it completes
            for uf in uploaded_files:
                with st.expander(f"📄 {uf.name}", expanded=True):
                    progress_bar = st.progress(0.0, text="Initialising…")
                    status_text = st.empty()
                    log_expander = st.expander("📋 Live log", expanded=False)
                    log_lines: list[str] = []

                    tmp_path: Path | None = None
                    try:
                        results, tmp_path, original_stem = _run_one_file(
                            uf, progress_bar, status_text, log_lines, log_expander
                        )
                        progress_bar.progress(1.0, text="✅ Complete!")
                        status_text.success(
                            f"Completed in **{results['elapsed_seconds']} s**"
                        )
                        _render_file_results(uf.name, results, tmp_path, original_stem)
                    except Exception as exc:
                        st.error(f"Pipeline failed: {exc}")
                        logger.exception("Pipeline error for %s", uf.name)
                    finally:
                        if tmp_path is not None:
                            tmp_path.unlink(missing_ok=True)

        else:
            # Process all files first, collect results
            all_results: list[tuple[object, dict, Path, str]] = []
            overall = st.progress(0.0, text="Starting…")

            for idx, uf in enumerate(uploaded_files):
                overall.progress(
                    idx / len(uploaded_files),
                    text=f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})…",
                )
                progress_bar = st.empty()
                status_text = st.empty()
                log_expander = st.expander(f"📋 Live log — {uf.name}", expanded=False)
                log_lines = []

                tmp_path: Path | None = None
                try:
                    results, tmp_path, original_stem = _run_one_file(
                        uf, progress_bar, status_text, log_lines, log_expander
                    )
                    all_results.append((uf, results, tmp_path, original_stem))
                except Exception as exc:
                    st.error(f"Pipeline failed for {uf.name}: {exc}")
                    logger.exception("Pipeline error for %s", uf.name)
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)

            overall.progress(1.0, text=f"✅ All {len(uploaded_files)} files complete!")

            # Now render all results
            for uf, results, tmp_path, original_stem in all_results:
                with st.expander(f"📄 {uf.name}", expanded=True):
                    st.success(f"Completed in **{results['elapsed_seconds']} s**")
                    _render_file_results(uf.name, results, tmp_path, original_stem)

else:
    st.info("Upload one or more audio files above to begin.", icon="👆")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Chorus Engine · Powered by [OpenAI Whisper](https://github.com/openai/whisper), "
    "librosa, NLTK, and Streamlit · "
    "All processing is performed locally — no audio data leaves your machine."
)
