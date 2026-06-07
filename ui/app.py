"""
ui/app.py — Chorus Streamlit Interface.

Provides a clean, single-page web UI for:
  - Uploading an audio file (any format supported by ffmpeg)
  - Configuring the Whisper model, language hint, and number of variants
  - Triggering the full Chorus pipeline with live progress feedback
  - Previewing and downloading the consensus Markdown document
  - Inspecting individual variant transcripts

Run with:
    streamlit run ui/app.py
or via Docker Compose (see README.md).
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path when running from the ui/ subdirectory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VARIANT_LABELS, WHISPER_MODEL  # noqa: E402
from pipeline_runner import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

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
    st.subheader("1 · Upload Audio File")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac", "webm"],
        help="Any audio format supported by FFmpeg is accepted.",
    )

with col_info:
    st.subheader("Pipeline Stages")
    st.markdown(
        """
1. 🎛️ **Audio Processing** — 3 cleaning filters applied
2. 🤖 **Transcription** — Whisper runs on each variant
3. 🗳️ **Consensus Merge** — Word-level voting & confidence scoring
4. 📄 **Output** — Annotated Markdown document
"""
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    st.divider()
    st.subheader("2 · Run Pipeline")

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶ Start Chorus", type="primary", use_container_width=True)

    if run_btn:
        # Save uploaded file to a temp location
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        # Override settings in environment for this run
        import os

        os.environ["WHISPER_MODEL"] = model_choice
        os.environ["ENABLE_NLP_RECONSTRUCTION"] = str(enable_nlp).lower()
        os.environ["ENABLE_DIARISATION"] = str(enable_diarisation).lower()

        # Progress tracking
        progress_bar = st.progress(0.0, text="Initialising…")
        status_text = st.empty()
        log_expander = st.expander("📋 Live log", expanded=False)
        log_lines: list = []

        def _ui_progress(label: str, frac: float) -> None:
            progress_bar.progress(min(frac, 1.0), text=label)
            status_text.markdown(f"**Status:** {label}")
            log_lines.append(f"`{frac*100:.0f}%` — {label}")
            with log_expander:
                st.markdown("\n\n".join(log_lines))

        try:
            results = run_pipeline(
                audio_path=tmp_path,
                language=language,
                progress_callback=_ui_progress,
            )

            progress_bar.progress(1.0, text="✅ Pipeline complete!")
            status_text.success(f"Completed in **{results['elapsed_seconds']} s**")

            # ── Results ──────────────────────────────────────────────────────
            st.divider()
            st.subheader("3 · Results")

            # Metrics row
            transcripts = results["transcripts"]
            m1, m2, m3, m4 = st.columns(4)
            for col, (key, meta) in zip(
                [m1, m2, m3, m4], transcripts.items(), strict=False
            ):  # noqa: E501
                wc = len(meta.get("text", "").split())
                col.metric(VARIANT_LABELS.get(key, key), f"{wc} words")

            # Consensus document
            st.subheader("📄 Consensus Transcript")
            consensus_path = results["consensus_path"]
            consensus_text = consensus_path.read_text(encoding="utf-8")

            st.markdown(consensus_text, unsafe_allow_html=False)

            st.download_button(
                label="⬇️ Download Consensus (.md)",
                data=consensus_text,
                file_name=consensus_path.name,
                mime="text/markdown",
                type="primary",
            )

            # Handle advanced exports
            formats_to_export = []
            if export_pdf:
                formats_to_export.append("pdf")
            if export_docx:
                formats_to_export.append("docx")
            if export_srt:
                formats_to_export.append("srt")

            if formats_to_export:
                st.markdown("### 📥 Additional Exports")
                from export_engine.exporter import export_all

                with st.spinner("Generating exports…"):
                    export_paths = export_all(
                        consensus_path,
                        transcripts["original"],
                        tmp_path.stem,
                        formats_to_export,
                    )

                cols = st.columns(len(formats_to_export))
                for col, fmt in zip(cols, formats_to_export, strict=False):
                    epath = export_paths.get(fmt)
                    if epath and epath.exists():
                        with col:
                            with open(epath, "rb") as f:
                                st.download_button(
                                    label=f"Download .{fmt.upper()}",
                                    data=f,
                                    file_name=epath.name,
                                    mime="application/octet-stream",
                                )

            if results.get("diarised_path") and results["diarised_path"].exists():
                st.markdown("### 🗣️ Speaker Diarisation")
                diar_text = results["diarised_path"].read_text(encoding="utf-8")
                with st.expander("Preview Diarised Transcript"):
                    st.markdown(diar_text)
                st.download_button(
                    label="⬇️ Download Diarised Transcript (.md)",
                    data=diar_text,
                    file_name=results["diarised_path"].name,
                    mime="text/markdown",
                )

            # Individual variant transcripts
            st.divider()
            st.subheader("🔍 Individual Variant Transcripts")
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
                        key=f"transcript_{key}",
                    )
                    st.download_button(
                        label=f"⬇️ Download {key}.json",
                        data=json.dumps(meta, ensure_ascii=False, indent=2),
                        file_name=f"{tmp_path.stem}_{key}.json",
                        mime="application/json",
                        key=f"dl_{key}",
                    )

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            logger.exception("Pipeline error")
        finally:
            tmp_path.unlink(missing_ok=True)

else:
    st.info("Upload an audio file above to begin.", icon="👆")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Chorus Engine · Powered by [OpenAI Whisper](https://github.com/openai/whisper), "
    "librosa, NLTK, and Streamlit · "
    "All processing is performed locally — no audio data leaves your machine."
)
