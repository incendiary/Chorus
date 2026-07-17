"""ui/upload.py — Audio upload area and pipeline-stage overview."""

from __future__ import annotations

import streamlit as st

from config import SUPPORTED_AUDIO_EXTENSIONS


def render_upload() -> list:
    """Render the upload area and stage overview, returning the uploaded files."""
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
        st.subheader("1 · Upload Audio Files")
        uploaded_files = st.file_uploader(
            "Drag and drop or click to browse — multiple files supported",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_AUDIO_EXTENSIONS),
            accept_multiple_files=True,
            help=(
                "Any audio format supported by FFmpeg, including video "
                "containers with an audio track. Upload multiple files to "
                "process them in one session."
            ),
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

    return uploaded_files
