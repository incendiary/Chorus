"""ui/pages/2_Logs.py — In-app log viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="Logs — Chorus",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Logs")

_LOG_BUFFER_KEY = "log_buffer"

buf: list[dict] = st.session_state.get(_LOG_BUFFER_KEY, [])

col_filter, col_clear, col_download = st.columns([2, 1, 1])

with col_filter:
    level_filter = st.selectbox(
        "Filter by level",
        options=["ALL", "INFO", "WARNING", "ERROR"],
        index=0,
        label_visibility="collapsed",
    )

with col_clear:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state[_LOG_BUFFER_KEY] = []
        st.rerun()

filtered = (
    buf if level_filter == "ALL" else [r for r in buf if r["level"] == level_filter]
)

with col_download:
    if filtered:
        log_text = "\n".join(
            f"{r['time']}  {r['level']:<8}  {r['logger']} — {r['message']}"
            for r in filtered
        )
        st.download_button(
            "⬇ Download",
            data=log_text,
            file_name="chorus_logs.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.button("⬇ Download", disabled=True, use_container_width=True)

st.divider()

if not filtered:
    st.info(
        "No log entries yet. Run a transcription and check back here."
        if not buf
        else f"No {level_filter} entries in the current buffer."
    )
else:
    level_colours = {
        "ERROR": "🔴",
        "WARNING": "🟡",
        "INFO": "🔵",
        "DEBUG": "⚪",
    }
    for record in reversed(filtered):
        icon = level_colours.get(record["level"], "⚪")
        st.markdown(
            f"`{record['time']}` {icon} **{record['level']}** "
            f"`{record['logger']}` — {record['message']}"
        )
