"""
ui/app.py — Chorus Streamlit Interface.

Thin entry point wiring the page together. Concerns are split across:
  - ui/theme.py               — theme presets, page config, CSS, header
  - ui/sidebar.py             — configuration controls and the run config
  - ui/upload.py              — audio upload area and stage overview
  - ui/pipeline_invocation.py — run controls and per-file pipeline glue
  - ui/results.py             — results rendering, status panels, summaries

Run with:
    streamlit run ui/app.py
or via Docker Compose (see README.md).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path when running from the ui/ subdirectory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.pipeline_invocation import render_run_section  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.theme import apply_page_chrome  # noqa: E402
from ui.upload import render_upload  # noqa: E402

_LOG_BUFFER_KEY = "log_buffer"
_LOG_BUFFER_MAX = 500


class _SessionLogHandler(logging.Handler):
    """Appends log records to st.session_state for the in-app Logs page."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import streamlit as _st

            buf = _st.session_state.setdefault(_LOG_BUFFER_KEY, [])
            buf.append(
                {
                    "time": self.formatTime(record, "%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
            )
            if len(buf) > _LOG_BUFFER_MAX:
                del buf[: len(buf) - _LOG_BUFFER_MAX]
        except Exception:  # noqa: BLE001, S110
            pass  # log handler must never crash the UI


_session_handler = _SessionLogHandler()
_session_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_session_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Page assembly
# ─────────────────────────────────────────────────────────────────────────────

apply_page_chrome()
sidebar_config = render_sidebar()
uploaded_files = render_upload()
render_run_section(uploaded_files, sidebar_config)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Chorus Engine · Powered by [OpenAI Whisper](https://github.com/openai/whisper), "
    "librosa, NLTK, and Streamlit · "
    "All processing is performed locally — no audio data leaves your machine."
)
