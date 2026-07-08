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
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path when running from the ui/ subdirectory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from config import (  # noqa: E402
    ALIGNMENT_STRATEGY,
    CONSENSUS_MODELS,
    NOISE_FLOOR_MODE,
    VARIANT_LABELS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from export_engine.exporter import (  # noqa: E402
    export_all,
    export_plain_text,
    export_zip,
)
from pipeline_runner import run_pipeline  # noqa: E402
from ui.hardware_survey import (  # noqa: E402  # type: ignore[import]
    detect_hardware,
    recommend_settings,
    recommend_settings_background,
    summarise,
)
from utils import sanitise_stem  # noqa: E402

logger = logging.getLogger(__name__)

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

SUMMARY_SUCCESS_MSG = (
    "All files completed successfully. Review confidence sections below, "
    "then download archives."
)
SUMMARY_FAILURE_MSG = (
    "Some files failed. Check technical details in the affected sections, "
    "then retry failed files in smaller batches."
)
PROCESSING_FAILURE_MSG = (
    "Processing failed for {file_name}. Review guidance below and retry when ready."
)
PROCESSING_REMEDIATION_STEPS = (
    "- Confirm audio file integrity and supported format\n"
    "- Try a smaller model or disable advanced features\n"
    "- Re-run this file alone to isolate the issue"
)

THEME_PRESETS: dict[str, dict[str, str]] = {
    "Ocean Professional": {
        "primary": "#0f3460",
        "surface": "#f8f9fa",
        "border": "#dee2e6",
        "header_a": "#1a1a2e",
        "header_b": "#16213e",
        "header_c": "#0f3460",
        "high_bg": "#d4edda",
        "high_fg": "#155724",
        "med_bg": "#fff3cd",
        "med_fg": "#856404",
        "low_bg": "#f8d7da",
        "low_fg": "#721c24",
    },
    "Slate Enterprise": {
        "primary": "#243447",
        "surface": "#f4f6f8",
        "border": "#d0d7de",
        "header_a": "#2c3e50",
        "header_b": "#34495e",
        "header_c": "#243447",
        "high_bg": "#d8f3dc",
        "high_fg": "#1b4332",
        "med_bg": "#fff1c1",
        "med_fg": "#7a5c00",
        "low_bg": "#fde2e4",
        "low_fg": "#8b1e3f",
    },
    "Forest Trust": {
        "primary": "#1f5f3f",
        "surface": "#f4f8f5",
        "border": "#cad7ce",
        "header_a": "#143d2b",
        "header_b": "#1f5f3f",
        "header_c": "#2f855a",
        "high_bg": "#d9f2e3",
        "high_fg": "#155d3b",
        "med_bg": "#fff4cf",
        "med_fg": "#775c00",
        "low_bg": "#f8dcdf",
        "low_fg": "#7a1f2d",
    },
    "Graphite Contrast": {
        "primary": "#1f2937",
        "surface": "#f3f4f6",
        "border": "#c7ccd1",
        "header_a": "#111827",
        "header_b": "#1f2937",
        "header_c": "#374151",
        "high_bg": "#d1fae5",
        "high_fg": "#065f46",
        "med_bg": "#fef3c7",
        "med_fg": "#78350f",
        "low_bg": "#fee2e2",
        "low_fg": "#991b1b",
    },
    "Sunrise Editorial": {
        "primary": "#7a3e00",
        "surface": "#faf7f2",
        "border": "#dfd6c8",
        "header_a": "#4e2a14",
        "header_b": "#7a3e00",
        "header_c": "#a35a1d",
        "high_bg": "#deefe4",
        "high_fg": "#1d5f3e",
        "med_bg": "#ffeccb",
        "med_fg": "#7a4f00",
        "low_bg": "#f9dede",
        "low_fg": "#7c2230",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chorus — Consensus Transcription Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "Ocean Professional"

if "recent_runs" not in st.session_state:
    st.session_state["recent_runs"] = []

_theme_name = str(st.session_state.get("ui_theme", "Ocean Professional"))
_theme = THEME_PRESETS.get(_theme_name, THEME_PRESETS["Ocean Professional"])

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    :root {
        --chorus-primary: __CHORUS_PRIMARY__;
        --chorus-surface: __CHORUS_SURFACE__;
        --chorus-border: __CHORUS_BORDER__;
        --chorus-header-a: __CHORUS_HEADER_A__;
        --chorus-header-b: __CHORUS_HEADER_B__;
        --chorus-header-c: __CHORUS_HEADER_C__;
        --chorus-high-bg: __CHORUS_HIGH_BG__;
        --chorus-high-fg: __CHORUS_HIGH_FG__;
        --chorus-med-bg: __CHORUS_MED_BG__;
        --chorus-med-fg: __CHORUS_MED_FG__;
        --chorus-low-bg: __CHORUS_LOW_BG__;
        --chorus-low-fg: __CHORUS_LOW_FG__;
    }

    .chorus-header {
        background: linear-gradient(135deg, var(--chorus-header-a) 0%, var(--chorus-header-b) 50%, var(--chorus-header-c) 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .chorus-header h1 { margin: 0; font-size: 2.4rem; letter-spacing: -0.5px; }
    .chorus-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 1rem; }

    .tier-badge-high   { background:var(--chorus-high-bg); color:var(--chorus-high-fg); padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }
    .tier-badge-medium { background:var(--chorus-med-bg); color:var(--chorus-med-fg); padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }
    .tier-badge-low    { background:var(--chorus-low-bg); color:var(--chorus-low-fg); padding:2px 8px;
                         border-radius:4px; font-size:0.8rem; font-weight:600; }

    .stProgress > div > div > div { background-color: var(--chorus-primary); }
    .metric-card { background:var(--chorus-surface); border-radius:8px; padding:1rem;
                   text-align:center; border:1px solid var(--chorus-border); }

    /* Streamlit chrome — sidebar and primary accent */
    section[data-testid="stSidebar"] { background-color: var(--chorus-surface) !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        border-right: 1px solid var(--chorus-border);
    }
    .stButton > button {
        border-color: var(--chorus-primary) !important;
        color: var(--chorus-primary) !important;
    }
    .stButton > button:hover {
        background-color: var(--chorus-primary) !important;
        color: white !important;
    }

    /* Improve keyboard navigation discoverability */
    button:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible,
    [tabindex]:focus-visible {
        outline: 2px solid var(--chorus-primary) !important;
        outline-offset: 2px !important;
    }

    .chorus-preflight {
        background: var(--chorus-surface);
        border: 1px solid var(--chorus-border);
        border-left: 4px solid var(--chorus-primary);
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0 1rem;
    }

    .chorus-confidence-note {
        font-size: 0.9rem;
        color: #4a4a4a;
        margin-top: 0.4rem;
    }

    .chorus-run-status {
        background: var(--chorus-surface);
        border: 1px solid var(--chorus-border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0 1rem;
    }

    .chorus-run-summary-ok {
        display: inline-block;
        background: var(--chorus-high-bg);
        color: var(--chorus-high-fg);
        border: 1px solid var(--chorus-border);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .chorus-run-summary-issues {
        display: inline-block;
        background: var(--chorus-low-bg);
        color: var(--chorus-low-fg);
        border: 1px solid var(--chorus-border);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .chorus-action-bar {
        position: sticky;
        top: 0.5rem;
        z-index: 20;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid var(--chorus-border);
        border-left: 4px solid var(--chorus-primary);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.5rem 0 0.9rem;
        backdrop-filter: blur(2px);
    }

    .chorus-action-bar a {
        color: var(--chorus-primary);
        font-weight: 600;
        text-decoration: none;
    }

    .chorus-action-bar a:hover {
        text-decoration: underline;
    }

    .chorus-skip-links {
        position: relative;
        z-index: 50;
        margin-bottom: 0.5rem;
    }

    .chorus-skip-links a {
        position: absolute;
        left: -9999px;
        top: 0;
        background: var(--chorus-primary);
        color: white;
        padding: 0.5rem 0.7rem;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 700;
    }

    .chorus-skip-links a:focus {
        left: 0;
    }

    @media (max-width: 900px) {
        .chorus-header {
            padding: 1.25rem 1rem;
            border-radius: 10px;
        }

        .chorus-header h1 {
            font-size: 1.7rem;
        }

        .chorus-header p {
            font-size: 0.9rem;
        }

        .chorus-action-bar {
            position: static;
            backdrop-filter: none;
            margin: 0.35rem 0 0.75rem;
        }

        .chorus-run-status {
            padding: 0.75rem;
            margin: 0.5rem 0 0.75rem;
        }

        .chorus-run-summary-ok,
        .chorus-run-summary-issues {
            font-size: 0.8rem;
            padding: 0.2rem 0.55rem;
        }

        .chorus-skip-links a,
        .chorus-skip-links a:focus {
            position: static;
            display: inline-block;
            margin: 0.15rem 0.3rem 0.15rem 0;
        }
    }
</style>
""".replace(
        "__CHORUS_PRIMARY__", _theme["primary"]
    )
    .replace("__CHORUS_SURFACE__", _theme["surface"])
    .replace("__CHORUS_BORDER__", _theme["border"])
    .replace("__CHORUS_HEADER_A__", _theme["header_a"])
    .replace("__CHORUS_HEADER_B__", _theme["header_b"])
    .replace("__CHORUS_HEADER_C__", _theme["header_c"])
    .replace("__CHORUS_HIGH_BG__", _theme["high_bg"])
    .replace("__CHORUS_HIGH_FG__", _theme["high_fg"])
    .replace("__CHORUS_MED_BG__", _theme["med_bg"])
    .replace("__CHORUS_MED_FG__", _theme["med_fg"])
    .replace("__CHORUS_LOW_BG__", _theme["low_bg"])
    .replace("__CHORUS_LOW_FG__", _theme["low_fg"]),
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="chorus-skip-links">
    <a href="#upload-section">Skip to upload</a>
    <a href="#run-section" style="margin-left: 11rem;">Skip to run controls</a>
    <a href="#results-section" style="margin-left: 25rem;">Skip to results</a>
</div>
<div id="top"></div>
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


def _pick_best_ollama_model(models: list[str]) -> str:
    """Return the model name best suited for low-confidence word correction.

    Scoring rules (higher is better):
    - Penalise specialised models (coder / code / math in the name): -10
    - Prefer larger param counts parsed from the name: 14b=14, 8b=8, 7b=7, etc.
    - Prefer known general-purpose families: llama=3, qwen=2, mistral=1
    """
    import re

    def _score(name: str) -> int:
        n = name.lower()
        score = 0
        if any(tag in n for tag in ("coder", "code", "math")):
            score -= 10
        m = re.search(r"(\d+)b", n)
        if m:
            score += int(m.group(1))
        for rank, family in enumerate(("llama", "qwen", "mistral", "gemma"), start=1):
            if family in n:
                score += 4 - rank
                break
        return score

    return max(models, key=_score)


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


def _render_preflight_summary(
    file_count: int,
    model_choice: str,
    consensus_models: tuple[str, ...],
    alignment_choice: str,
    noise_mode_choice: str,
    enable_nlp: bool,
    enable_llm: bool,
    enable_diarisation: bool,
) -> None:
    """Render run preflight summary with informed-choice guidance."""
    selected_features = []
    if enable_nlp:
        selected_features.append("NLP reconstruction")
    if enable_llm:
        selected_features.append("LLM reconstruction")
    if enable_diarisation:
        selected_features.append("speaker diarisation")
    feature_text = ", ".join(selected_features) if selected_features else "none"

    runtime_hint = (
        "longer runs expected due to max-accuracy configuration"
        if model_choice in {"medium", "small"}
        or enable_nlp
        or enable_llm
        or enable_diarisation
        else "balanced runtime expected"
    )
    model_set_text = ", ".join(consensus_models)

    st.markdown(
        (
            '<div class="chorus-preflight">'
            f"<b>Preflight:</b> {file_count} file{'s' if file_count > 1 else ''}, "
            f"model <b>{model_choice}</b>, alignment <b>{alignment_choice}</b>, "
            f"noise mode <b>{noise_mode_choice}</b>, consensus models <b>{model_set_text}</b>, "
            f"advanced features: <b>{feature_text}</b>.<br>"
            f"Expected profile: <b>{runtime_hint}</b>. "
            "Tip: use smaller batches if you need faster feedback cycles."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_confidence_overview(consensus_text: str) -> None:
    """Render confidence bars, metrics, and interpretation guidance."""
    import re as _re

    high_m = _re.search(r"HIGH\s*\|\s*(\d+)", consensus_text)
    med_m = _re.search(r"MEDIUM\s*\|\s*(\d+)", consensus_text)
    low_m = _re.search(r"LOW\s*\|\s*(\d+)", consensus_text)
    n_high = int(high_m.group(1)) if high_m else 0
    n_med = int(med_m.group(1)) if med_m else 0
    n_low = int(low_m.group(1)) if low_m else 0
    total_w = n_high + n_med + n_low or 1

    bar_cols = st.columns([n_high or 1, n_med or 1, n_low or 1])
    with bar_cols[0]:
        st.markdown(
            f'<div style="background:var(--chorus-high-bg);color:var(--chorus-high-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🟢 HIGH</b><br>{n_high} ({n_high*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )
    with bar_cols[1]:
        st.markdown(
            f'<div style="background:var(--chorus-med-bg);color:var(--chorus-med-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🟡 MED</b><br>{n_med} ({n_med*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )
    with bar_cols[2]:
        st.markdown(
            f'<div style="background:var(--chorus-low-bg);color:var(--chorus-low-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🔴 LOW</b><br>{n_low} ({n_low*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("High confidence words", n_high)
    c2.metric("Medium confidence words", n_med)
    c3.metric("Low confidence words", n_low)
    st.markdown(
        '<div class="chorus-confidence-note">'
        "High confidence terms generally reflect strong cross-variant agreement. "
        "Review medium confidence carefully. Validate low confidence terms against source audio."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_processing_error(
    file_name: str, exc: Exception, allow_retry: bool = False
) -> None:
    """Render consistent actionable processing error guidance."""
    st.error(PROCESSING_FAILURE_MSG.format(file_name=file_name))
    st.markdown(PROCESSING_REMEDIATION_STEPS)
    with st.expander(f"Technical details — {file_name}"):
        st.code(str(exc))

    if allow_retry and st.button(
        "Retry this file",
        key=f"retry_seq_{sanitise_stem(Path(file_name).stem, fallback='upload')}",
    ):
        st.rerun()


def _render_run_status(
    *,
    container: object,
    total_files: int,
    completed_files: int,
    failed_files: int,
    start_time: float,
    current_file: str | None = None,
) -> None:
    """Render a compact batch status panel with throughput and ETA guidance."""
    processed = completed_files + failed_files
    elapsed = max(time.time() - start_time, 0.001)
    progress = (processed / total_files) if total_files else 0.0

    rate_per_sec = processed / elapsed if processed else 0.0
    eta_seconds = (
        int((total_files - processed) / rate_per_sec) if rate_per_sec > 0 else None
    )

    with container.container():
        st.markdown('<div class="chorus-run-status">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed", f"{processed}/{total_files}")
        c2.metric("Completed", completed_files)
        c3.metric("Failed", failed_files)
        c4.metric("Elapsed", f"{int(elapsed)} s")

        progress_text = (
            f"Processing {current_file}…"
            if current_file
            else f"Batch progress: {processed}/{total_files} files"
        )
        st.progress(progress, text=progress_text)

        if eta_seconds is not None and processed < total_files:
            st.caption(
                f"Estimated time remaining: ~{eta_seconds} s (updates as files complete)."
            )
        elif processed == total_files:
            st.caption("Batch complete. Review results and download outputs below.")

        st.markdown("</div>", unsafe_allow_html=True)


def _render_batch_outcome_summary(
    *,
    total_files: int,
    completed_files: int,
    failed_files: int,
    duration_seconds: float,
    failed_file_names: list[str] | None = None,
    file_anchors: dict[str, str] | None = None,
) -> None:
    """Render concise outcomes guidance after a batch run completes."""
    st.markdown("#### Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total_files)
    c2.metric("Completed", completed_files)
    c3.metric("Failed", failed_files)
    c4.metric("Duration", f"{int(duration_seconds)} s")

    if failed_files == 0:
        st.markdown(
            '<div class="chorus-run-summary-ok">ALL FILES COMPLETED</div>',
            unsafe_allow_html=True,
        )
        st.success(SUMMARY_SUCCESS_MSG, icon="✅")
    else:
        st.markdown(
            '<div class="chorus-run-summary-issues">ATTENTION REQUIRED</div>',
            unsafe_allow_html=True,
        )
        st.warning(SUMMARY_FAILURE_MSG, icon="⚠️")
        if failed_file_names:
            if file_anchors:
                unique_failed = list(dict.fromkeys(failed_file_names))
                failed_links = " · ".join(
                    f"[{name}](#{file_anchors[name]})"
                    for name in unique_failed
                    if name in file_anchors
                )
                if failed_links:
                    st.markdown(f"**Failed files:** {failed_links}")
            else:
                st.markdown("**Failed files:** " + ", ".join(failed_file_names))


def _render_result_navigation(
    file_names: list[str], file_anchors: dict[str, str]
) -> None:
    """Render quick links to each file section for large result sets."""
    if len(file_names) < 3:
        return

    anchors = [(name, file_anchors[name]) for name in file_names]
    links = " · ".join(
        ["[Top](#top)", *[f"[{name}](#{anchor})" for name, anchor in anchors]]
    )
    st.markdown(
        (
            '<div class="chorus-action-bar">'
            "<b>Quick Navigation:</b> "
            f"{links}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_result_filter(total_files: int) -> str:
    """Render a simple result visibility filter for larger batches."""
    if total_files < 3:
        return "All"
    return st.radio(
        "Results filter",
        options=["All", "Completed", "Failed"],
        index=0,
        horizontal=True,
        help="Use this to focus on successful or failed files in larger batches.",
    )


def _build_file_anchors(file_names: list[str]) -> dict[str, str]:
    """Return deterministic unique anchors keyed by file name."""
    anchors: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name in file_names:
        base = sanitise_stem(Path(name).stem, fallback="upload")
        counts[base] = counts.get(base, 0) + 1
        suffix = f"-{counts[base]}" if counts[base] > 1 else ""
        anchors[name] = f"{base}{suffix}"
    return anchors


def _record_recent_run(
    *, total: int, completed: int, failed: int, duration: float
) -> None:
    """Store a compact run snapshot in session state."""
    stamp = time.strftime("%H:%M:%S")
    st.session_state["recent_runs"] = [
        {
            "time": stamp,
            "total": total,
            "completed": completed,
            "failed": failed,
            "duration": int(duration),
        },
        *st.session_state["recent_runs"],
    ][:5]


def _render_recent_runs() -> None:
    """Render recent in-session run snapshots."""
    runs = st.session_state.get("recent_runs", [])
    if not runs:
        return

    with st.expander("Recent Runs", expanded=False):
        for run in runs:
            st.markdown(
                f"- **{run['time']}** — {run['completed']}/{run['total']} completed, "
                f"{run['failed']} failed, {run['duration']} s"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Appearance ───────────────────────────────────────────────────────────
    st.subheader("Appearance")
    st.selectbox(
        "Theme preset",
        options=list(THEME_PRESETS.keys()),
        key="ui_theme",
        help=(
            "Choose a visual preset. Themes only change presentation; processing logic, "
            "confidence math, and exports are unchanged."
        ),
    )
    st.caption(
        "Themes apply to the header, confidence highlights, sidebar, and buttons. "
        "Tip: choose higher-contrast presets when reviewing low-confidence segments."
    )

    # ── Model & Device ────────────────────────────────────────────────────────
    st.subheader("Model & Device")

    # Initialise session-state defaults on first load so the survey button can
    # overwrite them and trigger a rerun that pre-selects the new values.
    _model_options = ["tiny", "base", "small", "medium", "large"]
    _device_options = ["auto", "cpu", "cuda", "mps"]
    if "cfg_model" not in st.session_state:
        st.session_state["cfg_model"] = (
            WHISPER_MODEL if WHISPER_MODEL in _model_options else "base"
        )
    if "cfg_device" not in st.session_state:
        st.session_state["cfg_device"] = (
            WHISPER_DEVICE if WHISPER_DEVICE in _device_options else "auto"
        )
    if "cfg_parallelism" not in st.session_state:
        st.session_state["cfg_parallelism"] = "auto"

    # Preset selector — surveys hardware and applies conservative or maximum settings.
    _preset_col, _apply_col = st.columns([3, 1])
    with _preset_col:
        _preset = st.selectbox(
            "Settings preset",
            options=["Max", "Background"],
            index=0,
            label_visibility="collapsed",
            help=(
                "**Max:** largest viable model and full parallelism — "
                "machine is dedicated to Chorus while running.\n\n"
                "**Background:** one model tier lower, parallelism pinned to 1 — "
                "machine stays responsive for other work."
            ),
        )
    with _apply_col:
        _apply = st.button("🔍 Apply", use_container_width=True)

    if _apply:
        with st.spinner("Surveying hardware…"):
            _hw = detect_hardware()
            _rec = (
                recommend_settings(_hw)
                if _preset == "Max"
                else recommend_settings_background(_hw)
            )
        st.session_state["cfg_model"] = _rec["whisper_model"]
        st.session_state["cfg_device"] = _rec["device"]
        st.session_state["cfg_parallelism"] = _rec["parallelism"]
        st.session_state["survey_summary"] = f"**{_preset}** — {summarise(_hw, _rec)}"
        st.rerun()

    if st.session_state.get("survey_summary"):
        st.info(st.session_state["survey_summary"])

    model_choice = st.selectbox(
        "Model size",
        options=_model_options,
        key="cfg_model",
        help=(
            "Larger models are more accurate but slower. 'base' is recommended for CPU. "
            "'large' requires ~10 GB RAM and a GPU — see docs/CONFIGURATION.md."
        ),
    )

    default_consensus = [m for m in CONSENSUS_MODELS if m in _model_options] or [
        model_choice
    ]
    if model_choice not in default_consensus:
        default_consensus.insert(0, model_choice)

    consensus_model_choice = st.multiselect(
        "Consensus models",
        options=_model_options,
        default=default_consensus,
        help=(
            "Choose one or more models for consensus voting. The first selected model "
            "is treated as primary for compatibility outputs."
        ),
    )
    if not consensus_model_choice:
        consensus_model_choice = [model_choice]
    if consensus_model_choice[0] != model_choice:
        consensus_model_choice = [
            model_choice,
            *[m for m in consensus_model_choice if m != model_choice],
        ]
    consensus_models = tuple(dict.fromkeys(consensus_model_choice))

    device_choice = st.selectbox(
        "Compute device",
        options=_device_options,
        key="cfg_device",
        format_func=lambda x: {
            "auto": "Auto-detect (recommended)",
            "cpu": "CPU",
            "cuda": "NVIDIA CUDA (GPU)",
            "mps": "Apple MPS (Apple Silicon)",
        }.get(x, x),
        help=(
            "**Auto:** probes CUDA → MPS → CPU and selects the best available.\n\n"
            "**CPU:** works everywhere; slowest.\n\n"
            "**CUDA:** NVIDIA GPU. Requires NVIDIA Container Toolkit (Docker) or native drivers.\n\n"
            "**MPS:** Apple Silicon GPU. Native macOS only — not available inside Docker. "
            "Note: a CPU fallback is triggered automatically for float64 operations."
        ),
    )

    parallelism_raw = st.session_state.get("cfg_parallelism", "auto")
    _par_is_auto = parallelism_raw == "auto"
    parallelism_auto = st.checkbox(
        "Auto parallelism",
        value=_par_is_auto,
        help=(
            "Let Chorus choose the worker count based on device and available capacity. "
            "Disable to pin an exact number of parallel transcription passes."
        ),
    )
    if parallelism_auto:
        st.session_state["cfg_parallelism"] = "auto"
        parallelism_choice = "auto"
    else:
        _par_default = int(parallelism_raw) if parallelism_raw.isdigit() else 2
        parallelism_choice = str(
            st.number_input(
                "Worker count",
                min_value=1,
                max_value=16,
                value=_par_default,
                step=1,
                help=(
                    "Number of concurrent transcription passes. "
                    "Pin to 1 on low-RAM machines to avoid memory pressure."
                ),
            )
        )
        st.session_state["cfg_parallelism"] = parallelism_choice

    # ── Language ──────────────────────────────────────────────────────────────
    st.subheader("Language")
    lang_input = st.text_input(
        "Language code (optional)",
        value="",
        placeholder="e.g. en, fr, de — leave blank for auto-detect",
        help="BCP-47 language code. Leave empty for automatic detection.",
    )
    language = lang_input.strip() or None

    # ── Processing Strategy ───────────────────────────────────────────────────
    st.subheader("Processing Strategy")
    alignment_choice = st.selectbox(
        "Alignment algorithm",
        options=["sequence", "positional"],
        index=0 if ALIGNMENT_STRATEGY == "sequence" else 1,
        format_func=lambda x: (
            "Sequence alignment (accurate)"
            if x == "sequence"
            else "Positional (fast, legacy)"
        ),
        help=(
            "**Sequence (Needleman-Wunsch):** Handles word insertions and deletions "
            "across variants. More accurate on noisy audio.\n\n"
            "**Positional (legacy):** Compares word-by-word at each index. "
            "Fast but sensitive to length differences between variants."
        ),
    )

    # ── Audio Cleaning ────────────────────────────────────────────────────────
    st.subheader("Audio Cleaning")
    noise_mode_choice = st.selectbox(
        "Noise floor detection",
        options=["vad", "fixed"],
        index=0 if NOISE_FLOOR_MODE == "vad" else 1,
        format_func=lambda x: "Auto (VAD)" if x == "vad" else "First 0.5 s (legacy)",
        help=(
            "**Auto (VAD):** Detects the quietest segment via energy analysis. "
            "Best when audio starts with speech.\n\n"
            "**First 0.5 s:** Assumes the first half-second is silence. "
            "Use if you know your recordings have a silent intro."
        ),
    )

    st.info(
        f"Chorus produces **{len(VARIANT_LABELS)} variants**:\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in VARIANT_LABELS.items()),
        icon="ℹ️",
    )

    # ── Advanced Features ─────────────────────────────────────────────────────
    st.subheader("Advanced Features")
    enable_nlp = st.checkbox(
        "🧠 NLP Reconstruction",
        help="Use spaCy to grammatically reconstruct LOW-confidence tokens.",
    )
    enable_llm = st.checkbox(
        "🤖 LLM Reconstruction (Ollama)",
        help=(
            "Use a local Ollama model to resolve LOW-confidence tokens. "
            "Requires Ollama installed and running separately — "
            "see the Help page (sidebar) for setup instructions."
        ),
    )
    ollama_model: str | None = None
    if enable_llm:
        from llm_reconstructor.ollama_client import list_models, probe_model

        _llm_ok, _llm_reason = probe_model()
        if not _llm_ok:
            enable_llm = False
            st.session_state["show_ollama_dialog"] = True
            st.session_state["ollama_fail_reason"] = _llm_reason
        else:
            _available = list_models()
            if _available:
                _default = _pick_best_ollama_model(_available)
                _prev = st.session_state.get("ollama_model")
                _default_idx = (
                    _available.index(_prev)
                    if _prev in _available
                    else _available.index(_default)
                )
                _labels = [
                    f"{m} (recommended)" if m == _default else m for m in _available
                ]
                _chosen_label = st.selectbox(
                    "Ollama model",
                    options=_labels,
                    index=_default_idx,
                    help=(
                        "All locally pulled models are available. "
                        "The recommended model is best suited for "
                        "low-confidence word correction."
                    ),
                )
                ollama_model = _available[_labels.index(_chosen_label)]
                st.session_state["ollama_model"] = ollama_model

    if st.session_state.get("show_ollama_dialog"):
        from llm_reconstructor.ollama_client import probe_model as _probe

        _reason = st.session_state.get("ollama_fail_reason", "Ollama is not reachable.")

        @st.dialog("LLM Reconstruction — Setup Required")
        def _ollama_setup_dialog():
            st.error(_reason)
            st.markdown(
                "**To enable LLM reconstruction, install and start Ollama:**\n\n"
                "```bash\n"
                "brew install ollama          # macOS\n"
                "ollama serve                 # keep this terminal open\n"
                "ollama pull llama3.1:8b      # ~4.7 GB — run once\n"
                "```\n\n"
                "On 8 GB machines use `llama3.2:3b` instead. "
                "See the **Help** page in the sidebar for full setup guidance."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Dismiss", use_container_width=True):
                    st.session_state["show_ollama_dialog"] = False
                    st.rerun()
            with col2:
                if st.button("Retry", type="primary", use_container_width=True):
                    _ok, _new_reason = _probe()
                    if _ok:
                        st.session_state["show_ollama_dialog"] = False
                        st.rerun()
                    else:
                        st.session_state["ollama_fail_reason"] = _new_reason
                        st.rerun()

        _ollama_setup_dialog()
    else:
        st.session_state.pop("ollama_fail_reason", None)
    enable_diarisation = st.checkbox(
        "🗣️ Speaker Diarisation",
        help="Identify multiple speakers (requires HUGGINGFACE_TOKEN).",
    )
    st.caption("ℹ️ Word-level timestamps are always enabled for precise subtitles.")

    # ── Export Formats ────────────────────────────────────────────────────────
    st.subheader("Export Formats")
    export_pdf = st.checkbox("PDF Document", value=False)
    export_docx = st.checkbox("Word Document (.docx)", value=False)
    export_srt = st.checkbox("Subtitles (.srt) — word-level", value=False)

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
    st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
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
    st.markdown('<div id="run-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("2 · Run Pipeline")

    # ── Processing mode (only shown for multiple files) ───────────────────────
    rec_mode, rec_reason = _hw_recommendation()

    if len(uploaded_files) > 1:
        # Auto-switch to batch view for 3+ files
        if len(uploaded_files) >= 3:
            mode_choice = "All at once — results shown at end"
            st.info(
                f"📁 **Batch mode** — {len(uploaded_files)} files detected. "
                "All files will be processed before results are displayed.",
                icon="📁",
            )
        else:
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

    _render_preflight_summary(
        file_count=n,
        model_choice=model_choice,
        consensus_models=consensus_models,
        alignment_choice=alignment_choice,
        noise_mode_choice=noise_mode_choice,
        enable_nlp=enable_nlp,
        enable_llm=enable_llm,
        enable_diarisation=enable_diarisation,
    )
    if n > 10:
        st.warning(
            "Large batch detected. For easier troubleshooting, consider processing in smaller groups of 5-10 files.",
            icon="⚠️",
        )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button(
            f"▶ Start Chorus ({n} file{'s' if n > 1 else ''})",
            type="primary",
            use_container_width=True,
        )

    _render_recent_runs()

    if run_btn:
        os.environ["WHISPER_MODEL"] = model_choice
        os.environ["CONSENSUS_MODELS"] = ",".join(consensus_models)
        os.environ["NOISE_FLOOR_MODE"] = noise_mode_choice

        # Apply device and parallelism overrides to the live config module so the
        # transcription engine and orchestrator pick them up without a restart.
        _effective_device = device_choice if device_choice != "auto" else WHISPER_DEVICE
        config.WHISPER_DEVICE = _effective_device
        os.environ["WHISPER_DEVICE"] = _effective_device
        config.TRANSCRIPTION_PARALLELISM = parallelism_choice
        os.environ["TRANSCRIPTION_PARALLELISM"] = parallelism_choice

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
            original_stem = sanitise_stem(Path(uf.name).stem, fallback="upload")
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
                consensus_models=consensus_models,
                enable_nlp=enable_nlp,
                enable_llm=enable_llm,
                ollama_model=ollama_model,
                enable_diarisation=enable_diarisation,
                alignment_strategy=alignment_choice,
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

            # ── Processing time ───────────────────────────────────────────────
            elapsed = results.get("elapsed_seconds", 0)
            st.caption(f"⏱️ Processed in **{elapsed} s**")

            # ── Metrics row ───────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            for col, (key, meta) in zip(
                [m1, m2, m3, m4], transcripts.items(), strict=False
            ):
                wc = len(meta.get("text", "").split())
                col.metric(VARIANT_LABELS.get(key, key), f"{wc} words")

            # ── Confidence Visualisation ──────────────────────────────────────
            st.markdown("#### 🎯 Confidence Overview")
            _render_confidence_overview(consensus_text)

            # ── Consensus document preview ────────────────────────────────────
            st.markdown("#### 📄 Consensus Transcript")
            view_mode = st.toggle(
                "Show raw Markdown",
                value=False,
                key=f"raw_md_{original_stem}",
                help="Toggle between rendered view and raw Markdown source.",
            )
            if view_mode:
                st.code(consensus_text, language="markdown")
            else:
                st.markdown(consensus_text, unsafe_allow_html=False)

            # ── Download buttons ──────────────────────────────────────────────
            # Stacked download controls remain readable across desktop and narrow layouts.
            st.download_button(
                label="⬇️ Download Consensus Markdown (.md)",
                data=consensus_text,
                file_name=consensus_path.name,
                mime="text/markdown",
                type="primary",
                key=f"dl_md_{original_stem}",
                use_container_width=True,
            )

            plain_path = export_plain_text(
                consensus_path, original_stem, include_low=show_low
            )
            st.download_button(
                label="⬇️ Download Most Likely Transcript (.txt)",
                data=plain_path.read_text(encoding="utf-8"),
                file_name=plain_path.name,
                mime="text/plain",
                key=f"dl_txt_{original_stem}",
                use_container_width=True,
            )

            with st.spinner("Building archive with selected outputs…"):
                zip_bytes = export_zip(
                    consensus_path,
                    transcripts["original"],
                    original_stem,
                    include_formats=formats_to_export or None,
                )
            st.download_button(
                label="⬇️ Download Full Output Archive (.zip)",
                data=zip_bytes,
                file_name=f"{original_stem}_chorus_all.zip",
                mime="application/zip",
                key=f"dl_zip_{original_stem}",
                use_container_width=True,
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

                # ── Editable speaker name table ───────────────────────────────
                speaker_labels = results.get("speaker_labels", [])
                if speaker_labels:
                    from diarisation.diariser import (
                        load_speaker_names,
                        save_speaker_names,
                    )

                    # Load existing names (previously saved or empty)
                    existing_names = load_speaker_names(original_stem)

                    st.markdown(
                        "**Speaker Names** — assign human-readable names to "
                        "each speaker. Names are saved and will be re-used on "
                        "reprocess."
                    )
                    name_cols = st.columns([1, 2])
                    with name_cols[0]:
                        st.markdown("**Label**")
                    with name_cols[1]:
                        st.markdown("**Name**")

                    updated_names: dict[str, str] = {}
                    for spk in speaker_labels:
                        row_cols = st.columns([1, 2])
                        with row_cols[0]:
                            st.code(spk, language=None)
                        with row_cols[1]:
                            name_val = st.text_input(
                                f"Name for {spk}",
                                value=existing_names.get(spk, ""),
                                placeholder="e.g. Interviewer, Guest…",
                                label_visibility="collapsed",
                                key=f"spk_name_{spk}_{original_stem}",
                            )
                            if name_val.strip():
                                updated_names[spk] = name_val.strip()

                    # Save button
                    if st.button(
                        "💾 Save Speaker Names",
                        key=f"save_spk_{original_stem}",
                        help="Saves names to a sidecar JSON file. They will be "
                        "automatically loaded next time you process this file.",
                    ):
                        save_speaker_names(original_stem, updated_names)
                        st.success(
                            f"Saved {len(updated_names)} speaker name(s) → "
                            f"`{original_stem}_speakers.json`"
                        )
                        st.rerun()

                # ── Diarised transcript preview ───────────────────────────────
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

            # AI Context Pack
            ai_ctx_path = results.get("ai_context_path")
            if ai_ctx_path and ai_ctx_path.exists():
                st.markdown("##### 🤖 AI Context Pack")
                st.caption(
                    "A structured document designed to be fed to an LLM alongside "
                    "questions about this transcript. Contains methodology, confidence "
                    "data, and uncertainty annotations."
                )
                ai_ctx_text = ai_ctx_path.read_text(encoding="utf-8")
                with st.expander("Preview AI Context Pack"):
                    st.markdown(ai_ctx_text)
                st.download_button(
                    label="⬇️ AI Context Pack (.md)",
                    data=ai_ctx_text,
                    file_name=ai_ctx_path.name,
                    mime="text/markdown",
                    key=f"dl_ai_ctx_{original_stem}",
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

        st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("3 · Results")
        run_started_at = time.time()
        run_status_slot = st.empty()
        completed_files = 0
        failed_files = 0
        failed_file_names: list[str] = []

        _render_run_status(
            container=run_status_slot,
            total_files=len(uploaded_files),
            completed_files=completed_files,
            failed_files=failed_files,
            start_time=run_started_at,
        )

        if sequential:
            file_names = [str(uf.name) for uf in uploaded_files]
            file_anchors = _build_file_anchors(file_names)
            _render_result_navigation(file_names, file_anchors)
            sequential_results: list[tuple[object, dict, Path, str]] = []

            # Process and render each file as it completes
            for uf in uploaded_files:
                section_anchor = file_anchors[str(uf.name)]
                st.markdown(
                    f'<div id="{section_anchor}"></div>', unsafe_allow_html=True
                )
                _render_run_status(
                    container=run_status_slot,
                    total_files=len(uploaded_files),
                    completed_files=completed_files,
                    failed_files=failed_files,
                    start_time=run_started_at,
                    current_file=uf.name,
                )
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
                        completed_files += 1
                        progress_bar.progress(1.0, text="✅ Complete!")
                        status_text.success(
                            f"Completed in **{results['elapsed_seconds']} s**"
                        )
                        sequential_results.append(
                            (uf, results, tmp_path, original_stem)
                        )
                    except Exception as exc:
                        failed_files += 1
                        failed_file_names.append(str(uf.name))
                        _render_processing_error(uf.name, exc, allow_retry=True)
                        logger.exception("Pipeline error for %s", uf.name)
                    finally:
                        if tmp_path is not None:
                            tmp_path.unlink(missing_ok=True)

                _render_run_status(
                    container=run_status_slot,
                    total_files=len(uploaded_files),
                    completed_files=completed_files,
                    failed_files=failed_files,
                    start_time=run_started_at,
                )

            # Show summary once processing is finished, then render detail sections.
            total_duration = time.time() - run_started_at
            _render_batch_outcome_summary(
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                duration_seconds=total_duration,
                failed_file_names=failed_file_names,
                file_anchors=file_anchors,
            )
            _record_recent_run(
                total=len(uploaded_files),
                completed=completed_files,
                failed=failed_files,
                duration=total_duration,
            )
            result_filter = _render_result_filter(len(uploaded_files))

            for uf, results, tmp_path, original_stem in sequential_results:
                if result_filter == "Failed":
                    continue
                with st.expander(f"📄 {uf.name}", expanded=True):
                    st.success(f"Completed in **{results['elapsed_seconds']} s**")
                    _render_file_results(uf.name, results, tmp_path, original_stem)

        else:
            file_names = [str(uf.name) for uf in uploaded_files]
            file_anchors = _build_file_anchors(file_names)
            _render_result_navigation(file_names, file_anchors)

            # Process all files first, collect results
            all_results: list[tuple[object, dict, Path, str]] = []
            overall = st.progress(0.0, text="Starting…")

            for idx, uf in enumerate(uploaded_files):
                _render_run_status(
                    container=run_status_slot,
                    total_files=len(uploaded_files),
                    completed_files=completed_files,
                    failed_files=failed_files,
                    start_time=run_started_at,
                    current_file=uf.name,
                )
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
                    completed_files += 1
                except Exception as exc:
                    failed_files += 1
                    failed_file_names.append(str(uf.name))
                    _render_processing_error(uf.name, exc)
                    logger.exception("Pipeline error for %s", uf.name)
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)

                _render_run_status(
                    container=run_status_slot,
                    total_files=len(uploaded_files),
                    completed_files=completed_files,
                    failed_files=failed_files,
                    start_time=run_started_at,
                )

            overall.progress(1.0, text=f"✅ All {len(uploaded_files)} files complete!")

            # Now render all results
            total_duration = time.time() - run_started_at
            _render_batch_outcome_summary(
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                duration_seconds=total_duration,
                failed_file_names=failed_file_names,
                file_anchors=file_anchors,
            )
            _record_recent_run(
                total=len(uploaded_files),
                completed=completed_files,
                failed=failed_files,
                duration=total_duration,
            )
            result_filter = _render_result_filter(len(uploaded_files))

            for uf, results, tmp_path, original_stem in all_results:
                if result_filter == "Failed":
                    continue
                section_anchor = file_anchors[str(uf.name)]
                st.markdown(
                    f'<div id="{section_anchor}"></div>', unsafe_allow_html=True
                )
                with st.expander(f"📄 {uf.name}", expanded=True):
                    st.success(f"Completed in **{results['elapsed_seconds']} s**")
                    _render_file_results(uf.name, results, tmp_path, original_stem)

else:
    st.info(
        "Upload one or more audio files above to begin. "
        "Then configure options in the sidebar and start Chorus.",
        icon="👆",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Chorus Engine · Powered by [OpenAI Whisper](https://github.com/openai/whisper), "
    "librosa, NLTK, and Streamlit · "
    "All processing is performed locally — no audio data leaves your machine."
)
