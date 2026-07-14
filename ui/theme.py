"""ui/theme.py — Theme presets and page chrome (config, CSS, header)."""

from __future__ import annotations

import streamlit as st

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


def apply_page_chrome() -> None:
    """Configure the page, inject themed CSS, and render the header."""
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

    theme_name = str(st.session_state.get("ui_theme", "Ocean Professional"))
    theme = THEME_PRESETS.get(theme_name, THEME_PRESETS["Ocean Professional"])

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
            "__CHORUS_PRIMARY__", theme["primary"]
        )
        .replace("__CHORUS_SURFACE__", theme["surface"])
        .replace("__CHORUS_BORDER__", theme["border"])
        .replace("__CHORUS_HEADER_A__", theme["header_a"])
        .replace("__CHORUS_HEADER_B__", theme["header_b"])
        .replace("__CHORUS_HEADER_C__", theme["header_c"])
        .replace("__CHORUS_HIGH_BG__", theme["high_bg"])
        .replace("__CHORUS_HIGH_FG__", theme["high_fg"])
        .replace("__CHORUS_MED_BG__", theme["med_bg"])
        .replace("__CHORUS_MED_FG__", theme["med_fg"])
        .replace("__CHORUS_LOW_BG__", theme["low_bg"])
        .replace("__CHORUS_LOW_FG__", theme["low_fg"]),
        unsafe_allow_html=True,
    )

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
