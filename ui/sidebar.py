"""ui/sidebar.py — Sidebar configuration controls and the resulting run config."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import (
    ALIGNMENT_STRATEGY,
    CONSENSUS_MODELS,
    CONSENSUS_THRESHOLD,
    NOISE_FLOOR_MODE,
    SIMILARITY_THRESHOLD,
    VARIANT_LABELS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from ui.hardware_survey import (  # type: ignore[import]
    detect_hardware,
    recommend_settings,
    recommend_settings_background,
    summarise,
)
from ui.theme import THEME_PRESETS


@dataclass
class SidebarConfig:
    """User-selected configuration collected from the sidebar controls."""

    model_choice: str
    consensus_models: tuple[str, ...]
    device_choice: str
    parallelism_choice: str
    language: str | None
    alignment_choice: str
    consensus_threshold: float
    similarity_threshold: float
    noise_mode_choice: str
    enable_nlp: bool
    enable_llm: bool
    ollama_model: str | None
    enable_diarisation: bool
    export_pdf: bool
    export_docx: bool
    export_srt: bool


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


def render_sidebar() -> SidebarConfig:
    """Render the configuration sidebar and return the selected run config."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── Appearance ───────────────────────────────────────────────────────
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

        # ── Model & Device ────────────────────────────────────────────────────
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
            st.session_state["survey_summary"] = (
                f"**{_preset}** — {summarise(_hw, _rec)}"
            )
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

        # ── Language ──────────────────────────────────────────────────────────
        st.subheader("Language")
        lang_input = st.text_input(
            "Language code (optional)",
            value="",
            placeholder="e.g. en, fr, de — leave blank for auto-detect",
            help="BCP-47 language code. Leave empty for automatic detection.",
        )
        language = lang_input.strip() or None

        # ── Processing Strategy ───────────────────────────────────────────────
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

        consensus_threshold = st.slider(
            "HIGH-confidence threshold",
            min_value=0.50,
            max_value=1.00,
            value=float(CONSENSUS_THRESHOLD),
            step=0.05,
            help=(
                "Fraction of transcription passes that must agree on a word for "
                "it to count as HIGH confidence (rendered plain, trusted). With "
                "the default 4 passes, 0.75 means 3-of-4 agreement. Lower it to "
                "trust more words; raise it to be flagged about more of them."
            ),
        )
        similarity_threshold = st.slider(
            "Word-match similarity",
            min_value=0.50,
            max_value=0.95,
            value=float(SIMILARITY_THRESHOLD),
            step=0.05,
            help=(
                "How similar two spellings must be (0–1, Levenshtein-based) to "
                "count as the same word when voting — e.g. grouping 'colour' "
                "and 'color'. Lower values merge more variant spellings; higher "
                "values treat near-misses as disagreements."
            ),
        )

        # ── Audio Cleaning ────────────────────────────────────────────────────
        st.subheader("Audio Cleaning")
        noise_mode_choice = st.selectbox(
            "Noise floor detection",
            options=["vad", "fixed"],
            index=0 if NOISE_FLOOR_MODE == "vad" else 1,
            format_func=lambda x: (
                "Auto (VAD)" if x == "vad" else "First 0.5 s (legacy)"
            ),
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

        # ── Advanced Features ─────────────────────────────────────────────────
        st.subheader("Advanced Features")
        enable_nlp = st.checkbox(
            "🧠 NLP Reconstruction",
            help="Use spaCy to grammatically reconstruct LOW-confidence tokens.",
        )
        if enable_nlp:
            from reconstruction import probe_spacy_model

            _nlp_ok, _nlp_reason = probe_spacy_model()
            if not _nlp_ok:
                enable_nlp = False
                st.session_state["show_spacy_dialog"] = True
                st.session_state["spacy_fail_reason"] = _nlp_reason

        if st.session_state.get("show_spacy_dialog"):
            from reconstruction import probe_spacy_model as _probe_spacy

            _spacy_reason = st.session_state.get(
                "spacy_fail_reason", "The spaCy model is not available."
            )

            @st.dialog("NLP Reconstruction — Setup Required")
            def _spacy_setup_dialog():
                st.error(_spacy_reason)
                st.markdown(
                    "**To enable NLP reconstruction, install spaCy and its model:**\n\n"
                    "```bash\n"
                    "pip install spacy\n"
                    "python -m spacy download en_core_web_md\n"
                    "```\n\n"
                    "See the **Help** page in the sidebar for full setup guidance."
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Dismiss", use_container_width=True, key="spacy_dismiss"
                    ):
                        st.session_state["show_spacy_dialog"] = False
                        st.rerun()
                with col2:
                    if st.button(
                        "Retry",
                        type="primary",
                        use_container_width=True,
                        key="spacy_retry",
                    ):
                        _ok, _new_reason = _probe_spacy()
                        if _ok:
                            st.session_state["show_spacy_dialog"] = False
                            st.rerun()
                        else:
                            st.session_state["spacy_fail_reason"] = _new_reason
                            st.rerun()

            _spacy_setup_dialog()
        else:
            st.session_state.pop("spacy_fail_reason", None)

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
            from reconstruction.ollama_client import list_models, probe_model

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
            from reconstruction.ollama_client import probe_model as _probe

            _reason = st.session_state.get(
                "ollama_fail_reason", "Ollama is not reachable."
            )

            @st.dialog("LLM Reconstruction — Setup Required")
            def _ollama_setup_dialog():
                st.error(_reason)
                st.markdown(
                    "**To enable LLM reconstruction, install and start Ollama:**\n\n"
                    "```bash\n"
                    "brew install ollama          # macOS\n"
                    "ollama serve                 # keep this terminal open\n"
                    "ollama pull qwen2.5:3b       # ~2 GB — run once\n"
                    "```\n\n"
                    "For technical/rare-vocabulary audio, `qwen2.5:14b` (~9 GB) "
                    "retains more long-tail vocabulary at the cost of speed. "
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

        # ── Export Formats ────────────────────────────────────────────────────
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

    return SidebarConfig(
        model_choice=model_choice,
        consensus_models=consensus_models,
        device_choice=device_choice,
        parallelism_choice=parallelism_choice,
        language=language,
        alignment_choice=alignment_choice,
        consensus_threshold=consensus_threshold,
        similarity_threshold=similarity_threshold,
        noise_mode_choice=noise_mode_choice,
        enable_nlp=enable_nlp,
        enable_llm=enable_llm,
        ollama_model=ollama_model,
        enable_diarisation=enable_diarisation,
        export_pdf=export_pdf,
        export_docx=export_docx,
        export_srt=export_srt,
    )
