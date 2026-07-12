"""ui/pages/1_Help.py — Chorus Help & FAQ page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="Help — Chorus",
    page_icon="❓",
    layout="wide",
)

st.title("❓ Help & FAQ")

# ── What is Chorus? ───────────────────────────────────────────────────────────
st.header("What is Chorus?")
st.markdown(
    "Chorus is a local, privacy-preserving audio transcription engine. "
    "Rather than relying on a single model pass, it generates multiple cleaned "
    "variants of your audio (high-pass filtered, normalised, denoised), "
    "transcribes each independently using OpenAI Whisper, then merges the results "
    "through a word-level consensus vote. Words agreed upon by more variants score "
    "higher confidence; words appearing in only one variant are flagged LOW. "
    "No audio leaves your machine."
)

# ── Quick start ───────────────────────────────────────────────────────────────
st.header("Quick Start")
st.markdown(
    "1. **Upload** one or more audio files using the uploader on the main page.\n"
    "2. **Configure** the Whisper model, language hint, and any advanced features "
    "in the sidebar.\n"
    "3. **Run** — click **Transcribe** (single file) or **Run Batch** (3+ files).\n"
    "4. **Review** the consensus transcript. Highlighted words indicate split or "
    "low-confidence regions.\n"
    "5. **Download** your chosen export formats (PDF, DOCX, SRT, VTT, JSON) "
    "individually or as a single zip archive."
)

# ── Supported formats ─────────────────────────────────────────────────────────
st.header("Supported Audio Formats")
st.markdown(
    "Any format supported by FFmpeg. Common formats that work out of the box:\n\n"
    "| Format | Extension | Notes |\n"
    "|--------|-----------|-------|\n"
    "| MPEG-4 Audio | `.m4a` | Default for iPhone/Mac recordings |\n"
    "| MP3 | `.mp3` | Universal |\n"
    "| WAV | `.wav` | Lossless; largest file size |\n"
    "| FLAC | `.flac` | Lossless, compressed |\n"
    "| OGG Vorbis | `.ogg` | Open format |\n\n"
    "> **Tip:** If you see a `PySoundFile failed. Trying audioread instead` warning "
    "in the console, this is expected for `.m4a` files — Chorus falls back to a "
    "compatible reader automatically. Transcription will proceed normally."
)

# ── Confidence tiers ──────────────────────────────────────────────────────────
st.header("Understanding Confidence Tiers")
st.markdown(
    "The consensus transcript annotates each word with a confidence tier based on "
    "how many of the four audio variants agreed on it:\n\n"
    "| Display | Tier | Agreement | Recommended Action |\n"
    "|---------|------|-----------|--------------------|\n"
    "| Plain text | **HIGH** (≥ 75 %) | 3–4 variants | Accept |\n"
    "| `==highlighted==` | **MEDIUM** (50 %) | 2 variants | Review |\n"
    "| **~~struck bold~~** | **LOW** (25 %) | 1 variant | Flag or discard |\n\n"
    "These thresholds are configurable in `config.py`."
)

# ── Export formats ────────────────────────────────────────────────────────────
st.header("Export Formats")
st.markdown(
    "| Format | Contents |\n"
    "|--------|----------|\n"
    "| **PDF** | Annotated consensus transcript with confidence highlighting |\n"
    "| **DOCX** | Same as PDF, editable in Word / LibreOffice |\n"
    "| **SRT** | Word-level subtitles, ≤ 6 words per cue |\n"
    "| **VTT** | WebVTT subtitles for web players |\n"
    "| **Plain text** | Clean transcript, no markup; optional `[word?]` for LOW tokens |\n"
    "| **Best-guess text** | Fully clean transcript — no brackets, markup, or "
    "statistics at all; every word resolved to its single best guess |\n"
    "| **JSON bundle** | All variant transcripts, word-vote sequence, and statistics |\n"
    "| **AI context pack** | Structured document for LLM consumption — methodology, "
    "confidence data, uncertainty annotations |\n"
    "| **ZIP** | All selected formats bundled together |\n"
)
st.markdown(
    "For a guide to every output file written for LLM-assisted analysis, see "
    "`docs/CHORUS_FOR_LLMS.md` in the repository — paste it alongside Chorus "
    "output when asking an LLM to summarise or fact-check a transcript."
)

# ── spaCy setup ────────────────────────────────────────────────────────────────
st.header("NLP Reconstruction — spaCy Setup")
st.markdown(
    "The **NLP Reconstruction** feature uses spaCy's `en_core_web_md` model "
    "(grammatical analysis and word vectors) to attempt to resolve "
    "LOW-confidence tokens. This model is not installed by default and must be "
    "downloaded once."
)

with st.expander("Step-by-step: install the spaCy model", expanded=True):
    st.markdown(
        "**1. Install spaCy** (already a project dependency, but confirm with):\n"
        "```bash\n"
        "pip install spacy\n"
        "```\n\n"
        "**2. Download the model** (run once; downloads ~40 MB):\n"
        "```bash\n"
        "python -m spacy download en_core_web_md\n"
        "```\n\n"
        "**3.** Restart Chorus and tick the **NLP Reconstruction** checkbox — "
        "the setup dialog should no longer appear.\n\n"
        "> **Docker users:** bake the model into your image by adding the same "
        "`python -m spacy download en_core_web_md` command to your Dockerfile, "
        "so it is available without a first-run download."
    )

# ── Ollama setup ──────────────────────────────────────────────────────────────
st.header("LLM Reconstruction — Ollama Setup")
st.markdown(
    "The **LLM Reconstruction** feature uses a locally running Ollama model to "
    "attempt to resolve LOW-confidence tokens. It is entirely optional and requires "
    "Ollama to be installed and running separately from Chorus."
)

with st.expander("Step-by-step: install and start Ollama (macOS)", expanded=True):
    st.markdown(
        "**1. Install Ollama:**\n"
        "```bash\n"
        "brew install ollama\n"
        "```\n\n"
        "**2. Start the Ollama server** (keep this terminal open, or run as a "
        "background service):\n"
        "```bash\n"
        "ollama serve\n"
        "```\n\n"
        "**3. Pull the model** (run once; downloads ~2 GB):\n"
        "```bash\n"
        "ollama pull qwen2.5:3b\n"
        "```\n\n"
        "> **Technical or rare-vocabulary audio?** Use `qwen2.5:14b` (~9 GB) "
        "instead, and set `OLLAMA_MODEL=qwen2.5:14b` in your `.env` file — it "
        "retains more long-tail vocabulary at the cost of speed.\n\n"
        "**4.** Restart Chorus and tick the **LLM Reconstruction** checkbox — "
        "the warning should be gone."
    )

with st.expander("Linux / Windows (Docker)"):
    st.markdown(
        "Install Ollama from [ollama.com](https://ollama.com/download) for your "
        "platform. The `ollama serve` and `ollama pull` steps are identical.\n\n"
        "When running Chorus via Docker, set `OLLAMA_BASE_URL` in your `.env` to "
        "point at the host machine:\n"
        "```\n"
        "OLLAMA_BASE_URL=http://host.docker.internal:11434\n"
        "```"
    )

with st.expander("Using a different model"):
    st.markdown(
        "Set `OLLAMA_MODEL` in your `.env` file to any model you have pulled:\n"
        "```\n"
        "OLLAMA_MODEL=qwen2.5:14b\n"
        "```\n"
        "Then restart Chorus. The model name must match exactly what "
        "`ollama list` shows."
    )

# ── FAQ ───────────────────────────────────────────────────────────────────────
st.header("FAQ")

with st.expander("Why does transcription fall back to CPU on Apple Silicon?"):
    st.markdown(
        "Whisper's word-timestamp alignment step uses a DTW algorithm that "
        "requires float64 precision. Apple's MPS GPU framework only supports "
        "float32. Chorus detects this automatically and retries on CPU for the "
        "alignment pass — the transcription output is identical, but the "
        "word-timestamp step runs at CPU speed. You will see a warning in the "
        "console: `MPS does not support float64 required for word-timestamp "
        "alignment — retrying on CPU.`"
    )

with st.expander("Batch mode — when does it activate?"):
    st.markdown(
        "The UI automatically switches to batch mode when you upload **3 or more** "
        "files. In batch mode, files are processed sequentially with a per-file "
        "progress panel. Single- and two-file uploads use the standard single-file "
        "view."
    )

with st.expander("What does the PySoundFile warning mean?"):
    st.markdown(
        "`UserWarning: PySoundFile failed. Trying audioread instead.` appears for "
        "`.m4a` and some compressed formats because `libsndfile` (used by "
        "PySoundFile) does not support them natively. Chorus falls back to "
        "`audioread` automatically. Transcription proceeds normally — this warning "
        "can be safely ignored."
    )

with st.expander("Why is the model loading slowly the first time?"):
    st.markdown(
        "Whisper model weights are downloaded from the internet on first use and "
        "cached locally (in Docker: in a persistent volume; bare-metal: in "
        "`~/.cache/whisper`). Subsequent runs load from the local cache and are "
        "significantly faster."
    )

with st.expander("Can I run Chorus on a machine without a GPU?"):
    st.markdown(
        "Yes. Set `WHISPER_DEVICE=cpu` in your `.env`, or leave it blank — Chorus "
        "will auto-detect and fall back to CPU if no GPU is available. Transcription "
        "will be slower but fully functional. The `base` model is recommended for "
        "CPU-only setups."
    )
