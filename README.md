# Chorus Engine

**Chorus is not a better transcriber — it's a transcriber that knows when it's wrong.**

A multi-pass consensus audio transcription engine powered by OpenAI Whisper. Chorus ingests raw audio, generates distinct cleaned variants (high-pass, normalised, denoised), transcribes each independently, and synthesises the results into a single, confidence-weighted "consensus transcript".

Voting across acoustic perspectives does not make the words more accurate — benchmarking showed transcript accuracy on par with a single Whisper pass (see [Measured accuracy](#measured-accuracy-v410-benchmark)). What it does produce is **calibrated uncertainty**: word-level confidence tiers that reliably separate the words you can trust from the words you should check. Plain Whisper hands you a transcript with its errors invisibly distributed; Chorus hands you the same transcript with the errors flagged — for your own review, or for a downstream LLM told exactly which words to distrust.

---

## Features

- **Acoustic Pre-processing Pipeline:** Applies dynamic range normalisation, high-pass filtering, and spectral subtraction denoising via `librosa` and `scipy`. Supports VAD-based noise floor detection.
- **Local Transcription:** Fully offline transcription using OpenAI's `whisper` models. No audio data leaves your machine. Word-level timestamps always enabled.
- **Dual Alignment Strategies:** Choose between Needleman-Wunsch sequence alignment (handles insertions/deletions) or legacy positional alignment (fast, word-by-word).
- **Consensus Voting Logic:** Multi-variant alignment that compares transcripts word-for-word, grouping near-matches using NLTK fuzzy similarity.
- **Confidence Highlighting:** The final output is an annotated Markdown document where words are highlighted based on inter-variant agreement.
- **AI Context Pack:** Machine-generated structured document for LLM consumption — includes methodology, confidence data, uncertainty annotations, and usage guidance.
- **Speaker Diarisation:** `pyannote.audio` integration for multi-speaker identification with persistent editable speaker names.
- **Word-Level Subtitles:** SRT/VTT exports use per-word timestamps for precise subtitle synchronisation.
- **Parallel Transcription Orchestration:** Configurable worker pool (`TRANSCRIPTION_PARALLELISM`) with device-aware assignment, including multi-CUDA-device round-robin support.
- **Memory-Optimised Pipeline:** Eager disk writes and prompt memory release for processing long recordings.
- **Streamlit Interface:** A clean, responsive web UI with confidence visualisation, strategy selectors, batch auto-switch, and processing time display.
- **Containerised:** Ready to deploy via Docker and `docker-compose`.

---

## Measured accuracy (v4.1.0 benchmark)

The consensus architecture was benchmarked against single-pass Whisper in July
2026 (15 LibriSpeech utterances, clean and SNR 5 dB noise-augmented conditions,
Whisper `base`, identical text normalisation — full method and per-file numbers
in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)):

| Condition | Single-pass WER | Chorus consensus WER |
|---|---|---|
| Clean | 0.0314 | **0.0288** |
| Noisy (SNR 5 dB) | **0.1024** | 0.1107 |

**The honest headline: Chorus is not a better transcriber — it's a transcriber
that knows when it's wrong.** Consensus does *not* reliably improve raw accuracy
over a single Whisper pass: most per-file scores are identical, and on noisy
audio single-pass was slightly better. What the multi-pass architecture *does*
buy, strongly, is **calibrated uncertainty**: HIGH-tier words were 97.8 %
correct on clean audio and 92.7 % on noisy audio, while every MEDIUM- and
LOW-tier word in the noisy condition was wrong. The confidence tiers tell you
exactly which words to distrust — something a single Whisper pass cannot do.
Treat the tiers, the annotated consensus document, and the machine-readable
`bundle.json` as the product; treat the transcript accuracy as equivalent to
plain Whisper. If raw accuracy is all you need, a larger Whisper model run
once is the cheaper path. (Caveats: single speaker, read speech, small
MEDIUM/LOW sample — see the results file.)

---

## Prerequisites

Native installation (recommended — see below) requires:

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) (must be available on your system `PATH`)

Docker installation instead requires:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

## Installation & Usage (Native / Bare-Metal)

This is the primary supported path — it's required for Apple Silicon (MPS) GPU
acceleration, and gives you direct access to the hardware survey script. See
[Docker installation](docs/DOCKER.md) instead if you'd rather not manage a Python
environment yourself.

1. **Install FFmpeg:**
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: install via `winget` or download binaries.

2. **Clone the repository at the current release:**

   ```bash
   git clone -b v4.1.0 https://github.com/incendiary/Chorus.git
   cd Chorus
   ```

3. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env to change WHISPER_MODEL (default is "base")
   # See docs/CONFIGURATION.md for a full explanation of every option.
   ```

   For local LLM-assisted reconstruction, run
   `bash devops-practices/survey-ollama-env.sh` — it surveys your hardware, recommends
   models, and can write `.env` for you. See
   [Local LLM Integration with Ollama](#local-llm-integration-with-ollama) below.

5. **Run the Streamlit UI:**

   ```bash
   streamlit run ui/app.py
   ```

   Open your browser at [http://localhost:8501](http://localhost:8501).

*Whisper model weights are cached under `~/.cache/whisper` after first use, so
subsequent runs are significantly faster.*

---

## Using the Web UI

Once the UI is open at [http://localhost:8501](http://localhost:8501), configuration lives in the sidebar and the workflow runs left-to-right in the main panel.

### 1. Configure (sidebar)

- **Settings preset — Max / Background:** click **🔍 Apply** to survey your machine's RAM, CPU, and GPU in-process and auto-select a model size, compute device, and parallelism for you. **Max** picks the largest model your hardware can run with full parallelism (machine dedicated to Chorus); **Background** steps one model tier down and pins parallelism to 1 (machine stays responsive for other work). This is the fastest way to get sensible settings without knowing the trade-offs yourself — use it first, then fine-tune the controls below if you want to.
- **Model size:** `tiny` → `large`. Larger is more accurate but slower; `base` is the recommended default for CPU-only machines.
- **Consensus models:** optionally select more than one Whisper model size to transcribe with — Chorus votes across all of them for extra confidence signal (slower, more accurate).
- **Compute device:** Auto-detect (recommended), CPU, NVIDIA CUDA, or Apple MPS.
- **Auto parallelism / worker count:** how many variants transcribe simultaneously.
- **Alignment strategy** and **noise floor mode:** advanced tuning — defaults are sensible for most audio.
- **Speaker Diarisation, LLM reconstruction, NLP reconstruction:** optional toggles. If a required local dependency (Ollama, a spaCy model) isn't set up yet, a setup dialog explains exactly what to install rather than failing silently.
- **Export formats:** PDF, Word (.docx), and word-level SRT subtitles are opt-in checkboxes — the annotated Markdown, best-guess plain text, and JSON bundle are always generated.

### 2. Upload and run (main panel)

1. **Upload Audio Files** — drag and drop, or browse. Any FFmpeg-supported format; multiple files at once.
2. For 2 uploaded files, choose **Sequential** (see each file's results as it finishes) or **All at once** (all files process before any results show); 3+ files always run in batch mode automatically.
3. Click **▶ Start Chorus** to begin. Progress and stage information appear as each file processes.

### 3. Understanding the output

Chorus produces a final `.md` file in the `outputs/consensus/` directory. This file uses standard Markdown and extended highlighting syntax to indicate confidence levels:

| Rendering | Confidence Tier | Meaning | Recommended Action |
|-----------|-----------------|---------|--------------------|
| Plain text | **HIGH** (≥ 75 %) | Word appears in 3 or 4 variants. | Accept — high agreement. |
| `==highlighted==` | **MEDIUM** (50 %) | Word appears in exactly 2 variants. | Review — split consensus. |
| **~~struck bold~~** | **LOW** (25 %) | Word appears in only 1 variant. | Flag — likely an artefact. |

*Note: The exact threshold percentages are configurable in `config.py`.*

Alongside the annotated Markdown, every run also writes a `{stem}_best_guess.txt` file — a clean, fully human-readable transcript with no brackets, highlighting, or statistics at all. Every MEDIUM/LOW-confidence position is resolved to its single best-guess word (the highest-agreement candidate already selected by the consensus vote), making it suitable for distribution to non-technical readers or downstream NLP processing.

Download individual formats from the results panel, or **Download All** for a ZIP bundle. Every completed run is also browsable later from the **Past Jobs** page in the sidebar — no need to keep the browser tab open.

---

## GPU Acceleration

Chorus probes hardware in this order and selects the first available option automatically: **CUDA** (NVIDIA GPU) → **MPS** (Apple Silicon) → **CPU**. Override explicitly via `.env`:

```bash
WHISPER_DEVICE=cuda  # force NVIDIA CUDA
WHISPER_DEVICE=mps   # force Apple Silicon MPS
WHISPER_DEVICE=cpu   # force CPU
# leave blank to auto-detect (default)
```

### Apple Silicon (macOS) — MPS

**Native only — MPS is fully supported and auto-detected**, no `.env` change needed. PyTorch's Metal backend gives roughly 3–5× the speed of CPU inference for the `base` and `small` models. If MPS fails to load (e.g. memory pressure), Chorus automatically falls back to CPU and logs a warning — transcription still completes.

**Not available in Docker at all** — Docker Desktop on macOS runs containers inside a Linux VM with no access to the Metal GPU. This is an architectural limitation with no workaround; use native installation for MPS.

> **Memory note:** The `large` model (~3 GB) may exceed unified memory on 8 GB M-series configurations. Use `WHISPER_MODEL=small` or `medium` on those machines.

### NVIDIA CUDA

Native installation auto-detects CUDA if PyTorch's CUDA build is installed and a compatible driver is present (Volta / GTX 1070 Ti / RTX series or newer). For **Docker** GPU setup (Linux, and Windows via WSL2), see [docs/DOCKER.md](docs/DOCKER.md#with-gpu-support).

---

## Local LLM Integration with Ollama

Chorus integrates with [Ollama](https://ollama.ai) for optional local LLM-based token reconstruction. This improves LOW-confidence word recovery without cloud dependencies.

### Survey Your System & Auto-Setup Ollama

Run this interactive script to survey your hardware and optionally set up Ollama:

```bash
bash devops-practices/survey-ollama-env.sh
```

This script:
1. **Surveys your system** for available RAM, CPU cores, GPU (NVIDIA CUDA, Apple Silicon MPS, etc.), and free disk space
2. **Recommends models** — both Ollama LLM models and a `WHISPER_MODEL` size — based on your hardware constraints
3. **Offers to start Ollama** if not running (`ollama serve`)
4. **Multi-select model menu** — presents all recommended Ollama models as a numbered list; enter space-separated numbers to pull several at once (e.g. `1 3`), or `0` to skip. Already-installed models are labelled in-line.
5. **Applies settings to `.env`** — offers the recommended `WHISPER_MODEL`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL` values, showing your current `.env` value alongside each. Enter numbers, `all`, or `0` to skip. If `.env` doesn't exist yet, it's created from `.env.example` first.
6. **Provides ready-to-run commands** for both Docker and bare-metal deployments

Example output:
```
Select models to install:
  0) Skip — don't pull any models
  1) qwen2.5:3b                          Qwen2.5 3B — fast default; reliably follows token-only output (~2GB)
  2) qwen2.5:14b                         Qwen2.5 14B — better for technical/rare vocabulary (~9GB)
Enter numbers to install (space-separated), or 0 to skip: 1

Apply Settings to .env
  1) WHISPER_MODEL         = medium
  2) OLLAMA_MODEL           = qwen2.5:3b
  3) OLLAMA_BASE_URL        = http://localhost:11434
Enter numbers to apply (space-separated), 'all', or 0 to skip: all

Starting Chorus (Docker)
  export OLLAMA_MODEL='qwen2.5:3b'
  docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up

Starting Chorus (Bare Metal/Native)
  export OLLAMA_MODEL='qwen2.5:3b'
  streamlit run ui/app.py
```

*Model recommendations favour reliable, low-latency instruction-following over raw size for this task (picking one word from a short candidate list) — see [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the reasoning, and the optional larger model offered for jargon-heavy transcripts.*

### Install Ollama

1. **Download and install:**
   - [macOS](https://ollama.ai/download/Ollama-darwin.zip)
   - [Linux](https://ollama.ai/download/ollama-linux-amd64.tgz)
   - [Windows (preview)](https://ollama.ai/download/OllamaSetup.exe)

2. **Verify installation:**
   ```bash
   ollama --version
   ```

### Troubleshooting

**"Connection refused" (Ollama not running):**
```bash
ollama serve  # Start server in a new terminal
```

**"Model not found" (model not pulled):**
```bash
ollama pull <model-name>
ollama list  # See installed models
```

**Out of memory:** use `qwen2.5:3b` (the default) rather than `qwen2.5:14b` — see
[Choosing a model](docs/CONFIGURATION.md#choosing-a-model) for the reasoning. Re-check
your system RAM with:
```bash
bash devops-practices/survey-ollama-env.sh
```

**Slow inference:** GPU acceleration helps (see [GPU Acceleration](#gpu-acceleration));
otherwise the 3B default is already the faster option — `qwen2.5:14b` is deliberately
slower in exchange for better rare-vocabulary handling.

---

## Docker

Prefer an isolated environment over managing a Python venv? See
[docs/DOCKER.md](docs/DOCKER.md) for the full Docker installation, GPU passthrough
(Linux and Windows/WSL2), and GHCR pre-built image instructions.

```bash
git clone -b v4.1.0 https://github.com/incendiary/Chorus.git
cd Chorus
docker-compose up --build
```

Note: Apple Silicon GPU (MPS) acceleration is **not available in Docker** — use native
installation above for that.

---

## Project Architecture

```text
chorus-engine/
├── chorus/                   # Stable public API façade (see "Library usage")
│   └── __init__.py           # Re-exports the supported 4.x entry points
├── audio_processor/          # Stage 1: Audio cleaning pipeline
│   ├── filters.py            # High-pass, norm, denoise (VAD + fixed modes)
│   └── pipeline.py           # Orchestrates variant generation (memory-optimised)
├── transcription_engine/     # Stage 2: Whisper integration
│   ├── whisper_engine.py     # Local model wrapper (word-level timestamps)
│   └── orchestrator.py       # Runs Whisper over all audio variants
├── consensus_merger/         # Stage 3: Voting and alignment
│   ├── alignment.py          # Strategy dispatcher (positional + sequence)
│   ├── sequence_alignment.py # Needleman-Wunsch word-level alignment
│   ├── renderer.py           # Markdown document generation
│   └── merger.py             # Consensus orchestrator
├── diarisation/              # Speaker identification
│   └── diariser.py           # pyannote integration + speaker name persistence
├── export_engine/            # Multi-format export
│   ├── exporter.py           # PDF, DOCX, SRT, VTT, ZIP, plain text
│   └── ai_context.py         # AI/LLM context pack generator
├── reconstruction/           # LOW-token reconstruction (strategy-based)
│   ├── __init__.py           # Unified reconstruct(votes, *, strategy) entry point
│   ├── nlp.py                # spaCy grammatical reconstruction ("nlp" strategy)
│   ├── llm.py                # Local Ollama reconstruction ("llm" strategy)
│   └── ollama_client.py      # Local Ollama API client wrapper
├── ui/                       # Stage 4: Web interface
│   └── app.py                # Streamlit dashboard (confidence vis, batch mode)
├── tests/                    # Test suite (244 tests)
│   ├── test_integration.py   # Full pipeline integration tests
│   ├── test_alignment.py     # Positional alignment tests
│   ├── test_sequence_alignment.py  # Needleman-Wunsch tests
│   ├── test_speaker_names.py # Speaker persistence tests
│   ├── test_ai_context.py    # AI context pack tests
│   └── ...                   # Audio processor, exporter, merger, reconstructor
├── config.py                 # Central configuration and thresholds
├── pipeline_runner.py        # End-to-end CLI entry point
├── Dockerfile                # CPU image (python:3.11-slim-bookworm)
├── Dockerfile.gpu            # CUDA image (nvidia/cuda:12.1.1-cudnn8)
├── docker-compose.yml        # CPU service definition
├── docker-compose.gpu.yml    # GPU override (NVIDIA device reservations)
└── requirements.txt          # Pinned Python dependencies
```

### Library usage

Once installed (`pip install .`), Chorus exposes a single, stable public API
through the top-level `chorus` package. Import the supported entry points from
there rather than reaching into the internal modules, whose paths may change
between minor releases:

```python
from chorus import run_pipeline, run_batch

# Transcribe a single file end to end.
results = run_pipeline(audio_path="meeting.wav", language="en")

# Or process a directory of files unattended.
run_batch(["recordings/"], recursive=True)
```

The full public surface is `run_pipeline`, `run_batch`,
`merge_transcripts_with_votes`, `export_all`, and `export_transcript_bundle`.

---

## Breaking changes in v4.0.0

Chorus 4.0.0 introduces breaking import-path changes as part of establishing a
stable public API. If you import Chorus internals directly (rather than through
the Streamlit UI or CLI), update as follows:

**New stable public API.** The recommended way to use Chorus as a library is now:

```python
from chorus import run_pipeline, run_batch, merge_transcripts_with_votes, export_all, export_transcript_bundle
```

These names are committed to staying stable; deeper module paths are internal
and may change without notice in future minor/patch releases.

**Reconstruction modules consolidated.** `nlp_reconstructor/` and
`llm_reconstructor/` have been merged into a single `reconstruction/` package
with one entry point:

```python
# Old (removed in 4.0.0):
from nlp_reconstructor.reconstructor import reconstruct as nlp_reconstruct
from llm_reconstructor.reconstructor import reconstruct as llm_reconstruct

# New:
from reconstruction import reconstruct
reconstruct(votes, strategy="nlp")   # was nlp_reconstructor
reconstruct(votes, strategy="llm")   # was llm_reconstructor
```

`enable_nlp` / `enable_llm` flag behaviour on `run_pipeline()` and in the UI is
unchanged — only the underlying import paths moved.

**Runtime dependencies now declared.** `pip install chorus-engine` (or `pip install .`)
now installs its runtime dependencies directly; previously only `requirements.txt`
carried them.

## Documentation

| Document | Description |
|---|---|
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Full reference for every configurable option — Whisper models, alignment strategy, noise mode, parallelism, LLM reconstruction, and more. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Internal module design and data-flow walkthrough. |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker build, tagging, and publish workflow. |
| [docs/SCALABILITY.md](docs/SCALABILITY.md) | Scaling guidance for large batch workloads. |
| [docs/CHORUS_FOR_LLMS.md](docs/CHORUS_FOR_LLMS.md) | Explains Chorus and its output files to a language model — paste it alongside Chorus output for downstream LLM analysis. |

---

## Roadmap Tracking

Roadmap planning and release completion tracking are maintained in [ROADMAP.md](ROADMAP.md) only.

Use [ROADMAP.md](ROADMAP.md) as the source of truth for:

- planned work items;
- completed items and their release versions;
- release progress status.

---

## Detailed Execution Plan

This section captures the active delivery framework for Chorus so decisions stay aligned across product, engineering, quality, and release operations.

### Planning Framework

Chorus uses a three-layer workflow inspired by the Karpathy operating model:

1. Spec layer: define outcomes, scope, constraints, and success criteria before implementation.
2. Verify layer: validate claims with tests, repository checks, and UX walkthroughs.
3. Environment layer: persist reusable decisions, conventions, and guardrails for future sessions.

### Product Priorities

1. Accuracy and confidence clarity for batch processing workflows.
2. Informed-user controls: every major option should explain trade-offs and likely impact.
3. Production-grade UX for enterprise/internal users, desktop-first.
4. Operational trust: predictable releases, synced docs, and explicit roadmap traceability.

### Delivery Phases

1. Alignment and discovery
   - Interview stakeholders on user profiles, critical workflows, non-goals, and acceptance criteria.
   - Map key user journeys and high-risk failure points.

2. Targeted implementation
   - Deliver high-impact, reviewable changes in small PRs.
   - Preserve existing functionality; avoid speculative rewrites.

3. Verification and hardening
   - Run focused tests first, then full test suite.
   - Validate docs, versioning, git tags, and roadmap metadata consistency.

4. Release hygiene
   - Merge through atomic PRs.
   - Tag releases and keep README, pyproject version, and roadmap status in sync.

### Quality Gates

A change is complete only when all of the following are true:

1. Functionality is preserved and tests pass.
2. User-facing copy is clear and reflects current behaviour.
3. Version references and release tags are consistent.
4. Completed work is reflected in [ROADMAP.md](ROADMAP.md).

### Current Focus Areas

1. UX clarity for batch consensus processing and confidence interpretation.
2. Better progress/error feedback to reduce support questions.
3. Continued modularisation and consistency in UI patterns.
4. Maintain strict release/documentation sync enforced by tests.

---

## Autonomous Maintenance Handoff

> **Note to Maintainers:** This codebase has been hardened and prepared for autonomous maintenance via Claude Code. It adheres to the security and tooling standards established by the [Bedrock](https://github.com/incendiary/Bedrock) repository.
>
> When initialising this repository:
>
> 1. Run `./gh_init.sh <your-org> <repo-name>` to create the GitHub repository, enforce branch protection, and require signed commits.
> 2. Open the project in Claude Code — the `CLAUDE.md` file will load automatically to enforce architectural rules, unit testing standards, and development guidelines.
> 3. The CI pipeline (`.github/workflows/ci.yml`) enforces a strict three-layer secret scan (gitleaks + TruffleHog + detect-secrets) before any code is linted or tested.

---

**Author:** Manus AI
**Licence:** [Prosperity Public License 4.0.0](LICENSE) — free for non-commercial use; a commercial licence must be arranged with the licensor for any commercial use.
