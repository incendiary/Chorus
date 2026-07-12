# Chorus Engine

**A multi-pass consensus audio transcription engine powered by OpenAI Whisper.**

Chorus automates high-fidelity audio transcription by applying a multi-pass methodology. It ingests raw audio, generates distinct cleaned variants (high-pass, normalised, denoised), transcribes each independently, and synthesises the results into a single, confidence-weighted "consensus transcript".

This approach dramatically reduces single-model hallucinations and improves word-error rates (WER) on challenging audio by mathematically voting on word-level alignment across multiple acoustic perspectives.

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

## Prerequisites

If running via Docker, you only need:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

If running natively (bare-metal), you require:

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) (must be available on your system `PATH`)

---

## Installation & Usage (Docker)

This is the recommended approach. The Docker image encapsulates the Python environment and FFmpeg dependencies.

1. **Clone the repository:**

   ```bash
   git clone -b v4.0.0 https://github.com/incendiary/Chorus.git
   cd Chorus
   ```

2. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env to change WHISPER_MODEL (default is "base")
   # See docs/CONFIGURATION.md for a full explanation of every option.
   ```

3. **Build and start the application:**

   ```bash
   docker-compose up --build
   ```

4. **Access the UI:**
   Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

### Stopping the service

```bash
docker-compose down
```

*Note: The Whisper model weights are cached in a persistent Docker volume, so subsequent starts will be significantly faster.*

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

### Linux — NVIDIA (Docker)

Follow the prerequisites in `docker-compose.gpu.yml`, then:

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires NVIDIA Container Toolkit and a Volta (GTX 1070 Ti / RTX series) or newer GPU.

---

### Windows — NVIDIA (Docker Desktop + WSL2)

The runtime command is identical to Linux, but the setup path is different. Docker Desktop on Windows uses a WSL2-based Linux VM to run containers, and GPU passthrough happens through that layer.

**Prerequisites:**

- Windows 10 (21H2 or later) or Windows 11
- Docker Desktop for Windows with the **WSL2 backend** enabled (Settings → General → *Use the WSL2 based engine*)
- NVIDIA driver **527.41 or later** installed on the **Windows host** — do not install CUDA inside WSL2, the host driver is all that is needed
- NVIDIA Container Toolkit installed **inside WSL2** (not on Windows itself)

**One-time WSL2 setup (run inside your WSL2 terminal, e.g. Ubuntu):**

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
curl -sL "https://nvidia.github.io/nvidia-docker/${distribution}/nvidia-docker.list" \
  | sed 's|deb |deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] |' \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

Restart Docker Desktop after installing the toolkit.

**Verify GPU passthrough:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If `nvidia-smi` fails, check that Docker Desktop is using the WSL2 backend and that your Windows NVIDIA driver is up to date.

**Start Chorus with GPU:**

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

---

### Apple Silicon (macOS) — MPS

**In Docker: CPU only.** Docker Desktop on macOS runs containers inside a Linux VM (Apple's Virtualization Framework). That VM has no access to the Metal GPU — `WHISPER_DEVICE=mps` cannot work inside a container. There is no workaround; this is an architectural limitation. The standard CPU image runs fine.

**Native (bare-metal): MPS is fully supported and auto-detected.** On Apple Silicon Macs, Chorus probes for MPS at startup and selects it automatically — no `.env` change needed. PyTorch's Metal backend gives roughly 3–5× the speed of CPU inference for the `base` and `small` models.

If you want to be explicit, or override to CPU for testing:

```bash
# .env
WHISPER_DEVICE=mps   # force MPS (Apple Silicon native)
WHISPER_DEVICE=cpu   # force CPU
# leave blank to auto-detect (default)
```

> **Memory note:** The `large` model (~3 GB) may exceed unified memory on 8 GB M-series configurations. Use `WHISPER_MODEL=small` or `medium` on those machines.

If Chorus is run natively on Apple Silicon and the MPS device fails to load (e.g. memory pressure), it automatically falls back to CPU and logs a warning — the transcription will still complete.

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

### Recommended Models (2024)

Select based on available RAM and your quality/speed trade-off:

| Model | RAM | Speed | Quality | Use Case |
|-------|-----|-------|---------|----------|
| **TinyLlama 1.1B** | 2GB | ⚡⚡⚡ Fast | ⭐ Low | Minimal systems (<4GB) |
| **Neural Chat 7B (q4)** | 4GB | ⚡⚡ Balanced | ⭐⭐⭐ Good | Best value (4-8GB systems) |
| **Mistral 7B** | 5GB | ⚡⚡ Balanced | ⭐⭐⭐ Good | Speed/quality balance |
| **Llama2 7B (q4)** | 4GB | ⚡⚡ Balanced | ⭐⭐⭐ Good | Strong reasoning |
| **Llama2 7B** | 15GB | ⚡ Slower | ⭐⭐⭐⭐ Excellent | Full precision (8GB+ RAM) |
| **Neural Chat 13B** | 28GB | ⚡ Slower | ⭐⭐⭐⭐ Excellent | High accuracy (16GB+ RAM) |
| **Dolphin Mixtral 8x7B** | 20GB | ⚡ Slower | ⭐⭐⭐⭐ Excellent | MoE power user (16GB+ RAM) |

**Note:** Model sizes shown are approximate. Quantized models (q4, q3) reduce VRAM by ~50% but slightly lower quality.

### Quick Start

1. **Start Ollama server** (runs on `http://localhost:11434`):
   ```bash
   ollama serve
   ```

2. **Pull a recommended model** (in another terminal):
   ```bash
   # Fast & efficient (recommended for most)
   ollama pull mistral

   # Or smaller for low-RAM systems
   ollama pull neural-chat:7b-v3.1-q4_0
   ```

3. **Start Chorus with Ollama** (uses the model for token recovery):
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up
   ```

4. **In the Chorus UI:**
   - Toggle **"Enable LLM reconstruction"** under Advanced Options
   - Select your model from the dropdown (auto-populated from Ollama)
   - Process audio as usual — LOW-confidence words are reconstructed by the LLM

### Environment Variables

Configure Ollama via `.env`:

```bash
# Ollama server endpoint
OLLAMA_BASE_URL=http://localhost:11434  # Default (native or Docker host)
# OLLAMA_BASE_URL=http://ollama:11434   # Use with docker-compose.ollama.yml

# Model to use for token reconstruction
OLLAMA_MODEL=mistral                    # Default: Mistral 7B

# Timeout for inference (seconds)
OLLAMA_TIMEOUT_SECONDS=30               # Default: 20
```

### Performance Tips

- **CPU only:** Use quantized models (q4_0, q4_K_M, q3_K_M) for 2-4× speedup with minimal quality loss
- **GPU acceleration:** Ollama auto-detects NVIDIA CUDA and Apple MPS. No configuration needed.
- **Multiple models:** Pre-pull multiple models to avoid download delays during processing:
  ```bash
  ollama pull mistral
  ollama pull neural-chat:7b-v3.1
  ```
- **Model tuning:** Lower `OLLAMA_TIMEOUT_SECONDS` if responses are slow, or increase for high-latency networks

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

**Out of memory:**
Use a quantized model (q4_0) or smaller model size. Check system RAM with:
```bash
bash devops-practices/survey-ollama-env.sh
```

**Slow inference:**
Enable GPU acceleration or use a smaller model. Native inference is much faster than Docker.

---

## Deploy from GHCR

Pre-built images are published to [GitHub Container Registry](https://ghcr.io/incendiary/chorus) on every tagged release. No local build step required.

### CPU

```bash
docker pull ghcr.io/incendiary/chorus:v4.0.0
docker run --rm -p 8501:8501 ghcr.io/incendiary/chorus:v4.0.0
```

### GPU (NVIDIA CUDA)

```bash
docker pull ghcr.io/incendiary/chorus:v4.0.0-gpu
docker run --rm -p 8501:8501 --gpus all ghcr.io/incendiary/chorus:v4.0.0-gpu
```

Access the UI at [http://localhost:8501](http://localhost:8501).

*See `docker-publish.sh` and `docker-test.sh` in the repo root for building, testing, and pushing images locally.*

---

## Native Installation (Bare-Metal)

1. **Ensure FFmpeg is installed.**
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Install via `winget` or download binaries.

2. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Streamlit UI:**

   ```bash
   streamlit run ui/app.py
   ```

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
├── tests/                    # Test suite (120+ tests)
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
