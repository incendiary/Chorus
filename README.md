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
   git clone -b v2.1.0 https://github.com/incendiary/Chorus.git
   cd Chorus
   ```

2. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env to change WHISPER_MODEL (default is "base")
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

## Deploy from GHCR

Pre-built images are published to [GitHub Container Registry](https://ghcr.io/incendiary/chorus) on every tagged release. No local build step required.

### CPU

```bash
docker pull ghcr.io/incendiary/chorus:v2.1.0
docker run --rm -p 8501:8501 ghcr.io/incendiary/chorus:v2.1.0
```

### GPU (NVIDIA CUDA)

```bash
docker pull ghcr.io/incendiary/chorus:v2.1.0-gpu
docker run --rm -p 8501:8501 --gpus all ghcr.io/incendiary/chorus:v2.1.0-gpu
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
   python -m venv .venv
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

## Understanding the Output

Chorus produces a final `.md` file in the `outputs/consensus/` directory. This file uses standard Markdown and extended highlighting syntax to indicate confidence levels:

| Rendering | Confidence Tier | Meaning | Recommended Action |
|-----------|-----------------|---------|--------------------|
| Plain text | **HIGH** (≥ 75 %) | Word appears in 3 or 4 variants. | Accept — high agreement. |
| `==highlighted==` | **MEDIUM** (50 %) | Word appears in exactly 2 variants. | Review — split consensus. |
| **~~struck bold~~** | **LOW** (25 %) | Word appears in only 1 variant. | Flag — likely an artefact. |

*Note: The exact threshold percentages are configurable in `config.py`.*

---

## Project Architecture

```text
chorus-engine/
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
├── nlp_reconstructor/        # NLP post-processing
│   └── reconstructor.py      # spaCy grammatical reconstruction
├── llm_reconstructor/        # Optional local LLM post-processing
│   ├── ollama_client.py      # Local Ollama API client wrapper
│   └── reconstructor.py      # LOW-token reconstruction helpers
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
**Licence:** MIT
