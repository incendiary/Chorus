# Chorus Engine — Docker Deployment Guide

Native installation (see the main [README](../README.md)) is the primary supported
path — it's required for Apple Silicon GPU acceleration and gives you direct access to
the survey script. Use Docker when you want an isolated environment or don't want to
manage a Python virtual environment yourself.

## Installation & Usage

This builds the image locally from source.

1. **Clone the repository:**

   ```bash
   git clone -b v4.0.0 https://github.com/incendiary/Chorus.git
   cd Chorus
   ```

2. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env to change WHISPER_MODEL (default is "base")
   # See CONFIGURATION.md for a full explanation of every option.
   ```

3. **Build and start the application:**

   ```bash
   docker-compose up --build
   ```

4. **Access the UI:**
   Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

**Stopping the service:**

```bash
docker-compose down
```

*Whisper model weights are cached in a persistent Docker volume, so subsequent starts
are significantly faster.*

### With GPU support

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires the NVIDIA Container Toolkit and a Volta (GTX 1070 Ti / RTX series) or newer
GPU. **Apple Silicon Macs cannot use GPU acceleration in Docker** — Docker Desktop on
macOS runs containers inside a Linux VM with no access to the Metal GPU; this is an
architectural limitation, not a configuration issue. Use native installation instead
for MPS acceleration.

**Windows (Docker Desktop + WSL2):** the runtime command is identical to Linux, but the
setup path differs — Docker Desktop uses a WSL2-based Linux VM, and GPU passthrough
happens through that layer.

Prerequisites:
- Windows 10 (21H2+) or Windows 11
- Docker Desktop with the **WSL2 backend** enabled (Settings → General → *Use the WSL2
  based engine*)
- NVIDIA driver **527.41+** on the **Windows host** — do not install CUDA inside WSL2,
  the host driver is all that's needed
- NVIDIA Container Toolkit installed **inside WSL2** (not on Windows itself)

One-time WSL2 setup (run inside your WSL2 terminal, e.g. Ubuntu):

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
curl -sL "https://nvidia.github.io/nvidia-docker/${distribution}/nvidia-docker.list" \
  | sed 's|deb |deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] |' \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

Restart Docker Desktop after installing the toolkit, then verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If `nvidia-smi` fails, check that Docker Desktop is
using the WSL2 backend and that your Windows NVIDIA driver is up to date.

### With Ollama LLM server

```bash
docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up
```

See the README's "Local LLM Integration with Ollama" section for model recommendations.

## Deploy from GHCR

Pre-built images are published to [GitHub Container Registry](https://ghcr.io/incendiary/chorus)
on every tagged release — no local build step required.

**CPU:**

```bash
docker pull ghcr.io/incendiary/chorus:v4.0.0
docker run --rm -p 8501:8501 ghcr.io/incendiary/chorus:v4.0.0
```

**GPU (NVIDIA CUDA):**

```bash
docker pull ghcr.io/incendiary/chorus:v4.0.0-gpu
docker run --rm -p 8501:8501 --gpus all ghcr.io/incendiary/chorus:v4.0.0-gpu
```

Access the UI at [http://localhost:8501](http://localhost:8501).

*See `docker-publish.sh` and `docker-test.sh` in the repo root for building, testing,
and pushing images locally.*

---

## Architecture

### Multi-stage Build (`Dockerfile`)

The Dockerfile uses a two-stage build for optimal image size and build performance:

**Stage 1: Builder**
- Installs system build tools (gcc, git, libsndfile1-dev)
- Creates a Python virtual environment
- Installs all Python dependencies from `requirements.txt`
- Pre-downloads NLTK data (punkt, punkt_tab, stopwords)
- **Result:** A heavyweight intermediate image (~2.5GB)

**Stage 2: Runtime**
- Slim base image (`python:3.11-slim-bookworm`)
- Copies only the venv from the builder stage
- Installs runtime dependencies (FFmpeg, libsndfile1)
- Copies application source code
- Sets up a non-root user (`chorus`) for security
- Configures healthcheck for container orchestration
- **Result:** Lightweight production image (~1.2GB)

### Benefits
- **Faster iteration:** Build cache preserves venv across code changes
- **Security:** Non-root user, no build tools in production image
- **Portability:** Single `FROM python:3.11-slim-bookworm` base
- **Size efficiency:** ~50% smaller than including build tools in final image

---

## Build Optimization

### Docker BuildKit
Enable BuildKit for better caching and parallelization:

```bash
export DOCKER_BUILDKIT=1
docker-compose up --build
```

### .dockerignore
The `.dockerignore` file excludes unnecessary files from the build context to reduce upload time to the Docker daemon.

### Layer Caching Strategy

1. System dependencies → rarely change
2. Python venv → changes when requirements.txt updated
3. Application source → changes frequently
4. NLTK data → rarely changes

This ordering maximizes cache hits during development.

---

## Configuration

### Environment Variables

**Whisper:**
- `WHISPER_MODEL` (default: `base`)
  - Options: `tiny`, `base`, `small`, `medium`, `large`
- `WHISPER_DEVICE` (default: auto-detect — probes CUDA, then MPS, then falls back to CPU)
  - Options: `cpu`, `cuda`, `mps` (Apple Silicon; native only, not available in Docker)

**Ollama Integration:**
- `OLLAMA_BASE_URL` (default: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:3b`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `20`)

See [CONFIGURATION.md](CONFIGURATION.md) for the full reference, including why
`qwen2.5:3b` is recommended over larger models for this task.

---

## Volume Management

### Named Volumes

**whisper_cache**
- Persists Whisper model weights across restarts
- Avoids re-downloading multi-gigabyte models

**ollama_models** (if using `docker-compose.ollama.yml`)
- Persists Ollama model artifacts

### Bind Mounts

**./outputs** - Pipeline results accessible on the host

**./sample_audio** - Test audio files
