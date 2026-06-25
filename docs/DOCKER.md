# Chorus Engine — Docker Deployment Guide

## Quick Start

### CPU-only (default)
```bash
docker-compose up --build
```
Streamlit UI: http://localhost:8501

### With GPU support
```bash
docker-compose -f Dockerfile.gpu up --build
```

### With Ollama LLM server
```bash
docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
```

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
- `WHISPER_DEVICE` (default: `cpu`)
  - Options: `cpu`, `cuda`, `mps` (Apple Silicon)

**Ollama Integration:**
- `OLLAMA_BASE_URL` (default: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL` (default: `llama3.1:8b`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `20`)

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
