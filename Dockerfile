# ─────────────────────────────────────────────────────────────────────────────
# Chorus Engine — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) — installs Python dependencies into a virtual environment
#   Stage 2 (runtime) — copies the venv and application code into a slim image
#
# The final image includes:
#   - Python 3.11 (slim-bookworm base)
#   - FFmpeg (required by Whisper and pydub for audio decoding)
#   - All Python dependencies pre-installed
#   - Streamlit UI exposed on port 8501
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libsndfile1-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download NLTK data packages required by the consensus merger
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True)"


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Chorus Engine"
LABEL description="Multi-pass consensus audio transcription engine"
LABEL version="1.0.0"

# Install runtime system dependencies (FFmpeg is mandatory)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/nltk_data /root/nltk_data

# Set PATH to use the venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Application setup ────────────────────────────────────────────────────────
WORKDIR /app

# Copy application source
COPY . /app/

# Create output directories
RUN mkdir -p /app/outputs/variants \
             /app/outputs/transcripts \
             /app/outputs/consensus \
             /app/sample_audio

# ── Whisper model cache ───────────────────────────────────────────────────────
# The model is downloaded on first run and cached in ~/.cache/whisper.
# Mount a named volume (see docker-compose.yml) to persist across container
# restarts and avoid repeated downloads.
ENV WHISPER_MODEL="base"
ENV WHISPER_DEVICE="cpu"

# ── Streamlit configuration ───────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Health check — verifies Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch the Streamlit UI
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
