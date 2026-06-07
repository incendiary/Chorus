#!/usr/bin/env bash
# docker-test.sh — Smoke-test Chorus Docker images.
#
# The container CMD is a long-running Streamlit server, so we override it with
# a one-shot Python import check that exits 0 on success.
#
# Usage:
#   bash docker-test.sh                                   # test latest CPU image
#   bash docker-test.sh --image ghcr.io/incendiary/chorus:v1.0.0
#   bash docker-test.sh --image ghcr.io/incendiary/chorus:v1.0.0-gpu --gpu
#   bash docker-test.sh --no-gpu                          # skip GPU test
#   bash docker-test.sh --timeout 90
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
IMAGE="ghcr.io/incendiary/chorus:latest"
GPU_MODE=0
SKIP_GPU=0
TIMEOUT=60

# ── Flags ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)    IMAGE="$2";   shift ;;
    --gpu)      GPU_MODE=1          ;;
    --no-gpu)   SKIP_GPU=1          ;;
    --timeout)  TIMEOUT="$2"; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

SMOKE_CMD='python -c "
from audio_processor.pipeline import process_audio
from config import WHISPER_MODEL
print(\"Chorus smoke test passed — WHISPER_MODEL:\", WHISPER_MODEL)
"'

# ── CPU smoke test ────────────────────────────────────────────────────────────
if [[ "${GPU_MODE}" -eq 0 ]]; then
  echo "  [CPU] smoke-testing ${IMAGE} …"
  if docker run --rm \
      --stop-timeout "${TIMEOUT}" \
      "${IMAGE}" \
      python -c "
from audio_processor.pipeline import process_audio
from config import WHISPER_MODEL
print('Chorus CPU smoke OK — WHISPER_MODEL:', WHISPER_MODEL)
"; then
    echo "  [CPU] PASS"
  else
    echo "  [CPU] FAIL" >&2
    exit 1
  fi
fi

# ── GPU smoke test ────────────────────────────────────────────────────────────
if [[ "${GPU_MODE}" -eq 1 ]]; then
  if [[ "${SKIP_GPU}" -eq 1 ]]; then
    echo "  [GPU] SKIP (--no-gpu)"
    exit 0
  fi

  # Detect NVIDIA container runtime
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "  [GPU] SKIP — no NVIDIA container runtime detected on this host"
    exit 0
  fi

  echo "  [GPU] checking nvidia-smi …"
  if ! docker run --rm --gpus all "${IMAGE}" nvidia-smi -L; then
    echo "  [GPU] FAIL — nvidia-smi returned non-zero" >&2
    exit 1
  fi

  echo "  [GPU] smoke-testing ${IMAGE} …"
  if docker run --rm --gpus all \
      --stop-timeout "${TIMEOUT}" \
      "${IMAGE}" \
      python -c "
from config import WHISPER_DEVICE
print('Chorus GPU smoke OK — WHISPER_DEVICE:', WHISPER_DEVICE)
"; then
    echo "  [GPU] PASS"
  else
    echo "  [GPU] FAIL" >&2
    exit 1
  fi
fi
