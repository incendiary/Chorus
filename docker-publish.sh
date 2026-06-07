#!/usr/bin/env bash
# docker-publish.sh — Build, smoke-test, and push Chorus images to GHCR.
#
# Prerequisites:
#   - GITHUB_TOKEN env var set with write:packages scope
#     (gh auth token may only have read:packages — create a PAT if pushes fail)
#   - Docker daemon running
#   - docker-test.sh in the same directory
#
# Usage:
#   bash docker-publish.sh            # build CPU + GPU, test, push
#   bash docker-publish.sh --dry-run  # print what would happen, exit 0
#   bash docker-publish.sh --no-gpu   # skip GPU variant
#   bash docker-publish.sh --skip-test
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
OWNER="incendiary"
REPO="chorus"
IMAGE="ghcr.io/${OWNER}/${REPO}"
CPU_DOCKERFILE="Dockerfile"
GPU_DOCKERFILE="Dockerfile.gpu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=0
NO_GPU=0
SKIP_TEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1    ;;
    --no-gpu)     NO_GPU=1     ;;
    --skip-test)  SKIP_TEST=1  ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

# ── Detect version ────────────────────────────────────────────────────────────
VERSION=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')
if [[ -z "${VERSION}" ]]; then
  echo "ERROR: no git tag found — tag a release first (e.g. git tag v1.0.0)" >&2
  exit 1
fi
TAG="v${VERSION}"

# ── Dry-run output ────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "=== docker-publish.sh — dry run ==="
  echo "  Version      : ${TAG}"
  echo "  Image        : ${IMAGE}"
  echo "  CPU tags     : ${IMAGE}:${TAG}  ${IMAGE}:latest"
  echo "  GPU tags     : ${IMAGE}:${TAG}-gpu  ${IMAGE}:latest-gpu"
  echo "  CPU Dockerfile : ${CPU_DOCKERFILE} (target: runtime)"
  echo "  GPU Dockerfile : ${GPU_DOCKERFILE} (target: runtime-gpu)"
  echo "  --skip-test  : ${SKIP_TEST}"
  echo "  --no-gpu     : ${NO_GPU}"
  echo "No changes made."
  exit 0
fi

# ── Token check ───────────────────────────────────────────────────────────────
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN is not set." >&2
  echo "  Export a PAT with write:packages scope:" >&2
  echo "  export GITHUB_TOKEN=ghp_..." >&2
  exit 1
fi

# ── GHCR login ────────────────────────────────────────────────────────────────
echo "→ Logging in to ghcr.io …"
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${OWNER}" --password-stdin

# ── CPU build ─────────────────────────────────────────────────────────────────
echo ""
echo "→ Building CPU image (${TAG}) …"
docker build \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:latest" \
  -f "${CPU_DOCKERFILE}" \
  --target runtime \
  .

if [[ "${SKIP_TEST}" -eq 0 ]]; then
  echo "→ Smoke-testing CPU image …"
  bash "${SCRIPT_DIR}/docker-test.sh" --image "${IMAGE}:${TAG}"
fi

# ── GPU build ─────────────────────────────────────────────────────────────────
if [[ "${NO_GPU}" -eq 0 ]]; then
  echo ""
  echo "→ Building GPU image (${TAG}-gpu) …"
  docker build \
    -t "${IMAGE}:${TAG}-gpu" \
    -t "${IMAGE}:latest-gpu" \
    -f "${GPU_DOCKERFILE}" \
    --target runtime-gpu \
    .

  if [[ "${SKIP_TEST}" -eq 0 ]]; then
    echo "→ Smoke-testing GPU image …"
    bash "${SCRIPT_DIR}/docker-test.sh" --image "${IMAGE}:${TAG}-gpu" --gpu
  fi
fi

# ── Push ──────────────────────────────────────────────────────────────────────
echo ""
echo "→ Pushing to GHCR …"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

if [[ "${NO_GPU}" -eq 0 ]]; then
  docker push "${IMAGE}:${TAG}-gpu"
  docker push "${IMAGE}:latest-gpu"
fi

echo ""
echo "✓ Published:"
echo "  ${IMAGE}:${TAG}"
echo "  ${IMAGE}:latest"
if [[ "${NO_GPU}" -eq 0 ]]; then
  echo "  ${IMAGE}:${TAG}-gpu"
  echo "  ${IMAGE}:latest-gpu"
fi
echo ""
echo "If this is the first push, set the package to public:"
echo "  GitHub → Packages → chorus → Package settings → Change visibility → Public"
