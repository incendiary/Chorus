#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# survey-ollama-env.sh — Environment survey and Ollama model recommendations
#
# This script probes your system for:
#   - Available memory (RAM)
#   - GPU availability and VRAM
#   - CPU core count
#   - Disk space
#   - Ollama installation status
#
# Based on findings, it recommends appropriate Ollama models and configuration,
# lets you select which models to pull, and optionally writes recommendations
# directly to your .env file.
#
# Usage:
#   bash devops-practices/survey-ollama-env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────────────────────────

# set_env_var KEY VALUE FILE — replace existing key or append a new one
set_env_var() {
  local key="$1" value="$2" file="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i.bak "s|^${key}=.*|${key}=${value}|" "$file" && rm -f "${file}.bak"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

# resolve_repo_root — walk up from the script's directory to the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Chorus Ollama Environment Survey${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

# ─────────────────────────────────────────────────────────────────────────────
# System Info
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}[1/6] System Information${NC}"
UNAME=$(uname -s)
echo "OS: $UNAME"

if [[ "$UNAME" == "Darwin" ]]; then
  TOTAL_MEM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
  CPU_CORES=$(sysctl -n hw.ncpu)
elif [[ "$UNAME" == "Linux" ]]; then
  TOTAL_MEM_GB=$(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE) / 1024 / 1024 / 1024))
  CPU_CORES=$(nproc)
else
  echo -e "${RED}Unsupported OS: $UNAME${NC}"
  exit 1
fi

echo "Total RAM: ${TOTAL_MEM_GB}GB"
echo "CPU Cores: $CPU_CORES"

# ─────────────────────────────────────────────────────────────────────────────
# GPU Detection
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[2/6] GPU Detection${NC}"

GPU_TYPE="none"
GPU_VRAM_GB=0

if [[ "$UNAME" == "Darwin" ]]; then
  if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Apple M"; then
    GPU_TYPE="Apple Silicon (MPS)"
    TOTAL_MEM_GB_UNIFIED=$TOTAL_MEM_GB
    echo "✓ Apple Silicon detected (MPS support)"
    echo "  Unified memory: ${TOTAL_MEM_GB_UNIFIED}GB (shared with CPU)"
  fi
elif [[ "$UNAME" == "Linux" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    GPU_TYPE="NVIDIA CUDA"
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
    echo "✓ NVIDIA GPU detected"
    echo "  Model: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  VRAM: ${GPU_VRAM_GB}GB"
  fi
fi

if [[ "$GPU_TYPE" == "none" ]]; then
  echo "✗ No GPU detected (CPU-only mode)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Ollama Installation
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[3/6] Ollama Installation Status${NC}"

OLLAMA_INSTALLED=false
OLLAMA_RUNNING=false

if command -v ollama &>/dev/null; then
  OLLAMA_INSTALLED=true
  echo "✓ Ollama is installed"
  OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
  echo "  Version: $OLLAMA_VERSION"

  if curl -s http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_RUNNING=true
    echo "✓ Ollama server is running (http://localhost:11434)"
    INSTALLED_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ' | sed 's/,$//')
    echo "  Installed models: $INSTALLED_MODELS"
  else
    echo "✗ Ollama server is not running"
    echo "  Start it with: ollama serve"
  fi
else
  echo "✗ Ollama is not installed"
  echo "  Install from: https://ollama.ai"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Disk Space
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[4/6] Disk Space${NC}"

if [[ "$UNAME" == "Darwin" ]]; then
  DISK_FREE_GB=$(df / | tail -1 | awk '{print $4/1024/1024}')
elif [[ "$UNAME" == "Linux" ]]; then
  DISK_FREE_GB=$(df / | tail -1 | awk '{print $4/1024/1024}')
fi

echo "Free space on /: ${DISK_FREE_GB%.0f}GB"

# ─────────────────────────────────────────────────────────────────────────────
# Model Recommendations
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[5/6] Model Recommendations${NC}"

RECOMMENDED_MODELS=()
REASONING=""

if [[ "$TOTAL_MEM_GB" -lt 4 ]]; then
  RECOMMENDED_MODELS=("tinyllama:latest" "qwen2.5:0.5b")
  REASONING="Low RAM (< 4GB): small, low-memory models"
elif [[ "$TOTAL_MEM_GB" -lt 8 ]]; then
  RECOMMENDED_MODELS=("qwen2.5:3b" "llama3.2:3b" "mistral:latest")
  REASONING="Medium RAM (4-8GB): compact models with quantization"
elif [[ "$TOTAL_MEM_GB" -lt 16 ]]; then
  RECOMMENDED_MODELS=("mistral:latest" "llama3.1:8b" "qwen2.5:7b")
  REASONING="Good RAM (8-16GB): full-precision ~7-8B models"
else
  RECOMMENDED_MODELS=("mistral:latest" "llama3.1:8b" "qwen2.5:14b" "gemma2:9b")
  REASONING="Excellent RAM (16GB+): larger models with headroom to spare"
fi

if [[ "$GPU_TYPE" != "none" ]]; then
  if [[ "$GPU_VRAM_GB" -ge 8 ]] || [[ "$GPU_TYPE" == "Apple Silicon (MPS)" ]]; then
    RECOMMENDED_MODELS+=("gemma2:27b")
    REASONING+=" | GPU detected: can use larger models"
  fi
fi

# check_model_exists MODEL:TAG — verify a tag actually resolves in Ollama's
# registry before recommending it. A previous version of this script
# recommended neural-chat:13b and tiny-llama:latest, neither of which ever
# existed — this check prevents that class of bug from silently recurring as
# tags get renamed or deprecated upstream. Network failures are treated as
# "assume valid" so an offline machine doesn't lose all recommendations.
check_model_exists() {
  local spec="$1" model tag code
  model="${spec%%:*}"
  tag="${spec#*:}"
  code=$(curl -s -o /dev/null -m 4 -w "%{http_code}" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://registry.ollama.ai/v2/library/${model}/manifests/${tag}" 2>/dev/null || echo "000")
  [[ "$code" == "200" || "$code" == "000" ]]
}

VALIDATED_MODELS=()
for m in "${RECOMMENDED_MODELS[@]}"; do
  if check_model_exists "$m"; then
    VALIDATED_MODELS+=("$m")
  else
    echo -e "${YELLOW}  (skipping $m — no longer available in the Ollama library)${NC}"
  fi
done
RECOMMENDED_MODELS=("${VALIDATED_MODELS[@]}")
if [[ "${#RECOMMENDED_MODELS[@]}" -eq 0 ]]; then
  RECOMMENDED_MODELS=("mistral:latest")
fi

# ── Whisper model recommendation ─────────────────────────────────────────────

WHISPER_RECOMMENDED="base"
WHISPER_REASONING=""

if [[ "$GPU_TYPE" == "NVIDIA CUDA" ]]; then
  if [[ "$GPU_VRAM_GB" -ge 10 ]]; then
    WHISPER_RECOMMENDED="large"
    WHISPER_REASONING="NVIDIA GPU with ${GPU_VRAM_GB}GB VRAM — large model fully supported"
  elif [[ "$GPU_VRAM_GB" -ge 5 ]]; then
    WHISPER_RECOMMENDED="medium"
    WHISPER_REASONING="NVIDIA GPU with ${GPU_VRAM_GB}GB VRAM — medium is the best practical fit"
  elif [[ "$GPU_VRAM_GB" -ge 2 ]]; then
    WHISPER_RECOMMENDED="small"
    WHISPER_REASONING="NVIDIA GPU with ${GPU_VRAM_GB}GB VRAM — small balances speed and accuracy"
  else
    WHISPER_RECOMMENDED="base"
    WHISPER_REASONING="NVIDIA GPU with low VRAM — base recommended to avoid OOM"
  fi
elif [[ "$GPU_TYPE" == "Apple Silicon (MPS)" ]]; then
  if [[ "$TOTAL_MEM_GB" -ge 32 ]]; then
    WHISPER_RECOMMENDED="large"
    WHISPER_REASONING="Apple Silicon with ${TOTAL_MEM_GB}GB unified memory — large model fits comfortably"
  elif [[ "$TOTAL_MEM_GB" -ge 16 ]]; then
    WHISPER_RECOMMENDED="medium"
    WHISPER_REASONING="Apple Silicon with ${TOTAL_MEM_GB}GB unified memory — medium recommended"
  elif [[ "$TOTAL_MEM_GB" -ge 8 ]]; then
    WHISPER_RECOMMENDED="small"
    WHISPER_REASONING="Apple Silicon with ${TOTAL_MEM_GB}GB unified memory — small is the safe choice (large may OOM)"
  else
    WHISPER_RECOMMENDED="base"
    WHISPER_REASONING="Apple Silicon with ${TOTAL_MEM_GB}GB unified memory — base keeps headroom for the OS"
  fi
else
  if [[ "$TOTAL_MEM_GB" -ge 16 ]]; then
    WHISPER_RECOMMENDED="medium"
    WHISPER_REASONING="CPU-only with ${TOTAL_MEM_GB}GB RAM — medium works but expect slower transcription"
  elif [[ "$TOTAL_MEM_GB" -ge 8 ]]; then
    WHISPER_RECOMMENDED="small"
    WHISPER_REASONING="CPU-only with ${TOTAL_MEM_GB}GB RAM — small gives a good accuracy/speed balance"
  elif [[ "$TOTAL_MEM_GB" -ge 4 ]]; then
    WHISPER_RECOMMENDED="base"
    WHISPER_REASONING="CPU-only with ${TOTAL_MEM_GB}GB RAM — base is the practical default"
  else
    WHISPER_RECOMMENDED="tiny"
    WHISPER_REASONING="Low RAM (${TOTAL_MEM_GB}GB) — tiny minimises memory footprint"
  fi
fi

get_model_info() {
  case "$1" in
  "tinyllama:latest") echo "TinyLlama 1.1B — ultra-fast, minimal memory (~1GB)" ;;
  "qwen2.5:0.5b") echo "Qwen2.5 0.5B — smallest capable option (~1GB)" ;;
  "qwen2.5:3b") echo "Qwen2.5 3B — compact, strong for its size (~2GB)" ;;
  "llama3.2:3b") echo "Llama 3.2 3B — Meta's small model, good general use (~2GB)" ;;
  "mistral:latest") echo "Mistral 7B — reliable speed/quality balance (~4GB)" ;;
  "llama3.1:8b") echo "Llama 3.1 8B — strong modern all-rounder (~5GB)" ;;
  "qwen2.5:7b") echo "Qwen2.5 7B — leading open-weight 7B for reasoning (~5GB)" ;;
  "qwen2.5:14b") echo "Qwen2.5 14B — top-tier reasoning at this size (~9GB)" ;;
  "gemma2:9b") echo "Gemma 2 9B — Google's model, strong instruction following (~5GB)" ;;
  "gemma2:27b") echo "Gemma 2 27B — high quality for larger/GPU systems (~16GB)" ;;
  *) echo "Unknown model" ;;
  esac
}

echo "Reasoning: $REASONING"
echo -e "\n${GREEN}Recommended models for your system:${NC}"

for model in "${RECOMMENDED_MODELS[@]}"; do
  echo "  • $model"
  echo "    $(get_model_info "$model")"
done

echo ""
echo "  These are a starting point, not an exhaustive ranking — browse the"
echo "  full current library: https://ollama.com/library?sort=popular"

echo ""
echo -e "${GREEN}Recommended Whisper model for transcription:${NC}"
echo "  WHISPER_MODEL=${WHISPER_RECOMMENDED}"
echo "  Reason: $WHISPER_REASONING"
echo ""
echo "  See docs/CONFIGURATION.md for full model comparison."

# ─────────────────────────────────────────────────────────────────────────────
# Setup Instructions & Interactive Actions
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[6/6] Setup & Installation${NC}"

SELECTED_OLLAMA_MODEL=""

# Step 1: Ollama Installation
if [[ "$OLLAMA_INSTALLED" == false ]]; then
  echo -e "${RED}✗ Ollama is not installed.${NC}"
  echo ""
  echo "Manual installation required:"
  echo "  1. Visit: https://ollama.ai/download"
  echo "  2. Download for your OS ($UNAME)"
  echo "  3. Follow installation instructions"
  echo "  4. Then re-run this script"
  echo ""
else
  echo -e "${GREEN}✓ Ollama is installed${NC}"
fi

# Step 2: Start Ollama Server
if [[ "$OLLAMA_INSTALLED" == true ]]; then
  if [[ "$OLLAMA_RUNNING" == false ]]; then
    echo ""
    echo -e "${YELLOW}Ollama server is not running.${NC}"
    printf "Would you like to start Ollama server now? (y/n) "
    read -r REPLY || true
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
      echo "Starting Ollama server..."
      nohup ollama serve >/tmp/ollama.log 2>&1 &
      OLLAMA_PID=$!
      echo "Ollama started with PID $OLLAMA_PID"
      echo "Waiting for server to start..."
      sleep 3
      if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo -e "${GREEN}✓ Ollama server is running${NC}"
        OLLAMA_RUNNING=true
      else
        echo -e "${RED}✗ Failed to start Ollama. Try manually: ollama serve${NC}"
      fi
    fi
  else
    echo -e "${GREEN}✓ Ollama server is running${NC}"
  fi
fi

# is_model_installed MODEL:TAG — exact match against installed models.
# A naive substring match on the model name (stripping ":tag") gives false
# positives whenever a differently-tagged or differently-named model shares a
# prefix — e.g. "qwen2.5:14b" would wrongly show as installed if only
# "qwen2.5-coder:14b" is present. Match the full "name:tag" string instead.
is_model_installed() {
  local spec="$1" installed
  for installed in $CURRENT_INSTALLED; do
    [[ "$installed" == "$spec" ]] && return 0
  done
  return 1
}

# Step 3: Multi-select model menu
if [[ "$OLLAMA_RUNNING" == true ]]; then
  CURRENT_INSTALLED=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')

  echo ""
  echo -e "${YELLOW}Select models to install:${NC}"
  echo "  0) Skip — don't pull any models"

  IDX=1
  for model in "${RECOMMENDED_MODELS[@]}"; do
    if is_model_installed "$model"; then
      STATUS="${GREEN}(already installed)${NC}"
    else
      STATUS=""
    fi
    printf "  %d) %-38s %s\n" "$IDX" "$model" "$(get_model_info "$model")"
    if [[ -n "$STATUS" ]]; then
      echo -e "     $STATUS"
    fi
    IDX=$((IDX + 1))
  done

  echo ""
  printf "Enter numbers to install (space-separated), or 0 to skip: "
  read -r MODEL_SELECTION || true

  MODELS_TO_PULL=()
  if [[ -n "$MODEL_SELECTION" ]] && [[ "$MODEL_SELECTION" != "0" ]]; then
    for num in $MODEL_SELECTION; do
      if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -ge 1 ]] && [[ "$num" -le "${#RECOMMENDED_MODELS[@]}" ]]; then
        MODELS_TO_PULL+=("${RECOMMENDED_MODELS[$((num - 1))]}")
      fi
    done
  fi

  if [[ "${#MODELS_TO_PULL[@]}" -gt 0 ]]; then
    echo ""
    for model in "${MODELS_TO_PULL[@]}"; do
      if is_model_installed "$model"; then
        echo -e "${GREEN}✓ $model is already installed — skipping${NC}"
      else
        echo "Pulling $model (this may take a few minutes)..."
        if ollama pull "$model"; then
          echo -e "${GREEN}✓ $model pulled successfully${NC}"
        else
          echo -e "${RED}✗ Failed to pull $model. Try manually: ollama pull $model${NC}"
        fi
      fi
    done
    # Use the first successfully-selected model as the recommended OLLAMA_MODEL
    SELECTED_OLLAMA_MODEL="${MODELS_TO_PULL[0]}"
  fi
fi

PRIMARY_MODEL="${SELECTED_OLLAMA_MODEL:-${RECOMMENDED_MODELS[0]}}"

echo ""
echo -e "${GREEN}Ready to use Chorus with Ollama!${NC}"
echo ""
echo -e "${YELLOW}Starting Chorus (Docker)${NC}"
echo "To start Chorus with LLM reconstruction enabled:"
echo ""
echo "  export OLLAMA_MODEL='$PRIMARY_MODEL'"
echo "  export OLLAMA_BASE_URL='http://localhost:11434'"
echo "  docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up"
echo ""
echo -e "${YELLOW}Starting Chorus (Bare Metal/Native)${NC}"
echo "To start Chorus with LLM reconstruction enabled:"
echo ""
echo "  export OLLAMA_MODEL='$PRIMARY_MODEL'"
echo "  export OLLAMA_BASE_URL='http://localhost:11434'"
echo "  streamlit run ui/app.py"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# .env Configuration
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}[7/7] Apply Settings to .env${NC}"

# Build the list of recommended env vars
ENV_REC_KEYS=()
ENV_REC_VALUES=()
ENV_REC_DESCS=()

ENV_REC_KEYS+=("WHISPER_MODEL")
ENV_REC_VALUES+=("$WHISPER_RECOMMENDED")
ENV_REC_DESCS+=("Whisper model size  (${WHISPER_REASONING})")

ENV_REC_KEYS+=("OLLAMA_MODEL")
ENV_REC_VALUES+=("$PRIMARY_MODEL")
ENV_REC_DESCS+=("Ollama model for LLM reconstruction")

ENV_REC_KEYS+=("OLLAMA_BASE_URL")
ENV_REC_VALUES+=("http://localhost:11434")
ENV_REC_DESCS+=("Ollama server address")

echo ""
echo "The following settings are recommended for your system:"
echo ""
IDX=1
for i in "${!ENV_REC_KEYS[@]}"; do
  # Show current .env value if the file exists
  CURRENT_VAL=""
  if [[ -f "$ENV_FILE" ]]; then
    CURRENT_VAL=$(grep "^${ENV_REC_KEYS[$i]}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 || true)
  fi
  if [[ -n "$CURRENT_VAL" ]] && [[ "$CURRENT_VAL" != "${ENV_REC_VALUES[$i]}" ]]; then
    CURRENT_NOTE="  (currently: ${CURRENT_VAL})"
  else
    CURRENT_NOTE=""
  fi
  printf "  %d) %-20s = %-20s  %s%s\n" \
    "$IDX" "${ENV_REC_KEYS[$i]}" "${ENV_REC_VALUES[$i]}" "${ENV_REC_DESCS[$i]}" "$CURRENT_NOTE"
  IDX=$((IDX + 1))
done

echo ""
printf "Enter numbers to apply (space-separated), 'all', or 0 to skip: "
read -r ENV_SELECTION || true

if [[ -n "$ENV_SELECTION" ]] && [[ "$ENV_SELECTION" != "0" ]]; then
  # Ensure .env exists
  if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_EXAMPLE" ]]; then
      echo ""
      echo "No .env found — copying from .env.example..."
      cp "$ENV_EXAMPLE" "$ENV_FILE"
      echo -e "${GREEN}✓ Created $ENV_FILE from .env.example${NC}"
    else
      echo ""
      echo "No .env or .env.example found — creating empty .env..."
      touch "$ENV_FILE"
    fi
  fi

  APPLY_INDICES=()
  if [[ "$ENV_SELECTION" == "all" ]]; then
    for j in $(seq 1 "${#ENV_REC_KEYS[@]}"); do
      APPLY_INDICES+=("$j")
    done
  else
    read -ra APPLY_INDICES <<<"$ENV_SELECTION"
  fi

  echo ""
  for num in "${APPLY_INDICES[@]}"; do
    if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -ge 1 ]] && [[ "$num" -le "${#ENV_REC_KEYS[@]}" ]]; then
      IDX=$((num - 1))
      set_env_var "${ENV_REC_KEYS[$IDX]}" "${ENV_REC_VALUES[$IDX]}" "$ENV_FILE"
      echo -e "  ${GREEN}✓ Set ${ENV_REC_KEYS[$IDX]}=${ENV_REC_VALUES[$IDX]}${NC}"
    fi
  done
  echo ""
  echo -e "${GREEN}✓ .env updated: $ENV_FILE${NC}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo "System: $UNAME | RAM: ${TOTAL_MEM_GB}GB | CPU: $CPU_CORES cores | GPU: $GPU_TYPE"
echo "Ollama: $([ "$OLLAMA_INSTALLED" = true ] && echo "Installed" || echo "Not installed") | Status: $([ "$OLLAMA_RUNNING" = true ] && echo "Running" || echo "Not running")"
echo "Recommended Ollama model: $PRIMARY_MODEL"
echo "Recommended Whisper model: $WHISPER_RECOMMENDED  (${WHISPER_REASONING})"
echo ""
echo -e "${GREEN}✓ Survey complete!${NC}"
