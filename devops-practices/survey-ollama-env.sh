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
# Based on findings, it recommends appropriate Ollama models and configuration.
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
  # Check for Apple Silicon
  if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Apple M"; then
    GPU_TYPE="Apple Silicon (MPS)"
    TOTAL_MEM_GB_UNIFIED=$TOTAL_MEM_GB
    echo "✓ Apple Silicon detected (MPS support)"
    echo "  Unified memory: ${TOTAL_MEM_GB_UNIFIED}GB (shared with CPU)"
  fi
elif [[ "$UNAME" == "Linux" ]]; then
  # Check for NVIDIA GPU
  if command -v nvidia-smi &> /dev/null; then
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

if command -v ollama &> /dev/null; then
  OLLAMA_INSTALLED=true
  echo "✓ Ollama is installed"
  OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
  echo "  Version: $OLLAMA_VERSION"

  if curl -s http://localhost:11434/api/tags &> /dev/null; then
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

# Decision logic based on available resources
if [[ "$TOTAL_MEM_GB" -lt 4 ]]; then
  RECOMMENDED_MODELS=("tiny-llama:latest" "neural-chat:7b-v3.1-q4_0")
  REASONING="Low RAM (< 4GB): Recommend small, quantized models"
elif [[ "$TOTAL_MEM_GB" -lt 8 ]]; then
  RECOMMENDED_MODELS=("mistral:latest" "neural-chat:7b-v3.1-q4_0" "llama2:7b-q4_K_M")
  REASONING="Medium RAM (4-8GB): Balanced models with quantization"
elif [[ "$TOTAL_MEM_GB" -lt 16 ]]; then
  RECOMMENDED_MODELS=("mistral:latest" "llama2:7b" "neural-chat:7b-v3.1" "dolphin-mixtral:latest")
  REASONING="Good RAM (8-16GB): Full-precision 7B models"
else
  RECOMMENDED_MODELS=("mistral:latest" "llama2:13b" "neural-chat:7b-v3.1" "dolphin-mixtral:latest" "llama2:70b-q3_K_M")
  REASONING="Excellent RAM (16GB+): Large models and quantized versions"
fi

# GPU adjustments
if [[ "$GPU_TYPE" != "none" ]]; then
  if [[ "$GPU_VRAM_GB" -ge 8 ]] || [[ "$GPU_TYPE" == "Apple Silicon (MPS)" ]]; then
    RECOMMENDED_MODELS+=("neural-chat:13b")
    REASONING+=" | GPU detected: Can use larger models"
  fi
fi

# ── Whisper model recommendation ─────────────────────────────────────────────
# Derive a recommended WHISPER_MODEL based on available RAM and GPU.
# Thresholds mirror the guidance in docs/CONFIGURATION.md.

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
  # Unified memory is shared; leave headroom for the OS and audio pipeline.
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
  # CPU-only path
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

echo "Reasoning: $REASONING"
echo -e "\n${GREEN}Recommended models for your system:${NC}"

# Define model info (bash 3.2 compatible - no associative arrays)
get_model_info() {
  case "$1" in
    "tiny-llama:latest") echo "TinyLlama 1.1B — Ultra-fast, low memory (~2GB)" ;;
    "neural-chat:7b-v3.1-q4_0") echo "Neural Chat 7B (4-bit) — Fast, high quality (~4GB)" ;;
    "mistral:latest") echo "Mistral 7B — Best speed/quality balance (~5GB)" ;;
    "llama2:7b-q4_K_M") echo "Llama2 7B (quantized) — Excellent reasoning (~4GB)" ;;
    "llama2:7b") echo "Llama2 7B (full) — High quality (~15GB)" ;;
    "neural-chat:7b-v3.1") echo "Neural Chat 7B — Quality conversation (~15GB)" ;;
    "dolphin-mixtral:latest") echo "Dolphin Mixtral 8x7B — Powerful MoE (~20GB)" ;;
    "neural-chat:13b") echo "Neural Chat 13B — High accuracy (~28GB)" ;;
    "llama2:13b") echo "Llama2 13B — Strong reasoning (~28GB)" ;;
    "llama2:70b-q3_K_M") echo "Llama2 70B (3-bit) — Expert level (~24GB)" ;;
    *) echo "Unknown model" ;;
  esac
}

for model in "${RECOMMENDED_MODELS[@]}"; do
  VRAM_EST=$(get_model_info "$model")
  echo "  • $model"
  echo "    $VRAM_EST"
done

echo ""
echo -e "${GREEN}Recommended Whisper model for transcription:${NC}"
echo "  WHISPER_MODEL=${WHISPER_RECOMMENDED}"
echo "  Reason: $WHISPER_REASONING"
echo ""
echo "  Set this in your .env file, or export it before starting Chorus:"
echo "    export WHISPER_MODEL='${WHISPER_RECOMMENDED}'"
echo "  See docs/CONFIGURATION.md for full model comparison."

# ─────────────────────────────────────────────────────────────────────────────
# Setup Instructions & Interactive Actions
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}[6/6] Setup & Installation${NC}"

PRIMARY_MODEL="${RECOMMENDED_MODELS[0]}"

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
    read -p "Would you like to start Ollama server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Starting Ollama server..."
      nohup ollama serve > /tmp/ollama.log 2>&1 &
      OLLAMA_PID=$!
      echo "Ollama started with PID $OLLAMA_PID"
      echo "Waiting for server to start..."
      sleep 3

      # Verify it started
      if curl -s http://localhost:11434/api/tags &> /dev/null; then
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

# Step 3: Pull Recommended Model
if [[ "$OLLAMA_RUNNING" == true ]]; then
  echo ""
  echo -e "${YELLOW}Checking for recommended model: $PRIMARY_MODEL${NC}"

  INSTALLED_MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')

  if echo "$INSTALLED_MODELS" | grep -q "${PRIMARY_MODEL%:*}"; then
    echo -e "${GREEN}✓ Model is already installed${NC}"
  else
    echo -e "${YELLOW}Model not found locally.${NC}"
    read -p "Would you like to pull $PRIMARY_MODEL now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Pulling $PRIMARY_MODEL (this may take a few minutes)..."
      ollama pull "$PRIMARY_MODEL"

      if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model pulled successfully${NC}"
      else
        echo -e "${RED}✗ Failed to pull model. Try manually: ollama pull $PRIMARY_MODEL${NC}"
      fi
    fi
  fi
fi

# Step 4: Ready for Chorus
echo ""
echo -e "${GREEN}Ready to use Chorus with Ollama!${NC}"
echo ""
echo -e "${YELLOW}Recommended models to pull:${NC}"
for model in "${RECOMMENDED_MODELS[@]:0:3}"; do
  echo "  ollama pull $model"
done
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
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo "System: $UNAME | RAM: ${TOTAL_MEM_GB}GB | CPU: $CPU_CORES cores | GPU: $GPU_TYPE"
echo "Ollama: $([ "$OLLAMA_INSTALLED" = true ] && echo "Installed" || echo "Not installed") | Status: $([ "$OLLAMA_RUNNING" = true ] && echo "Running" || echo "Not running")"
echo "Recommended Ollama model: $PRIMARY_MODEL"
echo "Recommended Whisper model: $WHISPER_RECOMMENDED  (${WHISPER_REASONING})"
echo ""
echo -e "${GREEN}✓ Survey complete!${NC}"
