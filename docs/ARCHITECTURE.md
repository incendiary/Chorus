# Chorus Engine — Architecture Guide

## System Overview

Chorus is a multi-pass consensus audio transcription engine:

1. Cleans audio with 3 filters (high-pass, normalization, denoise)
2. Transcribes each variant with Whisper (multiple models for consensus)
3. Aligns and votes on words across transcripts
4. Reconstructs LOW-confidence tokens via NLP or LLM
5. Exports results (PDF, DOCX, SRT, VTT, JSON)
6. Identifies speakers (optional diarisation)

## Core Modules

### Audio Processor
- High-pass filter (150 Hz): removes rumble, wind noise
- Normalization (-20dBFS): compresses dynamic range
- Spectral denoise: reduces background noise
- Output: 4 WAV variants (original + 3 cleaned)

### Transcription Engine
- Wraps OpenAI Whisper with multi-model support
- Device-aware scheduling (CPU/GPU/MPS with fallbacks)
- Model caching to avoid redundant loading
- Segment-level progress for UI responsiveness

### Consensus Merger
- Word-level alignment (sequence or positional)
- Voting: ≥75% → HIGH, ≥50% → MEDIUM, <50% → LOW
- Optional NLP reconstruction (spaCy)
- Optional LLM reconstruction (Ollama)

### Export Engine
- Markdown with confidence highlighting
- PDF (WeasyPrint), DOCX (python-docx)
- SRT/VTT subtitle formats
- JSON bundle (structured metadata)

### Diarisation
- Speaker identification (pyannote.audio)
- Aligns with Whisper segments via midpoint overlap
- Persists speaker names to JSON

### Batch Processor
- Multi-file processing with unified config
- Per-file output isolation (prevents collisions)
- Batch report summarizing results

### Streamlit UI
- Interactive pipeline control
- Live progress, speaker naming UI
- Browse past jobs, download outputs

## Data Flow

```
Audio File
  ↓ [Audio Processor]
  → 4 audio variants
  
  ↓ [Transcription]
  → 4 transcripts (one per variant)
  
  ↓ [Alignment + Voting]
  → Word-level confidence tiers
  
  ↓ [Optional Reconstruction]
  → Improved consensus
  
  ↓ [Export]
  → PDF, DOCX, SRT, VTT, JSON, Markdown
  
  ↓ [Optional Diarisation]
  → Speaker-labelled transcript
```

## Configuration

**Environment Variables:**
- `WHISPER_MODEL` — Model size (tiny→large)
- `CONSENSUS_MODELS` — Models to vote (comma-separated)
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` — LLM configuration
- `HUGGINGFACE_TOKEN` — Diarisation model access

**Pipeline Flags:**
- `enable_nlp`, `enable_llm` — Reconstruction
- `enable_diarisation` — Speaker identification
- `alignment_strategy` — Voting algorithm
- `output_dir` — Custom output location

## Testing

- Unit tests per module (audio, merger, exporter, etc.)
- Integration tests with mocked Whisper (no model downloads in CI)
- Synthetic audio via sine waves for deterministic testing
- Temporary directories prevent file system pollution

## Performance

| Stage | Bottleneck | Notes |
|-------|-----------|-------|
| Audio Processing | CPU filters | ~30s for 5min audio |
| Transcription | GPU inference | ~90s per variant (base model) |
| Consensus | Word alignment | O(n) sequence, O(n²) positional |
| Export | Disk I/O | Parallel format generation |

GPU acceleration: 10-50× speedup on inference.

## Security

- Local models only (no cloud dependencies)
- Non-root Docker user (process isolation)
- Input validation (file existence, format checks)
- Pre-commit secret scanning (GitLeaks)
- Configuration via environment (no hardcoded secrets)
