# Chorus Engine — Scalability Guide

## Batch Processing Tuning

### Sequential Processing (Default)

The batch processor runs files sequentially to minimize memory overhead:

```bash
# Process 10 audio files
python -m batch_processor.batch_runner /audio/*.mp3 --output-dir ./results
```

**Memory profile:**
- Peak RAM: ~1-2GB per file (depends on Whisper model size)
- Total time: linear with file count
- I/O bound: waits for disk writes between files

### Configuration for Scale

**For systems with 4GB RAM:**
```bash
WHISPER_MODEL=tiny python -m batch_processor.batch_runner /audio/*.mp3
```

**For systems with 16GB+ RAM:**
```bash
CONSENSUS_MODELS=base,small,medium python -m batch_processor.batch_runner /audio/*.mp3
```

---

## Model Caching Strategy

### First Use (Download)

When Whisper model is first run:
```
tiny:   ~40MB   → download + cache (2 min)
base:   ~150MB  → download + cache (5 min)
small:  ~460MB  → download + cache (10 min)
medium: ~1.5GB  → download + cache (20 min)
large:  ~3.1GB  → download + cache (40 min)
```

### Persistent Cache (Docker)

Named volume preserves models across container restarts:

```bash
# View cached models
docker volume inspect chorus_whisper_cache

# Clear cache
docker volume rm chorus_whisper_cache
```

### Multiple Models

Using consensus speeds up by loading all models once:

```bash
CONSENSUS_MODELS="base,small" ./batch_processor
# Loads both base and small once, then processes all files
```

---

## Performance Tuning

### Model Selection

| Use Case | Model | Speed | Quality |
|----------|-------|-------|---------|
| Real-time | tiny | ~5x | Low |
| Interactive | base | 1x | Good |
| Batch | small | 0.5x | Better |
| Archive | large | 0.1x | Excellent |

### GPU Acceleration

```bash
docker-compose -f Dockerfile.gpu up
```

Speedup: 10-50× faster depending on GPU model.

### Consensus Strategy

Fewer models = faster consensus:

```bash
CONSENSUS_MODELS="base"           # ~100s per file
CONSENSUS_MODELS="base,small"     # ~200s per file
CONSENSUS_MODELS="base,small,med" # ~400s per file
```

---

## Monitoring

### Docker Stats

```bash
docker stats chorus-engine --no-stream
```

### Batch Report

```bash
cat outputs/consensus/batch_report.md
```

Shows per-file timing and success/failure status.

---

## Future Enhancements

- Parallel worker pool for multi-file processing
- Streaming inference for large audio files
- Model quantization for reduced memory usage
- Distributed processing via task queue
