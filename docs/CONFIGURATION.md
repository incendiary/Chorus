# Chorus — Configuration Reference

This document describes every user-configurable option in Chorus: what each setting
does, how it influences output quality and processing time, and when to change it from
the default.

Options that are available in the UI sidebar are marked **(UI)**. Options that are only
accessible via environment variable are marked **(env only)**.

---

## Quick-start recommendations

| Scenario | Recommended settings |
|---|---|
| First run / CPU-only machine | `WHISPER_MODEL=base`, single consensus model, `sequence` alignment |
| Best accuracy, fast machine | `WHISPER_MODEL=medium`, consensus `small,medium`, `sequence` alignment |
| Meeting with multiple speakers | Add diarisation; use `medium` model |
| Low-RAM machine (< 8 GB) | `WHISPER_MODEL=tiny` or `base`; `TRANSCRIPTION_PARALLELISM=1` |
| Apple Silicon 16 GB, native install | Leave device blank (auto-selects MPS); use `small` or `medium` |
| Apple Silicon 32 GB + Ollama LLM | Use `large` model + large Ollama model — Whisper cache is released before Ollama loads |
| Non-English audio | Set `WHISPER_LANGUAGE` to the BCP-47 code; use `medium` or `large` |

---

## Whisper Models

**`WHISPER_MODEL`** **(UI + env)**
**`CONSENSUS_MODELS`** **(UI + env)**

Chorus uses OpenAI's [Whisper](https://github.com/openai/whisper) for all transcription.
Whisper is available in five sizes. The UI exposes tiny through medium; `large` is
available via the environment variable only.

### Model comparison

| Model | Parameters | Download size | Relative speed | Accuracy tier | Min RAM |
|---|---|---|---|---|---|
| `tiny` | 39 M | ~75 MB | ⚡⚡⚡⚡ Fastest | ⭐ Basic | 1 GB |
| `base` | 74 M | ~145 MB | ⚡⚡⚡ Fast | ⭐⭐ Good | 1 GB |
| `small` | 244 M | ~461 MB | ⚡⚡ Moderate | ⭐⭐⭐ Better | 2 GB |
| `medium` | 769 M | ~1.5 GB | ⚡ Slower | ⭐⭐⭐⭐ Best (UI) | 5 GB |
| `large` | 1 550 M | ~3 GB | 🐢 Slowest | ⭐⭐⭐⭐⭐ Highest | 10 GB |

*Speeds are relative on CPU. GPU acceleration narrows the gap significantly — `medium` on
an NVIDIA RTX card is faster than `base` on CPU.*

### Choosing a model

- **`tiny`** — use only when speed or memory is the hard constraint. Accuracy is noticeably
  lower on anything other than clean, studio-quality audio.
- **`base`** — the default. A strong balance for CPU-only or resource-constrained
  environments. Good enough for clear speech with minimal background noise.
- **`small`** — a meaningful accuracy improvement over `base` at roughly 2× the processing
  time. Recommended when you have a GPU or are willing to wait.
- **`medium`** — the best model available in the UI. Handles accents, overlapping speech,
  and background noise considerably better. Recommended as your primary model if hardware
  allows.
- **`large`** — highest accuracy, but requires 10 GB RAM and is slow on CPU. Set via
  `WHISPER_MODEL=large` in `.env`. Not available in the UI dropdown.

### Consensus models and how pass count compounds

`CONSENSUS_MODELS` allows you to run multiple Whisper models in the same pipeline run.
Chorus always generates four audio variants (original, high-pass, normalised, denoised),
and every selected model transcribes every variant:

| Models selected | Transcription passes | Votes per word |
|---|---|---|
| `base` (single) | 4 | 4 |
| `base, small` | 8 | 8 |
| `base, small, medium` | 12 | 12 |

More passes give more votes per word, which means:

- **Finer confidence discrimination.** With one model, a word either appears in all four
  variants (HIGH) or fewer (MEDIUM/LOW). With three models and twelve passes, the
  percentage scale is more granular — a word that consistently appears across all model
  sizes is very highly confident; one that only one model catches in one variant is flagged
  clearly as LOW.
- **Longer processing time.** Each additional model multiplies the transcription stage by
  4. On CPU, `small,medium` roughly doubles the time of `medium` alone.

**Recommended consensus pairs:**

- Balanced: `base,small` — good accuracy lift for ~2× the time.
- High accuracy: `small,medium` — maximises the quality/time trade-off.
- Research: `base,small,medium` — broadest voting base; expect 3× the time of a single
  medium pass.

---

## Alignment Strategy

**`ALIGNMENT_STRATEGY`** **(UI + env)** — values: `sequence` (default) | `positional`

After transcription, Chorus aligns the word sequences from all passes against each other
to vote on each word position. There are two strategies:

### Sequence alignment (default: `sequence`)

Uses the **Needleman-Wunsch** global sequence alignment algorithm — the same technique
used in bioinformatics to align DNA strands. Applied to words, it finds the optimal
global alignment of two word sequences by explicitly accounting for insertions and
deletions.

**When to use it:**
- Whenever transcripts may differ in word count between variants (almost always).
- Audio with hesitations, filler words, or variable-speed sections where one variant
  catches a word another misses.
- Any serious transcription work.

**Trade-off:** Slightly slower than positional due to the dynamic programming pass, but
the accuracy improvement is substantial on real-world audio.

### Positional alignment (`positional`)

Aligns words by position: word 1 from variant A versus word 1 from variant B, and so on.
Fast and simple, but assumes all variants produce transcripts of similar length. If one
variant drops or inserts a word, every subsequent word in the sequence is misaligned.

**When to use it:**
- Clean, studio-quality audio where all variants are expected to produce nearly identical
  transcripts.
- Speed is critical and the audio is simple enough that misalignment is unlikely.

---

## Noise Floor Mode

**`NOISE_FLOOR_MODE`** **(UI + env)** — values: `vad` (default) | `fixed`

The denoised audio variant uses spectral subtraction, which requires an estimate of the
background noise floor. This setting controls how that estimate is obtained.

### VAD mode (default: `vad`)

**Voice Activity Detection.** Chorus analyses the audio's energy envelope to
automatically locate silent regions, then measures the noise floor from those silences.

**When to use it:** Most recordings. Particularly effective when background noise is
consistent (fan hum, air conditioning, street noise) but the level may not be known in
advance.

### Fixed mode (`fixed`)

Uses the first 0.5 seconds of the recording as the noise floor reference.

**When to use it:** Recordings that begin with a known-silent segment (e.g. a deliberate
pre-roll of room tone). Faster than VAD; no energy analysis required. Unreliable if
speech or music starts immediately.

---

## Compute Device

**`WHISPER_DEVICE`** **(UI + env)** — values: `auto` | `cpu` | `cuda` | `mps`

Controls which hardware PyTorch uses for Whisper inference. The UI sidebar exposes a
**Compute device** selectbox; the `auto` option (default) probes the system and selects
the best available device automatically.

### Auto-detection order

Chorus probes hardware in this order and selects the first available option:

1. **CUDA** — NVIDIA GPU. Fastest option; requires the NVIDIA Container Toolkit on Linux
   or Docker Desktop + WSL2 on Windows.
2. **MPS** — Apple Silicon GPU (Metal Performance Shaders). Available on M-series Macs
   running Chorus natively (not in Docker). Gives roughly 3–5× the speed of CPU for
   `base` and `small` models.
3. **CPU** — Always available. Slowest, but works everywhere.

### Apple Silicon (MPS) caveat

Whisper's word-level timestamp alignment requires 64-bit floating point operations,
which MPS does not support. Chorus handles this automatically: it catches the error,
reloads the affected model pass on CPU, and continues. All other passes remain on MPS.
A warning is logged when this fallback occurs — it is expected behaviour, not a failure.

> **Memory note:** The `large` model (~3 GB) may exhaust unified memory on 8 GB
> M-series configurations. Use `small` or `medium` on those machines.

### Manual override

Select from the **Compute device** dropdown in the sidebar, or set in `.env` for
headless/Docker deployments:

```bash
WHISPER_DEVICE=cpu    # Force CPU regardless of GPU availability
WHISPER_DEVICE=cuda   # Force NVIDIA GPU
WHISPER_DEVICE=mps    # Force Apple MPS (native macOS only)
```

---

## Transcription Parallelism

**`TRANSCRIPTION_PARALLELISM`** **(UI + env)** — values: `auto` (default) | integer (e.g. `1`, `2`, `4`)

Controls how many audio variant passes run concurrently. Chorus uses a thread pool to
run multiple Whisper transcription passes in parallel, assigning each worker to a
device (or a specific CUDA device in multi-GPU setups).

The UI sidebar exposes an **Auto parallelism** toggle. Unchecking it reveals a
**Worker count** input (1–16) for explicit control.

### Auto mode (default)

Chorus chooses a sensible worker count based on the detected device and available
capacity. On CPU it typically defaults to a small pool to avoid memory pressure; on GPU
it exploits parallelism more aggressively.

### Manual override

Use the Worker count input in the sidebar, or set in `.env` for headless deployments:

```bash
TRANSCRIPTION_PARALLELISM=1   # Strictly sequential; minimum memory footprint
TRANSCRIPTION_PARALLELISM=4   # Run 4 passes in parallel
```

**When to pin to `1`:** Low-RAM machines (< 8 GB) where loading multiple Whisper model
instances simultaneously would cause swapping. Sequential processing is slower but
stable.

**Multi-GPU:** When multiple CUDA devices are present, Chorus assigns workers in
round-robin across all detected GPUs, maximising utilisation automatically.

---

## Language

**`WHISPER_LANGUAGE`** **(UI + env)** — BCP-47 language code (e.g. `en`, `fr`, `de`) or blank

Whisper can transcribe in dozens of languages and auto-detects the language from the
first 30 seconds of audio by default.

### When to leave it blank (auto-detect)

For single-language recordings where the language is clear from context. Auto-detection
works well for common languages with a clear speaker.

### When to set it explicitly

- **Mixed-language audio** — auto-detect may lock onto the wrong language.
- **Accuracy improvement** — specifying the language removes Whisper's language-detection
  overhead and can improve word-error rate slightly, particularly for `tiny` and `base`
  where the detection competes with limited capacity.
- **Non-Latin scripts** — Arabic, Chinese, Japanese, and similar scripts benefit from an
  explicit hint to avoid transliteration.
- **Forced English output** — set `WHISPER_LANGUAGE=en` to force transcription in English
  even if the audio is in another language (translation mode; transcription quality will
  vary).

---

## NLP Reconstruction

**`enable_nlp`** **(UI checkbox)**

When enabled, Chorus uses [spaCy](https://spacy.io/) (`en_core_web_md`) to
grammatically reconstruct LOW-confidence tokens after the consensus vote. The
reconstructor analyses the surrounding HIGH-confidence words as context and substitutes
the most grammatically plausible candidate for each LOW-confidence word.

### What it does

- Parses the consensus token sequence using spaCy's dependency parser and POS tagger.
- For each LOW-confidence word, scores candidate alternatives against the grammatical
  role the word plays in the sentence.
- Replaces the LOW token with the highest-scoring candidate.

### When it helps

- Grammatically predictable speech (formal presentations, read text, interviews).
- LOW tokens that are incorrect forms of a clearly implied word (e.g. `runned` → `ran`).

### When it is less useful

- Highly technical language, proper nouns, or domain jargon outside spaCy's vocabulary.
- Casual conversational speech with non-standard grammar.

### Setup requirement

The `en_core_web_md` model must be downloaded before use:

```bash
python -m spacy download en_core_web_md
```

In Docker this is handled automatically. For native installs, run this once after
`pip install -r requirements.txt`. If the model is missing, Chorus falls back safely
without crashing and logs a warning.

---

## LLM Reconstruction (Ollama)

**`enable_llm`** **(UI checkbox)**

When enabled, Chorus sends LOW-confidence tokens to a locally-running
[Ollama](https://ollama.com) language model for contextual reconstruction. Unlike NLP
reconstruction (which uses grammar rules), LLM reconstruction uses semantic
understanding of the surrounding passage to propose the most likely word.

### How it works

Chorus passes each LOW-confidence word and its surrounding context to Ollama with a
prompt that includes the confidence percentage. The model returns its best guess, which
replaces the LOW token in the final output.

### When LLM outperforms NLP

- Technical jargon, product names, or domain-specific vocabulary that spaCy does not
  know.
- Contextually ambiguous tokens where grammatical form alone is insufficient (e.g. a
  homophone that could be either of two valid words).

### Choosing a model

LLM reconstruction is a narrow, closed-set task: given roughly eight words of
surrounding context and a short list of two to four ASR candidate words, the
model must output exactly one candidate token and nothing else. This is not a
reasoning or open-ended generation task, so the usual "bigger model is smarter"
intuition does not straightforwardly apply, and getting it wrong is a real
failure mode — a chatty model that adds a preamble breaks the parsing logic.

Two things matter more than raw parameter count for this specific task:

- **Strict instruction-following.** The model must reliably emit only the
  requested token, with no explanation or commentary, across dozens of calls
  per transcript. The Qwen2.5 family measurably outperforms comparable
  Llama/Gemma/Mistral models on this (IFEval negative-constraint compliance),
  which is why it is the recommended default.
- **Vocabulary breadth for rare/technical words.** For ordinary spoken
  language, model size makes little practical difference — this saturates
  early. It does keep improving with scale specifically for technical,
  medical, legal, or otherwise rare vocabulary, since smaller models prune
  long-tail vocabulary during training and quantisation. `qwen2.5:14b` is
  offered as an explicit *option* for jargon-heavy transcripts, not a
  general "better" default — for typical audio it will not outperform the
  3B default, only run slower.

`devops-practices/survey-ollama-env.sh` recommends `qwen2.5:3b` as the default
and offers `qwen2.5:14b` when there's enough RAM headroom, following this
reasoning.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address. Use `http://ollama:11434` with `docker-compose.ollama.yml`. |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to use for reconstruction. See "Choosing a model" above. |
| `OLLAMA_TIMEOUT_SECONDS` | `20` | Seconds to wait for a response before skipping the reconstruction and retaining the original LOW token. |

### Behaviour on failure

If Ollama is unreachable, times out, or returns a malformed response, Chorus retains
the original LOW-confidence token unchanged and continues. The pipeline never fails due
to LLM reconstruction errors.

### Memory sequencing — Whisper and Ollama

Whisper and Ollama are strictly sequential: Whisper finishes all transcription passes
before Ollama is invoked. However, Whisper's model remains cached in the Chorus process
by default, so both models would occupy memory simultaneously during reconstruction.

When LLM reconstruction is enabled, Chorus explicitly clears the Whisper model cache
immediately after transcription completes. This means peak memory is whichever model
is larger, not the sum.

**Practical example — Apple Silicon 32 GB:**

| Configuration | Without cache release | With cache release |
|---|---|---|
| Whisper `large` + `neural-chat:13b` | ~31 GB peak | ~28 GB peak |
| Whisper `medium` + `llama3.1:8b` | ~6.5 GB peak | ~5 GB peak |

This makes large-model combinations viable on 32 GB unified memory machines. Whisper
reloads from disk on subsequent pipeline runs within the same session (the model weights
remain on disk; only the in-memory copy is freed).

---

## Hardware Survey

The **🔍 Detect recommended settings** button in the UI sidebar surveys your hardware
in-process (RAM, CPU cores, GPU via PyTorch) and pre-populates the model, device, and
parallelism controls with the best fit for your machine. Recommendations use the same
thresholds as the `devops-practices/survey-ollama-env.sh` script.

The same logic is available from the command line:

```bash
bash devops-practices/survey-ollama-env.sh
```

---

## Speaker Diarisation

**`enable_diarisation`** **(UI checkbox)**

When enabled, Chorus runs [pyannote.audio](https://github.com/pyannote/pyannote-audio)
to identify and label distinct speakers in the audio, then fuses speaker segments with
Whisper's word-level timestamps to produce a diarised transcript.

### What it produces

A `{stem}_diarised.md` file in which each line is attributed to a speaker label
(e.g. `SPEAKER_00`, `SPEAKER_01`). Speaker names are editable in the UI and persist to
a sidecar JSON file for future sessions.

### Requirements

Diarisation requires a HuggingFace access token with the pyannote model licence
accepted:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Accept the terms for `pyannote/speaker-diarization-3.1`.
3. Set `HUGGINGFACE_TOKEN=<your-token>` in `.env`.

### Performance

Diarisation adds a significant processing step — expect roughly 0.5–1× the audio
duration on CPU, less on GPU. It is recommended only when multiple speakers are
present and attribution matters.

---

## Export Formats

**PDF, DOCX, SRT, VTT** **(UI checkboxes)**

All export formats are generated from the same consensus transcript. Choose based on
how the output will be used:

| Format | Best used for |
|---|---|
| **Consensus (Markdown)** | Always generated. Source of truth; editable in any text editor. |
| **PDF** | Sharing with stakeholders who need a read-only, printable document. |
| **DOCX** | Downstream editing in Microsoft Word or Google Docs. |
| **SRT** | Subtitles for video editors (Premiere Pro, DaVinci Resolve, etc.). Word-level timestamps. |
| **VTT** | Web video subtitles (`<track>` element in HTML5). Word-level timestamps. |
| **Most Likely (Plain Text)** | Always generated. Clean text with no markup; suitable for copying or further NLP processing. |
| **AI Context Pack** | Always generated. Structured Markdown for supplying Chorus output to a language model. |
| **JSON Bundle** | Always generated. Machine-readable; contains all variant transcripts, vote sequences, and statistics. |

---

## Environment Variable Summary

A complete reference of all environment variables recognised by Chorus.

| Variable | Default | UI | Description |
|---|---|---|---|
| `WHISPER_MODEL` | `base` | ✓ | Primary Whisper model: `tiny` / `base` / `small` / `medium` / `large` |
| `CONSENSUS_MODELS` | *(same as `WHISPER_MODEL`)* | ✓ | Comma-separated list of models for multi-model consensus passes |
| `WHISPER_DEVICE` | *(auto)* | ✓ | Compute device: `auto` / `cpu` / `cuda` / `mps` |
| `WHISPER_LANGUAGE` | *(auto-detect)* | ✓ | BCP-47 language code (e.g. `en`, `fr`) |
| `TRANSCRIPTION_PARALLELISM` | `auto` | ✓ | Worker pool size: `auto` or an integer |
| `ALIGNMENT_STRATEGY` | `sequence` | ✓ | Consensus alignment: `sequence` (Needleman-Wunsch) or `positional` |
| `NOISE_FLOOR_MODE` | `vad` | ✓ | Noise floor detection: `vad` (auto) or `fixed` (first 0.5 s) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | — | Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | ✓ | Ollama model for LLM reconstruction |
| `OLLAMA_TIMEOUT_SECONDS` | `20` | — | Seconds before LLM reconstruction times out per token |
| `HUGGINGFACE_TOKEN` | *(none)* | — | Required for speaker diarisation |
