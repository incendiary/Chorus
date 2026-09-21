# Chorus Engine — Command-Line Reference

This document covers every command-line flag for Chorus Engine's two entry points.

## Entry Points

Chorus has two CLI tools for different workflows:

1. **`python -m batch_processor.batch_runner`**: process multiple files or
   directories in a single unattended run. Recommended for high-volume work,
   automation, and batch operations. Supports recursive directory scanning,
   concurrent transcription of multiple files, and locking to prevent parallel
   runs against the same output directory.

2. **`python pipeline_runner.py`**: process a single audio file with minimal
   configuration. Useful for quick, standalone transcriptions and integration
   into shell scripts or other tools.

The batch processor is the primary entry point and supports all of Chorus
Engine's features. The pipeline runner is a simpler interface for single-file
operation.

## Settings Precedence

When a setting can come from multiple sources, this order applies:

1. **Command-line flag** (highest priority)
2. **Hardware preset** (if `--hardware-preset max|background` is set)
3. **Process environment variable** (exported in your shell)
4. **`.env` file** (in the project root)
5. **Built-in default** (lowest priority)

Each effective setting displays its source when the run starts. For example:

```
  consensus threshold      0.75                 [default]
  parallelism             2                     [CLI (--parallelism)]
  device                  cuda                  [auto-detected default]
```

## Batch Processor: `python -m batch_processor.batch_runner`

### Input Discovery

#### `inputs` (positional argument, optional with `--check-diarisation`)

File paths, directory paths, or glob patterns to process. Chorus discovers
audio files from:

- Explicit file paths: `file.mp3`
- Directory paths: `/recordings/` (non-recursive by default)
- Glob patterns: `/audio/*.flac` (globbing is expanded by your shell or the
  tool, depending on quoting)

Supported formats: `.wav`, `.mp3`, `.m4a`, `.m4b`, `.mp4`, `.aac`, `.ogg`,
`.oga`, `.opus`, `.flac`, `.webm`, `.wma`, `.amr`, `.3gp`, `.aif`, `.aiff`,
`.caf`, `.mka`, `.mkv`, `.mov`.

Unsupported files in a directory are skipped with a warning. Corrupted or
unreadable files fail per-file with a clear error message.

**Examples:**

```bash
python -m batch_processor.batch_runner /recordings/
python -m batch_processor.batch_runner a.mp3 b.wav c.flac
python -m batch_processor.batch_runner /audio/*.mp3
```

### File Discovery Options

#### `--recursive` / `-r`

Scan input directories recursively, including subdirectories.

**Default:** Off (non-recursive).

**Web UI equivalent:** None.

**Example:**

```bash
python -m batch_processor.batch_runner /recordings/ --recursive
```

#### `--check-diarisation`

Check whether speaker diarisation can run right now without processing any
audio. Loads the full pipeline to verify that the diarisation model and
Hugging Face token are available, then exits with status 0 (ready) or 1 (not
ready).

Takes no inputs. Useful as a pre-flight check before running an unattended
batch with `--diarise`, because starting a real batch run only to discover
diarisation cannot load wastes time. The error message explains what is missing
(token, model access, or environment).

**Default:** Off (no check; batch runs normally).

**Web UI equivalent:** None (the Web UI checks diarisation readiness when you
enable the checkbox).

**Example:**

```bash
python -m batch_processor.batch_runner --check-diarisation
echo "Exit code: $?"  # 0 = ready, 1 = not ready
```

### Model and Device Configuration

#### `--consensus-models [MODEL ...]`

Ordered list of Whisper model names for consensus voting. Space-separated on
the command line. Example: `--consensus-models base small medium`.

Chorus transcribes the audio once with each model and votes word-for-word to
resolve uncertain regions. Using multiple models improves confidence but
increases processing time. The first model listed is treated as primary for
compatibility outputs (e.g. segment-level metadata).

**Default:** The value of `CONSENSUS_MODELS` environment variable (if set), or
`WHISPER_MODEL` if not. Typically `("base",)` (a single model, no consensus
voting).

**Source:** `CONSENSUS_MODELS` environment variable (checked in `.env` or
process environment).

**Web UI equivalent:** "Consensus models" multi-select in the Model & Device
section. The UI always includes the chosen "Model size" as the first option.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ \
  --consensus-models base small medium
```

#### `--whisper-model {tiny,base,small,medium,large}`

Shorthand for `--consensus-models`: use a single Whisper model for this run.

Mutually exclusive with `--consensus-models`. If both are given, the tool
errors.

**Default:** The value of `WHISPER_MODEL` environment variable (if set), or
`"base"`.

**Source:** `WHISPER_MODEL` environment variable (checked in `.env` or process
environment).

**Web UI equivalent:** "Model size" selector in the Model & Device section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --whisper-model large
```

#### `--device {auto,cpu,cuda,mps}`

Compute device for Whisper inference.

- **`auto`**: Probes CUDA (NVIDIA GPU) → MPS (Apple Silicon GPU) → CPU and
  uses the first available. Recommended.
- **`cpu`**: CPU only. Slowest, but works everywhere.
- **`cuda`**: NVIDIA GPU via CUDA. Requires NVIDIA drivers or NVIDIA Container
  Toolkit (Docker).
- **`mps`**: Apple Silicon GPU (macOS only). Falls back to CPU for float64
  operations.

**Default:** Auto-detected (`_detect_device()` in config, which probes torch
availability and picks the best device).

**Source:** `WHISPER_DEVICE` environment variable (checked in `.env` or process
environment). If unset, auto-detected at runtime.

**Web UI equivalent:** "Compute device" selector in the Model & Device section
(also offers "Auto-detect" as a friendly label).

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --device cuda
python -m batch_processor.batch_runner /audio/ --device auto
```

#### `--parallelism {auto|N}`

Number of concurrent transcription worker threads. Accepts `"auto"` (choose
based on device and capacity) or a positive integer.

More workers speed up batch processing of multiple files but increase memory
use and CPU pressure. On low-RAM machines, pin to `1`.

**Default:** `"auto"` (intelligent selection based on detected hardware).

**Source:** `TRANSCRIPTION_PARALLELISM` environment variable (checked in `.env`
or process environment).

**Web UI equivalent:** "Auto parallelism" checkbox and "Worker count" number
input in the Model & Device section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --parallelism auto
python -m batch_processor.batch_runner /audio/ --parallelism 2
```

#### `--hardware-preset {max,background}`

Detect hardware and apply a preset for model size, device, and parallelism.
Runs once, applies for this invocation only. Does not update configuration
files or environment variables.

- **`max`**: Largest viable model, full parallelism, for when the machine is dedicated to
  Chorus.
- **`background`**: One model tier smaller, parallelism pinned to 1, for when the machine
  stays responsive.

Explicit CLI flags (`--whisper-model`, `--device`, `--parallelism`) still take
precedence over the preset. For persistent tuning, use the interactive
`devops-practices/survey-ollama-env.sh` script to update `.env` after hardware
detection.

**Default:** None (presets are only applied if explicitly requested).

**Web UI equivalent:** "Settings preset" dropdown with "Max" / "Background"
options and "Apply" button in the Model & Device section. The Web UI applies
presets only to the current session, not to `.env`.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --hardware-preset max
python -m batch_processor.batch_runner /audio/ --hardware-preset background
```

### Language and Transcription Strategy

#### `--language {code}` / `-l {code}`

BCP-47 language code hint for Whisper (e.g. `en`, `fr`, `de`, `es`). Restricts
Whisper to a known language, which improves accuracy on noisy or short
recordings.

Omit (or set to `auto`) for auto-detection. Whisper's auto-detection is
reliable for clear speech.

**Default:** `None` (auto-detect).

**Source:** `WHISPER_LANGUAGE` environment variable (checked in `.env` or
process environment).

**Web UI equivalent:** "Language" selector in the Language section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --language en
python -m batch_processor.batch_runner /audio/ --language auto
```

#### `--alignment-strategy {sequence,positional}`

Consensus alignment algorithm for comparing variants.

- **`sequence`** (recommended): Needleman-Wunsch sequence alignment. Handles
  word insertions and deletions across variants. More accurate on noisy audio.
- **`positional`** (legacy): Compares word-by-word at each index. Fast but
  sensitive to length differences between variants.

**Default:** `"sequence"`.

**Source:** `ALIGNMENT_STRATEGY` environment variable (checked in `.env` or
process environment).

**Web UI equivalent:** "Alignment algorithm" dropdown in the Processing
Strategy section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --alignment-strategy sequence
```

### Confidence Thresholds

#### `--consensus-threshold {0.0...1.0}`

Minimum fraction of transcription passes that must agree on a word for it to be
marked HIGH confidence (rendered plain, trusted).

With the default 4 transcription passes, a threshold of `0.75` means 3-of-4
agreement is required for HIGH. Lower values trust more words but flag fewer
for review; raise it to be alerted about more uncertain regions.

Accepts floats in `[0.0, 1.0]`. Step size recommended: `0.05`.

**Default:** `0.75`.

**Source:** `config.CONSENSUS_THRESHOLD` (hardcoded in config).

**Web UI equivalent:** "HIGH-confidence threshold" slider in the Processing
Strategy section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --consensus-threshold 0.65
python -m batch_processor.batch_runner /audio/ --consensus-threshold 0.85
```

#### `--similarity-threshold {0.0...1.0}`

Fuzzy-match acceptance threshold for word-to-word comparison. Controls how
similar two spellings must be to count as the same word during voting.

Based on Levenshtein distance. Lower values merge more variant spellings (e.g.
`colour` and `color`); higher values treat near-misses as disagreements.

Accepts floats in `[0.0, 1.0]`. Typical range: `0.50–0.95`. Step size
recommended: `0.05`.

**Default:** `0.80`.

**Source:** `config.SIMILARITY_THRESHOLD` (hardcoded in config).

**Web UI equivalent:** "Word-match similarity" slider in the Processing
Strategy section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --similarity-threshold 0.75
```

### Audio Cleaning

#### `--noise-floor-mode {vad,fixed}`

Noise-floor detection strategy for audio cleaning filters.

- **`vad`** (recommended): Voice Activity Detection, which detects the quietest
  segment via energy analysis. Best when audio starts with speech.
- **`fixed`**: Assumes the first 0.5 seconds is silence (legacy). Use if you
  know your recordings have a silent intro.

**Default:** `"vad"`.

**Source:** `NOISE_FLOOR_MODE` environment variable (checked in `.env` or
process environment).

**Web UI equivalent:** "Noise floor detection" dropdown in the Audio Cleaning
section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --noise-floor-mode vad
```

#### `--word-timestamps` / `--no-word-timestamps`

Enable or disable Whisper word-level timestamps. By default, word timestamps
are off because they can cause alignment collapse on long-form audio (repeated
phrases), but are needed for word-level SRT/VTT subtitles.

Use `--word-timestamps` to enable, `--no-word-timestamps` to explicitly
disable.

**Default:** `False` (disabled).

**Source:** `WORD_TIMESTAMPS` environment variable: set to `1`, `true`, or
`yes` to enable. Checked in `.env` or process environment.

**Web UI equivalent:** None. The sidebar notes "Word-level timestamps are
always enabled for precise subtitles," but this is a documentation
inconsistency, so the CLI flag exists to allow disabling them. For Web UI runs,
word timestamps are always on to support SRT/VTT export.

**Command-line only:** To enable word timestamps, you must use
`--word-timestamps`. The `.env` variable applies only to code paths that read
the config directly (certain pipeline modes); the Web UI doesn't expose this
setting because it always needs word timestamps for subtitles.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --word-timestamps
python -m batch_processor.batch_runner /audio/ --no-word-timestamps
```

#### `--keep-variant-wavs` / `--no-keep-variant-wavs`

Retain or discard intermediate cleaned WAV variants (the four variants:
original, high-pass-filtered, normalised, denoised).

By default, these WAVs are deleted after every stage that reads them because
they are roughly 100 MB per recording and accumulate indefinitely. Set
`--keep-variant-wavs` to retain them for debugging the cleaning filters.

Use `--keep-variant-wavs` to enable, `--no-keep-variant-wavs` to explicitly
disable.

**Default:** `False` (delete after use).

**Source:** `KEEP_VARIANT_WAVS` environment variable: set to `1`, `true`, or
`yes` to keep. Checked in `.env` or process environment.

**Web UI equivalent:** None. The Web UI always deletes variant WAVs after use.

**Command-line only:** To retain variants, set `--keep-variant-wavs` or add
`KEEP_VARIANT_WAVS=1` to `.env` before running batch jobs that need them.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --keep-variant-wavs
python -m batch_processor.batch_runner /audio/ --no-keep-variant-wavs
```

### Optional Features: Token Reconstruction

#### `--nlp`

Enable spaCy NLP reconstruction for LOW-confidence tokens. Uses grammatical and
semantic analysis to infer missing or garbled words.

NLP reconstruction is optional and never fails a run: if the spaCy model is
missing, LOW-confidence tokens are simply left unreconstructed and flagged in
the consensus output.

**Default:** Off (disabled).

**Source:** CLI flag only.

**Web UI equivalent:** "NLP Reconstruction" checkbox in the Advanced Features
section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --nlp
```

#### `--llm`

Enable local LLM reconstruction using Ollama. Reconstructs LOW-confidence
tokens using a local language model.

LLM reconstruction is optional and never fails a run. If Ollama is not
reachable or the model is unavailable, LOW-confidence tokens are flagged but
not reconstructed.

**Important:** `--nlp` and `--llm` are not mutually exclusive. If both are
enabled, spaCy runs first on LOW-confidence tokens, then Ollama runs only on
whatever spaCy left at LOW confidence. This is useful for combining lightweight
NLP with heavier LLM inference for stubborn cases.

**Default:** Off (disabled).

**Source:** CLI flag only.

**Web UI equivalent:** "LLM Reconstruction (Ollama)" checkbox in the Advanced
Features section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --llm
python -m batch_processor.batch_runner /audio/ --nlp --llm  # both enabled
```

#### `--ollama-model {name}`

Ollama model name for LLM reconstruction (e.g. `mistral`, `qwen2.5:3b`,
`neural-chat`). Only used if `--llm` is set.

The Web UI automatically lists locally pulled models and recommends the
best-suited for low-confidence word correction.

**Default:** `"qwen2.5:3b"`.

**Source:** `OLLAMA_MODEL` environment variable (checked in `.env` or process
environment).

**Web UI equivalent:** "Ollama model" dropdown in the Advanced Features section
(only shown if `--llm` is enabled and Ollama is reachable).

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --llm --ollama-model mistral
python -m batch_processor.batch_runner /audio/ \
  --llm --ollama-model qwen2.5:14b
```

#### `--ollama-base-url {URL}`

Ollama API base URL for this run (e.g. `http://localhost:11434`). Only used if
`--llm` is set.

Allows pointing to a remote Ollama instance or a non-standard local port.

**Default:** `"http://localhost:11434"`.

**Source:** `OLLAMA_BASE_URL` environment variable (checked in `.env` or
process environment).

**Web UI equivalent:** None. To change the Ollama URL, set `OLLAMA_BASE_URL` in
`.env` before starting the Web UI.

**Command-line only:** The Web UI does not expose this setting. To use a
non-standard Ollama URL with the Web UI, add `OLLAMA_BASE_URL=...` to `.env`
and restart the UI.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --llm \
  --ollama-base-url http://localhost:11434
python -m batch_processor.batch_runner /audio/ --llm \
  --ollama-base-url http://remote-host:11434
```

#### `--ollama-timeout {SECONDS}`

Ollama request timeout in seconds (floating-point). Only used if `--llm` is
set.

If Ollama takes longer than this to respond, the request times out and
LOW-confidence tokens are left unreconstructed (they are still flagged).

**Default:** `20.0` seconds.

**Source:** `OLLAMA_TIMEOUT_SECONDS` environment variable (checked in `.env` or
process environment). Parsed as a float.

**Web UI equivalent:** None. To change the timeout, set
`OLLAMA_TIMEOUT_SECONDS` in `.env` before starting the Web UI.

**Command-line only:** The Web UI does not expose this setting. To change
timeout for Web UI runs, add `OLLAMA_TIMEOUT_SECONDS=...` to `.env` and
restart the UI.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --llm --ollama-timeout 30.0
python -m batch_processor.batch_runner /audio/ --llm --ollama-timeout 10.0
```

### Optional Features: Speaker Diarisation

#### `--diarise`

Enable speaker diarisation to identify and separate multiple speakers.

Requires a Hugging Face token (read-only) and accepted licences for gated
models. Set `HUGGINGFACE_TOKEN` in `.env` and visit the model repositories to
accept terms before running.

By default, `--diarise` refuses to start if diarisation cannot load (e.g.
missing token or model). Use `--allow-diarisation-stub` to proceed anyway with
a single-speaker placeholder.

**Default:** Off (disabled).

**Source:** CLI flag only.

**Web UI equivalent:** "Speaker Diarisation" checkbox in the Advanced Features
section.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --diarise
```

#### `--allow-diarisation-stub`

Skip the diarisation pre-flight check and proceed even if the diarisation
pipeline cannot load. Falls back to a single-speaker placeholder for every
file.

This is rarely needed. Use it only if you are aware that diarisation is
unavailable but want the batch to complete anyway. Every file will silently get
a fake single-speaker label; the error is logged but not surfaced in the final
report. Without this flag, `--diarise` blocks the batch from starting if
diarisation is not ready.

**Default:** Off (diarisation must be ready, or the batch refuses to start).

**Source:** CLI flag only.

**Web UI equivalent:** None. The Web UI checks diarisation readiness when you
enable the checkbox; if it is not ready, the checkbox unchecks and a setup
dialog appears.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --diarise \
  --allow-diarisation-stub
```

### Export Formats

#### `--export [{pdf,docx,srt,vtt} ...]`

Space-separated list of export formats to generate. Omit or leave empty to skip
export.

- **`pdf`**: PDF document (requires reportlab).
- **`docx`**: Word document (requires python-docx).
- **`srt`**: SRT subtitles with word-level timestamps.
- **`vtt`**: WebVTT subtitles with word-level timestamps.

The consensus Markdown is always generated. These formats are optional
additions.

**Default:** None (no export).

**Source:** CLI flag only.

**Web UI equivalent:** Checkboxes for "PDF Document", "Word Document (.docx)",
and "Subtitles (.srt)" in the Export Formats section. The Web UI does not offer
VTT as a separate option, but it is available via the CLI.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ --export pdf docx srt
python -m batch_processor.batch_runner /audio/ --export pdf
python -m batch_processor.batch_runner /audio/ --export srt vtt
```

### Output Location

#### `--output-dir {DIR}` / `-o {DIR}`

Root output directory. When supplied, each input file's outputs are written to
an isolated `<DIR>/<stem>/` subdirectory to prevent cross-job collisions.

If omitted, outputs are written to the global `outputs/` directory in the
project root, with all files dumped into `outputs/consensus/`,
`outputs/transcripts/`, etc.

Using `--output-dir` is recommended for batch operations to keep runs tidy.

**Default:** `None` (global project outputs).

**Source:** CLI flag only.

**Web UI equivalent:** None. The Web UI always writes to the global `outputs/`
directory.

**Example:**

```bash
python -m batch_processor.batch_runner /audio/ \
  --output-dir /data/chorus-run-2025-09-20
python -m batch_processor.batch_runner /audio/ -o ./batch-results
```

## Pipeline Runner: `python pipeline_runner.py`

The pipeline runner is a simpler interface for processing a single audio file.
It supports far fewer flags than the batch processor.

### Input

#### `audio` (positional argument, required)

Path to the input audio file.

**Supported formats:** Same as batch processor (`.wav`, `.mp3`, `.m4a`, `.flac`,
etc.).

**Example:**

```bash
python pipeline_runner.py /path/to/recording.mp3
```

### Options

#### `--language {code}` / `-l {code}`

BCP-47 language code hint for Whisper (e.g. `en`, `fr`). Omit for
auto-detection.

**Default:** `None` (auto-detect).

**Example:**

```bash
python pipeline_runner.py recording.mp3 --language en
```

#### `--output-dir {DIR}` / `-o {DIR}`

Root directory for pipeline outputs. Creates `variants/`, `transcripts/`, and
`consensus/` subdirectories inside.

If omitted, outputs go to the global `outputs/` directory.

**Default:** `None` (global project outputs).

**Example:**

```bash
python pipeline_runner.py recording.mp3 --output-dir /tmp/chorus-out
```

### Other Features

The pipeline runner does not expose CLI flags for:

- Model selection (uses configured default)
- Device selection (auto-detected)
- Consensus thresholds
- Audio cleaning strategy
- Token reconstruction (NLP/LLM)
- Diarisation
- Export formats

To access these features, use the **batch processor** (`python -m
batch_processor.batch_runner`) or the **Web UI** (`streamlit run ui/app.py`).

## Common Patterns

### Process a single file with all features enabled

```bash
python -m batch_processor.batch_runner recording.mp3 \
  --device cuda \
  --consensus-models base small medium \
  --language en \
  --diarise \
  --nlp \
  --llm --ollama-model mistral \
  --export pdf srt \
  --output-dir ./results
```

### Batch process a directory with conservative settings (background mode)

```bash
python -m batch_processor.batch_runner /recordings/ --recursive \
  --hardware-preset background \
  --export pdf
```

### Quick quality check on a single file before batch processing

```bash
python pipeline_runner.py sample.mp3 --language en
```

### Process audio, keep variant WAVs for debugging

```bash
python -m batch_processor.batch_runner /audio/ \
  --keep-variant-wavs \
  --output-dir /tmp/debug
```

### Verify diarisation is ready before starting a batch

```bash
python -m batch_processor.batch_runner --check-diarisation
if [ $? -eq 0 ]; then
  echo "Diarisation ready, starting batch"
  python -m batch_processor.batch_runner /audio/ --recursive --diarise
else
  echo "Diarisation not ready"
fi
```

### Adjust confidence thresholds per run

```bash
# Strict: flag everything except 4-of-4 agreement
python -m batch_processor.batch_runner /audio/ --consensus-threshold 1.0

# Lenient: trust anything 2-of-4 or better
python -m batch_processor.batch_runner /audio/ --consensus-threshold 0.50
```

## Debugging and Diagnostics

### View active settings and their source

When a batch run starts, it prints the effective configuration with sources:

```
Chorus batch — effective settings
  models                base                    [CLI (--consensus-models)]
  device                cuda                    [auto-detected default]
  parallelism           4                       [CLI (--parallelism)]
  language              en                      [CLI (--language)]
  consensus threshold   0.75                    [default]
  ...
  precedence            CLI > hardware preset > process environment > .env > default
```

### Check log file location

The batch processor writes a per-run log file and prints its path:

```
  log file               outputs/consensus/batch_20250920T143022Z.log
```

Check the log if files fail silently or processing seems stuck.

### Pre-flight check for diarisation

```bash
python -m batch_processor.batch_runner --check-diarisation
```

Returns 0 if ready, 1 if not. Useful for automation.
