# Chorus Engine Holistic Codebase Review

Date: 21 June 2026 · Re-assessed: 29 June 2026 (v3.3.0)

> **29 June 2026 re-assessment.** This review was re-validated against the v3.3.0 tree.
> The headline findings were routed into four v4.0.0 work packages, all now shipped
> in full — see [v3.3.0 re-assessment & v4.0.0 readiness](#v330-re-assessment--v400-readiness)
> below and `ROADMAP.md`'s "Completed — v4.0.0" section for what landed.

## Executive summary

Chorus is in a strong early-production state: the core audio, transcription, consensus, export, and release paths are modular, the test suite is broad for the non-UI core, dependencies are pinned in `requirements.txt`, and CI already runs tests, linting, secret scanning, scheduled drift checks, and a non-blocking dependency audit. The most urgent risk is output isolation drift: several helpers still write to global `outputs/consensus` even when callers supply an isolated `output_dir`, which can silently mix files across CLI, UI, and batch runs. The second-order scaling risk is the Streamlit UI and batch processor having little or no direct coverage despite being the main user-facing orchestration surfaces. Overall health is good, but the project should harden output routing, batch feature parity, export sanitisation, and dependency audit enforcement before scaling to larger unattended workloads.

## Context reviewed

### Entry points

| Entry point | Trigger | Notes |
|---|---|---|
| `pipeline_runner.py` | `python pipeline_runner.py <audio>` or programmatic `run_pipeline()` | Main single-file CLI and API entry point. |
| `batch_processor/batch_runner.py` | `python -m batch_processor.batch_runner ...` | Batch and directory-processing CLI. |
| `ui/app.py` | `streamlit run ui/app.py` or Docker Compose | Main interactive UI. |
| `tests/` | `pytest` | Test execution surface and regression harness. |
| Package `__init__.py` files | Imports | Mostly empty, low risk. |
| `.github/workflows/ci.yml` | Push, PR, and weekly schedule | CI, lint, test, secret scan, dependency audit, and version drift checks. |
| `.github/workflows/release.yml` | `v*` tag push | Release tests, Docker build and publish, GitHub release, and post-release checks. |
| `Dockerfile`, `Dockerfile.gpu`, `docker-compose.yml`, `docker-compose.gpu.yml` | Container runtime | CPU and CUDA deployment surfaces. |
| `docker-publish.sh`, `docker-test.sh`, `devops-practices/*.sh` | Manual operational scripts | Release and consistency helpers. |

### Dependency manifests

| Manifest | Purpose | Finding |
|---|---|---|
| `requirements.txt` | Runtime dependency pins | Direct runtime dependencies are pinned. |
| `pyproject.toml` | Build metadata, package discovery, tool config, dev extras | Runtime `dependencies = []`, so package installs do not receive runtime dependencies unless `requirements.txt` is installed separately. |
| `.pre-commit-config.yaml` | Local quality and secret hooks | Hooks are pinned and include layered secret scanning. |
| `Dockerfile`, `Dockerfile.gpu` | Container dependency installation | Runtime labels are stale (`1.0.0`), and GPU image installs a second `pyannote.audio` version outside `requirements.txt`. |

### CI definition

CI exists in `.github/workflows/ci.yml` and release automation exists in `.github/workflows/release.yml`. The pipeline runs on push, pull request, and a weekly schedule. It runs secret scanning, formatting checks, Ruff, isort, tests, version drift checks, and `pip-audit`, although `pip-audit` is currently non-blocking because the command ends with `|| true`.

### Existing tests and coverage

The repository has 188 tests under `tests/`. I ran:

```bash
.venv/bin/python -m pytest --cov=. --cov-report=term-missing --tb=short
```

Result: 188 passed, 7 warnings, 74% overall coverage. Notable blind spots: `ui/app.py` at 0%, `batch_processor/batch_runner.py` at 0%, `nlp_reconstructor/reconstructor.py` at 33%, `export_engine/exporter.py` at 58%, and `diarisation/diariser.py` at 66%.

### README and docs

`README.md` clearly documents purpose, Docker and native installation, GPU support, output interpretation, architecture, and roadmap governance. `ROADMAP.md` is the dedicated roadmap source of truth and is enforced by tests, so this review adds action items there rather than embedding roadmap checklists in `README.md`.

### Known pain points inferred from project docs

The current documented focus areas are batch consensus clarity, progress and error feedback, UI pattern consistency, and strict release and documentation synchronisation. The historical roadmap shows recent work on output directory isolation, LLM reconstruction hardening, batch processing, Docker environment documentation, and version consistency.

## Architecture

| Module / Path | Responsibility | Concerns |
|---|---|---|
| `config.py` | Central constants, environment parsing, device detection, and output directories | Runtime dependencies are not declared in `pyproject.toml`; environment-derived constants are imported early, which makes runtime UI changes via `os.environ` fragile. |
| `utils.py` | Shared filename stem sanitisation | Narrow, clear responsibility. |
| `audio_processor/` | Load audio, resample to mono 16 kHz, generate original, high-pass, normalised, and denoised WAV variants | Loads full audio into memory, so very large files scale linearly in RAM. |
| `transcription_engine/` | Load Whisper models, cache by model and device, transcribe variants, and write transcript JSON/TXT files | Model cache has no eviction policy; parallel CPU runs can overcommit RAM on large models. |
| `consensus_merger/` | Token alignment, fuzzy voting, sequence alignment, and Markdown rendering | Sequence alignment is banded for long inputs, but multi-model or highly divergent transcripts can still become CPU-heavy. |
| `nlp_reconstructor/` | Optional spaCy-based LOW-token reconstruction | English-only model assumption, low direct coverage, and no end-to-end assertion that reconstruction improves or preserves accuracy. |
| `llm_reconstructor/` | Optional local Ollama reconstruction and model probing | Local HTTP calls are bounded by timeout and covered, but prompt/output contracts remain heuristic. |
| `diarisation/` | Optional pyannote speaker diarisation, speaker labelling, speaker-name persistence, and diarised Markdown output | Output directory is global, and external model loading depends on Hugging Face token and network/model cache state. |
| `export_engine/` | PDF, DOCX, SRT, VTT, plain-text, ZIP, JSON bundle, and AI context exports | Several exports always write to global `CONSENSUS_DIR`; Markdown-to-HTML conversion does not explicitly sanitise untrusted transcript content before PDF generation. |
| `batch_processor/` | Batch discovery, sequential pipeline invocation, optional exports, and batch report generation | Reimplements optional NLP/diarisation/export work outside `run_pipeline()`, ignores some feature flags, and writes reports and exports globally. |
| `ui/` | Streamlit single-page workflow for upload, configuration, processing, preview, downloads, and speaker naming | Largest module, no direct automated coverage, and feature wiring depends on import-time config constants. |
| `.github/workflows/` | CI and release automation | Good baseline; dependency audit is advisory rather than blocking. |
| `Dockerfile*`, `docker-compose*.yml` | CPU/GPU runtime packaging | Docker labels are stale, and GPU dependency versioning diverges from `requirements.txt`. |

### Data flow

External input enters through uploaded browser files in `ui/app.py`, CLI file paths in `pipeline_runner.py`, or file/directory/glob inputs in `batch_processor/batch_runner.py`. The audio processor decodes the source file with librosa/FFmpeg, writes four WAV variants, and passes paths into the transcription orchestrator. Whisper produces one or more transcript dictionaries per model and variant, which are persisted as JSON and TXT companions. The consensus merger extracts transcript text, aligns tokens, computes confidence tiers, optionally applies NLP or LLM reconstruction, and renders a Markdown consensus document. Export helpers then generate AI context, JSON bundles, PDF, DOCX, SRT, VTT, plain text, ZIP archives, diarised Markdown, and speaker-name sidecars. Outputs leave the system as files under `outputs/` or a caller-provided `output_dir`, and as Streamlit download payloads.

## Risk inventory

| # | Category | Finding | Score | File / Location |
|---|---|---|---|---|
| 1 | Reliability | `output_dir` isolation is incomplete: AI context, diarisation, speaker names, batch reports, plain text, SRT, VTT, PDF, DOCX, and ZIP helper sidecars can still write to global `outputs/consensus`. | 4 | `pipeline_runner.py`, `export_engine/ai_context.py`, `export_engine/exporter.py`, `diarisation/diariser.py`, `batch_processor/batch_runner.py` |
| 2 | Reliability | Batch processing re-runs optional NLP and diarisation outside `run_pipeline()` and does not pass all feature flags into the pipeline, so batch behaviour can diverge from CLI and UI behaviour. | 4 | `batch_processor/batch_runner.py` |
| 3 | Maintainability | The main Streamlit app is a 1,400-line script with nested functions and zero direct coverage, making UI workflow regressions likely. | 4 | `ui/app.py` |
| 4 | Scalability | Audio processing loads each whole file into memory and materialises several full-length arrays; there is no duration, file-size, or decoded-sample guard. | 4 | `audio_processor/pipeline.py`, `audio_processor/filters.py`, `ui/app.py` |
| 5 | Dependency | `pyproject.toml` declares no runtime dependencies, so `pip install chorus-engine` or editable package installation without `requirements.txt` produces a broken runtime. | 4 | `pyproject.toml`, `requirements.txt` |
| 6 | Security | Transcript-derived Markdown is converted to HTML for PDF export without an explicit sanitisation policy; malformed or HTML-bearing transcript text may be rendered into PDF output. | 3 | `export_engine/exporter.py`, `consensus_merger/renderer.py`, `export_engine/ai_context.py` |
| 7 | Dependency | Dependency audit exists in CI but is non-blocking (`|| true`), and `pip-audit` is not installed in the local `.venv`, so the current manual audit could not verify CVEs. | 3 | `.github/workflows/ci.yml`, local environment |
| 8 | Reliability | Docker image labels are stale (`1.0.0`, `1.0.0-gpu`) while package and README version are `3.1.1`, which weakens operational traceability. | 3 | `Dockerfile`, `Dockerfile.gpu` |
| 9 | Dependency | The GPU Dockerfile installs `pyannote.audio==3.1.1` after installing `requirements.txt`, which pins `pyannote-audio==4.0.4`; this can downgrade or skew dependency resolution in GPU images. | 3 | `Dockerfile.gpu`, `requirements.txt` |
| 10 | Maintainability | Import-time config constants are mutated indirectly in the UI by setting `os.environ`, but imported modules have already captured many config values. | 3 | `ui/app.py`, `config.py`, `audio_processor/filters.py`, `transcription_engine/orchestrator.py` |
| 11 | Maintainability | Optional NLP reconstruction has low coverage and no quality regression harness; it can upgrade LOW tokens without an objective accuracy gate. | 3 | `nlp_reconstructor/reconstructor.py`, `tests/test_reconstructor.py` |
| 12 | Scalability | Whisper model cache has no memory budget, eviction, or explicit user-facing guard when multi-model consensus selects several large models. | 3 | `transcription_engine/whisper_engine.py`, `transcription_engine/orchestrator.py`, `ui/app.py` |
| 13 | Reliability | LLM and diarisation integrations depend on local/external model availability; they degrade gracefully in several cases, but long-running model operations are not covered by integration tests with realistic latency. | 2 | `llm_reconstructor/`, `diarisation/` |
| 14 | CI/CD | CI does not publish coverage artefacts or enforce a minimum coverage threshold for critical surfaces. | 2 | `.github/workflows/ci.yml`, `pyproject.toml` |
| 15 | Maintainability | Runtime and dev tool versions differ between `requirements.txt`, `pyproject.toml` dev extras, CI install commands, and the local `.venv`. | 2 | `requirements.txt`, `pyproject.toml`, `.github/workflows/ci.yml` |

## Predicted failure scenarios

### PF-1: Cross-run output contamination (Reliability, score 4)

**What happens:** A batch run or CLI run with `--output-dir` returns isolated consensus paths, but AI context packs, diarised transcripts, speaker names, plain-text downloads, subtitles, and batch reports appear in the global `outputs/consensus` directory or overwrite files from another run with the same stem.

**Trigger condition:** Processing two files with the same sanitised stem, processing simultaneous UI sessions, running batch jobs with `--output-dir`, or reprocessing the same uploaded filename.

**Estimated timeline:** This can fail today. The current tests only assert the consensus Markdown parent, not every secondary artefact.

**Minimum fix:** Add `output_dir` or `consensus_dir` parameters to every export, diarisation, speaker-name, AI context, and batch-report writer, and thread the caller-selected directory through the UI, CLI, and batch paths.

**Full fix (roadmap item):** Add a run-scoped output context object and integration tests that assert every artefact for two same-stem runs stays in its own directory.

### PF-2: Batch mode produces different results from single-file mode (Reliability, score 4)

**What happens:** Batch runs ignore or duplicate optional feature behaviour. NLP reconstruction can be applied after `run_pipeline()` has already generated consensus, diarisation can be run a second time, exports can land outside the batch output root, and LLM options are not represented in the batch API.

**Trigger condition:** Users process directories with `--nlp`, `--diarise`, `--export`, or a custom output directory.

**Estimated timeline:** This can fail today for batch users using optional features.

**Minimum fix:** Make `run_batch()` delegate all pipeline options to `run_pipeline()` and perform exports against the returned artefacts with the same output root.

**Full fix (roadmap item):** Define a shared processing-options dataclass used by CLI, batch, and UI, then add parity tests comparing single-file and batch output manifests.

### PF-3: UI regression reaches users without a failing test (Maintainability, score 4)

**What happens:** A change to sidebar option wiring, upload handling, error display, download generation, or speaker-name saving silently breaks the Streamlit workflow while all unit tests still pass.

**Trigger condition:** Any UI refactor, Streamlit upgrade, or feature flag addition.

**Estimated timeline:** Likely within the next few UI changes because `ui/app.py` has 0% direct coverage.

**Minimum fix:** Extract pure helpers for option construction, output manifest rendering, and file-processing orchestration, then unit-test them.

**Full fix (roadmap item):** Add Playwright or Streamlit app tests for upload, single-file processing, multi-file processing, failed-file display, and download buttons using mocked Whisper.

### PF-4: Process killed on long or high-resolution audio (Scalability, score 4)

**What happens:** The process is killed by the OS or Docker memory limit during `librosa.load()`, STFT denoising, or multi-variant array creation. In Docker, this appears as a container restart; in native runs, it can look like an abrupt shell or Streamlit failure.

**Trigger condition:** Long recordings, high sample-rate media, multi-file upload batches, or several large files processed on the 4 GB Docker memory limit.

**Estimated timeline:** Will occur when users move from short test recordings to long meetings, interviews, or lectures.

**Minimum fix:** Add a preflight metadata check using `soundfile.info()` or FFmpeg metadata to reject or warn on files above a configurable duration or decoded-sample limit.

**Full fix (roadmap item):** Move toward chunked audio processing and chunked transcription with progress, merge boundaries, and predictable memory limits.

### PF-5: Installed package cannot run outside the repo bootstrap path (Dependency, score 4)

**What happens:** A user installs the project with `pip install .` or `pip install -e .` and then imports or runs Chorus, but dependencies such as Whisper, librosa, Streamlit, pyannote, or WeasyPrint are missing.

**Trigger condition:** Any package-style installation that does not also install `requirements.txt`.

**Estimated timeline:** Will occur as soon as the project is consumed as a Python package rather than a Docker app or manually bootstrapped repo.

**Minimum fix:** Move runtime dependencies into `pyproject.toml` or generate them from one source of truth.

**Full fix (roadmap item):** Split extras into `ui`, `export`, `diarisation`, `llm`, and `dev`, and update Docker, CI, and README to install via package extras.

### PF-6: Transcript content renders unexpected HTML in PDF exports (Security, score 3)

**What happens:** Transcript text containing Markdown or HTML-like content is passed through Markdown conversion into HTML and then rendered by WeasyPrint. The local PDF renderer is less exposed than a browser, but output can include unintended markup, links, or layout-breaking content.

**Trigger condition:** Audio contains dictated HTML/Markdown, or imported transcript text includes angle brackets and Markdown table syntax.

**Estimated timeline:** Possible today when exporting adversarial or unusual transcripts to PDF.

**Minimum fix:** Escape transcript-derived tokens before Markdown rendering or sanitise generated HTML with an allowlist before PDF conversion.

**Full fix (roadmap item):** Add export sanitisation tests covering HTML tags, scripts as literal text, Markdown tables inside transcript content, and low-confidence annotations.

### PF-7: Vulnerable dependency ships because audit is advisory (Dependency, score 3)

**What happens:** CI reports a pip-audit issue but still passes, allowing a dependency with a known vulnerability to remain in main and release images.

**Trigger condition:** A new CVE appears in any pinned direct or transitive dependency.

**Estimated timeline:** Could happen on the next dependency disclosure. The weekly schedule helps visibility, but it does not block merges.

**Minimum fix:** Remove `|| true` from the audit step after documenting any intentional ignores with expiry dates.

**Full fix (roadmap item):** Add a dependency review workflow, publish audit artefacts, and create scheduled issues for drift or CVE findings.

### PF-8: GPU image dependency skew causes diarisation/runtime failures (Dependency, score 3)

**What happens:** GPU image builds with a different pyannote stack than CPU/local installs, causing diarisation import errors, model-loading errors, or subtly different speaker segmentation behaviour.

**Trigger condition:** Building `Dockerfile.gpu`, especially after dependency resolver changes or pyannote transitive updates.

**Estimated timeline:** Possible on the next GPU build.

**Minimum fix:** Remove the extra `pip install pyannote.audio==3.1.1` or align it with the pinned `pyannote-audio==4.0.4` requirement.

**Full fix (roadmap item):** Add a GPU Docker smoke test that imports Whisper, torch, pyannote, and runs a mocked diarisation path.

### PF-9: UI configuration changes do not reach processing code (Maintainability, score 3)

**What happens:** The UI updates `os.environ` for model and noise settings after modules have imported constants, but filter and orchestrator modules may continue using values captured at import time.

**Trigger condition:** User changes model, consensus model, or noise-floor mode in the sidebar after app start.

**Estimated timeline:** This can happen today for options backed by imported constants rather than explicit function parameters.

**Minimum fix:** Pass user-selected settings explicitly through `run_pipeline()` and into processing functions instead of mutating environment variables at run time.

**Full fix (roadmap item):** Introduce a typed runtime configuration object and remove user-option writes to `os.environ` from the Streamlit app.

### PF-10: Multi-model consensus exhausts memory (Scalability, score 3)

**What happens:** Selecting multiple large Whisper models loads several model/device cache entries and keeps them resident, leading to memory pressure or CPU/GPU fallback.

**Trigger condition:** Multi-model consensus using `medium` or `large`, parallel workers, or repeated runs in a long-lived Streamlit process.

**Estimated timeline:** Likely when users experiment with larger models after initial success with `base` or `small`.

**Minimum fix:** Add a model memory preflight warning and expose an unload/clear-cache operation after each run or batch.

**Full fix (roadmap item):** Implement a bounded model cache with explicit eviction and per-device memory policy.

### PF-11: NLP reconstruction reduces transcript accuracy (Maintainability, score 3)

**What happens:** LOW tokens are upgraded to MEDIUM based on weak semantic/POS scoring, and downstream users treat them as more reliable than they are.

**Trigger condition:** Domain-specific vocabulary, non-English audio, names, acronyms, or missing spaCy vectors.

**Estimated timeline:** Possible whenever `enable_nlp=True` is used on real-world transcripts.

**Minimum fix:** Keep reconstructed tokens visibly annotated and add tests for names, acronyms, non-English text, and no-vector candidates.

**Full fix (roadmap item):** Build a small golden transcript evaluation set and require reconstruction to improve or preserve word error rate before default use.

### PF-12: Docker/runtime version traceability is misleading (Reliability, score 3)

**What happens:** Inspecting image labels reports version `1.0.0` or `1.0.0-gpu` while README, tags, and package metadata report `3.1.1`, complicating incident response and support.

**Trigger condition:** Any deployed Docker image inspected by operators or release tooling.

**Estimated timeline:** This is already present.

**Minimum fix:** Source Docker labels from build arguments populated by CI release metadata.

**Full fix (roadmap item):** Add a CI check that Dockerfile labels, `VERSION`, `pyproject.toml`, README examples, and image tags agree.

## Test coverage gaps

| Path | Why critical | Test type needed |
|---|---|---|
| `ui/app.py` upload-to-download workflow | Main user-facing workflow and largest module; currently 0% coverage. | UI/integration: mocked upload, single run, batch run, failed file, and download checks. |
| `batch_processor/batch_runner.py:run_batch()` | Main unattended processing path; currently 0% coverage. | Unit/integration: discovery, output isolation, feature flags, export paths, failure report, and exit status. |
| `pipeline_runner.py:run_pipeline(output_dir=...)` secondary artefacts | Existing tests only assert consensus path parent, not AI context, diarisation, subtitles, plain text, ZIP sidecars, or speaker names. | Integration: same-stem two-run manifest isolation. |
| `export_engine/exporter.py:export_pdf()` and `export_docx()` | User-facing exports and sanitisation-sensitive path; many branches uncovered. | Unit: generated files, escaped transcript text, malformed Markdown, missing optional dependencies, and output directory override. |
| `diarisation/diariser.py:diarise()` real pipeline interaction | External model/token/GPU behaviour is high variance. | Integration with mocked pyannote pipeline plus stub fallback tests for missing token, model load failure, and empty speaker turns. |
| `nlp_reconstructor/reconstructor.py:reconstruct_low_tokens()` | Can change confidence tiers and transcript text. | Golden-data unit tests: names, acronyms, no-vector words, non-English text, and threshold boundaries. |
| `transcription_engine/whisper_engine.py` model cache lifecycle | Long-lived UI process can retain large models. | Unit: unload after multi-model runs, bounded cache policy once implemented. |
| Docker images | Deployment path can differ from native tests. | CI smoke: build or import-check CPU image and, where runner support exists, GPU image. |

Three highest-value additions:

1. Add batch processor tests for `run_batch()` option forwarding, output isolation, export paths, and failure reporting.
2. Add a same-stem, custom-output-dir integration test that asserts every returned and side-effect artefact stays under the selected run directory.
3. Add Streamlit workflow tests with mocked Whisper for upload, run, failure, and downloads.

## Dependency audit

| Dependency / Source | Pinning | Last-release status | CVE status | Notes |
|---|---|---|---|---|
| `requirements.txt` direct dependencies | Exact pins | Not verified live in this session | Not verified live in this session | Pins are good for reproducibility, but there is no generated lock file for transitives. |
| `openai-whisper==20250625` | Exact | Appears current by version date | Not verified | Central runtime dependency; model downloads and FFmpeg behaviour should be smoke-tested. |
| `librosa==0.11.0`, `soundfile==0.14.0`, `scipy==1.17.1`, `numpy==2.4.6` | Exact | Not verified live | Not verified | Heavy numerical stack; audio memory and Python-version compatibility matter. |
| `pydub==0.25.1` | Exact | Potentially mature/slow-moving | Not verified | Small direct dependency; current code primarily uses librosa/soundfile, so reassess whether it is still needed. |
| `nltk==3.9.4` | Exact | Not verified live | Not verified | Used for edit distance/token support; ensure NLTK data availability remains tested in Docker. |
| `streamlit==1.58.0` | Exact | Not verified live | Not verified | Main UI framework; high regression impact because UI has no direct tests. |
| `pyannote-audio==4.0.4` | Exact | Not verified live | Not verified | GPU Dockerfile separately installs `pyannote.audio==3.1.1`, creating skew. |
| `spacy==3.8.13` | Exact | Not verified live | Not verified | Runtime dependency is installed, but language models are external and optional. |
| `weasyprint==69.0`, `python-docx==1.2.0`, `Markdown==3.10.2` | Exact | Not verified live | Not verified | Export stack requires sanitisation tests and system-library awareness. |
| `tqdm==4.68.2`, `watchdog==6.0.0`, `psutil==7.2.2` | Exact | Not verified live | Not verified | Utility dependencies; reassess ongoing need as features stabilise. |
| `pyproject.toml` dev extras | Floating | Not verified live | Not verified | `black`, `ruff`, `isort`, `pytest`, and other dev tools float in extras while CI pins some versions separately. |

Live CVE audit status: `pip-audit` is not installed in the repository `.venv` (`Package(s) not found: pip-audit`). CI installs and runs `pip-audit`, but the job currently allows failure with `|| true`, so it is advisory.

Potential inline candidates: `pydub` should be reviewed because the observed processing path uses `librosa` and `soundfile`; keep it only if other supported formats or planned features need it. No other obvious single-purpose dependency is safe to inline because audio, ML, PDF, DOCX, and UI libraries carry substantial domain logic.

## CI/CD gaps

| Gap | Current state | Fix snippet |
|---|---|---|
| Dependency audit does not block | `pip-audit ... || true` means known CVEs can pass CI. | See snippet 1. |
| Coverage is measured manually, not enforced in CI | Tests run without coverage reporting or threshold. | See snippet 2. |
| Runtime dependency packaging is not validated | CI installs editable package plus hand-picked runtime dependencies, not the package as users would consume it. | See snippet 3. |
| Docker CPU image is not smoke-tested on PRs | Release builds images, but PR CI does not verify Docker runtime import/startup. | See snippet 4. |
| Docker version labels are not checked | Version-sync tests cover README, tags, and roadmap, but not Docker labels. | See snippet 5. |

Snippet 1: make dependency audit blocking after intentional ignores are documented.

```yaml
  dependency-audit:
    name: Dependency Audit
    runs-on: ubuntu-latest
    needs: secret-scan
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: pip-audit
        run: |
          python -m pip install --upgrade pip pip-audit
          pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969
```

Snippet 2: enforce coverage while allowing a ratchet period.

```yaml
      - name: Run tests with coverage
        run: |
          pytest --cov=. --cov-report=term-missing --cov-report=xml --cov-fail-under=75
```

Snippet 3: validate package install metadata.

```yaml
  package-install:
    name: Package Install Smoke
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install .
          python -c "import pipeline_runner, audio_processor, consensus_merger"
```

Snippet 4: add CPU Docker smoke test.

```yaml
  docker-smoke:
    name: Docker Smoke
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build CPU image
        run: docker build --target runtime -t chorus-smoke:ci .
      - name: Import check
        run: docker run --rm chorus-smoke:ci python -c "import streamlit, whisper, pipeline_runner"
```

Snippet 5: check Docker labels against `VERSION`.

```yaml
      - name: Docker label version check
        run: |
          VERSION=$(cat VERSION)
          grep -q "LABEL version=\"${VERSION}\"" Dockerfile
          grep -q "LABEL version=\"${VERSION}-gpu\"" Dockerfile.gpu
```

## Action roadmap

### RA-1: Thread output directory through every artefact writer

**Context:** `run_pipeline(output_dir=...)` isolates variants, transcripts, consensus Markdown, and JSON bundles, but AI context, diarisation, speaker names, plain text, subtitles, PDF, DOCX, ZIP sidecars, and batch reports still default to global `outputs/consensus`.

**Success criteria:**
- A same-stem two-run integration test proves every generated artefact stays under its selected output root.
- No export, diarisation, speaker-name, AI context, or batch-report helper writes to `CONSENSUS_DIR` when a caller-provided output directory is available.
- UI, CLI, and batch paths continue to return valid download paths.

**Files to change:** `pipeline_runner.py`, `export_engine/ai_context.py`, `export_engine/exporter.py`, `diarisation/diariser.py`, `batch_processor/batch_runner.py`, `ui/app.py`, `tests/test_integration.py`, `tests/test_exporter.py`, `tests/test_speaker_names.py`

**Estimated effort:** M

### RA-2: Unify batch option handling with the main pipeline

**Context:** `batch_processor/batch_runner.py` re-applies NLP and diarisation after `run_pipeline()` and does not represent all single-file options, which makes batch output diverge from CLI/UI output.

**Success criteria:**
- `run_batch()` passes supported feature flags directly to `run_pipeline()`.
- Batch exports are generated from returned pipeline artefacts without re-running NLP or diarisation.
- Tests cover `--nlp`, `--diarise`, `--export`, `--output-dir`, and failure reporting.

**Files to change:** `batch_processor/batch_runner.py`, `pipeline_runner.py`, `tests/test_batch_runner.py`, `tests/test_integration.py`

**Estimated effort:** M

### RA-3: Add Streamlit workflow regression tests

**Context:** `ui/app.py` is the main user-facing surface and currently has 0% direct coverage. Changes to upload handling, option wiring, error rendering, or downloads can pass the core test suite.

**Success criteria:**
- A mocked single-file UI run reaches the results section and exposes consensus, plain-text, ZIP, and JSON downloads.
- A mocked multi-file run shows success and failure summaries correctly.
- A failed processing run shows remediation guidance and does not leave temporary files behind.

**Files to change:** `ui/app.py`, `tests/test_ui_app.py`, optionally `tests/fixtures/`

**Estimated effort:** L

### RA-4: Add audio preflight limits

**Context:** `audio_processor/pipeline.py` decodes the full file into memory and then creates several full-length processed arrays. There is no configurable duration, file-size, or decoded-sample guard.

**Success criteria:**
- Files exceeding configured limits fail before full decode with a clear error message.
- Limits can be configured for CLI, UI, Docker, and tests.
- Tests cover acceptable files, over-limit files, and metadata-read failures.

**Files to change:** `config.py`, `audio_processor/pipeline.py`, `pipeline_runner.py`, `ui/app.py`, `README.md`, `tests/test_audio_processor.py`, `tests/test_integration.py`

**Estimated effort:** M

### RA-5: Move runtime dependencies into package metadata

**Context:** `requirements.txt` pins runtime dependencies, but `pyproject.toml` declares `dependencies = []`, so package-style installs miss required libraries.

**Success criteria:**
- `pip install .` installs enough dependencies for core CLI imports to succeed.
- Optional extras exist for UI, export, diarisation, LLM, and dev tooling, or an equivalent documented grouping is present.
- Docker, CI, and README installation commands use the chosen source of truth.

**Files to change:** `pyproject.toml`, `requirements.txt`, `README.md`, `.github/workflows/ci.yml`, `Dockerfile`, `Dockerfile.gpu`

**Estimated effort:** M

### RA-6: Make dependency audit blocking

**Context:** CI runs `pip-audit`, but the command is followed by `|| true`, so dependency vulnerabilities do not fail builds. The local `.venv` did not have `pip-audit` installed during this review.

**Success criteria:**
- CI fails on unignored `pip-audit` findings.
- Any ignored vulnerability has an inline reason and review date.
- Local setup docs include the command for maintainers to run the same audit.

**Files to change:** `.github/workflows/ci.yml`, `README.md`, `.pre-commit-config.yaml` or `pyproject.toml`

**Estimated effort:** S

### RA-7: Sanitise transcript-derived export content

**Context:** Consensus Markdown and AI context content can include transcript-derived text. PDF export converts Markdown to HTML and renders it with WeasyPrint without an explicit allowlist or escaping policy.

**Success criteria:**
- HTML-like transcript content renders as literal transcript text in Markdown preview, PDF, DOCX, AI context, and plain-text outputs.
- Tests cover angle brackets, Markdown table delimiters, links, and script-like strings.
- Export code documents which markup is generated by Chorus and which content is escaped.

**Files to change:** `consensus_merger/renderer.py`, `export_engine/ai_context.py`, `export_engine/exporter.py`, `tests/test_exporter.py`, `tests/test_ai_context.py`, `tests/test_merger.py`

**Estimated effort:** M

### RA-8: Align GPU Docker dependencies with requirements

**Context:** `Dockerfile.gpu` installs `requirements.txt`, then separately installs `pyannote.audio==3.1.1`, while `requirements.txt` pins `pyannote-audio==4.0.4`.

**Success criteria:**
- CPU and GPU images resolve the same intended pyannote version unless a documented GPU-specific constraint exists.
- CI or a local script verifies imports for `torch`, `whisper`, `pyannote.audio`, and `pipeline_runner` inside the GPU image.
- README GPU instructions mention any intentional version divergence.

**Files to change:** `Dockerfile.gpu`, `requirements.txt`, `README.md`, `.github/workflows/release.yml`, optionally `.github/workflows/ci.yml`

**Estimated effort:** S

### RA-9: Replace runtime environment mutation with explicit config

**Context:** The UI sets `os.environ` after importing modules that already captured configuration constants. This can make sidebar selections fail to affect processing consistently.

**Success criteria:**
- User-selected model, consensus models, noise mode, device, alignment strategy, and optional features are passed explicitly through a typed config object or function parameters.
- Processing modules do not rely on UI-time `os.environ` mutation.
- Tests prove sidebar-equivalent options reach audio processing and transcription orchestration.

**Files to change:** `config.py`, `pipeline_runner.py`, `audio_processor/filters.py`, `audio_processor/pipeline.py`, `transcription_engine/orchestrator.py`, `ui/app.py`, `tests/test_config_models.py`, `tests/test_integration.py`, `tests/test_orchestrator.py`

**Estimated effort:** L

### RA-10: Add a bounded Whisper model cache policy

**Context:** `transcription_engine/whisper_engine.py` caches models by `(model, device)` and only clears them when `unload_model()` is called manually. Multi-model consensus can keep several large models resident in a long-lived process.

**Success criteria:**
- Cache size or memory policy is configurable.
- UI and batch runs can release unused models after completion.
- Tests cover cache eviction, CPU fallback, and explicit unload after multi-model runs.

**Files to change:** `config.py`, `transcription_engine/whisper_engine.py`, `pipeline_runner.py`, `ui/app.py`, `tests/test_whisper_engine.py`, `tests/test_integration.py`

**Estimated effort:** M

### RA-11: Build a reconstruction quality harness

**Context:** `nlp_reconstructor/reconstructor.py` can upgrade LOW tokens to MEDIUM, but tests mainly cover graceful degradation. There is no golden-data gate for accuracy-preserving behaviour.

**Success criteria:**
- A small fixture set covers names, acronyms, technical terms, non-English text, and no-vector tokens.
- Reconstruction must preserve or improve expected token choices in the fixture set.
- Reconstructed tokens remain auditable in output metadata.

**Files to change:** `nlp_reconstructor/reconstructor.py`, `consensus_merger/renderer.py`, `tests/test_reconstructor.py`, `tests/fixtures/`

**Estimated effort:** M

### RA-12: Synchronise Docker image metadata with project version

**Context:** `Dockerfile` and `Dockerfile.gpu` labels still report `1.0.0` while `VERSION`, `pyproject.toml`, and README release examples report `3.1.1`.

**Success criteria:**
- Docker labels are populated from `VERSION` or CI build arguments.
- Version-sync tests or devops scripts fail when Docker labels drift.
- Release images expose the correct version through image inspection.

**Files to change:** `Dockerfile`, `Dockerfile.gpu`, `docker-publish.sh`, `tests/test_version_sync.py`, `devops-practices/check-version-sync.sh`, `.github/workflows/release.yml`

**Estimated effort:** S

---

## v3.3.0 re-assessment & v4.0.0 readiness

Re-validated 29 June 2026 against the v3.3.0 tree (193 test functions present;
`VERSION`, `pyproject.toml`, and README all report 3.3.0). The original findings were
checked against current code rather than recalled.

### Findings that still stand (routed into v4.0.0)

| Finding (original) | Status at v3.3.0 | Evidence | Routed to |
|---|---|---|---|
| Output isolation drift — global `CONSENSUS_DIR` writes/reads | **Still present** | `export_engine/exporter.py:600,605,610` (`build_export_zip` reads sidecars from `CONSENSUS_DIR`); `diarisation/diariser.py:337` (`_speaker_names_path` hardcodes `CONSENSUS_DIR`) | WP2 |
| `pyproject.toml` `dependencies = []` — wheel installs no runtime deps | **Still present** | `pyproject.toml:12` | WP1 (RA-1.1) |
| UI and batch surfaces near-zero coverage | **Still present** | no test imports `run_batch` or `ui/app.py` | WP3 |
| `pip-audit` non-blocking in CI | **Still present** | `.github/workflows/ci.yml:120` ends with `\|\| true` | WP3 (RA-3.3) |
| librosa audioread deprecation | **Still present** | `audio_processor/pipeline.py:69` `librosa.load(...)` falls back to deprecated path without `soundfile` | WP1 (RA-1.4) |

### New observation since the original review

- **Two reconstruction modules now coexist** — `nlp_reconstructor/` (spaCy) and
  `llm_reconstructor/` (Ollama, added v3.0.0). They have overlapping responsibility and
  separate pipeline wiring. Consolidating them behind one interface is the principal
  breaking change that justifies the 4.0.0 major bump. Routed to **WP1 (RA-1.3)**.
- `CLAUDE.md` "Core Modules" still lists only `nlp_reconstructor/`; it omits
  `llm_reconstructor/`. WP1 updates this for documentation parity.

### Path to 4.0.0

The four work packages constituted the 4.0.0 scope and have all shipped. WP1 was
breaking and earned the major version; WP2 and WP3 were correctness and
test-parity hardening; WP4 was the visible
feature payload. Recommended execution order: **WP2 → WP1 → WP3 → WP4**. The release
owner bumps `VERSION` to 4.0.0 and writes the migration note once all four merge.
