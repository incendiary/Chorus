# Chorus Engine — Roadmap

Tracked improvements identified during the June 2026 repository assessment.

---

## Architecture & Code Quality

- [x] **Extract shared `_sanitise_stem()` utility** (v2.0.1) — the filename sanitisation helper is duplicated in `pipeline_runner.py` and `ui/app.py`. ✓ Moved to `utils.py` module.

- [x] **Pipeline runner bypasses `merger.py`** (v2.0.2) — `run_pipeline()` reimplements merge logic inline instead of delegating to `consensus_merger/merger.py`. ✓ Refactored to call `merger.merge_transcripts_with_votes()` directly.

- [x] **Remove import-time side effects in `config.py`** (v2.0.3) — directory creation (`mkdir`) runs at import time (line 25), which complicates testing and causes side effects when any module imports `config`. ✓ Gated creation behind `ensure_output_dirs()` function called at pipeline start.

---

## Error Handling

- [x] **Narrow diarisation exception catch** (v2.0.4) — `diariser.py` catches bare `Exception`, which could swallow real bugs (memory leaks, GPU errors, corrupt output). ✓ Narrowed to `(RuntimeError, OSError, ValueError)`.

---

## Test Coverage

- [x] **Add `enable_nlp=True` integration test** (v2.0.5) — the spaCy NLP reconstruction path has no end-to-end test coverage. ✓ Added `TestOptionalPipelineFeatures`.

- [x] **Add `enable_diarisation=True` integration test** (v2.0.5) — diarisation is tested at the unit level but not through the full pipeline. ✓ Added `TestOptionalPipelineFeatures`.

- [x] **Add corrupt/unreadable audio test** (v2.0.5) — only "file not found" is tested; verify graceful handling of truncated, zero-byte, and format-mismatched files. ✓ Added `test_corrupt_audio_raises_runtime_error()`.

- [x] **Strengthen subtitle format assertions** (v2.0.5) — current SRT/VTT tests only check for prefix/separator presence. ✓ Validate full spec compliance (timestamp format, sequence numbering, blank line separators).

- [x] **Audio filter property-based tests** (v2.0.6) — verify that filters produce expected acoustic characteristics (e.g., high-pass actually attenuates below cutoff, normalisation hits target dBFS). ✓ Added `TestFilterAcousticProperties`.

---

## Future Enhancements

- [x] **Parallel transcription for multi-GPU setups** (v2.0.7) — variants are now transcribed with configurable parallelism and device-aware assignment, including multi-CUDA-device round-robin scheduling.

- [x] **Version sync across all release channels** (v2.0.8) — pyproject.toml, README.md, git tags, and ROADMAP completions enforced via Python tests and shell consistency guard (`tests/version_consistency_test.sh`).

- [x] **Streamlit UI polish and batch UX** (v2.0.8) — live batch status panel with ETA, post-run summary with success/issue badges, failed-file jump links, quick-navigation bar with sticky positioning, keyboard skip links, mobile-responsive layout, results filter control, centralised microcopy constants, and in-session recent run snapshots.

- [x] **Tighten output path coupling** (v2.0.9) — `output_dir: Path | None` parameter added to `run_pipeline()` and threaded through all stages. Tests use `tmp_path` isolation; two new `TestOutputDirIsolation` integration tests confirm correct sub-directory creation and run isolation.

- [x] **Multi-model consensus orchestration and UI controls** (v2.1.0) — model cache keyed by `(model, device)`, explicit `CONSENSUS_MODELS` configuration, orchestrator expansion to model×variant passes, and Streamlit sidebar controls that forward selected model sets through the pipeline with compatibility-preserving primary transcript keys.

- [x] **LLM reconstruction API and runtime integration** (v3.0.0) — added `llm_reconstructor` with local Ollama client, LOW-token recovery helpers, pipeline merge-stage wiring, and Streamlit toggle controls for optional LLM-assisted reconstruction.

- [x] **DevOps practices bootstrap** (v3.0.0) — added root `VERSION` file as single version source of truth; three portable enforcement scripts (`check-version-sync.sh`, `check-clone-refs.sh`, `check-test-baseline.sh`) in `devops-practices/`; weekly scheduled CI run for drift detection; extended Python version-sync tests for `VERSION`/pyproject parity.

---

## Completed — Post-v3 Hardening

- [x] **Ollama failure UX surfacing** (v3.1.0) — when the Ollama server is unreachable the UI auto-disables the toggle and shows a dismissible warning; timeout and connection errors are surfaced via `probe_model()` pre-flight check.

- [x] **LLM reconstruction timeout/fallback integration tests** (v3.1.0) — end-to-end tests covering Ollama timeout, HTTP error, and malformed JSON; votes are confirmed unchanged in all failure modes.

- [x] **Ollama model availability pre-flight** (v3.1.0) — `probe_model()` probes `/api/tags` before any pipeline run when `enable_llm=True`; surfaces a clear error if the model is not pulled, rather than failing mid-pipeline.

- [x] **`load_transcripts_from_disk` respects `output_dir`** (v3.1.0) — optional `transcripts_dir` parameter added; falls back to global `TRANSCRIPTS_DIR` when omitted.

---

## Completed — Next Feature Work

- [x] **Confidence-weighted LLM prompting** (v3.1.0) — per-candidate agreement percentages passed to Ollama prompt; reconstructor builds weights from variant occurrence counts.

- [x] **Streaming Whisper transcription progress** (v3.1.0) — optional `segment_callback(index, total, text)` added to `transcribe()` and threaded through orchestrator; pipeline wires a handler so the UI progress bar advances per decoded segment.

- [x] **Export to JSON transcript bundle** (v3.1.0) — `export_transcript_bundle()` writes `{stem}_bundle.json` with all variant transcripts, the word-vote sequence, and aggregate statistics; auto-generated after every pipeline run.

- [x] **CLI `--output-dir` flag** (v3.1.0) — `--output-dir / -o` added to `pipeline_runner.py` CLI; writes all outputs under the specified directory.

- [x] **Batch processor `output_dir` isolation** (v3.1.0) — `output_dir` parameter added to `run_batch()` and `--output-dir / -o` to the batch CLI; each file writes to an isolated `<dir>/<stem>/` subdirectory.

- [x] **Docker Compose environment documentation** (v3.1.0) — `CONSENSUS_MODELS`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS` documented in `docker-compose.yml` with defaults and usage notes.

- [x] **`ROADMAP.md` automated freshness check** (v3.1.0) — `version_consistency_test.sh` extended with check 11 that warns when the roadmap `Last updated` date exceeds 30 days.

---

## Completed — v3.1.1 Hardening (June 2026)

- [x] **MPS float64 CPU fallback** (v3.1.1) — Whisper's word-timestamp DTW alignment requires float64, which Apple MPS does not support. `transcription_engine/whisper_engine.py` now catches the `TypeError`, loads the model on CPU via the existing `(model, device)` cache, and retries the affected pass. All other passes continue on MPS.

- [x] **Ollama setup dialog popup** (v3.1.1) — replaced the silent `st.warning` on failed LLM checkbox with a `@st.dialog` modal that surfaces the specific failure reason (connection refused / model not pulled) and provides a step-by-step fix with Dismiss and Retry buttons.

- [x] **Help & FAQ page** (v3.1.1) — added `ui/pages/1_Help.py`: what Chorus does, quick start, confidence tiers, audio format table, Ollama setup walkthrough (macOS, Linux/Docker), export format reference, and FAQ.

- [x] **In-app logging page** (v3.1.1) — added `ui/pages/2_Logs.py`: session-state log buffer (capped at 500 entries) wired to root logger via `_SessionLogHandler`; filterable by level with clear and download buttons.

- [x] **Theme selector** (v3.1.1) — added `.streamlit/config.toml` (`base = "light"`) and extended CSS overrides to `section[data-testid="stSidebar"]` and `.stButton > button` so themes affect Streamlit's native elements, not only custom HTML blocks.

- [x] **Ollama environment variables documented** (v3.1.1) — `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS` added to `.env.example` with model guidance and quick-start commands.

- [x] **Past Jobs page** (v3.1.2) — added `ui/pages/3_Past_Jobs.py`: scans `outputs/consensus/` for completed runs, presents them newest-first as expandable cards with per-format download buttons and a "Download All" ZIP option. No pipeline re-execution; reads existing files from disk.

---

## Completed — v3.2.0 Output Isolation & Traceability

- [x] **Record original source filename in all output artefacts** (v3.2.0) — the original source filename (with extension, before sanitisation) is now captured at pipeline entry and threaded through all stages:
  - `source_filename` captured from `audio_path.name` in `pipeline_runner.py`.
  - Stored in `bundle.json` `meta` block via updated `export_transcript_bundle()`.
  - Included in `consensus.md` header via updated `renderer.render_consensus()`.
  - Passed to `merger.merge_transcripts_with_votes()` for end-to-end traceability.
  - Three integration tests added: output directory isolation, bundle metadata preservation, consensus markdown header verification.

## Upcoming

- [ ] **Suppress or optimise MPS float64 warnings** — on macOS with Apple Silicon, Whisper's word-level timestamp alignment requires float64, which Metal Performance Shaders does not support. The CPU fallback works correctly but generates a `UserWarning`. Investigate options:
  - Detect MPS availability earlier and suppress the redundant retry message.
  - Consider upstream Whisper patches to avoid float64 requirement entirely.
  - Document the performance trade-off in the Help page for macOS users.

- [x] **Live log window with user-configurable line count** (v3.2.1) — tail-N number input (default 50) added to the Logs page toolbar; shows only the last N entries. Consecutive identical messages deduplicated with ×N repeat badge. Existing Download and Clear buttons preserved.

> The spaCy setup, LLM context document, and best-guess export items previously
> listed here were folded into and shipped as v4.0.0's WP4 — see above.

---

## Completed — v3.2.1 UI Fixes & Code Quality

- [x] **Black formatting drift resolved** (v3.2.1) — 9 files flagged by CI Black check reformatted. No logic changes; CI Black step now passes on main.

- [x] **Past Jobs duplicate element key crash** (v3.2.1) — `StreamlitDuplicateElementKey` raised when the same source file was processed more than once. Download button keys now include `full_stem` to ensure per-run uniqueness.

- [x] **Past Jobs delete with confirmation** (v3.2.1) — inline delete button added to each run expander. Confirmation prompt shows file count before deletion; removes all associated artefacts from disk.

- [x] **Past Jobs grouped by date** (v3.2.1) — runs now rendered under date subheadings (newest first) instead of a flat list.

---

## Completed — v3.3.0 UI Controls & Configuration

- [x] **`large` Whisper model added to UI selector** (v3.3.0) — previously only accessible via `WHISPER_MODEL` env var. Now available in the sidebar Model size dropdown alongside tiny/base/small/medium.

- [x] **Compute device exposed in UI** (v3.3.0) — `WHISPER_DEVICE` was env-only. Now a sidebar selectbox: Auto-detect / CPU / NVIDIA CUDA / Apple MPS. Changes take effect immediately without restart.

- [x] **Transcription parallelism exposed in UI** (v3.3.0) — `TRANSCRIPTION_PARALLELISM` was env-only. Now an Auto toggle + Worker count input (1–16) in the sidebar.

- [x] **Hardware survey presets** (v3.3.0) — sidebar preset dropdown (Max / Background) with Apply button. Detects RAM, CPU, and GPU in-process via `ui/hardware_survey.py` (PyTorch + sysctl/procfs). Max applies the largest viable model and auto parallelism; Background steps one model tier down and pins parallelism to 1 for a responsive machine.

- [x] **Whisper model recommendation in survey script** (v3.3.0) — `devops-practices/survey-ollama-env.sh` now recommends a `WHISPER_MODEL` value alongside Ollama models, using the same hardware thresholds as the in-app survey.

- [x] **Whisper cache released before Ollama LLM reconstruction** (v3.3.0) — `clear_model_cache()` called after transcription completes when `enable_llm=True`. Prevents Whisper and Ollama models from holding unified memory simultaneously. Makes Whisper `large` + `neural-chat:13b` viable on 32 GB Apple Silicon (~28 GB peak vs ~31 GB previously).

- [x] **Full configuration reference** (v3.3.0) — `docs/CONFIGURATION.md` added: covers all 11 user-configurable options with trade-off explanations, hardware recommendations, quick-start table, and env var summary.

---

## Completed — v4.0.0 "Trustworthy outputs, stable surface"

The major release. It earns the major bump because **WP1 introduces breaking
changes** to import paths, packaging, and the reconstruction module layout. The theme:
make Chorus installable and integrable as a library, guarantee output isolation, and
test the surfaces users actually touch. See the README "Breaking changes in v4.0.0"
section for the user-facing migration note.

Each work package had a detailed, self-contained task specification at the time;
those specs have since been removed as their work completed (see the file changes
recorded under each item below).

### WP1 — Packaging & stable public API (BREAKING)

- [x] **Declare runtime dependencies in `pyproject.toml`** (v4.0.0) — `dependencies = []` today, so `pip install chorus-engine` installs no runtime deps. (RA-1.1) — files: `pyproject.toml`.
- [x] **Establish a stable top-level `chorus` public API** (v4.0.0) — re-export `run_pipeline`, `run_batch`, and the supported entry points; commit to keeping them stable. (RA-1.2) — files: `chorus/__init__.py`, `pyproject.toml`, `tests/test_public_api.py`, `README.md`.
- [x] **Consolidate `nlp_reconstructor` + `llm_reconstructor` into one `reconstruction` package** (v4.0.0) — single strategy-based interface; breaking import change. (RA-1.3) — files: `reconstruction/` (`__init__.py`, `nlp.py`, `llm.py`, `ollama_client.py`), `consensus_merger/merger.py`, `pipeline_runner.py`, `ui/app.py`, `pyproject.toml`, `tests/test_reconstructor.py`, `tests/test_llm_reconstructor.py`, `tests/test_integration.py`, `CLAUDE.md`, `README.md`.
- [x] **Retire the deprecated librosa audioread fallback** (v4.0.0) — make `soundfile` an explicit dependency and use the non-deprecated load path. (RA-1.4) — files: `audio_processor/pipeline.py`, `requirements.txt`, `pyproject.toml`, `tests/test_audio_processor.py`.

### WP2 — Output-routing correctness

- [x] **Thread `output_dir` through `build_export_zip`** (v4.0.0) — fixes `exporter.py:600/605/610` reading sidecars from the global `CONSENSUS_DIR`. (RA-2.1)
- [x] **Make speaker-name persistence honour `output_dir`** (v4.0.0) — fixes `diariser.py:337` hardcoding `CONSENSUS_DIR`. (RA-2.2)
- [x] **Add a global-directory leak regression guard** (v4.0.0) — assert an isolated run writes nothing to the global `CONSENSUS_DIR`. (RA-2.3)
  - **Files changed:** `export_engine/exporter.py`, `diarisation/diariser.py`, `pipeline_runner.py`, `ui/app.py`, `tests/test_exporter.py`, `tests/test_integration.py`, `tests/test_speaker_names.py`
  - **Tests:** 191 → 197 passing (output_dir isolation coverage)

### WP3 — User-facing test parity & CI hardening

- [x] **Batch processor test coverage** (v4.0.0) — `batch_runner.py` was at ~0 %; added 25 tests covering isolation, partial failure, and empty input. (RA-3.1) — files: `tests/test_batch_runner.py`.
- [x] **Make `pip-audit` blocking in CI** (v4.0.0) — removed the `|| true` that swallowed CVE findings. (RA-3.3) — files: `.github/workflows/ci.yml`.
- [x] **Streamlit UI smoke/behaviour tests** (v4.0.0) — `ui/app.py` was at 0 %; added 9 tests via `streamlit.testing.v1.AppTest` covering render smoke, sidebar controls (model/device/hardware-preset selectors, parallelism toggle), and the Ollama/spaCy setup-dialog paths. (RA-3.2) — files: `tests/test_ui_app.py`.

### WP4 — Headline user features

- [x] **Human-readable "best-guess" transcript export** (v4.0.0) — clean `{stem}_best_guess.txt`, no markup. (RA-4.1)
  - **Files changed:** `export_engine/exporter.py`, `pipeline_runner.py`, `ui/app.py`, `tests/test_exporter.py`, `README.md`, `ui/pages/1_Help.py`
- [x] **LLM context document** (v4.0.0) — `docs/CHORUS_FOR_LLMS.md` explaining the project and outputs to language models. (RA-4.2)
  - **Files changed:** `docs/CHORUS_FOR_LLMS.md`, `README.md`, `ui/pages/1_Help.py`
- [x] **Streamline spaCy model setup** (v4.0.0) — actionable guidance instead of a silent fallback warning. (RA-4.3)
  - **Files changed:** `reconstruction/nlp.py`, `reconstruction/__init__.py`, `pipeline_runner.py`, `ui/app.py`, `tests/test_reconstructor.py`, `ui/pages/1_Help.py`

> The three pre-existing "Upcoming" items above (best-guess export, LLM context doc,
> spaCy setup) are now folded into WP4. The MPS float64 warning cleanup remains a
> standalone 3.x maintenance item and is **not** required for 4.0.0.

---

## Completed — v4.0.1 Patch

- [x] **Fix nonexistent/stale Ollama model recommendations** (v4.0.1) — two hardcoded model tags (`neural-chat:13b`, `tiny-llama:latest`) never existed on the Ollama registry; replaced the whole recommendation philosophy with a research-backed default (`qwen2.5:3b`) plus an explicit jargon-heavy-transcript option (`qwen2.5:14b`), added runtime registry validation and a weekly CI staleness check, and fixed an "already installed" false-positive bug. (RA from user report)
  - **Files changed:** `devops-practices/survey-ollama-env.sh`, `.github/workflows/ollama-model-tags-check.yml`, `config.py`, `.env.example`, `docker-compose.yml`, `docker-compose.ollama.yml`, `ui/app.py`, `ui/pages/1_Help.py`, `docs/CONFIGURATION.md`
- [x] **Fix open nltk CVE in `pyproject.toml`** (v4.0.1) — `requirements.txt` was patched for PYSEC-2026-597 in v4.0.0, but `pyproject.toml`'s mirrored dependency list was never updated, leaving 2 open Dependabot alerts. (RA-1, partial — see below)
  - **Files changed:** `pyproject.toml`
- [x] **Restructure README native-first, split Docker into `docs/DOCKER.md`** (v4.0.1) — native installation is now the primary path (required for Apple Silicon MPS); removed a large stale/duplicate Ollama recommendations section; fixed a real bug in `docs/DOCKER.md` (invalid `docker-compose -f Dockerfile.gpu` syntax) and several stale defaults.
  - **Files changed:** `README.md`, `docs/DOCKER.md`, `tests/test_version_sync.py`, `tests/version_consistency_test.sh`
- [x] **Fresh holistic codebase review** (v4.0.1) — see `REVIEW.md` for full findings; added RA-1 through RA-9 below.

### From the 12 July 2026 holistic review

Full findings, risk scoring, and predicted failure scenarios in `REVIEW.md`.

- [x] **RA-1: Prevent pyproject.toml / requirements.txt drift** (v4.0.1) — automated CI check added to `check_dependency_drift.sh`; fails if any shared dependency has mismatched versions between the two files. (Effort: S)
- [x] **RA-2: Make pip-audit cover pyproject.toml's dependency list** (v4.0.1) — `security.yml`'s `pip-audit` step now installs both `requirements.txt` and `.[dev]` extras before scanning, ensuring vulnerabilities in `pyproject.toml` are visible to CI. (Effort: S)
- [x] **RA-3: Add SECURITY.md and enable private vulnerability reporting** (v4.0.1) — `SECURITY.md` added; private vulnerability reporting, secret scanning, and Dependabot security updates enabled via the GUI. (Effort: XS)
- [x] **RA-4: Add CodeQL scanning** (v4.0.1) — handled via GitHub's default CodeQL setup (Security & Analysis GUI) rather than a checked-in workflow; a custom `codeql.yml` was tried first but conflicts with default setup and was dropped. (Effort: XS)
- [ ] **RA-5: Test hardware_survey.py's detection and recommendation logic** — 14% coverage on the code directly behind the one-click hardware preset button. (Effort: M)
- [x] **RA-6: Verify ollama-model-tags-check.yml actually works under real CI** (v4.0.1) — workflow ran successfully on scheduled cron (2026-07-13 09:00) and validated three Ollama model tags (`qwen2.5:0.5b`, `qwen2.5:14b`, `qwen2.5:3b`) against the public Ollama registry; all tags resolved (HTTP 200).
- [ ] **RA-7: Expand export_engine/exporter.py test coverage** — 62% coverage; PDF/DOCX export paths have no direct test evidence. (Effort: M)
- [ ] **RA-8: Expand reconstruction/nlp.py test coverage beyond degradation paths** — 39% coverage; the actual grammatical-correction logic is thin on direct tests. (Effort: S)
- [x] **RA-9: Decompose ui/app.py** (v4.0.1) — split the single 1744-line file into focused modules: `ui/theme.py` (presets, page config, CSS, header), `ui/sidebar.py` (`render_sidebar` and the `SidebarConfig` dataclass), `ui/upload.py` (`render_upload`), `ui/pipeline_invocation.py` (`run_one_file` and `render_run_section`), and `ui/results.py` (render helpers, status panels, and `render_file_results`); `ui/app.py` is now a thin entry point wiring the page together and retaining the session-state log handler. Session-state keys, widget labels, and behaviour are unchanged; the nine `AppTest` smoke tests pass unmodified. (Effort: L)

---

*Last updated: 14 July 2026*
