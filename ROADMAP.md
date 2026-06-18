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

## Upcoming

- [ ] **Record original source filename in all output artefacts** — the original source filename (with extension, before sanitisation) is not currently stored anywhere. Only a sanitised stem is written. This means documents cannot be traced back to their source file without relying on the timestamped stem. Required changes:
  - Store `source_filename: str` in `bundle.json` `meta` block at pipeline run time.
  - Include a **Source file** field in the `consensus.md` header block.
  - Include a **Source file** field in the AI context pack (`ai_context.md`) header.
  - Pass the value to PDF and DOCX exports as document title metadata.
  - Update `ui/pages/3_Past_Jobs.py` to read `source_filename` from `bundle.json` and display it as the primary run identifier (falling back to the stem-derived name for runs produced before this change).

---

*Last updated: 18 June 2026*
