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

## Outstanding — Post-v3 Hardening

- [ ] **Ollama failure UX surfacing** — when the Ollama server is unreachable or returns a non-200 response, the Streamlit UI should surface a dismissible warning rather than silently skipping reconstruction. Timeout and connection errors should be user-readable.

- [ ] **LLM reconstruction timeout/fallback integration tests** — add end-to-end tests covering Ollama timeout (simulated via monkeypatched `urlopen`), HTTP error, and malformed JSON response; confirm votes are returned unchanged in all failure modes.

- [ ] **Ollama model availability pre-flight** — before running the pipeline when `enable_llm=True`, probe the configured Ollama model with a lightweight `/api/tags` call and surface a clear error if the model is not pulled, rather than failing mid-pipeline.

- [ ] **`load_transcripts_from_disk` respects `output_dir`** — the existing function always reads from the global `TRANSCRIPTS_DIR`; it should accept an optional `transcripts_dir` parameter to support the run-isolated path introduced in v2.0.9.

---

## Outstanding — Next Feature Work

- [ ] **Confidence-weighted LLM prompting** — pass the full variant list and per-variant word-count metadata to Ollama so the prompt context is richer than a bare candidate list; assess whether this materially improves LOW-token upgrade rates on real audio.

- [ ] **Streaming Whisper transcription progress** — expose word-level streaming results from Whisper during transcription so the UI progress bar advances word-by-word rather than only updating when a full variant completes.

- [ ] **Export to JSON transcript bundle** — provide a single structured `.json` export containing all variant transcripts, the consensus word-vote sequence, and confidence statistics, for downstream programmatic consumption.

- [ ] **CLI `--output-dir` flag** — expose the `output_dir` parameter already wired into `run_pipeline()` via a `--output-dir` argument in the CLI entry point so headless batch operators can control output placement.

- [ ] **Batch processor `output_dir` isolation** — `batch_processor/batch_runner.py` does not currently use per-run `output_dir`; each batch job should write to an isolated subdirectory to avoid cross-job file collision on concurrent runs.

- [ ] **Docker Compose environment documentation** — `CONSENSUS_MODELS` and `OLLAMA_BASE_URL` are new v3 environment variables not yet documented in `docker-compose.yml` comments or the README environment table.

- [ ] **`ROADMAP.md` automated freshness check** — extend `version_consistency_test.sh` to warn when the roadmap `Last updated` date is more than 30 days behind `HEAD` commit date, so the document does not silently drift from reality.

---

*Last updated: 16 June 2026*
