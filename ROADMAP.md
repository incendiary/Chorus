# Chorus Engine — Roadmap

Tracked improvements identified during the June 2026 repository assessment.

---

## Architecture & Code Quality

- [x] **Extract shared `_sanitise_stem()` utility** (v2.0.1) — the filename sanitisation helper is duplicated in `pipeline_runner.py` and `ui/app.py`. ✓ Moved to `utils.py` module.

- [x] **Pipeline runner bypasses `merger.py`** (v2.0.2) — `run_pipeline()` reimplements merge logic inline instead of delegating to `consensus_merger/merger.py`. ✓ Refactored to call `merger.merge_transcripts_with_votes()` directly.

- [x] **Remove import-time side effects in `config.py`** (v2.0.3) — directory creation (`mkdir`) runs at import time (line 25), which complicates testing and causes side effects when any module imports `config`. ✓ Gated creation behind `ensure_output_dirs()` function called at pipeline start.

- [ ] **Tighten output path coupling** — the global `OUTPUTS_DIR` is patched in 5+ locations in test fixtures. Consider passing the output path as a parameter through the pipeline to improve testability and flexibility.

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

- [ ] **Version sync across all release channels** — ensure pyproject.toml, README.md, git tags, and ROADMAP completions stay in sync. ✓ Added comprehensive version sync tests in v2.0.6+.

---

*Last updated: 15 June 2026*
