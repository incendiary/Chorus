# WP1 — Packaging & stable public API (BREAKING)

**Target release:** v4.0.0
**Breaking:** Yes — this is the work package that earns the major version bump.
**Depends on:** nothing. Do this before WP3.
**Branch:** `feat/v4-wp1-packaging`
**Overall effort:** L

Read `docs/tasks/README.md` ("Conventions every agent MUST follow") before starting.

## Why this exists

Chorus today cannot be installed and used as a library, and its internal module
layout leaks into callers. Three concrete problems:

1. `pyproject.toml` line 12 declares `dependencies = []`. A `pip install chorus-engine`
   therefore installs **none** of the runtime dependencies; the package only works if
   `requirements.txt` is installed separately by hand. (Confirmed in `REVIEW.md`.)
2. There is no stable public API. Callers and tests import from deep paths such as
   `from consensus_merger.merger import merge_transcripts_with_votes`. Any internal
   move breaks them.
3. There are **two** reconstruction modules with overlapping responsibility —
   `nlp_reconstructor/` (spaCy) and `llm_reconstructor/` (Ollama) — that grew
   separately. They should sit behind one coherent interface.

A major version is the correct, and only, time to fix 2 and 3, because both change
import paths.

---

### RA-1.1: Declare runtime dependencies in `pyproject.toml`

**Context:** `pyproject.toml` has `dependencies = []`. The real runtime deps live only
in `requirements.txt`. Move/declare them so the built wheel is self-sufficient.

**What to do:**
- Read `requirements.txt`. Copy the **runtime** pins into `pyproject.toml`'s
  `[project] dependencies` list. Keep the existing version pins exactly.
- Keep `requirements.txt` working (it is referenced by Dockerfiles and CI). The
  simplest non-duplicating option is to leave `requirements.txt` as the canonical pin
  file and have `dependencies` mirror it; if you prefer a single source, make
  `requirements.txt` a generated artefact — but **do not** change the Dockerfiles'
  `pip install -r requirements.txt` step in this task.
- Do not add new dependencies. Do not move dev-only tools (they belong in the existing
  `[project.optional-dependencies] dev`).

**Success criteria:**
- In a scratch venv, `pip install .` (no `-r requirements.txt`) succeeds and
  `python -c "import audio_processor, transcription_engine, consensus_merger"` works.
- `tests/test_version_sync.py` still passes (it checks VERSION/pyproject parity — do
  not touch the version field).
- `.venv/bin/python -m pytest` is green.

**Files to change:** `pyproject.toml`, possibly `requirements.txt`.
**Effort:** S

---

### RA-1.2: Establish a stable top-level `chorus` public API

**Context:** Callers reach into deep modules. Define one import surface that 4.0.0
commits to keeping stable.

**What to do:**
- Create a top-level package `chorus/` with an `__init__.py` that re-exports the
  supported public entry points. At minimum:
  - `run_pipeline` (from `pipeline_runner`)
  - `run_batch` (from `batch_processor.batch_runner`)
  - the consensus merge entry (`merge_transcripts_with_votes` from
    `consensus_merger.merger`)
  - the export entry points used by external callers (`export_all`,
    `export_transcript_bundle` from `export_engine.exporter`)
- Add `chorus*` to `[tool.setuptools.packages.find] include` in `pyproject.toml`.
- Define `__all__` in `chorus/__init__.py` and a module docstring stating this is the
  stable public API and deeper paths are internal.
- **Do not** delete or move the existing modules — `chorus/` is a thin façade. This
  keeps the change surgical and the diff reviewable.

**Success criteria:**
- `from chorus import run_pipeline, run_batch` works after `pip install .`.
- A new test `tests/test_public_api.py` imports every name in `chorus.__all__` and
  asserts each is callable.
- Existing tests unchanged and green.

**Files to change:** new `chorus/__init__.py`, `pyproject.toml`, new
`tests/test_public_api.py`. Update `README.md` "Architecture" / usage with the new
import style.
**Effort:** M

---

### RA-1.3: Consolidate the two reconstruction modules behind one interface (BREAKING)

**Context:** `nlp_reconstructor/reconstructor.py` (spaCy) and
`llm_reconstructor/reconstructor.py` + `llm_reconstructor/ollama_client.py` (Ollama)
both reconstruct LOW-confidence tokens via different strategies. They are wired
separately into the pipeline. Unify them under one package with a common entry point
so callers select a *strategy*, not a *module*.

**What to do:**
- Create `reconstruction/` package containing:
  - `__init__.py` exposing a single `reconstruct(votes, *, strategy, **opts)` function
    (or a small strategy registry) where `strategy ∈ {"nlp", "llm"}`.
  - Move the spaCy logic and the Ollama logic into `reconstruction/nlp.py` and
    `reconstruction/llm.py` respectively, keeping their existing function behaviour.
  - Keep `ollama_client.py` as `reconstruction/ollama_client.py`.
- Update the pipeline (`pipeline_runner.py`) and the UI toggles (`ui/app.py`) to call
  the unified entry point. Preserve the existing `enable_nlp` / `enable_llm` flag
  behaviour exactly — same flags, same outcomes.
- Update `pyproject.toml` `packages.find` include list (`reconstruction*`; remove the
  two old names once empty).
- **Migrate the tests:** `tests/test_reconstructor.py` and
  `tests/test_llm_reconstructor.py` must be updated to the new import paths. Do not
  weaken their assertions.
- Update `CLAUDE.md` "Core Modules" (it currently lists only `nlp_reconstructor/`) and
  `README.md` architecture section.

**Success criteria:**
- `from chorus import ...` and the pipeline run identically for NLP-only, LLM-only,
  and both-enabled paths (the existing integration tests in `tests/test_integration.py`
  for `TestOptionalPipelineFeatures` pass unchanged in behaviour, updated only for
  imports).
- No remaining references to `nlp_reconstructor` or `llm_reconstructor` anywhere
  (`grep -rn "nlp_reconstructor\|llm_reconstructor" --include='*.py' .` returns
  nothing outside `.venv`).
- This is the documented breaking change in the 4.0.0 migration note.

**Files to change:** new `reconstruction/` package, delete old
`nlp_reconstructor/` and `llm_reconstructor/`, `pipeline_runner.py`, `ui/app.py`,
`pyproject.toml`, `tests/test_reconstructor.py`, `tests/test_llm_reconstructor.py`,
`tests/test_ai_context.py` (if it imports these), `CLAUDE.md`, `README.md`.
**Effort:** L

---

### RA-1.4: Retire the deprecated librosa audioread fallback

**Context:** `audio_processor/pipeline.py:69` calls `librosa.load(...)`. When
`soundfile`/PySoundFile is unavailable, librosa silently falls back to the deprecated
`__audioread_load` path, which librosa will remove in v1.0. (Roadmap "Upcoming" item.)

**What to do:**
- Make `soundfile` an explicit, declared runtime dependency (it likely already
  installs transitively; declare it directly in `requirements.txt` and the new
  `pyproject.toml` dependencies from RA-1.1).
- In `audio_processor/pipeline.py`, load audio via the non-deprecated path. Add a
  clear, British-English error if audio loading fails (corrupt/unsupported file) that
  names the file and suggests a fix, rather than letting a deprecation warning or a
  bare exception surface. Reuse the existing corrupt-file handling pattern — see
  `tests/test_audio_processor.py::test_corrupt_audio_raises_runtime_error`.
- Do not change the resampling target or mono conversion behaviour.

**Success criteria:**
- `tests/test_audio_processor.py` passes, including the corrupt-audio test.
- No `librosa` deprecation warning is emitted when loading a standard WAV/MP3 in the
  test suite (assert via `pytest.warns`/`recwarn` in a new small test, or confirm the
  soundfile path is taken).

**Files to change:** `audio_processor/pipeline.py`, `requirements.txt`,
`pyproject.toml`, `tests/test_audio_processor.py`. Update the ROADMAP "Resolve librosa
audioread deprecation" item.
**Effort:** S
