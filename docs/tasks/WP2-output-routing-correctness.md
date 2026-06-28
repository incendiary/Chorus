# WP2 — Output-routing correctness

**Target release:** v4.0.0 (independently shippable as 3.4.x)
**Breaking:** No
**Depends on:** nothing. Recommended to do first.
**Branch:** `feat/v4-wp2-output-routing`
**Overall effort:** M

Read `docs/tasks/README.md` ("Conventions every agent MUST follow") before starting.

## Why this exists

This is the **top risk in `REVIEW.md`**. `run_pipeline()` accepts an `output_dir` to
isolate a run's artefacts, and most writers honour it (`output_dir or CONSENSUS_DIR`).
But a few helpers still hardcode the global `config.CONSENSUS_DIR`, so an isolated run
can silently **read stale files from a previous run** or **write to the shared
directory** — mixing artefacts across CLI, UI, and batch executions.

These are confirmed, concrete bugs at v3.3.0:

| Location | Bug |
|----------|-----|
| `export_engine/exporter.py:600` | `build_export_zip` reads `CONSENSUS_DIR / f"{stem}_speakers.json"` — ignores `output_dir`. |
| `export_engine/exporter.py:605` | same, for `{stem}_ai_context.md`. |
| `export_engine/exporter.py:610` | same, for `{stem}_diarised.md`. |
| `diarisation/diariser.py:337` | `_speaker_names_path(stem)` returns `CONSENSUS_DIR / f"{stem}_speakers.json"` — ignores `output_dir` everywhere it is used. |

Effect: a UI or batch run with an isolated `output_dir` builds a download ZIP that is
**missing** the speaker, AI-context, and diarised sidecars (they were written into the
isolated dir but read from the global one) — or, worse, bundles a **stale** sidecar
left in the global dir by an earlier run of the same filename.

---

### RA-2.1: Thread `output_dir` through `build_export_zip`

**Context:** `export_engine/exporter.py` `build_export_zip(...)` (around line 593)
reads three sibling artefacts from the global `CONSENSUS_DIR` (lines 600, 605, 610).

**What to do:**
- Add an `output_dir: Path | None = None` parameter to `build_export_zip`, matching the
  signature style already used by `export_all`, `export_transcript_bundle`, and the
  other functions in this file (`target_dir = output_dir or CONSENSUS_DIR`).
- Resolve the three sidecar paths from `output_dir or CONSENSUS_DIR`, not the bare
  constant.
- Find every caller of `build_export_zip` (UI download handler in `ui/app.py`, any
  Past Jobs page, batch code) and pass the run's `output_dir` through. Search:
  `grep -rn "build_export_zip" --include='*.py' .`
- Also pass `output_dir` into the nested `export_all(...)` call inside the ZIP builder
  (line ~616) so format exports resolve consistently.

**Success criteria:**
- A ZIP built from an isolated `output_dir` contains the speakers/ai_context/diarised
  sidecars that were written into *that* dir.
- A ZIP built from an isolated `output_dir` does **not** pick up a same-named file
  sitting in the global `CONSENSUS_DIR` (regression for cross-run contamination).
- New test in `tests/test_exporter.py` covers both above using `tmp_path`.

**Files to change:** `export_engine/exporter.py`, `ui/app.py` (and any other caller),
`tests/test_exporter.py`.
**Effort:** M

---

### RA-2.2: Make speaker-name persistence honour `output_dir`

**Context:** `diarisation/diariser.py:335` `_speaker_names_path(stem)` hardcodes
`CONSENSUS_DIR`. Both the writer and reader of the speaker-names sidecar go through it,
so speaker names always land in / load from the global directory regardless of the
run's `output_dir`.

**What to do:**
- Change `_speaker_names_path` to accept the run directory:
  `_speaker_names_path(stem, output_dir: Path | None = None) -> Path` returning
  `(output_dir or CONSENSUS_DIR) / f"{stem}_speakers.json"`.
- Update every caller (save and load of speaker names) to thread the run's `output_dir`
  through. Search: `grep -rn "_speaker_names_path\|_speakers.json" --include='*.py' .`
- Ensure the diarisation stage in `pipeline_runner.py` passes its `output_dir` down.

**Success criteria:**
- With an isolated `output_dir`, the speakers JSON is written to and read from that dir.
- New/extended test in `tests/test_speaker_names.py` (or `tests/test_integration.py`)
  asserts the sidecar lands under `tmp_path`, and that the global `CONSENSUS_DIR` is
  untouched.

**Files to change:** `diarisation/diariser.py`, `pipeline_runner.py`, any other caller,
`tests/test_speaker_names.py`.
**Effort:** S

---

### RA-2.3: Add a global-directory leak regression guard

**Context:** The existing `TestOutputDirIsolation` integration tests confirm files land
in the right place, but nothing asserts the *negative* — that an isolated run writes
**nothing** to the global `CONSENSUS_DIR`. After RA-2.1 and RA-2.2 this should be true
end-to-end.

**What to do:**
- Add an integration test that runs the full pipeline (mocking heavy stages as existing
  integration tests do — copy the pattern from `tests/test_integration.py`) with
  `output_dir=tmp_path`, enabling diarisation and AI-context, then asserts the global
  `CONSENSUS_DIR` gained **zero** new files during the run.
- Implement by snapshotting `set(CONSENSUS_DIR.iterdir())` before and after, or by
  monkeypatching `config.CONSENSUS_DIR` to a separate empty tmp dir and asserting it
  stays empty.

**Success criteria:**
- The new test fails on the pre-WP2 code (verify by stashing your RA-2.1/2.2 changes)
  and passes after. State this in the PR description.

**Files to change:** `tests/test_integration.py`.
**Effort:** S
