# RB-5: Test the UI run loop and results rendering

**Model tier:** Sonnet · **Effort:** M · **Branch:** `test/rb5-ui-run-loop-coverage`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first.

## Context

v4.0.1's RA-9 decomposed the 1744-line `ui/app.py` into focused modules. Two of them
are the least-tested code in the repo, and they implement the path the maintainer
exercises every single use:

- `ui/pipeline_invocation.py` — **13 % coverage** — `run_one_file()` (single-file
  pipeline invocation with progress callbacks and error capture) and
  `render_run_section()` (mode selection, pre-flight checks, the run button, the
  sequential and all-at-once processing loops, per-file status panels).
- `ui/results.py` — **12 % coverage** — `render_file_results()` and its helpers
  (variant tabs, consensus render, download buttons, diarisation table, AI context
  pack panel, message constants).

The RA-9 refactor deliberately hoisted these out of closures to module level, making
them mockable/testable for the first time. The refactoring agent explicitly flagged
this path as "verified by close reading only, not execution" — this task supplies the
missing execution evidence. The predicted failure (REVIEW.md PF-4): a Streamlit bump
or UI edit breaks progress/retry/downloads while the smoke tests stay green.

## What already exists (match its style)

`tests/test_ui_app.py` — 9 tests using `streamlit.testing.v1.AppTest.from_file("ui/app.py")`,
driving the app via widget labels and session-state keys, with `unittest.mock.patch`
targeting function-local import sites (e.g. `reconstruction.ollama_client.probe_model`).
Read it in full before writing anything; reuse its fixtures/patterns. Note its key
technique: patches must target where names are *looked up at call time*.

## What to build

Extend `tests/test_ui_app.py` (or a new `tests/test_ui_run_loop.py` if the file gets
unwieldy — your call, state the reason in the PR):

### A. `run_one_file` unit tests (no AppTest needed — it's a plain function)

Mock `chorus.run_pipeline` (check the actual import site in `ui/pipeline_invocation.py`
first) and verify:
1. Success path: returns the pipeline result, forwards config (model, device,
   reconstruction flags) from `SidebarConfig` into `run_pipeline`'s kwargs correctly —
   assert on the actual call kwargs, not just that it was called.
2. Failure path: pipeline raising `RuntimeError` is captured per-file (not raised
   through), and the returned/recorded error state contains the message.
3. Progress callback wiring: the `segment_callback`/progress hook (check its real
   name) is passed through and calling it doesn't crash.

### B. `render_run_section` via AppTest

With `run_pipeline` mocked to return a canned result dict (build it from the shapes
in `tests/test_integration.py` — grep for `bundle_path`/`consensus` result keys):
1. Uploading N=2 fake files then clicking the run button renders a per-file status
   panel for both, with no exceptions (`at.exception` empty).
2. One file succeeding + one raising → the failure is rendered (find the error
   element/marker used in the code) and the successful file's results still render —
   partial failure must not abort the batch.
3. Sequential vs all-at-once mode toggle both execute the mocked pipeline the
   expected number of times.

### C. `render_file_results` rendering

With a canned result (including votes with all three tiers, and paths to real temp
files for the download buttons — Streamlit download buttons need real bytes):
1. Download buttons render for each expected artefact (best guess, bundle, AI
   context at minimum) — assert by button label/key.
2. Tier statistics/summary panel shows the right counts.
3. With diarisation data absent → no diarisation table, no exception (degradation).

## Coverage target

`pytest --cov=ui.pipeline_invocation --cov=ui.results --cov-report=term-missing` —
aim to cover the real decision branches; **both modules ≥ 60 %** is the success bar
(from 13 %/12 %). Do not chase 100 % — the remaining tail is Streamlit layout code
with low regression value. Do not weaken existing tests. Known environment quirk: if
`pytest --cov` breaks `import spacy` locally ("cannot load module more than once"),
run coverage scoped to just these test files; full-suite coverage runs fine in CI.

## Verification (success criteria)

1. All new tests pass; existing 9 UI tests pass unmodified; full suite passes.
2. Coverage bar met (state the before/after numbers in the PR body).
3. `black`/`ruff`/`isort` clean; local CI mirror per conventions doc.
4. Tick RB-5 in `ROADMAP.md`. PR title: `test: cover UI run loop and results rendering`.
5. Honesty rule: if you find a real bug in the run loop while testing (plausible —
   this code has never been executed under test), report it in the PR body and STOP
   short of fixing it unless the fix is a one-liner; larger fixes are a separate PR.

## Files to change

- `tests/test_ui_app.py` (and optionally a new `tests/test_ui_run_loop.py`)
- `ROADMAP.md`
- Nothing under `ui/` unless a one-line bug fix, explained in the PR body
