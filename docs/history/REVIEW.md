# Chorus Engine — Holistic Review (28 July 2026)

> Third full review, at v4.1.0 with the background-run feature merged (WP0–WP3,
> WP-OUT). Supersedes the 15 July review, whose six action items (RB-1–RB-6) all
> shipped. Run **read-only** while a nine-file production batch was executing, so no
> test suite was run and no CPU-heavy analysis performed; findings come from source
> reading, live process inspection, and the running job's own artefacts.
>
> Spec unchanged from 15 July: Chorus is a **personal working tool** and a **learning
> vehicle**; "done" means **prove the core idea works, then wind down to maintenance**.

---

## Executive summary

The engineering position is strong — 412 tests, all CI gates green, dependencies fully
pinned, and the background-run feature working correctly in live production use (four
parallel workers, disk-backed logs, refresh-survivable state). The two most valuable
findings this round both came from *running the software*, not from reading it:
**14 GB of intermediate WAV files have accumulated with no cleanup path** (RC-1), and
**the sidebar's device and parallelism controls are silently inert** because the
orchestrator captures those settings at import time (RC-2) — the user had to set them
in `.env` to get a correct configuration. A third, cosmetic but corrosive, issue is
that Streamlit floods the console with `ScriptRunContext` warnings from the background
thread, roughly ten lines of noise per real log line (RC-3).

An important correction to the previous review's framing: the test-quality problem is
**not** systemic. A targeted audit found the vacuous assertions were confined to the
two agent-authored files caught during the WP3 salvage (already rewritten); the wider
suite's assertions are substantive. The suite's real weakness is what it *cannot*
reach — thread behaviour, disk growth, and live UI wiring — which is precisely where
every finding below originated.

**Verdict against the agreed bar:** the core idea has been proven (v4.1.0 benchmark:
accuracy on par with single-pass Whisper, but strongly calibrated confidence tiers),
and the tool is now genuinely usable unattended. RC-1 and RC-2 are worth fixing before
wind-down because both actively mislead or degrade real use; the rest is optional
polish.

---

## Architecture

| Module / Path | Responsibility | Concerns |
|---|---|---|
| `audio_processor/` | Three cleaning filters + ingest; ffmpeg fallback for MP4/AAC | Writes variant WAVs to a **global** dir even for isolated runs (RC-1) |
| `transcription_engine/` | Whisper wrapper, model cache, model×variant orchestration | **Captures `WHISPER_DEVICE`/`TRANSCRIPTION_PARALLELISM` at import** (RC-2) |
| `consensus_merger/` | Alignment (Needleman-Wunsch or positional), voting, tier assignment, rendering | Sound; thresholds correctly parameterised since RB/#185 |
| `reconstruction/` | LOW-token reconstruction via spaCy or Ollama | Sound. Correctly refuses out-of-candidate suggestions and demotes LOW→MEDIUM rather than laundering uncertainty |
| `diarisation/` | pyannote speaker separation | Token now read lazily (WP0a); 67 % covered |
| `export_engine/` | Consensus → PDF/DOCX/SRT/VTT/bundle/AI-context/parsing guide | `exporter.py` at 827 lines is the largest module; candidate for splitting |
| `batch_processor/` | Unattended CLI | Untouched by the background-run work, as intended |
| `ui/` | Streamlit app: run manager, worker, state file, status panel, indicator, pages | New `run_*` modules are clean and single-purpose. `results.py` (562) and `sidebar.py` (533) are large but cohesive |
| `.github/workflows/` | ci, security, release, ollama-tag-check | No gaps found |

**Data flow:** upload → spooled to `outputs/runs/<run_id>/` → background thread →
variants to **global** `outputs/variants/` → per-variant transcript JSON → aligned
votes → optional reconstruction/diarisation → renderers/exporters → `outputs/consensus/`
→ state file + `run.log` polled by the UI.

---

## Risk inventory

| # | Category | Finding | Score | Location |
|---|---|---|---|---|
| 1 | Scalability | Intermediate variant WAVs are never deleted. **14 GB / 156 files** already present; ~360 MB per file processed, so tonight's nine-file batch adds ~3 GB. Unbounded. | 4 | `audio_processor/pipeline.py`, `config.VARIANTS_DIR` |
| 2 | Reliability | `orchestrator.py` imports `WHISPER_DEVICE` and `TRANSCRIPTION_PARALLELISM` by value at import, so the sidebar's Compute-device and parallelism controls **cannot** affect them. Verified empirically: setting `config.WHISPER_DEVICE = "cpu"` leaves the orchestrator on `mps` and parallelism at 1. | 4 | `transcription_engine/orchestrator.py:21-28,62-85` |
| 3 | Maintainability | Streamlit emits `missing ScriptRunContext!` for every log record from the background thread — ~10 noise lines per real line, making live console monitoring impractical. (Confirmed *not* to affect `run.log`, which stays clean.) | 3 | `ui/run_worker.py` thread + Streamlit logging |
| 4 | Maintainability | Sidebar still captions the confidence thresholds "Configurable in `config.py`" although they became sliders in #185 — actively misdirects. | 2 | `ui/sidebar.py:504-505` |
| 5 | Maintainability | `export_engine/exporter.py` at 827 lines mixes PDF, DOCX, SRT/VTT, ZIP, bundle, and plain-text export. | 2 | `export_engine/exporter.py` |
| 6 | Maintainability | Two weak assertions survive: an `or "1" in text` clause that is near-tautological, and a redundant case-insensitive `or`. | 1 | `tests/test_ai_context.py:516`, `tests/test_exporter.py:142` |

Null findings worth recording: **Security** — no secrets in tree, GitLeaks/detect-secrets/
TruffleHog/CodeQL/bandit all wired and green, HF token correctly gitignored via `.env`.
**Dependency** — all 16 runtime deps exactly pinned and mirrored between
`requirements.txt` and `pyproject.toml` (enforced by the RA-1 drift check). **CI/CD** —
tests, lint, secrets, SAST, dependency audit, scheduled drift checks, and release
automation all present; no gaps.

---

## Predicted failure scenarios (score ≥ 3)

### PF-1: Disk exhaustion from intermediate WAVs (Scalability, 4)

**What happens:** `outputs/variants/` grows without limit. Every processed file leaves
four 16 kHz WAVs — ~360 MB per hour-ish recording — that are read only during
transcription and never again. At 14 GB after casual testing, sustained batch use fills
the disk; the pipeline then fails mid-run at the variant-write step with an `OSError`,
and because that happens *after* the audio is decoded, the failure wastes the most
expensive work done so far.

**Trigger condition:** routine repeated use. Already 14 GB; 188 GB free, so ~500 more
files before exhaustion — but nothing signals the growth or reclaims it.

**Timeline:** months at current use, immediate if the disk is otherwise near full.

**Minimum fix:** delete a file's variant WAVs in the worker's per-file `finally` block
once its transcripts are on disk (transcripts, not variants, are what
`load_transcripts_from_disk` rehydrates).

**Full fix:** a retention policy — keep variants only for the newest N runs, plus a
"Reclaim space" control on the Past Jobs page reporting reclaimable bytes.

### PF-2: Sidebar device/parallelism settings silently ignored (Reliability, 4)

**What happens:** a user selects "CPU" and a worker count in the sidebar; the
orchestrator ignores both and uses whatever was resolved at import. On Apple Silicon
that means auto-detected `mps` → parallelism pinned to **1**, while every pass then
falls back to CPU anyway because MPS cannot do float64 word-timestamp alignment. Net
effect: the slowest possible configuration, chosen silently, with the UI showing the
opposite. Measured impact: 4× fewer workers.

**Trigger condition:** any use of the sidebar device/parallelism controls. Present
today; only avoided by setting `.env` before launch, which is how the current
production batch was configured.

**Timeline:** now, on every run.

**Minimum fix:** in `orchestrator.py`, `import config` and read `config.WHISPER_DEVICE`
/ `config.TRANSCRIPTION_PARALLELISM` at call time instead of binding the values at
import — the same lazy-read pattern already applied to the HF token in WP0a.

**Full fix:** thread device and parallelism through `run_pipeline(...)` as explicit
parameters, as was done for the consensus thresholds in #185, so the values travel with
the run rather than through module state; then have the sidebar reflect the effective
value back to the user.

### PF-3: Console log unusable for live monitoring (Maintainability, 3)

**What happens:** every log record from the background thread is followed by ~10
`missing ScriptRunContext!` warnings, so a real run's console output is ~90 % noise.
Diagnosing a live problem by watching the terminal is impractical.

**Trigger condition:** every background run — i.e. all runs since WP2.

**Timeline:** now.

**Minimum fix:** attach a `logging.Filter` in `run_worker.execute_run` that drops
records from `streamlit.runtime.scriptrunner_utils.script_run_context`, or set that
logger to `ERROR` for the run's duration.

**Full fix:** as above, plus a console formatter matching `run.log`'s clean layout so
terminal and file agree.

---

## Test coverage gaps

The suite (412 tests) is healthy and its assertions are substantive — the vacuous
patterns found during the WP3 salvage were confined to two agent-authored files and
have been rewritten. The meaningful gaps are things unit tests structurally cannot
reach:

| Path | Why critical | Test type needed |
|---|---|---|
| Disk-space growth across runs | RC-1 is invisible to every existing test because each uses `tmp_path` | Integration: assert the worker removes a file's variants once its transcripts exist |
| `orchestrator._resolve_parallelism` / `_build_device_pool` under runtime config change | RC-2 shipped precisely because no test changes config *after* import | Unit: set `config.WHISPER_DEVICE` post-import, assert the resolved device and worker count follow |
| Whisper inference from a non-main thread | The whole background feature depends on it; verified only by a manual probe during this review | Integration (slow-marked): run `transcribe` on the `tiny` model from a thread |
| `export_engine/exporter.py` PDF/DOCX branches | 827-line module, 90 % covered but the largest single risk surface | Already reasonable; no action |

**Three highest-value additions:** (1) variant-cleanup integration test, (2) runtime
config-change unit test for the orchestrator, (3) threaded-transcription smoke test.

---

## Dependency audit

All 16 runtime dependencies are exactly pinned (`==`) and mirrored between
`requirements.txt` and `pyproject.toml`, with drift enforced in CI. `pip-audit` runs
both scoped and whole-environment (RA-2) and was last green on 14 July after the
setuptools CVE remediation. No abandoned packages; no single-purpose dependency worth
inlining. **Not re-run in this review** to avoid competing with the live batch for CPU —
CI's scheduled weekly audit covers it.

## CI/CD gaps

None. Tests, Black/Ruff/isort, GitLeaks + detect-secrets + TruffleHog, bandit, CodeQL
(default setup), `pip-audit`, dependency-drift check, version/tag consistency,
Dependabot, weekly security and Ollama-tag cron runs, and release automation with the
patch-release skip-cascade fixed (RB-1). Action versions pinned by major tag.

---

## Action roadmap

### RC-1: Reclaim intermediate variant WAVs

**Context:** `outputs/variants/` holds 14 GB across 156 files and never shrinks. The
four WAVs per processed file are inputs to Whisper only; once
`outputs/transcripts/<stem>_<variant>.json` exists they are dead weight
(`load_transcripts_from_disk` reads the JSON, not the WAVs).

**Success criteria:**
- After a run completes, that run's variant WAVs are gone while its transcripts and
  consensus outputs remain.
- An integration test processes a file (mocked transcription) and asserts the variant
  files are absent afterwards but `_bundle.json` and the transcript JSONs are present.
- Deletion failures are logged and never abort the run.
- Behaviour is opt-out via a config flag for anyone wanting to re-run consensus without
  re-processing audio.

**Files to change:** `ui/run_worker.py` (or `pipeline_runner.py` end-of-run),
`config.py` (retention flag), `tests/test_run_manager.py`.

**Estimated effort:** S

### RC-2: Make device and parallelism settings take effect at run time

**Context:** `transcription_engine/orchestrator.py:21-28` binds `WHISPER_DEVICE` and
`TRANSCRIPTION_PARALLELISM` by value at import, so the sidebar controls that set
`config.WHISPER_DEVICE` are inert. Verified: after setting `config.WHISPER_DEVICE =
"cpu"`, the orchestrator still reports `mps` and resolves 1 worker instead of 4.

**Success criteria:**
- Changing `config.WHISPER_DEVICE` after import changes what `_resolve_parallelism`
  and `_build_device_pool` return — asserted by a unit test that mutates config
  post-import (this test must fail against current code).
- Selecting CPU in the sidebar produces "Running transcription in parallel with 4
  workers on cpu" in the log without any `.env` change.
- Existing orchestrator tests pass unmodified.

**Files to change:** `transcription_engine/orchestrator.py`,
`tests/test_orchestrator.py`.

**Estimated effort:** S

### RC-3: Silence Streamlit's ScriptRunContext warnings in background runs

**Context:** Streamlit logs `missing ScriptRunContext!` for every log record emitted
from the worker thread, roughly ten lines per real line. `run.log` is unaffected; the
console is the casualty.

**Success criteria:**
- A background run's console output contains no `ScriptRunContext` lines.
- Genuine warnings from other loggers still appear.
- `run.log` content is unchanged.

**Files to change:** `ui/run_worker.py`, `tests/test_run_manager.py`.

**Estimated effort:** XS

### RC-4: Correct the stale confidence-threshold caption

**Context:** `ui/sidebar.py:504-505` still reads "Confidence Thresholds — Configurable
in `config.py`" although #185 made them sliders in that same sidebar.

**Success criteria:** the caption points at the sliders and names `config.py` only as
the source of defaults; a test asserts the phrase "Configurable in `config.py`" is
absent.

**Files to change:** `ui/sidebar.py`, `tests/test_ui_app.py`.

**Estimated effort:** XS

### RC-5: Tighten the two remaining weak assertions

**Context:** `tests/test_ai_context.py:516` accepts `or "1" in text`, which nearly any
document satisfies; `tests/test_exporter.py:142`'s second clause subsumes its first.

**Success criteria:** both assert one specific, meaningful condition; both still pass.

**Files to change:** `tests/test_ai_context.py`, `tests/test_exporter.py`.

**Estimated effort:** XS

### RC-6 (optional): Split `export_engine/exporter.py`

**Context:** 827 lines spanning PDF, DOCX, subtitles, ZIP, bundle, and plain text.
Lowest priority; the module is well-tested (90 %) and stable.

**Success criteria:** each format in its own module behind the existing public
functions; all export tests pass unmodified.

**Files to change:** `export_engine/`.

**Estimated effort:** M
