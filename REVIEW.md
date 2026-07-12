# Chorus Engine Holistic Codebase Review

Date: 12 July 2026 (v4.0.0, post-release)

> This supersedes the previous 21 June / 29 June 2026 review. That review's findings
> were routed into the four v4.0.0 work packages, all shipped. This is a fresh,
> ground-up pass against the current tree — see `ROADMAP.md`'s "Completed — v4.0.0"
> section for what shipped, and the action roadmap below for what's next.

## Executive summary

Chorus is in a healthy, well-tested state for its core pipeline (audio → transcription
→ consensus → export), with 244 passing tests and 82% overall coverage. The most
material risk found in this pass is not in the core pipeline but in **dependency
manifest drift**: `pyproject.toml` and `requirements.txt` must be kept in sync by hand,
and they silently drifted — a patched CVE in `requirements.txt` (nltk path traversal)
was never applied to `pyproject.toml`, leaving two open Dependabot alerts undetected
until this review (fix in progress, PR #129). The second-order risk is **coverage
concentration**: the two files with the least test coverage — `ui/hardware_survey.py`
(14%) and `ui/app.py` (36%) — are exactly the files behind the hardware-preset
selector and the main UI, i.e. the surfaces most users touch first. CI is otherwise
solid (secret scanning, pinned actions, scheduled drift checks), but has no code
scanning (CodeQL) and no security policy for coordinated disclosure.

## Architecture

| Module / Path | Responsibility | Concerns |
|---|---|---|
| `chorus/` | Stable public API façade (re-exports) | None — thin, low-risk by design |
| `audio_processor/` | Audio cleaning: high-pass, normalise, denoise | None significant |
| `transcription_engine/` | Whisper wrapper + multi-variant orchestration | None significant |
| `consensus_merger/` | Word-vote alignment (sequence + positional) and Markdown rendering | None significant |
| `diarisation/` | pyannote speaker ID + name persistence | 67% coverage; some untested branches (diariser.py:96-134, 164-194) |
| `export_engine/` | PDF/DOCX/SRT/VTT/ZIP/plain-text/best-guess export | 62% coverage — largest untested surface in the core pipeline (exporter.py:114-184, 218-294) |
| `reconstruction/` | Strategy-based LOW-token reconstruction (nlp/llm) | `nlp.py` at 39% — the actual spaCy grammatical-correction logic is thin on coverage vs. its degradation paths |
| `ui/` | Streamlit web UI + hardware survey | `app.py` is a single 1744-line file at 36% coverage; `hardware_survey.py` at 14% — see Risk Inventory |
| `batch_processor/` | Unattended multi-file CLI | 100% coverage (added WP3) |
| `devops-practices/` | Shell scripts: version sync, survey-ollama-env, clone-ref checks | Manually maintained; no test harness for the shell scripts themselves beyond `version_consistency_test.sh` |

**Data flow:** audio file → `audio_processor` (4 variants) → `transcription_engine`
(Whisper × variants × consensus models) → `consensus_merger` (word-vote alignment) →
optional `reconstruction` (LOW-token cleanup via spaCy or Ollama) → optional
`diarisation` → `export_engine` (all output formats) → filesystem (`output_dir` or
global `CONSENSUS_DIR`). Entry points: `pipeline_runner.py` (CLI/API), `ui/app.py`
(Streamlit), `batch_processor/batch_runner.py` (unattended batch).

## Risk inventory

| # | Category | Finding | Score | File / Location |
|---|---|---|---|---|
| 1 | Security | `pyproject.toml` and `requirements.txt` both declare runtime deps independently with no automated sync check; drifted silently, leaving an open CVE (nltk path traversal) unpatched in one manifest for weeks | 4 | `pyproject.toml`, `requirements.txt` |
| 2 | Security | `pip-audit` in `security.yml` only scans `requirements.txt` — a vulnerable pin that exists *only* in `pyproject.toml` (as just happened) would never be caught by CI at all | 4 | `.github/workflows/security.yml` |
| 3 | Security | No `SECURITY.md`, no private vulnerability reporting enabled, no CodeQL/code scanning configured | 3 | repo root, GitHub Security tab |
| 4 | Maintainability | `ui/app.py` is a single 1744-line file mixing sidebar config, upload handling, pipeline invocation, results rendering, and dialog logic | 3 | `ui/app.py` |
| 5 | Maintainability | `ui/hardware_survey.py` (RAM/CPU/GPU detection + Max/Background preset logic) is at 14% coverage — the exact code behind the one-click preset button documented as "the fastest way to get sensible settings" | 4 | `ui/hardware_survey.py` |
| 6 | Reliability | New GitHub Actions workflow (`ollama-model-tags-check.yml`) has never actually executed (confirmed via `gh run list` — zero runs since creation); its correctness under real CI conditions is unverified | 3 | `.github/workflows/ollama-model-tags-check.yml` |
| 7 | Maintainability | `export_engine/exporter.py` at 62% coverage — largest untested surface in the core (non-UI) pipeline; PDF/DOCX export paths specifically | 3 | `export_engine/exporter.py:114-184,218-294` |
| 8 | Reliability | Documentation (README, docs/DOCKER.md) drifted independently of code multiple times this session with no detection mechanism — stale Docker-compose syntax, stale model names, stale defaults sat unnoticed until manually found | 3 | `README.md`, `docs/DOCKER.md` (now fixed) |
| 9 | Dependency | `streamlit` (1.58.0 pinned, 1.59.1 latest) and `spacy` (3.8.13 pinned, 3.8.14 latest) are one release behind; not urgent, routine maintenance | 1 | `requirements.txt`, `pyproject.toml` |
| 10 | Maintainability | `reconstruction/nlp.py` at 39% coverage — spaCy reconstruction logic itself (not just its degradation path) is thin on direct tests | 3 | `reconstruction/nlp.py:132-288` |

## Predicted failure scenarios

### PF-1: A future dependency CVE fix applied to only one manifest goes undetected (Security, score 4)

**What happens:** Someone bumps a vulnerable package's pin in `requirements.txt` (the
file CI's `pip-audit` actually scans) but not in `pyproject.toml` (or vice versa), and
CI stays green while a real CVE remains exploitable via `pip install -e .`.

**Trigger condition:** Any future dependency security fix — this has already happened
once (nltk, this session) and the two manifests have no automated sync check.

**Estimated timeline:** Will recur at the next CVE fix unless addressed; it is not a
hypothetical, it already occurred.

**Minimum fix:** Add a CI check (or pre-commit hook) asserting every pinned version in
`pyproject.toml`'s `dependencies` matches the corresponding pin in `requirements.txt`.

**Full fix (roadmap item):** Generate `pyproject.toml`'s dependency list from
`requirements.txt` at build/lint time instead of maintaining two hand-written lists.

### PF-2: A user with a genuinely unusual hardware configuration gets a wrong preset recommendation silently (Reliability, score 4)

**What happens:** `ui/hardware_survey.py`'s `detect_hardware()`/`recommend_settings()`
functions have almost no direct test coverage (14%). A logic error in GPU/VRAM
detection or the recommendation thresholds would surface only as "the Max preset
picked a model that OOMs" — a bad user experience with no test to have caught it
first.

**Trigger condition:** Any hardware configuration not resembling the developer's own
test machine (e.g., unusual VRAM reporting, multi-GPU systems, non-standard `nvidia-smi`
output parsing).

**Estimated timeline:** Latent now; will surface as user-reported bugs, not CI
failures, since there's no test harness exercising the actual detection logic against
varied simulated hardware profiles.

**Minimum fix:** Add unit tests for `detect_hardware()` and `recommend_settings()`
mocking `nvidia-smi`/`system_profiler` output across a few representative hardware
profiles (low-RAM CPU-only, mid-range NVIDIA, Apple Silicon, high-VRAM NVIDIA).

**Full fix (roadmap item):** Extend to property-based testing across a wider input
matrix, given this function's output directly drives what model runs on a user's
machine.

### PF-3: `ui/app.py`'s 1744-line single-file structure makes future changes increasingly risky (Maintainability, score 3)

**What happens:** As features accumulate, the lack of separation between sidebar
config, upload handling, pipeline invocation, and results rendering makes it harder to
reason about side effects of a change — a change to one control's logic risks breaking
an unrelated one via shared `st.session_state` keys.

**Trigger condition:** Continued feature growth in the UI (already grown substantially
across v3.1-v4.0).

**Estimated timeline:** Not an immediate failure risk, but the maintenance cost is
already visible — most new UI logic this session (survey preset, LLM/NLP setup
dialogs) added to the same file rather than a decomposed module.

**Minimum fix:** None required immediately — flagging for awareness.

**Full fix (roadmap item):** Split `ui/app.py` into `ui/sidebar.py`, `ui/upload.py`,
`ui/results.py` modules called from a thin `ui/app.py` entry point, once the file
exceeds ~2000 lines or the next major UI feature is added.

## Test coverage gaps

| Path | Why critical | Test type needed |
|---|---|---|
| `ui/hardware_survey.py::detect_hardware/recommend_settings` | Drives the one-click preset button's model/device/parallelism choice for every user who clicks it | Unit: mocked hardware profiles across RAM/GPU tiers (see PF-2) |
| `ui/app.py` (lines 470-1726, most of the file) | Main UI; only render-smoke and dialog-trigger paths are covered (RA-3.2, this session) | Integration: `AppTest` coverage of results rendering, download buttons, batch progress |
| `export_engine/exporter.py::export_pdf/export_docx` (lines 114-184, 218-294) | User-facing export formats with zero direct test evidence | Unit: fixture-based export + structural validation of output files |
| `reconstruction/nlp.py` (lines 132-288) | The actual spaCy correction logic, as opposed to its already-tested degradation path | Unit: known LOW-token + context → expected correction assertions |

## Dependency audit

- `pip-audit -r requirements.txt`: clean (0 known vulnerabilities) as of this review.
- `pyproject.toml` had a real, undetected drift (see Risk #1) — now fixed in PR #129.
- All GitHub Actions in all 4 workflow files are version-pinned (no `@main`/`@master`
  floating refs) — good practice already in place.
- Minor staleness, not urgent: `streamlit==1.58.0` (latest 1.59.1), `spacy==3.8.13`
  (latest 3.8.14). No security releases missed; routine bump candidates.
- No abandoned dependencies identified (all actively maintained, recent releases).

## CI/CD gaps

| Check | Status |
|---|---|
| Tests run on every PR | ✅ `ci.yml` |
| Lint + format check | ✅ `ci.yml` (black, ruff, isort) |
| Secret scanning | ✅ `security.yml` (gitleaks + TruffleHog + detect-secrets, 3-layer) |
| Actions pinned to versions | ✅ confirmed across all 4 workflows |
| Scheduled dependency-drift detection | ✅ `ci.yml` weekly cron + `ollama-model-tags-check.yml` weekly cron |
| Code scanning (CodeQL / SAST) | ❌ not configured (`code-scanning/default-setup` reports `not-configured`) |
| `pip-audit` covers all dependency manifests | ❌ only scans `requirements.txt`, not `pyproject.toml` (Risk #2) |
| Security policy / private vulnerability reporting | ❌ no `SECURITY.md`; private vulnerability reporting disabled on the GitHub repo |
| Dependabot security updates (auto-PR on CVE) | ❌ disabled (routine version-update `dependabot.yml` is separate and already active) |

**Suggested CodeQL addition** (`.github/workflows/codeql.yml`):
```yaml
name: CodeQL
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 10 * * 3"
jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v7
      - uses: github/codeql-action/init@v3
        with:
          languages: python
      - uses: github/codeql-action/analyze@v3
```

## Action roadmap

### RA-1: Prevent pyproject.toml / requirements.txt drift

**Context:** The two dependency manifests are hand-maintained and already drifted
once, leaving a CVE open undetected (Risk #1, PF-1).

**Success criteria:** A CI check fails if any package version differs between the two
files; the check runs on every PR that touches either file.

**Files to change:** `.github/workflows/ci.yml` (new step), possibly a small Python
script under `devops-practices/`.

**Estimated effort:** S

### RA-2: Make pip-audit cover pyproject.toml's dependency list

**Context:** `security.yml`'s `pip-audit` step only scans `requirements.txt`; a
vulnerable pin unique to `pyproject.toml` is invisible to CI (Risk #2).

**Success criteria:** `pip-audit` (or an equivalent check) runs against the installed
package set from `pip install -e .`, not just `requirements.txt`.

**Files to change:** `.github/workflows/security.yml`

**Estimated effort:** S

### RA-3: Add SECURITY.md and enable private vulnerability reporting

**Context:** No coordinated-disclosure path exists for this public repo (Risk #3).

**Success criteria:** `SECURITY.md` exists with a reporting contact/process; private
vulnerability reporting is enabled in the GitHub repo settings.

**Files to change:** new `SECURITY.md`; GitHub repo settings (not code)

**Estimated effort:** XS

### RA-4: Add CodeQL scanning

**Context:** No SAST/code-scanning tool is configured (Risk #3, CI/CD gaps).

**Success criteria:** `.github/workflows/codeql.yml` runs on PR + weekly schedule and
reports to the Security tab.

**Files to change:** new `.github/workflows/codeql.yml`

**Estimated effort:** XS

### RA-5: Test hardware_survey.py's detection and recommendation logic

**Context:** 14% coverage on the code directly behind the one-click hardware preset
button (Risk #5, PF-2).

**Success criteria:** Unit tests cover `detect_hardware()` and
`recommend_settings()`/`recommend_settings_background()` across at least 4 mocked
hardware profiles (low-RAM CPU, mid NVIDIA, Apple Silicon, high-VRAM NVIDIA), asserting
the correct model/device/parallelism recommendation for each.

**Files to change:** new `tests/test_hardware_survey.py`

**Estimated effort:** M

### RA-6: Verify ollama-model-tags-check.yml actually works under real CI

**Context:** This workflow has zero recorded runs since creation; its `workflow_dispatch`
trigger and issue-filing logic are unverified in a real GitHub Actions environment
(Risk #6).

**Success criteria:** Manually trigger via `gh workflow run ollama-model-tags-check.yml`
and confirm it completes successfully end-to-end (including the no-op "all tags valid"
path).

**Files to change:** none expected unless a bug is found

**Estimated effort:** XS

### RA-7: Expand export_engine/exporter.py test coverage

**Context:** 62% coverage; PDF/DOCX export paths have no direct test evidence
(Risk #7).

**Success criteria:** Each export format (PDF, DOCX) has at least one test asserting
the output file is created and structurally valid (not just "doesn't crash").

**Files to change:** `tests/test_exporter.py`

**Estimated effort:** M

### RA-8: Expand reconstruction/nlp.py test coverage beyond degradation paths

**Context:** 39% coverage; existing tests cover graceful degradation (spaCy missing)
but not the actual grammatical-correction logic (Risk #10).

**Success criteria:** Tests cover known LOW-confidence-token + context inputs and
assert the expected corrected output.

**Files to change:** `tests/test_reconstructor.py`

**Estimated effort:** S

### RA-9 (lower priority): Decompose ui/app.py

**Context:** Single 1744-line file mixing multiple concerns (PF-3).

**Success criteria:** Sidebar config, upload/run, and results rendering split into
separate modules with a thin `ui/app.py` orchestrating them; existing `test_ui_app.py`
suite still passes unchanged.

**Files to change:** `ui/app.py`, new `ui/sidebar.py` / `ui/results.py` (naming TBD)

**Estimated effort:** L
