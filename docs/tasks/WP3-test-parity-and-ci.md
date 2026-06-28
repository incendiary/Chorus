# WP3 — User-facing test parity & CI hardening

**Target release:** v4.0.0 (independently shippable as 3.4.x)
**Breaking:** No
**Depends on:** WP1 — write tests against the post-WP1 import paths (the `chorus`
public API and the consolidated `reconstruction` package). If WP1 has not merged yet,
coordinate, or write tests against current paths and update imports when WP1 lands.
**Branch:** `feat/v4-wp3-test-parity`
**Overall effort:** M

Read `docs/tasks/README.md` ("Conventions every agent MUST follow") before starting.

## Why this exists

`REVIEW.md` measured 74 % overall coverage with the two **user-facing orchestration
surfaces at ~0 %**:

- `ui/app.py` — the Streamlit app, the primary way users run Chorus.
- `batch_processor/batch_runner.py` — the unattended batch CLI.

Scaling to larger unattended workloads (a stated project goal) is unsafe while the
batch path is untested. Separately, CI's dependency audit is non-blocking.

---

### RA-3.1: Add test coverage for the batch processor

**Context:** `batch_processor/batch_runner.py` has `run_batch(...)` and a CLI with
`--output-dir / -o`. No test imports it (confirmed: `grep -rn "run_batch" tests/`
returns nothing). This is the higher-priority of the two surfaces because it runs
unattended.

**What to do:**
- Create `tests/test_batch_runner.py`. Mock the per-file pipeline call (patch
  `run_pipeline` / the symbol `batch_runner` imports) so no real transcription occurs —
  follow the mocking style in `tests/test_integration.py`.
- Cover at minimum:
  - A directory of multiple audio files processes each one.
  - Each file writes to its **isolated** `<output_dir>/<stem>/` subdirectory (the
    batch isolation feature from v3.1.0).
  - One file failing does not abort the whole batch; the run reports per-file
    success/failure and exits with the correct status.
  - Empty directory / no matching files is handled gracefully (no crash, clear message).

**Success criteria:**
- `batch_processor/batch_runner.py` coverage rises from ~0 % to ≥ 70 %
  (`pytest --cov=batch_processor`).
- All new tests green; suite total still passes.

**Files to change:** new `tests/test_batch_runner.py`.
**Effort:** M

---

### RA-3.2: Add smoke/behaviour tests for the Streamlit UI

**Context:** `ui/app.py` is at 0 %. Streamlit ships `streamlit.testing.v1.AppTest`
for headless testing without a browser.

**What to do:**
- Create `tests/test_ui_app.py` using `AppTest.from_file("ui/app.py")`.
- Mock the pipeline entry so `AppTest` never runs real transcription.
- Cover the high-value paths:
  - App renders without exception (`at.run(); assert not at.exception`).
  - Empty/cancelled upload is handled (no crash) — see the CLAUDE.md unit-testing
    standard requiring this.
  - Sidebar controls added in v3.3.0 exist and forward correctly: model-size selector
    (incl. `large`), device selector, parallelism toggle/worker count, and the
    Max/Background preset dropdown.
  - The Ollama-unreachable path disables the LLM toggle / shows the warning dialog
    (mock `probe_model` to fail).
- If `streamlit.testing` is not already available in `.venv`, it ships with the pinned
  `streamlit` — confirm with `python -c "import streamlit.testing.v1"`. Do not add a
  new dependency for this.

**Success criteria:**
- `ui/app.py` coverage rises from 0 % to a meaningful level (≥ 40 % is realistic for a
  Streamlit script; aim higher on the control-wiring logic).
- Tests are deterministic and do not hit the network or load real models.

**Files to change:** new `tests/test_ui_app.py`.
**Effort:** M

---

### RA-3.3: Make `pip-audit` blocking in CI

**Context:** `.github/workflows/ci.yml:120` runs
`pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969 || true`. The `|| true`
swallows all findings, so a newly disclosed CVE in a pinned dependency will never fail
the build. (`REVIEW.md` flags this.)

**What to do:**
- Remove the `|| true` so `pip-audit` failures fail the job.
- Keep the existing documented `--ignore-vuln PYSEC-2022-42969` exception (it is an
  audited, accepted exception). If additional currently-failing vulns surface once the
  gate is live, add them as explicit `--ignore-vuln <ID>` entries **with a one-line
  comment each** stating why they are accepted — do not re-add a blanket `|| true`.
- Run `pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969` locally first to
  see what the gate will actually catch, and resolve or document each.

**Success criteria:**
- A simulated vulnerable pin would fail CI (reason it through in the PR description; do
  not actually introduce one).
- The job passes on the current `requirements.txt` with only documented, commented
  exceptions.

**Files to change:** `.github/workflows/ci.yml`. Note the change under the ROADMAP
v4.0.0 section.
**Effort:** S
