# Add smoke/behaviour tests for the Streamlit UI

**Breaking:** No
**Effort:** M

## Why this exists

`ui/app.py` — the Streamlit app, the primary way users run Chorus — is at 0 % test
coverage. `REVIEW.md` flagged this as one of the two user-facing orchestration
surfaces with no test coverage (the other, `batch_processor/batch_runner.py`, was
addressed in `tests/test_batch_runner.py`). Streamlit ships
`streamlit.testing.v1.AppTest` for headless testing without a browser, so this is
addressable without a browser-driving test framework.

## What to do

- Create `tests/test_ui_app.py` using `AppTest.from_file("ui/app.py")`.
- Mock the pipeline entry so `AppTest` never runs real transcription.
- Cover the high-value paths:
  - App renders without exception (`at.run(); assert not at.exception`).
  - Empty/cancelled upload is handled (no crash) — see the CLAUDE.md unit-testing
    standard requiring this.
  - Sidebar controls exist and forward correctly: model-size selector (incl.
    `large`), device selector, parallelism toggle/worker count, and the
    Max/Background hardware preset selector.
  - The Ollama-unreachable path disables the LLM toggle / shows the warning dialog
    (mock `probe_model` to fail).
- If `streamlit.testing` is not already available in `.venv`, it ships with the
  pinned `streamlit` — confirm with `python -c "import streamlit.testing.v1"`. Do
  not add a new dependency for this.

## Success criteria

- `ui/app.py` coverage rises from 0 % to a meaningful level (≥ 40 % is realistic for
  a Streamlit script; aim higher on the control-wiring logic).
- Tests are deterministic and do not hit the network or load real models.

**Files to change:** new `tests/test_ui_app.py`.
