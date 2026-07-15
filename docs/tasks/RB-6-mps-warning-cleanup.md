# RB-6 (optional): Silence redundant MPS float64 warning

**Model tier:** Haiku · **Effort:** XS · **Branch:** `fix/rb6-mps-warning`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first. This task is OPTIONAL — lowest
> priority in the wrap-up push; do it last or not at all.

## Context

On Apple Silicon, Whisper's word-timestamp DTW alignment needs float64, which MPS
does not support. `transcription_engine/whisper_engine.py` correctly catches the
`TypeError` and retries the affected pass on CPU (shipped in v3.1.1, "MPS float64 CPU
fallback" in `ROADMAP.md`). But each occurrence emits a `UserWarning`/log warning,
so a normal multi-pass run on a Mac prints the same warning repeatedly — noise that
trains the maintainer to ignore warnings. This is the last unchecked legacy roadmap
item ("Suppress or optimise MPS float64 warnings", `ROADMAP.md` ~line 114).

## The fix

1. Read `transcription_engine/whisper_engine.py` and find the fallback path and
   exactly what it emits (grep for `warn`, `float64`, `MPS`).
2. Demote the per-pass message to `logger.info`, and emit the user-facing warning
   **once per process**: module-level `_mps_float64_warned = False` flag (or
   `functools.lru_cache`-guarded helper) so the first fallback logs a clear
   `logger.warning("MPS does not support float64 word-timestamp alignment; affected "
   "passes will retry on CPU. This is expected on Apple Silicon.")` and subsequent
   ones stay at info level.
3. Do NOT change the fallback behaviour itself, and do NOT suppress warnings
   globally (`warnings.filterwarnings` on everything is forbidden). If the warning
   originates inside Whisper itself rather than our code, wrap only the specific
   call with `warnings.catch_warnings()` scoped to that exact category/message.

## Tests

Extend `tests/test_whisper_engine.py` (currently 100 % — keep it there):
1. Simulated MPS float64 `TypeError` on two consecutive passes → `caplog` shows
   exactly one WARNING-level record and the rest at INFO.
2. Existing fallback-behaviour tests still pass unmodified.

## Verification (success criteria)

1. Tests as above; full suite passes; lint clean; local CI mirror per conventions.
2. In `ROADMAP.md`: tick this RB-6 item AND the legacy "Suppress or optimise MPS
   float64 warnings" item (~line 114), cross-referencing each other.
3. PR title: `fix: warn once for MPS float64 CPU fallback`.

## Files to change

- `transcription_engine/whisper_engine.py`
- `tests/test_whisper_engine.py`
- `ROADMAP.md`
