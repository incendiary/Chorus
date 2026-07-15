# RB-2: WER + confidence-calibration benchmark

**Model tier:** Sonnet · **Effort:** L · **Branch:** `feat/rb2-wer-benchmark`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first — it covers local CI validation
> (shift-left), known false-positive patterns, and the PR flow.

## Why this task exists (read carefully — this is the project's exam)

Chorus's founding claim is that transcribing four audio variants (original, high-pass,
normalised, denoised) and taking a word-level consensus vote produces a *better*
transcript than a single Whisper pass — and that the HIGH/MEDIUM/LOW confidence tiers
tell a reader *where* the transcript can be trusted. Neither claim has ever been
measured. This task builds the measurement. **A negative result is an acceptable
outcome** — report whatever the numbers say; do not tune the benchmark until it
flatters the pipeline. The maintainer has explicitly said "prove it works, then stop",
and an honest "consensus ties with single-pass but the tiers are well-calibrated" is a
publishable conclusion.

## What to build

A `benchmarks/` directory (new, top-level) containing:

```
benchmarks/
  __init__.py          (empty)
  run_benchmark.py     (CLI entry point)
  data/                (gitignored — downloaded/generated audio lives here)
  RESULTS.md           (generated output, committed)
  README.md            (how to run, ~20 lines)
```

Add `benchmarks/data/` to `.gitignore`.

### Reference data

Use **LibriSpeech test-clean** (public domain, standard ASR benchmark, known
transcripts). Full archive: `https://www.openslr.org/resources/12/test-clean.tar.gz`
(~346 MB, one-time download, cache in `benchmarks/data/`). From the extracted set,
select the **first 15 utterances longer than 10 seconds** (deterministic selection —
sort by file path, no randomness) so the run is reproducible and completes in minutes
on the maintainer's Apple Silicon machine with the `base` Whisper model.

LibriSpeech layout gotchas: audio is FLAC (`soundfile` reads it natively — already a
dependency); transcripts are in `*.trans.txt` files, one line per utterance:
`<UTTERANCE-ID> <TEXT IN UPPER CASE>`.

### Conditions (two arms of audio quality)

1. **clean** — the LibriSpeech files as-is.
2. **noisy** — the same files with additive Gaussian noise at **SNR 5 dB**, generated
   deterministically (`numpy.random.default_rng(42)`), written as WAV into
   `benchmarks/data/noisy/`. SNR formula: scale noise so
   `10*log10(signal_power/noise_power) == 5`. The consensus architecture's whole
   premise is robustness on imperfect audio, so the noisy condition is the one that
   matters most.

### Pipelines (two arms of transcription)

1. **single** — one Whisper pass on the file, using
   `transcription_engine.whisper_engine` directly, `base` model. This is the baseline:
   what the user would get without Chorus.
2. **chorus** — the full consensus pipeline via `chorus.run_pipeline(path,
   output_dir=tmp)` with reconstruction and diarisation **disabled** (measure the core
   consensus mechanism, not the add-ons). Read the resulting `{stem}_bundle.json`:
   the transcript is `" ".join(w["word"] for w in bundle["consensus"])`, and the
   per-word tiers come from the same array.

Both arms MUST use the same Whisper model size (`base`) — check how
`run_pipeline`/config select the model and pin it explicitly so the comparison is fair.

### Metric 1 — Word Error Rate

Add `jiwer` to the dev extras in `pyproject.toml` (`[project.optional-dependencies]`,
dev list) — do NOT add it to runtime `requirements.txt`; it is benchmark-only. Apply
identical normalisation to reference and hypothesis before scoring:
lowercase → strip punctuation → collapse whitespace (use a
`jiwer.Compose([...])` transform; write it once, use it for every arm). Report WER per
(condition × pipeline) cell, averaged over the 15 utterances, plus per-file rows.

### Metric 2 — Confidence-tier calibration (chorus arm only)

For each word in the consensus array, determine whether it is **correct** — i.e.
appears in the aligned position of the reference. Use `jiwer`'s alignment output
(`jiwer.process_words(...).alignments`) to map hypothesis words to
correct/substitution/insertion, rather than inventing your own aligner. Then report,
per tier: word count, and precision (fraction correct). The claim under test:
`precision(HIGH) > precision(MEDIUM) > precision(LOW)`, with `precision(HIGH)` high in
absolute terms (≥ 0.95 would strongly validate the tier guidance in
`docs/CHORUS_FOR_LLMS.md` §3).

### Sanity gates (build these FIRST, as pytest tests in `tests/test_benchmark.py`)

These protect against a benchmark that is "wrong but green":

1. Feeding the normalised reference text as the hypothesis → WER == 0.0.
2. Feeding a completely different text → WER ≥ 0.9.
3. The SNR helper: generated noisy signal has measured SNR within ±0.5 dB of target.
4. Calibration counting: a hand-built 5-word case with known alignment produces the
   expected per-tier precision numbers.

These tests must NOT download LibriSpeech or load Whisper — pure unit tests on the
helper functions. Structure `run_benchmark.py` so the helpers (normalise, snr_mix,
wer, calibration_table) are importable functions, with the download/transcribe
orchestration behind `if __name__ == "__main__":` / a `main()` function.

### Output

`run_benchmark.py` writes `benchmarks/RESULTS.md`:

- date, whisper model, N files, total audio duration, machine (`platform.platform()`)
- WER table: rows = condition (clean/noisy), columns = pipeline (single/chorus), plus
  relative delta column
- calibration table: tier × (count, precision), per condition
- a short "Interpretation" section: 3–5 sentences stating plainly whether (a)
  consensus beat single-pass on noisy audio, (b) tiers are monotonically calibrated.
  State the numbers; do not editorialise beyond them.

## Verification (success criteria)

1. `pytest tests/test_benchmark.py` — all sanity gates pass.
2. Full suite still passes (`source .venv/bin/activate 2>/dev/null; python3 -m pytest -q`).
3. `python3 -m benchmarks.run_benchmark --limit 2` (add a `--limit` flag for a smoke
   run) completes end-to-end on this machine and writes RESULTS.md.
4. Run the full 15-file benchmark once; commit the generated `RESULTS.md`.
5. `black`, `ruff`, `isort` clean.

## Files to change

- `benchmarks/` (new), `tests/test_benchmark.py` (new)
- `pyproject.toml` (dev extras: `jiwer`), `.gitignore` (`benchmarks/data/`)
- `ROADMAP.md` — tick RB-2 with a one-line summary of the actual result

## Repo conventions & environment

- Branch → PR (`feat: add WER and confidence-calibration benchmark`) → leave open,
  do NOT merge.
- Squash-merge repo; never commit to main. British English, Oxford comma.
- Broken pyenv shim: always `source .venv/bin/activate 2>/dev/null; python3 …`.
- Whisper `base` model may need a one-time download on first run (~150 MB) — that is
  expected and fine.
- Do NOT loosen or skip the sanity-gate tests to make things pass. If a gate fails,
  the helper is wrong — fix the helper.
- If the full benchmark reveals a genuine bug in the pipeline itself (e.g. consensus
  crashes on some input), STOP, report the bug clearly in your final summary, and do
  not attempt drive-by fixes outside benchmark code.
