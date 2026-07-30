# Chorus benchmarks

Measures the project's two founding claims against LibriSpeech test-clean:

1. **WER** — does four-variant consensus beat single-pass Whisper (`base`),
   on clean audio and on noise-augmented audio (Gaussian noise, SNR 5 dB)?
2. **Calibration** — do the HIGH/MEDIUM/LOW confidence tiers predict word
   correctness (precision monotonically decreasing across tiers)?

## Running

```bash
source .venv/bin/activate
pip install -e ".[dev]"           # provides jiwer
python3 -m benchmarks.run_benchmark            # full 15-file run
python3 -m benchmarks.run_benchmark --limit 2  # quick smoke run
```

The first run downloads LibriSpeech test-clean (~346 MB) into
`benchmarks/data/` (gitignored) and caches it for subsequent runs. File
selection is deterministic: the first 15 utterances longer than 10 seconds,
sorted by path. Noise is generated with a fixed seed, so the noisy condition
is reproducible.

Results are written to `benchmarks/RESULTS.md` (committed). The sanity gates
for the helper functions live in `tests/test_benchmark.py` and run without
downloading data or loading Whisper.

## Representativeness — read before trusting these numbers

**This benchmark does not exercise the conditions Chorus was built for, and it
cannot detect the class of defect that has caused every real-world failure so
far.**

LibriSpeech test-clean is short (10–20 s), studio-recorded read speech. Chorus
was built for long, noisy, multi-speaker recordings. The gap shows up in the
one measurement that governs whether consensus has anything to work with —
how much the four audio variants actually disagree:

| Corpus | Mean pairwise variant WER vs `original` |
|---|---|
| This benchmark, noisy arm (SNR 5 dB) | **0.044** |
| A real 28.8-minute phone recording | **0.474** |

At 0.044 the four variants are near-identical, so alignment sees almost no
insertions and consensus has nothing to arbitrate — which is why the tables
above show ~550 HIGH words and 0–3 LOW. Two defects that made real transcripts
unusable (RC-10, forced word timestamps collapsing long-form audio into
repetition loops; RC-11, the multi-alignment merge displacing variants) moved
this benchmark's noisy WER by 0.0012 and its HIGH count by one word, while on
the real recording RC-11 alone moved HIGH from 4.6 % to 43.6 %.

Treat these results as a **regression gate on clean short-form audio**, not as
evidence about long-form or noisy performance. A representative benchmark
needs long-form, genuinely degraded, multi-speaker audio with ground-truth
transcripts; sourcing one is open work.
