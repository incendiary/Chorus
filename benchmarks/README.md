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
