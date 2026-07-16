# RB-2: WER + confidence-calibration benchmark results

- **Date**: 2026-07-16
- **Whisper model**: base
- **Files**: 15
- **Total audio duration**: 204.8 s
- **Machine**: macOS-26.5.1-arm64-arm-64bit-Mach-O

## Word error rate

| Condition | Single-pass WER | Chorus WER | Relative delta |
|---|---|---|---|
| clean | 0.0314 | 0.0288 | -8.2% |
| noisy | 0.1024 | 0.1107 | +8.2% |

## Per-file WER

| Utterance | Condition | Single-pass WER | Chorus WER |
|---|---|---|---|
| 1089-134686-0000 | clean | 0.0714 | 0.0714 |
| 1089-134686-0000 | noisy | 0.1071 | 0.1071 |
| 1089-134686-0006 | clean | 0.1250 | 0.1250 |
| 1089-134686-0006 | noisy | 0.1667 | 0.1667 |
| 1089-134686-0009 | clean | 0.0370 | 0.0370 |
| 1089-134686-0009 | noisy | 0.2593 | 0.3333 |
| 1089-134686-0011 | clean | 0.0000 | 0.0000 |
| 1089-134686-0011 | noisy | 0.0513 | 0.0513 |
| 1089-134686-0012 | clean | 0.0000 | 0.0000 |
| 1089-134686-0012 | noisy | 0.2500 | 0.2500 |
| 1089-134686-0018 | clean | 0.0000 | 0.0000 |
| 1089-134686-0018 | noisy | 0.0244 | 0.0244 |
| 1089-134686-0019 | clean | 0.0769 | 0.0769 |
| 1089-134686-0019 | noisy | 0.1026 | 0.1538 |
| 1089-134686-0020 | clean | 0.0000 | 0.0000 |
| 1089-134686-0020 | noisy | 0.0800 | 0.0800 |
| 1089-134686-0022 | clean | 0.0000 | 0.0000 |
| 1089-134686-0022 | noisy | 0.1562 | 0.1562 |
| 1089-134686-0023 | clean | 0.0000 | 0.0000 |
| 1089-134686-0023 | noisy | 0.0000 | 0.0000 |
| 1089-134686-0024 | clean | 0.0333 | 0.0333 |
| 1089-134686-0024 | noisy | 0.0333 | 0.0333 |
| 1089-134691-0002 | clean | 0.0000 | 0.0000 |
| 1089-134691-0002 | noisy | 0.1143 | 0.1143 |
| 1089-134691-0008 | clean | 0.0476 | 0.0476 |
| 1089-134691-0008 | noisy | 0.0714 | 0.0714 |
| 1089-134691-0009 | clean | 0.0408 | 0.0408 |
| 1089-134691-0009 | noisy | 0.0612 | 0.0612 |
| 1089-134691-0011 | clean | 0.0385 | 0.0000 |
| 1089-134691-0011 | noisy | 0.0577 | 0.0577 |

## Confidence-tier calibration (chorus arm)

### Clean

| Tier | Count | Precision |
|---|---|---|
| HIGH | 551 | 0.9782 |
| MEDIUM | 2 | 1.0000 |
| LOW | 0 | nan |

### Noisy

| Tier | Count | Precision |
|---|---|---|
| HIGH | 545 | 0.9266 |
| MEDIUM | 7 | 0.0000 |
| LOW | 3 | 0.0000 |

## Interpretation

On noisy audio (SNR 5.0 dB), single-pass WER was 0.1024 versus chorus consensus WER of 0.1107, so consensus did not beat single-pass Whisper on the condition the architecture is meant to help most. On clean audio, single-pass WER was 0.0314 versus chorus WER of 0.0288. Tier precision on noisy audio was HIGH=0.9266, MEDIUM=0.0000, LOW=0.0000, which is not monotonically calibrated (HIGH > MEDIUM > LOW).
