"""
benchmarks/run_benchmark.py — WER + confidence-calibration benchmark.

Measures two claims about the Chorus consensus pipeline against LibriSpeech
test-clean audio:

  1. Does four-variant consensus beat single-pass Whisper on word error rate
     (WER), on both clean and noise-augmented audio?
  2. Are the HIGH/MEDIUM/LOW confidence tiers calibrated — i.e. does
     precision(HIGH) > precision(MEDIUM) > precision(LOW)?

The helper functions below (``normalise_text``, ``wer``, ``snr_mix``,
``calibration_table``) are pure and importable without downloading any data
or loading Whisper, so they can be exercised directly by
``tests/test_benchmark.py``. Orchestration (download, transcription, report
generation) lives behind ``main()`` / ``if __name__ == "__main__":``.

Usage
-----
    python3 -m benchmarks.run_benchmark              # full 15-file run
    python3 -m benchmarks.run_benchmark --limit 2     # smoke run
"""

from __future__ import annotations

import argparse
import logging
import platform
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("benchmarks.run_benchmark")

BENCHMARKS_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARKS_DIR / "data"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_ARCHIVE = DATA_DIR / "test-clean.tar.gz"
LIBRISPEECH_ROOT = DATA_DIR / "LibriSpeech" / "test-clean"
NOISY_DIR = DATA_DIR / "noisy"

WHISPER_MODEL_NAME = "base"
SNR_DB = 5.0
MIN_DURATION_SECONDS = 10.0
DEFAULT_N_FILES = 15
NOISE_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation and WER scoring
# ─────────────────────────────────────────────────────────────────────────────

# Applied identically to reference and hypothesis text before every WER
# comparison: lowercase → strip punctuation → collapse whitespace.
NORMALISE_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def normalise_text(text: str) -> str:
    """Apply the shared normalisation transform and return flattened text."""
    tokens = NORMALISE_TRANSFORM([text])[0]
    return " ".join(tokens)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between *reference* and *hypothesis*, normalised identically."""
    return jiwer.wer(
        reference,
        hypothesis,
        reference_transform=NORMALISE_TRANSFORM,
        hypothesis_transform=NORMALISE_TRANSFORM,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SNR mixing
# ─────────────────────────────────────────────────────────────────────────────


def snr_mix(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise to *signal* scaled so the resulting SNR is *snr_db*.

    SNR (dB) = 10 * log10(signal_power / noise_power). Noise is drawn from
    *rng* (pass a seeded ``numpy.random.default_rng`` for reproducibility).
    """
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise = rng.standard_normal(signal.shape)
    noise_power = float(np.mean(noise**2))
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (signal.astype(np.float64) + noise * scale).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-tier calibration
# ─────────────────────────────────────────────────────────────────────────────


def calibration_table(
    hyp_words: list[dict[str, Any]], reference_text: str
) -> dict[str, dict[str, float | int]]:
    """
    Compute per-tier (count, precision) for a chorus consensus word sequence.

    Parameters
    ----------
    hyp_words : list[dict]
        Each item has at least ``"word"`` and ``"tier"`` (HIGH/MEDIUM/LOW).
    reference_text : str
        The ground-truth reference text (any casing/punctuation; normalised
        internally with the same transform used for WER).

    Returns
    -------
    dict
        ``{tier: {"count": int, "precision": float}}``. A hypothesis word is
        "correct" if jiwer's alignment marks it ``equal`` against the
        reference; ``substitute`` and ``insert`` words count as incorrect.
        Reference words with no hypothesis counterpart (``delete``) have no
        tier and are excluded.
    """
    kept_tokens: list[str] = []
    kept_tiers: list[str] = []
    for w in hyp_words:
        tokens = NORMALISE_TRANSFORM([w["word"]])[0]
        if not tokens:
            continue
        kept_tokens.append(" ".join(tokens))
        kept_tiers.append(w["tier"])

    hyp_text = " ".join(kept_tokens)
    ref_text = normalise_text(reference_text)

    correct = [False] * len(kept_tiers)
    if kept_tokens:
        output = jiwer.process_words(ref_text, hyp_text)
        for chunk in output.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    correct[i] = True

    tallies: dict[str, dict[str, int]] = {}
    for tier, is_correct in zip(kept_tiers, correct, strict=True):
        tally = tallies.setdefault(tier, {"count": 0, "correct": 0})
        tally["count"] += 1
        tally["correct"] += int(is_correct)

    result: dict[str, dict[str, float | int]] = {}
    for tier, tally in tallies.items():
        precision = tally["correct"] / tally["count"] if tally["count"] else 0.0
        result[tier] = {"count": tally["count"], "precision": round(precision, 4)}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LibriSpeech acquisition and selection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Utterance:
    """A single selected LibriSpeech utterance."""

    utterance_id: str
    audio_path: Path
    reference_text: str
    duration_seconds: float = field(default=0.0)


def download_librispeech(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract LibriSpeech test-clean into *data_dir* if absent."""
    root = data_dir / "LibriSpeech" / "test-clean"
    if root.exists():
        logger.info("LibriSpeech test-clean already present at %s", root)
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "test-clean.tar.gz"
    if not archive.exists():
        logger.info(
            "Downloading LibriSpeech test-clean (~346 MB) from %s", LIBRISPEECH_URL
        )
        urllib.request.urlretrieve(LIBRISPEECH_URL, archive)  # noqa: S310

    logger.info("Extracting %s …", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")  # noqa: S202
    return root


def _load_transcripts(trans_file: Path) -> dict[str, str]:
    """Parse a ``*.trans.txt`` file into ``{utterance_id: text}``."""
    mapping: dict[str, str] = {}
    for line in trans_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utterance_id, _, text = line.partition(" ")
        mapping[utterance_id] = text
    return mapping


def select_utterances(
    root: Path,
    n_files: int = DEFAULT_N_FILES,
    min_duration: float = MIN_DURATION_SECONDS,
) -> list[Utterance]:
    """
    Deterministically select the first *n_files* utterances longer than
    *min_duration* seconds, sorted by file path (no randomness).
    """
    flac_paths = sorted(root.rglob("*.flac"))
    selected: list[Utterance] = []
    transcript_cache: dict[Path, dict[str, str]] = {}

    for flac_path in flac_paths:
        if len(selected) >= n_files:
            break
        info = sf.info(str(flac_path))
        duration = info.frames / info.samplerate
        if duration <= min_duration:
            continue

        trans_file = (
            flac_path.parent
            / f"{flac_path.parent.parent.name}-{flac_path.parent.name}.trans.txt"
        )
        if trans_file not in transcript_cache:
            transcript_cache[trans_file] = _load_transcripts(trans_file)
        text = transcript_cache[trans_file].get(flac_path.stem)
        if text is None:
            continue

        selected.append(
            Utterance(
                utterance_id=flac_path.stem,
                audio_path=flac_path,
                reference_text=text,
                duration_seconds=duration,
            )
        )

    return selected


def make_noisy_wav(utterance: Utterance, noisy_dir: Path = NOISY_DIR) -> Path:
    """Write a noise-augmented WAV for *utterance* at ``SNR_DB``, return its path."""
    noisy_dir.mkdir(parents=True, exist_ok=True)
    out_path = noisy_dir / f"{utterance.utterance_id}.wav"
    if out_path.exists():
        return out_path

    signal, sample_rate = sf.read(str(utterance.audio_path), dtype="float32")
    rng = np.random.default_rng(NOISE_SEED)
    noisy = snr_mix(signal, SNR_DB, rng)
    sf.write(str(out_path), noisy, sample_rate)
    return out_path


def make_clean_wav(utterance: Utterance, clean_dir: Path) -> Path:
    """Write the LibriSpeech FLAC as a WAV (Whisper/pipeline expect WAV/FLAC alike)."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / f"{utterance.utterance_id}.wav"
    if out_path.exists():
        return out_path
    signal, sample_rate = sf.read(str(utterance.audio_path), dtype="float32")
    sf.write(str(out_path), signal, sample_rate)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline arms
# ─────────────────────────────────────────────────────────────────────────────


def transcribe_single_pass(audio_path: Path) -> str:
    """Baseline arm: one Whisper ``base`` pass on *audio_path*, via the engine directly."""
    from transcription_engine.whisper_engine import transcribe

    with tempfile.TemporaryDirectory() as tmp:
        result = transcribe(
            audio_path,
            variant_key="single",
            stem=audio_path.stem,
            model_name=WHISPER_MODEL_NAME,
            transcripts_dir=Path(tmp),
        )
    return result.get("text", "")


def transcribe_chorus(audio_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """
    Consensus arm: full pipeline via ``chorus.run_pipeline`` with reconstruction
    and diarisation disabled, pinned to the ``base`` model for every variant.
    """
    import json

    from chorus import run_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        results = run_pipeline(
            audio_path,
            consensus_models=(WHISPER_MODEL_NAME,),
            enable_nlp=False,
            enable_llm=False,
            enable_diarisation=False,
            output_dir=Path(tmp),
        )
        bundle = json.loads(results["bundle_path"].read_text(encoding="utf-8"))

    consensus_words = bundle["consensus"]
    transcript_text = " ".join(w["word"] for w in consensus_words)
    return transcript_text, consensus_words


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_tier_precision(
    per_file_tables: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    """Merge per-file calibration tables into a single count-weighted table."""
    totals: dict[str, dict[str, int]] = {}
    for table in per_file_tables:
        for tier, stats in table.items():
            entry = totals.setdefault(tier, {"count": 0, "correct": 0})
            entry["count"] += stats["count"]
            entry["correct"] += round(stats["precision"] * stats["count"])

    merged: dict[str, dict[str, float | int]] = {}
    for tier, entry in totals.items():
        precision = entry["correct"] / entry["count"] if entry["count"] else 0.0
        merged[tier] = {"count": entry["count"], "precision": round(precision, 4)}
    return merged


def write_results_md(
    results: dict[str, Any],
    output_path: Path = BENCHMARKS_DIR / "RESULTS.md",
) -> Path:
    """Render the benchmark results dict to ``RESULTS.md``."""
    tier_order = ["HIGH", "MEDIUM", "LOW"]
    lines: list[str] = []
    lines.append("# RB-2: WER + confidence-calibration benchmark results")
    lines.append("")
    lines.append(f"- **Date**: {results['date']}")
    lines.append(f"- **Whisper model**: {results['whisper_model']}")
    lines.append(f"- **Files**: {results['n_files']}")
    lines.append(
        f"- **Total audio duration**: {results['total_duration_seconds']:.1f} s"
    )
    lines.append(f"- **Machine**: {results['machine']}")
    lines.append("")

    lines.append("## Word error rate")
    lines.append("")
    lines.append("| Condition | Single-pass WER | Chorus WER | Relative delta |")
    lines.append("|---|---|---|---|")
    for condition in ("clean", "noisy"):
        single_wer = results["wer"][condition]["single"]
        chorus_wer = results["wer"][condition]["chorus"]
        delta = (
            (chorus_wer - single_wer) / single_wer * 100 if single_wer else float("nan")
        )
        lines.append(
            f"| {condition} | {single_wer:.4f} | {chorus_wer:.4f} | {delta:+.1f}% |"
        )
    lines.append("")

    lines.append("## Per-file WER")
    lines.append("")
    lines.append("| Utterance | Condition | Single-pass WER | Chorus WER |")
    lines.append("|---|---|---|---|")
    for row in results["per_file_wer"]:
        lines.append(
            f"| {row['utterance_id']} | {row['condition']} | "
            f"{row['single_wer']:.4f} | {row['chorus_wer']:.4f} |"
        )
    lines.append("")

    lines.append("## Confidence-tier calibration (chorus arm)")
    lines.append("")
    for condition in ("clean", "noisy"):
        lines.append(f"### {condition.capitalize()}")
        lines.append("")
        lines.append("| Tier | Count | Precision |")
        lines.append("|---|---|---|")
        table = results["calibration"][condition]
        for tier in tier_order:
            stats = table.get(tier, {"count": 0, "precision": float("nan")})
            lines.append(f"| {tier} | {stats['count']} | {stats['precision']:.4f} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(results["interpretation"])
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _build_interpretation(results: dict[str, Any]) -> str:
    """Compose the plain-language interpretation paragraph from the numbers."""
    noisy_single = results["wer"]["noisy"]["single"]
    noisy_chorus = results["wer"]["noisy"]["chorus"]
    clean_single = results["wer"]["clean"]["single"]
    clean_chorus = results["wer"]["clean"]["chorus"]

    noisy_beats = noisy_chorus < noisy_single

    noisy_table = results["calibration"]["noisy"]
    high_p = noisy_table.get("HIGH", {}).get("precision", float("nan"))
    med_p = noisy_table.get("MEDIUM", {}).get("precision", float("nan"))
    low_p = noisy_table.get("LOW", {}).get("precision", float("nan"))
    monotonic = high_p > med_p > low_p

    sentences = [
        f"On noisy audio (SNR {SNR_DB} dB), single-pass WER was {noisy_single:.4f} "
        f"versus chorus consensus WER of {noisy_chorus:.4f}, so consensus "
        f"{'beat' if noisy_beats else 'did not beat'} single-pass Whisper on the "
        "condition the architecture is meant to help most.",
        f"On clean audio, single-pass WER was {clean_single:.4f} versus chorus WER of "
        f"{clean_chorus:.4f}.",
        f"Tier precision on noisy audio was HIGH={high_p:.4f}, MEDIUM={med_p:.4f}, "
        f"LOW={low_p:.4f}, which is "
        f"{'monotonically' if monotonic else 'not monotonically'} calibrated "
        "(HIGH > MEDIUM > LOW).",
    ]
    return " ".join(sentences)


def run_benchmark(n_files: int = DEFAULT_N_FILES) -> dict[str, Any]:
    """Run the full benchmark end-to-end and return the results dict."""
    root = download_librispeech()
    utterances = select_utterances(root, n_files=n_files)
    if not utterances:
        raise RuntimeError("No LibriSpeech utterances selected — check dataset layout.")

    clean_dir = DATA_DIR / "clean"
    per_file_wer: list[dict[str, Any]] = []
    per_condition_wer: dict[str, dict[str, list[float]]] = {
        "clean": {"single": [], "chorus": []},
        "noisy": {"single": [], "chorus": []},
    }
    per_condition_calibration: dict[str, list[dict[str, dict[str, float | int]]]] = {
        "clean": [],
        "noisy": [],
    }
    total_duration = 0.0

    for utterance in utterances:
        total_duration += utterance.duration_seconds
        clean_wav = make_clean_wav(utterance, clean_dir)
        noisy_wav = make_noisy_wav(utterance)

        for condition, audio_path in (("clean", clean_wav), ("noisy", noisy_wav)):
            single_text = transcribe_single_pass(audio_path)
            chorus_text, chorus_words = transcribe_chorus(audio_path)

            single_wer = wer(utterance.reference_text, single_text)
            chorus_wer = wer(utterance.reference_text, chorus_text)

            per_condition_wer[condition]["single"].append(single_wer)
            per_condition_wer[condition]["chorus"].append(chorus_wer)
            per_file_wer.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "condition": condition,
                    "single_wer": single_wer,
                    "chorus_wer": chorus_wer,
                }
            )

            table = calibration_table(chorus_words, utterance.reference_text)
            per_condition_calibration[condition].append(table)

            logger.info(
                "%s [%s]: single WER=%.4f, chorus WER=%.4f",
                utterance.utterance_id,
                condition,
                single_wer,
                chorus_wer,
            )

    results: dict[str, Any] = {
        "date": datetime.now(UTC).date().isoformat(),
        "whisper_model": WHISPER_MODEL_NAME,
        "n_files": len(utterances),
        "total_duration_seconds": total_duration,
        "machine": platform.platform(),
        "wer": {
            condition: {
                pipeline: sum(values) / len(values) if values else float("nan")
                for pipeline, values in pipelines.items()
            }
            for condition, pipelines in per_condition_wer.items()
        },
        "per_file_wer": per_file_wer,
        "calibration": {
            condition: _aggregate_tier_precision(tables)
            for condition, tables in per_condition_calibration.items()
        },
    }
    results["interpretation"] = _build_interpretation(results)
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the WER + confidence-calibration benchmark."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N selected utterances, for a smoke run.",
    )
    args = parser.parse_args(argv)

    n_files = args.limit if args.limit is not None else DEFAULT_N_FILES
    results = run_benchmark(n_files=n_files)
    output_path = write_results_md(results)
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
