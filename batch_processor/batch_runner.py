"""
batch_processor/batch_runner.py — Batch and directory processing mode.

Enables unattended processing of multiple audio files in a single invocation.
Supports:
  - A list of explicit file paths
  - A directory glob (e.g., ``/data/recordings/*.mp3``)
  - Recursive directory scanning

Each file is processed sequentially through the full Chorus pipeline
(audio cleaning → transcription → consensus merge → optional export).
A summary report is written to outputs/consensus/batch_report.md upon
completion.

CLI Usage
─────────
    python -m batch_processor.batch_runner /path/to/audio_dir --recursive
    python -m batch_processor.batch_runner file1.mp3 file2.wav --language en
    python -m batch_processor.batch_runner /audio/*.flac --export pdf srt

Programmatic Usage
──────────────────
    from batch_processor.batch_runner import run_batch

    results = run_batch(
        inputs=["/audio/interview.mp3", "/audio/lecture.wav"],
        language="en",
        export_formats=["pdf", "srt"],
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import config
from config import CONSENSUS_DIR, SUPPORTED_AUDIO_EXTENSIONS, ensure_output_dirs

logger = logging.getLogger(__name__)

# Supported audio extensions — single source of truth in config, shared with
# the UI uploader. Kept under its historical name for backwards compatibility.
AUDIO_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parallelism(value: str) -> str:
    normalised = value.strip().lower()
    if normalised == "auto":
        return normalised
    try:
        workers = int(normalised)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a positive integer"
        ) from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return str(workers)


def _configured_models_source() -> str:
    if (
        "CONSENSUS_MODELS" in config.DOTENV_LOADED_KEYS
        or config.config_value_source("CONSENSUS_MODELS") == "process environment"
    ):
        return config.config_value_source("CONSENSUS_MODELS")
    return config.config_value_source("WHISPER_MODEL")


def _resolve_cli_settings(args: argparse.Namespace) -> dict[str, tuple[object, str]]:
    """Resolve effective batch settings and retain their provenance."""
    preset: dict[str, str] | None = None
    if args.hardware_preset:
        from ui.hardware_survey import (
            detect_hardware,
            recommend_settings,
            recommend_settings_background,
        )

        hardware = detect_hardware()
        preset = (
            recommend_settings(hardware)
            if args.hardware_preset == "max"
            else recommend_settings_background(hardware)
        )

    preset_source = f"hardware preset ({args.hardware_preset})"
    if args.consensus_models:
        models = tuple(dict.fromkeys(args.consensus_models))
        models_source = "CLI (--consensus-models)"
    elif args.whisper_model:
        models = (args.whisper_model,)
        models_source = "CLI (--whisper-model)"
    elif preset:
        models = (preset["whisper_model"],)
        models_source = preset_source
    else:
        models = config.CONSENSUS_MODELS
        models_source = _configured_models_source()

    if args.device:
        device = config._detect_device() if args.device == "auto" else args.device
        device_source = (
            "CLI (--device auto; auto-detected)"
            if args.device == "auto"
            else "CLI (--device)"
        )
    elif preset:
        device = preset["device"]
        device_source = preset_source
    else:
        device = config.WHISPER_DEVICE
        device_source = config.config_value_source("WHISPER_DEVICE")
        if device_source == "default":
            device_source = "auto-detected default"

    if args.parallelism:
        parallelism = args.parallelism
        parallelism_source = "CLI (--parallelism)"
    elif preset:
        parallelism = preset["parallelism"]
        parallelism_source = preset_source
    else:
        parallelism = config.TRANSCRIPTION_PARALLELISM
        parallelism_source = config.config_value_source("TRANSCRIPTION_PARALLELISM")

    def resolved(arg_name: str, config_name: str, env_name: str) -> tuple[object, str]:
        cli_value = getattr(args, arg_name)
        if cli_value is not None:
            return cli_value, f"CLI (--{arg_name.replace('_', '-')})"
        return getattr(config, config_name), config.config_value_source(env_name)

    language = (
        (None if args.language == "auto" else args.language, "CLI (--language)")
        if args.language is not None
        else (config.WHISPER_LANGUAGE, config.config_value_source("WHISPER_LANGUAGE"))
    )
    output_dir = (
        (Path(args.output_dir), "CLI (--output-dir)")
        if args.output_dir
        else (None, "default (project outputs)")
    )
    return {
        "models": (models, models_source),
        "device": (device, device_source),
        "parallelism": (parallelism, parallelism_source),
        "language": language,
        "alignment": resolved(
            "alignment_strategy", "ALIGNMENT_STRATEGY", "ALIGNMENT_STRATEGY"
        ),
        "consensus threshold": (
            (
                args.consensus_threshold
                if args.consensus_threshold is not None
                else config.CONSENSUS_THRESHOLD
            ),
            (
                "CLI (--consensus-threshold)"
                if args.consensus_threshold is not None
                else "default"
            ),
        ),
        "similarity threshold": (
            (
                args.similarity_threshold
                if args.similarity_threshold is not None
                else config.SIMILARITY_THRESHOLD
            ),
            (
                "CLI (--similarity-threshold)"
                if args.similarity_threshold is not None
                else "default"
            ),
        ),
        "noise floor": resolved(
            "noise_floor_mode", "NOISE_FLOOR_MODE", "NOISE_FLOOR_MODE"
        ),
        "word timestamps": resolved(
            "word_timestamps", "WORD_TIMESTAMPS", "WORD_TIMESTAMPS"
        ),
        "keep variant WAVs": resolved(
            "keep_variant_wavs", "KEEP_VARIANT_WAVS", "KEEP_VARIANT_WAVS"
        ),
        "Ollama model": resolved("ollama_model", "OLLAMA_MODEL", "OLLAMA_MODEL"),
        "Ollama URL": resolved("ollama_base_url", "OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
        "Ollama timeout": resolved(
            "ollama_timeout", "OLLAMA_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS"
        ),
        "NLP reconstruction": (args.nlp, "CLI flag"),
        "LLM reconstruction": (args.llm, "CLI flag"),
        "speaker diarisation": (args.diarise, "CLI flag"),
        "exports": (
            tuple(args.export or ()),
            "CLI (--export)" if args.export else "default",
        ),
        "recursive discovery": (args.recursive, "CLI flag"),
        "output directory": output_dir,
    }


def _attach_file_logging(
    output_dir: Path | None,
) -> tuple[logging.FileHandler, Path, int]:
    """Attach a per-batch log file, mirroring ``ui/run_worker.py``'s pattern.

    Unlike the Streamlit background-run path, batch runs have no run_id and
    can run for many hours unattended (or inside ``screen``/``tmux``, whose
    own scrollback is finite and easy to lose long before the batch ends).
    Without this, the only record of a run was the console — there was
    nothing on disk to check after the fact.
    """
    log_dir = output_dir if output_dir is not None else CONSENSUS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"batch_{stamp}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s", "%H:%M:%S"
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_root_level = root_logger.level
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)
    return handler, log_path, previous_root_level


def _apply_runtime_settings(settings: dict[str, tuple[object, str]]) -> None:
    """Apply settings consumed dynamically by downstream modules."""
    config.WHISPER_DEVICE = str(settings["device"][0])
    config.TRANSCRIPTION_PARALLELISM = str(settings["parallelism"][0])
    config.NOISE_FLOOR_MODE = str(settings["noise floor"][0])
    config.OLLAMA_BASE_URL = str(settings["Ollama URL"][0])
    config.OLLAMA_TIMEOUT_SECONDS = float(settings["Ollama timeout"][0])


def _display_settings(settings: dict[str, tuple[object, str]]) -> None:
    """Print effective non-secret settings before the first batch item runs."""
    print("\nChorus batch — effective settings")
    for name, (value, source) in settings.items():
        if name == "Ollama URL":
            display_value = "configured endpoint (value hidden)"
        elif isinstance(value, tuple):
            display_value = ", ".join(str(item) for item in value)
        elif value is None or (name == "language" and value == ""):
            display_value = "auto" if name == "language" else "project default"
        elif name == "Ollama timeout":
            display_value = f"{value} s"
        else:
            display_value = str(value)
        print(f"  {name:<22} {display_value:<24} [{source}]")
    print(
        "  precedence             CLI > hardware preset > process environment > .env > default"
    )
    print(
        "  hardware guidance      --hardware-preset max|background (this run only); "
        "devops-practices/survey-ollama-env.sh can update .env after confirmation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


class BatchResult:
    """Holds the outcome of processing a single file in a batch."""

    def __init__(self, path: Path):
        self.path = path
        self.success = False
        self.consensus_path: Path | None = None
        self.export_paths: dict[str, Path | None] = {}
        self.elapsed_seconds = 0.0
        self.error: str | None = None
        self.diarisation_error: str | None = None

    def __repr__(self) -> str:
        if not self.success:
            status = f"FAIL: {self.error}"
        elif self.diarisation_error:
            status = f"OK (diarisation failed: {self.diarisation_error})"
        else:
            status = "OK"
        return f"<BatchResult {self.path.name} [{status}]>"


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_audio_files(
    inputs: list[str | Path],
    recursive: bool = False,
) -> list[Path]:
    """
    Resolve a mixed list of file paths and directory paths to audio files.

    Parameters
    ----------
    inputs : list[str | Path]
        File paths, directory paths, or glob patterns.
    recursive : bool
        If True, directories are scanned recursively.

    Returns
    -------
    list[Path]
        Deduplicated, sorted list of audio file paths.
    """
    found: list[Path] = []

    for item in inputs:
        p = Path(item)

        if p.is_file():
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(p.resolve())
            else:
                logger.warning("Skipping non-audio file: %s", p)

        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in p.glob(pattern):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                    found.append(child.resolve())

        else:
            # Try as a glob pattern
            import glob as glob_mod

            matches = glob_mod.glob(str(p), recursive=recursive)
            for match in matches:
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in AUDIO_EXTENSIONS:
                    found.append(mp.resolve())

    # Deduplicate and sort
    seen = set()
    unique: list[Path] = []
    for f in sorted(found):
        if f not in seen:
            seen.add(f)
            unique.append(f)

    logger.info("Discovered %d audio file(s) for batch processing.", len(unique))
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────


def run_batch(
    inputs: list[str | Path],
    language: str | None = None,
    consensus_models: tuple[str, ...] | None = None,
    export_formats: list[str] | None = None,
    recursive: bool = False,
    alignment_strategy: str | None = None,
    enable_diarisation: bool = False,
    enable_nlp: bool = False,
    enable_llm: bool = False,
    ollama_model: str | None = None,
    output_dir: Path | None = None,
    consensus_threshold: float | None = None,
    similarity_threshold: float | None = None,
    keep_variant_wavs: bool | None = None,
    word_timestamps: bool | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[BatchResult]:
    """
    Process multiple audio files through the full Chorus pipeline.

    Parameters
    ----------
    inputs : list[str | Path]
        File paths, directory paths, or glob patterns.
    language : str, optional
        BCP-47 language code hint for Whisper.
    consensus_models : tuple[str, ...], optional
        Ordered Whisper model names to include in consensus transcription.
    export_formats : list[str], optional
        Subset of ``["pdf", "docx", "srt", "vtt"]``.  If None, no export.
    recursive : bool
        Scan directories recursively.
    alignment_strategy : str, optional
        Consensus alignment strategy: "sequence" or "positional".
    enable_diarisation : bool
        Run speaker diarisation on each file.
    enable_nlp : bool
        Run spaCy NLP reconstruction on LOW-confidence tokens.
    enable_llm : bool
        Run local LLM reconstruction (Ollama) on LOW-confidence tokens.
    ollama_model : str, optional
        Ollama model name for LLM reconstruction.
    output_dir : Path, optional
        Root directory for all batch outputs.  When supplied, each file's
        outputs are written to an isolated ``<output_dir>/<stem>/``
        subdirectory to prevent cross-job collisions.
    consensus_threshold, similarity_threshold : float, optional
        Per-run confidence and fuzzy-match thresholds in the range 0.0–1.0.
    keep_variant_wavs : bool, optional
        Retain intermediate cleaned WAVs. The configured default applies when None.
    word_timestamps : bool, optional
        Enable Whisper word timestamps. The configured default applies when None.
    progress_callback : callable, optional
        Called as ``progress_callback(current_index, total, filename)``
        after each file completes.

    Returns
    -------
    list[BatchResult]
        One result object per discovered audio file.
    """
    from pipeline_runner import run_pipeline

    audio_files = discover_audio_files(inputs, recursive=recursive)
    if not audio_files:
        logger.warning("No audio files found in the provided inputs.")
        return []

    total = len(audio_files)
    results: list[BatchResult] = []

    for idx, audio_path in enumerate(audio_files, start=1):
        result = BatchResult(audio_path)
        logger.info("[%d/%d] Processing: %s", idx, total, audio_path.name)
        t0 = time.perf_counter()

        try:
            # ── Core pipeline ─────────────────────────────────────────────
            file_output_dir: Path | None = None
            if output_dir is not None:
                from utils import sanitise_stem

                file_output_dir = Path(output_dir) / sanitise_stem(
                    audio_path.stem, fallback="audio"
                )
            pipeline_out = run_pipeline(
                audio_path,
                language=language,
                consensus_models=consensus_models,
                alignment_strategy=alignment_strategy,
                enable_nlp=enable_nlp,
                enable_llm=enable_llm,
                ollama_model=ollama_model,
                enable_diarisation=enable_diarisation,
                output_dir=file_output_dir,
                consensus_threshold=consensus_threshold,
                similarity_threshold=similarity_threshold,
                keep_variant_wavs=keep_variant_wavs,
                word_timestamps=word_timestamps,
            )
            result.consensus_path = pipeline_out["consensus_path"]
            result.export_paths = pipeline_out.get("export_paths", {})
            result.diarisation_error = pipeline_out.get("diarisation_error")

            # ── Optional: Export additional formats ────────────────────────
            # (pipeline handles NLP, LLM, and diarisation; export_all used here
            #  only for additional formats beyond those auto-generated)
            if export_formats and result.consensus_path:
                from export_engine.exporter import export_all

                result.export_paths.update(
                    export_all(
                        consensus_md_path=result.consensus_path,
                        whisper_result=pipeline_out["transcripts"]["original"],
                        stem=pipeline_out.get("stem", audio_path.stem),
                        formats=export_formats,
                        output_dir=(
                            file_output_dir / "consensus" if file_output_dir else None
                        ),
                    )
                    or {}
                )

            result.success = True

        except Exception as exc:
            logger.error("Failed to process '%s': %s", audio_path.name, exc)
            result.error = str(exc)

        result.elapsed_seconds = round(time.perf_counter() - t0, 2)
        results.append(result)

        if progress_callback:
            progress_callback(idx, total, audio_path.name)

    # ── Write batch summary report ────────────────────────────────────────────
    _write_batch_report(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch report
# ─────────────────────────────────────────────────────────────────────────────


def _write_batch_report(results: list[BatchResult]) -> Path:
    """Write a Markdown summary of the batch run."""
    ensure_output_dirs()

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    success = sum(1 for r in results if r.success)
    failures = total - success
    elapsed = sum(r.elapsed_seconds for r in results)

    lines = [
        "# Chorus — Batch Processing Report",
        "",
        f"> **Generated:** {now}",
        f"> **Files processed:** {total}  |  **Succeeded:** {success}  |  **Failed:** {failures}",  # noqa: E501
        f"> **Total elapsed:** {elapsed:.1f} s",
        "",
        "## Results",
        "",
        "| # | File | Status | Elapsed | Consensus |",
        "|---|------|--------|--------:|-----------|",
    ]

    for idx, r in enumerate(results, start=1):
        if not r.success:
            status = f"❌ {r.error or 'Unknown error'}"
        elif r.diarisation_error:
            status = f"⚠️ OK — diarisation failed: {r.diarisation_error}"
        else:
            status = "✅ OK"
        cons_lnk = f"`{r.consensus_path.name}`" if r.consensus_path else "—"
        lines.append(
            f"| {idx} | `{r.path.name}` | {status} | {r.elapsed_seconds} s | {cons_lnk} |"  # noqa: E501
        )

    lines += ["", "---", "", "*Generated by Chorus Engine — Batch Processor*", ""]

    out_path = CONSENSUS_DIR / "batch_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Batch report written → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chorus-batch",
        description="Chorus Batch Processor — process multiple audio files in one run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chorus-batch /recordings/                         # all audio in directory
  chorus-batch /recordings/ --recursive             # include subdirectories
  chorus-batch a.mp3 b.wav --language en            # explicit files
  chorus-batch /audio/*.flac --export pdf srt       # with export
  chorus-batch /audio/ --diarise --nlp              # all features enabled
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Audio files, directories, or glob patterns. Not required with "
        "--check-diarisation.",
    )
    parser.add_argument(
        "--check-diarisation",
        action="store_true",
        help="Check whether diarisation can actually run right now (loads "
        "the real pipeline; does not process any audio) and exit — 0 if "
        "ready, 1 if not. Takes no inputs. This is the normal way to verify "
        "readiness before an unattended run; starting a real --diarise batch "
        "just to see whether it immediately refuses is not.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="BCP-47 language code hint (e.g. 'en'), or 'auto' to override a "
        "configured language and auto-detect.",
    )
    parser.add_argument(
        "--consensus-models",
        nargs="*",
        default=None,
        help="Whisper model names for consensus (space-separated, e.g. 'base small medium').",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Use one Whisper model for this run (shorthand for --consensus-models).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Whisper compute device. 'auto' probes CUDA, MPS, then CPU.",
    )
    parser.add_argument(
        "--parallelism",
        type=_parallelism,
        default=None,
        metavar="AUTO|N",
        help="Transcription worker count ('auto' or a positive integer).",
    )
    parser.add_argument(
        "--hardware-preset",
        choices=["max", "background"],
        default=None,
        help="Detect hardware and apply the selected model/device/parallelism preset "
        "for this run. Explicit CLI settings still win.",
    )
    parser.add_argument(
        "--alignment-strategy",
        choices=["sequence", "positional"],
        default=None,
        help="Consensus alignment strategy.",
    )
    parser.add_argument(
        "--export",
        "-e",
        nargs="*",
        choices=["pdf", "docx", "srt", "vtt"],
        default=None,
        help="Export formats to generate (space-separated).",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=_unit_interval,
        default=None,
        metavar="0..1",
        help="Minimum transcript agreement fraction for a HIGH-confidence word.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=_unit_interval,
        default=None,
        metavar="0..1",
        help="Fuzzy-match acceptance threshold.",
    )
    parser.add_argument(
        "--noise-floor-mode",
        choices=["vad", "fixed"],
        default=None,
        help="Noise-floor detection strategy.",
    )
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable word-level Whisper timestamps (normally only for SRT/VTT).",
    )
    parser.add_argument(
        "--keep-variant-wavs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain or discard intermediate cleaned WAV variants.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Scan directories recursively.",
    )
    parser.add_argument(
        "--diarise",
        action="store_true",
        help="Enable speaker diarisation (requires HUGGINGFACE_TOKEN).",
    )
    parser.add_argument(
        "--allow-diarisation-stub",
        action="store_true",
        help="Skip the diarisation pre-flight check and proceed even if the "
        "pipeline can't load, falling back to a single-speaker stub for "
        "every file. Without this flag, --diarise refuses to start rather "
        "than silently produce fake speaker labels.",
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="Enable spaCy NLP reconstruction for LOW-confidence tokens.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable local LLM (Ollama) reconstruction for LOW-confidence tokens.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model name for LLM reconstruction (e.g. 'mistral', 'neural-chat').",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama API base URL for this run.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=_positive_float,
        default=None,
        metavar="SECONDS",
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Root output directory. Each file's outputs are written to an "
        "isolated <DIR>/<stem>/ subdirectory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code for testability."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.check_diarisation:
        from diarisation.diariser import check_diarisation_ready

        ready, reason = check_diarisation_ready()
        if ready:
            print("Diarisation is ready.")
            return 0
        print("Diarisation is NOT ready:\n")
        print(reason)
        return 1

    if not args.inputs:
        parser.error("inputs are required unless --check-diarisation is given")

    if args.whisper_model and args.consensus_models:
        parser.error("--whisper-model and --consensus-models cannot be used together")

    if args.diarise and not args.allow_diarisation_stub:
        from diarisation.diariser import check_diarisation_ready

        ready, reason = check_diarisation_ready()
        if not ready:
            parser.error(
                "--diarise was requested, but diarisation cannot run:\n\n"
                f"{reason}\n\n"
                "Without this, every file would silently complete with all "
                "audio assigned to a single fake speaker — the exact failure "
                "that went unnoticed on a real batch before this check "
                "existed. Fix the above and retry, or pass "
                "--allow-diarisation-stub to proceed anyway with informed "
                "consent."
            )

    settings = _resolve_cli_settings(args)
    _apply_runtime_settings(settings)
    _display_settings(settings)

    output_dir = settings["output directory"][0]
    consensus_models = settings["models"][0]

    handler, log_path, previous_root_level = _attach_file_logging(output_dir)
    print(f"  log file               {log_path}")
    try:
        batch_results = run_batch(
            inputs=args.inputs,
            language=settings["language"][0],
            consensus_models=consensus_models,
            export_formats=args.export,
            recursive=args.recursive,
            alignment_strategy=settings["alignment"][0],
            enable_diarisation=args.diarise,
            enable_nlp=args.nlp,
            enable_llm=args.llm,
            ollama_model=settings["Ollama model"][0],
            output_dir=output_dir,
            consensus_threshold=settings["consensus threshold"][0],
            similarity_threshold=settings["similarity threshold"][0],
            keep_variant_wavs=settings["keep variant WAVs"][0],
            word_timestamps=settings["word timestamps"][0],
        )
    finally:
        logging.getLogger().removeHandler(handler)
        logging.getLogger().setLevel(previous_root_level)
        handler.close()

    diarisation_failures = sum(1 for r in batch_results if r.diarisation_error)

    print(f"\n{'─'*60}")
    print(
        f"  Batch complete: {sum(r.success for r in batch_results)}/{len(batch_results)} succeeded"  # noqa: E501
    )
    if diarisation_failures:
        print(
            f"  ⚠️  Diarisation failed on {diarisation_failures}/{len(batch_results)} "
            "file(s) — see batch_report.md for details"
        )
    print(f"  Report: {CONSENSUS_DIR / 'batch_report.md'}")
    print(f"  Log:    {log_path}")
    print(f"{'─'*60}\n")
    return 0 if all(r.success for r in batch_results) else 1


if __name__ == "__main__":
    sys.exit(main())
