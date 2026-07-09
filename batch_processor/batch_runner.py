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

from config import CONSENSUS_DIR, ensure_output_dirs

logger = logging.getLogger(__name__)

# Supported audio extensions
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".webm",
    ".opus",
}


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

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL: {self.error}"
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
            )
            result.consensus_path = pipeline_out["consensus_path"]
            result.export_paths = pipeline_out.get("export_paths", {})

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
        status = "✅ OK" if r.success else f"❌ {r.error or 'Unknown error'}"
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
        nargs="+",
        help="Audio files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="BCP-47 language code hint (e.g. 'en'). Omit for auto-detect.",
    )
    parser.add_argument(
        "--consensus-models",
        nargs="*",
        default=None,
        help="Whisper model names for consensus (space-separated, e.g. 'base small medium').",
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
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Root output directory. Each file's outputs are written to an "
        "isolated <DIR>/<stem>/ subdirectory.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    consensus_models = tuple(args.consensus_models) if args.consensus_models else None
    batch_results = run_batch(
        inputs=args.inputs,
        language=args.language,
        consensus_models=consensus_models,
        export_formats=args.export,
        recursive=args.recursive,
        alignment_strategy=args.alignment_strategy,
        enable_diarisation=args.diarise,
        enable_nlp=args.nlp,
        enable_llm=args.llm,
        ollama_model=args.ollama_model,
        output_dir=output_dir,
    )

    print(f"\n{'─'*60}")
    print(
        f"  Batch complete: {sum(r.success for r in batch_results)}/{len(batch_results)} succeeded"  # noqa: E501
    )
    print(f"  Report: {CONSENSUS_DIR / 'batch_report.md'}")
    print(f"{'─'*60}\n")
    sys.exit(0 if all(r.success for r in batch_results) else 1)
