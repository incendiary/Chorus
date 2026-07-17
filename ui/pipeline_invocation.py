"""ui/pipeline_invocation.py — Run-section orchestration and per-file pipeline glue."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

import config
from config import WHISPER_DEVICE
from pipeline_runner import run_pipeline
from ui.results import (
    build_file_anchors,
    hw_recommendation,
    record_recent_run,
    render_batch_outcome_summary,
    render_file_results,
    render_preflight_summary,
    render_processing_error,
    render_recent_runs,
    render_result_filter,
    render_result_navigation,
    render_run_status,
)
from ui.sidebar import SidebarConfig
from utils import sanitise_stem

logger = logging.getLogger(__name__)


def run_one_file(
    uf: object,
    progress_slot: object,
    status_slot: object,
    log_lines: list[str],
    log_expander: object,
    config_obj: SidebarConfig,
) -> tuple[dict, Path, str]:
    """Process a single uploaded file.

    Returns (results, tmp_path, original_stem).
    """
    original_stem = sanitise_stem(Path(uf.name).stem, fallback="upload")
    suffix = Path(uf.name).suffix.lower()
    # Secure temp file: unique path, exclusive creation, no race condition
    tmp_fd = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=f"{original_stem}_", delete=False
    )
    tmp_path = Path(tmp_fd.name)
    tmp_fd.write(uf.read())
    tmp_fd.close()

    def _progress(label: str, frac: float) -> None:
        progress_slot.progress(min(frac, 1.0), text=label)
        status_slot.markdown(f"**Status:** {label}")
        log_lines.append(f"`{frac * 100:.0f}%` — {label}")
        with log_expander:
            st.markdown("\n\n".join(log_lines))

    results = run_pipeline(
        audio_path=tmp_path,
        language=config_obj.language,
        consensus_models=config_obj.consensus_models,
        enable_nlp=config_obj.enable_nlp,
        enable_llm=config_obj.enable_llm,
        ollama_model=config_obj.ollama_model,
        enable_diarisation=config_obj.enable_diarisation,
        alignment_strategy=config_obj.alignment_choice,
        consensus_threshold=config_obj.consensus_threshold,
        similarity_threshold=config_obj.similarity_threshold,
        progress_callback=_progress,
    )
    return results, tmp_path, original_stem


def render_run_section(uploaded_files: list, config_obj: SidebarConfig) -> None:
    """Render the upload prompt, run controls, and per-file results."""
    if not uploaded_files:
        st.info(
            "Upload one or more audio files above to begin. "
            "Then configure options in the sidebar and start Chorus.",
            icon="👆",
        )
        return

    st.markdown('<div id="run-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("2 · Run Pipeline")

    # ── Processing mode (only shown for multiple files) ───────────────────────
    rec_mode, rec_reason = hw_recommendation()

    if len(uploaded_files) > 1:
        # Auto-switch to batch view for 3+ files
        if len(uploaded_files) >= 3:
            mode_choice = "All at once — results shown at end"
            st.info(
                f"📁 **Batch mode** — {len(uploaded_files)} files detected. "
                "All files will be processed before results are displayed.",
                icon="📁",
            )
        else:
            mode_choice = st.radio(
                "Processing mode",
                options=[
                    "Sequential — results appear per file",
                    "All at once — results shown at end",
                ],
                index=0 if rec_mode.startswith("Sequential") else 1,
                horizontal=True,
                help=(
                    "**Sequential:** each file is fully processed and its results shown "
                    "before the next file starts. Lower peak memory — best for longer "
                    "recordings or machines with less RAM.\n\n"
                    "**All at once:** all files are processed back-to-back before any "
                    "results are displayed. Processing is still single-threaded; the only "
                    "difference is when results appear."
                ),
            )
        st.caption(rec_reason)
        sequential = mode_choice.startswith("Sequential")
    else:
        sequential = True

    # ── LOW-word display toggle ───────────────────────────────────────────────
    show_low = st.toggle(
        "Include uncertain words in plain transcript",
        value=True,
        help=(
            "Controls the **Most Likely Transcript** download only — "
            "the annotated consensus document is unaffected.\n\n"
            "**On:** LOW-confidence words appear as `[word?]`.\n"
            "**Off:** LOW-confidence words are omitted entirely.\n\n"
            "Both variants are always included in the Download All zip."
        ),
    )

    # ── Run button ────────────────────────────────────────────────────────────
    n = len(uploaded_files)

    render_preflight_summary(
        file_count=n,
        model_choice=config_obj.model_choice,
        consensus_models=config_obj.consensus_models,
        alignment_choice=config_obj.alignment_choice,
        noise_mode_choice=config_obj.noise_mode_choice,
        enable_nlp=config_obj.enable_nlp,
        enable_llm=config_obj.enable_llm,
        enable_diarisation=config_obj.enable_diarisation,
    )
    if n > 10:
        st.warning(
            "Large batch detected. For easier troubleshooting, consider processing in smaller groups of 5-10 files.",
            icon="⚠️",
        )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button(
            f"▶ Start Chorus ({n} file{'s' if n > 1 else ''})",
            type="primary",
            use_container_width=True,
        )

    render_recent_runs()

    if not run_btn:
        return

    os.environ["WHISPER_MODEL"] = config_obj.model_choice
    os.environ["CONSENSUS_MODELS"] = ",".join(config_obj.consensus_models)
    os.environ["NOISE_FLOOR_MODE"] = config_obj.noise_mode_choice

    # Apply device and parallelism overrides to the live config module so the
    # transcription engine and orchestrator pick them up without a restart.
    _effective_device = (
        config_obj.device_choice
        if config_obj.device_choice != "auto"
        else WHISPER_DEVICE
    )
    config.WHISPER_DEVICE = _effective_device
    os.environ["WHISPER_DEVICE"] = _effective_device
    config.TRANSCRIPTION_PARALLELISM = config_obj.parallelism_choice
    os.environ["TRANSCRIPTION_PARALLELISM"] = config_obj.parallelism_choice

    formats_to_export = [
        fmt
        for fmt, checked in [
            ("pdf", config_obj.export_pdf),
            ("docx", config_obj.export_docx),
            ("srt", config_obj.export_srt),
        ]
        if checked
    ]

    # ── Main run loop ─────────────────────────────────────────────────────────

    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("3 · Results")
    run_started_at = time.time()
    run_status_slot = st.empty()
    completed_files = 0
    failed_files = 0
    failed_file_names: list[str] = []

    render_run_status(
        container=run_status_slot,
        total_files=len(uploaded_files),
        completed_files=completed_files,
        failed_files=failed_files,
        start_time=run_started_at,
    )

    if sequential:
        file_names = [str(uf.name) for uf in uploaded_files]
        file_anchors = build_file_anchors(file_names)
        render_result_navigation(file_names, file_anchors)
        sequential_results: list[tuple[object, dict, Path, str]] = []

        # Process and render each file as it completes
        for uf in uploaded_files:
            section_anchor = file_anchors[str(uf.name)]
            st.markdown(f'<div id="{section_anchor}"></div>', unsafe_allow_html=True)
            render_run_status(
                container=run_status_slot,
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                start_time=run_started_at,
                current_file=uf.name,
            )
            with st.expander(f"📄 {uf.name}", expanded=True):
                progress_bar = st.progress(0.0, text="Initialising…")
                status_text = st.empty()
                log_expander = st.expander("📋 Live log", expanded=False)
                log_lines: list[str] = []

                tmp_path: Path | None = None
                try:
                    results, tmp_path, original_stem = run_one_file(
                        uf,
                        progress_bar,
                        status_text,
                        log_lines,
                        log_expander,
                        config_obj,
                    )
                    completed_files += 1
                    progress_bar.progress(1.0, text="✅ Complete!")
                    status_text.success(
                        f"Completed in **{results['elapsed_seconds']} s**"
                    )
                    sequential_results.append((uf, results, tmp_path, original_stem))
                except Exception as exc:
                    failed_files += 1
                    failed_file_names.append(str(uf.name))
                    render_processing_error(uf.name, exc, allow_retry=True)
                    logger.exception("Pipeline error for %s", uf.name)
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)

            render_run_status(
                container=run_status_slot,
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                start_time=run_started_at,
            )

        # Show summary once processing is finished, then render detail sections.
        total_duration = time.time() - run_started_at
        render_batch_outcome_summary(
            total_files=len(uploaded_files),
            completed_files=completed_files,
            failed_files=failed_files,
            duration_seconds=total_duration,
            failed_file_names=failed_file_names,
            file_anchors=file_anchors,
        )
        record_recent_run(
            total=len(uploaded_files),
            completed=completed_files,
            failed=failed_files,
            duration=total_duration,
        )
        result_filter = render_result_filter(len(uploaded_files))

        for uf, results, tmp_path, original_stem in sequential_results:
            if result_filter == "Failed":
                continue
            with st.expander(f"📄 {uf.name}", expanded=True):
                st.success(f"Completed in **{results['elapsed_seconds']} s**")
                render_file_results(
                    uf.name,
                    results,
                    tmp_path,
                    original_stem,
                    show_low=show_low,
                    formats_to_export=formats_to_export,
                )

    else:
        file_names = [str(uf.name) for uf in uploaded_files]
        file_anchors = build_file_anchors(file_names)
        render_result_navigation(file_names, file_anchors)

        # Process all files first, collect results
        all_results: list[tuple[object, dict, Path, str]] = []
        overall = st.progress(0.0, text="Starting…")

        for idx, uf in enumerate(uploaded_files):
            render_run_status(
                container=run_status_slot,
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                start_time=run_started_at,
                current_file=uf.name,
            )
            overall.progress(
                idx / len(uploaded_files),
                text=f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})…",
            )
            progress_bar = st.empty()
            status_text = st.empty()
            log_expander = st.expander(f"📋 Live log — {uf.name}", expanded=False)
            log_lines = []

            tmp_path: Path | None = None
            try:
                results, tmp_path, original_stem = run_one_file(
                    uf,
                    progress_bar,
                    status_text,
                    log_lines,
                    log_expander,
                    config_obj,
                )
                all_results.append((uf, results, tmp_path, original_stem))
                completed_files += 1
            except Exception as exc:
                failed_files += 1
                failed_file_names.append(str(uf.name))
                render_processing_error(uf.name, exc)
                logger.exception("Pipeline error for %s", uf.name)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

            render_run_status(
                container=run_status_slot,
                total_files=len(uploaded_files),
                completed_files=completed_files,
                failed_files=failed_files,
                start_time=run_started_at,
            )

        overall.progress(1.0, text=f"✅ All {len(uploaded_files)} files complete!")

        # Now render all results
        total_duration = time.time() - run_started_at
        render_batch_outcome_summary(
            total_files=len(uploaded_files),
            completed_files=completed_files,
            failed_files=failed_files,
            duration_seconds=total_duration,
            failed_file_names=failed_file_names,
            file_anchors=file_anchors,
        )
        record_recent_run(
            total=len(uploaded_files),
            completed=completed_files,
            failed=failed_files,
            duration=total_duration,
        )
        result_filter = render_result_filter(len(uploaded_files))

        for uf, results, tmp_path, original_stem in all_results:
            if result_filter == "Failed":
                continue
            section_anchor = file_anchors[str(uf.name)]
            st.markdown(f'<div id="{section_anchor}"></div>', unsafe_allow_html=True)
            with st.expander(f"📄 {uf.name}", expanded=True):
                st.success(f"Completed in **{results['elapsed_seconds']} s**")
                render_file_results(
                    uf.name,
                    results,
                    tmp_path,
                    original_stem,
                    show_low=show_low,
                    formats_to_export=formats_to_export,
                )
