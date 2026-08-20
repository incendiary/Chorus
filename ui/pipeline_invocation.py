"""ui/pipeline_invocation.py — Run-section orchestration and per-file pipeline glue."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

import config
from config import WHISPER_DEVICE
from pipeline_runner import run_pipeline
from transcription_engine.orchestrator import load_transcripts_from_disk
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
from ui.run_manager import RunManager, get_run_manager
from ui.run_state import RUNS_DIR, FileEntry, RunJob
from ui.run_status_panel import render_file_status_panel
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
        # O(1): render only the latest line, not the whole joined history
        # (a 197-segment file used to re-render ~19k cumulative lines).
        log_expander.markdown(log_lines[-1])

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


def spool_upload(uf: object, dest_dir: Path) -> tuple[Path, str]:
    """Spool an uploaded file into *dest_dir*; return (path, sanitised stem).

    Mirrors ``run_one_file``'s tempfile/sanitise logic, but writes into a
    run-specific directory (rather than the system temp dir) so the file
    survives on disk for the background thread to read after this script
    run has ended.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitise_stem(Path(uf.name).stem, fallback="upload")
    suffix = Path(uf.name).suffix.lower()
    tmp_fd = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=f"{stem}_", dir=str(dest_dir), delete=False
    )
    path = Path(tmp_fd.name)
    tmp_fd.write(uf.read())
    tmp_fd.close()
    return path, stem


def render_run_section(uploaded_files: list, config_obj: SidebarConfig) -> None:
    """Render the run section as a three-state dispatcher on ``RunManager``.

    Idle — preflight/upload/mode UI and the Start button.
    Running — a live, poll-driven status view fed from the on-disk state
      file, so progress survives a refresh or tab close.
    Finished — the results view, fed from the manager's in-memory registry
      with a disk-rehydration fallback.
    """
    manager = get_run_manager()
    state = manager.get_state()
    status = state.get("status") if state else None

    if status == "running":
        _render_running_fragment(manager)
        return

    if status in ("finished", "interrupted"):
        _render_finished_state(manager, state)
        return

    if not uploaded_files:
        st.info(
            "Upload one or more audio files above to begin. "
            "Then configure options in the sidebar and start Chorus.",
            icon="👆",
        )
        return

    _render_idle_state(uploaded_files, config_obj, manager)


def _render_idle_state(
    uploaded_files: list, config_obj: SidebarConfig, manager: RunManager
) -> None:
    """Preflight/upload/mode UI and the Start Chorus button."""
    st.markdown('<div id="run-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("2 · Run Pipeline")

    # ── Processing mode (only shown for multiple files) ───────────────────────
    # Background runs always process files one at a time (RunManager enforces
    # a single active run and run_worker.execute_run loops sequentially); this
    # choice is kept for preflight/expectation-setting parity but no longer
    # branches execution.
    rec_mode, rec_reason = hw_recommendation()

    if len(uploaded_files) > 1:
        # Auto-switch to batch view for 3+ files
        if len(uploaded_files) >= 3:
            st.info(
                f"📁 **Batch mode** — {len(uploaded_files)} files detected. "
                "All files will be processed before results are displayed.",
                icon="📁",
            )
        else:
            st.radio(
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
            disabled=manager.is_running(),
        )

    render_recent_runs()

    if not run_btn:
        return

    os.environ["WHISPER_MODEL"] = config_obj.model_choice
    os.environ["CONSENSUS_MODELS"] = ",".join(config_obj.consensus_models)
    os.environ["NOISE_FLOOR_MODE"] = config_obj.noise_mode_choice
    config.NOISE_FLOOR_MODE = config_obj.noise_mode_choice

    # Apply live config overrides so the current process picks them up without
    # a restart; environment values alone affect only later processes/imports.
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

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    files: list[FileEntry] = []
    for uf in uploaded_files:
        spool_path, stem = spool_upload(uf, run_dir)
        files.append(
            FileEntry(name=str(uf.name), stem=stem, spool_path=str(spool_path))
        )

    job = RunJob(
        run_id=run_id,
        config={
            "language": config_obj.language,
            "consensus_models": config_obj.consensus_models,
            "enable_nlp": config_obj.enable_nlp,
            "enable_llm": config_obj.enable_llm,
            "ollama_model": config_obj.ollama_model,
            "enable_diarisation": config_obj.enable_diarisation,
            "alignment_strategy": config_obj.alignment_choice,
            "consensus_threshold": config_obj.consensus_threshold,
            "similarity_threshold": config_obj.similarity_threshold,
        },
        files=files,
        show_low=show_low,
        formats_to_export=formats_to_export,
    )

    if not manager.start(job):
        st.warning(
            "A run is already in progress. Please wait for it to finish.",
            icon="⚠️",
        )
        return

    # Under CHORUS_SYNC_RUN=1, start() blocks until the run has finished, so
    # the state file already says "finished" here; under real threading it
    # says "running". Either way, a full-app rerun lets the dispatcher at the
    # top of render_run_section re-read the state and switch views.
    st.rerun()


@st.fragment(run_every=2)
def _render_running_fragment(manager: RunManager) -> None:
    """Poll-driven live view of the in-progress run (survives refresh/close)."""
    state = manager.get_state()
    if state is None:
        return
    if state.get("status") != "running":
        # The run finished between the last poll and this one — hand control
        # back to the full-page dispatcher so it can render the Finished view.
        st.rerun(scope="app")
        return

    files = state["files"]
    total_files = len(files)
    completed_files = sum(1 for f in files if f["status"] == "done")
    failed_files = sum(1 for f in files if f["status"] == "error")

    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("3 · Results")

    render_run_status(
        container=st.container(),
        total_files=total_files,
        completed_files=completed_files,
        failed_files=failed_files,
        start_time=state.get("started_at") or time.time(),
    )

    run_config = state.get("config", {})
    for file_state in files:
        with st.expander(f"📄 {file_state['name']}", expanded=True):
            panel_state = {**file_state, "_config": run_config}
            render_file_status_panel(panel_state)


def _render_finished_state(manager: RunManager, state: dict) -> None:
    """Reuse the results-rendering helpers, fed from the manager's in-memory
    registry with a disk-rehydration fallback for a restarted server."""
    run_id = state["run_id"]
    ui_options = state.get("ui_options", {})
    show_low = ui_options.get("show_low", True)
    formats_to_export = ui_options.get("formats_to_export", [])

    if state.get("status") == "interrupted":
        st.warning(
            "The previous run was interrupted (the server restarted mid-run). "
            "Partial outputs may be available below.",
            icon="⚠️",
        )

    files = state["files"]
    total_files = len(files)
    completed_files = sum(1 for f in files if f["status"] == "done")
    failed_files = sum(1 for f in files if f["status"] == "error")
    failed_file_names = [f["name"] for f in files if f["status"] == "error"]
    duration = (state.get("finished_at") or time.time()) - (
        state.get("started_at") or time.time()
    )

    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("3 · Results")

    file_names = [f["name"] for f in files]
    file_anchors = build_file_anchors(file_names)
    render_result_navigation(file_names, file_anchors)

    render_batch_outcome_summary(
        total_files=total_files,
        completed_files=completed_files,
        failed_files=failed_files,
        duration_seconds=duration,
        failed_file_names=failed_file_names,
        file_anchors=file_anchors,
    )

    session_key = f"_chorus_recorded_run_{run_id}"
    if not st.session_state.get(session_key):
        record_recent_run(
            total=total_files,
            completed=completed_files,
            failed=failed_files,
            duration=duration,
        )
        st.session_state[session_key] = True

    result_filter = render_result_filter(total_files)
    results_registry = manager.get_results(run_id)

    for file_state in files:
        name = file_state["name"]
        stem = file_state["stem"]
        if result_filter == "Failed" and file_state["status"] != "error":
            continue
        if result_filter == "Completed" and file_state["status"] != "done":
            continue

        section_anchor = file_anchors[name]
        st.markdown(f'<div id="{section_anchor}"></div>', unsafe_allow_html=True)
        with st.expander(f"📄 {name}", expanded=True):
            if file_state["status"] == "error":
                render_processing_error(
                    name, RuntimeError(file_state.get("error") or "Unknown error")
                )
                continue

            results = results_registry.get(name)
            if results is None:
                results = _rehydrate_results_from_disk(stem, file_state)
                if results is None:
                    st.error(
                        f"Results for {name} are no longer available "
                        "(in-memory cache cleared and outputs incomplete)."
                    )
                    continue

            st.success(f"Completed in **{results.get('elapsed_seconds', 0)} s**")
            render_file_results(
                name,
                results,
                None,
                stem,
                show_low=show_low,
                formats_to_export=formats_to_export,
            )

    if st.button("🧹 Clear results / start new run"):
        manager.clear_finished()
        st.rerun()


def _rehydrate_results_from_disk(stem: str, file_state: dict) -> dict | None:
    """Reconstruct a ``results`` dict from disk when the in-memory registry
    has been cleared (e.g. the server restarted after this run finished).

    Returns ``None`` when the on-disk transcripts don't include the
    ``"original"`` variant that ``export_zip`` requires (see
    ``ui/results.py::render_file_results``), or the consensus file is
    missing — the caller shows an error instead of guessing.
    """
    output_paths = file_state.get("output_paths", {})
    consensus_path = output_paths.get("consensus_path")
    if not consensus_path:
        return None

    transcripts = load_transcripts_from_disk(stem)
    if not transcripts.get("original"):
        return None

    def _opt_path(key: str) -> Path | None:
        value = output_paths.get(key)
        return Path(value) if value else None

    return {
        "transcripts": transcripts,
        "consensus_path": Path(consensus_path),
        "ai_context_path": _opt_path("ai_context_path"),
        "bundle_path": _opt_path("bundle_path"),
        "best_guess_path": _opt_path("best_guess_path"),
        "diarised_path": _opt_path("diarised_path"),
        "speaker_labels": [],
        "elapsed_seconds": file_state.get("elapsed", 0),
    }
