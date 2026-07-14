"""ui/results.py — Results rendering, run-status panels, and summary helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from config import VARIANT_LABELS
from export_engine.exporter import (
    export_all,
    export_best_guess,
    export_plain_text,
    export_zip,
)
from utils import sanitise_stem

SUMMARY_SUCCESS_MSG = (
    "All files completed successfully. Review confidence sections below, "
    "then download archives."
)
SUMMARY_FAILURE_MSG = (
    "Some files failed. Check technical details in the affected sections, "
    "then retry failed files in smaller batches."
)
PROCESSING_FAILURE_MSG = (
    "Processing failed for {file_name}. Review guidance below and retry when ready."
)
PROCESSING_REMEDIATION_STEPS = (
    "- Confirm audio file integrity and supported format\n"
    "- Try a smaller model or disable advanced features\n"
    "- Re-run this file alone to isolate the issue"
)


@st.cache_data
def hw_recommendation() -> tuple[str, str]:
    """Return (recommended_mode_label, reason_caption) based on available RAM."""
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        cores = psutil.cpu_count(logical=False) or 1
    except Exception:  # noqa: BLE001
        return (
            "Sequential — results appear per file",
            "ℹ️ Could not detect hardware — Sequential is the safer default.",
        )

    if ram_gb < 8:
        return (
            "Sequential — results appear per file",
            f"⚠️ {ram_gb:.1f} GB RAM detected ({cores} cores). "
            "**Sequential recommended** — All-at-once loads all recordings into memory "
            "simultaneously and may cause out-of-memory errors on this machine.",
        )
    if ram_gb < 16:
        return (
            "Sequential — results appear per file",
            f"ℹ️ {ram_gb:.1f} GB RAM / {cores} cores detected. "
            "Sequential is the safer choice for longer recordings. "
            "All-at-once is fine for short clips.",
        )
    return (
        "All at once — results shown at end",
        f"✅ {ram_gb:.1f} GB RAM / {cores} cores detected. Either mode is fine.",
    )


def render_preflight_summary(
    file_count: int,
    model_choice: str,
    consensus_models: tuple[str, ...],
    alignment_choice: str,
    noise_mode_choice: str,
    enable_nlp: bool,
    enable_llm: bool,
    enable_diarisation: bool,
) -> None:
    """Render run preflight summary with informed-choice guidance."""
    selected_features = []
    if enable_nlp:
        selected_features.append("NLP reconstruction")
    if enable_llm:
        selected_features.append("LLM reconstruction")
    if enable_diarisation:
        selected_features.append("speaker diarisation")
    feature_text = ", ".join(selected_features) if selected_features else "none"

    runtime_hint = (
        "longer runs expected due to max-accuracy configuration"
        if model_choice in {"medium", "small"}
        or enable_nlp
        or enable_llm
        or enable_diarisation
        else "balanced runtime expected"
    )
    model_set_text = ", ".join(consensus_models)

    st.markdown(
        (
            '<div class="chorus-preflight">'
            f"<b>Preflight:</b> {file_count} file{'s' if file_count > 1 else ''}, "
            f"model <b>{model_choice}</b>, alignment <b>{alignment_choice}</b>, "
            f"noise mode <b>{noise_mode_choice}</b>, consensus models <b>{model_set_text}</b>, "
            f"advanced features: <b>{feature_text}</b>.<br>"
            f"Expected profile: <b>{runtime_hint}</b>. "
            "Tip: use smaller batches if you need faster feedback cycles."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_confidence_overview(consensus_text: str) -> None:
    """Render confidence bars, metrics, and interpretation guidance."""
    import re as _re

    high_m = _re.search(r"HIGH\s*\|\s*(\d+)", consensus_text)
    med_m = _re.search(r"MEDIUM\s*\|\s*(\d+)", consensus_text)
    low_m = _re.search(r"LOW\s*\|\s*(\d+)", consensus_text)
    n_high = int(high_m.group(1)) if high_m else 0
    n_med = int(med_m.group(1)) if med_m else 0
    n_low = int(low_m.group(1)) if low_m else 0
    total_w = n_high + n_med + n_low or 1

    bar_cols = st.columns([n_high or 1, n_med or 1, n_low or 1])
    with bar_cols[0]:
        st.markdown(
            f'<div style="background:var(--chorus-high-bg);color:var(--chorus-high-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🟢 HIGH</b><br>{n_high} ({n_high*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )
    with bar_cols[1]:
        st.markdown(
            f'<div style="background:var(--chorus-med-bg);color:var(--chorus-med-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🟡 MED</b><br>{n_med} ({n_med*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )
    with bar_cols[2]:
        st.markdown(
            f'<div style="background:var(--chorus-low-bg);color:var(--chorus-low-fg);padding:8px;border-radius:4px;text-align:center">'
            f"<b>🔴 LOW</b><br>{n_low} ({n_low*100//total_w}%)</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("High confidence words", n_high)
    c2.metric("Medium confidence words", n_med)
    c3.metric("Low confidence words", n_low)
    st.markdown(
        '<div class="chorus-confidence-note">'
        "High confidence terms generally reflect strong cross-variant agreement. "
        "Review medium confidence carefully. Validate low confidence terms against source audio."
        "</div>",
        unsafe_allow_html=True,
    )


def render_processing_error(
    file_name: str, exc: Exception, allow_retry: bool = False
) -> None:
    """Render consistent actionable processing error guidance."""
    st.error(PROCESSING_FAILURE_MSG.format(file_name=file_name))
    st.markdown(PROCESSING_REMEDIATION_STEPS)
    with st.expander(f"Technical details — {file_name}"):
        st.code(str(exc))

    if allow_retry and st.button(
        "Retry this file",
        key=f"retry_seq_{sanitise_stem(Path(file_name).stem, fallback='upload')}",
    ):
        st.rerun()


def render_run_status(
    *,
    container: object,
    total_files: int,
    completed_files: int,
    failed_files: int,
    start_time: float,
    current_file: str | None = None,
) -> None:
    """Render a compact batch status panel with throughput and ETA guidance."""
    processed = completed_files + failed_files
    elapsed = max(time.time() - start_time, 0.001)
    progress = (processed / total_files) if total_files else 0.0

    rate_per_sec = processed / elapsed if processed else 0.0
    eta_seconds = (
        int((total_files - processed) / rate_per_sec) if rate_per_sec > 0 else None
    )

    with container.container():
        st.markdown('<div class="chorus-run-status">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed", f"{processed}/{total_files}")
        c2.metric("Completed", completed_files)
        c3.metric("Failed", failed_files)
        c4.metric("Elapsed", f"{int(elapsed)} s")

        progress_text = (
            f"Processing {current_file}…"
            if current_file
            else f"Batch progress: {processed}/{total_files} files"
        )
        st.progress(progress, text=progress_text)

        if eta_seconds is not None and processed < total_files:
            st.caption(
                f"Estimated time remaining: ~{eta_seconds} s (updates as files complete)."
            )
        elif processed == total_files:
            st.caption("Batch complete. Review results and download outputs below.")

        st.markdown("</div>", unsafe_allow_html=True)


def render_batch_outcome_summary(
    *,
    total_files: int,
    completed_files: int,
    failed_files: int,
    duration_seconds: float,
    failed_file_names: list[str] | None = None,
    file_anchors: dict[str, str] | None = None,
) -> None:
    """Render concise outcomes guidance after a batch run completes."""
    st.markdown("#### Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total_files)
    c2.metric("Completed", completed_files)
    c3.metric("Failed", failed_files)
    c4.metric("Duration", f"{int(duration_seconds)} s")

    if failed_files == 0:
        st.markdown(
            '<div class="chorus-run-summary-ok">ALL FILES COMPLETED</div>',
            unsafe_allow_html=True,
        )
        st.success(SUMMARY_SUCCESS_MSG, icon="✅")
    else:
        st.markdown(
            '<div class="chorus-run-summary-issues">ATTENTION REQUIRED</div>',
            unsafe_allow_html=True,
        )
        st.warning(SUMMARY_FAILURE_MSG, icon="⚠️")
        if failed_file_names:
            if file_anchors:
                unique_failed = list(dict.fromkeys(failed_file_names))
                failed_links = " · ".join(
                    f"[{name}](#{file_anchors[name]})"
                    for name in unique_failed
                    if name in file_anchors
                )
                if failed_links:
                    st.markdown(f"**Failed files:** {failed_links}")
            else:
                st.markdown("**Failed files:** " + ", ".join(failed_file_names))


def render_result_navigation(
    file_names: list[str], file_anchors: dict[str, str]
) -> None:
    """Render quick links to each file section for large result sets."""
    if len(file_names) < 3:
        return

    anchors = [(name, file_anchors[name]) for name in file_names]
    links = " · ".join(
        ["[Top](#top)", *[f"[{name}](#{anchor})" for name, anchor in anchors]]
    )
    st.markdown(
        (
            '<div class="chorus-action-bar">'
            "<b>Quick Navigation:</b> "
            f"{links}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_result_filter(total_files: int) -> str:
    """Render a simple result visibility filter for larger batches."""
    if total_files < 3:
        return "All"
    return st.radio(
        "Results filter",
        options=["All", "Completed", "Failed"],
        index=0,
        horizontal=True,
        help="Use this to focus on successful or failed files in larger batches.",
    )


def build_file_anchors(file_names: list[str]) -> dict[str, str]:
    """Return deterministic unique anchors keyed by file name."""
    anchors: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name in file_names:
        base = sanitise_stem(Path(name).stem, fallback="upload")
        counts[base] = counts.get(base, 0) + 1
        suffix = f"-{counts[base]}" if counts[base] > 1 else ""
        anchors[name] = f"{base}{suffix}"
    return anchors


def record_recent_run(
    *, total: int, completed: int, failed: int, duration: float
) -> None:
    """Store a compact run snapshot in session state."""
    stamp = time.strftime("%H:%M:%S")
    st.session_state["recent_runs"] = [
        {
            "time": stamp,
            "total": total,
            "completed": completed,
            "failed": failed,
            "duration": int(duration),
        },
        *st.session_state["recent_runs"],
    ][:5]


def render_recent_runs() -> None:
    """Render recent in-session run snapshots."""
    runs = st.session_state.get("recent_runs", [])
    if not runs:
        return

    with st.expander("Recent Runs", expanded=False):
        for run in runs:
            st.markdown(
                f"- **{run['time']}** — {run['completed']}/{run['total']} completed, "
                f"{run['failed']} failed, {run['duration']} s"
            )


def render_file_results(
    filename: str,
    results: dict,
    tmp_path: Path,
    original_stem: str,
    *,
    show_low: bool,
    formats_to_export: list[str],
) -> None:
    """Render the results section for one processed file."""
    transcripts = results["transcripts"]
    consensus_path = results["consensus_path"]
    consensus_text = consensus_path.read_text(encoding="utf-8")

    # ── Processing time ───────────────────────────────────────────────
    elapsed = results.get("elapsed_seconds", 0)
    st.caption(f"⏱️ Processed in **{elapsed} s**")

    # ── Metrics row ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, (key, meta) in zip([m1, m2, m3, m4], transcripts.items(), strict=False):
        wc = len(meta.get("text", "").split())
        col.metric(VARIANT_LABELS.get(key, key), f"{wc} words")

    # ── Confidence Visualisation ──────────────────────────────────────
    st.markdown("#### 🎯 Confidence Overview")
    render_confidence_overview(consensus_text)

    # ── Consensus document preview ────────────────────────────────────
    st.markdown("#### 📄 Consensus Transcript")
    view_mode = st.toggle(
        "Show raw Markdown",
        value=False,
        key=f"raw_md_{original_stem}",
        help="Toggle between rendered view and raw Markdown source.",
    )
    if view_mode:
        st.code(consensus_text, language="markdown")
    else:
        st.markdown(consensus_text, unsafe_allow_html=False)

    # ── Download buttons ──────────────────────────────────────────────
    # Stacked download controls remain readable across desktop and narrow layouts.
    st.download_button(
        label="⬇️ Download Consensus Markdown (.md)",
        data=consensus_text,
        file_name=consensus_path.name,
        mime="text/markdown",
        type="primary",
        key=f"dl_md_{original_stem}",
        use_container_width=True,
    )

    plain_path = export_plain_text(consensus_path, original_stem, include_low=show_low)
    st.download_button(
        label="⬇️ Download Most Likely Transcript (.txt)",
        data=plain_path.read_text(encoding="utf-8"),
        file_name=plain_path.name,
        mime="text/plain",
        key=f"dl_txt_{original_stem}",
        use_container_width=True,
    )

    best_guess_path = export_best_guess(consensus_path, original_stem)
    st.download_button(
        label="⬇️ Download Best-Guess Transcript (.txt)",
        data=best_guess_path.read_text(encoding="utf-8"),
        file_name=best_guess_path.name,
        mime="text/plain",
        help="Clean, human-readable transcript with no confidence markup.",
        key=f"dl_best_guess_{original_stem}",
        use_container_width=True,
    )

    with st.spinner("Building archive with selected outputs…"):
        zip_bytes = export_zip(
            consensus_path,
            transcripts["original"],
            original_stem,
            include_formats=formats_to_export or None,
            output_dir=None,
        )
    st.download_button(
        label="⬇️ Download Full Output Archive (.zip)",
        data=zip_bytes,
        file_name=f"{original_stem}_chorus_all.zip",
        mime="application/zip",
        key=f"dl_zip_{original_stem}",
        use_container_width=True,
    )

    # Additional format exports
    if formats_to_export:
        st.markdown("##### 📥 Additional Exports")
        with st.spinner("Generating exports…"):
            export_paths = export_all(
                consensus_path,
                transcripts["original"],
                original_stem,
                formats_to_export,
            )
        fmt_cols = st.columns(len(formats_to_export))
        for col, fmt in zip(fmt_cols, formats_to_export, strict=False):
            epath = export_paths.get(fmt)
            if epath and epath.exists():
                with col:
                    with open(epath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ .{fmt.upper()}",
                            data=f,
                            file_name=epath.name,
                            mime="application/octet-stream",
                            key=f"dl_{fmt}_{original_stem}",
                        )

    # Speaker diarisation
    if results.get("diarised_path") and results["diarised_path"].exists():
        st.markdown("##### 🗣️ Speaker Diarisation")

        # ── Editable speaker name table ───────────────────────────────
        speaker_labels = results.get("speaker_labels", [])
        if speaker_labels:
            from diarisation.diariser import load_speaker_names, save_speaker_names

            # Load existing names (previously saved or empty)
            existing_names = load_speaker_names(original_stem)

            st.markdown(
                "**Speaker Names** — assign human-readable names to "
                "each speaker. Names are saved and will be re-used on "
                "reprocess."
            )
            name_cols = st.columns([1, 2])
            with name_cols[0]:
                st.markdown("**Label**")
            with name_cols[1]:
                st.markdown("**Name**")

            updated_names: dict[str, str] = {}
            for spk in speaker_labels:
                row_cols = st.columns([1, 2])
                with row_cols[0]:
                    st.code(spk, language=None)
                with row_cols[1]:
                    name_val = st.text_input(
                        f"Name for {spk}",
                        value=existing_names.get(spk, ""),
                        placeholder="e.g. Interviewer, Guest…",
                        label_visibility="collapsed",
                        key=f"spk_name_{spk}_{original_stem}",
                    )
                    if name_val.strip():
                        updated_names[spk] = name_val.strip()

            # Save button
            if st.button(
                "💾 Save Speaker Names",
                key=f"save_spk_{original_stem}",
                help="Saves names to a sidecar JSON file. They will be "
                "automatically loaded next time you process this file.",
            ):
                save_speaker_names(original_stem, updated_names)
                st.success(
                    f"Saved {len(updated_names)} speaker name(s) → "
                    f"`{original_stem}_speakers.json`"
                )
                st.rerun()

        # ── Diarised transcript preview ───────────────────────────────
        diar_text = results["diarised_path"].read_text(encoding="utf-8")
        with st.expander("Preview Diarised Transcript"):
            st.markdown(diar_text)
        st.download_button(
            label="⬇️ Diarised Transcript (.md)",
            data=diar_text,
            file_name=results["diarised_path"].name,
            mime="text/markdown",
            key=f"dl_diar_{original_stem}",
        )

    # AI Context Pack
    ai_ctx_path = results.get("ai_context_path")
    if ai_ctx_path and ai_ctx_path.exists():
        st.markdown("##### 🤖 AI Context Pack")
        st.caption(
            "A structured document designed to be fed to an LLM alongside "
            "questions about this transcript. Contains methodology, confidence "
            "data, and uncertainty annotations."
        )
        ai_ctx_text = ai_ctx_path.read_text(encoding="utf-8")
        with st.expander("Preview AI Context Pack"):
            st.markdown(ai_ctx_text)
        st.download_button(
            label="⬇️ AI Context Pack (.md)",
            data=ai_ctx_text,
            file_name=ai_ctx_path.name,
            mime="text/markdown",
            key=f"dl_ai_ctx_{original_stem}",
        )

    # Individual variant transcripts
    st.markdown("##### 🔍 Individual Variant Transcripts")
    tabs = st.tabs([VARIANT_LABELS.get(k, k) for k in transcripts])
    for tab, (key, meta) in zip(tabs, transcripts.items(), strict=False):
        with tab:
            st.markdown(f"**Detected language:** `{meta.get('language', 'unknown')}`")
            st.markdown(f"**Model:** `{meta.get('model', 'unknown')}`")
            st.text_area(
                "Transcript text",
                value=meta.get("text", "").strip(),
                height=200,
                key=f"transcript_{key}_{original_stem}",
            )
            st.download_button(
                label=f"⬇️ {original_stem}_{key}.json",
                data=json.dumps(meta, ensure_ascii=False, indent=2),
                file_name=f"{original_stem}_{key}.json",
                mime="application/json",
                key=f"dl_json_{key}_{original_stem}",
            )
