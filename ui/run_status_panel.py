"""ui/run_status_panel.py — Per-file live status panel for background runs.

Replaces the old "Live log" expander (which re-appended the entire log
history on every progress callback — an O(n²) render for long files) with a
compact status display: what is happening now, forward-motion signals, and a
bounded, deduplicated history. Designed to be called from the Running
fragment in ``ui/pipeline_invocation.py``, once per poll, inside the
preserved ``📄 {name}`` expander.
"""

from __future__ import annotations

import time

import streamlit as st

_STAGE_LABELS = {
    "cleaning": "Audio cleaning",
    "loading_model": "Audio cleaning",  # folded into the same checklist item
    "transcribing": "Transcribing",
    "consensus": "Consensus",
    "reconstruction": "Reconstruction",
    "export": "Export",
    "diarisation": "Diarisation",
    "done": "Done",
}

_STALE_SECONDS = 60


def _format_elapsed(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _status_line(file_state: dict) -> str:
    """Build the ``st.progress`` text, e.g. ``"Transcribing — Whisper base ·
    Original (unprocessed) · segment 45/197"``."""
    stage = file_state.get("stage")
    label = _STAGE_LABELS.get(stage, (stage or "Working").replace("_", " ").title())
    detail = file_state.get("detail")
    parallel_workers = file_state.get("parallel_workers")
    passes_done = file_state.get("passes_done")
    passes_total = file_state.get("passes_total")
    segment = file_state.get("segment")
    segments_total = file_state.get("segments_total")

    text = f"{label} — {detail}" if detail else label

    if stage == "transcribing" and parallel_workers and parallel_workers > 1:
        piece = (
            f"{passes_done or 0}/{passes_total} passes complete"
            if passes_total
            else f"{passes_done or 0} passes complete"
        )
        text += f" · {piece} · {parallel_workers} running in parallel"
    elif segment is not None and segments_total:
        text += f" · segment {segment}/{segments_total}"
    elif stage == "transcribing" and passes_total:
        text += f" · pass {passes_done or 0}/{passes_total}"

    return text


def _caption_line(file_state: dict) -> str:
    """Build the ``st.caption`` line: stage counter, live elapsed, pass counter."""
    parts: list[str] = []

    stage_index = file_state.get("stage_index")
    stage_total = file_state.get("stage_total")
    if stage_index and stage_total:
        parts.append(f"Stage {stage_index}/{stage_total}")

    started_at = file_state.get("started_at")
    if started_at:
        elapsed = time.time() - started_at
    else:
        elapsed = file_state.get("elapsed")
    if elapsed is not None:
        parts.append(f"elapsed {_format_elapsed(elapsed)}")

    passes_total = file_state.get("passes_total")
    if passes_total:
        parts.append(f"pass {file_state.get('passes_done') or 0}/{passes_total}")

    return " · ".join(parts)


def _checklist_markdown(file_state: dict) -> str:
    """Build the one-line stage checklist, omitting disabled optional stages.

    The optional stages ("reconstruction"/"diarisation") aren't derivable
    from ``file_state`` alone, so the caller may attach the run's config
    snapshot under the ``"_config"`` key (see ``pipeline_invocation.py``'s
    Running fragment); absent that, both are omitted.
    """
    cfg = file_state.get("_config") or {}
    enable_reconstruction = bool(cfg.get("enable_nlp") or cfg.get("enable_llm"))
    enable_diarisation = bool(cfg.get("enable_diarisation"))

    items: list[tuple[str, str]] = [
        ("cleaning", "Audio cleaning"),
        ("transcribing", "Transcribing"),
        ("consensus", "Consensus"),
    ]
    if enable_reconstruction:
        items.append(("reconstruction", "Reconstruction"))
    items.append(("export", "Export"))
    if enable_diarisation:
        items.append(("diarisation", "Diarisation"))

    current_stage = file_state.get("stage")
    # "loading_model" is a brief sub-step of the cleaning bucket for display.
    display_stage = "cleaning" if current_stage == "loading_model" else current_stage
    order = [key for key, _ in items]
    current_pos = order.index(display_stage) if display_stage in order else -1
    is_done = current_stage == "done"

    parts = []
    for i, (key, label) in enumerate(items):
        if is_done or current_pos > i:
            glyph = "✅"
        elif current_pos == i:
            glyph = "🔵"
        else:
            glyph = "⚪"
        if key == "transcribing" and current_pos == i:
            passes_done = file_state.get("passes_done") or 0
            passes_total = file_state.get("passes_total")
            if passes_total:
                label = f"{label} ({passes_done}/{passes_total})"
        parts.append(f"{glyph} {label}")
    return " · ".join(parts)


def render_file_status_panel(file_state: dict) -> None:
    """Render the live status panel for one file inside its expander."""
    progress = min(max(file_state.get("progress") or 0.0, 0.0), 1.0)
    st.progress(progress, text=_status_line(file_state))

    caption = _caption_line(file_state)
    if caption:
        st.caption(caption)

    st.markdown(_checklist_markdown(file_state))

    last_event_at = file_state.get("last_event_at")
    if last_event_at is not None:
        idle_for = time.time() - last_event_at
        if idle_for > _STALE_SECONDS:
            st.warning(
                f"No progress events for {int(idle_for)}s — a long segment "
                "or model load may be in flight."
            )

    history = file_state.get("history") or []
    with st.expander("🕘 History", expanded=False):
        with st.container(height=240):
            if not history:
                st.caption("No stage transitions recorded yet.")
            for stage, detail in reversed(history):
                label = _STAGE_LABELS.get(
                    stage, (stage or "").replace("_", " ").title()
                )
                st.markdown(f"- {label} — {detail}" if detail else f"- {label}")
