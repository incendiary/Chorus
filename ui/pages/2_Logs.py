"""ui/pages/2_Logs.py — In-app log viewer (disk-backed + live session buffer)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import OUTPUTS_DIR  # noqa: E402
from ui.run_indicator import render_run_indicator  # noqa: E402
from ui.run_state import RUNS_DIR, load_state  # noqa: E402

st.set_page_config(
    page_title="Logs — Chorus",
    page_icon="📋",
    layout="wide",
)

render_run_indicator(is_subpage=True)

st.title("📋 Logs")

_LOG_BUFFER_KEY = "log_buffer"
_LEVEL_COLOURS = {
    "ERROR": "🔴",
    "WARNING": "🟡",
    "INFO": "🔵",
    "DEBUG": "⚪",
}

# Parse a log line: "HH:MM:SS  LEVEL    logger — message"
_LOG_LINE_PATTERN = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s{2}(\w+)-?\s*(.+?)\s—\s(.*)$"
)


def _parse_log_line(line: str) -> dict | None:
    """Parse a single log line into (timestamp, level, logger, message)."""
    match = _LOG_LINE_PATTERN.match(line.strip())
    if not match:
        return None
    time_str, level, logger, message = match.groups()
    return {
        "time": time_str,
        "level": level.strip(),
        "logger": logger.strip(),
        "message": message.strip(),
    }


def _find_run_logs() -> dict[str, Path]:
    """Find all on-disk run logs, keyed by run_id (newest first)."""
    if not RUNS_DIR.exists():
        return {}

    logs = {}
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir():
            log_file = run_dir / "run.log"
            if log_file.exists():
                logs[run_dir.name] = log_file

    return dict(
        sorted(logs.items(), key=lambda x: x[1].stat().st_mtime, reverse=True)
    )


def _load_and_parse_log(log_path: Path) -> list[dict]:
    """Load and parse a log file, returning list of parsed records."""
    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        records = [_parse_log_line(line) for line in lines if line.strip()]
        return [r for r in records if r is not None]
    except Exception:
        return []


def _strip_timestamp(line: str) -> str:
    """Strip leading HH:MM:SS timestamp from a log line."""
    # Format: "HH:MM:SS  LEVEL    logger — message"
    # Return just the "LEVEL logger — message" part
    match = re.match(r"^\d{2}:\d{2}:\d{2}\s{2}", line)
    if match:
        return line[match.end():]
    return line


def _render_disk_logs() -> None:
    """Render the disk-backed log viewer with dedup and newest-first order."""
    run_logs = _find_run_logs()

    if not run_logs:
        st.info("No run logs found yet. Run a transcription and check back here.")
        return

    # Determine default selection: active run if running, else newest
    state = load_state()
    if state and state.get("status") == "running":
        default_run_id = state.get("run_id")
        if default_run_id not in run_logs:
            default_run_id = None
    else:
        default_run_id = None

    default_idx = 0
    run_ids = list(run_logs.keys())
    if default_run_id and default_run_id in run_ids:
        default_idx = run_ids.index(default_run_id)

    col_select, col_tail, col_download = st.columns([2, 1, 1])

    with col_select:
        selected_run_id = st.selectbox(
            "Select run log",
            options=run_ids,
            index=default_idx,
            label_visibility="collapsed",
            format_func=lambda rid: f"{rid} ({run_logs[rid].stat().st_mtime})",
        )

    with col_tail:
        tail_n = st.number_input(
            "Last N entries",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            label_visibility="collapsed",
            help="Show only the last N log entries.",
        )

    selected_log = run_logs[selected_run_id]
    records = _load_and_parse_log(selected_log)

    # Apply tail window (newest N entries only).
    records = records[-int(tail_n):]

    # Deduplicate consecutive identical lines (timestamp-stripped comparison).
    # Keeps the most recent timestamp and accumulates a repeat count.
    deduped: list[tuple[dict, int]] = []
    for record in reversed(records):
        if deduped:
            prev_record = deduped[-1][0]
            # Compare timestamp-stripped versions
            prev_stripped = _strip_timestamp(
                f"{prev_record['time']}  {prev_record['level']:<8}  "
                f"{prev_record['logger']} — {prev_record['message']}"
            )
            curr_stripped = _strip_timestamp(
                f"{record['time']}  {record['level']:<8}  "
                f"{record['logger']} — {record['message']}"
            )

            if prev_stripped == curr_stripped:
                deduped[-1] = (record, deduped[-1][1] + 1)
            else:
                deduped.append((record, 1))
        else:
            deduped.append((record, 1))

    with col_download:
        if deduped:
            log_text = "\n".join(
                f"{r['time']}  {r['level']:<8}  {r['logger']} — {r['message']}"
                for r, _ in deduped
            )
            st.download_button(
                "⬇ Download",
                data=log_text,
                file_name=f"chorus_logs_{selected_run_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("⬇ Download", disabled=True, use_container_width=True)

    st.divider()

    # Render inside a bounded scrollable container
    if not deduped:
        st.info(
            f"No log entries for run {selected_run_id}."
        )
    else:
        with st.container(height=400):
            for record, count in deduped:
                icon = _LEVEL_COLOURS.get(record["level"], "⚪")
                repeat = f" ×{count}" if count > 1 else ""
                st.markdown(
                    f"`{record['time']}` {icon} **{record['level']}** "
                    f"`{record['logger']}` — {record['message']}{repeat}"
                )


def _render_session_logs() -> None:
    """Render the session-state live log buffer (secondary view)."""
    buf: list[dict] = st.session_state.get(_LOG_BUFFER_KEY, [])

    col_filter, col_tail, col_clear, col_download = st.columns([2, 1, 1, 1])

    with col_filter:
        level_filter = st.selectbox(
            "Filter by level",
            options=["ALL", "INFO", "WARNING", "ERROR"],
            index=0,
            label_visibility="collapsed",
        )

    with col_tail:
        tail_n = st.number_input(
            "Last N entries",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            label_visibility="collapsed",
            help="Show only the last N log entries.",
        )

    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state[_LOG_BUFFER_KEY] = []
            st.rerun()

    filtered = (
        buf if level_filter == "ALL" else [r for r in buf if r["level"] == level_filter]
    )

    # Apply tail window (most-recent N entries).
    filtered = filtered[-int(tail_n):]

    with col_download:
        if filtered:
            log_text = "\n".join(
                f"{r['time']}  {r['level']:<8}  {r['logger']} — {r['message']}"
                for r in filtered
            )
            st.download_button(
                "⬇ Download",
                data=log_text,
                file_name="chorus_logs_session.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("⬇ Download", disabled=True, use_container_width=True)

    st.divider()

    if not filtered:
        st.info(
            "No log entries yet. Run a transcription and check back here."
            if not buf
            else f"No {level_filter} entries in the current buffer."
        )
    else:
        # Deduplicate consecutive identical messages (same level + message text).
        # Keeps the most recent timestamp and accumulates a repeat count.
        deduped: list[tuple[dict, int]] = []
        for record in reversed(filtered):
            if (
                deduped
                and deduped[-1][0]["level"] == record["level"]
                and deduped[-1][0]["message"] == record["message"]
            ):
                deduped[-1] = (record, deduped[-1][1] + 1)
            else:
                deduped.append((record, 1))

        for record, count in deduped:
            icon = _LEVEL_COLOURS.get(record["level"], "⚪")
            repeat = f" ×{count}" if count > 1 else ""
            st.markdown(
                f"`{record['time']}` {icon} **{record['level']}** "
                f"`{record['logger']}` — {record['message']}{repeat}"
            )


# Auto-refresh when a run is active
@st.fragment(run_every=2)
def _disk_logs_auto_refresh() -> None:
    """Fragment for disk logs with auto-refresh while run is active."""
    _render_disk_logs()


# Determine if we should auto-refresh
state = load_state()
is_running = state and state.get("status") == "running"

# Create tabs: "Run Logs" (disk-backed) and "This Session" (live buffer)
tab_disk, tab_session = st.tabs(["📂 Run Logs", "🔄 This Session"])

with tab_disk:
    if is_running:
        _disk_logs_auto_refresh()
    else:
        _render_disk_logs()

with tab_session:
    _render_session_logs()
