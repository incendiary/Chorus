"""ui/run_indicator.py — Cross-page in-progress run indicator fragment.

Renders a compact pill when a run is active (running or interrupted), nothing
otherwise. On subpages, includes a link back to the main app to view full
progress. The interrupted state shows a warning with a dismiss button that
clears the state file.
"""

from __future__ import annotations

import streamlit as st

from ui.run_manager import get_run_manager
from ui.run_state import load_state


def render_run_indicator(is_subpage: bool = False) -> None:
    """Render the cross-page run-progress indicator (or nothing if no active run).

    Args:
        is_subpage: Whether this indicator is on a subpage (not app.py).
            When True, a "View progress" link is added after the indicator.
    """
    # Load state without any UI (pure read)
    state = load_state()

    # Render nothing if no active run or run is in a cleared/finished state
    if not state or state.get("status") not in ("running", "interrupted"):
        return

    @st.fragment(run_every=2)
    def _indicator_fragment() -> None:
        """Fragment-polled indicator that updates every 2 seconds."""
        current_state = load_state()
        if not current_state:
            return

        status = current_state.get("status")
        files = current_state.get("files", [])

        if status == "running":
            # Count files and derive progress
            done = sum(1 for f in files if f.get("status") in ("done", "error"))
            total = len(files)
            overall_progress = (
                sum(f.get("progress", 0.0) for f in files) / total if total > 0 else 0.0
            )

            percentage = int(round(overall_progress * 100))
            pill_text = f"⏳ Transcribing {done}/{total} — {percentage}%"

            # Render the pill using the existing .tier-badge-medium pattern
            st.markdown(
                f'<span class="tier-badge-medium">{pill_text}</span>',
                unsafe_allow_html=True,
            )

            if is_subpage:
                st.page_link("app.py", label="View progress", icon="📊")

        elif status == "interrupted":
            # Interrupted run: show warning with dismiss button
            col_warning, col_dismiss = st.columns([10, 1])
            with col_warning:
                st.warning(
                    "A previous run was interrupted. Partial outputs are in Past Jobs.",
                    icon="⚠️",
                )
            with col_dismiss:
                if st.button("✕", key="dismiss_interrupted", use_container_width=True):
                    manager = get_run_manager()
                    manager.clear_finished()
                    st.rerun()

    _indicator_fragment()
