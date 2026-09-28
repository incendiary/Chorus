"""batch_processor/progress_display.py — terminal progress bars for batch runs.

Renders two ``tqdm`` bars pinned at the bottom of the terminal while
``run_batch`` (see ``batch_processor/batch_runner.py``) works through a list
of files: an OVERALL bar tracking the whole batch, and a CURRENT TASK bar
tracking the active pipeline stage within the file being processed right
now. Without this, a long file (a 97-minute recording, say) sits with no
visible progress between log lines.

The event interpretation mirrors ``ui/run_worker.py``'s ``_event_cb``: both
read the same ``event_callback`` dict shape emitted by
``pipeline_runner.run_pipeline`` (stage, detail, frac, passes_done/total,
segment/segments_total).

Bars render only when stderr is a TTY (or when explicitly forced for
testing), so a redirected/piped run, CI, and the log file are completely
unaffected. Log lines are kept visible above the bars via
``tqdm.contrib.logging.logging_redirect_tqdm``.
"""

from __future__ import annotations

import sys

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Human labels for the stages emitted by pipeline_runner.active_stages().
STAGE_LABELS = {
    "cleaning": "Cleaning audio",
    "loading_model": "Loading model",
    "transcribing": "Transcribing",
    "consensus": "Building consensus",
    "reconstruction": "Reconstructing",
    "export": "Exporting",
    "diarisation": "Diarising",
    "done": "Done",
}

# tqdm's internal resolution for the current-task bar. Fractions from
# pipeline events (0.0-1.0) are scaled onto this range so partial progress
# within a stage is visible.
_TASK_BAR_RESOLUTION = 1000


def stage_label(event: dict) -> str:
    """Render a human label for a single pipeline event.

    Pure function of the event dict, kept separate from any tqdm object so
    the event-to-label mapping can be tested without a terminal.
    """
    stage = event.get("stage")
    label = STAGE_LABELS.get(stage, stage or "")
    passes_done = event.get("passes_done")
    passes_total = event.get("passes_total")
    segment = event.get("segment")
    segments_total = event.get("segments_total")

    if stage == "transcribing" and passes_done is not None and passes_total is not None:
        detail = f"pass {passes_done}/{passes_total}"
        if segment is not None and segments_total is not None:
            detail = f"{detail}, segment {segment}/{segments_total}"
        label = f"{label} ({detail})"
    elif stage == "diarisation" and event.get("detail"):
        label = f"{label} ({event['detail']})"

    return label


def overall_fraction(
    files_completed: int, total_files: int, current_file_frac: float
) -> float:
    """Fraction of the whole batch that is done.

    ``current_file_frac`` is the active file's own 0.0-1.0 progress (from a
    pipeline event's ``frac``), so a file half-way through counts as half a
    file towards the total. Pure function, no tqdm dependency, so the
    arithmetic can be tested directly.
    """
    if total_files <= 0:
        return 0.0
    frac = (files_completed + max(0.0, min(1.0, current_file_frac))) / total_files
    return max(0.0, min(1.0, frac))


class BatchProgressBars:
    """Owns the two tqdm bars (or does nothing, when disabled).

    ``enabled`` decides once, at construction, whether bars are actually
    drawn. When disabled, every method is a no-op, so callers never need to
    branch on whether progress display is active.
    """

    def __init__(self, total_files: int, *, enabled: bool):
        self.enabled = enabled
        self.total_files = total_files
        self.files_completed = 0
        self._current_index = 0
        self._current_filename = ""
        self._log_redirect = None
        self.overall = None
        self.task = None
        self._diarisation_start: float | None = None

        if not self.enabled:
            return

        self._log_redirect = logging_redirect_tqdm()
        self._log_redirect.__enter__()
        self.overall = tqdm(
            total=total_files,
            position=1,
            desc="Overall",
            unit="file",
            file=sys.stderr,
            leave=True,
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        self.task = tqdm(
            total=_TASK_BAR_RESOLUTION,
            position=0,
            desc="Current task",
            file=sys.stderr,
            leave=True,
            bar_format="{desc}: {bar} {postfix}",
        )

    def start_file(self, idx: int, total: int, filename: str) -> None:
        """Called once a file's processing begins."""
        self._current_index = idx
        self._current_filename = filename
        self._diarisation_start = None
        if not self.enabled:
            return
        self.overall.set_description_str(f"Overall  [{idx}/{total}] {filename}")
        self.task.n = 0
        self.task.total = _TASK_BAR_RESOLUTION
        self.task.set_postfix_str("starting")
        self.task.refresh()

    def on_event(self, event: dict) -> None:
        """Called for each ``pipeline_runner`` event during the current file.

        Diarisation has no real fraction to show (pyannote gives no
        sub-step hook — see ``diarisation/diariser.py``'s heartbeat), so
        its events switch the task bar into an indeterminate mode (no
        total), refreshed by the elapsed time carried in ``detail`` rather
        than a fraction it cannot provide.
        """
        if not self.enabled:
            return
        frac = event.get("frac") or 0.0
        if event.get("stage") == "diarisation":
            self.task.total = None
        else:
            self.task.total = _TASK_BAR_RESOLUTION
            self.task.n = int(max(0.0, min(1.0, frac)) * _TASK_BAR_RESOLUTION)
        self.task.set_postfix_str(stage_label(event))
        self.task.refresh()
        self._update_overall(frac)

    def _update_overall(self, current_file_frac: float) -> None:
        frac = overall_fraction(
            self.files_completed, self.total_files, current_file_frac
        )
        self.overall.n = frac * self.total_files
        self.overall.refresh()

    def finish_file(self) -> None:
        """Called once a file (success or failure) is fully processed."""
        self.files_completed += 1
        if not self.enabled:
            return
        self.overall.n = self.files_completed
        self.overall.refresh()

    def close(self) -> None:
        if not self.enabled:
            return
        self.task.close()
        self.overall.close()
        if self._log_redirect is not None:
            self._log_redirect.__exit__(None, None, None)
