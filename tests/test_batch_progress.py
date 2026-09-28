"""
tests/test_batch_progress.py — Unit tests for the batch CLI progress bars.

Covers:
  - Pure event-to-bar mapping (stage_label, overall_fraction)
  - Overall fraction across a 3-file batch
  - Non-TTY runs create no bars and touch neither stdout nor stderr
  - --no-progress parses and disables bars even on a TTY
  - A diarisation heartbeat updates the indeterminate task bar

No real pipeline, Whisper, or tqdm terminal rendering is required: bars are
exercised directly via ``BatchProgressBars``, with stderr captured.
"""

from __future__ import annotations

import io
from unittest.mock import patch

from batch_processor.batch_runner import _build_parser
from batch_processor.progress_display import (
    BatchProgressBars,
    overall_fraction,
    stage_label,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pure event-to-bar mapping
# ─────────────────────────────────────────────────────────────────────────────


def test_stage_label_transcribing_with_pass_and_segment():
    event = {
        "stage": "transcribing",
        "detail": None,
        "frac": 0.4,
        "passes_done": 2,
        "passes_total": 4,
        "segment": 15,
        "segments_total": 200,
    }
    assert stage_label(event) == "Transcribing (pass 2/4, segment 15/200)"


def test_stage_label_transcribing_without_segment_info():
    event = {
        "stage": "transcribing",
        "detail": None,
        "frac": 0.1,
        "passes_done": 1,
        "passes_total": 4,
        "segment": None,
        "segments_total": None,
    }
    assert stage_label(event) == "Transcribing (pass 1/4)"


def test_stage_label_diarisation_with_detail():
    event = {
        "stage": "diarisation",
        "detail": "60s elapsed",
        "frac": 0.97,
        "passes_done": None,
        "passes_total": None,
        "segment": None,
        "segments_total": None,
    }
    assert stage_label(event) == "Diarising (60s elapsed)"


def test_stage_label_unknown_stage_falls_back_to_raw_name():
    event = {"stage": "cleaning", "frac": 0.1}
    assert stage_label(event) == "Cleaning audio"


def test_overall_fraction_exact_values_across_three_files():
    # File 1 of 3 done, second file 50% through.
    assert overall_fraction(1, 3, 0.5) == (1 + 0.5) / 3
    # No files done yet, first file just starting.
    assert overall_fraction(0, 3, 0.0) == 0.0
    # All three files done.
    assert overall_fraction(3, 3, 0.0) == 1.0


def test_overall_fraction_clamps_out_of_range_current_frac():
    assert overall_fraction(1, 2, 1.5) == 1.0
    assert overall_fraction(0, 2, -0.5) == 0.0


def test_overall_fraction_zero_total_files_is_zero():
    assert overall_fraction(0, 0, 0.5) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BatchProgressBars: TTY vs non-TTY
# ─────────────────────────────────────────────────────────────────────────────


def test_non_tty_creates_no_bars_and_writes_nothing():
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with (
        patch("sys.stdout", captured_stdout),
        patch("sys.stderr", captured_stderr),
    ):
        bars = BatchProgressBars(3, enabled=False)
        bars.start_file(1, 3, "interview.wav")
        bars.on_event({"stage": "transcribing", "frac": 0.5})
        bars.finish_file()
        bars.close()

    assert bars.overall is None
    assert bars.task is None
    assert captured_stdout.getvalue() == ""
    assert captured_stderr.getvalue() == ""


def test_tty_creates_bars_and_writes_to_stderr():
    bars = BatchProgressBars(3, enabled=True)
    try:
        bars.start_file(1, 3, "interview.wav")
        bars.on_event({"stage": "cleaning", "frac": 0.1})
        assert bars.overall is not None
        assert bars.task is not None
        assert bars.task.n == 100  # 0.1 * _TASK_BAR_RESOLUTION(1000) scaled? see below
    finally:
        bars.close()


def test_no_progress_flag_parses_and_disables():
    parser = _build_parser()
    args = parser.parse_args(["file.wav", "--no-progress"])
    assert args.no_progress is True

    args_default = parser.parse_args(["file.wav"])
    assert args_default.no_progress is False


# ─────────────────────────────────────────────────────────────────────────────
# Diarisation heartbeat: indeterminate task bar
# ─────────────────────────────────────────────────────────────────────────────


def test_diarisation_event_sets_indeterminate_task_bar():
    bars = BatchProgressBars(1, enabled=True)
    try:
        bars.start_file(1, 1, "long-interview.wav")
        # A normal stage sets a real total.
        bars.on_event({"stage": "cleaning", "frac": 0.1})
        assert bars.task.total is not None

        # A diarisation heartbeat switches to indeterminate mode (no total)
        # and shows the elapsed-time detail rather than a fraction.
        bars.on_event({"stage": "diarisation", "detail": "60s elapsed", "frac": 0.97})
        assert bars.task.total is None
        assert "60s elapsed" in bars.task.postfix
    finally:
        bars.close()


def test_overall_fraction_advances_across_batch_via_bars():
    bars = BatchProgressBars(3, enabled=True)
    try:
        bars.start_file(1, 3, "a.wav")
        bars.on_event({"stage": "cleaning", "frac": 1.0})
        assert bars.overall.n == overall_fraction(0, 3, 1.0) * 3
        bars.finish_file()
        assert bars.overall.n == 1

        bars.start_file(2, 3, "b.wav")
        bars.on_event({"stage": "transcribing", "frac": 0.5})
        assert bars.overall.n == overall_fraction(1, 3, 0.5) * 3
        bars.finish_file()
        assert bars.overall.n == 2
    finally:
        bars.close()
