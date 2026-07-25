"""ui/run_state.py — Background-run state schema, atomic I/O, and path constants.

This module is the pure-data half of the background-run feature: dataclasses
describing a run job and its per-file state, plus atomic read/write helpers
for the on-disk state file that the UI polls to render live progress after a
tab refresh or close. It has no Streamlit dependency and is safe to import
under plain pytest.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR

SCHEMA_VERSION = 1

ACTIVE_RUN_FILE: Path = OUTPUTS_DIR / "active_run.json"
RUNS_DIR: Path = OUTPUTS_DIR / "runs"

# History is a ring buffer of (stage, detail) transitions only — segment
# ticks never append (that is the dedup that keeps it bounded).
HISTORY_MAX = 50


@dataclass
class FileEntry:
    """Per-file state tracked across a background run.

    ``stage``/``stage_index``/``stage_total``/``detail``/``passes_done``/
    ``passes_total``/``segment``/``segments_total``/``parallel_workers``/
    ``last_event_at``/``history`` are populated from the WP1a pipeline
    events emitted by ``run_pipeline(..., event_callback=...)``.
    """

    name: str
    stem: str
    spool_path: str
    status: str = "pending"  # pending | running | done | error
    progress: float = 0.0
    label: str | None = None
    elapsed: float | None = None
    error: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)

    # WP1a pipeline event fields
    stage: str | None = None
    stage_index: int | None = None
    stage_total: int | None = None
    detail: str | None = None
    passes_done: int | None = None
    passes_total: int | None = None
    segment: int | None = None
    segments_total: int | None = None
    parallel_workers: int | None = None
    last_event_at: float | None = None
    history: list[list[Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileEntry:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunJob:
    """A run to execute: a config snapshot plus the files it covers."""

    run_id: str
    config: dict[str, Any]
    files: list[FileEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "files": [entry.to_dict() for entry in self.files],
        }


def append_history(
    entry: FileEntry | dict[str, Any], stage: str, detail: str | None
) -> None:
    """Append a (stage, detail) transition to *entry*'s history ring buffer.

    Accepts either a ``FileEntry`` or its serialised ``dict`` form (the
    state file stores plain dicts). Callers must only invoke this when
    ``(stage, detail)`` has actually changed since the last transition —
    segment ticks must never grow history. Capped at ``HISTORY_MAX``,
    dropping the oldest entries first.
    """
    history = (
        entry.history
        if isinstance(entry, FileEntry)
        else entry.setdefault("history", [])
    )
    history.append([stage, detail])
    if len(history) > HISTORY_MAX:
        del history[: len(history) - HISTORY_MAX]


def new_state(
    run_id: str,
    boot_id: str,
    config_snapshot: dict[str, Any],
    files: list[FileEntry],
    started_at: float,
) -> dict[str, Any]:
    """Build a fresh ``running`` state dict for a new run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "boot_id": boot_id,
        "started_at": started_at,
        "finished_at": None,
        "config": config_snapshot,
        "log_path": None,
        "files": [entry.to_dict() for entry in files],
    }


def write_state_atomic(state: dict[str, Any]) -> None:
    """Write *state* to ``ACTIVE_RUN_FILE`` atomically.

    Writes to a temp file in the same directory, then ``os.replace``s it
    into place, so a concurrent reader never observes a partial write.
    """
    ACTIVE_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(ACTIVE_RUN_FILE.parent), prefix=".active_run_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp_name, ACTIVE_RUN_FILE)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def load_state() -> dict[str, Any] | None:
    """Load the active run state, or ``None`` if absent/unreadable."""
    if not ACTIVE_RUN_FILE.exists():
        return None
    try:
        with open(ACTIVE_RUN_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
