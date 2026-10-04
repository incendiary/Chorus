"""Shared utility helpers for the Chorus Engine."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

# Safe filename pattern — only alphanumeric, hyphens, and underscores retained
_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_-]")


def sanitise_stem(raw: str, fallback: str = "audio") -> str:
    """Sanitise a filename stem to safe filesystem characters."""
    sanitised = _SAFE_STEM_RE.sub("_", raw).strip("_")
    return sanitised or fallback


def _sha256_8(path: Path) -> str:
    """Return the first 8 hex characters of *path*'s SHA-256, streamed in
    chunks so large recordings never need to be held in memory whole."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()[:8]


def job_output_dir(
    audio_path: str | Path,
    source_name: str | None = None,
    root: str | Path | None = None,
) -> Path:
    """Return a fresh, never-overwritten per-run output directory.

    Layout: ``<root>/<stem>-<sha8>/<YYYYMMDD-HHMMSS>/``

    ``stem`` is ``sanitise_stem`` applied to *source_name* (or *audio_path*'s
    own name when *source_name* is omitted) — pass the ORIGINAL filename
    when *audio_path* is a spool/temp file with a mangled name, so the
    project directory is named after the recording, not the temp file.

    ``sha8`` is the first 8 hex characters of the SHA-256 of the audio
    bytes: the same name and the same bytes always map to the same
    ``<stem>-<sha8>`` project directory, while two different files that
    happen to share a name never collide. A renamed copy of a recording gets
    a different ``stem`` and so a different project directory.

    The timestamp directory is UTC and gets a ``-2``, ``-3``, … suffix if it
    already exists (two calls within the same second), so a run is never
    overwritten. The returned directory is created before being returned.

    *root* defaults to ``config.JOBS_DIR``.
    """
    audio_path = Path(audio_path)
    if root is None:
        from config import JOBS_DIR

        root = JOBS_DIR
    root = Path(root)

    name_source = source_name if source_name is not None else audio_path.name
    stem = sanitise_stem(Path(name_source).stem, fallback="audio")
    sha8 = _sha256_8(audio_path)
    project_dir = root / f"{stem}-{sha8}"

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    candidate = project_dir / stamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = project_dir / f"{stamp}-{suffix}"

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate
