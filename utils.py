"""Shared utility helpers for the Chorus Engine."""

from __future__ import annotations

import re

# Safe filename pattern — only alphanumeric, hyphens, and underscores retained
_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_-]")


def sanitise_stem(raw: str, fallback: str = "audio") -> str:
    """Sanitise a filename stem to safe filesystem characters."""
    sanitised = _SAFE_STEM_RE.sub("_", raw).strip("_")
    return sanitised or fallback
