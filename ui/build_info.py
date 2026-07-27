"""ui/build_info.py — Identify the build the running app was launched from.

Python imports modules once, at process start, so the code actually serving a
Streamlit session is whatever was on disk at launch — not whatever ``git`` says
now. Pulling or switching branches while the app runs changes the repository
but *not* the running code, which is a genuinely confusing failure mode: the
UI silently keeps serving the old build until it is restarted.

The build identity is therefore captured once at import time (equivalent to
process start) and never refreshed, so the banner always describes the code in
memory rather than the working tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_version() -> str:
    """Read the release version from the root VERSION file."""
    try:
        return (_REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _git(*args: str) -> str | None:
    """Run a read-only git command in the repo, or return None if unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _describe_build() -> str:
    """Compose a short build label: version, branch, commit, and dirty flag.

    Falls back gracefully when git is unavailable (for example inside the
    Docker image, where only the VERSION file ships).
    """
    version = _read_version()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "--short", "HEAD")

    if not commit:
        return f"v{version}"

    label = f"v{version} · {branch or 'detached'} @ {commit}"
    if _git("status", "--porcelain"):
        label += " · uncommitted changes"
    return label


# Captured once, at import — see the module docstring.
BUILD_LABEL = _describe_build()
