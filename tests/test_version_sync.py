"""
tests/test_version_sync.py — Version consistency tests.

Ensures the version string in pyproject.toml stays in sync with:
  - README.md clone instructions (git clone -b vX.Y.Z)
  - README.md Docker pull/run commands (ghcr.io/incendiary/chorus:vX.Y.Z)
  - README.md roadmap heading (Implemented Features (vX.Y.Z))

This test will fail any time pyproject.toml version is bumped without
updating the README, or vice versa.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent


def _get_pyproject_version() -> str:
    """Extract the version string from pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "Could not find version in pyproject.toml"
    return match.group(1)


def _get_readme_text() -> str:
    """Read the full README.md contents."""
    return (ROOT / "README.md").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionSync:
    """Ensure README references match pyproject.toml version."""

    def test_pyproject_version_is_valid_semver(self):
        """Version should be valid semver (X.Y.Z)."""
        version = _get_pyproject_version()
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"pyproject.toml version '{version}' is not valid semver"
        )

    def test_readme_clone_command_matches(self):
        """git clone -b vX.Y.Z should match pyproject version."""
        version = _get_pyproject_version()
        readme = _get_readme_text()
        expected = f"git clone -b v{version}"
        assert expected in readme, (
            f"README.md missing '{expected}'. "
            f"Update the clone command to match pyproject.toml version {version}."
        )

    def test_readme_docker_cpu_pull_matches(self):
        """docker pull ghcr.io/.../chorus:vX.Y.Z should match pyproject version."""
        version = _get_pyproject_version()
        readme = _get_readme_text()
        expected = f"ghcr.io/incendiary/chorus:v{version}"
        assert expected in readme, (
            f"README.md missing '{expected}'. "
            f"Update Docker pull commands to match pyproject.toml version {version}."
        )

    def test_readme_docker_gpu_pull_matches(self):
        """docker pull ghcr.io/.../chorus:vX.Y.Z-gpu should match pyproject version."""
        version = _get_pyproject_version()
        readme = _get_readme_text()
        expected = f"ghcr.io/incendiary/chorus:v{version}-gpu"
        assert expected in readme, (
            f"README.md missing '{expected}'. "
            f"Update GPU Docker pull commands to match pyproject.toml version {version}."
        )

    def test_readme_has_roadmap_heading_for_current_version(self):
        """README should have a roadmap section for the current version."""
        version = _get_pyproject_version()
        readme = _get_readme_text()
        # Match either "### Implemented Features (vX.Y.Z)" or similar heading
        expected = f"(v{version})"
        assert expected in readme, (
            f"README.md missing roadmap section for v{version}. "
            f"Add a '### Implemented Features (v{version})' heading."
        )

    def test_no_stale_version_references(self):
        """
        All vX.Y.Z references in clone/docker commands should be the current version.

        This catches the case where someone bumps pyproject.toml but forgets to
        update one of the README commands.
        """
        version = _get_pyproject_version()
        readme = _get_readme_text()

        # Find all version-tagged refs in clone and docker commands
        # Pattern: commands that reference a specific release version
        clone_refs = re.findall(r"git clone -b (v[\d.]+)", readme)
        docker_pull_refs = re.findall(r"ghcr\.io/incendiary/chorus:(v[\d.]+-?g?p?u?)", readme)

        all_refs = clone_refs + docker_pull_refs
        expected_tag = f"v{version}"

        stale = [
            ref for ref in all_refs
            if not ref.startswith(expected_tag)
        ]
        assert not stale, (
            f"README.md contains stale version references: {stale}. "
            f"All should reference v{version} (from pyproject.toml)."
        )
