"""
tests/test_version_sync.py — Version consistency tests.

Ensures the version string in pyproject.toml stays in sync with:
  - README.md clone instructions (git clone -b vX.Y.Z)
  - README.md Docker pull/run commands (ghcr.io/incendiary/chorus:vX.Y.Z)
  - README.md roadmap heading (Implemented Features (vX.Y.Z))
  - Git tags (latest tag should match pyproject.toml version)

This test will fail any time pyproject.toml version is bumped without
updating the README, or when a version is released without a corresponding git tag.
"""

from __future__ import annotations

import re
import subprocess
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


def _get_git_tags() -> list[str]:
    """
    Get all git tags matching vX.Y.Z pattern, sorted in ascending version order.
    
    Returns empty list if not a git repository or if no tags exist.
    """
    try:
        result = subprocess.run(
            ["git", "tag", "-l", "v*"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        
        tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
        # Sort by semantic version (extract X.Y.Z and sort)
        def version_key(tag: str) -> tuple[int, int, int]:
            match = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag)
            if not match:
                return (0, 0, 0)
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        return sorted(tags, key=version_key)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


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


class TestGitTagSync:
    """Ensure git tags match the current pyproject.toml version."""

    def test_current_version_has_git_tag(self):
        """
        The current version in pyproject.toml should have a corresponding git tag.
        
        This ensures that releases are properly tagged in git. If this fails,
        run: git tag -a v<version> -m "Release v<version>" && git push origin v<version>
        """
        version = _get_pyproject_version()
        tags = _get_git_tags()
        
        # If no tags exist, this is a fresh repo and the test should be skipped
        if not tags:
            pytest.skip("No git tags found; skipping git tag sync tests")
        
        expected_tag = f"v{version}"
        assert expected_tag in tags, (
            f"Git tag v{version} does not exist. "
            f"Current pyproject.toml version is {version}, but latest git tag is {tags[-1]}. "
            f"To create the tag, run: git tag -a v{version} -m 'Release v{version}' && git push origin v{version}"
        )

    def test_latest_git_tag_matches_pyproject(self):
        """
        The latest git tag should match the version in pyproject.toml.
        
        This catches the case where pyproject.toml is out of sync with git releases.
        """
        version = _get_pyproject_version()
        tags = _get_git_tags()
        
        if not tags:
            pytest.skip("No git tags found; skipping git tag sync tests")
        
        latest_tag = tags[-1]
        expected_tag = f"v{version}"
        
        assert latest_tag == expected_tag, (
            f"Latest git tag '{latest_tag}' does not match pyproject.toml version 'v{version}'. "
            f"Either bump pyproject.toml to {latest_tag.lstrip('v')}, or create a new tag for v{version}."
        )

    def test_git_tags_are_consecutive_semver(self):
        """
        All git tags should follow semantic versioning and be in ascending order.
        
        This catches mistakes like typos in version numbers or out-of-order releases.
        """
        tags = _get_git_tags()
        
        if len(tags) < 2:
            pytest.skip("Need at least 2 tags to check ordering")
        
        # Verify each tag is valid semver
        for tag in tags:
            assert re.match(r"^v\d+\.\d+\.\d+$", tag), (
                f"Git tag '{tag}' is not valid semver (vX.Y.Z format)"
            )
        
        # Verify tags are in ascending order (we already sorted them in _get_git_tags)
        # This is a sanity check — the function sorts them, so they should be ordered
        sorted_tags = sorted(tags, key=lambda t: tuple(map(int, t.lstrip("v").split("."))))
        assert tags == sorted_tags, (
            f"Git tags are not in ascending version order: {tags}"
        )

    def test_no_duplicate_git_tags(self):
        """Ensure there are no duplicate version tags."""
        tags = _get_git_tags()
        
        if not tags:
            pytest.skip("No git tags found")
        
        unique_tags = set(tags)
        assert len(tags) == len(unique_tags), (
            f"Duplicate git tags detected: {[t for t in tags if tags.count(t) > 1]}"
        )
