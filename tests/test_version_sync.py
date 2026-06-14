"""
tests/test_version_sync.py — Version and release consistency tests.

Ensures the version string in pyproject.toml stays in sync with:
    - README.md clone instructions (git clone -b vX.Y.Z)
    - README.md Docker pull/run commands (ghcr.io/incendiary/chorus:vX.Y.Z)
    - Git tags (latest tag should match pyproject.toml version)
    - ROADMAP.md completion metadata

Also enforces roadmap ownership:
    - README.md must link to ROADMAP.md
    - roadmap item lists belong in ROADMAP.md, not README.md
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


def _parse_semver(version: str) -> tuple[int, int, int]:
    """Parse X.Y.Z into comparable integer tuple."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    assert match, f"Invalid semver: {version}"
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


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


def _get_roadmap_text() -> str:
    """Read the full ROADMAP.md contents."""
    roadmap_path = ROOT / "ROADMAP.md"
    if not roadmap_path.exists():
        return ""
    return roadmap_path.read_text(encoding="utf-8")


def _get_completed_items_by_version() -> dict[str, list[str]]:
    """
    Parse ROADMAP.md to extract completed items grouped by version.

    Returns a dict mapping version (e.g., "2.0.1") to a list of completed item
    descriptions. Only items marked with [x] and containing (vX.Y.Z) are counted.
    """
    roadmap = _get_roadmap_text()
    completed: dict[str, list[str]] = {}

    pattern = r'-\s*\[x\]\s+\*\*([^*]+)\*\*\s+\(v([\d.]+)\)'
    matches = re.findall(pattern, roadmap)

    for description, version in matches:
        if version not in completed:
            completed[version] = []
        completed[version].append(description.strip())

    return completed

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

    def test_readme_links_to_roadmap(self):
        """README must contain a clickable link to ROADMAP.md."""
        readme = _get_readme_text()
        assert "[ROADMAP.md](ROADMAP.md)" in readme, (
            "README.md must link to ROADMAP.md using '[ROADMAP.md](ROADMAP.md)'."
        )

    def test_readme_does_not_embed_roadmap_items(self):
        """Roadmap item lists must live in ROADMAP.md, not README.md."""
        readme = _get_readme_text()
        assert "### Implemented Features (v" not in readme, (
            "README.md contains versioned roadmap headings. Move roadmap lists to ROADMAP.md."
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

    def test_current_version_has_git_tag_or_is_unreleased(self):
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
        if expected_tag in tags:
            return

        # Allow active development on a newer, not-yet-tagged version.
        latest_tag_version = tags[-1].lstrip("v")
        assert _parse_semver(version) > _parse_semver(latest_tag_version), (
            f"Current version {version} is not tagged and is not ahead of latest tag {latest_tag_version}. "
            f"Either tag v{version}, or set pyproject.toml to a newer unreleased version."
        )

    def test_latest_git_tag_matches_pyproject_or_is_previous_release(self):
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

        if latest_tag == expected_tag:
            return

        latest_tag_version = latest_tag.lstrip("v")
        assert _parse_semver(version) > _parse_semver(latest_tag_version), (
            f"Latest git tag '{latest_tag}' and pyproject version 'v{version}' are inconsistent. "
            f"Expected pyproject version to be equal to the latest tag, or a newer unreleased version."
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


class TestRoadmapSync:
    """Ensure ROADMAP.md completions are marked for implemented versions."""

    def test_roadmap_exists(self):
        """ROADMAP.md should exist for tracking feature completions."""
        roadmap_path = ROOT / "ROADMAP.md"
        assert roadmap_path.exists(), (
            "ROADMAP.md not found. Create a ROADMAP.md to track feature completions."
        )

    def test_roadmap_contains_checklist_items(self):
        """ROADMAP.md should contain at least one checklist item."""
        roadmap = _get_roadmap_text()
        if not roadmap:
            pytest.skip("ROADMAP.md is empty")

        has_item = bool(re.search(r"-\s*\[(?:x| )\]\s+", roadmap, re.IGNORECASE))
        assert has_item, "ROADMAP.md contains no checklist items. Add tracked roadmap entries."

    def test_current_version_items_are_marked_completed(self):
        """
        All roadmap items for the current pyproject.toml version should be marked [x].

        When a version is released, any completed items must be marked with [x]
        and tagged with the version (vX.Y.Z). This ensures the roadmap reflects
        what was actually delivered in each release.
        """
        version = _get_pyproject_version()
        roadmap = _get_roadmap_text()

        if not roadmap:
            pytest.skip("ROADMAP.md is empty")

        # Look for uncompleted items tagged with the current version
        # Pattern: - [ ] **Description** (vX.Y.Z)
        pattern = rf'-\s*\[\s*\]\s+\*\*([^*]+)\*\*\s+\(v{re.escape(version)}\)'
        incomplete_items = re.findall(pattern, roadmap)

        assert not incomplete_items, (
            f"ROADMAP.md has incomplete items tagged for v{version}. "
            f"Mark them as completed with [x] or remove the version tag:\n"
            f"  - {chr(10).join(incomplete_items)}"
        )

    def test_completed_items_have_version_tags(self):
        """
        All completed items should have a version tag indicating when they were done.

        This ensures the roadmap clearly documents the history of features and
        helps trace improvements to specific releases.
        """
        roadmap = _get_roadmap_text()

        if not roadmap:
            pytest.skip("ROADMAP.md is empty")

        # Find items marked [x] that don't have a version tag
        # Use a simpler approach: find completed items without (vX.Y.Z) nearby
        lines = roadmap.split('\n')
        problematic = []
        for line in lines:
            if re.search(r'-\s*\[x\]', line) and not re.search(r'\(v\d+\.\d+\.\d+\)', line):
                # This is a completed item without a version tag
                item_match = re.search(r'\*\*([^*]+)\*\*', line)
                if item_match:
                    problematic.append(item_match.group(1).strip())

        assert not problematic, (
            f"Completed items in ROADMAP.md are missing version tags. "
            f"Add (vX.Y.Z) to indicate when each was implemented:\n"
            f"  - {chr(10).join(problematic)}"
        )

    def test_roadmap_versions_match_git_tags(self):
        """
        All versions mentioned in ROADMAP.md should have corresponding git tags.

        This catches cases where the roadmap documents a completion for a version
        that hasn't been released yet.
        """
        completed_by_version = _get_completed_items_by_version()
        tags = _get_git_tags()

        if not tags or not completed_by_version:
            pytest.skip("Not enough git tags or roadmap items to validate")

        tag_versions = set(t.lstrip("v") for t in tags)
        roadmap_versions = set(completed_by_version.keys())

        current_version = _get_pyproject_version()
        # Allow current in-flight version to exist in roadmap before tag creation.
        allowed_missing = {current_version}
        missing_tags = roadmap_versions - tag_versions - allowed_missing
        assert not missing_tags, (
            f"ROADMAP.md references completed versions that don't have git tags: {sorted(missing_tags)}. "
            f"Either create git tags for these versions or remove the version annotations from ROADMAP.md."
        )
