#!/usr/bin/env sh
# check-version-sync.sh
#
# Enforces version alignment across VERSION, pyproject.toml, git tags,
# and GitHub releases.
#
# Usage:
#   ./devops-practices/check-version-sync.sh
#   ./devops-practices/check-version-sync.sh --allow-unreleased

set -eu

ALLOW_UNRELEASED=0
if [ "${1:-}" = "--allow-unreleased" ]; then
  ALLOW_UNRELEASED=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

warn() {
  printf 'WARN: %s\n' "$1" >&2
}

pass() {
  printf 'PASS: %s\n' "$1"
}

VERSION_FILE="$REPO_ROOT/VERSION"
[ -f "$VERSION_FILE" ] || fail "VERSION file not found at repo root"
VERSION=$(tr -d '[:space:]' < "$VERSION_FILE")
printf '%s' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || fail "VERSION is not valid semver: $VERSION"
pass "VERSION is valid semver ($VERSION)"

PYPROJECT_VERSION=$(grep -m1 -E '^[[:space:]]*version[[:space:]]*=' "$REPO_ROOT/pyproject.toml" | sed -E 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/' || true)
[ -n "$PYPROJECT_VERSION" ] || fail "Could not parse version from pyproject.toml"
[ "$PYPROJECT_VERSION" = "$VERSION" ] || fail "VERSION ($VERSION) != pyproject.toml ($PYPROJECT_VERSION)"
pass "VERSION matches pyproject.toml"

LATEST_TAG=$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || true)
if [ -z "$LATEST_TAG" ]; then
  if [ "$ALLOW_UNRELEASED" -eq 1 ]; then
    warn "No git tag found (allowed in unreleased mode)"
  else
    fail "No git tag found; expected v$VERSION"
  fi
elif [ "$LATEST_TAG" != "$VERSION" ]; then
  if [ "$ALLOW_UNRELEASED" -eq 1 ]; then
    warn "VERSION ($VERSION) != latest git tag ($LATEST_TAG) (allowed in unreleased mode)"
  else
    fail "VERSION ($VERSION) != latest git tag ($LATEST_TAG)"
  fi
else
  pass "VERSION matches latest git tag"
fi

if command -v gh >/dev/null 2>&1; then
  RELEASE_TAG=$(gh release view "v$VERSION" --json tagName --jq '.tagName' 2>/dev/null || true)
  if [ "$RELEASE_TAG" = "v$VERSION" ]; then
    pass "GitHub release exists for v$VERSION"
  else
    if [ "$ALLOW_UNRELEASED" -eq 1 ]; then
      warn "GitHub release v$VERSION not found (allowed in unreleased mode)"
    else
      fail "GitHub release v$VERSION not found"
    fi
  fi
else
  warn "gh CLI not available; skipped GitHub release check"
fi

printf 'All version sync checks completed.\n'
