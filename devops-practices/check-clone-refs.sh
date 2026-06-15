#!/usr/bin/env sh
# check-clone-refs.sh
#
# Enforces pinning by rejecting @main/@master refs in user-facing docs and
# GitHub Actions workflow references.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

pass() {
  printf 'PASS: %s\n' "$1"
}

TMP_FILE=$(mktemp)
trap 'rm -f "$TMP_FILE"' EXIT

# Docs and README references
DOC_TARGETS="$REPO_ROOT/README.md"
if [ -d "$REPO_ROOT/docs" ]; then
  DOC_TARGETS="$DOC_TARGETS $REPO_ROOT/docs"
fi

# shellcheck disable=SC2086
grep -RInE '@(main|master)\b' $DOC_TARGETS 2>/dev/null > "$TMP_FILE" || true

# Workflow action pinning
if [ -d "$REPO_ROOT/.github/workflows" ]; then
  grep -RInE 'uses:[[:space:]]*[^#]+@(main|master)\b' "$REPO_ROOT/.github/workflows" 2>/dev/null >> "$TMP_FILE" || true
fi

if [ -s "$TMP_FILE" ]; then
  printf 'Unpinned refs found:\n' >&2
  sed 's/^/  /' "$TMP_FILE" >&2
  fail "Found @main/@master references"
fi

pass "No @main/@master refs found in docs/workflows"
