#!/usr/bin/env bash
# tests/check_dependency_drift.sh
#
# Enforces dependency version consistency between requirements.txt and pyproject.toml.
#
# Both files use exact pins (package==version). This check extracts all package==version
# pairs from each file and verifies that packages appearing in both files have identical
# version pins. Packages may be present in only one file (e.g., dev-only dependencies
# in pyproject.toml[project.optional-dependencies.dev]); such packages do not trigger
# a failure.
#
# Exit code: 0 if all overlapping packages match, 1 if any drift is detected.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILURES=0
TOTAL=0

red()    { printf '\033[0;31m%s\033[0m\n' "$1"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }
info()   { printf '  %-70s' "$1"; TOTAL=$((TOTAL + 1)); }

fail() { red "FAIL"; FAILURES=$((FAILURES + 1)); }
pass() { green "PASS"; }

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║   Dependency Drift Check: requirements.txt ↔ pyproject ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Create temp files for parsed deps.
REQ_FILE=$(mktemp)
PYPROJ_FILE=$(mktemp)
trap "rm -f $REQ_FILE $PYPROJ_FILE" EXIT

# Extract package==version pairs from requirements.txt.
grep -E '^[a-zA-Z0-9._-]+==' "$REPO_ROOT/requirements.txt" | sort > "$REQ_FILE" 2>/dev/null || true

# Extract package==version pairs from pyproject.toml's [project].dependencies.
sed -n '/^\[project\]$/,/^\[/p' "$REPO_ROOT/pyproject.toml" | \
  grep -E '^\s*"[a-zA-Z0-9._-]+==' | \
  sed 's/^[[:space:]]*"//; s/"[,]*[[:space:]]*$//' | \
  sort > "$PYPROJ_FILE" 2>/dev/null || true

# Check for version mismatches in packages appearing in both files.
info "Checking for version drift (packages in both files)"

# Find packages in both files and compare their versions.
MISMATCHES=$(
  comm -12 <(cut -d= -f1 "$REQ_FILE") <(cut -d= -f1 "$PYPROJ_FILE") | while read -r pkg; do
    req_ver=$(grep "^${pkg}==" "$REQ_FILE" | cut -d= -f3)
    pyproj_ver=$(grep "^${pkg}==" "$PYPROJ_FILE" | cut -d= -f3)
    if [[ "$req_ver" != "$pyproj_ver" ]]; then
      echo "Version drift: $pkg: requirements.txt==$req_ver vs pyproject.toml==$pyproj_ver"
    fi
  done
)

if [[ -z "$MISMATCHES" ]]; then
  pass
else
  fail
  echo "$MISMATCHES" | while read -r line; do
    echo "    $line"
  done
fi

# Report package coverage (informational, not a failure).
info "Packages in requirements.txt"
TOTAL=$((TOTAL + 1))
REQ_COUNT=$(wc -l < "$REQ_FILE")
green "$REQ_COUNT"

info "Packages in pyproject.toml [project].dependencies"
TOTAL=$((TOTAL + 1))
PYPROJ_COUNT=$(wc -l < "$PYPROJ_FILE")
green "$PYPROJ_COUNT"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $((TOTAL - FAILURES))/${TOTAL} checks passed, ${FAILURES} failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $FAILURES -gt 0 ]]; then
  exit 1
fi
