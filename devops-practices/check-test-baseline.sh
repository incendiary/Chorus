#!/usr/bin/env sh
# check-test-baseline.sh
#
# Verifies CI and baseline test enforcement exist.

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

CI_FILE=""
if [ -f "$REPO_ROOT/.github/workflows/ci.yml" ]; then
  CI_FILE="$REPO_ROOT/.github/workflows/ci.yml"
elif [ -f "$REPO_ROOT/.github/workflows/ci.yaml" ]; then
  CI_FILE="$REPO_ROOT/.github/workflows/ci.yaml"
else
  fail "CI workflow not found at .github/workflows/ci.yml|ci.yaml"
fi
pass "CI workflow present"

grep -Eq 'pytest|python[[:space:]]+-m[[:space:]]+pytest' "$CI_FILE" || fail "CI workflow does not run pytest"
pass "CI workflow includes pytest"

[ -f "$REPO_ROOT/tests/test_version_sync.py" ] || fail "Missing baseline test: tests/test_version_sync.py"
pass "Baseline test file exists (tests/test_version_sync.py)"

[ -f "$REPO_ROOT/devops-practices/check-version-sync.sh" ] || fail "Missing script: devops-practices/check-version-sync.sh"
[ -f "$REPO_ROOT/devops-practices/check-clone-refs.sh" ] || fail "Missing script: devops-practices/check-clone-refs.sh"
pass "DevOps check scripts are present"

grep -q 'devops-practices/check-version-sync.sh' "$CI_FILE" || fail "CI missing check-version-sync.sh invocation"
grep -q 'devops-practices/check-clone-refs.sh' "$CI_FILE" || fail "CI missing check-clone-refs.sh invocation"
pass "CI enforces baseline devops checks"

printf 'All test baseline checks completed.\n'
