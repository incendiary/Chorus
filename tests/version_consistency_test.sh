#!/usr/bin/env bash
# tests/version_consistency_test.sh
#
# Enforces Chorus version/release consistency across:
#   pyproject.toml, README.md, ROADMAP.md, git tags, GitHub releases
#
# Usage:
#   ./tests/version_consistency_test.sh          # local (warns on pre-release mismatches)
#   ./tests/version_consistency_test.sh --ci     # strict (fails on release mismatches)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILURES=0
TOTAL=0
CI_MODE="${1:-}"

red()    { printf '\033[0;31m%s\033[0m\n' "$1"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }
info()   { printf '  %-70s' "$1"; TOTAL=$((TOTAL + 1)); }

fail() { red "FAIL"; FAILURES=$((FAILURES + 1)); }
pass() { green "PASS"; }
warn() { yellow "WARN"; }

version_key() {
  awk -F. '{ printf("%05d%05d%05d\n", $1, $2, $3) }'
}

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║      Chorus Version & Release Consistency Tests       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# --- 1. pyproject version ---
info "pyproject.toml has valid semver"
VERSION=$(grep -m1 -E '^[[:space:]]*version[[:space:]]*=' "$REPO_ROOT/pyproject.toml" | sed -E 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/' || true)
if [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  pass
else
  fail; echo "    Could not parse valid version from pyproject.toml"
fi

# --- 2. README references roadmap ---
info "README.md references ROADMAP.md"
if grep -q "\[ROADMAP.md\](ROADMAP.md)" "$REPO_ROOT/README.md" 2>/dev/null; then
  pass
else
  fail; echo "    README.md must link to ROADMAP.md"
fi

# --- 3. README does not embed roadmap items ---
info "README.md does not embed roadmap version sections"
if grep -q "### Implemented Features (v" "$REPO_ROOT/README.md" 2>/dev/null; then
  fail; echo "    README.md contains embedded roadmap item sections"
else
  pass
fi

# --- 4. README version references match pyproject ---
info "README clone command matches v${VERSION}"
if grep -q "git clone -b v${VERSION}" "$REPO_ROOT/README.md" 2>/dev/null; then
  pass
else
  fail; echo "    Missing 'git clone -b v${VERSION}' in README.md"
fi

info "README docker CPU tag matches v${VERSION}"
if grep -q "ghcr.io/incendiary/chorus:v${VERSION}" "$REPO_ROOT/README.md" 2>/dev/null; then
  pass
else
  fail; echo "    Missing CPU image tag v${VERSION} in README.md"
fi

info "README docker GPU tag matches v${VERSION}"
if grep -q "ghcr.io/incendiary/chorus:v${VERSION}-gpu" "$REPO_ROOT/README.md" 2>/dev/null; then
  pass
else
  fail; echo "    Missing GPU image tag v${VERSION}-gpu in README.md"
fi

# --- 5. ROADMAP exists and has checklist items ---
info "ROADMAP.md exists"
if [[ -f "$REPO_ROOT/ROADMAP.md" ]]; then
  pass
else
  fail; echo "    ROADMAP.md not found"
fi

info "ROADMAP.md contains checklist items"
if grep -Eq -- "-[[:space:]]*\[(x| )\]" "$REPO_ROOT/ROADMAP.md" 2>/dev/null; then
  pass
else
  fail; echo "    No checklist entries found in ROADMAP.md"
fi

# --- 6. Completed roadmap items have version tags ---
# Items under the still-open "## Planned — vNext" heading are exempt: this
# repo now ships a release as several incremental work-package PRs, each
# ticking off its own items on merge, well before the version bump and tag
# that only happen at the actual release cut. A tag can't exist yet for work
# that's real but not yet released, so these items are allowed to stay
# untagged until they move into a "## Completed — vX.Y.Z" section post-release
# (see check 9's identical "except current in-flight" reasoning).
info "Completed roadmap items include (vX.Y.Z) tags (excluding in-flight 'Planned' items)"
UNTAGGED=$(awk '
  /^## Planned/ { in_planned = 1; next }
  /^## / { in_planned = 0 }
  in_planned { next }
  /-[[:space:]]*\[x\]/ && !/\(v[0-9]+\.[0-9]+\.[0-9]+\)/ { print }
' "$REPO_ROOT/ROADMAP.md")
if [[ -z "$UNTAGGED" ]]; then
  pass
else
  fail; echo "    Completed roadmap items missing version tags:"; echo "$UNTAGGED" | sed 's/^/      /'
fi

# --- 7. Latest tag alignment ---
info "Latest git tag aligns with pyproject version"
LATEST_TAG=$(git -C "$REPO_ROOT" tag -l "v*" --sort=-v:refname | head -1 || true)
EXPECTED_TAG="v${VERSION}"
if [[ -z "$LATEST_TAG" ]]; then
  if [[ "$CI_MODE" == "--ci" ]]; then
    fail; echo "    No git tags found; expected ${EXPECTED_TAG}"
  else
    warn; echo "    No git tags found (expected ${EXPECTED_TAG} once released)"
  fi
elif [[ "$LATEST_TAG" == "$EXPECTED_TAG" ]]; then
  pass
else
  # Allow in-flight branch versions newer than latest tag.
  VERSION_KEY=$(echo "$VERSION" | version_key)
  TAG_KEY=$(echo "${LATEST_TAG#v}" | version_key)
  if [[ "$VERSION_KEY" > "$TAG_KEY" ]]; then
    warn; echo "    In-flight version ${EXPECTED_TAG} is ahead of latest tag ${LATEST_TAG}"
  else
    fail; echo "    Latest tag ${LATEST_TAG} is ahead of or incompatible with ${EXPECTED_TAG}"
  fi
fi

# --- 8. GitHub release for latest tag ---
info "GitHub release exists for latest git tag"
if [[ -n "$LATEST_TAG" ]] && command -v gh >/dev/null 2>&1; then
  RELEASE_TAG=$(gh release view "$LATEST_TAG" --json tagName --jq '.tagName' 2>/dev/null || true)
  if [[ "$RELEASE_TAG" == "$LATEST_TAG" ]]; then
    pass
  else
    if [[ "$CI_MODE" == "--ci" ]]; then
      fail; echo "    No GitHub release found for ${LATEST_TAG}"
    else
      warn; echo "    No GitHub release for ${LATEST_TAG} (create when publishing milestone)"
    fi
  fi
elif [[ -z "$LATEST_TAG" ]]; then
  warn; echo "    No tags to validate"
else
  warn; echo "    gh CLI not available — skipped"
fi

# --- 9. Completed roadmap versions are tagged (except current in-flight) ---
info "Completed roadmap versions are tagged (except current in-flight)"
ALL_TAGS=$(git -C "$REPO_ROOT" tag -l "v*" || true)
MISSING=()
while IFS= read -r v; do
  [[ -z "$v" ]] && continue
  if [[ "$v" == "$VERSION" ]]; then
    continue
  fi
  if ! grep -qx "v${v}" <<< "$ALL_TAGS"; then
    MISSING+=("$v")
  fi
done < <(grep -Eo '\(v[0-9]+\.[0-9]+\.[0-9]+\)' "$REPO_ROOT/ROADMAP.md" | sed -E 's/^\(v(.*)\)$/\1/' | sort -u)

if [[ ${#MISSING[@]} -eq 0 ]]; then
  pass
else
  fail; echo "    Missing tags for roadmap-completed versions: ${MISSING[*]}"
fi

# --- 10. Roadmap completed versions sorted ascending ---
info "Completed roadmap versions are in ascending semver order"
COMPLETED_VERSIONS=$(grep -Eo '\(v[0-9]+\.[0-9]+\.[0-9]+\)' "$REPO_ROOT/ROADMAP.md" | sed -E 's/^\(v(.*)\)$/\1/' | awk '!seen[$0]++')
SORTED_VERSIONS=$(echo "$COMPLETED_VERSIONS" | sort -t. -k1,1n -k2,2n -k3,3n)
if [[ "$COMPLETED_VERSIONS" == "$SORTED_VERSIONS" ]]; then
  pass
else
  fail; echo "    ROADMAP completed versions are not in ascending order"
fi

# --- 11. Roadmap Last updated date freshness ---
info "ROADMAP.md last-updated date is within 30 days"
LAST_UPDATED_LINE=$(grep -E '^\*Last updated:' "$REPO_ROOT/ROADMAP.md" | head -1 || true)
if [[ -z "$LAST_UPDATED_LINE" ]]; then
  warn; echo "    No 'Last updated:' line found in ROADMAP.md"
else
  # Extract date portion (expected format: *Last updated: DD Month YYYY*)
  ROADMAP_DATE=$(echo "$LAST_UPDATED_LINE" | sed -E 's/\*Last updated: ([0-9]+ [A-Za-z]+ [0-9]+)\*/\1/')
  # Convert to epoch seconds using date; fall back gracefully on parse failure
  if ROADMAP_EPOCH=$(date -j -f "%d %B %Y" "$ROADMAP_DATE" "+%s" 2>/dev/null || date -d "$ROADMAP_DATE" "+%s" 2>/dev/null); then
    NOW_EPOCH=$(date "+%s")
    DAYS_OLD=$(( (NOW_EPOCH - ROADMAP_EPOCH) / 86400 ))
    if [[ $DAYS_OLD -le 30 ]]; then
      pass
    else
      warn; echo "    ROADMAP.md last updated ${DAYS_OLD} days ago (>${30} day threshold) — consider refreshing"
    fi
  else
    warn; echo "    Could not parse ROADMAP.md date '${ROADMAP_DATE}'"
  fi
fi

# --- Summary ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $((TOTAL - FAILURES))/${TOTAL} passed, ${FAILURES} failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ $FAILURES -gt 0 ]]; then
  exit 1
fi
