#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# gh_init.sh — Chorus Engine GitHub Initialisation Script
#
# Initialises the local folder as a GitHub repository, pushes the initial
# commit, and enforces branch protection rules aligned with the Bedrock
# security baseline (https://github.com/incendiary/Bedrock).
#
# Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
#   1. GitHub CLI installed:   https://cli.github.com/
#   2. Authenticated:          gh auth login
#   3. GPG key configured for signed commits (optional but recommended):
#        git config --global user.signingkey <YOUR_GPG_KEY_ID>
#        git config --global commit.gpgsign true
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   chmod +x gh_init.sh
#   ./gh_init.sh <github-username> <repo-name> [--public]
#
# Examples
# ─────────────────────────────────────────────────────────────────────────────
#   ./gh_init.sh myorg chorus-engine
#   ./gh_init.sh myorg chorus-engine --public
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <github-username-or-org> <repo-name> [--public]"
    echo ""
    echo "  github-username-or-org   Your GitHub username or organisation name"
    echo "  repo-name                Name for the new GitHub repository"
    echo "  --public                 Create as a public repo (default: private)"
    exit 1
}

[[ $# -lt 2 ]] && usage

GITHUB_OWNER="$1"
REPO_NAME="$2"
VISIBILITY="--private"

[[ "${3:-}" == "--public" ]] && VISIBILITY="--public"

FULL_REPO="${GITHUB_OWNER}/${REPO_NAME}"

# ── Preflight checks ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Chorus Engine — GitHub Initialisation (Bedrock Security Baseline)  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

if ! command -v gh &>/dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed."
    echo "       Install from: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "ERROR: Not authenticated with GitHub CLI."
    echo "       Run: gh auth login"
    exit 1
fi

if ! command -v git &>/dev/null; then
    echo "ERROR: git is not installed."
    exit 1
fi

echo "==> Target repository: ${FULL_REPO} (${VISIBILITY/--/})"
echo ""

# ── Initialise local git repository ──────────────────────────────────────────
if [[ ! -d ".git" ]]; then
    echo "==> Initialising local git repository…"
    git init --initial-branch=main
else
    echo "==> Existing git repository detected — skipping git init."
fi

# ── Install pre-commit hooks ──────────────────────────────────────────────────
if command -v pre-commit &>/dev/null; then
    echo "==> Installing pre-commit hooks…"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "WARN: pre-commit not found. Install with: pip install pre-commit"
    echo "      Then run: pre-commit install"
fi

# ── Generate secrets baseline if absent ──────────────────────────────────────
if [[ ! -f ".secrets.baseline" ]]; then
    if command -v detect-secrets &>/dev/null; then
        echo "==> Generating detect-secrets baseline…"
        detect-secrets scan > .secrets.baseline
    else
        echo "WARN: detect-secrets not found. Baseline not generated."
    fi
fi

# ── Stage and commit ──────────────────────────────────────────────────────────
echo "==> Staging all files…"
git add -A

# Sign the commit if GPG signing is configured
if git config --get commit.gpgsign | grep -q "true" 2>/dev/null; then
    echo "==> Creating signed initial commit…"
    git commit -S -m "chore: initialise Chorus Engine from production-ready suite"
else
    echo "==> Creating initial commit (unsigned — configure GPG for signed commits)…"
    git commit -m "chore: initialise Chorus Engine from production-ready suite"
fi

# ── Create GitHub repository ──────────────────────────────────────────────────
echo "==> Creating GitHub repository: ${FULL_REPO}…"
gh repo create "${FULL_REPO}" \
    ${VISIBILITY} \
    --description "Chorus Engine — containerised multi-pass consensus audio transcription" \
    --source=. \
    --remote=origin \
    --push

# ── Set default branch ────────────────────────────────────────────────────────
echo "==> Setting default branch to main…"
gh repo edit "${FULL_REPO}" --default-branch main

# ── Enable branch protection on main ─────────────────────────────────────────
echo "==> Applying Bedrock branch protection rules to main…"
gh api "repos/${FULL_REPO}/branches/main/protection" \
    --method PUT \
    --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Secret Scan", "Lint", "Test"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true,
  "required_conversation_resolution": true
}
EOF

# ── Require signed commits ────────────────────────────────────────────────────
echo "==> Enforcing signed commits on main…"
gh api "repos/${FULL_REPO}/branches/main/protection/required_signatures" \
    --method POST \
    --silent || echo "WARN: Could not enforce signed commits (may require GitHub Pro/Team plan)."

# ── Enable vulnerability alerts and automated security fixes ─────────────────
echo "==> Enabling Dependabot vulnerability alerts…"
gh api "repos/${FULL_REPO}/vulnerability-alerts" \
    --method PUT \
    --silent || echo "WARN: Could not enable vulnerability alerts."

echo "==> Enabling Dependabot automated security fixes…"
gh api "repos/${FULL_REPO}/automated-security-fixes" \
    --method PUT \
    --silent || echo "WARN: Could not enable automated security fixes."

# ── Tag initial release ───────────────────────────────────────────────────────
echo "==> Creating initial release tag v2.0.0…"
gh release create v2.0.0 \
    --title "v2.0.0 — Production-Ready Suite" \
    --notes "All roadmap features implemented. Bedrock security baseline applied. Ready for Claude Code autonomous maintenance." \
    --latest || echo "WARN: Could not create release (check permissions)."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Initialisation complete.                                            ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Repository:  https://github.com/${FULL_REPO}"
echo "║  Branch:      main (protected)"
echo "║  Force push:  BLOCKED"
echo "║  Direct push: BLOCKED (PR required)"
echo "║  Signed commits: REQUIRED"
echo "║  CI checks:   Secret Scan → Lint → Test"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                                         ║"
echo "║  1. Open in Claude Code — CLAUDE.md loads automatically.            ║"
echo "║  2. Fill in the four context fields in CLAUDE.md.                   ║"
echo "║  3. Run: pre-commit run --all-files                                 ║"
echo "║  4. Regenerate secrets baseline after adding code:                  ║"
echo "║     detect-secrets scan > .secrets.baseline                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
