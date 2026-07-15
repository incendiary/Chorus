# RB-1: Fix release.yml skip-cascade for patch releases

**Model tier:** Haiku · **Effort:** XS · **Branch:** `ci/rb1-release-skip-cascade`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first — it covers local CI validation
> (shift-left), known false-positive patterns, and the PR flow.

## Context (read this first)

`.github/workflows/release.yml` runs on every `v*` tag push. Its job chain is:

```
test → docker-publish → github-release → post-release-consistency
```

`docker-publish` has `if: ${{ endsWith(github.ref, '.0.0') && !contains(github.ref, '-') }}`
— intentional: Docker images are only built for major releases. The bug: in GitHub
Actions, when a job in `needs:` is **skipped**, every dependent job is skipped too
unless it overrides the default status check. So for any patch tag (v4.0.1, v4.0.2…):

- `github-release` never runs → the tag gets no GitHub Release (this actually
  happened for v4.0.1; it was backfilled manually on 15 July 2026)
- `post-release-consistency` never runs → the strict consistency check
  (`tests/version_consistency_test.sh --ci`, whose check 8 verifies the release
  exists) is silently skipped — the exact check that would have caught the problem

## The fix

In `.github/workflows/release.yml`:

1. Change `github-release`'s dependency and add a condition so it runs when
   `docker-publish` succeeded **or was skipped**, but never when tests failed:

```yaml
  github-release:
    name: GitHub Release
    runs-on: ubuntu-latest
    needs: [test, docker-publish]
    if: ${{ !cancelled() && needs.test.result == 'success' && contains(fromJSON('["success", "skipped"]'), needs.docker-publish.result) }}
```

2. Apply the same pattern to `post-release-consistency`:

```yaml
  post-release-consistency:
    name: Post-release Consistency
    runs-on: ubuntu-latest
    needs: [test, github-release]
    if: ${{ !cancelled() && needs.test.result == 'success' && needs.github-release.result == 'success' }}
```

   (`github-release` itself now always runs on success-or-skip of docker, so a plain
   success check on it is sufficient here.)

3. Also make the release-creation step idempotent, so re-runs don't fail on an
   existing release. In the `Create GitHub Release` step, change the `gh release
   create` invocation to check first:

```bash
if gh release view "${TAG}" &>/dev/null; then
  echo "Release ${TAG} already exists — skipping creation."
else
  gh release create "${TAG}" \
    --title "Chorus ${TAG}" \
    --generate-notes \
    --verify-tag
fi
```

Keep all other content of the workflow unchanged. Do not touch the `docker-publish`
condition — its behaviour is intentional.

## Verification (success criteria)

1. `python3 -c "import yaml, sys; yaml.safe_load(open('.github/workflows/release.yml'))"`
   parses cleanly (use `source .venv/bin/activate 2>/dev/null; python3 …` — the bare
   `python` shim is broken on this machine).
2. Push the branch, open a PR titled `ci: run github-release for patch tags when
   docker-publish is skipped`, confirm CI is green.
3. After the PR merges, verify end-to-end by re-running the release workflow against
   the existing v4.0.1 tag: `gh workflow run` is not available (tag-push trigger
   only), so instead use `gh run rerun <run-id>` on the original v4.0.1 release run
   (find it with `gh run list --workflow release.yml`). Confirm the `GitHub Release`
   job **runs** (idempotent step will detect the existing backfilled release and
   skip creation) and `Post-release Consistency` runs and passes.

## Files to change

- `.github/workflows/release.yml` (only)

## Repo conventions

Squash-merge PRs only, never commit to main. British English in comments. Do NOT
merge the PR — leave it open for review. Local pre-commit hook output may flag
pip-audit CVEs in `black`/`torch`/`msgpack`/`pillow` — that is a known false-positive
(it scans the dev venv, not the CI scope); ignore it, real CI is authoritative.
