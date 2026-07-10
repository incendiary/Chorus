# Chorus Engine — v4.0.0 Work Packages

This directory contains the broken-out, self-contained task specifications for the
**Chorus 4.0.0** release. Each work package (`WP*.md`) is written so that a single
agent can execute it **without having read the rest of the codebase or this
conversation**. Read this index first, then the one work package you have been
assigned.

> Current released line: **3.3.0** (see `VERSION`). 4.0.0 is the next *major* bump.
> A major version is justified here because WP1 introduces **breaking changes** to
> import paths, packaging, and the optional-feature module layout.

---

## The 4.0.0 thesis — "Trustworthy outputs, stable surface"

Chorus 3.x is feature-rich but has three structural weaknesses that block confident,
unattended, at-scale use and library integration:

1. **You cannot `pip install` it and use it as a library.** `pyproject.toml` declares
   `dependencies = []`, and there is no stable public API — callers reach into deep
   module paths (`from consensus_merger.merger import ...`).
2. **Output isolation is not guaranteed.** Several helpers still read from and write to
   the global `outputs/consensus/` directory even when the caller supplies an isolated
   `output_dir`. This silently mixes artefacts across CLI, UI, and batch runs — the
   top risk recorded in `REVIEW.md`.
3. **The surfaces users actually touch are untested.** `ui/app.py` and
   `batch_processor/batch_runner.py` sit at ~0 % coverage, and `pip-audit` runs
   non-blocking (`|| true`) in CI.

4.0.0 fixes all three and ships two visible user features (a clean "best-guess"
transcript export, and an LLM context document) as the headline payload.

---

## Work packages and execution order

| WP | Title | Breaking? | Depends on | Effort |
|----|-------|-----------|------------|--------|
| [WP1](WP1-packaging-and-public-api.md) | Packaging & stable public API | **Yes** | — | L |
| [WP2](WP2-output-routing-correctness.md) | Output-routing correctness | No | — | M |
| [WP3](WP3-test-parity-and-ci.md) | User-facing test parity & CI hardening | No | WP1 (import paths) | M |
| [WP4](WP4-headline-features.md) | Headline user features | No | WP1 (reconstruction API), WP2 (output_dir) | L |

**Recommended order:** WP2 → WP1 → WP3 → WP4.

- **WP2 first** — it is pure correctness with no dependencies, and locks in the
  `output_dir` contract that WP4 relies on.
- **WP1 second** — the largest and the only breaking package; do it before WP3 writes
  tests against import paths, to avoid rework.
- **WP3 third** — its UI/batch tests should target the post-WP1 import surface.
- **WP4 last** — both new features build on the WP1 reconstruction API and the WP2
  `output_dir` guarantee.

WP2 and WP3 are independently shippable as `3.4.x` increments if you want to
de-risk; only WP1 *must* land under the `4.0.0` tag.

---

## Conventions every agent MUST follow

These are project rules from `CLAUDE.md`. Non-negotiable:

- **One branch + PR per work package.** Never commit to `main`. Branch name:
  `feat/v4-wp<N>-<slug>` (e.g. `feat/v4-wp2-output-routing`). Open a PR, squash-merge.
- **British English** in all comments, docstrings, and user-facing text. Active voice.
  Oxford comma enforced.
- **Surgical changes.** Touch only what the task requires. No speculative abstractions,
  no drive-by refactors of adjacent code.
- **Formatting gate before every commit:** `black .`, `ruff check .`, `isort .` must
  all pass. Then run the relevant tests.
- **Tests must pass:** `.venv/bin/python -m pytest` (193 tests pass on `main` today —
  do not regress that count; add to it).
- **VERSION is the single source of truth.** Do **not** bump `VERSION`, `pyproject.toml`,
  or README version strings inside a work-package PR. The release owner bumps to
  `4.0.0` once all WPs are merged (see "Release cut" below). Version-sync tests
  (`tests/test_version_sync.py`) will otherwise fail.
- **Documentation parity:** if you change a public API or behaviour, update the
  module docstrings, `README.md`, and `CLAUDE.md` in the same PR.
- **Mark roadmap items done:** when a WP merges, tick its checkboxes in `ROADMAP.md`
  under the `v4.0.0` section and record the file(s) touched.

### Environment

Use the repository virtual environment; do not install packages system-wide:

```bash
source .venv/bin/activate     # interpreter is Python 3.14 in .venv
python -m pytest              # run the suite
```

Heavy native deps (torch, whisper, pyannote, librosa) are already installed in
`.venv`. Most tests stub these out — read the existing tests in `tests/` for the
mocking patterns before writing new ones.

---

## Release cut (release owner only — not part of any WP)

Once WP1–WP4 are all merged to `main`:

1. `printf '4.0.0\n' > VERSION`
2. Update `pyproject.toml` `version`, and the `v3.3.0` strings in `README.md`
   (clone tag, Docker pull/run tags) to `v4.0.0`.
3. Add a dated `v4.0.0` "Completed" block to `ROADMAP.md`; move the v4 checklist there.
4. Run `bash tests/version_consistency_test.sh` and `.venv/bin/python -m pytest`.
5. Write a `## Breaking changes` migration note in `README.md` covering the new
   import paths and the consolidated reconstruction module (from WP1).
6. Tag `v4.0.0`; the release workflow (`.github/workflows/release.yml`) handles the
   Docker build, publish, and GitHub release.
