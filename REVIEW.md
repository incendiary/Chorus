# Chorus Engine — Holistic Review (15 July 2026)

> Second full review, at v4.0.1. Supersedes the 12 July 2026 review (see git history),
> whose nine action items (RA-1–RA-9) all shipped in v4.0.1. This review was run
> against a defined spec agreed with the maintainer: Chorus is a **personal working
> tool** and a **learning vehicle**; the bar for "done" is **prove the core idea
> works, then wind down to maintenance** after **one focused wrap-up push**.

---

## Executive summary

The engineering fundamentals are in excellent shape: 326 tests, 88 % line coverage,
clean dependency audit, layered CI with secret scanning, CodeQL, and blocking
`pip-audit`. The single most important gap is that **the project's founding claim has
never been measured**: nothing demonstrates that four-variant consensus transcription
is more accurate than a single Whisper pass, nor that the HIGH/MEDIUM/LOW confidence
tiers actually predict correctness. RB-2 (a WER + calibration benchmark) is the
centrepiece of the wrap-up push and answers "does this project meet its goals" with a
number. The most urgent operational bug is that **patch releases silently skip GitHub
Release creation** (a `needs:` skip-cascade in `release.yml` — v4.0.1's release had to
be backfilled by hand on 15 July). Output usability is largely a solved problem: clean
transcripts and machine-readable confidence data are always produced, and
`docs/CHORUS_FOR_LLMS.md` is an accurate, high-quality consumption contract — it just
has nothing keeping it honest as the schema evolves.

**Verdict against the maintainer's own bar:** as a learning vehicle, the project has
unambiguously succeeded. As a personal working tool it is functionally complete but
empirically unvalidated. Complete RB-1–RB-5 and the project has earned maintenance
mode regardless of what the benchmark shows — a negative result (consensus no better
than single-pass) would itself be a valid, honest conclusion, because the calibrated
uncertainty tiers are a deliverable single-pass Whisper cannot provide.

---

## Architecture

| Module / Path | Responsibility | Concerns |
|---|---|---|
| `audio_processor/` | Three cleaning filters (high-pass, normalise, denoise) + ingest validation | None — 92-93 % covered, property-based tests |
| `transcription_engine/` | Whisper wrapper + model×variant orchestrator, device-aware parallelism | MPS float64 CPU fallback emits a noisy `UserWarning` (RB-6) |
| `consensus_merger/` | Word-level alignment (Needleman-Wunsch or positional), voting, tier assignment, Markdown rendering | `alignment.py` (positional strategy) at 81 % — acceptable; non-default legacy path |
| `reconstruction/` | LOW-token reconstruction, `"nlp"` (spaCy) or `"llm"` (Ollama) strategy | Sound; 92-97 % covered since RA-8 |
| `diarisation/` | pyannote speaker separation fused with Whisper timestamps | 67 % coverage; degradation paths tested, happy path not — acceptable for wind-down |
| `export_engine/` | Consensus → PDF/DOCX/SRT/VTT/plain/bundle/AI-context | **PDF LOW-tier styling silently broken** (RB-3); bundle lacks a version field (RB-4) |
| `batch_processor/` | Unattended multi-file CLI | Sound, 83 % |
| `ui/` | Streamlit dashboard, decomposed in v4.0.1 (RA-9) | **`pipeline_invocation.py` 13 %, `results.py` 12 %** — the daily-use path is the least-tested code in the repo (RB-5) |
| `.github/workflows/` | CI, security, release automation | **Release skip-cascade breaks patch releases** (RB-1) |

Data flow: audio file → sanitised stem → 4 cleaned variants → N Whisper passes →
aligned word votes → (optional reconstruction, diarisation) → renderer/exporters →
`outputs/<stem>/` (or per-run `output_dir`). Input validation at entry, no network
exposure (local-first), no secrets in the tree.

---

## Risk inventory

| # | Category | Finding | Score | Location |
|---|---|---|---|---|
| 1 | Reliability | `github-release` and `post-release-consistency` jobs `needs: docker-publish`, which is skipped for non-`.0.0` tags — every patch release silently produces no GitHub Release and skips the strict consistency check. v4.0.1 was affected (backfilled 15 Jul). | 4 | `.github/workflows/release.yml:117,139` |
| 2 | Correctness (mission) | Core consensus claim unmeasured: no evidence multi-pass beats single-pass, no evidence confidence tiers are calibrated. All downstream guidance in `CHORUS_FOR_LLMS.md` §3 ("treat HIGH as reliable") is asserted, not demonstrated. | 4 | project-wide |
| 3 | Correctness (output) | PDF export: LOW-tier `~~word~~` markup is never converted to `<del>` (no strikethrough extension configured in `_md_to_html`), so the red-strikethrough CSS rule never fires. LOW words render unmarked in PDFs — a silent loss of the product's key signal in one of its formats. | 3 | `export_engine/exporter.py` (`_md_to_html`) |
| 4 | Maintainability | `ui/pipeline_invocation.py` (13 %) and `ui/results.py` (12 %) — run loop, retry/error rendering, download panels. Newly module-level (hence newly testable) after RA-9, but currently protected only by a render smoke test. | 3 | `ui/pipeline_invocation.py`, `ui/results.py` |
| 5 | Maintainability | `bundle.json` has no schema/producer version field (the docstring even promises "chorus version" in `meta` but the code never writes it), and no test ties `docs/CHORUS_FOR_LLMS.md` §5 to the real schema — the consumption contract can drift silently. | 2 | `export_engine/exporter.py:731,765` |
| 6 | Reliability | MPS float64 fallback emits a `UserWarning` on every affected pass on Apple Silicon — noise that trains the user to ignore warnings. Last unchecked legacy roadmap item. | 1 | `transcription_engine/whisper_engine.py` |

---

## Predicted failure scenarios (score ≥ 3)

### PF-1: Next patch release ships without a GitHub Release (Reliability, 4)

**What happens:** Tagging `v4.0.2` runs tests, then skips Docker (by design), then
skips `github-release` and `post-release-consistency` (by accident — skipped
dependencies cascade in GitHub Actions). `version_consistency_test.sh --ci` check 8
would have failed loudly, but it lives in the job that gets skipped. The tag exists,
the release page doesn't, and `ci.yml`'s Version-Tag Sync files a confusing issue on
the next push.

**Trigger condition:** any tag not ending `.0.0`. **Timeline:** the very next patch
release. **Fix:** RB-1.

### PF-2: The core value proposition quietly fails to exist (Mission, 4)

**What happens:** The maintainer (or anyone adopting the repo) spends 4× single-pass
compute per transcription on the assumption that consensus improves accuracy. If
consensus WER is not better than single-pass — plausible on clean audio, where
aggressive denoising can *introduce* errors — the extra cost buys nothing, and nobody
knows. Worse, if HIGH-tier words are not measurably more correct than average, the
guidance baked into `CHORUS_FOR_LLMS.md` misleads every downstream LLM.

**Trigger condition:** already latent; surfaces the first time anyone measures.
**Timeline:** unknown until RB-2 runs — which is exactly why it must.
**Fix:** RB-2. Note both possible outcomes are acceptable ends: "consensus wins on
noisy audio" validates the design; "consensus ties but tiers are calibrated"
repositions the product as *uncertainty-aware transcription*, and the docs get updated
to say so honestly.

### PF-3: PDF consumers act on unmarked low-confidence words (Correctness, 3)

**What happens:** A PDF is shared as the "formatted" transcript; a LOW word (often a
single-variant hallucination) renders as ordinary text. The reader quotes it.

**Trigger condition:** any PDF export containing LOW-tier words — i.e. most real
transcripts. **Timeline:** happening now. **Fix:** RB-3 (markdown extension + the
already-designed spy test asserting `<del>` reaches WeasyPrint).

### PF-4: UI regression in the run loop ships unnoticed (Maintainability, 3)

**What happens:** A future change (even a dependency bump — Streamlit minor versions
regularly change widget behaviour) breaks per-file progress, retry rendering, or a
download button. The 9 AppTest smoke tests exercise the sidebar and dialogs, not the
run/results path, so CI stays green.

**Trigger condition:** next Streamlit bump or UI edit. **Timeline:** months.
**Fix:** RB-5.

---

## Output usability assessment (maintainer's key question)

**Is Chorus producing usable output?** Yes, on both axes asked:

1. **Human transcript** — `{stem}_best_guess.txt` (clean, no markup) is always
   generated, alongside `most_likely` variants and the annotated consensus Markdown.
2. **AI-consumable ratings** — `{stem}_bundle.json` (full word-vote array with
   tier/confidence/alternatives per word) and `{stem}_ai_context.md` (prompt-ready
   methodology + statistics + uncertainty table) are always generated.

**Is the AI-consumption contract described and current?** `docs/CHORUS_FOR_LLMS.md`
is a genuinely strong contract document — schema reference, tier semantics, extraction
recipes, worked prompts. Verified line-by-line against the implementation in this
review: **accurate today**. Two things keep it from staying that way: the bundle
carries no version identifier (a consumer cannot tell which contract revision produced
it), and no test links the documented schema to the real one. RB-4 closes both.

---

## Test coverage

88 % overall, 326 tests, all passing. Gaps ranked by value:

| Path | Coverage | Why critical | Test needed |
|---|---|---|---|
| `ui/pipeline_invocation.py` / `ui/results.py` | 13 % / 12 % | The daily-use execution path; only smoke-tested | AppTest with mocked `run_pipeline`: sequential + all-at-once modes, per-file failure rendering, download panel presence (RB-5) |
| Bundle ↔ doc contract | n/a | Contract drift is silent | Structural test: doc §5 example keys match real bundle keys (RB-4) |
| `diarisation/diariser.py` | 67 % | Optional feature, degradation paths already tested | Accept as-is for wind-down |

---

## Dependency & CI audit

- All runtime pins exact and mirrored between `requirements.txt` and `pyproject.toml`,
  enforced by the RA-1 drift check. `pip-audit` (both scoped and whole-environment)
  green as of this review; setuptools CVE remediated 14 July.
- CI: tests, Black/Ruff/isort, GitLeaks, detect-secrets, bandit, CodeQL (default
  setup), Dependabot, weekly security and Ollama-tag cron runs. No gaps found beyond
  RB-1. Action versions pinned by major tag (acceptable).

---

## Action roadmap — the wrap-up push (target: v4.1.0)

Detailed, self-contained execution plans for each item live in `docs/tasks/RB-*.md`,
written for delegation to less-capable agents. Model assignments follow the
model-selection framework (verifiability × blast radius).

| ID | Title | Effort | Model | Why this tier |
|---|---|---|---|---|
| RB-1 | Fix release.yml skip-cascade for patch releases | XS | Haiku | Exact YAML given; verified by workflow re-run |
| RB-2 | WER + confidence-calibration benchmark | L | Sonnet | Judgment in data handling; "wrong but green" risk mitigated by designed sanity gates |
| RB-3 | Fix LOW-tier strikethrough in PDF export | S | Haiku | Known one-line fix + prescribed test |
| RB-4 | Version the bundle schema + contract test | S | Haiku | Exact spec, strong deterministic gate |
| RB-5 | Test the UI run loop and results rendering | M | Sonnet | Mocking judgment; hollow-test risk needs mid-tier |
| RB-6 | Silence redundant MPS float64 warning (optional) | XS | Haiku | Mechanical, gated by warning absence |

After RB-1–RB-5 merge: release v4.1.0, write the benchmark result into the README
(whatever it shows), and move the project to maintenance.
