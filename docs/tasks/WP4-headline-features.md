# WP4 — Headline user features

**Target release:** v4.0.0
**Breaking:** No
**Depends on:** WP1 (use the `chorus` public API and the `reconstruction` package),
WP2 (rely on the `output_dir` guarantee). Do this last.
**Branch:** `feat/v4-wp4-features`
**Overall effort:** L

Read `docs/tasks/README.md` ("Conventions every agent MUST follow") before starting.

## Why this exists

WP1–WP3 make 4.0.0 trustworthy under the hood. WP4 is the **visible payload** — the
two features a user or release note can point at. Both already sit, unstarted, in the
ROADMAP "Upcoming" section.

---

### RA-4.1: Human-readable "best-guess" transcript export

**Context:** Existing plain-text exports carry markup — `[word?]` for LOW-confidence
positions, omitted words, verbose statistics. There is no clean artefact suitable for a
non-technical reader. (`export_engine/exporter.py` already has
`export_plain_text(..., include_low=...)` — study it; this is a *new, cleaner* export,
not a tweak to that one.)

**What to do:**
- Add `export_best_guess(consensus_md_path, stem, *, output_dir=None) -> Path` to
  `export_engine/exporter.py`, writing `{stem}_best_guess.txt` to
  `output_dir or CONSENSUS_DIR` (honour `output_dir` per WP2).
- Output rules:
  - HIGH-confidence words: included verbatim.
  - MEDIUM/LOW-confidence positions: include the single best-guess word — the
    candidate with the **highest agreement across variants** (the word-vote structure
    already carries per-candidate counts; reuse it, do not recompute alignment).
  - **No** brackets, annotations, confidence markers, or statistics.
  - Preserve natural flow and paragraph breaks from the consensus document.
- Auto-generate it after every pipeline run (alongside the existing bundle/consensus
  outputs in `pipeline_runner.py`) and include it in the UI download ZIP
  (`build_export_zip`).

**Success criteria:**
- New test in `tests/test_exporter.py`: given a consensus document with mixed
  confidence tiers, the best-guess file contains the high-agreement word at each
  MEDIUM/LOW position and contains **no** `[`, `?]`, or statistics lines.
- Empty/silent transcript produces an empty (not crashing) best-guess file — covers the
  CLAUDE.md null-state requirement.
- File lands under `output_dir` when one is supplied.

**Files to change:** `export_engine/exporter.py`, `pipeline_runner.py`,
`export_engine/__init__.py` if it re-exports, `ui/app.py` (download wiring),
`tests/test_exporter.py`. Update README output-formats section and the Help page
(`ui/pages/1_Help.py`). Tick the ROADMAP item.
**Effort:** M

---

### RA-4.2: LLM context document (`docs/CHORUS_FOR_LLMS.md`)

**Context:** Chorus produces several artefacts (`consensus.md`, `bundle.json`,
`ai_context.md`, `diarised.md`, subtitle files). There is no single document that
explains the project and its outputs to a language model so a user can paste it
alongside Chorus output for downstream analysis. (Detailed spec is in the ROADMAP
"Upcoming" item — follow it.)

**What to do:** Write `docs/CHORUS_FOR_LLMS.md` covering:
- Project overview: what Chorus does, why, and high-level architecture.
- Output format reference: every generated file and what it contains.
- Confidence tier semantics: how to interpret HIGH/MEDIUM/LOW and what they say about
  reliability.
- Word-vote structure: how the consensus algorithm works and how variants drive
  confidence.
- Export formats: PDF, DOCX, SRT, VTT metadata and markup conventions.
- `bundle.json` schema: fields and how to extract structured data programmatically
  (verify against the real `export_transcript_bundle()` output — do not invent fields).
- Worked prompts: how to ask an LLM to use Chorus output for summarisation,
  fact-checking, and speaker-intent analysis.

**Success criteria:**
- Document exists, is accurate against the **current** code (cross-check every claimed
  field/file against `export_engine/` and `consensus_merger/`), and is in British
  English.
- Linked from `README.md` and the Help page.
- `tests/test_version_sync.py` / markdownlint pass (this is docs only).

**Files to change:** new `docs/CHORUS_FOR_LLMS.md`, `README.md`,
`ui/pages/1_Help.py`. Tick the ROADMAP item.
**Effort:** M

---

### RA-4.3: Streamline spaCy model setup

**Context:** The NLP reconstruction path needs `en_core_web_md`, fetched separately via
`python -m spacy download en_core_web_md`. Today a missing model surfaces as a silent
runtime fallback warning. (ROADMAP "Upcoming" item.)

**What to do (pick the lightest option that solves it; recommend in PR):**
- Detect the missing model at the point of use and surface a **clear, actionable**
  message (UI dialog like the existing Ollama setup dialog in `ui/app.py`, and a
  clean CLI error) telling the user the exact command to run — instead of a silent
  warning and degraded output.
- Optionally, offer a guarded auto-download with progress indication on first use.
- Document the requirement as a post-install step in `README.md` and the Help page,
  and note the option to bake the model into the Docker image for containerised
  deployments (do **not** change the Dockerfiles in this task unless trivial and
  tested).

**Success criteria:**
- With the model absent, the user gets an explicit instruction (not a silent
  degradation). Covered by a test that mocks the missing-model condition and asserts
  the actionable error/branch is taken — extends `tests/test_reconstructor.py` (post-WP1:
  the `reconstruction` package tests).
- The CLAUDE.md requirement — "`nlp_reconstructor` falls back safely if spaCy models
  are missing" — remains satisfied: the pipeline must not crash, it must inform.

**Files to change:** `reconstruction/nlp.py` (post-WP1) or
`nlp_reconstructor/reconstructor.py` (if WP1 not yet merged), `ui/app.py`, `README.md`,
`ui/pages/1_Help.py`, the relevant reconstruction test. Tick the ROADMAP item.
**Effort:** M
