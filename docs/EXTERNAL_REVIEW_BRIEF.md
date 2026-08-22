# Chorus Engine — External Review Brief and Handover

Paste everything below into a fresh session with a different model. It is written to be
self-contained: the reader has no memory of the work that produced this state.

It serves two purposes:

- **A review brief** (§0–§7) — an adversarial second opinion before a release.
- **A handover** (§8–§11) — everything needed to *take over* the project: environment,
  current state, decisions that are final, and the next chunk of work.

A reviewer needs §0–§7. Someone picking the work up needs all of it.

---

## 0. Your role and how to engage

You are reviewing a public Python repository before its maintainer cuts a release. The
maintainer is the Director of Offensive Security at a large telco group — a strong
engineer (Python primary, also C++/C#), working on macOS with JetBrains tooling. Write
in **British English, active voice, Oxford comma, no em dashes**. Be concise and direct;
skip flattery and ethics boilerplate.

**This is an adversarial review, not a rubber stamp.** The two most damaging defects in
this project's history were both found by *running the software*, not by reading it, and
both survived a purpose-built benchmark and a full test suite. Assume more of that class
remains.

**Engage, do not monologue.** Ask the maintainer questions where a decision is genuinely
theirs (hardware constraints, risk tolerance, scope, whether an unproven claim should
ship). Ask them up front, before producing findings, rather than burying them at the end.
Where a choice is purely technical with no user-facing trade-off, just decide and say so.

**Ground every claim in external signal.** Do not assert a defect you have not
demonstrated. If you believe something is broken, write the failing case and run it. If
you cannot run it, say "unverified hypothesis" explicitly. The maintainer has been burned
by three confidently-wrong root-cause claims in this project already, and by
agent-written tests that could not fail (`AppTest.get_text()` which does not exist, an
unresolvable patch target, `assert a or b` chains). **A test that passes against the
buggy code is worthless — prove your regression tests fail before the fix.**

---

## 1. The project

**Repository:** https://github.com/incendiary/Chorus (public)
**Review target:** `main` at commit `f701304`. Everything is merged and pushed; there is
no feature branch to check out. `VERSION` says `4.1.0`; there are 28 commits since that
tag, so a release bump is pending.

**What it is:** Chorus Engine is a local, containerised Streamlit application for
high-fidelity audio transcription using a multi-pass consensus method. It runs entirely
offline. Its founding premise: transcribe the same recording four ways and compare the
results to identify which words are trustworthy.

**Pipeline:**
1. `audio_processor/` — produces four audio variants (original, high-pass, dynamic-range
   normalised, denoised) via `pydub`/`librosa`
2. `transcription_engine/` — runs local Whisper over each variant
3. `consensus_merger/` — Needleman-Wunsch word alignment across variants, votes per word
   position, assigns HIGH / MEDIUM / LOW confidence tiers
4. `diarisation/` — `pyannote.audio` speaker separation, fused with Whisper timestamps
5. `reconstruction/` — repairs LOW-confidence tokens via spaCy or a local Ollama model
6. `export_engine/` — PDF, DOCX, SRT, VTT, plus a bundle JSON and an AI-context pack
7. `batch_processor/` — unattended CLI for directories
8. `ui/` — Streamlit dashboard with background runs that survive tab close

**Scale:** ~9,800 lines of application Python, ~8,200 lines of tests across 27 test
files, 430 tests passing. `ui/` (3,785) and `export_engine/` (1,191) are the largest
packages.

**Context that should shape your review:** this is a *personal working tool* and a
*learning vehicle*, not a product with users. The maintainer's stated bar is **"prove it
works, then stop."** Do not recommend enterprise scaffolding, speculative abstraction, or
features nobody asked for. Do recommend anything that bears on whether the core claim is
true.

---

## 2. The honest state of the founding claim

**Read this carefully — it is the crux of the review.**

The project exists to prove that four-variant consensus produces better transcription
than a single Whisper pass. **That claim is currently unproven, and the evidence leans
against it.**

Two severe defects were fixed immediately before this review:

- **RC-10** — `word_timestamps=True` was hardcoded on every Whisper pass. On long-form
  audio this collapses Whisper into a repetition loop. Measured on one 28.8-minute
  recording, same model, one parameter changed: **1,643 words with it off versus 137
  words with it on**, 47% of the latter being a single repeated phrase.
- **RC-11** — the multi-alignment merge called `columns.insert()` per insertion, which
  shifted every column to its right, while the next variant's counter still tracked raw
  reference positions. Each successive variant landed further out of place. On real audio
  this held HIGH confidence at **4.6% where the same data supports 43.6%**.

After both fixes, the benchmark (LibriSpeech test-clean, 15 utterances, Whisper `base`):

| Condition | Single-pass WER | Chorus WER |
|---|---|---|
| clean | 0.0314 | 0.0288 |
| noisy (SNR 5 dB) | 0.1024 | **0.1095** |

**Consensus still loses to single-pass Whisper on noisy audio** — the condition the
architecture is meant to help most.

And the benchmark itself is not representative. Mean pairwise divergence between the four
variants is **0.044 on the benchmark versus 0.474 on real phone audio** — roughly ten
times less. At 0.044 the variants are nearly identical, alignment sees almost no
insertions, and consensus has nothing to arbitrate. That is precisely why both defects
above survived it: RC-11 moved the real recording from 4.6% to 43.6% HIGH while moving
the benchmark's HIGH count by **one word**.

The current defensible positioning is **calibrated uncertainty** (the tiers reliably
indicate which words to trust — HIGH precision measured at 0.92–0.98), **not superior
accuracy**.

**Questions for the maintainer you should raise early:**
- Is "calibrated uncertainty" enough to ship on, or must the WER claim be proven first?
- Is there any long-form ground truth available (a human transcript of a real recording),
  or is `large-v3` output an acceptable reference proxy?
- Should README/docs be reworded so no reader infers an accuracy claim that is not
  evidenced?

---

## 3. Conventions this repo is held to

### 3.1 DevOps practices (the maintainer's standing standard)

**Versioning** — semantic. `VERSION` at repo root is the single source of truth, one line
`X.Y.Z`. Git tag is `vX.Y.Z` and must match. GitHub release must match the tag. Patch = a
batch of related PRs; minor = new capability; major = breaking change.

**PR discipline** — **never commit directly to `main`.** Branch, PR, CI green, squash
merge. One logical change per PR — not "fix bug and also refactor". Conventional commit
titles (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, `bench:`). Never rebase a published
branch; `git merge main` to catch up. Group related PRs into one version bump.

**Testing** — CI must include a test step. Never disable a failing test; fix the code or
fix the test. Flaky test = production bug. **Every production bug gets a regression test
in the same PR that fixes it.** Coverage direction must be up.

**Reference pinning** — no floating branch references in docs or Actions (an `@` followed
by a branch name such as the default branch); pin to `@vX.Y.Z` or a commit SHA. Install
commands versioned (`pip install pkg==X.Y.Z`). Note that `devops-practices/check-clone-refs.sh`
enforces this with a plain grep, so writing the literal form even as an example fails CI.

**Roadmap** — `ROADMAP.md` at root is the source of truth, checkbox format (`- [ ]` /
`- [x]`), README links to it rather than duplicating. Sync on merge. No stale items.

**Formatting** — Python: `black` + `ruff` (+`isort`), both gating CI. Nothing unformatted
merges.

**Documentation parity (project rule)** — if a module's public API or behaviour changes,
update its docstrings, `README.md`, and `CLAUDE.md` in the same change. British English,
active voice, Oxford comma throughout, including user-facing UI text.

**Surgical changes (project rule)** — touch only what you must. No speculative
abstractions, no wrapper functions that add no logic, no improving adjacent code. Every
changed line should trace to the request. Mention unrelated dead code rather than
deleting it.

**Existing CI:** `ci.yml`, `security.yml` (CodeQL, Bandit, GitLeaks, TruffleHog,
dependency audit), `release.yml`, `ollama-model-tags-check.yml`. Local mirror via
`bash local-ci.sh --list` / `--dry-run` / `--sync`. Note: a bare `local-ci.sh` run
auto-installs a pre-push hook, so use a flag unless hook installation is intended.

### 3.2 The Karpathy three-layer framework

The maintainer works this framework and expects you to operate inside it and to *engage
them with it*, not merely mention it.

**Layer 1 — Spec (`karpathy-spec`).** Define what "done" looks like *before* working.
Turn vague requests into precise, verifiable criteria. Prevents the dominant failure
mode: building the wrong thing precisely.

**Layer 2 — Verify (`karpathy-verify`).** *This review is a Layer 2 exercise* — you are
the second opinion, deliberately a different model with different blind spots. The method:

1. **Define evaluation criteria upfront, before evaluating.** Typically 4–6 dimensions
   (correctness, completeness, safety, usability, maintainability). Each must be
   *specific* and paired with a *runnable* verification method — not a subjective one.
   Format each as: `Criterion → Precise definition → How to verify`.
2. **Evaluate against those criteria**, pass/fail with evidence per criterion.
3. **Ground claims in external signal** — run the tests, execute the code, check the
   sources. Do not accept your own assertion as evidence.
4. **Identify the top three improvements**, ranked.

**Layer 3 — Environment (`karpathy-environment`).** Capture durable patterns into
`CLAUDE.md` and reusable skills so the next session starts further along. If this review
surfaces a recurring failure mode, propose the guardrail that would catch it next time.

**Start by proposing your Layer 1 criteria and asking the maintainer to confirm or amend
them before you produce findings.**

### 3.3 The holistic review method (`codebase-holistic-review`)

This repository has already been through **three** reviews using the maintainer's own
holistic-review method, and the findings you are being asked to check were produced by
it. You should follow the same method so your output is comparable and mergeable with
what already exists.

It is explicitly **predictive**, not just descriptive: the point is not only to find
today's bugs but to identify what will break as the project grows. A null finding in a
phase is still a finding — record it rather than skipping the phase.

**The eight phases, in order:**

1. **Architecture map** — layers, each module's one-line purpose, modules with no clear
   single responsibility, and the data-flow from external input through to
   response/file/side-effect. Output as a `Module | Responsibility | Concerns` table.
2. **Risk inventory** — classify every risk as **Security**, **Scalability**,
   **Reliability**, **Maintainability**, or **Dependency**, and score each **1–5**
   (5 = critical). Output as a `# | Category | Finding | Score | File/Location` table
   with precise `path:line` references.
3. **Predictive failure analysis** — for every risk scoring **≥ 3**, write: *what
   happens* (the exact observable failure — error, silent corruption, performance cliff,
   breach), *trigger condition* (what usage, data volume, or environment change causes
   it), *estimated timeline* (today? at 10× load? only after the next dependency
   upgrade?), *minimum fix*, and *full fix*.
4. **Test coverage gaps** — critical execution paths with no coverage, plus **tests that
   assert too broadly and would pass even if the logic were wrong**. This phase matters
   more than usual here: 430 tests passed while two severe defects were live. Rank the
   three highest-value additions.
5. **Dependency audit** — pinned versus ranged versus floating; anything with no upstream
   release in 18+ months; known CVEs; and any dependency doing one small thing that could
   be inlined.
6. **CI/CD gap analysis** — does the pipeline run tests on every PR, lint and
   format-check, scan for secrets, pin action versions, and run on a schedule to catch
   drift? Write the YAML snippet that fixes each "no".
7. **Write the review document** — executive summary (3–5 sentences: top risk, most
   urgent fix, overall health), then the phase outputs, then the action roadmap.
8. **Roadmap sync** — action items become checkboxes in `ROADMAP.md`.

**Action-item format.** Each item must be executable by a *less capable agent* who has
not read the review, so no item may assume context:

```markdown
### RD-1: <short imperative title>

**Context:** 1–2 sentences — what the problem is and exactly where.

**Success criteria:**
- Precisely how to verify it is done correctly.

**Files to change:** explicit paths

**Estimated effort:** XS / S / M / L / XL
```

**Prefix convention.** Each review gets its own letter series, and this repo has used
three already — `RA-` (first review), `RB-` (second), `RC-` (third, the one whose items
were just closed). **Use `RD-` for anything you raise**, so your findings do not collide
with the existing history in `REVIEW.md` and `ROADMAP.md`.

**Two gotchas the skill calls out that apply directly here:**
- **Read the actual files; do not recall them.** Start from source, not from this brief.
  This brief tells you what was *claimed*; your job is to check it.
- **Do not omit a finding because it is well known.** If something in §5 is worse than
  the maintainer believes, say so with evidence — "already known" is not the same as
  "correctly triaged".

---

## 4. What was just changed (verify this work critically)

Seven PRs merged in the session immediately before this review:

| PR | Change |
|---|---|
| #207 | RC-10 — word timestamps made opt-in; degeneracy guard added (flags any transcript where one 3-gram exceeds 20%) |
| #208 | RC-11 — alignment merge rewritten to key columns by reference position; shared insertions merged into one column |
| #209 | RB-2 benchmark re-run; benchmark's unrepresentativeness documented |
| #210 | RC-1 — intermediate variant WAVs reclaimed after each run (`outputs/variants/` had reached 21 GB across 232 files) |
| #211 | Weekly workflow was filing a duplicate GitHub issue every run — its dedup search used a title containing `(s)`, which GitHub's search parser treats as syntax, so the guard never matched |
| #212 | RC-8 closed as obsolete; MPS speed claim in docs replaced with measured figures (1.05× `base`, 1.7× `small`, 2.7× `medium` — the docs had claimed "3–5× for base and small", wrong in both magnitude and direction) |
| #213 | Roadmap brought back in line |

**Please attack these specifically:**

- **`consensus_merger/sequence_alignment.py::_build_multi_alignment`** is the highest-risk
  change — it is the heart of the consensus mechanism and it was rewritten. Reference-based
  (star) alignment is used, with the *longest* transcript chosen as reference. On real
  audio that meant the **most divergent variant anchored the alignment**. This was
  consciously left as a design question rather than fixed. Is that defensible? Would a
  progressive or consistency-based multiple alignment be materially better, and is that
  worth the complexity for a personal tool?
- **`pipeline_runner.py::_discard_variant_wavs`** deletes files. Verify it cannot delete
  anything outside the run's own variants directory, and that its placement after
  diarisation is correct (diarisation re-opens `variant_paths["original"]`).
- The **degeneracy guard** threshold (3-gram > 20%, minimum 30 words) is a heuristic
  pulled from one observed failure. Does it false-positive on legitimately repetitive
  audio?

---

## 5. Known open items — do not re-report these as discoveries

- **RC-6** — `export_engine/exporter.py` is 827 lines across six export formats.
  Reviewed as low priority, well-tested and stable. Deliberately not split.
- **Consensus WER claim unproven** — see §2. Known, not hidden.
- **No long-form benchmark** — no representative ground truth exists. Known gap.
- **21 GB of historical variant WAVs** still on the maintainer's disk. The fix prevents
  recurrence; the existing files are the maintainer's data to remove.
- **36 stale remote branches** on the origin from earlier merged work.
- **One open GitHub issue** (#194) — a genuine Ollama model tag that no longer resolves.
- **Test pollution** — some tests write artefacts (e.g. `test_srt_consensus.srt`) into the
  real `outputs/consensus/` rather than a temp directory.
- **`VERSION` says 4.1.0 with 28 commits since the tag** — release bump pending, deliberate.

---

## 6. Specific questions worth your attention

1. **Is the alignment fix actually correct?** Read `_build_multi_alignment` and its tests
   in `tests/test_sequence_alignment.py::TestMultiAlignmentColumnIntegrity`. Construct
   inputs the tests do not cover. Does any input produce dropped or reordered tokens?
2. **Is the confidence tiering sound?** `consensus_merger/alignment.py` computes
   `confidence = count / n_transcripts` where `count` is the largest fuzzy-matched group,
   then HIGH at `>= 0.75`, MEDIUM at `count >= 2`, else LOW. Note a column where three
   variants agree and a fourth is *absent* scores 3/4, the same as three agreeing with a
   fourth *dissenting*. Is conflating "absent" with "dissenting" defensible?
3. **Thread safety.** Whisper is not thread-safe (its KV cache is keyed by the model's own
   `Linear` module objects). `orchestrator.py::_max_safe_parallelism` caps concurrency to
   1 unless multiple CUDA devices are present. Verify the cap cannot be bypassed —
   corruption here previously destroyed six files in an overnight batch.
4. **Background run lifecycle.** `ui/run_manager.py`, `run_worker.py`, `run_state.py` run
   a pipeline in a plain thread with no Streamlit context, writing an atomic JSON state
   file. Look for races, orphaned threads, and stale-state handling across server restart.
5. **Is the test suite load-bearing or decorative?** 430 tests passed while both RC-10 and
   RC-11 were live. That is the single most important signal in this repo. Which tests
   would have caught them, and what is the smallest set of new tests that would catch the
   next one?
6. **Release readiness.** Given §2, what should the release notes claim, and what version
   number is right?

---

## 7. Deliverable

Produce a review document following §3.3's structure, with the additions below.

1. **Your Layer 1 evaluation criteria, and any questions for the maintainer — first,
   before any findings.** Wait for confirmation before proceeding.
2. **Executive summary** — 3–5 sentences: top risk, most urgent fix, overall health.
3. **The eight phases** (§3.3). Every risk gets a category and a 1–5 score; everything
   scoring ≥ 3 gets a full predictive failure scenario with trigger condition and
   timeline.
4. **Every finding labelled CONFIRMED or PLAUSIBLE.** CONFIRMED means you demonstrated
   it — you ran something and it failed. PLAUSIBLE means reasoned but unverified. Never
   blur the two, and never present a PLAUSIBLE finding in language that implies
   CONFIRMED. State the evidence for each.
5. **Action roadmap** in the §3.3 item format, prefixed **`RD-`**, each executable by an
   agent that has not read the review.
6. **Top three improvements**, ranked.
7. **An explicit release recommendation** — cut now or not, what version number, and what
   the release notes may honestly claim given §2.
8. **What you deliberately did not review, and why.**

Write the document to `REVIEW-RD.md` if you have file access; otherwise output it inline.
**Do not open PRs, do not push, do not modify `REVIEW.md` or `ROADMAP.md`.** Report
findings; the maintainer decides what gets actioned.

---

# Handover

Everything below is for taking the work over, not just reviewing it.

## 8. Environment and how to run it

**Repo:** `git clone https://github.com/incendiary/Chorus.git`
**Python:** 3.11+ required; the maintainer's venv runs 3.14.6.
**Never install anything system-wide** — always use the project-local venv.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # dev extras provide jiwer, pytest, black, ruff, isort
```

**Run the UI:**

```bash
streamlit run ui/app.py
```

Streamlit dies when its terminal closes. For an overnight run use `tmux` or `nohup` —
this has bitten the maintainer before.

**Run the batch CLI:** `python3 -m batch_processor.batch_runner <path>`
**Run the benchmark:** `python3 -m benchmarks.run_benchmark` (downloads LibriSpeech
test-clean, ~346 MB, cached in `benchmarks/data/`; `--limit 2` for a smoke run)

**Tests and gates — all four must pass before any PR:**

```bash
python3 -m pytest tests/ -q
python3 -m black --check .
python3 -m ruff check .
python3 -m isort --check-only .
```

**Configuration** lives in `.env` at repo root (not committed; `.env.example` is the
template). `config.py` parses it with a small stdlib loader — real environment variables
always win over `.env`. The maintainer's `.env` sets `WHISPER_MODEL`, `WHISPER_DEVICE`,
`WHISPER_LANGUAGE`, `TRANSCRIPTION_PARALLELISM`, the three `OLLAMA_*` values, and
`HUGGINGFACE_TOKEN` (needed for diarisation; a **read** token, never a write token).

**Note:** there is no `local-ci.sh` in this repo, despite the maintainer's general
workflow referencing one. CI runs on GitHub only.

## 9. Current state

- **Branch:** `main`, in sync with `origin/main`, clean working tree
- **Last commit:** `f701304` — *docs: record RC-1 and RC-8 to RC-11 in the roadmap (#213)*
- **`VERSION`:** `4.1.0` — **28 commits behind the tag**; a release bump is pending and
  deliberate
- **Tests:** 430 passing; black/ruff/isort clean
- **Open issues:** one (#194), a genuine Ollama tag that no longer resolves
- **Stale remote branches:** 36, from earlier merged work — safe to prune, nobody has

## 10. Decisions that are final — do not re-litigate

Re-opening any of these wastes time that has already been spent:

- **Word timestamps stay off by default.** They are opt-in via `WORD_TIMESTAMPS=1` or
  `transcribe(word_timestamps=True)`. Turning them back on globally reintroduces RC-10.
- **The denoise filter is not the problem.** It was investigated and cleared — 100% of
  signal energy is retained. This was a wrong hypothesis; do not revisit it.
- **Do not "recalibrate" the confidence thresholds to raise the HIGH rate.** This was
  proposed and rejected: with the bugs live it would have relabelled hallucinations as
  trustworthy. The tiers were never miscalibrated; the inputs were broken.
- **`condition_on_previous_text` is not the cause of anything.** Tested and disproved —
  the `large` model produced *more* words with it on (2,404 versus 1,890).
- **Transcription parallelism stays capped at 1** unless multiple CUDA devices are
  present. Whisper is not thread-safe; raising this destroyed six files in an overnight
  batch. Do not "optimise" it.
- **RC-6 (splitting the 827-line `export_engine/exporter.py`) is deliberately not done.**
  Low priority, well-tested, stable. Leave it unless there is a concrete reason.
- **The benchmark is known to be unrepresentative** and is retained as a regression gate
  on clean short-form audio only. Do not present its numbers as evidence about long-form
  or noisy performance.

## 11. The next chunk of work

**Objective:** cut release **v4.2.0**.

Minor, not patch: RC-10 and RC-11 materially change output quality, and RC-1 changes
on-disk behaviour. Seven PRs (#207–#213) have landed since v4.1.0.

**Before tagging — the one thing genuinely outstanding.** The last full production run
was on the *broken* code. Everything since is verified by tests plus targeted
measurements against saved transcripts, but **nobody has run the whole UI pipeline
end-to-end on real audio since RC-11 landed.** Both headline bugs were found by running
the software, not by reading it or by tests. Ask the maintainer whether to do this run
before tagging; recommend that they do.

**Steps:**
1. Confirm with the maintainer: proceed to release, or validate on real audio first.
2. Branch. Bump `VERSION` to `4.2.0`. Check `README.md` for any version pin that must
   move with it.
3. Move the RC items into a `## Completed — v4.2.0` section in `ROADMAP.md`, following
   the existing per-version structure.
4. Draft release notes. **They must not claim consensus improves accuracy** — see §2.
   The defensible claim is calibrated uncertainty plus two significant correctness fixes.
5. PR, CI green, squash merge.
6. Tag `v4.2.0` and create the GitHub release; the tag, `VERSION`, and the release must
   all agree.

**Done when:**
- [ ] `VERSION`, git tag, and GitHub release all read `4.2.0`
- [ ] `ROADMAP.md` has no open item that is actually complete
- [ ] Release notes make no unevidenced accuracy claim
- [ ] CI green on `main`

**Optional follow-on work, in rough priority order:**
1. **Source long-form ground truth** so the founding WER claim can finally be tested —
   the single highest-value thing left. A human transcript of a real recording, or
   `large-v3` output as a reference proxy if the maintainer accepts that.
2. Reconsider **reference selection in alignment** — the longest transcript anchors it,
   which on real audio meant the most divergent variant became the reference.
3. Fix **test pollution** — some tests write into the real `outputs/consensus/`.
4. Prune the 36 stale remote branches.

**Return signal:** commit as `chore: release v4.2.0`, then report the tag, the release
URL, and explicitly what the notes do and do not claim.
