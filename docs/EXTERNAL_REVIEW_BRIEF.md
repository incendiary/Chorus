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
**Review target:** `main` at commit `4d13219`. Everything is merged and pushed; there is
no feature branch to check out. `VERSION` says `4.2.0`, but **no `v4.2.0` git tag or
GitHub release exists yet** — a scheduled workflow has an open issue (#242) flagging
exactly this. Whether to tag now or gate it behind this review's findings is the first
decision for the maintainer; see §11.

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

**Scale:** ~19,900 lines of Python total, ~9,100 lines of it tests across 29 test files,
491 tests passing (2 skipped). `ui/` (3,873) and `export_engine/` (1,199) are still the
largest packages.

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

**Nothing here has changed since the 28 July 2026 review.** No new benchmark has run and
no long-form ground truth has been sourced — this section is carried forward verbatim
because it is still true, not because it was re-verified this round. The work since then
(§4) was almost entirely about diarisation correctness and CLI/Web UI reliability, not
about the WER claim. Do not treat "unchanged" as "resolved."

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
dependency audit), `release.yml`, `ollama-model-tags-check.yml`. There is no
`local-ci.sh` in this repo. Real pre-commit and pre-push git hooks were installed this
session (`pre-commit install` for both `--hook-type pre-commit` and
`--hook-type pre-push`) — the hook previously present was an unrelated stub from a
different tool that referenced a missing script and was designed to always exit `0`
regardless, so none of GitLeaks/Ruff/isort/file-hygiene was actually enforced locally
before now. Verify the hooks are still real, not silently reverted to a no-op stub.

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

Twenty-three commits landed since the 28 July review's baseline (`f701304`), across two
version bumps. **This review has not happened yet against any of it** — nobody has
adversarially checked this work the way the 28 July review checked RC-10/RC-11.

### v4.1.1 — determinism and release hardening (#214–#221)

Closed out the 28 July review's remaining action items: deterministic consensus
alignment (stable tie-breaking, anchor-local insertion alignment, order-independent
fuzzy groups), process-wide `RunManager` run exclusion, honest agreement-not-accuracy
language in generated output, dependency/import-time cleanup, and Streamlit binding to
loopback by default with a warning on deliberate exposure.

### v4.2.0 — diarisation correctness and CLI/Web UI parity (#222–#237)

Prompted by a real, ~20-hour production run on nine real recordings for an active
financial ombudsman complaint (evidentiary audio — see `ROADMAP.md`'s Validation note
under this version), which surfaced two distinct diarisation failure modes that every
prior test suite had missed:

| PR | Change |
|---|---|
| #226 | Batch CLI config overrides, hardware presets, and effective-setting provenance, matching the Web UI |
| #227 | `survey-ollama-env.sh` never actually recommended or wrote `WHISPER_DEVICE` — always detected GPU type correctly but never turned it into a device recommendation |
| #228 | Batch CLI runs persist a log file — previously the only record of an unattended run was console output, already lost by the time a multi-hour slowdown needed investigating |
| #229 | `--diarise` refuses to start rather than silently completing every file with a single-speaker stub, if `check_diarisation_ready()` (a real pipeline load) fails |
| #230–#231 | The pre-flight check's guidance stopped hardcoding which HuggingFace repos matter — an early version named specific gated repos, then the installed `pyannote.audio` version turned out to route through a *different* set of repos than its own Hub `config.yaml` implied, within the same evening |
| #232 | `--check-diarisation` — a standalone readiness check, so verifying diarisation works no longer means starting a real batch and watching it refuse |
| #233 | The Web UI got the same pre-flight check as the CLI (previously CLI-only) |
| #234 | `diarisation_repo_status()` checks every known dependency repo via an authenticated HEAD request to an actual weight file — `HfApi.model_info()`'s gating field had reported every repo accessible while one still hard-403'd for hours, and "the first file returned" is `.gitattributes` for some repos, always public even when gated |
| #235 | **The actual diarisation crash.** `pyannote.audio` 4.x's default pipeline wraps its result in a `DiarizeOutput` dataclass instead of the bare `Annotation` older versions returned. `diarise()` called `.itertracks()` on whatever it got back, unconditionally — diarisation could run for over an hour of real compute and then crash at the final parsing step, caught by a broad `except` in `pipeline_runner.py`, so the file "succeeded" with no `diarised.md` at all |

### Post-v4.2.0, not yet in any tagged release (#238–#243)

Found and fixed *after* the nine-file production run, while preparing to process a new,
unrelated recording for the same case:

- **#238 — single-run lock for the batch CLI.** Two overlapping `batch_runner`
  invocations against the same `--output-dir` collided during the nine-file run itself:
  both processed the same file within seconds of each other, and one's variant-WAV
  cleanup deleted a file the other still needed for diarisation, deadlocking for **over
  ten hours** before being killed manually. `main()` now refuses to start if a live PID
  already holds the lock for that directory.
- **#239 — a second, distinct diarisation silent-failure mode.** On the very next
  production run (a single new recording), diarisation failed again — this time
  `torchcodec` 0.14.0 couldn't decode audio against the host's installed ffmpeg 9 (it
  only shipped decoder support for ffmpeg 4–8), and `pipeline_runner.py`'s bare
  `except Exception: logger.warning(...)` around the whole diarisation block produced a
  transcript with no speaker labels and no visible error — the batch still reported
  "1/1 succeeded". Note that **this is the *third* distinct diarisation failure mode
  found in as many weeks** (credential kwarg naming, `DiarizeOutput` shape, now a
  runtime decode failure) — see §6 for the question this should raise about the pattern.
  `run_pipeline()` now returns the error as `diarisation_error` rather than only logging
  it; surfaced in the batch report, the console summary, and a Web UI warning.
  `torchcodec` was bumped 0.14.0 → 0.16.0 and pinned explicitly (previously an unpinned
  transitive dependency of `pyannote-audio`).
- **#240 — `PYSEC-2026-3624`/`CVE-2026-58659` documented as an accepted risk.**
  `pip-audit` flags an RCE in `lightning` (pulled in by `pyannote-audio`) on every CI
  run. The fix merged upstream 2026-07-14 but has not shipped in a release — `2.6.5`
  (2026-05-27) is still the latest, and predates the fix. No new tag has been cut since,
  over 50 days and 15+ unrelated commits later. Recorded in `SECURITY.md`, cross-linked
  with tracking issue #219. **Verify the reachability claim**: Chorus never calls
  `LightningModule.load_from_checkpoint` directly and only loads pyannote's own pinned
  weights — is that actually the whole exploit surface, or does something in the
  dependency chain call it indirectly?
- **#241 — real pre-commit/pre-push git hooks installed.** See §3.1's note — the hook
  previously present did nothing.
- **#243 — the previous version of *this document*, plus a working-notes file, folded
  into `ROADMAP.md`.**
- **#236, #224, #223, #215, #214 — five Dependabot bumps**, one of which (`streamlit`
  1.59.2 → 1.63.0, landed via two successive Dependabot force-pushes to the same PR
  branch, 1.62.0 then 1.63.0) had a genuine, undocumented breaking change:
  `AppTest.from_file`'s relative-path resolution changed from CWD-relative to
  relative-to-the-calling-file, breaking 26 tests across 4 files that passed literal
  strings like `"ui/app.py"`. Fixed by resolving every such path via `Path(__file__)`.
  Separately, Dependabot's `spacy` target (3.8.15) had been yanked from PyPI entirely by
  the time it was actioned — bumped to the next real release (3.8.16) instead.

**Please attack these specifically:**

- **`consensus_merger/sequence_alignment.py::_build_multi_alignment`** is unchanged
  since 28 July and remains the highest-risk code in the repo — see the original
  question below, still open. Reference-based (star) alignment is used, with the
  *longest* transcript chosen as reference. On real audio that meant the **most
  divergent variant anchored the alignment**. Is that defensible? Would a progressive or
  consistency-based multiple alignment be materially better, and is that worth the
  complexity for a personal tool?
- **`diarisation/diariser.py::diarise`** and **`pipeline_runner.py`'s diarisation
  block.** Three failure modes found in three weeks in this exact code path. Read §6's
  question about whether the pattern itself, not just each individual bug, needs
  addressing.
- **`batch_processor/batch_runner.py`'s new lock functions**
  (`check_batch_lock`/`acquire_batch_lock`/`release_batch_lock`). Check for a
  check-then-act race: two processes could both call `check_batch_lock()`, both see no
  lock, and both proceed to `acquire_batch_lock()` — is there an actual window for this
  on a local filesystem, and does it matter given the lock is advisory (not `flock`-based)?
- **`pipeline_runner.py::_discard_variant_wavs`** deletes files. Verify it cannot delete
  anything outside the run's own variants directory, and that its placement after
  diarisation is correct (diarisation re-opens `variant_paths["original"]`).
- The **degeneracy guard** threshold (3-gram > 20%, minimum 30 words) is a heuristic
  pulled from one observed failure. Does it false-positive on legitimately repetitive
  audio?

---

## 5. Known open items — do not re-report these as discoveries

- **RC-6** — `export_engine/exporter.py` is 1,199 lines across six export formats.
  Reviewed as low priority, well-tested and stable. Deliberately not split.
- **Consensus WER claim unproven** — see §2. Known, not hidden.
- **No long-form benchmark** — no representative ground truth exists. Known gap.
- **`VERSION` says `4.2.0` with no matching git tag or GitHub release** — issue #242,
  auto-filed. Deliberately unresolved pending this review; see §11.
- **Four CLI settings have no Web UI equivalent** (word-level timestamps, WAV retention,
  Ollama base URL/timeout) — `ROADMAP.md`'s "Planned — post-v4.2.0 cleanup" section.
- **A comprehensive CLI/Web UI flag reference doc doesn't exist yet** — README documents
  roughly 6 of 22 flags in passing. Same ROADMAP section.
- **A cosmetic provenance-label bug** — paired boolean flags
  (`--word-timestamps`/`--no-word-timestamps`) always report the positive flag's name in
  the settings-table source column regardless of which was actually passed. The printed
  *value* is correct; only the label is wrong. Same ROADMAP section.
- **An unexplained process death, 2026-08-23/24** — a batch process died silently
  between files mid-run, no error, no OOM signal captured. Not diagnosed; nothing
  prescribed to fix until it recurs with more evidence. Same ROADMAP section.
- **Open GitHub issues:** #194 (a genuine Ollama model tag that no longer resolves), #219
  (tracking the `lightning` CVE, cross-referenced with `SECURITY.md`), #220 (a bounded
  AMI multi-speaker benchmark, proposed but not built), #242 (the missing v4.2.0 tag,
  above).
- **Test pollution** — some tests write artefacts (e.g. `test_srt_consensus.srt`) into the
  real `outputs/consensus/` rather than a temp directory. Confirmed directly during the
  September production runs; not yet fixed.
- **Remote branch hygiene is currently clean** — down from 36 stale branches at the last
  review to zero; every merged PR branch was deleted on merge, and one stale
  fully-superseded duplicate (`codex/batch-cli-config-provenance`, byte-identical to the
  already-merged #226) was found and deleted this session. Re-verify this hasn't drifted
  by the time you read this.

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
   RC-11 were live; 491 tests pass today, after three more diarisation bugs and a
   10.5-hour deadlock all reached production before being caught. Which tests would have
   caught each of them, and what is the smallest set of new tests that would catch the
   next one in this same code path?
6. **Is the diarisation failure pattern itself the real finding, not any individual bug?**
   Three distinct failure modes in the same subsystem in three weeks: a wrong credential
   kwarg name (#229-era), a pyannote result-shape mismatch (#235), and a runtime audio-
   decode failure (#239). Each was fixed individually and each fix added another
   layer of defence (pre-flight checks, then per-file error surfacing), but nobody has
   asked whether `diarisation/diariser.py`'s fundamental approach — wrapping an external,
   fast-moving library (`pyannote.audio`) with no version-compatibility shim — is
   structurally prone to this, and what would actually stop a fourth one.
7. **Is the single-run lock (#238) actually safe, or does it just narrow the window?**
   It is a plain PID-file check, not `flock`/`fcntl`-based. Confirm whether the
   check-then-write gap between `check_batch_lock()` and `acquire_batch_lock()` is a real
   TOCTOU race on a local filesystem with two processes started close together, and
   whether that matters in practice given how the collision that motivated it actually
   happened (two manually-started overnight runs, not a tight race).
8. **Release readiness.** Given §2, and given three post-release diarisation bugs found
   in the two weeks after v4.2.0's feature work merged, what should the release notes
   claim, and is `v4.2.0` even the right version number for what's actually shipping, or
   should the post-release fixes (#238–#241) bump it further?

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
**Python:** 3.11+ required; the maintainer's venv runs 3.14.7.
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
python3 -m black --check .   # never pass --exclude here — it overrides pyproject.toml's
                              # own excludes and wipes the .venv exclusion, which has
                              # twice caused black to reformat-check the entire virtualenv
python3 -m ruff check .
python3 -m isort --check-only .
```

Real pre-commit and pre-push git hooks are installed as of #241
(`pre-commit install` + `pre-commit install --hook-type pre-push`) — GitLeaks/Ruff/isort/
file-hygiene run on commit, the full `pytest` suite runs on push. Confirm they are still
live and not reverted to the earlier no-op stub before trusting a clean local run.

**Configuration** lives in `.env` at repo root (not committed; `.env.example` is the
template). `config.py` parses it with a small stdlib loader — real environment variables
always win over `.env`. The maintainer's `.env` sets `WHISPER_MODEL`, `WHISPER_DEVICE`,
`WHISPER_LANGUAGE`, `TRANSCRIPTION_PARALLELISM`, the three `OLLAMA_*` values, and
`HUGGINGFACE_TOKEN` (needed for diarisation; a **read** token, never a write token).

**Note:** there is no `local-ci.sh` in this repo, despite the maintainer's general
workflow referencing one. CI runs on GitHub only.

## 9. Current state

- **Branch:** `main`, in sync with `origin/main`, clean working tree
- **Last commit:** `4d13219` — *docs: fold the Coventry Case fix-list into ROADMAP.md,
  document the pre-push hook (#243)*
- **`VERSION`:** `4.2.0` — **no git tag or GitHub release exists yet** (issue #242);
  whether to cut one is a live decision, not settled
- **Tests:** 491 passing, 2 skipped; black/ruff/isort clean; real pre-commit/pre-push
  hooks installed and verified working
- **Open issues:** #194 (stale Ollama tag), #219 (tracking the `lightning` CVE), #220
  (proposed AMI multi-speaker benchmark, not built), #242 (missing v4.2.0 tag)
- **Remote branches:** just `main` and this review branch — clean, one stale duplicate
  pruned this session

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
- **The unpatched `lightning` CVE is an accepted, documented risk, not something to
  silently suppress in CI.** No upstream fix has shipped. The `Python Dependency Audit`
  check is expected to stay red on every PR until it does — do not propose CI config
  changes to make it pass; do check whether the reachability analysis in `SECURITY.md`
  and issue #219 is actually still correct.
- **A hook that never blocks is worse than no hook.** The stub previously in
  `.git/hooks/pre-commit` always exited `0` regardless of what it found. Do not
  reintroduce anything with that property, even as a stopgap.

## 11. The next chunk of work

**Objective:** decide whether to tag **v4.2.0** now, and either cut it or say precisely
why not — this review is the gate the maintainer is deliberately waiting on before that
decision, per issue #242.

Unlike the last handover, the release's *feature* work (#222–#237) is already merged,
tested, and validated against a real 9-file, ~20-hour production run on evidentiary
audio (`ROADMAP.md`'s Validation note). What's genuinely unresolved is everything found
*after* that validation run, on the very next production use (#238–#241): a 10.5-hour
deadlock, a second distinct silent diarisation failure, and the discovery that the local
git hooks had never actually been enforcing anything. All of that is now fixed and
merged to `main` too — but nobody has adversarially reviewed whether those fixes are
sound, or whether the pattern behind three diarisation bugs in three weeks (§6, question
6) needs something structural rather than another patch.

**Before recommending tag or no-tag, address directly:**
1. Is the diarisation subsystem's repeated-failure pattern (§6, question 6) a
   release-blocking concern, or three independent, now-fixed bugs?
2. Is the single-run lock (§6, question 7) actually correct, or does it need `flock`
   before it's trustworthy for unattended overnight runs?
3. Does anything in #238–#241 warrant its own regression the way #235's `DiarizeOutput`
   fix and #239's diarisation-error surfacing already got one? Check coverage, don't
   assume it from the PR having tests.
4. Given three post-release fixes landed *after* the feature work that would have
   defined v4.2.0, is `4.2.0` still the right version number, or should the tag include
   `.1`/`.2` patch history, or should the whole thing bump to `4.3.0`? The maintainer's
   own versioning rule (§3.1) says patch = "a batch of related PRs" — #238–#241 read as
   exactly that.

**Steps once a recommendation is reached:**
1. Report the recommendation to the maintainer with the reasoning above; this is their
   decision, not yours to make unilaterally (§0).
2. If proceeding: confirm `VERSION`, the intended tag, and any final `ROADMAP.md`
   entries are in sync (`bash tests/version_consistency_test.sh` checks most of this
   automatically).
3. Tag and create the GitHub release; the tag, `VERSION`, and the release must all
   agree. Close issue #242 referencing the release.
4. If not proceeding: say exactly what must happen first, and file it the same way
   `ROADMAP.md`'s existing "Planned" sections are structured, not as a vague caveat.

**Done when:**
- [ ] A tag/no-tag recommendation has been given, with reasoning grounded in this
      review's findings, not just repeated from this brief
- [ ] If tagging: `VERSION`, git tag, and GitHub release all agree, and issue #242 is
      closed
- [ ] `ROADMAP.md` has no open item that is actually complete
- [ ] CI green on `main` (aside from the accepted `lightning` CVE)

**Return signal:** report the recommendation and, if a tag was cut, its URL and exactly
what the release notes do and do not claim (§2 still applies — no unevidenced accuracy
claim).
