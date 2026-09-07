# Chorus Engine — Holistic Review (RD series)

**Reviewed:** `main` at `681b6e5`, 6 September 2026
**Method:** `codebase-holistic-review`, eight phases
**Prefix:** `RD-` (fourth review; `RA-`/`RB-`/`RC-` precede it)

Every finding is labelled **CONFIRMED** (demonstrated by running something or by direct
code-read of the executing path) or **PLAUSIBLE** (reasoned, not executed). Null findings
are recorded rather than skipped.

---

## Executive summary

The repository is in good health: 27 test files against 11 packages, all 16 runtime
dependencies pinned, three layers of secret scanning, genuine pre-commit and pre-push
hooks, and a README that is already honest about the unproven accuracy claim. The top
risk is not a new defect but a **structural one: two of the three subsystems that failed
in production this quarter can still fail silently**, because the repeat-loop guard only
logs and the diarisation loader still degrades to a fake speaker. The most urgent fix is
the repeat-loop guard, because a degenerate transcript is not merely counted — it is
*preferentially selected* as the alignment backbone, since the aligner picks the longest
transcript and a repetition loop is artificially long. Separately, the standing decision
to let CI's dependency audit stay red has done exactly what that pattern always does: a
second, undocumented advisory (`nltk`) has been sitting in the red output unnoticed.
Overall: ship-worthy after three small fixes, with a clear-eyed set of deferred items.

---

## Phase 1 — Architecture map

Layers: **ingestion** (`audio_processor`) → **inference** (`transcription_engine`,
`diarisation`) → **synthesis** (`consensus_merger`, `reconstruction`) → **presentation**
(`export_engine`, `ui`, `batch_processor`), with `config.py` and `pipeline_runner.py` as
cross-cutting infrastructure.

**Data flow.** Audio enters via the Streamlit uploader (`ui/`) or a CLI path argument
(`batch_processor/`). `pipeline_runner.run_pipeline()` orchestrates: produce four audio
variants → transcribe each → align and vote → optionally diarise → optionally reconstruct
LOW tokens → export. Output leaves as files under `outputs/`, plus a JSON bundle and an
AI-context pack. Side effects: model downloads, HuggingFace network calls during
diarisation, variant WAV deletion, and a PID lock file for batch runs.

| Module | LOC | Responsibility | Concerns |
|---|---|---|---|
| `audio_processor/` | 467 | Produce four audio variants via pydub/librosa | None material. Clean single responsibility. |
| `transcription_engine/` | 685 | Whisper wrapper, model cache, parallelism policy | Owns the repeat-loop guard, which detects but does not act (**RD-1**). |
| `consensus_merger/` | 953 | Needleman-Wunsch alignment, fuzzy grouping, tier assignment | Highest-risk code. Star alignment anchored on the longest variant; gap/dissent conflation (**RD-5**). |
| `diarisation/` | 638 | pyannote wrapper, pre-flight checks, speaker labelling | Three production failures in three weeks. Wraps a fast-moving library with an ad-hoc `hasattr` shim, no version gate (**RD-2**, **RD-6**). |
| `reconstruction/` | 590 | Repair LOW tokens via spaCy or Ollama | None material. Clean strategy dispatch. |
| `export_engine/` | 1,199 | PDF, DOCX, SRT, VTT, bundle, context pack | Large but stable and well-tested. RC-6 deliberately closed; not re-raised. |
| `batch_processor/` | 929 | Unattended CLI, config resolution, single-run lock | Lock is advisory PID-file with a TOCTOU window (**RD-3**). |
| `ui/` | 3,873 | Streamlit dashboard, background run lifecycle | Largest package. Mixes rendering, state, and orchestration; `_results` is shared across threads unsynchronised (**RD-7**). |
| `pipeline_runner.py` | 548 | Stage orchestration | Broad `except Exception` around diarisation, now defensible because the error is surfaced. |
| `security/`, `benchmarks/`, `chorus/` | 672 | Scanning helpers, WER harness, package shim | None material. |
| `tests/` | 9,112 (27 files) | — | Strong in places, decorative in others (**RD-8**). |

**Modules without a single clear responsibility:** `ui/` only. Splitting it is not
recommended for a personal tool; noted, not actioned.

---

## Phase 2 — Risk inventory

| # | Category | Finding | Score | Location |
|---|---|---|---|---|
| 1 | Reliability | Repeat-loop guard logs only; degenerate variant votes at full weight **and** is preferentially chosen as the alignment reference | **5** | `transcription_engine/whisper_engine.py:249`, `consensus_merger/sequence_alignment.py:173` |
| 2 | Reliability | Diarisation still degrades silently to a single fake `SPEAKER_00` with `diarisation_error=None` | **4** | `diarisation/diariser.py:304-307`, `:343-344` |
| 3 | Reliability | Test suite deletes the real `outputs/active_run.json` before and after every test | **4** | `tests/conftest.py:18-32` |
| 4 | Security | `nltk` PYSEC-2026-3740 flagged by CI, absent from `SECURITY.md`, masked by the accepted-red policy | **3** | `SECURITY.md:41-52`, CI dependency audit |
| 5 | Reliability | Batch lock is advisory PID-file: TOCTOU window, fails open on a malformed file, no release on SIGKILL | **3** | `batch_processor/batch_runner.py:244-302`, `:880-883` |
| 6 | Maintainability | Diarisation pre-flight never runs inference, so it cannot catch result-shape or decode failures | **3** | `diarisation/diariser.py:183-204` |
| 7 | Correctness | `n_transcripts` fixed globally and computed before empty variants are filtered; absence and dissent are numerically identical | **3** | `consensus_merger/sequence_alignment.py:279`, `:282`, `:293` |
| 8 | Maintainability | Tests that pass against broken logic: length-only assertions, either-way tier chains, a soft 80% statistical gate | **3** | `tests/test_sequence_alignment.py:135`, `:315`; `tests/test_alignment.py:102`, `:113` |
| 9 | Dependency | No GitHub Action is SHA-pinned; `ci.yml:101` and `release.yml:47` `pip install` five packages unpinned, bypassing `requirements.txt` | **3** | `.github/workflows/*.yml` |
| 10 | Dependency | 10 of 13 dev-extra tools float; CI pins `ruff==0.4.7` for lint while `[dev]` floats — two ruff versions | **2** | `pyproject.toml:48-61`, `ci.yml:56` |
| 11 | Maintainability | 11 test artefacts written into the real `outputs/consensus/` | **2** | `tests/test_exporter.py:78-140`, `tests/test_merger.py:59-80` |
| 12 | Reliability | `clear_finished` unlinks state outside the lock; `_results` dict mutated by worker and read by Streamlit threads unsynchronised | **2** | `ui/run_manager.py:35`, `:95-101` |
| 13 | Maintainability | `TypeError` absent from `_try_load_pipeline`'s except tuple; `ui/sidebar.py:504` calls the checker unguarded | **2** | `diarisation/diariser.py:150-180` |
| 14 | Maintainability | Stale duplicate package tree at `build/lib/consensus_merger/`; `_score_pair` is dead code | **1** | `build/lib/`, `consensus_merger/sequence_alignment.py:53` |
| 15 | Dependency | No lockfile or hashes; `lightning` and `torch` float transitively | **2** | `requirements.txt` |
| 16 | Reliability | Both Docker images fail to build: a stale NLTK pre-download step survives although `nltk` is neither a declared dependency nor imported anywhere | **4** | `Dockerfile:38-43`, `:75-77`; `Dockerfile.gpu:47-52`, `:85-87` |

### Null findings (checked, nothing wrong)

- **Whisper parallelism cap holds.** `orchestrator.py:44-61` feeds `_clamp` (`:73-85`),
  which bounds the executor at `:318`. No config, env, or CLI path bypasses it. The
  six-file corruption incident cannot recur through this route. **CONFIRMED.**
- **`_discard_variant_wavs` cannot escape its directory.** `pipeline_runner.py:78-84`
  compares `path.parent.resolve()` against `variants_dir.resolve()` by exact equality —
  no traversal or sibling escape. Placement after the diarisation block (`:494` vs
  block ending `:486`) is correct and commented. **CONFIRMED.**
- **README makes no unevidenced accuracy claim.** `README.md:8-11` and `:180-199`
  already state consensus was broadly on par with single-pass and that tiers are not
  calibrated error probabilities. **CONFIRMED.**
- **RC-11's column-shifting bug is genuinely fixed.** Columns are pre-created per
  reference token (`sequence_alignment.py:203`) and insertions accumulate into
  `insertion_runs` rather than mutating the list. No shifting is possible. **CONFIRMED.**
- **Branch hygiene is clean.** `origin` carries only `main`. **CONFIRMED.**
- **Hooks are real.** Both `.git/hooks/pre-commit` and `pre-push` are genuine
  pre-commit-framework hooks, not the old always-exit-0 stub. **CONFIRMED.**
- **`.env` is gitignored and was never committed.** **CONFIRMED.**
- **All 16 runtime dependencies are `==` pinned** in both manifests. **CONFIRMED.**

---

## Phase 3 — Predictive failure scenarios (score ≥ 3)

### PF-1: A repetition loop silently becomes the alignment backbone (Risk 1, score 5) — CONFIRMED

**What happens.** Whisper collapses into a repetition loop on one variant. The guard at
`whisper_engine.py:249` writes a log warning and nothing else — it sets no flag on the
result and does not exclude the variant. The looping transcript is *longer* than the
healthy ones, so `sequence_alignment.py:173`'s `min(..., key=lambda k: (-len(...), k))`
selects it as the alignment reference. Every healthy variant is then aligned against
garbage. The observable output is a transcript with a collapsed HIGH-confidence rate and
scattered agreement — the exact signature of RC-11, arriving by a different route.

**Trigger condition.** Any run where one variant degenerates. Historically triggered by
`word_timestamps=True` on long-form audio (RC-10), now off by default — but the loop is
a property of Whisper on difficult long-form audio, not solely of that flag. Denoised or
high-pass variants of a poor phone recording are the likely candidates.

**Estimated timeline.** Today, on the next long-form real recording that pushes one
variant into a loop. Not hypothetical: this is how the nine-file batch produced 137-word
transcripts of 29-minute recordings.

**Minimum fix.** Exclude a flagged variant from reference selection only.

**Full fix.** Additionally record the ratio on the result, surface it in the run report
and UI, and consider dropping the variant from the vote pool behind a threshold.

### PF-2: Diarisation reports a single fake speaker with no error (Risk 2, score 4) — CONFIRMED

**What happens.** `_load_pipeline` fails, logs a warning, returns `None`
(`diariser.py:304-307`). `diarise()` at `:343-344` then returns `_stub_diarisation()` —
one synthetic `SPEAKER_00` spanning the file — and `diarisation_error` stays `None`. The
run reports success. The user receives a diarised document asserting the recording has
exactly one speaker.

**Trigger condition.** Any pipeline-load failure on a path where the pre-flight was not
run: `--allow-diarisation-stub`, a direct `run_pipeline()` API call, or the Web UI if
the pre-flight is skipped. Expired HuggingFace token, revoked gating, or an offline host
all produce it.

**Estimated timeline.** Today. This is the same class of failure as #229/#235/#239, on a
path those fixes did not close.

**Minimum fix.** Make the stub reachable only under the explicit opt-in flag; otherwise
propagate a real error into `diarisation_error`.

**Full fix.** As above, plus stamp the stub output itself so a stub transcript is
self-identifying even if a caller ignores the error field.

### PF-3: Running the test suite destroys a live UI run (Risk 3, score 4) — CONFIRMED

**What happens.** `tests/conftest.py:18-32` is an `autouse` fixture, so it runs for
*every* test in the suite. It unlinks the real `ui/run_state.ACTIVE_RUN_FILE`
(`outputs/active_run.json`) before and after each test. A background run in flight loses
its state file; the UI shows no run, and the run's own `try/finally` later writes state
for a run the UI has forgotten.

**Trigger condition.** Running `pytest` while a Streamlit run is active. Given the
pre-push hook now runs the full suite, this fires on any `git push` during a long run —
and long runs here are multi-hour.

**Estimated timeline.** Today, and made *more* likely by #241's pre-push hook.

**Minimum fix.** Monkeypatch `ACTIVE_RUN_FILE` to a `tmp_path` for the fixture's scope.

**Full fix.** As above, plus assert in a test that the real path is never touched.

### PF-4: The next CVE hides in the permanently-red audit (Risk 4, score 3) — CONFIRMED

**What happens.** The dependency audit is expected to be red because of the unpatched
`lightning` advisory, so a red result carries no information. `nltk` PYSEC-2026-3740 has
been appearing in that output and is documented nowhere.

**Trigger condition.** Any new advisory against any transitive dependency. Already
happened once.

**Estimated timeline.** Already occurred; recurrence is certain over the life of the repo.

**Minimum fix.** Document the `nltk` advisory in `SECURITY.md` with its reachability
assessment. *Assessment for this one:* `nltk` enters only via `safety` (a dev/CI tool)
and `torchmetrics`' opt-in extras — it is absent from `requirements.txt` and never ships
to a user. No upstream fix exists; 3.10.3 is the latest release and is the flagged
version.

**Full fix.** Change the audit step to fail on any advisory *not* on an explicit
allowlist, so the signal returns.

### PF-5: Two batch runs collide despite the lock (Risk 5, score 3) — CONFIRMED (code-read)

**What happens.** `main()` (`batch_runner.py:880-883`) calls `check_batch_lock` then
`acquire_batch_lock` as two unsynchronised steps, and `acquire` unconditionally
overwrites. Two processes starting close together both see no lock and both proceed. A
malformed or truncated lock file returns "safe to proceed" (`:269-271`) — fail-open. The
lock is never released on SIGKILL, and PID-liveness is reuse-vulnerable in both
directions.

**Trigger condition.** Two overlapping invocations against the same `--output-dir`. The
motivating incident was two manually-started overnight runs, which the current lock
*does* catch. The residual window needs near-simultaneous starts — a shell loop, a
scheduler, or a double-click.

**Estimated timeline.** Not today under manual use. Likely the first time the CLI is
driven by anything automated.

**Minimum fix.** Replace the check-then-write pair with a single atomic
`O_CREAT|O_EXCL` create.

**Full fix.** `fcntl.flock` on the lock file, plus hostname and process start-time in
the payload so PID reuse and network output dirs are handled.

### PF-6: A fourth diarisation failure reaches production (Risk 6, score 3) — CONFIRMED (code-read)

**What happens.** The pre-flight verifies imports, token, and `Pipeline.from_pretrained`
succeeding. It never calls the pipeline on audio, never decodes, never touches
`itertracks`. So it passes, the user starts a multi-hour run, and the run fails at the
parsing or decoding step — precisely the shape of #235 and #239.

**Trigger condition.** Any pyannote or torchcodec upgrade that changes result shape or
decoder support. `pyannote-audio` is pinned at 4.0.7 and `torchcodec` at 0.16.0, so this
is dormant until the next bump — but `lightning` and `torch` float transitively beneath
them.

**Estimated timeline.** At the next dependency upgrade, not today.

**Minimum fix.** Extend the pre-flight to run inference over a one-second synthetic
silent WAV and parse the result, exercising load, decode, and result shape in about a
second.

**Full fix.** As above, plus an explicit supported-version range for `pyannote.audio`
with a clear error outside it, replacing the ad-hoc `hasattr` shim at `:359-362`.

### PF-7: Confidence understates agreement when a variant is empty or absent (Risk 7, score 3) — CONFIRMED (code-read)

**What happens.** Gaps are stripped before voting (`:293`), but `n_transcripts` is fixed
globally at `:279` — and computed *before* empty transcripts are filtered at `:282`. A
variant that produced nothing therefore permanently depresses every column's confidence,
capping unanimity at 0.75 in a four-variant run: exactly on the HIGH boundary. Separately,
absence and dissent cost the winner identically, so the tool cannot distinguish "this
variant did not hear the word" from "this variant heard something else".

**Trigger condition.** Any run where one variant yields an empty transcript (silence
detection, a decode failure) or has many gaps.

**Estimated timeline.** Today, on any run with a weak variant. The effect is silent
under-reporting of confidence, not a crash.

**Minimum fix.** Compute `n_transcripts` after the empty-variant filter at `:282`.

**Full fix.** Additionally track present-vs-absent per column and report them
separately, so a gap and a dissent are distinguishable in the vote record.

### PF-8: A future alignment regression passes the test suite (Risk 8, score 3) — CONFIRMED

**What happens.** Several tests assert only that output has a given length, or that a
tier is in a two-element set, or that at least 80% of agreed words reached HIGH. A change
that mis-tiers a fifth of all words, or reorders tokens while preserving count, passes.

**Trigger condition.** Any future edit to `consensus_merger/`.

**Estimated timeline.** At the next change to that package. Note 430 tests passed while
RC-10 and RC-11 were both live — this is the established failure mode here.

**Minimum fix.** Replace the length-only and either-way assertions with exact expected
token-and-tier sequences.

**Full fix.** As above, plus a property test asserting round-trip token preservation per
variant across randomly generated divergent inputs.

### PF-9: A supply-chain change lands through an unpinned action or install (Risk 9, score 3) — CONFIRMED

**What happens.** Every third-party action floats on a mutable tag, `gitleaks-action@v2`
loosest of all. A compromised or simply changed tag alters what runs in CI, including in
`release.yml`, which has `contents: write` and `packages: write`. Separately, `ci.yml:101`
and `release.yml:47` run a bare `pip install librosa soundfile scipy numpy openai-whisper`,
so the release's own test job validates versions other than the pinned ones shipped.

**Trigger condition.** An upstream tag move, or an upstream release of any of those five
packages.

**Estimated timeline.** The unpinned `pip install` drifts continuously and is drifting
today. The action-tag risk is low-probability, high-impact.

**Minimum fix.** Delete the ad-hoc `pip install` lines so `requirements.txt` governs.

**Full fix.** Additionally SHA-pin every third-party action, per the repo's own
reference-pinning rule.

---

### PF-10: The v5.0.0 release publishes no image because the build fails (Risk 16, score 4) — CONFIRMED

**What happens.** `docker build -f Dockerfile .` fails at the NLTK pre-download step with
`ModuleNotFoundError: No module named 'nltk'`. `nltk` is not a declared dependency and no
code imports it: `consensus_merger/alignment.py:90-92` documents inlining the Levenshtein
routine specifically to avoid pulling NLTK in. The dependency was removed but both
`Dockerfile` and `Dockerfile.gpu` kept the download step, the
`COPY --from=builder /root/nltk_data`, and the `NLTK_DATA` environment variable.

**Trigger condition.** Any image build. It is already broken; the only reason it has gone
unnoticed is that `release.yml:60` publishes images only on a `.0.0` tag, and the last one
was `v4.0.0`. `docs/DOCKER.md` has meanwhile been advertising `v4.1.0` images that this
workflow never built.

**Estimated timeline.** Immediately, on the `v5.0.0` tag. The GHCR publish would have
failed in public, after the tag was already pushed and therefore not cleanly retractable.

**Minimum fix.** Delete the NLTK steps from both Dockerfiles and rebuild to confirm.

**Full fix.** As above, plus a CI job that builds the image on pull requests touching a
Dockerfile, so image rot is caught without waiting for a major release.

**Note.** This also removes the only route by which `nltk` would have entered a shipped
artefact, which independently strengthens PF-4's reachability conclusion.

---

## Phase 4 — Test coverage gaps

| Path | Why critical | Test needed |
|---|---|---|
| `whisper_engine.py:249` guard → `sequence_alignment.py:173` reference choice | The highest-severity finding; nothing connects the two today | Unit: a token set where the longest variant is a repetition loop asserts a healthy variant becomes the reference |
| `diariser.py:343-344` stub path | The original silent-failure mode, still reachable | Unit: force `_load_pipeline` to return `None`; assert a non-`None` error and no `SPEAKER_00` output |
| `pipeline_runner.py:78-84` escape guard | Deletes files; guard is correct but untested | Unit: point a variant path outside `variants_dir`; assert it survives |
| `sequence_alignment.py:279` denominator | Silently depresses all confidence | Unit: four variants, one empty; assert unanimity among the other three reaches HIGH |
| `batch_runner.py:269-271` fail-open | Lock returns "proceed" on a malformed file | Unit: write a truncated lock file; assert the run refuses |

**Tests that assert too broadly** (would pass against broken logic):
`tests/test_sequence_alignment.py:36` and `:49` (gap presence, not position), `:135-160`
(length-only, three tests), `:99` (docstring promises a LOW assertion it never makes),
`:113` (vacuous: `0 <= c <= 1` holds by construction), `:315` (tolerates 20% mis-tiering);
`tests/test_alignment.py:86` (`count >= 1`), `:102` (`tier in ("HIGH","MEDIUM")` — passes
either way). `TestSequencePerformance:161` allows 30s for 500 words, ~100× too loose.

**Strongest existing tests**, worth preserving as the model:
`TestMultiAlignmentColumnIntegrity::test_every_token_is_preserved_in_order` (`:299`) —
per-variant round-trip equality, the best assertion in the suite; and
`TestDiarizeOutputCompatibility` (`test_diariser_preflight.py:204`), whose fixture
asserts the object genuinely lacks `itertracks`.

**Highest-value additions, ranked:** (1) degenerate-reference exclusion, (2) diarisation
stub refusal, (3) exact token-and-tier assertions replacing the length-only trio.

---

## Phase 5 — Dependency audit

- **Runtime: 16 direct dependencies, all `==` pinned** in both `requirements.txt` and
  `pyproject.toml:14-31`, duplicated by hand and guarded by
  `tests/check_dependency_drift.sh`. Clean.
- **No lockfile and no hashes.** `requirements.txt:6` references `pip-compile` but no
  `requirements.in` exists. The transitive graph — including `lightning` and `torch` —
  floats. `lightning` is unpinned anywhere, which matters because it is the subject of
  the accepted CVE.
- **Dev extras: 11 entries, 10 floating**, 1 ranged (`black>=26.3.1`). CI pins
  `ruff==0.4.7` for lint while `[dev]` floats — two ruff versions between local and CI.
- **Advisories:**
  - `lightning` PYSEC-2026-3624 / CVE-2026-58659 — documented and accepted. No upstream
    release since 2.6.5. Reachability claim (Chorus never calls
    `LightningModule.load_from_checkpoint`) not independently re-verified this review;
    flagged for the optional security pass.
  - `nltk` PYSEC-2026-3740 — **undocumented**. No upstream fix (3.10.3 is latest and is
    flagged). Enters only via `safety` and `torchmetrics` extras; absent from the shipped
    runtime.
- **Nothing abandoned:** no direct dependency lacks an upstream release in 18 months.
- **Nothing worth inlining:** no single-purpose micro-dependency present.

---

## Phase 6 — CI/CD gap analysis

| Question | Answer |
|---|---|
| Tests on every PR? | **Yes** — `ci.yml:104`, though gated behind `lint` ← `secret-scan`, so a lint failure means tests never run |
| Lint and format check? | **Yes** — black, ruff, isort at `ci.yml:58-65` |
| Secret scanning? | **Yes** — gitleaks, TruffleHog, detect-secrets baseline, plus a separate `security.yml` job |
| Actions pinned? | **No** — every third-party action floats on a mutable tag. Only TruffleHog (`@v3.97.0`) and `pre-commit/action` (`@v3.0.1`) carry full versions, and neither is SHA-pinned |
| Scheduled drift run? | **Yes** — three (ci Mon 08:00, security Mon 09:00, ollama Mon 09:00) |

Two further gaps: `security.yml:61` runs bandit with `|| true`, so it can never fail the
build; and the ad-hoc `pip install` at `ci.yml:101` / `release.yml:47` overrides the
pinned manifest.

The fix for action pinning is mechanical — replace each `uses: owner/repo@vN` with
`uses: owner/repo@<40-char-sha>  # vN`, resolvable via
`gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`. The fix for the install override
is to delete the line, since `pip install -e ".[dev]"` on the preceding line already
resolves the pinned set.

---

## Phase 7 — Action roadmap

### RD-1: Bar a degenerate transcript from anchoring the alignment

**Context:** `transcription_engine/whisper_engine.py:249` detects a Whisper repetition
loop and only writes a log warning. The transcript still votes, and because
`consensus_merger/sequence_alignment.py:173` selects the *longest* transcript as the
alignment reference, a repetition-looped transcript — which is artificially long — is the
most likely to be chosen as the backbone for every other variant.

**Success criteria:**
- `whisper_engine.transcribe()` records the repetition ratio and a boolean flag on the
  returned result dict, alongside the existing `variant` / `model` / `device` keys.
- The flag reaches `_build_multi_alignment` via an optional
  `degenerate_keys: set[str] | None = None` parameter threaded through
  `merge_transcripts_with_votes` → `align_transcripts` → `align_transcripts_sequence`.
- `_build_multi_alignment` excludes flagged keys when choosing `ref_key`, and falls back
  to current behaviour if every variant is flagged.
- A new test builds four token lists where the longest is a repetition loop and asserts a
  non-degenerate variant is chosen as reference. **The test must be demonstrated failing
  before the fix.**
- Full suite green; black, ruff, and isort clean.

**Files to change:** `transcription_engine/whisper_engine.py`,
`consensus_merger/merger.py`, `consensus_merger/alignment.py`,
`consensus_merger/sequence_alignment.py`, `tests/test_sequence_alignment.py`

**Estimated effort:** M

---

### RD-2: Stop diarisation degrading silently to a fake speaker

**Context:** `diarisation/diariser.py:304-307` logs a warning and returns `None` when the
pyannote pipeline fails to load. `diarise()` at `:343-344` then returns
`_stub_diarisation()` — one synthetic `SPEAKER_00` covering the whole file — leaving
`diarisation_error` as `None`, so the run reports success with a fabricated single-speaker
result.

**Success criteria:**
- A pipeline-load failure produces an error that reaches `diarisation_error` in
  `run_pipeline()`'s result dict; it is never `None` when the stub was used.
- The stub is reachable only under the explicit `--allow-diarisation-stub` opt-in.
- `TypeError` is added to `_try_load_pipeline`'s caught exception tuple so a future
  kwarg mismatch returns `(False, reason)` rather than raising through
  `ui/sidebar.py:504`, which has no try/except.
- A new test forces `_load_pipeline` to return `None` and asserts both a non-`None`
  `diarisation_error` and the absence of `SPEAKER_00` stub output. **Demonstrated failing
  before the fix.**

**Files to change:** `diarisation/diariser.py`, `pipeline_runner.py`,
`tests/test_diariser_preflight.py`

**Estimated effort:** M

---

### RD-3: Stop the test suite deleting live UI run state

**Context:** `tests/conftest.py:18-32` is an `autouse` fixture that unlinks the real
`ui/run_state.ACTIVE_RUN_FILE` (`outputs/active_run.json`) before and after every test in
the suite. Running `pytest` while a Streamlit run is in flight destroys that run's state —
and since #241 the pre-push hook runs the full suite, so any `git push` during a
multi-hour run triggers it.

**Success criteria:**
- The fixture redirects `ui.run_state.ACTIVE_RUN_FILE` to a per-test temporary path via
  `monkeypatch`; the real `outputs/active_run.json` is never read or unlinked.
- Existing `get_run_manager.clear()` isolation behaviour is preserved and the UI tests
  still pass.
- Verified by creating a sentinel `outputs/active_run.json` with known content, running
  the full suite, and confirming the file exists unchanged afterwards.

**Files to change:** `tests/conftest.py`

**Estimated effort:** S

---

### RD-4: Document the nltk advisory as an accepted, unreachable risk

**Context:** CI's `Python Dependency Audit` reports `nltk 3.10.3` /
`PYSEC-2026-3740` (a file-sandbox bypass in model-persistence APIs) on every run.
`SECURITY.md` documents only the `lightning` advisory, so this one has gone unrecorded —
the standing "the audit is expected to stay red" decision removed the signal.

**Success criteria:**
- `SECURITY.md` gains an `nltk` entry in the same format as the existing `lightning` one,
  stating: no upstream fix exists (3.10.3 is the latest release and is the flagged
  version); `nltk` is not in `requirements.txt`; it enters only through the `safety`
  dev/CI tool and `torchmetrics`' opt-in extras, so it is absent from the shipped runtime.
- A tracking issue is filed mirroring the structure of issue #219 and cross-linked from
  the new entry.

**Files to change:** `SECURITY.md`

**Estimated effort:** XS

---

### RD-5: Restore signal to the dependency audit

**Context:** The audit step fails on every run because of the accepted `lightning`
advisory, so a red result conveys nothing and a genuinely new advisory (see RD-4) sat
unnoticed in the output.

**Success criteria:**
- The audit step passes an explicit ignore list containing only the accepted advisory
  IDs, and fails the job on any advisory outside that list.
- Adding a fake ID to the list and removing a real one demonstrably flips the job result,
  proving the gate is live.
- The accepted advisories remain visible in the log rather than being suppressed.

**Files to change:** `.github/workflows/ci.yml`, `.github/workflows/security.yml`

**Estimated effort:** S

---

### RD-6: Make the batch lock atomic

**Context:** `batch_processor/batch_runner.py:880-883` calls `check_batch_lock()` and
then `acquire_batch_lock()` as two unsynchronised steps, and `acquire` unconditionally
overwrites any existing lock (`:286-296`). A malformed or empty lock file is treated as
"safe to proceed" (`:269-271`), so the failure mode is fail-open.

**Success criteria:**
- Acquisition is a single atomic operation: `os.open` with `O_CREAT|O_EXCL`, or
  `fcntl.flock` on the lock file.
- A malformed or empty lock file causes the run to refuse, not proceed.
- The lock payload records hostname and process start time so a recycled PID is not
  mistaken for the original owner.
- Tests cover: second acquisition refused while the first is held; refusal on a truncated
  lock file; successful acquisition when the recorded PID is genuinely dead.

**Files to change:** `batch_processor/batch_runner.py`, `tests/test_batch_runner.py`

**Estimated effort:** M

---

### RD-7: Give the diarisation pre-flight a real smoke test

**Context:** `check_diarisation_ready()` (`diarisation/diariser.py:183-204`) verifies
imports, token presence, and `Pipeline.from_pretrained` succeeding, but never runs
inference. It therefore could not have caught either the `DiarizeOutput` result-shape
failure or the torchcodec/ffmpeg decode failure — the two most recent production
incidents. The ad-hoc `hasattr` shim at `:359-362` recovers only one wrapper attribute
name.

**Success criteria:**
- The pre-flight runs the loaded pipeline over a short synthetic silent WAV generated at
  runtime, and parses the result through the same code path `diarise()` uses.
- A result-shape mismatch or an audio-decode failure is reported by the pre-flight, with
  a message naming which of the two occurred.
- The check completes in a few seconds and requires no fixture audio committed to the repo.
- A test injects a result object lacking both `itertracks` and `speaker_diarization` and
  asserts the pre-flight reports failure rather than raising.

**Files to change:** `diarisation/diariser.py`, `tests/test_diariser_preflight.py`

**Estimated effort:** M

---

### RD-8: Fix the confidence denominator

**Context:** `consensus_merger/sequence_alignment.py:279` computes
`n_transcripts = len(token_lists)` *before* empty transcripts are filtered out at `:282`.
A variant that produced no output therefore permanently depresses every column's
confidence: in a four-variant run with one empty variant, unanimity among the remaining
three scores 0.75, exactly on the HIGH boundary.

**Success criteria:**
- `n_transcripts` is computed after the empty-variant filter.
- A test with four variants, one empty, asserts that unanimous agreement among the other
  three is tiered HIGH. **Demonstrated failing before the fix.**
- Existing alignment tests still pass, confirming no tier changes for runs without empty
  variants.

**Files to change:** `consensus_merger/sequence_alignment.py`,
`tests/test_sequence_alignment.py`

**Estimated effort:** S

---

### RD-9: Replace assertions that cannot fail

**Context:** Several alignment tests would pass against broken logic: length-only
assertions (`tests/test_sequence_alignment.py:135-160`, three tests), a vacuous range
check (`tests/test_alignment.py:113`, where `0 <= c <= 1` holds by construction), an
either-way tier chain (`tests/test_alignment.py:102`), and a statistical gate tolerating
20% mis-tiering (`tests/test_sequence_alignment.py:315`). 430 tests passed while RC-10
and RC-11 were both live.

**Success criteria:**
- Each listed test asserts the exact expected token sequence and tier per position, not a
  length or a set membership.
- Each rewritten test is demonstrated to fail when a deliberate off-by-one is introduced
  into `_build_multi_alignment`, then to pass once reverted.
- `TestSequencePerformance:161`'s 30-second budget for 500 words is tightened to
  something that would actually catch a regression.

**Files to change:** `tests/test_sequence_alignment.py`, `tests/test_alignment.py`

**Estimated effort:** M

---

### RD-10: Pin CI actions and stop overriding pinned dependencies

**Context:** No third-party GitHub Action is SHA-pinned; all float on mutable tags
including `gitleaks/gitleaks-action@v2`, and this contradicts the repo's own
reference-pinning rule. Separately, `.github/workflows/ci.yml:101` and
`.github/workflows/release.yml:47` run
`pip install librosa soundfile scipy numpy openai-whisper` with no versions, overriding
every pin in `requirements.txt` — so the release's own test job validates versions other
than the ones shipped.

**Success criteria:**
- Every `uses:` referencing a third-party action names a 40-character commit SHA with the
  human-readable version in a trailing comment. Resolve with
  `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`.
- The bare `pip install` lines are removed; the preceding `pip install -e ".[dev]"`
  supplies the pinned set. CI still passes, proving nothing depended on the override.
- `devops-practices/check-clone-refs.sh` still passes.

**Files to change:** `.github/workflows/ci.yml`, `.github/workflows/security.yml`,
`.github/workflows/release.yml`

**Estimated effort:** M

---

### RD-11: Stop tests writing into the real outputs directory

**Context:** Eleven artefacts land in the real `outputs/consensus/` on every suite run,
because `export_srt`, `export_vtt`, and `merge_transcripts` default `output_dir` to the
global `CONSENSUS_DIR` and these call sites omit it. The same file gets it right
elsewhere — `tests/test_exporter.py:203`, `:214`, `:249` and others pass
`output_dir=tmp_path` — so this is unconverted, not undecided.

**Success criteria:**
- Every call at `tests/test_exporter.py:78`, `:82`, `:86`, `:101`, `:113`, `:117`, `:122`,
  `:140` and `tests/test_merger.py:59`, `:70`, `:80` passes `output_dir=tmp_path`.
- After a full suite run on a clean checkout, `outputs/consensus/` contains no
  `test_*` files.

**Files to change:** `tests/test_exporter.py`, `tests/test_merger.py`

**Estimated effort:** S

---

### RD-12: Tidy dead and duplicated code

**Context:** `build/lib/consensus_merger/` is a stale duplicate of the live package left
by an old build; `consensus_merger/sequence_alignment.py:53`'s `_score_pair` has no
callers.

**Success criteria:**
- `build/` is removed from the working tree and added to `.gitignore` if not already
  covered, and no import resolves to it.
- `_score_pair` is deleted and the suite still passes.

**Files to change:** `consensus_merger/sequence_alignment.py`, `.gitignore`

**Estimated effort:** XS

---

## Top three improvements, ranked

1. **RD-1 — bar a degenerate transcript from anchoring the alignment.** The only finding
   that can silently corrupt the primary output, and the guard that should prevent it
   already exists and is simply not wired to anything.
2. **RD-2 — stop diarisation degrading to a fake speaker.** Three production failures in
   three weeks came from this subsystem, and this path reproduces the original silent
   failure mode wherever the pre-flight is bypassed.
3. **RD-7 — give the pre-flight a real smoke test.** The structural answer to why there
   were three failures rather than one: a load check cannot catch a result-shape or
   decode fault, and adding one second of real inference closes the whole class.

---

## Release recommendation

**Cut the release. Tag `v5.0.0`, after RD-1, RD-2, RD-3, and RD-4.**

**On the version number.** `VERSION` reads `4.2.0` with no matching tag, and the last
release was `v4.1.1`. Because nothing was ever published as `4.2.0`, there is no
installable artefact whose history needs preserving, and no consumer holding a `4.2.0`
without the later fixes. The maintainer has designated this the final release, which is
sufficient grounds for a major bump independent of whether a breaking change exists.
`v5.0.0` it is; issue #242 closes as superseded.

**Two mechanical consequences of choosing a `.0.0` tag**, neither of which applies to a
`4.2.x`:
- `tests/version_consistency_test.sh:71-80` requires `docs/DOCKER.md` to name
  `ghcr.io/incendiary/chorus:v5.0.0` and the `-gpu` variant. It currently says `v4.1.0`,
  so this edit is mandatory or the release check fails.
- `.github/workflows/release.yml:60` gates Docker publish on `endsWith(ref, '.0.0')`, so
  tagging fires a GHCR CPU and GPU image build — the first since `v4.0.0`. Build the
  image locally before tagging so a failure surfaces early.

**What the release notes may honestly claim:** diarisation correctness and result-shape
compatibility; CLI and Web UI parity; batch reliability including the single-run lock and
persisted logs; error surfacing for previously silent diarisation failures; and the fixes
above.

**What they must not claim:** any accuracy improvement. Consensus still loses to
single-pass Whisper on noisy audio (0.1095 versus 0.1024 WER), the benchmark is known
unrepresentative (0.044 mean pairwise divergence versus 0.474 on real phone audio), and
no long-form ground truth exists. The defensible positioning remains **calibrated
uncertainty, not superior accuracy** — unchanged since 28 July, and not re-verified this
review.

---

## What I deliberately did not review, and why

- **The WER claim and the benchmark.** No new benchmark was run and no long-form ground
  truth was sourced. §2 of the brief is carried forward unverified, not re-confirmed.
- **`export_engine/exporter.py`'s size (RC-6).** Deliberately closed by the maintainer;
  re-raising it would waste a settled decision.
- **The `lightning` reachability claim.** `SECURITY.md` asserts Chorus never reaches
  `LightningModule.load_from_checkpoint`. I did not trace the pyannote dependency chain to
  confirm no indirect call exists. Flagged for the optional security pass rather than
  asserted either way.
- **Runtime behaviour of the pipeline.** No audio was processed during this review, so
  every finding labelled "code-read" is a reading of the executing path, not an observed
  failure. Given this project's history of defects that survive both the test suite and
  the benchmark, that is a real limitation of this review.
- **`reconstruction/`, `audio_processor/`, and `export_engine/` internals.** Read for the
  architecture map only; no adversarial pass. They have no production-incident history.
- **The Streamlit UI as a user.** The dashboard was not exercised in a browser.
