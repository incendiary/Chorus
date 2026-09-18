# v5.0.0 Final Release — Triage of Open Items

**Status:** for review. No code has been changed on the strength of this document.
**Date:** 18 September 2026

## Why this exists

Every previous triage in this project assumed a next release. Items were sorted into
"blocks the tag" and "post-release cleanup", and the roadmap still carries headings that
say *Planned — post-v5.0.0*.

v5.0.0 is the final release. There is no post-release, so "deferred" does not mean "later",
it means **shipped permanently, never fixed**. That changes the ordering substantially:

- A defect that produces **silently wrong output** is much worse, because the person
  holding it has no upgrade path.
- **Maintainability** work is much less valuable, because there is no future change for it
  to protect.
- **Documentation** is much more valuable, because it becomes the permanent record of what
  this software does and does not do.

The 29 open roadmap entries below are re-triaged on that basis. Two pairs are duplicates
(the `survey-ollama-env.sh` shell-test item appears twice, and generic "test pollution"
duplicates RD-11), so there are 27 distinct items, plus 5 newly found and not yet tracked.

## Decision key

| Decision | Meaning |
|---|---|
| **FIX** | Ships wrong or loses data otherwise. Must land before the tag. |
| **DOCUMENT** | Will not be fixed. The permanent record must state it plainly so a user is not surprised. |
| **ACCEPT** | Genuinely fine to ship as-is. Recorded as a known limitation, no further work. |

---

## Tier 1 — Silently wrong output or silent data loss

These produce a plausible-looking result that is incorrect or incomplete, with no signal
to the user. On evidentiary material that is the worst failure mode available, and it is
the exact shape of every defect that has already cost this project real time.

| Item | Decision | Why it cannot ship |
|---|---|---|
| **RD-8** — confidence denominator computed before empty variants are filtered | **FIX** | The tiers are the product's entire defensible claim. One empty variant permanently depresses every confidence and caps unanimity at exactly 0.75, the HIGH boundary. Roughly a one-line change. |
| **RD-21** — spaCy tokens indexed against vote positions | **FIX** | Contractions and punctuation shift spaCy's tokenisation, so part-of-speech is read off the wrong token. Silently wrong reconstruction, not a crash. |
| **RD-19** — reconstruction no-ops when spaCy or Ollama is absent | **FIX** | `--nlp` or `--llm` reports nothing wrong while doing nothing at all. Indistinguishable from "ran and found nothing to fix". |
| **RD-20** — `en_core_web_sm` fallback has no word vectors | **FIX** | Semantic scoring silently returns 0.0 for every candidate, degrading to part-of-speech plus Levenshtein with no warning. Same file as RD-19. |
| **RD-18** — a failed export still reports success | **FIX** | A WeasyPrint crash or full disk shows in the batch report as `OK` with the format simply missing. Verified not to have fired on the September casework run, so latent, not active. |

## Tier 2 — Robustness and release integrity

Not wrong answers, but failure modes a user will hit with no recourse, plus the integrity
of the one build that produces the permanent artefact.

| Item | Decision | Why |
|---|---|---|
| **RD-6** — batch lock check-then-act, fail-open on a malformed file | **FIX** | This class already cost a 10.5-hour deadlock once. Unattended overnight batches are the primary use. |
| **RD-23** — `audio_processor` unguarded numerical edges | **FIX** | Empty-array reductions, division by sample rate, zero frame length. Guards turn an obscure `ValueError` into a clear message. |
| **RD-22** — Web UI validates no input | **FIX** | A zero-byte or renamed file fails as a generic per-file error. `CLAUDE.md`'s own testing standard requires this. |
| **release.yml installs unpinned packages** (part of RD-10) | **FIX** | `release.yml:47` runs a bare `pip install librosa soundfile scipy numpy openai-whisper`, so the release's own test job validates versions other than those shipped. For a final release, the build that makes the artefact must test the artefact. |
| **Pin the actions used by `release.yml`** (part of RD-10) | **FIX** | That workflow holds `contents: write` and `packages: write` and every third-party action floats on a mutable tag. Narrow scope: pin the release workflow only. |
| **RD-7** — diarisation pre-flight never runs inference | **FIX (judgement call)** | The structural cause of three separate failures. Mitigated by pinned dependencies, so a user installing the pinned set is less exposed. Worth it if scope allows; the honest alternative is DOCUMENT. |
| **RD-17** — `Dockerfile.gpu` discards the torch it installs | **FIX** | The image is permanent. It wastes roughly 2.5 GB per build, and ships CUDA 13.0 on a CUDA 12.1 base, making the file header and the documented driver requirement wrong. |

## Tier 3 — The permanent record

For a final release the documentation is the deliverable, not a support artefact.

| Item | Decision | Why |
|---|---|---|
| **`SECURITY.md` nltk entry is inaccurate** *(new)* | **FIX** | My own wording says "no patched version published". OSV's record contradicts it by listing a 3.10.3 fix. I verified from the 3.10.3 source that `AveragedPerceptron.save`/`load` still use raw `open()`, so OSV is wrong and the entry is right by accident. It must cite the source-level check, not an absence of data. |
| **`PYSEC-2026-3967` in pytorch-lightning** *(new)* | **FIX** | A genuinely different advisory from the tracked RCE, and it *is* fixed in 2.6.6. `lightning` is not pinned anywhere, so the fix is not reachable without pinning it. |
| **ROADMAP headings promise work that will never happen** *(new)* | **FIX** | Sections titled *Planned — post-v5.0.0 cleanup* and *Planned — deferred* are false under a final release. They must read as accepted limitations. |
| **Comprehensive CLI flag reference** (roadmap item 5) | **FIX** | README documents roughly 6 of 22 flags. This is the user's only guide and there will be no later edition. |
| **Four CLI settings with no Web UI equivalent** (roadmap item 4) | **DOCUMENT** | Building four new UI controls immediately before a final tag adds more risk than it removes. Document the gap and the `.env` route instead. |
| **`docs/DOCKER.md` GPU guidance** *(new, with RD-17)* | **FIX** | States a CUDA 12.1 / driver expectation that the shipped image contradicts. |
| **Unexplained process death, 2026-08-23/24** (roadmap item 6) | **DOCUMENT** | Undiagnosed, with no evidence to act on. Record it as a known unexplained event rather than implying it is pending investigation. |

## Tier 4 — Accepted, shipped as-is

Recorded so the permanent record is honest, but no work.

| Item | Why accepted |
|---|---|
| **RD-9** — assertions that cannot fail | Tests protect future change. There is none. The suite's weakness is worth recording, not rewriting. |
| **RD-10** (remainder) — SHA-pin all CI actions | Only the release workflow's integrity matters now; the rest of CI has little future to protect. |
| **RD-14** — build Docker images in CI on Dockerfile changes | Guards future pull requests. There will be none. |
| **RD-24** — UI results shared across threads without a lock | Real but low-probability, and the worst outcome is a stale read in a single-user local tool. |
| **RD-25** — interrupted runs leave spool files | Disk hygiene on the user's own machine. |
| **RD-27** — `run_one_file` is dead code kept alive by tests | Its risk is drift; nothing will drift after the final tag. |
| **RD-12** — `build/lib/` duplicate and `_score_pair` | Verified: `build/` is gitignored with zero tracked files, so it never ships. Local tidiness only. |
| **RC-6** — split `export_engine/exporter.py` | Explicitly closed by the maintainer. Not reopened. |
| **Shell tests for `survey-ollama-env.sh`** (items 1 and 8, duplicated) | Test coverage for a helper script, with no future changes to protect. |
| **`large` vs `large+medium` comparison** (item 7) | Curiosity-driven, explicitly optional. |
| **Paired boolean flags mislabel provenance** (item 3) | Borderline. The printed *value* is correct and only the source label is wrong. Promote to FIX if provenance accuracy matters for casework records. |
| **Local venv drift** *(new)* | The development environment carries 107 advisories across 11 packages, almost all dev tooling absent from `requirements.txt`, plus stale `chorus-engine 4.1.1` metadata. Local only, never shipped. |

## Borderline, needs your call

- **RD-11 / test pollution** — eleven test artefacts land in the real `outputs/consensus/`,
  which is where casework output lives. Cheap to fix (pass `output_dir=tmp_path` at eleven
  call sites). Filed under Tier 4 by the "maintainability does not matter" rule, but the
  directory it pollutes argues for fixing it.
- **RD-26 / benchmark dirties the tree** — trivial to fix and trivial to accept.
- **RD-7** — listed as FIX above, but it is the single largest item in Tier 2 and the one
  most reasonably dropped if scope needs cutting.

## Honest effort estimate

Tier 1 is five changes, mostly small, two of them in the same file. Tier 2 is seven,
including the two largest pieces of work in the list (RD-6 and RD-7). Tier 3 is largely
writing, with the CLI reference being the substantial piece.

Realistically this is **12 to 16 pull requests**, not the two or three implied by the
earlier "blockers versus deferred" framing. That is the honest cost of the final-release
view, and it is materially more than was previously represented.

## What shipping the Tier 4 list actually means

A user of v5.0.0, permanently:

- gets a test suite with assertions that would pass against broken alignment logic
- may accumulate spool directories after interrupted Web UI runs
- runs a Web UI whose run registry has an unsynchronised cross-thread read
- has no automated coverage for the environment survey script

None of those produce a wrong transcript. That is the line this triage draws.
