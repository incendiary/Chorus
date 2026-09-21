# Security Policy

## Supported Versions

Only the latest released version of Chorus is supported with security fixes. Older
versions are not patched — please upgrade to the latest release before reporting an
issue tied to an older version.

| Version | Supported |
|---------|-----------|
| Latest (see [ROADMAP.md](ROADMAP.md) / [releases](https://github.com/incendiary/Chorus/releases)) | ✅ |
| Older releases | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private vulnerability reporting instead:

1. Go to the [Security tab](https://github.com/incendiary/Chorus/security) of this
   repository.
2. Click **Report a vulnerability**.
3. Describe the issue, including steps to reproduce, affected version, and potential
   impact.

This creates a private advisory visible only to the maintainer and you, so the issue
can be discussed and fixed before any public disclosure.

If you cannot use GitHub's private reporting for any reason, open a regular issue with
**no exploit details** and request a private channel to continue the conversation.

## What to Expect

- Acknowledgement of your report as soon as reasonably possible.
- An assessment of severity and, if valid, a fix released as a patch version.
- Credit in the release notes, if you'd like it (let us know your preference when
  reporting).

## Known Accepted Risks

- **`lightning` — `CVE-2026-58659` / `PYSEC-2026-3624`** (arbitrary code execution
  via the `_instantiator` hyperparameter in `LightningModule.load_from_checkpoint`,
  bypassing `torch.load(weights_only=True)`). `lightning` is a transitive
  dependency pulled in by `pyannote-audio` for speaker diarisation. The fix
  merged upstream on 2026-07-14 but has not shipped in a `lightning` release. A
  newer release, `2.6.6` (2026-09-10), was checked directly against the fix commit
  and confirmed to diverge from it rather than include it — it was cut from an
  older maintenance branch point, not from `master` after the fix. Re-verify
  against the fix PR before assuming any future point release includes it, rather
  than trusting the version number alone. Chorus does not call `LightningModule.load_from_checkpoint`
  or load any user-supplied checkpoint; diarisation only loads pyannote's own
  pinned, first-party model weights from Hugging Face. `ci/security.yml`'s
  `pip-audit` step will keep flagging this until `lightning` cuts a release
  containing the fix — treat that failure as expected and check
  [Lightning-AI/pytorch-lightning#21832](https://github.com/Lightning-AI/pytorch-lightning/pull/21832)
  for release status before assuming a new dependency audit failure is this one.

- **`nltk` — `PYSEC-2026-3740`** (file sandbox bypass: several model-persistence
  APIs, including `TransitionParser.train`, `AveragedPerceptron.save`, and
  `PerceptronTagger.save_to_json`, use built-in `open()` on caller-controlled paths
  instead of the `pathsec`-aware helpers, so they read and write outside the
  configured allowed roots). **No upstream fix exists, and the advisory databases
  disagree about that.** OSV records `PYSEC-2026-3740` as "fixed in 3.10.3", while
  the copy `pip-audit` consults lists no patched version and keeps flagging
  `3.10.3`. The tie was broken by reading the release rather than trusting either:
  in `nltk` 3.10.3, `AveragedPerceptron.save` and `.load` still call the built-in
  `open()` directly, `nltk/pathsec.py` is not imported by that module, and the
  advisory's own remediation asks for exactly the opposite. OSV's fix version is
  therefore wrong and the advisory stands. Re-read the source before accepting any
  future claim that this is fixed, as issue
  [#219](https://github.com/incendiary/Chorus/issues/219) already warns for
  `lightning`: an inconsistent fix-version string is not evidence. Chorus
  does not depend on `nltk`. It is absent from `requirements.txt` and
  `pyproject.toml`'s runtime dependencies, and reaches an installed environment only
  through the `safety` development and CI tool, and through `torchmetrics`' opt-in
  `text`, `all`, and `dev` extras, none of which Chorus requests. No shipped Chorus
  code path imports `nltk` or calls the affected APIs, so the sandbox bypass is not
  reachable from the application. Tracked in
  [#244](https://github.com/incendiary/Chorus/issues/244) so that a future dependency
  audit failure naming `nltk` is recognised as this accepted risk rather than a new one.

- **`cuda-toolkit` — `CVE-2025-33228`** (command injection via unsanitised input).
  Reported by the `safety` step in `security.yml`, which scans the whole installed
  environment rather than the declared dependency set, so it appears only in CI.
  No fixed version is published. Chorus does not depend on `cuda-toolkit`: it is
  absent from `requirements.txt` and `pyproject.toml`, no Chorus code imports it or
  invokes any CUDA toolkit executable, and it reaches the CI environment only as a
  transitive of the CUDA-enabled PyTorch wheels. The vulnerable surface is the
  toolkit's own command-line entry points, which Chorus never calls.

`PYSEC-2026-3967` in `pytorch-lightning` was also found during this review. It is
distinct from the `lightning` RCE above and genuinely fixed upstream in `2.6.6`, so it
is **not** an accepted risk: `lightning` and `pytorch-lightning` are now pinned to
`2.6.6` explicitly. Both had resolved transitively through `pyannote-audio` with no
version constraint, so the installed version depended on when the environment happened
to be built. Note that they are two separate distributions and `lightning` requires
`pytorch-lightning` without a constraint, so pinning only one leaves the advisory live.

### Why the dependency audit is expected to fail

Two separate jobs scan dependencies, and they disagree by design:

- `ci.yml`'s `pip-audit` step reads `requirements.txt`, which lists only Chorus's
  16 direct pins. It reports `nltk`.
- `security.yml`'s `safety` step scans the entire installed environment, so it also
  reports transitive packages Chorus never declares, such as `cuda-toolkit`.

A permanently red job carries no signal, which is precisely how the `nltk` advisory
went unrecorded for weeks and how `CVE-2025-33228` was found only by reading a log
line by line. Every advisory currently reported is listed above; anything that is
not listed here is new and should be investigated rather than assumed accepted.

## Scope

This policy covers the Chorus Engine codebase in this repository — the audio
processing, transcription, consensus, export, and UI code, plus its CI/CD
configuration. It does not cover:

- Third-party dependencies (report those to the upstream project; `pip-audit` and
  Dependabot already monitor known CVEs in Chorus's pinned dependencies).
- The Ollama or Whisper model weights themselves.
- Issues that require local, unauthenticated access to a machine already running
  Chorus (Chorus is a local-first tool with no exposed network service by default).
