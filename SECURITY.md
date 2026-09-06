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
  merged upstream on 2026-07-14 but has not shipped in a `lightning` release —
  the latest available version (`2.6.5`) predates the fix and is the version
  named as vulnerable. Chorus does not call `LightningModule.load_from_checkpoint`
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
  configured allowed roots). No upstream fix exists: `3.10.3` is the latest release
  and is the version the advisory names, with no patched version published. Chorus
  does not depend on `nltk`. It is absent from `requirements.txt` and
  `pyproject.toml`'s runtime dependencies, and reaches an installed environment only
  through the `safety` development and CI tool, and through `torchmetrics`' opt-in
  `text`, `all`, and `dev` extras, none of which Chorus requests. No shipped Chorus
  code path imports `nltk` or calls the affected APIs, so the sandbox bypass is not
  reachable from the application. Tracked in
  [#244](https://github.com/incendiary/Chorus/issues/244) so that a future dependency
  audit failure naming `nltk` is recognised as this accepted risk rather than a new one.

## Scope

This policy covers the Chorus Engine codebase in this repository — the audio
processing, transcription, consensus, export, and UI code, plus its CI/CD
configuration. It does not cover:

- Third-party dependencies (report those to the upstream project; `pip-audit` and
  Dependabot already monitor known CVEs in Chorus's pinned dependencies).
- The Ollama or Whisper model weights themselves.
- Issues that require local, unauthenticated access to a machine already running
  Chorus (Chorus is a local-first tool with no exposed network service by default).
