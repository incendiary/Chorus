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

## Scope

This policy covers the Chorus Engine codebase in this repository — the audio
processing, transcription, consensus, export, and UI code, plus its CI/CD
configuration. It does not cover:

- Third-party dependencies (report those to the upstream project; `pip-audit` and
  Dependabot already monitor known CVEs in Chorus's pinned dependencies).
- The Ollama or Whisper model weights themselves.
- Issues that require local, unauthenticated access to a machine already running
  Chorus (Chorus is a local-first tool with no exposed network service by default).
