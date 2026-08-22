"""
tests/test_diariser_preflight.py — Tests for the diarisation pre-flight check.

Covers the bug this check exists to catch: a real casework batch run where
speaker-diarization-3.1's internal dependency on a separately-gated model
failed to load. The failure was swallowed by a broad except clause and
silently downgraded to a single-speaker stub, so every file in the batch
"succeeded" while quietly assigning the entire recording to one fake
speaker. check_diarisation_ready() exists so that failure is caught before
any file is processed, not discovered afterwards by reading the output.

Which repo is actually involved is deliberately not hardcoded anywhere in
the check or these tests: a config.yaml fetched from the Hub named one set
of internal dependencies, but the installed pyannote.audio's actual default
pipeline bundled them into an entirely different, third checkpoint instead
— a library-version-dependent detail no fixed list can track. The tests
below use a repo name that has never appeared anywhere in this codebase to
prove the message genuinely relays whatever failed, rather than one that
happens to match a name the implementation was written to expect.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from diarisation.diariser import _load_pipeline, check_diarisation_ready


def test_ready_when_pipeline_loads(monkeypatch):
    fake_pipeline = MagicMock()
    monkeypatch.setattr(
        "diarisation.diariser._try_load_pipeline",
        lambda: (fake_pipeline, None),
    )
    ok, reason = check_diarisation_ready()
    assert ok is True
    assert reason == ""


def test_not_ready_reports_the_real_failure_reason(monkeypatch):
    monkeypatch.setattr(
        "diarisation.diariser._try_load_pipeline",
        lambda: (None, "Cannot access gated repo for url .../segmentation-3.0"),
    )
    ok, reason = check_diarisation_ready()
    assert ok is False
    assert "segmentation-3.0" in reason


def test_not_ready_when_token_missing(monkeypatch):
    monkeypatch.setattr("diarisation.diariser._get_hf_token", lambda: None)
    ok, reason = check_diarisation_ready()
    assert ok is False
    assert "HUGGINGFACE_TOKEN" in reason


def test_gated_repo_failure_names_whatever_actually_failed(monkeypatch):
    """The message must relay the real failing repo verbatim rather than
    assume a fixed dependency list. Uses a repo name that has never
    appeared anywhere in this codebase or its history — proof that nothing
    here special-cases known repo names, since the actual dependency set
    has already changed once between when this check was written and when
    it was first run against the real, installed pyannote.audio."""
    monkeypatch.setattr("diarisation.diariser._get_hf_token", lambda: "hf_faketoken")

    class _FakePipelineCls:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError(
                "Cannot access gated repo for url "
                "https://huggingface.co/some-org/never-seen-before-repo/resolve/"
                "main/weights.bin."
            )

    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "pyannote.audio": MagicMock(Pipeline=_FakePipelineCls),
        },
    ):
        ok, reason = check_diarisation_ready()

    assert ok is False
    assert "never-seen-before-repo" in reason
    assert "segmentation-3.0" not in reason
    assert "wespeaker" not in reason
    assert "speaker-diarization-community-1" not in reason


def test_non_gating_failure_does_not_falsely_blame_licences(monkeypatch):
    """A network or cache failure has nothing to do with gated-model
    licences. The message must not tell the user to go accept licence
    terms as if that were the fix — the check must not overfit to the one
    failure mode it happened to be discovered from."""
    monkeypatch.setattr("diarisation.diariser._get_hf_token", lambda: "hf_faketoken")

    class _FakePipelineCls:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("Connection timed out while fetching model files.")

    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "pyannote.audio": MagicMock(Pipeline=_FakePipelineCls),
        },
    ):
        ok, reason = check_diarisation_ready()

    assert ok is False
    assert "Connection timed out" in reason
    assert "accepting licences will not fix it" in reason


def test_load_pipeline_still_returns_none_on_failure_for_existing_callers(monkeypatch):
    """_load_pipeline's simple pipeline-or-None contract must not change —
    diarise() and existing tests depend on it."""
    monkeypatch.setattr(
        "diarisation.diariser._try_load_pipeline",
        lambda: (None, "some reason"),
    )
    assert _load_pipeline() is None
