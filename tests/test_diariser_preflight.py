"""
tests/test_diariser_preflight.py — Tests for the diarisation pre-flight check.

Covers the bug this check exists to catch: a real casework batch run where
speaker-diarization-3.1's internal dependency on the separately-gated
pyannote/segmentation-3.0 repo failed to load. The failure was swallowed by
a broad except clause and silently downgraded to a single-speaker stub, so
every file in the batch "succeeded" while quietly assigning the entire
recording to one fake speaker. check_diarisation_ready() exists so that
failure is caught before any file is processed, not discovered afterwards
by reading the output.
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


def test_gated_repo_failure_names_all_three_required_repos(monkeypatch):
    """Reproduces the exact failure mode seen on real audio: the top-level
    pipeline's licence was accepted, but its internal dependency on
    segmentation-3.0 was not, so pyannote raised on that second repo.

    The pipeline's own config.yaml declares three dependencies (itself,
    segmentation-3.0, wespeaker-voxceleb-resnet34-LM); the message must name
    all three, not just the one that happened to fail on real audio — a
    different pyannote release could gate the embedding model instead."""
    monkeypatch.setattr("diarisation.diariser._get_hf_token", lambda: "hf_faketoken")

    class _FakePipelineCls:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError(
                "Cannot access gated repo for url "
                "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/"
                "pytorch_model.bin."
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
    assert "segmentation-3.0" in reason
    assert "speaker-diarization-3.1" in reason
    assert "wespeaker-voxceleb-resnet34-LM" in reason


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
