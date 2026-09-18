"""tests/test_reconstruction_status.py — tests for reconstruction status reporting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from consensus_merger.alignment import WordVote
from reconstruction import reconstruct
from reconstruction.nlp import get_reconstruction_status as get_nlp_status


def _vote(word: str, tier: str = "HIGH", variants: list[str] | None = None) -> WordVote:
    return WordVote(
        word=word,
        count=1 if tier == "LOW" else 3,
        total=4,
        confidence=0.25 if tier == "LOW" else 0.75,
        tier=tier,
        variants=variants or [word],
    )


class TestNLPStatusReporting:
    """Test status reporting for spaCy-based reconstruction."""

    def test_nlp_status_unavailable_when_spacy_missing(self, monkeypatch):
        """get_nlp_status should report unavailable when spaCy is not installed."""
        monkeypatch.setattr(
            "reconstruction.nlp.probe_spacy_model",
            lambda: (False, "spaCy is not installed"),
        )

        status = get_nlp_status()
        assert status["status"] == "unavailable"
        assert "spaCy" in status["reason"]

    def test_nlp_status_degraded_when_sm_loaded(self, monkeypatch):
        """get_nlp_status should report degraded when sm model (no vectors) is used."""
        # Simulate the situation where en_core_web_md falls back to sm (no vectors)
        mock_nlp = MagicMock()
        # The sm model has no word vectors
        mock_nlp.vocab.vectors_length = 0

        # Patch both probe_spacy_model and _get_nlp to simulate fallback
        monkeypatch.setattr(
            "reconstruction.nlp.probe_spacy_model",
            lambda: (True, ""),  # Probe succeeds (md is available nominally)
        )
        monkeypatch.setattr(
            "reconstruction.nlp._get_nlp",
            lambda: mock_nlp,  # But _get_nlp returns sm model (no vectors)
        )

        status = get_nlp_status()
        assert status["status"] == "degraded"
        assert "word vectors" in status["reason"]

    def test_nlp_status_complete_when_md_model_available(self, monkeypatch):
        """get_nlp_status should report complete when en_core_web_md is available.

        ``_get_nlp`` is patched alongside the probe because otherwise this
        would perform a real spaCy load and assert "complete" only on a machine
        that happens to have ``en_core_web_md`` installed. The pre-push hook
        runs the suite under a different interpreter from the project venv,
        which is where that environment dependency surfaced.
        """
        model_with_vectors = MagicMock()
        model_with_vectors.vocab.vectors_length = 300

        monkeypatch.setattr(
            "reconstruction.nlp.probe_spacy_model",
            lambda: (True, ""),
        )
        monkeypatch.setattr(
            "reconstruction.nlp._get_nlp",
            lambda: model_with_vectors,
        )

        status = get_nlp_status()
        assert status["status"] == "complete"
        assert status["reason"] == ""

    def test_reconstruct_returns_votes_unchanged_when_backend_missing(self):
        """reconstruct should return votes unchanged when backend is missing."""
        votes = [
            _vote("the", tier="HIGH"),
            _vote("garbl", tier="LOW", variants=["garbl", "garble"]),
            _vote("world", tier="HIGH"),
        ]

        with patch(
            "reconstruction.nlp.probe_spacy_model",
            return_value=(False, "Not installed"),
        ):
            result = reconstruct(votes, strategy="nlp")

        assert result[1].word == "garbl"
        assert result[1].tier == "LOW"


class TestLLMStatusReporting:
    """Test status reporting for Ollama-based reconstruction."""

    def test_llm_status_unavailable_when_ollama_unreachable(self, monkeypatch):
        """Ollama status should report unavailable when unreachable."""
        # Patch inside the llm module where it's imported
        monkeypatch.setattr(
            "reconstruction.llm.probe_model",
            lambda model=None: (False, "Cannot reach Ollama at http://localhost:11434"),
        )

        from reconstruction.llm import get_reconstruction_status as get_llm_status

        status = get_llm_status()
        assert status["status"] == "unavailable"
        assert "Ollama" in status["reason"]

    def test_llm_status_unavailable_when_model_not_pulled(self, monkeypatch):
        """Ollama status should report unavailable when model is not pulled."""
        # Patch inside the llm module where it's imported
        monkeypatch.setattr(
            "reconstruction.llm.probe_model",
            lambda model=None: (False, "Model 'llama3.1' is not pulled"),
        )

        from reconstruction.llm import get_reconstruction_status as get_llm_status

        status = get_llm_status()
        assert status["status"] == "unavailable"
        assert "pulled" in status["reason"]

    def test_llm_status_complete_when_model_available(self, monkeypatch):
        """Ollama status should report complete when model is available."""
        # Patch inside the llm module where it's imported
        monkeypatch.setattr(
            "reconstruction.llm.probe_model",
            lambda model=None: (True, ""),
        )

        from reconstruction.llm import get_reconstruction_status as get_llm_status

        status = get_llm_status()
        assert status["status"] == "complete"
        assert status["reason"] == ""


class TestReconstructionStatusIsSurfaced:
    """Status has to reach the user, not just the result dict.

    A field that nothing reads is the same defect as no field at all: the
    pipeline already carried ``diarisation_error`` for a while that the Web UI
    silently dropped. These pin the consumers so the same gap cannot reopen.
    """

    def test_batch_report_flags_an_unavailable_backend(self):
        from batch_processor.batch_runner import _reconstruction_warning

        warning = _reconstruction_warning(
            {
                "reconstruction_status_nlp": {
                    "status": "unavailable",
                    "reason": "spaCy is not installed.",
                }
            }
        )

        assert warning == "NLP reconstruction unavailable"

    def test_batch_report_flags_a_degraded_backend(self):
        from batch_processor.batch_runner import _reconstruction_warning

        warning = _reconstruction_warning(
            {
                "reconstruction_status_nlp": {
                    "status": "degraded",
                    "reason": "no word vectors",
                }
            }
        )

        assert warning == "NLP reconstruction degraded"

    def test_batch_report_reports_both_backends_together(self):
        from batch_processor.batch_runner import _reconstruction_warning

        warning = _reconstruction_warning(
            {
                "reconstruction_status_nlp": {"status": "degraded", "reason": ""},
                "reconstruction_status_llm": {"status": "unavailable", "reason": ""},
            }
        )

        assert warning == (
            "NLP reconstruction degraded, LLM reconstruction unavailable"
        )

    def test_a_complete_backend_produces_no_warning(self):
        from batch_processor.batch_runner import _reconstruction_warning

        assert (
            _reconstruction_warning(
                {
                    "reconstruction_status_nlp": {"status": "complete", "reason": ""},
                    "reconstruction_status_llm": {"status": "complete", "reason": ""},
                }
            )
            is None
        )

    def test_a_run_without_reconstruction_produces_no_warning(self):
        from batch_processor.batch_runner import _reconstruction_warning

        assert _reconstruction_warning({"consensus_path": "x"}) is None
