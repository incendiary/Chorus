"""tests/test_config_models.py — tests for consensus model configuration parsing."""

from __future__ import annotations

import importlib

import config as config_module


def _reload_config(
    monkeypatch,
    *,
    whisper_model: str | None = None,
    consensus_models: str | None = None,
):
    """Reload config after applying environment overrides."""
    if whisper_model is None:
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
    else:
        monkeypatch.setenv("WHISPER_MODEL", whisper_model)

    if consensus_models is None:
        monkeypatch.delenv("CONSENSUS_MODELS", raising=False)
    else:
        monkeypatch.setenv("CONSENSUS_MODELS", consensus_models)

    return importlib.reload(config_module)


def test_consensus_models_defaults_to_whisper_model(monkeypatch):
    cfg = _reload_config(monkeypatch, whisper_model="base", consensus_models=None)
    assert cfg.CONSENSUS_MODELS == ("base",)
    assert cfg.CONSENSUS_MODEL_LABELS == {"base": "Whisper base"}


def test_consensus_models_parses_deduplicated_order(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        whisper_model="base",
        consensus_models=" base, small,base, MEDIUM ",
    )
    assert cfg.CONSENSUS_MODELS == ("base", "small", "medium")
    assert cfg.CONSENSUS_MODEL_LABELS == {
        "base": "Whisper base",
        "small": "Whisper small",
        "medium": "Whisper medium",
    }


def test_consensus_models_empty_value_falls_back_to_default(monkeypatch):
    cfg = _reload_config(monkeypatch, whisper_model="small", consensus_models=" , , ")
    assert cfg.CONSENSUS_MODELS == ("small",)
    assert cfg.CONSENSUS_MODEL_LABELS == {"small": "Whisper small"}


class TestDotenvLoader:
    """config._load_dotenv populates os.environ from a .env file."""

    def test_loads_values_and_strips_quotes(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            'HUGGINGFACE_TOKEN="hf_abc123"\n' "# a comment\n" "\n" "PLAIN=value\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        monkeypatch.delenv("PLAIN", raising=False)

        config_module._load_dotenv(env)

        assert config_module.os.environ["HUGGINGFACE_TOKEN"] == "hf_abc123"
        assert config_module.os.environ["PLAIN"] == "value"

    def test_real_environment_takes_precedence(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("HUGGINGFACE_TOKEN=from_dotenv\n", encoding="utf-8")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "from_real_env")

        config_module._load_dotenv(env)

        assert config_module.os.environ["HUGGINGFACE_TOKEN"] == "from_real_env"

    def test_missing_file_is_a_noop(self, tmp_path):
        config_module._load_dotenv(tmp_path / "does_not_exist.env")  # no raise
