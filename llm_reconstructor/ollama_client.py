"""Ollama client helpers for low-confidence token reconstruction."""

from __future__ import annotations

import json
import logging
from urllib import error, request

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def probe_model() -> tuple[bool, str]:
    """Check whether the configured Ollama model is available.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the model is available, or
        ``(False, reason)`` with a human-readable reason string.
    """
    tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    req = request.Request(url=tags_url, method="GET")
    try:
        with request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except TimeoutError:
        return False, f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS:.0f} s"
    except error.URLError as exc:
        return False, f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {exc.reason}"
    except error.HTTPError as exc:
        return False, f"Ollama returned HTTP {exc.code}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False, "Ollama /api/tags response was not valid JSON"

    pulled = [m.get("name", "") for m in data.get("models", [])]
    # Match on exact name or name-without-tag (e.g. "llama3.1:8b" matches "llama3.1")
    base = OLLAMA_MODEL.split(":")[0]
    if any(OLLAMA_MODEL == p or base == p.split(":")[0] for p in pulled):
        return True, ""

    if pulled:
        return False, (
            f"Model '{OLLAMA_MODEL}' is not pulled. "
            f"Available: {', '.join(pulled)}. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )
    return False, (
        f"No models are pulled in Ollama. Run: ollama pull {OLLAMA_MODEL}"
    )


def suggest_token(*, context: str, candidates: list[str]) -> str | None:
    """Ask Ollama to pick the most plausible token from candidates."""
    if not candidates:
        return None

    prompt = (
        "Choose exactly one token from the candidate list that best fits the context. "
        "Respond with only the token and nothing else.\n\n"
        f"Context: {context}\n"
        f"Candidates: {', '.join(candidates)}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except (TimeoutError, error.URLError, error.HTTPError) as exc:
        logger.warning("Ollama request failed: %s", exc)
        return None

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Ollama response was not valid JSON")
        return None

    raw = str(parsed.get("response", "")).strip().lower()
    if not raw:
        return None

    # Keep only first token to prevent multi-word outputs.
    return raw.split()[0]
