"""Ollama client helpers for low-confidence token reconstruction."""

from __future__ import annotations

import json
import logging
from urllib import error, request

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


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
