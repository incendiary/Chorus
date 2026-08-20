"""Deployment exposure policy and user-facing security guidance."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass

LOCALHOST_MODE_ENV = "CHORUS_LOCALHOST_ONLY"
UPSTREAM_ISSUE_ENV = "CHORUS_UPSTREAM_SECURITY_ISSUE_URL"
DEFAULT_ISSUE_URL = "https://github.com/incendiary/Chorus/issues/219"


def _parse_bool(value: str | None, *, default: bool) -> bool:
    """Parse common environment-style booleans, rejecting ambiguous values."""
    if value is None or not value.strip():
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{LOCALHOST_MODE_ENV} must be true or false, not {value!r}.")


def localhost_only_enabled(value: str | None = None) -> bool:
    """Return the configured exposure policy; localhost-only is the default."""
    raw_value = os.environ.get(LOCALHOST_MODE_ENV) if value is None else value
    return _parse_bool(raw_value, default=True)


def is_loopback_address(address: str | None) -> bool:
    """Return whether a bind address names only the local machine."""
    if not address:
        return False
    candidate = address.strip().lower().strip("[]")
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ExposureState:
    """Effective security posture presented by the UI."""

    localhost_only: bool
    bind_address: str
    policy_mismatch: bool
    issue_url: str


def current_exposure_state(bind_address: str | None) -> ExposureState:
    """Classify the effective bind against the declared Chorus policy."""
    address = (bind_address or "").strip()
    localhost_only = localhost_only_enabled()
    issue_url = os.environ.get(UPSTREAM_ISSUE_ENV, DEFAULT_ISSUE_URL).strip()
    return ExposureState(
        localhost_only=localhost_only,
        bind_address=address,
        policy_mismatch=localhost_only and not is_loopback_address(address),
        issue_url=issue_url,
    )


def exposure_warning(state: ExposureState) -> str | None:
    """Return the persistent warning required for an unsafe exposure state."""
    if state.policy_mismatch:
        return (
            "Localhost-only mode is enabled, but Streamlit is bound to "
            f"`{state.bind_address or 'an unspecified address'}`. Stop Chorus and "
            "restore a loopback bind before processing sensitive audio."
        )
    if state.localhost_only:
        return None
    return (
        "Network exposure is enabled. Chorus has not been hardened as an "
        "internet-facing or multi-user service. Review authentication, TLS, "
        "upload isolation, and upstream component security before exposing it. "
        "Speaker diarisation currently depends on a Lightning checkpoint loader "
        f"with an open upstream security concern. Tracking: {state.issue_url}"
    )
