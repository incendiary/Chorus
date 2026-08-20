"""Security regression tests for the Chorus deployment exposure policy."""

from __future__ import annotations

import pytest

from security.exposure import (
    current_exposure_state,
    exposure_warning,
    is_loopback_address,
    localhost_only_enabled,
)


@pytest.mark.parametrize("address", ["127.0.0.1", "127.7.8.9", "::1", "localhost"])
def test_loopback_addresses_are_local(address):
    assert is_loopback_address(address)


@pytest.mark.parametrize("address", [None, "", "0.0.0.0", "::", "192.0.2.20"])
def test_non_loopback_addresses_are_not_local(address):
    assert not is_loopback_address(address)


def test_localhost_only_is_the_default(monkeypatch):
    monkeypatch.delenv("CHORUS_LOCALHOST_ONLY", raising=False)
    assert localhost_only_enabled()


def test_network_exposure_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("CHORUS_LOCALHOST_ONLY", "false")
    state = current_exposure_state("0.0.0.0")

    assert not state.localhost_only
    assert not state.policy_mismatch
    assert "Network exposure is enabled" in (exposure_warning(state) or "")
    assert "upstream security concern" in (exposure_warning(state) or "")


def test_declared_local_mode_cannot_hide_non_loopback_bind(monkeypatch):
    monkeypatch.setenv("CHORUS_LOCALHOST_ONLY", "true")
    state = current_exposure_state("0.0.0.0")

    assert state.policy_mismatch
    assert "restore a loopback bind" in (exposure_warning(state) or "")


def test_local_bind_in_local_mode_has_no_warning(monkeypatch):
    monkeypatch.setenv("CHORUS_LOCALHOST_ONLY", "true")
    state = current_exposure_state("127.0.0.1")

    assert not state.policy_mismatch
    assert exposure_warning(state) is None


def test_ambiguous_mode_is_rejected():
    with pytest.raises(ValueError, match="must be true or false"):
        localhost_only_enabled("perhaps")
