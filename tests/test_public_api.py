"""
tests/test_public_api.py — Contract tests for the stable ``chorus`` public API.

These tests assert that every name advertised in ``chorus.__all__`` is importable
and callable.  They guard the 4.x public surface against accidental removal or
breakage.
"""

from __future__ import annotations

import importlib

import chorus


def test_all_names_are_exported():
    """Every name in ``__all__`` is a genuine attribute of the package."""
    for name in chorus.__all__:
        assert hasattr(chorus, name), f"chorus.{name} is advertised but missing"


def test_all_exported_names_are_callable():
    """Every public entry point is callable."""
    for name in chorus.__all__:
        obj = getattr(chorus, name)
        assert callable(obj), f"chorus.{name} is not callable"


def test_top_level_imports():
    """The documented top-level import style works."""
    module = importlib.import_module("chorus")
    from chorus import run_batch, run_pipeline

    assert run_pipeline is module.run_pipeline
    assert run_batch is module.run_batch
