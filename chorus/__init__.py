"""
chorus — the stable, public API for the Chorus Engine.

This package is a thin façade over the internal modules.  The names re-exported
here are the supported entry points that the 4.x release line commits to keeping
stable.  Anything reached through a deeper path (for example
``consensus_merger.merger`` or ``export_engine.exporter``) is internal and may
move between minor releases without notice.

Typical usage::

    from chorus import run_pipeline

    results = run_pipeline(audio_path="meeting.wav", language="en")

Supported entry points:

  - :func:`run_pipeline`              — process a single audio file end to end.
  - :func:`run_batch`                 — process several files or directories.
  - :func:`merge_transcripts_with_votes` — run the consensus merge directly.
  - :func:`export_all`                — export a consensus document to PDF, DOCX,
                                        SRT, and VTT.
  - :func:`export_transcript_bundle`  — write a structured JSON transcript bundle.
"""

from __future__ import annotations

from batch_processor.batch_runner import run_batch
from consensus_merger.merger import merge_transcripts_with_votes
from export_engine.exporter import export_all, export_transcript_bundle
from pipeline_runner import run_pipeline

__all__ = [
    "run_pipeline",
    "run_batch",
    "merge_transcripts_with_votes",
    "export_all",
    "export_transcript_bundle",
]
