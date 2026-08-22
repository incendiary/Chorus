"""
diarisation/diariser.py — Speaker diarisation module.

Integrates pyannote.audio to identify and separate multiple speakers in an
audio file.  The diarisation result is fused with the Whisper segment-level
timestamps to produce a speaker-labelled transcript.

Architecture
────────────
  1. ``diarise(audio_path)``        — runs pyannote speaker diarisation and
                                      returns a list of ``SpeakerSegment`` objects.
  2. ``label_transcript(segments, whisper_result)``
                                    — aligns pyannote speaker turns with Whisper
                                      timed segments using midpoint overlap.
  3. ``render_diarised_md(labelled, stem)``
                                    — writes a speaker-labelled Markdown document
                                      to outputs/consensus/.

Graceful Degradation
────────────────────
pyannote.audio requires a Hugging Face access token to download the
``pyannote/speaker-diarization-3.1`` model weights.  If the token is absent
or the model cannot be loaded, the module falls back to a ``SPEAKER_00``-only
stub so the rest of the pipeline continues uninterrupted.

Set the token via the environment variable:
    HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import CONSENSUS_DIR

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Minimum speaker-turn duration to retain (seconds)
MIN_SEGMENT_DURATION: float = 0.5


def _get_hf_token() -> str | None:
    """Read HUGGINGFACE_TOKEN lazily, at pipeline-load time.

    Reading at call time (rather than caching a module-level constant at
    import) ensures the token is always seen, regardless of import order
    relative to ``.env`` being loaded (see ``config.py``).
    """
    return os.environ.get("HUGGINGFACE_TOKEN")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SpeakerSegment:
    """A single speaker turn with start/end timestamps."""

    speaker: str  # e.g. "SPEAKER_00", "SPEAKER_01"
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class LabelledSegment:
    """A Whisper segment annotated with a speaker label."""

    speaker: str
    start: float
    end: float
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Diarisation
# ─────────────────────────────────────────────────────────────────────────────


def _try_load_pipeline() -> tuple[object | None, str | None]:
    """
    Attempt to load the pyannote speaker-diarization pipeline.

    Returns ``(pipeline, None)`` on success or ``(None, reason)`` on failure,
    where *reason* is a specific, actionable message. This is the single
    place that attempts the real load, so both the silent-fallback path
    (``_load_pipeline``) and the pre-flight check (``check_diarisation_ready``)
    see exactly the same failure a real run would hit — not a lighter proxy
    that could pass while the real load still fails.
    """
    try:
        import torch
        from pyannote.audio import Pipeline  # type: ignore

        hf_token = _get_hf_token()
        if not hf_token:
            return None, (
                "HUGGINGFACE_TOKEN is not set. Create a read-only token at "
                "https://huggingface.co/settings/tokens/new and set it in .env."
            )

        logger.info("Loading pyannote speaker-diarization-3.1 pipeline…")
        # pyannote.audio 4.x renamed the credential argument from
        # ``use_auth_token`` to ``token``. Passing the old name raised
        # TypeError, and because diarisation degrades gracefully the failure
        # was invisible: every run reported success while silently producing
        # no speaker labels. Detect the accepted name rather than pinning to
        # one, so neither an upgrade nor a downgrade of pyannote breaks it.
        token_kwarg = (
            "token"
            if "token" in inspect.signature(Pipeline.from_pretrained).parameters
            else "use_auth_token"
        )
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **{token_kwarg: hf_token},
        )

        # Use GPU if available
        try:
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                logger.info("Diarisation pipeline running on CUDA.")
            else:
                logger.info("Diarisation pipeline running on CPU.")
        except (RuntimeError, OSError, ValueError):
            logger.debug("Could not move diarisation pipeline to CUDA.", exc_info=True)

        return pipeline, None

    except ImportError:
        return None, (
            "pyannote.audio is not installed. Install with: pip install pyannote.audio"
        )
    except (RuntimeError, OSError, ValueError) as exc:
        # This is the failure that went unnoticed on real casework audio.
        # Do not enumerate the pipeline's dependencies here by name — that
        # was tried twice and was wrong both times within the same evening:
        # a config.yaml fetched from the Hub named two internal models
        # (segmentation-3.0, wespeaker-voxceleb-resnet34-LM), but the
        # installed pyannote.audio's actual default __init__ signature
        # (pyannote/audio/pipelines/speaker_diarization.py) bundles
        # segmentation, embedding, and PLDA scoring into a third, different
        # checkpoint (pyannote/speaker-diarization-community-1) instead —
        # a library-version-dependent architecture change that a hardcoded
        # repo list cannot track. str(exc) already names whichever repo
        # actually failed, correctly, on every pyannote version, because it
        # comes from the real request that just happened — that is the only
        # part of this message that generalises.
        return None, (
            f"Failed to load the diarisation pipeline: {exc}\n\n"
            "If the message above names a gated repo (a huggingface.co URL "
            "with 'restricted' or '403'), visit that exact URL while signed "
            "in with the account that owns HUGGINGFACE_TOKEN and accept its "
            "terms, then retry. Which repos are involved depends on the "
            "installed pyannote.audio version, so there is no fixed list to "
            "check in advance — this message always reflects the actual "
            "failure just now, not a guess. If nothing above mentions "
            "gating, the failure is something else — a network or cache "
            "problem, most likely — and accepting licences will not fix it."
        )


def check_diarisation_ready() -> tuple[bool, str]:
    """
    Verify diarisation can actually run before a batch starts.

    Attempts the real pipeline load (not just a metadata/gating check), since
    that is the only way to be sure both models are accessible. Returns
    ``(True, "")`` when ready, or ``(False, reason)`` with the actionable
    message from ``_try_load_pipeline``.

    Callers should surface *reason* and require an explicit, informed choice
    before continuing in stub mode — the silent fallback previously produced
    a fully "successful" run that had quietly assigned an entire recording's
    audio to a single fake speaker for every file, with no error and no
    output-level marker that anything had degraded.
    """
    pipeline, reason = _try_load_pipeline()
    return pipeline is not None, reason or ""


def _load_pipeline():
    """
    Attempt to load the pyannote speaker-diarization pipeline.

    Returns the pipeline object or ``None`` if unavailable. Logs the same
    reason ``check_diarisation_ready`` would report, at WARNING level, so a
    caller that skips the pre-flight check still sees why it fell back.
    """
    pipeline, reason = _try_load_pipeline()
    if pipeline is None:
        logger.warning("%s Using stub diarisation.", reason)
    return pipeline


def _stub_diarisation(audio_path: Path) -> list[SpeakerSegment]:
    """
    Fallback stub that assigns the entire audio to a single speaker.

    Used when pyannote.audio is unavailable or unconfigured.
    """
    import soundfile as sf

    info = sf.info(str(audio_path))
    logger.info("Stub diarisation: assigning %.1f s to SPEAKER_00.", info.duration)
    return [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=info.duration)]


def diarise(audio_path: str | Path) -> list[SpeakerSegment]:
    """
    Run speaker diarisation on *audio_path*.

    Parameters
    ----------
    audio_path : str | Path
        Path to a WAV audio file (16 kHz mono recommended).

    Returns
    -------
    list[SpeakerSegment]
        Chronologically ordered list of speaker turns.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_pipeline()

    if pipeline is None:
        return _stub_diarisation(audio_path)

    logger.info("Running diarisation on: %s", audio_path.name)
    diarization = pipeline(str(audio_path))

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        if dur < MIN_SEGMENT_DURATION:
            continue
        segments.append(
            SpeakerSegment(
                speaker=speaker,
                start=round(turn.start, 3),
                end=round(turn.end, 3),
            )
        )

    logger.info(
        "Diarisation complete: %d segments, %d unique speakers.",
        len(segments),
        len({s.speaker for s in segments}),
    )
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Transcript labelling
# ─────────────────────────────────────────────────────────────────────────────


def label_transcript(
    speaker_segments: list[SpeakerSegment],
    whisper_result: dict[str, Any],
) -> list[LabelledSegment]:
    """
    Fuse pyannote speaker turns with Whisper timed segments.

    For each Whisper segment, the speaker whose turn contains the segment's
    midpoint is assigned.  If no speaker turn covers the midpoint, the
    nearest speaker turn by start time is used.

    Parameters
    ----------
    speaker_segments : list[SpeakerSegment]
        Output of ``diarise()``.
    whisper_result : dict
        Whisper transcription result dict containing ``"segments"`` list.

    Returns
    -------
    list[LabelledSegment]
        Whisper segments annotated with speaker labels.
    """
    labelled: list[LabelledSegment] = []
    whisper_segs = whisper_result.get("segments", [])

    for ws in whisper_segs:
        mid = (ws["start"] + ws["end"]) / 2.0
        label = "SPEAKER_00"  # default

        # Find the speaker turn that contains the midpoint
        for sp in speaker_segments:
            if sp.start <= mid <= sp.end:
                label = sp.speaker
                break
        else:
            # Nearest by start time
            if speaker_segments:
                nearest = min(speaker_segments, key=lambda s: abs(s.start - mid))
                label = nearest.speaker

        labelled.append(
            LabelledSegment(
                speaker=label,
                start=ws["start"],
                end=ws["end"],
                text=ws["text"].strip(),
            )
        )

    return labelled


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def render_diarised_md(
    labelled: list[LabelledSegment],
    stem: str,
    speaker_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Write a speaker-labelled Markdown transcript to CONSENSUS_DIR.

    Parameters
    ----------
    labelled : list[LabelledSegment]
        Output of ``label_transcript()``.
    stem : str
        Base filename stem.
    speaker_map : dict[str, str], optional
        Optional mapping of ``"SPEAKER_XX"`` → human-readable name
        (e.g., ``{"SPEAKER_00": "Interviewer", "SPEAKER_01": "Guest"}``).

    Returns
    -------
    Path
        Path to the written ``.md`` file.
    """
    speaker_map = speaker_map or {}
    lines = [
        "# Chorus — Speaker-Diarised Transcript",
        "",
        f"> **Source:** `{stem}`",
        "",
        "---",
        "",
    ]

    current_speaker = None
    for seg in labelled:
        display = speaker_map.get(seg.speaker, seg.speaker)
        ts_start = _format_timestamp(seg.start)
        ts_end = _format_timestamp(seg.end)

        if seg.speaker != current_speaker:
            lines.append(f"\n### 🎙️ {display}")
            current_speaker = seg.speaker

        lines.append(f"**[{ts_start} → {ts_end}]** {seg.text}")

    lines += [
        "",
        "---",
        "",
        "*Generated by Chorus Engine — Speaker Diarisation Module*",
        "",
    ]

    target_dir = output_dir or CONSENSUS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{stem}_diarised.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Diarised transcript written → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Speaker name persistence
# ─────────────────────────────────────────────────────────────────────────────


def _speaker_names_path(stem: str, output_dir: Path | None = None) -> Path:
    """
    Return the path to the speaker names sidecar JSON for *stem*.

    Parameters
    ----------
    stem : str
        Base filename stem.
    output_dir : Path | None
        Root directory for outputs. If None, uses the global CONSENSUS_DIR.

    Returns
    -------
    Path
        Path to the speaker names JSON file.
    """
    target_dir = output_dir or CONSENSUS_DIR
    return target_dir / f"{stem}_speakers.json"


def load_speaker_names(stem: str, output_dir: Path | None = None) -> dict[str, str]:
    """
    Load a previously saved speaker name mapping for *stem*.

    The mapping is stored as a JSON file alongside the consensus outputs.

    Parameters
    ----------
    stem : str
        Base filename stem.
    output_dir : Path | None
        Root directory for outputs. If None, uses the global CONSENSUS_DIR.

    Returns
    -------
    dict[str, str]
        Mapping of diarisation label (``"SPEAKER_00"``) → human-readable name.
        Returns an empty dict if no sidecar file exists or is unreadable.
    """
    path = _speaker_names_path(stem, output_dir=output_dir)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Speaker names file has unexpected format: %s", path)
            return {}
        # Ensure all values are strings
        return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read speaker names from %s: %s", path, exc)
        return {}


def save_speaker_names(
    stem: str, speaker_map: dict[str, str], output_dir: Path | None = None
) -> Path:
    """
    Save a speaker name mapping to the sidecar JSON file.

    Only entries where the user has provided a non-empty custom name are
    persisted. Entries mapping to the original label (e.g.,
    ``"SPEAKER_00" → "SPEAKER_00"``) are excluded to keep the file clean.

    Parameters
    ----------
    stem : str
        Base filename stem.
    speaker_map : dict[str, str]
        Mapping of diarisation label → human-readable name.
    output_dir : Path | None
        Root directory for outputs. If None, uses the global CONSENSUS_DIR.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    # Filter out identity mappings and empty names
    cleaned = {
        k: v.strip() for k, v in speaker_map.items() if v.strip() and v.strip() != k
    }

    path = _speaker_names_path(stem, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Speaker names saved → %s (%d entries)", path, len(cleaned))
    return path


def get_unique_speakers(labelled: list[LabelledSegment]) -> list[str]:
    """
    Extract the unique speaker labels from a labelled transcript, in order
    of first appearance.

    Parameters
    ----------
    labelled : list[LabelledSegment]
        Output of ``label_transcript()``.

    Returns
    -------
    list[str]
        Unique speaker labels in order of first appearance.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for seg in labelled:
        if seg.speaker not in seen:
            seen.add(seg.speaker)
            ordered.append(seg.speaker)
    return ordered
