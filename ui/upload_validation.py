"""ui/upload_validation.py — Reject unusable uploads before they are spooled."""

from __future__ import annotations

from pathlib import Path

from config import SUPPORTED_AUDIO_EXTENSIONS


def _upload_size(uf: object) -> int:
    """Return the upload's size in bytes without consuming it.

    Streamlit's ``UploadedFile`` exposes ``size`` directly. Falling back to
    ``read()`` would leave the stream at EOF, and ``spool_upload`` reads the
    same object immediately afterwards, so every validated upload would be
    written to disk as zero bytes. Where only ``read()`` is available, the
    position is restored.
    """
    size = getattr(uf, "size", None)
    if isinstance(size, int):
        return size

    data = uf.read()
    seek = getattr(uf, "seek", None)
    if callable(seek):
        seek(0)
    return len(data)


def validate_upload(uf: object) -> str | None:
    """Check an uploaded file is usable, returning None when it is.

    Returns a message naming the problem otherwise. This reports rather than
    raises: one unusable file must not discard the rest of the batch.
    """
    file_name = str(getattr(uf, "name", "unknown"))

    if _upload_size(uf) == 0:
        return (
            f"'{file_name}' is empty (zero bytes), so there is nothing to "
            "transcribe. Check the file exported correctly, then upload it again."
        )

    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        described = suffix or "no extension"
        return (
            f"'{file_name}' has an unsupported format ({described}). "
            f"Chorus accepts: {supported}."
        )

    return None
