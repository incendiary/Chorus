"""tests/test_upload_validation.py — Validation of file uploads in the UI."""

from __future__ import annotations

from unittest.mock import MagicMock

from ui.upload_validation import validate_upload


class TestUploadValidation:
    """Test the upload validation logic for the file uploader."""

    def test_rejects_zero_byte_file(self):
        """A zero-byte file should be rejected with a clear message."""
        mock_file = MagicMock()
        mock_file.name = "empty.wav"
        mock_file.read.return_value = b""

        reason = validate_upload(mock_file)
        assert reason is not None
        assert "empty" in reason.lower()

    def test_rejects_unsupported_extension(self):
        """A file with an unsupported extension should be rejected."""
        mock_file = MagicMock()
        mock_file.name = "document.txt"
        mock_file.read.return_value = b"This is not audio"

        reason = validate_upload(mock_file)
        assert reason is not None
        assert ".txt" in reason or "extension" in reason.lower()

    def test_accepts_valid_wav_file(self):
        """A non-empty WAV file should be accepted."""
        mock_file = MagicMock()
        mock_file.name = "valid.wav"
        mock_file.read.return_value = b"RIFF" + b"\x00" * 100  # Minimal WAV header

        reason = validate_upload(mock_file)
        assert reason is None

    def test_accepts_valid_mp3_file(self):
        """A non-empty MP3 file should be accepted."""
        mock_file = MagicMock()
        mock_file.name = "valid.mp3"
        mock_file.read.return_value = b"ID3" + b"\x00" * 100  # Minimal ID3 header

        reason = validate_upload(mock_file)
        assert reason is None

    def test_accepts_valid_m4a_file(self):
        """A non-empty M4A file should be accepted."""
        mock_file = MagicMock()
        mock_file.name = "valid.m4a"
        mock_file.read.return_value = b"\x00" * 100

        reason = validate_upload(mock_file)
        assert reason is None

    def test_empty_filename_with_supported_extension(self):
        """An empty file is rejected regardless of extension."""
        mock_file = MagicMock()
        mock_file.name = ""
        mock_file.read.return_value = b""

        reason = validate_upload(mock_file)
        assert reason is not None

    def test_rejects_uppercase_unsupported_extension(self):
        """Unsupported extension should be rejected even if uppercase."""
        mock_file = MagicMock()
        mock_file.name = "document.TXT"
        mock_file.read.return_value = b"content"

        reason = validate_upload(mock_file)
        assert reason is not None

    def test_rejects_unsupported_extension_lowercase(self):
        """Unsupported extension should be rejected in lowercase."""
        mock_file = MagicMock()
        mock_file.name = "document.txt"
        mock_file.read.return_value = b"content"

        reason = validate_upload(mock_file)
        assert reason is not None


class TestValidationDoesNotConsumeTheUpload:
    """Validation must leave the stream readable.

    ``spool_upload`` reads the same object immediately after validation. If
    validation consumed it, every accepted upload would be written to disk as
    zero bytes and the whole Web UI upload path would be silently broken. A
    MagicMock cannot catch this, because its ``read()`` returns the same value
    on every call: these use a real file-like object.
    """

    def _upload(self, name: str, payload: bytes):
        import io

        stream = io.BytesIO(payload)
        stream.name = name
        return stream

    def test_a_valid_upload_is_still_fully_readable_afterwards(self):
        payload = b"RIFF....WAVEfmt " + b"\x00" * 64
        upload = self._upload("recording.wav", payload)

        assert validate_upload(upload) is None
        assert upload.read() == payload

    def test_a_size_attribute_is_preferred_over_reading(self):
        """Streamlit's UploadedFile exposes size, so the stream is untouched."""

        class _Sized:
            name = "recording.wav"
            size = 1024

            def read(self):  # pragma: no cover - must never be called
                raise AssertionError("validation read the stream despite .size")

        assert validate_upload(_Sized()) is None

    def test_an_empty_upload_is_rejected_by_content_not_name(self):
        upload = self._upload("recording.wav", b"")

        reason = validate_upload(upload)

        assert reason is not None
        assert "zero bytes" in reason


class TestUnsupportedFormatMessage:
    def test_a_renamed_text_file_is_accepted_on_extension_alone(self):
        """Documents a deliberate limit rather than implying otherwise.

        Validation checks the name and the size, not the container. A text
        file renamed to .wav passes here and fails later in decoding. Content
        sniffing was out of scope; this pins the actual behaviour so the gap
        is visible rather than assumed closed.
        """
        import io

        upload = io.BytesIO(b"this is plain text, not audio")
        upload.name = "fake.wav"

        assert validate_upload(upload) is None
