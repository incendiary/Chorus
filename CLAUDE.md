# Chorus Engine — Autonomous Maintenance Guidelines

## Architecture Summary

The **Chorus Engine** is a local, containerised Python application that automates high-fidelity audio transcription using a multi-pass consensus methodology.

### Core Modules
1. **`audio_processor/`**: Ingests raw audio and applies three distinct cleaning filters via `pydub`/`librosa` (High-Pass Focus, Dynamic Range Normalization, Denoise Filter).
2. **`transcription_engine/`**: Wraps a local instance of OpenAI's Whisper model. Orchestrates sequential transcription over the original audio and the three cleaned variants.
3. **`consensus_merger/`**: Performs a word-for-word sliding-window consensus analysis across the four transcript variants. Calculates confidence weights and renders a unified Markdown document with tier-based highlighting.
4. **`diarisation/`**: Integrates `pyannote.audio` to identify and separate multiple speakers, fusing them with Whisper segment timestamps.
5. **`nlp_reconstructor/`**: Uses `spaCy` grammatical and semantic analysis to reconstruct LOW-confidence tokens.
6. **`export_engine/`**: Converts the consensus Markdown into PDF, DOCX, SRT, and VTT formats.
7. **`batch_processor/`**: CLI tool for unattended processing of multiple files or entire directories.
8. **`ui/`**: A Streamlit interface providing an interactive dashboard to trigger the pipeline and review/download variants.

---

## Development Rules

When maintaining, refactoring, or extending this codebase, Claude Code must adhere strictly to the following rules:

- **Language & Style**: Use professional British English in all documentation, comments, and user-facing text. Use the active voice and strictly enforce the Oxford comma.
- **Documentation Parity**: You must maintain project documentation during any refactoring. If a module's public API or behaviour changes, update the docstrings, `README.md`, and this file accordingly.
- **Surgical Changes**: Touch only what you must. Do not introduce speculative abstractions or wrapper functions that add no logic.
- **Formatting**: Adhere to the established `black`, `ruff`, and `isort` configurations.

---

## Unit Testing Standard

Before pushing any code to the repository, you must execute and pass the following testing checklist. If tests do not exist for modified components, you must write them.

### Mandatory Checklist

1. **Input Validation**:
   - Verify that the `audio_processor` gracefully rejects unsupported file formats or corrupted audio files.
   - Ensure the Streamlit UI properly handles empty uploads or cancelled operations.

2. **Null/Empty States**:
   - Verify that the `consensus_merger` handles empty transcripts (e.g., silence) without throwing `ZeroDivisionError` or index out-of-bounds exceptions.
   - Ensure the `nlp_reconstructor` falls back safely if `spaCy` models are missing or return no valid tokens.

3. **Performance Benchmarks**:
   - Verify that the `audio_processor` cleaning filters process a standard 5-minute audio file within acceptable memory limits (under 500MB peak RAM).
   - Ensure the `consensus_merger` sliding-window alignment logic completes in under 2 seconds for a 10,000-word transcript.

4. **Security & Secrets**:
   - Run `pre-commit run --all-files` to verify no secrets, API keys, or sensitive tokens are present in the working tree.
   - Run `detect-secrets scan --baseline .secrets.baseline` and ensure no new un-audited secrets are flagged.
