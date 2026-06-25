# Contributing to Chorus Engine

Thank you for your interest in Chorus! This guide covers development setup, testing, and PR submission.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/incendiary/Chorus.git
cd Chorus
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev tools
```

### 4. Set Up Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Run checks on all files
```

This ensures code quality and secret detection before commits.

---

## Code Quality Standards

### Style Guide

- **Python:** PEP 8 via Ruff + Black
- **Imports:** Alphabetical via isort (Black-compatible profile)
- **Line length:** 88 characters
- **Type hints:** Optional but encouraged

### Linting & Formatting

```bash
# Check code style
ruff check .

# Auto-fix style issues
ruff check --fix .

# Format code
black .

# Sort imports
isort . --profile black
```

### Type Checking (Optional)

```bash
mypy . --ignore-missing-imports
```

---

## Testing

### Run Test Suite

```bash
pytest tests/
pytest tests/ -v  # Verbose output
pytest tests/ --cov  # With coverage report
```

### Test Organization

- `tests/test_*.py` — Unit and integration tests
- Tests use temporary directories via pytest's `tmp_path` fixture
- Mock Whisper transcription to avoid downloading models in CI

### Test Coverage Standards

- Aim for > 80% coverage on new code
- All public APIs should have at least one test
- Error cases should be tested (FileNotFoundError, ValueError, etc.)

---

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
```

Branch naming conventions:
- `feat/*` — New features
- `fix/*` — Bug fixes
- `docs/*` — Documentation
- `chore/*` — Maintenance, dependencies
- `refactor/*` — Code restructuring (no functional change)

### 2. Make Your Changes

- Keep commits focused and logical
- Write descriptive commit messages (see below)
- Update tests for new functionality
- Update docstrings and comments as needed

### 3. Commit Message Format

```
type: brief description (under 70 chars)

Longer explanation if needed. Explain the WHY, not the WHAT.
- Use bullet points for multiple reasons
- Reference related issues: "Fixes #123"
- Mention breaking changes: "BREAKING: renamed X to Y"

Co-Authored-By: Your Name <your.email@example.com>
```

**Type keywords:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `refactor:` Code refactoring (no behavior change)
- `chore:` Maintenance, deps, CI
- `test:` Test additions/fixes
- `perf:` Performance improvement

### 4. Push and Open a PR

```bash
git push origin feat/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference any related issues
- Mention if this is a breaking change

---

## Code Review Checklist

When submitting a PR, ensure:

- [ ] Tests pass: `pytest tests/`
- [ ] Code is formatted: `black . && isort .`
- [ ] Linting passes: `ruff check .`
- [ ] No secrets committed: `detect-secrets scan`
- [ ] Documentation is updated (docstrings, README, ROADMAP)
- [ ] Commit messages follow the format above
- [ ] No hardcoded paths, API keys, or sensitive data

---

## Architecture Overview

For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Key Modules

- `pipeline_runner.py` — Main orchestrator
- `audio_processor/` — Audio cleaning filters
- `transcription_engine/` — Whisper model wrapper
- `consensus_merger/` — Word-level voting and alignment
- `export_engine/` — Multi-format export (PDF, DOCX, SRT, VTT, JSON)
- `diarisation/` — Speaker identification (pyannote)
- `nlp_reconstructor/` — spaCy-based token recovery
- `llm_reconstructor/` — Ollama-based token recovery
- `batch_processor/` — Multi-file processing
- `ui/` — Streamlit interface

---

## Running Chorus Locally

### CLI Mode

```bash
python pipeline_runner.py /path/to/audio.wav --language en
```

### UI Mode

```bash
streamlit run ui/app.py
```

### Docker Mode

```bash
docker-compose up --build
# Then visit http://localhost:8501
```

---

## Debugging

### Enable Debug Logging

```bash
export LOGLEVEL=DEBUG
streamlit run ui/app.py
```

Or in code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Pipeline State

```python
from pipeline_runner import run_pipeline
results = run_pipeline("/audio.wav")
print(results.keys())  # ['variant_paths', 'transcripts', 'consensus_path', ...]
```

### Check Outputs

```bash
ls -lh outputs/consensus/        # Final transcripts
ls -lh outputs/transcripts/       # Raw Whisper outputs
ls -lh outputs/variants/          # Cleaned audio
```

---

## Release Process

### Version Bumping

Chorus uses semantic versioning: `MAJOR.MINOR.PATCH`

Update `VERSION` file and `pyproject.toml[project]version`:

```bash
echo "3.2.0" > VERSION
# Update pyproject.toml version field
```

### Creating a Release

Push a git tag and GitHub Actions handles Docker image publishing (major releases only):

```bash
git tag v3.2.0
git push origin v3.2.0
```

See [.github/workflows/release.yml](.github/workflows/release.yml) for full CI pipeline.

---

## Questions?

- Check [README.md](README.md) for project overview
- Review [ROADMAP.md](ROADMAP.md) for planned work
- See [docs/](docs/) for detailed guides (Docker, scalability, etc.)
- Open an issue for bugs or feature requests

Thanks for contributing! 🎉
