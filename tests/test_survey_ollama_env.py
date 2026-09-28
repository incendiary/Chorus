"""tests/test_survey_ollama_env.py — subprocess coverage for
devops-practices/survey-ollama-env.sh.

The script has no automated tests and shells out to real hardware-probing
and networking commands (uname, sysctl, getconf, nproc, system_profiler,
nvidia-smi, curl, ollama). Every one of those is replaced here with a small
fake executable placed at the front of PATH, so hardware detection is
deterministic and no real hardware or network is queried. PATH is otherwise
restricted to /usr/bin:/bin so a real `ollama` (or `nvidia-smi`) installed
elsewhere on the host machine (e.g. Homebrew's /opt/homebrew/bin) is never
picked up.

The script resolves its .env location as one directory above wherever it is
run from (``dirname "$SCRIPT_DIR"``). Rather than touching the script, each
test copies it into a fresh ``tmp_path/devops-practices/`` directory, so that
resolved path lands entirely inside tmp_path. HOME and the subprocess cwd are
also pointed at tmp_path. The repository's real .env is never read or
written by these tests.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ui.hardware_survey import _recommend_device

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="bash script, not supported on Windows"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_SOURCE = REPO_ROOT / "devops-practices" / "survey-ollama-env.sh"


def _write_fake(bin_dir: Path, name: str, body: str) -> None:
    """Write a small fake executable named ``name`` into ``bin_dir``."""
    path = bin_dir / name
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_script(tmp_path: Path) -> Path:
    """Copy the real script into tmp_path/devops-practices/ so its resolved
    REPO_ROOT (one directory up from the script) is tmp_path itself."""
    script_dir = tmp_path / "devops-practices"
    script_dir.mkdir(exist_ok=True)
    dest = script_dir / "survey-ollama-env.sh"
    shutil.copy2(SCRIPT_SOURCE, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR)
    return dest


def _build_fake_path(
    tmp_path: Path, *, uname: str, mem_gb: int, cpu: int, gpu: str, vram_gb: int = 0
) -> str:
    """Populate tmp_path/fakebin with fake hardware/network probes and return
    a PATH string with it prepended to a minimal, hermetic system PATH.

    gpu is one of "apple", "nvidia", "none".
    """
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)

    _write_fake(fakebin, "uname", f'echo "{uname}"')

    page_size = 4096
    phys_pages = mem_gb * 1024 * 1024 * 1024 // page_size
    _write_fake(
        fakebin,
        "sysctl",
        f"""
if [[ "$2" == "hw.memsize" ]]; then
  echo $(( {mem_gb} * 1024 * 1024 * 1024 ))
elif [[ "$2" == "hw.ncpu" ]]; then
  echo "{cpu}"
fi
""",
    )
    _write_fake(
        fakebin,
        "getconf",
        f"""
case "$1" in
  _PHYS_PAGES) echo "{phys_pages}" ;;
  PAGE_SIZE) echo "{page_size}" ;;
esac
""",
    )
    _write_fake(fakebin, "nproc", f'echo "{cpu}"')

    if gpu == "apple":
        _write_fake(fakebin, "system_profiler", 'echo "Chip: Apple M2 Pro"')
    else:
        _write_fake(fakebin, "system_profiler", 'echo "Processor Name: Intel Core i7"')

    if gpu == "nvidia":
        vram_mb = vram_gb * 1024
        _write_fake(
            fakebin,
            "nvidia-smi",
            f"""
case "$*" in
  *memory.total*) echo "{vram_mb}" ;;
  *) echo "Fake GPU 9000" ;;
esac
""",
        )

    # Always report the registry probe (and any other curl use) as
    # unreachable, so check_model_exists()'s documented "assume valid on
    # network failure" branch fires deterministically without a real
    # network call.
    _write_fake(fakebin, "curl", 'echo -n "000"')

    # /usr/bin:/bin carries the real coreutils (sed, grep, cut, paste, tr,
    # head, awk, df, seq, bash) needed for the rest of the script, without
    # exposing a host-installed ollama/nvidia-smi (e.g. Homebrew's
    # /opt/homebrew/bin/ollama).
    return f"{fakebin}:/usr/bin:/bin"


def _run_script(
    tmp_path: Path, path_value: str, stdin_text: str
) -> subprocess.CompletedProcess:
    script = _install_script(tmp_path)
    env = {"PATH": path_value, "HOME": str(tmp_path)}
    return subprocess.run(
        ["/bin/bash", str(script)],
        cwd=tmp_path,
        env=env,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


# Two prompts always fire regardless of hardware, since no fake `ollama` is
# ever on PATH (OLLAMA_INSTALLED is always false): the hidden Hugging Face
# token prompt, then the .env apply-selection prompt.
def _stdin(env_selection: str) -> str:
    return f"\n{env_selection}\n"


# ── (a) device recommendation matches ui.hardware_survey._recommend_device ──


class TestDeviceRecommendation:
    def test_cuda(self, tmp_path):
        path_value = _build_fake_path(
            tmp_path, uname="Linux", mem_gb=32, cpu=16, gpu="nvidia", vram_gb=12
        )
        result = _run_script(tmp_path, path_value, _stdin("0"))

        assert result.returncode == 0, result.stderr
        assert (
            "GPU: NVIDIA CUDA" in result.stdout
            or "detected GPU type is 'NVIDIA CUDA'" in result.stdout
        )
        expected = _recommend_device("nvidia_cuda")
        assert expected == "cuda"
        assert "  WHISPER_DEVICE=cuda" in result.stdout

    def test_mps(self, tmp_path):
        path_value = _build_fake_path(
            tmp_path, uname="Darwin", mem_gb=16, cpu=10, gpu="apple"
        )
        result = _run_script(tmp_path, path_value, _stdin("0"))

        assert result.returncode == 0, result.stderr
        expected = _recommend_device("apple_mps")
        assert expected == "mps"
        assert "  WHISPER_DEVICE=mps" in result.stdout

    def test_cpu(self, tmp_path):
        path_value = _build_fake_path(
            tmp_path, uname="Linux", mem_gb=8, cpu=4, gpu="none"
        )
        result = _run_script(tmp_path, path_value, _stdin("0"))

        assert result.returncode == 0, result.stderr
        expected = _recommend_device("none")
        assert expected == "cpu"
        assert "  WHISPER_DEVICE=cpu" in result.stdout


# ── (b) / (c) applying settings to .env ──────────────────────────────────────

ORIGINAL_ENV_CONTENT = """# Chorus environment
WHISPER_MODEL=large
WHISPER_DEVICE=cuda
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://old-host:1234
CUSTOM_VAR=keep-me
ANOTHER_LINE=unchanged
"""


class TestApplyToEnvFile:
    def test_apply_writes_only_selected_keys(self, tmp_path):
        # CPU-only, 8GB RAM: WHISPER_MODEL -> "small", WHISPER_DEVICE -> "cpu"
        # (see the elif TOTAL_MEM_GB >= 8 branches in the script).
        path_value = _build_fake_path(
            tmp_path, uname="Linux", mem_gb=8, cpu=4, gpu="none"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(ORIGINAL_ENV_CONTENT)

        # Options are, in order: 1) WHISPER_MODEL 2) WHISPER_DEVICE
        # 3) OLLAMA_MODEL 4) OLLAMA_BASE_URL. Select only the first two.
        result = _run_script(tmp_path, path_value, _stdin("1 2"))

        assert result.returncode == 0, result.stderr

        expected_lines = ORIGINAL_ENV_CONTENT.splitlines()
        expected_lines[1] = "WHISPER_MODEL=small"
        expected_lines[2] = "WHISPER_DEVICE=cpu"
        expected_content = "\n".join(expected_lines) + "\n"

        assert env_file.read_text() == expected_content

    def test_decline_leaves_env_byte_identical(self, tmp_path):
        path_value = _build_fake_path(
            tmp_path, uname="Linux", mem_gb=8, cpu=4, gpu="none"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(ORIGINAL_ENV_CONTENT)

        result = _run_script(tmp_path, path_value, _stdin("0"))

        assert result.returncode == 0, result.stderr
        assert env_file.read_text() == ORIGINAL_ENV_CONTENT
