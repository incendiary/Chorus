"""tests/test_dependency_audit.py — unit tests for scripts/dependency_audit.py.

Covers the known-advisory filtering (including alias/CVE matching), new-advisory
reporting via GITHUB_OUTPUT, the missing/empty/garbage report failure modes, safety's
non-JSON preamble tolerance, and the cross-check that every id in
.github/known-advisories.txt is documented in SECURITY.md.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "dependency_audit.py"
KNOWN_ADVISORIES_PATH = REPO_ROOT / ".github" / "known-advisories.txt"
SECURITY_MD_PATH = REPO_ROOT / "SECURITY.md"

spec = importlib.util.spec_from_file_location("dependency_audit", SCRIPT_PATH)
dependency_audit = importlib.util.module_from_spec(spec)
sys.modules["dependency_audit"] = dependency_audit
spec.loader.exec_module(dependency_audit)


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestLoadKnownIds:
    def test_loads_ids_ignoring_comments_and_blanks(self, tmp_path):
        known_file = tmp_path / "known.txt"
        known_file.write_text(
            "# header comment\n\nPYSEC-2026-3624   # lightning\nCVE-2026-58659\n",
            encoding="utf-8",
        )
        assert dependency_audit.load_known_ids(known_file) == {
            "PYSEC-2026-3624",
            "CVE-2026-58659",
        }


class TestExtractFindings:
    def test_pip_audit_findings(self):
        report = {
            "dependencies": [
                {
                    "name": "nltk",
                    "version": "3.10.3",
                    "vulns": [{"id": "PYSEC-2026-3740", "aliases": []}],
                },
                {"name": "numpy", "version": "2.4.6", "vulns": []},
            ]
        }
        findings = dependency_audit.extract_findings("pip-audit", report)
        assert findings == [
            {
                "id": "PYSEC-2026-3740",
                "aliases": [],
                "package": "nltk",
                "version": "3.10.3",
            }
        ]

    def test_safety_findings_use_cve_as_alias(self):
        report = {
            "vulnerabilities": [
                {
                    "vulnerability_id": "SFTY-20260715-05023",
                    "package_name": "lightning",
                    "analyzed_version": "2.6.6",
                    "CVE": "CVE-2026-58659",
                }
            ]
        }
        findings = dependency_audit.extract_findings("safety", report)
        assert findings == [
            {
                "id": "SFTY-20260715-05023",
                "aliases": ["CVE-2026-58659"],
                "package": "lightning",
                "version": "2.6.6",
            }
        ]


class TestIsKnown:
    def test_matches_on_primary_id(self):
        finding = {"id": "PYSEC-2026-3740", "aliases": []}
        assert dependency_audit.is_known(finding, {"PYSEC-2026-3740"})

    def test_matches_on_alias(self):
        finding = {"id": "SFTY-20260715-05023", "aliases": ["CVE-2026-58659"]}
        assert dependency_audit.is_known(finding, {"CVE-2026-58659"})

    def test_unknown_finding_does_not_match(self):
        finding = {"id": "PYSEC-2099-0001", "aliases": ["CVE-2099-0001"]}
        assert not dependency_audit.is_known(finding, {"PYSEC-2026-3740"})


class TestMainKnownFinding:
    def test_known_finding_exits_zero_and_reports_no_new(
        self, tmp_path, monkeypatch, capsys
    ):
        report_path = tmp_path / "pip-audit.json"
        _write_json(
            report_path,
            {
                "dependencies": [
                    {
                        "name": "nltk",
                        "version": "3.10.3",
                        "vulns": [{"id": "PYSEC-2026-3740", "aliases": []}],
                    }
                ]
            },
        )
        output_path = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "pip-audit", str(report_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "accepted: PYSEC-2026-3740" in captured.out
        assert "::warning::" not in captured.out
        assert output_path.read_text(encoding="utf-8") == "new=false\n"


class TestMainNewFinding:
    def test_new_finding_exits_zero_warns_and_sets_output_true(
        self, tmp_path, monkeypatch, capsys
    ):
        report_path = tmp_path / "pip-audit.json"
        _write_json(
            report_path,
            {
                "dependencies": [
                    {
                        "name": "some-pkg",
                        "version": "1.0.0",
                        "vulns": [
                            {"id": "PYSEC-2099-9999", "aliases": ["CVE-2099-9999"]}
                        ],
                    }
                ]
            },
        )
        output_path = tmp_path / "github_output"
        summary_path = tmp_path / "step_summary"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "pip-audit", str(report_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "::warning::New dependency advisory PYSEC-2099-9999" in captured.out
        assert output_path.read_text(encoding="utf-8") == "new=true\n"
        assert "some-pkg" in summary_path.read_text(encoding="utf-8")


class TestMainReportFailureModes:
    def test_missing_report_exits_two(self, tmp_path, monkeypatch, capsys):
        missing_path = tmp_path / "does-not-exist.json"
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "pip-audit", str(missing_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 2
        assert "::error::" in capsys.readouterr().out

    def test_empty_report_exits_two(self, tmp_path, monkeypatch, capsys):
        report_path = tmp_path / "empty.json"
        report_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "pip-audit", str(report_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 2
        assert "::error::" in capsys.readouterr().out

    def test_garbage_report_exits_two(self, tmp_path, monkeypatch, capsys):
        report_path = tmp_path / "garbage.json"
        report_path.write_text("this is not json at all", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "pip-audit", str(report_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 2
        assert "::error::" in capsys.readouterr().out


class TestSafetyTextPreamble:
    def test_tolerates_non_json_preamble_before_first_brace(
        self, tmp_path, monkeypatch
    ):
        report_path = tmp_path / "safety.json"
        report_path.write_text(
            "Safety v3.8.1 scanning your environment...\n"
            '{"vulnerabilities": [{"vulnerability_id": "SFTY-20260120-40557", '
            '"package_name": "cuda-toolkit", "analyzed_version": "12.0", "CVE": "CVE-2025-33228"}]}',
            encoding="utf-8",
        )
        output_path = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
        monkeypatch.setattr(
            sys, "argv", ["dependency_audit.py", "safety", str(report_path)]
        )

        exit_code = dependency_audit.main()

        assert exit_code == 0
        assert output_path.read_text(encoding="utf-8") == "new=false\n"


class TestKnownAdvisoriesDocumentedInSecurityMd:
    def test_every_known_id_appears_in_security_md(self):
        known_ids = dependency_audit.load_known_ids(KNOWN_ADVISORIES_PATH)
        security_text = SECURITY_MD_PATH.read_text(encoding="utf-8")
        missing = [
            advisory_id for advisory_id in known_ids if advisory_id not in security_text
        ]
        assert not missing, f"IDs missing from SECURITY.md: {missing}"

    def test_cross_check_fails_on_a_bogus_id(self, tmp_path):
        """Prove the cross-check above actually catches a mismatch.

        A known-advisories file with an id SECURITY.md does not document must
        fail this same assertion, otherwise the real test above could pass
        vacuously.
        """
        bogus_file = tmp_path / "known-with-bogus.txt"
        bogus_file.write_text(
            "PYSEC-2026-3624\nCVE-9999-00000-BOGUS\n", encoding="utf-8"
        )
        known_ids = dependency_audit.load_known_ids(bogus_file)
        security_text = SECURITY_MD_PATH.read_text(encoding="utf-8")
        missing = [
            advisory_id for advisory_id in known_ids if advisory_id not in security_text
        ]
        assert missing == ["CVE-9999-00000-BOGUS"]
