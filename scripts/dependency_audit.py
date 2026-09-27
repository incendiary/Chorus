#!/usr/bin/env python3
"""Judge a pip-audit or safety JSON report against the known-advisories list.

Usage:
    python scripts/dependency_audit.py {pip-audit|safety} REPORT.json

Known advisories (see .github/known-advisories.txt) are printed as accepted
and never fail the run. Any other finding is new: it is printed as a GitHub
``::warning::`` annotation, appended to a markdown table in
$GITHUB_STEP_SUMMARY (if set), and reported via ``new=true`` written to
$GITHUB_OUTPUT (if set), so a caller can raise it separately without failing
CI on it.

The script exits 0 whenever the report file could be parsed, whether or not
new findings were found: known and new advisories are both expected outcomes
of a normal run, not audit failures. It exits 2, with a ``::error::``
annotation, only when the scanner itself failed to produce something
readable: a missing, empty, or unparsable report file. That is the one
condition that must fail CI, since it means the scanner crashed rather than
ran clean.
"""

import json
import os
import sys
from pathlib import Path

KNOWN_ADVISORIES_PATH = (
    Path(__file__).resolve().parent.parent / ".github" / "known-advisories.txt"
)


def load_known_ids(path=KNOWN_ADVISORIES_PATH):
    """Return the set of accepted advisory identifiers from the known list."""
    known = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                known.add(line)
    return known


def load_report(path):
    """Read the report file and parse the first JSON object found in it.

    Returns the parsed object. Raises ValueError if the file is missing,
    empty, or contains no parseable JSON object.
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise ValueError(f"report file could not be read: {exc}") from exc

    if not text.strip():
        raise ValueError("report file is empty")

    # Tolerate non-JSON text around the report: safety prints a deprecation
    # banner both before and after its JSON, so parse from the first "{" and
    # stop at the end of that object.
    start = text.find("{")
    if start == -1:
        raise ValueError("report file does not contain a JSON object")

    try:
        report, _end = json.JSONDecoder().raw_decode(text[start:])
        return report
    except json.JSONDecodeError as exc:
        raise ValueError(f"report file is not valid JSON: {exc}") from exc


def extract_findings(tool, report):
    """Return a list of findings as dicts with id, aliases, package, version."""
    findings = []
    if tool == "pip-audit":
        for dep in report.get("dependencies", []) or []:
            for vuln in dep.get("vulns", []) or []:
                findings.append(
                    {
                        "id": vuln.get("id", ""),
                        "aliases": vuln.get("aliases", []) or [],
                        "package": dep.get("name", ""),
                        "version": dep.get("version", ""),
                    }
                )
    elif tool == "safety":
        for vuln in report.get("vulnerabilities", []) or []:
            aliases = []
            cve = vuln.get("CVE")
            if cve:
                aliases.append(cve)
            findings.append(
                {
                    "id": vuln.get("vulnerability_id", ""),
                    "aliases": aliases,
                    "package": vuln.get("package_name", ""),
                    "version": vuln.get("analyzed_version", ""),
                }
            )
    else:
        raise ValueError(f"unknown tool: {tool}")
    return findings


def is_known(finding, known_ids):
    ids = {finding["id"], *finding["aliases"]}
    return bool(ids & known_ids)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("pip-audit", "safety"):
        print(
            "usage: dependency_audit.py {pip-audit|safety} REPORT.json", file=sys.stderr
        )
        return 2

    tool, report_path = sys.argv[1], sys.argv[2]

    try:
        report = load_report(report_path)
    except ValueError as exc:
        print(f"::error::{tool} report at {report_path} could not be parsed: {exc}")
        return 2

    known_ids = load_known_ids()
    findings = extract_findings(tool, report)

    known_findings = [f for f in findings if is_known(f, known_ids)]
    new_findings = [f for f in findings if not is_known(f, known_ids)]

    for f in known_findings:
        print(f"accepted: {f['id']} in {f['package']} {f['version']} (known advisory)")

    for f in new_findings:
        print(
            f"::warning::New dependency advisory {f['id']} in {f['package']} "
            f"{f['version']} is not in .github/known-advisories.txt"
        )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"\n### {tool} dependency audit\n\n")
            if new_findings:
                f.write("| Package | Version | Advisory |\n")
                f.write("|---------|---------|----------|\n")
                for finding in new_findings:
                    f.write(
                        f"| {finding['package']} | {finding['version']} | {finding['id']} |\n"
                    )
            else:
                f.write("No new advisories.\n")
            if known_findings:
                f.write(
                    f"\n{len(known_findings)} known advisory finding(s) accepted, see SECURITY.md.\n"
                )

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"new={'true' if new_findings else 'false'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
