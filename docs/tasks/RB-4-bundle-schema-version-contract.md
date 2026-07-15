# RB-4: Version the bundle schema and add a doc contract test

**Model tier:** Haiku · **Effort:** S · **Branch:** `feat/rb4-bundle-schema-contract`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first.

## Context

`{stem}_bundle.json` (produced by `export_engine/exporter.py::export_transcript_bundle`,
around line 719) is Chorus's machine-readable output for LLM/programmatic consumers,
documented in detail in `docs/CHORUS_FOR_LLMS.md` §5. Two gaps:

1. The function's docstring promises `meta` contains "chorus version", but the code
   writes only `stem`, `source_filename`, and `generated_at`. A consumer cannot tell
   which producer version or contract revision generated a bundle.
2. Nothing ties `docs/CHORUS_FOR_LLMS.md` §5's documented schema to the real output —
   the contract can drift silently when fields change.

## The fix

### Part A — version fields

In the `bundle` dict in `export_transcript_bundle`, extend `meta`:

```python
"meta": {
    "stem": stem,
    "source_filename": source_filename,
    "generated_at": datetime.now(UTC).isoformat(),
    "chorus_version": _read_version(),
    "schema_version": 1,
},
```

For `_read_version()`: the single source of truth is the root `VERSION` file
(`Path(__file__).resolve().parent.parent / "VERSION"`, read + `.strip()`; fall back to
`"unknown"` on any `OSError`). Check first whether a version-reading helper already
exists in the codebase (`grep -rn "VERSION" config.py utils.py chorus/`) and reuse it
if so rather than writing a new one.

`schema_version` is a hard-coded integer literal `1` — bump it manually in future
whenever a field is renamed/removed (adding fields is backwards-compatible and does
not require a bump).

### Part B — contract test

New test in `tests/test_exporter.py` (same class as the existing bundle tests):

1. `test_bundle_meta_versioned` — generated bundle's `meta` has `chorus_version`
   matching the `VERSION` file's contents and `schema_version == 1`.
2. `test_bundle_matches_documented_contract` — parse the fenced JSON example in
   `docs/CHORUS_FOR_LLMS.md` §5 (extract the first ```json code block after the
   heading `## 5.`; `json.loads` will fail on its `"...": "..."` placeholder entries
   inside `variants`, so compare at the level that matters instead: assert the
   **top-level keys** documented (`meta`, `variants`, `consensus`, `statistics`), the
   **consensus-entry keys** (`word`, `tier`, `confidence`, `count`, `total`,
   `variants`), and the **statistics keys** all appear as string literals in the doc
   section, AND that a real generated bundle contains exactly those top-level keys,
   exactly those consensus-entry keys, and at least those statistics keys. The test's
   purpose: if someone adds/renames a bundle field without updating the doc — or vice
   versa — this test fails with a message pointing at both files.

### Part C — documentation parity

Update `docs/CHORUS_FOR_LLMS.md` §5's example JSON to include the two new `meta`
fields, and add one sentence explaining `schema_version`. Update the docstring of
`export_transcript_bundle` so it matches reality.

## Verification (success criteria)

1. Both new tests fail against pre-change code (Part B's real-bundle key assertion
   fails because `chorus_version`/`schema_version` are documented but absent —
   confirm the failure direction makes sense), pass after.
2. Full suite passes; `black`/`ruff`/`isort` clean; local CI mirror per conventions.
3. Tick RB-4 in `ROADMAP.md`. PR title: `feat: version bundle schema and enforce doc contract`.

## Files to change

- `export_engine/exporter.py`
- `tests/test_exporter.py`
- `docs/CHORUS_FOR_LLMS.md`
- `ROADMAP.md`
