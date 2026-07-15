# RB-3: Fix LOW-tier strikethrough in PDF export

**Model tier:** Haiku · **Effort:** S · **Branch:** `fix/rb3-pdf-low-strikethrough`

> Read `docs/tasks/AGENT-CONVENTIONS.md` first.

## Context

Chorus marks LOW-confidence words in consensus Markdown as `**~~word~~**[^…]`
(bold + strikethrough + footnote). The PDF exporter
(`export_engine/exporter.py::export_pdf`) converts that Markdown to HTML via
`_md_to_html()` and ships it to WeasyPrint with CSS that styles LOW words red and
struck through via a rule like `del strong, strong del { … }`.

**The bug (found during RA-7, confirmed by test):** the `markdown` library is invoked
without any strikethrough/tilde extension, so `~~word~~` is never converted to a
`<del>` element — the tildes survive as literal characters inside `<strong>`, and the
CSS rule never fires. LOW-confidence words therefore render as ordinary bold text
with tildes in the PDF: the product's most important warning signal is silently lost
in that format. The word text itself is not dropped; only the styling and the
literal-tilde noise are wrong.

## The fix

In `_md_to_html()` in `export_engine/exporter.py`, pre-process the strikethrough
syntax before (or after) the `markdown.markdown(...)` call. The dependency-free
approach — do NOT add `pymdownx`/new packages — is a regex substitution on the
Markdown input:

```python
import re
md_text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", md_text)
```

Raw inline HTML passes through the `markdown` library untouched by default, so
`**<del>word</del>**` becomes `<strong><del>word</del></strong>` — which the existing
`del strong, strong del` CSS already targets. Check the existing CSS block in
`export_pdf` to confirm the selector, and check `export_docx` for comparison: the
DOCX path parses tiers itself and is NOT affected — do not touch it.

## Tests (the gate — write these first)

In `tests/test_exporter.py`, extend the existing PDF test class (it already has a spy
pattern capturing the HTML passed to `weasyprint.HTML` — reuse it):

1. A consensus Markdown containing a LOW word (`**~~garbl~~**`) produces HTML where
   `<del>garbl</del>` appears and the literal string `~~` does **not** appear.
2. `_md_to_html` unit test: `==word==` still maps to `<mark>` (regression guard for
   the MEDIUM path), and text containing a single `~` (not doubled) passes through
   unchanged.

Both tests must FAIL against the current code before your fix, and pass after —
verify and state this in the PR body.

## Verification (success criteria)

1. New tests fail pre-fix, pass post-fix; full suite passes.
2. Generate a real PDF once (the existing tests create sample votes — write a small
   throwaway script or reuse a test fixture) and visually confirm file size is sane
   and no `~~` appears; you cannot assert colour from bytes, the spy test covers the
   structural claim.
3. `black`/`ruff`/`isort` clean; local CI mirror per conventions doc.
4. Tick RB-3 in `ROADMAP.md`. PR title: `fix: render LOW-tier strikethrough in PDF export`.

## Files to change

- `export_engine/exporter.py` (`_md_to_html` only)
- `tests/test_exporter.py`
- `ROADMAP.md`
