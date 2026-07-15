# Agent conventions for RB-* tasks (read before starting any task)

Shared rules for every `docs/tasks/RB-*.md` execution plan. Each task file assumes
you have read this once.

## 1. Validate against CI locally BEFORE committing (shift left)

Run the local CI mirror from the repo/worktree you are working in:

```bash
bash ~/.claude/skills/github-actions-locally/github-actions-locally.sh
```

It parses `.github/workflows/*.yml`, runs every runnable step, and reports what
passed. Interpret its output with these **known local false-positive patterns** —
all three were hit during the v4.0.1 cycle and cost CI round-trips:

| Symptom | Cause | What to do |
|---|---|---|
| `black local X ≠ CI pin 24.4.2 — auto-fix disabled` followed by Black FAILs listing `build/lib/...` and other files | You are in a worktree with **no `.venv`**, so the tool fell back to a newer global Black | Re-run with `--sync` (builds CI's exact pinned toolchain in an ephemeral venv), or run from a checkout whose `.venv` exists. Never hand-format with the unpinned Black. |
| `security.yml › Run pip-audit ✗ FAIL` flagging `black`, `torch`, `msgpack`, `pillow` | The local hook audits the entire dev venv; real CI installs a fresh scoped environment | Ignore locally. Real CI is authoritative. Verify the true state with `pip-audit -r requirements.txt --desc` if in doubt. |
| `Install Python dependencies ✗ TOOLCHAIN ISSUE` + `pyenv: No such file or directory` | This machine's bare `python` shim is broken | Not a code failure. Always use `source .venv/bin/activate 2>/dev/null; python3 …` in your own commands. |

A commit is ready to push when: the **lint** steps pass with a version-verified
toolchain, the **test** steps pass, and the only failures remaining match the
false-positive table above.

## 2. Git and PR flow

- Branch off current `main`; never commit to `main` directly.
- One PR per RB task, titled per the task file. Push, confirm **real** GitHub CI is
  green (`gh pr checks <n>`), and **leave the PR open — do not merge**.
- This repo squash-merges. Expect `ROADMAP.md` conflicts if another RB PR merges
  first; resolve by keeping both sides' completed `[x]` lines.

## 3. Environment

- Python: `source .venv/bin/activate 2>/dev/null; python3 …` — never bare `python`.
- Never install packages globally; use the project venv. New dev-only deps go in
  `pyproject.toml` dev extras, never `requirements.txt` (which is runtime-scoped and
  drift-checked against `pyproject.toml` dependencies by CI).
- If your worktree lacks `.venv`, the main checkout's interpreter at
  `<repo-root>/.venv/bin/python3` works from any directory (worktrees live under
  `<repo-root>/.claude/worktrees/`, so the main checkout is two levels up).

## 4. Style and honesty

- British English, active voice, Oxford comma in all docs/comments.
- Surgical changes: touch only the files your task lists, unless something breaks
  and you explain why in the PR body.
- Update `ROADMAP.md`: tick your RB item `[x]` with a one-line factual summary,
  matching the style of the completed RA-1–RA-9 lines above it.
- Report honestly: distinguish what you verified by execution from what you verified
  by reading. If a gate fails, fix the code — never weaken the gate. If you find a
  bug outside your task's scope, report it in your final summary; do not fix it.
