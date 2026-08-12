# Agentic RAG — Working Agreement

This file governs how work gets done in this repository: process, workflow, and
agent behavior. It does not describe the product itself — for the system's
architecture, functional requirements, and technical specifications, see
[`docs/REQUIREMENTS.md`](../docs/REQUIREMENTS.md). For the phased build plan and
current status, see [`PROJECT_TRACKER.md`](../PROJECT_TRACKER.md).

---

## 1. Core Principles

- **Never invent architecture or requirements.** Everything built must trace back
  to `docs/REQUIREMENTS.md`, an ADR, or an explicit instruction from the user.
  If something is unspecified or ambiguous, ask — do not assume, do not guess.
- **Read before you write.** Always read the existing, relevant code before
  editing it. Before fixing a bug: read the code, understand why it's failing,
  and plan the fix before touching anything.
- **Do no harm to working code.** A fix or a new feature must not break existing,
  passing functionality. Run the relevant tests before and after a change.
- **Simplicity is a requirement, not a preference.** Code must be simple, human
  readable, and human understandable. Prefer the obvious solution over the
  clever one. No speculative abstraction, no unused flexibility.
- **One task, one concern.** Don't bundle unrelated changes together.

## 2. Test-Driven Development

- Tests are written **before** implementation code, for every feature and every
  bug fix. No exceptions.
- A feature is not "done" until its tests exist, pass, and cover the edge cases
  implied by `docs/REQUIREMENTS.md` (e.g. permission boundaries, empty corpus,
  no-match queries).
- Test files mirror source structure (e.g. `src/retrieval/reranker.py` →
  `tests/retrieval/test_reranker.py`).
- Tests must be deterministic — no reliance on live external services, real
  wall-clock timing, or non-seeded randomness.

## 3. Feature Branching & Pull Requests

- Work is divided into features. Each feature lives on its own branch and gets
  its own pull request — never bundle multiple features into one PR.
- Every PR is reviewed automatically by **Kodus.io**.
  - If Kodus.io approves / marks the PR mergeable, it merges.
  - If Kodus.io returns **"Request changes,"** address every point raised,
    push the fix, and let it re-review. Do not merge over an outstanding
    request-changes status and do not dismiss its feedback without reason.
- Branch naming: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`,
  `chore/<short-name>`, `test/<short-name>` — matching the commit type below.

## 4. Commit Messages

Conventional commits: `type(scope): description`

| Type | When |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | No behavior change |
| `chore` | Tooling, deps, config |

- Description: lowercase, imperative, no trailing period.
- Body explains *why* when the change isn't self-evident from the diff.

## 5. Secrets & Configuration

- All API keys, tokens, and other sensitive values live in `.env`, which is
  gitignored. `.env.example` documents every required variable with a
  placeholder value and is committed.
- Never commit a real secret. Before pushing, check `git status`/`git diff`
  for anything that looks like a credential, even in files that don't look
  like config.
- **All configuration lives in exactly one place** — a single, central config
  module (proposed: `config.py`, loaded via `pydantic-settings` from `.env`).
  No settings scattered across files, no magic numbers duplicated at call
  sites. If a value is likely to change between environments (model names,
  chunk sizes, top-k values, thresholds), it belongs in that one config
  module, not hardcoded inline.

## 6. Tracking Progress

- `PROJECT_TRACKER.md` holds the phased roadmap. When a phase (or a feature
  within a phase) is completed and merged, update its status there in the same
  PR or a prompt follow-up — not as an afterthought later.
- `README.md`'s architecture section and phase log are updated with a short
  summary whenever a phase ships, so the README always reflects current
  reality, not the original plan.

## 7. When Confused

If a requirement is ambiguous, missing, or contradicts something else in
`docs/REQUIREMENTS.md` — stop and ask the user. Do not fill the gap with an
assumption, even a reasonable-sounding one.
