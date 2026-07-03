# Phase 1 — Consolidate docs: design specs + reviews under `.claude/docs/`, add `docs/README.md`

[← back to PLAN.md](PLAN.md) · [overview](overview.md#target-structure)

Docs-only phase (no code, no test impact). Establishes the doc-type→home scheme and
moves the out-of-place doc classes (design specs, historical plans, reviews) into it.

**Inputs to read first:**

- [overview.md](overview.md#target-structure) — the target tree and the doc-TYPE→home table this phase realizes.
- `docs/superpowers/specs/` (4 files), `docs/superpowers/plans/` (1 historical plan),
  and `.claude/review/` (5 files) — the design specs, historical plan, and reviews
  being relocated.
- `.claude/research/content_model_design.md` — a design spec currently mis-filed in the exploratory tree.

## Tasks

- **Create `docs/design/`** (human-discoverable — scientific/methodological design must be
  findable in `docs/`, not a tool-specific hidden dir) and `git mv` into it:
  `docs/superpowers/specs/*.md` (4 files) and `.claude/research/content_model_design.md`.
- **Move the historical Superpowers plan into the plan home**:
  `git mv docs/superpowers/plans/2026-05-27-figure01-revision.md .claude/docs/plans/figure01-revision/PLAN.md`.
  This preserves it as a historical
  implementation plan while satisfying the "implementation plan" → `.claude/docs/plans/`
  rule. After moving both `specs/` and `plans/`, remove the now-empty
  `docs/superpowers/` tree.
- **Create `.claude/docs/reviews/`** and `git mv .claude/review/*.md` (5 files) into it.
  Remove the now-empty `.claude/review/`.
- **Fix cross-references to the moved files.** Grep the repo for links to the old paths
  and update them:
  `grep -rn "superpowers/specs\|superpowers/plans\|\.claude/review\|research/content_model_design" --include=*.md --include=*.py .`
  Known referrers: `.claude/docs/plans/covariate-harmonization/phase-4-promote-content-models.md`
  (links `../../review/04_...`), the synthesis note, and any plan STATUS banners. Update
  active docs. For the moved historical figure01 plan, update navigational links to the
  moved design spec/plan path, but leave old path mentions in task prose if they are
  clearly point-in-time instructions.
- **Write `docs/README.md`** — the scheme. One short table mapping each doc TYPE to its
  home (copy the table from [overview.md](overview.md#target-structure)), then a one-line
  purpose for each current `docs/*.md` file tagged by type (catalog / definitions /
  methods / results / reference). This is the artifact that stops the drift.
- **Update `CLAUDE.md`** "Project Structure" (`CLAUDE.md:64`) docs portion to point at
  `docs/README.md` and name the split: `docs/design/` (scientific design), `.claude/docs/plans`
  + `.claude/docs/reviews` (agent process), `.claude/research` (exploratory). (Scripts portion
  updated in phase 3; full pass in phase 6.)

## Deliberately not in this phase

- Scripts moves (phase 3) and model-spec extraction (phase 2) — this phase is docs only.
- Renaming or restructuring the `docs/*.md` files themselves — only `README.md` is added;
  existing files stay put (they're already in the right tree).

## Validation slice

| Test | Asserts |
| --- | --- |
| `grep -rn "superpowers/specs\|superpowers/plans\|\.claude/review/\|research/content_model_design" --include=*.md .` | no stale links remain in active docs (historical plan prose excepted, noted inline) |
| manual: open `docs/README.md` | every current `docs/*.md` file is listed with a type tag; the doc-type→home table is present |
| `ls docs/design/ .claude/docs/reviews/` | design specs (5) in `docs/design/`; reviews (5) in `.claude/docs/reviews/` |
| `ls docs/superpowers .claude/review` | both directories are gone (fully migrated) |

## Fixtures

None (documentation move).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- All 4 design specs + `content_model_design.md` are under `.claude/docs/designs/`; the
  historical figure01 plan is under `.claude/docs/plans/figure01-revision/PLAN.md`; all 5
  reviews are under `.claude/docs/reviews/`; old dirs removed.
- No stale links to the moved paths in active docs; historical-plan exceptions are called out.
- `docs/README.md` types every current `docs/*.md` file and carries the home table.
- CLAUDE.md docs section reflects the new `.claude/docs/{plans,designs,reviews}` split.
- Docstrings / module names don't reference this plan.
