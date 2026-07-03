# Phase 6 — Document the results/figure output conventions; final CLAUDE.md rewrite + straggler sweep

[← back to PLAN.md](PLAN.md) · [overview](overview.md#target-structure)

Capstone. Closes the "scattered outputs" problem by *documenting* where result and figure
artifacts live (the descriptive names phase 5 produced), then does the single holistic CLAUDE.md "Project
Structure" rewrite and a repo-wide sweep for references the per-phase fixes missed.

**Inputs to read first:**

- [overview.md](overview.md#target-structure) — the finished tree that CLAUDE.md must now describe.
- `Processed-Data/replay_content/` — the descriptive output dirs from phase 5 (`content_arm`, `content_patch`,
  `content_arm_{swr,nonswr}`, `content_patch_{swr,nonswr}`, `rate_per_bin`, `rate_per_stop`)
  plus the `swr_contrast/` dir (phase 5 un-mixed those pkls out of the old `models/`).
- [phase-4-figures.md](phase-4-figures.md) — the figure-output convention this capstone folds
  into `docs/README.md` and CLAUDE.md.
- `docs/README.md` (from phase 1) — gains the results + figure output subsections.
- `docs/per_bin_rate_results.md`, `docs/replay_rate_content_swr_synthesis.md` — the results
  writeups whose data sources this convention names.

## Tasks

- **Add a "Where outputs live" section to `docs/README.md`** covering all three output kinds:
  (1) result **data** = `Processed-Data/replay_content/<descriptive>/*.pkl|.nc` (list the dirs +
  which runner writes each); (2) result **docs** — auto-generated (`per_bin_rate_results.md`,
  from `summarize_per_bin_rate_models.py`) vs narrative (`replay_rate_content_swr_synthesis.md`,
  hand-maintained); (3) **figures** — `paper/figures/<topic>/` = publication,
  `.claude/research/*_outputs/` = exploratory (the phase-4 convention). State that
  `.claude/research/*_outputs/` is never results-of-record.
- **Rewrite `CLAUDE.md` "Project Structure"** (`CLAUDE.md:64-90`) to the final tree from
  [overview.md](overview.md#target-structure): the `scripts/` subdirs, the
  `.claude/docs/{plans,designs,reviews}` + `.claude/research` split, `docs/` typed by README,
  the `model_specs.py` single-source-of-truth note, and the `paper/figures/` topic-subdir
  convention. Point readers at `docs/README.md` and `scripts/README.md` rather than
  duplicating them.
- **Repo-wide straggler sweep.** Run and resolve (note the `tests/`, paired notebook
  `.ipynb`, script docstrings/runtime literals, and `pyproject` scope — earlier phases fix
  them, this confirms):
  `rg -n 'scripts/[A-Za-z0-9_./-]+\.(py|sh)|superpowers/(specs|plans)|\.claude/review/|research/content_model_design' CLAUDE.md README.md docs src scripts tests notebooks pyproject.toml --glob '*.md' --glob '*.py' --glob '*.sh' --glob '*.toml' --glob '*.ipynb'`
  Every hit outside historical `.claude/docs/plans/*` bodies must either point to an
  existing current path or be documented as a historical/pre-existing exception. The broad
  script-path regex intentionally catches digits (`generate_figure01.py`) and nested paths
  (`scripts/figures/archive/...`). **Expected noise (pre-existing rot, NOT move-breakage):**
  `src/…/palettes.py` and `posterior_io.py` docstrings cite
  `scripts/plot_replay_content_time_slice.py` / `prepare_replay_content_data.py`, neither of
  which exists today — leave with an inline note, or fix separately.
- **Verify the phase-5 rename is fully reflected**: no runner/reader/doc still writes or reads an old `models*/` name (grep confirms phase 5 was complete).

## Deliberately not in this phase

- Renaming data dirs (that was phase 5) — this capstone only documents the new names + sweeps.
- Figure-output moves (phase 4) and any script/doc moves (phases 1–3) — this phase only
  documents and sweeps.

## Validation slice

| Test | Asserts |
| --- | --- |
| straggler `rg` (command above) | zero stale refs outside historical plan bodies (pre-existing `src/` rot excepted, noted) |
| manual: `CLAUDE.md` "Project Structure" | matches the [overview target tree](overview.md#target-structure); references `docs/README.md` + `scripts/README.md` |
| manual: `docs/README.md` "Where outputs live" | all phase-5 descriptive dirs named with producer; data/auto-doc/narrative/figure tiers distinguished |
| `uv run pytest` | green (documentation-only phase) |

## Fixtures

None (documentation + grep).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- CLAUDE.md "Project Structure" is the single accurate description of the final tree and
  defers detail to the two READMEs (no duplication).
- The outputs convention covers all three kinds (data / docs / figures) and separates
  publication from exploratory.
- The straggler sweep is clean; left-in historical/pre-existing references are justified.
- No old `models*/` name survives anywhere (phase-5 rename fully swept).
- Docstrings / module names don't reference this plan.
