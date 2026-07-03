# Overview — Scope, target structure, move-safety, risks

[← back to PLAN.md](PLAN.md)

## Target structure

The canonical end state every phase implements a slice of. **Rule: one home per
artifact type.**

```
docs/                                    # PUBLISHED reference (stable, rendered, human-discoverable)
  README.md                              # NEW — the scheme: what each doc TYPE is and where it lives
  design/                                # NEW — scientific/methodological design specs (moved OUT of .claude so collaborators/reviewers find them)
    <- docs/superpowers/specs/*.md       #   (moved in)
    <- .claude/research/content_model_design.md
  models_table.md                        #   catalog   (mirrors model_specs.py)
  covariate_dictionary.md                #   definitions
  behavior_model_covariates.md           #   definitions (qvals_table data dict)
  rate_models.md                         #   methods
  per_bin_rate_results.md                #   results (auto-generated)
  replay_rate_content_swr_synthesis.md   #   results (living note)
  task_description.md                    #   reference
  cached_trials_for_testing.md           #   reference
  known_issues/                          #   reference
.claude/docs/                            # AGENT/PROCESS artifacts (not scientific record)
  plans/                                 # implementation/execution plans
  reviews/                               # NEW: code/methods review outputs <- .claude/review/*.md
  (design specs do NOT live here — they go to docs/design/, above)
.claude/research/                        # EXPLORATORY ONLY: scratch notes + *_outputs/ figures
  fig2b_outputs/  peri_departure_outputs/  fig2b_nonswr_robustness.md  *fastplotlib*
scripts/
  README.md                              # NEW — index of entry points by domain
  decoding/ rate/ content/ behavior/ figures/ qa/ pfc/   # NEW domain subdirs
  exploratory/                           # NEW: <- all scripts/_*.py one-offs
  figures/archive/                       # <- scripts/archive/
src/continuum_swr_replay/analysis/replay/model_specs.py   # SINGLE source of truth for ALL model ladders
Processed-Data/replay_content/           # data outputs — dirs RENAMED to descriptive names (phase 5)
  content_arm/  content_patch/           #   content models, all events (was models/, models_patch/)
  content_arm_swr/  content_arm_nonswr/  #   is_swr split (T→swr, F→nonswr; was models_swr_{T,F})
  content_patch_swr/  content_patch_nonswr/   #   (was models_patch_swr_{T,F})
  rate_per_bin/  rate_per_stop/          #   rate ladders (was models_per_bin_rate/, models_per_stop_rate/)
  swr_contrast/                          #   NEW — un-mixes swr_contrast_{rate,content}.pkl out of models/
paper/figures/                           # PUBLICATION figures ONLY — subdirs, no loose files
  figure01/ … figure04/                  #   main-text panels
  supp/                                  #   supplements (supp_decoder_validation, qc_diagnosis, …)
  rate/ content/ reality/ quality/ …     #   analysis figures grouped by topic
.claude/research/<topic>_outputs/        # EXPLORATORY figures (fig2b_outputs, peri_departure_outputs, …)
```

Figure-output rule (phase 4): `paper/figures/` holds only publication figures in
**topic subdirs** — no loose top-level files, and **no exploratory one-off output**
(those go to `.claude/research/<topic>_outputs/`, same as the scripts that write them).

Doc-TYPE → home (stated in `docs/README.md`, phase 1):

| type | home |
|---|---|
| catalog / definitions / methods / results / reference | `docs/` (flat; README states each file's type) |
| design spec (scientific/methodological) | `docs/design/` (human-discoverable) |
| implementation/execution plan (agent process) | `.claude/docs/plans/` |
| review (code/methods output, process) | `.claude/docs/reviews/` |
| exploratory notes + figures | `.claude/research/` |
| **model spec (code)** | `model_specs.py` |
| results DATA (pkl/nc) | `Processed-Data/…` |

Note: `docs/` stays **flat** (only ~9 files) — the README types each file rather than
adding subdirectories (avoided churn for a small dir).

## Current codebase integration points

- `scripts/fit_swr_contrast_rate.py:79-169` — `_SPLINE`/`_REWK_PREV`/`_HISTCOS`, `_BLOCK_F`,
  `_BLOCK_COLS`, `STOP_COVARS`, `_g_specs()`, `_s_specs()`, `G_SPECS`/`S_SPECS`: the rate
  `is_swr`-contrast ladder — a **contiguous** block (`_SPLINE` @79 → `G_SPECS,S_SPECS` @169;
  `_fit` begins @170) that **moves to `model_specs.py`** (phase 2). The data-assembly/fit
  code (`_load`, `_fit`, `_fit_one_session`) stays.
- `scripts/fit_swr_contrast_content.py:57-200` — `ARMFE`, `BASEc`, `_TSC`, `ARM_LADDER`,
  `ARM_KEY_TERM`, `BASEc_PATCH`, `_LR_PATCH`, `PATCH_LADDER`, `PATCH_KEY_TERM`: the
  content ladder — **moves to `model_specs.py`** (phase 2). The `_assemble_*`
  column-builders stay.
- `src/continuum_swr_replay/analysis/replay/model_specs.py` — defines `MODEL_HIERARCHY`,
  `MODEL_PATCH`, `MODEL_PER_BIN_RATE`, `MODEL_PER_STOP_RATE`, etc.: **gains** the
  `SWR_CONTRAST_*` ladders (phase 2).
- `CLAUDE.md:64` — "Project Structure" section: rewritten to the target tree (refs
  updated per-phase; final holistic pass in phase 4).
- `docs/models_table.md` — the is_swr section's provenance note ("single source of
  truth is model_specs.py") becomes TRUE after phase 2.

## Move-safety analysis (drives phase 3)

Audited how scripts resolve paths, so the move doesn't silently break them:

- **61 scripts import `PROCESSED_DATA_DIR` from the package** → move-safe (package
  import resolves regardless of file location). No change needed.
- **10 scripts use `Path(__file__).resolve().parents[1]`** to reach repo-root / `src` /
  `.claude/research` / `paper` → **BREAK** when moved one level deeper (`parents[1]`
  then resolves to `scripts/`, not repo-root). Must become `parents[2]`. Files:
  `analyze_content_occupancy.py`, `analyze_event_spike_evidence.py`, `fit_cross_history_rate.py`,
  `plot_uncertainty_rate.py`, and the 5 `_fig2b_*` one-offs, `_peri_departure_rate.py`.
- **2 scripts load a sibling via `with_name("fit_uncertainty_rate.py")`**
  (`_audit_covariate_accuracy.py:52`, `plot_uncertainty_rate.py:45`) → break if
  separated from `fit_uncertainty_rate.py`. `plot_uncertainty_rate.py` co-locates in
  `rate/`; `_audit_covariate_accuracy.py` goes to `exploratory/`, so its `with_name`
  must be repointed to the new `rate/fit_uncertainty_rate.py` path.
- **~100 `scripts/<path>.(py|sh)` path mentions** across CLAUDE.md, README.md,
  `docs/*.md`, tests, notebooks (`.py` and paired `.ipynb`), scripts, pyproject, and
  historical plans. Use the broad pattern `scripts/[A-Za-z0-9_./-]+\.(py|sh)` so
  figure names with digits and nested paths like `scripts/archive/...` are included.
  Fix **active** refs (CLAUDE.md, README.md, `docs/`, `tests/`, `notebooks/`, the
  `.sh` runners, script docstrings/runtime literals, `scripts/README.md`, pyproject).
  **Do NOT** rewrite incidental old paths inside historical/completed `.claude/docs/plans/*`
  bodies — those are point-in-time records; note this exclusion in the phase.

### External runtime references (found by independent review — these BREAK, and the
above was a scripts-internal audit only)

- **`tests/` exec-load moved scripts at import time — 47 tests across 8 files.** These build
  `REPO_ROOT/"scripts"/…` and `spec_from_file_location`/`read_text().exec` the script at
  **collection** time, so a moved script is a pytest *collection error*, not a soft failure —
  phase 3's own `uv run pytest` gate goes red until fixed. The 8 files + the script each
  loads are enumerated in [phase-3](phase-3-scripts.md). `tests/` must be added to both the
  ref-fixing scope AND the straggler grep (which never covered it).
- **`scripts/generate_all_figures.py:36-48,77` subprocess-invokes its 4 sibling figure
  scripts** by hard-coded `"scripts/generate_figureNN.py"` string + `subprocess.run([...])`
  with no `cwd`. Moving them to `figures/` breaks the orchestrator at runtime. A `.py`
  orchestrator, not a doc mention — the doc-only ref sweep misses it. Phase 3 fixes the 4
  strings.
- **`pyproject.toml:98`** ruff per-file-ignore `"scripts/archive/**" = ["ALL"]` stops
  matching once `archive/` → `figures/archive/`, so ruff starts linting intentionally
  un-lint-clean archived scripts. Phase 3 updates the glob to `"scripts/figures/archive/**"`.
  (CI runs only pytest, so this doesn't fail CI — but breaks the documented `ruff check`.)
- **`notebooks/single_epoch_walkthrough.py` and its paired `.ipynb`** reference `scripts/`
  paths — add `notebooks/` and `--include=*.ipynb`/`--glob '*.ipynb'` to the straggler
  sweep (stale, non-fatal).
- **`scripts/check_data_availability.py:320`** writes the default CSV to
  `scripts/data_availability.csv`; after the move that would recreate a loose file in
  the scripts root. Make it relative to the moved script, e.g.
  `Path(__file__).with_name("data_availability.csv")`, so the output lands in `scripts/qa/`.
- **`scripts/fit_cross_history_rate.py:37`** is a `sys.path.insert(0, …parents[1]/"src")` —
  already in the `parents[1]→[2]` list, but flag it's a `sys.path` feed, not the repo-root
  constant idiom, so a mechanical fixer doesn't skip it.
- **Pre-existing rot (not caused by the move):** `src/…/palettes.py` and `posterior_io.py`
  docstrings cite `scripts/plot_replay_content_time_slice.py` / `prepare_replay_content_data.py`
  which **do not exist**. The phase-5 grep will surface these as noise — note them so they're
  not mistaken for move-breakage.

### git-mv caveat

The 13 **untracked** `_*.py` one-offs cannot be `git mv`'d (git-mv requires tracked files) —
plain `mv` those; `git mv` only the 15 tracked one-offs and the domain scripts.

## Scope and dependency policy

### Goals

- Every artifact type has exactly one documented home; the location is discoverable
  from `docs/README.md`, `scripts/README.md`, and CLAUDE.md.
- `model_specs.py` is the true single source of truth for **all** model ladders.
- `scripts/` root shows domain entry points, not a flat wall of ~80 files.
- No behavior change: every runner, test, and figure script works from its new path.

### Non-Goals

- No consolidation/merging of the runners themselves (7 rate/content entry points stay
  as-is, just relocated). Reducing runner count is a separate future effort.
- (Data-dir renaming IS now in scope — phase 5 renames the cryptic `models*/` dirs to
  descriptive names + updates all readers/writers. The capstone [phase 6] documents them.)
- No pruning of `_*.py` one-offs beyond moving them; delete only clearly-dead ones that a
  grep proves are unreferenced, and list each deletion for approval (don't delete silently).
- No content/analysis changes — this is organization only.

## Metrics

- `uv run pytest` green at every phase boundary (unchanged pass count).
- The rate + content `is_swr`-contrast smokes reproduce their committed outputs
  bit-identically after phase 2 (spec extraction is behavior-preserving).
- Broad script-ref sweep (`scripts/[A-Za-z0-9_./-]+\.(py|sh)`) over active docs,
  tests, notebooks, scripts, and pyproject has **zero stale paths** after phase 3
  (new subdir paths are allowed; old root-level/script-archive paths are not).
- Every moved script runs `--help` (or imports) from its new location without error.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Moving scripts breaks `parents[1]` repo-root resolution | Enumerated the 10 affected files (above); phase 3 converts each to `parents[2]` and verifies by running the script. |
| External refs break the move silently (tests exec-load scripts, `generate_all_figures` subprocess, ruff glob) | Independent review enumerated all three (above); phase 3 fixes them as explicit tasks and its pytest gate catches the test-ref class. |
| `with_name` sibling loads break | Enumerated the 2 (above); co-locate or repoint explicitly in phase 3. |
| Spec extraction (phase 2) silently changes a model | Phase 2 re-runs both smokes and asserts bit-identical parity vs the committed pkls before merge. |
| A running tmux job references an old script path | Phase 3 notes: check `tmux ls` for in-flight jobs before moving; none are load-bearing at plan time, but re-check at execution. |
| Historical plan files hold stale paths after the move | Explicitly excluded from ref-fixing (they are records); documented so the straggler grep's hits there are expected. |

## Rollout Strategy

All-at-once per phase (no feature flags — this is file organization). Each phase is a
self-contained PR that leaves the repo green. Recommended order 1→6; phases 1 and 2 are
mutually independent, phase 3 should follow phase 2 (so the thinned runners move), phase 4
(figure outputs) should follow phase 3 (figure scripts are relocated first), phase 5 (data-dir
rename) is independent, phase 6 is the capstone consistency pass (documents the final names).

## Open Questions

1. **`docs/` flat vs subdirs** — resolved: flat + typed README (dir is small). Revisit only
   if `docs/` grows past ~15 files.
2. **Delete vs keep dead one-offs** — deferred to phase 3 per-file: move by default, delete
   only with a grep proving zero references, each listed for approval.

## Estimated Effort

Mostly `git mv` + reference edits. Phase 1 ~11 file moves + 1 new README. Phase 2 ~250
LOC relocated (spec dicts) + import lines. Phase 3 ~80 script moves + ~15 idiom fixes + 8
test-ref fixes + `generate_all_figures`/`pyproject` fixes + ~30 doc-reference files + 1 new
README. Phase 4 ~27 figure-file moves + writer-target edits. Phase 5 CLAUDE.md rewrite +
1 convention note. No net new logic.
