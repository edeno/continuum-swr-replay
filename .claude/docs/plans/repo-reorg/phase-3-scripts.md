# Phase 3 — Group `scripts/` into domain subdirs + `exploratory/`, add `scripts/README.md`, fix references

[← back to PLAN.md](PLAN.md) · [overview](overview.md#move-safety-analysis-drives-phase-3)

The largest phase: ~80 script files move into domain subdirectories. The risk is
reference breakage; the [move-safety analysis](overview.md#move-safety-analysis-drives-phase-3)
enumerates exactly what breaks (10 `parents[1]`, 2 `with_name`, ~100 path mentions).

**Inputs to read first:**

- [overview.md](overview.md#move-safety-analysis-drives-phase-3) — the three break-classes
  and the exact files in each. This phase is mechanical *given* that analysis.
- `git ls-files scripts/` plus `find scripts -maxdepth 1 -type f -name '_*.py'` — together
  these are the authoritative current file list, including untracked one-offs that `git ls-files`
  omits.
- The 2 `with_name` sites: `scripts/_audit_covariate_accuracy.py:52`,
  `scripts/plot_uncertainty_rate.py:45`.

## Tasks

- **Create the subdirs** and move each script per the mapping below. `git mv` the tracked
  files (history follows); the **13 untracked `_*.py` one-offs must be plain `mv`** (git-mv
  requires tracked files). Edge cases the executor resolves by the rule "domain of the
  script's primary output," listed here as best categorization:

  | subdir | scripts |
  |---|---|
  | `decoding/` | `run_decoding_bandit.py`, `run_decoding_contfrag.py`, `regenerate_stop_table.py`, `build_timebin_tables.py`, `prepare_contfrag_replay_content_data.py` |
  | `rate/` | `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py`, `fit_swr_contrast_rate.py`, `fit_uncertainty_rate.py`, `fit_cross_history_rate.py`, `fit_swr_power_model.py`, `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_uncertainty_rate.py`, `plot_model_checking.py`, `summarize_per_bin_rate_models.py` |
  | `content/` | `fit_replay_content_models.py`, `fit_swr_contrast_content.py`, `compare_event_source_models.py`, `analyze_content_occupancy.py`, `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py` |
  | `behavior/` | `fetch_spatial_bandit_behavior.sh` |
  | `figures/` | `generate_figure01.py`…`04.py`, `generate_all_figures.py`, `generate_supp_decoder_validation.py`, `generate_supp_pipeline_data_flow.py`, `generate_supp_pipeline_schematic.py`, `generate_event_gallery.py`, `run_event_gallery.sh`; and `scripts/archive/` → `scripts/figures/archive/` |
  | `qa/` | `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`, `run_quality_metrics.py`, `analyze_event_spike_evidence.py` |
  | `pfc/` | `run_pfc_analysis_bandit.py`, `run_stage5b_retrospective.py`, `run_stage6_learning.py` |
  | `exploratory/` | every `scripts/_*.py` (15 tracked + 13 untracked) |

- **Fix the 10 `parents[1]` repo-root idioms → `parents[2]`** (moved one level deeper).
  Files listed in [overview.md](overview.md#move-safety-analysis-drives-phase-3). Prefer,
  where the target is `Processed-Data`, converting to the move-safe
  `from continuum_swr_replay.parameters import PROCESSED_DATA_DIR` instead of a relative
  climb; for `.claude/research` / `paper` targets, `parents[2]` is the minimal fix.
  **Note:** `fit_cross_history_rate.py:37` is `sys.path.insert(0, …parents[1]/"src")` (feeds
  `sys.path`, not a data path) — same `parents[2]` fix, but don't skip it as "not a repo-root
  constant."
- **Fix the 2 `with_name` sibling loads**: `plot_uncertainty_rate.py` co-locates with
  `fit_uncertainty_rate.py` in `rate/` → `with_name` still resolves, no change.
  `_audit_covariate_accuracy.py` (→ `exploratory/`) must repoint to
  `…/scripts/rate/fit_uncertainty_rate.py` (e.g. `Path(__file__).parents[1] / "rate" /
  "fit_uncertainty_rate.py"`; verify the parent depth from `exploratory/`).
- **Fix the 8 test files that exec-load moved scripts at import time** (else phase-3's own
  `pytest` gate is a *collection error*). Update the `REPO_ROOT/"scripts"/…` path constants:

  | test file:line | script → new subdir |
  |---|---|
  | `tests/unit/test_contfrag_paths.py:23,50,53` | `fit_replay_content_models.py`→`content/`, `prepare_contfrag_replay_content_data.py`→`decoding/` |
  | `tests/unit/test_compare_event_source_models.py:12` | `compare_event_source_models.py`→`content/` |
  | `tests/unit/test_fit_per_bin_rate_models.py:25` | `fit_per_bin_rate_models.py`→`rate/` |
  | `tests/unit/test_prep_zero_event_session_keeps_stop_rows.py:89` | `prepare_contfrag_replay_content_data.py`→`decoding/` |
  | `tests/unit/test_uncertainty_covariates.py:22` | `fit_uncertainty_rate.py`→`rate/` |
  | `tests/unit/test_augment_discover.py:8` | `_augment_data_over_time.py`→`exploratory/` |
  | `tests/unit/test_run_quality_metrics_empty_calib.py:28,57` | `run_quality_metrics.py`→`qa/` |
  | `tests/unit/test_statescript_contingency.py:19` | `_statescript_contingency.py`→`exploratory/` |

  (Grep `rg -n '"scripts"|scripts/' tests/` to confirm none are missed before moving.)
- **Fix `scripts/generate_all_figures.py:36-48`** — the hard-coded `"scripts/generate_figureNN.py"`
  strings its `subprocess.run` (`:77`) invokes → `"scripts/figures/generate_figureNN.py"` (all 4).
  This is a runtime `.py` orchestrator, not a doc mention.
- **Update `pyproject.toml:98`** ruff per-file-ignore `"scripts/archive/**"` →
  `"scripts/figures/archive/**"` (else ruff starts linting the intentionally un-clean archive).
- **Fix non-code path literals that would recreate loose files in `scripts/` root.** Known
  case: `scripts/check_data_availability.py:320` defaults to `scripts/data_availability.csv`;
  after moving the script to `qa/`, make the default `Path(__file__).with_name("data_availability.csv")`
  (or equivalent) so it writes under `scripts/qa/`.
- **Fix ~100 `scripts/<path>.(py|sh)` path mentions** in ACTIVE docs and script strings:
  `CLAUDE.md`, `README.md`, `docs/*.md`, `tests/`, the `.sh` runners (`run_event_gallery.sh`,
  `fetch_spatial_bandit_behavior.sh` — check they invoke sibling scripts by path), script
  docstrings/runtime literals, and both `notebooks/single_epoch_walkthrough.py` and the paired
  `notebooks/single_epoch_walkthrough.ipynb`. **Do NOT** rewrite paths inside
  `.claude/docs/plans/*` historical bodies (records). Drive from
  `rg -n 'scripts/[A-Za-z0-9_./-]+\.(py|sh)' CLAUDE.md README.md docs tests notebooks scripts pyproject.toml --glob '*.md' --glob '*.py' --glob '*.sh' --glob '*.toml' --glob '*.ipynb'`.
  This broader pattern intentionally catches digits (`generate_figure01.py`) and nested
  paths (`scripts/archive/...`).
- **Write `scripts/README.md`** — a table of entry points grouped by the subdirs above,
  one line each ("what it produces / when to run it"), so `scripts/` is navigable. Mark
  `exploratory/` as unsupported one-offs.
- **Update `CLAUDE.md`** "Project Structure" scripts portion to show the subdirs (full
  holistic pass is phase 5).
- **Dead one-offs:** for each `_*.py`, `grep -rn "<name>"` across the repo; if zero
  references and it's superseded, list it for deletion in the PR description and delete —
  do not delete silently. Default is move to `exploratory/`.

## Deliberately not in this phase

- Merging/consolidating runners (a non-goal). Each script only moves.
- Renaming `Processed-Data/…/models*/` output dirs (phase 5 documents; never renames).
- Figure-output organization (phase 4) and the final CLAUDE.md rewrite + straggler sweep (phase 5).

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest` | green — **only after** the 8 test path-constants are repointed (they exec-load scripts at collection time; unfixed = collection error, not soft fail) |
| `find scripts -path '*/__pycache__' -prune -o -name '*.py' -exec python -m py_compile {} +` | every moved script still parses, including untracked one-offs |
| `uv run python scripts/rate/fit_swr_contrast_rate.py --max-per-animal 1` | a `parents`-touching + package-importing runner works from its new path |
| `uv run python scripts/exploratory/_peri_departure_rate.py` (slow) | a moved `parents[1]→[2]` one-off writes to `.claude/research/peri_departure_outputs/` correctly |
| `rg -n 'scripts/(archive/|[A-Za-z0-9_-]+\.(py|sh))' CLAUDE.md README.md docs tests notebooks scripts pyproject.toml --glob '*.md' --glob '*.py' --glob '*.sh' --glob '*.toml' --glob '*.ipynb'` | zero old root-level script paths or old `scripts/archive/` paths in active files (new `scripts/<subdir>/...` paths are allowed) |
| manual: `scripts/README.md` | every subdir + entry point listed |

Mark the `_peri_departure_rate.py` run **slow** (loads the 593k-row arm table).

## Fixtures

Uses existing `Processed-Data/` tables for the runner smokes. No new fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every script is under a subdir; `scripts/` root holds only `README.md` and the subdirs.
- All 10 `parents[1]` fixed and verified by running at least one per target-type
  (Processed-Data, .claude/research, paper, src).
- Both `with_name` sites resolve from their new locations.
- Straggler grep over active files has no old root-level script paths; historical-plan
  exclusions noted.
- Any deleted one-off is listed in the PR body with its zero-reference evidence.
- `scripts/README.md` + CLAUDE.md scripts section reflect the new layout.
