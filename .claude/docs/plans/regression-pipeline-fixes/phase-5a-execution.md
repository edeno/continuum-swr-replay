# Phase 5a execution plan (task-structured view of phase-5a-readability-hygiene.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-5a-readability-hygiene.md`,
`shared-contracts.md` §"Standardization entry point" (enforced here), and the review
`.claude/docs/reviews/2026-09-03-regression-pipeline-review.md` (findings D1, D4, D5, D7, D8).
Where this file and those differ, they win. Line numbers in the phase file predate phases 1–4;
the anchors below were re-verified on 2026-09-06 and are still approximate — grep before editing.

## Global Constraints

- **Behavior-preserving phase.** No fitted output, no persisted table value, no persisted name
  changes. The parity gate is the proof: freeze one cached session before the first edit
  (`scripts/qa/freeze_regression_baseline.py`), and after the last edit
  `scripts/qa/compare_to_golden_baseline.py` must report byte-identical tables; additionally one
  content smoke fit (`scripts/content/fit_replay_content_models.py --max-per-animal 1 --models M0,M0a`
  to a scratch `--output-dir`) before and after must agree to 1e-12 on every coefficient.
- Run Python with `uv run …`; `ruff check`/`ruff format --diff` clean on added lines.
- Docstrings, test names, module names, commit messages: no plan/phase/task references.
- Tests value-level; `slow` marks over 20 s.
- Every deletion is preceded by `grep -rn <symbol> src/ scripts/ tests/ docs/` recorded in the
  report (zero hits outside the definition, or the hits are the callers you rewire).
- Nothing under `Processed-Data/`, `.superpowers/`, `paper/figures/**/*.png` committed;
  `paper/claims_table.md` untouched.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` then
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`. No `git checkout` /
  `stash` / `clean`; do not sweep in untracked user files.
- "Deliberately not in this phase" (binding): no persisted column/value renames (`in_transit`,
  `is_prev_arm`, `dot`, `tc`); no structural moves (`_helpers` split, registry trim, CLI
  relocation — phase 5b); no change to any fitter or spec formula.

### Task 0: Parity baseline

Run `scripts/qa/freeze_regression_baseline.py` for one cached session (the one it already uses)
into a fresh `_golden_baseline_<date>_pre5a` under `Processed-Data/replay_content/` (check the
script's flags), and the content smoke fit to a scratch dir; record paths and hashes in the
report. This task edits nothing.

### Task 1: Narrative out of `model_specs.py`; delete the `tables.py` shim; dead symbols

Files: `src/continuum_swr_replay/analysis/replay/model_specs.py` (1535 lines today; narrative
blocks at ~127-186 [content M-ladder history], ~200-212, ~383-400 [patch renumbering], ~469-480,
~677 [switch retirement], ~866 [M8 retirement], ~1062-1095 [rate contrast ladder preamble],
~1198-1270 [content contrast preamble], ~1416-1430 [CP2/CPS3 retirement]), `docs/design/model_history.md`
(sections at 17, 105, 153 — add dated headings), `src/continuum_swr_replay/analysis/replay/tables.py`
(the shim; 27 import sites across `src/ scripts/ tests/` — `grep -rn "replay.tables\|from .tables import"`),
`_helpers.py:136` (`immobility_bout_starts`), `retrospective.py:79` (`_PATCH_TO_ARM_SET`),
`simulation.py:101-103` (`_BASE_BINARY`, `_CONTINUOUS_ARM`, `_EVENT_LEVEL`),
`analysis/behavior/_spatial_bandit_rl.py:18, 236` (`set_nan_at_max2` — read line 236 first: if the
apply is a no-op delete both, else keep both and say why).

- Move each narrative block verbatim into `model_history.md` under a dated heading; leave a
  one-line pointer per rung in code. Fix the last-rewarded-family contradiction (~152-159) to the
  current state (family refused as unidentified; `CS5b` n = 132 with 94 separated). Target
  `model_specs.py` < 900 lines (report the count; if the ladders alone exceed 900, report the
  number and stop trimming — do not move executable content).
- Delete the shim: point every importer at `tables_event` / `tables_stop` / `tables_arm_patch` /
  `_tables_common`; tests import column tuples from `_tables_common`.
- Delete the dead symbols after the grep evidence.

### Task 2: Test-only surfaces, `__all__` trim, version check, one z-score path, drop count

Files: `covariates/history/visit_recency.py:287` (`visit_recency_per_arm`),
`covariates/history/reward_history.py:334, 348` (`rolling_reward_count_per_arm`,
`recent_reward_rate_per_arm`), `covariates/context.py:364` (`behavioral_location_at_times`),
`covariates/clocks.py:13, 36, 141, 150` (`_trial_arrival_times`, `_trial_outcome_times` and the
public wrappers `trial_cutoff_times`, `trial_outcome_times` — rename the private pair to public,
delete the wrappers, update the callers: `visualization/session_tables.py`,
`scripts/figures/generate_event_gallery.py`, `scripts/exploratory/_audit_covariate_accuracy.py`,
`scripts/exploratory/_lagged_reward_kernel.py`, `covariates/context.py`, and any test),
`covariates/__init__.py` and `covariates/history/__init__.py` (13 `_x as _x` re-exports),
`analysis/replay/__init__.py` (233 lines; trim `__all__` to names imported through the package —
measure with a grep over `src/ scripts/ tests/`; keep the star-import test passing; keep the lazy
DM exports), `analysis/replay/versions.py` (`assert_versions_compatible`),
`covariate_registry.py:1506` (`require_covariate_schema`), `prepare_data.py:212`
(`load_behavioral_context`), `scripts/rate/fit_uncertainty_rate.py:640-645` (inline z),
`scripts/rate/fit_swr_power_model.py:274` (inline z of the design matrix),
`analysis/hippocampus/event_immobility.py:16-18, 203` (`filter_event_intervals_by_speed`).
Tests: `test_history_public_api.py`, `test_covariates.py`, `test_contfrag_arm_fraction_denominator.py`,
`tests/benchmarks/bench_per_arm_helpers.py`, new `test_versions.py::test_require_schema_rejects_old_version`,
`test_standardization.py::test_runner_references_written`.

- Remove the per-arm scalar views, `behavioral_location_at_times`, the clock wrappers, and the
  `_x as _x` re-exports; rewrite the affected tests against the vectorized functions (the
  benchmark file too).
- Wire `assert_versions_compatible(covariate_schema_version=…)` into `require_covariate_schema`
  and `assert_versions_compatible(behavioral_context_cache_version=…)` into
  `load_behavioral_context` when reading a cache; test that an old-version stamp raises through
  `require_covariate_schema`.
- One z-score path: replace the two inline z-scorings with
  `standardization.attach_standardized(..., support_label="fit_cohort_all_rows")` and persist the
  reference JSON next to each output (`<artifact-stem>.z_reference.json`); test that the
  uncertainty-rate smoke writes it with `support == "fit_cohort_all_rows"`. Ruling: the power
  model's design-matrix standardization becomes the same helper applied column-wise; its fitted
  OLS coefficients must be identical to 1e-12 before/after (standardization is linear) — assert in
  the report with a smoke.
- `filter_event_intervals_by_speed` returns (or logs at INFO through the caller in
  `hippocampus/event_table.py`) the number of intervals removed; delete the "Filtering is silent"
  caveat.

### Task 3: Parity and suite

Run `scripts/qa/compare_to_golden_baseline.py` against the Task-0 baseline (byte-identical across
all tables), the content smoke diff (1e-12), the uncertainty-rate and power-model smokes
(coefficients identical to 1e-12 vs a pre-edit smoke run in Task 0 — add those two smokes to
Task 0), and the full `uv run pytest tests/unit -q` (report the collected count). Write
`docs/audits/<date>-readability-hygiene-parity.md` with the evidence. Report `wc -l model_specs.py`.

## Review

Before the final review, confirm: every symbol listed for deletion is gone (grep evidence in the
reports); nothing scientific moved (formulas, ladders, fitters untouched — diff `model_specs.py`
executable content before/after by evaluating each ladder dict and comparing `repr`); the parity
report is committed; no persisted names changed; no plan/phase references; `__all__` trim did not
break any script (run `python -c "import <each script module>"` or the scripts' `--help`).
