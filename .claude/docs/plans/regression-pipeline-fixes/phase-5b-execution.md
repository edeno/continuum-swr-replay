# Phase 5b execution plan (task-structured view of phase-5b-readability-structure.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-5b-readability-structure.md`,
every contract in `shared-contracts.md` (preserved, not changed), and the review
`.claude/docs/reviews/2026-09-03-regression-pipeline-review.md` (findings D2, D3, D5, D8). Where
this file and those differ, they win. Anchors below were verified on 2026-09-06 BEFORE phase 5a
landed; re-grep after the 5a deletions (the `tables.py` shim, dead symbols, the `_x as _x`
re-exports, the clock wrappers) before editing.

## Global Constraints

- **Behavior-preserving, structure-only phase.** No fitter, spec, formula, persisted column, or
  value changes. Proof: the same parity gate as phase 5a — freeze one cached session before the
  first edit (`scripts/qa/freeze_regression_baseline.py --label pre5b`), and after the last edit
  `scripts/qa/compare_to_golden_baseline.py` byte-identical across all tables; plus one content
  smoke fit before/after identical to 1e-12 on every coefficient.
- Import layering (binding, tested): no module under `covariates/` imports `model_specs`;
  `data_loaders/` imports nothing from `analysis/`; `hippocampus/` imports nothing from `replay/`
  except `bouts` and `event_ids`.
- Every symbol deletion preceded by `grep -rn <symbol> src/ scripts/ tests/ docs/` recorded in the
  report; every move keeps the public name importable from the new module and the old import
  sites updated (no compatibility re-exports left behind except where the phase file says so:
  `model_specs.__all__` keeps re-exporting the design constants for scripts).
- `uv run pytest tests/unit -q` passes; `ruff check`/`ruff format --diff` clean on added lines;
  no plan/phase/task references; nothing under `Processed-Data/` committed; `paper/claims_table.md`
  untouched.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` then
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`. No `git checkout` /
  `stash` / `clean`; do not sweep in untracked user files.
- "Deliberately not in this phase" (binding): no persisted-name renames; no fitter/spec change; no
  deletion of `simulation.py` test-only symbols beyond phase 5a's list.

### Task 0: Parity baseline

`freeze_regression_baseline.py --label pre5b`; content smoke (`fit_replay_content_models.py
--max-per-animal 1 --models M0,M0a --output-dir <scratch>/pre5b/content`) recorded with commands,
paths, sha256, coefficient columns. No edits.

### Task 1: Design-constant module and the import-layering test

Files: new `src/continuum_swr_replay/analysis/replay/design_constants.py`;
`model_specs.py` (constants at ~92 `ARM_FE_PREDICTORS`, ~232-240 `RECENCY_SPLINE_*`, ~538-549
`REWARD_HISTORY_*`/`N_REWARD_LAGS`, ~643-671 `BIN_S`, `TMAX`, `SPLINE_KNOTS`, `TMAX_ARRIVAL`,
`ARRIVAL_SPLINE_KNOTS`, `ARRIVAL_CLOCK_COLUMN`, `N_HISTORY_LAGS`, `N_HISTORY_BASIS`;
`raised_cosine_history_basis` now lives in `_helpers.py` ~784 — decide its home per the phase
file: `design_constants.py` holds the constants and re-exports the basis function from its final
home, or the basis moves to `per_bin_design.py` in Task 2 — choose one, state it);
importers that must switch to `design_constants`: `covariates/definitions.py`,
`covariates/candidate_arm.py`, `covariates/history/visit_recency.py`,
`covariates/history/value_uncertainty.py`, `covariates/history/reward_history.py`,
`tables_stop.py` (six files currently import `model_specs`); `analysis/constants.py:51`
`OMISSION_KNOWN_LATENCY_SEC` → `data_loaders/constants.py` (so `data_loaders/trial_clocks.py` no
longer imports from `analysis`; update every importer — grep). New
`tests/unit/test_import_layering.py` (AST-based: the three rules in Global Constraints; must FAIL
before the moves and pass after — record the pre-move failure list in the report).

### Task 2: Split `_helpers.py`; one session key; one `is_stay`; event timing computed once

Files: `_helpers.py` (1287 lines today; function index: `nearest_well_distance_cm` ~61,
`immobility_bout_intervals` ~150, `time_since_stop` ~174, `single_arm_from_well` ~190,
`arm_to_patch` ~200, `align_posterior_to_dot` ~210, reducers ~237-296, `hash_event_id` ~297,
`validate_event_ids` ~314, `hash_stop_id` ~328, `assign_bout` ~360, `eq_indicator*` ~375/392,
`align_count_matrix` ~423, `assert_event_label_alignment` ~454, `assert_single_session_alignment`
~496, `add_arm_fixed_effects` ~546, `build_well_to_arm_map` ~585, `attach_is_swr_to_onsets` ~606,
`raised_cosine_history_basis` ~784, `reward_basis_from_lags` ~815, `reward_kernel_weights` ~847,
`reconstruct_reward_kernels` ~871, `reconstruct_self_history_kernels` ~894,
`_arrival_spanning_stop_ids` ~926, `aggregate_to_timebins` ~958, `POPULATIONS` ~1278) → new
`bouts.py` (`immobility_bout_intervals`, `time_since_stop`, `assign_bout`,
`nearest_well_distance_cm`), `event_ids.py` (`hash_event_id`, `hash_stop_id`,
`validate_event_ids`), `per_bin_design.py` (`aggregate_to_timebins`, `_arrival_spanning_stop_ids`,
`attach_is_swr_to_onsets`, `POPULATIONS`, the basis/kernel functions unless Task 1 put them in
`design_constants`); `_helpers.py` keeps the reducers, `eq_indicator*`, `align_count_matrix`,
`add_arm_fixed_effects`, `build_well_to_arm_map`, `single_arm_from_well`, `arm_to_patch`,
`align_posterior_to_dot`, and the alignment asserts; target < 400 lines. Update every importer
(grep `_helpers`; ~12 modules + tests). `hippocampus/event_table.py` imports from
`bouts`/`event_ids` only (it currently defines `_SESSION_COLS = list(SESSION_COLUMNS_DAY)` at ~78).
- One session key: replace the local definitions (`_helpers.py:33`, `table_contracts.py:22`,
  `choice_prediction.py:51`, `covariates/history/value_uncertainty.py:20`,
  `hippocampus/event_table.py:78`, `scripts/rate/fit_per_bin_rate_models.py:117`,
  `scripts/rate/fit_cross_history_rate.py:64`, `scripts/rate/plot_uncertainty_rate.py:61`, and any
  in `scripts/content/fit_replay_content_models.py`, `prepare_data.py`, `retrospective.py`,
  `timebin_tables.py`) with `session_keys.SESSION_COLUMNS` / `SESSION_COLUMNS_DAY` /
  `session_key_frame`; delete `_helpers._SESSION_COLS`. Test
  `test_session_keys.py::test_no_local_session_key_definitions` (grep-based over `src/ scripts/`).
- One `is_stay`: `stay_indicator(next_arm, current_arm) -> float` (NaN when either is NaN) in
  `_helpers.py`, used at `tables_stop.py:327`, `shuffles.py:325`, `tables_event.py` (find),
  `analysis/contfrag/tables.py` (two sites). Test `test_stay_indicator.py` (NaN propagation,
  equality semantics, ContFrag and Replay agree on a fixture).
- Event timing once: in `tables_event.py`, take `event_start_s`/`event_end_s`/`event_duration_s`
  from the classified table and delete `_merge_classified_and_verify_timing` (~467-566) — but FIRST
  run its bit-for-bit comparison on the cached session and record that it passes (the function's
  check is the proof the two derivations agree; keep one `assert_event_label_alignment` call).
  NB `label_event_*` columns are dropped after that check today (`tables_event.py` ~487-566); the
  registry alias stays (`covariate_registry.py` ~186-190).

### Task 3: Trim the registry; relocate CLIs; relocate validators

Files: `covariate_registry.py` (1535 lines; ledger `Entry` rows, derived views, the qvals
ingestion quartet, `stamp_covariate_schema`, `require_covariate_schema` ~1506,
`assert_unique_scientific_rows`, `finalize_covariate_schema`, `row_keys`, `_ROW_KEYS`,
`assert_canonical_vocabulary`) → keep the functions and only the `Entry` rows the qvals native →
canonical mapping needs; replace `assert_canonical_vocabulary`'s ledger-derived deny-list with a
fixed `LEGACY_COLUMN_NAMES` frozenset extracted once from the current ledger (record the
extraction command and the count); move the full ledger to
`docs/design/2026-08-21-covariate-naming-scheme.md` as an appendix table; target < 400 lines;
`tests/unit/test_covariate_registry.py` rewritten (qvals ingestion still canonicalizes the native
fixture; the deny-list rejects a legacy column). Production callers to keep working:
`qvals_table.py` (ingestion quartet), `require_covariate_schema` ×~42, `stamp_covariate_schema`
×~12, `row_keys` ×2.
- Relocate CLIs: `prepare_data.py` `_build_arg_parser`/`main` (~894+) →
  `scripts/decoding/prepare_replay_content_data.py` (the name `table_io.py` already points at —
  verify `~196`); `qvals_table.main` (~965+) → `scripts/behavior/build_qvals_table.py`; the modules
  keep library functions only; update `docs/README.md`, `scripts/README.md`, the docstrings citing
  the module invocation (`scripts/rate/fit_swr_contrast_rate.py` ~267, the audits that quote
  `prepare_data --from-cache` are historical records — leave them), and `docs/regression_pipeline_map.md`
  (regenerate with `scripts/qa/write_pipeline_map.py`).
- Relocate validators: `_validate_requested_standardization` (~245),
  `_validate_canonical_table_contract` (~289), `_validate_and_match_fit_results` (~1169) from
  `scripts/content/fit_replay_content_models.py` (2599 lines) into
  `analysis/replay/table_contracts.py` as public functions with identical behavior; the runner's
  science path (`_load_and_filter_cohort` ~2018 → `_fit_content_ladder` ~2224 →
  `fit_model_per_session`) should read in under 200 lines — report the line count; if it cannot
  reach 200 without a behavior change, report and stop.

### Task 4: Parity, suite, and line counts

`compare_to_golden_baseline.py` vs the Task-0 baseline (byte-identical); content smoke diff
(1e-12); `uv run pytest tests/unit -q` (report collected count); `wc -l` for `_helpers.py` (< 400)
and `covariate_registry.py` (< 400); `test_import_layering.py` passes; the pipeline map `--check`
passes. Write `docs/audits/<date>-readability-structure-parity.md`.

## Review

Before the final review, confirm: every task implemented as specified; "Deliberately not in this
phase" honored; the layering test fails on the pre-move tree (recorded) and passes after; the six
former session-key sites are gone; `_merge_classified_and_verify_timing`, the ledger, `_SESSION_COLS`,
and the in-`src` CLIs are gone; parity and the 1e-12 coefficient diff are committed as an audit;
no plan/phase references.
