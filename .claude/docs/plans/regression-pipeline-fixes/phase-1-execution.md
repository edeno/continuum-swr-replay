# Phase 1 execution plan (task-structured view of phase-1-rate-runners.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-1-rate-runners.md`,
`shared-contracts.md`, `designs.md`, and the review
`.claude/docs/reviews/2026-09-03-regression-pipeline-review.md`. This file regroups the phase's
tasks into dependency-ordered dispatch units; where it and the phase file differ, the phase file
and contracts win.

## Global Constraints

- Run Python with `uv run …` from the repo root. Never install into the base environment.
- Lint/format: `uv run ruff check <paths>` and `uv run ruff format <paths>` on every file touched.
- Docstrings, test names, module names, and commit messages must not mention this plan, phase
  numbers, task numbers, or `.claude/docs/plans/`.
- Tests are value-level: assert numbers and behaviour, not column presence or "does not raise".
  No test-only methods in production code. Shared setup in fixtures.
- Fail loud: contracts are enforced with `raise`, not silent fallbacks or warnings.
- Canonical tables are parquet under `Processed-Data/replay_content/`; read through
  `continuum_swr_replay.table_io.read_table`. Never modify a canonical artifact without first
  copying it to a dated `_pre_fix_backup_<YYYYMMDD>/` directory. Never delete data.
- Every persisted canonical column has an entry in
  `analysis/replay/covariates/definitions.py` and appears in the `_tables_common.py` column tuples.
- Commit after each coherent unit with a message ending in
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`.
- Long runs (> 5 min): smoke first (`--max-per-animal 1`), record wall time and peak RSS with
  `/usr/bin/time -v`, extrapolate, then run the full job inside `tmux` and report the session name.
- The reward-history kernel contract (verbatim from shared-contracts.md):
  - `B = raised_cosine_history_basis(n_lags=10, n_basis=3)` (log-spaced bumps; supports with
    weight > 0.05: bump 0 lags 1–7 with weights 1.00, 0.79, 0.54, …; bump 1 lags 1–10; bump 2 lags 2–10).
  - `lags[t, j-1]` = reward (0/1) of the j-th most recent outcome known at query time `t`, NaN
    when fewer than `j` outcomes are known (`lagged_trial_rewards`).
  - `reward_history_rcos_k[t] = Σ_j B[k, j-1]·lags[t, j-1] / Σ_j B[k, j-1]` over finite lags only;
    undefined (no finite lag in the bump's support) → 0.0 with `reward_history_incomplete = 1`.
  - `reward_history_incomplete` is 1 for stops with fewer than two known outcomes, else 0. It is a
    structural-missingness nuisance main effect: present in every rate rung that carries the
    basis, never inside an `is_swr *` expansion, never aggregated or reported.
  - Features are weighted reward rates in [0, 1]; coefficients per unit, not per SD.
  - Feature 0 is the immediate component; `is_swr:reward_history_rcos_0` and
    `reward_history_rcos_0:bs(tc, …)` replace the former lag-1 interactions.
  - `known_reward_outcome_lag_1..10` and `reward_fraction_last10_known_trials` stay persisted as
    stop metadata; they are never ladder predictors.
- The common-cohort contract (verbatim from shared-contracts.md): `freeze_per_bin_cohort(design, specs, *, extra_columns=())`
  keeps only bins complete for the union of every spec's design columns and with `exposure > 0`,
  returns `(frozen, report)` with
  `{"n_bins_before","n_bins_after","n_stops_before","n_stops_after","n_event_bins_before","n_event_bins_after","required_columns"}`
  and sets `attrs["per_bin_cohort_frozen"]`. `fit_per_bin_rate_model` and
  `cross_validate_per_bin_rate_model` raise `ValueError` on any NaN in a needed column or
  `exposure <= 0`; they never subset. Every rung of one (session, population) design reports equal
  `n_bins`/`n_stops`; `compare_nested_models` NaNs comparisons when `n_events`, `n_bins`, or
  `n_stops` differ. The per-stop family uses `freeze_common_cohort` once per session over the union
  of `MODEL_PER_STOP_RATE` predictors with `offset_col="immobility_duration_s"`, and
  `fit_per_stop_rate_model` raises on NaN. Cohort audits attribute union drops to
  `n_drop_common_cohort_other` when the row is complete for the model's own predictors and to
  `n_drop_own_predictor_missing` otherwise.
- The provenance contract (verbatim from shared-contracts.md): every runner that writes a fitted
  artifact writes `<artifact-stem>.provenance.json` beside it with `artifact`, `written_at`,
  `git_hash`, `estimator`, `inputs` (`{name: {sha256, mtime, covariate_schema_version}}`), `cli`.
  `analysis/replay/fit_provenance.py` exposes
  `write_fit_provenance(artifact_path, inputs: dict[str, Path], estimator: str, cli: list[str])`
  and `check_fit_provenance(artifact_path, inputs) -> list[str]` (names of inputs whose current
  SHA-256 differs). Readers raise unless `--allow-stale`.

### Task 1: Reward-history basis features and stop-table columns

Read first: `designs.md` §"Reward-history basis features" (complete code for
`reward_basis_from_lags` and `reward_history_basis_features`), and the kernel contract in Global
Constraints.

Files: `src/continuum_swr_replay/analysis/replay/covariates/history/reward_history.py`
(`lagged_trial_rewards` at 231-276), `src/continuum_swr_replay/analysis/replay/_helpers.py:752-781`
(`raised_cosine_history_basis`), `src/continuum_swr_replay/analysis/replay/model_specs.py:597`
(`N_REWARD_LAGS`), `src/continuum_swr_replay/analysis/replay/tables_stop.py:136-147`,
`src/continuum_swr_replay/analysis/replay/_tables_common.py:206-247`,
`src/continuum_swr_replay/analysis/replay/covariates/definitions.py:729-743`,
`src/continuum_swr_replay/analysis/replay/covariates/history/__init__.py` and
`covariates/__init__.py` (re-exports).

- Implement `reward_basis_from_lags` and `reward_history_basis_features` exactly as in
  `designs.md`; the latter returns `(features, incomplete)`. Import `raised_cosine_history_basis`
  from `_helpers` (that import path already exists in the module).
- Add to `model_specs.py` beside `N_REWARD_LAGS`: `REWARD_HISTORY_N_LAGS = 10`,
  `REWARD_HISTORY_N_BASIS = 3`, `REWARD_HISTORY_COLUMNS = ("reward_history_rcos_0", "reward_history_rcos_1", "reward_history_rcos_2")`,
  `REWARD_HISTORY_INCOMPLETE = "reward_history_incomplete"`; add them to `__all__`.
- In `build_stop_table` after the lag columns (line 147) compute the features at `bout_starts`
  and write the four columns. Append them to `_STOP_TABLE_BUILD_COLUMNS` and `_STOP_TABLE_COLUMNS`
  immediately after `known_reward_outcome_lag_10`.
- Catalog entries in `definitions.py` for the three features (grain `immobility stop`, units
  "weighted reward fraction", clock STOP_START, PRE_OUTCOME, support "all stops", window
  "log-spaced raised-cosine bump over the last 10 known outcomes, normalized over the available
  lags", missingness "none; undefined bumps are zero-filled and flagged by reward_history_incomplete",
  producer `covariates.history.reward_history_basis_features`) and for the indicator (units
  "binary indicator", missingness "none", description stating it is a structural-missingness
  nuisance term for stops with fewer than two known outcomes).
- Re-export the two functions from `covariates/history/__init__.py` and `covariates/__init__.py`
  (`__all__` included).
- Tests in `tests/unit/test_covariates.py` using the hand-built experience streams already in
  `tests/unit/_port_experience_fixtures.py`: (i) with ten known outcomes each feature equals the
  bump-weighted mean of the lag vector and `incomplete == 0`; (ii) with one known outcome
  features 0 and 1 equal that outcome, feature 2 is 0.0, `incomplete == 1`; (iii) with two known
  outcomes feature 2 equals the second outcome and `incomplete == 0`; (iv) a query before any
  outcome gives all zeros and `incomplete == 1`; (v) values are in [0, 1]; (vi) the outcome clock
  is respected (a query inside the arrival→outcome gap does not see the pending outcome);
  (vii) `reward_basis_from_lags` on a lag row with NaNs returns NaN only for bumps whose support is
  entirely NaN. Test in `tests/unit/test_replay_tables.py`: the stop table carries the four columns
  and they equal the function evaluated at `bout_starts` for the synthetic session.
- Do not regenerate any table in this task.

### Task 2: Regenerate the stop table with the new columns

Files: `scripts/decoding/regenerate_stop_table.py`, `src/continuum_swr_replay/analysis/replay/prepare_data.py`
(`build_stop_table_scope` at 607, `run_prepare_data` at 959), `scripts/qa/compare_to_golden_baseline.py`,
`scripts/qa/freeze_regression_baseline.py`, `Processed-Data/replay_content/stop_table.parquet`.

- Back up `Processed-Data/replay_content/stop_table.parquet` to
  `Processed-Data/replay_content/_pre_fix_backup_<YYYYMMDD>/stop_table.parquet` (copy, not move).
- Determine the cheapest regeneration path that does not need a database connection: prefer
  `scripts/decoding/regenerate_stop_table.py` if it can run from the behavioral-context cache;
  otherwise `prepare_data --from-cache` into a staging directory (`--output-dir`), keeping only the
  stop table from staging. Smoke with `--max-per-animal 1` first and record wall time / peak RSS.
- Run the full 218-session regeneration in `tmux` (session name `stop_table_regen`).
- Parity: every pre-existing column of the new stop table must be byte-identical to the backup
  (same rows, same `stop_id` set and order, same dtypes, same values including NaN positions);
  the only differences are the four added columns. Write the comparison as
  `docs/audits/<YYYY-MM-DD>-stop-table-reward-basis-regen.md` with row counts, the SHA-256 of old
  and new files, and the per-column parity result. Also report: share of stops and events with
  `reward_history_incomplete == 1` overall and per animal (expected ≈ 1.3 % of stops, 1.6 % of
  events; the affected stops must be an unbroken prefix of each epoch — assert it).
- If `event_arm_table.parquet` or any other table would be rebuilt by the chosen path, it must
  come out byte-identical to the live file; do not promote any other table.
- Promote the new stop table into place only after parity passes.

### Task 3: Respecify the rate ladders on the basis kernel

Files: `src/continuum_swr_replay/analysis/replay/model_specs.py:551-608` (`MODEL_PER_STOP_RATE`),
`:631-650`, `:652-760` (`PerBinRateModelSpec`, `formula`), `:766-909` (`MODEL_PER_BIN_RATE`),
`:995-1034` (`_REWK_PREV`, `_BLOCK_F`, `_BLOCK_COLS`, `SWR_CONTRAST_RATE_STOP_COVARS`),
`:1060-1087` (`_s_specs`); `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:60-82`
(`_design_columns`), `:141-147` (the lag-1 contrast guard); `scripts/rate/fit_uncertainty_rate.py:140-165`
(`MU_UNCERTAINTY`); `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:60-75`
(`BINARY_TERMS`); `docs/models_table.md:113-153` (R notation for Family A/B and `BASE`).

- `PerBinRateModelSpec`: replace `reward_lags: int` with `reward_basis: int` (0 or
  `REWARD_HISTORY_N_BASIS`); `formula()` emits `reward_history_rcos_0 + reward_history_rcos_1 + reward_history_rcos_2 + reward_history_incomplete`
  when `reward_basis > 0`; `reward_time_interaction` emits `reward_history_rcos_0:{_SPLINE}`;
  `_design_columns` follows. The "fewer than two reward levels" guard in `fit_per_bin_rate_model`
  becomes "reward_history_rcos_0 has fewer than two distinct values".
- `MODEL_PER_BIN_RATE`: `M1_rew1` and `M2_rewhist` collapse into `M1_rew` (basis, parent
  `M0_const`); delete `M2r_rewrate`; `M4_hist_rew` (parent `M3_hist`), `M7_time_hist_rew`
  (parent `M6_time_hist`), `M7r_time_histcos_rew` (parent `M5_time`), `M9_rew_x_time`
  (parent `M7_time_hist_rew`) use the basis. Keep every other rung unchanged.
- `MODEL_PER_STOP_RATE`: `R0` unchanged; `R1` carries the three features plus the indicator;
  delete `R2` and `Rlag`.
- SWR-rate contrast: `_REWBASIS = " + ".join(REWARD_HISTORY_COLUMNS)`;
  `_BLOCK_F["BASE"] = f"{_SPLINE} + {_REWBASIS} + {_HISTCOS}"`; in `_g_specs` append
  ` + reward_history_incomplete` to every formula; in `_s_specs` every formula is
  `f"is_swr * ({...}) + reward_history_incomplete"` so the indicator never enters the
  interaction; `S1_rew1` becomes `S1_rew` with `is_swr:reward_history_rcos_0`; `REWXTIME` becomes
  `reward_history_rcos_0:{_SPLINE}`; `_BLOCK_COLS["BASE"]` lists `tc`, the three features, the
  indicator, and the four `hist_rcos_*`; `_BLOCK_COLS["REWXTIME"] = ["reward_history_rcos_0", "tc"]`.
  Delete `_REWK_PREV`. `STOP_COVARIATES` becomes `()`.
- `fit_uncertainty_rate.MU_UNCERTAINTY`: `reward_basis=REWARD_HISTORY_N_BASIS` (positional
  construction there must be updated to keywords); the comment about `reward_fraction_last10_known_trials`
  collinearity is deleted with the rung that needed it.
- `contrast_aggregation.BINARY_TERMS`: remove `known_reward_outcome_lag_1` and
  `is_swr:known_reward_outcome_lag_1`. `aggregate_contrast` and `specs_from_ladder` never emit a
  row for a term named `reward_history_incomplete`.
- `N_REWARD_LAGS` survives only as the basis span (equal to `REWARD_HISTORY_N_LAGS`); delete every
  formula string or column list built from `reward_lags` that no longer has a consumer.
- Tests: `tests/unit/test_model_specs.py` (create if absent): no shipped spec references
  `known_reward_outcome_lag_*` or `reward_fraction_last10_known_trials`; every `MODEL_PER_BIN_RATE`
  formula's columns resolve through `_design_columns`; `S` formulas contain
  `+ reward_history_incomplete` outside the parenthesized interaction; `G`/`S` `need` lists
  include the indicator. Update `tests/unit/test_per_bin_rate_model.py`,
  `test_per_stop_rate_model.py`, `test_fit_per_bin_rate_models.py`, `test_per_bin_rate_helpers.py`,
  `test_rate_cohort_audits.py` fixtures to carry the four basis columns; a planted effect on
  `reward_history_rcos_0` (rate 12 vs 6 Hz) must be recovered by `M1_rew` with
  `rr_reward_history_rcos_0` in [1.5, 2.7] in `test_fit_per_bin_rate_models.py::test_tidy_schema`.
- `docs/models_table.md:113-153`: R notation for the new Family A/B rungs and `BASE`; define
  `REWBASIS` in "Conventions".

### Task 4: One cohort per ladder in both rate families

Read first: the common-cohort contract in Global Constraints.

Files: `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:84-193, 195-306`,
`per_stop_rate_model.py:63-173`, `cohort_selection.py:33-60`, `model_comparison.py:20-98`,
`scripts/rate/fit_per_bin_rate_models.py:108-168, 237-283`, `scripts/rate/fit_per_stop_rate_models.py:69-156`,
`scripts/rate/fit_swr_contrast_rate.py:152-197, 200-241`, `scripts/rate/fit_uncertainty_rate.py`
(design assembly), `tests/unit/test_replay_tables_audit_regressions.py:299-330`.

- Implement `freeze_per_bin_cohort` in `per_bin_rate_model.py`. Replace the `dropna` at 135-136
  and 235-236 with the raising check. In `fit_per_bin_rate_models._fit_one_session` freeze right
  after `aggregate_to_timebins`; assert the freeze removed zero stops for the shipped
  `MODEL_PER_BIN_RATE` (raise with the offending columns otherwise); carry the report into
  `_cohort_records` so `n_drop_common_cohort_other` is emitted. In
  `fit_swr_contrast_rate._fit_one_session` freeze each stratum design over the union of `G_SPECS`
  columns and the stacked design over the union of `S_SPECS` columns plus `is_swr`; delete the
  `dropna` in `_fit` (167) and raise instead; the qvals-dependent rungs' drops are reported as
  `n_drop_qvals_gap`. `fit_uncertainty_rate.py` freezes over `MU_UNCERTAINTY`'s columns.
- Per-stop: in `fit_per_stop_rate_hierarchy` before the outcome/model loops call
  `freeze_common_cohort(group, unit_col="stop_id", required_cols=[*all_predictors, *RESPONSE_COLS.values()], offset_col="immobility_duration_s")`;
  replace `dropna` at `per_stop_rate_model.py:103` with a raise; rewrite the accounting at 98-115
  per the contract.
- `compare_nested_models`: include `n_bins`/`n_stops` in `fit_cols` when present and NaN the
  comparison columns when any of `n_events`/`n_bins`/`n_stops` differ.
- Tests: `test_per_bin_rate_model.py::test_freeze_per_bin_cohort_makes_rungs_share_rows` (all
  specs equal `n_bins`/`n_stops`; unfrozen design raises); `test_fit_per_bin_rate_models.py::test_no_stop_dropped_by_freeze`
  (a design with an epoch-prefix of `reward_history_incomplete == 1` loses zero stops);
  `test_per_stop_rate_model.py::test_hierarchy_shares_cohort` (R0 and R1 share `n_stops`; the
  indicator coefficient is present in output and absent from the aggregated summary);
  `test_model_comparison.py::test_mismatched_n_bins_invalidates`; remove the `xfail` decorator at
  `test_replay_tables_audit_regressions.py:299-310` and keep the real-data test (skips when the
  pickle is absent; it will pass only after Task 8's refit — mark it `slow`).
- `scripts/rate/fit_per_bin_rate_models.py:213-218` docstring: replace the "identical bins" claim
  with a pointer to `freeze_per_bin_cohort`.

### Task 5: Separation flag, post-outcome gate, aggregation drop rule, onset edges, output-dir keying

Files: `scripts/rate/fit_swr_contrast_rate.py:182, 244-291`, `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:100-140`,
`src/continuum_swr_replay/analysis/replay/_helpers.py:868-892`, `scripts/content/fit_replay_content_models.py:1113-1196`,
`src/continuum_swr_replay/analysis/replay/model_specs.py:112-118` (`default_content_model_names`).

- `fit_swr_contrast_rate.py:182` → `fit_status(res, "glm", predictor_sd=np.asarray(res.model.exog, dtype=float).std(axis=0))`.
- In `fit_swr_contrast_rate._load` after the qvals merge: set `rpe_postoutcome` and
  `reward_belief_mean_update_l1_all_ports_postoutcome` to NaN where
  `stop["stop_start"] < stop["outcome_known_time"]`; log the count by `occupancy_state`.
- `contrast_aggregation.py:117-121`: delete the `unstable` rule; `kept = present & (|coef| < 10) & ~flagged`;
  rewrite the comment: `separated` is the effect-scale identifiability rule, `|coef| < 10` a backstop.
- `_helpers.py:876-882`: `src_binj = np.clip(np.floor(t / bin_s + 1e-9).astype(int), 0, nmax - 1)`;
  after the merge assert `int(out["count"].sum()) == n_expected` where `n_expected` counts onsets
  with `t_since_stop_onset < min(tmax, immobility_duration_s)` (join `src` to
  `s[["stop_id", "immobility_duration_s"]]`); raise `ValueError` otherwise.
- `_resolve_run_config`: when `--models` names a strict subset of
  `default_content_model_names(hierarchy)` and `--output-dir` is absent, the output directory
  becomes `<base>_<models joined by '-'>` (e.g. `models_M4-Mv-Mu-Mvu`); writing a subset into the
  canonical `models/` or `models_patch/` raises before any fit.
- Tests: `tests/unit/test_contrast_aggregation.py` (create): coef 0.01 / SE 0.5 kept,
  `separated=True` dropped; `test_per_bin_rate_helpers.py::test_onset_on_bin_edge` (onset at
  exactly `2*bin_s` lands in bin 2; an onset at `t == immobility_duration_s` raises);
  `test_fit_replay_content_models.py::test_subset_models_do_not_write_canonical_dir`;
  `test_swr_contrast_rate.py` (create): the post-outcome gate NaNs the two columns only where
  `stop_start < outcome_known_time`; a `fit_status` call on a design with a small-SD column is not
  flagged separated at an ordinary effect size.

### Task 6: Fit provenance sidecars

Read first: the provenance contract in Global Constraints.

Files: new `src/continuum_swr_replay/analysis/replay/fit_provenance.py`; `scripts/rate/fit_per_bin_rate_models.py:285-345`
(`main`), `scripts/rate/fit_per_stop_rate_models.py:159-` (`main`), `scripts/rate/fit_swr_contrast_rate.py:353-`
(`main`), `scripts/rate/fit_uncertainty_rate.py` (`main`), `scripts/rate/summarize_per_bin_rate_models.py`,
`scripts/rate/plot_per_bin_rate_models.py`; `src/continuum_swr_replay/table_io.py` (how parquet
attrs carry `covariate_schema_version`).

- Implement `write_fit_provenance` and `check_fit_provenance`; `inputs` maps a display name to the
  parquet path actually read (resolve `.pkl` names through `table_io.parquet_path` so the hashed
  file is the one read). Record `git_hash` via `git rev-parse HEAD`.
- Call the writer at the end of `main` in the four rate runners (inputs: the tables each loaded;
  `estimator="poisson_glm"`); call the checker in the summarizer and plotter, raising with the
  list of stale inputs unless `--allow-stale` is passed.
- Tests: `tests/unit/test_fit_provenance.py`: writer emits one sha256 per input matching
  `hashlib.sha256` of the file; checker returns `[]` when unchanged and the input name after one
  byte changes; the summarizer's check raises on a stale input and passes with `--allow-stale`.

### Task 7: Cross-validation and orchestrator tests

Files: `tests/unit/test_per_bin_rate_model.py:235-257`, `tests/unit/test_fit_per_bin_rate_models.py`,
`src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:195-306`.

- Replace `test_cv_folds_split_on_stop` with a test that monkeypatches `per_bin_rate_model.smf.glm`
  with a recorder: it calls the real `smf.glm`, records `set(data["stop_id"])` as the train set,
  and wraps the returned result so `.predict(te, ...)` records `set(te["stop_id"])` as the test
  set. Run `cross_validate_per_bin_rate_model` on `make_per_bin_rate_design(n_stops=120)` (with
  the basis columns) and assert every fold's train and test stop sets are disjoint and the union
  of test sets equals all stops.
- In `test_fit_per_bin_rate_models.py::test_tidy_schema` confirm the planted-effect assertion
  from Task 3 is present and passes on the full ladder with CV and GOF on.

### Task 8: Baseline backup and refit of Family A, Family B, SWR-rate, and MU

Files: `Processed-Data/replay_content/models_per_bin_rate/`, `models_per_stop_rate/`,
`scripts/rate/*.py`, new `scripts/qa/compare_fits.py`, `docs/audits/`, `paper/figures/rate/`.

- Move `models_per_bin_rate/all_models.pkl`, `models_per_bin_rate/swr_contrast_rate*.pkl`,
  `models_per_bin_rate/uncertainty_rate.pkl`, and the whole `models_per_stop_rate/` directory into
  `Processed-Data/replay_content/_pre_fix_backup_<YYYYMMDD>/` (preserve relative paths).
- Smoke each runner with `--max-per-animal 1` under `/usr/bin/time -v`; record wall time and peak
  RSS; extrapolate to 218 sessions and write the estimate in the report before launching.
- Full runs in `tmux` sessions `rate_family_a`, `rate_family_b`, `rate_swr_contrast`,
  `rate_uncertainty` (`--workers 16` where the runner supports it; check free cores first with
  `nproc` and `uptime`). Wait for completion; confirm provenance sidecars exist and
  `swr_contrast_rate_audited.pkl` is non-empty.
- Regenerate `docs/per_bin_rate_results.md` with `scripts/rate/summarize_per_bin_rate_models.py`.
- Write `scripts/qa/compare_fits.py`: join backup and new `all_models.pkl` on (session, outcome,
  model); because the reward block changed, compare the joint reward step (ΔAIC of the reward rung
  vs its parent per session: old `M2_rewhist` vs `M0_const` against new `M1_rew` vs `M0_const`;
  old `M7_time_hist_rew` vs `M6` against new), the `M5_time` KS statistic, `n_bins` before/after,
  and for Family A `R1` vs `R0`. Save `docs/audits/<YYYY-MM-DD>-rate-refit-comparison.md`.
- Plot the fitted kernel per population (three coefficients × basis, with per-session medians
  and per-animal medians) as `paper/figures/rate/reward_kernel_basis.png` (use the
  `scientific-figures-paper` conventions already followed by `scripts/rate/plot_per_bin_rate_models.py`).
- Run the real-data test `test_replay_tables_audit_regressions.py::test_rate_ladder_rungs_share_a_cohort`
  and report its result.

### Task 9: Documentation

Files: `docs/rate_models.md`, `docs/models_table.md`, `docs/covariate_dictionary.md`,
`docs/per_bin_rate_results.md` (regenerated in Task 8), `scripts/README.md`.

- `docs/rate_models.md`: rewrite the Family-A/B ladder descriptions for the basis kernel (what each
  feature weights; the structural indicator for the epoch prefix; the common-cohort rule; the
  per-bin cohort is every stop with positive immobility); replace every number in the "What the
  ladder actually shows" and Family-A paragraphs with values from the Task 8 comparison; remove
  the ten-lag "boost-then-suppression" narrative in favour of the fitted kernel figure.
- `docs/models_table.md`: confirm the Task 3 R notation; state in "Cohort-audit tables" that rate
  audits now populate `n_drop_common_cohort_other` and that the shipped ladders report zero there.
- `docs/covariate_dictionary.md`: entries for the three features and the indicator; mark the ten
  lags and the window-10 fraction as stop metadata, not predictors.
- `scripts/README.md`: add `scripts/qa/compare_fits.py`.
