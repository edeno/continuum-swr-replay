# Phase 1 — Rate runners: spline reward kernel, one cohort per ladder, honest flags, refit Family A/B and SWR-rate

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#reward-history-basis-features)

Findings addressed: A2, A3, A4, A5, A6, A7 (onset edges), B4, E1, E3, E4.

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/covariates/history/reward_history.py:231-276` — `lagged_trial_rewards` (the lag matrix the basis features are built from; NaN where fewer than *j* outcomes are known).
- `src/continuum_swr_replay/analysis/replay/_helpers.py:752-781` — `raised_cosine_history_basis` (reused for the reward kernel).
- `src/continuum_swr_replay/analysis/replay/tables_stop.py:136-147` — stop reward columns at bout start; the three basis columns are added here.
- `src/continuum_swr_replay/analysis/replay/_tables_common.py:206-247` — stop column tuples.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:551-608` (`MODEL_PER_STOP_RATE`), `:631-650` (constants), `:652-760` (`PerBinRateModelSpec`, `formula`), `:766-909` (`MODEL_PER_BIN_RATE`), `:995-1034` (`_REWK_PREV`, `_BLOCK_F`, `_BLOCK_COLS`), `:1060-1087` (`_s_specs`).
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:60-82` (`_design_columns`), `:84-193` (`fit_per_bin_rate_model`; spec-specific `dropna` at 135), `:195-306` (`cross_validate_per_bin_rate_model`; `dropna` at 235).
- `src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:63-173` — `fit_per_stop_rate_model`; `dropna` at 103.
- `src/continuum_swr_replay/analysis/replay/cohort_selection.py:33-60` — `freeze_common_cohort`.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:783-935` — `aggregate_to_timebins` (`reward_lags` carry at 849; onset binning 876-882).
- `scripts/rate/fit_per_bin_rate_models.py:108-168, 285-345`; `scripts/rate/fit_per_stop_rate_models.py:69-156`; `scripts/rate/fit_swr_contrast_rate.py:152-197, 200-241, 244-291`; `scripts/rate/fit_uncertainty_rate.py:140-165` (`MU_UNCERTAINTY` uses the reward kernel).
- `src/continuum_swr_replay/analysis/replay/model_comparison.py:20-98`; `contrast_aggregation.py:60-75` (`BINARY_TERMS`), `:100-140` (drop rules).
- `src/continuum_swr_replay/analysis/replay/covariates/definitions.py:729-743` (lag catalog entries), `covariate_timing.py:64-70` (`omission_streak` sources).
- `scripts/content/fit_replay_content_models.py:1113-1196` — `_resolve_run_config`.
- `scripts/decoding/regenerate_stop_table.py` — stop-only regeneration.
- `tests/unit/test_replay_tables_audit_regressions.py:299-330`, `tests/unit/test_per_bin_rate_model.py:235-257`, `tests/unit/test_fit_per_bin_rate_models.py:31-107`.

**Contracts referenced:**

- [Reward-history kernel](shared-contracts.md#reward-history-kernel) — implemented here.
- [Common-cohort freeze for rate designs](shared-contracts.md#common-cohort-freeze-for-rate-designs) — implemented here.
- [Fit provenance sidecar](shared-contracts.md#fit-provenance-sidecar) — module introduced here.

**Designs referenced:** [designs.md#reward-history-basis-features](designs.md#reward-history-basis-features), [designs.md#output-directory-keying](designs.md#output-directory-keying).

## Tasks

- **Reward-history basis features.** Add `reward_basis_from_lags` and
  `reward_history_basis_features(query_times, port_experiences, *, n_lags=10, n_basis=3)` to
  `covariates/history/reward_history.py` (design section 5): the lag matrix from
  `lagged_trial_rewards`, projected onto `raised_cosine_history_basis(10, 3)` with per-feature
  normalization over the available lags, undefined entries zero-filled, plus the
  `reward_history_incomplete` indicator. Expose the constants `REWARD_HISTORY_N_LAGS = 10`,
  `REWARD_HISTORY_N_BASIS = 3`, `REWARD_HISTORY_COLUMNS = ("reward_history_rcos_0", "_1", "_2")`,
  `REWARD_HISTORY_INCOMPLETE = "reward_history_incomplete"` in `model_specs.py` beside
  `N_REWARD_LAGS`. In `build_stop_table` after line 147 write the four columns at `bout_starts`;
  append them to `_STOP_TABLE_BUILD_COLUMNS` / `_STOP_TABLE_COLUMNS`; add catalog entries (grain
  stop, clock STOP_START, PRE_OUTCOME, producer `covariates.history.reward_history_basis_features`,
  window "log-spaced raised-cosine bump over the last 10 known outcomes, normalized over the
  available lags"; the indicator's entry states it is a structural-missingness nuisance term). Keep
  `known_reward_outcome_lag_1..10` and `reward_fraction_last10_known_trials` persisted (consumed by
  `omission_streak`, the uncertainty covariates, and the qvals cross-checks) but remove them from
  every ladder spec. Regenerate the stop table with `scripts/decoding/regenerate_stop_table.py`
  (smoke `--max-per-animal 1` first) and confirm every pre-existing column and `stop_id` are
  byte-identical to the 2026-09-03 table.
- **Respecify the ladders.** `PerBinRateModelSpec`: replace `reward_lags: int` with
  `reward_basis: int` (0 or `REWARD_HISTORY_N_BASIS`); `formula()` emits
  `reward_history_rcos_0 + _1 + _2 + reward_history_incomplete`; the reward × time interaction becomes
  `reward_history_rcos_0:bs(tc, …)`; `_design_columns` follows. `MODEL_PER_BIN_RATE`: `M1_rew1` and
  `M2_rewhist` collapse into one rung `M1_rew` (three basis features, parent `M0_const`);
  `M2r_rewrate` is deleted; `M4_hist_rew`, `M7_time_hist_rew`, `M7r_time_histcos_rew`,
  `M9_rew_x_time` use the basis. `MODEL_PER_STOP_RATE`: `R1` carries the three features; `R2` and
  `Rlag` are deleted. SWR-rate contrast: `_BLOCK_F["BASE"] = f"{_SPLINE} + {_REWBASIS} + {_HISTCOS}"`,
  `S1_rew1` becomes `S1_rew` with `is_swr:reward_history_rcos_0`, `REWXTIME` becomes
  `reward_history_rcos_0:{_SPLINE}`; `_BLOCK_COLS` and `SWR_CONTRAST_RATE_STOP_COVARS` follow.
  In the `S` specs the indicator stays outside the `is_swr * (...)` expansion:
  `f"is_swr * ({base_without_indicator}) + reward_history_incomplete"`. `contrast_aggregation`
  and every summarizer skip terms named `reward_history_incomplete`.
  `fit_uncertainty_rate.MU_UNCERTAINTY` uses the basis. `STOP_COVARIATES` becomes `()`.
  `contrast_aggregation.BINARY_TERMS`: remove the two lag-1 entries; the basis features are
  continuous rates in [0, 1] and are reported per unit. `covariate_timing.DERIVED_COVARIATE_SOURCES`:
  `omission_streak` keeps its lag sources; add nothing for the basis (they are atoms in the catalog).
  Delete the `reward_lags`/`_REWK_PREV`/`N_REWARD_LAGS`-driven formula strings that no longer have a
  consumer; `N_REWARD_LAGS` survives only as the basis span.
- **Freeze the per-bin cohort once per (session, population).** Add `freeze_per_bin_cohort` to
  `per_bin_rate_model.py` (contract). In `fit_per_bin_rate_model` replace lines 135-136 with a
  check that raises `ValueError("design is not frozen: NaN in {cols} / nonpositive exposure")`;
  same in `cross_validate_per_bin_rate_model` (235-236). In
  `scripts/rate/fit_per_bin_rate_models.py:_fit_one_session` call
  `design, report = freeze_per_bin_cohort(design, MODEL_PER_BIN_RATE.values())` right after
  `aggregate_to_timebins` (130-139) and carry `report` so `_cohort_records` (237) emits
  `n_drop_common_cohort_other`. In `scripts/rate/fit_swr_contrast_rate.py:_fit_one_session` freeze
  each stratum design over the union of `G_SPECS` columns and the stacked design over the union of
  `S_SPECS` columns plus `is_swr`; delete the `dropna` at 167 in `_fit` and raise instead.
  `fit_uncertainty_rate.py` freezes its design over `MU_UNCERTAINTY`'s columns the same way. With
  the zero-filled basis features and the indicator, the union cohort of the shipped ladder is
  every stop with positive immobility; assert in the runner that the freeze removed zero stops
  for `MODEL_PER_BIN_RATE` (a nonzero drop means a new NaN source and must be explained). The
  qvals-dependent rungs (`MU_uncertainty`, G/S value rungs) still lose their qvals gap, reported
  as `n_drop_qvals_gap`.
- **Freeze the per-stop cohort once per session.** In `fit_per_stop_rate_hierarchy`, before the
  outcome/model loops (82), call
  `group, sel = freeze_common_cohort(group, unit_col="stop_id", required_cols=[*all_predictors, *RESPONSE_COLS.values()], offset_col="immobility_duration_s")`
  with `all_predictors` the union over `MODEL_PER_STOP_RATE`. Replace `dropna` at
  `per_stop_rate_model.py:103` with a raise. Rewrite the accounting at 98-115 so `n_own` counts
  stops missing the model's own predictors and `n_drop_common_cohort_other` the remainder of
  `sel["n_missing_units"]`.
- **Guard comparisons on row counts.** `model_comparison.compare_nested_models`: include `n_bins`
  and `n_stops` in `fit_cols` when present (53-56) and NaN the four comparison columns when any of
  `n_events`/`n_bins`/`n_stops` differ (84-97).
- **Effect-scale separation in the SWR-rate runner.** `fit_swr_contrast_rate.py:182` →
  `fit_status(res, "glm", predictor_sd=np.asarray(res.model.exog, dtype=float).std(axis=0))`.
- **Gate post-outcome qvals covariates to post-outcome stops.** In `fit_swr_contrast_rate._load`
  after the qvals merge (270-290): set `rpe_postoutcome` and
  `reward_belief_mean_update_l1_all_ports_postoutcome` to NaN where
  `stop["stop_start"] < stop["outcome_known_time"]`; log the count by `occupancy_state`. Those rows
  leave G5/G7/S7/S9 through the common cohort of those rungs only.
- **Replace the estimate-dependent drop in aggregation.** `contrast_aggregation.py:117-121`: delete
  the `unstable` rule; `kept = present & (|coef| < 10) & ~flagged`; update the comment.
- **Onset bins on exact edges.** `_helpers.py:876-882`: `np.floor(t / bin_s + 1e-9).astype(int)`;
  after the merge assert `int(out["count"].sum()) == n_expected` (onsets with
  `t_since_stop_onset < min(tmax, immobility_duration_s)`), raise otherwise.
- **Key content output directories on the requested model set** (design section 4); the canonical
  base ladder is restored in phase 2a.
- **Provenance module.** Create `analysis/replay/fit_provenance.py` (contract); call the writer at
  the end of `main` in the three rate runners and `fit_uncertainty_rate.py`; call the checker in
  `scripts/rate/summarize_per_bin_rate_models.py` and `plot_per_bin_rate_models.py`
  (raise unless `--allow-stale`).
- **Tests.**
  - `tests/unit/test_covariates.py`: `reward_history_basis_features` on a hand-built experience
    stream: (i) with ten known outcomes each feature equals the weighted mean of the lag vector and
    `incomplete == 0`; (ii) with one known outcome features 0 and 1 equal that outcome, feature 2
    is 0.0, `incomplete == 1`; (iii) with two known outcomes feature 2 equals the second outcome
    (only lag 2 in its support) and `incomplete == 0`; (iv) a query before any outcome gives all
    zeros and `incomplete == 1`; (v) values are in [0, 1]; (vi) the outcome clock is respected (an
    event in the arrival→outcome gap does not see the pending outcome); (vii) `reward_basis_from_lags`
    on a lag row with a NaN returns NaN only for bumps whose support is entirely NaN.
  - `tests/unit/test_replay_tables.py`: the stop table carries the three columns and they equal the
    function output at `bout_starts`.
  - `tests/unit/test_per_bin_rate_model.py:235-257`: replace with a recorder-based test of
    `cross_validate_per_bin_rate_model` (monkeypatch `per_bin_rate_model.smf.glm` to record train
    `stop_id`s and wrap `.predict` to record test `stop_id`s); assert disjoint folds and every stop
    held out once.
  - `tests/unit/test_fit_per_bin_rate_models.py`: fixture gains the three basis columns (planted
    effect on `reward_history_rcos_0`); assert `rr_reward_history_rcos_0` for `M1_rew` in [1.5, 2.7].
  - New `test_freeze_per_bin_cohort_makes_rungs_share_rows`: all specs report identical
    `n_bins`/`n_stops`; unfrozen design raises.
  - Remove the xfail at `test_replay_tables_audit_regressions.py:299-310`; keep the real-data test.
  - `test_per_stop_rate_model.py`: R0 and R1 share `n_stops` and no stop is lost when early stops
    have `reward_history_incomplete == 1`; the indicator's coefficient is present in the output
    and absent from the aggregated summary.
  - `test_per_bin_rate_helpers.py`: onset at `k*bin_s` lands in bin `k`; onset at `t == duration` raises.
  - `test_contrast_aggregation.py` (create): coef 0.01 / SE 0.5 kept; `separated=True` dropped.
  - `test_model_specs.py`: no shipped spec references `known_reward_outcome_lag_*` or
    `reward_fraction_last10_known_trials`; every formula's columns resolve through `_design_columns`.
- **Baseline capture, then refit.** Move `models_per_bin_rate/all_models.pkl`,
  `models_per_bin_rate/swr_contrast_rate*.pkl`, `models_per_bin_rate/uncertainty_rate.pkl`,
  `models_per_stop_rate/*` into `Processed-Data/replay_content/_pre_fix_backup_<YYYYMMDD>/`. Smoke
  each runner with `--max-per-animal 1`, record wall time and peak RSS (`/usr/bin/time -v`),
  extrapolate to 218 sessions, then run the full fits in `tmux` sessions `rate_family_a`,
  `rate_family_b`, `rate_swr_contrast`, `rate_uncertainty` (`--workers 16` where supported).
  Regenerate `docs/per_bin_rate_results.md`.
- **Before/after table.** `scripts/qa/compare_fits.py`: join backup and new `all_models.pkl` on
  (session, outcome, model); because the reward block changed, compare the *joint* reward step
  (ΔAIC of the reward rung vs its parent, per session) and the `M5_time` KS statistic rather than
  individual lag coefficients; save to `docs/audits/<date>-rate-refit-comparison.md`. Plot the
  fitted kernel (three coefficients × basis) per population as `paper/figures/rate/reward_kernel_basis.png`.
- **Docs.** `docs/rate_models.md`: rewrite the Family-A/B ladder descriptions for the basis kernel
  (what each feature weights, the structural indicator for the epoch prefix, the common-cohort
  rule); replace the
  numbers with refit values. `docs/models_table.md:113-153` R-notation for Family A/B and the
  SWR-rate `BASE`; conventions define `REWBASIS`. `docs/covariate_dictionary.md`: entries for the
  three features; mark the ten lags and the window-10 fraction as stop metadata, not predictors.
  `scripts/rate/fit_per_bin_rate_models.py:213-218` docstring → pointer to `freeze_per_bin_cohort`.

## Deliberately not in this phase

- Any content-family change (phase 2a), including the response column.
- Arrival clock and per-bin recomputation of the basis features at the outcome boundary
  (phase 3); this phase uses bout-start features so the phase-3 comparison has a corrected baseline.
- Rewriting `paper/claims_table.md` (phase 4); the lag-1 language there is replaced when the
  refits exist.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_covariates.py::test_reward_history_basis_features_*` (seven cases above) | normalization, zero-fill + indicator by bump support, [0, 1] range, outcome-clock gating |
| `test_fit_per_bin_rate_models.py::test_no_stop_dropped_by_freeze` | the shipped ladder's freeze removes zero stops from a design with a session-prefix of incomplete history |
| `test_replay_tables.py::test_stop_table_reward_basis_columns` | stop columns equal the function evaluated at `bout_starts` |
| `test_per_bin_rate_model.py::test_cv_folds_split_on_stop` (rewritten) | disjoint train/test stops per fold; union equals all stops |
| `test_per_bin_rate_model.py::test_freeze_per_bin_cohort_makes_rungs_share_rows` | equal `n_bins`, `n_stops` across specs; unfrozen design raises |
| `test_fit_per_bin_rate_models.py::test_tidy_schema` | planted RR = 2 on `reward_history_rcos_0` recovered in [1.5, 2.7] |
| `test_replay_tables_audit_regressions.py::test_rate_ladder_rungs_share_a_cohort` (real data, skip if absent) | `n_bins` per model identical across rungs |
| `test_per_stop_rate_model.py::test_hierarchy_shares_cohort` | R0 and R1 `n_stops` equal |
| `test_per_bin_rate_helpers.py::test_onset_on_bin_edge` | edge onset in the right bin; `t == duration` raises |
| `test_contrast_aggregation.py::test_small_coefficient_is_not_dropped` | coef 0.01 / SE 0.5 retained; `separated=True` dropped |
| `test_model_comparison.py::test_mismatched_n_bins_invalidates` | `delta_aic` NaN when `n_bins` differ |
| `test_model_specs.py::test_no_lag_dummies_in_shipped_specs` | no ladder references the ten lags or the window fraction |
| `test_fit_provenance.py` | writer emits sha256 per input; checker reports changed inputs |
| Stop-table regeneration parity (slow, real data) | pre-existing columns and `stop_id` byte-identical |
| Refit smoke (slow) | all four rate runners complete on `--max-per-animal 1`; `swr_contrast_rate_audited.pkl` non-empty |

## Fixtures

- Hand-built port-experience streams (existing `_port_experience_fixtures.py`) for the basis tests.
- `make_per_bin_rate_design` extended with `reward_history_rcos_0..2` and a `n_missing_feature2_stops` option.
- `_toy_tables` in `test_fit_per_bin_rate_models.py` updated to plant the effect on `reward_history_rcos_0`.
- Real-data tests gate on `PROCESSED_DATA_DIR`, marked `slow`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal are actually removed: the three spec-specific `dropna` calls, the `unstable` rule, `M1_rew1`/`M2_rewhist`/`M2r_rewrate`/`R2`/`Rlag`, `_REWK_PREV`, and every formula string built from `reward_lags`.
- User-facing documentation listed as tasks is updated, not deferred.
- The before/after comparison table and the kernel figure are attached to the PR description.
