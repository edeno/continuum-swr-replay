# Phase 3 execution plan (task-structured view of phase-3-timing-provenance.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-3-timing-provenance.md`,
`designs.md` §"Arrival clock and per-bin reward lags" (complete code for the per-bin clocks, the
boundary-shifted reward features, and the arrival spline), `shared-contracts.md` (Reward-history
kernel; Common-cohort freeze for rate designs; Fit provenance sidecar), `overview.md` (decisions:
keep stops as immobility fragments and add an arrival clock; arrival window [0, 8] s with knots
1, 2, 4, 6), and the review `.claude/docs/reviews/2026-09-03-regression-pipeline-review.md`
(findings C1–C4, E2, A4). Where this file and those differ, they win.

## Global Constraints

- Run Python with `uv run …` from the repo root. Never install into the base environment.
- `uv run ruff check` and `uv run ruff format` on every file touched (check added lines with
  `ruff format --diff`; do not reformat pre-existing debt).
- Docstrings, test names, module names, and commit messages must not mention this plan, phase
  numbers, task numbers, or `.claude/docs/plans/`.
- Tests are value-level. No test-only methods in production code. Shared setup in fixtures.
  Tests over 20 s are marked `slow`.
- Fail loud: contracts are enforced with `raise`, never warn-and-continue.
- Canonical tables are parquet under `Processed-Data/replay_content/`; read through
  `continuum_swr_replay.table_io.read_table`. Never modify a canonical artifact without first
  moving/copying it into a dated `_pre_phase3_backup_<YYYYMMDD>/` directory. Never delete data.
  Nothing under `Processed-Data/` or `paper/figures/**/*.png` is committed. `paper/claims_table.md`
  is NOT edited (phase 4).
- Every persisted canonical column has a catalog entry in
  `analysis/replay/covariates/definitions.py` (grain, clock, timing class); new model-facing
  columns are registered in `covariate_timing.DERIVED_COVARIATE_SOURCES`.
- Long runs (> 5 min): smoke first (`--max-per-animal 1`), record wall time and peak RSS with
  `/usr/bin/time -v`, extrapolate to 218 sessions, then run the full job inside `tmux` and report
  the session name. Poll with bounded waits.
- Do not run `git checkout` / `git stash` / `git clean`. Only edit files your task names plus their
  tests. Commit after each coherent unit with the trailer
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` then
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`.
- State inherited from phases 1, 2a, 2b (all merged to master): rate runners freeze a common
  cohort (`per_bin_rate_model.freeze_per_bin_cohort` ~100, `cohort_selection.freeze_common_cohort`)
  before any rung is fit, and the fitters raise on NaN/`exposure <= 0`; the reward-history block
  is the three raised-cosine features `REWARD_HISTORY_COLUMNS` + `REWARD_HISTORY_INCOMPLETE`
  (`model_specs.py` ~535-546; `raised_cosine_history_basis`, `reward_basis_from_lags`,
  `reward_history_basis_features` in `covariates/history/reward_history.py` — verify the module);
  `aggregate_to_timebins` (`_helpers.py` ~834-960) carries `[*REWARD_HISTORY_COLUMNS,
  REWARD_HISTORY_INCOMPLETE]` plus `stop_covariates` by `stop_id`; provenance sidecars via
  `fit_provenance.py` (`record_inputs`, `write_fit_provenance(..., extra=)`,
  `check_fit_provenance`, `require_fresh_inputs`, `--allow-stale`) are already wired into the rate
  readers (`scripts/rate/plot_*.py`, `summarize_per_bin_rate_models.py`) and the content readers
  (`generate_replay_content_figures.py`, `analyze_uncertainty_content.py`,
  `plot_swr_contrast_coefs.py`); `scripts/qa/compare_fits.py` has rate modes (per-bin, per-stop,
  SWR contrast) and a `--content` mode; content models are on the fraction response with `nt`
  twins and a DM estimator (`_dm` bundles) — none of that is touched here.
- The reward-history kernel contract (binding, verbatim from shared-contracts.md): the same
  function computes the stop-level values (bout start) and the per-bin values after the
  outcome-boundary shift, from the same lag matrix; features 0 and 1 exist from the first known
  outcome, feature 2 from the second; undefined features are 0.0 with `reward_history_incomplete = 1`;
  nothing is dropped for it.
- The common-cohort contract (binding): arrival-clock rungs freeze on the union like every other
  rung, THEN apply their window; every rung of a ladder still reports its `n_bins`/`n_stops`
  (the arrival rungs additionally report `n_bins_arrival_window`), and `compare_nested_models`
  NaNs a comparison whose `n_bins` differ — so arrival-clock rungs are compared only with each
  other or with a parent refit on the same window (see Task 3).

### Task 1: Outcome time per timebin, the two stop clocks, and stop-table regeneration

Files: `src/continuum_swr_replay/analysis/replay/covariates/context.py` (`add_trial_context` ~453;
empty-trials column list ~504-518; `outcome_times` ~531; `_goal_arrival_time` written ~615-618;
`current_outcome_known` ~626-630), `src/continuum_swr_replay/analysis/replay/tables_stop.py`
(`build_stop_table` ~58; `per_bout` columns ~140-147; `_strict_first` imported ~36),
`src/continuum_swr_replay/analysis/replay/_tables_common.py` (`_STOP_TABLE_BUILD_COLUMNS` ~206,
`_STOP_TABLE_COLUMNS` ~249), `covariates/definitions.py` (catalog),
`scripts/decoding/regenerate_stop_table.py` (stop-only regeneration, keeps `stop_id`),
`scripts/qa/compare_to_golden_baseline.py` / `table_parity.diff_table(..., expected_added_columns=)`.
Tests: `tests/unit/test_behavioral_location_context.py`, `tests/unit/test_replay_tables.py`.

- In `add_trial_context`, next to `_goal_arrival_time`, write
  `dot["_outcome_known_time"] = np.where(has_completed & outcome_observed[completed_clipped], outcome_times[completed_clipped], np.nan)`
  (reuse the same `has_completed`/`completed_clipped` arrays the arrival column uses); add it to
  the empty-trials column list.
- In `build_stop_table` add `per_bout["last_well_arrival_time_s"] = grouped["_goal_arrival_time"].apply(_strict_first)`
  and `per_bout["outcome_known_time_s"] = grouped["_outcome_known_time"].apply(_strict_first)`,
  reindexed over the bouts like the SWR columns; append both to `_STOP_TABLE_BUILD_COLUMNS` (and
  therefore `_STOP_TABLE_COLUMNS`); catalog entries: grain stop, clocks GOAL_ARRIVAL /
  OUTCOME_AVAILABILITY, timing PRE_OUTCOME / POST_OUTCOME respectively, absolute seconds.
- Tests: `test_outcome_known_time_per_timebin` — `_outcome_known_time` NaN before the outcome,
  equal to the pump onset (rewarded) / arrival + `OMISSION_KNOWN_LATENCY_SEC` (omission) after;
  `test_stop_table_persists_arrival_and_outcome_clocks` — the two stop columns equal the trial
  clocks on a synthetic session and are NaN for a pre-trial stop; `_strict_first` raises when a
  bout's rows disagree.
- Regenerate the stop table: back up `Processed-Data/replay_content/stop_table.parquet` (and any
  stop-derived cache the script names) to `_pre_phase3_backup_<YYYYMMDD>/`; smoke
  `regenerate_stop_table.py --max-per-animal 1` (verify the flag) under `/usr/bin/time -v`; full
  run in tmux `stop_table_regen`; then parity: every pre-existing column byte-identical and the
  `stop_id` set identical to the backup (`diff_table(..., expected_added_columns=
  ["last_well_arrival_time_s", "outcome_known_time_s"])`); write
  `docs/audits/<date>-stop-table-clock-columns-regen.md` with the parity result and the coverage of
  the two new columns (share finite by occupancy state; expected: arrival finite on every at-well
  stop, outcome finite on ~80 % of at-well stops at bout start).

### Task 2: Per-bin clocks and boundary-shifted reward features

Read first: `designs.md` §"Arrival clock and per-bin reward lags" (complete code).

Files: `src/continuum_swr_replay/analysis/replay/_helpers.py` (`aggregate_to_timebins` ~834-960:
`carry` ~923, `out` construction), `covariates/history/reward_history.py` (the basis functions —
reuse, do not duplicate), `covariates/definitions.py` (two per-bin clocks),
`covariate_timing.py` (register `time_since_last_well_arrival_s`, `time_since_outcome_known_s`),
`tests/unit/test_per_bin_rate_helpers.py` and the `make_per_bin_rate_design` fixture (extend with
`last_well_arrival_time_s`, `outcome_known_time_s`, `completed_trial_reward`, `stop_start`,
`occupancy_state`, the ten `known_reward_outcome_lag_j` columns, and the reward basis columns).

- When the stop frame carries `last_well_arrival_time_s` / `outcome_known_time_s`, emit
  `time_since_last_well_arrival_s` and `time_since_outcome_known_s` per bin exactly as designed
  (bin centre minus clock; NaN where the clock is NaN).
- When the spec carries the reward basis AND the outcome boundary falls inside the stop
  (`stop_start < outcome_known_time_s <= stop_end`) AND `occupancy_state == "at_well"`, shift the
  per-bin lag matrix at the boundary (lag_1 ← `completed_trial_reward`, lag_j ← lag_{j−1}) for bins
  whose left edge is ≥ the boundary, recompute the three basis features and
  `reward_history_incomplete` from the shifted matrix with the SAME functions the stop level uses,
  and leave bins before the boundary on the bout-start values. Stops with `occupancy_state ==
  "mixed"` (arrival-spanning) are not shifted and are counted in
  `out.attrs["n_stops_arrival_spanning"]`. The ten lag columns, `completed_trial_reward`,
  `occupancy_state`, and the two clocks are carried by `stop_id` only when present and the lags are
  dropped before return.
- Tests: `test_reward_features_shift_at_outcome_boundary` — bins before the boundary carry the
  bout-start features; bins at/after carry features recomputed from `[completed_trial_reward,
  lag_1, …, lag_9]` (compare to a direct call of the basis function on the shifted row);
  `reward_history_incomplete` flips 1 → 0 for a stop containing the epoch's second known outcome;
  arrival-spanning stops unshifted and counted; a stop whose boundary is outside the stop is
  untouched. `test_time_since_arrival_and_outcome` — values equal `bin_center_abs − clock` per bin,
  NaN where the clock is NaN. A regression test that a design WITHOUT the new stop columns is
  byte-identical to the current output (freeze one fixture output before the change and compare).

### Task 3: Arrival-anchored rungs

Files: `src/continuum_swr_replay/analysis/replay/model_specs.py` (`BIN_S`/`TMAX`/`SPLINE_KNOTS`/
`_SPLINE` ~640-650; `PerBinRateModelSpec` ~663 and `formula()`; `MODEL_PER_BIN_RATE` ~787-910),
`src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py` (`_design_columns` ~75;
`freeze_per_bin_cohort` ~100), `scripts/rate/fit_per_bin_rate_models.py` (`_fit_one_session`
~127; the freeze call ~164), `src/continuum_swr_replay/analysis/replay/model_comparison.py`
(`compare_nested_models` guard on `n_bins`). Tests: `tests/unit/test_per_bin_rate_model.py`,
`tests/unit/test_model_specs.py`, `tests/unit/test_fit_per_bin_rate_models.py` (or the runner's
existing test module).

- Add `time_clock: str = "stop"` to `PerBinRateModelSpec` (`"arrival"` swaps `_SPLINE` for
  `_ARRIVAL_SPLINE` in `formula()`), `TMAX_ARRIVAL = 8.0`, `ARRIVAL_SPLINE_KNOTS = (1.0, 2.0,
  4.0, 6.0)`, `_ARRIVAL_SPLINE`, and the rungs `M5a_time_arrival` (parent `M0_const`) and
  `M7ra_time_arrival_histcos_rew` (parent `M5a_time_arrival`; same blocks as
  `M7r_time_histcos_rew` with the arrival spline). Export the new constants. `_design_columns` adds
  `time_since_last_well_arrival_s` for arrival-clock specs.
- Runner: after `freeze_per_bin_cohort` over the union (which now includes the arrival column,
  so bins with a NaN arrival clock — pre-trial stops — leave the common cohort for EVERY rung;
  report how many in the freeze report), apply the window `0 ≤ time_since_last_well_arrival_s ≤
  TMAX_ARRIVAL` for arrival-clock specs only; record `n_bins_arrival_window` and the retained
  share by population (`retained_share_arrival_{all,swr,nonswr}`) in the row (expected ≈ 88 %,
  ≈ 89 % SWR / 87 % non-SWR). Because the parent `M0_const` is fit on the full window, ALSO fit
  `M0_const` on the arrival window inside the arrival rungs' own comparison (`M0a_const_arrival`,
  not persisted as a ladder rung but as the `llf_null`/parent for `M5a`), so `compare_nested_models`
  compares equal-`n_bins` rows; document this in the spec docstring. Ruling to apply: arrival rungs
  are compared within the arrival window only; the stop-clock rungs are never numerically compared
  with them (the doc reports both ladders side by side).
- Catalog entries for the two per-bin clocks in `definitions.py` (if not done in Task 2).
- Tests: `test_arrival_clock_spec_formula_and_columns` — `M5a_time_arrival.formula()` uses
  `time_since_last_well_arrival_s` with the arrival knots/bounds; `_design_columns` includes it;
  on a simulated per-bin design with a planted rate peak at 2 s after arrival the arrival rung
  recovers the peak (argmax of the fitted time course within 0.5 s) while the stop-clock rung's
  time course is flatter (state the numeric criterion); the window keeps ≈ the planted share; the
  runner row carries `n_bins_arrival_window` and the retained shares; a stop-clock rung's `n_bins`
  still equals every other stop-clock rung's.

### Task 4: qvals alignment guard, fixture omission clock, DIO end-to-end test, stale comments, provenance check script

Files: `src/continuum_swr_replay/analysis/replay/covariates/history/value_block.py` (new
`assert_qvals_aligned(stop_table, contingency)`), `src/continuum_swr_replay/analysis/replay/prepare_data.py`
(`_process_session` ~345; `build_stop_table` call ~424; `load_contingency` — verify),
`scripts/rate/fit_swr_contrast_rate.py` (`_load`), `scripts/rate/fit_uncertainty_rate.py` (after
its qvals merge), `tests/unit/_port_experience_fixtures.py` (~575-600, omission clock), new
`tests/unit/test_dio_to_tables_end_to_end.py`, `src/continuum_swr_replay/data_loaders/trial_clocks.py`
(`outcome_known_times_from_task_visits` ~17; `OMISSION_KNOWN_LATENCY_SEC`), stale comments at
`covariates/history/value_uncertainty.py` ~446-454 (and any twin near ~345), `analysis/behavior/qvals_table.py`
~1003-1006, `data_loaders/trials.py` ~495-501; new `scripts/qa/check_fit_provenance.py`;
`scripts/qa/check_data_availability.py` (one-line epoch-edge poke check); `scripts/exploratory/_timeresolved_psth.py:13`
comment. Tests: `tests/unit/test_value_block.py`, `tests/unit/test_fit_provenance.py`.

- `assert_qvals_aligned`: join on `stop_id`; raise `ValueError` naming the first offending stop if
  any `abs(qvals_context_time − stop_start) > 1e-6` or if a qvals `stop_id` is absent from the stop
  table; called in `_process_session` after the stop table is built and the contingency loaded,
  and in the two rate runners after their qvals merge. Test: shifted `qvals_context_time` raises;
  matching passes; a missing `stop_id` raises.
- Fixture omission clock: compute `outcome_known_time` with
  `outcome_known_times_from_task_visits(rewarded, first_poke_in_time, final_poke_out_time, reward_on_time)`
  instead of `end_time`. Then `test_dio_to_tables_end_to_end.py`: beam-break and pump events →
  `make_well_visits_df` → `make_trials_df_from_well_visits` → `make_port_experiences_df` →
  `build_stop_table` / `build_event_table` on a synthetic session (3 wells, 4 trials, one omission
  with a stop starting 0.1 s after poke-in and another starting 0.5 s after), asserting
  `known_reward_outcome_lag_1` for the stop starting inside `[poke_in, poke_in +
  OMISSION_KNOWN_LATENCY_SEC)` equals the PREVIOUS outcome and for the later stop equals the
  omission (0.0); also that the new `outcome_known_time_s` stop column equals
  `poke_in + OMISSION_KNOWN_LATENCY_SEC` for the omission and the pump time for a reward.
  Every existing table test must still pass after the fixture clock change; if any asserted value
  changes, the change must be explained in the report as the fixture having been wrong (the
  reviewed finding E2), not silently updated.
- Stale comments: replace the "missing at essentially every at-well stop" text with the measured
  fact (finite at bout start for ~80 % of at-well stops; the remainder start inside the
  arrival→outcome window; cite the Task 1 audit); `trials.py` ~495-501: state the measured pump
  latency (p99 91 ms, max 236 ms) and that the 0.3 s guard is the binding rule;
  `_timeresolved_psth.py:13`: pointer to `time_since_outcome_known_s`.
- `scripts/qa/check_fit_provenance.py`: lists every `Processed-Data/replay_content/models*/**/*.provenance.json`,
  re-hashes the recorded inputs, prints one line per artifact (fresh / stale: which inputs /
  missing sidecar), exit code 1 if any stale; `--json` output option. Confirm every reader listed
  in the inherited-state block already calls `require_fresh_inputs`; wire any `scripts/figures/*`
  reader of `models*/` pickles that does not. Test: a temp tree with one fresh and one stale
  sidecar → correct classification and exit code.
- `check_data_availability.py`: one line reporting, per epoch, the count of pokes within 1 s of
  the epoch edges (review C7); no builder change.

### Task 5: Refit Family B and SWR-rate on the new design, A4 decision, comparison, docs

Files: `Processed-Data/replay_content/models_per_bin_rate/`, the SWR-rate contrast outputs
(`scripts/rate/fit_swr_contrast_rate.py` — find its output paths), `scripts/qa/compare_fits.py`
(extend), `docs/rate_models.md`, `docs/per_bin_rate_results.md` (if it carries numbers),
`docs/covariate_dictionary.md`, `docs/audits/<date>-rate-refit-arrival-clock.md` (generated),
`src/continuum_swr_replay/analysis/replay/model_specs.py` (A4 outcome).

- Back up `models_per_bin_rate/` and the SWR-rate contrast artifacts to
  `_pre_phase3_backup_<YYYYMMDD>/` (move, with sidecars). Smoke both runners (`--max-per-animal 1`,
  `/usr/bin/time -v`), extrapolate, then full runs in tmux `rate_family_b_arrival` and
  `rate_swr_contrast_arrival` (logs under `logs/phase3_rate_refit/` — name the dir without the
  word "phase": `logs/rate_refit_arrival/`). Verify: sessions fitted, every rung incl. `M5a`/`M7ra`,
  provenance sidecars fresh, retained arrival share ≈ 88 % balanced across SWR classes.
- A4 decision: from the refit's `cv_bits_per_event`, if it is NaN in > 90 % of sessions for the
  `onset_lag` dummy rungs (`M3_hist`, `M4_hist_rew`, `M6_time_hist`, `M7_time_hist_rew`,
  `M9_rew_x_time`), remove them from `MODEL_PER_BIN_RATE`, re-parent `M9` onto `M7r` as
  `M9r_rew_x_time`, refit the affected rungs, and record the retirement in
  `docs/design/model_history.md`; otherwise keep them and record the CV status in
  `docs/rate_models.md`. Either outcome is executed in this task; the report states the measured
  NaN share.
- `compare_fits.py`: report `M5_time` vs `M5a_time_arrival` (per-session CV bits, and the
  time-course shape: KS distance between the two fitted time courses evaluated on their own
  windows, plus the position of the peak), and the `reward_history_rcos_0..2` coefficients before
  (bout-start features) vs after (boundary-recomputed) for `M7r`; generated audit committed.
- Docs: `docs/rate_models.md` gains "Stops are immobility fragments" (1.96 fragments/visit, 69 %
  of events in later fragments, 81 % of stops post-outcome at start — re-derive these three from
  the new stop table and state the numbers you measured), the arrival-clock variant, and the
  per-bin recomputation rule; `docs/covariate_dictionary.md` entries for `tc`,
  `time_since_last_well_arrival_s`, `time_since_outcome_known_s`, the two stop columns, and the
  boundary shift. `paper/claims_table.md` untouched.

## Review

Before the final review, confirm: every task implemented as specified, including the A4 decision
executed one way or the other; "Deliberately not in this phase" honoured (stop definition
unchanged; `time_since_stop_start_s` in the event table unchanged; C7 is a one-line check only);
stop-table parity evidence attached; validation-slice tests pass with `slow` marks; tests
value-level; no plan/phase references; stale comments actually replaced; docs updated; every
refit artifact has a fresh provenance sidecar; smoke timings and full wall times in the report.
