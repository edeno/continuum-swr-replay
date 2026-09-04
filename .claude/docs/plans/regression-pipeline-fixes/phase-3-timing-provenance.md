# Phase 3 — Timing and provenance: arrival clock, per-bin reward lags, qvals alignment, fit hashes, fixture clock

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#arrival-clock-and-per-bin-reward-lags)

Findings addressed: C1, C2, C3, C4 (stale comments), E2, A4 (decision point).

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/covariates/context.py:453-656` — `add_trial_context`; `_goal_arrival_time` written at 615-618; `outcome_times` array at 531; `current_outcome_known` at 626-630; empty-trials column list at 504-518.
- `src/continuum_swr_replay/analysis/replay/tables_stop.py:53-200` — `build_stop_table`; `per_bout` reward columns at 136-147; `grouped` (per-bout groupby) at 129.
- `src/continuum_swr_replay/analysis/replay/_tables_common.py:206-247` — stop column tuples.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:783-935` — `aggregate_to_timebins` (the `carry` list at 849, `out` construction 858-872, reward columns carried by `stop_id`).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:631-650` (`BIN_S`, `TMAX`, `SPLINE_KNOTS`, `_SPLINE`), `:652-760` (`PerBinRateModelSpec`, `formula`), `:766-909` (`MODEL_PER_BIN_RATE`).
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:60-82` — `_design_columns`.
- `scripts/decoding/regenerate_stop_table.py` — the stop-only regeneration path (keeps `stop_id`).
- `src/continuum_swr_replay/analysis/replay/prepare_data.py:1019-1020` — `load_contingency`; `:345` `_process_session` (where the stop table is built) — add the alignment guard there.
- `src/continuum_swr_replay/analysis/behavior/qvals_table.py:759-856` — `qvals_context_time` / `outcome_known_time` per stop.
- `src/continuum_swr_replay/analysis/replay/fit_provenance.py` (phase 1).
- `tests/unit/_port_experience_fixtures.py:575-600` — fixture omission clock.
- `src/continuum_swr_replay/data_loaders/trial_clocks.py:17-80` — `outcome_known_times_from_task_visits`, `OMISSION_KNOWN_LATENCY_SEC`.
- Stale comments to fix: `covariates/history/value_uncertainty.py:345-350, 446-454`; `analysis/behavior/qvals_table.py:1003-1006`; `data_loaders/trials.py:495-501`.

**Contracts referenced:**

- [Common-cohort freeze for rate designs](shared-contracts.md#common-cohort-freeze-for-rate-designs) — arrival-clock rungs freeze on the union like every other rung, then apply their window.
- [Fit provenance sidecar](shared-contracts.md#fit-provenance-sidecar) — extended to content runners' readers here.

**Designs referenced:** [designs.md#arrival-clock-and-per-bin-reward-lags](designs.md#arrival-clock-and-per-bin-reward-lags).

## Tasks

- **Outcome time per timebin.** In `add_trial_context` add `dot["_outcome_known_time"]` next to
  `_goal_arrival_time` (615-618): `np.where(has_completed & outcome_observed[completed_clipped], outcome_times[completed_clipped], np.nan)`.
  Add it to the empty-trials column list (504-518).
- **Stop columns.** In `build_stop_table` after line 147 add
  `per_bout["last_well_arrival_time_s"] = grouped["_goal_arrival_time"].apply(_strict_first)`
  and `per_bout["outcome_known_time_s"] = grouped["_outcome_known_time"].apply(_strict_first)`
  (reindexed over `n_bouts` like the SWR columns). Append both to `_STOP_TABLE_BUILD_COLUMNS` and
  `_STOP_TABLE_COLUMNS`, add catalog entries (grain stop, clock GOAL_ARRIVAL / OUTCOME_AVAILABILITY,
  PRE_OUTCOME / POST_OUTCOME). Regenerate the stop table with
  `scripts/decoding/regenerate_stop_table.py` (smoke `--max-per-animal 1` first); confirm
  `stop_id` and every pre-existing column are byte-identical to the 2026-09-03 table
  (`scripts/qa/compare_to_golden_baseline.py` on the stop table with the two new columns excluded).
- **Per-bin clocks and reward features.** Implement `time_since_last_well_arrival_s`,
  `time_since_outcome_known_s`, and the boundary-shifted recomputation of
  `reward_history_rcos_0..2` / `reward_history_incomplete` in `aggregate_to_timebins` (design
  section 3); `stop_start`, `last_well_arrival_time_s`, `outcome_known_time_s`,
  `completed_trial_reward`, `occupancy_state`, and the ten lag columns are carried via the `carry`
  list when present (the lags are dropped before return). Return the count of arrival-spanning
  stops excluded from the shift in `out.attrs["n_stops_arrival_spanning"]`.
- **Arrival-anchored rungs.** Add `time_clock` to `PerBinRateModelSpec`, `TMAX_ARRIVAL = 8.0`,
  `ARRIVAL_SPLINE_KNOTS = (1.0, 2.0, 4.0, 6.0)`, `_ARRIVAL_SPLINE`, and the two rungs
  `M5a_time_arrival`, `M7ra_time_arrival_histcos_rew` to `MODEL_PER_BIN_RATE`. In
  `_design_columns` add `time_since_last_well_arrival_s` for arrival-clock specs. In
  `scripts/rate/fit_per_bin_rate_models.py:_fit_one_session`, after the common-cohort freeze,
  apply the arrival window for arrival-clock specs only and record `n_bins_arrival_window` and the
  retained share by population in the row (expected ≈ 88 %, balanced across SWR classes). Add
  the two per-bin clocks to `definitions.py`.
- **Decision on `onset_lag` dummies (A4).** After the refit below, if `cv_bits_per_event` remains
  NaN in > 90 % of sessions for the dummy rungs, remove `M3_hist`, `M4_hist_rew`, `M6_time_hist`,
  `M7_time_hist_rew`, `M9_rew_x_time` from `MODEL_PER_BIN_RATE` and re-parent `M9` onto `M7r`
  (as `M9r_rew_x_time`); otherwise keep them and record the CV status in `docs/rate_models.md`.
  Either outcome is a task in this phase, not a deferral.
- **qvals alignment guard.** In `prepare_data._process_session` after the stop table is built and
  the contingency loaded, call a new `assert_qvals_aligned(stop_table, contingency)` in
  `covariates/history/value_block.py`: join on `stop_id` and raise if any
  `abs(qvals_context_time - stop_start) > 1e-6` or if a qvals `stop_id` is absent from the stop
  table. Also call it from `scripts/rate/fit_swr_contrast_rate._load` and
  `scripts/rate/fit_uncertainty_rate` after their qvals merge.
- **Provenance readers.** Wire `check_fit_provenance` into `scripts/content/generate_replay_content_figures.py`,
  `scripts/rate/plot_*.py`, `scripts/rate/summarize_per_bin_rate_models.py` (if not done in phase 1),
  and `scripts/figures/*` that read `models*/` pickles; add `scripts/qa/check_fit_provenance.py`
  that lists every `models*/**/*.provenance.json` and reports stale inputs.
- **Fixture omission clock.** `tests/unit/_port_experience_fixtures.py:579-595`: compute
  `outcome_known_time` with `outcome_known_times_from_task_visits(rewarded, first_poke_in_time, final_poke_out_time, reward_on_time)`
  instead of `end_time`. Add `tests/unit/test_dio_to_tables_end_to_end.py`: beam-break and pump
  events → `make_well_visits_df` → `make_trials_df_from_well_visits` → `make_port_experiences_df`
  → `build_stop_table` / `build_event_table`, asserting `known_reward_outcome_lag_1` for a stop that
  starts inside `[poke_in, poke_in + OMISSION_KNOWN_LATENCY_SEC)` equals the *previous* outcome and
  for a stop starting after it equals the omission (0.0).
- **Stale comments.** Replace the "missing at essentially every at-well stop" text at
  `value_uncertainty.py:345-350, 446-454` and `qvals_table.py:1003-1006` with the measured fact
  (finite at bout start for ~80 % of at-well stops; the remainder start inside the arrival→outcome
  window). Fix `trials.py:495-501` to state the measured pump latency (p99 91 ms, max 236 ms) and
  that the 0.3 s guard is the binding rule.
- **Refit Family B and SWR-rate** (both consume the per-bin design): back up phase-1 outputs to
  `_pre_phase3_backup_<YYYYMMDD>/`, smoke, then full in `tmux` `rate_family_b_arrival`,
  `rate_swr_contrast_arrival`. Extend `scripts/qa/compare_fits.py` to report `M5_time` vs
  `M5a_time_arrival` KS-within-band and CV bits per session, and the `reward_history_rcos_0` coefficient before/after
  the per-bin boundary recomputation.
- **Docs.** `docs/rate_models.md`: a "Stops are immobility fragments" section with the measured
  numbers (1.96 fragments/visit, 69 % of events in later fragments, 81 % of stops post-outcome at
  start), the arrival-clock variant, and the per-bin recomputation rule; `docs/covariate_dictionary.md`
  entries for `tc`, `time_since_last_well_arrival_s`, `time_since_outcome_known_s`, the two stop
  columns, and the boundary shift; `scripts/exploratory/_timeresolved_psth.py:13` comment on `is_reward_stop` replaced by a
  pointer to `time_since_outcome_known_s`.

## Deliberately not in this phase

- Redefining the stop as a well visit (non-goal).
- Changing `time_since_stop_start_s` in the event table (documented median-bin definition; content
  rungs use it only as a control).
- The epoch-edge poke hypothesis (C7 in the review): add a one-line check to
  `scripts/qa/check_data_availability.py` only; no builder change.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_behavioral_location_context.py::test_outcome_known_time_per_timebin` | `_outcome_known_time` NaN before the outcome, equal to pump onset / arrival+0.30 s after |
| `test_replay_tables.py::test_stop_table_persists_arrival_and_outcome_clocks` | new stop columns equal the trial clocks for a synthetic session; NaN for a pre-trial stop |
| `test_per_bin_rate_helpers.py::test_reward_features_shift_at_outcome_boundary` | bins before the boundary carry the bout-start features; bins at/after carry features recomputed from `[completed_trial_reward, lag_1, …, lag_9]`, and `reward_history_incomplete` flips 1→0 for a stop containing the epoch's second outcome; arrival-spanning stops unshifted and counted |
| `test_per_bin_rate_helpers.py::test_time_since_arrival_and_outcome` | values equal `bin_center_abs − clock` for each bin |
| `test_per_bin_rate_model.py::test_arrival_clock_spec_formula_and_columns` | `M5a_time_arrival.formula()` uses `time_since_last_well_arrival_s`; `_design_columns` includes it; fit recovers a planted peak at 2 s after arrival |
| `test_value_block.py::test_assert_qvals_aligned` | shifted `qvals_context_time` raises; matching passes |
| `test_dio_to_tables_end_to_end.py` | `known_reward_outcome_lag_1` inside vs after the omission latency window as specified |
| `test_fit_provenance.py::test_readers_refuse_stale` | summarizer raises on a changed input hash unless `--allow-stale` |
| Stop-table regeneration parity (slow, real data) | all pre-existing columns byte-identical; `stop_id` set identical |
| Refit smoke (slow) | `M5a_time_arrival` fits; retained-bin share by population reported |

## Fixtures

- `_port_experience_fixtures.py` (updated omission clock) reused by every table test.
- A synthetic session for the end-to-end DIO test: 3 wells, 4 trials, one omission with a stop
  starting 0.1 s after poke-in and another starting 0.5 s after.
- `make_per_bin_rate_design` extended with `last_well_arrival_time_s`, `outcome_known_time_s`, `completed_trial_reward`, `stop_start`, the ten lag columns, and the reward basis columns.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified, including the A4 decision.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (stale comments; dummy rungs if the decision was to retire them).
- User-facing documentation listed as tasks is updated, not deferred.
- Stop-table parity evidence is attached to the PR.
