# Phase 8 — Canonical trial timebase and covariate repair

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#trial-timebase)

Replace stop-row-derived behavioral history with exact trial/timestamp covariates, add explicit
outcome/decision timing, make visit recency suffix-invariant, distinguish unvisited from
observed-zero/exposure histories at arm and patch grain, and remove misleading names. This is the
first number-changing phase and writes versioned candidate tables only after synthetic and one-animal
validation; Phase 13 alone promotes a global canonical release.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/covariates.py:126](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L126) — last-ten-trial count currently named cumulative reward.
- [src/continuum_swr_replay/analysis/replay/covariates.py:47-119](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L47) — session-length sentinel currently makes early recency depend on future trials.
- [src/continuum_swr_replay/analysis/replay/covariates.py:456](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L456) — stop-row switch calculation and unused stop-grain WSLS/leaky-Q outputs.
- [src/continuum_swr_replay/analysis/replay/_helpers.py:29-132](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L29) — current-trip reward/switch fields are attached from trial start.
- [src/continuum_swr_replay/analysis/replay/tables.py:777](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L777) — stop builder and canonical column order.
- [src/continuum_swr_replay/analysis/replay/tables.py:1277](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1277) — arm history attachment.
- [src/continuum_swr_replay/analysis/replay/tables.py:1296-1307](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1296) and [:1421-1450](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1421) — full-session recency transform and patch aggregation to replace.
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:74-75](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L74) and [:311-340](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L311) — integer source-epoch keys, exact poke/reward timestamps, and the model-inner-joined trial frame that must not define the complete behavior timebase.
- [src/continuum_swr_replay/analysis/replay/covariate_timing.py:27](../../../../src/continuum_swr_replay/analysis/replay/covariate_timing.py#L27) — timing registry to correct after materialization changes.
- [src/continuum_swr_replay/analysis/replay/model_specs.py:200](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L200) — prior decision to encode unvisited arms without a history indicator, now reversed by the audit.

**Contracts referenced:**

- [Covariate timing](shared-contracts.md#covariate-timing) — every post-outcome row must satisfy the physical timestamp gate.
- [Run artifact contract](shared-contracts.md#run-artifact-contract) — regenerated tables and audits live in a Phase-7 run.

**Designs referenced:** [trial timebase](designs.md#trial-timebase) and [post-outcome gating](designs.md#outcome-gating).

## Tasks

- Add a public, offline-testable `build_behavior_trial_table` that starts from the complete
  `TrialsInfoByEpoch.ByTrial`/trial-source rows, not `_per_trial`'s model inner join. Persist
  `session_epoch_crosswalk.parquet` first, with one-to-one source
  `(nwb_file_name, epoch_number)` and canonical `(session_id, animal, day, epoch)` keys; hash it into
  the table run. Join through that crosswalk—never filename construction or string splitting—then
  persist `behavior_trial_table.parquet`: one row per
  `(session_id, trial_number_by_epoch)` carrying both source/canonical epoch fields, exact poke/reward
  timestamps, chosen leaf/patch, observed outcome, nominal probabilities, and
  `outcome_known_time/source`. Assert source-row cardinality and preserve every non-Q terminal trial.
  Phase 9a owns the separate Beta-model table and its join.
- Implement `recent_completed_switch_rate` as a pure vectorized helper. Materialize
  `recent_completed_switch_rate_k5` at stop start in `stop_table`; compute it at event/bin query time in
  their builders. Remove `recent_switch_rate` from `add_uncertainty_covariates`, remove its `switch_window`
  API, and delete the fitted stop-row path. Delete unused stop-grain `wsls_deviation` and
  `leaky_q_value_diff` outputs rather than preserving misleading duplicates; retain only genuinely used,
  correctly defined covariates.
- Split the overloaded trial-context labels. Materialize
  `previous_completed_trial_is_switch` for predictive rate/content specifications and rename the
  eventual current-trip field to `current_trip_is_switch`, descriptive only. Likewise, current-trip
  reward/outcome fields must carry their `outcome_known_time`; predictive reward history continues to
  use `reward_lag*`. Remove `is_after_switch` and ambiguous `is_after_reward`/`is_after_omission`
  predictor use rather than keeping aliases.
- Add stable `context_trial_number`, `decision_trial_number`, and hashed `decision_id` to the canonical
  stop/event tables using the sorted trial timebase after the epoch crosswalk. Assert all stops/events
  with a finite `decision_id` agree with the behavior trial table's canonical session/trial key.
- Replace the ambiguous reward-history names with window-explicit fields across replay and ContFrag
  builders, model specs, visualizations, tests, and docs:
  `reward_count_arm_global_k10`, `visit_count_arm_global_k10`,
  `reward_count_arm_visits_k10`, `visit_count_arm_visits_k10`, and
  `reward_rate_arm_visits_k10`. The global pair uses the last ten completed trials; the visit pair/rate
  uses the last up-to-ten completed visits to that arm. Rename helpers and use an `rg` removal gate;
  do not retain aliases.
- Materialize `has_arm_history`; keep `reward_rate_arm_visits_k10=0` for an unvisited leaf but
  distinguish it with `visit_count_arm_visits_k10=0` and the indicator. A visited zero-reward arm has
  a positive denominator/history and rate 0. Add denominator and indicator to reward-history content
  and choice baselines; document that the rate coefficient is conditional on history/exposure.
- Replace the future-session sentinel and persisted demeaning with the exact recency design:
  `visit_recency_trials` is trials since the last completed visit, or
  `n_completed_trials_at_query + 1` for an unvisited leaf. Materialize `is_reward_site` and use zero
  for trunk history fields; pair leaf novelty with `has_arm_history`. Delete
  `visit_recency_demeaned`; persist raw `visit_recency_trials` only. The fitter applies `log1p` and
  learns scaling on the declared estimation cohort or each CV training fold in Phase 9b, never on the
  full materialized session.
- At patch grain, materialize the exact
  `reward_count_patch_global_k10`/`visit_count_patch_global_k10` and
  `reward_count_patch_visits_k10`/`visit_count_patch_visits_k10` windows,
  `has_patch_history`, and
  `reward_rate_patch_visits_k10 = reward_count_patch_visits_k10 /
  visit_count_patch_visits_k10` where exposure is positive. Recompute from completed patch visits;
  never average leaf-arm rates. Replace the P3/P4 reward-history term with visit-window rate plus
  exposure/history when the interpretation is experienced reward rate. Retain a global-count
  specification only under an explicit “recent rewarded exposure” sensitivity name.
- Carry `outcome_known_time` into event and per-bin designs. Gate G5/S7, G7/S9, C6/C9, MU post-outcome
  terms, and any power/Stage-6 consumer by actual row time. In `aggregate_to_timebins`, add
  `bin_start_time = stop_start + binj * bin_s` and `bin_end_time = bin_start_time + exposure`;
  construct trial/history covariates at the bin left edge instead of copying stop-start values. A
  post-outcome per-bin row requires `bin_start_time >= outcome_known_time`; drop a boundary-straddling
  bin and never gate on `tc`, which is the center. Keep exposure/count/self-history mechanics
  unchanged. Keep peri-switch rungs descriptive. Remove code/comments that treat `descriptive=True`
  alone as sufficient for current-outcome variables.
- Update every affected registry/spec/runner (`model_specs.py`, `covariate_timing.py`, SWR contrast
  runners, uncertainty runner), including timing assertions for runner-local MU/power specifications,
  `docs/covariate_dictionary.md`, `docs/models_table.md`, and
  `docs/design/content_model_design.md`. Explicitly invalidate the old switch-rate result in the docs.
- Capture before/after table validation slices: row/cardinality keys must match; record how many rows
  change switch history, enter/leave post-outcome cohorts, and change reward-history interpretation.
  Directional coefficient preservation is not an acceptance criterion.

## Deliberately not in this phase

- Choice/Stage-6 fitting on `decision_id` (Phase 11); this phase only supplies their canonical inputs.
- Common selection cohorts or model ranking (Phases 9a/9b).
- Refitting full production ladders; one-animal smoke only, followed by Phase 13's full run.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recent_completed_switch_rate_ignores_duplicate_stops` | duplicating/splitting stops leaves values unchanged at identical timestamps |
| `test_switch_rate_uses_only_completed_trials` | a future patch change cannot affect an earlier query; boundary uses `side="right"` |
| `test_previous_switch_and_current_trip_switch_are_distinct` | predictive context uses the last completed decision; current eventual choice is descriptive-only |
| `test_outcome_known_time_reward_and_omission` | reward uses finite pump onset; omission uses poke-in; source is recorded |
| `test_complete_trial_timebase_does_not_depend_on_model_rows` | deleting the final Beta-model rows leaves behavior-trial keys/timestamps unchanged and complete |
| `test_post_outcome_gate_at_event_and_bin_grain` | event-start and bin-left-edge gates retain exact-boundary/post rows; a bin straddling the boundary drops even when its center is post-outcome |
| `test_per_bin_history_is_queried_at_bin_start` | completed-trial switch/reward/history fields change only on eligible bin left edges and are not repeated from stop start |
| `test_epoch_crosswalk_is_bijective` | integer DataJoint and canonical string epoch keys map one-to-one; duplicates, missing rows, filename synthesis, and string-split inference fail |
| `test_decision_id_stable_and_unique` | same session/trial hashes identically; different sessions/trials do not collide in fixture |
| `test_unvisited_and_observed_zero_are_distinct` | both have reward rate 0 but different `has_arm_history` |
| `test_recency_is_suffix_and_duplicate_stop_invariant` | appending future trials/events or duplicating stops cannot change an earlier recency value |
| `test_recency_leaf_trunk_and_no_history_contract` | visited/unvisited leaves, trunks, and no-completed-trial queries receive the designed fields |
| `test_reward_history_windows_are_not_mixed` | global-trial and per-target-visit counts match hand calculations; visit-window rate reconstructs only from its own numerator/denominator |
| `test_patch_reward_rate_has_exact_exposure` | patch history is recomputed from patch visits; zero-history differs from observed-zero and no mean-of-arm-rates path remains |
| `rg "recent_switch_rate|cumulative_reward|visit_recency_demeaned|is_after_switch|reward_rate_arm_k10" src scripts tests docs --glob '!docs/reviews/**' --glob '!docs/known_issues/**' --glob '!docs/archive/**'` | zero legacy references in live code, schemas, tests, or current docs; historical audit/archive text is explicitly excluded |
| one-animal table smoke (slow) | exact keys/cardinalities, timing audit, and behavior-trial alignment all pass |

## Fixtures

Synthetic trials must include multiple stops in one trial, rewarded/omitted arrivals, same-patch and
cross-patch choices, an unvisited leaf, a visited zero-reward leaf, a trunk, unequal patch visit
exposure, appended future trials, a stop spanning an outcome boundary, and a final trial without a
next decision. Reuse one validated real-animal source bundle from Phase 7 for the slow validation.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- No stop-row shift/rolling path remains for fitted switch history.
- No current trip's eventual switch is presented as a lagged “after switch” predictor.
- The physical outcome-time gate is applied to rows, not merely represented by a metadata label.
- No table covariate depends on total future session length or a full-session mean.
- Patch value/rate claims condition on an explicit visit denominator/history; count-only terms are
  labeled exposure.
- Old misleading columns/functions are removed everywhere, including figures and docs.
- Same-arm/duplicate-stop fixtures exercise the original production failure mode.
- The before/after report explains cohort changes without treating numerical similarity as correctness.
- Every task is implemented, deliberately deferred model fitting stays out of scope, and old column/helper paths are removed.
- Validation tests pass, slow tests are marked, and fixtures reproduce duplicate-stop and boundary failures rather than tautologies.
- Production names/docstrings do not mention this plan or phase numbers; all listed docs are updated.
