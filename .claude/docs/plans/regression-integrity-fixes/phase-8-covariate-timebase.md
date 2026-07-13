# Phase 8 — Canonical trial timebase and covariate repair

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#trial-timebase)

Replace stop-row-derived behavioral history with exact trial/timestamp covariates, add an explicit
outcome-known time, distinguish unvisited from observed-zero arms, and remove misleading names. This
is the first number-changing phase and regenerates canonical tables only after synthetic and one-animal
validation.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/covariates.py:126](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L126) — last-ten-trial count currently named cumulative reward.
- [src/continuum_swr_replay/analysis/replay/covariates.py:456](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L456) — stop-row switch calculation and unused stop-grain WSLS/leaky-Q outputs.
- [src/continuum_swr_replay/analysis/replay/tables.py:777](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L777) — stop builder and canonical column order.
- [src/continuum_swr_replay/analysis/replay/tables.py:1277](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1277) — arm history attachment.
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:90](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L90) and [:380](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L380) — exact poke/reward timestamps and currently stop-only output.
- [src/continuum_swr_replay/analysis/replay/covariate_timing.py:27](../../../../src/continuum_swr_replay/analysis/replay/covariate_timing.py#L27) — timing registry to correct after materialization changes.
- [src/continuum_swr_replay/analysis/replay/model_specs.py:200](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L200) — prior decision to encode unvisited arms without a history indicator, now reversed by the audit.

**Contracts referenced:**

- [Covariate timing](shared-contracts.md#covariate-timing) — every post-outcome row must satisfy the physical timestamp gate.
- [Run artifact contract](shared-contracts.md#run-artifact-contract) — regenerated tables and audits live in a Phase-7 run.

**Designs referenced:** [trial timebase](designs.md#trial-timebase) and [post-outcome gating](designs.md#outcome-gating).

## Tasks

- Refactor `qvals_table.py` so `_per_trial` has a public, offline-testable table builder and persist
  `behavior_trial_table.parquet`: one row per `(animal, day, epoch, trial_number_by_epoch)` with exact
  poke/reward timestamps, chosen leaf/patch, outcome, pre-outcome Beta/HMM value vectors, nominal
  probabilities, entropy/update variables, and `outcome_known_time/source` per the design.
- Implement `recent_completed_switch_rate` as a pure vectorized helper. Materialize
  `recent_completed_switch_rate_k5` at stop start in `stop_table`; compute it at event/bin query time in
  their builders. Remove `recent_switch_rate` from `add_uncertainty_covariates`, remove its `switch_window`
  API, and delete the fitted stop-row path. Delete unused stop-grain `wsls_deviation` and
  `leaky_q_value_diff` outputs rather than preserving misleading duplicates; retain only genuinely used,
  correctly defined covariates.
- Add stable `context_trial_number`, `decision_trial_number`, and hashed `decision_id` to the canonical
  stop/event tables using the sorted trial timebase. Assert all stops/events with a finite `decision_id`
  agree with the behavior trial table's session/trial key.
- Rename `cumulative_reward` → `reward_count_arm_k10` and `cumulative_reward_patch` →
  `reward_count_patch_k10` across builders, model specs, visualizations, tests, and docs. Rename the helper
  functions as well. Use an `rg` removal gate; do not retain aliases.
- Materialize `arm_visit_count_prior_k10` and `has_arm_history`; keep `reward_rate_arm_k10=0` for an
  unvisited leaf but distinguish it with the indicator. Add the indicator to reward-history content and
  choice baselines; document that the reward-rate coefficient is conditional on history availability.
- Carry `outcome_known_time` into event and per-bin designs. Gate G5/S7, G7/S9, C6/C9, MU post-outcome
  terms, and any power/Stage-6 consumer by actual row time. Keep peri-switch rungs descriptive. Remove
  code/comments that treat `descriptive=True` alone as sufficient for current-outcome variables.
- Update every affected registry/spec/runner (`model_specs.py`, `covariate_timing.py`, SWR contrast
  runners, uncertainty runner), `docs/covariate_dictionary.md`, `docs/models_table.md`, and
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
| `test_outcome_known_time_reward_and_omission` | reward uses finite pump onset; omission uses poke-in; source is recorded |
| `test_post_outcome_gate_at_event_and_bin_grain` | pre-time rows drop, boundary/post rows retain, and partial stops are not mislabeled |
| `test_decision_id_stable_and_unique` | same session/trial hashes identically; different sessions/trials do not collide in fixture |
| `test_unvisited_and_observed_zero_are_distinct` | both have reward rate 0 but different `has_arm_history` |
| `rg "recent_switch_rate|cumulative_reward" src scripts tests docs` | zero live-code/schema references after intentional historical docs are excluded |
| one-animal table smoke (slow) | exact keys/cardinalities, timing audit, and behavior-trial alignment all pass |

## Fixtures

Synthetic trials must include multiple stops in one trial, rewarded/omitted arrivals, same-patch and
cross-patch choices, an unvisited arm, a visited zero-reward arm, a stop spanning an outcome boundary,
and a final trial without a next decision. Reuse real one-animal cached tables for the slow validation.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- No stop-row shift/rolling path remains for fitted switch history.
- The physical outcome-time gate is applied to rows, not merely represented by a metadata label.
- Old misleading columns/functions are removed everywhere, including figures and docs.
- Same-arm/duplicate-stop fixtures exercise the original production failure mode.
- The before/after report explains cohort changes without treating numerical similarity as correctness.
- Every task is implemented, deliberately deferred model fitting stays out of scope, and old column/helper paths are removed.
- Validation tests pass, slow tests are marked, and fixtures reproduce duplicate-stop and boundary failures rather than tautologies.
- Production names/docstrings do not mention this plan or phase numbers; all listed docs are updated.
