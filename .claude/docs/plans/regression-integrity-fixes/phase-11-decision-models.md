# Phase 11 — Decision-grain choice and Stage-6 models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#decision-models)

Delete the stop-grain behavioral analyses and rebuild them around one canonical replay window per
actual choice. Include all six well arms, same-arm choices, zero-replay decisions, pre-outcome value and
geometry controls, exact common cohorts, and blocked held-out scores.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:246-418](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L246) — current replay-selected stop dataset, current-arm exclusion, and silent invalid-group drop.
- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:423-525](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L423) — conditional-logit fitter to retain only after generalizing group name/diagnostics.
- [src/continuum_swr_replay/analysis/replay/model_specs.py:760-822](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L760) — old C models and geometry caveat to replace.
- [src/continuum_swr_replay/analysis/replay/learning.py:125-370](../../../../src/continuum_swr_replay/analysis/replay/learning.py#L125) — stop replay summary.
- [src/continuum_swr_replay/analysis/replay/learning.py:383-643](../../../../src/continuum_swr_replay/analysis/replay/learning.py#L383) — old stop Logit/session-GEE path to delete.
- [scripts/pfc/run_stage6_learning.py:130-195](../../../../scripts/pfc/run_stage6_learning.py#L130) — stale output and session-Wilcoxon reporting; move replacement runner under `scripts/behavior/`.
- [scripts/content/compare_event_source_models.py:477-493](../../../../scripts/content/compare_event_source_models.py#L477) — current paper-facing choice block wiring.

**Contracts referenced:**

- [Decision table](shared-contracts.md#decision-table) — exact six-row choice set and one-row summary.
- [Comparison identity](shared-contracts.md#comparison-identity) and [folds](shared-contracts.md#fold-assignment) — D0/D1 rows/folds are exact.
- [Fit validity](shared-contracts.md#fit-validity) and [animal summary](shared-contracts.md#animal-summary).

**Designs referenced:** [decision-grain replay models](designs.md#decision-models).

## Tasks

- Add `analysis/replay/decision_tables.py` building `decision_arm_table` and
  `decision_summary_table` from the Phase-8 behavior trial table and canonical event/arm tables. Enforce
  the risk window, six candidate arms, one chosen arm, stable `decision_id`, and zero-filled absent replay
  mass. Include an event only when
  `event_start_time >= outcome_known_time[t-1]` and
  `event_end_time <= poke_out_ts[t-1]`; retain exact-boundary equality, require positive duration, and
  drop events that straddle either boundary. Aggregate soft posterior mass as primary; carry hard-count
  equivalents and all/SWR/non-SWR mass.
- Include every eligible decision, not only replay-containing stops. Report exclusions by reason:
  missing behavior-model state, invalid timestamps, first/no-prior trial, or incomplete candidate
  covariates. Same-arm choices are ordinary retained outcomes.
- Replace `MODEL_CHOICE` with D0/D1/D2 specs from the design. Include Beta-model `q_arm` and distance rank in
  the primary baseline, plus raw-recency's fit-time `log1p` transform,
  `reward_rate_arm_visits_k10`, `visit_count_arm_visits_k10`, and `has_arm_history`; standardize
  continuous terms on training data within each fold. Add nominal-Q and hard-mass sensitivity specs
  without co-fitting mechanically redundant value constructs.
- Generalize `fit_choice_model` to `decision_id`, Phase-10a diagnostics, structured failure rows, robust
  metadata, and held-out conditional log likelihood using Phase-9b fold assignments. Fit D0/D1 on an
  exact common cohort; formally test the two split replay coefficients against each other rather than
  comparing their separate p-values.
- Delete `build_choice_dataset`'s stop-grain construction and old C specs. Update
  `compare_event_source_models.py` or replace its choice block with the decision-run artifact; do not
  preserve an alternate switch-conditional path.
- Replace Stage-6 stop summaries/fits with a decision-summary model under the prespecified descriptive
  formula. Include previous outcome, value advantage, completed-switch history, window duration, and
  trial/block context. Use per-animal/session-aware estimation and Phase-10b summaries; label the result
  association, never learning mediation or replay gating.
- Move `run_stage6_learning.py` from `scripts/pfc/` to `scripts/behavior/` with a new decision-grain name,
  remove the old output files/specs, and register both decision analyses in Phase-7 run manifests.
- Update `docs/models_table.md`, `docs/covariate_dictionary.md`, claims/figure plans, and method text.
  Report calibration, held-out log score, decision counts, same-arm counts, zero-replay counts, and all
  five animal effects before any paper-facing replay-choice claim.

## Deliberately not in this phase

- A causal replay→choice or mediation claim; no observational regression here identifies it.
- Imputing the behavior-model tail. Value-controlled models define their cohort explicitly.
- PFC code despite the legacy Stage-6 script location.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_decision_table_has_six_arms_one_choice` | exact key/cardinality contract for every retained decision |
| `test_same_arm_and_zero_replay_decisions_retained` | both original selection failures remain valid rows |
| `test_repeated_stops_collapse_to_one_decision` | adding 1 or 236 stops cannot duplicate the response |
| `test_replay_window_requires_full_event_containment` | exact-boundary events retain; pre-outcome, postdeparture, and either-boundary-straddling events drop |
| `test_choice_uses_target_trial_preoutcome_values` | value vector is trial `t`, not stale trial `t-1` or future outcome |
| `test_choice_cv_common_rows_folds_and_calibration` | D0/D1 exact hashes match and held-out score/calibration are produced |
| `test_split_replay_difference_is_formal_contrast` | equality contrast is computed from one fitted model |
| `test_stage6_formula_uses_one_row_per_decision` | no stop identifier or duplicate target enters likelihood |
| one-animal decision smoke (slow) | counts reconcile to behavior trials; same-arm/zero-replay rates reported |

## Fixtures

Synthetic trial sequence includes same-arm repeat, patch stay/switch, reward and omission, two stops for
one decision, no-replay window, SWR-only and non-SWR-only windows, exact-start/exact-end events,
pre-outcome and postdeparture straddlers, and missing Q on the final trial. The slow fixture uses one
animal's behavior and canonical event tables.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- Exactly one likelihood choice set/summary row represents each actual decision.
- Same-arm and zero-replay decisions are not hidden exclusions.
- Replay always precedes the target choice and value covariates are pre-outcome for that target trial.
- The old stop-grain builders/specs/runner/output readers are deleted, not retained as a compatibility path.
- “Prediction” appears only with held-out metrics; Stage 6 is consistently labeled descriptive association.
- Every task is implemented, causal claims remain out of scope, and old stop-grain builders/specs/runners are removed.
- Validation tests pass, the real-data smoke is marked slow, and same-arm/zero-replay fixtures are substantive.
- Production names/docstrings do not mention this plan or phase numbers; all behavioral docs/claims listed above are updated.
