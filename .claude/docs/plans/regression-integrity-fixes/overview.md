# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Source audit: [07_comprehensive_regression_model_statistical_audit.md](../../reviews/07_comprehensive_regression_model_statistical_audit.md).

## Current codebase integration points

- [`analysis/replay/covariates.py:456-548`](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L456) — `add_uncertainty_covariates` currently rolls `is_stay` over stop rows; Phase 8 removes that path.
- [`analysis/replay/tables.py:777-1009`](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L777) — canonical stop builder; Phase 8 adds stable trial/decision context and stop-start history.
- [`analysis/behavior/qvals_table.py:380-430`](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L380) — exact poke/reward timestamps and stop-to-trial mapping; Phase 8 also emits the canonical per-trial decision-covariate table.
- [`analysis/replay/cohort_audit.py:25-118`](../../../../src/continuum_swr_replay/analysis/replay/cohort_audit.py#L25) — landed drop-accounting infrastructure; Phase 9a extends it with exact identity fields.
- [`scripts/content/compare_event_source_models.py:198-383`](../../../../scripts/content/compare_event_source_models.py#L198) — useful script-local cohort hashing/freezing implementation moved to shared code in Phase 9a.
- [`analysis/replay/model_checking.py:21-164`](../../../../src/continuum_swr_replay/analysis/replay/model_checking.py#L21) — content CV currently blocks by event rather than stop; Phase 9b replaces its fold construction.
- [`analysis/replay/per_bin_rate_model.py:129-302`](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L129) — model-specific row dropping, random stop folds, and no Patsy-design rank gate; Phases 9a, 9b, and 10a modify distinct parts.
- [`analysis/replay/model_comparison.py:20-157`](../../../../src/continuum_swr_replay/analysis/replay/model_comparison.py#L20) — comparison helper checks counts rather than identities and clips invalid LR inputs; Phase 9b replaces that contract.
- [`analysis/replay/fitting.py:37-64`](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L37) and [`:437-454`](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L437) — conditional rank uses within-group centering, but VIF/condition diagnostics do not; Phase 10a unifies them.
- [`analysis/replay/choice_prediction.py:246-418`](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L246) — stop-grain, replay-selected, non-current-arm choice-set construction deleted in Phase 11.
- [`analysis/replay/learning.py:383-643`](../../../../src/continuum_swr_replay/analysis/replay/learning.py#L383) — stop-grain Stage-6 Logit/GEE deleted in Phase 11.
- [`scripts/rate/fit_swr_power_model.py:45-161`](../../../../scripts/rate/fit_swr_power_model.py#L45) — unidentified OLS and session-Wilcoxon reporting replaced in Phase 12.
- [`analysis/replay/contrast_aggregation.py:57-135`](../../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L57) — current animal-balanced pattern and term-only coefficient filter; Phase 10b generalizes aggregation after Phase 10a defines fit validity.
- [`analysis/pfc/glm.py:930-1060`](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L930) — PFC event construction and GLMs are owned by the dedicated PFC plan, not duplicated here.

## Scope and dependency policy

### Goals

- Every promoted result is traceable to one complete run, exact inputs/specifications, exact likelihood rows, and exact folds.
- Every likelihood-based comparison either proves row/fold compatibility or fails with a reason; no plausible-looking invalid AIC/LRT/CV values are emitted.
- All fitted covariates have accurate names, trial grain, and time availability relative to their response.
- Core content/rate estimates have design-rank gates, dependence-aware uncertainty, and animal-balanced summaries.
- Choice and Stage-6 operate on one observation per actual behavioral decision and include zero-replay and same-arm decisions.
- SWR-power coefficients come from algebraically identified specifications with residual/rank/uncertainty diagnostics.
- One clean refit replaces all stale/mixed-generation results and regenerates model tables, figures, and claims.

### Non-Goals

- No re-decoding or detector change; canonical event labels and posterior outputs are inputs.
- No causal mediation claim for Stage 6. It remains a post-outcome behavioral association.
- No claim that five animals provide high-powered population inference; the plan improves the unit of analysis, not sample size.
- No attempt to infer value for the trailing trials without fitted behavior-model state; those rows remain an explicit definitional exclusion.
- No duplicate PFC implementation here. Canonical events, controls, GLM mechanics, dependence-aware inference, and Figure 4 remain in [the PFC plan](../pfc-fig4-recut/PLAN.md).
- No backward-compatible readers, aliases, old column names, or legacy artifact fallback. The user explicitly chose replacement in place.

### Dependency policy

No new runtime dependency is required. Use NumPy, pandas, Patsy/statsmodels, SciPy, and scikit-learn already in the environment. Bootstrap/permutation routines are local, seeded implementations. PFC work follows its existing plan and can proceed in parallel until Phase 13's promotion gate.

## Locked decisions

1. **Artifact cutover:** versioned run directories plus an atomic `CURRENT.json`; fixed-path legacy outputs are never read by canonical consumers.
2. **Fit purposes:** `estimation` uses the scientifically appropriate maximal cohort; `selection` uses a declared exact common cohort/folds. Only `selection` outputs support AIC/BIC/LRT/CV ranking.
3. **Timing:** post-outcome RPE/update analyses are physically gated after the recorded outcome-known timestamp. Future-referencing peri-switch variables remain descriptive only.
4. **Switch history:** compute over completed trials at the query timestamp, never over stop rows; rename it `recent_completed_switch_rate_k5`.
5. **Reward history:** rename `cumulative_reward` to `reward_count_arm_k10`; add `has_arm_history` so unvisited and observed-zero arms are distinct.
6. **Content primary:** the stable arm M4-style spatial/reward-history specification is primary; sparse last-rewarded and value/uncertainty terms are prespecified sensitivities. Mvu is not a headline exploit-versus-explore model.
7. **Content response:** soft posterior mass is the primary robustness target alongside the existing hard-count estimate; agreement is required for a content claim.
8. **Decision window:** for target trial `t`, aggregate replay in `[outcome_known_time(t-1), poke_out_time(t-1))`; use trial `t`'s pre-outcome value vector and all six leaf arms. Decisions with no replay stay in the dataset.
9. **Population summary:** all five animal effects and an animal-balanced center/interval are primary; session/neuron signed-rank summaries are descriptive.
10. **SWR power:** primary decomposition is realized reward + pre-outcome Q. RPE-alone is a separate sensitivity; reward, Q, and RPE are never co-fit.

## Metrics

- No canonical consumer can load a missing, incomplete, stale-schema, mixed-run, or hash-mismatched artifact.
- Every selection comparison has identical `cohort_hash`, `group_hash`, and, for CV, `fold_hash`; mismatches raise.
- Same-N/different-ID, partial candidate-set, duplicate-key, nonnested-LRT, negative-LR, and mixed-run fixtures all fail closed.
- `recent_completed_switch_rate_k5` is unchanged by adding duplicate stops to a trial and depends only on trials completed by the query time.
- Every post-outcome fitted row satisfies `response_time >= outcome_known_time`; every pre-outcome row fails the gate.
- Conditional diagnostics are invariant to event-level offsets and predictor rescaling; all fitted designs report rank and scaled condition diagnostics.
- Rank-deficient/nonconverged fits never enter comparisons or headline summaries; excluded cells remain explicitly represented with failure reasons.
- Dependence-aware simulation intervals improve nominal coverage over model-based intervals under the repository's autocorrelated trajectory fixtures.
- Decision tables contain one row per `(decision_id, arm)`, six arms and exactly one chosen arm per decision, including same-arm and zero-replay decisions.
- SWR-power deterministic-identity fixtures are rejected and planted identified effects are recovered within stated Monte Carlo tolerance.
- The production promotion audit reports complete expected model/population/session coverage and all five animals before updating `CURRENT.json`.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Common cohorts shrink high-dimensional ladders and change apparent winners. | Keep maximal-cohort estimation separate; rank only declared selection sets and report retention/hash. |
| The trial/decision rebuild changes many headline numbers. | Save pre-change descriptive baselines, smoke one animal, and publish before/after deltas without requiring direction preservation. |
| Stop/block bootstrap is computationally expensive. | Fixed-seed small-bootstrap CI tests in CI; smoke one session with timing/memory, then choose the production replicate count before the full run. |
| Five-animal intervals remain wide. | Show all five effects and leave-one-animal-out sensitivity; do not substitute session p-values. |
| Penalized sparse-term sensitivity cannot use ordinary AIC/LRT. | Label it `estimation` only and compare by shared-fold predictive score; never mix it with unpenalized likelihood tests. |
| Atomic-run migration touches many plotting scripts. | Centralize resolution/validation first, migrate consumers in Phase 7, and test interrupted promotion before any number-changing phase. |
| PFC work drifts from regression promotion. | Phase 12 records an explicit PFC completion contract; Phase 13 refuses to promote cross-area claims until the dedicated plan's manifest passes. |

## Rollout Strategy

Replace in place with no compatibility layer. Execute in order:

`7 → 8 → 9a → 9b → 10a → 10b → 11 → 12 → 13`.

The dedicated PFC plan may run in parallel after Phase 7 establishes the shared manifest fields, but must finish before Phase 13 promotes Figure-4/cross-area claims. Each number-changing phase smoke-tests synthetic data first and then one real session/animal. Only Phase 13 runs the full canonical analysis and atomically advances `CURRENT.json`.

## Open Questions

None. The scientific and compatibility choices needed to execute these phases are locked above. Unexpected empirical failures are reported as failed validation gates, not resolved by silently changing the estimand.

## Estimated Effort

Approximately 1,400-2,000 changed LOC plus tests across the nine active PRs, excluding the separately planned PFC implementation and production compute. The largest diffs are Phase 7 consumer migration, Phase 10b bootstrap/aggregation, and Phase 11 decision-table replacement.
