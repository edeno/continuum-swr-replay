# Comprehensive Regression-Validity Remediation Implementation Plan

**Status:** Not started.

This continuation plan turns the 2026-07-10 cross-family statistical audit into a fail-closed,
decision-grain, dependence-aware regression pipeline and replaces all mixed-generation result
artifacts with one promoted canonical run.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific active phase?** Open the matching phase-7 through phase-13 file.
2. **Need shared schemas or validity rules?** [shared-contracts.md](shared-contracts.md).
3. **Need an algorithm or exact estimand?** [designs.md](designs.md).
4. **Need broader scope, integration points, dependencies, or risks?** [overview.md](overview.md).
5. **Working on PFC?** Use the dedicated [PFC Fig-4 recut plan](../pfc-fig4-recut/PLAN.md);
   this plan only defines its integration gate.

## Files

- [overview.md](overview.md) — active scope, locked scientific decisions, dependencies, risks, and rollout
- [shared-contracts.md](shared-contracts.md) — artifact, cohort, fold, decision, validity, and population-summary contracts
- [designs.md](designs.md) — canonical algorithms for time alignment, comparison cohorts, robust inference, decision models, and SWR power
- Active phases (each ships as a separable PR):
  - [phase-7-run-artifacts.md](phase-7-run-artifacts.md) — versioned manifests, atomic promotion, and fail-closed loading
  - [phase-8-covariate-timebase.md](phase-8-covariate-timebase.md) — trial-grain switch history, outcome timing, and accurate reward-history semantics
  - [phase-9a-cohort-identity.md](phase-9a-cohort-identity.md) — exact comparison cohorts and runner wiring
  - [phase-9b-blocked-comparisons.md](phase-9b-blocked-comparisons.md) — shared temporal folds and strict AIC/LRT/CV contracts
  - [phase-10a-estimability.md](phase-10a-estimability.md) — centered diagnostics, rank gates, and one fit-validity policy
  - [phase-10b-dependence-inference.md](phase-10b-dependence-inference.md) — cluster/bootstrap uncertainty and animal-balanced summaries
  - [phase-11-decision-models.md](phase-11-decision-models.md) — replace stop-grain choice and Stage-6 models with decision-grain analyses
  - [phase-12-swr-power.md](phase-12-swr-power.md) — identified event-power models and PFC integration gate
  - [phase-13-refit-reporting.md](phase-13-refit-reporting.md) — clean production refit, promotion, model comparison, and claims cutover
- Historical baseline files retained for audit trail (do not execute as active work):
  - [phase-1-labeling-fixes.md](phase-1-labeling-fixes.md)
  - [phase-2-fit-diagnostics.md](phase-2-fit-diagnostics.md)
  - [phase-3-cohort-audit-tables.md](phase-3-cohort-audit-tables.md)
  - [phase-4-covariate-causal-hygiene.md](phase-4-covariate-causal-hygiene.md)
  - [phase-5-inference-validity.md](phase-5-inference-validity.md)
  - [phase-6-repro-plumbing.md](phase-6-repro-plumbing.md)
