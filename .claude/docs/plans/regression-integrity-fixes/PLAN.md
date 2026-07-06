# Regression-Integrity & Provenance Fixes — Implementation Plan

**Status:** Not started.

This plan implements the fixes from the 2026-07-04 data-provenance / regression-integrity audit
(`.claude/docs/reviews/06_data_provenance_regression_audit.md`) and a convergent independent review.
The pipeline's relational plumbing and leakage quarantine are already sound; this work fixes
(1) stale result labels, (2) uneven fit diagnostics, (3) unreported/non-representative complete-case
cohorts, (4) covariates that are not strictly pre-outcome used as predictors, (5) session-level
pseudoreplication in the base ladders, and (6) reproducibility hazards in the canonical builder.
Nothing here requires re-decoding; the heaviest cost is table/model regeneration.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / dependency policy / the two open scientific decisions?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, risks, rollout, the two open estimand decisions
- [shared-contracts.md](shared-contracts.md) — cohort-audit-table schema; covariate pre/post-outcome timing reference
- Phases (each ships as a separable PR):
  - [phase-1-labeling-fixes.md](phase-1-labeling-fixes.md) — correct stale `q_arm_hmm` provenance, choice-doc claim, interval docstring
  - [phase-2-fit-diagnostics.md](phase-2-fit-diagnostics.md) — convergence + separation checks; scope down warning suppression
  - [phase-3-cohort-audit-tables.md](phase-3-cohort-audit-tables.md) — per-model retained-N + drop-profile emitted with every fit run
  - [phase-4-covariate-causal-hygiene.md](phase-4-covariate-causal-hygiene.md) — pre/post-outcome split, `reward_lag1` swap, value-rung time-gating, mechanism-A encoding
  - [phase-5-inference-validity.md](phase-5-inference-validity.md) — per-animal aggregation as primary for base ladders; common-cohort + `swr_x_value` cohort/construct sensitivity
  - [phase-6-repro-plumbing.md](phase-6-repro-plumbing.md) — fold `_z`/`q_arm_hmm` into canonical builder; document SWR/MUA count convention
