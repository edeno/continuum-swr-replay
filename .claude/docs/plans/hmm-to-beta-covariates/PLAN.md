# HMM → beta-bernoulli covariate migration

**Status:** Complete (2026-07-13). Phases 1–5A + doc reconciliation + PR-review
hardening executed and merged into `covar-harmon-phase1-definitions` (merge
`82cdabf`). All HMM behavioral covariates dropped, beta equivalents fitted (218
sessions), full test suite green (948 passed).

The behavioral model has switched from the HMM to the beta-bernoulli fit, and the
paper now uses the beta-bernoulli belief. This plan removes every HMM-derived
behavioral covariate (`q_arm_hmm`, `hmm_relative_value`, `hmm_rpe`,
`hmm_value_update`, `reward_entropy`, `state_entropy`) from the rate and content
model ladders and the coefficient figures, replacing each with a beta-bernoulli
equivalent (value `q_arm`, relative value / RPE / value-update from the vendored
beta decision variables, and a derived beta belief-entropy scalar). Uses the beta
fit already ingested (`q_vals_*_beta_stable_withleaf.csv`) — no re-ingest. After
migration, ladders are re-fit, the coefficient forest plot regenerated, and the
results docs/memory reconciled to the beta value sign (content *avoids* high value,
where the HMM had said it *prefers*).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained.
2. **Need the HMM→beta name mapping or the beta-entropy definition?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / integration points?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, risks, rollout
- [shared-contracts.md](shared-contracts.md) — canonical HMM→beta covariate mapping + beta belief-entropy definition
- Phases (each ships as a separable PR):
  - [phase-1-beta-covariates.md](phase-1-beta-covariates.md) — standardize/attach beta equivalents; regenerate tables with zero-drift verification
  - [phase-2-model-specs-swap.md](phase-2-model-specs-swap.md) — swap HMM names in ladders + timing registry + runners; delete HMM attach/fetch; re-fit
  - [phase-3-coefficient-plot.md](phase-3-coefficient-plot.md) — swap HMM literals in the coefficient forest plot; regenerate figures
  - [phase-4-docs-reconciliation.md](phase-4-docs-reconciliation.md) — flip value narrative to beta sign; drop construct-dependence apparatus; update dictionary/synthesis/memory
  - [phase-5-beta-exploit-explore.md](phase-5-beta-exploit-explore.md) — new science: within-arm beta value×uncertainty exploit/explore content model; promote the exploratory estimator; add a numbered claim
