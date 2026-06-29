# Covariate Harmonization Implementation Plan

**Status:** Not started.

Make the behavioral covariates **comparable across the rate / content / choice /
coupling model families** by giving each construct one canonical definition (per
[docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md), decisions
D1–D5) and feeding every family from a single canonical covariate source, then
re-fitting and verifying that covariates whose definition did *not* change reproduce
their current coefficients within tolerance. Net behavioral changes: per-arm recent
reward 5→10 visits; add per-arm nominal value `p_reward_arm`; replace
`leaky_q_value_diff` (α=0.3) with a fitted-α leaky-Q value (`q_current`/`q_arm`);
standardize exploration on `recent_switch_rate` while keeping `is_after_switch` as a
comparison.

## Reading order

1. **Working a phase?** Open its file — each is self-contained (inputs, tasks, validation).
2. **Need the canonical definitions / entry-rule contract?** [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the spec) and [shared-contracts.md](shared-contracts.md) (the precompute schema + naming).
3. **Need scope / integration points / risks / the key approach decision?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, the precompute-vs-regen decision.
- [shared-contracts.md](shared-contracts.md) — the canonical covariate names + which canonical-table columns they live in (no sidecar) + the in-place-update safety contract.
- Phases (each ships as a separable PR):
  - [phase-1-canonical-covariate-layer.md](phase-1-canonical-covariate-layer.md) — build the canonical covariate source + update model specs/runners to consume it + tests + smoke fits.
  - [phase-2-refit-verify.md](phase-2-refit-verify.md) — re-fit all families on current tables, verify parity for unchanged covariates, refresh docs/claims numbers.
