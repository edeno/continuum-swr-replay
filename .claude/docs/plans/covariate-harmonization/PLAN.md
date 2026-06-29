# Covariate Harmonization Implementation Plan

**Status:** Not started.

Give each behavioral covariate **one canonical definition** (per
[docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md), decisions
D1–D5) computed by the table **builders** and written into the canonical tables every
model family reads — so the same construct is entered the same way across rate /
content / choice / coupling (interpret *within* an estimator class; see the dictionary
§A). Net behavioral changes: per-arm recent reward 5→10 visits (`reward_rate_arm_k10`);
add per-arm nominal value `p_reward_arm` (populating the deferred `nominal_reward_prob`);
**add** a fitted-α leaky-Q level (`q_current`/`q_arm`) **and keep** `leaky_q_value_diff`
(the per-stop margin) as a comparison; `recent_switch_rate` canonical with
`is_after_switch` kept as a comparison. Implemented by **rebuilding the canonical tables**
(replay **and** contfrag, reusing existing decodes — no re-decode), then re-fitting all
families and verifying unchanged-definition covariates via design-matrix identity.

## Reading order

1. **Working a phase?** Open its file — each is self-contained (inputs, tasks, validation).
2. **Need the canonical definitions / entry-rule contract?** [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the spec) and [shared-contracts.md](shared-contracts.md) (names + which canonical-table columns they live in + the rebuild-integrity contract).
3. **Need scope / integration points / risks / the key approach decision?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, the rebuild-vs-backfill decision.
- [shared-contracts.md](shared-contracts.md) — the canonical covariate names + which canonical-table columns they live in + the rebuild-integrity safety contract.
- Phases (each ships as a separable PR):
  - [phase-0-extract-covariates.md](phase-0-extract-covariates.md) — behavior-preserving: consolidate the covariate functions + promoted Q-learner into `covariates.py` with a re-export shim (no definition changes).
  - [phase-1-canonical-covariate-layer.md](phase-1-canonical-covariate-layer.md) — change the definitions in `covariates.py` to canonical (D1–D5), update **both** table builders (replay + contfrag) + regenerate the tables + update model specs/runners + tests + smoke fits.
  - [phase-2-refit-verify.md](phase-2-refit-verify.md) — re-fit all families on current tables, verify parity for unchanged covariates, refresh docs/claims numbers.
  - [phase-3-canonical-is-swr-sweep.md](phase-3-canonical-is-swr-sweep.md) — point the ancillary analyses (reality, occupancy, PFC) at the canonical event table's `is_swr` via one adapter + divergence tests, so every claim rests on one event definition. (Independent of phases 0–2.)
  - [phase-4-promote-content-models.md](phase-4-promote-content-models.md) — promote the exploratory content scripts (`_content_reward_interactions`, `_content_patch_destination`) into the audited pipeline with full diagnostics, so claim #19f + patch-destination become non-exploratory. (Depends on phase 2.)
