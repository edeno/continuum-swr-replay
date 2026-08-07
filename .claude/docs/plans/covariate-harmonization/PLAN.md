# Covariate Harmonization Implementation Plan

**Status (updated 2026-07-01):**
- **Phase-0 — COMPLETE, merged** to `master` at `54e9ed7` (2026-06-29). Extracted
  `covariates.py` + re-export shim behavior-preservingly (`3c236bf`, reviewed/revised
  `54e9ed7`; 11 functions + `UNCERTAINTY_COVARIATES` byte-identical; 989 passed).
  Earlier: baseline `f6c5585`; pre-merge cleanup `5c2becf` (red tests greened, `TMAX`
  ceil/exposure fix, guards).
- **Phase-1 — COMPLETE** on branch `covar-harmon-phase1-definitions` (**not yet
  merged**). Canonical definitions (`reward_rate_arm_k10` 5→10, `p_reward_arm`
  populated), both builders updated, canonical tables **regenerated** (replay + contfrag
  SWR/HPC), specs/runners/tests updated, smoke fits green.
- **Phase-1.5 — COMPLETE** (same branch): **behavior-model (RL) covariate
  integration** — see [phase-1.5](phase-1.5-behavior-model-covariates.md). This
  **supersedes D3-as-planned**: `q_current`/`q_arm` are the external Beta-Bernoulli
  **model belief** from DataJoint `qvals_table`, not a promoted fitted-α leaky-Q;
  `_bandit_csv_contingency.py` was **deleted**, not promoted; the nominal block also
  moved to `qvals_table`. Adds the accuracy-audit + invariant-bounds verification layer
  and the base-covariate catalog. Multi-agent reviewed; tests green.
- **Phase-2 — PARTIAL.** Only `uncertainty_rate` refit; the full refit of the other
  families + claim-number refresh were not completed here. **The remaining refit,
  verification, and claims cutover are superseded by**
  [`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md);
  do not execute this phase independently. Its stop-grain switch and unseen-arm
  assumptions are corrected by that plan's Phase 8.
- **Phase-3 — NARROWED / NOT started.** Only the non-PFC ancillary reality/occupancy
  `is_swr` adapters remain separate, after regression Phase 13. PFC event plumbing is
  owned by [`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md).
- **Phase-4 — SUPERSEDED.** Its generic diagnostics/cohort/refit work and proposed
  content-model promotion are now owned by regression-integrity Phases 8–10 and 13.
  Do not execute the old phase file; reconsider any still-novel exploratory
  destination estimand only after the corrected canonical run exists.

**Ownership refresh (2026-07-29):** this plan remains the historical owner of the
landed covariate extraction/harmonization. The comprehensive
[`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md) plan is the
sole owner of corrective covariate definitions and the coherent final refit.
The narrowed Phase 3 remains a separate optional analysis and must consume its
promoted outputs. Phase 4 is retained only as historical design context.

This plan originally set out to give each behavioral covariate one builder-owned
definition shared across rate/content/choice/coupling. The completed Phase 1/1.5
result uses the external Beta-Bernoulli `qvals_table` belief for `q_current`/`q_arm`
(not the superseded fitted-α leaky-Q proposal) and rebuilt the replay and ContFrag
canonical tables without re-decoding. Its historical `recent_switch_rate` /
`is_after_switch` decisions and its unfinished refit/parity work are not the final
contract; regression-integrity Phases 8 and 13 replace them.

## Reading order

1. **Working a phase?** Open its file — each is self-contained (inputs, tasks, validation).
2. **Need the canonical definitions / entry-rule contract?** [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the spec) and [shared-contracts.md](shared-contracts.md) (names + which canonical-table columns they live in + the rebuild-integrity contract).
3. **Need scope / integration points / risks / the key approach decision?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, the rebuild-vs-backfill decision.
- [shared-contracts.md](shared-contracts.md) — the canonical covariate names + which canonical-table columns they live in + the rebuild-integrity safety contract.
- Phases (each ships as a separable PR):
  - [phase-0-extract-covariates.md](phase-0-extract-covariates.md) — behavior-preserving: consolidate the covariate functions + promoted Q-learner into `covariates.py` with a re-export shim (no definition changes).
  - [phase-1-canonical-covariate-layer.md](phase-1-canonical-covariate-layer.md) — change the definitions in `covariates.py` to canonical (D1–D5), update **both** table builders (replay + contfrag) + regenerate the tables + update model specs/runners + tests + smoke fits. **(COMPLETE, unmerged.)**
  - [phase-1.5-behavior-model-covariates.md](phase-1.5-behavior-model-covariates.md) — behavior-model (RL) covariate integration from DataJoint `qvals_table`; **supersedes D3's leaky-Q** for `q_current`/`q_arm`; adds the accuracy-audit / invariant-bounds layer + base-covariate catalog. **(COMPLETE, unmerged.)**
  - [phase-2-refit-verify.md](phase-2-refit-verify.md) — re-fit all families on current tables, verify parity for unchanged covariates, refresh docs/claims numbers. **(PARTIAL — only `uncertainty_rate` refit.)**
  - [phase-3-canonical-is-swr-sweep.md](phase-3-canonical-is-swr-sweep.md) — **narrowed:** only non-PFC ancillary reality/occupancy adapters remain; PFC is owned by its dedicated plan.
  - [phase-4-promote-content-models.md](phase-4-promote-content-models.md) — **superseded historical design; do not execute.**
