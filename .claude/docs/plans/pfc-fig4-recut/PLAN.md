# PFC Fig 4 Recut Implementation Plan

**Status:** Not started.

Rebuild the mPFC cross-area analysis (Figure 4 / claim #20) on a correct,
provenance-tracked footing. It repairs spike-bin and fit-validity mechanics,
replaces the non-canonical `is_swr` split, adds a matched immobile non-event
control, accounts for spike history/drift and temporal dependence, performs
animal-level inference, and regenerates Fig 4 with an engagement—not
equivalence—claim. Scope comes from reviews 05 and 07.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — provenance, matched-control, fit-validity, and temporal-inference contracts.
- Phases (each ships as a separable PR):
  - [phase-0-glm-mechanics.md](phase-0-glm-mechanics.md) — fix spike-bin edges, rank/convergence, and theta fitting.
  - [phase-1-canonical-events.md](phase-1-canonical-events.md) — swap in canonical `is_swr`, fix `<=`/`<`, write provenance, old-vs-new diagnostic.
  - [phase-2-matched-control.md](phase-2-matched-control.md) — time-shifted-onset matched non-event control model + wiring.
  - [phase-2b-dependence-inference.md](phase-2b-dependence-inference.md) — spike-history/drift controls, autocorrelation-preserving nulls, and blocked scores.
  - [phase-3-batch-rerun.md](phase-3-batch-rerun.md) — smoke-test then full Dask rerun; regenerate artifacts with provenance.
  - [phase-4-inference-figure.md](phase-4-inference-figure.md) — per-animal inference, Fig 4 regen (engagement framing), claim/doc reconciliation.
