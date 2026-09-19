# mPFC Marked-Family v2 Implementation Plan

**Status:** Not started.

Makes the mPFC marked-kernel family answer the questions it was built for. The first corpus batch (2026-09-19) showed that four "content effects" were stationary firing-rate differences between stops, that correlated marks split and shrink each other, and that no reference exists for the small population slopes. This plan adds per-stop baseline terms so event-locked and context effects are estimated separately, replaces correlated marks with the constructs they stand for, adds a marginal tier so every effect is reported both alone and jointly, adds a shuffled-mark reference for the population slope kernels, and adds the two analyses the family cannot ask on its own: whether mPFC drive scales with the number of replays in a stop, and whether the mPFC population reactivates its representation of the replayed place.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals, non-goals, metrics, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — baseline-column layout, contract version 2, construct mark names, bundle suffixes.
- Phases (each ships as a separable PR):
  - [phase-1a-baseline-terms.md](phase-1a-baseline-terms.md) — per-stop well and reward-outcome columns in every design; prior and contract plumbing.
  - [phase-1b-construct-marks.md](phase-1b-construct-marks.md) — graded distance and trajectory-size marks, centered interactions, rare-level eligibility.
  - [phase-1c-corpus-refit.md](phase-1c-corpus-refit.md) — smoke, 71-session refit under contract 2, audit and summary regenerated.
  - [phase-2-marginal-tier.md](phase-2-marginal-tier.md) — every target fit alone; joint and marginal reported side by side.
  - [phase-3-slope-permutation.md](phase-3-slope-permutation.md) — shuffled-mark reference for population slope kernels.
  - [phase-4-stop-rate-link.md](phase-4-stop-rate-link.md) — per-stop mPFC firing against replay count (independent of phases 1–3).
  - [phase-5-reactivation.md](phase-5-reactivation.md) — does the mPFC population pattern during replay of a place match its pattern at that place (independent of phases 1–3).
