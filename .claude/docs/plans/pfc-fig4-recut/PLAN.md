# PFC Fig 4 Recut Implementation Plan

**Status:** Phases 1 and 2 landed on branch `pfc-recut-canonical-events` (2026-09-14),
executed per the decisions in [phase-1-2-execution.md](phase-1-2-execution.md): the PFC path
reads the canonical event table and the speed-gated `SWR_event` intervals, the sidecar and the
skip check state the event definition, the retired construction survives only in
`scripts/qa/diagnose_pfc_event_definition.py`, and a duration-matched non-event control is drawn
inside the fit mask and entered as its own event class with the nested engagement test
(`matched_control + non_local_no_SWR vs. matched_control`) and `matched_control vs. constant`.
No batch rerun happened, so the 71 shipped at-well unit bundles still predate both.

**Corrected premise (2026-09-14):** Phase 1's motivating discrepancy does not hold. On the
well-anchored mPFC population the retired ripple-overlap split and the canonical `is_swr` agree
**exactly in 71 of 71 sessions** (median SWR fraction 0.447 both ways); the earlier 44.7 % vs
43.0 % comparison set the PFC split against the canonical fraction over *all* events, off-well
ones included, and was wrong. Phase 1 therefore buys traceability — the canonical `event_id`s
and input hashes on Figure 4 — not a relabelling; the matched control is the substantive change.

Phases 0, 2b, 3 and 4 are open.

Rebuild the mPFC cross-area analysis (Figure 4 / claim #20) on a correct,
provenance-tracked footing. It repairs spike-bin and fit-validity mechanics,
replaces the non-canonical `is_swr` split, adds a matched immobile non-event
control, accounts for spike history/drift and temporal dependence, performs
animal-level inference, and regenerates Fig 4 with an engagement—not
equivalence—claim. Scope comes from reviews 05 and 07.

**Integration boundary:** this plan alone owns PFC mechanics, rerun, Figure 4, claim 20, and the
immutable `pfc_completion.json` producer. Production Phases 3–4 wait for the frozen release candidate
from `regression-integrity-fixes` Phase 13; they create no PFC-specific current pointer. Phase 13 only
validates/attaches the completion (or an unavailable reason) and mechanically composes shared staged
documents from its adjudication; it does not implement or rerun PFC analysis.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — provenance, matched-control, fit-validity, and temporal-inference contracts.
- Phases (0–2b each ship as a separable PR; Phases 3–4 each declare a pre-freeze code PR and a
  post-freeze, no-code production record):
  - [phase-0-glm-mechanics.md](phase-0-glm-mechanics.md) — fix spike-bin edges, rank/convergence, and theta fitting.
  - [phase-1-canonical-events.md](phase-1-canonical-events.md) — swap in canonical `is_swr`, fix `<=`/`<`, write provenance, old-vs-new diagnostic.
  - [phase-2-matched-control.md](phase-2-matched-control.md) — time-shifted-onset matched non-event control model + wiring.
  - [phase-2b-dependence-inference.md](phase-2b-dependence-inference.md) — spike-history/drift controls, autocorrelation-preserving nulls, and blocked scores.
  - [phase-3-batch-rerun.md](phase-3-batch-rerun.md) — pre-freeze runner/audit plumbing, then post-freeze smoke and full Dask rerun.
  - [phase-4-inference-figure.md](phase-4-inference-figure.md) — pre-freeze loader/finalizer/Fig-4 code, then post-freeze inference, staged outputs, and completion.
