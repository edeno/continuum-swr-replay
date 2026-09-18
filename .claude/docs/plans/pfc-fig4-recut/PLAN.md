# PFC Fig 4 Recut Implementation Plan

**Status:** Phases 1 and 2 landed on branch `pfc-recut-canonical-events` (2026-09-14),
executed per the decisions in [phase-1-2-execution.md](phase-1-2-execution.md): the PFC path
reads the canonical event table and the speed-gated `SWR_event` intervals, the sidecar and the
skip check state the event definition, the retired construction survives only in
`scripts/qa/diagnose_pfc_event_definition.py`, and a duration-matched non-event control is drawn
inside the fit mask and entered as its own event class with the nested engagement test
(`matched_control + non_local_no_SWR vs. matched_control`) and `matched_control vs. constant`.
**Phase 0 (fitter) landed 2026-09-15** per [phase-0-fitter-execution.md](phase-0-fitter-execution.md):
the unit GLMs are fit by a batched Poisson IRLS in JAX (equivalence-gated against statsmodels at
1e-10 on three real sessions), and **the batch rerun ran** — all 71 at-well bundles refit at
`9879ff0` under the canonical event definition and with the matched control, one worker per card on
six A100s, 13.7 h. Audited in `docs/audits/2026-09-15-pfc-canonical-rerun.md`: nothing the old and
new trees share moves outside the gate tolerances in any session, and the two engagement tests read
for the first time. Figure 4 is regenerated from the new bundles.

**What the rerun exposed, and what comes next.** 14.8 % of the shipped model fits (22,902 of
154,350, over 4,640 neurons) run to `maxiter` without converging — separating fits, which the
rank-deficiency gate cannot see — and their LR tests and AICs are inside every number the figure and
claim 20 quote. That is the subject of
[phase-0b-ridge-heldout-execution.md](phase-0b-ridge-heldout-execution.md) (informative ridge +
blocked held-out deviance), the next phase, which supersedes the statsmodels hand-back and the
asymptotic LR/AIC machinery. Claim 20's numbers are pending it.

**Phase 0b status (2026-09-18): complete.** Task A, the Task B integration and Task C all landed
on `pfc-ridge-heldout`. The production batch ran at `db01df0` — 71/71 sessions, launched 2026-09-16
and finished 2026-09-17 on five idle A100s (cards 3, 4, 6, 7, 9), 13 h 52 m–16 h 59 m per worker,
every worker exit 0 — and the maximum-likelihood tree is kept at
`Processed-Data/_pre_ridge_backup_20260915`. Non-converged fits fell from 14.8 % of shipped model
fits to 0.28 %. The before/after audit is `docs/audits/2026-09-18-pfc-ridge-heldout.md`, Figure 4 is
regenerated from these bundles, claim 20 is requoted from that audit, and the null's geometry is
measured (`scripts/qa/measure_pfc_null_geometry.py`): engagement-null/control overlap median 11.0 %,
6.2–8.4 % of called units with no positive held-out improvement, 0.3–1.4 % of events unmatchable.
The 2026-09-18 descriptive-population revision removes the 1 Hz and inferential-availability
screens: Figure 4 now contains all 17,280 units with four finite primary curves; each p-value or
best-model summary uses only its own finite tested denominator.
Population inference (neuron → session → animal) remains Phase 4. The original smoke selected
σ = 3, K = 5 and R = 40. A
follow-up GPU optimization gate reduced that smoke's null phase from 8:26 to 1:49 and made
**R = 1,000** the production default (minimum p = 1/1,001; the 40-draw artifact remains only a
pre-production equivalence gate). Review
corrected the Monte Carlo test to `(b + 1)/(R + 1)`, excludes incompletely placed null draws,
requires every requested production draw to be usable (with 20 retained only as a diagnostic
floor), groups runs whose event-kernel supports overlap, and keeps the
unit-level raw-p fraction explicitly descriptive pending Phase 4's session/animal inference. Those
changes passed the corrected three-session GPU smoke at `00765b2`: every class produced 40/40 exact
draws on its fixed matchable event subset, the three artifacts passed the combined population
loader and estimator audit, and the largest session completed in 27:25 at 8.89 GiB peak RSS. The
optimized one-session rerun was scientifically equivalent to the gated artifact and completed in
5:23 at 4.66 GiB peak RSS. The production-scale R = 1,000 gate then completed on the largest
510-unit session in 1:33:07, with a 1:27:22 null phase, 95.05% average GPU compute, 9.1 GiB peak
device memory and 8.4 GiB peak RSS. All 506 estimable units had all 1,000 usable draws for all five
tests and every finite p-value lay on the k/1,001 grid.

**Corrected premise (2026-09-14):** Phase 1's motivating discrepancy does not hold. On the
well-anchored mPFC population the retired ripple-overlap split and the canonical `is_swr` agree
**exactly in 71 of 71 sessions** (median SWR fraction 0.447 both ways); the earlier 44.7 % vs
43.0 % comparison set the PFC split against the canonical fraction over *all* events, off-well
ones included, and was wrong. Phase 1 therefore buys traceability — the canonical `event_id`s
and input hashes on Figure 4 — not a relabelling; the matched control is the substantive change.

Phase 0's fitter half, Phase 3's rerun and Phase 0b are done; Phase 0's GLM mechanics, 2b and 4
are open.

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
  - [phase-0-fitter-execution.md](phase-0-fitter-execution.md) — batched JAX Poisson GLMs, the equivalence gate, and the batch rerun (landed).
  - [phase-0b-ridge-heldout-execution.md](phase-0b-ridge-heldout-execution.md) — informative ridge and blocked held-out deviance, replacing the hand-back and the asymptotic LR/AIC (next).
  - [phase-1-canonical-events.md](phase-1-canonical-events.md) — swap in canonical `is_swr`, fix `<=`/`<`, write provenance, old-vs-new diagnostic.
  - [phase-2-matched-control.md](phase-2-matched-control.md) — time-shifted-onset matched non-event control model + wiring.
  - [phase-2b-dependence-inference.md](phase-2b-dependence-inference.md) — spike-history/drift controls, autocorrelation-preserving nulls, and blocked scores.
  - [phase-3-batch-rerun.md](phase-3-batch-rerun.md) — pre-freeze runner/audit plumbing, then post-freeze smoke and full Dask rerun.
  - [phase-4-inference-figure.md](phase-4-inference-figure.md) — pre-freeze loader/finalizer/Fig-4 code, then post-freeze inference, staged outputs, and completion.
