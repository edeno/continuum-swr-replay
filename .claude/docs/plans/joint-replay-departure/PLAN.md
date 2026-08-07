# Joint replay-intensity + stop-departure model Implementation Plan

**Status:** Not started.

**Dependency/ownership (2026-07-29):** this plan owns only the new
duration/departure estimand. It starts from the corrected rate inputs promoted by
[`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md)
and must reuse that plan's covariate, cohort, fold, fit-validity, and artifact
contracts. It does not reopen or independently refit those repairs.

A stop is two interleaved processes — recurrent **replay onsets** and a single
terminal **departure** (the animal leaves the well). Family B models only the
replay process and treats stop duration as fixed exposure, but duration is
endogenous (rewarded stops are ~3.5× longer; at large time-since-stop only long,
≈rewarded stops survive), so the within-stop time course and the reward effect
can conflate "replay rises with time" with "long stops are different stops."
This plan **builds up** to a joint replay-intensity + departure-hazard model
(optionally with a shared per-stop frailty) **with a decision gate at each
phase**, so the heavy joint/frailty model is only fit if the simpler steps show
it is needed. Additive to Family A/B — nothing in the shipped ladder changes.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a phase?** Open its file — each is self-contained (inputs, tasks,
   gate, validation, fixtures).
2. **Need the hazard design / joint-likelihood interface?** [shared-contracts.md](shared-contracts.md).
3. **Need motivation, the questions answered, the escalation logic, risks?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — motivation, the questions this answers, the phase-gate escalation logic, integration points, risks, metrics.
- [shared-contracts.md](shared-contracts.md) — the discrete-time departure-hazard design + `fit_departure_hazard` signature + the Phase-4 joint-likelihood interface.
- Phases (each ships as a separable PR; **each ends with a gate that decides whether the next phase is warranted**):
  - [phase-1-duration-controlled-timecourse.md](phase-1-duration-controlled-timecourse.md) — cheapest: is the replay ramp real within constant-duration strata, or composition? (reuses Family B)
  - [phase-2-departure-hazard.md](phase-2-departure-hazard.md) — the departure process alone (discrete-time survival): what governs leaving; does reward prolong stops?
  - [phase-3-two-process-coupling-test.md](phase-3-two-process-coupling-test.md) — fit both independently + test residual coupling (replay→departure, duration↔replay). **The key gate: build Phase 4 only if coupling exists.**
  - [phase-4-joint-frailty.md](phase-4-joint-frailty.md) — the full shared-frailty joint model (gated): is replay↔dwell one latent drive; does it de-confound the time course; SWR vs non-SWR.
