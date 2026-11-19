# Overview — Scope, motivation, escalation, risks

[← back to PLAN.md](PLAN.md)

> **Execution prerequisite (2026-07-29):** do not run any phase against fixed
> `Processed-Data` paths or pre-remediation tables. This plan starts only after
> [`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md)
> promotes a current release. Every phase consumes that release through the
> [post-Phase-13 input adapter](shared-contracts.md#post-phase-13-input-adapter)
> and records its release/manifest identity.

## Motivation

The per-bin replay-rate model (Family B, [docs/rate_models.md](../../../../docs/rate_models.md))
treats each immobility stop as a recurrent point process of replay onsets with a
`log(exposure)` offset, and reads a within-stop time course `λ(tc)` (the `bs(tc)`
spline). But a stop is **two interleaved processes**: the recurrent replay onsets
*and* a single **terminal departure** (the animal leaves the well, ending
immobility). Family B models only the first and treats the stop's duration as
fixed exposure. That is a problem because **duration is endogenous**:

- Stops have very different durations; rewarded stops are ~3.5× longer
  (consummatory). So at large `tc` only the long (≈rewarded) stops survive — the
  time course there is read over a **selected subpopulation**, not "the same
  stops over time."
- Replay and dwell may be **co-determined** — by reward, or by a shared latent
  engagement/consummatory state — so the time course and the reward effect can
  conflate "replay rises with time" with "long stops are different stops."

The principled fix is to model the **departure hazard** explicitly and, if
warranted, **jointly** with the replay intensity (a recurrent-events +
terminal-event model, optionally with a shared per-stop frailty). But that is a
heavy model; this plan **builds up to it with decision gates** so we never fit
the full joint/frailty model unless the simpler steps show it is needed.

This plan owns only replay-onset **intensity** and terminal **departure
hazard**. Replay direction, represented arm/patch, posterior content marks, and
rate×content models remain separate downstream scientific questions.

## Questions this plan can answer

- **Is the within-stop replay ramp a genuine time-since-stop effect, or stop-
  duration composition?** (Phase 1)
- **What governs leaving?** Does reward lower the departure hazard (prolong
  stops)? Is the hazard time-varying (e.g. rises after consummation)? (Phase 2)
- **Are replay and dwell independent given reward, or coupled?** Does a replay
  burst **precede departure** (replay → leaving)? Does "destined duration"
  predict early replay (replay ↔ duration)? (Phase 3)
- **Is replay↔dwell a shared latent (consummatory/engagement) drive**, and does
  modeling it **de-confound** the time course and change the SWR-vs-non-SWR
  conclusions? (Phase 4)
- **Do SWR-coupled vs non-SWR non-local events relate to departure
  differently** — e.g. does SWR-coupled replay wrap up a stop before leaving
  while non-SWR does not? (Phases 3–4)

## Escalation logic (the point of the phasing)

Each phase ends with an explicit **gate**: a result that determines whether the
next, more complex phase is justified. Stop early if a gate fails.

1. **Phase 1 (cheapest, reuses Family B):** condition the existing replay time
   course on stop duration. *Gate:* does the ramp survive within constant-
   duration strata? If it fully dissolves, the "time course" is composition —
   document and stop. If it survives (and/or long-destined stops show higher
   early replay), a duration process matters → continue.
2. **Phase 2:** fit the departure hazard alone (a standard discrete-time
   survival GLM). *Gate:* is it well-calibrated and does reward move it? This is
   the building block; it has standalone value (what drives leaving).
3. **Phase 3:** fit both processes independently on shared covariates and test
   **residual coupling** (replay history → hazard; duration → replay) beyond
   shared covariates. *Gate — the key one:* if there is **no** residual coupling,
   independent models suffice → **do not build the frailty model**. Only proceed
   if coupling is present.
4. **Phase 4:** the shared-frailty joint model. *Gate:* does the frailty change
   conclusions (time course, reward effect, SWR-vs-non-SWR) vs the independent
   fit, on held-out data? If not, the independent models are the reportable result.

## Current codebase integration points

- `src/continuum_swr_replay/analysis/replay/_helpers.py:880` `aggregate_to_timebins`
  — the `(stop × 15 ms bin)` design both processes reuse; the departure model
  adds a per-bin "last bin of the stop" indicator on the same bins.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:58`
  `fit_per_bin_rate_model`; `:185` `per_bin_rate_gof` — the replay-intensity
  half + time-rescaling GOF to reuse for both processes.
- `src/.../model_specs.py` `BIN_S`, `TMAX`, `SPLINE_KNOTS`, `STOP_COVARIATES` —
  shared bin width, truncation, fixed spline knots (so the hazard and intensity
  splines share a reconstructable basis).
- The Phase-13 current release supplies source-valid stop/per-bin inputs,
  source manifests, exact cohort keys, and Phase-9b fold infrastructure. The
  shared adapter resolves those artifacts, rejects legacy fixed paths, and joins
  `reward_rate_trial_k10`, `previous_completed_trial_is_switch`,
  `recent_completed_switch_rate_k5`, `outcome_known_time`, and stable
  stop/decision IDs without re-deriving covariates.
- `non_local_detector.model_checking.sorted_spikes.TimeRescaling` — GOF for both
  the replay and (rescaled) departure processes.

## Non-Goals

- **Not the promoted rate ladder's replacement.** The Phase-13 rate results
  remain canonical; this is an additive downstream analysis with its own
  versioned run linked to the source release.
- **No continuous-time Cox / external survival packages** — discrete-time
  per-bin hazard reusing the existing binned design (no new heavy deps).
- **No cross-/ensemble spike modeling** — event-onset and departure processes
  only.
- **No replay content/direction model.** This plan does not use represented
  arm/patch or posterior content as a mark; it owns rate/timing only.

## Metrics
- Phase 1: does the duration-conditioned time course retain a positive
  time-since-stop slope within high-survival `tc`, per population (report, not a
  fixed threshold).
- Phase 2: departure-hazard model calibration — predicted vs empirical survival
  `S(tc)` within tolerance; reward-on-hazard sign + per-animal consistency.
  Current-outcome effects use only bins at or after `outcome_known_time`.
- Phase 3: a pre-registered residual-coupling test (replay-history coefficient
  in the hazard; duration coefficient in the intensity) with per-animal n/5 and
  a paired/signed-rank p — the gate to Phase 4. Both processes use one frozen
  cohort and fold assignment.
- Phase 4: held-out improvement of the joint (frailty) model over independent
  (per-bin/`cv_bits`-style + departure log-likelihood), and whether the time
  course / SWR-vs-non-SWR conclusions change.
- Every attempted fit remains represented by a structured success/failure row
  carrying source-release, cohort, and fold hashes; no `None` disappearance or
  fixed-path result is allowed.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Conditioning on total duration induces collider bias (duration is downstream of replay) | Phase 1 compares the *early* time course (high-survival `tc`) across duration strata rather than conditioning on the full duration; the unbiased handling is the *joint* model (Phase 3–4), which models duration instead of conditioning on it. |
| Building the frailty model when it isn't needed | The Phase-3 residual-coupling gate must pass before Phase 4; if independent models fit, stop. |
| Discrete-time hazard separation (few/zero departures in early bins) | cloglog link + the fixed-knot spline (bounded, no extrapolation); pool short sessions; report unfit counts. |
| Frailty estimation cost / non-convergence | Per-session random effect via Laplace/EM; smoke-test timing on a few sessions before any full run; fall back to a fixed-effect duration covariate if the frailty won't identify. |
| A stale table or private fold recreates the pre-remediation analysis | Resolve only the Phase-13 current release, validate manifest/cohort hashes, and pass one persisted fold table to every replay/departure fit. |

## Open Questions
1. **Hazard link** — cloglog (grouped continuous-time hazard) vs logistic;
   default cloglog (the discrete analog of a proportional continuous hazard).
   Settle in Phase 2.
2. **Frailty distribution** — gamma (conjugate, standard for joint frailty) vs
   log-normal (couples naturally to the log-linear intensity). Default gamma;
   revisit if coupling sign needs the log-normal. Phase 4.
3. **Per-session vs hierarchical frailty** — start per-session (one `u` per
   stop, estimated within session); a cross-session hierarchical version is a
   later refinement.

## Estimated Effort
- Phase 1: ~120 LOC (runner + duration strata) reusing Family B + ~60 LOC tests.
- Phase 2: ~150 LOC (`fit_departure_hazard` + GOF) + ~100 LOC tests.
- Phase 3: ~120 LOC (shared-covariate fits + coupling tests) + tests.
- Phase 4: ~250–400 LOC (joint frailty EM/Laplace) + tests — the heavy one,
  gated.
