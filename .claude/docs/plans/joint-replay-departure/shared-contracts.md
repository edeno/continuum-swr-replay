# Shared contracts

[← back to PLAN.md](PLAN.md)

The two coupling points across phases: the discrete-time **departure-hazard
design** (Phases 2–4) and the **joint-likelihood interface** (Phase 4). Both
reuse the existing `(stop × BIN_S bin)` design from `aggregate_to_timebins`.

## departure-hazard-design

A per-(stop, bin) discrete-time survival design on the **same bins** as the
replay design, fit once per session.

- **At-risk set:** every bin of a stop up to and including the bin containing the
  departure (i.e. all bins the `aggregate_to_timebins` design already emits — a
  stop is "at risk of leaving" in each immobility bin).
- **Response `is_last_bin`:** 1 in each stop's final emitted bin (the bin in
  which immobility ends), 0 otherwise. Derived per `stop_id` as
  `binj == max(binj)` within the stop (subject to the `TMAX` truncation — stops
  censored at `TMAX` have `is_last_bin = 0` in all kept bins; carry a
  `censored` flag).
- **Covariates:** `tc` spline (`SPLINE_KNOTS`, fixed/bounded — shared basis with
  the replay intensity), `reward_lag1` (or `reward_outcome`),
  `recent_trial_reward_rate`, `is_after_switch`; optionally (Phase 3) lagged
  replay-onset counts `onset_lag{j}` to test replay → departure.
- **Model:** `is_last_bin ~ <covariates>` with a **cloglog** link (grouped
  proportional-hazards discrete time); `exposure` is NOT an offset here (each bin
  is one Bernoulli trial at risk). Returns the per-bin hazard
  `h(tc) = 1 − exp(−exp(η))` and `S(tc) = Π(1 − h)`.

`fit_departure_hazard(design, spec) -> dict | None` mirrors
`fit_per_bin_rate_model`: `coef_*` / `bse_*` / `pval_*` per term, `n_bins`,
`n_departures`, `llf`, `aic`, and `survival` (the fitted `S(tc)` on a `tc` grid,
for calibration vs the empirical Kaplan-Meier-style survival). Returns `None` on
missing columns / too few departures / separation, like the replay fitter.

Invariants:
- exactly one `is_last_bin == 1` per *uncensored* stop; zero for censored stops.
- the `tc` spline uses `SPLINE_KNOTS` + `[0, TMAX]` bounds (same basis as the
  replay intensity, so hazard and intensity time courses are directly comparable
  and both reconstructable from coefficients).

## joint-likelihood-interface (Phase 4)

The joint model factorizes per stop as
`L_stop = [recurrent replay Poisson over the at-risk bins] ×
[terminal departure hazard over the at-risk bins]`, coupled by a per-stop frailty
`u` that multiplies the replay intensity (`λ_R · u`) and the departure hazard
(`h_L · u^{ξ}`, sign of `ξ` free). `u ~ Gamma(1/θ, 1/θ)` (mean 1, variance θ).
Estimation: per-session EM / Laplace over `{u_stop}`; `θ → 0` recovers the
independent fits (Phase 3) — the **nested null** for the gate. The interface
returns both sub-models' coefficients plus `theta` (frailty variance, with a
likelihood-ratio test vs `θ = 0`) so Phase 4 can report whether coupling is real.
