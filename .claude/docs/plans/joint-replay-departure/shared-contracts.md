# Shared contracts

[← back to PLAN.md](PLAN.md)

Three contracts span phases: the post-Phase-13 input adapter, the discrete-time
**departure-hazard design** (Phases 2–4), and the **joint-likelihood interface**
(Phase 4). The scientific models reuse the existing `(stop × BIN_S bin)` design
from `aggregate_to_timebins`; no phase may bypass the adapter with a fixed path.

## Post-Phase-13 input adapter {#post-phase-13-input-adapter}

This is a pre-Phase-1 prerequisite, not a second covariate builder. Add one
adapter under `analysis/replay/` that resolves the Phase-13 promoted current
release through the strict run-artifact loader and returns:

- source `release_id`/`run_id` plus exact manifest hashes;
- the source-valid canonical stop/per-bin rows and stable
  `stop_id`/`decision_id` keys;
- the frozen joint-analysis cohort keys;
- one persisted chronological stop-fold table and its `fold_hash`.

The adapter joins, but never recomputes, the canonical fields
`reward_lag1`, `reward_outcome`, `reward_rate_trial_k10`,
`previous_completed_trial_is_switch`,
`recent_completed_switch_rate_k5`, `outcome_known_time`, and the absolute
per-bin `response_time`. It rejects a missing/incomplete current release,
schema or hash mismatch, absent contributing-session manifest, duplicate join
key, and every path not resolved and validated through that release—including
legacy fixed `Processed-Data/.../*.pkl` inputs.

Freeze one cohort and fold assignment per `(source_release_id, session_id,
timing_family, comparison_set)`. Replay-intensity and departure-hazard members
of the same comparison consume the identical row/group keys and fold table;
they never construct private random folds. Any downstream analysis run records
the source release/manifest, cohort, and fold hashes and writes through the
versioned run-artifact API.

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
- **Pre-outcome-safe covariates:** `tc` spline (`SPLINE_KNOTS`, fixed/bounded —
  shared basis with the replay intensity), `reward_lag1`,
  `reward_rate_trial_k10`, `previous_completed_trial_is_switch`, and
  `recent_completed_switch_rate_k5`; optionally (Phase 3) lagged replay-onset
  counts `onset_lag{j}` to test replay → departure.
- **Current outcome:** `reward_outcome` is a separate post-outcome estimand. A
  row may use it only when `response_time >= outcome_known_time`. Freeze the
  post-outcome cohort before fitting its null and reward models; never compare
  its likelihood/AIC/CV with a pre-outcome or full-stop model.
- **Model:** `is_last_bin ~ <covariates>` with a **cloglog** link (grouped
  proportional-hazards discrete time); `exposure` is NOT an offset here (each bin
  is one Bernoulli trial at risk). Returns the per-bin hazard
  `h(tc) = 1 − exp(−exp(η))` and `S(tc) = Π(1 − h)`.

`fit_departure_hazard(design, spec, *, cohort_identity, folds) -> dict` always
returns one structured result row. Successful rows include `coef_*` / robust
`bse_*` / `pval_*`, `n_bins`, `n_departures`, `llf`, `aic`, held-out score,
`survival`, source/cohort/fold hashes, rank/convergence fields, and covariance
metadata. Missing columns, too few departures, rank deficiency, separation,
nonconvergence, or another exception returns the same schema with a non-`ok`
`fit_status`, non-empty `failure_reason`, and unavailable numeric fields—not
`None`. Only `fit_status == "ok"` enters comparisons or headline summaries.

Invariants:
- exactly one `is_last_bin == 1` per *uncensored* stop; zero for censored stops.
- the `tc` spline uses `SPLINE_KNOTS` + `[0, TMAX]` bounds (same basis as the
  replay intensity, so hazard and intensity time courses are directly comparable
  and both reconstructable from coefficients).
- every current-outcome row passes the physical outcome-time gate;
- every compared intensity/hazard fit has identical cohort and fold hashes;
- every expected fit cell has either an `ok` row or an explicit structured
  failure row.

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
This interface has no replay-content mark or represented-arm/patch likelihood;
adding one belongs to a separate content plan.
