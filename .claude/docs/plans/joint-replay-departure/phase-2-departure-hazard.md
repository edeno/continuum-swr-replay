# Phase 2 — Departure-hazard model (standalone survival)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Build the **departure process** on its own: a discrete-time per-bin hazard for
when the animal leaves the well (ends immobility). The building block for the
joint model, with standalone value.

**Questions:** What governs leaving? Does reward **lower** the departure hazard
(prolong stops — the consummatory stay)? Is the hazard time-varying (e.g. a low
hazard during early consummation, rising later)? Does it differ by well/value?

**Inputs to read first:**

- [shared-contracts.md#departure-hazard-design](shared-contracts.md#departure-hazard-design) — the per-bin `is_last_bin` design + `fit_departure_hazard` signature.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:58` `fit_per_bin_rate_model` — mirror its structure (column checks, separation→None, dict shape) for `fit_departure_hazard`.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:880` `aggregate_to_timebins` — the bins; add the `is_last_bin` indicator from `binj == max(binj)` per `stop_id` (+ a `censored` flag for `TMAX`-truncated stops).

## Tasks

- `src/.../departure_hazard.py` (new, tested): `fit_departure_hazard(design,
  spec)` — a per-session **cloglog** GLM `is_last_bin ~ bs(tc, SPLINE_KNOTS) +
  reward_lag1 + recent_trial_reward_rate + is_after_switch`, one Bernoulli trial
  per at-risk bin (no exposure offset). Returns `coef_/bse_/pval_` per term,
  `n_bins`, `n_departures`, `llf`, `aic`, and the fitted `survival` `S(tc)` on a
  `tc` grid (reconstructable, fixed knots). `None` on too-few-departures /
  separation. A helper to build `is_last_bin` + `censored` from the design.
- A spec set (small ladder): `D0` baseline hazard (spline only), `D1` + reward,
  `D2` + recent reward + switch — to read each block's contribution.
- GOF: predicted `S(tc)` vs the empirical survival (fraction still immobile);
  report calibration (max |S_pred − S_emp|) per session; rescaled departure-time
  KS via `TimeRescaling` (the terminal event rescaled like a spike train).
- Runner `scripts/fit_departure_hazard_models.py` (mirror
  `fit_per_bin_rate_models.py`: per session, populations are not needed — the
  departure process is shared across event types; one fit per session) →
  `models_per_bin_rate/departure_hazard.pkl`. Figure: hazard `h(tc)` and `S(tc)`,
  rewarded vs omitted.

## Gate
- Is the hazard model **calibrated** (predicted vs empirical survival close)?
- Does **reward lower the departure hazard** (per-animal consistent)? If the
  hazard is flat / reward has no effect, the "duration is endogenous to reward"
  premise is weak — note it, but the building block still stands for Phase 3.

## Deliberately not in this phase
- No coupling to the replay process (Phase 3); no frailty (Phase 4). The hazard
  is fit independently here.
- No continuous-time Cox / external survival packages — discrete-time per-bin only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_departure::test_is_last_bin` | On a synthetic 2-stop design, `is_last_bin` marks exactly the max-`binj` bin per uncensored stop; `TMAX`-censored stop has all-zero `is_last_bin` + `censored=True`. |
| `test_departure::test_hazard_recovers_survival` | On stops simulated from a known constant hazard, fitted `S(tc)` matches the analytic `exp(−h·tc)` within tolerance. |
| `test_departure::test_reward_lowers_hazard` | On simulated stops where rewarded stops are drawn longer, `coef_reward_lag1` in the hazard is negative. |
| runner smoke (slow) | Runs on `--max-per-animal 1`; writes `departure_hazard.pkl` with the schema; calibration reported. |

## Fixtures
- Synthetic: stops with known durations from specified hazards (constant + reward-
  dependent) for the recovery tests. Real-data smoke via one session.

## Review
Dispatch `code-reviewer`. Confirm: `is_last_bin`/censoring correct; cloglog (no
exposure offset); `S(tc)` reconstructable (fixed knots); calibration reported vs
empirical survival; separation → None; no coupling/frailty added; names free of
plan references.
