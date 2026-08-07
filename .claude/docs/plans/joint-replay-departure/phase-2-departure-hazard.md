# Phase 2 — Departure-hazard model (standalone survival)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **Prerequisite:** execute only from the Phase-13 promoted current release via
> the [post-Phase-13 input adapter](shared-contracts.md#post-phase-13-input-adapter).
> A fixed `stop_table.pkl`, legacy covariate alias, or privately generated fold
> table is a hard error.

Build the **departure process** on its own: a discrete-time per-bin hazard for
when the animal leaves the well (ends immobility). The building block for the
joint model, with standalone value.

**Questions:** What governs leaving? Does reward **lower** the departure hazard
(prolong stops — the consummatory stay)? Is the hazard time-varying (e.g. a low
hazard during early consummation, rising later)? Does it differ by well/value?

**Inputs to read first:**

- [shared-contracts.md#post-phase-13-input-adapter](shared-contracts.md#post-phase-13-input-adapter) — strict current-release resolution, canonical covariates, exact cohort identity, and shared folds.
- [shared-contracts.md#departure-hazard-design](shared-contracts.md#departure-hazard-design) — the per-bin `is_last_bin` design + `fit_departure_hazard` signature.
- [`regression-integrity-fixes` fit-validity and fold contracts](../regression-integrity-fixes/shared-contracts.md#fit-validity) — structured failures, exact cohort/fold hashes, and no disappearing cells.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:58` `fit_per_bin_rate_model` — reuse the corrected post-Phase-13 fitter/result conventions, not its historical `None` failure behavior.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:880` `aggregate_to_timebins` — the bins; add the `is_last_bin` indicator from `binj == max(binj)` per `stop_id` (+ a `censored` flag for `TMAX`-truncated stops).

## Tasks

- Wire the runner through the post-Phase-13 adapter. Validate the promoted
  release and contributing-session manifests before constructing a design;
  persist the source release/manifest, cohort, and fold hashes in every output.
  Delete/refuse any direct fixed-pickle input path.
- `src/.../departure_hazard.py` (new, tested): `fit_departure_hazard(design,
  spec, *, cohort_identity, folds)` — a per-session **cloglog** GLM with one
  Bernoulli trial per at-risk bin and no exposure offset. Always return the
  shared structured result schema: successful estimates/diagnostics or an
  explicit `fit_status`/`failure_reason`; never `None`.
- Use two declared timing families with separate exact cohorts:
  - pre-outcome-safe: `D0_pre = bs(tc)`, `D1_pre = D0_pre + reward_lag1`, and
    `D2_pre = D1_pre + reward_rate_trial_k10 +
    previous_completed_trial_is_switch +
    recent_completed_switch_rate_k5`;
  - current-outcome: freeze rows satisfying
    `response_time >= outcome_known_time`, then fit
    `D0_post = bs(tc) + reward_lag1 + reward_rate_trial_k10 +
    previous_completed_trial_is_switch +
    recent_completed_switch_rate_k5` and
    `D1_post = D0_post + reward_outcome`.
  Never treat current and lagged reward as aliases or compare models across the
  two timing families.
- Freeze the cohort once per timing family and construct/load one chronological
  stop-fold table through the shared Phase-9b API. Pass the same keys and folds
  to every hazard/intensity member in the comparison; a fitter-side row drop or
  private fold is a structured failure.
- Return `coef_/bse_/pval_`, `n_bins`, `n_departures`, `llf`, `aic`, held-out
  score, and fitted `survival` `S(tc)` on a reconstructable fixed-knot grid,
  together with rank/convergence/covariance and identity metadata. Add a helper
  to build `is_last_bin` + `censored` from the design.
- GOF: predicted `S(tc)` vs the empirical survival (fraction still immobile);
  report calibration (max |S_pred − S_emp|) per session; rescaled departure-time
  KS via `TimeRescaling` (the terminal event rescaled like a spike train).
- Runner `scripts/fit_departure_hazard_models.py` (mirror
  `fit_per_bin_rate_models.py`: per session, populations are not needed — the
  departure process is shared across event types; one fit per session) writes a
  versioned `joint_replay_departure` analysis run through the shared artifact
  API. Do not write `models_per_bin_rate/departure_hazard.pkl`. Figure: hazard
  `h(tc)` and `S(tc)`, rewarded vs omitted.

## Gate
- Is the hazard model **calibrated** (predicted vs empirical survival close)?
- Does **reward lower the departure hazard** (per-animal consistent)? If the
  hazard is flat / reward has no effect, the "duration is endogenous to reward"
  premise is weak — note it, but the building block still stands for Phase 3.
- Do source-release, cohort, fold, timing-gate, and structured-failure audits all
  pass? A scientific result cannot advance the gate if any identity check fails.

## Deliberately not in this phase
- No coupling to the replay process (Phase 3); no frailty (Phase 4). The hazard
  is fit independently here.
- No continuous-time Cox / external survival packages — discrete-time per-bin only.
- No replay-content mark, represented-arm/patch response, or content model.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_departure::test_current_release_adapter_rejects_legacy_input` | Missing/stale manifests and direct fixed-pickle paths fail before design construction. |
| `test_departure::test_is_last_bin` | On a synthetic 2-stop design, `is_last_bin` marks exactly the max-`binj` bin per uncensored stop; `TMAX`-censored stop has all-zero `is_last_bin` + `censored=True`. |
| `test_departure::test_hazard_recovers_survival` | On stops simulated from a known constant hazard, fitted `S(tc)` matches the analytic `exp(−h·tc)` within tolerance. |
| `test_departure::test_completed_switch_fields_only` | Design uses `previous_completed_trial_is_switch` / `recent_completed_switch_rate_k5`; removed stop-row aliases are rejected. |
| `test_departure::test_current_reward_is_post_outcome_only` | `D0_post`/`D1_post` share exact rows, and no current-outcome row precedes `outcome_known_time`. |
| `test_departure::test_shared_cohort_and_folds` | Compared hazard/intensity fits have identical row/group/fold hashes; a private drop/fold fails. |
| `test_departure::test_failure_is_structured` | Separation, insufficient departures, or nonconvergence emits one failure row rather than `None`. |
| `test_departure::test_reward_lowers_hazard` | On simulated stops where current reward prolongs dwell, the post-outcome reward coefficient is negative. |
| runner smoke (slow) | Runs on one promoted session; writes/resolves a versioned run with source-release identity and reports calibration. |

## Fixtures
- Synthetic: stops with known durations from specified hazards (constant + reward-
  dependent) for the recovery tests. Real-data smoke via one session.

## Review
Dispatch `code-reviewer`. Confirm: `is_last_bin`/censoring correct; cloglog (no
exposure offset); `S(tc)` reconstructable (fixed knots); calibration reported vs
empirical survival; current reward is physically post-outcome; completed-switch
fields, exact shared folds, and structured failures follow the regression
contracts; no fixed path, replay-content term, coupling, or frailty was added;
names are free of plan references.
