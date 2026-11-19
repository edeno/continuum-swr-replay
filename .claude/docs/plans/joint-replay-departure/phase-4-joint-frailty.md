# Phase 4 — Shared-frailty joint model (build only if Phase 3 gated through)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The full model: replay intensity and departure hazard estimated **jointly** with
a per-stop **frailty** `u` coupling them. **Only build this if Phase 3 found
residual coupling** — otherwise the independent models are the result.

**Questions:** Is replay↔dwell a **shared latent (consummatory/engagement)
drive** rather than two independent reward effects? Does modeling it
**de-confound the within-stop time course** (does `λ(tc)` change once the
duration process and frailty are in)? Do **SWR-coupled vs non-SWR** events couple
to departure differently (separate `ξ`/frailty loading per type)?

**Inputs to read first:**

- [shared-contracts.md#joint-likelihood-interface](shared-contracts.md#joint-likelihood-interface) — the factorized likelihood + gamma frailty + `θ→0` nested null.
- `src/.../departure_hazard.py` (Phase 2) and `per_bin_rate_model.py` (replay half) — the two sub-models the joint model couples.
- `non_local_detector.model_checking.sorted_spikes.TimeRescaling` — GOF for both processes under the joint fit.

## Tasks

- `src/.../joint_replay_departure.py` (new, tested): per-session joint estimator.
  Likelihood per stop = `[replay Poisson over at-risk bins with intensity λ_R·u]
  × [departure cloglog hazard over at-risk bins with h_L·u^ξ]`, `u ~
  Gamma(1/θ, 1/θ)`. Estimate by EM (E-step: posterior `E[u|data]` per stop in
  closed form for gamma-Poisson, numeric for the hazard part; M-step: weighted
  GLMs reusing `fit_per_bin_rate_model` / `fit_departure_hazard` machinery) or
  Laplace. Return both sub-models' coefficients, `theta`, `xi`, and a
  likelihood-ratio test vs `θ=0` (the independent fit).
- Comparison to Phase 3 independent fit: held-out replay `cv_bits_per_event` +
  departure log-likelihood; whether the `λ(tc)` spline and the reward effect
  shift when the frailty is added; per population.
- **SWR vs non-SWR:** fit the joint model for `non_local_swr` and
  `non_local_no_swr` replay against the (shared) departure process; compare the
  frailty loading `ξ` — does SWR-coupled replay relate to leaving differently?
- Runner `scripts/fit_joint_replay_departure.py` →
  `models_per_bin_rate/joint_replay_departure.pkl`. Figures: frailty-adjusted
  `λ(tc)` vs the independent `λ(tc)`; `θ`/`ξ` by population.
- **Smoke-test the estimator's cost** on 2–3 sessions (EM iterations × bins) and
  extrapolate before any full run — the per-stop frailty makes this the
  expensive phase.

## Gate (what to report)
- Likelihood-ratio test `θ > 0`: is the coupling statistically supported?
- Does the joint model **change conclusions** (time course, reward effect, SWR-
  vs-non-SWR) vs the independent fit on held-out data? If `θ≈0` / no change, the
  **independent models (Phase 2/3) are the reportable result** — the joint model
  confirms they suffice.

## Deliberately not in this phase
- No continuous-time / Cox; no external frailty packages (implement the EM on the
  existing discrete-time design). If the per-stop frailty won't identify, fall
  back to a fixed-effect duration covariate (noted, not a silent switch).
- No cross-session hierarchical frailty (a later refinement); per-session only.
- No replay direction; no change to shipped models / `all_models.pkl`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_joint::test_theta_zero_recovers_independent` | With the frailty variance pinned to 0, the joint fit's coefficients match the independent Phase-2/3 fits (the nested null). |
| `test_joint::test_recovers_planted_frailty` | On data simulated with a known gamma frailty coupling replay↑/duration↑, the estimator recovers `θ`/`ξ` within tolerance and the LRT rejects `θ=0`. |
| `test_joint::test_em_converges` | EM monotonically increases the (penalized) joint log-likelihood on a small fixture; stops at tolerance. |
| smoke (slow) | 2–3 real sessions fit; timing reported + extrapolated; outputs written. |

## Fixtures
- Synthetic: coupled replay+departure with a known frailty (recovery + LRT) and
  an independent version (θ→0 null). Real-data smoke on a few sessions.

## Review
Dispatch `code-reviewer`. Confirm: `θ=0` exactly recovers the independent fits
(nested null verified); EM convergence checked; the LRT and held-out comparison
are reported rather than assuming the joint model wins; estimator cost smoke-
tested before any full run; fallback to fixed-effect duration is explicit if the
frailty fails to identify; names free of plan references.
