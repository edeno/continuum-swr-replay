# Designs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Per-component designs. Names inside `analysis/regression/` are the builder's to choose (design section 10);
the ones here are proposals, marked new. Code samples are NumPy float64 reference code; they show the
algorithm, not final style.

- [Penalized Newton](#penalized-newton)
- [Likelihoods](#likelihoods)
- [Smooth terms and bases](#smooth-terms-and-bases)
- [REML with exact derivatives](#reml-with-exact-derivatives)
- [Pooled REML: one strength per term across sessions](#pooled-reml)
- [Penalty selection: grid start, then Newton](#penalty-selection)
- [Support rule](#support-rule)
- [Collinearity measures](#collinearity-measures)
- [Trial folds](#trial-folds)
- [Simultaneous bands](#simultaneous-bands)
- [Session to day to animal](#session-to-day-to-animal)
- [Rate design on the visit](#rate-design)
- [Content design](#content-design)
- [JAX batch twin](#jax-batch-twin)

## Penalized Newton

Objective to minimize, for fixed penalty strengths `lam_j = exp(rho_j)`:

    F(beta) = -l(beta) + 1/2 beta' S_lam beta,     S_lam = sum_j lam_j S_j  (each S_j embedded in p x p)

```python
def penalized_newton(lik, X, y, aux, S_lam, beta0, *, tol=NEWTON_DECREMENT_TOL,
                     max_iter=100, max_halvings=30, max_step=10.0):
    beta = beta0.copy()
    F = -lik.loglik(X @ beta, y, aux) + 0.5 * beta @ S_lam @ beta
    for it in range(max_iter):
        eta = X @ beta
        g = X.T @ lik.score_eta(eta, y, aux) - S_lam @ beta      # gradient of +l - penalty
        A = lik.information(X, eta, y, aux) + S_lam              # H + S, PD for a penalized fit
        L = np.linalg.cholesky(A)                                # fails -> not converged
        step = scipy.linalg.cho_solve((L, True), g)
        decrement = 0.5 * g @ step                                # predicted gain
        if decrement < tol:
            return Fit(beta, A, converged=True, newton_decrement=decrement, n_iter=it)
        scale = min(1.0, max_step / np.max(np.abs(step)))
        for _ in range(max_halvings):
            cand = beta + scale * step
            F_new = -lik.loglik(X @ cand, y, aux) + 0.5 * cand @ S_lam @ cand
            if np.isfinite(F_new) and F_new <= F:
                break
            scale *= 0.5
        else:
            return Fit(beta, A, converged=False, newton_decrement=decrement, n_iter=it)
        beta, F = cand, F_new
    return Fit(beta, A, converged=False, ...)
```

- No clipping of `eta` at the returned optimum; an infinite `F` during halving is a rejected step.
- Start: Poisson intercept `log(sum y / sum exposure)`, others 0; during a penalty search, warm start from
  the previous `rho`'s solution of the same session.
- Unpenalized fits (S = 0) use the same code; then `A = H` must be PD, which the support rule makes likely
  and a failed Cholesky reports as not converged.

## Likelihoods

One small class per family with `loglik(eta, y, aux)`, `score_eta` (dl/deta), `information(X, eta, y, aux)`
(= -d2l/dbeta2 = X' W X, or the grouped form), and, for REML, the directional derivatives
`dinformation(X, eta, d)` and `d2information(X, eta, d1, d2)` (d, d1, d2 are vectors on the eta scale, n).

| Family | aux | l | w = information weight | w' = dw/deta | w'' |
|---|---|---|---|---|---|
| Poisson, log link, offset | `offset = log(exposure)` | `sum y (eta+o) - exp(eta+o)` | mu | mu | mu |
| Bernoulli, logit | none | `sum y eta - log(1+e^eta)` | `mu(1-mu)` | `w(1-2mu)` | `w(1-6w)` |
| Gaussian, identity | `phi` (scale) | `-sum (y-eta)^2/(2 phi) - n/2 log(2 pi phi)` | `1/phi` | 0 | 0 |
| Grouped softmax | `groups` (event id per row), `n_g = sum_k y_gk` | `sum_g sum_k y_gk log p_gk`, `p_g = softmax(eta_g)` | per group `n_g G_g`, `G = diag(p) - p p'` | see below | see below |

For the diagonal families: `dinformation(X, eta, d) = X' diag(w' * d) X` and
`d2information(X, eta, d1, d2) = X' diag(w'' * d1 * d2) X`.

Grouped softmax (content, choice). Within a group with probabilities `p` and directions `d1`, `d2`:

    D_d p      = G d                                   (G = diag(p) - p p')
    D_d G      = diag(G d) - (G d) p' - p (G d)'
    D_d2 D_d1 G = diag(q) - q p' - p q' - (G d1)(G d2)' - (G d2)(G d1)',   q = (D_d2 G) d1

so `dinformation = sum_g n_g X_g' (D_d G_g) X_g` and `d2information = sum_g n_g X_g' (D_d2 D_d1 G_g) X_g`.
The response is the event's share over its candidates (fractions summing to one, `n_g = 1`), as the
shipped fits use; the likelihood is then a quasi-likelihood with the scale fixed at one, as now. The
conditional Poisson of `analysis/replay_models/fitting.py:677` is the same likelihood (check 3 of Phase 1).
Group-level constants (an intercept, any event-level covariate) are not identified and must not be in X;
the design builder drops them and the support rule reports it.

Gaussian scale: REML includes `log phi` as one more parameter of the outer search
([REML](#reml-with-exact-derivatives)); `w = 1/phi` makes the inner fit penalized least squares.

## Smooth terms and bases

- Cubic B-spline on equally spaced knots over a fixed range `[lo, hi]` per term (the knots are the same
  for every session), with a second-order difference penalty `P = D2' D2`. Beyond `hi` the clock is clamped
  to `hi` (the curve is flat there), below `lo` to `lo`. Scratch reference:
  `proto_basic/pspline.py` (`pspline_knots`, `bspline_basis`, `diff_penalty`), see
  [appendix.md](appendix.md#scratch-map).
- Identifiability: one sum-to-zero constraint per smooth, computed once on the pooled cohort rows of the
  model and kind (`c = sum_rows B`; `Z` = null space of `c'` from a complete QR), and the same `Z` used for
  every session, so that the coefficients mean the same thing in every session and the session curves can be
  summarized together. The constrained penalty is `Z' P Z` (rank `n_basis - 3` after the constraint for a
  second-order penalty on a cubic basis with an intercept in the model; compute the rank numerically once).
- Model D (reward history as a linear functional): `eta_i += sum_{l=1}^{10} f(log l) r_{i,l}`,
  `f(x) = sum_b B_b(x) gamma_b` over `x in [0, log 10]`, so column `b` of the design is
  `sum_l B_b(log l) r_{i,l}` with `r_{i,l}` the outcome (1 rewarded, 0 omitted) at lag `l`. No sum-to-zero
  constraint (the term is not confounded with the intercept unless every outcome is the same; the support
  rule checks that). Second-order difference penalty on `gamma`. mgcv's equivalent is
  `s(L, by = R)` with matrix arguments `L` (lags) and `R` (outcomes), the "linear functional" form of
  `?linear.functional.terms`. The log scale lets the curve change fast between lags 1 and 2 (design call).
- Initial basis sizes (calls, checked as below): time since arrival `[0, 20] s`, knots 0.5 s apart; time
  until leaving `[0, 3] s`, 0.1 s apart (the design asks for closer knots here, section 2.4: 0.33 s was too
  smooth for a 0.05 s change); time until next movement `[0, 1.5] s`, 0.1 s apart; log trials since last
  visit `[log 1, log 200]`, 10 basis functions; model D lags 8 basis functions over `[0, log 10]`.
- Basis check: after the fit, a term whose edf exceeds 0.8 times its basis dimension is refit with twice
  the basis functions; the rule and both sizes are recorded in `basis` of the fit record.

## REML with exact derivatives

For one session (Laplace approximation; exact for the Gaussian):

    V(rho) = -l(b) + 1/2 b' S b - 1/2 log|S|_+ + 1/2 log|H + S|          (b = beta_hat(rho), S = S_lam, H = H(b))

Penalties of different terms never overlap (each smooth has its own columns), so
`log|S|_+ = sum_j (r_j rho_j + log|S_j|_+)` with `r_j = rank(S_j)`; the code asserts the blocks are
disjoint. Write `A = H + S`, `S_j~ = lam_j S_j`, and the derivatives of the optimum:

    b_j  = db/drho_j        = -A^-1 S_j~ b
    H_j  = dH/drho_j        = dinformation(X, eta, X b_j)
    A_j  = S_j~ + H_j
    b_jk = d2b/drho_j drho_k = -A^-1 (A_k b_j + S_j~ b_k) + delta_jk b_j
    H_jk = d2information(X, eta, X b_j, X b_k) + dinformation(X, eta, X b_jk)
    A_jk = delta_jk S_j~ + H_jk

Gradient and Hessian (the envelope theorem removes the derivative of the first two terms through `b`):

    g_j  = 1/2 [ b' S_j~ b - r_j + tr(A^-1 A_j) ]
    V_jk = 1/2 [ delta_jk b' S_j~ b + 2 b_k' S_j~ b - tr(A^-1 A_k A^-1 A_j) + tr(A^-1 A_jk) ]

```python
def reml_and_derivatives(lik, X, y, aux, blocks, rho, beta0):
    S = assemble(blocks, rho)                                  # blocks: list of (cols, S_j, rank_j)
    fit = penalized_newton(lik, X, y, aux, S, beta0)
    if not fit.converged:
        return np.inf, None, None, fit                          # the search rejects this rho
    b, A = fit.beta, fit.A
    L = np.linalg.cholesky(A)
    eta = X @ b
    V = -lik.loglik(eta, y, aux) + 0.5 * b @ S @ b \
        - 0.5 * sum(r * rj for (_, _, r), rj in zip(blocks, rho)) + np.sum(np.log(np.diag(L)))
    Sj = [embed(cols, np.exp(rj) * Sj_) for (cols, Sj_, _), rj in zip(blocks, rho)]
    bj = [-cho_solve(L, Sjj @ b) for Sjj in Sj]
    Aj = [Sjj + lik.dinformation(X, eta, X @ bjj) for Sjj, bjj in zip(Sj, bj)]
    AinvAj = [cho_solve(L, Ajj) for Ajj in Aj]
    g = 0.5 * np.array([b @ Sjj @ b - r + np.trace(M)
                        for Sjj, (_, _, r), M in zip(Sj, blocks, AinvAj)])
    m = len(blocks); Hs = np.empty((m, m))
    for j in range(m):
        for k in range(j, m):
            bjk = -cho_solve(L, Aj[k] @ bj[j] + Sj[j] @ bj[k]) + (j == k) * bj[j]
            Ajk = (j == k) * Sj[j] + lik.d2information(X, eta, X @ bj[j], X @ bj[k]) \
                  + lik.dinformation(X, eta, X @ bjk)
            Hs[j, k] = Hs[k, j] = 0.5 * ((j == k) * (b @ Sj[j] @ b) + 2 * bj[k] @ Sj[j] @ b
                                         - np.sum(AinvAj[k] * AinvAj[j].T)
                                         + np.trace(cho_solve(L, Ajk)))
    return V, g, Hs, fit
```

Gaussian: profile the scale. With `D_p = ||y - X b||^2 + b' S b` and `M_p` the dimension of the penalty
null space (unpenalized columns plus null spaces of the `S_j`),
`V(rho, phi) = D_p/(2 phi) + (n - M_p)/2 log(2 pi phi) - 1/2 log|S/phi|_+ + 1/2 log|(X'X + S)/phi|`
(Wood 2011, eq. 4), minimized jointly over `(rho, log phi)`. Check against mgcv's `gam(..., method = "REML")`
scale estimate on the hand case of Phase 1.

Tests (Phase 1): the gradient and Hessian against central finite differences of `V` (relative error below
1e-5) for every family with one, two and three penalties; `V` against mgcv on the stored fixtures.

## Pooled REML

One penalty strength per smooth term, shared by all sessions of a model and kind (design 4.3; the
per-session choice was unstable, section 2.5). The pooled score is the sum of the sessions' scores, each
with its own coefficients and its own `log|S|_+` term, as
`/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_numpy.py:408-478` (`_reml_objective_numpy`)
does over units (its convergence gate is at lines 467-468):

    V_pool(rho) = sum_s V_s(rho),   g_pool = sum_s g_s,   H_pool = sum_s H_s

A session whose inner fit fails at some `rho` makes `V_pool(rho) = inf` (rejected), as that file does. The session loop is embarrassingly parallel; on CPU use a `ProcessPoolExecutor` over sessions only
if one evaluation exceeds a few seconds (measure first).

## Penalty selection

The REML score can have two optima (design 2.4: a wiggly one and the straight-line limit). Start from a
grid, then Newton:

1. Grid over `[-10, 20]` per log penalty: one term, every integer (31 points); two or three terms, the
   product of `{-2, 2, 6, 10, 14}` (25 or 125 points); more than three, coordinate-wise 1-D integer grids
   starting from all terms at 6, two passes. Keep the best point.
2. Newton on `rho` from there with the exact Hessian: eigen-decompose, floor eigenvalues at
   `1e-6 * max(1, max|eig|)` (flip negative ones), step halving on `V_pool`, clamp to the bounds.
3. Stop when `max |g_pool| < 1e-6 * max(1, |V_pool|)` or 50 steps; record `at_boundary` per term
   (within 0.5 of a bound) and the number of outer steps.

Fallback is not allowed: if no grid point gives a finite score, raise (as `_glm_numpy.py:481-560`,
`select_penalty_by_reml`, does) and record the model as not fitted.

## Support rule

Decided from the data before the fit, per unpenalized declared term and session (design 4.3):

- A 0/1 term (rate: last outcome rewarded): count the units (visits) with the term 1 that hold at least one
  event of the kind, and those with the term 0 that do; the term passes when both counts are at least 5.
- A continuous slope restricted to an arm role (content): among the role's candidate rows of events of the
  kind, count events with the covariate above and below the session median of those rows (ties to neither);
  pass when both are at least 5 and the covariate is not constant within the role.
- A role category (content: previous arm, next arm): pass when at least 5 events of the kind have a
  candidate in the role.
- A term that fails in more than 20 % of sessions with a fit (constant of [C5](shared-contracts.md#c5-the-convention))
  leaves the per-session model. The design says it is then asked in a pooled model; that model is stage 3,
  outside this plan, so the phase stops and reports the term to the lead.

Smooth terms carry a penalty and have no support rule; their basis check and edf are recorded.

## Collinearity measures

Measured on the quantity the likelihood sees, with the information weights at the fit: for the diagonal
families `X~ = diag(sqrt(w)) X`; for the grouped softmax, per event,
`X~_g = diag(sqrt(p_g)) (X_g - 1 p_g' X_g)` (then `X~' X~ = X' (diag p - p p') X` summed over events).

For a term `i` with columns `Xi` and the rest of the design `X0` (intercept or arm effects included), QR of
`[X~0, X~i]` gives `R = [[R00, R0i], [0, Rii]]`. Then (mgcv's three concurvity measures; mgcv reports them by
`concurvity(fit, full = TRUE)`):

    observed = ||R0i b_i||^2 / (||R0i b_i||^2 + ||Rii b_i||^2)          (b_i the fitted coefficients)
    worst    = largest eigenvalue mu of  R0i' R0i v = mu (R0i' R0i + Rii' Rii) v
    estimate = ||R0i||_F^2 / (||R0i||_F^2 + ||Rii||_F^2)

For one column the three equal `R^2` of that column on the rest, and `VIF = 1 / (1 - R^2)`. The rule of
[C5](shared-contracts.md#c5-the-convention) reads `R^2` for one-column terms and `estimate` for smooths
(call), and stores all three. Phase 1's mgcv script prints `concurvity(full = TRUE)` for the fixture models,
and Phase 2 tests the three numbers against it.

## Trial folds

One fold function for every family (design 4.4), in `analysis/regression/folds.py` (new). Same semantics and
signature as `trial_folds(time, segment_starts, segment_classes, n_folds=5, max_segment_s=30.0)` of branch
`swr-trial-folds` (`analysis/event_evidence/tetrode_holdout.py:297` there, with `trial_segments` `:129`,
`capped_groups` `:255`, `task_segment_starts` `:177`): scikit-learn `StratifiedGroupKFold(n_splits=5,
shuffle=False)` with one sample per row, class = the destination well of the row's trial segment, group =
the row's piece of that segment (a segment longer than 30 s cut into equal pieces). Deterministic. For a
regression the rows are bins (rate, mPFC) or events (content, choice), placed by their start time.

When that branch is on master, its `trial_folds` calls the shared function (or the shared function is that
one, moved), and a test asserts both give the same folds on a synthetic epoch. Folds compare designs and
check over-fitting; they do not choose penalties and are not the test that an effect exists.

## Simultaneous bands

For a curve `f(x_1..x_G) = C beta` (C the basis rows for the term, a fixed grid):

1. Per session, the pointwise standard error `se_g = sqrt(diag(C V C'))` with `V = (H + S)^-1`.
2. A band valid for the whole curve: draw `M = 10000` vectors `z ~ N(0, V)` (seed fixed and stored), compute
   `max_g |C z|_g / se_g`, take its 0.95 quantile `c`, band `f +/- c se`.
3. For the animal-balanced curve across sessions: nested bootstrap draws of the curve (animals, then
   sessions; `shared_statistics.nested_bootstrap_draws`, `analysis/shared_statistics.py:377`), `sd_g` over
   the draws, `c` = 0.95 quantile of `max_g |draw_g - mean_g| / sd_g`, then the band is
   `estimate +/- c * sd_g * k` with `k = t(0.975, A - 1) sqrt(A / (A - 1)) / z(0.975)`, the same small-sample
   correction `bootstrap_interval` (`shared_statistics.py:456`) applies to a scalar (call: combining the two
   is this plan's choice; Phase 9 checks its coverage on simulated data).

## Session to day to animal

New functions in `analysis/shared_statistics.py` (task-agnostic, so they belong there; the module's rule,
`docs/README.md:100-103`, is kept):

- `animal_day_session_values(values, animal, day, session, statistic)`: session values -> recording-day
  values -> animal values (a mean or median at each level), the three-level analog of
  `animal_session_values` (`shared_statistics.py:191`).
- `day_level_test(values, animal, day)`: the signed-rank p over recording-day values
  (`signed_rank_p`, `shared_statistics.py:51`, which returns NaN below six non-zero differences).
- The estimate is the animal-balanced value; its interval is `bootstrap_interval` over
  `nested_bootstrap_draws` (animals, then sessions); "animals with the sign" counts animal values with the
  estimate's sign. No claim rests on a p-value over five animals.

The scratch script `verify_rate/s5_inference.py` (`infer`) shows the pattern on real values.

## Rate design

The visit at-risk bin design (Phase 3 builds it; Phase 5a fits it). Per session and kind:

- Visits from the port experiences (`arrival_time` = first poke-in, `departure_time` = last poke-out). The
  at-risk set is the union over the session's immobility stops of `[stop_start, stop_end]` intersected with
  each visit interval, less excluded time (`analysis/event_definition/analyzed_time.py`), less time inside an
  ongoing event of the outcome's kind (as `aggregate_to_timebins` does with `at_risk_events`,
  `analysis/replay_models/per_bin_design.py:394`).
- 15 ms bins (`RATE_BIN_S`, `analysis/behavior/design_constants.py:106`) laid from each in-visit immobile
  stretch's start; the last bin is partial (its `exposure` the remainder). Each bin carries
  `visit_context = "in_visit"`, `well_visit_id`, `stretch_index` within the visit, and the three clocks at
  the bin center: `s` = t - first poke-in, `u` = last poke-out - t, `m` = end of the current immobile stretch
  (clipped at the last poke-out) - t. In the visit's last stretch `m = u`.
- Counts: event starts of the kind whose event is `in_visit` (the event table's label). An event that starts
  in a bin but crosses the last poke-out is not counted and its time is not at risk once it has started; the
  rate simulation of Phase 5a plants this rule, so its effect is measured, not assumed.
- Reward covariates per bin: the last known outcome and the last ten known outcomes at the bin's time, with
  the existing outcome-boundary shift (the outcome of the current trial becomes lag 1 at
  `outcome_known_time_s`; `per_bin_design.py:85-91`, `_OUTCOME_SHIFT_STOP_COLUMNS`).
- Model A: `count ~ offset(log exposure) + f_s(s) + f_u(u) [+ f_m(m)] + last_outcome_rewarded`.
  Model D: the same with the model D functional in place of the 0/1 term, on bins with ten known lags.
  `f_m` stays only if the collinearity rule separates it from `f_u` (design 4.1, "to try").

## Content design

Per session and kind, one row per (event, candidate arm), all nine arms (lead's decision 1):

- Response: the event's share on each arm (`arm_fraction` of the event-arm table), weight one per event.
- Roles, exclusive, per (event, arm): `current` (the arm of the well being visited), `previous` (the arm of
  the previous trial's well, when different), `other_leaf` (any other leaf arm), `trunk` (arms 6 to 8).
  Each role is a category (reference: `other_leaf`). Arm fixed effects stay in every content model, per kind
  (`model_specs.py:2072`, `add_arm_fixed_effects`); a role contrast is then read against them.
- Leaf-arm covariates enter as one slope per role (current, previous, other_leaf): reward fraction over the
  last ten visits, value (beta-belief mean), and the uncertainty measure of the model. Trunk rows carry none
  (set to 0 with the trunk role absorbing the level).
- Recency: a smooth of `log(trials since last visit)` on leaf rows, not centered within the event (lead:
  absolute recency), with the never-visited indicator; recency is clamped at 200 trials. The shipped
  within-event centering (`behavior/covariates/history/visit_recency.py:23`) is not used.
- Q7b: a smooth of time until leaving (`u` at the event start) for the current role, per kind.
- Length-adjusted comparison of kinds: every role slope and role category also interacts with
  `log(event duration) - L0`, `L0` the pooled median log duration over both kinds (fixed, stored), so the
  per-kind estimate is at the same reference length; the unadjusted model omits the interactions. Both are
  stored and reported (see [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct)).
- Uncertainty: three models, each the same design plus one uncertainty measure with its role slopes: the
  belief variance (`reward_belief_variance_candidate_arm_preoutcome`, built by
  `behavior/covariates/history/value_block.py:206`), the arm's reward variability, and the number of times
  the animal has sampled the arm (Phase 3 builds the last two; definitions are decision L2).
- Q6 (Phase 7): one more role, `next`, for the arm of the next trial's well when it is neither current nor
  previous (call: when the next arm is the current or previous arm, the event adds nothing to this term).

## JAX batch twin

For the mPFC unit models only (Phase 8): the same objective, Newton rule and pooled REML as the NumPy
reference, vectorized over units with `jax.vmap` and run in float64 on CPU for tests and in float32 on the
GPU (design 5, check 6). Read for method, not imported:
`/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_jax.py:215-459` (`_newton_loop_jax`,
`_reml_score_jax`, `select_penalty_by_reml_jax`) and this repository's
`analysis/pfc/batched_glm.py:460-640` (`_sweep`, `_solve`, `_parameter_covariance`). Known pitfalls with
the pinned library stack are in the project memory: float32 drift in marginals, A100 matmul precision (set
`jax.default_matmul_precision("highest")` for the check).
