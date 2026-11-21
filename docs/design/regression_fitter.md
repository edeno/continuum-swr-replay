# The shared regression fitter

`src/continuum_swr_replay/analysis/regression/` fits penalized generalized linear models,
one per session, with penalty strengths shared by the sessions. It knows nothing of the
task: it takes design matrices, responses and penalty blocks and returns fits. Which
columns and rows a model uses is decided by the code that calls it. The design behind it
is [2026-11-20-regression-convention-and-figure04.md](2026-11-20-regression-convention-and-figure04.md)
(sections 2.4, 4.3, 5 and 6).

| Module | What it holds |
| --- | --- |
| `constants.py` | every numerical rule below; change a value only there |
| `likelihoods.py` | the four likelihoods and their derivatives on the linear-predictor scale |
| `newton.py` | the penalized Newton fit at fixed penalty strengths |
| `reml.py` | the REML score with exact derivatives, pooled over sessions, and the penalty search |
| `pspline.py` | the cubic B-spline basis on fixed, equally spaced knots, its second-order difference penalty and the sum-to-zero constraint |

## Likelihoods

| Family | Extra data | Used for |
| --- | --- | --- |
| Poisson, log link | the offset, `log(exposure)` | event rates per bin |
| Bernoulli, logit link | none | 0/1 outcomes |
| Gaussian, identity link | none; the scale is profiled | continuous outcomes |
| grouped softmax | the group (event) id of each row, rows of a group contiguous | content: an event's share on each candidate arm |

Each family gives the log-likelihood, its derivative in the linear predictor, the
observed information `X' W X` (the grouped form for the softmax) and the first and
second directional derivatives of the information, which REML's exact derivatives need.
Constants that do not depend on the coefficients are left out (the Poisson `log y!`, the
multinomial coefficient). The grouped softmax on fractional shares is the likelihood of
the current content fitter's conditional Poisson, a quasi-likelihood with the scale
fixed at one.

## The fit and its convergence rule

At fixed strengths `lam_j` the fit minimizes `-l(beta) + beta' S beta / 2`, with
`S = sum_j lam_j S_j`, by Newton in float64. Each step solves `A step = g` by Cholesky
(`A = H + S`); a step is first shortened so that no coefficient moves more than 10, then
halved (at most 30 times) until the objective does not rise. The linear predictor is
never clipped; a step whose objective is not finite is rejected like one that rises.

The fit has converged when the Newton decrement `g' A^-1 g / 2`, the gain the next step
predicts, is below 1e-8 (the current content fitter's value). It then takes that last
full step, and reports `A` and the decrement where it lands: without it a coefficient
would stay off by up to `sqrt(2 tol / lambda_min)` of `A`, more than the comparisons
below allow, and REML's derivatives assume the exact optimum. A matrix `A` that is not
positive definite (an unpenalized design the data cannot identify) or not finite ends
the fit as not converged; nothing is regularized to make it fit.

## How the penalties are chosen

A smooth term is a cubic P-spline: B-splines on equally spaced knots over a fixed range,
the same in every session, with the clock clamped to the range; a second-order
difference penalty, whose null space is the straight lines; and one sum-to-zero
constraint computed once on the pooled rows and used for every session.

Each smooth has one strength, shared by every session of a model (pooled REML). The
pooled score is the sum of the sessions' Laplace-approximate REML scores, each with its
own coefficients; one session whose inner fit fails makes it infinite. The gradient and
Hessian in the log strengths are exact. The penalty blocks must be disjoint (each smooth
has its own columns); the code refuses overlapping blocks.

The search (`select_penalties`):

1. a grid over the log strengths in `[-10, 20]`: every integer for one term (31 points),
   the product of `{-2, 2, 6, 10, 14}` for two or three terms (25 or 125 points), and
   for more, two passes of one-term integer grids from all terms at 6. The score can
   have two optima, a wiggly one and the straight line, and a search started at one
   point can end at the worse;
2. Newton on the log strengths from the best grid point, with the Hessian's eigenvalues
   made positive and floored at `1e-6 * max(1, max|eigenvalue|)`, step halving on the
   pooled score and the bounds. A log strength at a bound whose gradient points outward
   is held there;
3. it stops when the largest free gradient component is below
   `1e-6 * max(1, |score|)`, then takes one last Newton step if the score does not
   rise, or after 50 steps (not converged).

There is no fallback: if no grid point gives a finite score, `select_penalties` raises
and the model is not fitted. `fit_at_penalties` fits at given strengths without a search.
Each session's fit carries its covariance `A^-1` (mgcv's `Vp`), the edf per term and in
all, and two flags per term: `at_boundary` (the log strength within 0.5 of a bound) and
`in_null_space` (the edf within 0.01 of the dimension of the penalty's null space, for a
P-spline a straight line). Only the second catches a term shrunk to a line by a search
that stopped on the score's plateau short of the bound.

## What the score includes

For one session,

    V = -l(b) + b' S b / 2 - log|S|_+ / 2 + log|H + S| / 2 - M_p log(2 pi) / 2

with `b` the fit, `H` the information there, `log|S|_+` the log pseudo-determinant
(the sum over the blocks of `rank_j rho_j + log|S_j|_+`) and `M_p` the dimension of the
penalty's null space. It is mgcv's REML score given the same penalty matrices, less the
constants the likelihood leaves out: for the Poisson, `V + sum log y!` is mgcv's
`gcv.ubre`. For the Gaussian the scale is profiled: the fit minimizes
`||y - X b||^2 + b' S b` (`lam_j` on the residual-sum-of-squares scale, as mgcv's `sp`),
the scale is `phi = D_p / (n - M_p)` with `D_p` that minimum, and `V` is Wood (2011,
eq. 4) at that scale.

## Agreement with mgcv, neurospatial and the current fitters

**mgcv** (R 4.5.3, mgcv 1.9-4, `gam(..., bs = "ps", method = "REML")` with the same
knots). Four synthetic data sets with fixed seeds: one Poisson smooth with a 0/1
covariate, two Poisson smooths with it, one Bernoulli smooth and one Gaussian smooth.
mgcv scales each penalty matrix before it estimates the strength, so its `sp` is
compared as `log(sp * tr(S_mgcv) / tr(S))`. Measured on 2026-11-20, the largest
differences over the four sets: log strength 2e-8, edf 3e-8, REML score 1e-11, centered
curves 6e-10 and their standard errors 2e-10 on the link scale, the 0/1 coefficient
2e-11 and its standard error 3e-12 relative, the Gaussian scale 5e-10 relative. The
largest are the Gaussian set's; the Poisson and Bernoulli sets agree to 1e-9 or better.
The comparison needs mgcv's convergence control tightened
(`gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10))`): with its defaults
mgcv stops early, and on the Bernoulli set its log strength moves by 4e-5. The fixtures
and the command that remakes them are in
[tests/analysis/regression/data/mgcv/README.md](../../tests/analysis/regression/data/mgcv/README.md);
`test_reml.py::test_matches_mgcv_fixtures` checks them without R.

```bash
uv run --frozen python scripts/checks/compare_regression_to_mgcv.py   # runs the R script, rewrites the fixtures, prints the agreement
```

**neurospatial** (`encoding/_glm_numpy.py`, commit `d8a5643a2`), whose pooled REML over
units is the model for this one. On one P-spline term shared by three units, its score
equals ours less the terms that do not depend on the strength (to 6e-9 on scores near
-240; it also adds a 1e-10 ridge to the Hessian), and our selected log strength equals
the minimum of its objective to 9e-8, the log rates to 3e-8. Its own
`select_penalty_by_reml` stops at `xatol = 1e-3` in the log strength and lands 1.3e-4
from that minimum. The details are in
[tests/analysis/regression/data/neurospatial/README.md](../../tests/analysis/regression/data/neurospatial/README.md):

```bash
/cumulus/edeno/neurospatial/.venv/bin/python scripts/checks/compare_regression_to_neurospatial.py
```

**The current fitters.** Without a penalty the Poisson fit gives the per-bin rate
fitter's coefficients (statsmodels' GLM, `fit_per_bin_glm`) and the grouped softmax the
content fitter's (`fit_conditional_poisson`). On one real session each of chimi, j16 and
wilbur (2026-11-20; `M7r_time_histcos_rew` for the SWR outcome, 14 coefficients, and
`M0a`, 8), the largest relative difference was 1.1e-8 for the per-bin model and 2.3e-7
for the content model. The content fitter stops without the last step, at decrements of
1e-13 to 1e-15, against 1e-19 or less for the shared fit. On those sessions the per-bin
design's `reward_history_incomplete` column is zero on every bin; statsmodels gives it
the minimum-norm coefficient, zero, and the shared fitter, which refuses an unidentified
design, is fit without it.

```bash
uv run --frozen python scripts/checks/compare_regression_to_current_fitters.py
```

The suite's checks on simulated data: `test_simulation.py::test_coverage_and_size`
(slow) fits 200 data sets per truth, 350 stops each with about 300 events; on its seeds
the 95 % interval of a planted 1.3-fold reward effect covers in 93.0 % of the sets, a
null reward term is significant at 5 % in 3.5 %, and a flat clock truth shrinks to a
line (edf below 2.5) in 95 %. On 1,000 data sets per truth with other seeds the rates
were 95.6 %, 5.4 % and 95.8 %.

## Cost

Measured on 2026-11-20 on synthetic sessions of the real size: 242 sessions of 40,000
bins, a Poisson model with two smooths and 56 coefficients. One pooled evaluation took
8 s (score only, warm start) to 32 s (with the gradient and Hessian, cold start), one
penalty selection (25 grid points and 4 Newton steps) 6.3 min, with about 5 cores busy on
average and 4.5 GB resident, almost all of it the design matrices. 82 % of an
evaluation's time is the information `X' W X`; the session loop is independent across
sessions and can be spread over processes if a model needs it.
