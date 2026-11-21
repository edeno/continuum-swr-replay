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

A fit checks its inputs once, at its entry, and refuses with a message that names the
problem: a design, response, penalty or start that is not finite or of the wrong shape;
for the Poisson, a count below zero, an offset that is not finite, or a row of zero
exposure (offset minus infinity: drop the row, it carries no likelihood); for the
Bernoulli, a response outside {0, 1} (it takes 0/1 outcomes, not proportions); for the
grouped softmax, group ids that are not finite or not sorted, a negative share, or a
group whose shares sum to zero; for the Gaussian, a scale that is not finite and
positive. A penalty block whose matrix is not symmetric positive semidefinite (to
`1e-10` of its largest entry and eigenvalue) is refused when it is built. In a pooled
fit the message names the session.

## The fit and its convergence rule

At fixed strengths `lam_j` the fit minimizes `-l(beta) + beta' S beta / 2`, with
`S = sum_j lam_j S_j`, by Newton in float64. Each step solves `A step = g` by Cholesky
(`A = H + S`); a step is first shortened so that no coefficient moves more than 10, then
halved (at most 30 times) until the objective does not rise. The Gaussian, whose
log-likelihood is quadratic in the linear predictor, takes its full step without the
cap: that step is the exact optimum whatever the outcome's scale (capped, an outcome of
spread 1,000 did not converge in 100 steps). The linear predictor is never clipped; a
step whose objective is not finite is rejected like one that rises.

Before the first step the design is checked: the coefficients the penalty leaves free
(the unpenalized columns and each penalty's null space) must be identified by `X`. Those
columns of `X`, each scaled to unit length, must have a Gram matrix whose smallest
eigenvalue is above `1e-12`; a duplicated column, a column zero on every row, or a
linear term beside a smooth of the same variable (the smooth's penalty leaves the line
free) fails, whatever the penalty strengths, and the fit is not attempted. The Cholesky
factor of such a design's `A` exists or not by the luck of rounding (it did on 11 of 20
seeds of a duplicated column), so the test is made before it.

The fit has converged when the Newton decrement `g' A^-1 g / 2`, the gain the next step
predicts, is below 1e-8 (the current content fitter's value), and the step that goes
with it moves the linear predictor by at most 0.1 on every row (`max |X step|`).
Where the likelihood has no finite maximum along a direction (a logistic 0/1 column
whose rows are all y = 1, a Poisson 0/1 cell with no events, a continuous column that
separates the outcome) the decrement goes to zero while the coefficient runs to
infinity and each step still moves the linear predictor by about one: such a fit is not
converged (status "no finite maximum"), where before it was reported converged with a
coefficient near 23 and a standard error near 1e4. The threshold is on the link scale
and does not depend on a column's units. The largest `max |X step|` at the stop of a
regular fit was 1.7e-3 over the regression tests and the mgcv fixtures (1,384 stops),
2.4e-5 over 30 synthetic sessions of 40,000 bins (1,020 stops) and 1.8e-3 over the
slow simulation's 2,000 data sets (68,127 stops); the smallest at a separated one was
1.0. The Gaussian is not held to this rule: once `A` is positive definite its maximum
is finite, and its linear predictor is on the outcome's scale, where rounding steps
exceed 0.1 for an outcome of spread 1e9.

At the stop the fit takes that last full step when the objective does not rise by more
than its rounding (`8 eps max(1, |F|)`) and the decrement where it lands is below the
tolerance, and reports `A` and the decrement there; otherwise it stays where it was.
Without the step a coefficient would stay off by up to `sqrt(2 tol / lambda_min)` of
`A`, more than the comparisons below allow, and REML's derivatives assume the exact
optimum. Nothing is regularized to make a fit converge.

Every fit ends with a status (`NewtonFit.status`; `converged` is true only for the
first): "converged"; "design not identified"; "information not positive definite"
(`A` not positive definite or not finite, as from a start whose predictions
overflow); "step halving failed"; "step limit reached" (100 steps); "no finite
maximum".

**The columns running away.** A fit that stops as "no finite maximum" keeps its last
Newton step (`NewtonFit.step`), the direction along which the likelihood rises without
bound, and the columns running away along it (`NewtonFit.running_away`,
`running_away_columns`): those whose part of the step, `max_i |X_ij step_j|`, moves the
linear predictor by more than 0.1 on some row, the threshold of the stop itself, the
largest first. Along the direction each such column moves the linear predictor by about
one a step and the others by about the square root of the decrement; on the simulated
content model's separated sessions the columns named moved it by 1.0 and every other by
less than 5e-4. A separation that combines columns (an arm column at -19.6 with a role
column, each with part of the response on its rows, which a rule on one column at a time
cannot see) names each of them. The list can name only part of a direction spread over
near-collinear columns (of ten noisy copies of a 0/1 column, the first fit named 7, the
refit 3). The REML fit also gives the direction in the coordinates the penalty
leaves free or penalizes (`RemlScore.running_away_directions`, `FreeDirection`): an
unpenalized design column, a block's null space (for a P-spline, its straight line) or
a block's penalized direction. Error texts give both
(`session 4 (s4): no finite maximum (columns running away: [11, 12]; along column 11,
column 12)`; a separation by a smooth's own line reads `... along block 0's null space
(direction 0), column 0`). Nothing is regularized to hide a separation.

A separation can also end at the step limit: a 0/1 outcome that is a step in `x`,
beside a P-spline of `x`, is separated by the smooth's unpenalized line; every Newton
step is cut to the cap of 10, the progress stays linear, the decrement stays near 0.3
and the fit ran out of its 100 steps as "step limit reached", with no columns named. A
fit that reaches the step limit is "no finite maximum" when each of its last 10 steps
(`NO_MAXIMUM_CAPPED_STEPS`) was cut to the cap and lowered the objective while the full
Newton step's movement of the linear predictor did not shrink: along a direction
without a maximum the curvature vanishes and the full step grows (from 17 to 211 on
that example), while toward a finite maximum, however far in a column's units, the full
step shrinks (a Poisson slope of 2,000 on a column of units 1e-3 stays "step limit
reached" at 40 steps and converges at 400).

**What the likelihood identifies.** The design check uses the design the likelihood
sees (`Likelihood.identifying_design`): `X` for the Poisson, Bernoulli and Gaussian;
for the grouped softmax, `X` less each group's mean, since a combination of columns
constant within every group (a session's role columns that together mark every
candidate) cancels in each group. Checked on `X` itself, such a design passed and its fit
ended as "information not positive definite" (one of 242 simulated content sessions);
it is now "design not identified".

## How the penalties are chosen

A smooth term is a cubic P-spline: B-splines on equally spaced knots over a fixed range,
the same in every session, with the clock clamped to the range; a second-order
difference penalty, whose null space is the straight lines; and one sum-to-zero
constraint computed once on the pooled rows and used for every session.

Each smooth has one strength, shared by every session of a model (pooled REML). The
pooled score is the sum of the sessions' Laplace-approximate REML scores, each with its
own coefficients; one session whose inner fit fails makes it infinite, and the score
records the first such session's index and its fit's status (`PooledScore`); the
searches set aside, before they start, the sessions that fail at a reference point
(below). Every error
a pooled fit raises names the session (its index, and `SessionData.name` when given)
and the reason. The gradient and Hessian in the log strengths are exact. The penalty
blocks must be disjoint (each smooth has its own columns); the code refuses overlapping
blocks.

**Strong penalties.** The fit, the score and its derivatives are computed with each
block's coefficients in the eigenbasis of its penalty (`reml_and_derivatives`): an
orthogonal change of coefficients `gamma_j = V_j' beta_j`, which leaves the score, its
derivatives and the edf as they are, and in which the penalty is
`diag(PenaltyBlock.eigenvalues)`, exactly zero on the `k - rank` null-space directions
that `rank` defines. The coefficients, `A` and the covariance (computed in that basis)
are rotated back. With `S` as given, two rounding errors grow with `exp(rho)`: `b' S b`
is a difference of terms of size `|S| |b|^2` (at log penalty 16 on a content session it
came out -8.4e-8 where it is 2.1e-7, an error ten times the last Newton step's predicted
gain of 3.4e-8, and the fit stopped as "step halving failed"), and the stored `S_j` has
rounding of about `1e-16` of its largest eigenvalue on its null space, which `exp(rho)`
turns into a penalty on the straight lines (on 242 content sessions the pooled score was
3.4e-6 too high at log penalty 10, 1.1e-4 at 14, and rose from 14 to 16 where it should
fall). In the eigenbasis `b' S b` is a sum of positive terms and the fit at a very
strong penalty is the fit in the penalty's null space: at log penalty 20 the linear
predictor equals that fit's to 1e-6 (`test_inner_fits_converge_at_strong_penalties`).
On a simulated content model (242 sessions of each of two kinds, log penalties 14 to
20, 3,388 fits) 47 fits stopped as "step halving failed" at ef98019d; in the eigenbasis
41 of them converge and 6 stop as "no finite maximum", in the sessions whose unpenalized
columns separate (below); none fails otherwise. The bounds `[-10, 20]` are kept: the
fit at 20 is obtainable, and the score there is within its last few `1e-8` of the limit.

A session may lack support for an unpenalized term (a 0/1 term none of its rows has):
`SessionData.active`, a boolean mask over the columns, marks the columns the session
estimates. Its fit, score and derivatives are those of its design built without the
inactive columns, and their coefficients and covariance rows and columns are NaN. A
column of a penalty block may not be inactive (the penalty identifies it); that is
refused.

A stacked design, one `SessionData` per animal holding its sessions' rows with a set of
columns per session (zero on the other sessions' rows), is a session like any other: a
separated per-session column gives the whole animal no finite maximum, the animal is set
aside (below) with that session's columns named as running away and refitted with them
inactive, and marking them inactive gives the fit, score, derivatives, covariance and
edf of the stack built without them (`test_a_stack_with_inactive_columns_is_the_stack_without_them`, five animals of
three sessions). The mask already did this at ef98019d; what was missing was the names of
the columns, and the whole five-animal search raised.

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
3. it stops when the Newton decrement in the log strengths, `g' H^-1 g / 2` on the free
   terms with the floored Hessian (the gain in the score the next step predicts, in
   log-likelihood units), is below `1e-8`, then takes one last Newton step if the score
   does not rise by more than its rounding (`8 eps max(1, |score|)`). The rule does not
   depend on units: the gradient rule it replaces, largest free gradient component
   below `1e-6 * max(1, |score|)`, moved with the score's size, which depends on the
   units of the exposure (the Poisson) or of the outcome (the Gaussian's profiled
   scale). On 30 synthetic sessions of 40,000 bins (score 63,852) the old rule stopped
   4.6e-6 from the optimum in one log strength; the new one ends within 1e-11 of it.
   On the four mgcv fixtures the selected log strengths are the same to the last bit,
   and over the 17 searches of the REML tests they move by at most 3e-9. A decrement
   below the rounding of the score also ends the search: no halving can show a gain
   that small, and the point is the optimum to rounding (with the tolerance set to zero
   the search ends there instead of on a step too small to move `rho`).

**A flat score.** When the data do not support curvature the score falls toward its
limit in the penalty's null space as `c exp(-rho)`: the gradient is `-c exp(-rho)`,
the gain left on the way to the bound, and the Hessian the same, soon below the
eigenvalue floor of `1e-6`. The search then stops where the gain left is below
`sqrt(2 * 1e-8 * 1e-6) = 1.4e-7`, or at the bound, which a grid start on such a score
usually is (the bound is its lowest grid point) and where a gradient pointing out holds
the term. Either point is converged: the term's edf is its null space's dimension
(`in_null_space`; `at_boundary` only at the bound) and the score is within about
`1.4e-7` of its limit. On 242 simulated content sessions with a flat score from log
penalty 5 on, the search raised at ef98019d (the score's rounding at strong penalties,
above, made it rise where it falls, by 3e-4 from 14 to 16, and inner fits failed from
17); it now ends at the bound, the edf 1.000 in every session.

**Sessions without a finite maximum are set aside before the search.** A session whose
unpenalized columns, or a smooth's unpenalized line, separate the outcome has no finite
maximum at any penalty (the penalty does not touch those coordinates), and one such
session made every pooled score infinite, so the whole model raised. Before the grid,
`select_penalties` fits every session at one reference point, every log strength at the
grid's center (6: a fixed point, the same for every model and data set, where the
penalty identifies the smooth without dominating it; `refine_penalties` uses its
start).

- A session whose fit converges there takes part in the penalty choice
  (`PenaltySearch.in_choice`); the search runs over these sessions only, the same ones
  from the grid's first point to the last step, so that the objective it minimizes does
  not change as it goes. A session that passes the reference point but fails at another
  point makes that point's score infinite.
- A session whose fit has no finite maximum there is set aside (`PenaltySearch.set_aside`).
- A session whose design is not identified is an error that names the session and the
  columns the likelihood cannot see, with the remedy: mark the columns the session
  cannot estimate inactive (`SessionData.active`). It is the caller's design, not the
  data's lack of a maximum.
- A session whose fit fails otherwise ("step halving failed", "step limit reached",
  "information not positive definite") is an error naming it: a numerical failure is not
  a property of the data to be reported around.

After the search, each set-aside session is fitted at the chosen log strengths (a
session's penalty does not depend on its own data). While that fit has no finite
maximum along unpenalized design columns only, those columns are made inactive and the
session is fitted again, at most `REFIT_MAX_ROUNDS` (5) times. A direction through a
penalty block (a smooth's own line, or its penalized coordinates) cannot be left out,
and the session stays without a fit; so does one that still has no finite maximum
after the rounds, or whose refit fails otherwise. Each set-aside session's record keeps
its fit at the reference point, every refit, the columns left out, and the reason in
words (`SetAsideSession`). `PooledFit.sessions` keeps one entry per session given, in
order: a set-aside session that was fitted has its fit, with `in_penalty_choice` False
(it did not take part in choosing the penalties) and NaN at the columns left out; one
without a fit has NaN coefficients, covariance, edf and score. `PooledFit.score` is the
pooled score over `search.in_choice` only, so two models' scores are compared only on
the same set. When more than `MAX_FAILING_SHARE` (20 %, the project's rule for reporting
a model) of the sessions end without a fit (`PenaltySearch.share_without_fit`), the
search raises, naming each with its statuses and the reason.

In exact arithmetic the first refit, before any column is left out, cannot succeed:
the direction of a session without a finite maximum lies in the coordinates the
penalty leaves free, whatever its strength. It is kept as a check, and on the
reproductions it never succeeded; the sessions fitted were fitted with columns left out.

There is no fallback: if no grid point gives a finite score over the sessions kept,
`select_penalties` raises, naming the sessions that failed and why, and the model is not
fitted. A search that does not converge (50 steps, or a step no halving accepts) raises
too, with the last log strengths, the gradient and the steps taken; there is no argument
that returns an unconverged result. When sessions fail at only part of the grid, the
search goes on from the finite points and records how many failed and why
(`PenaltySearch.n_grid_not_finite`, `grid_failures`: the count per status).
`fit_at_penalties` fits at given strengths without a search (and without the score's
derivatives).
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
eq. 4) at that scale. The Newton fit behind it runs at the working scale `var(y)` (with
the penalty divided by it, the same minimizer), so its decrement does not depend on the
outcome's units: for an outcome scaled by 1, 1e3 and 1e6 the curve is the same up to
the scale and the selected log strength the same to 3e-15 (and at 1e9 too), with every
grid point finite (before, at 1e6, the inner fit at the four largest strengths stopped at 100
steps on a decrement of rounding, 5e-8 to 9e-5 in squared units of the outcome).

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
(slow, about a minute) fits 1,000 data sets per truth, 350 stops each with about 300
events. On its seeds the 95 % interval of a planted 1.3-fold reward effect covers in
94.4 % of the sets (the test asserts 93-97 %), a null reward term is significant at 5 %
in 4.4 % (3-7 %), and a flat clock truth shrinks to a line (edf below 2.5) in 95.2 %
(at least 90 %). The smooth's own 95 % Bayesian interval, from `A^-1`, covers the
centered rising truth at 93.7 % of the cells averaged over the sets (Wood's
across-the-function coverage; the test asserts 92-98 %): the interval leaves out the
intercept's uncertainty, which Marra and Wood (2012) add to bring it to the nominal
rate.

## Cost

Measured on 2026-11-20 on synthetic sessions of the real size: 242 sessions of 40,000
bins, a Poisson model with two smooths and 56 coefficients. One pooled evaluation took
8 s (score only, warm start) to 32 s (with the gradient and Hessian, cold start), one
penalty selection (25 grid points and 4 Newton steps) 6.3 min, with about 5 cores busy on
average and 4.5 GB resident, almost all of it the design matrices. 82 % of an
evaluation's time is the information `X' W X`; the session loop is independent across
sessions and can be spread over processes if a model needs it.

These were measured before the fits were moved to the penalties' eigenbasis. The copy
of the design in that basis, and the covariance computed at each evaluation, add to one
session's evaluation (40,000 rows, 53 columns, two smooths, one thread, warm start,
median of 5; 2026-11-21) 18 ms to the score alone (54 to 72 ms) and 26 ms with the
gradient and Hessian (165 to 191 ms).
