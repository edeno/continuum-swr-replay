# The shared regression fitter

`src/continuum_swr_replay/analysis/regression/` fits penalized generalized linear models,
one per unit of the fit (a session, a recording day or an animal; `fit_by`), with penalty
strengths shared by the units. It knows nothing of the
task: it takes design matrices, responses and penalty blocks and returns fits. Which
columns and rows a model uses is decided by the code that calls it. The design behind it
is [2026-11-20-regression-convention-and-figure04.md](2026-11-20-regression-convention-and-figure04.md)
(sections 2.4, 4.3, 5 and 6).

| Module | What it holds |
| --- | --- |
| `constants.py` | every numerical rule below; change a value only there |
| `likelihoods.py` | the four likelihoods and their derivatives on the linear-predictor scale, as JAX kernels on padded rows |
| `newton.py` | the penalized Newton fit at fixed penalty strengths: a host loop around a JAX iteration kernel |
| `reml.py` | the REML score with exact derivatives, pooled over sessions, and the penalty search (the score at a fit, `A^-1` and the edf in JAX) |
| `pspline.py` | the cubic B-spline basis on fixed, equally spaced knots, its second-order difference penalty and the sum-to-zero constraint |
| `terms.py` | the model terms, the design they build on the pooled rows, the names of terms and coefficients, the standardization and the design check |
| `collinearity.py` | how well the other terms explain each term, on the weighted design |
| `support.py` | the support rule (the row rule for an inactive level is applied in `sessions.py`) and the guard that a tested coefficient carries a penalty |
| `sessions.py` | the fitting loop, one fit for each unit of the fit, and the row rule for an inactive level |
| `calibration.py` | the declared log penalty of a per-session smooth: the one at which its median edf over the sessions is a target |
| `record.py` | the fit-row record (one per unit of the fit, with its sessions) and the closed set of causes of a fit not kept |
| `contrasts.py` | scalar estimands as linear functionals of a fit, with their exact variance |
| `folds.py` | the one fold rule |
| `bands.py` | simultaneous bands for a curve |
| `simulation.py` | the simulation engine: an outcome drawn from a planted truth on a model's own rows, fitted and reduced by the real code |
| `units.py` | one estimate per unit of a fit, indexed by the unit's key, and its reduction (`unit_summary`) |
| `runtime.py` | how the fitter's JAX runs: float64 for each call, the device guard, the float64 kernel guard and the log of traces |
| `padding.py` | rows and groups padded to bucket sizes, with the weight that marks the real rows |

The reduction of a per-session estimate to recording days and animals is in
`analysis/shared_statistics.py`, beside the project's other statistics. The declared
family of tests names the task's models and roles, so it is in
`analysis/replay_models/declared_tests.py`.

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
cap: that step is the exact optimum whatever the outcome's scale. The linear predictor is never clipped; a
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
converged (status "no finite maximum"). The threshold is on the link scale
and does not depend on a column's units. The movement is the one the likelihood sees
(`Likelihood.identifying_design`, below): for the grouped softmax it is `X step` less
each event's mean of it, since a movement common to an event's candidates cancels in its
softmax. Measured on `X step` itself, a slope with a shift of 1e4 common to every
candidate (100 events of two candidates, shares 0.2 and 0.8) moved it by 0.13 at the
maximum, all but 1.3e-5 of it common to each event, and the fit stopped as "no finite
maximum" with its coefficient short of `log(4) / 2` by 1.3e-5. The largest `max |X step|`
at the stop of a regular fit was 1.7e-3 over the regression tests and the mgcv fixtures
(1,384 stops), 2.4e-5 over 30 synthetic sessions of 40,000 bins (1,020 stops) and 1.8e-3
over the slow simulation's 2,000 data sets (68,127 stops); the smallest at a separated
one was 1.0. Those were measured on `X step` itself, before the softmax's movement was
taken within events; within an event it is at most the range of `X step` over the event's
rows, so at most twice `max |X step|`. The Gaussian is not held to this rule: once `A` is positive definite its maximum
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
`running_away_columns`): those whose part of the step, `max_i |X_ij step_j|` on the
design the likelihood identifies (for the grouped softmax, `X` less each event's column
means), moves the linear predictor by more than 0.1 on some row, the threshold of the stop itself, the
largest first (the columns of one direction have parts that differ in the last digit only,
and which is larger depends on the machine: parts whose logarithms agree to 9 decimals are
a tie, in column order). Along the direction each such column moves the linear predictor by about
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
The coefficients of a fit with no finite maximum are a point on a diverging path, set by
where the rule stopped it and by the rounding along the way: only its status and its
`running_away` set are stable. Between two versions of the fitter that differ only in
rounding (c169e039 and d42ad2a0, 51 such fits of random samples) the status, the step
count and the columns running away were the same in every one, while a coefficient moved
by up to 7e-5 of itself, the log-likelihood by up to 1e-4 relative and `step` by up to
5e-4; the stored references keep such a fit's status, steps, log-likelihood and columns,
not its coefficients.

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
candidate) cancels in each group. The no-finite-maximum rule, the cap rule and the
columns running away measure the step on the same design (in the iteration kernel,
`Likelihood._jax_identified_change` takes each group's mean off `X step`).

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
is a difference of terms of size `|S| |b|^2`, which can exceed the gain of the last
Newton steps, and the stored `S_j` has rounding of about `1e-16` of its largest
eigenvalue on its null space, which `exp(rho)` turns into a penalty on the straight
lines. In the eigenbasis `b' S b` is a sum of positive terms and the fit at a very
strong penalty is the fit in the penalty's null space: at log penalty 20 the linear
predictor equals that fit's to 1e-6 (`test_inner_fits_converge_at_strong_penalties`).
The fit at the upper bound, 20, is obtainable, and the score there is within its last
few `1e-8` of the limit. The measurements behind this, and the rules it replaced, are in
[the fitter's history](../../.claude/docs/reviews/2026-11-21-regression-fitter-history.md).

A session may lack support for a term (a 0/1 term none of its rows has):
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
three sessions).

The search (`select_penalties`):

1. a grid over the log strengths in `[-10, 20]`: every integer for one term (31 points),
   the product of `{-2, 2, 6, 10, 14}` for two or three terms (25 or 125 points), and
   for more, two passes of one-term integer grids from all terms at 6. The score can
   have two optima, a wiggly one and the straight line, and a search started at one
   point can end at the worse;
2. Newton on the log strengths from the best grid point, step halving on the pooled
   score and the bounds. The Hessian is first scaled by each term's own curvature,
   `D^-1 H D^-1` with `D_jj = sqrt(max(1, |H_jj|))`, and its eigenvalues are made positive
   and floored at `1e-6 * max(1, max|eigenvalue|)` there: a term of curvature at most
   one keeps the absolute floor of `1e-6`, and a stiff term does not set the floor of a
   weak one. A log strength at a bound whose gradient points outward is held there;
3. it stops only where the unfloored Hessian on the free terms is positive
   semidefinite to rounding (the same floor, on the scaled Hessian); where the
   decrement is small but the curvature negative (near a maximum or a saddle of the
   score) it takes a step of one scaled unit
   along the direction of most negative curvature, downhill first, and goes on. It
   stops when the Newton decrement in the log strengths, `g' H^-1 g / 2` on the free
   terms with the floored Hessian (the gain in the score the next step predicts, in
   log-likelihood units), is below `1e-8`, then takes one last Newton step if the score
   does not rise by more than its rounding (`8 eps max(1, |score|)`). The rule is in
   log-likelihood units and does not depend on the size of the score, which moves with
   the units of the exposure (the Poisson) or of the outcome (the Gaussian's profiled
   scale). A decrement below the rounding of the score also ends the search: no halving
   can show a gain that small, and the point is the optimum to rounding.

**A flat score.** When the data do not support curvature the score falls toward its
limit in the penalty's null space as `c exp(-rho)`: the gradient is `-c exp(-rho)`,
the gain left on the way to the bound, and the Hessian the same, soon below the
eigenvalue floor of `1e-6`. The search then stops where the gain left is below
`sqrt(2 * 1e-8 * 1e-6) = 1.4e-7`, or at the bound, which a grid start on such a score
usually is (the bound is its lowest grid point) and where a gradient pointing out holds
the term. Either point is converged: the term's edf is its null space's dimension
(`in_null_space`; `at_boundary` only at the bound) and the score is within about
`1.4e-7` of its limit.

**Sessions without a finite maximum are set aside before the search.** A session whose
unpenalized columns, or a smooth's unpenalized line, separate the outcome has no finite
maximum at any penalty (the penalty does not touch those coordinates), and one such
session would make every pooled score infinite. Before the grid,
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
  coordinates that take part (columns the likelihood cannot see, or the columns and
  penalty directions that are collinear), with the remedy: mark the columns the session
  cannot estimate inactive (`SessionData.active`). It is the caller's design, not the
  data's lack of a maximum. A coordinate is named when more than a tenth of its unit
  length (`NAMED_SHARE`) lies in the space of the combinations that are not identified;
  that set does not depend on the basis in which the space is computed.
- A session whose fit fails otherwise ("step halving failed", "step limit reached",
  "information not positive definite") is an error naming it: a numerical failure is not
  a property of the data to be reported around.

After the search, each set-aside session is fitted at the chosen log strengths (a
session's penalty does not depend on its own data), by the refit rule of
`refit_session` (section "Sessions without a finite maximum" below): columns running away
are made inactive and the session fitted again, and a session whose direction cannot be
left out (a penalty block's, or the intercept's) stays without a fit. Each set-aside session's record keeps
its fit at the reference point, every refit, the columns left out, and the reason in
words (`SetAsideSession`). `PooledFit.sessions` keeps one entry per session given, in
order (each session's `RemlScore`): a set-aside session that was fitted has its fit,
with NaN at the columns left out (it is not in `search.in_choice`: it did not take part
in choosing the penalties); one without a fit has NaN coefficients, covariance, edf and
score. `PooledFit.score` is the
pooled score over `search.in_choice` only, so two models' scores are compared only on
the same set. When more than `MAX_FAILING_SHARE` (20 %, the project's rule for reporting
a model) of the sessions end without a fit (set aside and not fitted by their refits), the
search raises, naming each with its statuses and the reason.

In exact arithmetic the first refit, before any column is left out, cannot succeed:
the direction of a session without a finite maximum lies in the coordinates the
penalty leaves free, whatever its strength. It is kept as a check, and on the
reproductions it never succeeded; the sessions fitted were fitted with columns left out.
How the simulated content model's searches ended on 2026-11-21 is in the fitter's history.

There is no fallback: if no grid point gives a finite score over the sessions kept,
`select_penalties` raises, naming the sessions that failed and why, and the model is not
fitted. A search that does not converge (50 steps, or a step no halving accepts) raises
too, with the last log strengths, the gradient and the steps taken; there is no argument
that returns an unconverged result. When sessions fail at only part of the grid, the
search goes on from the finite points and records how many failed and why
(`PenaltySearch.grid_failures`: the count per status).
`fit_at_penalties` fits at given strengths without a search (and without the score's
derivatives).
Each session's fit carries its covariance `A^-1` (mgcv's `Vp`) and the edf per term and in
all. Two flags per term say a term was shrunk: `PenaltySearch.at_boundary` (the log
strength within 0.5 of a bound) and `in_null_space(session, blocks)` (the edf within 0.01
of the dimension of the penalty's null space, for a P-spline a straight line). Only the
second catches a term shrunk to a line by a search that stopped on the score's plateau
short of the bound.

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
grid point finite. A constant outcome (`var(y) = 0`) has no scale to profile and is
refused as such ("the Gaussian outcome is constant"). So are, after the fit, a session
with no residual degrees of freedom (`n - M_p <= 0`: two rows, an intercept and a slope,
had a scale, score, log-likelihood and covariance of NaN, yet ended "converged") and a
fit whose scale is not finite and positive (a zero residual: three rows on a line, one
residual degree of freedom, divided by a zero scale). The test of the scale is `phi > 0`,
not a threshold relative to the outcome: a noise-free target fitted with a smooth (a
straight line `1 + 2x` on 400 rows, in the smooth's null space) leaves only rounding,
`D_p` = 9e-28, and is fitted. Fewer rows than `M_p` leave the design not
identified, which the fit names first.

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

## Terms and their names

A model is a list of terms (`terms.py`: `Smooth`, `LinearFunctional`, `Category`, `Slope`,
`Offset`) and, for the diagonal families, an intercept. `build_design` computes once,
from the pooled rows of the model and kind, what every session shares: the knots, the
sum-to-zero constraints, the penalty blocks and the standardizations. `Design.build`
maps a set of rows to its design matrix and offset, with NaN where a value read is
missing. The module docstring states what each term builds and the rule that names every
term and coefficient:

| Term | Term name | Coefficient names |
| --- | --- | --- |
| smooth | `s(<column>)` | `s(<column>).<b>`, `b` from 0 |
| slope | `<column>` | `<column>` |
| linear functional | `lf(<first lag column>..<last lag column>)` | `lf(...).<b>` |
| category | `<column>` | `<column>[<level>]` |
| intercept | `(Intercept)` | `(Intercept)` |

A smooth or a slope fitted for one level of a category appends `:<by column>=<level>`
to its term name, for example `s(clock_s):role=current`. The declared estimands name
their terms by this rule.

**Baselines.** A model declares its baseline terms to `build_design` (`baseline`): the
intercept or a category (the rate models' intercept, the content models' arm category;
`declared_tests.BASELINES`). In a fit of several sessions
(`Design.for_sessions`) each is copied for each session, zero on the other sessions' rows:
the intercept becomes one 0/1 column per session, each coded level of a category one
column per session, so each session keeps the reference level. A copy is named
`<name>@<animal>/<session_date>/<epoch>`, and the record's `sessions` lists each session's
copies by name, so a reader does not split the names. A copy is a level of its session:
the loop judges and drops it as a level. A baseline carries no penalty, and no term may be
fitted `by` a baseline category; `build_design` refuses both. No declared contrast reads a baseline
(`test_declared_tests.py::test_no_contrast_reads_a_baseline`), and the simulation engine
refuses an estimand that does. The collinearity record projects the copies out, as it does
the intercept; the edf and `Design.term_columns` do not hold them. In a fit of one session
nothing is copied. A category's reference level must have rows in every session of a fit
that copies it: a session without rows of the reference arm makes the first fit an error
that names the fit and the copies
(`test_sessions.py::test_a_session_without_the_reference_arm_is_an_error_naming_the_fit`).

A built design describes each term once, as a `DesignTerm`: its kind (intercept, level,
0/1 slope, continuous slope, smooth, lag functional), its columns, its level, its
covariate, its penalty block and, for a level, the terms that leave a session with it.
The support rule, the design check and the loop over fits read these descriptions.

Every term but the offset takes a required penalty: `"none"` or a fixed log penalty. No
penalty is chosen from the data in a per-session fit. A smooth's penalty is its
second-order difference penalty. A slope's and a category's is the identity on each of
its columns, each column its own penalty block: one block for a slope, one for each
level of a slope by a category, one for each coded level of a category. It enters
`Design.blocks`, `Design.log_penalties` and `Design.log_penalty_record()` as a smooth's
does; the record names every term's penalty, `"none"` included. A category used only as
the `by` of another term is not a term, and its penalty is not read.

A fixed log penalty `rho` on a simple coefficient is a Normal prior of mean 0 and SD
`exp(-rho / 2)`: the fit minimizes `-l + exp(rho) b**2 / 2`
(`terms.log_penalty_from_prior_sd`). The
coefficient is in the unit the convention reports: a 0/1 column enters as it is (a step
from 0 to 1), and a continuous column enters divided by its pooled SD (centered first
when it is by a category), so its coefficient is per pooled SD and the prior SD is too.
A covariate given in units 1000 times larger builds the same column
(`test_terms.py::test_the_penalty_acts_in_the_reported_unit`). Under the Gaussian the
penalty is on the scale of the residual sum of squares, so the prior SD is in units of
the residual SD. There a penalized category, a penalized slope of a covariate of mean near 100 and SD
near 50, a penalized slope by the category and a penalized 0/1 slope equal the ridge
solution `(X'X + diag(exp(rho))) b = X'y`, with `X` built by hand with the pooled SD and
the by-category centering, to 1e-10
(`test_sessions.py::test_a_gaussian_penalized_fit_is_the_ridge_solution`).

Each record holds the edf of each penalized simple coefficient (`edf`; a level of a
penalized category by its coefficient name as well as in its category's sum). Near 1 the
data decide the value; near 0 the penalty does. The category's entry is the sum of its
levels' entries, so a total over all the entries counts those levels twice; a total sums
the model terms only. For one coefficient with a 1x1 penalty the edf is the diagonal
element `[(H + S)^-1 H]_jj` of the influence of the penalized information: in a Gaussian
case and a grouped-softmax case the records equal it computed from the fit alone
(`test_sessions.py::test_a_penalized_category_edf_is_the_sum_of_its_levels`,
`test_penalized_terms_record_their_edf`).

Measured results, two cells (an intercept, one penalized 0/1 column, an offset; 20
visits; `test_sessions.py`):

- The fit equals a one-dimensional root of the penalized score equations to 1e-8, with
  counts on both sides and with no event on one side.
- At the weakest log penalty, -10, the estimate equals the plain maximum likelihood value
  to 1e-4 and its edf is 1 to 1e-4; from -10 to 10 the estimate falls toward zero and its
  edf falls, monotonically.
- The intercept, the coefficient and its edf equal mgcv 1.9-4's
  `gam(..., paraPen = list(flag = list(diag(1), sp = exp(rho))))` at the same fixed
  strength to 1e-8, in four cases (`test_a_penalized_binary_slope_matches_mgcv`; mgcv's
  `sp b**2` on the deviance is the fitter's `exp(rho) b**2 / 2` on the log-likelihood).

## Standardization

Each covariate has one scale and one centering reference, computed once over the pooled
rows of every level on which any slope of the design reads it, so that "per SD" means
the same for every term of the covariate. The rule is in `terms.py`'s docstring
(`standardization`); the record holds, per covariate, its SD, `ddof` (0), its reference
and the terms centered at it.

## The design check

The fit limits each Newton step to `MAX_STEP_LENGTH` in coefficient units, so a column
whose values are all very small ends at the step limit. A term's size is its largest
absolute value over the rows, a smooth's columns measured together; the threshold is
`SMALLEST_COLUMN_SIZE = 1e-2`. On the pooled rows that are not missing, a term below it is
the caller's error and `build_design` refuses it; in one session it fails the support
rule there. The rule, and which terms are measured, is in `terms.py`'s docstring.

Measured results:

- The threshold, on simulated designs of 600 rows (an intercept, three N(0,1) columns and
  a P-spline of 7 columns; Poisson, Bernoulli and grouped softmax; five seeds), with each
  N(0,1) column multiplied in turn by 1e-4 to 1e-2 and fitted at log penalty 2: every
  case whose column was at most 3.3e-3 ended at the step limit in at least one fit, and
  every case at 6.0e-3 or more converged. The threshold is three times the largest that
  failed (`terms.SMALLEST_COLUMN_SIZE`).
- A smooth is measured as one term because one of its columns can be tiny in a session
  that covers a small part of its range (0.0015 on the synthetic fixtures) while the term
  is not (`test_terms.py::test_a_smooth_is_measured_as_one_term`).
- A 0/1 slope and a level hold only 0 and 1 and are not measured. A continuous slope is,
  although standardized: two constructed cases are below the threshold on the pooled
  rows that are not missing, a slope sharing its scale with a slope of the covariate on
  another level (3.4e-6) and a plain slope whose spread lies on rows missing for another
  term (8.2e-4) (`test_terms.py::test_a_slope_is_measured_after_its_standardization`).

## Collinearity

`collinearity.py` measures each term on what the likelihood sees: the design weighted so
that `X~' X~` is the information at the fit, the intercept projected out first, then
mgcv's three concurvity measures (`concurvity(full = TRUE)`: observed, worst, estimate)
for a term of several columns and the R² and VIF for a term of one. The module docstring
states the weighting, the measures and when a term is separable (`MAX_EXPLAINED = 0.8`)
or not measured (`VARIES_TOL`).

On the mgcv fixtures, at unit weights, the smooths' three measures equal mgcv's to
3.5e-12 (measured 2026-11-21, `test_collinearity.py::test_matches_mgcv_concurvity`).
mgcv's "para" term keeps the intercept, so it is not compared.

In a fit of several sessions the intercept and every active baseline copy are projected out
first. A copy is zero on the other sessions' rows, also once weighted, so each session's copies
are projected out of its own rows: for each session the range of its copies (by a singular value
decomposition, so that a rank-deficient set of copies is handled) is taken out and what is left is
reduced by a QR, the sessions' blocks are stacked and reduced, and the same
per-term measures on them (`term_collinearity`, `session_copies`). It equals the projection of
every copy out of the whole design to 1e-10 (`test_collinearity.py::
test_copies_by_session_equal_the_sequential_projection`); a copy that its session's other copies
explain wholly adds no direction there
(`::test_a_copy_with_nothing_left_is_skipped_by_session`).

## The support rule

`support.py` decides from the design, before the fit, whether a session can estimate a
simple term (a level, a 0/1 slope, a continuous slope), penalized or not. No rule reads
the event counts before a fit. A level of a category passes when `MIN_UNITS = 5` units
have a row of the level. A 0/1 slope passes when 5 units have a row where it is 1 and 5
have a row where it is 0. A continuous slope passes when 5 units have a row above the
median of its rows and 5 have a row below it; a constant column fails. The units and the
rule are in the module docstring. No share of sessions failing the rule drops a term
from the model (the function that computed it, `terms_failing_share`, was never called and
was removed on 2026-11-24); the constant `MAX_FAILING_SHARE` remains, because the penalty
search and the rate model's third-clock decision (`visit_rate.third_clock_decision`) read
it.
A smooth and a lag functional have no counts; every term is judged by the design check.

The rule replaced, on 2026-11-23, a rule that counted the units with an event on each
side. The lead chose a weak penalty on the reward coefficient and on every simple tested
coefficient; a guard that reads the design only is the author's call that goes with it
(design, section 9, twenty-second round).
Under the grouped softmax every candidate row belongs to an event, so the counts are the
same as before (`test_support.py::test_softmax_counts_are_those_of_the_event_rule`, the
records of two content sessions against the old rule's counts at commit `ec1ae69c`).
Under the Poisson and the Bernoulli a side can have rows and few events or none. A
penalized term always has a finite maximum and stays in the fit. An unpenalized term has
no finite maximum only when it is a 0/1 slope or a level with no event on a side (under
the Bernoulli, also with an event on every row of a side); the refit then leaves it out
(below). In every other case it has a finite maximum and is fitted: a continuous slope, or
a level whose events are in one to four units. The event-based rule left these out; this
rule fits them, with a large standard error. A Poisson session with a level on 10 of 60
visits and one event there, the category unpenalized, fits the level at -3.67 with a
standard error of 1.0, the two-cell maximum likelihood
(`test_sessions.py::test_an_unpenalized_level_with_one_event_is_fitted`). So an
unpenalized simple term is a baseline or a nuisance term, whose value is not reported or
tested. A tested coefficient must carry the penalty. `support.require_tested_penalized(design,
tested)` raises an error that names every coefficient or term in `tested` without a fixed
penalty; each model module calls it with the coefficients of its declared tests.

A level that fails is removed with its rows, which are dropped and counted, not merged
into the reference level: in simulation, merging inflated the belief variance's false
positives, to 20 % in one case. The terms fitted for the level leave with it. How the
loop applies the rule (the levels to a fixed point, then every other term once on the
final rows) is in `sessions.py`'s docstring.

## Sessions without a finite maximum

`reml.refit_session` fits a session at given log penalties and, while the fit has no
finite maximum along unpenalized design columns only, makes the columns running away
inactive and fits again, at most `REFIT_MAX_ROUNDS` rounds; a direction through a penalty
block, any other status, or the intercept among the columns running away leaves the
session without a fit. Its docstring states the rule. The penalty search's refit of a
session it set aside and the loop over fits both call it, so the rule is in one place.

A session with columns left out is reported with those terms missing, not left out of a
summary, because leaving it out would depend on the outcome: the sessions that fail are
those with no events in one condition. Its record names each column left out with the
cause and the round, and holds NaN for those coefficients and their covariance. A term of
several unpenalized columns leaves a session whole (`sessions.py`'s docstring).

## The loop over fits

`sessions.fit_sessions(design, sessions, lik, model=..., kind=..., response=..., unit=...,
table_sha256=..., fit_by="session")` groups the sessions given into units of the fit
(`tables.session_keys.unit_key`: a session, a recording day `(animal, session_date)` or an
animal), the units in the order of their first session, and returns one record per unit.
The design of the behavior models is decided in
[2026-11-24-unit-of-fit.md](2026-11-24-unit-of-fit.md) (design C, fitted at the day and the
animal unit; `analysis/replay_models/fit_by.py`). Each unit is one fit through the same
steps (the module docstring): missing rows; the levels to a fixed point; every other term
once on the final rows; the fit; a level the refit leaves out is removed and the steps run
again. In a fit of several sessions each session's rows are sorted by unit (under the
softmax) and then joined, the support unit is coded within each session and then offset
(a well visit id starts at 0 in every session), and the baselines are copied for each
session (above). A fit by day or animal of a design without a baseline is refused at entry:
it would fit one intercept, or one set of arm effects, for all the unit's sessions. An
intercept copy that fails the support rule, or that the refit leaves
out (a session with no event), drops that session's rows, counted under `rows_dropped`.
A unit of one session is the per-session fit, bit for bit: the stored per-session
references pass unedited (`scripts/checks/_regression_reference.py::_records` checks that
`fit_by` is `"session"` and that `sessions` repeats the record's own session before it
leaves the two fields out of the comparison).

A session's design is built once, on all its rows; one mask with one reason code per row
removed says which rows are kept, and the kept rows' design is a slice. Measured on
2026-11-21 on a busy machine (load 20 to 23; medians), against the loop that rebuilt the
design on every pass and stacked the pooled design for the check: four Poisson sessions
of 50,000 rows and 54 columns, 2.65 s to 1.33 s; `build_design` at 1,000,000 rows, 6.84 s
to 2.26 s; its peak resident memory at 2,000,000 rows, 2.25 GB to 1.09 GB (0.44 GB before
the build).

## What is exact and what is not

A session's design equals the pooled design on its rows bitwise, and so does the design
of the rows a session keeps, which is a slice of it. The exception is a design built on
very few rows: a smooth by a level with fewer than about 8 rows of the level, or a
session with fewer than about 8 rows, can differ in the last bit (at most 5.6e-17,
measured by an independent review, 2026-11-21), because numpy computes a product with so
few rows by another path. Against the code before the cleanup, no kept fit differed in
1,612 compared (the same review: coefficients and covariances bitwise equal).

## The fit record

`record.FitRecord` is one row per (model, kind, unit of the fit); its docstring lists every
field. `fit_by` names the unit (at `"day"` `epoch` is `""`, at `"animal"` `session_date` and
`epoch` are too). `sessions` has one entry per session of the fit: its key, the rows, units
and events the fit saw of it, its rows and events removed by reason, and the names of its
baseline copies.
A session with no rows, or no events of the kind, left is not fitted: no fit is tried,
and its record has the status "not fitted" and NaN coefficients. A fit tried and not
kept keeps the last fit's status and the true counts, with NaN coefficients, and lists
what the refit had left out before it ended, as a record of what was tried (the rule is
in `sessions.py`'s docstring). A record
whose status is not "converged" is kept; its readers flag it. Every cause is a code of a
closed set, so that sessions can be counted by cause: `support.Removed` for a term left
out (`left_out`, with the refit's round in `left_out_round`), `record.NoFit` for a
session without a kept fit (`no_fit`).

## Contrasts

Every scalar estimand of a fit is a linear functional `c' beta` of its coefficients, so
its variance in a session is exact: `c' V c`, with `V` the Bayesian covariance. There are
two kinds. A curve contrast is the mean of a curve over one range of its variable less
its mean over another, each mean over a grid of step `GRID_STEP = 0.01`; a range whose
two ends are equal is one point. A coefficient difference is a sum of named coefficients
less another sum. A curve enters only through the function that maps values of its
variable to rows of the coefficient vector, so the basis and its freedom do not matter.
The rules (a range must lie inside the term's range; a coefficient left out of a session
makes NaN only the contrasts that use it) are stated in `analysis/regression/contrasts.py`.
The declared family of estimands built from these contrasts is
`analysis/replay_models/declared_tests.py`.

## Folds

There is one fold rule in the project, in `analysis/regression/folds.py`; its rules are
stated there. Inside an epoch each row (a time bin, or an event placed by its start)
belongs to a trial segment: from a departure from a well to the next departure, so the
run to a well and the whole stay there are one segment, its class the destination well.
A segment longer than `MAX_SEGMENT_S = 30` s (a number of rows at the median row
spacing) is cut into equal pieces, so a long stay is spread over the folds and can span
two of them. The folds are scikit-learn's `StratifiedGroupKFold` with `N_FOLDS = 5` and
no shuffling, the class a row's segment's and the group its piece: every row of a piece
is in one fold, each class is spread over the folds as evenly as whole pieces allow, and
the folds do not depend on a seed. The rows' times must be strictly increasing; a model
with several rows per event (one per candidate arm) makes the folds on the events and
gives each event's rows its fold.

The held-out scores of the event evidence use the same rule:
`analysis/event_evidence/tetrode_holdout.py`'s `epoch_folds` gives each epoch's decode
bins their folds (`trial_folds_and_groups`, which also returns the segments and groups the
counts are recorded from) for the held-out tetrode score and the non-local restriction
run. `scripts/checks/check_stored_folds_reproduce.py` checks both runs against the rule.
On 2026-11-21, against the restriction run made at commit 3953304c and the held-out
tetrode run made at commit be732dee (`event_nonlocal_restriction/` and
`event_tetrode_holdout/`), it found in 242 of 242 epochs that the folds made from the
session cache's trials equal the restriction run's stored folds, with the segment and
group counts it records, and that the held-out tetrode run records the same segment and
group counts, trial shares and margins.

Folds compare designs and check over-fitting. They do not choose penalties, and they are
not the test that an effect exists. A held-out claim is made on the sum over sessions.

## Bands

A band of `bands.py` is built for the whole curve, not each point alone, at a nominal level
of 95 %. There are two forms, stated in the module's docstring:

- one fit's curve `f = C beta`: the critical value is the 0.95 quantile of the largest
  standardized deviation `max |C z| / se` over draws `z ~ N(0, V)`, and the band is
  `f ± c se`;
- the animal-balanced curve across sessions (session to day to animal, with the nested
  bootstrap draws of the next section): the band is `estimate ± c sd sqrt(A / (A - 1))`,
  `sd` the draws' standard deviation over the `A` animals, with `c` from a pivot simulated
  at the level of the animals. At one grid point the pivot is `|t|` with `A - 1` degrees
  of freedom, so the band is `estimand_summary`'s interval.

Each band records its form, the draws behind it, the pivot's draws and the seed (`Band`).
Whether a band reaches its level depends on the curve. What was measured:

- One fit's band: a slow test simulates 500 Poisson fits of a P-spline curve and
  requires that the band holds the whole true curve in 93 to 97 % of them
  (`tests/analysis/regression/test_bands.py`).
- The across-session band, in simulation on the cohort's shape
  (`scripts/checks/simulate_interval_coverage.py --band-out`; 300 data sets per setting,
  table `tests/scripts/checks/data/band_coverage_full.csv`):
  - With 20 grid points, smooth animal and day curves (correlation length 0.2 of the
    range) and session noise of SD 0.3 or 1, it held the whole curve in 0.950 to 0.997 of
    the data sets.
  - When the animals carried almost all the variance (session noise SD 0.02) and the
    curve was rough for its 100-point grid (correlation length 0.05 or 0.02, or white
    noise), it held the whole curve in only 0.91 to 0.92 of them (standard error about
    0.016), below 95 %. The pivot's grid correlation is estimated from the draws, and five
    animals show at most four independent directions of difference between curves: a
    curve with more independent directions than that has a larger maximum than the
    simulated pivot.
  - With three animals it held the whole curve in every data set but was very wide (mean
    critical value about 40).
  - At a single grid point it is conservative: it covered in 0.997 to 1.0 of the data
    sets.

Its level for the curves of the models, low-dimensional spline curves, is not yet
measured: it is checked when those curves exist.

## From session to day to animal

The 242 epochs are not independent, so each per-session estimate is reduced session to
recording day to animal (`analysis/shared_statistics.py`): the mean over animals of each
animal's mean over its days of each day's mean over its sessions. Its draws are the
project's nested bootstrap, animals and then each drawn animal's sessions, an animal
drawn twice being two animals, each with its own sessions and its own day means; the
days are not resampled but stay in the estimate. The rule is stated once, in
`animal_balanced_bootstrap`. A simulation (`scripts/checks/simulate_interval_coverage.py`;
5 animals, 45 days, 244 sessions, 2,000 data sets per setting; table
`tests/scripts/checks/data/interval_coverage_full.csv`) found the t-corrected interval of
this rule at or above 95 % coverage in every setting of animal, day and session variance,
missing sessions and a paired kind difference included; resampling days as a third level
gave intervals up to 40 % wider that over-covered (up to 100 %) when the animals varied
little.

`estimand_summary` summarizes a scalar estimand: the animal-balanced estimate, its 95 %
interval (`bootstrap_interval` over the nested draws), the number of animals whose own
value has the estimate's sign, and the signed-rank p over recording days. The day and
animal values behind the last two are the same means as the estimate's, and
`animal_day_session_values` and `day_level_test` give them alone. The day test and the
animals' signs read those means to 12 significant digits of the largest of them, so that
a tie or a zero does not depend on the order of a sum (the rule is stated in
`_snapped`); an animal at zero does not count as having the estimate's sign. A missing
animal, day or session label is refused. A kind difference is
the difference of the two kinds' session estimates, reduced the same way. With five
animals a sign test cannot go below p = 0.0625, so no claim rests on a p over animals
alone.

`units.unit_summary(values, fit_by, n_draws=..., seed=...)` (in the fitter's package, as it
knows the units; `units.unit_values` indexes a fit's values by `FitRecord.unit_key`, and
`units.unit_estimate` is its estimate alone) reduces one estimate per unit of a
fit (`tables.session_keys.FitBy`; `docs/design/2026-11-24-unit-of-fit.md`, section 7), with
the same keys at every unit. At `"session"` it is `estimand_summary`. At `"day"` it is
`estimand_summary` with each day as its own session. At `"animal"` it is
`animal_mean_interval`: the mean over the animals, `mean +/- t(0.975, A - 1) SD / sqrt(A)`,
and no day-level test. A band over units is `bands.across_session_band` with one row per
day, or one row per animal whose day label is the animal
(`test_bands.py::test_band_over_days`, `::test_band_over_animals`).

## Simulation

**The rule.** A simulation replaces only the outcome column of a model's own rows. The
rows, the design, the fit of each unit (`fit_sessions` at the model's `fit_by`), the
estimands (`contrasts`) and the reduction over the units with the project's interval
(`units.unit_summary`) are the real functions. Models that differ only in
their `fit_by` are fitted to the same drawn data sets in one run.

**Two targets.** Each data set has the present target, `planted` (the planted value, the
session to day to animal mean of the sessions' values), and the unit's own target,
`planted_unit`: the model at its `fit_by` fitted by the real fitter to the expected outcome,
with the weak penalty off, reduced over the units as the estimates are
(`2026-11-24-unit-of-fit.md`, section 6). The summary gives the bias and the coverage
against both (`bias`, `coverage`; `bias_unit`, `coverage_unit`). The target is fitted once
per run, in the parent before any worker starts, for an inside scenario and once per data set for an outside one. The
noise-free target never drops a unit for lack of events, and a drawn data set can, so
`bias_unit` can compare slightly different sets of units. The Bernoulli
takes only 0/1 outcomes, so its `planted_unit` is NaN. The engine,
`analysis/regression/simulation.py`, builds no model: the caller gives a `Model` (the
`Design` as the model's own code builds it, the likelihood, the response and unit columns,
and each estimand's `c`). Every coefficient that an estimand uses must carry a fixed
penalty (`support.require_tested_penalized`; the engine refuses the model otherwise).

**The draws.** `likelihoods.draw(lik, eta, aux, rng)` draws outcomes at a linear
predictor: Poisson counts at `exp(eta + offset)`, Bernoulli 0/1, Gaussian with a given
variance, and under the grouped softmax each row's share of its group (shares sum to
one, as the content response does). The softmax has three noise models, which a scenario
chooses (`n_trials`, `concentration`; a number, or a column that holds one per event):
one categorical choice per event (a share is 0 or 1, variance `p (1 - p)`); a multinomial
of `n` choices, such as the event's decode bins (variance `p (1 - p) / n`); and a
Dirichlet-multinomial of `n` choices with concentration `a` (variance
`p (1 - p) (n + a) / (n (1 + a))`). Each has the model's mean share `p`. The fit treats
each event as one choice, so the noise model decides what a content simulation shows. The draws use only a NumPy `Generator`, never the fitter, so the same
seeds give the same data after the solver changes. Data set `i` of the scenario at place
`s` of the list is drawn from `SeedSequence([seed, s, i])`, the sessions in order; each
(model, estimand) of the data set has its own bootstrap seed, spawned from it. A run of
some scenarios only keeps each scenario's place, and so its data.

**Two kinds of scenario.**

1. Inside the model (`InsideScenario`): planted coefficients on the model's design. The
   planted value of each estimand is `c' beta` at them.
2. Outside the model (`OutsideScenario`): any expected outcome as a function of the rows
   (a curve the model cannot hold exactly, an effect drawn anew per animal in each data
   set). It has no planted coefficients, so it states the planted value of each estimand
   itself, as a number or a function of the model. An estimand it does not name has the
   planted value NaN, and the table gives its estimate without a bias or a coverage.

Several models with the same coefficients (for example one model at several declared
penalties) are fitted to the same data sets of one run.

**The results table** (`simulation.summarize`; one row per scenario, model and
estimand): `planted`; `mean_estimate`, `bias`, `bias_share`, `sd` (over data sets),
`bias_se`; `coverage` (the share of data sets whose 95 % interval holds the planted
value), `excludes_zero`, `reject_day` (the day-level signed-rank p below 0.05, a NaN p
counted as no rejection), each with its binomial standard error; `share_without_value`
and, for each `record.NoFit` and `support.Removed` code, the share of sessions without a
value for that cause; `share_not_converged` (of the session fits tried);
`n_sessions`, `n_sessions_used_min`, `n_sessions_used_mean`, `n_data_sets`,
`n_data_sets_with_estimate`, `n_draws` and `seed`. The engine also returns one row per
data set and one per session.

**How a phase adds a list.** A module beside the model, in
`analysis/replay_models/<family>/`, defines `simulation(quick, workers) -> Simulation` (the
model's real rows, its models, its scenarios, the data sets and the seed) and, if the
model needs its own columns, `table(result, simulation)`. The phase adds the module to
`MODELS` in `scripts/checks/run_simulations.py`, which runs it
(`--model`, `--scenarios`, `--n-data-sets`, `--seed`, `--workers` in a `spawn` pool,
`--output-dir`, `--quick`) and writes `<model>.parquet` with the seed, the commit and
each model's declared penalties in its `attrs`. The first list is the declared-penalty
check, `scripts/checks/simulate_declared_penalty.py` (a synthetic cohort; one model per
target edf). Fixed-seed tables are stored in
`tests/analysis/regression/data/numpy_reference/simulation_*.json`, so that a change of
the solver is checked on the whole pipeline of bias and coverage.

**Cost.** The fits take almost all the time. What depends only on the rows (each
session's offset, missing rows and linear predictor under each inside scenario) is
computed once per run; `fit_sessions` builds each session's design again for every data
set, since its interface takes rows, not a built design.

## Precision, devices, threads and padding

The fit at fixed penalties and REML's score with its gradient and Hessian in the log
penalties run in JAX, one fitter for every model: the likelihoods' kernels
(`likelihoods.py`), the Newton iteration and the evaluation of the objective (`newton.py`),
and REML's fixed-penalty summary, its derivatives and the profiled Gaussian (`reml.py`).
What is still NumPy: the design check, the columns running away, the penalty search and
the refits, and REML's rotation of the design into the penalties' eigenbasis (below,
"REML's derivatives").

**The kernels and the host loop.** Each family has JAX functions of the padded rows and
their weights (`_jax_loglik`, `_jax_score`, `_jax_information`, `_jax_dinformation`,
`_jax_d2information`, `_jax_weighted_design`). The one public method on them,
`weighted_design`, takes NumPy arrays, pads them and runs one jitted kernel; the tests call
the other functions on NumPy rows the same way (`conftest.family_kernel`). A likelihood
made for a test overrides the JAX functions. `newton.penalized_newton` pads the session's rows
once and runs a host loop in Python and NumPy that holds every rule, status and count of
"The fit and its convergence rule", around two jitted kernels: one evaluates `F` and `l`
at given coefficients; one makes a Newton iteration (`A`, its Cholesky factor, the step,
the decrement, the step's largest movement of the linear predictor as the likelihood
sees it, and the step-halving search as a `lax.while_loop`). A trial
of the search takes its linear predictor as `eta + scale * (X @ step)`, and the accepted
trial's starts the next iteration, kept on the device, so the line search multiplies a
vector by `X` once (`X @ step`) whatever its halvings, where it did once for `eta`, once
for the step and once for each trial; the gradient `score @ X` is the iteration's second
product with `X`, beside the Gram. The iteration returns the seven numbers per fit the loop
reads at every step (`ok`, the decrement, that movement, whether the step was capped, the
new `F` and `l`, whether a halving was found) stacked in one array, brought to the host in
one transfer; `A`, the step and the candidate stay on the device until the loop reads them
(`A` and the step only where a fit stops). The
loop is written for a batch of fits on one design (state arrays with a leading axis of
fits, each rule a vectorized expression over it), and the kernels map over that axis
(`jax.vmap`); a session is a batch of one. A family is a static argument of the kernels
(two families are equal when they are of the same class, so a family keeps no parameter
on the object: what changes its result travels in `aux`, and an object with an attribute
of its own is refused). The rule constants `NEWTON_DECREMENT_TOL`, `MAX_STEP_HALVINGS`,
`NO_MAXIMUM_ETA_STEP`, `NO_MAXIMUM_CAPPED_STEPS`, `IDENTIFIABILITY_TOL` and
`RUNNING_AWAY_TIE_DECIMALS` are read from `newton` at each call; the first two, with the
step cap, are passed to the iteration kernel as arguments, so a changed value takes effect
in a kernel already compiled (`tests/analysis/regression/test_newton.py`). The defaults of
`max_iter` and `max_step` are `NEWTON_MAX_STEPS` and `MAX_STEP_LENGTH` as imported. JAX's Cholesky gives NaN
where SciPy raises, so `A` is "not positive definite" when `A` or its factor is not
finite; on the CPU both call LAPACK's `potrf` (JAX 0.8.0 replaces the factor by NaN when
`potrf` reports a failure, `jax/_src/lax/linalg.py`, `_cholesky_cpu_lowering`).
`jax.scipy.linalg.cho_factor` reads one triangle, as SciPy's does.

**No kernel multiplies by the transposed design.** The gradient is written `score @ X`,
and every weighted Gram `X.T @ (w[:, None] * X)` (`likelihoods._gram`, for weights of
either sign, as the information's derivatives need; the grouped softmax's per-group
products alike). XLA folds that transpose into the product. Written `X.T @ score` and
`(X.T * w) @ X`, the NumPy forms, it built the transposed design (36,864 x 76 at the rate
model's size) on every iteration. One Newton iteration of the rate model took 90 ms (the
least of four processes, each the least of 40 calls; one thread, under load, 2026-11-23),
51 ms with the gradient as `score @ X`, and 46 ms with the Gram as well. The jaxpr cannot
tell the forms apart (`X.T @ (w * X)` has a `transpose` there that XLA removes, and
`dot_general(w * X, X)` has none, yet inside the iteration's program it compiled to a
`transpose` of the weighted design; compiled alone it had none).
`tests/analysis/regression/test_newton.py` keeps a canary on the CPU only: it reads the
optimized program of the iteration kernel and of the three information kernels of each
family and fails on an instruction named `transpose` whose printed shape has the padded
row count. It is not a proof: XLA can write the same copy as a `copy` to another layout
(the old `X.T @ score` compiled to a `transpose` and then a `copy`), which it would not
see if the `transpose` went away, and a form whose copy appears only in some programs is
caught only in those it compiles.

**REML's derivatives.** `reml_and_derivatives` fits by `newton._fit_one`, which returns
the fit with its problem, whose padded rows are on the device. The score, `A^-1`, the edf
and the gradient and Hessian in `rho` are then one call of one kernel on those rows and on
one Cholesky factor of `A` (`reml._laplace_derivatives`, the family a static argument;
the profiled Gaussian's are in `reml._profiled_gaussian`, which needs no product with the
design beyond its Gram; it is given the stack of penalties only when the derivatives are
asked for, so it has two programs per shape, the score alone and the score with its
derivatives). The rows are padded and moved once
for a fit and its derivatives; the NumPy loops called the public `dinformation` and
`d2information` (since removed) once for each penalty and pair, each padding and moving `X` again. The
penalties are the leading axis of the kernel's arrays (`S_j` of shape `(m, p, p)`): a new
number of penalties is a new shape and so a new program, and a model has one. With
`b_j = -A^-1 S_j b` and `E = X [b_1 ... b_m]`, the `m` matrices `A_j = S_j + dH(E_j)` are
made one at a time (`jax.lax.map`), and the Hessian is computed for `j <= k` and mirrored,
so it is exactly symmetric. With `A_jk = d2H(E_j, E_k) + dH(X b_jk) + delta_jk S_j`, the
Hessian's trace `tr(A^-1 A_jk)` is the sum of three terms: `tr(A^-1 S_j)` is its own term,
and the traces of the two derivatives of the information, `tr(A^-1 d2H(E_j, E_k))` and
`tr(A^-1 dH(X b_jk))`, are read by the family's `_jax_information_traces`; the cross term
`tr(A^-1 A_k A^-1 A_j)` is computed from the `m` matrices `A^-1 A_j` in full. For the
Poisson and the Bernoulli (the diagonal families) those two traces come from `h_i = x_i' A^-1 x_i` without making the derivatives:
`tr(A^-1 X' diag(c) X) = sum_i c_i h_i`, so the traces of all pairs are
`(weight w' h) @ (X b_jk)` and `E' diag(weight w'' h) E`. An evaluation with derivatives
then makes `m + 1` products of the size of a Gram (the `A_j` and `X A^-1`) where the
NumPy loops made `m + m (m + 1)` (8 for two penalties, now 3), beside smaller products of
the design with vectors, `X b_j` and `X b_jk` (`n p m` and `n p m (m + 1) / 2` multiplies).
For the grouped softmax
they come from the derivatives made in full, one at a time, as before. The hand-written
gradient and Hessian equal `jax.grad` and `jax.hessian` of the score written in JAX
(the information from `jax.hessian` of the log-likelihood, the fit as four Newton steps
from the optimum, written in the penalties' eigenbasis) to 7e-12 of their largest entry or
less (most below 1e-13),
one, two and three penalties (the third a ridge, without a null space) on 120 rows at
moderate and strong log penalties, every family (`tests/analysis/regression/test_reml.py`);
the central differences of the score
check them as before. The kernels hold few copies of the design: XLA's memory analysis of
the compiled programs (20,000 x 40 rows, CPU) gives temporaries of 1.1 to 1.2 times the
padded design for the Poisson and the Bernoulli at two or three penalties (2.1 at one,
where XLA keeps the one weighted copy and `X A^-1` at once), 1.0 for the Gaussian, and 3.4
to 4.0 for the grouped softmax, whose sandwich forms keep several weighted copies `u X`
for their segment sums at once (as its information kernels do).

The rotation of the design into the penalties' eigenbasis is a NumPy product, `X Q` with
`Q` the block-diagonal rotation (identity off the blocks). Assigning each block's columns
by index, as it was, cost 140 to 160 ms at 39,792 x 56 (two blocks); the product 24 ms.
The two are equal to the bit when each block's columns are contiguous and ascending, as the
designs of `terms` make them; with a block's columns permuted or interleaved with another's
they differed by up to 8.9e-16 in 6 of 28 layouts tried (an independent review,
2026-11-24), far inside the references' rule. A value of the design that is not finite
would be spread by the product over its row's columns of the block, so the design is
checked before it is rotated, and the count of entries not finite in the error is the
design's own. Moving it to the device would mean padding and moving the
design before the rotation and reading the rotated design back for the design check
(`newton._unidentified`) and for the columns running away; on the CPU XLA's product is
slower than BLAS's, so it would gain nothing there. What costs more at the rate model's
size is that each REML evaluation of a session rotates, checks, pads and moves its design
again; keeping a session's rotated design on the device from one evaluation to the next
is the step that would remove it.

**How the log-likelihood is summed.** The last step of a fit is taken when `F` does not
rise by more than `8 eps max(1, |F|)`, which only the rounding of `F` decides at the
optimum. Written as two dot products (`y' log mu - 1' mu`, NumPy's form), XLA's sum of a
Poisson session of 240 rows moved `F` by 2.0e-12 between two points 1e-15 apart in the
decrement, above the allowance (1.3e-12), and the fit refused a last step NumPy took
(`select.poisson_two_smooths`, a step count one lower). Each row's terms are therefore
combined before one sum (`sum w (y log mu - mu)`; for the grouped softmax `y log p` per
row): there the same two points differed by 0 to 1.1e-13.

The JAX fitter therefore takes some last steps the NumPy fitter refused, and the two can
differ by one Newton step outside the stored references. In 216 fits drawn at random by a
first independent review (200 Newton fits, 16 REML scores; 2026-11-23), 2 took one more
step (a Bernoulli fit of 2,500 rows, 9 steps to 10; the inner fit of a Poisson REML score
of 4,096 rows, 4 to 5). In 970 random fits of a second review (four families, 40 to 20,000
rows, 2 to 13 columns), 4 of 650 in one sample and 0 of 320 in another took one more step,
all of them Poisson (about 2.5 % of the Poisson fits of the first sample, none of the 480
of the other families). No status differed. Where it happens, the outputs differ by about
the stop's tolerance, 1e-8 (1.6e-8 in the REML score, log-likelihood, gradient and Hessian
of the Poisson case), not by the references' rule of `1e-11 + 1e-9 |b|`; the rate depends
on the data, the family and the platform. The NumPy sum had moved `F` by 16.8 eps |F|
along the Bernoulli step; under an exact sum (`math.fsum`) the step changes `F` by -0.99
eps |F| there and lowers it by 4.1 eps |F| in the Poisson case, so the JAX fit is the
better converged one. The per-row sum is within 1.5 eps |F| of the exact sum on fits of
500 to 20,000 rows. The rule compares two evaluations of the evaluation kernel, both on `X
@ beta`: `F` at the point where the fit stands and at the moved point, and the
log-likelihood the fit reports is that evaluation's. The iteration's own `F`, from the
line search on the carried `eta + scale * (X @ step)`, differs from it at the same
coefficients by up to 1.5 eps |F| on one sample (the last step of 299 random fits) and 5.4
eps |F| on another (298 fits; 7 above 1.5, the largest a Poisson fit of 100 rows and 3
columns from a far start), too close to the allowance to compare one with the other, as
the rule did for a while. Evaluating `F` again costs one more call of the evaluation
kernel per fit (rate size: 252 to 260 ms for `penalized_newton`, 3 %; 300 rows: 2.3 to 2.4
ms) and changed no status, step count or coefficient on 400 random fits. The carried
linear predictor drifts from `X @ beta` by the rounding of each step: 1 to 3 eps of the
largest `|X @ beta|` along the path after 15 to 49 capped steps from a far start (which is
up to 51 eps of the final `max |eta|`, the early predictors being larger); the second
review measured 1.3 to 2.6 eps max|eta| after 20 iterations, 3.0 to 3.3 after 50 and 9.4
after 100, growing about as the square root of the iterations. Nothing bounds it but
`max_iter`; `tests/analysis/regression/test_newton.py` pins it to 8 eps of the path's
largest `|X @ beta|` on a fit of 45 steps. Every stored case matches
(`tests/analysis/regression/data/numpy_reference/README.md`, "Differences from the NumPy
fitter outside the stored cases").

**Float64 for each call.** Every public entry that makes a JAX array or runs a kernel
carries `runtime.in_float64`, which enters a fresh `jax.enable_x64(True)` for that call.
The flag is never set for the whole process, because the decoders in the same process
compute in float32. In JAX 0.8.0 the flag is thread-local: a thread started inside the
scope sees it off, and the calling thread has its old value back after the call, also
when the call raises (measured, and `tests/analysis/regression/test_runtime.py`). The jit
caches are keyed on the flag (`jax/_src/config.py`, `include_in_jit_key=True`), so a
function called inside and outside the scope has two compiled programs. With the flag
off, `jnp.asarray` of a float64 NumPy array silently gives float32. So every kernel is
jitted through `runtime.float64_kernel`, which refuses when it is traced with the flag off
or with a floating input that is not float64. A float64 array made inside the scope stays
float64 after it, but arithmetic on it outside the scope is in float32.

**One device.** `in_float64` first calls `runs.device.require_one_jax_device`: the
fitter refuses to start unless this process configured a card (`configure_jax_gpu`),
`JAX_PLATFORMS` excludes CUDA, or `CUDA_VISIBLE_DEVICES` names at most one card. Only the environment is read, so the check
starts no backend. The message says what to set. A runner that fits calls
`runs.device.default_jax_platforms("cpu")` in `main` unless it asks for a GPU
(`simulate_declared_penalty.py`, `compare_regression_to_mgcv.py`,
`compare_regression_to_current_fitters.py`). The GPU's own check is a later step.

**Threads.** JAX 0.8.0 has no setting for the number of CPU threads. Its CPU client takes
no thread count (`jax/_src/xla_bridge.py`, `make_cpu_client`), and its pool follows the
process's CPU set: 7 threads on 1 CPU, 18 on 4 and 30 on 8, and a float64 `X'X` took 1.0
second of CPU time per second of wall time on 1 CPU and 6.5 on 8.
`--xla_cpu_multi_thread_eigen=false` changed neither (6.7 on 8; measured 2026-11-23, under
load). `os.sched_setaffinity(0, ...)` moves only the calling thread: called after the
backend started, it left the pool on 8 CPUs (7.3). The pool's threads are made when JAX
starts its backend and keep the CPU set they were made with, so a process is pinned
before its first JAX computation. `runs.threads.pin_to_cpus(cpus)` moves every thread of
the process to `cpus` and sets the BLAS thread variables to their number; it refuses when
numpy was already imported with another count, which its BLAS keeps. A pool whose
processes compute with JAX starts them by spawn, `runs.threads.pool_context()`: a fork of
a process with JAX's threads running can deadlock. `tests/repository/test_no_fork_pools.py`
fails when `src/` or `scripts/` asks for fork explicitly (`get_context("fork")`,
`set_start_method("fork")`). A pool made with no start method also forks on Linux, so the
same file fails when a module that imports `jax`, the regression fitter or
`non_local_detector` (directly or through this repository's modules) makes a `Pool` or a
`ProcessPoolExecutor` with no `mp_context=` or `context=`, other than by `pool_context()` or
`get_context(...)`. The present offenders are listed there with their reasons; the test
also fails when a listed one is gone.

**Padding.** `jit` compiles once for each new shape, so a session's rows are padded to a
bucket (`padding.bucket_size`): below a step, the next power of two and at least 256 rows
(`SMALLEST_ROW_BUCKET`); from the step up, the next multiple of it. The step is 4,096 rows
for the Poisson, Bernoulli and Gaussian families (`padding.pad_rows`: 256, 512, 1,024,
2,048, then 4,096, 8,192, ...), and 1,024 rows for the grouped softmax, whose groups go to
a multiple of 64 (`padding.pad_groups`: 256, 512, then 1,024, 2,048, ...). Columns are not
padded. A Bernoulli fit of 300 rows padded to 4,096 spent most of its time on the padded
rows; padded to 512 it took about half the time. Fits of 40 rows took the same time padded
to 64, 128 or 256 rows (one thread, 100 calls each, 2026-11-23), so a smaller bucket would
add compiled shapes for nothing. The grouped softmax's fits of 10 to 50 events (90 to 450
rows) took 15 to 25 % less time below its step; at 100 events and more the bucket is the
same. Groups stay at multiples of 64: a session has at least as many rows as groups.
A padded row has a design row of zeros, a
response of 0, an offset of 0 and a weight of 0, after the real rows. A kernel multiplies
each row's terms by the weight, so a padded row adds exactly 0 to the log-likelihood, its
gradient and its information; with a zero design row it adds nothing to the gradient or
the information even unweighted. Its terms are finite: an offset of minus infinity would
make a zero weight give `0 * -inf = nan`. For the grouped softmax the padded rows form
one group after the real ones, with a response total of 0, and the groups after it to the
bucket have no rows. A group with no rows has `log(sum exp) = -inf`; the log-likelihood is
summed by row (`y log p`), so no row reads that group and it adds 0 (a per-group sum would
give `n_g log(sum exp) = 0 * -inf`). Every quantity that depends on the number of rows uses
the real count: the Gaussian's normalizing constant and the profiled Gaussian's degrees of
freedom. No rows pad to one bucket of padded rows, every sum 0. A fit without rows is then
what the penalty alone makes it: where the penalty leaves a coefficient free (no penalty,
or a null space) it is "design not identified" at the start, log-likelihood 0, no step;
where the penalty has full rank it is "converged" at its minimum, `beta = 0` whatever the
start, with `A = S`, a decrement of 0 and a log-likelihood of 0 (the NumPy fitter raised
there, by accident); a grouped softmax without rows is refused (`ValueError`, no groups).
A per-row argument of the weighted design (`eta`, `y`, a per-row offset) needs one value per
row on one axis: a column of shape (n, 1) is refused with its name and shape, since it
would broadcast against the padded rows (a Poisson log-likelihood of 6 rows came out
-1536.6 for -5.82). So is a scalar or a one-element array given for a per-row vector, which
the NumPy code broadcast: that narrowing is intended. When the rows are padded
(`Poisson.pad`) a single Poisson offset is every row's, but a fit refuses it:
`Poisson.check`, which `penalized_newton` calls at its entry, needs one offset per row. Junk in the padded rows (a random design, a response of 5, an offset of 2) leaves
every quantity unchanged, exactly, and the kernels equal the formulas written in NumPy in
the test. The padded and unpadded log-likelihood, gradient, information and its two
directional derivatives agree to 1e-12 of their largest entry for each family, the kernels
on the padded rows against the same kernels on the rows unpadded, with groups that have no
rows, and so do the profiled Gaussian's scale and score
(`tests/analysis/regression/test_padding.py`). The hand-written score, information and
its directional derivatives equal JAX's automatic derivatives of the log-likelihood
(`jax.grad`, `jax.hessian`, `jax.jvp`) to 1e-12 of their largest entry, at most 8.2e-16
measured (`tests/analysis/regression/test_likelihoods.py`).

**Compilations.** `runtime.COMPILE_LOG` counts each kernel's traces by input signature.
JAX traces a jitted function once for each new signature and compiles each trace; a
program found in the persistent compilation cache is still traced (measured). So the count
is the number of programs a process needed, whatever the cache holds: ten Poisson fits of
ten row counts in four buckets trace each Newton kernel once per bucket
(`tests/analysis/regression/test_newton.py`). Each new bucket of a family costs its first
fit about 2 to 3 s of compilation on one CPU thread.

**How many programs a process keeps.** Each compiled XLA CPU program holds memory mappings
of its own: 111 for an iteration kernel, 24 for an evaluation kernel and 6 for an
information kernel (CPU, 30 shapes each, 2026-11-23), and 1.5 to 4.3 MB of resident
memory. Linux stops a process at `vm.max_map_count` (65,530 by default; the allocation
fails inside LLVM and the process dies): a stream of 800 random fits of 2 to 13 columns
and 40 to 20,000 rows reached 735 programs and 64,823 mappings at about fit 775 and
aborted, and the finer row buckets had made it come about 1.7 times sooner.
`jax.clear_caches()` releases them (5,048 mappings to 672 after 93 programs;
`gc.collect()` adds nothing), though not the resident memory the allocator keeps (786 to
774 MB). So `runtime.in_float64` counts the programs traced since the last clear and, at
the entry of a call when no call of the fitter is running in any thread (never inside a
fit, never while a kernel runs), clears JAX's caches once the count has reached
`runtime.LIVE_PROGRAM_LIMIT`, 256: 256 iteration kernels are 28,400 mappings, under half
the default limit with the process's own and one call's new programs added; the 800-fit
stream peaked at 21,021 mappings and 1.3 GB (2.4 GB without the bound at fit 675) and ran
to the end. `jax.clear_caches()` clears every jitted function of the process, another
library's too. A process that fits many designs of different column counts and row buckets
recompiles each after a clear: the first 700 fits of that stream took 849 s with the bound
against 595 s without it (two processes at once on two CPUs under load). JAX's persistent
compilation cache, which this project does not configure, makes the recompilation cheap:
with `JAX_COMPILATION_CACHE_DIR` set (and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`,
so that every program is stored) the same 700 fits took 554 s, a long driver's choice.
`tests/analysis/regression` without its slow tests traces about 435 programs in one
process, so it clears once or twice; the tests that count traces fix the limit high for
their own run.

The clear can fire only at the entry of an outermost call, so which functions carry
`in_float64` decides where the bound holds. The loop over fits (`sessions.fit_sessions`)
and the calibration (`calibration.log_penalty_for_edf`, which runs that loop at each trial
penalty) do not carry it: each unit's fit is an outermost call, the caches can be cleared
between two units, and the live programs stay bounded through a whole cohort
(`test_the_program_bound_holds_between_the_sessions_of_one_loop`,
`test_the_program_bound_holds_during_a_calibration`). The penalty searches
(`select_penalties`, `refine_penalties`, `fit_at_penalties`, `pooled_reml`), `refit_session`
and `term_collinearity` do carry it, so that no clear falls between two evaluations of one
search, which would compile every session's programs again at each evaluation
(`test_a_search_clears_the_caches_at_most_once_at_its_entry`). **Inside one search the
bound therefore does not hold**: a search keeps one set of programs for each row bucket and
each number of active columns among its sessions. How many that is for a full cohort has
not been measured; measure it (`runtime.live_programs()`) before a search over a whole
cohort is run, and see the planned fixed-width kernels (inactive columns as a mask), which
make the count the number of row buckets.

**Importing the fitter imports JAX.** `likelihoods`, `newton` and `reml` import JAX, and
so every module of the fitter that imports them (`sessions`, `calibration`, `record`,
`terms`, `support`, `collinearity`, `simulation`) and the modules that import those (the
rate model's `visit_rate`, `visit_rate_statistics` and `visit_rate_scenarios`). `folds`,
`bands`, `contrasts`, `pspline`, `padding` and `constants` do not. Importing `sessions`
took 1.6 s against 1.1 s before (three runs each, under load, 2026-11-23).

## Cost

**A fit of a larger unit** (scratch measurements of 2026-11-24, whose script is not in the
repository: synthetic rows, CPU, one thread, a shared machine at a load near 32; timings, not
results; `fit_sessions` at the day unit with baseline copies, before and after the collinearity
record was made one session at a time):

| Unit | Rows x columns | `fit_sessions`, before / after | collinearity record, before / after (its peak memory) | the fit | peak memory of the process, before / after |
|---|---|---|---|---|---|
| content-like, 60 sessions x 110 events x 9 arms, arm baselines | 59,400 x 494 | 115 s / 6.8 s | 109 s (1.4 GB) / 0.34 s (1.5 MB) | 5.3 s | 2.6 / 2.2 GB |
| rate-like, 5 sessions x 40,000 bins, intercept baselines, 3 smooths | 200,000 x 97 | 6.0 s / 4.2 s | 3.2 s (0.94 GB) / 1.6 s (0.15 GB) | 1.9 s | 2.1 / 1.5 GB |
| rate-like, 58 sessions x 40,000 bins | 2,320,000 x 150 | 203 s / 47 s | 172 s (16.8 GB) / 15 s (0.16 GB) | 19 s | 20.8 / 14.6 GB |

The dense design fits at the animal size. Before, the collinearity record projected each
baseline copy out of the whole weighted design; now each session's copies are projected out of
its own rows (`collinearity.term_collinearity`, `session_copies`). The copies, the support count
and the record of the sessions take at most 12 s (the animal-size rate unit). A loop over 45
content-like day units and 5 animal units took 32 s and left at most 37 live programs (the
limit is 256; 48 before, when the whole unit was weighted at once).

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

The fit at fixed penalties in JAX, against the NumPy solver before it, by
`scripts/checks/time_regression_fitter.py` (its `--child` cases) and two small Bernoulli
fits of 23 columns through `reml.refit_session` (300 and 1,500 rows, 200 calls each). One
thread, each process pinned to one CPU, the code versions run in turn in the same minutes
(NumPy, JAX before, JAX after each change, then again), on a machine under heavy load
(1-minute load average 60 to 130 on 96 CPUs, no idle core; 2026-11-23). Each cell is the
least time over all calls of all rounds, then the median of the rounds' medians, in ms.
NumPy at commit 7959aca7; JAX before the changes at 2592ef01 and after them at efc2bb3a:

| Case | Rounds | NumPy: least, median | JAX before: least, median | JAX after: least, median | After / NumPy: least, median |
|---|---|---|---|---|---|
| fit_rate (37,000 x 76) | 9 | 242, 298 | 534, 690 | 245, 381 | 1.0, 1.3 |
| fit_content (400 events x 9) | 8 | 12.4, 14.6 | 16.6, 20.0 | 11.3, 16.0 | 0.9, 1.1 |
| fit_bernoulli (20,000 x 23) | 3 | 35, 53 | 45, 57 | 31, 52 | 0.9, 1.0 |
| fit_gaussian (20,000 x 23) | 3 | 21, 37 | 33, 43 | 27, 35 | 1.2, 1.0 |
| fit_sessions_30 | 3 | 8,855, 9,968 | 11,724, 13,778 | 7,249, 9,353 | 0.8, 0.9 |
| Bernoulli, 300 x 23 | 8 | 0.92, 1.00 | 6.4, 9.0 | 2.4, 3.3 | 2.6, 3.3 |
| Bernoulli, 1,500 x 23 | 8 | 1.8, 3.1 | 6.8, 9.3 | 3.9, 4.9 | 2.2, 1.6 |

The times of one round can be twice those of another, so read the ratios of the least
times; the Bernoulli and Gaussian cases at 20,000 rows, with three rounds, are within the
noise of NumPy. What each change gave:

- The gradient as `score @ X` and the Gram as `X.T @ (w[:, None] * X)` (no copy of the
  transposed design, above): the rate model's Newton iteration from 90 to 46 ms, the whole
  rate fit from 534 to 287 ms; the content fit's iteration from 2.4 to 2.1 ms.
- The row buckets below the step: the fit of 300 rows from 5.9 to 3.2 ms, of 1,500 rows
  from 6.9 to 5.1 ms. They added 44 programs to the 389 that `tests/analysis/regression`
  without its slow tests traced, each new bucket of a family costing its first fit about
  2 to 3 s of compilation.
- The iteration's scalars in one transfer, and its arrays fetched when read: measured on
  `penalized_newton` alone, the two versions called in turn in one process, the fit of 300
  rows from 3.1 to 2.3 ms, of 1,500 rows from 3.8 to 3.2 ms, nothing at the rate size.
- One transfer of the padded arrays: the same way, 300 rows from 2.4 to 2.0 ms, 1,500 rows
  from 5.5 to 4.6 ms, nothing at the rate size.
- The line search on the carried `eta + scale * (X @ step)`: the same way, the rate
  model's `penalized_newton` from 244 to 220 ms (median 254 to 230), about 7 % of its whole
  refit; the small fits did not change.

Peak memory did not change (the rate case's process 856 to 876 MB before, 853 to 858
after). `tests/analysis/regression` without its slow tests traced 373 programs before and
435 after (44 of them the new buckets, 16 the test of the compiled programs, two the test
of a batch of fits) and took 402 and 418 s on one CPU in the same minutes, so its time is
in its compilations, not its fits. Its slow test of penalty selections on small cells
(`test_simulation.py::test_coverage_and_size`, 2,000 selections of about 100 summed cells)
took 415 s before and 183 s after, in the same minutes. What remains of a small fit is the
host: the cost of each kernel call (about 0.3 ms for an iteration of 300 rows), the design
check (`_unidentified`) and REML's rotations. REML's copy of the design into the
penalties' eigenbasis (`reml_and_derivatives`, in NumPy) is now one product, 24 ms at
39,792 x 56 where assigning each block's columns took 140 to 160 ms (the 55 to 60 ms of a
rate refit measured before it were of that older form, at 37,000 x 76), and XLA's product for the Gram is slower than BLAS's `syrk` for an unweighted
`X' X`, which XLA does not call.

REML's gradient and Hessian in JAX, and the rotation by one product ("REML's
derivatives", above), against the code before them (2e0ea0d0), by the `--child` cases of
`scripts/checks/time_regression_fitter.py` run in one process per version and round. One
thread, each process pinned to one CPU, the versions in turn on the same CPU in the same
minutes, under heavy load (1-minute load average 70 to 82 on 96 CPUs; 2026-11-24). Each
cell is the least time over all calls of all rounds, then the median of the rounds'
medians. "Derivatives" is the derivatives in JAX alone (0bc534a0), "after" with the
rotation as well (44d9ec2e). Under that load the table resolves differences of about
25 %, and smaller ones are not to be read: `fit_rate` computes no derivatives, so its
"derivatives" column, slower than "before", is noise.

| Case | Rounds | Before: least, median | Derivatives: least, median | After: least, median | Traces before, after |
|---|---|---|---|---|---|
| select_cost, 2 sessions of 39,792 x 56 | 3 | 23.0 s, 25.0 s | 21.9 s, 24.8 s | 15.7 s, 18.9 s | 5, 4 |
| fit_sessions_30 | 3 | 15.6 s, 18.3 s | 16.9 s, 17.5 s | 11.4 s, 11.9 s | 16, 16 |
| fit_rate (37,000 x 76) | 8, 4, 8 | 430, 544 ms | 468, 693 ms | 342, 491 ms | 3, 3 |
| calibration_30 | 1 | 247 to 254 s | | 238 s | 16, 16 |

One REML evaluation of one `select_cost` session with its derivatives (warm start, seven
calls) took 688 ms before and 418 ms with the derivatives in JAX; the score alone 245 and
238 ms. The search spends most of its evaluations on the score alone (the grid), so the
selection gains less than an evaluation; the rotation, which every evaluation pays, gave
the rest (it took 140 to 160 ms of each, now 24 ms). Evaluations of 120 rows and two
smooths with derivatives (200 calls each, only two rounds, so these few milliseconds are
less certain than the table): Poisson 3.7 to 2.3 ms, Bernoulli
3.6 to 2.3 ms, Gaussian 2.6 to 2.5 ms, grouped softmax 5.3 to 3.1 ms; what is left is
the host (two Newton iterations, three evaluations of `F`, the design check, the
transfers). The slow test of 2,000 small selections
(`test_simulation.py::test_coverage_and_size`) took 406 s before and 343 s after, run at
the same time on two CPUs (290 s before, alone, earlier the same day: the load moves it
more than the change does). `tests/analysis/regression` without its slow tests, over the
464 tests both versions have, took 605 s before and 591 s after (at the same time) and
traced 610 and 519 programs: a Laplace family's REML derivatives are one program where the
public `dinformation` and `d2information` were two, and a selection's traces fell from 5
to 4. The 19 tests added with the change (the derivatives against automatic
differentiation, the padding of the new kernels, the traces, the canary) bring the whole
to 583 programs and about 820 s. The calibration and the per-session fits compute no
derivatives; they gain from the rotation alone, and calibration_30's single calls are
within the noise.
