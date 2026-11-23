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
| `terms.py` | the model terms, the design they build on the pooled rows, the names of terms and coefficients, the standardization and the design check |
| `collinearity.py` | how well the other terms explain each term, on the weighted design |
| `support.py` | the support rule (the row rule for an inactive level is applied in `sessions.py`) |
| `sessions.py` | the per-session fitting loop and the row rule for an inactive level |
| `calibration.py` | the declared log penalty of a per-session smooth: the one at which its median edf over the sessions is a target |
| `record.py` | the fit-row record and its parquet storage |
| `contrasts.py` | scalar estimands as linear functionals of a fit, with their exact variance |
| `folds.py` | the one fold rule |
| `bands.py` | simultaneous bands for a curve |

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
candidate) cancels in each group.

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
a model) of the sessions end without a fit (`PenaltySearch.share_without_fit`), the
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
(`PenaltySearch.n_grid_not_finite`, `grid_failures`: the count per status).
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
refused as such ("the Gaussian outcome is constant").

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

A built design describes each term once, as a `DesignTerm`: its kind (intercept, level,
0/1 slope, continuous slope, smooth, lag functional), its columns, its level, its
covariate, its penalty block and, for a level, the terms that leave a session with it.
The support rule, the design check and the per-session loop read these descriptions.

A smooth's penalty is required: `"none"` or a fixed log penalty. No penalty is chosen
from the data in a per-session fit: the rule for a smooth term there is open with the
lead.

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

## The support rule

`support.py` decides from the rows, before the fit, whether a session can estimate an
unpenalized term: a level of a category passes on the units with an event on its rows, a
0/1 slope and a continuous slope when both of their sides reach `MIN_UNITS = 5` units with
an event. The rule, the units and what a row holding an event is are in its docstring. A
term that fails in more than `MAX_FAILING_SHARE` (20 %) of the sessions with a fit leaves
the per-session model (`terms_failing_share`).

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
session it set aside and the per-session loop both call it, so the rule is in one place.

A session with columns left out is reported with those terms missing, not left out of a
summary, because leaving it out would depend on the outcome: the sessions that fail are
those with no events in one condition. Its record names each column left out with the
cause and the round, and holds NaN for those coefficients and their covariance. A term of
several unpenalized columns leaves a session whole (`sessions.py`'s docstring).

## The per-session loop

`sessions.fit_sessions(design, sessions, lik, model=..., kind=..., response=..., unit=...,
table_sha256=...)` fits every session alone and returns one record per session. Its
module docstring states the steps: missing rows; the levels to a fixed point; every other
term once on the final rows; the fit; a level the refit leaves out is removed and the
steps run again.

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

`record.FitRecord` is one row per (model, kind, session); its docstring lists every field.
A session with no rows, or no events of the kind, left is not fitted: no fit is tried,
and its record has the status "not fitted" and NaN coefficients. A fit tried and not
kept keeps the last fit's status and the true counts, with NaN coefficients, and lists
what the refit had left out before it ended, as a record of what was tried (the rule is
in `sessions.py`'s docstring). A record
whose status is not "converged" is kept; its readers flag it. Every cause is a code of a
closed set, so that sessions can be counted by cause: `support.Removed` for a term left
out (`left_out`, with the refit's round in `left_out_round`), `record.NoFit` for a
session without a kept fit (`no_fit`); `record.describe` gives their words. `write_records` and
`read_records` store the records through `tables.io` (`write_table`, `read_table`), the
structured fields as JSON strings.

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
