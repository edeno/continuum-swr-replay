# The shared regression fitter: how its rules came to be (to 2026-11-21)

The measurements and earlier rules behind
[docs/design/regression_fitter.md](../../../docs/design/regression_fitter.md), moved
out of it on 2026-11-21 so that it describes the method as it is. Commit `ef98019d` is
the fitter before the eigenbasis, the set-aside sessions and the decrement rule of the
penalty search. The text is as it stood in the design document.

## The fit

- The Gaussian takes its full step without the cap: capped, an outcome of spread 1,000
  did not converge in 100 steps.
- A fit whose decrement is small while its step still moves the linear predictor by
  more than 0.1 was reported converged before the rule, with a coefficient near 23 and
  a standard error near 1e4.
- The design check uses the design the likelihood sees: checked on `X` itself, a
  grouped-softmax design with a combination of columns constant within every group
  passed and its fit ended as "information not positive definite" (one of 242 simulated
  content sessions); it is now "design not identified".

## Strong penalties

With `S` as given, two rounding errors grow with `exp(rho)`: `b' S b` is a difference of
terms of size `|S| |b|^2` (at log penalty 16 on a content session it came out -8.4e-8
where it is 2.1e-7, an error ten times the last Newton step's predicted gain of 3.4e-8,
and the fit stopped as "step halving failed"), and the stored `S_j` has rounding of
about `1e-16` of its largest eigenvalue on its null space, which `exp(rho)` turns into a
penalty on the straight lines (on 242 content sessions the pooled score was 3.4e-6 too
high at log penalty 10, 1.1e-4 at 14, and rose from 14 to 16 where it should fall).

On a simulated content model (242 sessions of each of two kinds, log penalties 14 to
20, 3,388 fits) 47 fits stopped as "step halving failed" at ef98019d; in the eigenbasis
41 of them converge and 6 stop as "no finite maximum", in the sessions whose unpenalized
columns separate; none fails otherwise. The bounds `[-10, 20]` are kept: the fit at 20
is obtainable, and the score there is within its last few `1e-8` of the limit.

A stacked design (one `SessionData` per animal): the mask already gave the stack built
without the inactive columns at ef98019d; what was missing was the names of the columns,
and the whole five-animal search raised.

## The penalty search

- The eigenvalue floor on the scaled Hessian: floored over the unscaled Hessian, a term
  of curvature 1,000 floored one of curvature 2e-5 or 2e-6 at 1e-3: the weak term
  crawled for 50 steps and the search raised, and one of 2e-7 was reported converged
  eight units short of its optimum (now: one step to the optimum, and for 2e-7 the stop
  of the term alone).
- The search does not stop where the curvature is negative: started 1e-5 from a maximum
  of the score, it was reported converged at the maximum.
- The decrement rule replaced a gradient rule, largest free gradient component below
  `1e-6 * max(1, |score|)`, which moved with the score's size, and so with the units of
  the exposure (the Poisson) or of the outcome (the Gaussian's profiled scale). On 30
  synthetic sessions of 40,000 bins (score 63,852) the old rule stopped 4.6e-6 from the
  optimum in one log strength; the new one ends within 1e-11 of it. On the four mgcv
  fixtures the selected log strengths are the same to the last bit, and over the 17
  searches of the REML tests they move by at most 3e-9. With the decrement's tolerance
  set to zero the search ends where the decrement is below the score's rounding instead
  of on a step too small to move `rho`.
- A flat score: on 242 simulated content sessions with a flat score from log penalty 5
  on, the search raised at ef98019d (the score's rounding at strong penalties made it
  rise where it falls, by 3e-4 from 14 to 16, and inner fits failed from 17); it now
  ends at the bound, the edf 1.000 in every session.
- Before the screen, one session without a finite maximum made every pooled score
  infinite, so the whole model raised.

## The simulated content model (2026-11-21)

On the simulated content model (242 sessions per model and kind) the M0 and M1 models
without SWR set aside 4 sessions each, all "no finite maximum"; each was fitted at the
chosen log strength (2.57 and 2.95) with 5 or 6 arm and role columns inactive, in one
round, so no session is without a fit and 238 sessions chose the penalty. The models
with SWR set aside none and end at the bound, every session a line. With two smooths
(M3) the search raises at the reference point: one session's design is not identified
for the grouped softmax (columns 21 to 24 and 28 are collinear within events, which the
study's per-column support rule did not see); it was set aside before, and earlier still
its fit failed as "information not positive definite". With that session and the
separated ones left out by the study's own screen, M3 returns `[5.30, 19.38]` without
SWR and `[19.70, 20]` with it.

## The Gaussian scale

Until 2026-11-21 the Newton fit behind the profiled Gaussian score ran at the working
scale `var(y)` (with the penalty divided by it, the same minimizer), so that its
decrement, and the tolerance it was held to, did not depend on the outcome's units: for
an outcome scaled by 1, 1e3 and 1e6 the curve was the same up to the scale and the
selected log strength the same to 3e-15 (and at 1e9 too), with every grid point finite.
Before that, at an outcome scaled by 1e6, the inner fit at the four largest strengths
stopped at 100 steps on a decrement of rounding (5e-8 to 9e-5 in squared units of the
outcome). The working scale was removed when the Gaussian fit became one exact Newton
step, which needs no tolerance. A constant outcome was refused as a working scale that
is not positive before it was named ("the Gaussian outcome is constant").
