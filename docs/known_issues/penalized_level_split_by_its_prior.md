# A penalized level can be split from a session's intercept by its prior alone

**Status:** open, recorded 2026-11-25; no production design has this shape.
**Affects:** the regression fitter's reference-level check
(`analysis/regression/sessions.py`, `_Session._require_reference_rows`), for a fit of
several sessions (by day or animal) with a baseline intercept per session and a
penalized category shared by the sessions.

## What

The fit loop refuses a category without rows of its reference level. For a category
that is not a baseline it asks for those rows in one session of the fit, because the
category's levels are shared. That identifies the levels only when the intercept is
shared too. With a baseline intercept per session, a level that a session holds alone
is collinear with that session's intercept; an unpenalized fit is refused as not
identified, but a penalized one passes the check and the level's prior alone sets the
split between the level and the session's intercept.

## Measured

A Poisson fit by day of two sessions of one day, a baseline `(Intercept)` per session,
`Category("level", ("a", "b", "c"), "a")` at prior SD 2 and an exposure of 1: the first
session's 20 visits at `a` and `c`, the second's 12 at `b` (counts drawn at mean 2,
`default_rng(0)` and `default_rng(1)`). 2026-11-25:

| level penalty | result |
| --- | --- |
| prior SD 2 | converged; `level[b]` = -0.0, nothing left out |
| none | refused: design not identified, in `(Intercept)@sim/20260101/04_r2`, `level[b]` |

## What is open

The condition the check should test is that each shared level has rows in some session
that also has rows of another level (so the session's intercept does not absorb it), or,
more generally, that the design stays identified with the penalized simple terms counted
as free. Both change which fits are refused, so the choice is the lead's or the
regression code's author's.
