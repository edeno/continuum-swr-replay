# A smooth by a level stays in a fit however few units the level has

**Status:** open, recorded (lead, 2026-11-25): not fixed now; giving a smooth by a
level a support count would be a design change.
**Affects:** the regression fitter's support rule (`analysis/regression/support.py`),
for a model with a smooth fitted `by` a level, such as visit-rate model A with
`reward_form="time_varying"` (`analysis/replay_models/rate/visit_rate.py`, the arrival
clock smooth by the rewarded outcome).

## What

The support rule counts units only for simple terms: a level, a 0/1 slope and a
continuous slope pass when `MIN_UNITS = 5` units have a row on each side (a level, on
its own side). A smooth and a lag functional have no counts. Since 2026-11-25 one
whose columns are zero on every row of a session (a session with no rows of its
level) fails the rule there and is inactive; one with rows of its level, however
few, stays in the fit.

So in a session where the level has rows on 1 to 4 units, the slope by that level
fails the support rule and leaves the fit, while the smooth by the same level stays
in and is fitted from those few units.

## Measured

A simulated Poisson session of 60 visits of 10 bins (0.05 s each), with rewarded rows
on 2 visits (20 rows, 2 events there); terms: a clock smooth, `Slope("rewarded")` at
prior SD 2, and the clock smooth by `rewarded = 1` at log penalty 2, the design built
on this session's rows and those of a session rewarded on half its visits (log rate
`log 3 - 0.1 clock + 0.4 rewarded`, one `default_rng(0)` drawing the sessions in turn;
measured 2026-11-25 on master at `e1ccae02` with the empty-level fix applied):

| | value |
| --- | --- |
| status | converged |
| `rewarded` (the slope) | left out, `support` (`n_high` 2, `n_low` 58) |
| edf of `s(clock):rewarded=1` | 1.11 |
| the by-smooth on the rewarded rows | mean 0 (its sum-to-zero constraint), from -0.67 to +0.49 in log rate |

The smooth carries no mean shift, which is the slope's, but it fits a time course of
the reward effect from two visits with two events, where the slope would not be
fitted at all.

## What is open

Whether a smooth by a level gets a unit count like a slope by a level: the units with
a row of its level, failing below `MIN_UNITS`. That changes which fits keep the
smooth, so it is a decision for the lead, not a repair. Until then, a session's
by-smooth fitted from few units of its level should be read with the slope's support
entry beside it.
