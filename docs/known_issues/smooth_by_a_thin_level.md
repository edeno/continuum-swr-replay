# A smooth by a level stays in a fit however few units the level has

**Status:** open, recorded (lead, 2026-11-25): not fixed now; giving a smooth by a
level a support count would be a design change.
**Affects:** the regression fitter's support rule (`analysis/regression/support.py`),
for a model with a smooth fitted `by` a level, such as visit-rate model A in its declared
form, `reward_form="by_outcome"` (`analysis/replay_models/rate/visit_rate.py`: an arrival
curve for each last outcome, the convention's twenty-sixth round). The form of the
measurement below, a shared curve plus a curve by the rewarded outcome
(`"time_varying"`), was removed on 2026-11-26.

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
measured 2026-11-25 with the empty-level fix applied):

| | value |
| --- | --- |
| status | converged |
| `rewarded` (the slope) | left out, `support` (`n_high` 2, `n_low` 58) |
| edf of `s(clock):rewarded=1` | 1.11 |
| the by-smooth on the rewarded rows | mean 0 (its sum-to-zero constraint), from -0.67 to +0.49 in log rate |

The smooth carries no mean shift, which is the slope's, but it fits a time course of
the reward effect from two visits with two events, where the slope would not be
fitted at all.

## The mirror case stops the whole fit

A session whose rows are all of the level (every visit rewarded) is not made inactive
either. Its reward slope fails the support rule (no unrewarded units), and its smooth
by `rewarded = 1` then equals the clock smooth on every row, so the two smooths' null
spaces are collinear and the session's first fit raises "design not identified, in the
coefficients s(clock) ..., s(clock):rewarded=1 ...", which stops `fit_sessions` for
every session (the measurement above with the second session rewarded on all 60
visits; 2026-11-25). It needs every visit of a session (or of a fit by day or animal)
rewarded, which the task's reward probabilities make unlikely; it is not handled.

## The by-outcome form (2026-11-26)

The declared form of the rate model meets the first case and not the second.

- **A unit with 1 to 4 visits after an omission** (a session, or a day): the outcome
  coefficient fails the support rule and is left out, so Q2 and the onset test have no
  value; the arrival curve after an omission stays in the fit, fitted from those visits.
  Q1 reads that curve, so the rate model's window support rule
  (`visit_rate.window_support`, the author's rule) marks Q1, Q2 and the onset test
  unsupported where fewer than `MIN_UNITS` visits of a side reach a window the contrast
  reads. The values are kept, with the counts beside them, and left out of the reduction
  over units.
- **A unit with no visit after an omission**, or every visit rewarded: the curve after an
  omission has zero columns there and fails the support rule; the outcome coefficient
  fails it too. The form has no shared arrival curve, so nothing is collinear with the
  curve after a reward, and the unit is fitted
  (`test_visit_rate.py::test_a_unit_with_few_omission_visits`). The mirror case above does
  not stop a fit of this form.

## What is open

Whether a smooth by a level gets a unit count like a slope by a level: the units with
a row of its level, failing below `MIN_UNITS`. That changes which fits keep the
smooth, so it is a decision for the lead, not a repair. Until then, a session's
by-smooth fitted from few units of its level should be read with the slope's support
entry beside it.
