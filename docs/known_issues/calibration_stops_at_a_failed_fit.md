# The calibration of a penalty stops when one unit's fit fails at a trial penalty

**Status:** open, recorded 2026-11-26; to be settled on the real rows in the declared-penalty
step, before any real fit of the rate model.
**Affects:** the declared-penalty calibration (`analysis/regression/calibration.py`,
`log_penalty_for_edf`), and through it `visit_rate.calibrate_log_penalty` and the rate
simulation's `visit_rate_simulation.calibrate_clocks`.

## What

`log_penalty_for_edf` searches by bisection for the log penalty at which a smooth's median edf
over the units of the fit is the target. At each trial penalty it fits every unit through
`sessions.fit_sessions`. When a unit's FIRST fit ends neither converged nor with no finite
maximum (for example "step limit reached"), `sessions.py`, `_Session.fit`, raises
`ValueError("<unit>: at its first fit its fit ended as step limit reached, a numerical
failure")`. Nothing in `calibration.py` catches it (it passes through `_term_edf` and the
search's `at`), so the whole calibration stops. A unit whose fit is tried and not kept, or in
which the term is left out, has an edf of NaN and drops out of the median; a unit whose first
fit fails does not drop out: it stops the search.

## Where it was seen

The numbers below are simulation numbers on synthetic rows (no real table was read).

- On a synthetic cohort whose visits after an omission lasted 0.5 to 8 s (5 animals of 4 days
  of 2 sessions, 100 visits and 200 expected events a session), the rate model's curve after
  an omission, at the day and at the animal unit, with targets above the edf at a log penalty
  of -2.5: the bisection tried -6.25, where the curve has no row beyond the longest visit
  after an omission, and a unit's first fit ended at the step limit (2026-11-26).
- On the present cohort shape (visits after an omission of 0.5 to 12 s), the review reproduced
  it for one day unit (2 sessions of 100 visits, 200 expected events each) of one seed of three
  at a log penalty of -6.25 for the omitted curve, with the curves centered on Q2's window and
  with the curves centered on the rows alike: the centering is not its cause.
- On `visit_rate_simulation.VALIDATION_COHORT` the search reached every target of
  `CLOCK_TARGETS` (2026-11-26): the bisection did not try a penalty at which a unit fails.

## How to reproduce

On synthetic rows: one day of two sessions from `visit_rate_simulation.synthetic_visit_rows`
(100 visits each, visit lengths 1 to 14 s, after an omission 0.5 to 12 s, counts drawn from a
constant ratio of 1.5 with 200 expected events a session), fitted by
`visit_rate.fit_visit_rate(..., settings=VisitRateSettings(fit_by="day",
log_penalties={OMITTED_ARRIVAL_TERM: -6.25}), allow_provisional="test")`, for seeds 0 to 2: one
seed raises. The review's script is `steplimit_check.py` (scratch,
`scratchpad/rate_fix_review/`, not in the repository).

## What is not decided

- How a unit whose fit fails at a trial penalty should count in the median: left out (as a
  fit not kept is), counted as out of reach on that side of the bracket, or a stop as today.
- Whether the search's lower bound (`LOG_PENALTY_BOUNDS`, -10) should move for these curves, or
  the search should start from a bracket known to fit.
- Both are settled on the real rows, in the declared-penalty step, before any real fit. Nothing
  in `calibration.py` is changed until then.
