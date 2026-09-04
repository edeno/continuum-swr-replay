# Builder test floor

The covariate builders in `analysis/replay/covariates/` compute the values that
feed every replay regression. The schema/timing gates check a covariate's *name*
and *availability window*, not its *value* — so a units error, a wrong window
edge, or a flipped `searchsorted` side would ship silently unless a test pins the
value. This doc records the measured floor and the target.

The measurement tool is `scripts/qa/mutation_check.py`: it mechanically mutates
value-critical operations (comparison operators, `searchsorted side=`, `+`/`-`)
and reports which the covariate test subset catches ("killed") versus not
("survived"). Survivors are coverage gaps.

## Starting score (baseline)

Run over `covariates/candidate_arm.py` and `covariates/history.py` against
`test_covariates.py test_uncertainty_covariates.py test_covariate_timing.py`:

- **Killed 83 / 119 (70%). 36 survivors.**
- Highest-signal survivors: five `searchsorted side="right"→"left"` flips at
  `history.py:462, 547, 698, 747, 1519`. `side` decides whether the current
  outcome falls inside a reward-history window (convention A). The tests do not
  catch flipping it.

### Survivor list (to triage and close)

`side=` (highest priority): `history.py:462, 547, 698, 747, 1519`.

Comparison operators: `history.py:147, 149, 167, 304, 336, 337, 338, 339, 1448,
1524, 1532`; `candidate_arm.py:173, 185`.

`+`/`-` binops: `history.py:359, 444, 445, 700, 702, 748, 1313, 1314, 1421,
1514, 1525`; `candidate_arm.py:170, 203, 260`.

## After the builder-test phase

`tests/unit/test_builder_value_floor.py` adds value-level boundary tests, and
`scripts/qa/mutation_check.py`'s `_DEFAULT_TESTS` now runs **six** files —
`test_covariates.py test_uncertainty_covariates.py test_covariate_timing.py
test_builder_value_floor.py test_recent_trial_reward_rate.py
test_recent_reward_rate.py`.

**Score: killed 93/119 (78%), 26 survivors, zero `side=` survivors** (measured
against that six-file set). The starting 70% / 36 above was measured against the
first three files only, so the raw 70%→78% is not a clean apples-to-apples delta
(the six-file set is the current, larger kill set); the meaningful result is the
**zero `side=` survivors** — all five `searchsorted`/join-type mutants
(`history.py:462, 547, 698, 747, 1519`) are killed, verified by applying each
flip and confirming a named test fails. Re-run the tool for the current number;
it uses the six-file set by default.

The raw rate stays at 78% by design, not for lack of trying: the honest triage
below shows the remaining survivors are value gaps that need the full event
fixture (deferred to the I/O-purity phase, where the builders become unit-
testable in memory) plus input-integrity guards and equivalent mutants. Padding
the rate with contrived fixtures here is explicitly avoided.

The remaining survivors are triaged below. Not every survivor is a value gap:
the raw kill rate is a weaker target than a clean triage, so per the plan we
kill the value gaps we can isolate, defer the value gaps that need a full event
fixture to the I/O-purity phase (which makes the enclosing builders unit-
testable in memory), and document the guards and equivalents.

### Killed by the value floor

`history.py`: all five `side=`/join sites; `700` (window denominator is `k`,
not total count); `1524` (trunk arms take 0.0, not a leaf variance); `1532`
(Beta-variance range guard); `147` (non-monotonic-arrival integrity guard).

### Window-arithmetic survivors not yet distinguished by the fixtures

- `history.py:702` — the window *numerator* `cum[counts] - cum[window_starts]`.
  The denominator test uses a window whose start is 0 (`cum[window_starts] == 0`),
  so `+` and `-` agree there; killing it needs a fixture where the window start
  is past a rewarded experience (e.g. `[1,1,0,1,1]`, `k=2`). Real value gap,
  closeable with one more fixture.
- `history.py:748` — the lag index `idx = counts - j` in `lagged_trial_rewards`.
  Under `+`, `idx = counts + j` runs off the end for the most-recent lag; the
  boundary lag test exercises the region but does not force the out-of-range
  read on every lag. Real value gap, same fixture family.

### Value gaps deferred to the I/O-purity phase (need a full event fixture)

- `candidate_arm.py:203` — `is_last_rewarded_noncurrent_arm = is_last_rewarded *
  (1 - is_current)`; the `-` masks out the current arm.
- `candidate_arm.py:260` — `omitted = 1 - rewarded`; the omission indicator.
- `history.py:359` + `336-339` — `spline_visit_recency_within_event`: within-
  event mean-centering (`out - means`) and the eligibility mask
  (`arm < len(WELL_ARMS)`, `never_visited == 0`, `is_current == 0`,
  `is_prev == 0`).
- `history.py:1314` — `_fit_alpha`'s Bernoulli NLL (`reward*log(p) +
  (1-reward)*log(1-p)`); the per-animal RL learning-rate objective.

These live inside `attach_candidate_arm_covariates` and the patsy spline, which
today require the on-disk contingency/event fixtures. Phase 5 (I/O-pure
builders) passes those frames in, at which point a value assertion on each is a
plain in-memory test; the tests are written there, not padded in here.

### Input-integrity guards (flipping weakens validation, not a value)

`history.py:149` (arrival/departure overlap), `167` (non-decreasing outcome),
`1448` (q in [0,1]); `candidate_arm.py:173, 185` (row-count / shape guards).
A wrong flip lets malformed input through; it does not change a covariate value
on valid input. Worth a raise-test but lower priority than value gaps.

### Equivalent mutants (no covariate-value effect)

- `history.py:304` — `where=counts > 0` only avoids a divide-by-zero warning;
  the `counts == 0` rows are overwritten with `NaN` immediately after, so the
  value is identical either way.
- `history.py:1313`, `1525` — clip bounds (`1 - 1e-4`, `len(WELL_ARMS) - 1`)
  whose mutated range agrees with the original on every value that reaches them.
- `history.py:444, 445, 1421, 1514`; `candidate_arm.py:170` — set differences
  and counts inside error-message f-strings, reached only on the raise path.

Re-run `mutation_check.py` and update the score here whenever the builders or
their tests change; the check is CI-runnable so the floor cannot silently erode.
