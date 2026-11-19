# Builder test floor

The covariate builders in `analysis/behavior/covariates/` compute the values that
feed every replay regression. The schema/timing gates check a covariate's *name*
and *availability window*, not its *value* — so a units error, a wrong window
edge, or a flipped `searchsorted` side would ship silently unless a test pins the
value. This doc records the measured floor and the target.

The measurement tool is `scripts/checks/mutation_check.py`: it mechanically mutates
value-critical operations (comparison operators, `searchsorted side=` and
`merge how=` string literals, `+`/`-`) and reports which the covariate test
subset catches ("killed") versus not ("survived"). Survivors are coverage gaps.

## Current score

Measured 2026-09-06 over the tool's default targets — `covariates/candidate_arm.py`
and the six modules of the `covariates/history/` package — against its default
six-file test set (`test_covariates.py`, `test_uncertainty_covariates.py`,
`test_covariate_timing.py`, `test_builder_value_floor.py`,
`test_recent_trial_reward_rate.py`, `test_recent_reward_rate.py`).

**Killed 89 / 124 (72 %). 35 survivors.** 124 mutation runs, ~42 min wall.

| module | mutations | killed | survivors |
|---|---|---|---|
| `covariates/candidate_arm.py` | 5 | 0 | 5 |
| `covariates/history/_common.py` | 12 | 8 | 4 |
| `covariates/history/reward_history.py` | 28 | 27 | 1 |
| `covariates/history/visit_recency.py` | 27 | 19 | 8 |
| `covariates/history/value_uncertainty.py` | 23 | 23 | 0 |
| `covariates/history/value_block.py` | 23 | 10 | 13 |
| `covariates/history/diagnostics.py` | 6 | 2 | 4 |

**Read the 72 % against the previous 78 % carefully — it is not an erosion.**
The earlier figure was measured over 119 mutation sites in
`candidate_arm.py` plus a single `covariates/history.py` module; that module is
now the six-module `history/` package, the site count is 124, and the default
test set is unchanged. Two things account for
the difference, and neither is a builder regression:

1. **Six of the 35 survivors are measurement scope, not coverage.**
   `tests/analysis/behavior/covariates/history/test_value_block.py` exists and exercises the qvals-alignment
   validator, but it is not in the tool's `_DEFAULT_TESTS`. Re-running those six
   mutants against it directly kills all six (`value_block.py:298`, `299`,
   `343`, `346`, `359`, `371`). Including that file would put the floor at
   **95 / 124 (77 %)**. Adding it to `_DEFAULT_TESTS` is the cheapest real
   improvement available here.
2. **The `searchsorted`/join family grew and is still almost fully killed.**
   There are now nine `side=`/`how=` sites (five previously). Eight are killed.
   The ninth, `value_block.py:359`, is a `merge(how="left")` inside the
   `assert_qvals_aligned` validator — killed by `test_value_block.py`, per point 1.

The remaining true survivors are triaged below. The raw rate is a weaker target
than a clean triage: what matters is that no *covariate value* can be wrong
without a test failing. Padding the rate with contrived fixtures is avoided.

## Survivor triage

### Value gaps that need a full event fixture

These live inside `attach_candidate_arm_covariates` and the patsy spline, which
still require on-disk contingency/event fixtures. Making the builders I/O-pure
(passing those frames in) turns each into a plain in-memory value assertion;
that is where the tests belong, not padded in here.

- `candidate_arm.py:203` — `is_last_rewarded_noncurrent_arm = is_last_rewarded *
  (1 - is_current_arm)`; the `-` masks out the current arm.
- `candidate_arm.py:260` — `omitted = 1 - rewarded`; the omission indicator.
- `visit_recency.py:126-129` and `:149` — `spline_visit_recency_within_event`:
  the eligibility mask (`arm < len(WELL_ARMS)`, `never_visited == 0`,
  `is_current == 0`, `is_prev == 0`) and the within-event mean-centering
  (`out[:, column] - means`).

### Window-arithmetic survivors not yet distinguished by the fixtures

- `reward_history.py:177` — the window count `count = n_vis_before - start` in
  `recent_reward_rate_all_events`. The existing fixtures use a window whose
  start is 0, where `+` and `-` agree; killing it needs a fixture whose window
  start is past a rewarded experience (e.g. `[1,1,0,1,1]`, `k=2`). Real value
  gap, closeable with one more fixture.

### Diagnostic-only objective (not a cataloged covariate)

- `diagnostics.py:77-78` — the clip bound and the Bernoulli NLL
  (`reward*log(p) + (1-reward)*log(1-p)`) of `nll`, the per-animal fitted-alpha
  leaky-Q learner. That module produces no cataloged covariate and is used only
  for offline model checking, so a wrong value here cannot reach a fitted model.
  Worth a test, lowest priority.

### Input-integrity guards (flipping weakens validation, not a value)

A flip lets malformed input through; it does not change a covariate value on
valid input. Worth a raise-test, lower priority than value gaps.

- `_common.py:83, 85 (×2), 103` — arrival strictly increasing, arrival/departure
  overlap, non-decreasing outcome time.
- `candidate_arm.py:173, 185` — row-count and shape guards.
- `value_block.py:198` (the `q > 1.0` half of the belief-mean range guard) and
  `value_block.py:262` (both halves of the belief-variance `var ∈ [0, 0.25]`
  guard) — only the *boundary* is mutated, so a mutant admits an
  exactly-boundary value. The `q < 0.0` half at `:198` is killed.
- `value_block.py:301 (×2)` — the session-span boundary
  (`time >= low`, `time <= high`) in `_qvals_rows_expected_in`; the mutant
  differs only for a stop exactly `SESSION_SPAN_PAD_SEC` from the span edge.
  Survives even `test_value_block.py`.
- `value_block.py:372` — the drift threshold `drift > tolerance` in
  `assert_qvals_aligned`; the mutant differs only at exactly the tolerance.
  Survives even `test_value_block.py`.

### Equivalent mutants (no covariate-value effect)

- `visit_recency.py:94` — `where=counts > 0` only avoids a divide-by-zero
  warning; the `counts == 0` rows are overwritten with `NaN` immediately after,
  so the value is identical either way.
- `candidate_arm.py:170`, `visit_recency.py:234, 235`, `value_block.py:87` — set
  differences and counts inside error-message f-strings, reached only on the
  raise path.

## Re-running

```bash
uv run python scripts/checks/mutation_check.py           # ~42 min, 124 mutations
uv run python scripts/checks/mutation_check.py \
    --targets <files> --tests <test files>           # a narrower slice
```

It is long enough to want a `tmux` session and a log under `logs/` (gitignored).
Each mutation restores the source in a `finally`; a run killed mid-mutation
leaves one modified file, which `git diff` will show.

Re-run and update the score here whenever the builders or their tests change, so
the floor cannot silently erode.
