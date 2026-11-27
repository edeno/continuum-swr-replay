# NumPy reference outputs of the shared fitter

The outputs of the shared regression fitter (`src/continuum_swr_replay/analysis/regression/`)
on the cases of `scripts/checks/_regression_reference.py`, one JSON file per group
(`newton`, `reml`, `refit`, `select`, `sessions`, `calibration`; the module docstring lists
what each covers). Each case builds its inputs from a fixed seed, so only the outputs are
stored. Each file names the commit that made it, the command, the versions of Python, NumPy
and SciPy, the BLAS library each was built with and its version, the machine, and the BLAS
thread and kernel variables set when it was written; each case is one line. A float that is
not finite is stored as the string `"NaN"`, `"Infinity"` or `"-Infinity"` (strict JSON).

`tests/analysis/regression/test_reference_fixtures.py` runs every case again and compares
it with these files: a float `a` matches its reference `b` when `|a - b| <= 1e-11 + 1e-9 |b|`;
an int, a bool, a string or None must be equal and of the same type, and a float reference
needs a float (`3` does not match `3.0`, nor `3.0` `3`); statuses, reasons, step counts, the
columns left out and every count must be equal (`_regression_reference.differences`). They
are the reference a fitter with another solver is checked against.

Made from the repository root, with a checkout whose tracked files equal its commit and one
BLAS thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --frozen python scripts/checks/write_regression_reference.py
```

## Changed since they were written

- `refit.along_a_penalty` (a grouped softmax): the order of the columns running away in
  its two fits, columns 3 and 8 swapped (`[7, 6, 5, 0, 4, 3, 8, 1, 2, 10, 9]`), when the
  softmax's columns running away came to be measured on the design less each event's
  means (`Likelihood.identifying_design`; their parts there are 24.3 and 22.2, on `X`
  23.2 and 26.1). Only that list was edited, from a trial build of the `refit` group
  (`--out-dir`) whose every other output matched the stored file; the header still names
  the commit that wrote the rest.

## Where they were verified

Only on x86-64 Linux (an Intel Xeon Gold 6342, with AVX-512) with OpenBLAS: NumPy 2.3.4 built
with scipy-openblas 0.3.30 and SciPy 1.16.3 with scipy-openblas 0.3.29.dev, Python 3.11.13.
Every case matched under four settings of the kernels:

- OpenBLAS's own choice for the CPU (SkylakeX), at 1 and at 8 threads;
- the AVX2 kernels a CI runner uses, `OPENBLAS_CORETYPE=Haswell` with NumPy's AVX-512
  features off (`NPY_DISABLE_CPU_FEATURES`, `tests/README.md`), at 1 and at 6 threads.

They are untested on macOS with Apple's BLAS (Accelerate), and the tests are not skipped
there. A failure on such a machine may come from the BLAS, not from the fitter: read the
ratio the message prints before reading it as a fault.

## What is not stored, and why

Two kinds of output did not reproduce across those kernels, and are not stored:

- the coefficients, `A` and step of a fit with no finite maximum: they are a point on a
  diverging path, set by the rounding (a coefficient of 22 moved by 3e-6, a step of 1 by
  2e-5). Its status, steps, log-likelihood and the columns running away are stored.
  Outside the stored cases only its status and its columns running away are stable:
  between c169e039 and d42ad2a0 (51 such random fits) the status, steps and columns were
  the same while a coefficient moved by up to 7e-5 of itself, the log-likelihood by up to
  1e-4 and `step` by up to 5e-4.
- a covariance where `A` is too badly conditioned. A P-spline of 40 columns gave
  covariances beyond the rule between the two kernels at log penalty -8 for the Poisson
  (condition number of `A` 9.5e8, 1.4e-7 apart relative) and the Gaussian (8e6, 2.7 times
  the allowed difference), and at -6 for the grouped softmax (6.1e6, 2.7 times). The REML
  cases stored are the same splines at -4 (Poisson 1.7e7, Bernoulli 3.3e6, grouped softmax
  8.6e5) and at -6 (Gaussian 1.1e6). The Newton cases store `A`, not its inverse, and
  reproduce at weaker penalties: 40 columns at -6 for the Poisson (1.1e7), the Bernoulli
  (1.3e7) and the grouped softmax (1.8e7), and at -8 for the Gaussian (5.6e6).

## Paths no stored output checks

- The last step of a Newton fit rejected because `F` rises by more than its rounding
  allowance (`newton._last_step`): a real likelihood reaches it only through the rounding
  of `F`. The separated session of `select.bernoulli_one_fails_at_the_weakest_grid_point`
  takes it in fits of the grid and the search, which are not stored: accepting every last
  step leaves every stored output unchanged. Stored, a rejection would flip with the
  rounding of `F`.
- The last step rejected because it lands where `A` is not positive definite or the
  decrement is not below the tolerance, and the status "step halving failed". With a step
  that moves the linear predictor by at most `NO_MAXIMUM_ETA_STEP` on every row, the third
  derivative of the Poisson, Bernoulli and grouped softmax likelihoods is too small beside
  the second for the quadratic model to fail that way; only rounding could. No simulated
  fit tried reached them. `tests/analysis/regression/test_newton.py` reaches them with a
  likelihood made for it or a step of its choosing.
- The search's step along negative curvature (`_Curvature.downhill_probes`): it needs the
  search to stop within about 1e-4 of a maximum or a saddle of the score, which Newton on
  the absolute eigenvalues moves away from; no simulated set of sessions tried reached it.
  `tests/analysis/regression/test_reml.py` reaches it with a made-up score.

## Cases whose stopping decision is close to its tolerance

A port that rounds differently can flip these decisions and change a stored step count or
status; a failure in one of these cases alone, at that decision, is read first as rounding.
They were found by moving each stopping tolerance of the fitter (`NEWTON_DECREMENT_TOL`,
`NO_MAXIMUM_ETA_STEP`, `IDENTIFIABILITY_TOL`, `REML_DECREMENT_TOL`, `REML_EIGEN_FLOOR`,
`NAMED_SHARE`, the last step's rounding allowance) by 20 % either way and listing the cases
whose stored outputs changed, and by measuring the margin of each decision. The cap rule
(`_runs_away_at_the_cap`) has no tolerance: its margin is the smallest relative change of
the full step over the last capped steps (the rule asks that it not shrink).

| Case | Decision | Margin |
|---|---|---|
| `refit.along_the_intercept` | the decrement one step before the stop with no finite maximum | 1.14 times the tolerance |
| `select.poisson_one_without_a_fit` | the same, in its session 0 (the same session) | 1.14 times the tolerance |
| `newton.no_finite_maximum_quasi` | the decrement one step before the stop with no finite maximum | 1.20 times the tolerance |
| `refit.along_a_penalty` | the cap rule: the full step still grows (no finite maximum) | +0.47 % |
| `newton.far_maximum_step_limit` | the cap rule: the full step shrinks (step limit, toward a far maximum) | -1.0 % |
| `select.bernoulli_one_fails_at_the_weakest_grid_point` | the cap rule at log penalty -10 in the separated session: the full step shrinks, so that grid point fails as "step limit reached" | -5.8 % |

Along a direction with no finite maximum the decrement falls by `e` per step, so such a
stop always lies between a decrement within a factor `e` above the tolerance and one below
it. In the empty reference level of `_rate_levels` (along the intercept) the rows and the
exposure fix that decrement whatever the seed, so those two cases keep it.

Some cases were moved off a close decision to a nearby seed or size on the same path (the
comments in `_regression_reference.py` give the old margins): `select.poisson_two_smooths`
(a decrement 1.0009 times the tolerance), `select.bernoulli_one_smooth` (0.95),
`newton.far_maximum_*` (1.08), `newton.no_finite_maximum_logistic_cell` (1.18) and
`refit.columns_left_out_in_rounds` (a column's part of the step 1.06 times
`NO_MAXIMUM_ETA_STEP`). A search with a term stopping on the score's plateau was not kept:
its log penalty there is set by the tolerance (a last decrement 0.93 times it), so
`select.gaussian_two_smooths_six_steps` gives its second smooth an effect instead.

## Differences from the NumPy fitter outside the stored cases

Every stored case matches the JAX fitter by the rule above. Outside them the two fitters
can differ by one Newton step, which then moves the outputs by about the convergence
tolerance rather than by the rule.

- **What differs.** In 216 fits drawn at random by a first independent review (2026-11-23:
  200 Newton fits, 50 of each family, and 16 REML scores at fixed log penalties), 2 took
  one step more than the NumPy fitter at 7959aca7: a Bernoulli fit of 2,500 rows (9 steps
  to 10) and the inner fit of a Poisson REML score of 4,096 rows (4 to 5). In 970 random
  fits of a second review (2026-11-23; the four families, 40 to 20,000 rows, 2 to 13
  columns), 4 of 650 in one sample and 0 of 320 in another took one more step, all of them
  Poisson (about 2.5 % of that sample's Poisson fits, none of the other three families).
  No status differed. Where a step count differs the outputs differ by about the tolerance
  of the stop, 1e-8 (the REML score, log-likelihood, gradient and Hessian of the Poisson
  case by about 1.6e-8 absolute), not by `1e-11 + 1e-9 |b|`. The rate depends on the data,
  the family and the platform, since it is set by rounding.
- **Why.** At the stop the fit takes its last full step when `F` does not rise by more
  than `8 eps max(1, |F|)` (`newton._last_step`). The NumPy fitter summed the
  log-likelihood as two dot products (`y' log mu - 1' mu`), whose rounding moved `F` by
  more than that allowance between two points a decrement of about 1e-15 apart: in the
  Bernoulli case `F` rose 16.8 eps |F| with the step, and the step was refused. The JAX
  fitter sums each row's terms once; its `F` is within 1.5 eps |F| of an exact sum
  (`math.fsum`) on fits of 500 to 20,000 rows, and it takes the step. Under the exact
  sum the step moves `F` by -0.99 eps |F| in the Bernoulli case and lowers it by
  4.1 eps |F| in the Poisson case: the JAX fit is the better converged one.
- **Where it is checked.** `tests/analysis/regression/test_newton.py`
  (`test_a_last_step_the_numpy_sum_refused_is_decided_by_the_exact_sum`,
  `test_a_reml_fits_last_step_is_decided_by_the_exact_sum`) rebuilds both cases from their
  seeds and checks that the decision, whichever the platform's rounding makes it, agrees
  with the exact sum: a step taken does not raise the exact `F` by more than the
  allowance, a step refused does not lower it by more. The step counts (10 and 5 on the
  CPU where they were written) are not asserted. No stored file was changed.

## The simulation engine's fixed-seed tables

Two more files in this folder are written by the simulation engine and compared by its
own tests, not by `write_regression_reference.py` and `test_reference_fixtures.py`. They
are compared by the same rule (`1e-11 + 1e-9 |b|`, counts and codes exactly equal).

- `simulation_declared_penalty.json`: the table of the declared-penalty check's quick
  run (`scripts/checks/simulate_declared_penalty.py`: 5 animals of 1 day of 2
  sessions, 2 data sets per curve, every target), through the simulation engine.
  Made, from the repository root, with
  `python scripts/checks/run_simulations.py --model declared_penalty --quick --output-dir OUT --reference tests/analysis/regression/data/numpy_reference/simulation_declared_penalty.json`.
  Compared in `tests/scripts/checks/test_simulate_declared_penalty.py::test_matches_the_fixed_seed_fixture`.
- `simulation_engine.json`: the summary tables of the simulation engine's own
  synthetic models (a Poisson rate model under a scenario inside and one outside the
  model; a grouped-softmax content model inside), 4 data sets each. Made with
  `python -m tests.analysis.regression.test_simulation_engine`. Compared in
  `tests/analysis/regression/test_simulation_engine.py::test_matches_the_fixed_seed_fixture`.
- `simulation_engine_tables.json`: the `summary`, `data_sets` and `sessions` tables of
  fixed small runs of the engine with one model or several models of one design (the
  rate model inside and outside; the content model inside; the day cohort's model at the
  session, the day and the animal unit with a support hook, inside and outside), 2 data
  sets each, written before the difference of two models was added to the engine. They
  are compared exactly, not by the rule above: every float to the bit (by its repr), a
  float that is not finite by its name, an int not equal to a float. That holds only on
  the machine that wrote them (the CPU and BLAS above), so the test is skipped on
  GitHub's runners. Made with
  `python -m tests.analysis.regression.test_simulation_engine tables`. Compared in
  `tests/analysis/regression/test_simulation_engine.py::test_tables_unchanged_bit_for_bit`.
