# NumPy reference outputs of the shared fitter

The outputs of the shared regression fitter (`src/continuum_swr_replay/analysis/regression/`)
on the cases of `scripts/checks/_regression_reference.py`, one JSON file per group
(`newton`, `reml`, `refit`, `select`, `sessions`, `calibration`; the module docstring lists
what each covers). Each case builds its inputs from a fixed seed, so only the outputs are
stored. Each file names the commit that made it, the command and the versions of Python,
NumPy and SciPy; each case is one line. A float that is not finite is stored as the string
`"NaN"`, `"Infinity"` or `"-Infinity"` (strict JSON).

`tests/analysis/regression/test_reference_fixtures.py` runs every case again and compares
it with these files: a float `a` matches its reference `b` when `|a - b| <= 1e-11 + 1e-9 |b|`;
statuses, reasons, step counts, the columns left out and every count must be equal
(`_regression_reference.differences`). They are the reference a fitter with another solver
is checked against.

Made from the repository root, with a checkout whose tracked files equal its commit and one
BLAS thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --frozen python scripts/checks/write_regression_reference.py
```

## What is not stored, and why

On the machine that made them, every case also matched with 8 BLAS threads and with the
AVX2 kernels a CI runner uses (`OPENBLAS_CORETYPE=Haswell` and the AVX-512 features off,
`tests/README.md`), at 1 and 6 threads. Two kinds of output did not, and are not stored:

- the coefficients, `A` and step of a fit with no finite maximum: along the direction of
  unbounded increase they are set by the rounding (a coefficient of 22 moved by 3e-6, a step
  of 1 by 2e-5). Its status, steps, log-likelihood and the columns running away are stored.
- a covariance where `A` is too badly conditioned: a P-spline of 40 columns at log penalty
  -8 (condition number of `A` 9.5e8) gave covariances 1.4e-7 apart (relative) between the
  two kernels, beyond the rule. The case stored is the same spline at -4 (condition number
  1.8e7); the Newton case of 40 columns at -6 has a condition number of 1.1e7.
