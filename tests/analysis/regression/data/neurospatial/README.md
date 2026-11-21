# neurospatial reference fit

`one_term_poisson.json`: neurospatial's REML penalty selection
(`neurospatial/encoding/_glm_numpy.py`, commit `d8a5643a2`) on one synthetic one-term
Poisson model, three units sharing one penalty: a cubic P-spline on 12 basis functions
over [0, 1] with a second-order difference penalty, written in the penalty's eigenbasis
(neurospatial's fitter takes a diagonal penalty). The data are rebuilt by
`make_data` in `scripts/checks/compare_regression_to_neurospatial.py` (NumPy and SciPy
only, seed 11), so the suite does not need neurospatial.

Made with neurospatial's own interpreter, from the repository root:

```bash
/cumulus/edeno/neurospatial/.venv/bin/python scripts/checks/compare_regression_to_neurospatial.py
```

## What is compared

neurospatial minimizes, summed over units,
`-sum(n eta - o exp(eta)) + lam/2 sum(d gamma^2) - rank/2 log(lam) + 1/2 log|H + lam diag(d) + 1e-10 I|`
over `log lam` in [-8, 20] with SciPy's bounded scalar minimizer (`xatol = 1e-3`). It
is the same Laplace REML score as the shared fitter's, less terms that do not depend on
`lam` (the offset term `sum n log o`, `-1/2 log|S|_+` at unit strength, `-M_p/2 log(2 pi)`
per unit), and with a 1e-10 ridge on the Hessian. Its inner fit stops on the relative
decrease of the objective (`tol = 1e-13` here) rather than the Newton decrement.

`tests/analysis/regression/test_reml.py::test_one_penalty_matches_neurospatial_fixture`
checks, with the measured agreement in brackets:

- the two scores differ by exactly those constants on a grid of `log lam` from -2 to 10
  (to 1e-7 absolute; measured 6e-9, on scores near -240);
- the shared fitter's selected `log lam` equals the minimum of neurospatial's objective
  found with `xatol = 1e-10` (to 1e-6; measured 9e-8);
- it equals `select_penalty_by_reml`'s own choice to within that function's `xatol` of
  1e-3 (measured 1.3e-4: neurospatial's default stopping rule, not a difference of
  method, is what keeps that comparison from 1e-4);
- the log rate of each unit in each bin at the selected penalty (to 1e-6; measured 3e-8).
