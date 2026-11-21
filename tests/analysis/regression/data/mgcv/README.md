# mgcv reference fits

mgcv's REML fits of the four synthetic data sets of
`scripts/checks/compare_regression_to_mgcv.py` (`poisson_one_smooth`,
`poisson_two_smooths`, `bernoulli_one_smooth`, `gaussian_one_smooth`), one JSON file
each: the smoothing parameters `sp`, the trace of each smooth's penalty matrix as mgcv
scaled it (`trace_S`), the edf, the REML score (`gcv.ubre`), the scale, the 0/1
coefficient `rew` and its standard error (Poisson sets), and each centered smooth and
its standard error on the grid 0.05, 0.10, ..., 3.95 (`predict(type = "terms")`).
`n_rows` and `y_sum` let the test check that it rebuilt the same data.

`tests/analysis/regression/test_reml.py::test_matches_mgcv_fixtures` rebuilds each data
set from its seed, fits it with the shared fitter and compares; it does not need R.

Made with R 4.5.3 and mgcv 1.9-4 (the `csr-mgcv` conda environment), from the
repository root:

```bash
uv run --frozen python scripts/checks/compare_regression_to_mgcv.py
```

which runs `scripts/checks/compare_regression_to_mgcv.R` with
`/home/edeno/miniconda3/envs/csr-mgcv/bin/Rscript` (`--rscript` to change it), writes
these files and prints the agreement. The R script tightens mgcv's convergence control
(`gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10))`): with the defaults
mgcv's outer search stops early, and on these data sets its log penalty and edf move by
up to 4e-5 and its curve by up to 4e-6 (Bernoulli set).
