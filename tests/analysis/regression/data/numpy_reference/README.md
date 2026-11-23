# NumPy fitter reference outputs

Outputs of the present NumPy fitter on fixed seeds, kept so that a change of the
solver (the JAX port) is checked against them. Each file records the commit and the
command that made it.

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

The comparison is to `1e-11 + 1e-9 |b|`, with counts and codes exactly equal.
