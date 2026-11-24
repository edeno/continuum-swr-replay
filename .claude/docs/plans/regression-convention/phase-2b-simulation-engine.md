# Phase 2b — The simulation package: one engine, real regression code, scenario lists per model

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [JAX fitter design](jax-fitter-design.md)

Branch: `regression-simulation` from master after Phase 2 and the branch `regression-declared-penalty`.
CPU only. No canonical table is read by the engine; the first scenario list reads the real rate bins.

**Status:** not started. Decided by the lead (design, twentieth round, 2026-11-23): "yeah, maybe we
should have a simulation package so that we have these for the rate, content, etc models, all with the
same principle, we should always be using the real regression code", and "yes" to the shape below. The
order (this phase before the JAX fitter, [Phase 2c](phase-2c-jax-fitter.md)) is the author's call; the
lead did not object (design, twenty-first round). It is the port design's Phase 1
([jax-fitter-design.md](jax-fitter-design.md), section 12).

**Inputs to read first:**

- The design, section 6 (check 5: known truth), and section 9, the seventeenth to twenty-third rounds.
- [jax-fitter-design.md](jax-fitter-design.md), sections 9 and 12 (Phase 1): why the draws must not
  depend on the fitter.
- `src/continuum_swr_replay/analysis/regression/`: `likelihoods.py` (the four families), `sessions.py`
  (`fit_sessions`), `calibration.py` (`log_penalty_for_edf`), `contrasts.py`, `record.py`;
  `analysis/replay_models/declared_tests.py`; `analysis/shared_statistics.py` (`estimand_summary`,
  `animal_day_session_values`).
- `scripts/checks/simulate_declared_penalty.py` and `tests/scripts/checks/test_simulate_declared_penalty.py`
  (branch `regression-declared-penalty`, 86ad3fc5): the first stored check written on the real code; its
  cohort shape and its output columns are the pattern.
- `scripts/checks/simulate_interval_coverage.py`: the real cohort shape (animals, days, sessions).
- The project memory note "simulation uses real regression code" (`feedback_simulation_uses_real_regression_code.md`).
- Scratch: `S/proto_rate_realistic/` (the realistic rate scenarios; running on 2026-11-23),
  `S/proto_rate_smoothing/REPORT.md` and its follow-up section (the reward-term estimators; simulation).

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C8](shared-contracts.md#c8-the-declared-family-of-tests), [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session),
[C12](shared-contracts.md#c12-layering-of-the-new-package), [C13](shared-contracts.md#c13-no-legacy-left-behind).

## The rule

A simulation replaces only the outcome column of a model's real rows. Everything else is the real code:
the design, the support rule, the per-session fit, the refits, the declared tests and the reduction.
Nothing rebuilds a model by hand.

## Tasks

- **Draw functions** (on each likelihood in `analysis/regression/likelihoods.py`): `draw(eta, aux, rng)`
  returns outcomes from the family at a linear predictor. Poisson (with the offset), Bernoulli, Gaussian
  (with a given scale) and the grouped softmax (one multinomial draw, or shares, per group; the content
  response is shares, so the form of a share draw is stated in the docstring). The draws use a NumPy
  `Generator` only, never the fitter's code, so that the same seeds give the same data before and after
  the JAX port ([Phase 2c](phase-2c-jax-fitter.md)). Failing test first:
  `test_likelihoods.py::test_draws_have_the_family_mean` (the mean of many draws equals the family's mean
  at a fixed `eta`).
- **The engine** (`analysis/regression/simulation.py`, new; task-free, C12). It takes a model (a
  `Design` and the names of its response and unit columns), its real rows by session, a scenario and a
  seed. For each data set it draws the outcome, runs `fit_sessions`, computes the declared estimands with
  `contrasts`, reduces them with `estimand_summary`, and returns one results table per scenario and
  estimand: bias, spread (SD over data sets), coverage of the planted value, rejections at 0.05, and the
  sessions without a value (by `record.NoFit` cause and by `left_out` cause).
- **Scenarios** (in `simulation.py`): a scenario is one of two kinds.
  1. Inside the model: planted coefficients on the model's own design.
  2. Outside the model: any expected outcome as a function of the rows (for example a rate curve of a
     shape the model cannot hold exactly, or an effect that differs between sessions and animals).
  Each scenario names its planted value for each estimand, or NaN when the estimand has no true value
  under it.
- **Scenario lists** live beside each model, in `analysis/replay_models/<family>/` (task knowledge). The
  first list is the rate model's, `analysis/replay_models/rate/visit_rate_scenarios.py` (new), with the
  scenarios that `simulate_declared_penalty.py` plants now (a threefold rise after arrival, a flat curve).
  The realistic scenarios of `S/proto_rate_realistic/` become that list in
  [Phase 5a](phase-5a-rate-models.md). Phases 5b, 6a, 7, 9 and 11 add their own lists.
- **One runner** (`scripts/checks/run_simulations.py`, new): `--model <family.model>`, `--scenarios`,
  `--n-data-sets`, `--seed`, `--workers`, `--output-dir`. It writes one parquet of results per model under
  `Processed-Data/replay_content/models_convention/simulations/` (or `--output-dir`), with the seeds, the
  commit and the declared penalties. A process pool uses `spawn`, not `fork`
  ([jax-fitter-design.md](jax-fitter-design.md), section 8).
- **Move the stored check onto the engine**: `scripts/checks/simulate_declared_penalty.py` becomes a
  scenario list and a call of the runner (its output columns kept). Remove the script's own simulation
  and fitting code in this phase (C13). Its test moves with it.
- **Docs**: a section "Simulation" in `docs/design/regression_fitter.md`: the rule, the two kinds of
  scenario, the results table's columns, and how a phase adds a list.

## Tests that gate it

| Test | Asserts |
| --- | --- |
| `tests/analysis/regression/test_likelihoods.py::test_draws_have_the_family_mean` | for each family, the mean of 20,000 draws at a fixed `eta` is the family's mean within 4 standard errors |
| `test_simulation_engine.py::test_same_seed_same_data` | two runs with one seed give the same outcome columns, bit for bit |
| `test_simulation_engine.py::test_only_the_outcome_changes` | the rows given to `fit_sessions` equal the real rows in every column but the outcome |
| `test_simulation_engine.py::test_planted_coefficients_recovered` (tiny cohort) | a planted coefficient inside the model is recovered (bias within 2 simulation SEs) on 5 animals of 2 sessions |
| `test_simulation_engine.py::test_outside_scenario` (tiny cohort) | an expected outcome given as a function of the rows is drawn and fitted; the table names the estimand's planted value NaN when it has none |
| `test_simulation_engine.py::test_sessions_without_a_value_counted` | a scenario with no events in one condition in some sessions reports those sessions by cause |
| `tests/scripts/checks/test_run_simulations.py` | the runner's `--quick` run writes the table with the seeds, the commit and the declared penalties |
| the moved test of `simulate_declared_penalty.py` | the same output columns as before; its fixed-seed numbers are the fixture that [Phase 2c](phase-2c-jax-fitter.md) compares with |

The engine's outputs on fixed seeds are stored as a fixture
(`tests/analysis/regression/data/numpy_reference/simulation_*.json`), so that the JAX port is checked on
the whole pipeline of bias and coverage (port design, section 9).

## Deliberately not in this phase

- Any scenario list but the rate model's first one. Each family phase writes its own.
- The realistic rate scenarios themselves (Phase 5a turns them into the list).
- Any change to the fitter. The engine runs on the present NumPy fitter.
- A decision on the open question `OPEN (lead): rate model form` (the penalty width is no longer open: the rule decided by the lead, 2026-11-23, 19:34, "lets just pick a penalty that's reasonable"; the value declared per model by the author from its simulation, with a stiffer and a looser value: rate 2 (1, 4), content 8 (4, 16), the other families not picked yet). The
  engine reports what a scenario gives; it does not choose a rule. (The reward term's rule is decided: a
  weak penalty, and a support rule on the design; design, twenty-second round. The engine runs it, and
  each family's scenario list checks the penalty before the family's fit is used.)

## Verification and runtime

- Tiny cohorts in the tests. The first real run: the moved declared-penalty check on its full cohort,
  timed (wall time and peak memory, `/usr/bin/time -v`), in `tmux` (`csr-2b-simulation`) if over five
  minutes (C11). Compare its table with the one the script wrote before the move (same seeds): equal.

## Rebuilt

`models_convention/simulations/declared_penalty.parquet` (the moved check). Nothing in `paper/reported/`.

## Exit criteria

The tests above pass; the declared-penalty check runs through the engine and gives the same table as
before the move; the old script's own simulation code is removed; the fixed-seed fixture is committed for
Phase 2c.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
