# Phase 1 — The shared fitter and its correctness checks

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#the-shared-fitter-built)

**Status:** done. Merged to master in 6e54eda8 (branch `regression-fitter`), cleaned up in 98fe5c06.

**As built (2026-11-21).** This file is the record of the work as planned; it is not rewritten. The code
(`src/continuum_swr_replay/analysis/regression/`, its docstrings) and `docs/design/regression_fitter.md`
are the truth. Where they differ from the text below:

- `penalized_newton(lik, X, y, aux, S, beta0, *, free_basis=None, max_iter, max_step)` has no `tol` and no
  `max_halvings` argument (the rules are in `constants.py`). It returns a `NewtonFit` with a `status`, one
  of six (`newton.FitStatus`); `converged` is a property, true only for "converged". A fit with no finite
  maximum also carries its `step` and `running_away` columns. The design's identification is checked
  before the first step, and the fit takes a last full step at the stop.
- `SessionData(X, y, aux, name, active)` holds no start (`beta0`); `active` marks the columns a session
  estimates.
- The fixed-penalty path is `fit_at_penalties(lik, sessions, blocks, rho)`; it raises at the first session
  whose fit does not converge. `refine_penalties` searches from a given start without the grid.
- `select_penalties` sets aside the sessions with no finite maximum at the grid's center before the search
  and refits them after it with their running-away unpenalized columns inactive. A `PooledFit`'s
  `sessions` are `RemlScore` records (there is no `SessionFit`); the sessions that chose the penalties are
  `search.in_choice`, and `in_null_space(session, blocks)` flags a term shrunk to a line.
- The search stops on its Newton decrement in log-likelihood units (`REML_DECREMENT_TOL`), not on
  `max |g| < 1e-6 max(1, |V|)`; fits and scores are computed in each penalty's eigenbasis. The Gaussian
  scale is profiled, not an outer parameter.
- `PSplineBasis.constrained(x)` returns a `ConstrainedSmooth`: the constrained design, its penalty and the
  constraint.
- `constants.py` holds only the fitter's own rules. The convention's constants that the fitter does not
  read (the collinearity threshold, the support minimum, the longest stay piece, the number of folds) are
  not there; each is defined in the Phase 2 module that reads it.
- Check 7 runs four synthetic data sets (a Gaussian smooth added) and its fixtures are JSON; the R script
  does not print concurvity (Phase 2 adds it). The coverage simulation fits 1,000 data sets per truth, not
  200.
- The cleanup fixed two defects: the penalty search could end at negative curvature on its last allowed
  step (df1d0625), and grid failures of two sessions with one status were not summed (52d5b633). A one-step
  Gaussian fit was tried (726adc45) and reverted (538f83e9).
- Follow-ups are in `TODO.md`, section "Regression fitter" (among them, the step cap's dependence on column
  scaling). The designs this file links to are replaced by
  [designs.md#the-shared-fitter-built](designs.md#the-shared-fitter-built), an interface summary.

Branch: `regression-fitter` from master. Reads no canonical table except in check 1 (three real sessions).
Can start before the visit labels merge.

**Inputs to read first:**

- The design, sections 2.4, 5 and 6.
- `src/continuum_swr_replay/analysis/replay_models/fitting.py:65-160` — the current convergence and
  identifiability constants (`NEWTON_DECREMENT_TOL` `:78`, `_MAX_STEP_HALVINGS` `:83`, `_MAX_STEP_LENGTH`
  `:89`); `:344` (`_newton_finish`) and `:677` (`fit_conditional_poisson`): the content fitter check 3
  compares against.
- `src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_model.py:293` (`fit_per_bin_glm`):
  the statsmodels per-bin fitter check 1 compares against.
- `src/continuum_swr_replay/analysis/replay_models/rate/ridge_history_cv.py:130-206` (`_objective`,
  `fit_ridge_poisson`): a NumPy penalized Poisson Newton already in the repository.
- `src/continuum_swr_replay/analysis/pfc/batched_glm.py:129-155` (`_in_float64`), `:546-640` (truncated
  solve, covariance): how the repository keeps float64.
- `/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_numpy.py:260-560` — Newton fit, pooled REML,
  convergence gate ([appendix](appendix.md#outside-references-read-for-method-not-imported)).
- Scratch: `S/proto_basic/pspline.py` (the REML prototype that matched mgcv), `S/proto_basic/fit_mgcv.R`,
  `S/proto_basic/check/two_disagree.py`, `S/proto_basic/check_closed_form.py`, `S/proto_basic/sim.py`
  ([appendix scratch map](appendix.md#scratch-map)).
- `tests/repository/test_import_layering.py:80-115`: the rule tables the new package joins.

**Contracts referenced:**

- [C1](shared-contracts.md#c1-running-code-and-tests), [C2](shared-contracts.md#c2-data-the-database-git-hygiene).
- [C5](shared-contracts.md#c5-the-convention) rule 3 (fit) and its shared constants.
- [C12](shared-contracts.md#c12-layering-of-the-new-package) — the package imports only the foundation,
  `analysis.constants` and `analysis.shared_statistics`.

**Designs referenced:** [penalized Newton](designs.md#penalized-newton),
[likelihoods](designs.md#likelihoods), [REML](designs.md#reml-with-exact-derivatives),
[pooled REML](designs.md#pooled-reml), [penalty selection](designs.md#penalty-selection), and the P-spline
basis of [smooth terms](designs.md#smooth-terms-and-bases) (Phase 2 adds the rest of that section).

## Tasks

- **First commit: delete the older all-bins cross-history model** (lead: "ok delete"; nothing reports it).
  Remove `src/continuum_swr_replay/analysis/replay_models/rate/cross_history.py`,
  `scripts/replay_models/rate/fit_cross_history_rate.py`, `tests/analysis/replay_models/rate/test_cross_history.py`,
  the run at `scripts/workflows/run_refits.py:173`, the entries in
  `tests/analysis/event_definition/test_artifact_exposure.py:503` and `:729`, and the mentions in
  `scripts/README.md:85`, `docs/README.md:186` (the `cross_history_models.pkl` part only),
  `docs/models_table.md:287`, `docs/design/cohort_and_timing_rules.md:504`, and the comments at
  `analysis/replay_models/model_specs.py:1108` and `rate/duration_history_simulation.py:441` (which say
  what they mirror; reword to name the duration-aware model). Keep `MODEL_CROSS_HISTORY_RATE`
  (`model_specs.py:1141`): `rate/duration_aware_history.py:295` uses it. Add a row to
  `docs/history/2026-module-moves.md` ("removed: retired"). Regenerate
  `docs/regression_pipeline_map.md` with `scripts/checks/write_pipeline_map.py`. Leave the stored file
  `models_per_bin_rate/cross_history_models.pkl` on disk. Run the full suite once after this commit.
- **Package skeleton** `src/continuum_swr_replay/analysis/regression/` (new): `__init__.py`,
  `constants.py` (the shared constants of C5), `likelihoods.py`, `newton.py`, `reml.py`,
  `pspline.py` (cubic B-spline on fixed equally spaced knots, second-order difference penalty, the
  sum-to-zero constraint from given rows). Add the layering rule of C12 to
  `tests/repository/test_import_layering.py` and the layer text to `docs/README.md:93-121`.
- **Likelihoods** (`likelihoods.py`): Poisson with offset, Bernoulli, Gaussian with scale, grouped softmax,
  each with `loglik`, `score_eta`, `information`, `dinformation`, `d2information` exactly as
  [designs.md#likelihoods](designs.md#likelihoods). Grouped softmax works on sorted group ids with
  `np.add.reduceat`, no Python loop over events.
- **Penalized Newton** (`newton.py`): [designs.md#penalized-newton](designs.md#penalized-newton), returning
  a frozen dataclass with `beta`, `A` (H + S), `converged`, `newton_decrement`, `n_iter`, `loglik`.
- **REML** (`reml.py`): `reml_and_derivatives` for one session; `pooled_reml` summing over a list of
  sessions; `select_penalties` with the grid start and the Newton refinement of
  [designs.md#penalty-selection](designs.md#penalty-selection); the Gaussian scale profiled as there. A
  fixed-penalty path (`lam` given) skips selection. Assert the penalty blocks are disjoint.
- **Outside reference script** (check 7): `scripts/checks/compare_regression_to_mgcv.py` (new) writes three
  small synthetic data sets (one Poisson smooth with a 0/1 covariate; two Poisson smooths; one Bernoulli
  smooth) to CSV, runs `scripts/checks/compare_regression_to_mgcv.R` (new, adapted from
  `S/proto_basic/fit_mgcv.R`: `bs = "ps"` with the same knots, `method = "REML"`) with the `csr-mgcv`
  Rscript, then fits the same with the shared fitter and prints agreement (edf, log lambda, the curve on a
  grid, the 0/1 coefficient and its standard error, the REML score). Also run on two real sessions' visit
  designs once Phase 3 exists (a flag; skip if the visit design is absent). Commit the mgcv outputs of the
  three synthetic sets as test fixtures under `tests/analysis/regression/data/mgcv/` (CSV, a few kB each).
- **Check 2 fixture**: `scripts/checks/compare_regression_to_neurospatial.py` (new) is run with
  neurospatial's own interpreter (`/cumulus/edeno/neurospatial/.venv/bin/python`, commit `d8a5643a2`), fits
  one synthetic one-term Poisson model with `select_penalty_by_reml` (`_glm_numpy.py:481`), and writes the
  selected lambda and curve to `tests/analysis/regression/data/neurospatial/`. The suite compares to that
  file; it never imports neurospatial.
- **Docs**: a short `docs/design/regression_fitter.md` (new) saying what the package fits, its convergence
  and penalty rules, and how to rerun check 7; link it from `docs/README.md`.

## Deliberately not in this phase

- Collinearity, support rule, fit-row record, folds, bands, day-level reduction (Phase 2).
- The JAX batch twin and the GPU check (Phase 8).
- Any change to a production model or stored fit. The current fitters stay as they are; they are the
  baselines of check 1 and 3.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/regression/test_likelihoods.py::test_derivatives_match_finite_differences` | for each family, `score_eta`, `information`, `dinformation`, `d2information` equal central finite differences of `loglik` to 1e-6 relative on random small designs |
| `test_newton.py::test_hand_case_poisson_two_parameters` | check 4: a 2-parameter Poisson (intercept + one 0/1 covariate, no penalty) equals the closed form (log of the group rate ratio) to 1e-10 |
| `test_newton.py::test_hand_case_ridge_one_parameter` | a 1-parameter Poisson with a quadratic penalty equals the root of its scalar stationarity equation (solved by `scipy.optimize.brentq`) to 1e-10 |
| `test_newton.py::test_softmax_equals_conditional_poisson` | check 3: on a synthetic fractional-response data set, the grouped softmax coefficients equal `fitting.fit_conditional_poisson` (`fitting.py:677`) to 1e-6 relative |
| `test_newton.py::test_decrement_rule_and_halving` | the fit stops with decrement below 1e-8; a start with saturated probabilities still converges (the step cap) |
| `test_reml.py::test_gradient_and_hessian_finite_differences` | for one, two and three disjoint penalties and each family, `g` and `V_jk` equal central differences of `V` to 1e-5 relative |
| `test_reml.py::test_grid_start_finds_better_optimum` | a data set built as in `S/proto_basic/check/two_disagree.py`: a Newton search from `rho = 0` ends at the worse optimum, `select_penalties` ends at the better one (lower `V`) |
| `test_reml.py::test_pooled_invariant_to_duplication` | duplicating every session leaves the selected penalties unchanged (as `_glm_numpy.py` states for units) |
| `test_reml.py::test_matches_mgcv_fixtures` | check 2 and 7 in the suite: on the committed mgcv fixtures, edf to 1e-6, log lambda to 1e-4, curve to 1e-4, the 0/1 coefficient and its standard error to 1e-4 relative |
| `test_reml.py::test_gaussian_scale_matches_closed_form` | the Gaussian with one smooth: the profiled scale equals the closed form at the selected penalty, and the fit equals penalized least squares |
| `test_reml.py::test_one_penalty_matches_neurospatial_fixture` | check 2: on a one-term Poisson model, the selected lambda and curve match a committed fixture produced by neurospatial's `select_penalty_by_reml` with the same basis and penalty (see the reference-script task; neurospatial is not installed in this environment and is not imported) to 1e-4 |
| `test_simulation.py::test_coverage_and_size` (slow) | check 5: over 200 simulated data sets (the `rise` and `flat` truths of `S/proto_basic/sim.py`), the 95 % pointwise interval of a 0/1 coefficient covers 93-97 %, a null 0/1 term is significant in 3-7 %, and a flat truth shrinks to a line (edf below 2.5 in at least 90 %) |
| `tests/analysis/regression/test_no_penalty_equals_current.py` | check 1 in the suite: on a small synthetic per-bin design and a small synthetic content design, the shared fitter with no penalty gives the coefficients of `fit_per_bin_glm` (`M7r_time_histcos_rew`, `model_specs.py:956`) and of `fit_conditional_poisson` (M0a) to 1e-6 relative. When a later phase removes one of those fitters it first commits that fitter's coefficients on the same synthetic design as a fixture (C13) and the test reads the fixture; no test keeps a removed fitter alive |
| `scripts/checks/compare_regression_to_current_fitters.py` (new; real data, run once) | check 1 on three real sessions (one each of chimi, j16, wilbur): the same comparison; its printed agreement goes in the PR. Removed in the phase that removes the second of the two old fitters (Phase 7) |
| `tests/repository/test_import_layering.py` | the new rule holds |

## Fixtures

Synthetic data sets built in the tests with fixed seeds (no data files) except the mgcv outputs under
`tests/analysis/regression/data/mgcv/`. Check 1's real-data run is a script, not a test.

## Verification and runtime

- Baseline before: none needed for new code; for check 1, save the current fitters' coefficients on the three
  sessions to the phase's scratch directory first, then compare.
- Measure: one pooled REML selection for a two-smooth Poisson model on 242 synthetic sessions of the real
  size (about 40,000 bins each); record wall time and memory in the PR. The design's 0.08 s per fit implies
  about 20 s per pooled evaluation and minutes per selection on one core.

## Rebuilt

Nothing in Processed-Data or `paper/reported/` (the deleted model reports nothing). Run the full suite.

## Exit criteria

All rows of the validation slice pass, including the slow ones; the mgcv script runs and prints agreement
within the metric tolerances of [overview.md#metrics](overview.md#metrics); the full suite, ruff and the
layering test pass.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
