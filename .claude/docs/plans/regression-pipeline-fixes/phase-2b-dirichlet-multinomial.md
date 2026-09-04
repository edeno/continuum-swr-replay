# Phase 2b — Dirichlet-multinomial content fitter and the three-estimator comparison

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#dirichlet-multinomial-content-fitter)

Findings addressed: A1 (alternative estimator), B1 (concentration as the "SWR broadens" test), A8 (estimand comparison).

**Inputs to read first:**

- [designs.md#dirichlet-multinomial-content-fitter](designs.md#dirichlet-multinomial-content-fitter) — the model, likelihood, and fitter code.
- `src/continuum_swr_replay/analysis/replay/fitting.py:83-243` — result-dict conventions to mirror.
- `src/continuum_swr_replay/analysis/replay/model_checking.py:21-170` — `cross_validate_conditional_poisson` (GroupKFold over events) to mirror for DM.
- `src/continuum_swr_replay/analysis/replay/simulation.py:106-145, 307-440` — `_softmax`, `sample_arm_design`, `simulate_with_event_covariates`.
- `scripts/content/fit_swr_contrast_content.py:281-343, 366-452` — `_fit_session`/`_run`, which gain an estimator switch.
- `scripts/content/fit_replay_content_models.py:552-662` — `fit_model_per_session` (`grain` branch), where the DM branch is added.
- `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:60-90` — `BINARY_TERMS`, `std_factor` (the new `log_phi_*` terms are binary-scale).
- `tests/unit/test_replay_content_model_selection.py` — the existing recovery-test pattern.

**Contracts referenced:**

- [Content estimator result schema](shared-contracts.md#content-estimator-result-schema) — DM output must match.
- [Fraction response](shared-contracts.md#fraction-response) — DM uses the count response; the comparison pairs it with the fraction fit on the same events.
- [Fit provenance sidecar](shared-contracts.md#fit-provenance-sidecar).

**Designs referenced:** [designs.md#dirichlet-multinomial-content-fitter](designs.md#dirichlet-multinomial-content-fitter).

## Tasks

- **Fitter module.** Create `analysis/replay/dirichlet_multinomial.py` with `fit_dirichlet_multinomial`,
  `multinomial_log_constant(y, groups)`, `event_loglik(theta, X, Y, Z)`, and
  `cross_validate_dirichlet_multinomial(long_df, predictors, ...)` (GroupKFold over events, held-out
  per-event log-likelihood; report `cv_ll_per_event_mean` and bits vs the DM null with β = 0).
  Effect-scale separation flag and Wald p-values as in the design. Export from
  `analysis/replay/__init__.py`.
- **Small-session fallback.** In the runner, if a session has fewer than 40 modelled events, fit
  with `fixed_log_phi` set to the animal-pooled intercept obtained from a first pass over that
  animal's sessions with ≥ 40 events; record `phi_source ∈ {"session","animal_pooled"}` in the row.
- **Simulator.** Add `simulate_dirichlet_multinomial_counts` to `simulation.py` (design). Reuse the
  design sampler so `is_swr` and the arm-varying predictors match the real vocabulary.
- **Runner integration.** `fit_model_per_session` (content runner) gains `estimator="dirichlet_multinomial"`
  as a third branch beside timebin/event grain, selected by `--estimator {conditional_poisson,dirichlet_multinomial}`;
  `fit_swr_contrast_content._run` gains the same switch, and for the DM estimator the SWR contrast
  rungs use `concentration_cols=("is_swr",)` so `coef_log_phi_is_swr` is the concentration
  contrast. Output directory suffix `_dm`. Add `log_phi_intercept`, `log_phi_is_swr` to
  `BINARY_TERMS` so `std_factor` treats them per-category.
- **Comparison script.** `scripts/content/compare_content_estimators.py`: on the phase-2a frozen
  at-well cohort, per session fit (i) conditional Poisson on `arm_count_hard` (retired count model),
  (ii) conditional Poisson on `arm_fraction` with clustered SEs, (iii) DM on `arm_count_hard` with
  `is_swr` concentration, for rungs M2, M3, M4, CS1, CS4, CS5 (leaf-only where the spec says so).
  Equalize llf constants before any AIC comparison. Emit:
  - `content_estimator_comparison.parquet`: one row per (session, rung, term, estimator) with coef, se, n_events;
  - `content_estimator_comparison_summary.csv`: per (rung, term) median coef by estimator, median SE ratio vs the count model, `n_animals_pos` by estimator, median per-session ΔAIC vs parent by estimator, and the DM `coef_log_phi_is_swr` animal median with sign count;
  - an animal-level bootstrap (resample animals with replacement, 2000 draws) of the per-animal-median coefficient for each estimator, reported as a 95 % interval;
  - two figures under `paper/figures/content_estimators/`: coefficient agreement scatter (count vs fraction vs DM) per term, and `log_phi` by session with SWR modifier.
- **Result doc.** `docs/design/content_estimator_comparison.md`: what each estimator assumes, the
  simulation recovery table, the real-data summary table, and the recommendation (fractions remain
  shipped; DM concentration reported as the direct "SWR broadens" test). Numbers in this doc are the
  only place they live until phase 4 lifts them into `models_table.md` / the claims table.
- **Docs.** `docs/models_table.md` conventions row for the DM estimator; `docs/covariate_dictionary.md`
  entry for `log_phi_*`.

## Deliberately not in this phase

- Making DM the shipped primary (decided: fractions primary; revisit after the comparison if the
  DM concentration term changes a headline claim).
- A Dirichlet-multinomial rate model or any Family-B change.
- Hierarchical modelling across animals (open question 3); the bootstrap here is descriptive.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_dirichlet_multinomial.py::test_recovers_beta_and_phi` | on `simulate_dirichlet_multinomial_counts` with φ ∈ {1, 3, 10, 100}, β within 3 SE in ≥ 95 % of 40 replicates; `log_phi` within 0.3 |
| `test_dirichlet_multinomial.py::test_large_phi_matches_conditional_poisson` | with φ = 1e4 simulated data, DM β equals `fit_conditional_poisson` β within 0.02 and llf (constants equalized) within 1.0 |
| `test_dirichlet_multinomial.py::test_count_model_se_too_small_on_dm_data` (slow) | on φ = 2 data, count-model `bse` / empirical SD < 0.5 while DM `bse` / empirical SD in [0.8, 1.25] |
| `test_dirichlet_multinomial.py::test_is_swr_concentration_recovered` | planted `log_phi_is_swr = -1.0` recovered within 0.3 |
| `test_dirichlet_multinomial.py::test_event_constant_predictor_rejected` | a within-event-constant column raises (same rule as `within_group_full_rank`) |
| `test_dirichlet_multinomial.py::test_cv_folds_over_events` | held-out folds partition events; bits vs null positive on planted data |
| `test_compare_content_estimators.py` | on a synthetic two-session cohort the script produces the parquet + summary with one row per (session, rung, term, estimator) |
| Real-data smoke (slow) | comparison runs on `--max-per-animal 1`; wall time and RSS recorded in the PR |

## Fixtures

- `simulate_dirichlet_multinomial_counts` (new) with seeds pinned.
- The phase-2a frozen cohort (real data) for the smoke, gated on `PROCESSED_DATA_DIR`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- No old code path is removed in this phase; confirm nothing was.
- User-facing documentation listed as tasks is updated, not deferred.
- The comparison summary and both figures are attached to the PR description.
