# Phase 2 — The convention's tools: terms, collinearity, support, record, folds, bands, inference

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#smooth-terms-and-bases)

Branch: `regression-convention-tools` from master after Phase 1. Reads no canonical table except one
real-data smoke test.

**Inputs to read first:**

- The design, sections 4.2 to 4.7.
- Phase 1's package `src/continuum_swr_replay/analysis/regression/`.
- `src/continuum_swr_replay/analysis/shared_statistics.py:51` (`signed_rank_p`), `:75`
  (`benjamini_hochberg`), `:191` (`animal_session_values`), `:294-510` (`nested_codes`,
  `nested_bootstrap_draws`, `bootstrap_interval`): what the inference reuses.
- `src/continuum_swr_replay/tables/standardization.py`: the repository's current standardization helpers
  (one of the six schemes the design wants replaced by one).
- `src/continuum_swr_replay/analysis/replay_models/contrast_aggregation.py:90-100`
  (`MAX_UNIDENTIFIED_SHARE`, `MIN_SESSIONS_PER_TERM`): the current per-term gate's constants.
- Branch `swr-trial-folds`: `git show swr-trial-folds:src/continuum_swr_replay/analysis/event_evidence/tetrode_holdout.py`
  lines 109-372 (`time_blocks`, `trial_segments`, `task_segment_starts`, `capped_groups`, `trial_folds`).
- Scratch: `S/proto_basic/check_aggregation.py` (shared against per-session penalty),
  `S/verify_rate/s5_inference.py` (`infer`), `S/verify_uncertainty/s07_inference.py`.

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C8](shared-contracts.md#c8-the-declared-family-of-tests), [C12](shared-contracts.md#c12-layering-of-the-new-package).

**Designs referenced:** [smooth terms](designs.md#smooth-terms-and-bases),
[support rule](designs.md#support-rule), [collinearity](designs.md#collinearity-measures),
[trial folds](designs.md#trial-folds), [simultaneous bands](designs.md#simultaneous-bands),
[session to day to animal](designs.md#session-to-day-to-animal).

## Tasks

- **Terms and design assembly** (`analysis/regression/terms.py`, new). A small, explicit term vocabulary:
  `Smooth(column, lo, hi, n_basis, clamp=True)`, `LinearFunctional(lag_columns, lags, lo, hi, n_basis)`
  (model D), `Category(column, levels, reference)`, `Slope(column, by=None)` (a slope, optionally one per
  level of a category column such as the arm role), `Offset(column)`. A `Design` built from a list of terms
  and pooled reference rows holds the fixed knots, the pooled sum-to-zero constraints, the penalty blocks,
  the pooled standardization (population SD, `ddof = 0`) and centering references, and maps any session's
  rows to `(X, offset, blocks)`. Pooled quantities are computed once from the cohort rows and stored; never
  per session.
- **Standardization**: one helper in `terms.py` (pooled SD with `ddof = 0`; centering of a continuous
  factor at a fixed pooled reference when it enters an interaction; a 0/1 factor never centered). Do not
  change `tables/standardization.py`; new models do not call it.
- **Collinearity** (`analysis/regression/collinearity.py`, new): `term_collinearity(fit, design)` returning,
  per term, `r2` and `vif` (one column) or `observed`, `worst`, `estimate` (several), on the weighted design
  of [designs.md#collinearity-measures](designs.md#collinearity-measures); `separable(record, threshold=0.8)`
  applying the rule.
- **Support rule** (`analysis/regression/support.py`, new): the three cases of
  [designs.md#support-rule](designs.md#support-rule), evaluated before the fit from the rows; a
  `terms_failing_share(records)` that applies the 20 % rule over sessions.
- **Fit-row record** (`analysis/regression/record.py`, new): build one row of
  [C7](shared-contracts.md#c7-the-fit-row-record) from a fit, its design, support and collinearity; write
  and read a parquet of rows (JSON columns as strings) with `tables/io.py:215` (`write_atomically`).
- **Per-session fitting loop** (`analysis/regression/sessions.py`, new): given one `Design`, a mapping
  session -> rows, a likelihood and the kind, run support, pooled REML (`reml.select_penalties` over all
  sessions), the final per-session fits at the shared penalties, collinearity, and return the records. No
  task knowledge here.
- **Linear functionals of a fit** (`analysis/regression/contrasts.py`, new): the estimate and exact
  variance of `c' beta` for a curve contrast (the mean of a smooth over one range minus over another, on a
  grid) and for a coefficient difference; used for every scalar estimand of C8.
- **Folds** (`analysis/regression/folds.py`, new): the fold rule of
  [designs.md#trial-folds](designs.md#trial-folds), same signature as the `swr-trial-folds` function.
  Copy its logic (with attribution in the docstring to the module it came from); do not import
  `analysis/event_evidence/` (layering). When that branch is on master, Phase 4 makes the event-evidence code
  call this one and adds the parity test.
- **Simultaneous bands** (`analysis/regression/bands.py`, new): both forms of
  [designs.md#simultaneous-bands](designs.md#simultaneous-bands) (per fit; across sessions from nested
  bootstrap draws), seed stored.
- **Day-level inference** (in `analysis/shared_statistics.py`): `animal_day_session_values` and
  `day_level_test` of [designs.md#session-to-day-to-animal](designs.md#session-to-day-to-animal), plus
  `estimand_summary(values, animal, day, session, *, n_draws, seed)` returning the animal-balanced estimate,
  the t-corrected nested bootstrap interval, the animals with the estimate's sign, and the day-level p.
- **Declared tests** (`analysis/regression/declared_tests.py`, new): the list of
  [C8](shared-contracts.md#c8-the-declared-family-of-tests) as decided in L1, each entry with its question,
  estimand id, model, and contrast definition (range pairs or coefficient names). One tuple; nothing else
  hard-codes the count.
- **Stored analysis: shared against per-session penalty** (`scripts/checks/simulate_shared_penalty.py`,
  new; adapted from `S/proto_basic/check_aggregation.py`): simulated sessions of the real size, curve
  recovery error with a shared penalty against per-session penalties; writes
  `Processed-Data/replay_content/models_convention/simulations/shared_penalty.parquet`.
- **Docs**: extend `docs/design/regression_fitter.md` with the term vocabulary, the collinearity and
  support rules, the fold rule and its use, and the band construction.

## Deliberately not in this phase

- Any task design (which columns, which rows, which roles): Phases 3, 5a, 6a.
- The family's FDR over the declared tests: Phase 9 (it needs all estimands).
- Changing the event-evidence code to call the shared fold function: Phase 4, after the other session's
  branch is on master.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/regression/test_terms.py::test_fixed_knots_and_pooled_constraint` | two sessions with different ranges get the same knots and the same constraint matrix; a session's curve equals the pooled-design curve at the same coefficients |
| `test_terms.py::test_clamp_beyond_range` | values above `hi` give the basis row at `hi` |
| `test_terms.py::test_linear_functional_matches_explicit_sum` | the model D columns equal `sum_l B_b(log l) r_l` computed by a loop on a small case |
| `test_terms.py::test_standardization_ddof0_and_centering` | pooled SD uses `ddof = 0`; an interaction's continuous factor is centered at the stored reference; a 0/1 factor is not |
| `test_collinearity.py::test_single_column_equals_r2` | for one column, observed = worst = estimate = R² of an OLS of the weighted column on the rest; VIF = 1/(1 - R²) |
| `test_collinearity.py::test_matches_mgcv_concurvity` | on the Phase 1 fixtures with two smooths, observed, worst and estimate equal mgcv's `concurvity(full = TRUE)` (add those numbers to the fixture outputs) to 1e-4 |
| `test_collinearity.py::test_softmax_weighting` | for a grouped design, the weighted cross-product equals the softmax information |
| `test_support.py::test_binary_term_counts` | hand-built visits: counts of units with events on each side and the pass flag |
| `test_support.py::test_role_slope_counts_and_constant` | a role slope constant within the role fails; median split counts are right |
| `test_support.py::test_failing_share_rule` | a term failing in 21 % of sessions is flagged, 20 % is not |
| `test_record.py::test_round_trip` | a record written and read back is identical, JSON columns included |
| `test_sessions.py::test_shared_penalty_recovers_planted_curve` | 30 simulated sessions with one planted smooth: the shared penalty is selected once, every session's fit converges, the animal-balanced curve covers the truth |
| `test_contrasts.py::test_curve_contrast_variance` | the variance of a range-mean contrast equals the empirical variance over 2,000 draws from N(beta, V) to 2 % |
| `test_folds.py::test_trial_folds_rules` | every bin of a group in one fold; long stays cut at 30 s; classes balanced; deterministic; raises with fewer groups than folds |
| `test_bands.py::test_simultaneous_band_coverage` (slow) | over 500 simulated fits, the simultaneous band covers the whole true curve in 93-97 %, and the pointwise band less often |
| `tests/analysis/test_shared_statistics.py::test_animal_day_session_values` | three-level reduction on a hand table (mean and median) |
| `test_shared_statistics.py::test_day_level_test` | equals `signed_rank_p` of the day values; NaN below six non-zero differences |
| `test_shared_statistics.py::test_estimand_summary` | estimate, interval (as `bootstrap_interval`), sign count, day p on a hand table with a fixed seed |
| `tests/analysis/regression/test_declared_tests.py::test_list_is_unique_and_complete` | every estimand has a model and contrast; ids unique; each is tested for SWR, No SWR and the difference |

## Fixtures

All synthetic, built in `tests/analysis/regression/conftest.py` (fixed seeds), plus the Phase 1 mgcv
fixtures extended with concurvity numbers.

## Verification and runtime

- Smoke test on one real session: build a two-smooth design from the current stop bins
  (`per_bin_design.aggregate_to_timebins`, `per_bin_design.py:394`, as now) and run the per-session loop on
  that one session; record time. This is a smoke test of the tools, not a model.
- The stored simulation runs in minutes on CPU; record its wall time.

## Rebuilt

`Processed-Data/replay_content/models_convention/simulations/shared_penalty.parquet` only. Nothing in
`paper/reported/`.

## Exit criteria

The validation slice passes; the simulation output exists and shows the shared penalty no worse than
per-session penalties (state the numbers in the PR from the stored file); the declared list matches the
lead's decision L1.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
