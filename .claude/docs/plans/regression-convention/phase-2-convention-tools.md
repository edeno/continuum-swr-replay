# Phase 2 — The convention's tools: terms, collinearity, support, record, folds, bands, inference

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#smooth-terms-and-bases)

Branch: `regression-convention-tools` from master after Phase 1. Reads no canonical table except one
real-data smoke test.

**Status:** done. The tools are on master (merged with 492dffb4, 2026-11-21). The declared-penalty
parts are built on branch `regression-declared-penalty` (pushed, not merged on 2026-11-23):

- b47faec9 and 39a51d3a: `analysis/regression/calibration.py:log_penalty_for_edf`, the declared log
  penalty of a per-session smooth, found by its median edf over sessions (with
  `tests/analysis/regression/test_calibration.py`);
- 86ad3fc5: the recovery test `test_sessions.py::test_the_declared_penalty_recovers_a_planted_curve` and
  the stored check `scripts/checks/simulate_declared_penalty.py` (with
  `tests/scripts/checks/test_simulate_declared_penalty.py`).

The smooth-term rule is decided (design, seventeenth round; the author's reading of the lead's "For
decision 2, I think I would prefer the penalized splines" and "I want to stick with one fit per session
for now", 2026-11-22, told to the lead and not objected to): a per-session smooth term is the cubic
P-spline at one fixed log penalty per term, declared in the model and the same in every session; no
penalty is chosen from the data for a per-session smooth term. How a session without a finite maximum is
reported is decided (lead, 2026-11-21: with those terms missing).

**Inputs to read first:**

- The design, sections 4.2 to 4.7.
- Phase 1's package `src/continuum_swr_replay/analysis/regression/` (module docstrings),
  `docs/design/regression_fitter.md`, [designs.md#the-shared-fitter-built](designs.md#the-shared-fitter-built)
  (its interface) and `TODO.md`, section "Regression fitter" (the follow-ups this phase meets: the step cap
  and column scaling, who refits the set-aside sessions).
- The design, ninth to fourteenth rounds of section 9, and its fifteenth round (the two open questions).
- Scratch `proto_uncertainty_measures/REPORT.md` (the per-session recency variants and the row rule for an
  inactive role column; simulation).
- `src/continuum_swr_replay/analysis/shared_statistics.py:51` (`signed_rank_p`), `:75`
  (`benjamini_hochberg`), `:191` (`animal_session_values`), `:294-510` (`nested_codes`,
  `nested_bootstrap_draws`, `bootstrap_interval`): what the inference reuses.
- `src/continuum_swr_replay/tables/standardization.py`: the repository's current standardization helpers
  (one of the six schemes the design wants replaced by one).
- `src/continuum_swr_replay/analysis/replay_models/contrast_aggregation.py:90-100`
  (`MAX_UNIDENTIFIED_SHARE`, `MIN_SESSIONS_PER_TERM`): the current per-term gate's constants.
- Branch `swr-trial-folds`: `git show swr-trial-folds:src/continuum_swr_replay/analysis/event_evidence/tetrode_holdout.py`
  lines 109-372 (`time_blocks`, `trial_segments`, `task_segment_starts`, `capped_groups`, `trial_folds`).
- Scratch: `S/proto_basic/check_aggregation.py` (shared against per-session penalty; history only, the declared-penalty check replaced it),
  `S/verify_rate/s5_inference.py` (`infer`), `S/verify_uncertainty/s07_inference.py`.

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C8](shared-contracts.md#c8-the-declared-family-of-tests), [C12](shared-contracts.md#c12-layering-of-the-new-package).

**Designs referenced:** [smooth terms](designs.md#smooth-terms-and-bases),
[support rule](designs.md#support-rule), [collinearity](designs.md#collinearity-measures),
[trial folds](designs.md#trial-folds), [simultaneous bands](designs.md#simultaneous-bands),
[session to day to animal](designs.md#session-to-day-to-animal).

## Tasks

- **The smooth-term rule** (design, seventeenth round). A `Smooth` term is a cubic P-spline
  (`pspline.PSplineBasis`) with a required `penalty`: one fixed log penalty, declared in the model and the
  same in every session, or `"none"`. There is no `"reml"` option for a per-session term, and no
  natural-spline basis is added. As built (`terms.py`): `penalty` is required, `"none"` or a fixed log
  penalty inside `LOG_PENALTY_BOUNDS`.
- **Terms and design assembly** (`analysis/regression/terms.py`, new). A small, explicit term vocabulary:
  `Smooth(column, lo, hi, n_basis, clamp=True, by=None, penalty=<required>)` (`by`: one smooth per level
  of a category column such as the arm role, each with its own penalty; `penalty`: the declared fixed log
  penalty, or `"none"` for an unpenalized B-spline of the same basis), `LinearFunctional(lag_columns, lags, lo, hi, n_basis)` (model D),
  `Category(column, levels, reference)`, `Slope(column, by=None)` (a slope, optionally one per level of a
  category column such as the arm role), `Offset(column)`. A `Design` built from a list of terms
  and pooled reference rows holds the fixed knots, the pooled sum-to-zero constraints, the penalty blocks,
  the pooled standardization (population SD, `ddof = 0`) and centering references, and maps any session's
  rows to `(X, offset, blocks)`. Pooled quantities are computed once from the cohort rows and stored; never
  per session. A P-spline term is built with `pspline.PSplineBasis(...).constrained(pooled_x)`, whose
  `ConstrainedSmooth.constraint` maps any session's rows (`basis.design(x_s) @ constraint`); its penalty
  becomes a `reml.PenaltyBlock`. Each session goes to the fitter as a `reml.SessionData(X, y, aux, name,
  active)` (no start: the fitter chooses it).
- **The design check: no column whose values are all very small** (in `terms.py`, where a session's `X`
  is built). The fitter's step cap is in coefficient units, so a column whose values are all very small
  makes a fit stop at the step limit (`TODO.md`, "The step cap depends on column scaling"). Two
  replacement rules for the cap failed, so the fitter keeps its rule and the design refuses such a column,
  with a message that names it, before any fit. Measured on simulated designs of 600 rows: an N(0,1)
  column multiplied by 1e-2 converges; multiplied by 1e-4, the Poisson, Bernoulli and softmax fits end at
  the step limit. Set the threshold here, on a stated measure of a column's size (for example its largest
  absolute value over the session's rows), as a constant of `terms.py` (not of the fitter's
  `constants.py`). Failing test first: `test_terms.py::test_tiny_column_is_refused`, which fits the same
  design through `reml.fit_at_penalties` (it raises, the fit ending at "step limit reached") and then
  asserts that the design check refuses the 1e-4 column and accepts the 1e-2 one. Verify that every
  column the convention builds (pooled-SD slopes, 0/1 terms, constrained spline columns, model D's
  functional) passes on the synthetic fixtures.
- **Standardization**: one helper in `terms.py` (pooled SD with `ddof = 0`; centering of a continuous
  factor at a fixed pooled reference when it enters an interaction; a 0/1 factor never centered). Do not
  change `tables/standardization.py`; new models do not call it.
- **Collinearity** (`analysis/regression/collinearity.py`, new): `term_collinearity(fit, design)` returning,
  per term, `r2` and `vif` (one column) or `observed`, `worst`, `estimate` (several), on the weighted design
  of [designs.md#collinearity-measures](designs.md#collinearity-measures); `separable(record, threshold=...)`
  applying the rule. The threshold, R² 0.8, is a constant defined in this module (new; the fitter's
  `constants.py` does not hold it). The weights come from the fit (`RemlScore.fit`). Add
  `concurvity(full = TRUE)` to `scripts/checks/compare_regression_to_mgcv.R` and its fixtures (it does not
  print it today).
- **Support rule** (`analysis/regression/support.py`, new): the three cases of
  [designs.md#support-rule](designs.md#support-rule), evaluated before the fit from the rows; a
  `terms_failing_share(records)` that applies the 20 % rule over sessions (the value of the fitter's
  `MAX_FAILING_SHARE`, imported, not copied). The minimum of 5 units each side is a constant defined in
  this module (new). A term that fails is made inactive in that session (`SessionData.active`); when it is
  a 0/1 role column, the session's rows with that role are dropped from its fit and counted in the
  record's `cohort_counts`, not merged into the reference role (found by simulation: merging inflated the
  belief variance's false positives, to 20 % in one case; scratch `proto_uncertainty_measures/REPORT.md`).
  A smooth term carries its declared penalty and has no support rule of its own (designs.md, "Support
  rule"). A support rule per grid point for curves averaged over sessions is an idea, not decided
  (`OPEN (lead): grid-point support`).
- **Sessions without a finite maximum** (in `sessions.py`, below). Under the smooth-term rule the
  models are fitted at declared penalties or with none, so the fitter's search does not run, and
  `reml.fit_at_penalties` raises at the first session whose fit has no finite maximum. Today only
  `reml.select_penalties` handles such a session (it sets it aside and refits it at the chosen penalties
  with the running-away unpenalized columns inactive, at most `REFIT_MAX_ROUNDS` rounds; `reml._refit`).
  This task gives the per-session loop the same handling outside the search:
  1. Fit each session alone (`reml.pooled_reml(lik, [session], blocks, rho, derivatives=False)` returns
     the failing session's `RemlScore` in `failed` instead of raising). When its status is "no finite
     maximum" along unpenalized design columns only (no `FreeDirection` with a `block` in
     `running_away_directions`), make `fit.running_away` inactive and fit again, at most
     `REFIT_MAX_ROUNDS` rounds. Follow `select_penalties`' rules: at the session's first fit a design not
     identified, or a numerical failure, is an error that names the session; in a refit a direction
     through a penalty block, or any other status, leaves the session without a fit, with the reason in
     words. When a column made inactive is a 0/1 role column,
     drop that role's rows from the session, as the support rule does (fact found by simulation, above).
  2. One implementation only (C13, C14): the loop's rule and the search's must not be two copies. Make
     the refit a public function of `reml.py` that both call (the author's call: the search's behavior
     and its tests do not change, and the rule stays in one place; this settles `TODO.md`, "Who refits
     the set-aside sessions": remove that entry). Add the new tests beside the fitter's.
  3. What is recorded for such a session (C7): its `status`, the columns left out with the reason
     (`left_out`), NaN for those terms' coefficients and covariance, and the round count.
  4. How such a session is reported (lead, 2026-11-21: "do terms missing"): it is kept, with those terms
     missing, and it is not left out. The caller owns the handling, since it owns the support rule.
     Leaving the session out would depend on the outcome (the sessions that fail are those with no events
     in one condition). In simulated content data 4 of 242 sessions were of this kind, with 5 or 6 arm and
     role columns left out in each (simulation, not a result).
  Failing tests first, in `tests/analysis/regression/test_sessions.py`:
  `test_a_separated_session_is_kept_with_its_terms_missing` (synthetic sessions at a given penalty, one
  with a 0/1 column whose rows hold no event: `fit_at_penalties` raises on them; the loop returns a fit for
  every session, the separated one with that column in `left_out` and NaN there, the others equal to
  their `fit_at_penalties` fits alone) and `test_an_inactive_role_drops_its_rows` (a session whose role
  column is made inactive fits on the rows without that role, and the count of dropped rows is recorded).
  Verify on the simulated content model of the gate study (scratch) that the sessions it reported as
  without a finite maximum are fitted with the same columns left out.
- **Fit-row record** (`analysis/regression/record.py`, new): build one row of
  [C7](shared-contracts.md#c7-the-fit-row-record) from a fit, its design, support and collinearity; write
  and read a parquet of rows (JSON columns as strings) with `tables/io.py:215` (`write_atomically`).
- **Per-session fitting loop** (`analysis/regression/sessions.py`, new): given one `Design`, a mapping
  session -> rows, a likelihood and the kind, run the design check, support, the per-session fits,
  the handling of sessions without a finite maximum (task above), collinearity, and return the records. No
  task knowledge here. The loop fits each session at the design's declared log penalties
  (`reml.refit_session`); it chooses no penalty. `reml.select_penalties` stays in the package for a
  family that may still choose a penalty from the data (`OPEN (lead): mPFC and history penalties`).
- **Linear functionals of a fit** (`analysis/regression/contrasts.py`, new): the estimate and exact
  variance of `c' beta` for a curve contrast (the mean of a smooth over one range minus over another, on a
  grid) and for a coefficient difference; used for every scalar estimand of C8.
- **Folds** (`analysis/regression/folds.py`, new): the fold rule of
  [designs.md#trial-folds](designs.md#trial-folds), same signature as the `swr-trial-folds` function.
  Copy its logic (with attribution in the docstring to the module it came from); do not import
  `analysis/event_evidence/` (layering). The number of folds (5) and the longest stay piece (30 s) are
  constants defined in this module (new). Phase 4 is done without that branch (it is not on master); when
  it merges, the phase open at that time makes the event-evidence code call this one and adds the parity
  test (Phase 4's fold task, which its text sends to Phase 5a).
- **Simultaneous bands** (`analysis/regression/bands.py`, new): both forms of
  [designs.md#simultaneous-bands](designs.md#simultaneous-bands) (per fit; across sessions from nested
  bootstrap draws), seed stored.
- **Day-level inference** (in `analysis/shared_statistics.py`): `animal_day_session_values` and
  `day_level_test` of [designs.md#session-to-day-to-animal](designs.md#session-to-day-to-animal), plus
  `estimand_summary(values, animal, day, session, *, n_draws, seed)` returning the animal-balanced estimate,
  the t-corrected nested bootstrap interval, the animals with the estimate's sign, and the day-level p.
- **Declared tests** (`analysis/replay_models/declared_tests.py`, new; not in the fitter package, which knows nothing of the task): the list of
  [C8](shared-contracts.md#c8-the-declared-family-of-tests) (twelve estimands, 36 tests; settled), each entry with its question,
  estimand id, model, and contrast definition (range pairs or coefficient names). One tuple; nothing else
  hard-codes the count.
- **The declared penalty, found by a target edf** (built, b47faec9, 39a51d3a):
  `analysis/regression/calibration.py:log_penalty_for_edf(design, sessions, lik, *, term, target_edf,
  response, unit)` bisects the log penalty of one fixed-penalty term inside `LOG_PENALTY_BOUNDS` until the
  median edf over sessions (from the records of `fit_sessions`, the loop the model uses) is the target,
  the other penalties held. Each model phase calls it on that model's own real rows, writes the value
  into the term as a constant, and reports the fit also at a stiffer and a looser value. A target does
  not carry from one model to another (design, seventeenth round; simulation).
- **Stored check of the declared-penalty rule** (built, 86ad3fc5; replaces the planned
  shared-against-per-session check): `scripts/checks/simulate_declared_penalty.py` simulates the real
  cohort's shape (5 animals, 45 days, 256 sessions of real-sized visit bins and event numbers), declares
  the arrival smooth's log penalty at a median edf of 4, 6 and 8 on one data set, and writes, per target
  and curve (a planted threefold rise, a flat curve), the bias, SD, rejections over animals and interval
  coverage of the animal-balanced Q1 contrast, to `declared_penalty.parquet` in `--output-dir`. It moves
  onto the simulation engine in [Phase 2b](phase-2b-simulation-engine.md).
- **Docs**: extend `docs/design/regression_fitter.md` with the term vocabulary, the design check, the
  collinearity and support rules (with the row rule for an inactive role column), the handling of sessions
  without a finite maximum outside the search, the fold rule and its use, and the band construction.
  Mark the `TODO.md` items this phase closes (the step cap's design check; who refits the set-aside
  sessions).

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
| `test_sessions.py::test_the_declared_penalty_recovers_a_planted_curve` (built, 86ad3fc5) | 30 simulated sessions with a threefold rise after arrival: at the log penalty that gives a median edf of 6 (`log_penalty_for_edf`), every session's fit converges and records its penalty as fixed, and the animal-balanced Q1 contrast is the planted one to 0.15 |
| `test_calibration.py` (built, b47faec9) | the found log penalty gives the target median edf; the edf falls as the penalty rises; deterministic; the other penalties are held; a target out of reach, or a term without a fixed penalty, is refused |
| `tests/scripts/checks/test_simulate_declared_penalty.py` (built, 86ad3fc5) | the summed bins fit as the bins on one session; the planted contrast; the table on a tiny cohort |
| `test_sessions.py::test_a_separated_session_is_kept_with_its_terms_missing` | a session without a finite maximum at a given penalty is fitted with its running-away columns left out and recorded so; the others are unchanged |
| `test_sessions.py::test_an_inactive_role_drops_its_rows` | a 0/1 role column made inactive drops that role's rows from the session's fit, with the count recorded |
| `test_terms.py::test_tiny_column_is_refused` | a column scaled by 1e-4 is refused before the fit (unrefused, the fit stops at the step limit); one scaled by 1e-2 is accepted |
| `test_contrasts.py::test_curve_contrast_variance` | the variance of a range-mean contrast equals the empirical variance over 2,000 draws from N(beta, V) to 2 % |
| `test_folds.py::test_trial_folds_rules` | every bin of a group in one fold; long stays cut at 30 s; classes balanced; deterministic; raises with fewer groups than folds |
| `test_bands.py::test_simultaneous_band_coverage` (slow) | over 500 simulated fits, the simultaneous band covers the whole true curve in 93-97 %, and the pointwise band less often |
| `tests/analysis/test_shared_statistics.py::test_animal_day_session_values` | three-level reduction on a hand table (mean and median) |
| `test_shared_statistics.py::test_day_level_test` | equals `signed_rank_p` of the day values; NaN below six non-zero differences |
| `test_shared_statistics.py::test_estimand_summary` | estimate, interval (as `bootstrap_interval`), sign count, day p on a hand table with a fixed seed |
| `tests/analysis/replay_models/test_declared_tests.py::test_list_is_unique_and_complete` | every estimand has a model and contrast; ids unique; each is tested for SWR, No SWR and the difference |

## Fixtures

All synthetic, built in `tests/analysis/regression/conftest.py` (fixed seeds), plus the Phase 1 mgcv
fixtures extended with concurvity numbers.

## Verification and runtime

- Smoke test on one real session: build a two-smooth design from the visit bins
  (`replay_models/visit_design.py:visit_timebins`, Phase 3, on master) and run the per-session loop on that
  one session; record time. This is a smoke test of the tools, not a model.
- The stored check runs in minutes on CPU with `--workers`; record its wall time.

## Rebuilt

`declared_penalty.parquet` of the stored check, in the `--output-dir` given (on 2026-11-23 the author's
run wrote it to a scratch folder only; Phase 2b stores it under
`models_convention/simulations/`). Nothing in `paper/reported/`.

## Exit criteria

The validation slice passes (done on master, 492dffb4, and on `regression-declared-penalty`); the
stored check of the declared-penalty rule runs and its numbers are stated in the PR from its output file;
the declared list matches the lead's decisions (twelve estimands, 36 tests); the branch
`regression-declared-penalty` is merged after the phase-boundary review and the lead's approval.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
