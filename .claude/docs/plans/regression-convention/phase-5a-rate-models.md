# Phase 5a — Rate on the visit: three clocks, models A and D (Q1, Q2, Q7a)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#rate-design)

Branch: `visit-rate-models` from master after Phases 2 and 3. CPU only.

**Inputs to read first:**

- The design, sections 2.1, 2.4, 2.5, 2.7, 3 (Q1, Q2, Q7), 4.1 to 4.7, and decision 3 (both forms of the
  reward effect).
- Phase 2's `analysis/regression/` (`terms.py`, `sessions.py`, `contrasts.py`, `collinearity.py`,
  `support.py`, `record.py`, `declared_tests.py`) and Phase 3's
  `analysis/replay_models/visit_design.py` (`visit_timebins`).
- The family it replaces, to compare against: `analysis/replay_models/rate/per_bin_rate_ladder.py:98`
  (`fit_ladder_session`), `:254` (`fit_per_bin_rate_models`); `model_specs.py:879` (`MODEL_PER_BIN_RATE`);
  `reported/behavior/rate.py` (`per_bin_values`, `per_bin_detail_values`) and the `rate_*` keys of
  `paper/reported/behavior.json`; `docs/rate_models.md`.
- `src/continuum_swr_replay/analysis/replay_models/rate/rate_fit_statistics.py` and
  `analysis/replay_models/coefficient_statistics.py:135` (`coefficient_statistics`): how the current values
  are summarized (to know what changes).
- `src/continuum_swr_replay/reported/questions.py:320-358` (Part 3 questions), `reported/behavior/common.py:20-35`
  (`TAGS`), `reported/behavior/__init__.py:132` (`summarize`), `reported/values.py` (`value_factory`, `pend`).
- Scratch: `S/proto_rate/proto_core.py` (`simulate`, `calibrate_b0`, `bin_events`, `design_proposed`),
  `S/proto_rate/p2_m7r_try.py`, `p3_run.py`, `log_full_null.txt`, `S/verify_rate/s4_model_free_cells.py`,
  `s7_cells_tte.py`, `S/my_checks/three_clock_overlap.py`.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct),
[C7](shared-contracts.md#c7-the-fit-row-record), [C8](shared-contracts.md#c8-the-declared-family-of-tests),
[C9](shared-contracts.md#c9-where-new-fits-are-stored), [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

**Designs referenced:** [rate design](designs.md#rate-design), [smooth terms](designs.md#smooth-terms-and-bases),
[pooled REML](designs.md#pooled-reml), [session to day to animal](designs.md#session-to-day-to-animal).

**Decision needed first:** L1 (the declared family), fixed in `declared_tests.py` and committed before the
first real fit of this phase.

## Tasks

- **Baseline, before any code change**: copy `paper/reported/behavior.json` to the phase scratch directory;
  read and save the current per-bin rate values (the `rate_*` keys) and Phase 3's model-free clock curves
  (`visit_clock_curves.parquet`). These are the comparison, not targets.
- **Model specs** (`analysis/replay_models/rate/visit_rate.py`, new): the two models of
  [designs.md#rate-design](designs.md#rate-design), per kind (`swr`, `no_swr`), as term lists:
  - A: `Offset(log exposure) + Smooth(s, 0, 20) + Smooth(u, 0, 3) [+ Smooth(m, 0, 1.5)] + Slope(last_outcome_rewarded)` (a 0/1 column, not centered);
  - D: the same smooths + `LinearFunctional(lag_1..lag_10, log lags, 0, log 10)`.
  The rule for the third clock is written in the module before any fit: fit A with `f_m`; if the
  concurvity `estimate` of `f_m` or `f_u` is at least 0.8 in more than 20 % of sessions, refit without `f_m`
  and report the time-until-movement curve only from the model-free curves (Figure 4c second half). Store
  both fits.
- **Cohort step** (in the same module): `visit_timebins` per session -> `select_visit_context` -> drop
  rows with a missing covariate (counted per kind) -> D additionally keeps bins with ten known lags
  (counted). Store `count_visit_context` and the drop counts in each record.
- **Fit** through `analysis/regression/sessions.py`: support rule for the 0/1 term; one pooled REML
  selection per (model, kind); per-session fits at the shared penalties; collinearity measures; records.
- **Estimands** (`analysis/replay_models/rate/visit_rate_statistics.py`, new), each per session with its
  exact variance, then `shared_statistics.estimand_summary` per kind and for the difference:
  - Q1: mean of `f_s` over `[2, 4] s` minus mean over `[0, 1] s` (log rate ratio);
  - Q2: the coefficient of `last_outcome_rewarded` in model A (log rate ratio), the reward effect at fixed
    time before leaving;
  - Q7a: mean of `f_u` over `[0, 0.25] s` minus mean over `[1, 2] s`;
  - model D: the curve over lags 1 to 10 with its band (described, not tested).
- **Simulations** (stored analysis `scripts/replay_models/rate/simulate_visit_rate.py`, new, from
  `S/proto_rate/proto_core.py`): event starts drawn on the real visits' at-risk time of every session from a
  known rate `exp(b0 + f_s + f_u + beta * rewarded)`, with the counting rule (an event crossing the last
  poke-out is not counted, nor at risk once started). Scenarios: (a) no reward effect; (b) rate ratio 1.3;
  (c) a sharp lag-1 effect only, to see what D returns; (d) a drop on `u` in the last 0.3 s with no reward
  effect, which must not move to `s` or to the reward term; (e) a drop before small movements only (`f_m`)
  with flat `f_u`, which the model must separate or the rule above must catch. Twenty data sets each over
  all sessions; outputs to `models_convention/simulations/visit_rate.parquet`.
- **Runner** `scripts/replay_models/rate/fit_visit_rate_models.py` (new): reads the tables, fits both
  models and kinds, writes `models_convention/rate/{A,D}.parquet` and a manifest; registered in
  `scripts/workflows/run_refits.py` (`RUNS`, `:152`) as `rate.visit.A` and `rate.visit.D`.
- **Questions**: add Q1 to Q7 of the design's section 3 as new Part 3 questions in
  `reported/questions.py` (call: ids `3.11` to `3.17`, wording from the design's table); add their tags to
  `reported/behavior/common.py` (`TAGS`).
- **Summary values** (`reported/behavior/visit_rate.py`, new; called from `reported/behavior/__init__.py`):
  per kind and difference, the estimate as a rate ratio, its interval, animals with the sign, day-level p,
  sessions kept and left out (support, convergence, collinearity), cohort counts per label. No q-values
  yet (Phase 9). Pending notes follow C10.
- **Docs**: `docs/rate_models.md` gains a section on the visit model (unit, clocks, A and D, the rule for
  the third clock, what is tested and what is described), written with value markers; a model table for
  the new models (in `docs/models_table.md`, a new section, or a new doc linked from it).

## Deliberately not in this phase

- The per-stop (total) reward effect, the SWR-contrast covariates, the uncertainty rate model (Phase 5b).
- FDR over the family and figure bands (Phase 9), Figure 4 (Phase 10).
- Removing the per-bin ladder, its runner, its values (Phase 12). Its stored fits stay as the baseline.
- The duration-aware and ridge history models (Phase 11).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/rate/test_visit_rate.py::test_null_reward_with_counting_rule` | on synthetic visits whose ends are earlier after omissions, events from a rate with no reward effect and the counting rule on: model A's reward ratio is within 2 standard errors of 1 over 20 data sets, where a one-clock model (no `f_u`) is biased upward |
| `test_visit_rate.py::test_planted_reward_ratio_recovered` | planted ratio 1.3 recovered (bias within 2 SE; 95 % interval covers in 18 of 20) |
| `test_visit_rate.py::test_end_drop_stays_on_u` | a planted `u` drop does not move the reward term or `f_s` beyond nominal |
| `test_visit_rate.py::test_third_clock_rule` | with `m = u` in every bin the rule drops `f_m`; with half the at-risk time before small movements it keeps it |
| `test_visit_rate.py::test_model_d_sharp_lag1` | with a lag-1-only truth, D's curve at lag 1 covers the truth and its band includes 0 at lags 3 to 10 |
| `test_visit_rate_statistics.py::test_estimands_are_linear_functionals` | Q1 and Q7a equal the grid means of the fitted smooths; variances equal `c' V c` |
| `test_visit_rate_statistics.py::test_difference_between_kinds` | the difference estimand is the difference of the per-kind session values, reduced the same way |
| `tests/reported/behavior/test_visit_rate_values.py::test_values_and_tags` | the builder emits one value per estimand, kind and difference, tagged with the new questions; refuses when an epoch has no row |
| `tests/reported/test_reported_questions.py` | still passes with the new questions |
| full-cohort run (script, slow) | every session either fits or has a recorded reason; support and collinearity records stored |

## Fixtures

Synthetic visits, stops, outcomes and events with fixed seeds (`tests/analysis/replay_models/rate/conftest.py`),
reusing the Phase 3 synthetic helpers. Real data only in the runner's smoke test.

## Verification and runtime

- Simulated first (the tests and the stored simulation), then one real session per animal (five), then all
  242 epochs.
- Smoke test on one session: build, fit A and D for both kinds at a fixed penalty, then one pooled REML
  evaluation over the five sessions. Time both (the design measured about 0.08 s per two-clock fit). The
  selection over 242 sessions costs (grid points + Newton steps) x 242 x (one fit); with up to 125 grid
  points that is about 40 minutes per (model, kind) on one core; parallelize over sessions with a process
  pool if the smoke test confirms it; run in `tmux` (`csr-5a-rate`).
- Compare with the baseline: Q1's sign and the rise over the first seconds against the model-free
  `s` curve; Q7a against the model-free `u` curve; Q2 for SWR against the old lag-1 value (the design's
  section 2.1 says it falls when the end-of-stop time is modeled; state the old and new values from the two
  summary files in the PR, with the simulation's null result beside them); the kind difference of Q2 against
  the old one. A change of sign or a difference far beyond the simulation's spread is investigated before the
  PR.
- Spot-check three sessions: plot the fitted `f_s` and `f_u` over the model-free points with the bands.

## Rebuilt

`Processed-Data/replay_content/models_convention/rate/` and `simulations/visit_rate.parquet`;
`paper/reported/behavior.json`, `reported_values.tex`, `questions.md`.

## Exit criteria

Simulations show model A's reward term null under the counting rule alone and unbiased under a planted
effect; the full-cohort fits are stored with records; the new values are in the behavior summary; the third
clock's fate is decided by the written rule and stated in the PR.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
