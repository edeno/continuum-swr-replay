# Phase 5a — Rate on the visit: three clocks, models A and D (Q1, Q2, Q7a)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#rate-design)

Branch: `visit-rate-models` from master after Phases 2c and 3 (the JAX fitter and the simulation engine
of Phase 2b are on master). CPU only.

**Inputs to read first:**

- The design, sections 2.1, 2.4, 2.5, 2.7, 3 (Q1, Q2, Q7), 4.1 to 4.7, and decision 3 (both forms of the
  reward effect).
- Phase 2's `analysis/regression/` (`terms.py`, `sessions.py`, `contrasts.py`, `collinearity.py`,
  `support.py`, `record.py`, `calibration.py`, `declared_tests.py`), Phase 2b's `simulation.py` and the rate
  model's first scenario list (`replay_models/rate/visit_rate_scenarios.py`), the fitter's interface
  ([designs.md#the-shared-fitter-built](designs.md#the-shared-fitter-built), `docs/design/regression_fitter.md`)
  and Phase 3's `analysis/replay_models/visit_design.py`
  (`visit_timebins(stop_table, events, port_experiences, *, kind, ...)`, on master since d14ff888).
- The family it replaces, to compare against: `analysis/replay_models/rate/per_bin_rate_ladder.py:98`
  (`fit_ladder_session`), `:254` (`fit_per_bin_rate_models`); `model_specs.py:879` (`MODEL_PER_BIN_RATE`);
  `reported/behavior/rate.py` (`per_bin_values`, `per_bin_detail_values`) and the `rate_*` keys of
  `paper/reported/behavior.json`; `docs/rate_models.md`.
- `src/continuum_swr_replay/analysis/replay_models/rate/rate_fit_statistics.py` and
  `analysis/replay_models/coefficient_statistics.py:135` (`coefficient_statistics`): how the current values
  are summarized (to know what changes).
- `src/continuum_swr_replay/reported/questions.py:320-358` (Part 3 questions), `reported/behavior/common.py:20-35`
  (`TAGS`), `reported/behavior/__init__.py:132` (`summarize`), `reported/values.py` (`value_factory`, `pend`).
- The design, section 9, seventeenth to twenty-first rounds and "Open after the twenty-first round" (a
  and b).
- Scratch: `S/proto_rate_realistic/` (model A on the real bins with planted curves shaped like the real
  ones; its `REPORT.md` when it is written), `S/proto_rate_smoothing/REPORT.md` and its follow-up section,
  `S/proto_rate_smoothing/firth_check/model_form_diagnostics.py` and `outcome_curves/` (diagnostics on the
  real bins, not results), `S/proto_rate/proto_core.py` (`simulate`, `calibrate_b0`, `bin_events`, `design_proposed`),
  `S/proto_rate/p2_m7r_try.py`, `p3_run.py`, `log_full_null.txt`, `S/verify_rate/s4_model_free_cells.py`,
  `s7_cells_tte.py`, `S/my_checks/three_clock_overlap.py`.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct),
[C7](shared-contracts.md#c7-the-fit-row-record), [C8](shared-contracts.md#c8-the-declared-family-of-tests),
[C9](shared-contracts.md#c9-where-new-fits-are-stored), [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

**Designs referenced:** [rate design](designs.md#rate-design), [smooth terms](designs.md#smooth-terms-and-bases),
[the shared fitter](designs.md#the-shared-fitter-built), [session to day to animal](designs.md#session-to-day-to-animal).

**Questions for the lead at this phase** (calls of the plan's author, not decided):

- An event that starts on a stop's first sample has no at-risk time before it, and the rate design does not
  count it (`visit_timebins`' `attrs["n_starts_not_at_risk_by_reason"]["stop_start"]`): about 0.2 % of the
  No SWR events and none of the SWR events, 28 events. The rate and content models would then differ by
  those events, against the lead's "rate and content models should analyze the same events" (C4). Ask
  before the first real fit whether that stands, or whether the content model leaves them out too; state
  the answer in `docs/regression_models.md`.
- **OPEN (lead): reward-term support** (design, "Open after the twenty-first round", a). The planned rule
  for the 0/1 term (5 visits with an event on each side, `support.binary_support`) fails in 86 % of the
  real SWR sessions and 42 % of the No SWR sessions (counts of the input tables, not results), over the
  20 % limit, so the term would leave the per-session model and Q2 would have no per-session answer. The
  author proposed to count visits with at-risk time on each side (all 242 sessions pass; a session with
  no event after an omission then has no finite maximum, 12.8 % of SWR sessions, and is reported with the
  term missing). Simulation with the plain fit: bias +0.05 to +0.10 on the log scale; the project's
  interval covered the planted value in 0.975 to 1.00 of data sets; the planned rule gave -0.23 at a
  threefold effect. The lead has not said yes in words. Ask before the first real fit.
- **OPEN (lead): rate model form** (b). Diagnostics on the real bins (pooled; not results): omission
  visits have a median of 0.9 s against 5.4 s; SWR events almost never start in the first second after
  arrival after either outcome (none in 93 % of sessions), so Q1's reference window `[0, 1] s` is empty
  for SWR events; the reward ratio at the same time since arrival is about 2 to 3 from 1 to 6 s and about
  1 beyond 10 s, so it is not constant; visits longer than 10 s are 705 of 44,435, ten sessions hold 72 %
  of that time, and it holds 56 % of the SWR events after an omission; the two clocks add from 2 s on but
  not between 1 and 2 s. Open: whether the model covers a declared time range; whether the reward effect
  may change with time since arrival; Q1's reference window; the value of each clock's penalty. The
  realistic simulation (scratch `proto_rate_realistic/`) is the evidence to bring. Ask before the first
  real fit; nothing of the model's form changes before the lead answers.

**Decisions this phase applies** (settled 2026-11-20): L1, the declared family of twelve estimands and 36
tests (C8), committed in `declared_tests.py` before the first real fit of this phase; L4, the event set (C3);
ASK-1, the ContFrag event sources; C14.

## Tasks

- **Baseline, before any code change**: copy `paper/reported/behavior.json` to the phase scratch directory;
  read and save the current per-bin rate values (the `rate_*` keys) and Phase 3's model-free clock curves
  (`visit_clock_curves.parquet`). These are the comparison, not targets.
- **Model specs** (`analysis/replay_models/rate/visit_rate.py`, new): the two models of
  [designs.md#rate-design](designs.md#rate-design), per kind (`swr`, `no_swr`), as term lists:
  - A: `Offset(log exposure) + Smooth(s, 0, 20) + Smooth(u, 0, 3) [+ Smooth(m, 0, 1.5)] + Slope(last_outcome_rewarded)` (a 0/1 column, not centered);
  - D: the same smooths + `LinearFunctional(lag_1..lag_10, log lags, 0, log 10)`.
  Each `Smooth` term and model D's functional is a P-spline at one declared fixed log penalty, the same
  in every session (design, seventeenth round). Each value is found once on this model's own real bins,
  per kind, by `calibration.log_penalty_for_edf` at a target median edf, checked by the scenario list
  below, and written into the module as a constant with the target, the commit and the rows it was found
  on. Do not reuse a target from another model or from `simulate_declared_penalty.py` (simulation: the
  same target gave different penalties and biases of opposite sign in two designs). The ranges stay.
  The fit is also reported at a stiffer and a looser value (one step of the target each way; call).
  The rule for the third clock is written in the module before any fit: fit A with `f_m`; if the
  concurvity `estimate` of `f_m` or `f_u` is at least 0.8 in more than 20 % of sessions, refit without `f_m`
  and report the time-until-movement curve only from the model-free curves (Figure 4c second half). Store
  both fits.
- **Cohort step** (in the same module): `visit_timebins` per session -> `select_visit_context` -> drop
  rows with a missing covariate (counted per kind) -> D additionally keeps bins with ten known lags
  (counted). Store `count_visit_context` and the drop counts in each record.
- **Fit** through `analysis/regression/sessions.py`: the plain penalized fit (no bias-reduced fit; lead,
  2026-11-23: "I don't want to use firth") of each session at the declared penalties; no penalty is
  chosen from the data, and no selection runs. The support rule for the 0/1 term as the lead answers
  `OPEN (lead): reward-term support`; sessions without a finite maximum reported with those terms missing
  (lead, 2026-11-21); collinearity measures; records.
- **Storage order of the design** (in the design builder, `visit_rate.py`): store each session's design in
  column (Fortran) order if a timing confirms it. A reviewer measured one weighted Gram product (`X' W X`, 82
  % of an evaluation's time in `docs/design/regression_fitter.md`, "Cost") at 7.0 ms with the design in column
  order against 9.8 ms in row order, on a synthetic session of 40,000 rows and 56 columns; we have not
  verified it. Timing step: on the smoke-test session, time 20 fits of model A with the design in each order
  (median, the same thread count), check that the coefficients agree to 1e-12 relative, and keep column order
  only if it is faster; state both timings in the PR.
- **The fold task from Phase 4** (its text sends it here): if `swr-trial-folds` is on master, make its
  `trial_folds` call `analysis/regression/folds.py`'s function and add the parity test
  (`tests/analysis/event_evidence/test_trial_folds_shared.py::test_same_folds`); otherwise leave it and say
  so in the PR.
- **Estimands** (`analysis/replay_models/rate/visit_rate_statistics.py`, new), each per session with its
  exact variance, then `shared_statistics.estimand_summary` per kind and for the difference:
  - Q1: mean of `f_s` over `[2, 4] s` minus mean over `[0, 1] s` (log rate ratio);
  - Q2: the coefficient of `last_outcome_rewarded` in model A (log rate ratio), the reward effect at fixed
    time before leaving;
  - Q7a: mean of `f_u` over `[0, 0.25] s` minus mean over `[1, 2] s`;
  - model D: the curve over lags 1 to 10 with its band (described, not tested).
- **A scenario list for the simulation package (Phase 2b)** (`replay_models/rate/visit_rate_scenarios.py`,
  extended; run by Phase 2b's runner, which replaces only the outcome column of the real bins and runs the
  real fit, tests and reduction). First the realistic scenarios of `S/proto_rate_realistic/`, with curves
  shaped like the real ones: no reward effect; a constant effect; a fading effect (the ratio falling to 1
  with time since arrival); an effect taken from the data (the model-free ratio by time since arrival);
  an effect that differs between sessions and between animals; a later onset after an omission than after
  a reward. Each is run at the declared penalties and at the stiffer and looser values, on all bins and
  on a declared time range if `OPEN (lead): rate model form` asks for one. Then the scenarios of the
  earlier plan: event starts drawn on the real visits' at-risk time of every session from a
  known rate `exp(b0 + f_s + f_u + beta * rewarded)`, with the event definition's counting rule (a candidate
  that starts in a stop and runs into movement is not an event, at every stop end, small movements and the
  last one; an event that crosses the final poke-out while immobile is counted, as the event set has it). Scenarios: (a) no reward effect; (b) rate ratio 1.3;
  (c) a sharp lag-1 effect only, to see what D returns; (d) a drop on `u` in the last 0.3 s with no reward
  effect, which must not move to `s` or to the reward term; (e) a drop before small movements only (`f_m`)
  with flat `f_u`, which the model must separate or the rule above must catch. Twenty data sets each over
  all sessions (more where the realistic prototype shows the spread needs it); outputs to
  `models_convention/simulations/visit_rate.parquet`.
- **ContFrag event sources** (ASK-1, lead: "Move them"): the same models A and D fit to the SWR-sourced and
  MUA-sourced ContFrag events of the rebuilt ContFrag tables, as two more event sources (`event_source` in
  the fit-row record; each source is one stream, with no SWR and No SWR split, call), same bins, same selection, same penalty rule. Their values are a supplementary
  comparison (Phase 7 builds the comparison values); they are not in the declared family.
- **Runner** `scripts/replay_models/rate/fit_visit_rate_models.py` (new): reads the tables, fits both
  models, both kinds and the three event sources (non-local, ContFrag SWR, ContFrag MUA), writes `models_convention/rate/{A,D}.parquet` and a manifest; registered in
  `scripts/workflows/run_refits.py` (`RUNS`, `:152`) as `rate.visit.A` and `rate.visit.D`.
- **Questions**: add Q1 to Q7 of the design's section 3 as new Part 3 questions in
  `reported/questions.py` (call: ids `3.11` to `3.17`, wording from the design's table); add their tags to
  `reported/behavior/common.py` (`TAGS`).
- **Summary values** (`reported/behavior/visit_rate.py`, new; called from `reported/behavior/__init__.py`):
  per kind and difference, the estimate as a rate ratio, its interval, animals with the sign, day-level p,
  sessions kept and left out (support, convergence, collinearity), cohort counts per label. No q-values
  yet (Phase 9). Pending notes follow C10.
- **Docs**: create `docs/regression_models.md` (new; the one doc for the convention's models, linked from
  `docs/README.md` in place of `rate_models.md` once that is gone) with the visit rate model (unit, clocks,
  A and D, the rule for the third clock, what is tested and what is described), written with value markers,
  and its model table. Later family phases add their sections to it.

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-5a](legacy-inventory.md#phase-5a)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. First commit the fixture of C13 for Phase 1's check 1 (the `fit_per_bin_glm` coefficients of
     `M7r_time_histcos_rew` on the synthetic design of `tests/analysis/regression/test_no_penalty_equals_current.py`),
     switch the test to it, then remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- The per-stop (total) reward effect, the SWR-contrast covariates, the uncertainty rate model (Phase 5b).
- FDR over the family and figure bands (Phase 9), Figure 4 (Phase 10).
- Removing what other families still call (`per_bin_design.py`, `PerBinRateModelSpec`, `_fit_diagnostics.py`,
  `coefficient_statistics.py`, the cohort selectors): each leaves with its last caller (the inventory names
  the phase). The per-bin ladder's stored fits stay on disk, unread.
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
- Smoke test on one session: build, fit A and D for both kinds at the declared penalties, and time it
  (the design measured about 0.08 s per two-clock fit with the NumPy fitter; time it again on the JAX
  fitter). There is no penalty selection: the cost is one fit per session, model and kind, plus the
  calibration of each declared penalty (about 18 passes of `fit_sessions` per term) and the scenario
  list's data sets. Time one calibration on the five sessions. Parallelize over sessions with a `spawn`
  process pool only if the measured run exceeds about 30 minutes (C11); run in `tmux` (`csr-5a-rate`).
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

The lead has answered questions a and b, and the model's form and the reward term's support rule follow
the answers; the scenario list shows model A's reward term null under the counting rule alone and unbiased
under a planted effect (on the realistic scenarios too), at the declared penalties; the full-cohort fits are stored with records; the new values are in the behavior summary; the third
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
