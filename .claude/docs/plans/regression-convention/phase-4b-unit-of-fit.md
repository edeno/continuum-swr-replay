# Phase 4b — The unit of the fit: session, day or animal, one setting for the behavior models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design note](../../../../docs/design/2026-11-24-unit-of-fit.md)

Branch: `unit-of-fit` from master, after the rate branch (`visit-rate-validation`) and the content branch
(`role-content-models`) have each taken master and been merged (see "Order of work"). CPU only. No real
fit is run in this phase.

**Status:** tasks 1, 2, 3, 5 and 7 started 2026-11-24 on branch `unit-of-fit` (see "Amendments,
2026-11-24" at the end, which holds where it and a task above disagree). The what and the why are in the design note,
`docs/design/2026-11-24-unit-of-fit.md` (read all of it first). This file is the work.

**Decided by the lead (2026-11-24):** "I'd like to be consistent with what we do with the behavior
models" / "By behavior models, I mean rate, content, etc" (02:09, 02:10); "Does not include the mpfc models
for now." (02:14); "I'm inclined to do C and then attempt D as we are doing C, but we will see if it is
prohibitive" (03:13; C is one fit per animal with a baseline for each session, D one model with random
session effects and REML penalties); "Maybe we make it easy in the code to do day or animal grouping"
(03:33). On the penalties: "Should we keep the fixed penalty?" / "Ok decide then" (03:18, 03:20): the
author decides after the running simulations report. After the simulations: "I suspect with will end up
doing design C for all. And then running both the day and animal model and compare" (09:47); "So maybe
design C with day and animal model and then design D later to check." (09:49); "I did say maybe, lets
just make this the decision now and we can change if the evidence is against it" (09:55). So the lead
decided: design C with the day and the animal model, both run and compared; design D later as a check;
changeable if the evidence is against it; consistent across the behavior models; not the mPFC models for
now. Still in force from 2026-11-23: no Firth fit; a regression model; the weak penalty on every simple
tested coefficient; simulations use the real code.

**Not the lead's words; labeled as the author's** (the design note, 11.1):

- *Fixed, declared penalties*: the author's decision, on "Should we keep the fixed penalty?" / "Ok
  decide then" (03:18, 03:20), from the evidence in the design note, 11.2.
- *The curves are among the terms shared inside a unit*: the author's reading. The author's message to
  the lead just before 09:55 said that design C shares the curves inside a unit, that the lead had
  declined "one curve for each animal" on 2026-11-21 for the per-session design, and that the author
  read the lead's messages of that morning as changing that for design C; the lead's next message was
  the 09:55 one. The rate model's clock curves are the point the running scenario may change (design
  note, 11.4).
- *Which models*: the lead's words are "rate, content, etc" (02:10) and "Does not include the mpfc
  models for now." (02:14); that "etc" means the choice, stay-or-leave and ripple-power models is the
  author's reading (stated to the lead several times, not contradicted).

**The gate stays:** simulation results go to the lead before any real fit at a new unit.

## Open questions for the lead

Each is OPEN. The phase builds what is needed to answer it; it does not answer it.

1. **Rate: are the clock curves shared inside a unit, or per session with only the tested coefficients
   (the reward coefficient) shared?** Awaited: the new scenario with session-specific curves (task 10),
   fitted in both forms at a day and an animal, Q1, Q2 and Q7a against both targets. The design's
   fourteenth round records that one curve per animal was declined on 2026-11-21; the lead's
   2026-11-24 words reopen the unit, not, in the author's reading, that point.
2. **Content: are the arm effects shared inside a unit or per session?** Awaited: the running scratch
   simulation `S/content_validation/unit_of_fit/` (forms "S" and "B" at a day and an animal, with arm
   effects that differ by session); then its repeat with the real code (validation slice).
3. **Penalties under a larger unit: fixed and recalibrated on the unit (to which target edf), or REML?**
   The lead left this to the author ("Ok decide then"). Awaited: the unit run's "pS" against "pU" arms;
   the design D trials (`S/design_d_trial/rate/`, `S/design_d_trial/content/`). The author reports the
   choice and its evidence to the lead before task 6's losing path is removed.
4. *(Answered, 09:55: both units. The behavior models are fitted at the day and the animal unit and the
   two are compared.)* **The declared value of `fit_by`: `"day"` or `"animal"`** (the lead's 03:31
   question, "And grouping by day wouldn't work?", and 03:33).
5. **Which target the paper states when sessions differ**: the plain mean of session effects (today's)
   or the unit's information-weighted effect. Awaited: both targets, reported side by side, under the
   scenarios that plant session differences (task 7, task 10).
6. **Whether design D follows, and for which models.** Awaited: the design D trials and their cost.
7. **The rate branch's per-animal profile fit** (`visit_rate.fit_animal_rewards`): removed, or moved into
   the fitter as the path for per-session curves. Follows from question 1 (task 12). The branch's own
   design round already lists whether it stays as a question for the lead.
8. **A session without support for its baseline**: its rows dropped from the unit (the author's default,
   as a failing level is dropped today), or a baseline shared with the unit's other such sessions (the
   scratch driver's choice). Awaited: the counts of such sessions at each unit (task 3) and the slice.
9. **The two forms of interval at the animal unit** (recorded, not changed): a scalar uses the
   closed-form t interval over the animals (`unit_summary`), a curve the bootstrap band with the animal
   as its own day (`bands.across_session_band`). They agree only approximately.
10. **A session without rows of a baseline category's reference level** (recorded, not changed): today's
   rule holds, so the unit's first fit is an error that stops the loop. At the animal unit one such
   session stops an animal's fit.

## Inputs to read first

- The design note (all of it). The convention: `docs/design/2026-11-20-regression-convention-and-figure04.md`,
  sections 4.3, 4.5, 7 (stage 3) and section 9, rounds thirteen, fourteen, seventeen and eighteen to
  twenty-three.
- `docs/design/regression_fitter.md`, sections "The support rule", "The per-session loop", "The fit
  record", "From session to day to animal", "Simulation", "Cost".
- The code in the inventory below, master first, then the two branches.
- `S` = `/cumulus/edeno/continuum-swr-replay/.claude/research/2026-11-20-regression-prototypes/`:
  `rate_validation/REPORT.md` (sections 1, 3, 4, 6, 8), `content_validation/realistic_noise/REPORT.md`,
  `content_validation/unit_of_fit/driver.py` (how a larger unit was emulated: form S concatenates a unit's
  sessions as one "session"; form B relabels the arm term per (session, arm) and builds one design per
  unit), `content_validation/unit_of_fit/design_units.json` (counts of the design),
  `content_validation/unit_of_fit/launch.log` (timings), `design_d_trial/gpu_size/run.log`.

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session), [C12](shared-contracts.md#c12-layering-of-the-new-package),
[C13](shared-contracts.md#c13-no-legacy-left-behind), [C14](shared-contracts.md#c14-one-of-everything).
This phase changes C5 (inference) and C7 (the record); see the last section.

## Inventory: where the session is the unit today

Master is 2e0ea0d0. Branch lines are at `visit-rate-validation` 5d97fd98 and `role-content-models`
d758a17e; both branches start from master 7959aca7, before the JAX fitter.

### Master: the fitter (`src/continuum_swr_replay/analysis/regression/`)

| Place | What it does | What changes |
|---|---|---|
| `sessions.py:92` `SessionKey` | the key of a fit, `(animal, session_date, epoch)` | a fit's key becomes the unit key (`tables.session_keys.unit_key`); the sessions stay the input keys |
| `sessions.py:126-151` `_Session.__init__` | one fit's state; `:142` factorizes the support unit on the rows given; `:136-137` sorts by unit under the softmax | the support unit is factorized on (session, unit) when the fit holds more than one session (task 2); the sort keeps rows of one event together |
| `sessions.py:334-341` | the first-fit error names a category's reference level "in every session" | the message names the unit |
| `sessions.py:440-458` `_fit_session` | one session through the steps | one unit through the same steps; renamed `_fit_unit` (no alias, C13) |
| `sessions.py:461-523` `_record` | the record's identity from the key (`:495`) | identity from the unit key; adds `fit_by` and the per-session counts (task 2) |
| `sessions.py:526-586` `fit_sessions` | the loop over sessions | takes `fit_by`, groups the sessions given into units, fits each (task 2). At `"session"` the records equal today's |
| `record.py:103-203` `FitRecord` | one row per (model, kind, session); the session at `:180-182` | adds `fit_by` and `sessions` (task 2); `read_records` (`:248-250`) refuses a file without them (C13: old fits are not read) |
| `support.py:1-53` | the rule "per column and session"; `:81` `MIN_UNITS` | counted per fit as today; a per-session baseline copy is counted on its session's rows (task 3) |
| `support.py:305-331` `terms_failing_share` | the share of sessions in which a term fails | the share of fits (units); docstring and argument names (task 4) |
| `calibration.py:73-100` `_term_edf`, `:103-214` `log_penalty_for_edf` | the median edf over sessions | the median over the fits at the given `fit_by` (task 4) |
| `terms.py:15-18` | a reference level must have rows in every session | in every fit; a per-session baseline has its own reference handling (task 3) |
| `terms.py:30-35`, `:52-58`, `:683-700` `_covariate_scales` | knots, constraints and standardizations from the pooled rows of every session given | nothing: the scale rows are already pooled over the model's rows, so they are the same at every unit |
| `terms.py:985-1075` `build_design` | one design for all fits | marks the model's baseline terms (task 3) |
| `simulation.py:1-63`, `:583-614` `_session_rows`, `:617-665` `_data_set_row`, `:668-703` `_run_task`, `:879-968` `summarize` | draws on each session, fits each session, reduces with `estimand_summary` | the model's `fit_by`; one row per unit; the reduction per unit; both planted targets (task 7) |
| `bands.py:210` `across_session_band` | a curve per session, animal-balanced | at `"day"`, a day is passed as its own session; at `"animal"`, a curve per animal (task 5 tests both) |
| `reml.py:1883` `select_penalties` | REML over the given sessions' blocks | the REML option of task 6 calls it on one unit |
| `folds.py` | trial folds within a session | nothing: no behavior model reads it (used by `analysis/event_evidence/`) |

### Master: foundation and statistics

| Place | What it does | What changes |
|---|---|---|
| `tables/session_keys.py:27` `SESSION_COLUMNS`, `:30` `SESSION_COLUMNS_DAY` | the session key | adds `FitBy`, `FIT_BY_COLUMNS`, `unit_key` (task 1) |
| `tables/session_keys.py:66` `subset_max_per_animal` | smoke subsets by session | nothing |
| `analysis/shared_statistics.py:1013-1097` `estimand_summary` | session to day to animal, bootstrap, day test | nothing: a day value is passed as a day with one session |
| `analysis/shared_statistics.py:978` `day_level_test`, `:928` `animal_day_session_values`, `:621` `bootstrap_interval` | the day test, the means, the interval | nothing |
| (rate branch) `analysis/shared_statistics.py:679` `animal_mean_interval` | mean over animals, t interval | kept; the one animal-level reduction (task 5) |

### Master: the models

| Place | What it does | What changes |
|---|---|---|
| `analysis/replay_models/declared_tests.py:1-10` | "its variance in a session is exact", "the difference of the two kinds' session estimates" | docstring: "in a fit", "unit estimates". No contrast changes; a test that no declared contrast reads a baseline column (task 3) |
| `analysis/replay_models/rate/visit_rate.py:3` | "One fit for each session and each kind" | one fit for each unit and kind (task 8) |
| `visit_rate.py:160` `UNIT = "well_visit_id"` | a port experience number from 0 in each session (`data_loaders/trials.py:595`) | nothing in the model; the fitter keys it by session (task 2) |
| `visit_rate.py:232-246` `CLOCK_LOG_PENALTIES` | provisional per-session penalties | a real fit at a unit needs penalties calibrated at that unit; `CalibratedPenalty` (`:287`) records its `fit_by` (task 8) |
| `visit_rate.py:313` `VisitRateSettings` | the model's settings | gains `fit_by` (default `BEHAVIOR_FIT_BY`) and, until question 1 is decided, `clock_curves` (`"shared"` or `"per_session"`) (task 8) |
| `visit_rate.py:443` `model_terms`, `:505` `build_visit_rate_design` | the terms; the intercept from `build_design` | the intercept is the declared baseline (task 3, 8) |
| `visit_rate.py:689` `_lag_curve_identified`, `:705` `model_d_rows` | model D drops a session whose outcomes cannot identify the lag curve | judged on the unit's rows (task 8) |
| `visit_rate.py:882-955` `fit_visit_rate` | per-session fit; `:951-954` adds each session's cohort counts | passes `fit_by`; the cohort counts go to each session's entry of the unit's record (task 8) |
| `visit_rate.py:958` `fit_visit_rate_session` | one session alone | nothing (a unit of one session) |
| `visit_rate.py:979` `calibrate_log_penalty` | median edf over sessions | over units (task 4, 8) |
| `visit_rate.py:1043` `third_clock_decision` | share of sessions | share of units (task 4, 8) |
| `rate/visit_rate_statistics.py:70` `session_estimands`, `:113` `_summary`, `:127` `summarize_estimands`, `:173` `lag_curves`, `:222` `lag_curve_band` | per session, then `estimand_summary` | `unit_estimands` and the reduction of task 5 (renamed, no alias) |
| `rate/visit_rate_scenarios.py:291` `poisson_projection` | a hand-built per-session target | the session target stays; the unit target comes from the engine (task 7) |
| `rate/visit_rate_scenarios.py:501` `session_log_ratios`, `:538` `differing_ratio` | reward effects that differ by session and animal | kept; reported against both targets (task 10) |
| `scripts/checks/run_simulations.py` | the runner | nothing beyond the model's config carrying `fit_by` |

### Rate branch (`visit-rate-validation`, not on master)

| Place | What it does | What changes |
|---|---|---|
| `visit_rate.py:320` `REWARD_LEVELS`; `:1092` `AnimalReward`; `:1134` `AnimalRewardFit`; `:1167` `_profile_terms`; `:1196`, `:1208` the side checks; `:1218` `_fit_animal`; `:1273` `fit_animal_rewards` | one unpenalized reward coefficient per animal, each session's baseline and curves its own, by Newton steps on the summed profile score | question 7, task 12 |
| `rate/visit_rate_statistics.py:187` `animal_reward_values`, `:214` `summarize_animal_rewards` | the animal-level Q2 | replaced by task 5's reduction at `fit_by="animal"` (task 12) |
| `rate/visit_rate_simulation.py:670` `_animal_truth`, `:718` `animal_level_fit`, `:820` `_models` (the `reward_level="animal"` model) | the animal check in the simulation | replaced by a model at `fit_by="animal"` with the engine's unit target (task 7, 12) |
| `rate/visit_rate_scenarios.py:403` `shared_ratio_projection` | the animal target, by hand | replaced by the engine's unit target (task 7) |
| `regression/simulation.py:144` `UNIT_COLUMNS`, `:162` `OwnFit`, `:753` `_animal_data_set_row`, `:786` `_own_units`, `:817` `_own_fit_rows`, `:1085` `summarize` (no day test for animal units) | a model's own fit, with animal units detected by a missing `session_date` | the animal-unit branch is replaced by the engine's own unit handling (task 7). **Call**: `OwnFit` stays for the joint run of both kinds (`visit_rate_simulation.py:1337` `joint_fit`), which needs each unit's SWR minus No SWR difference |

### Content branch (`role-content-models`, not on master)

| Place | What it does | What changes |
|---|---|---|
| `content/role_content.py:340` `SESSION_COLUMNS` | a second copy of the session columns | removed; imports `tables.session_keys.SESSION_COLUMNS` (C14) (task 9) |
| `role_content.py:662` `session_rows` | the cohort rows by session | nothing (the fitter groups) |
| `role_content.py:746-787` `role_terms`; `:761` the arm `Category` | the arm effects, unpenalized, reference arm 0 | the arm term is the declared baseline; per session or shared (question 2, task 9) |
| `role_content.py:1032-1146` `fit_role_content`; `:1132` collinearity per session; `:1199` `tested_coefficient_collinearity` | per-session fit and collinearity | per unit (task 9) |
| `role_content.py:1149` `fit_role_content_session` | one session alone | nothing |
| `content/role_content_statistics.py:61` `session_estimands`, `:106` `penalized_edf`, `:160` `penalized_edf_summary`, `:193` `estimand_summaries` | per session, then `estimand_summary` | per unit (task 9) |
| `content/role_content_scenarios.py:350` `planted_target` | the present target | kept as target 1 (task 7) |
| `role_content_scenarios.py:382` `arm_effects`; `:761` `spread_truths` | one set of arm effects; effects that differ by animal and session | a scenario with per-session arm deviations (task 10) |
| `content/role_content_simulation.py:327` `engine_model`, `:447` `build_simulation`, `:561` `calibrate_smooth_penalties` | the engine models; the calibration over sessions | `fit_by`; calibration over units (task 9) |

The scratch driver (`S/content_validation/unit_of_fit/driver.py`) is not in the repository. Its form B
built one design per unit to dodge the reference-level rule; task 3 makes that unneeded.

## Order of work

1. **The rate branch takes master** (it holds the JAX fitter, 2e0ea0d0) and merges after its own
   phase-boundary review. It brings `animal_mean_interval`, the engine's own-fit path and the animal
   check (`reward_level="animal"`) as they are. Its simulations are running: it merges master only after
   they finish (C11).
2. **The content branch the same**, after its simulations finish.
3. **This phase branches from master after both.** If one has not merged, tasks 1 to 7 (task-free) go
   ahead, and the model tasks (8 to 12) wait for it. Nothing here is built on an unmerged branch.
4. **What this phase replaces and removes** (C13):
   - the engine's animal-unit branch of the own fit, `_animal_data_set_row`, and the reduction it holds
     (task 7);
   - `shared_ratio_projection` and `_animal_truth` (task 7);
   - `summarize_animal_rewards`, `animal_reward_values`, `animal_level_fit` (task 12);
   - `REWARD_LEVELS`, `fit_animal_rewards` and its helpers: removed if question 1 gives shared curves;
     moved into `analysis/regression/` and generalized if it gives per-session curves (task 12). Until the
     answer they stay, as the check, because they are the only per-session-curve fit that fits in memory
     for an animal (design note, section 8);
   - `role_content.SESSION_COLUMNS` (task 9);
   - the losing penalty path of task 6, after the author's decision (question 3);
   - the losing form of questions 1 and 2 (`clock_curves`, the arm-baseline choice), after the lead's
     answer. Until then both forms are named settings, as the rate model's open settings are today.

## Tasks

Each task is one implementer's PR-sized piece. Its gating tests are written first and fail first.

1. **The setting and the key.** In `tables/session_keys.py`: `FitBy = Literal["session", "day",
   "animal"]`, `FIT_BY_COLUMNS` (`"session"`: `SESSION_COLUMNS`; `"day"`: `("animal", "session_date")`;
   `"animal"`: `("animal",)`) and `unit_key(session_key, fit_by)`. In `analysis/replay_models/fit_by.py`
   (new): `BEHAVIOR_FIT_BY: tuple[FitBy, ...] = ("day", "animal")` (the units every behavior model is
   fitted at, in that order; question 4), with the decision record in its docstring. `"session"` stays a
   value of `FitBy` and the default of `fit_sessions`, with today's behavior exactly. Tests:
   `tests/tables/test_session_keys.py::test_unit_key_of_each_fit_by`, `::test_unknown_fit_by_refused`,
   `tests/analysis/replay_models/test_fit_by.py::test_behavior_units_are_day_and_animal`.
2. **The grouping key through the fitter's loop and its record.** `fit_sessions(..., fit_by="session")`
   groups the sessions given by `unit_key`, in the order of each unit's first session, concatenates each
   unit's rows (each row keeping its session key) and runs one fit per unit. The support unit is
   factorized on (session, unit). `FitRecord` gains `fit_by: str` and `sessions: list` (JSON: per session
   in the fit, its key, `n_rows`, `n_units`, `n_events` and the rows dropped, by cause); at `"day"`
   `epoch` is `""`, at `"animal"` `session_date` and `epoch` are `""` (**call**). Tests (in
   `tests/analysis/regression/test_sessions.py` unless named):
   - `test_fit_by_session_equals_today` (the records at `"session"` equal those of the fitter before the
     change, field by field, on the fixtures of `tests/analysis/regression/data/numpy_reference/`);
   - `test_visits_of_two_sessions_are_two_units` (two sessions with the same visit ids: the support counts
     at `"day"` are the sum of the two sessions' counts);
   - `test_day_fit_equals_one_concatenated_session` (at `"day"`, with no baseline term, one fit equals
     `fit_sessions` of the concatenated rows given as one session);
   - `test_record_holds_its_sessions` (`sessions` lists each session with its counts);
   - `tests/analysis/regression/test_record.py::test_round_trip_with_fit_by` and
     `::test_old_file_refused`.
3. **A baseline for each session.** A model declares its baseline terms (`build_design(...,
   baseline=(...))`: the intercept for the rate model, the arm category for content). In a fit of k
   sessions each baseline term is replaced by k copies, each zero outside its session's rows; the
   intercept becomes k session indicators; the arm category keeps its reference arm in each copy. A
   copy's coefficient is named `<name>@<animal>/<session_date>/<epoch>`. A baseline carries no
   penalty and no term is fitted `by` a baseline category (`build_design` refuses both; amended
   2026-11-24; a penalized baseline, as design D needs, adds one penalty block over a term's copies
   then), is never tested, and counted by the support rule on its session's rows; a session whose copy fails has its rows
   dropped from the unit and counted (question 8). In a fit of one session nothing changes. Tests:
   - `test_terms.py::test_baseline_copies_per_session` (names, columns, zeros outside the session);
   - `test_sessions.py::test_rate_baselines_match_separate_intercepts` (Poisson: k sessions sharing only a
     slope give each session's intercept as a separate fit with that slope fixed would, to 1e-8);
   - `test_sessions.py::test_session_without_support_dropped_and_counted`;
   - `test_sessions.py::test_reference_level_per_session_copy` (a session whose rows lack another
     session's levels fits);
   - `tests/analysis/replay_models/test_declared_tests.py::test_no_contrast_reads_a_baseline`.
4. **The shares and the calibration over units.** `terms_failing_share` and the rate model's
   `third_clock_decision` count units; `log_penalty_for_edf` takes `fit_by` and takes the median over
   units. Tests: `test_support.py::test_failing_share_over_units`, `test_calibration.py::test_median_over_units`
   (at `"animal"` with five animals, the median of five edfs; the bracket rule still raises when the
   median jumps).
5. **The reduction per unit.** One function, `regression.units.unit_summary(values, fit_by, *, n_draws,
   seed)` (first built in `shared_statistics`, moved into the fitter's package on 2026-11-24 because it
   knows the units): `"session"` is `estimand_summary`; `"day"` is `estimand_summary` with each day as its own
   session; `"animal"` is `animal_mean_interval`, with `day_p` NaN. Its output has the same keys at every
   unit. The kind difference is taken unit by unit. The bands follow the same rule (`across_session_band`
   with a day or an animal per row). Tests (`tests/analysis/regression/test_units.py`):
   `test_unit_summary_session_is_estimand_summary`, `test_unit_summary_day_one_session_per_day`,
   `test_unit_summary_animal_is_t_interval` (hand-checked on five values),
   `test_unit_summary_same_keys_at_every_unit`; `tests/analysis/regression/test_bands.py::test_band_over_animals`.
6. **Penalties per unit.** A real fit at a unit reads penalties calibrated at that unit
   (`CalibratedPenalty` records `fit_by` and the target; a fit refuses a penalty calibrated at another
   unit). Until question 3 is decided, a model's settings may instead ask for REML on its smooth blocks
   within each unit (`penalty_rule = "fixed" | "reml"`, `reml.select_penalties` on the unit's data), so
   that the simulations can compare. The weak penalty on simple tested coefficients is never selected.
   After the decision the losing path is removed in this phase (C13). Tests:
   `test_visit_rate.py::test_penalty_of_another_unit_refused`, `test_sessions.py::test_reml_rule_selects_only_smooth_blocks`.
7. **The simulation engine per unit, with both targets.** `Model` gains `fit_by` (from the model's
   settings). The engine fits each data set at that unit, writes one `sessions`-table row per unit (with
   `fit_by`, its sessions and events) and reduces with `unit_summary`. Each data set row carries two
   planted values: `planted` (named `planted_session_mean` here before; the code keeps the name
   `planted`, which the stored fixture has; today's target: the session -> day -> animal mean of the
   sessions' planted values) and `planted_unit` (the unit model fitted by the real fitter to the expected
   outcome, without noise and with the weak penalty off, reduced as the estimates are; design note,
   section 6). The summary gives bias and coverage against both. The animal-unit branch of the own fit
   and `shared_ratio_projection` are removed. Tests (`tests/analysis/regression/test_simulation_engine.py`):
   `test_same_seed_same_data_at_every_unit` (the drawn outcomes do not depend on `fit_by`),
   `test_unit_target_equals_planted_when_sessions_agree` (an effect the same in every session: the two
   targets equal the planted value), `test_two_targets_differ_when_information_differs` (a tiny cohort
   with two sessions of unequal size and unequal effects), `test_unit_rows_and_counts`,
   `test_animal_unit_has_no_day_test`.
8. **The rate model reads the setting.** `VisitRateSettings.fit_by` and `clock_curves`; the intercept is
   the baseline; at `clock_curves="per_session"` the clock smooths are baseline terms too (dense; for an
   animal this exceeds memory, see task 12); model D's identifiability is judged on the unit's rows; the
   statistics module maps each contrast onto a fit's coefficients by name (`terms.Design.contrast_on`, which also refuses a contrast that reads a baseline), reduces with `unit_summary`, and checks `record.fit_by` (it refuses a record of a larger unit
   until then: `visit_rate_statistics._session_records`); the model builder declares its baseline from
   `declared_tests.BASELINES`. Tests (`tests/analysis/replay_models/rate/`):
   `test_visit_rate.py::test_fit_by_from_the_one_setting` (the default is `BEHAVIOR_FIT_BY`),
   `::test_day_fit_has_one_intercept_per_session`, `::test_model_d_identified_on_the_unit`,
   `test_visit_rate_statistics.py::test_unit_estimands_at_each_unit`.
9. **The content model reads the setting.** `RoleContentSettings.fit_by` and `arm_effects`
   (`"per_session"` or `"shared"`, question 2); the arm category is the baseline when per session; the
   collinearity and the statistics per unit; the calibration of the smooths over units;
   `role_content.SESSION_COLUMNS` removed. As in task 8: the statistics map each contrast by
   coefficient name, reduce with `unit_summary` and check `record.fit_by`, and the builder declares
   the baseline from `declared_tests.BASELINES`. Nothing reads `BASELINES` or `BEHAVIOR_FIT_BY` before
   tasks 8 and 9; that is expected. Tests (`tests/analysis/replay_models/content/`):
   `test_role_content.py::test_fit_by_from_the_one_setting`, `::test_arm_effects_per_session_and_shared`,
   `::test_collinearity_per_unit`, `test_role_content_statistics.py::test_unit_estimands_at_each_unit`.
10. **Scenarios that plant session-to-session differences.** Rate: clock curves that differ by session (a
    shift and a scale of the arrival and leaving curves per session, drawn once with a fixed seed), with
    the share of rewarded visits left as the rows give it; the existing `differing_ratio` kept. Content:
    arm effects that differ by session (a Normal deviation per session and arm, as the scratch driver
    planted with SD 0.3); the existing `spread_truths` kept. Each reports both targets. Tests:
    `test_visit_rate_scenarios.py::test_differing_curves_differ_by_session`,
    `test_role_content_scenarios.py::test_arm_deviations_per_session`, each also checking the seed fixes
    the draw.
11. **The leave-one-session-out report.** A task-free function in `analysis/regression/` (new module
    `influence.py`, `leave_one_session_out(design, sessions, lik, *, fit_by, ...)`) refits each unit once
    per session left out, with the same design and penalties. A model's statistics module turns it into
    one row per unit and estimand: the full value, the largest change, the session that made it, and the
    numbers of sessions and events behind the value. Tests: `tests/analysis/regression/test_influence.py::
    test_identical_sessions_give_no_change`, `::test_one_outlying_session_is_named`, `::test_counts_per_unit`.
12. **The rate branch's per-animal fit, after question 1.** Shared curves: remove `REWARD_LEVELS`,
    `fit_animal_rewards` and every helper of the inventory, and their tests (C13). Per-session curves:
    move the profile step into `analysis/regression/` as the path for a fit whose shared columns are few
    and whose per-session columns are many (the summed profile score and information, generalized from
    one coefficient to a vector, with the weak penalty on the shared tested coefficients), and remove the
    rate-specific copy. Tests for the second case: `test_sessions.py::test_profile_path_equals_dense_fit`
    (on a day small enough for the dense design, the two give the same coefficients to 1e-8).
13. **Docs.** `docs/design/regression_fitter.md`: "The per-session loop" becomes "The loop over fits",
    with the unit of the fit; "The fit record" gains `fit_by` and `sessions`; "From session to day to
    animal" gains the table of the design note, section 7; "Simulation" gains the two targets. One row
    per removal in `docs/history/2026-module-moves.md`; `docs/regression_pipeline_map.md` regenerated
    (`scripts/checks/write_pipeline_map.py`).

## Tests that gate it

| Task | Gate |
|---|---|
| 1 | `test_session_keys.py` as above |
| 2 | `test_fit_by_session_equals_today` on the stored fixtures; the support-unit test; `test_record.py`; all of `tests/analysis/regression` |
| 3 | the baseline tests; `test_no_contrast_reads_a_baseline` |
| 4 | the share and calibration tests |
| 5 | the four `unit_summary` tests and the band test |
| 6 | the refusal test; the REML-rule test |
| 7 | the five engine tests; the Phase 2b fixed-seed fixture still reproduces at `fit_by="session"` |
| 8, 9 | the model tests; each model's existing suite at `fit_by="session"` unchanged |
| 10 | the scenario tests |
| 11 | the influence tests |
| 12 | removal: `git grep -n -w` gives no match for each removed name (C13); or the profile-path test |
| all | `make check`; the American-spelling test; the import-layering test (C12: `analysis/regression/` imports only the foundation) |

## Deliberately not in this phase

- Design D (random session effects with REML strengths, the factor smooth, the chunked or per-session
  Gram on one GPU). The baseline copies of task 3 are the blocks it will penalize; nothing else of D is
  built.
- The mPFC models (lead, 02:14).
- The choice, stay-or-leave and ripple-power models: they read `BEHAVIOR_FIT_BY` when Phase 7 builds them.
- Any decision on the open questions. The phase builds both forms where a question is open.
- Any real fit at a day or an animal, and any change to `paper/reported/`.
- A change to the fitter's Newton step or memory use.

## Validation slice

Every simulation below runs with the real code at the unit named, after the phase's tests pass and
before any real fit. The results go to the lead. Costs are rough: they come from the timings in the
reports (CPU-seconds unless marked) and are to be measured on a smoke test first (C11: one data set,
`/usr/bin/time -v`, in `tmux` as `csr-4b-<what>` when over five minutes).

| Model | Scenarios | Units and forms | Rough cost, from |
|---|---|---|---|
| rate, both kinds | no effect, constant ratio, differing ratio, differing curves (new), later onset, drop before leaving | session; day and animal, each with shared and per-session curves; penalties pS, pU (and REML while question 3 is open) | the per-session run of 8 SWR scenarios took 1 h 47 min on 12 processes for 40 sessions and 5 models; the animal profile model took 47 s per data set on 10 sessions against 9 s for a session-level model (`rate_validation/REPORT.md`, section 8). A dense animal fit with shared curves is about 2.3 million rows by 150 columns (design note, section 8): to be timed on the largest animal (j16) first |
| content, both kinds, both models | the plain model's no effect, small and large current-arm level, large leaving curve, truth from the data, spread; the uncertainty model's no effect, belief variance with recency, recency only, small recency curve; arm effects differing by session (new) | session; day and animal, arm effects shared and per session; pS, pU (and REML) | per 10 data sets of the scratch unit run: session 150 to 163 s, day shared 90 to 108 s, day per session 179 to 225 s, animal shared 145 to 153 s, animal per session 1,336 to 2,748 s (`content_validation/unit_of_fit/launch.log`). So the animal per-session form costs about 10 to 20 times the session fit |
| both | leave one session out, on one data set of the no-effect scenario | the declared unit | one unit refit per session: about 242 unit fits per model and kind |

Each table reports, per scenario, unit and estimand: bias against both targets, spread, coverage of
both, the share of intervals excluding zero, and the sessions dropped for their baseline.

## Risks

| Risk | What shows it early |
|---|---|
| Shared clock curves push session differences in the curves into the reward coefficient | task 10's differing-curves scenario: Q2 at no planted reward effect, shared against per-session curves |
| Per-session arm effects bias the content levels (many small strata in a softmax) | Q3a and Q3b at no effect, per-session against shared arm effects, at a day (about 5 sessions) and an animal |
| An animal's rate fit does not fit in memory | the smoke test on j16 records peak memory; per-session curves are known not to fit densely (design note, section 8) |
| The calibration over five animals cannot meet `CALIBRATION_EDF_TOL` (the median of five jumps) | task 4's test; the first calibration on the slice raises and names the gap |
| Visits of different sessions merge as one support unit | task 2's test fails |
| Dropping sessions without baseline support changes the cohort silently | the `sessions` field and the slice's dropped counts |
| The two targets differ and coverage looks wrong against one of them | both are always reported (task 7) |
| The animal interval on 4 degrees of freedom is wide, and the day test is gone | the slice's interval widths and rejection rates, by unit |
| More, larger JAX programs per process | the fitter's compile log (`runtime.COMPILE_LOG`) on the slice's smoke test |
| The record's schema change breaks a reader of stored fits | intended (C13): `read_records` refuses an old file; the stored fits are refit |

## Rebuilt

Nothing in `paper/reported/` and no stored real fit. Simulation tables under
`models_convention/simulations/` (or `--output-dir`), one per model and unit.

## Exit criteria

The gating tests pass. At `fit_by="session"` every model's records and every simulation fixture equal
the ones before the phase. Every behavior model reads `BEHAVIOR_FIT_BY`. The validation slice has run and
its tables are with the lead. Every name the phase removes has no match (C13).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

## Edits owed elsewhere when the branches merge

Not made here, because the rate and content branches edit these files. Make them when the branches are on
master.

- `overview.md`: a row for Phase 4b in "Phases and dependencies" (delivers: the unit of the fit as one
  setting; depends on 2b, 2c, 5a's and 6a's branches merged); 5a, 6a, 7, 9 and 10 depend on 4b; the
  Status list and the risk table (the risks above).
- `PLAN.md`: the link to this file.
- `phase-5a-rate-models.md` and `phase-6a-content-models.md`: their fits and values read `fit_by`; their
  "per session" wording; their simulations name the unit.
- `phase-7-choice-stay-power.md`: the three models read `BEHAVIOR_FIT_BY` and declare their baselines.
- `phase-8-mpfc.md`: one line that the mPFC models stay per session (lead, 2026-11-24, 02:14).
- `phase-9-inference-family.md`: the declared family's values are unit values; the reduction is
  `unit_summary`; at the animal unit there is no day test.
- `phase-10-figure04.md`: a panel of per-session values needs a session-level fit or goes.
- `shared-contracts.md`: C5 (fit and inference: one fit per unit; the reduction of the design note's
  section 7); C7 (the record: `fit_by`, `sessions`; one row per (model, kind, unit)); C14 ("one
  inference").
- `designs.md`: the fitter's interface (`fit_sessions(..., fit_by)`).
- `legacy-inventory.md`: rows for the removals of "Order of work", item 4.
- The design, `docs/design/2026-11-20-regression-convention-and-figure04.md`: a new round (the
  twenty-sixth; the rate branch adds the twenty-fifth and the content branch the twenty-fourth) with the
  lead's 2026-11-24 words, a note on its fourteenth and seventeenth rounds ("one fit for each session")
  that this round changes them, and the open questions above in "Open after".
- `docs/README.md`: an index line for `docs/design/2026-11-24-unit-of-fit.md`.
- `docs/design/regression_fitter.md` (edited by the rate branch): task 13's sections.
- `docs/regression_pipeline_map.md` (edited by both branches): regenerated after task 13.

## Amendments, 2026-11-24 (after the simulations and the lead's 09:47 to 09:49 words)

The design note's section 11 has the evidence. What changes in this file:

- **Order of work.** The lead: "I suspect with will end up doing design C for all. And then running both
  the day and animal model and compare" and "So maybe design C with day and animal model and then design D
  later to check." The task-free tasks (1, 2, 3, 5, 7, then 11) start now from master 5ad82a02, which has
  the JAX fitter and the cleanup of 2026-11-24; the rate and content branches take master and this branch
  afterward (tasks 8 to 10, 12).
- **Question 2 is answered: per-session arm effects** (the baseline). `RoleContentSettings.arm_effects`
  is not built as a choice (task 9).
- **Question 3 is answered: fixed, declared penalties.** Task 6's `penalty_rule = "reml"` path is not
  built. The content model keeps today's per-session values at every unit (recalibrating on the unit was
  not better); a penalty is a declared value of the model, not tied to a unit, so the refusal of "a
  penalty calibrated at another unit" is not built either. `log_penalty_for_edf` still gains `fit_by`
  (the median over units), for the rate clocks, whose values are open.
- **Question 6: design D is a later check**, on this mechanism (design note, section 9). Nothing of D is
  built here.
- **Task 2.** The record's parquet storage was removed on 2026-11-24 (`record.write_records`,
  `read_records`; `tests/analysis/regression/test_record.py` went with it), so the two `test_record.py`
  tests are not written. `FitRecord` gains `fit_by` and `sessions` as written.
- **Task 4.** `support.terms_failing_share` was removed on 2026-11-24 (no fit applied it); only the rate
  model's `third_clock_decision` counts a share over units (task 8), and `log_penalty_for_edf` takes the
  median over units.
- **Task 5.** `shared_statistics.animal_mean_interval` is on the rate branch only; this branch adds it to
  master with its tests (copied from `origin/visit-rate-validation`), and the rate branch drops its copy
  when it takes master. Master's `animal_mean` (a table's mean per animal, 2026-11-24) is another
  function; `unit_summary` does not replace it.
- **Task 7.** The engine's own-fit path and its animal-unit branch are on the rate branch only; on master
  the engine gains `fit_by`, `unit_summary` and the two targets, and the rate branch's copies go when it
  takes master.
- **Task 10 gains the lead's 09:48 point** ("a decent amount of variation from day to day (or even session
  to session) from learning, motivation, etc."): each model's scenarios plant a day-level component of the
  tested effects, a trend over an animal's recording days, and a day-level shift of the level, and the
  statistics modules report, at the day unit, the between-day SD (the variance of the day estimates within
  an animal less their mean squared standard error, floored at 0) and the slope of the day estimates on
  the day's rank, and the difference between an animal's value and the mean of its day values.
- **The program bound** (`runtime.LIVE_PROGRAM_LIMIT`): a unit's fit is one outermost call per unit, as a
  session's is today; `fit_sessions` stays without `in_float64` (`docs/design/regression_fitter.md`, "How
  many programs a process keeps").
- **Question 4 is answered (lead, 09:55): both units.** "I did say maybe, lets just make this the decision
  now and we can change if the evidence is against it". Task 1 declares `BEHAVIOR_FIT_BY = ("day",
  "animal")`, not one value. `"session"` stays a value of `FitBy` and the default of `fit_sessions`. A
  model's settings carry one `fit_by` for each run (tasks 8 and 9). Task 7: since both units are always
  run, a run of the engine fits the same drawn data sets at several units where the engine's structure
  allows it.

## Notes from the /simplify round of 2026-11-24 (for tasks 8 to 10 and later)

The four reviews and the fix list are in
`.claude/research/2026-11-20-regression-prototypes/unit_of_fit_simplify/` (main checkout). Skipped there,
recorded here:

- **The general form of a baseline, if per-session smooths or design D's factor smooth are wanted.** The
  session becomes an ordinary row factor (the fit's rows carry a session column whose levels are the unit's
  sessions), and a baseline becomes "a term by session" through the existing term builders: the intercept
  is a fully coded session category that replaces it; a category gets `by` the session, each session with
  its own reference; a per-session curve is `Smooth(by=session)`, which exists and takes its sum-to-zero
  constraint on each session's rows. It removes `Design._base`, `_source`, `_build_copies`,
  `DesignTerm.session` and the special names of `for_sessions`, and it keeps "a term leaves with its
  level". Not built: the behavior models need only an intercept or a category as the baseline, which the
  copies serve; per-session curves are not planned (the rate follow-up's interim numbers: they bias the
  reward term more than shared curves do). About a day or two. The main session's call; the lead can
  overrule it.
- **A penalized baseline** (design D: one strength over a term's copies) is refused by `build_design`
  today. When it is built, one penalty block covers a term's copies over the sessions, and the fit's data
  take the block's active columns, as a session's copy can be dropped.
- **A category baseline in the record (A6).** At the session unit the content arm category is an ordinary
  term (in `edf` and `collinearity`); at a day or an animal its copies are projected out and are in
  neither. A rule keyed on the declared `design.baseline` would change the content model's session
  records: the lead's decision. Until then this is the documented behavior.
- **Task 7 follow-ups (A7b, A7c).** The unit target is NaN under the Bernoulli because `Bernoulli.check`
  refuses proportions, not for a reason of the mathematics. The noise-free target fit runs the support rule
  and the refit on the expected outcome, so its active set can differ from a data fit's.
- **Several units as one model (A8).** Today several units in one run are models that differ only in
  `fit_by`, with distinct names. A `Simulation.fit_by` tuple, rows keyed by (model, fit_by), would let one
  `Model` run at `BEHAVIOR_FIT_BY`: for tasks 8 to 10 if the runners want it.
- **Typed session entries (A9).** `FitRecord.sessions` entries are plain dicts; a `TypedDict` or a small
  dataclass, with a helper for the sessions with rows, if tasks 8 to 10 read them often.
- **A copy's support counted on its session's rows (E4) is not built.** The counts would be identical
  (`n_high` unchanged; `n_low` = the session's units with a zero row plus the distinct kept units of the
  other sessions), but the support rule would need each session's rows, which only the copies of the
  intercept carry today: not a local change. It would save about 4 s of 47 s in an animal-size rate fit
  (measured 2026-11-24).

