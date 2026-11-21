# Overview — scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

The approved design is `docs/design/2026-11-20-regression-convention-and-figure04.md` (branch
`figure04-behavior-models`). This file does not restate it; it says how the work is cut, ordered and checked.

## Goals

In the lead's order (design section 9, eighth round):

1. Figures 2 and 3 and `paper/reported/events.json` use the same data subset as Figures 4 and 5: events
   inside a well visit, chosen by the one selection function.
2. Then the regression plan for Figures 4 and 5: one convention (design section 4) and one shared fitter
   (section 5), shown correct (section 6), applied to every regression family the paper reports
   (section 7), with inference reduced session to recording day to animal and one declared family of tests,
   and Figure 4 drawn.

## Non-goals

- Building the visit labels or the selection function (branch `event-visit-labels`; a precondition,
  [shared-contracts.md#c3](shared-contracts.md#c3-precondition-the-visit-labels-are-on-master)).
- Figure 2's held-out analysis (the other session's branches `swr-restricted-models`, `swr-trial-folds`).
- The design's non-goals: a claim that uncertainty, as distinct from recency, shapes content (it returns
  only as the declared tests); No SWR events as decoding spillover (question 2c.6, the wider-Local-kernel
  re-decode); any change to event definitions, cohorts, the decoder or the tables' meaning; a joint model of
  all sessions first.
- Stage 3 of the design (a pooled model with session, day and animal grouping terms). It starts only if
  Phase 9's two-stage summary is not enough, by the lead's decision, in a later plan.
- A description of the events outside the visits in a supplement (sixth round: not for now).
- Any tool to avoid pending values (seventh round).

## Phases and dependencies

| Phase | Delivers | Depends on | Size (code + tests, LOC) |
|---|---|---|---|
| [1](phase-1-shared-fitter.md) | `analysis/regression/` core: likelihoods, penalized Newton, exact REML derivatives, pooled REML, grid start; checks 1 to 5; mgcv script; older all-bins history model deleted | nothing (no table) | L, ~1,800 |
| [2](phase-2-convention-tools.md) | bases, collinearity, support rule, fit-row record, folds, bands, day-level reduction, declared-test list | 1 | M, ~1,100 |
| [3](phase-3-visit-tables.md) | visit clocks and stop-end kinds in the stop and event tables, two new uncertainty covariates, the visit at-risk bin design; one table rebuild | precondition C3 (code independent of 1 and 2) | M, ~900 + a rebuild |
| [4](phase-4-figures-2-3-common-subset.md) | Figures 2, 3 and `events.json` on the common subset | C3; the other session's branch on master | M, ~500 + producer re-runs |
| [5a](phase-5a-rate-models.md) | visit rate model, models A and D, Q1, Q2, Q7a values | 2, 3 | L, ~1,300 |
| [5b](phase-5b-rate-families-moved.md) | per-stop (total) reward effect, SWR-contrast covariates, uncertainty rate model, on the convention | 5a | M, ~700 |
| [6a](phase-6a-content-models.md) | nine-arm content by role, smooth recency, three uncertainty measures, Q3, Q4, Q5, Q7b values | 2, 3 | L, ~1,500 |
| [6b](phase-6b-content-robustness.md) | count and Dirichlet-multinomial estimators on the new design; within-arm comparison deleted | 6a | M, ~600 |
| [7](phase-7-choice-stay-power.md) | Q6, stay or leave per visit, ripple power, on the convention | 6a | M, ~900 |
| [8](phase-8-mpfc.md) | mPFC unit models on the convention (JAX twin, check 6, REML over units, trial folds); Figure 5 redrawn | 2, 3 | L, ~1,400 + GPU run |
| [9](phase-9-inference-family.md) | the declared family with FDR, bands, inference values | 5a, 6a, 7 | M, ~600 |
| [10](phase-10-figure04.md) | Figure 4 | 9 (and 5a, 6a, 7) | L, ~1,000 |
| [11](phase-11-history-models.md) | duration-aware and ridge cross-validated history models on the shared fitter | 10 | M, ~600 |
| [12](phase-12-retire-replaced.md) | replaced ladders, runners, values and docs removed | 10, and the lead's yes on any cited value | M, mostly deletions |

```
1 ──> 2 ──┬──> 5a ──> 5b
          ├──> 6a ──┬──> 6b
          │         └──> 7 ──┐
          └──> 8             ├──> 9 ──> 10 ──┬──> 11
3 ───────────> (5a, 6a, 8)   │               └──> 12
C3 ──> 3, 4;  other session's merge ──> 4    5a, 6a ──┘
```

Parallel tracks: Phases 1, 3 and 4 can run at the same time (1 reads no table; 3 and 4 touch different
files). Phases 5a, 6a and 8 can run in parallel once 2 and 3 are in. Phase 8 is independent of the Figure 4
chain and can be scheduled around GPU availability.

**Deviation from the design's stage 0 (call).** The design lists check 6 (GPU against CPU) in stage 0. This
plan builds the JAX twin and check 6 in Phase 8, the only family that needs a GPU (rate and content run on
CPU, design section 5). Everything else of stage 0 is in Phase 1. The lead may move it back.

## Current codebase integration points

What the plan touches (line numbers on `figure04-behavior-models` at `02df809f`):

- `src/continuum_swr_replay/analysis/replay_models/fitting.py:677` (`fit_conditional_poisson`), `:78`
  (`NEWTON_DECREMENT_TOL`): the content fitter; check 3 compares to it; replaced in Phase 12.
- `src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_model.py:293` (`fit_per_bin_glm`,
  statsmodels), `:509` (shuffled `KFold` over stops): the per-bin family; check 1 compares to it; replaced.
- `src/continuum_swr_replay/analysis/replay_models/per_bin_design.py:394` (`aggregate_to_timebins`): the
  stop bin design, first 4 s (`behavior/design_constants.py:111`, `TMAX`); the visit design of Phase 3 reuses
  its at-risk and outcome-shift logic (`:85-91`); left in place until Phase 12.
- `src/continuum_swr_replay/analysis/replay_models/model_specs.py`: every ladder (`MODEL_PER_BIN_RATE` `:879`,
  `MODEL_PER_STOP_RATE` `:490`, `MODEL_HIERARCHY` `:247`, `MODEL_UNCERTAINTY_RATE` `:1090`,
  `MODEL_CROSS_HISTORY_RATE` `:1141`, `MODEL_CHOICE` `:1251`); `docs/models_table.md` mirrors it
  (`tests/repository/test_models_table_mirrors_specs.py`). New models get their own spec module in
  `replay_models/` (call) and their own docs table.
- `src/continuum_swr_replay/analysis/replay_models/rate/cross_history.py` and
  `scripts/replay_models/rate/fit_cross_history_rate.py`: the older all-bins history model, deleted in
  Phase 1 (the spec `MODEL_CROSS_HISTORY_RATE` stays: the duration-aware model uses it,
  `rate/duration_aware_history.py:295`).
- `src/continuum_swr_replay/analysis/replay_models/content/uncertainty_content.py:705`
  (`within_arm_horserace`), `:856` (`summarize_horserace`); `reported/behavior/content.py:718`
  (`horserace_values`); deleted in Phase 6b.
- `src/continuum_swr_replay/analysis/event_tables/stop_table.py:61` (`build_stop_table`),
  `event_table.py:545` (`build_event_table`), `arm_patch_tables.py:57` (`build_event_arm_table`),
  `behavior/covariates/candidate_arm.py:76-113` (`_history_by_event`): gain columns in Phase 3.
- `scripts/figures/generate_figure02.py:1326`, `scripts/figures/generate_figure03.py:178`,
  `src/continuum_swr_replay/reported/events.py:2172`: the event-table reads that Phase 4 moves onto the
  selection.
- `src/continuum_swr_replay/analysis/pfc/batched_glm.py:167` (`DEFAULT_PENALTY_SIGMA = 3.0`),
  `analysis/pfc/fit.py:112` (`model_penalties`), `analysis/pfc/heldout.py:222` (`assign_blocked_folds`),
  `analysis/pfc/cohort.py:65` (`load_session_stop_cohort`): changed in Phase 8.
- `src/continuum_swr_replay/reported/behavior/__init__.py:132` (`summarize`): gains the new families'
  values phase by phase; loses the old ones in Phase 12.
- `src/continuum_swr_replay/reported/questions.py:93` (`FIGURES`), `:320-358` (Part 3 questions): Q1 to Q7
  added in Phase 5a, Figure 4 in Phase 10.
- `scripts/workflows/run_refits.py:152-200` (`RUNS`): each new runner is registered there; old ones leave in
  Phase 12.
- Untouched: the decoders, the event rule, the SWR source, `event_definition/`, the event evidence code
  (owned by the other session until Phase 4), Figures 1 and the supplements.

## What happens to each existing family

The design's table (section 7), with the phase that does it. Nothing is removed before its replacement's
values are in a summary, and a value `paper/*.tex` cites is removed only with the lead's yes.

| Family | Fate | Phase |
|---|---|---|
| Per-bin rate ladder (M0 to M9r, arrival rungs) | replaced by the visit model | 5a builds, 12 removes |
| Per-stop rate ladder | refit by the shared fitter as the per-second-of-stop reward effect | 5b |
| SWR-contrast rate ladder | its kind contrast is in the visit model; its other covariates moved to the convention | 5b |
| Uncertainty rate model | moved to the convention | 5b |
| Duration-aware and ridge cross-validated history | kept; moved to the shared fitter after Figure 4 | 11 |
| Older all-bins cross-history model | deleted first (lead: "ok delete"; nothing reports it) | 1 |
| Content ladders (arm, patch, contrast, leaf) | replaced by the role model | 6a builds, 12 removes |
| Dirichlet-multinomial content | robustness estimator on the new design | 6b |
| Within-arm comparison | deleted when Q5 values are in the summary (lead: "ok delete") | 6b |
| Choice | moved | 7 |
| Stay or leave | refit on the convention; the lead then decides if it stays | 7 |
| Ripple power per event | moved onto the shared cohort | 7 |
| mPFC peri-event ladder, marked kernels, stop context | moved (REML over a session's units, trial folds, shared rules) | 8 |

## How reported values and figures are rebuilt

After each phase that writes fits: commit, then rebuild only the summary the phase changes
(`behavior` for 5a to 7 and 9; `events` for 4; `mpfc` for 8), then emit the macros
([C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures)). Figures that read changed values are
redrawn in the same phase (Figures 2 and 3 in 4, Figure 5 in 8, Figure 4 in 10). Pending values from the
table rebuilds of C3 and Phase 3 are accepted. `make release-check` is expected to fail on pending cited
values until Phase 12; that is not a blocker for any phase before it.

## Metrics

- Fitter: agreement with mgcv on the fixtures (edf to 1e-6, curves to 1e-4 on the link scale, a parametric
  coefficient and its standard error to 1e-4 relative; the prototype reached tighter); REML gradient and
  Hessian against finite differences to 1e-5 relative; no-penalty coefficients equal to the current fitters'
  to 1e-6 relative on three real sessions.
- Simulation (each family): planted effects recovered (bias within 2 standard errors of the simulation mean);
  95 % intervals cover 93 % to 97 % over 200 data sets; a null term significant in 3 % to 7 %; an effect
  planted on one term does not appear on another beyond the same rate.
- Data: row counts before and after each join, kept and dropped per label, stored with every fit and
  summary.
- Runtime: measured on one session and on the full cohort, recorded in each phase PR.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Stops partly inside a visit (`spans_visit_end`) hold a large share of immobile time; selecting stops by label drops the seconds before leaving | bins carry their own label and are selected by the same function ([C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter)); decision L4 |
| The table rebuild of Phase 3 changes files the other session reads, and its summing step refuses until its runs record the new event table (design, eighth round) | build into a staging root (`--output-dir`), check parity, message the other session before swapping the files in, and swap once |
| A declared estimand fails the support or collinearity rule in many sessions | the rule is applied before any coefficient is read; the phase stops and reports to the lead; no silent pooling |
| Pooled REML is slow (every evaluation refits every session) | measure one evaluation; the grid has at most 125 points; parallelize over sessions only if measured above a few seconds |
| The fractional softmax response is a quasi-likelihood, so Laplace REML is approximate for content | check against mgcv on the same fractional design (mgcv `multinom` does not take fractions; compare with the count response and the shared fitter's own simulation coverage); report coverage from simulation |
| Removing old ladders removes values the manuscript cites | Phase 12 lists cited macros first; the lead decides each text change |
| The other session's branch is not merged when Phase 4 is ready | Phase 4 does not start; Phases 5a to 8 proceed; see Phase 4's "if not merged" |
| GPU contention with other users and the other session | [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session); Phase 8 measures three sessions before planning cards |
| Model D on a log-lag basis smooths away a sharp lag-1 effect | Phase 5a's simulation plants a sharp effect at lag 1 and reports what D returns; A remains the tested estimand |

## Rollout

Replace in place, family by family, with old and new fits side by side until Phase 12. No backwards
compatibility is required for code; old stored fits stay on disk until Phase 12 so they can serve as
baselines. Each phase is one PR into master with the lead's approval.

## Decisions for the lead

Each must be settled before the named phase starts.

| # | Decision | Plan's default if the lead agrees | Before |
|---|---|---|---|
| L1 | The declared family: the design says "ten estimands ... 30 tests", its table lists eleven (33 tests), and three uncertainty measures each "tested once" make Q5b three (39). Which list? | eleven estimands with Q5b counted three times: thirteen estimands, 39 tests | 5a |
| L2 | The two new uncertainty measures: "how variable an arm's rewards were" (over which window? note that for 0/1 outcomes the sample variance over the last k visits is `p(1-p) k/(k-1)`, a function of the reward fraction already in the model) and "how many times the animal has sampled the arm" (visits in the epoch so far, or in the last N trials; log or raw) | variability: the variance of the arm's outcomes over its last ten visits, flagged for its collinearity with reward fraction; samples: `log1p` of completed visits to the arm earlier in the epoch | 3 (to build them in the one rebuild; else 6a builds them at design time) |
| L3 | "Arrival at the well": the first nose-poke in (the visit labels' `arrival_time`) or the trial arrival (`last_well_arrival_time_s`, `stop_table.py:188`). The design says the unit runs "from the animal's arrival at a well", the labels from the first poke-in | the first poke-in, so the unit and the label agree | 3 |
| L4 | At-risk time of stops partly inside a visit: count their in-visit seconds (bins labeled `in_visit`), or drop such stops whole (what `select_visit_context` on the stop table does) | count the in-visit seconds | 3 |
| L5 | Figure 2's SWR-grain and time-grain panels (SWR counts 2c, detector overlap 2e, immobile time by ripple power 2j, the brain-state reference): do they also move to the subset (SWRs and immobile time inside visits)? The labels mark non-local events and stops, not SWRs or time bins | yes, by a sample-level visit mask from the same visit intervals, with the counts per label recorded | 4 |
| L6 | The per-second-of-stop ("total") reward effect: per stop, as the design says, or per visit now that the visit is the unit | per visit (exposure = the visit's at-risk seconds), with the per-stop variant as the literature-comparable check | 5b |
| L7 | Stay or leave under the visit unit: one decision per visit (`is_stay` of the next trial), with content summarized over the visit's in-visit events | yes | 7 |

## Things in the design that are unclear or contradict each other

- The count of the declared tests (L1).
- "Direct" and "total" name two different contrasts (event length held fixed between kinds; the reward
  effect at fixed time before leaving against per second of stop). This plan names them apart
  ([C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct)).
- Section 4.1 says the stop table gains "each stretch's kind of end (a small movement or the animal
  leaving)" and the stop table's in-visit label marks a stop that crosses the last poke-out as outside the
  visit (L4).
- Decision 2 is recorded as "still with the lead" in the first and second rounds, but section 4.1 states the
  unit ends at the last nose-poke out (lead) and the poke times are cached for every epoch (later
  exploration). This plan treats section 4.1 as settled.
- The reward variability measure may not be separable from the reward fraction (L2).
- Section 6 places check 6 in stage 0; only the mPFC family needs a GPU (see the deviation above).
- Figure 4's panel c asks for the rate against "time until a small movement", but the third clock is kept
  only if the collinearity rule separates it; panel c's second half depends on Phase 5a's result.

## Estimated effort

About 13,000 lines of code and tests over fourteen PRs, plus one table rebuild, the producer re-runs of
Figures 2 and 3, CPU refits of the rate, content, choice and power families, and one mPFC GPU run (the last
full one took about 14 GPU-hours; Phase 8 measures before committing to it).
