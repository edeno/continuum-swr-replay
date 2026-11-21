# Overview — scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

The approved design is `docs/design/2026-11-20-regression-convention-and-figure04.md` (branch
`figure04-behavior-models`). This file does not restate it; it says how the work is cut, ordered and checked.

## Goals

In the lead's order (design section 9, eighth round):

1. Figures 2 and 3 and `paper/reported/events.json` use the same data subset as Figures 4 and 5: events
   that start inside a well visit, chosen by the one selection function.
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

## Status

As of 2026-11-21 (branch `figure04-behavior-models`, at master 98fe5c06 plus one `TODO.md` commit):

| Phase | State | Commit on master |
|---|---|---|
| 1, the shared fitter | done; interface differs from the plan's sketch (the "as built" note at the top of [phase-1-shared-fitter.md](phase-1-shared-fitter.md)) | merged 6e54eda8, cleaned up 98fe5c06 |
| 3, the well visit in the tables | done; the 13 canonical tables published | d14ff888 |
| 4, Figures 2 and 3 on the common subset | done, except 20 held-out tetrode values in `paper/reported/events.json`, pending until another session's rerun ends; the fold task waits for `swr-trial-folds` and Phase 2 | 5dc74fc6 |
| 2, the convention's tools | **next**; can start now | |
| 5a to 12 | not started | |

What Phase 2 waits for, and only in the tasks named:

- **Open with the lead: the rule for a smooth term in a per-session fit.** The design stays one fit for
  each session (design, fourteenth round). Option A: a natural cubic spline with 4 columns, knots from the
  pooled data, no penalty (the author's recommendation, with a check that the conclusions hold at 3, 4
  and 5 columns). Option B: a P-spline with one fixed weak penalty per term. Known from simulation: a
  penalty chosen from the data by REML makes each session's curve a line when a session has about 140
  events, and the belief-variance test is then wrong; a fixed, declared smoothness keeps that test
  correct. Either way the REML choice is not used for per-session smooth terms with few events; it stays
  in the package. Every place that depends on it carries a line `WAITS FOR THE LEAD (smooth-term rule)`.
- **Decided (lead, 2026-11-21): a session whose simple terms have no finite maximum is reported with
  those terms missing**, not left out. Only the penalty search handles one today; `fit_at_penalties` raises
  at the first, so Phase 2 builds the handling outside the search
  ([phase-2-convention-tools.md](phase-2-convention-tools.md)).

Calls of the author still to confirm with the lead, each at its phase: at Phase 5a, the events that start
on a stop's first sample, which the rate design does not count (about 0.2 % of the No SWR events, none of
the SWR events: 28 events); at Phase 6a, Q4a read as the change over one standard deviation of the
reward-fraction smooth, and the trial number defined as the number of visits arrived.

## Phases and dependencies

| Phase | Delivers | Depends on | Size (code + tests, LOC) |
|---|---|---|---|
| [1](phase-1-shared-fitter.md) (done) | `analysis/regression/` core: likelihoods, penalized Newton, exact REML derivatives, pooled REML, grid start; checks 1 to 5; mgcv script; older all-bins history model deleted | nothing (no table) | L, ~1,800 |
| [2](phase-2-convention-tools.md) | bases, collinearity, support rule, fit-row record, folds, bands, day-level reduction, declared-test list | 1 | M, ~1,100 |
| [3](phase-3-visit-tables.md) (done) | visit clocks and stop-end kinds in the stop and event tables, two new uncertainty covariates, the visit at-risk bin design; one table rebuild; the same columns in the ContFrag tables; the dense stop time-bin table and table-side legacy removed (compatibility branches, re-exports, one-time table scripts) | precondition C3 (code independent of 1 and 2) | M, ~900 added, ~800 removed + a rebuild |
| [4](phase-4-figures-2-3-common-subset.md) (done) | Figures 2, 3 and `events.json` on the common subset; the duplicate fold function removed | C3; the other session's branch on master | M, ~500 + producer re-runs |
| [5a](phase-5a-rate-models.md) | visit rate model, models A and D, Q1, Q2, Q7a values; the per-bin ladder, its plotters, values and docs removed | 2, 3 | L, ~1,300 added, ~5,000 removed |
| [5b](phase-5b-rate-families-moved.md) | per-visit total reward effect, SWR-contrast covariates, uncertainty rate model, on the convention; the three old families removed | 5a | M, ~700 added, ~3,500 removed |
| [6a](phase-6a-content-models.md) | nine-arm content by role, smooth recency, one uncertainty model (belief variance and sample count with the trial number, after the simulation gate), smooth reward fraction, Q3, Q4, Q5, Q7b values, the ContFrag sources; the content ladders removed | 2, 3 | L, ~1,500 added, ~8,000 removed |
| [6b](phase-6b-content-robustness.md) | count and Dirichlet-multinomial estimators on the new design; within-arm comparison, old estimator paths and the per-term gate removed | 6a | M, ~600 added, ~2,500 removed |
| [7](phase-7-choice-stay-power.md) | Q6, stay or leave per visit, ripple power, the event-source comparison (ContFrag sources), on the convention; old choice, stay or leave (with the pooled GEE), power, the old event-source suite and the conditional fitters removed | 5a, 6a | M, ~900 added, ~4,000 removed |
| [8](phase-8-mpfc.md) | mPFC unit models on the convention (JAX twin, check 6, REML over units, trial folds); Figure 5 redrawn; statsmodels backend, fixed-sigma ridge, blocked folds, stop cohort and old-bundle readers removed | 2, 3 | L, ~1,400 added, ~3,000 removed + GPU run |
| [9](phase-9-inference-family.md) | the declared family with FDR, bands, inference values | 5a, 6a, 7 | M, ~600 |
| [10](phase-10-figure04.md) | Figure 4 | 9 (and 5a, 6a, 7) | L, ~1,000 |
| [11](phase-11-history-models.md) | duration-aware and ridge cross-validated history models on the shared fitter; the old history models, `per_bin_design.py`, the stop-bin specs and constants removed | 10 | M, ~600 added, ~4,000 removed |
| [12](phase-12-legacy-sweep.md) | final sweep: cross-cutting leftovers, the ASK list answered, five audits proving nothing replaced is left | 10, 11 | S to M, deletions and audits |

```
1 ──> 2 ──┬──> 5a ──> 5b
          ├──> 6a ──┬──> 6b
          │   (5a + 6a) ──> 7 ──┐
          └──> 8                ├──> 9 ──> 10 ──┬──> 11
3 ───────────> (5a, 6a, 8)      │               └──> 12
C3 ──> 3, 4;  other session's merge ──> 4    5a, 6a ──┘
6a's real fits wait for the uncertainty-measure gate (the lead sees the study first)
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
  (`NEWTON_DECREMENT_TOL`): the content fitter; check 3 compares to it; removed in Phase 7 (after a fixture).
- `src/continuum_swr_replay/analysis/replay_models/rate/per_bin_rate_model.py:293` (`fit_per_bin_glm`,
  statsmodels), `:509` (shuffled `KFold` over stops): the per-bin family; check 1 compares to it; removed in
  Phase 5a or 5b (after a fixture).
- `src/continuum_swr_replay/analysis/replay_models/per_bin_design.py:394` (`aggregate_to_timebins`): the
  stop bin design, first 4 s (`behavior/design_constants.py:111`, `TMAX`); the visit design of Phase 3 reuses
  its at-risk and outcome-shift logic (`:85-91`); removed in Phase 11 with its last caller.
- `src/continuum_swr_replay/analysis/replay_models/model_specs.py`: every ladder (`MODEL_PER_BIN_RATE` `:879`,
  `MODEL_PER_STOP_RATE` `:490`, `MODEL_HIERARCHY` `:247`, `MODEL_UNCERTAINTY_RATE` `:1090`,
  `MODEL_CROSS_HISTORY_RATE` `:1141`, `MODEL_CHOICE` `:1251`); `docs/models_table.md` mirrors it
  (`tests/repository/test_models_table_mirrors_specs.py`). New models get their own spec module in
  `replay_models/` (call) and their own docs table.
- `src/continuum_swr_replay/analysis/replay_models/rate/cross_history.py` and
  `scripts/replay_models/rate/fit_cross_history_rate.py`: the older all-bins history model, deleted in
  Phase 1 (done; the spec `MODEL_CROSS_HISTORY_RATE` stays: the duration-aware model uses it,
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
  `analysis/pfc/cohort.py:65` (`load_session_stop_cohort`): replaced and removed in Phase 8.
- `src/continuum_swr_replay/reported/behavior/__init__.py:132` (`summarize`): gains the new families'
  values phase by phase and loses the old ones in the same phases.
- `src/continuum_swr_replay/reported/questions.py:93` (`FIGURES`), `:320-358` (Part 3 questions): Q1 to Q7
  added in Phase 5a, Figure 4 in Phase 10.
- `scripts/workflows/run_refits.py:152-200` (`RUNS`): each new runner is registered there; an old one leaves in
  the phase that replaces it.
- [legacy-inventory.md](legacy-inventory.md): every other piece of legacy in the touched areas, with its
  callers, replacement, phase and class (REMOVE, ASK, KEEP).
- Untouched: the decoders, the event rule, the SWR source, `event_definition/` (except
  `cohort_selection.py`'s old selectors), the event evidence code
  (owned by the other session until Phase 4), Figures 1 and the supplements.

## What happens to each existing family

The design's table (section 7), with the phase that does it. The phase that builds a replacement removes
what it replaces ([C13](shared-contracts.md#c13-no-legacy-left-behind)); the item-by-item list, with what
else counts as legacy, is [legacy-inventory.md](legacy-inventory.md). Nothing is removed before its
replacement's values are in a summary, and a value `paper/*.tex` cites is removed only with the lead's yes,
asked at that phase's boundary.

| Family | Fate | Phase |
|---|---|---|
| Per-bin rate ladder (M0 to M9r, arrival rungs) | replaced by the visit model | 5a builds and removes |
| Per-stop rate ladder | refit by the shared fitter as the per-second-of-stop reward effect | 5b builds and removes |
| SWR-contrast rate ladder | its kind contrast is in the visit model; its other covariates moved to the convention | 5b builds and removes |
| Uncertainty rate model | moved to the convention | 5b builds and removes |
| Duration-aware and ridge cross-validated history | kept; moved to the shared fitter after Figure 4 | 11 builds and removes |
| Older all-bins cross-history model | deleted first (lead: "ok delete"; nothing reports it) | 1 |
| Content ladders (arm, patch, contrast, leaf) | replaced by the role model; the ContFrag ladders too (ASK-1: their sources move onto it) | 6a builds and removes |
| Dirichlet-multinomial content | robustness estimator on the new design | 6b builds and removes the old path |
| Within-arm comparison | deleted when Q5 values are in the summary (lead: "ok delete") | 6b |
| Choice | moved | 7 builds and removes |
| Stay or leave | refit on the convention (the pooled GEE goes); the lead then decides if it stays | 7 builds and removes |
| Ripple power per event | moved onto the shared cohort | 7 builds and removes |
| mPFC peri-event ladder, marked kernels, stop context | moved (REML over a session's units, trial folds, shared rules) | 8 builds and removes |

## How reported values and figures are rebuilt

After each phase that writes fits: commit, then rebuild only the summary the phase changes
(`behavior` for 5a to 7 and 9; `events` for 4; `mpfc` for 8), then emit the macros
([C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures)). Figures that read changed values are
redrawn in the same phase (Figures 2 and 3 in 4, Figure 5 in 8, Figure 4 in 10). Pending values from the
table rebuilds of C3 and Phase 3 are accepted. `make release-check` may fail on pending values until Phase 12
runs it last; that is not a blocker for any phase before it.

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
| Stops that cross the final poke-out hold 31 % of the immobile time inside visits; selecting stops by label would drop the seconds before leaving, and counting only whole-in-visit events would make the rate fall before leaving | bins carry their own label and are selected by the same function, and the event set is the events that start inside a visit, so rate and content analyze the same events ([C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter); L4) |
| The belief-variance test moves recency into its slope (a shrunk recency curve) or has too little power to be informative | the repeat with the shared fitter's REML penalty was run (design, thirteenth round; simulation): a penalty chosen from the data makes each session's curve a line at about 140 events and the test wrong. WAITS FOR THE LEAD (smooth-term rule): the recency curve is fitted by the rule the lead chooses (a natural spline with no penalty, or a fixed weak penalty); the rule and the power statement are written before the fits; a null is reported as "no evidence beyond recency; the design cannot detect an effect of this size" |
| The table rebuild of Phase 3 changes files the other session reads, and its summing step refuses until its runs record the new event table (design, eighth round) | build into a staging root (`--output-dir`), check parity, message the other session before swapping the files in, and swap once |
| A declared estimand fails the support or collinearity rule in many sessions | the rule is applied before any coefficient is read; the phase stops and reports to the lead; no silent pooling |
| Pooled REML is slow (every evaluation refits every session) | measured on synthetic sessions of the real size: one selection over 242 sessions took 6.3 min (`docs/design/regression_fitter.md`, "Cost"); prepare each session once per search before the real searches (`TODO.md`); parallelize over sessions only if measured above a few seconds. WAITS FOR THE LEAD (smooth-term rule): under either option the per-session smooth terms are fitted without a search, so this risk may apply only to the searches that remain (the mPFC units, Phase 8) |
| The fractional softmax response is a quasi-likelihood, so Laplace REML is approximate for content | check against mgcv on the same fractional design (mgcv `multinom` does not take fractions; compare with the count response and the shared fitter's own simulation coverage); report coverage from simulation |
| A cited value is removed by mistake | each retire task lists the macros `paper/*.tex` cites from the values it removes (none of behavior, mpfc or events were cited on 2026-11-20) and asks the lead at that phase's boundary; `tests/reported/test_manuscript_macros.py` fails on a cited macro that no longer exists |
| A baseline is removed too early (the old fits are the only baseline, and Phase 1's checks 1 and 3 compare with old fitters) | the old-against-new comparison is written before the removal (C13); a test that compares with an old fitter switches to a committed fixture of its coefficients first; old stored fits stay on disk, unread |
| Removing a family breaks a caller outside it (ContFrag, Part 1 and 2, the pipeline figure) | the inventory lists callers per item; ContFrag moves with its families (ASK-1, settled); the zero-match check and the full suite run at each phase end |
| The other session's branch is not merged when Phase 4 is ready | Phase 4 does not start; Phases 5a to 8 proceed; see Phase 4's "if not merged" |
| GPU contention with other users and the other session | [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session); Phase 8 measures three sessions before planning cards |
| Model D on a log-lag basis smooths away a sharp lag-1 effect | Phase 5a's simulation plants a sharp effect at lag 1 and reports what D returns; A remains the tested estimand |

## Rollout

Replace in place, family by family: a family phase builds the replacement, records the old-against-new
comparison, and removes the old code in the same PR or a second PR of the phase that merges right after.
Old and new code never live side by side past the phase that replaces the old. No backward compatibility,
shim or flag back to an old path is allowed ([C13](shared-contracts.md#c13-no-legacy-left-behind)); old
stored fits stay on disk, unread. Phase 12 is the final sweep and audit. Each phase is one PR (or two) into
master with the lead's approval.

## Decisions for the lead

Settled by the lead on 2026-11-20 (18:51 and 18:58), with the lead's words. The design's tenth round
(`docs/design/2026-11-20-regression-convention-and-figure04.md`, section 9) records L2, L3 and L4.

| # | Decision | Lead's words | Settled as | Applied in |
|---|---|---|---|---|
| L1 | The declared family | "Ok" (18:51); then "Yes" (20:39) to dropping the reward variability | twelve estimands, 36 tests; Q5b has two members, the belief variance (Q5b-var) and the sample count (Q5b-n) | C8; Phases 2, 5a, 6a, 9 |
| L2 | The uncertainty measures | "Do your rec. Maybe we should check on simulated" (18:51); after the simulation, "Yes" and "Yes" (20:39) | the simulation (scratch `proto_uncertainty_measures/REPORT.md`, real data structure, six leaf arms, 100 data sets per truth) found the belief variance about 90 % predicted by recency within a session (VIF 30 to 43; calibrated only with a flexible, unshrunk recency curve; power 30 to 50 % at 0.25 per SD), the reward variability not separable from reward-fraction curvature, and the sample count a drift with the trial number unless the trial number is in the model. Settled: the reward variability is dropped (not built); the reward fraction gets a smooth per role; the belief variance and the sample count = log(1 + visits to the well with a known outcome earlier in the epoch) are tested together in one model with the trial number in the epoch. Calls: before any real fit the simulation is repeated with the shared fitter's REML recency penalty, and if that penalty moves recency into the belief variance, the recency curve of that model is fitted unpenalized or with a held penalty, stated before the fit; the low power is stated before the fits; the belief variance is reported beyond recency (C5's rule for a term that fails the collinearity rule by design). The repeat and its committed test are Phase 6a's gate | designs.md, C5, C8; Phases 2, 3, 6a |
| L3 | "Arrival at the well" | no decision needed | on all 242 epochs `last_well_arrival_time_s` equals the visit's first poke-in for every stop that starts inside a visit where both exist (93,302 stops, no difference; missing for 554). The visit starts at the first poke-in; `last_well_arrival_time_s` and the clocks built on it are replaced by the visit clocks | C3; Phase 3; inventory C-5 |
| L4 | Stops and events across the final poke-out | option A, and "rate and content models should analyze the same events" | a stop that starts inside a visit contributes its bins up to the final poke-out; the event set for every family and figure is **the events that start inside a well visit**, `STARTS_IN_WELL_VISIT = ("in_visit", "spans_visit_end")` (69,011 + 189 = 69,200 events), the default of `select_visit_context`; `well_visit_id` is set for every row that starts inside a visit; a `spans_visit_end` event's current arm is the arm of the visit's well. The lead chose the start over the whole event because the 189 are 8 to 11 % of the starts in the last 0.25 s before the poke-out and most in the last 0.05 s | C3, C4, C5; designs.md; Phases 3 to 8 |
| L5 | Figure 2's SWR and time panels | "Ok" | every Figure 2 panel that counts SWRs or seconds uses only time inside a well visit, by a sample-level mask from the same visit intervals | C4; Phase 4 |
| L6 | The total reward effect | "We need to be consistent in the repo" | read as: the visit is the unit everywhere; the total reward effect is per visit; no per-stop variant is kept beside it, not even as a literature check; a per-stop number, if ever wanted, is a question for the lead | C6, C14; Phase 5b |
| L7 | Stay or leave | not answered in words | by the same rule (one unit), one decision per visit: the coordinator's application of the lead's rule, **to be confirmed at Phase 7's boundary** | Phase 7 |
| ASK-1 | The ContFrag regressions | "Move them" | the ContFrag SWR- and MUA-sourced events become two more event sources of the new rate and content runners, reported as a supplementary comparison; the ContFrag tables get the visit labels (lead: "Label them now", on the labels branch) and the new covariates and clocks in Phase 3's rebuild. ASK-3 follows: the comparison the moved sources need is built in Phase 7, the rest of the old event-source suite is removed | Phases 3, 5a, 6a, 7; inventory |
| ASK-4 | The dense stop time-bin table | "Remove" | the `stop_timebin` table, its builder, its canonical name and registry entry, and `scripts/exploratory/_timeresolved_psth.py` go in Phase 3; the event-onset table stays; the file on disk is not deleted; Phase 3's visit bins are what a raw-rate-against-clock check is drawn from | Phase 3; inventory C-7 |
| Rule | Consistency | "We need to be consistent in the repo" | one unit, one event set, one selection function, one fitter, one set of clocks in every family, summary and figure, the mPFC models and the ContFrag sources included | C14 |

Still open, for the lead, by Phase 12 at the latest: ASK-2 and ASK-5 to ASK-12
([legacy-inventory.md#ask-list](legacy-inventory.md#ask-list)), and the confirmation of L7 at Phase 7.
Open now, before Phase 2 can finish: the rule for a smooth term in a per-session fit, and how a session
without a finite maximum is reported ([Status](#status); the design's fifteenth round).

## Things in the design that are unclear or contradict each other

- "Direct" and "total" name two different contrasts (event length held fixed between kinds; the reward
  effect at fixed time before leaving against per visit). This plan names them apart
  ([C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct)).
- Decision 2 is recorded as "still with the lead" in the first and second rounds, but section 4.1 states the
  unit ends at the last nose-poke out (lead) and the poke times are cached for every epoch (later
  exploration). This plan treats section 4.1 as settled.
- Section 4.1 says "an event counts when it lies wholly inside the visit"; the lead's L4 rule (events that
  start inside the visit) overrides it.
- Q4a was a slope on the reward fraction; with the reward fraction now a smooth, Q4a is read as the change of
  the smooth over one pooled SD (designs.md; a call). The lead may prefer another summary of the curve.
- Section 6 places check 6 in stage 0; only the mPFC family needs a GPU (see the deviation above).
- Figure 4's panel c asks for the rate against "time until a small movement", but the third clock is kept
  only if the collinearity rule separates it; panel c's second half depends on Phase 5a's result.
- The design's section 4.3 chooses each smooth's strength by REML shared by all sessions. The fourteenth
  round keeps one fit per session, and the simulation shows the REML choice makes a per-session curve a
  line at few events; the rule that replaces it is open with the lead (see [Status](#status)).
- Contract C4 says rate and content analyze the same events; as built, the rate design does not count an
  event with no at-risk time before its start (28 events start on a stop's first sample). A question for
  the lead at Phase 5a.

## Estimated effort

About 13,000 lines of code and tests added and about 30,000 removed (code, tests and docs of the replaced
families; a rough count from the inventory's modules) over fourteen phases, plus one table rebuild, the producer re-runs of
Figures 2 and 3, CPU refits of the rate, content, choice and power families, and one mPFC GPU run (the last
full one took about 14 GPU-hours; Phase 8 measures before committing to it).
