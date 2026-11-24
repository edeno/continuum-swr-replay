# Shared contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Every phase obeys these. Do not weaken one inside a phase; if a phase cannot meet one, stop and ask the
lead. "The design" means `docs/design/2026-11-20-regression-convention-and-figure04.md`, with section
numbers as there.

- [C1. Running code and tests](#c1-running-code-and-tests)
- [C2. Data, the database, git hygiene](#c2-data-the-database-git-hygiene)
- [C3. Precondition: the visit labels are on master](#c3-precondition-the-visit-labels-are-on-master)
- [C4. The selection function is the only filter](#c4-the-selection-function-is-the-only-filter)
- [C5. The convention (six rules and the collinearity rule)](#c5-the-convention)
- [C6. Kinds, contrasts and the two meanings of "direct"](#c6-kinds-contrasts-and-the-two-meanings-of-direct)
- [C7. The fit-row record](#c7-the-fit-row-record)
- [C8. The declared family of tests](#c8-the-declared-family-of-tests)
- [C9. Where new fits are stored](#c9-where-new-fits-are-stored)
- [C10. Rebuilding reported values and figures](#c10-rebuilding-reported-values-and-figures)
- [C11. GPUs, long runs and the other session](#c11-gpus-long-runs-and-the-other-session)
- [C12. Layering of the new package](#c12-layering-of-the-new-package)
- [C13. No legacy left behind](#c13-no-legacy-left-behind)
- [C14. One of everything](#c14-one-of-everything)

## C1. Running code and tests

A worktree has no environment of its own. Run with the main checkout's interpreter, from the worktree root:

```bash
PY=/cumulus/edeno/continuum-swr-replay/.venv/bin/python
PYTHONPATH=src JAX_PLATFORMS=cpu $PY scripts/<path>.py ...
PYTHONPATH=src JAX_PLATFORMS=cpu $PY -c "import sys, pytest; sys.exit(pytest.main(['-q', 'tests/analysis/regression']))"
PYTHONPATH=src JAX_PLATFORMS=cpu $PY -m ruff check src scripts tests   # and ruff format --check
```

- `JAX_PLATFORMS=cpu` on every command that is not a measured GPU job (a JAX import otherwise grabs every
  card on the host).
- The mgcv reference runs in the separate conda environment `csr-mgcv`:
  `/home/edeno/miniconda3/envs/csr-mgcv/bin/Rscript <script>.R ...`. It is not part of the uv environment
  and nothing in `src/` or `tests/` calls R.
- Slow tests carry `@pytest.mark.slow` (the marker is already used in `tests/`; see `tests/README.md`).

## C2. Data, the database, git hygiene

- The worktree's `Processed-Data` is a symlink to the main checkout's data
  (`/cumulus/edeno/continuum-swr-replay/Processed-Data`). Writing there writes the shared data every
  session reads. Data roots are read from `$CSR_PROCESSED_DATA_DIR` through `paths.py`.
- Stage files path by path. Never `git add -A` or `git add .` in a worktree that links Processed-Data.
- Nothing writes to the lab's Spyglass database. A table rebuild uses `--from-cache`
  (`scripts/tables/prepare_replay_content_data.py:517`); a fallback read of Spyglass is allowed, a write
  never. `tests/repository/test_no_database_writes.py` must keep passing.
- No commit to master and no merge or fast-forward into master without the lead's explicit approval.
  Push the phase branch as commits land.
- American spelling in names and prose (`tests/repository/test_american_spelling.py`).
- Code, docstrings and test names do not mention this plan, its phases or `.claude/docs/plans/`.

## C3. Precondition: the visit labels are on master

The visit labels and the selection function are built on branch `event-visit-labels` (worktree
`/cumulus/edeno/continuum-swr-replay-visit-labels`) and merged, with the canonical tables rebuilt, before
this plan starts. Do not rebuild them. Every phase that reads the tables first runs this check and stops if
it fails:

```bash
git log master --oneline | head -20          # a commit that adds visit_context is there
git merge-base --is-ancestor master HEAD      # the phase branch holds it
PYTHONPATH=src JAX_PLATFORMS=cpu $PY - <<'EOF'
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context, count_visit_context)
from continuum_swr_replay.analysis.constants import IN_WELL_VISIT, VISIT_CONTEXTS
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.paths import replay_content_dir, table_path
for name in ("event_table", "stop_table", "event_arm_table"):
    t = read_swr_table(table_path(name, replay_content_dir()))
    assert "visit_context" in t.columns, name
    print(name, count_visit_context(t, unit_col="event_id" if name == "event_arm_table" else None))
EOF
```

Names as built on that branch on 2026-11-20 (uncommitted then; confirm on master): `visit_context`,
`well_visit_id`, `last_completed_well_visit_id`, `time_since_last_poke_out_s`,
`time_until_next_poke_in_s` (`analysis/event_tables/visit_context.py`, `VISIT_CONTEXT_COLUMNS`), the
vocabulary `analysis/constants.py:VISIT_CONTEXTS` with `IN_WELL_VISIT = "in_visit"`, and
`analysis/event_definition/cohort_selection.py:select_visit_context` and `count_visit_context`. If master
uses other names, use master's and note it in the phase PR.

Facts about those labels that this plan relies on (read from that branch's code):

- A window is `in_visit` exactly when its `occupancy_state` is `at_well`
  (`table_contracts.require_visit_context_labels`). A window that starts inside a visit and ends after its
  final poke-out is `spans_visit_end`; one that starts before a first poke-in and ends inside the visit is
  `spans_visit_start`. On the labels branch `well_visit_id` was set for `in_visit` rows only; Phase 3 sets
  it for every row that starts inside a visit (`in_visit` and `spans_visit_end`).
- The default event set (lead, 2026-11-20, decision L4): **the events that start inside a well visit**,
  `analysis.constants.STARTS_IN_WELL_VISIT = ("in_visit", "spans_visit_end")`, which `select_visit_context(table)`
  selects with no argument (being added on the labels branch; if master lacks it, Phase 3 adds it).
  Diagnostic of 2026-11-20 (not a summary value): 69,011 `in_visit` plus 189 `spans_visit_end` events. The
  lead chose the start over the whole event because the 189 are 8 to 11 % of the starts in the last 0.25 s
  before the poke-out and most of those in the last 0.05 s; leaving them out would make the rate fall before
  the animal leaves by a counting rule.
- A visit is one port experience, from `arrival_time` (first nose-poke in) to `departure_time` (final
  poke-out) of the port experiences.
- The label columns are required in every canonical table, the ContFrag tables included (lead: "Label
  them now"; being added on the labels branch). The allowance that skipped a table with no `visit_context`
  ("built before the label existed", labels branch head `9b460a90`) is being removed there; if master still
  has it, remove it in the first phase that reads the tables (C13).
- The stop table's arrival clock `last_well_arrival_time_s` equals the visit's first poke-in for every stop
  that starts inside a visit where both exist (all 242 epochs: 93,302 stops, no difference; missing for
  554), so arrival needs no decision (L3). The visit clocks replace it and the clocks built on it
  ([inventory C-5](legacy-inventory.md#cohort-selectors-and-stop-clocks)).

## C4. The selection function is the only filter

- Every summary statistic, summary plot and model of Figures 2 to 5, in every family (rate, content,
  choice, stay or leave, ripple power, history, mPFC) and for every event source (the hippocampal non-local
  events and the ContFrag SWR- and MUA-sourced events), selects its events with `select_visit_context(table)`
  (default: `STARTS_IN_WELL_VISIT`, C3), and records `count_visit_context(table)` (rows or
  `event_id`s per label, kept or not) in its output. No other code filters by occupancy, position or visit.
  New code never calls the older `select_stop_cohort` / `select_event_cohort` (`cohort_selection.py:322`,
  `:378`); each remaining caller moves to the selection function or retires in the phase the inventory names
  ([legacy-inventory.md](legacy-inventory.md#cohort-selectors-and-stop-clocks)).
- Time is selected at bin grain. Time bins carry their own `visit_context` (Phase 3 adds it to the visit
  bin design), and the rate and mPFC designs select bins with the same function. A bin is `in_visit` when
  it lies inside a visit interval (first poke-in to final poke-out) and inside an immobility stop. So a stop
  that crosses the final poke-out contributes its bins up to the poke-out (lead, L4, option A), and a stop
  that starts before the first poke-in contributes its bins after it. With this rule every event that is
  counted starts in an at-risk bin and is in the default set: rate and content analyze the same events
  (lead: "rate and content models should analyze the same events"), with one exception as built
  (`visit_design.visit_timebins`): an event whose start has no at-risk time before it, such as one that
  starts on a stop's first sample, is not counted by the rate design. Those on a stop's first sample are
  about 0.2 % of the No SWR events and none of the SWR events (28 events). Whether the two families may
  differ by them is a question for the lead at Phase 5a
  ([designs.md#rate-design](designs.md#rate-design)).
  Diagnostic of 2026-11-20 (all 242 epochs, not summary values): 93,738 stops start inside a visit, 15,803
  of them cross the final poke-out (part before it median 1.75 s, after it 0.07 s); those parts before the
  poke-out are 31 % of the immobile time inside visits (11.4 of 36.8 h) and hold 20,497 of the 69,011
  `in_visit` events; 44,592 visits, median 3.11 s.
- Figure 2's panels that count SWRs or seconds (L5, lead: "Ok") use only time inside a well visit, by a
  sample-level mask from the same visit intervals; an SWR counts when it starts inside a visit (the event
  set's start rule applied to SWRs; Phase 4).
- A row with a missing value is never dropped inside a fitter. The cohort step drops it, counts it per kind
  and stores the counts with the fit.
- Figure 5's default is the same subset restricted to the epochs with mPFC units.

## C5. The convention

The design's section 4. In short, with the phase that implements each piece:

| Rule | What every model does | Built in |
|---|---|---|
| 1. Data | Unit is the well visit (first poke-in to last poke-out); the events are those that start inside a visit (C3); at-risk time is immobile in-visit time less time inside an ongoing event of the outcome's kind; three clocks per bin (time since arrival = first poke-in, time until leaving = until the last poke-out, time until the next movement of any size); whole visit modeled | Phase 3 |
| 2. Design | Clocks, trials since last visit and peri-event lags are cubic P-spline smooths (second-order difference penalty) at one fixed log penalty per term, declared in the model and the same in every session (the smooth-term rule below); arm role (current, previous, other leaf, trunk) is a category with a separate slope per role; kind is a category with separate terms per kind; one pooled standardization (population SD, ddof = 0) stored with the fit; a continuous factor in an interaction is centered at a fixed pooled reference, a 0/1 factor is not; reward history by model A (last outcome, 0/1) and model D (penalized smooth over the last ten outcomes, log lag) | Phases 2, 5a, 6a |
| 3. Fit | Plain penalized maximum likelihood by the shared fitter (no bias-reduced fit: lead, 2026-11-23), one fit for each session (design, fourteenth and seventeenth rounds), each smooth at its declared penalty (no penalty chosen from the data for a per-session smooth); Newton decrement convergence in float64, on the CPU and on the GPU (lead, 2026-11-23); a weak penalty on every simple tested coefficient (the weak-penalty rule below); a support rule per simple term and session that reads the design only (units with a row on each side), never the events, stored; a session without a finite maximum handled by one rule (Phase 2) | Phases 1, 2 |
| 4. Folds | Five folds of whole trials balanced by destination well, long stays cut into pieces (one function); used to compare designs and check over-fitting only; held-out claims summed over sessions | Phase 2 |
| 5. Inference | Session to recording day to animal; animal-balanced estimate with the nested bootstrap interval (t-corrected); animals with the estimate's sign; recording-day-level test; one declared family with FDR; simultaneous bands for curves | Phases 2, 9 |
| 6. Report | Rate effects as rate ratios, content effects as ratios of shares, each with an interval; every fit carries the record of C7; each statistic lives in `analysis/` and the figure calls the function the summary reads | every family phase |
| Collinearity | Before a coefficient is read: VIF for one-column terms, concurvity for smooths, on the quantity the likelihood sees; a term the others explain at R² ≥ 0.8 (VIF ≥ 5) is reported only jointly with its partner, or leaves by a rule written before the fit; a planted-effect simulation over the real design must not move to another term. A declared term that fails the rule **by design** (the question is what it adds beyond its partner, as the belief variance beyond recency) is not dropped: it is reported as a conditional effect, "beyond <partner>", with its VIF, the partner's curve fitted flexibly as the simulation requires, the simulated false-positive rate and power on the real design, and a null stated as "no evidence beyond <partner>; the design cannot detect an effect of this size". Which terms fall under this, and how they are reported, is written before the fit | Phase 2 (measures), each family phase (rule) |

> **The smooth-term rule** (design, seventeenth round; the author's reading of the lead's "For decision
> 2, I think I would prefer the penalized splines" and "I want to stick with one fit per session for
> now", 2026-11-22, told to the lead on 2026-11-22 and 2026-11-23 and not objected to). It replaces the
> design's section 4.3 (one REML strength per term shared by the sessions). A penalty chosen from the
> data makes a per-session curve a line when a session has few events (simulation, about 140 events), so
> no penalty is chosen from the data for a per-session smooth term; the REML selection stays in the
> package. Each term has one fixed log penalty, declared in the model and the same in every session. Its
> value is set per model, on that model's own real rows, by a target median edf over the sessions
> (`regression/calibration.py:log_penalty_for_edf`), checked by a simulation on those rows (Phase 2b's
> scenario list for the model), and the fit is reported also at a stiffer and a looser value. A target
> does not carry from one model to another (simulation: the same target gave different penalties and
> biases of opposite sign in two designs). The mPFC kernels and the history kernels are open at their
> phases (`OPEN (lead): mPFC and history penalties`, Phases 8 and 11).

> **The weak-penalty rule** (design, twenty-second round). The lead, 2026-11-23, 13:00, on the reward
> coefficient of the rate model: "I think weak penalty". At 13:02, asked whether the same weak penalty
> applies to every simple tested coefficient in every model, with a simulation check for each family
> before use: "yes". The author's calls that go with it (**call**, each): baselines are not penalized (the
> intercept, the arm effects, other nuisance categories; Greenland and Mansournia 2015, Statistics in
> Medicine 34:3133, doi 10.1002/sim.6537); the support rule reads the design only ("No rule reads the event
> counts before a fit"): a term is absent in a session when its column does not vary enough there (fewer
> than 5 units with a row on a side); each fit records the edf of each penalized simple coefficient; the
> width is one declared Normal prior SD for all models, on the coefficient in the reported unit (a 0/1
> step; one pooled SD of a continuous covariate), so the log penalty is `log(1 / SD^2)`
> (`terms.log_penalty_from_prior_sd`). **The SD's value is open** (`OPEN (lead): penalty width`; a
> simulation with SD 1, 2 and 4 is running). A penalized coefficient always has a finite maximum, so a
> session with no event on one side of it keeps it. An unpenalized term has no finite maximum only as a
> 0/1 slope or a level with no event on a side, and the no-finite-maximum rule leaves it out there; with
> few events on a side (a level whose events are in one to four units), or as a continuous slope, it is
> fitted with a large standard error. So an unpenalized simple term is a baseline or nuisance term whose
> value is not reported or tested. Built on branch `weak-penalty`, aa8f8425. The guard (**call**, the
> author's, not the lead's words): each model module calls `regression/support.py:require_tested_penalized(design,
> tested)` with the coefficients of its declared tests; it raises an error that names every one without a
> fixed penalty.

Numerical constants, each in one home; change a value only there:

- The fitter's own rules are in `analysis/regression/constants.py` (built): the Newton decrement
  tolerance `1e-8` (the value of `analysis/replay_models/fitting.py:78`), at most 100 Newton steps and 30
  halvings, the step cap, log-penalty bounds `[-10, 20]`, the penalty search's grid and stopping rule,
  `REFIT_MAX_ROUNDS` and the largest failing share `MAX_FAILING_SHARE = 0.20` (the value of
  `analysis/replay_models/contrast_aggregation.py:MAX_UNIDENTIFIED_SHARE`).
- The convention's constants that the fitter does not read are defined in the Phase 2 module that reads
  each, not in `constants.py`: the collinearity threshold R² 0.8 in `collinearity.py`, the support minimum
  of 5 units each side in `support.py`, the number of folds (5) and the longest stay piece (30 s) in
  `folds.py`, and the smallest column scale of the design check in the module that holds that check
  ([Phase 2](phase-2-convention-tools.md)).

## C6. Kinds, contrasts and the two meanings of "direct"

- "SWR" and "No SWR" are non-local events with and without strict positive-duration SWR overlap
  (`is_swr` of the event table), named and colored as in Figures 2, 3 and 5
  (`visualization/palettes.py:42`, `COLORS["No SWR"]`).
- Each model is fit per kind. Within a session the two kinds' likelihoods are separate (rate: two event
  streams on the same bins; content: disjoint events), so separate fits equal a stacked fit with
  block-diagonal covariance; the kind difference of an estimand is the difference of the two kinds'
  session estimates, reduced like any other estimand. The SWR events alone (all SWRs) are not a kind here.
- The design uses "direct" and "total" for two different things. This plan names them apart:
  - **length-adjusted / unadjusted** (content kind comparisons): with or without the event-length
    interaction of [designs.md#content-design](designs.md#content-design), evaluated at one pooled
    reference length. Both are reported.
  - **reward effect at fixed time before leaving / per visit** (rate, design decision 3): model A in the
    three-clock visit design ("direct"), and the total effect per second of in-visit at-risk time of
    Phase 5b ("total"). Both are reported, each named. There is no per-stop variant (L6, lead: "We need to
    be consistent in the repo"); a per-stop number, if ever wanted, is a question for the lead.

## C7. The fit-row record

Every stored fit of every family is a table with one row per (model, kind, session) and these columns,
written by one function (new, `analysis/regression/record.py`, Phase 2):

| Column | Content |
|---|---|
| `model`, `kind`, `animal`, `session_date`, `epoch` | identity |
| `likelihood` | `poisson`, `grouped_softmax`, `bernoulli`, `gaussian` |
| `n_rows`, `n_units`, `n_events` | what the likelihood saw (units: visits, events or units) |
| `cohort_counts` | JSON: `count_visit_context` output (added by the caller) and the rows removed from the session's fit, by cause (a missing value; a level's rows dropped; an event with no response; an event emptied by dropped rows) |
| `coef`, `vcov` | JSON arrays with `coef_names`; `vcov` is the Bayesian covariance `(H + S)^-1` |
| `log_penalty` | JSON: term -> `{"set": "fixed", "log_penalty": value}` or `{"set": "none", "log_penalty": null}`, for every term: smooths, lag functionals, slopes (each level's for a slope by a category) and categories (as built, `terms.Design.log_penalty_record`; the simple terms since aa8f8425). A per-session smooth has a fixed declared value, the same in every session; how it was set (the target median edf, the calibration's rows and commit) is stated with the model in `docs/regression_models.md`. No value is selected by REML for a per-session smooth |
| `edf` | JSON: term -> effective degrees of freedom; a level of a penalized category also by its coefficient name. A penalized simple coefficient's edf is between 0 (the penalty decides) and 1 (the data decide) |
| `status`, `newton_decrement`, `n_iter` | the convergence measure of C5 rule 3: the fit's `NewtonFit.status` (one of the six statuses of `newton.FitStatus`; only "converged" is a fit), or `not fitted` when no fit was tried |
| `left_out` | JSON: coefficient name -> why it is inactive in this session, a code of `support.Removed` (`support`, `too_small`, `with_level`, `no_finite_maximum`); its `coef` and `vcov` entries are NaN |
| `left_out_round` | JSON: coefficient name -> the refit round in which it left, for `no_finite_maximum` |
| `support` | JSON: coefficient name (a column of a simple term, penalized or not; a smooth term's name when the term is too small in the session) -> {`n_low`, `n_high`, `passes`, `reason` (a `Removed` code, or empty), `size`} from the support rule: units with a row on each side (a level: with a row of the level), counted on the rows the fit used |
| `collinearity` | JSON: term -> {`r2`, `vif`} or {`observed`, `worst`, `estimate`} |
| `basis` | JSON: term -> {`kind`, `knots`, `degree`, `range`, `constraint`, `clamp`} |
| `standardization` | JSON: covariate -> {`sd`, `ddof`: 0, `reference`} |
| `table_sha256` | the canonical tables the fit read (`reported.values.file_record` hashes) |
| `no_fit` | empty when a fit is kept; otherwise a code of `record.NoFit`: `no_rows` or `no_events` (no fit was tried; `status` is `not fitted`), or `along_the_intercept`, `along_a_penalty`, `too_many_rounds`, `other_status` (a fit was tried and not kept: `status` and the counts are the last fit's, the coefficients are NaN, and `left_out` holds what the refit had left out before it ended) |

A record whose `status` is not "converged", a failing support entry for a declared term, or a declared term
over the collinearity threshold is kept and flagged; the summary reads only passing rows and states the count
left out. A session with a term in `left_out` is kept for its other terms and reported with that term
missing (lead, 2026-11-21). A term with several columns and no penalty leaves a session whole. The
causes are closed sets of codes, so that a summary counts sessions by cause; `record.describe` (removed 2026-11-24, with the parquet storage; only tests used it) gave the
words.

## C8. The declared family of tests

The family is declared before any fit of Phase 5a and is not changed after a fit is seen. Each member is a
scalar estimand, tested for SWR, for No SWR and for their difference. The design's list (section 4.5):

| Question | Scalar estimand | Model (phase) |
|---|---|---|
| Q1 | log rate 2 to 4 s after arrival minus log rate in the first second | visit rate, smooth of time since arrival (5a) |
| Q2 | log rate ratio, rewarded against omitted last outcome | model A (5a) |
| Q3a | log share ratio, current arm against another leaf arm | content (6a) |
| Q3b | log share ratio, previous arm against another leaf arm | content (6a) |
| Q4a | change of the current arm's log share over one pooled SD of its reward fraction, read from the role's reward-fraction smooth | content (6a) |
| Q4b | slope of another arm's log share on its value (per pooled SD) | content (6a) |
| Q5a | log share ratio of an arm 20 trials since its last visit against one 3 trials since | content, recency smooth (6a) |
| Q5b-var | slope on the belief variance (per pooled SD), in the one uncertainty model (recency smooth, smooth reward fraction, trial number, and the sample count beside it) | content, uncertainty model (6a) |
| Q5b-n | slope on the sample count (log of one plus the visits to the well with a known outcome earlier in the epoch; per pooled SD), in the same model | content, uncertainty model (6a) |
| Q6 | log share ratio, next arm against an arm that is neither current nor previous | content with a next-arm role (7) |
| Q7a | log rate in the last 0.25 s before the last poke-out minus log rate 1 to 2 s before | visit rate, smooth of time until leaving (5a) |
| Q7b | the same for the current-arm log share | content (6a) |

A curve-derived estimand is a linear functional of the coefficients (the mean of the smooth over one range
of the clock minus its mean over another, on a 0.01 s or 0.01-log-unit grid), so its session variance is
exact.

**Settled (lead, 2026-11-20).** L1 ("Ok") settled the list; after the uncertainty simulation the lead dropped
the reward variability ("Yes"), so the family is **twelve estimands, each for SWR, No SWR and their
difference: 36 tests**. Q5b has two members, the belief variance and the sample count, tested together in one
model with the trial number and the smooth reward fraction ("Yes"). Q4a is read from the smooth reward
fraction ([designs.md#content-design](designs.md#content-design)). (The design's "ten estimands ... 30 tests"
predates the Q3, Q4 and Q7 splits and the uncertainty changes.) The code reads the list from one place (new
`analysis/replay_models/declared_tests.py`, Phase 2: it names roles and visits, so it is outside the fitter package), so the count is not hard-coded anywhere else. If Phase
6a's gate changes how a member is fitted, the member keeps its id and the change is made before any real
fit (the family is still declared before the fits).

## C9. Where new fits are stored

New fits go under `Processed-Data/replay_content/models_convention/<family>/` (`rate`, `rate_total`,
`content`, `content_robustness`, `choice`, `stay_leave`, `ripple_power`, `simulations`,
`descriptive` for model-free reference curves), one parquet of
fit-row records per model plus a `manifest.json`, each written with `tables/io.py:215` (`write_atomically`). mPFC bundles keep their own directory layout
(`analysis/pfc/bundle_files.py`) with the record of C7 added. Old fits are not overwritten; they stay
on disk, unread once their family is removed (C13).

## C10. Rebuilding reported values and figures

After a phase's fits are written and its code is committed:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu $PY scripts/reported/summarize.py --summary <events|behavior|mpfc>
PYTHONPATH=src JAX_PLATFORMS=cpu $PY scripts/reported/emit_reported_values.py
```

- `summarize.py` writes into `paper/reported/` only from a checkout whose files equal the commit; commit
  first, or use `--output-dir` for a trial build.
- Values read from outputs made on older tables become pending. The lead accepts that ("pending is fine");
  build no tool to avoid it, and do not re-run an unrelated family only to clear a pending note.
- A new value tags a question in `src/continuum_swr_replay/reported/questions.py` and, when it has one, a
  figure (`questions.py:93`, `FIGURES`). Q1 to Q7 are added as new Part 3 questions in Phase 5a (call:
  ids `3.11` to `3.17`); an old 3.x question is marked `dropped` (reason: superseded by its new id) in the
  phase that removes its last value.
- A number in a doc is written inside a `<!--v:<summary>.key-->…<!--/v-->` marker; never typed.
- Before removing any value, list the macros `paper/main.tex` cites (`make reported-values` prints the
  count; grep the macro names in `paper/*.tex`). A cited value is removed only with the lead's yes on the
  text change.

## C11. GPUs, long runs and the other session

- Rate, content, choice, stay or leave and ripple power fits run on CPU (measured in the design: about
  0.08 s per two-clock rate fit and 0.03 s per content fit, one session). Parallelize over sessions with a
  process pool only when the measured run exceeds about 30 minutes.
- Before any GPU job: read `nvidia-smi` on zephyr and on breeze (`ssh breeze.cin.ucsf.edu nvidia-smi`), use
  only cards with no other process, and message the other session (the one that owns
  `swr-restricted-models` / `swr-trial-folds`; it plans a 242-epoch run on 16 cards) with the cards and the
  expected hours. If no card is free, wait; never stop another job.
- Never stop or relaunch a running batch without agreeing first.
- A job longer than five minutes runs inside `tmux` with a named session (`csr-<phase>-<what>`), after a
  one-session smoke test that records wall time and peak memory (`/usr/bin/time -v`).

## C12. Layering of the new package

`src/continuum_swr_replay/analysis/regression/` knows nothing of the task: arrays in, fits out. It imports
only the foundation, `analysis.constants` and `analysis.shared_statistics`, and any `analysis/` package may
import it (the event-evidence code of Figure 2 included, for the shared fold function). Phase 1 adds this
rule to `tests/repository/test_import_layering.py` (beside `EVENT_DEFINITION_MAY_IMPORT`,
`test_import_layering.py:91`) and to the layer list in `docs/README.md:93-133`. Task-aware code (which
columns, which rows) stays in `replay_models/`, `event_tables/` and `pfc/`.

## C13. No legacy left behind

Lead (2026-11-20): "Let's make sure to cleanup any legacy code." Every phase obeys these rules, and the
phase that replaces a thing removes it ([legacy-inventory.md](legacy-inventory.md) lists what and where):

- **No compatibility shims.** No wrapper, adapter or alias that keeps an old call working.
- **No flag or parameter that selects the old path** (no `--legacy`, no `inference="legacy"`, no
  `fitter="statsmodels"`, no `--stop-cohort` choice among old cohorts). A sensitivity variant the design
  keeps is a named model with its own record, not a switch back to old code.
- **No allowance for older files.** No "skip when the column or field is absent", "built before", "older
  bundle" or version-fallback branch for tables or bundles made before a change. Tables are rebuilt and fits
  are refit; an old file fails loudly (`require_columns`, a version check that raises).
- **No re-export of a moved or renamed name.** Callers import from the new home.
- **A renamed column is renamed everywhere in the same phase** (builders, registry, readers, tests, docs);
  no deny-list of the old spelling is added for it.
- **Nothing new under `scripts/exploratory/`.** A check worth keeping is a test or a committed stored
  analysis; a one-off stays in the phase's scratch directory.
- **Zero matches at the end of the phase.** Every symbol, file and column the phase removes has no match in
  `git grep -n -w <name> -- src scripts tests docs` (dated records exempt: `docs/history/`, dated
  `docs/design/<YYYY-MM-DD>-*.md`, `docs/design/model_history.md`, `paper/reported/removed_part*.md`).
- **Record every removal**: a row in `docs/history/2026-module-moves.md` ("removed: replaced by ..."), and
  `docs/regression_pipeline_map.md` regenerated with `scripts/checks/write_pipeline_map.py`.
- **Old against new before removal.** A phase that removes a family first writes, for the lead, each old
  reported value beside its replacement with the reason for the change (cohort, clock, scaling, design),
  read from the two summary files (keep a copy of the old `paper/reported/behavior.json` or `mpfc.json` in
  the phase scratch directory before rebuilding). The old fits are the only baseline, so this comes first.
- **A correctness check never keeps dead code alive.** Where a test compares the new fitter with an old one
  that is being removed (Phase 1's check 1 and check 3), the phase that removes the old fitter first commits
  the old fitter's coefficients on the same small data as a fixture under
  `tests/analysis/regression/data/legacy_fitters/` (CSV, with the script line that made them in a README
  beside it), switches the test to the fixture, then deletes the old fitter.

The guards:

- **Cited values need the lead's yes.** A value `paper/*.tex` cites is removed only after the lead approves
  the text change, asked at the boundary of the phase that removes it (C10), not saved for the end.
- **An analysis this plan does not replace is not removed without asking.** The lead's standing rule: "an
  analysis nothing cites is a question for the lead, not a removal." Those are the ASK items of the
  inventory; they go to the lead as one list at the latest in Phase 12.
- Stored outputs in Processed-Data are never deleted by this plan; they are only no longer read.

## C14. One of everything

Lead (2026-11-20): "We need to be consistent in the repo." In every model family, summary and figure,
including the mPFC models and the ContFrag event sources:

- **one unit**: the well visit (first poke-in to final poke-out). No per-stop, per-trial-segment or
  per-bout variant is kept beside a per-visit one;
- **one event set**: the events that start inside a well visit (C3);
- **one selection function**: `select_visit_context` (C4), at event, stop, event-arm and bin grain;
- **one fitter**: `analysis/regression/`, in JAX for every family after [Phase 2c](phase-2c-jax-fitter.md)
  (lead, 2026-11-23); no NumPy solver beside it and no separate twin for the mPFC units;
- **one simulation engine**: `analysis/regression/simulation.py` ([Phase 2b](phase-2b-simulation-engine.md)),
  which replaces only the outcome column and runs the real fit, tests and reduction;
- **one set of clocks**: time since the first poke-in, time until the final poke-out, time until the end of
  the current immobile stretch (Phase 3); no stop-start or trial-arrival clock;
- **one fold rule, one convergence rule, one support rule, one standardization, one inference** (C5).

Where a phase finds code that uses another unit, another filter, another clock or another fitter, it moves
it onto these in the same phase, or, if the thing is outside the phase's family, lists it in the PR and in
[legacy-inventory.md](legacy-inventory.md) for the lead.

