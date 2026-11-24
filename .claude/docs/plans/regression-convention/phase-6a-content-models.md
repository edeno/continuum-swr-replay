# Phase 6a — Content on nine arms by arm role (Q3, Q4, Q5, Q7b)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#content-design)

Branch: `role-content-models` from master after Phases 2b and 3 (the simulation engine on master). It
runs on the present fitter, beside Phase 5a and beside Phase 2c, the JAX port (Figure 4 does not wait
for the port; lead, twenty-third round). CPU only.

**Inputs to read first:**

- The design, sections 2.2, 2.3, 3 (Q3, Q4, Q5, Q7), 4.2 (roles, recency, nine arms, uncertainty,
  standardization), 4.7, and decisions 1, 5 and 6.
- Phase 2's `analysis/regression/` and the grouped softmax likelihood of Phase 1.
- `src/continuum_swr_replay/analysis/replay_models/content/cohort.py:263` (`select_content_cohort`),
  `:99` (`prepare_canonical_fit_request`), `:511` (`content_cohort_record`): the current cohort step to
  replace with the selection function.
- `src/continuum_swr_replay/analysis/replay_models/content/session_fits.py:125` (`fit_model_per_session`),
  `:664` (`fit_contrast_ladder`), `:837` (`fit_content_ladder`); `fitting.py:677`
  (`fit_conditional_poisson`).
- `src/continuum_swr_replay/analysis/replay_models/model_specs.py:175-363` (content ladders), `:1795-1945`
  (`leaf_only_predictors`, `restrict_to_leaf_arms`, `leaf_only_cohort`), `:2072` (`add_arm_fixed_effects`).
- `src/continuum_swr_replay/analysis/behavior/covariates/candidate_arm.py:116-300`
  (`attach_candidate_arm_covariates`: `is_current_arm` `:190`, `is_prev_arm` `:191`, `is_next_arm` `:192`,
  recency, reward fraction, value, belief variance), `behavior/covariates/history/visit_recency.py:23`
  (the within-event centering that is not used), `behavior/design_constants.py:66-73` (recency knots and
  bounds), `behavior/covariates/history/value_block.py:206` (`attach_arm_uncertainty`).
- `src/continuum_swr_replay/analysis/replay_models/content/simulation.py:139` (`sample_arm_design`),
  `:230` (`simulate_counts`), `:306` (`simulate_with_event_covariates`): the repository's content simulator.
- `reported/behavior/content.py` (`content_contrast_values` `:326`, `leaf_values` `:607`) and the content
  keys of `paper/reported/behavior.json`; `docs/models_table.md:367-457` (content arm and patch ladders) and
  `docs/design/content_model_design.md` (the design being replaced).
- Scratch: `S/proto_content/sim_lib.py`, `run_sim.py`, `diag_centering.py`, `calibrate.py`,
  `S/verify_uncertainty/s06_null.py`, `s01_covariate.py`, `S/verify_content/42_role_slopes.py`,
  `41_value_recency.py`, `S/my_checks/role_slopes_check.py`, `departure_timecourse.py`,
  `other_arms_check.py`.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct),
[C7](shared-contracts.md#c7-the-fit-row-record), [C8](shared-contracts.md#c8-the-declared-family-of-tests),
[C9](shared-contracts.md#c9-where-new-fits-are-stored), [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

**Designs referenced:** [content design](designs.md#content-design), [likelihoods](designs.md#likelihoods)
(grouped softmax), [collinearity](designs.md#collinearity-measures), [support rule](designs.md#support-rule).

**Decisions this phase applies** (settled 2026-11-20): the declared family of twelve estimands, 36 tests
(C8; Q5b-var and Q5b-n); after the uncertainty simulation (lead, 20:39, "Yes" twice) the reward variability
is dropped, the reward fraction gets a smooth per role, and the belief variance and the sample count are
tested together in one model with the trial number
([designs.md#content-design](designs.md#content-design)); L4
(the event set: events that start inside a visit, C3; a `spans_visit_end` event's current arm is the visit's
well's arm), ASK-1 (lead: "Move them": the ContFrag SWR- and MUA-sourced events are two more event sources of
this runner), C14.

**Decided at this phase** (design, twenty-fourth round): the lead's words, 2026-11-23, 18:17: "do option
B". The rest of this paragraph is the author's reading and text, not the lead's words. The recency curve
and the never-visited indicator cover the other leaf arms only; the role terms carry everything for the
current and previous arms (every current-arm row has recency 1 trial and every previous-arm row 2); Q5a
reads the curve at its two points among other leaf arms. The author proposes that the paper says that Q3a
and Q3b compare with an average other leaf arm, and that in this task "current arm" cannot be separated
from "most recently visited arm". What was shown before the decision includes two limits (simulation,
`content_decisions/recency_options/REPORT.md`): with recency alone planted the current-arm term comes
back +1.12 to +1.17, as a true role effect would, so neither option tells the two apart; and Q5a (planted
+0.50) is +0.50 to +0.51 at prior SD 8 with a flexible curve, +0.64 to +0.67 at SD 2 and +0.33 to +0.40 at
the stiff penalty. The setting `recency_rows`, built to inform the decision, is removed.

**Questions for the lead at this phase** (calls of the plan's author, not decided; ask before the first
real fit and state the answers in `docs/regression_models.md`):

- Q4a, first declared as a slope on the reward fraction, is read as the change of the current role's
  reward-fraction smooth over one pooled standard deviation of the fraction, `f(m + sd/2) - f(m - sd/2)`
  ([designs.md#content-design](designs.md#content-design)). The lead may prefer another summary of the
  curve.
- The trial number in the epoch is the number of visits the animal has arrived at by the event start
  (Phase 3's `n_visits_arrived`), not the trials completed (`completed_trial_id`).
- The declared penalty of each per-session smooth term (gate item 2 below): the rule is decided (design,
  seventeenth round); the values found on this model's rows, with the targets, go to the lead with the
  gate's results.
- (**call**, 2026-11-23) The collinearity rule of a tested coefficient is measured on the penalized problem:
  the information at the fit plus the smooths' declared penalties, the weak penalties left out
  (`role_content.tested_coefficient_collinearity`). On the likelihood's information alone `role[current]`
  failed in 87 % of the recency simulation's session fits (R^2 0.9999994; `content_decisions/
  recency_options/REPORT.md`, 40 sessions under option B) while the fits recovered it (the content
  validation's part 2, a different run on 40 sessions, found 90 %):
  each by-role smooth's sum-to-zero constraint is computed on the scale rows, so in a session that does
  not see one of its basis functions the curve's columns hold the role's indicator, and only the curve's
  penalty fixes that direction (the weak penalties do not). Evidence and the remaining failures (arm
  effects concentrated on few arms): `content_validation/REPORT.md` in the scratch folder, part 2.

**Gate before any real fit: the uncertainty simulations.** Two semi-synthetic studies on the real data
structure, in scratch (`/cumulus/edeno/continuum-swr-replay/.claude/research/2026-11-20-regression-prototypes/proto_uncertainty_measures/`):

1. The first is done (`REPORT.md`, `sim_main_key_table.csv`, `sim_control_table.csv`; six leaf arms, 100 data
   sets per truth). It gave the decisions above: the belief variance is about 90 % predicted by recency
   within a session (VIF 30 to 43) and its test is calibrated only with a flexible, unshrunk recency curve;
   the reward variability cannot be told from reward-fraction curvature; the sample count is a drift with the
   trial number unless the trial number is in the model; with all three and the trial number every
   false-positive rate is 1 to 9 % except the reward variability under curvature; the belief variance's power
   is 30 to 50 % at 0.25 per SD with a t test over five animals. (Simulation results, not summary values.)
2. The repeat with the shared fitter's REML-penalized recency smooth is done (design, thirteenth and
   fourteenth rounds; scratch `REPORT.md`, same folder; simulation): a penalty chosen from the data makes
   each session's curve a straight line when a session has about 140 events of a kind, and the
   belief-variance test is then wrong (61 % and 68 % false positives). A small fixed penalty, or a small
   basis with no penalty, keeps that test correct (false positives 0 to 7 %) with low power (23 to 50 %
   at an effect of the size of the recency effect's linear part). The lead kept one fit for each session
   (fourteenth round).
   The rule that follows (design, seventeenth round): the uncertainty model's recency curve, and every
   other per-session smooth of this phase, is a P-spline at one declared fixed log penalty, the same in
   every session. Each value is found on this model's own real rows by `calibration.log_penalty_for_edf`
   at a target median edf, and checked by this phase's scenario list: the recency penalty must be weak
   enough that the belief-variance test keeps its size (3 to 7 %). (Simulation, recency: log penalty -4,
   about 4.3 effective parameters, was cleaner than -2, about 3.0, which biased the slope by +0.05 to
   +0.10.) Do not reuse a target from the rate model. Write the values, their targets and the stiffer
   and looser values into the phase's PR and `docs/regression_models.md` before the first real fit.

Both results go to the lead before any real fit of this phase. The statements that must be written before the
fits (C5's rule for a term that fails the collinearity rule by design): the belief variance is reported
beyond recency, with its VIF, the recency curve as fitted, and the simulated false-positive rate and power; a
null is reported as "no evidence beyond recency; the design cannot detect an effect of this size", with the
simulated power beside it. The phase commits the studies as its slow simulation test (task "Uncertainty
identifiability test" below), since scratch is not kept.

## Tasks

- **Baseline**: save the content keys of `paper/reported/behavior.json`. Compute and store the model-free
  references (stored analysis `scripts/replay_models/content/describe_role_content.py`, new, method from
  `S/my_checks/role_slopes_check.py` and `departure_timecourse.py`): per kind and animal, the mean share
  on the current, previous, other-leaf and trunk arms; the within-session, within-arm slope of share on
  reward fraction and on value by role; the current-arm share in bins of time until leaving. Output
  `models_convention/descriptive/role_content.parquet`.
- **Recency check** (decision 6: "the stage 1 comparison stays as a check"): fit the role design with
  absolute recency (the smooth of log trials since the last visit) and with order-only recency (the rank of
  the arm's recency among the event's leaf arms, as a category); compare the summed held-out score over
  sessions on trial folds (Phase 2's fold function) and report. Absolute recency stays the design either
  way; the result is stored (`models_convention/content/recency_check.parquet`) and stated in the PR.
- **Design** (`analysis/replay_models/content/role_content.py`, new), per kind, from the event-arm table:
  `select_visit_context` (default `STARTS_IN_WELL_VISIT`) -> nine candidate rows per event -> roles (current, previous,
  other_leaf, trunk; exclusive; the previous role only when it differs from the current) -> drop rows with a
  missing covariate in the cohort step (counted per kind) -> terms:
  `Category(arm)` (arm fixed effects) + `Category(role)` + `Smooth(reward_fraction, 0, 1, 6, by=role)` +
  `Slope(value, by=role)` + `Smooth(log_recency, log 1, log 200)` and the never-visited indicator on the
  other-leaf rows only (design, twenty-fourth round: the lead, 2026-11-23, 18:17, "do option B") +
  `Smooth(u, 0, 3)` for the current role (Q7b). The three `Smooth` terms are P-splines at declared
  fixed log penalties (gate item 2); their ranges stay. Two variants per model: length-unadjusted, and
  length-adjusted (every role term and role slope interacting with `log(duration) - L0`, `L0` the pooled
  median, stored). One uncertainty model: the main design plus, on the leaf roles, the belief variance,
  the sample count (Phase 3's column) and the trial number in the epoch (Phase 3's `n_visits_arrived`), each
  a pooled-SD slope per role, with the recency curve fitted as the gate decides.
  Each simple tested coefficient (a role level that a declared test reads, a role slope, the uncertainty
  slopes) has the declared weak penalty (design, twenty-second round: the lead, 2026-11-23, 13:02, "yes" to the same weak penalty on every simple tested coefficient; width `OPEN (lead): penalty width`). The arm fixed effects and other nuisance
  categories are baselines and are not penalized (**call**). The nine-arm scenario list below checks the
  penalty before the fit is used.
- **Nine-arm simulation first** (design 4.2: "the nine-arm design goes through the same simulations
  before it is used"): a scenario list for the simulation package (Phase 2b),
  `replay_models/content/role_content_scenarios.py` (new; scenarios from `S/proto_content/sim_lib.py`),
  run by Phase 2b's runner on the real nine-arm design of every session:
  (a) opposite true slopes by role, where a pooled slope returns about zero and the role slopes recover
  both; (b) recency-only truth, where the uncertainty slope's false-positive rate is near nominal with the
  uncentered smooth and above it with the shipped centered spline; (c) a curved recency truth, which a
  straight-line recency term misfits; (d) a planted belief-variance slope and a planted sample-count slope,
  each recovered in the one uncertainty model; (e) a planted rise of the
  current-arm share toward leaving, recovered on `u` and not on the role slopes. Twenty data sets each;
  output `models_convention/simulations/role_content.parquet`. Do not fit the real data until (a) to (e)
  pass.
- **Uncertainty identifiability test** (committed form of the gate studies): a slow test
  `tests/analysis/replay_models/content/test_uncertainty_measures_identifiable.py` and a scenario list for
  the simulation package (Phase 2b), `replay_models/content/uncertainty_scenarios.py` (new; scenarios
  from the scratch study's scripts: `sim.py`, `build_measures.py`, `check_measures.py` and what
  `REPORT.md` names; its own fitting code is not copied), run by Phase 2b's runner with the shared fitter and the uncertainty model as it will be fit to the real data (the
  recency curve as the gate decides). Truths: recency only; the belief variance; the sample count; a drift
  with the trial number; a curved reward-fraction effect. Asserts: under each truth without a measure's
  effect, that measure's slope is significant in 3 to 7 % of data sets (100 in the stored analysis, fewer in
  the test with a wider band); a planted measure is recovered without bias; the stored analysis records the
  power of each test at the planted size, which the summary's description of Q5b-var and Q5b-n quotes from
  `models_convention/simulations/uncertainty_measures.parquet`. A planted curvature of the reward fraction
  must not appear on either measure (the reason the variability measure was dropped).
- **ContFrag event sources** (ASK-1): the same role design fit to the SWR- and MUA-sourced ContFrag events
  of the rebuilt ContFrag event-arm tables, as two more event sources (`event_source` in the fit-row
  record; each source is one stream, without an SWR and No SWR split, call). Their values are a
  supplementary comparison (Phase 7 builds the comparison values), not members of the declared family.
- **Fit** through `analysis/regression/sessions.py` with the grouped softmax: support rule per role slope
  and role category, read from the design (units with a row on each side; under the softmax the same
  counts as the events with a candidate row; a role column made inactive drops that role's rows from the
  session's fit, Phase 2);
  sessions without a finite maximum handled by Phase 2's rule (the simulated content model had 4 of 242
  such sessions; simulation); collinearity per term (recency against uncertainty is the known concern,
  design 2.3); records. The plain penalized fit at the declared penalties (no bias-reduced fit; lead,
  2026-11-23); no penalty is chosen from the data, and no selection runs.
- **Storage order of the design** (in the design builder, `role_content.py`): store each session's design
  in column (Fortran) order if a timing confirms it. A reviewer measured one weighted Gram product at
  7.0 ms in column order against 9.8 ms in row order on a synthetic session of 40,000 rows and 56 columns
  (a rate-sized design, not verified by us; the content design has fewer rows). Timing step: on the
  smoke-test session, time 20 fits of the main model with the design in each order (median, the same
  thread count), check that the coefficients agree to 1e-12 relative, and keep column order only if it is
  faster; state both timings in the PR.
- **Estimands** (`analysis/replay_models/content/role_content_statistics.py`, new): Q3a (current role minus
  other_leaf), Q3b (previous minus other_leaf), Q4a (the current role's reward-fraction smooth, `f(m + sd/2) -
  f(m - sd/2)`, designs.md), Q4b (value slope on the other_leaf role), Q5a (recency smooth at log 20 minus at
  log 3, other_leaf rows), Q5b-var and Q5b-n (the belief-variance and sample-count slopes on other_leaf, both
  from the one uncertainty model), Q7b (mean of the current role's `f_u` over `[0, 0.25] s` minus over `[1, 2]
  s`), each per kind and difference, length-adjusted and unadjusted, with `estimand_summary`. The recency
  curve the paper shows (and Figure 4g) is read from the uncertainty model, the one that also holds the belief
  variance, the sample count and the trial number: read from the model without them, the curve absorbs a true
  belief-variance effect or a drift (simulation, bias 0.3 to 0.9). Its supported range in the simulation was 5
  to 80 trials since the last visit; at 150 only about a quarter of the sessions contribute. Show it over that
  range. Q5a's reference of 3 trials lies below that range: a question for the lead before the first real fit.
  A term over the collinearity threshold is reported jointly with its partner, except a declared term that
  fails the rule by design: the belief variance is reported beyond recency as C5 says, with its simulated
  power.
- **Runner** `scripts/replay_models/content/fit_role_content_models.py` (new; event sources non-local,
  ContFrag SWR, ContFrag MUA), registered in `run_refits.py`.
- **Summary values** (`reported/behavior/role_content.py`, new, called from
  `reported/behavior/__init__.py`): per estimand, kind and difference, as ratios of shares with intervals,
  animals with the sign, day-level p, sessions kept and left out, cohort counts; tagged with the new
  questions of Phase 5a.
- **Docs**: `docs/regression_models.md` (the role design, nine arms, recency form, the smooth reward fraction,
  the uncertainty model with its two measures and the trial number, the recency penalty rule and the power
  statement, all written before the fit, the length-adjusted and unadjusted comparisons), with value markers;
  the model table.

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-6a](legacy-inventory.md#phase-6a)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- The count and Dirichlet-multinomial estimators, and deleting the within-arm comparison (Phase 6b).
- Q6 and the next-arm role (Phase 7).
- Removing what Phase 6b or 7 still calls (the Dirichlet-multinomial and estimator-comparison inputs,
  `fitting.py`'s conditional fitters used by choice): the inventory names the phase.
- FDR (Phase 9).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/content/test_role_content.py::test_roles_exclusive_and_complete` | every candidate row has one role; previous only when it differs from current; trunk arms get the trunk role and no leaf covariates |
| `test_role_content.py::test_pooled_slope_hides_opposite_roles` | simulated opposite role slopes: a pooled slope within 2 SE of 0, the role slopes each within 2 SE of truth |
| `test_role_content.py::test_recency_only_truth_uncertainty_null` | recency-only truth: the uncertainty slope significant in 3-7 % of 100 data sets with the uncentered smooth |
| `test_role_content.py::test_centered_spline_leaks_to_uncertainty` | the same truth with the within-event centered spline gives a rate well above nominal (the reason for the change, kept as a regression test) |
| `test_role_content.py::test_curved_recency_needs_smooth` | a straight-line recency term leaves a biased uncertainty slope on a curved truth; the smooth does not |
| `test_role_content.py::test_current_share_toward_leaving_on_u` | a planted rise on `u` is recovered by `f_u` and not by the role slopes |
| `test_role_content.py::test_length_adjusted_at_reference` | with a planted length interaction, the adjusted per-kind estimate equals the truth at `L0` |
| `test_role_content_statistics.py::test_share_ratio_from_softmax` | `exp(beta_current - beta_other)` equals the ratio of fitted shares within an event |
| `tests/analysis/replay_models/content/test_uncertainty_measures_identifiable.py` (slow) | in the one uncertainty model: false positives of each measure at 3 to 7 % under recency-only, the other measure, trial-number drift and reward-fraction curvature truths; each planted measure recovered |
| `test_role_content.py::test_spans_visit_end_events_in_cohort` | events labeled `spans_visit_end` are in the cohort with the visit's well's arm as current |
| `tests/reported/behavior/test_role_content_values.py::test_values_and_joint_reporting` | an estimand over the collinearity threshold gives a joint value and no single one |

## Fixtures

Synthetic nine-arm events from `content/simulation.py` (`sample_arm_design`, `simulate_counts`) with fixed
seeds; the synthetic visit helpers of Phase 3 for `u`.

## Verification and runtime

- Simulated first (tests, then the stored nine-arm simulation over the real designs), then five sessions,
  then 242.
- The design measured about 0.03 s per content fit and session; a pooled selection with two smooths (25 grid
  points plus Newton steps) over 242 sessions is minutes; no selection runs under the declared-penalty
  rule, so the cost is one fit per session, model, variant and kind, plus one calibration per smooth and
  kind (about 18 passes each) and the scenario lists. Time the smoke test; run the full fits in `tmux`
  (`csr-6a-content`).
- Compare with the model-free references: Q3a's direction and Q4a's sign (negative for SWR on the current arm
  in the design's check) against the within-arm slopes; Q7b against the model-free current-arm share by time
  until leaving. Compare the old pooled-slope values (baseline) with the role slopes and explain each
  difference by role in the PR.

## Rebuilt

`models_convention/content/`, `descriptive/role_content.parquet`, `simulations/role_content.parquet`;
`paper/reported/behavior.json` and the emitted files.

## Exit criteria

The lead has seen both uncertainty simulations (the first and its repeat with the shared fitter's recency
penalty) and the recency rule and power statement were written before the first real fit; the committed
identifiability test passes; the nine-arm simulations pass before any real fit; the full fits are stored with
records; Q3, Q4 (from the smooth reward fraction), Q5a, Q5b-var, Q5b-n and Q7b values are in the summary, the
belief variance reported beyond recency with its simulated power, and any other collinear pair reported
jointly.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
