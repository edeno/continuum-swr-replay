# Phase 6a — Content on nine arms by arm role (Q3, Q4, Q5, Q7b)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#content-design)

Branch: `role-content-models` from master after Phases 2 and 3. CPU only. Can run beside Phase 5a.

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
  keys of `paper/reported/behavior.json`; `docs/behavior_models.md`.
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
  `select_visit_context` (default) -> nine candidate rows per event -> roles (current, previous,
  other_leaf, trunk; exclusive; the previous role only when it differs from the current) -> drop rows with a
  missing covariate in the cohort step (counted per kind) -> terms:
  `Category(arm)` (arm fixed effects) + `Category(role)` + `Slope(reward_fraction, by=role)` +
  `Slope(value, by=role)` + `Smooth(log_recency, log 1, log 200)` on leaf rows + never-visited indicator +
  `Smooth(u, 0, 3)` for the current role (Q7b). Two variants per model: length-unadjusted, and
  length-adjusted (every role term and role slope interacting with `log(duration) - L0`, `L0` the pooled
  median, stored). Three uncertainty models, each adding one measure's role slopes (belief variance; the
  two Phase 3 columns, or built here from the port experiences if Phase 3 shipped without them).
- **Nine-arm simulation first** (design 4.2: "the nine-arm design goes through the same simulations
  before it is used"): stored analysis `scripts/replay_models/content/simulate_role_content.py` (new, from
  `S/proto_content/sim_lib.py` and `content/simulation.py`) on the real nine-arm design of every session:
  (a) opposite true slopes by role, where a pooled slope returns about zero and the role slopes recover
  both; (b) recency-only truth, where the uncertainty slope's false-positive rate is near nominal with the
  uncentered smooth and above it with the shipped centered spline; (c) a curved recency truth, which a
  straight-line recency term misfits; (d) a planted uncertainty slope, recovered; (e) a planted rise of the
  current-arm share toward leaving, recovered on `u` and not on the role slopes. Twenty data sets each;
  output `models_convention/simulations/role_content.parquet`. Do not fit the real data until (a) to (e)
  pass.
- **Fit** through `analysis/regression/sessions.py` with the grouped softmax: support rule per role slope
  and role category; one pooled REML selection per smooth, model and kind; collinearity per term (recency
  against uncertainty is the known concern, design 2.3); records.
- **Estimands** (`analysis/replay_models/content/role_content_statistics.py`, new): Q3a (current role
  minus other_leaf), Q3b (previous minus other_leaf), Q4a (reward-fraction slope on the current role), Q4b
  (value slope on the other_leaf role), Q5a (recency smooth at log 20 minus at log 3, other_leaf rows), Q5b
  (the uncertainty slope on other_leaf, one per measure), Q7b (mean of the current role's `f_u` over
  `[0, 0.25] s` minus over `[1, 2] s`), each per kind and difference, length-adjusted and unadjusted, with
  `estimand_summary`. A term over the collinearity threshold is reported jointly with its partner (for
  example, recency and uncertainty) and its single value is not reported (C5).
- **Runner** `scripts/replay_models/content/fit_role_content_models.py` (new), registered in `run_refits.py`.
- **Summary values** (`reported/behavior/role_content.py`, new, called from
  `reported/behavior/__init__.py`): per estimand, kind and difference, as ratios of shares with intervals,
  animals with the sign, day-level p, sessions kept and left out, cohort counts; tagged with the new
  questions of Phase 5a.
- **Docs**: `docs/behavior_models.md` (the role design, nine arms, recency form, the three uncertainty
  measures declared before the fit, the length-adjusted and unadjusted comparisons), with value markers;
  the model table.

## Deliberately not in this phase

- The count and Dirichlet-multinomial estimators, and deleting the within-arm comparison (Phase 6b).
- Q6 and the next-arm role (Phase 7).
- Removing the old content ladders, runners and values (Phase 12).
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
| `tests/reported/behavior/test_role_content_values.py::test_values_and_joint_reporting` | an estimand over the collinearity threshold gives a joint value and no single one |

## Fixtures

Synthetic nine-arm events from `content/simulation.py` (`sample_arm_design`, `simulate_counts`) with fixed
seeds; the synthetic visit helpers of Phase 3 for `u`.

## Verification and runtime

- Simulated first (tests, then the stored nine-arm simulation over the real designs), then five sessions,
  then 242.
- The design measured about 0.03 s per content fit and session; a pooled selection with two smooths (25 grid
  points plus Newton steps) over 242 sessions is minutes; three uncertainty models, two length variants and
  two kinds make about twelve selections. Time the smoke test; run the full fits in `tmux`
  (`csr-6a-content`).
- Compare with the model-free references: Q3a's direction and Q4a's sign (negative for SWR on the current arm
  in the design's check) against the within-arm slopes; Q7b against the model-free current-arm share by time
  until leaving. Compare the old pooled-slope values (baseline) with the role slopes and explain each
  difference by role in the PR.

## Rebuilt

`models_convention/content/`, `descriptive/role_content.parquet`, `simulations/role_content.parquet`;
`paper/reported/behavior.json` and the emitted files.

## Exit criteria

The nine-arm simulations pass before any real fit; the full fits are stored with records; Q3, Q4, Q5 (three
uncertainty measures) and Q7b values are in the summary, with any collinear pair reported jointly.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
