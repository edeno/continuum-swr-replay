# Phase 7 — Content and the next choice (Q6), stay or leave, ripple power

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#content-design)

Branch: `choice-stay-power-convention` from master after Phase 6a. CPU only.

**Inputs to read first:**

- The design, sections 3 (Q6: "null once behavior is in the model; next arm equals previous arm in over
  80 % of events"), 4.2, 4.3, and section 7's rows for choice, stay or leave and ripple power.
- Phase 6a's role design (`analysis/replay_models/content/role_content.py`) and Phase 2's tools.
- Choice: `analysis/replay_models/choice/choice_prediction.py:220` (`build_choice_dataset`), `:442`
  (`fit_choice_model`), `:591` (`fit_choice_blocks`); `model_specs.py:1251` (`MODEL_CHOICE`);
  `scripts/replay_models/content/compare_event_source_models.py` (the event-source suite that holds the
  choice block); `reported/behavior/choice.py:41` (`choice_values`).
- Stay or leave: `analysis/replay_models/choice/stay_leave.py:151` (`stop_replay_summary`), `:405`
  (`fit_is_stay_logistic_per_session`), `:534` (`fit_is_stay_pooled_gee`), `:687` (`fit_stay_leave_roles`);
  `scripts/replay_models/choice/run_stay_leave.py`; `reported/behavior/choice.py:117` (`stay_leave_values`).
- Ripple power: `analysis/replay_models/coupling/swr_power.py:46-57` (`PREDICTORS`, the post-outcome gate),
  `:63` (`post_outcome_masks`), `:210` (`fit_power_model`), `:357` (`assemble_power_frame`);
  `scripts/replay_models/coupling/fit_swr_power_model.py`; `reported/behavior/coupling.py:29`
  (`power_values`).
- Scratch: `S/verify_content/43_next_controlled.py` (next against previous), `31_stay.py` (stay fits).

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C7](shared-contracts.md#c7-the-fit-row-record),
[C8](shared-contracts.md#c8-the-declared-family-of-tests) (Q6).

**Decision needed first:** L7 (stay or leave per visit).

## Tasks

- **Baseline**: save the `behavior.event_source_suite.*`, `behavior.stay_leave.*` and
  `coupling.ripple_power` values; count the stay or leave fits that converge now (the design: a small
  share) with the current code.
- **Q6, content and the next arm** (in `role_content.py`): one more role, `next`, for the arm of the next
  trial's well when it is neither current nor previous (`is_next_arm`, `candidate_arm.py:192`); the
  estimand is the `next` role minus `other_leaf`, per kind and difference, length-adjusted and unadjusted;
  support rule on the role (events with such a candidate). Fit with Phase 6a's runner as an added model.
- **Choice** (`analysis/replay_models/choice/visit_choice.py`, new): one row per (visit, candidate next
  arm), the grouped softmax over the candidates of `build_choice_dataset` (reuse its candidate set and
  behavioral covariates), covariates at pooled SD, plus each kind's in-visit content share on the candidate
  (the mean over the visit's selected events; a visit with no event of the kind gets 0 and a no-event
  indicator, counted). Estimand: the content slope per kind and its difference (not in the declared family;
  Q6's declared member is the content model above). Support rule; records.
- **Stay or leave** (`analysis/replay_models/choice/visit_stay_leave.py`, new; per L7): one row per visit,
  Bernoulli `is_stay` (the next trial stays in the patch) on the omitted outcome, each role's in-visit
  content share per kind, and their products (the continuous factor centered at a pooled reference; the 0/1
  factor not), through the shared fitter with the Newton decrement rule. Report the share of sessions that
  converge and pass support, beside the baseline count, so the lead can decide whether it stays in the paper.
  The pooled GEE is not moved (it is the stage 3 kind of model); say so in the docs.
- **Ripple power** (`analysis/replay_models/coupling/visit_swr_power.py`, new): the Gaussian likelihood per
  in-visit event of the kind, the `PREDICTORS` of `swr_power.py:46` at pooled SD (replacing the per-session
  z-score), `time_since_stop_start_s` replaced by a smooth of time since visit start and a smooth of time
  until leaving (Phase 3's event clocks), the post-outcome gate of `swr_power.py:56` kept for post-outcome
  predictors (the cohort step restricts and counts). REML with the profiled scale.
- **Runners** (new): `scripts/replay_models/choice/fit_visit_choice.py`, `fit_visit_stay_leave.py`,
  `scripts/replay_models/coupling/fit_visit_swr_power.py`; registered in `run_refits.py`.
- **Summary values**: Q6 per kind and difference (declared); choice, stay or leave and ripple power
  slopes with intervals, labeled outside the declared family; the stay or leave convergence counts.
- **Docs**: `docs/behavior_models.md` sections for the three.

## Deliberately not in this phase

- Removing the old choice, stay or leave and power code and values (Phase 12).
- The pooled GEE (not moved).
- FDR (Phase 9).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/content/test_role_content.py::test_next_role_only_when_distinct` | the next role is set only when the next arm is neither current nor previous; the support count matches a hand count |
| `test_role_content.py::test_next_role_null_when_next_equals_previous` | simulated content tied to the previous arm and choices that repeat it: the next-role estimate is within 2 SE of 0 |
| `tests/analysis/replay_models/choice/test_visit_choice.py::test_planted_content_slope` | a planted content effect on choice is recovered; with behavior that predicts the choice and no content effect, the slope is null at the nominal rate |
| `test_visit_stay_leave.py::test_converges_on_separable_free_data` | per-session fits converge by the decrement rule on simulated visits; a separated session fails support, not the fit |
| `tests/analysis/replay_models/coupling/test_visit_swr_power.py::test_gaussian_reml_and_gate` | the Gaussian fit equals penalized least squares at the selected penalty; post-outcome predictors see only post-outcome events |

## Fixtures

Synthetic visits, choices and events (fixed seeds), reusing Phases 3 and 6a helpers.

## Verification and runtime

- Simulated first, then five sessions, then 242. All CPU and small (one row per visit or event).
- Compare with the baseline: Q6 against the old choice block's content term (the design expects a null
  with behavior in the model); stay or leave convergence share before and after; ripple power slopes' signs
  against the old fits.

## Rebuilt

`models_convention/choice/`, `stay_leave/`, `ripple_power/`, Q6 rows in `content/`;
`paper/reported/behavior.json` and the emitted files.

## Exit criteria

Q6 is in the summary with its support counts; choice, stay or leave and ripple power each have records and
values or recorded reasons; the stay or leave convergence count is reported for the lead's decision.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
