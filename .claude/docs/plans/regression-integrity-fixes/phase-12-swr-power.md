# Phase 12 — Identified SWR-power model and PFC integration gate

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#swr-power)

Replace the algebraically unidentified SWR-power script with tested, separate event-mark estimands and
make the independently planned PFC recut an explicit prerequisite for cross-area promotion.

**Inputs to read first:**

- [scripts/rate/fit_swr_power_model.py:45-87](../../../../scripts/rate/fit_swr_power_model.py#L45) — reward/RPE/Q co-fit and unchecked `lstsq` to delete.
- [scripts/rate/fit_swr_power_model.py:127-161](../../../../scripts/rate/fit_swr_power_model.py#L127) — marginal duration check and session-Wilcoxon report to replace.
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:207-228](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L207) — exact Q/RPE identity and construction.
- [src/continuum_swr_replay/analysis/replay/tables.py](../../../../src/continuum_swr_replay/analysis/replay/tables.py) — event duration, time, stop, and power columns.
- [PFC Fig-4 recut plan](../pfc-fig4-recut/PLAN.md) — separate canonical-event/control/population plan whose completion manifest Phase 13 requires.
- [src/continuum_swr_replay/analysis/pfc/glm.py:167-245](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L167) and [:347-480](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L347) — PFC mechanical/inferential gaps that must be covered by that plan before promotion.

**Contracts referenced:**

- [Fit validity](shared-contracts.md#fit-validity), [fold assignment](shared-contracts.md#fold-assignment),
  [animal summaries](shared-contracts.md#animal-summary), and [run artifacts](shared-contracts.md#run-artifact-contract).

**Designs referenced:** [identified SWR-power models](designs.md#swr-power).

## Tasks

- Add `analysis/replay/swr_power_model.py` containing data validation, formula construction, scaled-rank
  preflight, statsmodels OLS fitting, stop-cluster covariance, residual/influence diagnostics, common
  blocked CV scoring, and structured result/failure rows. Keep the script as a thin run/report wrapper.
- Implement P0-P3 exactly as designed. Use aligned `trial_reward` and `q_current`; never substitute
  `reward_lag1` for current outcome and never co-fit deterministic reward/Q/RPE components. Gate all
  outcome-dependent models after `outcome_known_time`.
- Register row key `event_id`, dependence/fold group `stop_id`, and two exact selection sets:
  `POWER_REWARD_Q = {P0, P1, P3}` with declared LRT edges `P0→P1→P3`, and
  `POWER_RPE_SENS = {P0, P2}` with edge `P0→P2`. Refit each set's P0 on that set's frozen
  common cohort. Never compare P1 with P2 by LRT/AIC, and never move P2 into the primary set after
  seeing results.
- Include `log(event_duration)` in every non-null design. Save R-squared/adjusted R-squared, robust
  intervals, residual normality/heteroskedasticity flags, leverage/Cook's-distance summary, cluster count,
  condition/rank fields, and held-out score. Model event occurrence separately; this response is a mark
  conditional on event detection.
- Remove the Sarle bimodality threshold as evidence for event classes. Keep the power distribution as a
  descriptive QC plot and report detector-threshold relationships without turning them into an OLS
  validation test.
- Migrate reporting to animal-balanced effects and correct multiplicity across the prespecified power
  terms. Label P2 RPE-alone a sensitivity rather than an LRT competitor to P1.
- Treat the refreshed dedicated PFC plan as an external dependency. Its completion manifest must cover
  canonical event labels, spike-bin edge contract, rank/convergence, spike history/slow drift, matched
  immobile control, autocorrelation-preserving null/blocked score, and neuron→session→animal aggregation.
  Verify those fields here rather than duplicating PFC code.
- Implement validation of the `pfc_completion.json` schema owned and produced only by
  [`pfc-fig4-recut` Phase 4](../pfc-fig4-recut/phase-4-inference-figure.md): require its frozen
  source/table-generation hashes, the same clean frozen code/config git identity as the candidate,
  null dirty-diff hash, output hashes, complete expected-session accounting, five-animal coverage,
  event/control/fold/null fields, the fit-diagnostics path/hash, exact expected-versus-accounted
  cell counts, complete status counts, zero missing statuses, zero headline-invalid
  rank/nonconvergence cells, and the forbidden stale LR-column check. Phase 13 may
  promote non-PFC analyses while marking cross-area claims unavailable, but it cannot
  regenerate/promote Figure 4 without this gate.
- Update `docs/models_table.md` with the identified power formulas and clarify event-mark versus
  occurrence questions. Remove prior coefficient claims until Phase 13 refits.

## Deliberately not in this phase

- Implementing PFC GLMs/controls/figure inside this plan; the dedicated plan owns them.
- Treating event-mean power as a detector-independent latent SWR magnitude.
- Comparing P1 and P2 by AIC/LRT as if they were the same parameterization question.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_power_spec_never_contains_reward_q_and_rpe` | deterministic three-way design is impossible through registry/API |
| `test_rank_deficient_power_design_returns_failure` | exact identity or constant term cannot yield coefficients |
| `test_planted_reward_and_q_effects_recovered` | identified P1 recovers simulation effects within tolerance |
| `test_duration_adjustment_removes_planted_confound` | duration coefficient/effect behaves as expected |
| `test_power_rows_are_post_outcome` | outcome-dependent fit contains no pre-outcome event |
| `test_power_selection_sets_and_rows_are_exact` | P0/P1/P3 and P0/P2 use their set-specific identical event/group/fold hashes; P1-vs-P2 comparison is rejected |
| `test_cluster_covariance_and_blocked_cv_metadata` | cluster/fold hashes and robust fields are complete |
| `test_pfc_completion_gate_rejects_stale_or_incomplete_run` | missing control/animal evidence, fit-table hash/count/status drift, headline rank/nonconvergence, or stale LR column blocks Figure-4 promotion |
| one-session power smoke (slow) | fit/residual/manifest outputs are finite, auditable, and current-schema |

## Fixtures

Simulate event marks with planted reward, Q, duration, and stop-cluster effects, plus exact
`reward = rpe + q` dependence and a constant covariate. PFC gate tests use tiny manifest/pickle fixtures;
actual PFC compute remains in its dedicated plan.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- The old `np.linalg.lstsq` path and reward/RPE/Q formula are deleted.
- Each power formula maps to one explicit scientific estimand and correct outcome timing.
- Event-mark conclusions are not generalized to event occurrence or causality.
- PFC work is neither duplicated nor waved through; the promotion gate checks concrete outputs.
- All power and PFC claims remain unavailable until their fresh manifests validate.
- Every task is implemented, PFC code remains in its dedicated plan, and the old unidentified script fitter is removed.
- Validation tests pass, the real-data smoke is marked slow, and deterministic-identity tests fail if the formula guard is removed.
- Production names/docstrings do not mention this plan or phase numbers; power/PFC integration docs are updated.
