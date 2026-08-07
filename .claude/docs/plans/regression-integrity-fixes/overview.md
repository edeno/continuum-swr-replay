# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Source audit:
[07_comprehensive_regression_model_statistical_audit.md](../../reviews/07_comprehensive_regression_model_statistical_audit.md).
Executable pipeline map:
[08_regression_pipeline_code_map.md](../../reviews/08_regression_pipeline_code_map.md).

## Current codebase integration points

- [`data_loaders/position.py:106-115`](../../../../src/continuum_swr_replay/data_loaders/position.py#L106)
  and [`:221-293`](../../../../src/continuum_swr_replay/data_loaders/position.py#L221) — unrestricted
  linear interpolation, broad position/track-graph fallbacks, and outer-envelope valid-time trimming;
  Phase 7 makes source selection, graph geometry, and the time envelope explicit and auditable.
- [`scripts/decoding/run_decoding_bandit.py:346-407`](../../../../scripts/decoding/run_decoding_bandit.py#L346)
  and [`decoding_common.py:153-167`](../../../../src/continuum_swr_replay/decoding_common.py#L153) —
  decoding reuse is currently based on file presence; Phases 7 and 13 add source-aware validation
  and selectively refresh/re-decode invalid sessions.
- [`analysis/replay/prepare_data.py:150-195`](../../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L150)
  and [`:222-240`](../../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L222) —
  presence-only trial caching and independently assembled table inputs; Phase 7 replaces this with
  a versioned session-input bundle.
- [`analysis/replay/covariates.py:47-119`](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L47) —
  never-visited recency currently depends on total future session length; Phase 8 replaces it with
  query-time, suffix-invariant history.
- [`analysis/replay/covariates.py:456-548`](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L456) — `add_uncertainty_covariates` currently rolls `is_stay` over stop rows; Phase 8 removes that path.
- [`analysis/replay/_helpers.py:29-132`](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L29) —
  current-trip reward/switch labels are broadcast from trial start; Phase 8 separates completed-trial,
  current-trip, and post-outcome semantics.
- [`analysis/replay/tables.py:777-1009`](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L777) — canonical stop builder; Phase 8 adds stable trial/decision context and stop-start history.
- [`analysis/replay/tables.py:1296-1307`](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1296)
  and [`:1421-1450`](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1421) —
  full-session recency demeaning and patch reward aggregation; Phase 8 replaces both estimand contracts.
- [`analysis/behavior/qvals_table.py:380-430`](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L380) — exact poke/reward timestamps and stop-to-trial mapping; Phase 8 also emits the canonical per-trial decision-covariate table.
- [`analysis/replay/cohort_audit.py:25-118`](../../../../src/continuum_swr_replay/analysis/replay/cohort_audit.py#L25) — landed drop-accounting infrastructure; Phase 9a extends it with exact identity fields.
- [`scripts/content/compare_event_source_models.py:198-383`](../../../../scripts/content/compare_event_source_models.py#L198) — useful script-local cohort hashing/freezing implementation moved to shared code in Phase 9a.
- [`analysis/replay/model_checking.py:21-164`](../../../../src/continuum_swr_replay/analysis/replay/model_checking.py#L21) — content CV currently blocks by event rather than stop; Phase 9b replaces its fold construction.
- [`analysis/replay/per_bin_rate_model.py:129-302`](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L129) — model-specific row dropping, random stop folds, and no Patsy-design rank gate; Phases 9a, 9b, and 10a modify distinct parts.
- [`analysis/replay/model_comparison.py:20-157`](../../../../src/continuum_swr_replay/analysis/replay/model_comparison.py#L20) — comparison helper checks counts rather than identities and clips invalid LR inputs; Phase 9b replaces that contract.
- [`analysis/replay/fitting.py:37-64`](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L37) and [`:437-454`](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L437) — conditional rank uses within-group centering, but VIF/condition diagnostics do not; Phase 10a unifies them.
- [`analysis/replay/choice_prediction.py:246-418`](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py#L246) — stop-grain, replay-selected, non-current-arm choice-set construction deleted in Phase 11.
- [`analysis/replay/learning.py:383-643`](../../../../src/continuum_swr_replay/analysis/replay/learning.py#L383) — stop-grain Stage-6 Logit/GEE deleted in Phase 11.
- [`scripts/rate/fit_swr_power_model.py:45-161`](../../../../scripts/rate/fit_swr_power_model.py#L45) — unidentified OLS and session-Wilcoxon reporting replaced in Phase 12.
- [`analysis/replay/contrast_aggregation.py:57-135`](../../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L57) — current animal-balanced pattern and term-only coefficient filter; Phase 10b generalizes aggregation after Phase 10a defines fit validity.
- [`analysis/pfc/glm.py:930-1060`](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L930) — PFC event construction and GLMs are owned by the dedicated PFC plan, not duplicated here.

## Scope and dependency policy

### Goals

- Every decoded/table input is proven to lie within its exact Spyglass valid-time union and to match
  the source/query/cache identity recorded in its manifest.
- Every promoted result is traceable to one complete run, exact inputs/specifications, exact likelihood rows, and exact folds.
- Every likelihood-based comparison either proves row/fold compatibility or fails with a reason; no plausible-looking invalid AIC/LRT/CV values are emitted.
- All fitted covariates have accurate names, trial grain, and time availability relative to their response.
- Core content/rate estimates have design-rank gates, dependence-aware uncertainty, and animal-balanced summaries.
- Choice and Stage-6 operate on one observation per actual behavioral decision and include zero-replay and same-arm decisions.
- SWR-power coefficients come from algebraically identified specifications with residual/rank/uncertainty diagnostics.
- One clean refit replaces all stale/mixed-generation results and regenerates model tables, figures, and claims.

### Non-Goals

- No decoder, event detector, bin-width, spline, or self-history redesign. Existing decoder results
  are reused only when source, detector, parameter, output, and time-axis provenance can all be
  reconstructed and matched to the corrected source bundle; otherwise Phase 13 re-decodes the
  session into a new staging root.
- No HMM re-ingestion, behavior-model migration, second uncertainty family, second per-bin
  architecture, or duplicate parity/cohort/GOF framework.
- No causal mediation claim for Stage 6. It remains a post-outcome behavioral association.
- No claim that five animals provide high-powered population inference; the plan improves the unit of analysis, not sample size.
- No attempt to infer value for the trailing trials without fitted behavior-model state; those rows remain an explicit definitional exclusion.
- No duplicate PFC implementation here. Canonical events, controls, GLM mechanics, dependence-aware inference, and Figure 4 remain in [the PFC plan](../pfc-fig4-recut/PLAN.md).
- No backward-compatible artifact readers, covariate aliases, old column names, or legacy artifact
  fallback. Stable public module paths/CLI entry points are preserved where their semantics remain
  valid; before the first release promotion, development and smoke commands select an explicit
  finalized `run_id` rather than pretending a legacy artifact is current.
- No full-dataset W2b/W3 spike-evidence run, matched non-event reality baseline, or multi-bin-local
  decoder sensitivity. Those remain explicitly unowned backlog items; this plan must mark the
  corresponding claims unavailable rather than absorbing that work.

### Dependency policy

No new runtime dependency is required. Use NumPy, pandas, Patsy/statsmodels, SciPy, and scikit-learn
already in the environment. Bootstrap/permutation routines are local, seeded implementations.
Code-only PFC phases may proceed after Phase 7 establishes the manifest interface. Its production
rerun must wait for Phase 13's validated, staged source/table generation and emit the dedicated
completion manifest before the release decision.

## Locked decisions

1. **Source contract:** exact-one source selection, valid-interval-union containment, exact
   decoder/position time alignment, and source/cache fingerprints are promotion prerequisites.
   Presence-only cache hits are never trusted.
2. **Artifact and compatibility cutover:** preserve existing public module paths and CLI entry points
   where their semantics remain valid, but replace corrected columns/schemas in place without aliases.
   Archive exactly one read-only pre-change baseline, then use versioned run directories plus one
   global, atomic regression-release `CURRENT.json`; fixed-path or independently-current family outputs
   are never read by canonical consumers.
3. **Fit purposes:** `estimation` uses the scientifically appropriate maximal cohort; `selection` uses a declared exact common cohort/folds. Only `selection` outputs support AIC/BIC/LRT/CV ranking.
4. **Timing:** post-outcome RPE/update analyses are physically gated after the recorded outcome-known timestamp. Future-referencing peri-switch variables remain descriptive only.
5. **Switch context:** compute recent switching over completed trials at the query timestamp, never
   over stop rows; rename it `recent_completed_switch_rate_k5`. A current trip's eventual switch is
   descriptive context only; predictive models use `previous_completed_trial_is_switch`.
6. **Recency:** a never-visited leaf uses `n_completed_trials_at_query + 1`, with
   `has_arm_history=0`; visited leaves use trials since last completed visit. Trunks use
   `is_reward_site=0` and recency 0. Full-session demeaning is removed.
7. **Reward history:** make the window part of every name. Preserve
   `reward_rate_arm_visits_k10` as rewards among the last ten visits to that arm, with its
   `visit_count_arm_visits_k10` denominator and `has_arm_history`. Separately name counts over the
   last ten global trials `reward_count_arm_global_k10` and `visit_count_arm_global_k10`. Use the
   corresponding patch fields. Never reconstruct a rate from a numerator and denominator that use
   different windows, and label count-only terms recent rewarded exposure rather than value.
8. **Q tail:** do not impute behavior-model state after the last fitted poke-out. Non-Q models retain
   those rows; Q models use a declared mapped cohort and report the observable terminal-tail profile.
9. **Content primary:** the stable arm M4-style spatial/reward-history specification is primary; sparse last-rewarded and value/uncertainty terms are prespecified sensitivities. Mvu is not a headline exploit-versus-explore model.
10. **Content response:** soft posterior mass is the primary robustness target alongside the existing hard-count estimate; agreement is required for a content claim.
11. **Decision window:** for target trial `t`, aggregate replay in `[outcome_known_time(t-1), poke_out_time(t-1))`; use trial `t`'s pre-outcome value vector and all six leaf arms. Decisions with no replay stay in the dataset.
12. **Population summary:** all five animal effects and an animal-balanced center/interval are primary; session/neuron signed-rank summaries are descriptive.
13. **SWR power:** primary decomposition is realized reward + pre-outcome Q. RPE-alone is a separate sensitivity; reward, Q, and RPE are never co-fit.

## Covariate-construction audit routing {#covariate-audit-routing}

The five requested workstreams route into the existing PR sequence; this table is the non-overlap
contract for the 2026-07-29 review.

| Review workstream | Canonical owner | Scope added by this refresh |
| --- | --- | --- |
| 1. Input integrity and provenance | [Phase 7](phase-7-run-artifacts.md), then [Phase 13](phase-13-refit-reporting.md) | Valid-time/frozen-tail audit, source/cache identities, selective refresh/re-decode |
| 2. Temporal correctness | [Phase 8](phase-8-covariate-timebase.md) | Existing trial-grain switch/outcome gate plus explicit prior-vs-current switch semantics |
| 3. Content covariate semantics | [Phase 8](phase-8-covariate-timebase.md) and [Phase 9a](phase-9a-cohort-identity.md) | Past-only recency, patch exposure, hard joins, Q-terminal-tail profile |
| 4. Regression safeguards | [Phases 9b](phase-9b-blocked-comparisons.md), [10a](phase-10a-estimability.md), [10b](phase-10b-dependence-inference.md), and [12](phase-12-swr-power.md) | No duplicate implementation; these phases already own common folds, rank/convergence, dependence, and identified power |
| 5. Controlled rebuild and reporting | [Phase 13](phase-13-refit-reporting.md) | Source-QC gate and schema-exact rebuild before promotion |

## Related-plan ownership

| Existing plan | Boundary |
| --- | --- |
| [`covariate-harmonization`](../covariate-harmonization/PLAN.md) | Its Phase 1/1.5 builder and Beta-model work must be merged or this plan must be based on the branch that contains it before Phase 7/8 begins. Its unfinished all-family refit and Phase-4 model-promotion work are superseded here; its stop-grain switch and unseen-arm contracts are corrected in Phase 8. Only the non-PFC ancillary `is_swr` adapter slice of its Phase 3 remains separate. |
| [`hmm-to-beta-covariates`](../hmm-to-beta-covariates/PLAN.md) | Complete and authoritative for Beta covariate construction. This plan adds timing enforcement and rebuilds stale materializations; it does not redo the migration. |
| [`nonswr-uncertainty-rate`](../nonswr-uncertainty-rate/PLAN.md) | Historical/superseded. Reuse the existing runner; Phase 8 replaces its invalid stop-grain switch definition. |
| [`per-bin-rate-models`](../per-bin-rate-models/PLAN.md) | Complete baseline infrastructure. Phase 8 changes only information-time querying/gating at bin left edges; exposure, counts, and lag-history mechanics remain. Phases 9–10 own exact sets, folds, diagnostics, and inference. |
| [`rate-content-dissociation`](../rate-content-dissociation/PLAN.md) | Historical/superseded router. Its artifact audit, refits, choice work, and reporting are not executable; active replacements live here and in the single surviving followup below. |
| [`rate-content-followups`](../rate-content-followups/PLAN.md) | Its choice-CV phase is superseded by Phase 11. Phase 13 owns the valid cross-history refit; only the SWR/non-SWR onset cross-correlogram and continuum interpretation remain separate downstream work. |
| [`joint-replay-departure`](../joint-replay-departure/PLAN.md) | Separate duration estimand; it begins only from Phase-13 promoted rate inputs. |
| [`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md) | Sole owner of PFC mechanics, immutable completion-manifest production, claim 20, and Figure 4. Phase 12 implements the external validator; Phase 13 invokes it after the PFC production phases. |
| [`review-findings-fixes`](../review-findings-fixes/PLAN.md) | Retains decoder-identifier/ripple/trajectory history. This plan owns source-bundle validation; Phase 11 supersedes that plan's older switch-conditional choice decision. |
| [`integrative-replay-roadmap`](../integrative-replay-roadmap.md) | Roadmap only, not an implementation owner. Its rate/choice sketches consume this plan's promoted outputs; only the separately routed joint-departure plan is currently executable. |

The full W2b/W3 reality run and matched-baseline strengthening are intentionally not assigned to any
active phase here. Claim rows that depend on them stay `pending`/`unavailable`; a later plan may own
that work, but no executor should infer it from Phase 7's source re-decode.

## Audit baseline

- Four saved `*_HPC_80_data_over_time.pkl` files extend beyond the current Spyglass valid interval.
  Three frozen low-speed tails contribute 473.44 seconds of false immobility exposure.
- The canonical event-arm artifact contains three retired HMM columns absent from the current source
  schema, and several regression results predate their current table/Q inputs.
- Q coverage is 105,344/107,400 stops; all 2,056 missing rows are terminal-tail rows and are
  omission-enriched.
- These figures are baseline evidence to reproduce in Phase 7 and close in Phase 13, not hard-coded
  unit-test constants.

## Metrics

- Zero promoted `data_over_time` sample lies outside the exact union of its declared valid intervals;
  decoder time, position index, and derived-table time hashes match exactly.
- Every source/cache change invalidates the affected session bundle; no presence-only cache hit can
  enter a promoted table.
- No canonical consumer can load a missing, incomplete, stale-schema, mixed-release, or
  hash-mismatched artifact; all paper-facing families resolve through one release ID.
- Every selection comparison has identical `cohort_hash`, `group_hash`, and, for CV, `fold_hash`; mismatches raise.
- Same-N/different-ID, partial candidate-set, duplicate-key, nonnested-LRT, negative-LR, and mixed-run fixtures all fail closed.
- `recent_completed_switch_rate_k5` is unchanged by adding duplicate stops to a trial and depends only on trials completed by the query time.
- Appending future trials or events cannot change any already-materialized recency/history covariate.
- Arm/patch visit-window reward-rate terms carry matching visit-window exposure/history; a
  global-trial-window count is never labeled value or used as that rate's numerator.
- Every post-outcome event satisfies `event_start_time >= outcome_known_time`; every post-outcome
  bin satisfies `bin_start_time >= outcome_known_time`. A bin that straddles the boundary is dropped;
  bin centers are never used for eligibility.
- Conditional diagnostics are invariant to event-level offsets and predictor rescaling; all fitted designs report rank and scaled condition diagnostics.
- Rank-deficient/nonconverged fits never enter comparisons or headline summaries; excluded cells remain explicitly represented with failure reasons.
- Dependence-aware simulation intervals improve nominal coverage over model-based intervals under the repository's autocorrelated trajectory fixtures.
- Decision tables contain one row per `(decision_id, arm)`, six arms and exactly one chosen arm per decision, including same-arm and zero-replay decisions.
- SWR-power deterministic-identity fixtures are rejected and planted identified effects are recovered within stated Monte Carlo tolerance.
- The production promotion audit reports complete expected model/population/session coverage and all five animals before updating `CURRENT.json`.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Correct source bounds invalidate saved decoder time axes. | Phase 7 audits all sessions first; Phase 13 reuses only fully attested aligned results and selectively re-decodes all other sessions into a separate staging root, preserving an archived baseline. |
| Position/DIO QC exposes additional sessions beyond the four known stale tails. | Fail with explicit session/reason records; do not relax thresholds during the production run. |
| Common cohorts shrink high-dimensional ladders and change apparent winners. | Keep maximal-cohort estimation separate; rank only declared selection sets and report retention/hash. |
| The trial/decision rebuild changes many headline numbers. | Save pre-change descriptive baselines, smoke one animal, and publish before/after deltas without requiring direction preservation. |
| Stop/block bootstrap is computationally expensive. | Fixed-seed small-bootstrap CI tests in CI; smoke one session with timing/memory, then choose the production replicate count before the full run. |
| Five-animal intervals remain wide. | Show all five effects and leave-one-animal-out sensitivity; do not substitute session p-values. |
| Penalized sparse-term sensitivity cannot use ordinary AIC/LRT. | Label it `estimation` only and compare by shared-fold predictive score; never mix it with unpenalized likelihood tests. |
| Atomic-release migration touches many plotting scripts. | Centralize release resolution/validation first, migrate consumers in Phase 7, require an explicit finalized `run_id` before the first global release, and test interrupted promotion before any number-changing phase. |
| PFC work drifts from regression promotion. | Phase 13 first freezes a validated release candidate. PFC production consumes that exact canonical-table generation and emits the completion manifest; only then may Phase 13 promote Figure-4/cross-area claims. |

## Rollout Strategy

Replace in place with no compatibility layer. Execute in order:

`7 → 8 → 9a → 9b → 10a → 10b → 11 → 12 → 13`.

First merge, or base this work on, the unmerged Phase-1/1.5 covariate-builder changes. Implement
regression code in `7 → 8 → 9a → 9b → 10a → 10b → 11 → 12`. In parallel after Phase 7, land all
PFC code-bearing changes (Phases 0–2b, Phase-3 runner/provenance glue, and Phase-4
loader/finalizer/figure code) with synthetic tests. Freeze one clean analysis-code/config commit; no
formula, schema, or analysis code changes are allowed during production. Phase 13 then executes the
source/table/model build and freezes a release candidate; PFC Phases 3–4 execute against that exact
candidate and emit the immutable completion manifest; Phase 13 then generates and hashes all
paper-facing bytes through the explicit candidate, finalizes the release-output tree, and promotes
the global release, including either the validated PFC component or an explicit unavailable reason.
Only after `CURRENT.json` advances are those exact bytes copied to live repository destinations. Any
source/table or analysis-code change invalidates the candidate and PFC completion and starts a new run. Each
number-changing implementation phase smoke-tests synthetic data first and then one real
session/animal. Only Phase 13 atomically advances the global regression-release `CURRENT.json`.

For the five-part covariate review, the same order is grouped as:
`input/provenance (7) → timing/constructs (8) → cohorts/comparisons (9a–9b) →
fit safeguards (10a–10b, 12) → controlled rebuild (13)`. Phase 11 is the already-owned
decision-model correction and is not duplicated by this review slice.

## Open Questions

None. The scientific and compatibility choices needed to execute these phases are locked above. Unexpected empirical failures are reported as failed validation gates, not resolved by silently changing the estimand.

## Estimated Effort

Approximately 1,550-2,200 changed LOC plus tests across the nine active PRs, excluding the separately
planned PFC implementation and production compute. The source-bundle checks enlarge Phase 7; the other
largest diffs remain Phase 10b bootstrap/aggregation and Phase 11 decision-table replacement.
