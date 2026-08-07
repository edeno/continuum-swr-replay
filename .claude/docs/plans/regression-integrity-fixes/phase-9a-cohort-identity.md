# Phase 9a — Exact comparison cohorts and identity plumbing

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#cohorts-folds)

Create one shared exact-row identity implementation, harden Q/value joins, and wire every selection
runner to a frozen common cohort. This phase fixes likelihood-row comparability and makes the
behavior-model terminal tail explicit, but deliberately leaves fold construction and comparison
statistics to Phase 9b.

**Inputs to read first:**

- [scripts/content/compare_event_source_models.py:198-383](../../../../scripts/content/compare_event_source_models.py#L198) — existing sorted-ID hashing, whole-unit freezing, and all-model success policy to generalize.
- [src/continuum_swr_replay/analysis/replay/cohort_audit.py:25-118](../../../../src/continuum_swr_replay/analysis/replay/cohort_audit.py#L25) — current permissive audit schema/reconciliation.
- [scripts/content/fit_replay_content_models.py:468-540](../../../../scripts/content/fit_replay_content_models.py#L468) — content union-complete cohort that already behaves mostly correctly.
- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:129-138](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L129) and [scripts/rate/fit_per_bin_rate_models.py:115-153](../../../../scripts/rate/fit_per_bin_rate_models.py#L115) — urgent model-specific-drop defect.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:86-99](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L86) and [scripts/rate/fit_per_stop_rate_models.py:80-122](../../../../scripts/rate/fit_per_stop_rate_models.py#L80) — same defect at stop grain.
- [scripts/content/fit_swr_contrast_content.py:208-365](../../../../scripts/content/fit_swr_contrast_content.py#L208) and [scripts/rate/fit_swr_contrast_rate.py:143-232](../../../../scripts/rate/fit_swr_contrast_rate.py#L143) — per-rung contrast filtering.
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:271-308](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L271) and [:335-378](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L335) — deterministic post-final-poke exclusion, first-match model fetch, and per-trial join coverage.
- [src/continuum_swr_replay/analysis/replay/covariates.py:725-823](../../../../src/continuum_swr_replay/analysis/replay/covariates.py#L725) — contingency/value attachment currently warns/deduplicates instead of enforcing key cardinality.

**Contracts referenced:**

- [Comparison identity](shared-contracts.md#comparison-identity) — declared sets and exact compatibility fields.
- [Cohort audit schema](shared-contracts.md#cohort-audit-schema) — identity fields and strict schema.
- [Run artifact contract](shared-contracts.md#run-artifact-contract) — identity tables are required run outputs.

**Designs referenced:** [exact cohorts and blocked folds](designs.md#cohorts-folds), cohort portion only.

## Tasks

- Add `analysis/replay/analysis_identity.py` with typed canonical key encoding, duplicate rejection,
  complete candidate-set validation, stable row/group hashes, `AnalysisIdentity`, and
  `freeze_comparison_cohort`. Move the useful implementation out of the event-source script; keep that
  script as a consumer and delete its private duplicate.
- Extend `ModelSpec`/`PerBinRateModelSpec` or add a registry alongside them declaring
  `fit_purpose`, `comparison_set`, response/family/link/offset/grain/category semantics, and parent.
  Model definitions remain the single formula source of truth.
- Add strict fitter mode: a selection fitter receives exact allowed row keys and raises if its normal
  validation/drop path would change them. A missing predictor or incomplete group is a configuration
  failure recorded for the cell, not a warning followed by a smaller fit.
- Wire per-bin and per-stop runners to build the declared selection cohort once per
  `(session, outcome, comparison_set)` using the union of required columns. Emit separate maximal-cohort
  `estimation` fits only where scientifically useful; never put their AIC in a selection table.
- Refactor base arm/patch content to the shared helper and split its current all-requested-model union
  into the exact `ARM_*`/`PATCH_*` sets in shared contracts, so Mv/Mu/Mvu missingness no longer
  changes `ARM_BASE = {M0,M0a,M1,M2,M3,M4}`. Preserve every declared parent edge—especially
  `M5→Mv`, `M5→Mu`, and `Mu→Mvu`—and freeze contrast cohorts separately for predictive,
  retrospective, and descriptive timing families.
- Build `behavior_model_trial_table.parquet` separately from Phase 8's complete behavior timebase.
  Require exactly one external behavior-model result for the declared model key and unique model-trial
  keys. Normalize its integer `(nwb_file_name, epoch_number)` key through Phase 8's persisted,
  bijective `session_epoch_crosswalk` before joining. Left-join it one-to-one onto
  `(session_id, trial_number_by_epoch)`, classify coverage before any fit, and fail on internal gaps,
  crosswalk gaps, or source/canonical key disagreement; do not require model coverage for terminal
  trials that are outside the fitted behavior-model window.
- Make downstream Q/value joins fail closed: require unique `stop_id` and `validate=` cardinality on
  every stop/event/arm merge. Remove warn-and-keep-first, silent `drop_duplicates`, and any inner join
  that can shrink the non-Q base table.
- Treat stops after the last fitted poke-out as the declared `q_terminal_tail`, never as generic
  missingness and never by imputation. Base/non-Q models use a left join and retain those rows.
  Q/value/uncertainty sets use one explicit mapped cohort. Persist a terminal-tail profile by animal,
  session position, reward outcome, stop duration, event presence/type, and available past-only
  covariates, plus a common-observable-window sensitivity for primary non-Q estimates.
- Extend `cohort_audit.py` and all producers to emit the schema-versioned identity fields. Reject missing
  required fields; remove `write_cohort_audit` behavior that silently creates absent columns.
- Persist the exact sorted row/group key tables next to results, not just hashes. Register them in the
  Phase-7 manifest and require one identity per selection cell.
- Join every family-specific expected cohort to Phase 7's `expected_sessions` snapshot before
  filtering. An absent session must appear as either a structured fit/source failure or an approved
  family-scoped exclusion from `config/regression_session_exclusions.yaml`; never infer exclusions
  from which files or successful fits happen to exist.
- Update `docs/models_table.md` to label estimation versus selection outputs and list declared sets;
  remove all model-ranking text until Phase 13 supplies a valid refit.

## Deliberately not in this phase

- Fold creation/CV and AIC/LRT calculation (Phase 9b).
- Rank/separation exclusion policy (Phase 10a); cohort identity is about response rows, not fit success.
- Decision-grain choice cohorts (Phase 11), though the shared helper must support their key shape.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_typed_hash_same_n_different_ids` | equal counts with different IDs produce different hashes |
| `test_hash_row_order_invariant_and_duplicate_strict` | permutation preserves hash; duplicate key raises |
| `test_complete_arm_patch_choice_groups` | partial/duplicate category sets fail before fit |
| `test_strict_fitter_cannot_drop_frozen_rows` | internal NaN/exposure filtering of supplied keys raises |
| `test_per_bin_selection_set_has_exact_keys` | every model/session/outcome shares row and group hashes |
| `test_content_sets_and_parent_edges_are_exact` | `ARM_BASE` includes M0a; value/uncertainty missingness does not shrink it; Mv/Mu/Mvu and patch parents match the registry |
| `test_q_fetch_and_joins_require_exact_cardinality` | zero/multiple model-result rows, duplicate/internal-missing trial or stop keys, and key disagreement fail |
| `test_behavior_timebase_survives_model_left_join` | all complete behavior-trial keys remain after the model join; only model columns are null in the declared tail |
| `test_behavior_model_join_uses_epoch_crosswalk` | integer source epochs map bijectively to canonical sessions before the one-to-one left join; label parsing and filename construction cannot pass |
| `test_q_terminal_tail_is_classified_not_imputed` | post-final-poke rows enter the named bucket; Q is absent and non-Q columns remain usable |
| `test_non_q_base_retains_q_terminal_tail` | Q missingness cannot shrink the base content/rate cohort |
| `test_common_observable_window_uses_identical_non_q_rows` | full-vs-mapped-window sensitivity differs only by the declared terminal rows |
| existing cohort reconciliation suite | all mutually exclusive drop accounting still closes with new identity columns |
| `test_expected_session_accounting_survives_failed_fit` | a failed or absent session remains represented and cannot be silently converted to an exclusion |

## Fixtures

Generalize `tests/unit/test_compare_event_source_models.py` and `test_cohort_audit.py` fixtures. Add
same-N/different-ID, partial nine-arm/three-patch/six-choice groups, duplicate typed keys, and a rate
design where `reward_lag10` is missing on a known subset. Add a two-session behavior table with an
internal missing trial (hard error), a legitimate terminal tail, duplicate external model matches, and
omission-enriched observable tail fields.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- Every selection runner freezes once, then fits all members without another silent row drop.
- Maximal-cohort estimation and common-cohort selection outputs cannot be confused by schema or path.
- Timing families are not forced into one scientifically incoherent all-rung intersection.
- Non-Q rows never disappear merely because a Q artifact is shorter; Q-tail sensitivity is observable
  and does not invent latent state.
- No external behavior/model join can select the first duplicate or silently reduce cardinality.
- Exact key tables, not just hashes/counts, are persisted and manifest-registered.
- Existing event-source semantics remain source-specific rather than forcing incompatible event IDs.
- Every task is implemented, fold/statistical work remains out of scope, and the private duplicate cohort helper is removed.
- Validation tests pass and genuinely distinguish same-N/different-ID and incomplete-group cases.
- Production names/docstrings do not mention this plan or phase numbers; comparison-set docs are updated.
