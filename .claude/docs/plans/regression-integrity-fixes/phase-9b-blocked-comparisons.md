# Phase 9b — Shared blocked folds and fail-closed model comparison

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#cohorts-folds)

Use one chronological block assignment per comparison set and replace permissive comparison logic with
strict eligibility checks. This phase makes held-out and likelihood rankings defensible; it does not
change coefficient uncertainty.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/model_checking.py:21-164](../../../../src/continuum_swr_replay/analysis/replay/model_checking.py#L21) — content `GroupKFold` by event and existing bad-fold invalidation.
- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:195-303](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L195) — shuffled stop folds.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:175-249](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L175) — shuffled stop-row folds.
- [src/continuum_swr_replay/analysis/replay/model_comparison.py:20-157](../../../../src/continuum_swr_replay/analysis/replay/model_comparison.py#L20) — inner-drop, count-only guard, LR clipping, and session-Wilcoxon summary to replace.
- [tests/unit/test_model_comparison.py:70](../../../../tests/unit/test_model_comparison.py#L70) — tests currently enshrining clipped/inner-drop behavior.
- [tests/unit/test_per_bin_rate_model.py:208](../../../../tests/unit/test_per_bin_rate_model.py#L208) and [tests/unit/test_per_stop_rate_model.py:363](../../../../tests/unit/test_per_stop_rate_model.py#L363) — good whole-metric bad-fold policy to preserve.

**Contracts referenced:**

- [Fold assignment](shared-contracts.md#fold-assignment) — immutable contiguous groups and hash.
- [Comparison identity](shared-contracts.md#comparison-identity) — eligibility rules.
- [Animal summary](shared-contracts.md#animal-summary) — aggregation of paired model differences.

**Designs referenced:** [exact cohorts and blocked folds](designs.md#cohorts-folds).

## Tasks

- Add `analysis/replay/blocked_folds.py` with deterministic contiguous balanced group assignment,
  serialization, typed hashing, and mapping to descendant likelihood rows. It consumes Phase-9a group
  tables and never infers order from current dataframe order.
- Construct folds once per `(run_id, comparison_set, session)` and pass the immutable table into content,
  per-bin, and per-stop CV. Content blocks all events from a stop together. Preserve the rule that any
  failed/nonconverged/separated training fold invalidates the whole metric without dropping that fold.
- Add explicit `blocked_interpolation` and optional `forward_chaining` scheme names. Paper-facing
  “future prediction” language requires the latter; otherwise report held-out blocked association.
- Rewrite `compare_nested_models` around a `ComparisonContractError`. Validate exact session sets,
  unique cells, run/purpose/set, estimator/response/link/offset/grain/category semantics, row/group hash,
  fold hash for CV, declared nesting, positive df, and monotone nested LL before computing metrics.
  Remove LR/df clipping and inner-merge disappearance.
- Produce paired per-session `delta_aic`, `delta_bic`, and held-out `delta_log_score`, then aggregate
  within animal before across-animal reporting. Remove Wilcoxon-on-delta-AIC as a model-selection test;
  retain session distributions only under descriptive names.
- Update all model-summary/plot scripts so they display paired deltas, Akaike weights, held-out score
  deltas, hashes, valid-pair denominators, and unavailable reasons—not median absolute AIC across
  heterogeneous sessions.
- Persist `folds.parquet` and comparison eligibility/result tables through the Phase-7 manifest. Update
  `docs/rate_models.md` and comparison docs to remove the claim that a shared bin width alone makes
  models comparable.

## Deliberately not in this phase

- Robust/cluster coefficient SEs (Phase 10b).
- Choice-model CV implementation (Phase 11 uses this fold API at decision grain).
- Selecting the final paper model before the clean refit (Phase 13).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_contiguous_folds_keep_stops_and_events_together` | no temporal interleaving and no stop leakage |
| `test_fold_assignment_is_input_order_invariant` | shuffle leaves assignments/hash unchanged |
| `test_every_group_scored_once` | folds form an exact partition with no omissions/duplicates |
| `test_models_share_fold_hash` | every member of a selection set receives the same assignment |
| `test_comparison_rejects_same_n_different_ids_or_folds` | exact identity, not counts, governs validity |
| `test_comparison_rejects_session_mismatch_and_duplicate_cells` | inner-drop behavior is gone |
| `test_lrt_rejects_nonnesting_nonpositive_df_and_negative_lr` | invalid inputs raise rather than clip |
| `test_exact_contract_computes_expected_deltas` | known valid nested fixture returns correct values |

## Fixtures

Use a temporally ordered three-stop/multi-event session, same-N/different-ID fit frames, one model with a
different fold assignment, nonnested predictor sets, and a valid planted nested likelihood pair. Extend
existing bad-fold fixtures instead of replacing their coverage.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- No fitter constructs private random folds after a shared fold table is supplied.
- Content CV cannot place events from one stop in train and test.
- Comparison mismatch paths raise or report unavailable without computing numbers.
- No absolute-AIC cross-session ranking or session-Wilcoxon model-selection claim remains.
- Docs name blocked interpolation accurately and reserve future-prediction language for forward chaining.
- Every task is implemented, robust-SE/model-selection decisions remain out of scope, and private random-fold/clipping paths are removed.
- Validation tests pass and fail if fold/identity enforcement is deleted; slow tests are marked.
- Production names/docstrings do not mention this plan or phase numbers; all listed reporting docs are updated.
