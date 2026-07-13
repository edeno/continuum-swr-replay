# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Run artifact and promotion contract](#run-artifact-contract)
- [Cohort audit schema](#cohort-audit-schema)
- [Comparison identity contract](#comparison-identity)
- [Fold assignment contract](#fold-assignment)
- [Covariate timing contract](#covariate-timing)
- [Decision-table contract](#decision-table)
- [Fit-validity and diagnostics contract](#fit-validity)
- [Term diagnostics](#term-diagnostics)
- [Animal-balanced summary contract](#animal-summary)

## Run artifact and promotion contract {#run-artifact-contract}

Every canonical runner writes into `<analysis>/runs/<run_id>.staging/`, validates the run, atomically
renames it to `<analysis>/runs/<run_id>/`, then atomically updates `<analysis>/CURRENT.json`. A run is
invisible to canonical consumers until the pointer changes.

`manifest.json` is written last and contains:

```json
{
  "schema_version": 1,
  "run_id": "<UTC>-<git8>-<spec8>-<input8>",
  "analysis": "per_bin_rate",
  "completed": true,
  "started_utc": "...",
  "completed_utc": "...",
  "git_sha": "...",
  "dirty": false,
  "dirty_diff_sha256": null,
  "argv": ["..."],
  "software_versions": {"python": "...", "statsmodels": "..."},
  "model_spec_sha256": "...",
  "model_specs": {},
  "inputs": [{"path": "...", "size": 0, "sha256": "..."}],
  "comparison_sets": {},
  "expected_cells": {},
  "failed_cells": [],
  "outputs": [{"path": "...", "size": 0, "sha256": "...", "columns": []}]
}
```

The strict loader requires `schema_version == 1`, `completed is true`, a matching run ID in every
output, exact output hashes/schemas, all expected cells or explicit failure records, and the audit,
identity, and fold tables declared by the manifest. There is no legacy fallback.

## Cohort-audit table schema {#cohort-audit-schema}

The landed row-drop fields remain, with exact identity added. One record is emitted per
`(run_id, fit_purpose, comparison_set, model, population)`:

| Column | Meaning |
| --- | --- |
| `schema_version`, `run_id` | artifact contract identity |
| `fit_purpose` | `estimation` or `selection` |
| `comparison_set` | declared set ID; empty only for standalone estimation |
| `model`, `population`, `grain` | fit cell and counting unit |
| `n_in`, mutually exclusive `n_drop_*`, `n_modeled` | existing reconciled row/group accounting |
| `n_rows_modeled`, `n_groups_modeled` | exact likelihood rows and conditioning/block groups |
| `row_key_columns`, `group_key_columns` | typed keys used to construct hashes |
| `cohort_hash`, `group_hash` | exact canonical typed hashes |
| `fold_hash` | selection CV assignment hash; null only when CV is not run |
| `n_sessions_*`, `n_animals_modeled`, `drop_notes` | existing coverage/profile fields |

The invariant remains `n_in - sum(mutually exclusive drop buckets) == n_modeled`. Schema-versioned
writes reject absent identity fields rather than filling them with NaN.

## Comparison identity contract {#comparison-identity}

Selection results may be compared only when all of these match: `run_id`, `fit_purpose=selection`,
`comparison_set`, response, estimator/family, link, offset/exposure semantics, grain, category-set
semantics, `cohort_hash`, and `group_hash`. CV also requires equal `fold_hash`.

Declared sets:

| Analysis | Row key | Group/block | Selection sets |
| --- | --- | --- | --- |
| Per-stop rate | `stop_id` | chronological stop/decision | `R0-R1-R2`, `R0-Rlag` |
| Per-bin rate | `(stop_id, binj)` | whole stop | explicit parent-child sets plus one prespecified paper-ranking set |
| Arm content | `(event_id, arm)` | whole stop | base `M0-M4`; sparse-memory `M4-M5`; value `M4-Mv`; uncertainty sensitivities |
| Patch content | `(event_id, patch)` | whole stop | base `P0-P4`; value `P3-Pv` |
| SWR contrasts | native likelihood key | whole stop | separate `predictive`, `retrospective`, and `descriptive` sets |
| Decision choice | `(decision_id, arm)` | whole decision | behavior/value baseline versus +replay; split replay separately |

AIC/BIC require the same likelihood and rows. LRT additionally requires a declared strict parent,
positive parameter difference, and child log likelihood no smaller than the parent beyond numerical
tolerance. Invalid input raises `ComparisonContractError`; it is never clipped or converted to a
plausible p-value.

## Fold assignment contract {#fold-assignment}

`folds.parquet` contains one row per block group:

| Column | Meaning |
| --- | --- |
| `run_id`, `comparison_set`, `session_id` | owner |
| `group_id` | stop or decision ID; all descendant rows inherit its fold |
| `order_time` | canonical start time used for ordering |
| `fold_id` | contiguous balanced test block `0..K-1` |

Assignments are deterministic and invariant to input row order: sort by `(order_time, group_id)`,
split unique groups into contiguous balanced blocks, and score every group exactly once. The typed
hash of `(session_id, group_id, fold_id)` is `fold_hash`. Missing order time, duplicated group IDs, or
an unassigned group raises. Forward-chaining, if used, is a separately named scheme and is never
silently substituted for blocked interpolation.

## Covariate pre/post-outcome timing contract {#covariate-timing}

The existing `PRE_OUTCOME_SAFE`, `POST_OUTCOME`, and `FUTURE_REFERENCING` registry remains the
authoritative model-build gate. Phase 8 changes two details:

- Replace `recent_switch_rate` with `recent_completed_switch_rate_k5`, computed at response/query time
  from completed trials.
- A `POST_OUTCOME` predictor is accepted only when the fitted row carries
  `response_time >= outcome_known_time`; setting `descriptive=True` without a physical gate is not a
  substitute for a post-outcome analysis.

Future-referencing peri-switch predictors remain descriptive; no within-stop timing gate makes them
predictive-safe.

## Decision-table contract {#decision-table}

`decision_arm_table` has exactly six rows per eligible decision, keyed by `(decision_id, arm)` with
arms `0..5` and exactly one `chosen == 1`. `decision_id` is the stable session/trial identifier for the
target trial. Decisions with the same chosen and current arm and decisions with no replay are retained.

Required decision-level fields include target/current trial indices and arms, previous outcome,
`window_start`, `window_end`, duration, session/animal keys, and behavior-model coverage. Arm-varying
fields include pre-outcome Q/value, nominal probability, visit/reward history, `has_arm_history`,
distance rank, and hard/soft replay mass for all/SWR/non-SWR events. Replay mass is zero—not NaN—when
a decision window contains no event of that type.

`decision_summary_table` contains one row per decision for Stage 6, derived from the same arm table and
window. No stop-grain response is allowed in either canonical model.

## Fit-validity and diagnostics contract {#fit-validity}

Every attempted cell produces a row, including failures. Required fields:

| Field | Rule |
| --- | --- |
| `fit_status` | `ok`, `rank_deficient`, `nonconverged`, `separated`, `insufficient_data`, or `fit_error` |
| `failure_reason` | non-empty unless `ok` |
| `design_rank`, `n_design_columns` | computed on the actual fitted design |
| `condition_number_scaled`, singular-value fields | computed after centering appropriate to estimator and unit-RMS scaling |
| `vif_centered_*` | conditional estimators use within-group-centered design |
| `n_clusters`, `covariance_type` | actual uncertainty unit/method |
| `converged`, `separated` | explicit booleans |

Rank-deficient or nonconverged fits invalidate all inference/comparison. Separated fits retain raw
diagnostics but invalidate affected-term inference and all likelihood comparisons. Only `fit_status=ok`
enters headline aggregation. A penalized sensitivity is labeled as a different estimator and is never
used in an ordinary LRT/AIC comparison.

## Term-level diagnostics table {#term-diagnostics}

The existing table is extended with `fit_status`, scaled term coefficient, estimability, robust-SE
availability, and exclusion reason. The old universal raw `abs(coef) < 10` rule is removed; scale-free
fit validity and explicit warnings own exclusion.

## Animal-balanced summary contract {#animal-summary}

For each estimand/term, first aggregate valid session estimates within animal, then report:

- all five per-animal effects and contributing session counts;
- median of animal effects;
- `n_animals_positive / n_animals_used`;
- fixed-seed animal-block bootstrap interval and leave-one-animal-out range where meaningful;
- an incomplete-animal warning when `n_animals_used < 5`.

Session signed-rank p-values may be retained only under names ending `_descriptive`. Neuron-level PFC
effects first aggregate to session and then animal under the dedicated PFC plan.
