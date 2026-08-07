# Phase 10b — Dependence-aware uncertainty and animal-balanced inference

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#dependence)

Replace optimistic model-based uncertainty with stop/trial-aware covariance or bootstrap methods and
make animal-balanced effects the primary cross-session summary. This phase preserves point-estimate
formulas and exact cohorts from earlier phases.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:151-188](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L151) — ordinary GLM covariance and result schema.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:138-170](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L138) — ordinary per-stop Poisson uncertainty/dispersion outputs.
- [scripts/rate/fit_swr_contrast_rate.py:143-188](../../../../scripts/rate/fit_swr_contrast_rate.py#L143) — stacked/SWR-rate model-based SEs.
- [tests/unit/test_replay_content_simulation.py:417-460](../../../../tests/unit/test_replay_content_simulation.py#L417) — explicit demonstration that trajectory autocorrelation undercovers nominal content SEs.
- [src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:57-135](../../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L57) — current animal-balanced pattern to generalize.
- [scripts/content/fit_replay_content_models.py:239-271](../../../../scripts/content/fit_replay_content_models.py#L239), [scripts/rate/summarize_per_bin_rate_models.py:51-95](../../../../scripts/rate/summarize_per_bin_rate_models.py#L51), and [scripts/rate/fit_cross_history_rate.py:144](../../../../scripts/rate/fit_cross_history_rate.py#L144) — remaining session-as-replicate summaries.

**Contracts referenced:**

- [Fit validity](shared-contracts.md#fit-validity) — only valid cells enter uncertainty/aggregation.
- [Animal-balanced summaries](shared-contracts.md#animal-summary) — required output fields.
- [Fold assignment](shared-contracts.md#fold-assignment) — bootstrap is inference; folds remain prediction and are not reused as pseudo-replicates.

**Designs referenced:** [dependence-aware estimation](designs.md#dependence).

## Tasks

- Implement the exact `alpha=0.05`, SHA-256-derived seed, CR1 correction, cluster minimum, reference
  distribution, bootstrap counts/success gate, animal aggregation, sign-flip tests, and BH rules in
  [the dependence design](designs.md#dependence). These are schema constants, not runner options.
- Make stop-cluster CR1 covariance primary for per-bin and stacked SWR-rate GLMs. Cluster on
  `stop_id` so temporal bins and paired event-type rows share a cluster. Save naive covariance only as
  explicit diagnostics (`bse_model_*`, `pval_model_*`). Record cluster count, correction, degrees of
  freedom, and covariance method.
- For per-stop Poisson, cluster on Phase-8 decision/trial ID. Retain Pearson/deviance dispersion and
  add only the locked quasi-Poisson sensitivity when the Pearson threshold is met; do not implement an
  outcome-selected negative-binomial alternative or compare AIC across likelihood types.
- Add the deterministic 2,000-attempt stop-block bootstrap for conditional content: sample whole stops
  within session, carry all events/category rows, assign fresh event group IDs, refit, and report the
  percentile interval, replicate covariance, and successful fraction. Wire both hard-count and
  soft-posterior-mass responses; fewer than 1,800 joint-valid replicates fails robust inference.
- Add a dependence-calibration simulation using the existing trajectory autocorrelation generator.
  Run the locked 400-replicate grid and require the stated 90% coverage / five-percentage-point
  improvement gates. A regression that restores anti-conservative coverage fails.
- Extract `aggregate_session_effects` from `contrast_aggregation.py` under the animal-summary contract.
  Migrate base content, per-stop/per-bin, uncertainty, cross-history, SWR power consumers, and Stage-6
  reporting. Require all five animals or an incomplete-animal warning. Relabel every session signed-rank
  field `_descriptive`.
- Implement paired animal aggregation for formal SWR interactions and model-score deltas. Separate fits
  remain descriptive; summary code cannot infer a difference by comparing significance labels.
- Add the schema loader/validator for the checked-in
  `analysis/replay/confirmatory_hypotheses.json` contract. Validate every entry against exact
  model/set/population/term identities, construct declared joint tests, apply one BH-FDR family across
  all entries, and hash the canonical registry bytes into run/release metadata. Initialize the file as
  `registry_status="locked"` with an empty `entries` array unless the analysis authors approve a
  non-empty reviewed commit before any corrected production fit. The safe default is therefore no
  confirmatory labels—not an executor-invented hypothesis or a family chosen after results.
- Construct confirmatory term/joint raw p-values only from the exact five-animal sign-flip procedures,
  then apply the locked single-family BH rule. Reject joint directional alternatives, missing animals,
  zero-RMS joint terms, or any attempt to feed session-level Wald/bootstrap p-values into FDR.
- Update reporting docs to state covariance/bootstrap unit, cluster count, bootstrap success, five animal
  effects, and the limitation of n=5. Remove language calling a session-cluster GEE population-valid.

## Deliberately not in this phase

- Choice/Stage-6 dataset/formula replacement (Phase 11), though their final reporting uses this helper.
- PFC resampling and neuron→session→animal aggregation, owned by the dedicated PFC plan.
- A new mixed-effects dependency. Animal-balanced summaries and resampling are sufficient for this plan.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_per_bin_cluster_groups_include_all_stop_rows` | bins and SWR/non-SWR pairs share stop cluster |
| `test_clustered_and_naive_fields_are_not_confused` | primary/diagnostic columns have explicit names/methods |
| `test_cr1_cluster_threshold_and_df_are_locked` | correction, G-1 reference df, and fewer-than-20-cluster failure follow the exact contract |
| `test_content_bootstrap_resamples_stops_and_rekeys_events` | no group collision or partial-stop resample |
| `test_content_bootstrap_attempts_seed_and_success_gate` | production uses 2,000 deterministic attempts, percentile endpoints, and fails below 1,800 valid vectors |
| `test_dependence_coverage_improves` | fixed-seed block interval coverage exceeds nominal model-based coverage within tolerance |
| `test_overdispersion_triggers_only_quasipoisson_sensitivity` | the 1.5/30-cluster boundary is exact and no NB/AIC branch exists |
| `test_animal_summary_balances_unbalanced_sessions` | duplicating sessions for one animal does not change across-animal center |
| `test_missing_animal_warns_and_blocks_headline` | 4-animal result cannot masquerade as 5-animal evidence |
| `test_paired_effects_aggregate_within_animal_first` | pairing/order and denominator are correct |
| `test_confirmatory_registry_resolves_and_hashes_exactly` | unknown set/model/population/term, draft status, hash drift, or a second family fails; a locked empty registry yields no confirmatory labels |
| `test_confirmatory_signflip_and_bh_are_exact` | all 32 synchronized animal sign patterns, alternatives, joint statistic, p-value ties, and q=0.05 BH output match hand calculations |
| one-session bootstrap smoke (slow) | records timing, memory, successful fraction, and stable interval at production-like event count |

## Fixtures

Use an autocorrelated event-trajectory simulation, clustered stop-bin counts, five animals with strongly
unbalanced session counts, one missing-animal case, and one bootstrap replicate with a failed fit. The
slow smoke uses one cached session after checking available CPU/memory.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- The primary covariance unit matches the repeated-data unit for each family.
- Model-based p-values remain only as clearly named diagnostics.
- Bootstrap samples whole scientific units and reports failed-replicate fraction.
- Across-animal summaries cannot be dominated by animals with more sessions/neurons.
- No claim of greater power is made; uncertainty may widen and significance may disappear.
- Every task is implemented, decision/PFC dataset changes remain out of scope, and session-as-primary reporting paths are removed.
- Validation and calibration tests pass, slow tests are marked, and Monte Carlo tolerances are prespecified.
- Production names/docstrings do not mention this plan or phase numbers; uncertainty/inference docs are updated.
