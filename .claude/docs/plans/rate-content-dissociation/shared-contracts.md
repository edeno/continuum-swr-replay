# Shared contracts

> **SUPERSEDED — DO NOT EXECUTE (2026-07-29).** This file is retained only as
> historical design context. Canonical corrective work is owned by
> [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md); the only
> surviving downstream rate/content task is
> [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md).

[<- back to PLAN.md](PLAN.md)

This plan joins three model families without collapsing them into a single
meaningless "effect size." The contracts below are the shared surface every
phase must preserve.

## estimand-labels

| family | estimator | question | transformed effect |
| --- | --- | --- | --- |
| `rate_per_bin` | Poisson GLM with `log(exposure)` offset | How often do non-local event onsets occur per immobility time bin? | `rate_ratio = exp(coef)` |
| `rate_swr_contrast` | stratified/stacked per-bin Poisson GLM | Does a task effect differ between SWR-overlap and non-SWR streams? | `rate_ratio = exp(coef)` |
| `rate_continuous_swr` | per-bin Poisson GLM with continuous ripple mark | Does rate scale with ripple overlap/power instead of only binary `is_swr`? | `rate_ratio = exp(coef)` per mark unit |
| `content_arm` | conditional Poisson over 9 arms within event | Given an event, which arm gets decoded content? | `allocation_ratio = exp(coef)` |
| `content_patch` | conditional Poisson over 3 patches within event | Given an event, which patch gets decoded content? | `allocation_ratio = exp(coef)` |
| `content_continuous_swr` | conditional Poisson with continuous ripple mark interactions | Does content allocation scale with ripple overlap/power? | `allocation_ratio = exp(coef)` per mark unit |
| `choice` | conditional logit over candidate next arms | Does replay content predict the next chosen arm? | `choice_odds_ratio = exp(coef)` |

Never compare effect magnitudes across rows with different `effect_scale`.
Comparisons across families are directional and inferential: same session,
same covariate family, compatible sign interpretation, and explicit label.

## model-summary-schema

All downstream phases consume a long table with one row per coefficient term:

| column | meaning |
| --- | --- |
| `family` | One of the estimand families above. |
| `artifact_path` | Source pickle/CSV path. |
| `artifact_hash` | Hash of the source artifact bytes or provenance manifest. |
| `model` | Model name from `model_specs.py` or runner output. |
| `population` | `non_local`, `non_local_swr`, `non_local_no_swr`, `all`, `SWR`, `nonSWR`, `contrast`, or choice outcome label. |
| `animal`, `day`, `epoch` | Session key; nullable only for already-aggregated rows. |
| `term` | Predictor/contrast term without the `coef_` prefix. |
| `coef`, `se`, `pval` | Fitted coefficient and uncertainty when present. |
| `effect` | `exp(coef)`. |
| `effect_scale` | `rate_ratio`, `allocation_ratio`, or `choice_odds_ratio`. |
| `n_events`, `n_rows`, `n_bins`, `n_stops` | Available sample-size fields copied from source. |
| `complete_case_policy` | `event_wise`, `stop_wise`, `bin_wise`, or `not_applicable`. |
| `fit_status` | `fit`, `skipped`, `failed`, or `artifact_only`. |
| `skip_reason` | Machine-readable reason when `fit_status != "fit"`. |
| `is_key_term` | True for pre-registered terms used in headline tests. |
| `stale_flag` | True if artifact predates the current spec/contract or lacks required terms. |
| `notes` | Short machine-readable reason for skips/stale flags. |

Aggregated report tables are derived from this table. Do not hand-read
`coef_*` columns in later phases.

## fit-availability-contract

Every fitting runner touched by this plan must write a companion availability
table, even when the coefficient artifact contains only successful fits. This
prevents a missing coefficient from being silently interpreted as "no effect."

Availability rows are keyed by:

| column | meaning |
| --- | --- |
| `family`, `artifact_path`, `model`, `population` | Same labels as the summary table. |
| `animal`, `day`, `epoch` | Session key attempted. |
| `fit_status` | `fit`, `skipped`, or `failed`. |
| `skip_reason` | One of `missing_predictors`, `too_few_rows`, `too_few_events`, `zero_count`, `rank_deficient`, `within_group_constant`, `no_reward_contrast`, `fit_error`, `not_requested`, or another documented value. |
| `n_rows_before`, `n_rows_after` | Candidate rows before and after complete-case filtering. |
| `n_events_before`, `n_events_after` | Events before and after complete-case filtering when applicable. |
| `n_bins`, `n_stops` | Rate/choice sample-size fields when applicable. |
| `n_params` | Number of model predictors attempted. |
| `message` | Short diagnostic string for human triage. |

The Phase 2 summary builder must join this availability table. A missing key
coefficient is acceptable only when the corresponding availability row explains
why it is unavailable.

## complete-case-policy

Content and choice models condition on a full candidate set. Therefore:

- Arm content fits retain an event only if all required columns are observed for
  all 9 arm rows.
- Patch content fits retain an event only if all required columns are observed
  for all 3 patch rows.
- Choice fits retain a stop only if all required columns are observed for every
  candidate row in that stop's choice set.
- Rate fits can drop invalid bins/rows under the existing `log(exposure)`
  design, but must report retained `n_bins` and `n_events`.

Any downstream feature aggregation must propagate NaN rather than fabricating
zeros when a source event/stop failed complete-case filtering.

## artifact-freshness-contract

A content artifact is stale if any of the following are true:

- It contains coefficients for deprecated aliases such as
  `coef_recent_reward_rate` where the current spec expects
  `coef_reward_rate_arm_k10`.
- A current key term from `model_specs.py` is missing without an explicit
  unfit/skip reason.
- It was produced before event-wise complete-case filtering for SWR-contrast
  content fits.
- It lacks provenance after Phase 1 introduces provenance manifests.
- It lacks a fit-availability table after Phase 1 introduces that contract.

Phase 1 may write regenerated artifacts to an audited output directory first.
Only after parity and contract checks pass should canonical artifact paths be
replaced or documented as current.

## provenance-contract

Every new artifact directory gets a small manifest, preferably
`metadata.json`, containing:

- command line and script path;
- git commit and dirty-worktree summary;
- input artifact paths and hashes;
- model spec version/source file hash;
- date/time and machine-readable complete-case policy;
- output row/session/event counts;
- staged output directory or canonical output path.

Figure and report scripts must fail if a required source artifact has no
manifest or has `stale_flag=True`.

## support-classification-rule

Phase 3 uses these labels for each pre-registered covariate family:

| label | rule |
| --- | --- |
| `supported` | Non-stale rows, at least 5 animals and 20 sessions, Wilcoxon/FDR `q < 0.05` or animal-bootstrap 95% CI excluding 0, and at least 4/5 animals with the median-effect sign. |
| `supported_secondary` | Non-stale rows and a consistent effect, but sparse support, exploratory status, or only one of the statistical criteria above. |
| `not_supported` | Meets the sample-size thresholds but does not meet support criteria. |
| `underpowered` | Fewer than 5 animals, fewer than 20 sessions, or too many skipped/failed fits to distinguish absence from lack of power. |
| `blocked_stale` | Required artifact or key term fails freshness/provenance checks. |

Rate/content categories are derived only after assigning family support:

- `rate_only`: rate is `supported` or `supported_secondary`; content meets the
  sample-size thresholds and is `not_supported`.
- `content_only`: content is `supported` or `supported_secondary`; rate meets
  the sample-size thresholds and is `not_supported`.
- `both`: rate and content are both supported.
- `neither`: rate and content both meet sample-size thresholds and are
  `not_supported`.
- `unresolved`: any required family is `blocked_stale`, unavailable, or
  `underpowered`.

## key-terms

Primary terms for the first implementation:

| question | terms |
| --- | --- |
| immediate reward rate gate | `reward_lag1`, `is_swr:reward_lag1` |
| delayed reward-history rate suppression | `reward_lag2` through `reward_lag10` |
| SWR/non-SWR rate contrast | `is_swr`, `is_swr:<task_term>` |
| content locality | `is_current`, `is_current_patch`, `swr_x_current`, `swr_x_current_patch` |
| content reward/value | `reward_rate_arm_k10`, `q_arm`, `swr_x_reward_rate_arm_k10`, `swr_x_q_arm` |
| task-state content | `rpe_x_current`, `relval_x_current`, `entropy_x_current`, `update_x_current`, `switchrate_x_current` |
| behavior prediction | `replay_frac`, `replay_frac_swr`, `replay_frac_non_swr`, `replay_frac_residual` |
