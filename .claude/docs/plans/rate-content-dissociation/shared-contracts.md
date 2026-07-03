# Shared contracts

[<- back to PLAN.md](PLAN.md)

This plan joins three model families without collapsing them into a single
meaningless "effect size." The contracts below are the shared surface every
phase must preserve.

## estimand-labels

| family | estimator | question | transformed effect |
| --- | --- | --- | --- |
| `rate_per_bin` | Poisson GLM with `log(exposure)` offset | How often do non-local event onsets occur per immobility time bin? | `rate_ratio = exp(coef)` |
| `rate_swr_contrast` | stratified/stacked per-bin Poisson GLM | Does a task effect differ between SWR-overlap and non-SWR streams? | `rate_ratio = exp(coef)` |
| `content_arm` | conditional Poisson over 9 arms within event | Given an event, which arm gets decoded content? | `allocation_ratio = exp(coef)` |
| `content_patch` | conditional Poisson over 3 patches within event | Given an event, which patch gets decoded content? | `allocation_ratio = exp(coef)` |
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
| `is_key_term` | True for pre-registered terms used in headline tests. |
| `stale_flag` | True if artifact predates the current spec/contract or lacks required terms. |
| `notes` | Short machine-readable reason for skips/stale flags. |

Aggregated report tables are derived from this table. Do not hand-read
`coef_*` columns in later phases.

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
- output row/session/event counts.

Figure and report scripts must fail if a required source artifact has no
manifest or has `stale_flag=True`.

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
