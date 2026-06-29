# Shared contracts

[← back to PLAN.md](PLAN.md)

The canonical covariate **names** and the **`canonical_covariates.pkl` schema** that
both phases depend on. Definitions/windows/units are in
[docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the spec);
this file fixes the column names and the on-disk shape so phase 1 produces and
phase 2 consumes exactly the same thing.

## canonical-covariate-names

Renames (old → canonical) and additions:

| old name | canonical name | change |
| --- | --- | --- |
| `recent_reward_rate` (per-arm, k=5 visits) | `reward_rate_arm_k10` | window 5→10 visits |
| `recent_trial_reward_rate` (all-arm, k=10 trials) | `reward_rate_trial_k10` | rename only |
| `leaky_q_value_diff` (α=0.3) | **removed** | replaced by `q_current` / `q_arm` |
| — | `p_reward_arm` | **new** per-arm nominal contingency value |
| — | `q_current` (scalar), `q_arm` (per-arm) | **new** fitted-α leaky-Q value |

Unchanged (keep names/definitions): `reward_lag{1..10}`, `p_reward_current`, `rpe`,
`value_diff_patch`, `recent_switch_rate`, `is_after_switch`, `is_current`, `is_prev`,
`is_last_rewarded_elsewhere`, `arm_fe_*`, `arm_dist_rank`, `bs(tc)`, `onset_lag*`,
`hist_rcos*`, the patch analogues.

## canonical-covariates-pkl

`Processed-Data/replay_content/canonical_covariates.pkl` — the single source every
runner merges. Two row granularities in one file is ambiguous, so write **two**
frames (or two files):

- `canonical_covariates_arm.pkl` — one row per **(animal, day, epoch, event_id, arm)**.
  Columns: keys + `reward_rate_arm_k10`, `p_reward_arm`, `q_arm`. Merged onto
  `event_arm_table` / `event_patch_table`(aggregated) / choice dataset by `(event_id, arm)`.
- `canonical_covariates_stop.pkl` — one row per **stop_id**. Columns: `stop_id` +
  `reward_rate_trial_k10`, `q_current`, plus the existing per-stop scalars
  (`p_reward_current`, `rpe`, `value_diff_patch`, `recent_switch_rate`,
  `is_after_switch`) re-exported so the rate/coupling runners have one merge source.
  Merged by `stop_id`.

Invariants:
- `q_arm` for the current arm equals `q_current` for that stop (same Q-learner).
- `reward_rate_arm_k10` keeps the structural conventions of the current per-arm
  function: trunk arms `0.0`, never-visited leaf arms `NaN`.
- α (per animal) used for `q_*` is recorded in the file's `.attrs` (or a sidecar)
  so the per-animal learning rate is reproducible.
- Both frames cover the full session set; coverage gaps (e.g., stops without a
  parsed contingency row) are left as `NaN`, not dropped, so the merge is lossless.
