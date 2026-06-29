# Shared contracts

[← back to PLAN.md](PLAN.md)

The canonical covariate **names** and **where they live**. Definitions/windows/units
are in [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the
spec); this file fixes the column names and the rule that they are **columns in the
canonical tables** (not a sidecar), so every consumer reads one source.

## canonical-covariate-names

Renames (old → canonical) and additions:

| old name | canonical name | change |
| --- | --- | --- |
| `recent_reward_rate` (per-arm, k=5 visits) | `reward_rate_arm_k10` | window 5→10 visits |
| `recent_trial_reward_rate` (all-arm, k=10 trials) | `reward_rate_trial_k10` | rename only |
| `leaky_q_value_diff` (α=0.3) | **removed** | replaced by `q_current` / `q_arm` |
| — | `p_reward_arm` | **new** per-arm nominal contingency value |
| — | `q_current` (per-stop), `q_arm` (per-arm) | **new** fitted-α leaky-Q value |

Unchanged (keep names/definitions): `reward_lag{1..10}`, `p_reward_current`, `rpe`,
`value_diff_patch`, `recent_switch_rate`, `is_after_switch`, `is_current`, `is_prev`,
`is_last_rewarded_elsewhere`, `arm_fe_*`, `arm_dist_rank`, `bs(tc)`, `onset_lag*`,
`hist_rcos*`, the patch analogues.

## where-the-covariates-live

Canonical covariates are **columns in the existing canonical tables**, written by an
in-place covariate-only update (no posterior re-rollup):

- `event_arm_table.pkl` — per **(event_id, arm)**: add/replace `reward_rate_arm_k10`,
  `p_reward_arm`, `q_arm`. (Drop/ignore the old `recent_reward_rate`.)
- `event_patch_table.pkl` — per **(event_id, patch)**: the patch aggregates of the
  above (sum/mean per patch, matching the existing patch-aggregation rule).
- `stop_table.pkl` — per **stop_id**: `reward_rate_trial_k10`, `q_current`, and the
  already-present `recent_switch_rate` / `is_after_switch` / contingency scalars
  (`p_reward_current`, `rpe`, `value_diff_patch`).

Update contract (the safety rule for the in-place rewrite):
- Recompute **only** the behavioral covariate columns; the **response/posterior
  columns** (`arm_count_hard`, `arm_count_soft`, `arm_posterior_mass`, the offsets,
  `is_swr`, timing) and all keys must be **byte-identical** before/after — assert this,
  write to a temp file, validate, then swap.
- Inputs are the trial-level SpatialBanditTask CSV (per-arm visit/reward history),
  the contingency CSV (`p_reward_port{0..5}`), and the per-animal-α Q-learner — **no
  `load_data`, no decoder rollup**.
- `q_arm` at the current arm == `q_current`; `p_reward_arm` at the current arm ==
  `p_reward_current`; trunk arms `reward_rate_arm_k10` = 0.0, never-visited = NaN;
  per-animal α recorded (table `.attrs` or a sidecar note).
