# Shared contracts

[← back to PLAN.md](PLAN.md)

Canonical covariate **names** and **where they live**. Definitions/windows/units are
in [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) (the spec);
this file fixes column names and the rule that they are **columns in the canonical
tables**, produced by the table **builders** (replay + contfrag) and the table
**regeneration** — one source every consumer reads.

## canonical-covariate-names

| old name | canonical name | change |
| --- | --- | --- |
| `recent_reward_rate` (per-arm, k=5 visits) | `reward_rate_arm_k10` | window 5→10 visits |
| `recent_trial_reward_rate` (all-arm, k=10 trials) | `reward_rate_trial_k10` | rename only (assert byte-identical) |
| `nominal_reward_prob` (per-arm, deferred NaN) | `p_reward_arm` | populated (per-arm nominal value) + renamed |
| `q_chosen` (contingency table) | `q_current` | rename + unify |
| — | `q_arm` | **new** per-arm fitted-α leaky-Q level |
| `leaky_q_value_diff` (α=0.3, per-stop margin) | *kept* | **not dropped** — retained comparison (MU) |

Unchanged: `reward_lag{1..10}`, `p_reward_current`, `rpe`, `value_diff_patch`,
`recent_switch_rate`, `is_after_switch`, `is_current`, `is_prev`,
`is_last_rewarded_elsewhere`, `arm_fe_*`, `arm_dist_rank`, `bs(tc)`, `onset_lag*`,
`hist_rcos*`, the patch analogues. **Both** the replay (`tables.py`) and contfrag
(`analysis/contfrag/tables.py`) builders use these definitions.

## where-the-covariates-live

Canonical covariates are **columns in the canonical tables**, emitted by the builders:

- `event_arm_table.pkl` (+ contfrag `event_arm_table_{swr,mua}.pkl`) — per **(event_id, arm)**: `reward_rate_arm_k10`, `p_reward_arm`, `q_arm` (+ existing `visit_recency*`, `cumulative_reward`, spatial role).
- `event_patch_table.pkl` (+ contfrag) — per **(event_id, patch)**: patch aggregates. **Aggregator per column:** `reward_rate_arm_k10` and `p_reward_arm` are fractions/probabilities → **patch mean** (not sum), trunk arms excluded; `q_arm` → patch mean; match the existing per-patch rule for the others.
- `stop_table.pkl` — per **stop_id**: `reward_rate_trial_k10`, `q_current`, plus existing `recent_switch_rate` / `is_after_switch` / contingency scalars (`p_reward_current`, `rpe`, `value_diff_patch`) and the retained `leaky_q_value_diff`.

## rebuild-integrity contract (the safety rule)

The tables are **regenerated** by the builders (reusing existing decodes — no
re-decode). Because the rollup is deterministic on fixed decodes:

- the regenerated **response/posterior columns** (`arm_count_hard`, `arm_count_soft`,
  `arm_posterior_mass`, offsets, `is_swr`, keys) must **reproduce** the pre-rebuild
  tables → confirms the rebuild changed *only* covariates;
- the **covariate columns** are the new canonical ones;
- keep a **backup** of the pre-rebuild tables for the parity check (phase 2).

Invariants: `q_arm` at the current arm == `q_current`; `p_reward_arm` at the current
arm == `p_reward_current`; trunk arms `reward_rate_arm_k10` = 0.0, never-visited = NaN;
per-animal α recorded in a **sidecar JSON** (not `.attrs` — pandas drops those); a
single canonical `recent_switch_rate` window (5 stops) used by every family.
