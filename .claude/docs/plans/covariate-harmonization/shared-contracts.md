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
| `nominal_reward_prob` (per-arm, deferred NaN) | `p_reward_arm` | populated (per-arm nominal value) + renamed; source now `qvals_table` (phase-1.5) |
| `q_chosen` (contingency table) | `q_current` | rename + unify; now the **behavior-model belief** from `qvals_table` (phase-1.5) |
| — | `q_arm` | per-arm **behavior-model belief** (`Q`, Beta-Bernoulli) from `qvals_table` — **not** leaky-Q (phase-1.5 supersedes D3) |
| `leaky_q_value_diff` (α=0.3, per-stop margin) | *kept* | **not dropped** — retained comparison (MU); the only leaky-Q construct that remains |

Unchanged: `reward_lag{1..10}`, `p_reward_current`, `rpe`, `value_diff_patch`,
`recent_switch_rate`, `is_after_switch`, `is_current`, `is_prev`,
`is_last_rewarded_elsewhere`, `arm_fe_*`, `arm_dist_rank`, `bs(tc)`, `onset_lag*`,
`hist_rcos*`, the patch analogues. **Both** the replay (`tables.py`) and contfrag
(`analysis/contfrag/tables.py`) builders use these definitions.

## where-the-covariates-live

Canonical covariates are **columns in the canonical tables**, emitted by the builders:

- `event_arm_table.pkl` (+ contfrag `event_arm_table_{swr,mua}.pkl`) — per **(event_id, arm)**: `reward_rate_arm_k10`, `p_reward_arm`, `q_arm` (+ existing `visit_recency*`, `cumulative_reward`, spatial role). Phase-1.5: `p_reward_arm`/`q_arm` are joined in from `qvals_table` by `attach_contingency_value_block` (`p_reward_port{arm}`/`q_port{arm}`), not computed in the builder.
- `event_patch_table.pkl` (+ contfrag) — per **(event_id, patch)**: patch aggregates. **Aggregator per column:** `reward_rate_arm_k10` and `p_reward_arm` are fractions/probabilities → **patch mean** (not sum), trunk arms excluded; `q_arm` → patch mean; match the existing per-patch rule for the others.
- `stop_table.pkl` — per **stop_id**: `reward_rate_trial_k10`, `q_current`, plus existing `recent_switch_rate` / `is_after_switch` / contingency scalars (`p_reward_current`, `rpe`, `value_diff_patch`) and the retained `leaky_q_value_diff`.

## rebuild-integrity contract (the safety rule)

The tables are **regenerated** by the builders (reusing existing decodes — no
re-decode). Because the rollup is deterministic on fixed decodes:

- the regenerated **response/posterior columns** must **reproduce** the pre-rebuild
  tables (the rollup has no RNG, so this is logically deterministic) → confirms the
  rebuild changed *only* covariates. Tolerance: **exact equality** on integer/key
  columns (`arm_count_hard`, offsets, `is_swr`, ids) and **`np.allclose` (atol=1e-9,
  rtol=1e-6, equal NaN)** on the float reductions (`arm_count_soft`,
  `arm_posterior_mass`) — to absorb summation-order/library drift across a months-later
  re-run; any non-`allclose` column is a **hard failure to investigate**, not a pass;
- the **covariate columns** are the new canonical ones;
- keep a **backup** of the pre-rebuild tables for the parity check (phase 2).

Invariants: `q_arm` at the current arm == `q_current`; `p_reward_arm` at the current
arm == `p_reward_current` (both verified in `_verify_covariate_invariants.py` /
`test_qvals_table.py`); trunk arms `reward_rate_arm_k10` = 0.0, never-visited = NaN;
a single canonical `recent_switch_rate` window (5 stops) used by every family.
(Phase-1.5 note: there is **no fitted α** to record — `q_*` come from the external
behavior model in `qvals_table`, not a local leaky-Q fit; the model fit identity is
recorded per [phase-1.5](phase-1.5-behavior-model-covariates.md).)
