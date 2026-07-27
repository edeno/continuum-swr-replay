# Shared contracts — HMM→beta covariate mapping

[← back to PLAN.md](PLAN.md)

Every phase references this mapping. It is the single home for (a) which beta
column replaces each HMM covariate, (b) the beta belief-entropy derivation, and
(c) the standardized-name / timing-class each replacement carries. **Do not
re-derive these anywhere else.**

## Contract 1 — the covariate mapping

| Role | HMM covariate (remove) | Beta replacement | Source column status |
| --- | --- | --- | --- |
| per-arm value | `q_arm_hmm` | `q_arm` | attached in `tables.py:1315`; already `_z` via `EVENT_ARM_STANDARDIZED_COLS` (`tables.py:163`) |
| relative value (ΔV) | `hmm_relative_value` | `switch_advantage_mu_preoutcome_sess` | in `qvals_table.COVARIATES` (`qvals_table.py:106`); **needs `_z`** |
| reward prediction error | `hmm_rpe` | `rpe` (= `reward − q_chosen`, beta) | created `qvals_table.py:245`; in `COVARIATES:97`; **needs `_z`** |
| value-update magnitude | `hmm_value_update` | `global_mu_update_abs_postoutcome_sess` | in `COVARIATES` (`qvals_table.py:108`); **needs `_z`** |
| belief entropy (scalar) | `reward_entropy` | `beta_reward_entropy` (derived — Contract 2) | **does not exist — derive in Phase 1** |
| (HMM state entropy) | `state_entropy` | — (drop; used in no active ladder) | remove, no replacement |

Notes that bind the executor:

- **`q_arm` is the ONLY value construct after this migration.** The paper's beta
  value result is that content *avoids* high value (`swr_x_q_arm ≈ −2.5`), the
  opposite sign of the retired `swr_x_q_arm_hmm ≈ +1.06`. Phase 4 rewrites the
  narrative to the beta sign; Phases 1–3 must not preserve any "cite HMM / flag
  construct-dependence" hedge.
- **`swr_x_q_arm` already exists** as a pre-materialized column and is registered
  in `covariate_timing.DERIVED_COVARIATE_SOURCES` (`covariate_timing.py:67`)
  alongside `swr_x_q_arm_hmm`. The content `CS5` rung just switches which one it
  names.
- **Two distinct uncertainty covariates — do not conflate them.**
  (a) `arm_uncertainty` / `arm_uncertainty_z` = per-arm Beta variance, a per-arm
  *main effect* (`C13_armuncert`, `Mu`) — already beta, unchanged.
  (b) `beta_reward_entropy` = the per-stop *scalar* belief entropy that replaces
  the scalar `reward_entropy`. It enters as `beta_reward_entropy` in the rate
  rungs (`G6_uncert`/`S8_uncert`) AND as the scalar×current interaction
  `entropy_x_current = is_current * beta_reward_entropy_z` in content rung `C8`
  (matching the original `reward_entropy`×`is_current` structure). **`C8` is not
  redundant with `C13`** — scalar-entropy modulation of current-arm focus vs
  per-arm-variance main effect are different quantities. Keep both.

## Contract 2 — `beta_reward_entropy` definition

`reward_entropy` was the HMM reward-state Shannon entropy: a per-stop scalar
"how uncertain is the reward landscape right now." The beta equivalent is the
mean **differential** entropy of the per-leaf Beta posteriors at that stop.

The beta fit exposes the posterior shape parameters **directly** — columns
`betadist_α{i}` / `betadist_β{i}` (Greek names, `i` in 1..6), both ≥ 1 in this
fit — so compute the entropy from `(α, β)` rather than recovering concentration
from mean+variance (confirmed present in the build frame `vals`; source
`_spatial_bandit_rl.py:36`):

```python
import numpy as np
from scipy.special import betaln, digamma

def add_beta_belief_entropy(vals):
    """Add `beta_reward_entropy` = mean per-leaf Beta differential entropy.
    H_i = betaln(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b), averaged over 6 leaves.
    """
    ent = np.zeros(len(vals))
    for i in range(1, 7):
        a = vals[f"betadist_α{i}"].to_numpy(float)
        b = vals[f"betadist_β{i}"].to_numpy(float)
        ent += betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) \
               + (a + b - 2) * digamma(a + b)
    vals["beta_reward_entropy"] = ent / 6.0
    return vals
```

Notes: (1) differential entropy **can be negative** (unlike the HMM discrete
Shannon entropy) — expected and harmless; the covariate is used standardized.
(2) `α,β ≥ 1` here, so `digamma`/`betaln` are finite — no clipping needed; guard
with a loud error if the `betadist_α{i}`/`betadist_β{i}` columns are absent.
This per-trial scalar is computed in `_per_trial` (after
`add_experienced_value_columns`), added to `COVARIATES` so it survives the
`model_cols` filter, and added to `QVALS_STANDARDIZED_COLS` so
`beta_reward_entropy_z` exists (mirroring how `reward_entropy_z` was produced).

## Contract 3 — timing class of each replacement

The leakage/timing registry in `covariate_timing.py` must classify the
replacements exactly as the HMM originals were:

| beta column | timing class | registry set (`covariate_timing.py`) |
| --- | --- | --- |
| `q_arm` | pre-outcome-safe | already in `PRE_OUTCOME_SAFE` (line 41) |
| `switch_advantage_mu_preoutcome_sess` | pre-outcome-safe (name says `preoutcome`) | add to `PRE_OUTCOME_SAFE` |
| `rpe` | POST_OUTCOME | already in `POST_OUTCOME` (line 30) |
| `global_mu_update_abs_postoutcome_sess` | POST_OUTCOME (name says `postoutcome`) | add to `POST_OUTCOME` |
| `beta_reward_entropy` | pre-outcome-safe | add to `PRE_OUTCOME_SAFE` |

The `_x_current` interaction sources in `DERIVED_COVARIATE_SOURCES`
(`covariate_timing.py:49-54`) repoint: `rpe_x_current → ["rpe"]`,
`relval_x_current → ["switch_advantage_mu_preoutcome_sess"]`,
`update_x_current → ["global_mu_update_abs_postoutcome_sess"]`,
`entropy_x_current → ["beta_reward_entropy"]` (scalar, see Contract 1 note).
