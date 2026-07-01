# Covariate dictionary — canonical definitions across replay model families

Single source of truth for how each behavioral covariate is defined and how it
enters each model family, so coefficients are **comparable across rate / content /
choice / coupling**. Agreed decisions D1–D5 (2026-06-29) are recorded at the end.
This is the target the model specs and covariate-computation layer implement
against; the harmonization refactor brings the code to this spec.

**Status: implemented (2026-06-29, harmonization phase-1).** The covariate
definitions live in `analysis/replay/covariates.py`; the per-arm value block
(`reward_rate_arm_k10` at k=10, `p_reward_arm`, `q_arm`) is emitted by the event
builders from the per-stop contingency artifact (`p_reward_port`/`q_port`,
per-animal fitted α in `contingency_csv_alpha.json`); the value rungs are `Mv`
(arm) / `Pv` (patch); `q_current` enters the MU/coupling runners; the choice
baseline is `arm_dist_rank`. Coefficients are produced by the phase-2 re-fit on
the regenerated canonical tables.

## A. Entry-rule contract (forced by estimator type)

A covariate that is constant across arms within an event **cancels** in a
conditional/multinomial model, so it cannot be a main effect there. Hence:

| estimator | family | scalar covariates (reward / value / switch) | arm- or patch-relative covariates |
|---|---|---|---|
| **absolute** | per-bin rate (B), per-stop rate (A), coupling/power | **main effects** | — |
| **conditional (content)** | arm hierarchy, patch hierarchy | only as **`× is_current`** (or `× role`) interactions | **main effects** |
| **conditional (choice)** | next-arm logit | (cancel — omit) | **main effects** over the choice set |

"Parallel" therefore means **same construct, same window and units**, entered in
the form each estimator allows — *not* literally the same regressor everywhere. This
makes coefficients **admissible and same-construct**, not numerically interchangeable:
an absolute-rate main effect (log rate ratio) and a conditional-content interaction
(arm-relative log-odds) are different estimands — **interpret within an estimator
class**, compare *direction/consistency* across classes, not raw magnitudes.

## B. Canonical constructs

### 1. Reward memory
Named by **scope + window** so the same word never hides a different definition.

| canonical name | definition | window (N=10) | scope | units | enters |
|---|---|---|---|---|---|
| `reward_lag{1..10}` | outcome of the j-th most recent trial | 10 trials, per-lag | all-arm (per-trial) | binary | rate A (Rlag), rate B, MU, coupling — main effects |
| `reward_rate_trial_k10` | rewarded fraction of the last 10 **trials** | 10 trials | all-arm | fraction | rate A (R2), standalone only — **never co-fit with the kernel** (it is the kernel's mean → VIF ∞) |
| `reward_rate_arm_k10` | rewarded fraction of the last 10 **visits to that arm** | 10 arm-visits | per-arm | fraction | content (main effect), choice (main effect); rate/coupling via `× is_current` |

Change from prior code (implemented): the per-arm rate was **5-visit**
(`recent_reward_rate`) and is now **10-visit** (`reward_rate_arm_k10`). The all-arm
10-trial rate (`recent_trial_reward_rate`) was renamed `reward_rate_trial_k10`. Trunk
arms keep the structural `0.0`; never-visited leaf arms keep `NaN`.

### 2. Value — both nominal and experienced, available to every family

| canonical name | definition | scope | source | enters |
|---|---|---|---|---|
| `p_reward_current` | nominal contingency p(reward) of the current arm | scalar | `qvals_table` (DataJoint) | rate/coupling main effect |
| `p_reward_arm` | nominal contingency p(reward) of **each** arm | per-arm | `qvals_table` (DataJoint) | content/choice main effect; rate via `× is_current` |
| `q_current` | beta-model **belief** (value) of the current/chosen arm (`q_chosen` = `Q{leaf}`) | scalar | behavior model (`qvals_table`, `beta_stable_withleaf`) | rate/coupling main effect |
| `q_arm` | beta-model **belief** of **each** arm (`q_port{i}` = `Q{i+1}`) | per-arm | same behavior model | content/choice main effect |
| `leaky_q_value_diff` | per-**stop** leaky-Q best-minus-second **margin** (α=0.3) | scalar | `_leaky_q_value_diff` | **retained comparison** (MU only), like `is_after_switch` under D4 |

Changes (implemented): `p_reward_arm` **replaced the formerly-deferred
`nominal_reward_prob` column** and is now populated (`nominal_reward_prob`→`p_reward_arm`
in the `tables.py` schema + the viz registry, no orphan). `q_current`/`q_arm` are the
experienced-value covariate — now the **beta-model belief** (`beta_stable_withleaf`),
not a local leaky-Q learner: all value covariates were migrated to `qvals_table.pkl`
(DataJoint, exact `poke_out` alignment; retires the difflib contingency CSV). The
model's per-port `Q{i}` populate `q_port{i}` → `q_arm`, and `Q{leaf}` = `q_chosen` →
`q_current`. **`leaky_q_value_diff` is NOT this** — it is a *different*, still-local
construct (per-stop best-minus-second margin, α=0.3, computed in `covariates.py`), kept
as a reported MU-only comparison. `value_diff_patch` (nominal) remains a scalar
rate/coupling covariate. **`rpe` is now the model RPE everywhere** (`reward − q_chosen`,
from `qvals_table.pkl`; see [behavior_model_covariates.md](behavior_model_covariates.md)) —
after the migration every consumer (MU-uncertainty as `rpe_model`, plus `fit_swr_power`
and `_covariate_ranking`) uses the model RPE, not a nominal one.

### 3. Exploration — canonical continuous + retained binary comparison

| canonical name | definition | window | units | enters |
|---|---|---|---|---|
| `recent_switch_rate` | rolling mean of `(1 − is_stay)` over the last 5 stops, lagged 1 (no leakage) | 5 stops | fraction | **canonical** cross-family: rate/coupling main effect; content via `× is_current` |
| `is_after_switch` | current trial is a patch change | 1 trial | binary | **retained comparison** (per D4): the immediate-binary complement, available alongside `recent_switch_rate` wherever the estimator allows |

(D4: both kept — `recent_switch_rate` is the canonical graded covariate; `is_after_switch`
is a valued immediate-contrast comparison, not dropped.)

### 4. Spatial / role — already consistent, no change

`is_current`, `is_current_patch`, `is_prev`, `is_prev_patch`,
`is_last_rewarded_elsewhere` / `is_last_rewarded_patch_elsewhere` (disjoint
support), `arm_fe_{0..7}` (content baseline), `arm_dist_rank` (choice baseline,
replaces `dist_to_arm` for identifiability). Definitions unchanged.

### 5. Time / self-history — rate-only, no change

`bs(tc)` (fixed knots (1,2,3) s, [0,4]), the reward-lag kernel, `onset_lag{j}` /
`hist_rcos` self-history. These are per-bin within-stop terms and live only in the
rate families; conditional models access within-stop time only as `tsc_x_current`
(`× is_current`) when a within-stop content effect is tested.

## C. Per-family covariate matrix (target)

| construct | rate B (per-bin) | rate A (per-stop) | coupling | content (arm/patch) | choice |
|---|---|---|---|---|---|
| reward kernel `reward_lag{j}` | main | main (Rlag) | main | `× is_current` | — |
| `reward_rate_trial_k10` | (M2r standalone) | main (R2) | — | — | — |
| `reward_rate_arm_k10` | `× is_current` | — | `× is_current` | **main** | **main** |
| `p_reward_current` / `p_reward_arm` | main / `×is_current` | — | main | **`p_reward_arm` main** | **`p_reward_arm` main** |
| `q_current` / `q_arm` | main | — | main | **`q_arm` main** | **`q_arm` main** |
| `rpe`, `value_diff_patch` | main | — | main | (nominal via interaction if tested) | — |
| `recent_switch_rate` | main | main | main | `× is_current` | (main if predictive) |
| `is_after_switch` | main (M8) | main (R1+) | comparison | `× is_current` comparison | — |
| spatial role / `arm_fe` | — | — | — | main | choice-set + `arm_dist_rank` |
| `bs(tc)`, self-history | main | — | — | `tsc_x_current` only | — |

## D. Decisions of record (2026-06-29)

- **D1** — recent-reward-rate window standardized to **N = 10** in each construct's
  natural unit: `reward_rate_trial_k10` (10 trials, all-arm) and `reward_rate_arm_k10`
  (10 visits, per-arm). Per-arm window changes 5→10. **Caveat:** "10" is not the same
  *real-time* span (a single arm is visited far less often than trials elapse) — it is
  10 in each construct's own unit, not 10 in time.
- **D2** — add **per-arm nominal value** `p_reward_arm` to content/choice — by
  **replacing the formerly-deferred `nominal_reward_prob` column** (renamed and now
  populated), not a new parallel column.
- **D3** — **add** the experienced-value covariate `q_current` / `q_arm` (now the
  **beta-model belief** from `qvals_table`, not a local leaky-Q learner) as canonical,
  and **keep `leaky_q_value_diff`** (the per-stop margin, a different, still-local
  construct) as a *reported comparison*. (Revised from "drop": the two are different
  algorithms, and the margin is deliberately kept in the MU dissociation table.)
- **D4** — exploration canonical = **`recent_switch_rate`**; **keep `is_after_switch`**
  as the immediate-binary comparison (not dropped).
- **D5** — adopt the **entry-rule contract** (§A) as the standing rule.

## E. Notes / non-goals

- Coefficients remain comparable **only within an estimator class**; across classes
  (absolute vs conditional) the same construct is read as a rate ratio vs an
  arm-relative log-odds — interpret accordingly.
- The Q-learner α is fit per animal (not shared), so `q_*` is the animal's own
  learning timescale; report the per-animal α range.
- Everything stays **correlational**; this dictionary harmonizes *definitions*, not
  causal status.
- The full SpatialBanditTask **behavioral-model** latent variables (Beta-Bernoulli
  value, ΔV / Stay / Switch, value updates, Beta variance, HMM reward entropy) live
  in a separate per-stop table; see
  [`behavior_model_covariates.md`](behavior_model_covariates.md).

## F. Temporal causality (pre- vs post-outcome) — leakage audit

Each stop-level covariate is broadcast to the non-local events within that stop.
A covariate is safe iff its value is determined by information available **before**
the event it predicts. Classification of the covariates fitted by the rate models:

| covariate | class | available before event? |
|---|---|---|
| `p_reward_current`, `p_reward_arm`, `value_diff_patch`, `trials_since_block_change` | nominal schedule | yes — fixed by the block, known at trial start |
| `q_current` / `q_arm`, `switch_advantage_*_preoutcome_sess`, Beta `*_mu`/`*_var` `preoutcome` | model, **pre-outcome** belief (row *t* = belief entering trial *t*) | yes |
| `recent_switch_rate` | rolling, **lag-1** | yes — by construction excludes the current stop |
| `reward_entropy`, `state_entropy` | HMM belief entropy | yes (pre-outcome posterior) |
| `rpe` (= reward − q_chosen), `*_mu_update_abs_postoutcome_sess` | model, **post-outcome** (uses the current trial's realized reward) | **only if the event follows reward delivery** |

The three post-outcome covariates are the only leakage risk. **Empirically there is
no leakage:** on rewarded stops, 0.0% of non-local/SWR/MUA events occur before the
reward pump (min `t_since_pump` ≈ 1.0 s, median ≈ 3.8 s; `event_onset_table`), so
the realized outcome is always known ≥ ~1 s before any modeled event. Re-check this
exposure if the immobility/event-window definition changes.
