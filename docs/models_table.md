# Model reference — every regression model, in R formula notation

The complete set of regression models across all families, written as R formulas.
The **single source of truth is `model_specs.py`** (+ the per-family runners); this
doc is a human-readable mirror — if a spec changes there, update here. Covariate
definitions are in [covariate_dictionary.md](covariate_dictionary.md) and
[behavior_model_covariates.md](behavior_model_covariates.md).

## Conventions & shorthands

Expanded once; used throughout:

```
BS_TC   := bs(tc, knots=c(1,2,3), Boundary.knots=c(0,4))         # within-stop time-course spline
REWK    := reward_lag1 + reward_lag2 + ... + reward_lag10         # 10-trial reward kernel
SELFLAG := onset_lag1 + ... + onset_lag10                         # raw self-history (autoregressive)
HISTCOS := hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3  # raised-cosine self-history
ARMFE   := arm_fe_0 + ... + arm_fe_7                              # 8 dummies for the 9 arms
```

Per-family fit conventions:

| family | estimator | response | link | offset | grouping / replication |
|---|---|---|---|---|---|
| Per-bin rate (B) | Poisson GLM | `count` (onsets per 15 ms bin) | log | `log(exposure)` | per session × population {swr, non_local_swr, non_local_no_swr} |
| MU uncertainty | Poisson GLM | `count` | log | `log(exposure)` | per session × 3 populations |
| Per-stop rate (A) | Poisson GLM | `n_events` per stop | log | `log(immobility_time)` | per session |
| Content arm | conditional Poisson over 9 arms/event ≡ `clogit(…, strata(event_id))` | `arm_count_hard` | log | — | per session × is_swr split {all, T, F} |
| Content patch | conditional Poisson over 3 patches/event | `patch_count_hard` | log | — | per session × {all, T, F} |
| Choice | conditional logit over candidate arms/stop ≡ `clogit(…, strata(stop))` | `chosen` | logit | — | per event-source |
| SWR power | OLS (`lm`) | `ripple_power_z` (event-level) | identity | — | per session; predictors z-scored |

`M0`/`P0` are the null (intercept-only, no arm/patch terms); the `arm_fe` main
effects first enter at `M0a`/`P1`. "conditional Poisson over arms within event" is fit
as Poisson with per-event fixed effects — the multinomial/`clogit` equivalent.

Every per-session fit carries `converged` / `separated` status flags and an
`n_convergence_warnings` count (`analysis.replay._fit_diagnostics`); the audited
aggregation surfaces per-(model, population) `n_sessions_flagged_nonconverged` /
`n_sessions_flagged_separated`. These are diagnostics on the retained cohort — the only
coefficient exclusion is the `|coef|>=10` quasi-separation filter.

## Per-bin rate — Family B
`fit_per_bin_rate_models.py` → `models_per_bin_rate/all_models.pkl`

```r
M0_const              count ~ 1
M1_rew1               count ~ reward_lag1
M2_rewhist            count ~ REWK
M2r_rewrate           count ~ reward_rate_trial_k10
M3_hist               count ~ SELFLAG
M4_hist_rew           count ~ REWK + SELFLAG
M5_time               count ~ BS_TC
M6_time_hist          count ~ BS_TC + SELFLAG
M7_time_hist_rew      count ~ BS_TC + REWK + SELFLAG
M8_full_switch        count ~ BS_TC + REWK + SELFLAG + is_after_switch
M9_rew_x_time         count ~ BS_TC + REWK + SELFLAG + reward_lag1:BS_TC
M3r_histcos           count ~ HISTCOS
M7r_time_histcos_rew  count ~ BS_TC + REWK + HISTCOS
```

## MU uncertainty — per-bin Poisson (M7r base + RL/value block)
`fit_uncertainty_rate.py` → `models_per_bin_rate/uncertainty_rate.pkl`

```r
MU_uncertainty  count ~ BS_TC + REWK + HISTCOS
                        + p_reward_current + value_diff_patch + recent_switch_rate
                        + trials_since_block_change + rpe_model
                        + switch_advantage_mu_preoutcome_sess + reward_entropy
                        + global_mu_update_abs_postoutcome_sess
                        + unchosen_patches_mu_update_abs_postoutcome_sess
```

## Per-stop rate — Family A
`fit_per_stop_rate_models.py`

```r
R0    n_events ~ 1
R1    n_events ~ is_after_omission + is_after_switch
R2    n_events ~ is_after_omission + is_after_switch + reward_rate_trial_k10
Rlag  n_events ~ REWK + is_after_switch
```

## Content — arm
`fit_replay_content_models.py --hierarchy arm` (per is_swr split {all, T, F})

```r
M0   arm_count ~ 1                        # null (uniform over arms) | strata(event_id)
M0a  arm_count ~ ARMFE
M1   arm_count ~ ARMFE + is_current
M2   arm_count ~ ARMFE + is_current + is_prev
M3   arm_count ~ ARMFE + is_current + is_prev + reward_rate_arm_k10
M4   arm_count ~ ARMFE + is_current + is_prev + reward_rate_arm_k10 + visit_recency_demeaned
M5   arm_count ~ ARMFE + is_current + is_prev + reward_rate_arm_k10 + visit_recency_demeaned
                 + is_last_rewarded_elsewhere
M6   arm_count ~ [M5 terms] + swr_x_current + swr_x_prev
M7   arm_count ~ [M6 terms] + swr_x_reward_rate_arm_k10 + swr_x_visit_recency_demeaned
                 + swr_x_is_last_rewarded_elsewhere
Mv   arm_count ~ [M5 terms] + p_reward_arm + q_arm_hmm
```

## Content — patch
`fit_replay_content_models.py --hierarchy patch` (per {all, T, F})

```r
P0   patch_count ~ 1                       # null | strata(event_id)
P1   patch_count ~ is_current_patch
P2   patch_count ~ is_current_patch + is_last_rewarded_patch_elsewhere
P3   patch_count ~ is_current_patch + is_last_rewarded_patch_elsewhere + cumulative_reward_patch
P4   patch_count ~ [P3 terms] + swr_x_current_patch + swr_x_last_rewarded_patch_elsewhere
Pv   patch_count ~ [P3 terms] + p_reward_arm_patch + q_arm_patch
```

## Contfrag content (SWR / MUA event sources)
`prepare_contfrag_replay_content_data.py` + content runner, `--event-source {SWR,MUA}`.
Same conditional-Poisson forms on the contfrag tables, truncated ladders:
`MODEL_HIERARCHY_SWR/MUA` = arm **M0–M5**; `MODEL_PATCH_SWR/MUA` = patch **P0–P3**.

## Choice
`compare_event_source_models.py` (`clogit(chosen ~ …, strata(stop))`)

```r
C_replay_only      chosen ~ replay_frac
C_behavior         chosen ~ visit_recency + cumulative_reward + reward_rate_arm_k10 + arm_dist_rank
C_full             chosen ~ visit_recency + cumulative_reward + reward_rate_arm_k10 + arm_dist_rank
                            + replay_frac
C_full_swr_split   chosen ~ visit_recency + cumulative_reward + reward_rate_arm_k10 + arm_dist_rank
                            + replay_frac_swr + replay_frac_non_swr
```

## SWR power
`fit_swr_power_model.py` (per-session `lm`; predictors z-scored per session)

```r
ripple_power_z ~ scale(reward_lag1) + scale(p_reward_current) + scale(rpe)
                 + scale(recent_switch_rate) + scale(value_diff_patch)
                 + scale(q_current) + scale(time_since_stop)
```

## Per-bin non-local rate — `is_swr` version (phase-2, first pass)

The primary phase-2 analysis of the **non-local event rate** and its SWR-vs-non-SWR
dissociation, per bin. Response `count` = non-local onset count per **15 ms bin**;
`family = poisson(log)`, `offset = log(exposure)`; rows are (stop × bin) to
`tmax = 4 s`. **Fit per session, aggregate coefficients across sessions** (signed-rank).
Specs: `model_specs.py` (`SWR_CONTRAST_RATE_G_SPECS` / `_S_SPECS`; content ladders
`SWR_CONTRAST_CONTENT_ARM_LADDER` / `_PATCH_LADDER`) — the single source of truth.
Runners: `scripts/rate/fit_swr_contrast_rate.py`, `scripts/content/fit_swr_contrast_content.py`
(import the specs; keep only data-assembly + fitting).

Shorthands / covariate blocks. The task dimensions are **the paper's variables
(Comrie et al.), all from one model (`default_hmm`)** — see `qvals_table.add_hmm_decision_variables`:
```
BASE      := bs(tc,knots=c(1,2,3),Boundary.knots=c(0,4)) + reward_lag1 + REWK_prev + HISTCOS   # = M7r nuisance base
REWK_prev := reward_lag2 + ... + reward_lag10          # former trials (dwell-clean)
HISTCOS   := hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3
PERISWITCH:= trials_since_switch + is_post_switch + trials_since_switch:is_post_switch  # peri-switch broken-stick (one construct; see Note 1b)
BLOCK     := trials_since_block_change  # learning across a contingency block
SWITCHRATE:= recent_switch_rate         # local switch volatility
VALUE     := hmm_relative_value        # relative value ΔV (paper's single value+decision axis; see Note 2)
RPE       := hmm_rpe                    # reward - Q_current_leaf (HMM; unscaled canonical RPE, see Note 6)
UNCERT    := reward_entropy            # HMM native reward-state entropy; NOT the port-specific reward_entropy_current_leaf (see Note 4)
UPDATE    := hmm_value_update          # global value update Σ_leaf|ΔQ| (HMM)
REWXTIME  := reward_lag1:bs(tc,knots=c(1,2,3),Boundary.knots=c(0,4))  # reward × within-stop time course (dwell-shape test; see Note 7)
```

### A. Stratified — each formula fit 3× to `NL_all`, `NL_swr`, `NL_noswr`
Comparing a coefficient across the three populations = the descriptive dissociation.

| model | formula (`count ~ … + offset(log exposure)`) | question |
|---|---|---|
| `G0_base` | `BASE` | Does the non-local rate have within-stop time structure, reward-history dependence, and self-history (refractoriness) — separately for all / SWR / non-SWR events? |
| `G1_periswitch` | `BASE + PERISWITCH` | Does the rate ramp around patch switches (asymmetric before/after)? |
| `G2_block` | `BASE + BLOCK` | Does the rate shift across a contingency block (learning)? |
| `G3_switchrate` | `BASE + SWITCHRATE` | Is the rate modulated by recent local switch volatility? |
| `G4_value` | `BASE + VALUE` | Is the rate modulated by relative value (ΔV = Switch − Stay, the paper's decision variable)? |
| `G5_rpe` | `BASE + RPE` | Is the rate modulated by reward prediction error (reward − expected value)? |
| `G6_uncert` | `BASE + UNCERT` | Is the rate higher when reward-state entropy (outcome uncertainty) is high? |
| `G7_update` | `BASE + UPDATE` | Is the rate higher when global value updating is large? |
| `G8_rewxtime` | `BASE + REWXTIME` | Does the current reward reshape the within-stop rate *profile* (not just shift its level)? — the dwell-shape control (rewarded stops are ~3.6× longer). |

### B. `is_swr` contrast — pooled two non-local streams (design B)
`is_swr:X` coefficients are the SWR-vs-non-SWR dissociation tests.

| model | formula (`count ~ … + offset(log exposure)`) | question |
|---|---|---|
| `S0_rate` | `is_swr + BASE` | Do SWR-coincident non-local events occur at a different **overall rate** than non-SWR ones, holding the shared time/reward/history structure? |
| `S1_rew1` | `BASE + is_swr + is_swr:reward_lag1` | Does the **current (consummatory) reward** gate the rate **differently** for SWR vs non-SWR events? (headline reward-gating dissociation) |
| `S2_base` | `is_swr * BASE` | Do the time-course, reward-history, and self-history each modulate the rate differently by SWR status? |
| `S3_periswitch` | `is_swr * (BASE + PERISWITCH)` | Does the peri-switch ramp of rate differ by SWR? |
| `S4_block` | `is_swr * (BASE + BLOCK)` | Does the across-block (learning) modulation differ by SWR? |
| `S5_switchrate` | `is_swr * (BASE + SWITCHRATE)` | Does the switch-volatility modulation differ by SWR? |
| `S6_value` | `is_swr * (BASE + VALUE)` | Does relative-value modulation of rate differ by SWR? |
| `S7_rpe` | `is_swr * (BASE + RPE)` | Does RPE modulation of rate differ by SWR? |
| `S8_uncert` | `is_swr * (BASE + UNCERT)` | Does entropy (uncertainty) modulation differ by SWR? |
| `S9_update` | `is_swr * (BASE + UPDATE)` | Does value-update modulation differ by SWR? |
| `S10_rewxtime` | `is_swr * (BASE + REWXTIME)` | Does the reward × within-stop-time-course (dwell-shape) effect differ by SWR? |

**Notes.** (1) **Single-model provenance:** the task dimensions all come from the
paper's HMM (`default_hmm`) — `hmm_relative_value` / `hmm_rpe` / `hmm_value_update`
(ported from Alison's `add_rpe_deltaq_to_hmm_plus` onto native HMM `Q`) + native
`reward_entropy` — so we don't mix the beta model's ΔV with HMM entropy. (1b) **CONTEXT
is split into three near-orthogonal rungs** (`PERISWITCH`, `BLOCK`, `SWITCHRATE`; pairwise
|r| ≤ 0.14, VIF ≈ 1 except within peri-switch), each tested individually rather than as one
lumped block. Peri-switch's three terms (`trials_since_switch`, `is_post_switch`, their
interaction; internal r ≈ 0.71) stay together — the interaction is meaningless without its
main effects, and jointly they are the paper's broken-stick ramp around a switch. (2) **Value
and decision are one dimension:** the paper fits a single relative-value axis (ΔV),
not separate value and decision covariates, so we follow it with the single
`hmm_relative_value`. Empirically this is corroborated cross-model: the beta model's
`q_chosen` and HMM ΔV are r=−0.67 collinear (measured on the production table, n≈103k;
a cross-model correlation, not the paper's intra-model value/decision collinearity).
`q_chosen` is a robustness re-param, not co-fit. (3) The reward kernel is split `reward_lag1` (current,
dwell-confounded) + `REWK_prev` (former, dwell-clean) so the current-vs-former gating
contrast separates. (4) `reward_entropy` is the HMM's native reward-state entropy; the
paper's port-specific `reward_entropy_current_leaf` is not in the trials-aligned fit and
appears only in Alison's exploratory notebooks (not `rl_helpers.py`), so whether it is a
clean reproducible column is unconfirmed — a follow-up if the exact variant is wanted.
(5) Per-session fit + across-session signed-rank (no random effects); every rung isolates
one dimension (no joint `*_all` kitchen-sink specs — they were dropped as overparameterized
on low-event sessions). (6) **RPE is the unscaled canonical
`reward − Q_leaf`.** Alison rescales Q before subtracting (`(Q+0.6)/1.25`, a cosmetic
"get into 0–1 zone" transform), yielding `reward − 0.8·Q_leaf − 0.48` — *not* an affine
transform of `reward − Q_leaf` (it reweights Q vs reward). We deliberately fit the
textbook RPE, not the plotting-rescaled variant. (7) **REWXTIME = `reward_lag1:bs(tc)`**
tests whether the current reward changes the *shape* of the within-stop rate profile, not
just its level. The per-bin design + `offset(log exposure)` already removes the pure
*duration* confound (rewarded stops are ~3.6× longer → more bins, not a higher per-bin
rate); BASE's additive `reward_lag1 + bs(tc)` still assumes a shared profile shape, and
this rung is what tests that assumption directly.

## Phase-2 refit classification

Which specs change under the phase-1/1.5 harmonization (→ coefficients move, true
refit + report) vs are covariate-unchanged (→ **parity-only**: refit reproduces the
design columns; verified by `_verify_harmonization_parity`, see
[plan phase-2](../.claude/docs/plans/covariate-harmonization/phase-2-refit-verify.md)).

| family | CHANGED (report old→new) | parity-only |
|---|---|---|
| MU uncertainty | MU_uncertainty (**done**) | — |
| SWR power | ripple_power_z model (model `rpe`/`q_current`) | — |
| Content arm | M3, M4, M5, M6, M7, Mv | M0, M0a, M1, M2 |
| Content patch | Pv | P0–P4 |
| Contfrag content | arm M3–M5 | arm M0–M2, patch P0–P3 |
| Choice | C_behavior, C_full, C_full_swr_split | C_replay_only |
| Per-bin rate (B) | — covariates unchanged; **verify** the 07-01 table regen / TMAX fix didn't move onset counts, else refit | all specs |
| Per-stop rate (A) | — | R0, R1, R2, Rlag |

Changed drivers: `reward_rate_arm_k10` (5→10 visits), the value block
`p_reward_arm`/`q_arm_hmm` (the value rungs C4/CS5/Mv now use the HMM per-arm
value `q_arm_hmm`, harmonized with the is_swr rate ladder; the beta `q_arm`
column is retained for robustness), model `rpe`/`q_current`, and the choice
`dist_to_arm`→`arm_dist_rank` swap.
