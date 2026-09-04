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
REWK    := known_reward_outcome_lag_1 + ... + known_reward_outcome_lag_10
                                                                # 10-known-outcome kernel
SELFLAG := onset_lag1 + ... + onset_lag10                         # raw self-history (autoregressive)
HISTCOS := hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3  # raised-cosine self-history
ARMFE   := is_candidate_arm_0 + ... + is_candidate_arm_7          # arm 8 is reference
```

Per-family fit conventions:

| family | estimator | response | link | offset | grouping / replication |
|---|---|---|---|---|---|
| Per-bin rate (B) | Poisson GLM | `count` (onsets per 15 ms bin) | log | `log(exposure)` | per session × population {swr, non_local_swr, non_local_no_swr} |
| MU uncertainty | Poisson GLM | `count` | log | `log(exposure)` | per session × 3 populations |
| Per-stop rate (A) | Poisson GLM | `n_selected_events` per stop | log | `log(immobility_duration_s)` | per session |
| Content arm | conditional Poisson over 9 arms/event ≡ `clogit(…, strata(event_id))` | `arm_count_hard` | log | — | per session × is_swr split {all, T, F} |
| Content patch | conditional Poisson over 3 patches/event | `patch_count_hard` | log | — | per session × {all, T, F} |
| Choice | conditional logit over candidate arms/stop ≡ `clogit(…, strata(stop))` | `chosen` | logit | — | per event-source |
| SWR power | OLS (`lm`) | `ripple_power_z` (event-level) | identity | — | per session; predictors z-scored |

`M0`/`P0` are the null (intercept-only, no arm/patch terms); candidate-arm fixed
effects first enter at `M0a`/`P1`. "conditional Poisson over arms within event" is fit
as Poisson with per-event fixed effects — the multinomial/`clogit` equivalent.

Every per-session fit carries `converged` / `separated` status flags and an
`n_convergence_warnings` count (`analysis.replay._fit_diagnostics`); the audited
aggregation surfaces per-(model, population) `n_sessions_flagged_nonconverged` /
`n_sessions_flagged_separated`. These are diagnostics on the retained cohort — the only
coefficient exclusion is the `|coef|>=10` quasi-separation filter.

### Covariate causal-hygiene (phase 4)

Every ladder runner enforces covariate timing at model-build via
`analysis.replay.covariate_timing.assert_predictor_timing`; a rung whose predictor is
post-outcome or future-referencing must be **declared descriptive**, else the run fails.

- **Per-stop R1/R2** use `known_reward_outcome_lag_1`, the newest physically
  known outcome at bout start, instead of a current-outcome flag. This is not
  a fixed current/previous-trial relabeling (see [rate_models.md](rate_models.md)).
- **Descriptive rungs** (retained under an explicit *association* / *peri-switch
  alignment* claim, NOT a forward prediction — `DESCRIPTIVE_CONTENT_RUNGS` /
  `DESCRIPTIVE_RATE_RUNGS` in `model_specs.py`):
  content `C6_rpe`, `C9_update` (RPE / value-update association), `C10_periswitch` +
  patch `CP2`/`CP3`/`CPS3` (peri-switch alignment); rate `G5`/`G7`/`S7`/`S9`
  (RPE/update association), `G1`/`S3` (peri-switch).
  `periswitch_index_trials` / `is_periswitch_post` are future-referencing
  (sign toward the *next* switch) and can only be descriptive;
  `rpe_postoutcome` / `reward_belief_mean_update_l1_all_ports_postoutcome`
  are post-outcome and were relabeled
  descriptive here rather than gated.
- **Never-visited encoding:** an unvisited leaf's
  `reward_fraction_candidate_arm_last10_visits` is `0.0` (not NaN), while
  `is_candidate_arm_never_visited` explicitly identifies novelty. The two enter
  together wherever recency/reward history is modeled.
- Stage-6 `is_stay ~ outcome * content` legitimately uses the post-outcome
  `is_after_omission`: its modeled unit (the stay/leave decision) is post-outcome by
  construction, so the timing gate does not apply.

### Cohort-audit tables

The **primary** fit runners emit a `<run>_cohort_audit.csv` next to their outputs: **one
row per (model, population)** stating who entered the model and who was dropped, so
complete-case filtering is visible rather than implied. Covered runners:
`fit_replay_content_models.py` (content base ladder, common cohort),
`fit_swr_contrast_content.py` and `fit_swr_contrast_rate.py` (the is_swr-contrast
ladders), `fit_per_stop_rate_models.py` (Family A) and `fit_per_bin_rate_models.py`
(Family B). The emitter is `analysis.replay.cohort_audit`. Counts are in the row's
`grain` (`event` for content, `stop` for the rate families). Drops are **mutually
exclusive**, assigned by priority:

`n_drop_qvals_gap` (mechanism B — a value covariate is missing on the ~896-event /
2056-stop tail) → `n_drop_own_predictor_missing` (this model's own predictor NaN, e.g.
an early-session reward-history gap) → `n_drop_nonfinite`
(a required value is present but invalid, such as ±Inf) → `n_drop_zero_response`
(Σcount==0) → `n_drop_common_cohort_other` (union complete-case: dropped only for
*another* rung's predictor) → `n_excluded_session` (data-level `min_*`/rank gates).

**Reconciliation invariant (asserted, fail-loud):**
`n_in − Σ n_drop_* == n_modeled`. Convergence/separation counts
(`n_sessions_flagged_*`) and the `|coef|>=10` quasi-separation counts (the separate
`<run>_term_diagnostics.csv`) are **session/term-level** and excluded from the
reconciliation. `scripts/qa/summarize_cohort_audits.py` stacks every audit CSV and flags
low-retention (`n_modeled/n_in < 0.9`) or biased (mechanism-A early / mechanism-B
qvals-tail) drops.

Also covered: `compare_event_source_models.py` maps its native `_freeze_common_cohort`
provenance (the content / patch / rate / **choice** blocks) onto the standardized schema
via `records_from_common_cohort_provenance`, writing `*_comparison_fits_cohort_audit.csv`.
**This audit is block-grained, not per-rung:** every model in a block shares ONE frozen
common cohort, so it emits one row per block with `model` = the block/analysis name
(`choice`, `choice_swr_split`, `patch_hierarchy`, …) rather than duplicating identical
counts per rung. The block's up-front selection — e.g. choice's valid-replay-stop filter
— is *definitional*, so `n_in` is the post-selection candidate count. The qvals-merging
auxiliary runners `fit_uncertainty_rate.py` (inner join → `n_drop_qvals_gap`) and
`fit_swr_power_model.py` (left join + per-session dropna on qvals-derived predictors →
`n_drop_qvals_gap`) each emit their own cohort audit. `fit_cross_history_rate.py` (a
bivariate self-/cross-history variant with no new qvals dependence) is **not** audited —
documented, not silently dropped.

## Per-bin rate — Family B
`fit_per_bin_rate_models.py` → `models_per_bin_rate/all_models.pkl`

```r
M0_const              count ~ 1
M1_rew1               count ~ known_reward_outcome_lag_1
M2_rewhist            count ~ REWK
M2r_rewrate           count ~ reward_fraction_last10_known_trials
M3_hist               count ~ SELFLAG
M4_hist_rew           count ~ REWK + SELFLAG
M5_time               count ~ BS_TC
M6_time_hist          count ~ BS_TC + SELFLAG
M7_time_hist_rew      count ~ BS_TC + REWK + SELFLAG
M9_rew_x_time         count ~ BS_TC + REWK + SELFLAG + known_reward_outcome_lag_1:BS_TC
M3r_histcos           count ~ HISTCOS
M7r_time_histcos_rew  count ~ BS_TC + REWK + HISTCOS
```

## MU uncertainty — per-bin Poisson (M7r base + RL/value block)
`fit_uncertainty_rate.py` → `models_per_bin_rate/uncertainty_rate.pkl`

```r
MU_uncertainty  count ~ BS_TC + REWK + HISTCOS
                        + task_reward_probability_chosen_arm + task_reward_probability_patch_mean_gap_best_vs_second
                        + patch_switch_fraction_last5_completed_trips
                        + trials_since_block_change + rpe_postoutcome
                        + relative_value_switch_minus_stay_preoutcome
                        + reward_belief_differential_entropy_leaf_mean_preoutcome
                        + reward_belief_mean_update_l1_all_ports_postoutcome
                        + reward_belief_mean_update_l1_alternative_patches_postoutcome
```

## Per-stop rate — Family A
`fit_per_stop_rate_models.py`

```r
R0    n_selected_events ~ 1
R1    n_selected_events ~ known_reward_outcome_lag_1
R2    n_selected_events ~ known_reward_outcome_lag_1 + reward_fraction_last10_known_trials
Rlag  n_selected_events ~ REWK
```

## Content — arm
`fit_replay_content_models.py --hierarchy arm` (per is_swr split {all, T, F})

The runner accepts only the current canonical table schema and validates every
requested rung before fitting. A stale/legacy table or one missing any requested
predictor aborts the complete request; models are never warning-skipped into a
partial ladder. Candidate identities, current-location indicators, and hard-count
responses are checked as scientific invariants. All rungs are then restricted to
the same successfully fitted session/event cohort and published as one staged
directory bundle, so a late failed rung cannot leave mixed-generation outputs.

```r
M0   arm_count ~ 1                        # null (uniform over arms) | strata(event_id)
M0a  arm_count ~ ARMFE
M1   arm_count ~ ARMFE + is_current_arm
M2   arm_count ~ ARMFE + is_current_arm + is_prev_arm
M3   arm_count ~ ARMFE + is_current_arm + is_prev_arm
                 + reward_fraction_candidate_arm_last10_visits
M4   arm_count ~ [M3 terms] + visit_recency_trials_event_leaf_centered_spline_1
                 + visit_recency_trials_event_leaf_centered_spline_2
                 + visit_recency_trials_event_leaf_centered_spline_3
                 + is_candidate_arm_never_visited
M5   arm_count ~ [M4 terms] + swr_x_is_current_arm + swr_x_is_prev_arm
M6   arm_count ~ [M5 terms]
                 + swr_x_reward_fraction_candidate_arm_last10_visits
                 + swr_x_visit_recency_trials_event_leaf_centered_spline_{1..3}
Mv   arm_count ~ [M4 terms] + task_reward_probability_candidate_arm
                 + reward_belief_mean_candidate_arm_preoutcome
```

## Content — patch
`fit_replay_content_models.py --hierarchy patch` (per {all, T, F})

```r
P0   patch_count ~ 1                       # null | strata(event_id)
P1   patch_count ~ is_current_patch
P2   patch_count ~ is_current_patch + reward_fraction_candidate_arm_last10_visits_patch
P3   patch_count ~ [P2 terms] + swr_x_is_current_patch
Pv   patch_count ~ [P2 terms] + task_reward_probability_candidate_arm_patch
                            + reward_belief_mean_candidate_arm_preoutcome_patch
```

## Contfrag content (SWR / MUA event sources)
`prepare_contfrag_replay_content_data.py` + content runner, `--event-source {SWR,MUA}`.
Same conditional-Poisson forms on the contfrag tables, truncated ladders:
`MODEL_HIERARCHY_SWR/MUA` = arm **M0–M4**; `MODEL_PATCH_SWR/MUA` = patch **P0–P2**.

## Choice
`compare_event_source_models.py` (`clogit(chosen ~ …, strata(stop))`)

```r
C_replay_only      chosen ~ replay_fraction
C_behavior         chosen ~ visit_recency_trials + is_candidate_arm_never_visited
                            + reward_fraction_candidate_arm_last10_visits
                            + reward_belief_mean_candidate_arm_preoutcome
                            + candidate_arm_distance_rank
C_full             chosen ~ [C_behavior terms] + replay_fraction
C_full_swr_split   chosen ~ [C_behavior terms]
                            + replay_fraction_swr + replay_fraction_non_swr
```

## SWR power
`fit_swr_power_model.py` (per-session `lm`; predictors z-scored per session)

```r
ripple_power_z ~ scale(known_reward_outcome_lag_1) + scale(task_reward_probability_chosen_arm)
                 + scale(rpe_postoutcome)
                 + scale(patch_switch_fraction_last5_completed_trips)
                 + scale(task_reward_probability_patch_mean_gap_best_vs_second)
                 + scale(reward_belief_mean_chosen_arm_preoutcome)
                 + scale(time_since_stop_start_s)
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
(Comrie et al.), all from the **beta-bernoulli** model (HMM retired 2026-07-13):
```
BASE      := bs(tc,knots=c(1,2,3),Boundary.knots=c(0,4))
             + known_reward_outcome_lag_1 + REWK_prev + HISTCOS
REWK_prev := known_reward_outcome_lag_2 + ... + known_reward_outcome_lag_10
HISTCOS   := hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3
PERISWITCH:= periswitch_index_trials + is_periswitch_post
             + periswitch_index_trials:is_periswitch_post
BLOCK     := trials_since_block_change  # learning across a contingency block
SWITCHRATE:= patch_switch_fraction_last5_completed_trips
VALUE     := relative_value_switch_minus_stay_preoutcome
RPE       := rpe_postoutcome
UNCERT    := reward_belief_differential_entropy_leaf_mean_preoutcome
UPDATE    := reward_belief_mean_update_l1_all_ports_postoutcome
REWXTIME  := known_reward_outcome_lag_1:bs(tc,knots=c(1,2,3),Boundary.knots=c(0,4))
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
| `G6_uncert` | `BASE + UNCERT` | Is the rate higher when belief entropy (outcome uncertainty) is high? |
| `G7_update` | `BASE + UPDATE` | Is the rate higher when global value updating is large? |
| `G8_rewxtime` | `BASE + REWXTIME` | Does the current reward reshape the within-stop rate *profile* (not just shift its level)? — the dwell-shape control (rewarded stops are ~3.6× longer). |

### B. `is_swr` contrast — pooled two non-local streams (design B)
`is_swr:X` coefficients are the SWR-vs-non-SWR dissociation tests.

| model | formula (`count ~ … + offset(log exposure)`) | question |
|---|---|---|
| `S0_rate` | `is_swr + BASE` | Do SWR-coincident non-local events occur at a different **overall rate** than non-SWR ones, holding the shared time/reward/history structure? |
| `S1_rew1` | `BASE + is_swr + is_swr:known_reward_outcome_lag_1` | Does the **most-recent-known outcome at bout start** gate the rate differently for SWR vs non-SWR events? |
| `S2_base` | `is_swr * BASE` | Do the time-course, reward-history, and self-history each modulate the rate differently by SWR status? |
| `S3_periswitch` | `is_swr * (BASE + PERISWITCH)` | Does the peri-switch ramp of rate differ by SWR? |
| `S4_block` | `is_swr * (BASE + BLOCK)` | Does the across-block (learning) modulation differ by SWR? |
| `S5_switchrate` | `is_swr * (BASE + SWITCHRATE)` | Does the switch-volatility modulation differ by SWR? |
| `S6_value` | `is_swr * (BASE + VALUE)` | Does relative-value modulation of rate differ by SWR? |
| `S7_rpe` | `is_swr * (BASE + RPE)` | Does RPE modulation of rate differ by SWR? |
| `S8_uncert` | `is_swr * (BASE + UNCERT)` | Does belief-entropy (uncertainty) modulation differ by SWR? |
| `S9_update` | `is_swr * (BASE + UPDATE)` | Does value-update modulation differ by SWR? |
| `S10_rewxtime` | `is_swr * (BASE + REWXTIME)` | Does the reward × within-stop-time-course (dwell-shape) effect differ by SWR? |

**Notes.** (1) **Single-model provenance:** the task dimensions all come from the
**beta-bernoulli** model — `relative_value_switch_minus_stay_preoutcome` (ΔV) /
`rpe_postoutcome` /
`reward_belief_mean_update_l1_all_ports_postoutcome` plus
`reward_belief_differential_entropy_leaf_mean_preoutcome` (the HMM
was retired 2026-07-13; there is one model now). (1b) **CONTEXT
is split into three near-orthogonal rungs** (`PERISWITCH`, `BLOCK`, `SWITCHRATE`; pairwise
|r| ≤ 0.14, VIF ≈ 1 except within peri-switch), each tested individually rather than as one
lumped block. Peri-switch's three terms (`periswitch_index_trials`,
`is_periswitch_post`, their
interaction; internal r ≈ 0.71) stay together — the interaction is meaningless without its
main effects, and jointly they are the paper's broken-stick ramp around a switch. (2) **Value
and decision are one dimension:** the paper fits a single relative-value axis (ΔV),
not separate value and decision covariates, so we follow it with the single beta
`relative_value_switch_minus_stay_preoutcome` (**Switch − Stay**; it is
approximately −(current stay
value), r = −0.952 with the stay value and −0.890 with `q_chosen`, so a positive
coefficient on
`relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm` means content
sharpens onto the current arm when
that arm's value is **low**). Chosen/candidate-arm reward-belief mean is a
robustness re-parameterization, not co-fit on the rate ladder. (3) The reward
kernel is split `known_reward_outcome_lag_1` (newest
outcome known at bout start) + `REWK_prev` (successively older known outcomes).
Because feedback can occur during a visit, lag 1 is not uniformly a fixed
current- or previous-trial label. (4)
`reward_belief_differential_entropy_leaf_mean_preoutcome` is the mean per-leaf
**Beta** differential entropy (from
`betadist_α/β`, `qvals_table.add_beta_belief_entropy`) — the beta analog of the retired
HMM `reward_entropy`; candidate-arm reward-belief variance is the separate
content-uncertainty covariate (see the content ladder).
(5) Per-session fit + across-session signed-rank (no random effects); every rung isolates
one dimension (no joint `*_all` kitchen-sink specs — they were dropped as overparameterized
on low-event sessions). (6) **RPE is the unscaled canonical
`reward − Q_leaf`.** Alison rescales Q before subtracting (`(Q+0.6)/1.25`, a cosmetic
"get into 0–1 zone" transform), yielding `reward − 0.8·Q_leaf − 0.48` — *not* an affine
transform of `reward − Q_leaf` (it reweights Q vs reward). We deliberately fit the
textbook RPE, not the plotting-rescaled variant. (7) **REWXTIME =
`known_reward_outcome_lag_1:bs(tc)`**
tests whether the current reward changes the *shape* of the within-stop rate profile, not
just its level. The per-bin design + `offset(log exposure)` already removes the pure
*duration* confound (rewarded stops are ~3.6× longer → more bins, not a higher per-bin
rate); BASE's additive `known_reward_outcome_lag_1 + bs(tc)` still assumes a
shared profile shape, and
this rung is what tests that assumption directly.

## Phase-2 refit classification

Which specs change under the phase-1/1.5 harmonization (→ coefficients move, true
refit + report) vs are covariate-unchanged (→ **parity-only**: refit should reproduce the
design columns; checked by `_verify_harmonization_parity`, see
[plan phase-2](../.claude/docs/plans/covariate-harmonization/phase-2-refit-verify.md)).

> **Current materialized status (2026-08-20): not yet a parity pass.** The
> key-aligned gate reproduces every declared unchanged column on shared rows but
> fails coverage because a transient rebuild error omitted three Replay sessions
> (218 → 215). Regenerate those sessions before treating any row as verified.

| family | CHANGED (report old→new) | parity-only |
|---|---|---|
| MU uncertainty | MU_uncertainty (**done**) | — |
| SWR power | ripple_power_z model (canonical model `rpe_postoutcome` and `reward_belief_mean_chosen_arm_preoutcome`) | — |
| Content arm | M3, M4, M5, M6, Mv | M0, M0a, M1, M2 |
| Content patch | P2, P3, Pv (patch reward rate replaces count) | P0–P1 |
| Contfrag content | arm M3–M4 | arm M0–M2, patch P0–P2 |
| Choice | C_behavior, C_full, C_full_swr_split | C_replay_only |
| Per-bin rate (B) | — covariates unchanged; **verify** the 07-01 table regen / TMAX fix didn't move onset counts, else refit | all specs |
| Per-stop rate (A) | — | R0, R1, R2, Rlag |

Changed drivers: candidate-arm reward fraction (5→10 visits), the
candidate-arm task-probability/reward-belief block, post-outcome RPE, and the
choice geometry control now named `candidate_arm_distance_rank`.
