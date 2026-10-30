# Model reference — every regression model, in R formula notation

The complete set of regression models across all families, written as R
formulas. The **single source of truth is
`src/continuum_swr_replay/analysis/replay/model_specs.py`** (plus the runner of
the SWR-power model, the one auxiliary model instantiated in its runner); this
file is the human-readable mirror. Covariate definitions live in
[covariate_dictionary.md](covariate_dictionary.md) and
[behavior_model_covariates.md](behavior_model_covariates.md); the design
narrative (why each rung carries what it does, and what was retired) is
[design/model_history.md](design/model_history.md).

This is a **catalog**: specifications only. Fitted effects live in the audits
under [`.claude/docs/reviews/`](../.claude/docs/reviews/), in [per_bin_rate_results.md](per_bin_rate_results.md),
and in `paper/claims_table.md`.

## How to read this file

Every section that mirrors an executable ladder in `model_specs.py` carries
**exactly one** fenced block whose info string is `rungs`:

- a line that starts in column 0 and is not a `#` comment begins with a **rung
  name** — its first whitespace-delimited token;
- indented lines are continuations of the formula above them;
- blank lines and `#` comment lines are ignored.

`tests/unit/test_models_table_mirrors_specs.py` parses those blocks and asserts,
per family, that the rung set equals the executable ladder's keys. Adding a rung
to `model_specs.py` without adding it here fails that test. A section without a
`rungs` block (the SWR-power model below) has no ladder to mirror.

## Conventions & shorthands

Expanded once; used throughout:

```
BS_TC   := bs(tc, knots=c(1,2,3), Boundary.knots=c(0,4))          # within-stop time-course spline
BS_ARR  := bs(time_since_last_well_arrival_s, knots=c(1,2,4,6),   # arrival-anchored time course,
              Boundary.knots=c(0,8))                              # fit on the [0,8] s window only
REWBASIS:= reward_history_rcos_0 + reward_history_rcos_1 + reward_history_rcos_2
           + reward_history_incomplete
                                                                  # raised-cosine reward-history
                                                                  # kernel over the last 10 known
                                                                  # outcomes + its structural-
                                                                  # missingness nuisance indicator
HISTCOS := hist_rcos_0 + hist_rcos_1 + hist_rcos_2 + hist_rcos_3  # raised-cosine self-history over
                                                                  # a 10-bin (150 ms) window
CROSSCOS:= cross_hist_rcos_0 + ... + cross_hist_rcos_3            # HISTCOS over the OTHER stream's
                                                                  # onsets (cross-history coupling)
OFFCOS  := offhist_rcos_0 + ... + offhist_rcos_3                  # HISTCOS over the stream's own
                                                                  # event ENDS (duration-aware)
XOFFCOS := cross_offhist_rcos_0 + ... + cross_offhist_rcos_3      # CROSSCOS over the other stream's
                                                                  # event ENDS (duration-aware)
ARMFE   := is_candidate_arm_0 + ... + is_candidate_arm_7          # arm 8 is reference
RECSPL  := visit_recency_trials_event_leaf_centered_spline_{1,2,3}
           + is_candidate_arm_never_visited
```

`BS_TC` / `BS_ARR` are `patsy` `bs(...)` bases in the code
(`bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)` and
`bs(time_since_last_well_arrival_s, knots=(1.0, 2.0, 4.0, 6.0), lower_bound=0.0,
upper_bound=8.0)`); the knots are fixed globally, never per-session quantiles, so
a fitted curve is reconstructable from the saved coefficients and the spline
coefficients are comparable across sessions. All per-bin models share one bin
width (`BIN_S = 15 ms`) so the ladder is nested and AIC / CV / deviance / GOF are
directly comparable.

### Rung-name suffixes

| suffix | means |
|---|---|
| `nt` | **n**o **t**ime/length controls — the twin fitted on `BASEc_NT`, i.e. without the within-stop-time and event-length current/previous-arm products. Reports the **total** SWR effect where the `BASEc` rung reports the length-adjusted (direct) one. |
| `b` | carries the **previous-goal** control (`is_last_rewarded_noncurrent_arm` and its SWR interaction) on top of its parent rung. |
| `r` | self-history block is the **r**aised-cosine basis (`HISTCOS`) rather than per-lag dummies. |
| `a` | time-course block is anchored on the **a**rrival clock (`BS_ARR`) instead of the bout clock. |

### Bundle names and suffixes

Content bundles are directories under `Processed-Data/replay_content/`. The
**base name** is `models/` (arm) or `models_patch/` (patch); a ContFrag run
(`--event-source {SWR,MUA}`) extends the base name itself with
`_{source}_{area}` — `models_swr_HPC`, `models_patch_mua_HPC` — so SWR/HPC,
SWR/mPFC and MUA/HPC never overwrite each other. The suffixes below then append
to whichever base name that produced, in this order:

| suffix | added when |
|---|---|
| `_swr_T` / `_swr_F` | `--swr-subset {T,F}` — the ladder refit on SWR-only / non-SWR-only events (`models_swr_T`, `models_patch_swr_F`, and on a ContFrag base `models_swr_HPC_swr_T`) |
| `_event_grain` | `--grain event` |
| `_hard_count` | `--response hard_count`, where that is not the estimator's own default response |
| `_leaf` | `--leaf-only` — the six-leaf candidate set |
| `_dm` | `--estimator dirichlet_multinomial`; appended **last** so an estimator's artifacts sort together |
| `-` joined rung names | a **partial** ladder (any rung set other than the default) is keyed by the rungs it fits, because it is fitted on its own complete-case cohort and must never overwrite a canonical bundle |

So the documented belief chain — `--leaf-only --models
M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu` — is both leaf-only and partial and lands in
`models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/`, **not** a bare `models_leaf/`.

The `is_swr` content-contrast runner writes flat artifacts into `models_swr_contrast/` instead (outside `models/`, which the arm ladder replaces whole):
`swr_contrast_content_{arm,patch,audited}{,_dm,_dm_len}{,_rewarded,_omitted}.pkl`,
each with its `_cohort_audit*.csv`,
`_term_diagnostics*.csv` and `_provenance*.json` sidecars. The estimator suffix
precedes the outcome-stratum suffix. The rate contrast (`swr_contrast_rate_*`) and the per-bin family write to
`models_per_bin_rate/`; the per-stop family to `models_per_stop_rate/`.

**Stop-cohort suffixes (rate artifacts).** Every rate runner takes
`--stop-cohort {at_well,all_stops,in_transit}`, and the cohort it was given
names the artifact: the canonical **`at_well`** (well-anchored stops,
`occupancy_state ∈ {at_well, mixed}`) keeps the unsuffixed name, `all_stops`
appends **`_allstops`** and `in_transit` appends **`_intransit`** — so
`models_per_bin_rate/all_models.pkl`, `all_models_allstops.pkl` and
`all_models_intransit.pkl` are the same ladder on three populations. The cohort
is written onto every fitted row as a `stop_cohort` column, recorded in the
provenance sidecar, and carried in `model_comparison.METADATA_COLUMNS`, so
`compare_nested_models` NaNs a comparison across two cohorts instead of
differencing it. The two content runners take no such flag: the content ladders
are defined only at a well, so they always apply the event-grain rule. The rules
themselves are in
[design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md).

### Estimators

| family | estimator | status | response | link | offset | grouping / replication |
|---|---|---|---|---|---|---|
| Per-bin rate (B) | Poisson GLM | shipped | `count` (onsets per 15 ms bin) | log | `log(exposure)` | per session × population {swr, non_local_swr, non_local_no_swr} |
| MU uncertainty | Poisson GLM | shipped (auxiliary) | `count` | log | `log(exposure)` | per session × 3 populations |
| Cross-history | Poisson GLM | auxiliary (written only with `--write`) | `count` (the target stream's onsets per bin) | log | `log(exposure)` | per session × direction {non-SWR ← with-SWR history, with-SWR ← non-SWR history} |
| Per-stop rate (A) | Poisson GLM | shipped | `n_selected_events` per stop | log | `log(immobility_duration_s)` | per session |
| Content arm | conditional Poisson over 9 arms/event ≡ `clogit(…, strata(event_id))` | **shipped** | `arm_fraction` (per-event allocation fraction) | log | — | per session × `is_swr` split {all, T, F}; SEs event-clustered |
| Content patch | conditional Poisson over 3 patches/event | **shipped** | `patch_fraction` | log | — | per session × {all, T, F}; SEs event-clustered |
| Content, timebin-count response | same conditional Poisson | **retired as primary**; retained behind `--response hard_count` for the estimator comparison and the before/after audit | `arm_count_hard` / `patch_count_hard` | log | — | as above, but weighted by decoded timebins |
| Content, DM alternative | Dirichlet-multinomial over the same 9 arms / 3 patches per event | comparison estimator | `arm_count_hard` / `patch_count_hard` | log (softmax mean) | — | per session × population; SEs event-level by construction |
| Choice | conditional logit over candidate arms/stop ≡ `clogit(…, strata(stop))` | shipped | `chosen` (event grain) | logit | — | per event-source |
| SWR power | OLS (`lm`) | shipped (auxiliary) | `ripple_power_z` (event-level) | identity | — | per session; predictors z-scored |

`M0` / `P0` are the null (intercept-only, no arm/patch terms); candidate-arm
fixed effects first enter at `M0a` / `P1`. "Conditional Poisson over arms within
event" is fit as Poisson with per-event fixed effects — the
multinomial / `clogit` equivalent.

**Content response = the per-event allocation fraction.** Each event contributes
one observation of unit mass (`arm_fraction` sums to 1 over its nine arms,
`patch_fraction` over its three patches), so a content fit's likelihood, AIC,
BIC and standard errors are stated **per event**, not per decoded timebin. Every
content fit carries the `estimator` (`conditional_poisson_fraction`) and
`response_col` it was produced with, and nested comparisons refuse to mix the
two. Alongside the model SE the fitters write an event-clustered sandwich SE
(`bse_cluster_<term>`), which is the SE the audited aggregation reads: on a
fraction response the model SE describes a design in which nothing varied within
an event. The timebin **count** response weights an event by its number of
decoded timebins and treats those autocorrelated bins as independent draws, so
its AIC scale is not comparable with a fraction bundle's.

**Dirichlet-multinomial alternative (`--estimator dirichlet_multinomial`).** A
second, non-primary content estimator that keeps the timebin **count** response
(it has no fraction form, so the flag forces `--response hard_count`) and adds
an explicit within-event concentration,
`log φ_e = log_phi_intercept + log_phi_is_swr · is_swr_e`. Overdispersion is then
a fitted parameter rather than a sandwich correction, so the standard errors are
event-level by construction and `coef_log_phi_is_swr` is a direct test of
whether SWR events spread their mass over more arms. It is fit on the **full**
likelihood including the multinomial coefficient
(`llf_includes_constants = True`), whereas the conditional fits drop it
(`False`): **a DM AIC is not comparable to a conditional fit's until
`multinomial_log_constant` — computed from the same count matrix — is added to
the conditional side.** Every DM row carries `phi_source` ∈ {`session`,
`animal_pooled`, `pooled_all_animals`}: a session with fewer modelled events
than `--min-events-session-phi` (default 40) is refit with its animal's median
concentration intercept pinned, and the column records which. `--n-jobs` sizes
the process pool that fits the per-session DM models (ignored by the conditional
Poisson). **`--concentration-length` (`is_swr`-contrast runner only)** adds
`log_n_nonlocal_bins_z` to the concentration design wherever `is_swr` is
identified, writing a separate **`_dm_len`** bundle: the likelihood otherwise
treats φ as **length-invariant** (`L_e` enters only as the multinomial index)
and SWR events run longer, so the plain `_dm` `log_phi_is_swr` is the **total**
concentration contrast and the `_dm_len` one is the length-adjusted twin,
reported next to `log_phi_log_n_nonlocal_bins_z` (per SD). The flag is refused
for the conditional Poisson, which has no concentration design. The estimator
comparison behind all of this is
[design/content_estimator_comparison.md](design/content_estimator_comparison.md).

### Fit diagnostics

Every per-session fit carries `converged` / `separated` status flags, the
`|`-joined `separated_terms` naming which coefficients are unidentified, one
`sd_<term>` per coefficient, and an `n_convergence_warnings` count
(`analysis.replay._fit_diagnostics`). Separation is judged **per term** — a term
is separated when its SE is non-finite or `> 1e3`, when `SE × the SD of its own
predictor` reaches `1.0` (the SE read on the effect scale, i.e. `SE >= 1 / SD`),
or — for a zero-SD column, which cannot be judged on that scale — when
`|coef| >= 10`. The structural nuisance terms `Intercept` and
`reward_history_incomplete` (`NUISANCE_TERMS`) are excluded from the judgment
entirely, since a blown-up baseline or structural-missingness coefficient says
nothing about whether the model's scientific terms are identified. `separated`
is simply "at least one non-nuisance term". The audited aggregation drops a
session for a term only when that term itself is named in `separated_terms`. A
product term (`is_swr:<covariate>`, `swr_x_<covariate>`) is judged on its **own**
coefficient and does not inherit a factor's separation: its `bse·SD` already
measures how well the interaction is pinned down, and a poorly identified
`is_swr` main effect is a statement about the two streams' baseline rates, not
about whether the covariate's slope differs between them. The
per-(model, population) `n_sessions_flagged_nonconverged` /
`n_sessions_flagged_separated` counts stay fit-level diagnostics. The remaining
coefficient exclusion is the `|coef| >= 10` backstop for fits that carry no flag.

### Cohort and timing rules

The covariate causal-hygiene rules (which covariates may be a forward predictor,
and which rungs are retained only as descriptive) and the cohort-audit table
schema are in
[design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md). The rung
membership lists those rules refer to are enumerated at the end of this file.

## Per-bin rate — Family B
`scripts/rate/fit_per_bin_rate_models.py` → `models_per_bin_rate/all_models.pkl`

A point-process conditional-intensity ladder (Truccolo–Eden–Brown 2005), one row
per (stop, 15 ms bin) with an equal-exposure offset, so the within-stop time
course is modelled explicitly rather than averaged into a per-stop dwell
correction. `tmax = 4 s` on the bout clock.

```rungs
M0_const              count ~ 1
M1_rew                count ~ REWBASIS
M5_time               count ~ BS_TC
M3r_histcos           count ~ HISTCOS
M6r_time_histcos      count ~ BS_TC + HISTCOS
M7r_time_histcos_rew  count ~ BS_TC + REWBASIS + HISTCOS
M9r_rew_x_time        count ~ BS_TC + REWBASIS + HISTCOS + reward_history_rcos_0:BS_TC

# Arrival-anchored sub-ladder (time_clock = "arrival"), fit on the [0, 8] s
# arrival window -- a strict subset of the frozen cohort's rows. AIC, deviance
# and held-out likelihood all scale with the number of rows, so these rungs are
# compared ONLY with each other; M0a_const_arrival is the window's own null.
M0a_const_arrival              count ~ 1
M5a_time_arrival               count ~ BS_ARR
M7ra_time_arrival_histcos_rew  count ~ BS_ARR + REWBASIS + HISTCOS
```

`M6r_time_histcos` is `M7r_time_histcos_rew` without the reward block, so the
reward step *given* time and self-history is the nested pair
`M7r_time_histcos_rew − M6r_time_histcos`; `M1_rew − M0_const` is the same
block's marginal step. The per-lag `onset_lag1..10` self-history rungs are
retired — the dummy block separates on the refractory bins, which voids the
rung's held-out metric; `HISTCOS` spans the same window and is scored. The
per-bin switch main-effect rung is retired too, because
`did_last_completed_trip_switch_patch` is often constant within a session's
complete-case cohort and makes the design rank-deficient. See
[design/model_history.md](design/model_history.md).

## MU uncertainty — per-bin Poisson (auxiliary)
`scripts/rate/fit_uncertainty_rate.py` → `models_per_bin_rate/uncertainty_rate.pkl`

One model, `MODEL_UNCERTAINTY_RATE` in `model_specs.py`, kept out of the shipped
`MODEL_PER_BIN_RATE` ladder so `all_models.pkl`'s schema is untouched.
`M7r_time_histcos_rew` (its `parent`) plus the curated value/uncertainty block
`MU_UNCERTAINTY_COVARIATES`:

```rungs
MU_uncertainty  count ~ BS_TC + REWBASIS + HISTCOS
                        + task_reward_probability_chosen_arm
                        + task_reward_probability_patch_mean_gap_best_vs_second
                        + patch_switch_fraction_last5_completed_trips
                        + trials_since_block_change
                        + rpe_postoutcome
                        + relative_value_switch_minus_stay_preoutcome
                        + reward_belief_differential_entropy_leaf_mean_preoutcome
                        + reward_belief_mean_update_l1_all_ports_postoutcome
                        + reward_belief_mean_update_l1_alternative_patches_postoutcome
```

## Cross-history coupling — per-bin Poisson (auxiliary)
`scripts/rate/fit_cross_history_rate.py` → `models_per_bin_rate/cross_history_models.pkl`
(only with `--write`)

`MODEL_CROSS_HISTORY_RATE` in `model_specs.py`. Fit per session in both
directions — the non-SWR non-local rate given the with-SWR stream's history, and
the reverse. Reward enters through the kernel only. Both rungs carry
`parent = None`; the runner compares the pair directly (self → self + cross AIC
drop and held-out bits per event).

```rungs
Mself       count ~ BS_TC + REWBASIS + HISTCOS
Mselfcross  count ~ BS_TC + REWBASIS + HISTCOS + CROSSCOS
```

## Cross-history coupling, duration-aware — per-bin Poisson (auxiliary)
`scripts/rate/fit_duration_aware_history_rate.py` →
`models_per_bin_rate/duration_aware_history_models.pkl` (per session, `--mode session`) and
`duration_aware_history_pooled.pkl` (each animal's sessions stacked with a session fixed effect,
`--mode pooled`), each with its `.provenance.json`.
Simulation behind it: `analysis/replay/duration_history_simulation.py`, review
`.claude/docs/reviews/2026-10-26-duration-aware-history-simulation.md`.

`MODEL_CROSS_HISTORY_DURATION_RATE` in `model_specs.py`. Events have duration
and one detector's events never overlap, so these rungs are fit on the at-risk
design: `aggregate_to_timebins(at_risk_events=<every event of the detector>,
offset_history=True)` drops the bins inside an ongoing event and sets each bin's
exposure to the time no event covers. History is measured from past events'
onsets, their ends, or both (`*_history_anchors`), so an end-locked dependence
(fragmentation) is not read as an onset-locked one. The fixed blocks are
`Mnohist_dur` → `Mself_dur` → `Mselfcross_dur`; the one-anchor rungs give the
onset-vs-offset contrasts. Self-history rungs are fit on all four populations
(detected SWR, at risk outside Kay SWRs; the non-local ones, at risk outside
non-local events); the cross rungs on the two non-local streams.

```rungs
Mnohist_dur            count ~ BS_TC + REWBASIS
Mself_onset_dur        count ~ BS_TC + REWBASIS + HISTCOS
Mself_offset_dur       count ~ BS_TC + REWBASIS + OFFCOS
Mself_dur              count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS
Mselfcross_onset_dur   count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS + CROSSCOS
Mselfcross_offset_dur  count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS + XOFFCOS
Mselfcross_dur         count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS + CROSSCOS + XOFFCOS
```

## Cross-history coupling, long window — ridge cross-validated (auxiliary)
`scripts/rate/fit_cross_history_ridge_cv.py` → `cross_history_ridge_cv.pkl` in its `--out-dir`, with its
`.provenance.json`. Module: `analysis/replay/ridge_history_cv.py`.

`MODEL_CROSS_HISTORY_LONG_RATE` in `model_specs.py`: `Mself_dur` and `Mselfcross_dur` with the cross
block (onset- and end-locked) reaching 1 s after the other stream's event (`N_CROSS_LONG_LAGS` = 67
lags, `N_CROSS_LONG_BASIS` = 6 log-spaced bumps), both frozen on the 1 s window's cohort. Each animal's
sessions are stacked with a session effect; the history coefficients carry a ridge prior whose SD is
chosen by inner cross-validation, and the pair is compared by outer held-out log-likelihood over five
folds of whole stops, not by AIC.

```rungs
Mself_long_dur         count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS + C(session)
Mselfcross_long_dur    count ~ BS_TC + REWBASIS + HISTCOS + OFFCOS + CROSSCOS_1S + XOFFCOS_1S + C(session)
```

## Per-stop rate — Family A
`scripts/rate/fit_per_stop_rate_models.py` → `models_per_stop_rate/`

Ordinary Poisson on the per-stop event count with a `log(immobility_duration_s)`
offset. It carries an intercept, so — unlike the conditional Poisson —
stop-level constants do not cancel.

```rungs
R0  n_selected_events ~ 1
R1  n_selected_events ~ REWBASIS
```

`R1` is the stop-level analog of the content hierarchy's reward-history axis:
`R1` vs `R0` tests whether replay *occurrence* tracks reward history at all, and
is the confound check for whether an `M3` content effect is partly a rate
effect. Candidate-arm visit recency has no stop-level analog (every trial visits
some arm), so the ladder stops at `R1`.

## Content — arm
`scripts/content/fit_replay_content_models.py --hierarchy arm` (per `is_swr` split {all, T, F})

The runner accepts only the current canonical table schema and validates every
requested rung before fitting. A stale/legacy table, or one missing any
requested predictor, aborts the complete request; models are never
warning-skipped into a partial ladder. Candidate identities, current-location
indicators and the response's per-event unit mass are checked as scientific
invariants. All rungs are then restricted to the same successfully fitted
session/event cohort and published as one staged directory bundle, so a late
failed rung cannot leave mixed-generation outputs.

```rungs
M0   arm_fraction ~ 1                      # null (uniform over arms) | strata(event_id)
M0a  arm_fraction ~ ARMFE
M1   arm_fraction ~ ARMFE + is_current_arm
M2   arm_fraction ~ ARMFE + is_current_arm + is_prev_arm
M3   arm_fraction ~ [M2 terms] + reward_fraction_candidate_arm_last10_visits
M4   arm_fraction ~ [M3 terms] + RECSPL
M5   arm_fraction ~ [M4 terms] + swr_x_is_current_arm + swr_x_is_prev_arm
M6   arm_fraction ~ [M5 terms] + swr_x_reward_fraction_candidate_arm_last10_visits
                  + swr_x_visit_recency_trials_event_leaf_centered_spline_{1,2,3}
Mv   arm_fraction ~ [M4 terms] + task_reward_probability_candidate_arm
                  + reward_belief_mean_candidate_arm_preoutcome
Mu   arm_fraction ~ [M4 terms] + reward_belief_variance_candidate_arm_preoutcome_z
Mvu  arm_fraction ~ [M4 terms] + reward_belief_mean_candidate_arm_preoutcome
                  + reward_belief_variance_candidate_arm_preoutcome_z
```

`M0`–`M6` are the default ladder (`models/`); under
`--estimator dirichlet_multinomial` the same eight default rungs publish to
`models_dm/`. `Mv` / `Mu` / `Mvu` are the belief-dependent rungs
(`BELIEF_DEPENDENT_CONTENT_MODELS`): their covariates are NaN on the unmapped-stop
tail, so co-fitting them would shrink the shared complete-case cohort for the
whole base ladder, and they are excluded from a default run either way. Request
them with their parents in one run, on their own matched cohort.

**Leaf-only belief chain.** The Beta-belief covariates are structural zeros on
the three trunk arms, so the belief rungs are fitted over the six leaves:

```
--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu
```

The whole chain must be requested in one run so the AIC comparisons stay nested
on one candidate set — a leaf-only `Mv` cannot be compared to a nine-arm `M4` by
ΔAIC, because changing the candidate set changes the conditional likelihood. A
leaf-only fit is a **six**-category conditional likelihood (null baseline
`log(6)`, not `log(9)`), the arm dummies for arms 5–7 are dropped (arm 5 becomes
the reference), and the fraction response is renormalised to unit mass over the
six leaves — without that, an event would be weighted by its leaf share, and SWR
events put more of their bins on the trunks. The bundle is
`models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/` (leaf-only ∧ partial ladder; see
"Bundle names and suffixes").

## Content — patch
`scripts/content/fit_replay_content_models.py --hierarchy patch` (per {all, T, F})

The nine arms collapsed to three patches (arm counts summed within patch). The
conditional Poisson conditions on the event total as before, but over three
candidate patches. Patch predictors carry a `_patch` suffix to distinguish them
from the arm-level versions.

```rungs
P0  patch_fraction ~ 1                     # null | strata(event_id)
P1  patch_fraction ~ is_current_patch
P2  patch_fraction ~ is_current_patch + reward_fraction_candidate_arm_last10_visits_patch
P3  patch_fraction ~ [P2 terms] + swr_x_is_current_patch
Pv  patch_fraction ~ [P2 terms] + task_reward_probability_candidate_arm_patch
                   + reward_belief_mean_candidate_arm_preoutcome_patch
```

`Pv` is belief-dependent and is excluded from a default run for the same reason
as `Mv` / `Mu` / `Mvu`; request it as `--models P2,Pv`.

## Post-switch patch
`MODEL_PATCH_SWITCH` — fit ONLY on events where
`did_last_completed_trip_switch_patch == 1`.

On switch trials the current and previous patch differ, so `is_current_patch`
and `is_prev_patch` are jointly identifiable; on stay trials they are collinear
and must not be combined. This is the dedicated test of the
previous-vs-current patch contrast.

```rungs
PS0  patch_fraction ~ 1                     # null (switch events only)
PS1  patch_fraction ~ is_current_patch
PS2  patch_fraction ~ is_current_patch + is_prev_patch
PS3  patch_fraction ~ [PS2 terms] + swr_x_is_current_patch + swr_x_is_prev_patch
```

## ContFrag content — arm
`scripts/tables/prepare_contfrag_replay_content_data.py` + the content runner,
`--event-source {SWR,MUA}`.

Same conditional-Poisson forms as the non-local arm hierarchy, on the ContFrag
tables. When events are externally restricted to SWR or MUA windows the
corresponding flag is constant by construction, so its interactions with
arm-varying predictors collapse onto the main effects and are unidentifiable —
the ladder therefore mirrors `M0`–`M4` and stops there.

```rungs
M0   arm_fraction ~ 1
M0a  arm_fraction ~ ARMFE
M1   arm_fraction ~ ARMFE + is_current_arm
M2   arm_fraction ~ ARMFE + is_current_arm + is_prev_arm
M3   arm_fraction ~ [M2 terms] + reward_fraction_candidate_arm_last10_visits
M4   arm_fraction ~ [M3 terms] + RECSPL
```

`MODEL_HIERARCHY_SWR` and `MODEL_HIERARCHY_MUA` are the same rung set. A
cross-modality rung (`mua_x_*` inside SWR events, `swr_x_*` inside MUA events)
is potentially identifiable, but is deferred until the cross-modality
variability is verified on the built ContFrag tables.

## ContFrag content — patch
Same restriction one level up: `P3`'s `swr_x_` interaction is unidentifiable
under event-source restriction, so the patch ladder stops at `P2`.

```rungs
P0  patch_fraction ~ 1
P1  patch_fraction ~ is_current_patch
P2  patch_fraction ~ is_current_patch + reward_fraction_candidate_arm_last10_visits_patch
```

`MODEL_PATCH_SWR` and `MODEL_PATCH_MUA` are the same rung set.

## Choice
`scripts/content/compare_event_source_models.py`
(`clogit(chosen ~ …, strata(stop))`)

Asks whether replay content predicts behaviour beyond a behavioural baseline.
These are not a single linear nested chain: `C_full` nests both `C_behavior`
(+ `replay_fraction`) and `C_replay_only` (+ the behavioural block). The
declared parent encodes the primary test, `C_full` vs `C_behavior`;
`C_replay_only` is a standalone replay-only reference.

```rungs
C_replay_only     chosen ~ replay_fraction
C_behavior        chosen ~ visit_recency_trials + is_candidate_arm_never_visited
                  + reward_fraction_candidate_arm_last10_visits
                  + reward_belief_mean_candidate_arm_preoutcome
                  + candidate_arm_distance_rank
C_full            chosen ~ [C_behavior terms] + replay_fraction
C_full_swr_split  chosen ~ [C_behavior terms] + replay_fraction_swr + replay_fraction_non_swr
```

`is_same_patch` and the raw `distance_to_candidate_arm_cm` are deliberately
excluded from the behavioural block: the two were near-collinear, and the raw
distance quasi-separated (the animal almost always chose the closest arm) and
crashed the fit in a substantial minority of sessions.
`candidate_arm_distance_rank` is the identifiable geometric baseline that
replaces them. Trade-off: the replay coefficient is no longer cleanly controlled
for distance.

## SWR power (auxiliary)
`scripts/coupling/fit_swr_power_model.py` — per-session `lm`, predictors z-scored per
session.

```r
ripple_power_z ~ scale(known_reward_outcome_lag_1)
                 + scale(task_reward_probability_chosen_arm)
                 + scale(rpe_postoutcome)
                 + scale(patch_switch_fraction_last5_completed_trips)
                 + scale(task_reward_probability_patch_mean_gap_best_vs_second)
                 + scale(reward_belief_mean_chosen_arm_preoutcome)
                 + scale(time_since_stop_start_s)
```

## The `is_swr`-contrast ladders

The analysis of the **non-local event rate and content** and their
SWR-vs-non-SWR dissociation. Specs: `SWR_CONTRAST_RATE_G_SPECS` /
`SWR_CONTRAST_RATE_S_SPECS` (rate) and `SWR_CONTRAST_CONTENT_ARM_LADDER` /
`SWR_CONTRAST_CONTENT_PATCH_LADDER` (content). Runners:
`scripts/rate/fit_swr_contrast_rate.py`,
`scripts/content/fit_swr_contrast_content.py` — they import the specs and keep
only data-assembly and fitting.

Covariate blocks for the rate ladders. The task dimensions are the paper's
variables (Comrie et al.), all from the **beta-Bernoulli** behavioural model:

```
BASE      := BS_TC + reward_history_rcos_0 + reward_history_rcos_1
             + reward_history_rcos_2 + HISTCOS
             # reward_history_incomplete is appended to every G/S formula as a
             # plain main effect, OUTSIDE any is_swr expansion
PERISWITCH:= periswitch_index_trials + is_periswitch_post
             + periswitch_index_trials:is_periswitch_post
BLOCK     := trials_since_block_change            # learning across a contingency block
SWITCHRATE:= patch_switch_fraction_last5_completed_trips
VALUE     := relative_value_switch_minus_stay_preoutcome
RPE       := rpe_postoutcome
UNCERT    := reward_belief_differential_entropy_leaf_mean_preoutcome
UPDATE    := reward_belief_mean_update_l1_all_ports_postoutcome
REWXTIME  := reward_history_rcos_0:BS_TC
```

Response `count` = non-local onset count per 15 ms bin; `family = poisson(log)`,
`offset = log(exposure)`; rows are (stop × bin) to `tmax = 4 s`. Fit per session,
aggregate coefficients across sessions (signed-rank).

Notes on the covariate choices. (1) **Single-model provenance:** every task
dimension comes from the beta-Bernoulli model; there is one behavioural model.
(2) **Context is split into three near-orthogonal rungs** (`PERISWITCH`,
`BLOCK`, `SWITCHRATE`), each tested individually rather than as one lumped
block. Peri-switch's three terms stay together — the interaction is meaningless
without its main effects, and jointly they are the broken-stick ramp around a
switch. (3) **Value and decision are one dimension:** a single relative-value
axis (ΔV, **Switch − Stay**), not separate value and decision covariates. The
sign convention matters: ΔV is approximately the negative of the current stay
value, so a positive coefficient means the current option's value is **low**.
Chosen/candidate-arm reward-belief mean is a robustness re-parameterization, not
co-fit on the rate ladder. (4) The reward kernel is the raised-cosine basis:
`reward_history_rcos_0` (the immediate component, weighted toward the newest
outcomes known at bout start) plus two progressively slower bumps, each a
weighted fraction in [0, 1]. Because feedback can occur during a visit, the
newest known outcome is not uniformly a fixed current- or previous-trial label.
`reward_history_incomplete` flags stops with fewer than two known outcomes
(features zero-filled) and is fit, never reported. (5)
`reward_belief_differential_entropy_leaf_mean_preoutcome` is the mean per-leaf
Beta differential entropy; candidate-arm reward-belief variance is the separate
content-uncertainty covariate. (6) **RPE is the unscaled canonical
`reward − Q_leaf`**, not the plotting-rescaled variant used elsewhere, which
reweights Q against reward and is therefore not an affine transform of it. (7)
`REWXTIME` tests whether recent reward changes the *shape* of the within-stop
rate profile, not just its level: the per-bin design and exposure offset already
remove the pure duration confound, but `BASE`'s additive
`reward_history_rcos_0 + BS_TC` still assumes a shared profile shape, and this
rung tests that assumption directly. (8) Every rung isolates one dimension —
joint kitchen-sink specs were dropped as overparameterized on low-event
sessions.

### Rate `is_swr` contrast — stratified (G)

Each formula is fit three times, to `NL_all`, `NL_swr` and `NL_noswr`.
Comparing a coefficient across the three populations is the descriptive
dissociation.

```rungs
G0_base
G1_periswitch
G2_block
G3_switchrate
G4_value
G5_rpe
G6_uncert
G7_update
G8_rewxtime
```

| model | formula (`count ~ … + offset(log exposure)`) | question |
|---|---|---|
| `G0_base` | `BASE` | Does the non-local rate have within-stop time structure, reward-history dependence, and self-history (refractoriness) — separately for all / SWR / non-SWR events? |
| `G1_periswitch` | `BASE + PERISWITCH` | Does the rate ramp around patch switches (asymmetric before/after)? |
| `G2_block` | `BASE + BLOCK` | Does the rate shift across a contingency block (learning)? |
| `G3_switchrate` | `BASE + SWITCHRATE` | Is the rate modulated by recent local switch volatility? |
| `G4_value` | `BASE + VALUE` | Is the rate modulated by relative value (ΔV = Switch − Stay)? |
| `G5_rpe` | `BASE + RPE` | Is the rate modulated by reward prediction error (reward − expected value)? |
| `G6_uncert` | `BASE + UNCERT` | Is the rate higher when belief entropy (outcome uncertainty) is high? |
| `G7_update` | `BASE + UPDATE` | Is the rate higher when global value updating is large? |
| `G8_rewxtime` | `BASE + REWXTIME` | Does the current reward reshape the within-stop rate *profile* (not just shift its level)? — the dwell-shape control. |

### Rate `is_swr` contrast — pooled (S)

Both non-local streams pooled, with `is_swr:X` coefficients as the
SWR-vs-non-SWR dissociation tests. Every `S` formula appends
`+ reward_history_incomplete` **outside** the `is_swr` expansion, so no
`is_swr:reward_history_incomplete` term exists — it is a structural-missingness
nuisance, and its contrast would be a contrast of missingness. `S0_rate` and
`S1_rew` are not full `is_swr *` expansions by design: they carry the `is_swr`
main effect, and `S1_rew` one named interaction, against an additive `BASE`.

```rungs
S0_rate
S1_rew
S2_base
S3_periswitch
S4_block
S5_switchrate
S6_value
S7_rpe
S8_uncert
S9_update
S10_rewxtime
```

| model | formula (`count ~ … + offset(log exposure)`) | question |
|---|---|---|
| `S0_rate` | `is_swr + BASE` | Do SWR-coincident non-local events occur at a different **overall rate** than non-SWR ones, holding the shared time/reward/history structure? |
| `S1_rew` | `BASE + is_swr + is_swr:reward_history_rcos_0` | Does the **immediate reward-history component** gate the rate differently for SWR vs non-SWR events? |
| `S2_base` | `is_swr * BASE` | Do the time-course, reward-history and self-history each modulate the rate differently by SWR status? |
| `S3_periswitch` | `is_swr * (BASE + PERISWITCH)` | Does the peri-switch ramp of rate differ by SWR? |
| `S4_block` | `is_swr * (BASE + BLOCK)` | Does the across-block (learning) modulation differ by SWR? |
| `S5_switchrate` | `is_swr * (BASE + SWITCHRATE)` | Does the switch-volatility modulation differ by SWR? |
| `S6_value` | `is_swr * (BASE + VALUE)` | Does relative-value modulation of rate differ by SWR? |
| `S7_rpe` | `is_swr * (BASE + RPE)` | Does RPE modulation of rate differ by SWR? |
| `S8_uncert` | `is_swr * (BASE + UNCERT)` | Does belief-entropy (uncertainty) modulation differ by SWR? |
| `S9_update` | `is_swr * (BASE + UPDATE)` | Does value-update modulation differ by SWR? |
| `S10_rewxtime` | `is_swr * (BASE + REWXTIME)` | Does the reward × within-stop-time-course (dwell-shape) effect differ by SWR? |

### Content `is_swr` contrast — arm (C / CS)

Conditional Poisson on `arm_fraction` over the nine candidate arms within an
event, one fit per session, event-clustered SEs.

```
_TSC     := log1p_time_since_stop_start_s_z:is_current_arm    # within-stop time course
_LENC    := log_n_nonlocal_bins_z:is_current_arm              # event-length control
_LENP    := log_n_nonlocal_bins_z:is_prev_arm
BASEc_NT := ARMFE + is_current_arm + is_prev_arm + RECSPL
BASEc    := BASEc_NT + _TSC + _LENC + _LENP
SWR_ARMFE    := swr_x_ARMFE                                   # SWR-specific arm baselines
BASEc_SWR    := BASEc + SWR_ARMFE                             # base of every pooled rung
BASEc_NT_SWR := BASEc_NT + SWR_ARMFE
```

**Why `BASEc` carries the two length products and `_TSC`.** SWR-coupled events
run longer than non-SWR ones, and a longer trajectory visits more arms, so an
uncontrolled `swr_x_is_current_arm` partly reads "longer events leave the
current arm" rather than "SWR content is less current-arm-locked". `_TSC`
removes the same channel for the within-stop time course, which also differs
between the two streams. Both enter `BASEc` **unconditionally**, so every rung
built on the base reports its SWR interaction *beyond* event length and *beyond*
the general time course. Because `n_nonlocal_bins` is measured after `is_swr`
and differs with it, that is a **post-treatment adjustment**: a `BASEc` rung's
`swr_x_…` is the length-adjusted (DIRECT) SWR effect, and length may be a
mediator (SWR → longer replay → the trajectory leaves the current arm). The `nt`
twins on `BASEc_NT` report the same term as the TOTAL SWR effect. Both are
shipped; report the pair and say which estimand a number is.

`candidate_arm_distance_rank` is deliberately absent from this ladder: it takes
five tied values over a fixed track topology, its rank-1 level is identical to
`is_current_arm`, and carrying it collapsed the current-arm coefficient and made
every `C*`/`CS*` `is_current_arm` non-comparable with `M1` / `C1_current`.

**Split vs pooled.** A `split` rung is fit three times (`all` / SWR-only /
non-SWR-only) and the dissociation is the comparison across those populations. A
`pooled` rung fits both streams together, so its `swr_x_…` coefficient is the
formal SWR-vs-non-SWR difference.

**Every pooled rung carries SWR-specific arm baselines** (`SWR_ARMFE`, from
2026-10-30): SWR and non-SWR events each get their own arm fixed effects, so a
rung's `swr_x_…` compares the two kinds of event at the same arm. With one
shared set, SWR's shift of content onto the trunk arms (about twice the trunk
share, 5/5 animals) loaded on the only SWR-specific term: `CS1nt`'s
`swr_x_is_current_arm` read −0.41 (0/5 animals positive), "SWR content is off
the current arm" when it was "SWR content is on the trunks". On a leaf-only rung
the baselines are the five leaf dummies' `swr_x_` twins. The pooled rungs below
are on `BASEc_SWR` (`BASEc_NT_SWR` for `nt`); the "added to" column lists what
is added on top of that.

```rungs
C0_armfe
C1_current
C2_base
C3_armreward
C4_armvalue
C5_lastrew
C6_rpe
C7_value
C8_uncert
C9_update
C10_periswitch
C11_block
C12_switchrate
C13_armuncert
CS1_current
CS1nt_current
CS2_prev
CS3_lastrew
CS4_armreward
CS4nt_armreward
CS5_armvalue
CS5nt_armvalue
CS5b_valuevsprevgoal
CS6_armuncert
CS6nt_armuncert
```

| model | added to `BASEc` | scope | key term |
|---|---|---|---|
| `C0_armfe` | (`ARMFE` only) | split | — |
| `C1_current` | (`ARMFE` +) `is_current_arm` | split | `is_current_arm` |
| `C2_base` | — | split | `is_current_arm` |
| `C3_armreward` | `reward_fraction_candidate_arm_last10_visits` | split | that term |
| `C4_armvalue` | `reward_belief_mean_candidate_arm_preoutcome` | split | that term |
| `C5_lastrew` | `is_last_rewarded_noncurrent_arm` | split | that term |
| `C6_rpe` | `rpe_postoutcome_z:is_current_arm` | split | that term |
| `C7_value` | `relative_value_switch_minus_stay_preoutcome_z:is_current_arm` | split | that term |
| `C8_uncert` | `reward_belief_differential_entropy_leaf_mean_preoutcome_z:is_current_arm` | split | that term |
| `C9_update` | `reward_belief_mean_update_l1_all_ports_postoutcome_z:is_current_arm` | split | that term |
| `C10_periswitch` | `PERISWITCH` terms × `is_current_arm` | split | `periswitch_index_trials_z:is_current_arm` |
| `C11_block` | `trials_since_block_change_z:is_current_arm` | split | that term |
| `C12_switchrate` | `patch_switch_fraction_last5_completed_trips_z:is_current_arm` | split | that term |
| `C13_armuncert` | `reward_belief_variance_candidate_arm_preoutcome_z` | split | that term |
| `CS1_current` | `swr_x_is_current_arm` | pooled | `swr_x_is_current_arm` |
| `CS1nt_current` | on `BASEc_NT`: `swr_x_is_current_arm` | pooled | `swr_x_is_current_arm` |
| `CS2_prev` | `swr_x_is_prev_arm` | pooled | that term |
| `CS3_lastrew` | `is_last_rewarded_noncurrent_arm` + its `swr_x_` | pooled | `swr_x_is_last_rewarded_noncurrent_arm` |
| `CS4_armreward` | `reward_fraction_candidate_arm_last10_visits` + its `swr_x_` | pooled | `swr_x_reward_fraction_candidate_arm_last10_visits` |
| `CS4nt_armreward` | on `BASEc_NT`: `CS4` terms | pooled | `swr_x_reward_fraction_candidate_arm_last10_visits` |
| `CS5_armvalue` | `reward_belief_mean_candidate_arm_preoutcome` + its `swr_x_` | pooled | `swr_x_reward_belief_mean_candidate_arm_preoutcome` |
| `CS5nt_armvalue` | on `BASEc_NT`: `CS5` terms | pooled | `swr_x_reward_belief_mean_candidate_arm_preoutcome` |
| `CS5b_valuevsprevgoal` | `CS5` + `is_last_rewarded_noncurrent_arm` + its `swr_x_` (the previous-goal control) | pooled | `swr_x_reward_belief_mean_candidate_arm_preoutcome` |
| `CS6_armuncert` | `reward_belief_variance_candidate_arm_preoutcome_z` + its `swr_x_` | pooled | `swr_x_reward_belief_variance_candidate_arm_preoutcome_z` |
| `CS6nt_armuncert` | on `BASEc_NT`: `CS6` terms | pooled | `swr_x_reward_belief_variance_candidate_arm_preoutcome_z` |

**Outcome strata.** The whole ladder is refit within `--outcome-stratum
rewarded` and `omitted` (an event whose current outcome is not yet known is in
neither), writing `_rewarded` / `_omitted` artifacts. The per-SD units and the
fit-time `_z` reference are frozen over the pooled cohort first, so a stratified
refit states its effects in the same unit as the pooled one. Rungs whose
`n_sessions_modeled` is 0 in a stratum are named in the cohort audit.

**Dirichlet-multinomial bundles.** Under the DM the audited table gains a
`log_phi_is_swr` row per rung × population, aggregated as its own BH-FDR family
rather than folded into the β contrast: it is not any rung's key term but the
same quantity on every rung (how tightly an SWR event's mass is concentrated),
and the conditional-Poisson fits produce no such row. `--concentration-length`
writes the `_dm_len` bundle, whose audited table carries **both**
`log_phi_is_swr` — now the length-adjusted estimand — and
`log_phi_log_n_nonlocal_bins_z` per rung × population. Each artifact's
provenance sidecar records the `concentration_cols` it was fit with, so the two
estimands are told apart from the record rather than the filename.

### Content `is_swr` contrast — patch (CP / CPS)

The same ladder one level up: conditional Poisson on `patch_fraction` over the
three patches, with the patch analogues of the arm within-stop-time and
event-length controls.

```
BASEc_PATCH_NT := is_current_patch + is_prev_patch + visit_recency_trials_patch
                  + is_patch_never_visited
BASEc_PATCH    := BASEc_PATCH_NT + log1p_time_since_stop_start_s_z:is_current_patch
                  + log_n_nonlocal_bins_z:is_current_patch
                  + log_n_nonlocal_bins_z:is_prev_patch
```

```rungs
CP0_base
CP1_lastrew
CP3_newgrowth
CP4_switchrate
CPS1_current
CPS1nt_current
CPS2_lastrew
```

| model | added to `BASEc_PATCH` | scope | key term |
|---|---|---|---|
| `CP0_base` | — | split | `is_current_patch` |
| `CP1_lastrew` | `is_last_rewarded_noncurrent_patch` | split | that term |
| `CP3_newgrowth` | `periswitch_index_trials_z:is_current_patch` | split | that term |
| `CP4_switchrate` | `patch_switch_fraction_last5_completed_trips_z:is_current_patch` | split | that term |
| `CPS1_current` | `swr_x_is_current_patch` | pooled | `swr_x_is_current_patch` |
| `CPS1nt_current` | on `BASEc_PATCH_NT`: `swr_x_is_current_patch` | pooled | `swr_x_is_current_patch` |
| `CPS2_lastrew` | `is_last_rewarded_noncurrent_patch` + its `swr_x_` | pooled | `swr_x_is_last_rewarded_noncurrent_patch` |

The two rungs that crossed `is_last_rewarded_noncurrent_patch` with a
peri-switch term are retired: the indicator is almost perfectly anti-correlated
with its own interaction, so the design was rank-deficient in a large fraction
of sessions, and a low-support indicator crossed with a time term is not
estimable per session in a three-alternative model.

## Rung subsets

Membership lists referenced by the cohort and timing rules in
[design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md), and by
the leaf-only candidate-set rule above. All three are frozensets in
`model_specs.py`.

### Leaf-only contrast rungs

`LEAF_ONLY_CONTRAST_RUNGS` — standalone contrast rungs conditioned on the six
leaf arms rather than all nine. Their belief covariates are structural zeros on
the trunks, `is_candidate_arm_6` / `is_candidate_arm_7` become identically zero,
and the fraction response is renormalised to unit mass over the leaves so an SWR
event is not downweighted for putting more of its bins on a trunk. The per-SD
units the audited table reports these coefficients in are likewise taken over
the leaf arms only. Restricted to standalone rungs because they are fitted one
at a time and so can switch candidate sets individually; the nested arm-ladder
belief rungs use the `--leaf-only` chain instead.

```rungs
C4_armvalue
C13_armuncert
CS5_armvalue
CS5nt_armvalue
CS5b_valuevsprevgoal
CS6_armuncert
CS6nt_armuncert
```

### Descriptive rungs

`DESCRIPTIVE_CONTENT_RUNGS` and `DESCRIPTIVE_RATE_RUNGS` — rungs carrying a
post-outcome covariate (RPE, value update) or a future-referencing one
(peri-switch), retained under an explicit *association* / *alignment* claim
rather than a forward-prediction one. The runners declare exactly these names
descriptive; a new rung with a leaky covariate that is not listed fails the
timing assertion.

```rungs
C6_rpe
C9_update
C10_periswitch
CP3_newgrowth
G1_periswitch
G5_rpe
G7_update
S3_periswitch
S7_rpe
S9_update
```

### Post-outcome cohort rungs

`POST_OUTCOME_RATE_RUNGS` — the `is_swr`-contrast rate rungs carrying a
`POST_OUTCOME_COVARIATES` column (`rpe_postoutcome`,
`reward_belief_mean_update_l1_all_ports_postoutcome`). These are undefined on a
stop that begins before its trial's outcome is known, so they are **not**
members of the nested ladder frozen on the primary cohort: they are fit on the
post-outcome cohort together with a refit of their ladder's null (`G0_base` /
`S0_rate`) so their ΔAIC is computed on identical rows. Every fitted row carries
a `cohort` label (`primary` / `post_outcome`).

```rungs
G5_rpe
G7_update
S7_rpe
S9_update
```

### Cohort-audit tables

Moved to
[design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md#cohort-audit-tables):
the `<run>_cohort_audit.csv` schema, its mutually-exclusive drop buckets, the
runners that emit it, and the `n_in − Σ n_drop_* == n_modeled` reconciliation
invariant.
