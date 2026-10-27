# Replay rate × content × SWR-coupling: integrative synthesis

**Status:** living results note. Confirmed findings, open hypotheses, and unknowns
for how the **rate**, **content**, and **SWR-coupling** of non-local replay events
relate. Companion to the per-layer references: rate ladder
[per_bin_rate_results.md](per_bin_rate_results.md) / [rate_models.md](rate_models.md);
content claims registry [../paper/claims_table.md](../paper/claims_table.md).

> **Refit status (2026-09-12).** Both quantitative layers now share one
> population. The rate families and the `is_swr`-contrast content ladder were
> refit on the **well-anchored stop cohort** (`occupancy_state ∈ {at_well,
> mixed}`, <!--v:part3.stops_well_anchored-->93,560<!--/v--> of <!--v:part3.stops_total-->120,320<!--/v--> stops), which is the population the content family
> was already fit on at the event grain, so a rate claim and a content claim
> about non-SWR replay are now claims about the same events
> ([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md);
> [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md),
> [2026-09-12-content-cohort-comparison.md](../.claude/docs/reviews/2026-09-12-content-cohort-comparison.md)).
> The rate numbers moved the most: the non-SWR stream's immediate reward
> component rises once the in-transit stops — which carried a disproportionate
> share of non-SWR onsets — leave the cohort, so the reward dissociation is
> weaker and is now a difference between two responding streams rather than
> response-vs-none. The content coefficients moved at the third decimal, and
> their per-SD units are now defined on the at-well fit cohort. The all-stops
> rate fits survive as a sensitivity bundle; an in-transit bundle exists but is
> thin (<!--v:part3.rate_intransit_sessions-->52<!--/v--> of <!--v:part3.rate_sessions-->242<!--/v--> sessions) and is not evidence of absence.
>
> Before that, both quantitative layers were refit in the 2026-09
> fix wave, and the numbers below have been restated from those refits wherever
> the audits report them. The rate ladders were refit on **2026-09-06** with a
> smooth past-only reward-history basis, boundary-recomputed reward features,
> per-coefficient separation judging and shared-row rung scoring
> ([2026-09-06-rate-refit-arrival-clock.md](../.claude/docs/reviews/2026-09-06-rate-refit-arrival-clock.md));
> the content ladders were refit on **2026-09-05** onto the per-event allocation
> fraction with event-clustered standard errors
> ([2026-09-05-content-refit-comparison.md](../.claude/docs/reviews/2026-09-05-content-refit-comparison.md)).
> Any quantity this document still carries that was **not** re-derived from those
> refits is marked **PENDING** and must not be cited; where a claim rested on a
> rung that no longer exists, the claim is marked retired in place rather than
> deleted.

## Provenance & methods (read first)

- **Data:** event tables (`event_table.parquet`, `event_arm_table.parquet`,
  `stop_table.parquet`) on the canonical <!--v:part3.rate_sessions-->242<!--/v-->-session, 5-animal corpus. Every
  number below is read from `paper/reported/part3.json`.
- **Inference convention (all layers):** fit **per session**, then test the
  per-session coefficients **across sessions** (the session is the exchangeable
  unit), with **per-animal sign consistency (n/5)**. Content aggregates are the
  **animal-balanced median** (the median over animals of that animal's
  per-session median), animals-positive-of-5, and a session sign-rank **BH-FDR
  q**; an animal bootstrap (2000 draws, seed 0) supplies intervals where the
  source reports one. This is robust to within-session clustering; it is *not*
  an event-level test.
- **Content response and standard errors.** The shipped content response is the
  **per-event allocation fraction** — one event, one unit of mass — fit with
  **event-clustered** standard errors. The retired per-timebin count response
  treated a median ~<!--v:part3.event_bins_median-->32<!--/v--> autocorrelated bins per event as independent draws, so
  its standard errors and its likelihood-scale quantities (AIC, ΔAIC) are not
  interpretable at the event scale: a count-response number and a
  fraction-response number must never be differenced. The per-SD unit for the
  arm belief terms also changed with the leaf-only fitting, so standardized
  coefficients are not comparable across the change either. **Per-SD units are
  now defined on the at-well fit cohort (2026-09-12); pre-refit values differ at
  the third decimal.**
- **Content estimand — direct vs total, and why both are shipped.** `BASEc`
  rungs carry the event-length products (`log_n_nonlocal_bins_z ×
  is_current_arm` / `× is_prev_arm`). Event length is measured **after**
  `is_swr` and differs with it, so it is **post-treatment**: a `BASEc`
  `swr_x_...` coefficient is the length-adjusted **direct** effect, and length
  is plausibly a mediator (SWR to longer replay to a trajectory that leaves the
  current arm). The `nt` twins on `BASEc_NT` omit those products and report the
  **total** effect. Both are shipped, neither supersedes the other, and every
  number below says which it is.
- **Content estimator comparison.** One cohort, three likelihoods — retired
  count, shipped fraction, Dirichlet-multinomial (DM) — are compared term for
  term in
  [design/content_estimator_comparison.md](design/content_estimator_comparison.md).
  **Exactly one conclusion is estimator-dependent:** the *direct* SWR by
  arm-value interaction (`CS5_armvalue`
  `swr_x_reward_belief_mean_candidate_arm_preoutcome`), where count <!--v:part3.est_swr_armvalue_count-->−0.17<!--/v--> (<!--v:part3.est_swr_armvalue_count_animals_positive-->2<!--/v-->/5
  animals) and fraction <!--v:part3.est_swr_armvalue_fraction-->−0.014<!--/v--> (<!--v:part3.est_swr_armvalue_fraction_animals_positive-->2<!--/v-->/5) are negative while the DM
  is <!--v:part3.est_swr_armvalue_dm-->+0.046<!--/v--> (<!--v:part3.est_swr_armvalue_dm_animals_positive-->3<!--/v-->/5), all near zero, on the <!--v:part3.est_swr_armvalue_fraction_sessions-->238<!--/v-->-session cohort. Every other key term
  agrees in sign across all three, and the magnitudes shrink monotonically from
  count to fraction to DM on the SWR interactions
  (`CS1` <!--v:part3.est_swr_current_count-->−0.093<!--/v--> → <!--v:part3.est_swr_current_fraction-->−0.071<!--/v--> → <!--v:part3.est_swr_current_dm-->−0.036<!--/v-->; `CS4` <!--v:part3.est_swr_armreward_count-->−1.3<!--/v--> → <!--v:part3.est_swr_armreward_fraction-->−1.0<!--/v--> → <!--v:part3.est_swr_armreward_dm-->−0.78<!--/v-->) while
  `is_current_arm` does not shrink (<!--v:part3.est_current_count-->+0.67<!--/v--> → <!--v:part3.est_current_fraction-->+0.73<!--/v--> → <!--v:part3.est_current_dm-->+0.73<!--/v-->). Those three
  columns are one fitting run on the same tables as the shipped fits, and its
  fraction column matches the shipped value (<!--v:part3.base_current_all-->+0.73<!--/v-->); the comparison is between likelihoods and is read on its own rows. The DM also
  carries a parameter the conditional estimators do not — `log_phi_is_swr`, the
  within-event concentration contrast — which is the direct measurement of "SWR
  broadens content" (Layer 3). The fraction's event-clustered SE runs above the
  DM's model SE on every term (the comparison summary's `median_se_ratio_*`
  columns), so the shipped estimator is if
  anything the more conservative of the two honest options, and the fraction
  stays the shipped response.
- **Everything here is correlational.** Covariate-to-replay associations, not
  established causation. Two confounds are explicitly controlled where noted:
  **occupancy** (via the within-well outcome contrast) and **within-stop time /
  dwell** (via a `time_since_stop_start_s` control); **collinearity** was
  audited.
- **One process, three layers.** A non-local event is a *point* (occurrence)
  carrying *marks*: its **SWR-coupling** (ripple power) and its **content**
  (represented arm). "Rate" = occurrence intensity; "coupling" and "content" =
  marks. The layers are separable factors of one marked-point-process likelihood
  (see *Open* — we have not fit the full joint model).

## Layer 1 — Rate (occurrence)

Per-bin conditional-intensity ladder (Family B, `MODEL_PER_BIN_RATE`; 15 ms
bins, <!--v:part3.rate_sessions-->242<!--/v--> sessions, the well-anchored stop cohort — <!--v:part3.rate_stops-->92,942<!--/v--> stops over <!--v:part3.rate_bins-->7,535,050<!--/v-->
bins). Baseline rates (`M0_const`, events per second of
immobility, median across sessions): non-local (all) **<!--v:part3.rate_baseline_non_local-->0.43<!--/v-->**, detected SWR
**<!--v:part3.rate_baseline_swr-->0.27<!--/v-->**, non-local no-SWR **<!--v:part3.rate_baseline_non_local_no_swr-->0.22<!--/v-->**, non-local w/ SWR **<!--v:part3.rate_baseline_non_local_swr-->0.21<!--/v-->**. The
non-local split is roughly additive, and the two non-local streams are the
rarest types. The dominant structure in every population is the within-stop
**time course** (rise over the first ~1.5–2 s then plateau; median peak <!--v:part3.rate_time_peak_s_swr-->2.62<!--/v--> s
detected SWR / <!--v:part3.rate_time_peak_s_non_local-->2.96<!--/v--> s non-local / <!--v:part3.rate_time_peak_s_non_local_swr-->2.48<!--/v--> s w/ SWR / <!--v:part3.rate_time_peak_s_non_local_no_swr-->3.00<!--/v--> s no-SWR, read only on the
supported interval [1, 3] s, where a peak at 3.00 s is the interval's edge rather
than a located maximum), **self-history** clustering, and a smooth
**reward-history** kernel. The time spline is what calibrates the process
(KS-in-band <!--v:part3.rate_ladder_ks_band_const-->0.45<!--/v--> → <!--v:part3.rate_ladder_ks_band_time-->0.93<!--/v--> at `M5_time`, non-local). `M5_time` is the best-generalizing rung
(<!--v:part3.rate_ladder_cv_bits_time-->0.24<!--/v--> held-out bits/event over <!--v:part3.rate_ladder_cv_sessions_time-->226<!--/v--> of <!--v:part3.rate_sessions-->242<!--/v--> sessions);
`M7r_time_histcos_rew` is the rung to read coefficients off (median AIC <!--v:part3.rate_ladder_aic_full-->2128<!--/v-->,
KS-in-band <!--v:part3.rate_ladder_ks_band_full-->0.96<!--/v-->), but its CV is scored on a selected <!--v:part3.rate_ladder_cv_sessions_full-->61<!--/v--> sessions and is **not** a
like-for-like win over `M5_time`.

**Reward history is one smooth three-feature basis, not per-lag dummies.** The
five lag-dummy rungs (`M3_hist`, `M4_hist_rew`, `M6_time_hist`,
`M7_time_hist_rew`, `M9_rew_x_time`) were retired on 2026-09-06 —
`cv_bits_per_event` was NaN in **99.08 %** of the non-local fits — so every
number this document previously carried from them is withdrawn. Read the reward
effect off the **per-session reconstructed kernel**
(`paper/figures/rate/reward_kernel_basis.png`), never off one basis
coefficient: the three are strongly collinear.

| population | kernel lag 1 | kernel lags 3–10 | immediate coefficient RR (`M7r`) |
| --- | --- | --- | --- |
| detected SWR | <!--v:part3.rate_kernel_lag_one_swr-->1.38<!--/v--> | <!--v:part3.rate_kernel_late_min_swr-->0.90<!--/v-->–<!--v:part3.rate_kernel_late_max_swr-->1.04<!--/v--> | <!--v:part3.rate_immediate_session_swr-->6.4<!--/v--> (<!--v:part3.rate_immediate_session_positive_fraction_swr-->83 %<!--/v--> of sessions, <!--v:part3.rate_immediate_session_animals_swr-->5<!--/v-->/5, p <!--v:part3.rate_immediate_session_p_swr-->7.9e-28<!--/v-->) |
| non-local (all) | <!--v:part3.rate_kernel_lag_one_non_local-->1.19<!--/v--> | <!--v:part3.rate_kernel_late_min_non_local-->0.91<!--/v-->–<!--v:part3.rate_kernel_late_max_non_local-->1.04<!--/v--> | <!--v:part3.rate_immediate_session_non_local-->3.3<!--/v--> (<!--v:part3.rate_immediate_session_positive_fraction_non_local-->79 %<!--/v-->, <!--v:part3.rate_immediate_session_animals_non_local-->5<!--/v-->/5, p <!--v:part3.rate_immediate_session_p_non_local-->1.3e-22<!--/v-->) |
| non-local, w/ SWR | **<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v-->** | <!--v:part3.rate_kernel_late_min_non_local_swr-->0.89<!--/v-->–<!--v:part3.rate_kernel_late_max_non_local_swr-->1.07<!--/v--> | **<!--v:part3.rate_immediate_session_non_local_swr-->9.9<!--/v-->** (<!--v:part3.rate_immediate_session_positive_fraction_non_local_swr-->87 %<!--/v-->, <!--v:part3.rate_immediate_session_animals_non_local_swr-->5<!--/v-->/5, p <!--v:part3.rate_immediate_session_p_non_local_swr-->8.0e-33<!--/v-->) |
| non-local, no SWR | **<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->** | <!--v:part3.rate_kernel_late_min_non_local_no_swr-->0.93<!--/v-->–<!--v:part3.rate_kernel_late_max_non_local_no_swr-->1.01<!--/v--> | **<!--v:part3.rate_immediate_session_non_local_no_swr-->1.4<!--/v-->** (<!--v:part3.rate_immediate_session_positive_fraction_non_local_no_swr-->58 %<!--/v-->, <!--v:part3.rate_immediate_session_animals_non_local_no_swr-->5<!--/v-->/5, p <!--v:part3.rate_immediate_session_p_non_local_no_swr-->0.0026<!--/v-->) |

The kernel lag-1 column is `rr_lag1` — the reconstructed kernel evaluated at lag
1 — and the last column is the across-session median of `exp` of the single
bump-0 coefficient. They are different quantities; the audited aggregate
summarizes the latter animal-balanced instead, where it reads <!--v:part3.rate_immediate_animal_swr-->5.4<!--/v--> / <!--v:part3.rate_immediate_animal_non_local-->2.9<!--/v--> / <!--v:part3.rate_immediate_animal_non_local_swr-->8.8<!--/v--> / <!--v:part3.rate_immediate_animal_non_local_no_swr-->1.2<!--/v--> in the same row order. Never substitute one for another.

**The dissociation lives in the immediate component, and it is paired within
session:** the within-session comparison of `coef_reward_history_rcos_0` between
the two non-local streams is positive in **<!--v:part3.rate_paired_positive_fraction-->77 %<!--/v--> of <!--v:part3.rate_paired_sessions-->240<!--/v--> paired sessions, <!--v:part3.rate_paired_animals_positive-->5<!--/v-->/5 animals, signed-rank p = <!--v:part3.rate_paired_p-->1.1e-19<!--/v-->**. After full control the non-SWR immediate
component is still a response of its own (RR <!--v:part3.rate_immediate_session_non_local_no_swr-->1.4<!--/v-->, p = <!--v:part3.rate_immediate_session_p_non_local_no_swr-->0.0026<!--/v-->, <!--v:part3.rate_immediate_session_animals_non_local_no_swr-->5<!--/v-->/5 animals) —
about <!--v:part3.rate_immediate_ratio_swr_to_no_swr-->7.0<!--/v-->× smaller than the SWR-coupled one, not absent, though its lag-1
kernel (<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->) is close to 1. On all stops it read as
absent (RR <!--v:part3.rate_allstops_immediate_session_no_swr-->0.83<!--/v-->, p = <!--v:part3.rate_allstops_immediate_session_p_no_swr-->0.74<!--/v-->, <!--v:part3.rate_allstops_immediate_session_animals_no_swr-->3<!--/v-->/5, lag 1 at <!--v:part3.rate_allstops_kernel_lag_one_no_swr-->0.95<!--/v-->, paired in <!--v:part3.rate_allstops_paired_positive_fraction-->83 %<!--/v-->, p = <!--v:part3.rate_allstops_paired_p-->4.6e-26<!--/v-->);
**that framing belonged to a population including in-transit stops and must not
be carried forward.** The dissociation survives on the shared cohort, weaker. The slower part of the kernel — a mild suppression from lag 3
on, the lowest per-population RR over lags 3–10 being <!--v:part3.rate_kernel_late_min_low-->0.89<!--/v-->–<!--v:part3.rate_kernel_late_min_high-->0.93<!--/v-->, back to ~1 by lag 10 — is **shared** by all four
populations. Whether that slower part is an RPE / relative-reward signal or
satiety / adaptation is **not** established here: the GLM uses past binary
outcomes, not a prediction error.

**The reward block is a small AIC step marginally and a negative one
conditionally** (negative ΔAIC = child better). `M1_rew − M0_const` is **<!--v:part3.rate_marginal_reward_delta_aic_swr-->−6.8<!--/v-->**
(detected SWR) / **<!--v:part3.rate_marginal_reward_delta_aic_non_local-->−7.8<!--/v-->** (non-local) / **<!--v:part3.rate_marginal_reward_delta_aic_non_local_swr-->−5.4<!--/v-->** (w/ SWR) / **<!--v:part3.rate_marginal_reward_delta_aic_non_local_no_swr-->+0.089<!--/v-->** (no
SWR); `M7r_time_histcos_rew − M6r_time_histcos`, the same block given time
**and** self-history, is **<!--v:part3.rate_conditional_reward_delta_aic_swr-->+0.94<!--/v-->** / **<!--v:part3.rate_conditional_reward_delta_aic_non_local-->+1.0<!--/v-->** / **<!--v:part3.rate_conditional_reward_delta_aic_non_local_swr-->+1.6<!--/v-->** / **<!--v:part3.rate_conditional_reward_delta_aic_non_local_no_swr-->+2.4<!--/v-->** — every
population moves the wrong way. Family A (per-stop Poisson with a
`log(immobility)` offset, the literature-comparable layer) agrees in *direction*
only. On the **well-anchored** cohort its median per-session `R1 − R0` ΔAIC is
**<!--v:part3.per_stop_delta_aic_animal_swr-->−2.7<!--/v-->** (detected SWR) / **<!--v:part3.per_stop_delta_aic_animal_non_local-->−4.3<!--/v-->** (non-local) / **<!--v:part3.per_stop_delta_aic_animal_non_local_swr-->−2.6<!--/v-->** (w/ SWR) /
**<!--v:part3.per_stop_delta_aic_animal_non_local_no_swr-->+0.88<!--/v-->** (no SWR) animal-balanced, and **<!--v:part3.per_stop_delta_aic_session_swr-->−4.3<!--/v-->** / **<!--v:part3.per_stop_delta_aic_session_non_local-->−4.2<!--/v-->** / **<!--v:part3.per_stop_delta_aic_session_non_local_swr-->−3.1<!--/v-->** /
**<!--v:part3.per_stop_delta_aic_session_non_local_no_swr-->+0.88<!--/v-->** session-pooled, over <!--v:part3.per_stop_sessions_non_local-->241<!--/v--> sessions each — the block earns its
parameters in three populations but not in the non-SWR one (<!--v:part3.per_stop_child_better_fraction_non_local_no_swr-->46 %<!--/v--> of
sessions child-better, <!--v:part3.per_stop_animals_child_better_non_local_no_swr-->2<!--/v-->/5 animals, against <!--v:part3.per_stop_child_better_fraction_non_local-->74 %<!--/v--> and <!--v:part3.per_stop_animals_child_better_non_local-->5<!--/v-->/5 on the pooled
non-local stream).
**Retired with the outcome-flag
rung:** the per-stop omission RR 0.36 and recent-reward RR 0.62 headline came
from a rung built at the trial *start*, which could encode the not-yet-delivered
outcome on approach stops; it is no longer fit and neither its magnitude nor its
session share survives.

**Events are strongly non-Poisson.** The controlled self-history kernel is
refractory at short lags then bursty: peak RR **<!--v:part3.rate_burst_peak_swr-->2.67<!--/v-->** (detected SWR, <!--v:part3.rate_burst_peak_lag_ms_swr-->105<!--/v--> ms),
**<!--v:part3.rate_burst_peak_non_local-->1.95<!--/v-->** (non-local, <!--v:part3.rate_burst_peak_lag_ms_non_local-->120<!--/v--> ms), **<!--v:part3.rate_burst_peak_non_local_swr-->1.98<!--/v-->** (w/ SWR, <!--v:part3.rate_burst_peak_lag_ms_non_local_swr-->120<!--/v--> ms, dead time ≤<!--v:part3.rate_refractory_ms_non_local_swr-->60<!--/v--> ms),
**<!--v:part3.rate_burst_peak_non_local_no_swr-->1.56<!--/v-->** (no SWR, <!--v:part3.rate_burst_peak_lag_ms_non_local_no_swr-->105<!--/v--> ms). Detected SWRs are the most tightly
bursty stream and non-SWR events the flattest.

**Reward shifts the rate LEVEL, not the within-stop profile SHAPE.**
`M9r_rew_x_time` costs held-out bits and buys no AIC against its parent
`M7r_time_histcos_rew` (<!--v:part3.rate_ladder_aic_reward_by_time-->2133<!--/v--> vs <!--v:part3.rate_ladder_aic_full-->2128<!--/v-->; <!--v:part3.rate_ladder_cv_bits_reward_by_time-->0.18<!--/v--> vs <!--v:part3.rate_ladder_cv_bits_full-->0.22<!--/v--> bits/event, on a smaller CV
cohort still: <!--v:part3.rate_ladder_cv_sessions_reward_by_time-->40<!--/v--> sessions).

**Arrival clock.** Sister rungs (`M0a_const_arrival`, `M5a_time_arrival`,
`M7ra_time_arrival_histcos_rew`) re-anchor the same spline on time since well
arrival over [0, 8] s. They are fit on a strict subset of the rows and are
therefore **never differenced** against the stop-clock rungs. On their supported
interior the two clocks put the median peak at broadly the same delay on the
well-anchored cohort (<!--v:part3.rate_time_peak_s_low-->2.48<!--/v-->–<!--v:part3.rate_time_peak_s_high-->3.00<!--/v--> s on the stop clock vs <!--v:part3.rate_arrival_peak_s_low-->2.94<!--/v-->–<!--v:part3.rate_arrival_peak_s_high-->4.00<!--/v--> s after arrival, KS distance
<!--v:part3.rate_clock_ks_low-->0.08<!--/v-->–<!--v:part3.rate_clock_ks_high-->0.09<!--/v--> on the well-anchored cohort; <!--v:part3.rate_allstops_time_peak_s_low-->2.57<!--/v-->–<!--v:part3.rate_allstops_time_peak_s_high-->3.00<!--/v--> s vs <!--v:part3.rate_allstops_arrival_peak_s_low-->2.91<!--/v-->–<!--v:part3.rate_allstops_arrival_peak_s_high-->5.73<!--/v--> s, KS <!--v:part3.rate_allstops_clock_ks_low-->0.08<!--/v-->–<!--v:part3.rate_allstops_clock_ks_high-->0.09<!--/v--> on
all stops, where the arrival-clock band is wider), so the early-in-stop rise is not an artifact of the bout clock. They
retain a **different** share of each population's event bins — **<!--v:part3.rate_arrival_event_share_swr-->0.95<!--/v-->**
detected SWR, **<!--v:part3.rate_arrival_event_share_non_local_swr-->0.94<!--/v-->** w/ SWR, **<!--v:part3.rate_arrival_event_share_non_local-->0.91<!--/v-->** non-local, **<!--v:part3.rate_arrival_event_share_non_local_no_swr-->0.88<!--/v-->** no SWR (on all
stops: <!--v:part3.rate_allstops_arrival_event_share_swr-->0.91<!--/v--> / <!--v:part3.rate_allstops_arrival_event_share_non_local_swr-->0.91<!--/v--> / <!--v:part3.rate_allstops_arrival_event_share_non_local-->0.86<!--/v--> / <!--v:part3.rate_allstops_arrival_event_share_non_local_no_swr-->0.81<!--/v-->) — so a
between-population contrast on that ladder is partly a contrast between windows;
and they are heavily flagged (<!--v:part3.rate_arrival_separated_fraction_low|pctnum-->58<!--/v-->–<!--v:part3.rate_arrival_separated_fraction_high-->63 %<!--/v--> of `M5a_time_arrival` fits carry a
separated non-nuisance term), so read them pooled, on the fitted curve, never
per session.

### Behavioral-state covariates on rate — PENDING

**The shipped per-bin ladder carries none.** Value, uncertainty, switch-rate and
block covariates are fit in dedicated ladders — `MU_uncertainty`
(`scripts/rate/fit_uncertainty_rate.py`) and the per-bin `is_swr`-contrast
ladder (`scripts/rate/fit_swr_contrast_rate.py`) — on their own frozen cohorts
and with the post-outcome gating they need.

Every coefficient this document previously reported from those two ladders is
**PENDING**: the nominal-value null, the RPE 1.69×, the exploration 0.57×/0.55×
suppression, the patch-gap 0.52/0.77 pair, and the whole per-session
likelihood-ratio table of the `is_swr`-contrast rate ladder (the 53 % / 5.1× /
3.1× reward figures, the dissociation hierarchy, the stratified G-ladder
pattern). They were fit against a reward block and a cohort that no longer
exist. Those ladders were refit on 2026-09-06 alongside the main ladder and now
carry the raised-cosine basis and per-coefficient separation judging, but the
audit reports only fit counts and separation shares for them, not coefficients —
see [per_bin_rate_results.md](per_bin_rate_results.md) §5 and
[2026-09-06-rate-refit-arrival-clock.md](../.claude/docs/reviews/2026-09-06-rate-refit-arrival-clock.md)
("is_swr-contrast"). Note also that separation is heavy there: **<!--v:part3.rate_contrast_separated_fraction_post_outcome-->67 %<!--/v-->** of the contrast cohort's
post-outcome fits and **<!--v:part3.rate_contrast_separated_fraction_primary-->71 %<!--/v-->** of its primary fits carry a separated
non-nuisance term, and `is_swr:reward_history_rcos_1` is flagged in **<!--v:part3.rate_contrast_term_separated_fraction-->75 %<!--/v-->**
of the fits that carry it — so any re-derived aggregate must state
its excluded share.

## Layer 2 — SWR-coupling (ripple power as a continuous mark)

`ripple_power_z` (event-mean consensus ripple power — the signal the Kay detector
thresholds, so `is_swr` is its thresholded version). Per-event regression
(`scripts/coupling/fit_swr_power_model.py`, <!--v:part3.power_sessions-->242<!--/v--> sessions). **Scope note:** this layer was refit
on the current tables, but its fit has **no provenance sidecar**, so every number
below is pending.

- **Reward (<!--v:part3.power_reward-->+0.040<!--/v-->/SD, <!--v:part3.power_reward_animals_positive-->5<!--/v-->/5 animals positive, p <!--v:part3.power_reward_p-->1.1e-5<!--/v-->)** and **RPE (<!--v:part3.power_rpe-->+0.10<!--/v-->/SD, <!--v:part3.power_rpe_animals_positive-->5<!--/v-->/5, p <!--v:part3.power_rpe_p-->5.9e-22<!--/v-->)** raise SWR power;
  **exploration lowers it (<!--v:part3.power_exploration-->−0.073<!--/v-->/SD, <!--v:part3.power_exploration_animals_positive-->0<!--/v-->/5 positive, p <!--v:part3.power_exploration_p-->9.8e-11<!--/v-->)**. Combined with Layer 1 (the reconstructed
  occurrence kernel puts the immediate reward effect at RR <!--v:part3.rate_kernel_lag_one_swr-->1.38<!--/v--> for detected
  SWRs, <!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v--> for SWR-coupled non-local events and <!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v--> for non-SWR non-local
  events), reward acts on **coupling** more than on bare occurrence —
  i.e. the binary "with-SWR rate" dissociation is largely reward pushing events'
  ripple power up across the threshold. *Caveat:* the power-model reward covariate
  is the current trial's known outcome (`known_reward_outcome_lag_1`, the
  **current-well** reward, dwell-confounded), so "reward raises coupling" may be partly a **consummatory
  LFP artifact**, not a clean pre-event mark.
- **Scope:** this is the *mark factor only*, by per-event OLS — **not** a full
  marked point process (no joint intensity, no test of power↔history independence,
  no shared latent).

## Layer 3 — Content (represented arm)

Conditional-Poisson arm ladder on the **per-event allocation fraction** with
event-clustered SEs (refit 2026-09-05). Content predictors are *arm-relative* (a
constant-across-arms covariate cancels in the per-event normalization), so
scalar behavioral covariates enter only as **interactions with
`is_current_arm`** (the arm the animal occupies). Every coefficient below is an
animal-balanced median with animals-positive-of-5 and a session sign-rank FDR q,
and every `swr_x_...` term states whether it is the length-adjusted **direct**
effect (`BASEc`) or the **total** effect (`nt` twin on `BASEc_NT`).

**The anchors:**

| finding | rung / term | value |
| --- | --- | --- |
| current-arm bias, pooled | `C2_base` `is_current_arm` | **<!--v:part3.base_current_all-->+0.73<!--/v-->** (<!--v:part3.base_current_all_sessions-->242<!--/v--> sessions, <!--v:part3.base_current_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.base_current_all_q-->1.9e-34<!--/v-->) |
| current-arm bias, non-SWR | same, nonSWR population | **<!--v:part3.base_current_non_swr-->+0.53<!--/v-->** (<!--v:part3.base_current_non_swr_sessions-->239<!--/v--> sessions, <!--v:part3.base_current_non_swr_animals_positive-->5<!--/v-->/5) |
| current-arm bias, SWR | same, SWR population | **<!--v:part3.base_current_swr-->+0.88<!--/v-->** (<!--v:part3.base_current_swr_sessions-->241<!--/v--> sessions, <!--v:part3.base_current_swr_animals_positive-->4<!--/v-->/5) |
| SWR less current-locked (direct) | `CS1_current` `swr_x_is_current_arm` | **<!--v:part3.swr_current_direct-->−0.071<!--/v-->** (<!--v:part3.swr_current_direct_sessions-->242<!--/v--> sessions, <!--v:part3.swr_current_direct_animals_positive-->1<!--/v-->/5, q <!--v:part3.swr_current_direct_q-->4.3e-7<!--/v-->) |
| SWR less current-locked (total) | `CS1nt_current` | **<!--v:part3.swr_current_total-->−0.41<!--/v-->** (<!--v:part3.swr_current_total_sessions-->242<!--/v--> sessions, <!--v:part3.swr_current_total_animals_positive-->0<!--/v-->/5, q <!--v:part3.swr_current_total_q-->7.0e-28<!--/v-->) |
| SWR less prev-locked (direct) | `CS2_prev` `swr_x_is_prev_arm` | **<!--v:part3.swr_prev_direct-->−0.42<!--/v-->** (<!--v:part3.swr_prev_direct_sessions-->240<!--/v--> sessions, <!--v:part3.swr_prev_direct_animals_positive-->1<!--/v-->/5, q <!--v:part3.swr_prev_direct_q-->3.7e-15<!--/v-->) |
| arm reward fraction, non-SWR | `C3_armreward` `reward_fraction_candidate_arm_last10_visits` | **<!--v:part3.armreward_non_swr-->+0.027<!--/v-->** (<!--v:part3.armreward_non_swr_sessions-->239<!--/v--> sessions, <!--v:part3.armreward_non_swr_animals_positive-->3<!--/v-->/5, q <!--v:part3.armreward_non_swr_q-->0.35<!--/v-->) |
| arm reward fraction, SWR | same, SWR population | **<!--v:part3.armreward_swr-->−0.26<!--/v-->** (<!--v:part3.armreward_swr_sessions-->241<!--/v--> sessions, <!--v:part3.armreward_swr_animals_positive-->0<!--/v-->/5, q <!--v:part3.armreward_swr_q-->3.3e-4<!--/v-->) |
| SWR avoids recently-rewarded arms (direct) | `CS4_armreward` `swr_x_reward_fraction_candidate_arm_last10_visits` | **<!--v:part3.swr_armreward_direct-->−1.0<!--/v-->** (<!--v:part3.swr_armreward_direct_sessions-->242<!--/v--> sessions, <!--v:part3.swr_armreward_direct_animals_positive-->0<!--/v-->/5, q <!--v:part3.swr_armreward_direct_q-->1.7e-35<!--/v-->) |
| ... (total) | `CS4nt_armreward` | **<!--v:part3.swr_armreward_total-->−1.2<!--/v-->** (<!--v:part3.swr_armreward_total_sessions-->242<!--/v--> sessions, <!--v:part3.swr_armreward_total_animals_positive-->0<!--/v-->/5, q <!--v:part3.swr_armreward_total_q-->1.3e-38<!--/v-->) |
| current-patch bias | `CP0_base` `is_current_patch` | **<!--v:part3.patch_current-->+1.1<!--/v-->** (<!--v:part3.patch_current_sessions-->230<!--/v--> sessions, <!--v:part3.patch_current_animals_positive-->5<!--/v-->/5, q <!--v:part3.patch_current_q-->4.4e-38<!--/v-->) |
| SWR less patch-locked (direct / total) | `CPS1_current` / `CPS1nt_current` `swr_x_is_current_patch` | **<!--v:part3.swr_patch_current_direct-->+0.015<!--/v-->** (<!--v:part3.swr_patch_current_direct_sessions-->236<!--/v--> sessions, <!--v:part3.swr_patch_current_direct_animals_positive-->3<!--/v-->/5, q <!--v:part3.swr_patch_current_direct_q-->0.0016<!--/v-->) / **<!--v:part3.swr_patch_current_total-->−0.21<!--/v-->** (<!--v:part3.swr_patch_current_total_sessions-->239<!--/v--> sessions, <!--v:part3.swr_patch_current_total_animals_positive-->1<!--/v-->/5, q <!--v:part3.swr_patch_current_total_q-->1.2e-6<!--/v-->) |

**The population-split rows and the `swr_x_...` rows are different estimands,
not a contradiction.** A `SWR` or `nonSWR` row is a *separately baselined* fit on
that population's own, unpaired, differently sized session set (<!--v:part3.base_current_swr_sessions-->241<!--/v--> vs <!--v:part3.base_current_non_swr_sessions-->239<!--/v--> on
`C2_base`), so it answers "is the bias present in this population". A
`swr_x_...` row is the *pooled single-fit* SWR contrast against the shared
non-SWR baseline, so it answers "does SWR change it". Both `C2_base` splits being
strongly positive while `swr_x_is_current_arm` is negative is therefore
consistent, not conflicting.

So **non-SWR content is reward-value-neutral while with-SWR content avoids
recently rewarded arms**, and SWR content is less locked to the current arm, the
previous arm and the current patch — with the caveat that the patch-level
current contrast survives only as a **total** effect. The current-arm bias is
not an estimator artifact: count **<!--v:part3.est_current_count-->+0.67<!--/v-->**, fraction **<!--v:part3.est_current_fraction-->+0.73<!--/v-->**, DM **<!--v:part3.est_current_dm-->+0.73<!--/v-->**
(<!--v:part3.est_current_count_animals_positive-->5<!--/v-->/5, <!--v:part3.est_current_fraction_animals_positive-->5<!--/v-->/5, <!--v:part3.est_current_dm_animals_positive-->5<!--/v-->/5 animals), fraction bootstrap [<!--v:part3.est_current_fraction_low-->+0.25<!--/v-->, <!--v:part3.est_current_fraction_high-->+1.4<!--/v-->] — the three-estimator run, on the same tables as
the shipped fit (**<!--v:part3.base_current_all-->+0.73<!--/v-->**).

**"SWR broadens content", measured directly rather than inferred from arm
coefficients.** The Dirichlet-multinomial adds a within-event **concentration**
parameter the conditional estimators do not have. `log_phi_is_swr` is **<!--v:part3.dm_breadth-->+1.7<!--/v-->**
on `C2_base` (animal bootstrap [<!--v:part3.est_breadth_dm_low-->+1.5<!--/v-->, <!--v:part3.est_breadth_dm_high-->+2.1<!--/v-->]), positive for **<!--v:part3.dm_breadth_animals_positive-->5<!--/v-->/5 animals and
<!--v:part3.dm_breadth_positive_fraction-->100 %<!--/v--> of sessions**, and positive on every rung of the pooled table. Large φ
means the event's timebins sit closer to independent draws from its own arm
distribution and its mass is spread over more arms, so an SWR event's φ is
≈ e^<!--v:part3.dm_breadth|sig2-->1.7<!--/v--> ≈ **<!--v:part3.dm_breadth_ratio-->5.6<!--/v-->×** a non-SWR event's. **Caveat — φ is assumed length-invariant.**
The likelihood uses event length only as the multinomial index, and length is
post-treatment with respect to `is_swr`: SWR events run **<!--v:part3.event_bins_ratio-->2.0<!--/v-->×** longer (median
<!--v:part3.event_bins_median_swr-->47<!--/v--> vs <!--v:part3.event_bins_median_non_swr-->24<!--/v--> bins). The length-adjusted twin (`_dm_len`) puts the contrast at
**<!--v:part3.dm_breadth_length_adjusted-->+1.4<!--/v-->** on `C2_base` (<!--v:part3.dm_breadth_length_adjusted_animals_positive-->5<!--/v-->/5, q <!--v:part3.dm_breadth_length_adjusted_q-->2.7e-41<!--/v-->) — ≈ **<!--v:part3.dm_breadth_length_adjusted_ratio-->3.9<!--/v-->×** at the same length — with
the length term itself at **<!--v:part3.dm_breadth_length_term-->+0.55<!--/v--> per SD**. About <!--v:part3.dm_breadth_length_share-->21 %<!--/v--> of the total runs
through event length and the finding survives. Quote the total (<!--v:part3.dm_breadth-->+1.7<!--/v-->) or the
length-adjusted (<!--v:part3.dm_breadth_length_adjusted-->+1.4<!--/v-->) figure, but say which. This is a statement about
**within-event timebin redundancy**, not about which arms are represented.

**The reward-memory (last-rewarded) family is refused as unidentified.** It has
no usable rung in the per-session arm ladder (see
[design/model_history.md](design/model_history.md)), and the previous avoidance
coefficients — −1.485 at the arm grain, −1.62 SWR-split, −1.49 patch-grain — are
**withdrawn with no replacement number**.

**Behavioral modulation of the current-arm bias is PENDING.** The
`is_current_arm` interaction block fit by an exploratory script (reward x
current, scheduled-contingency x current, patch-switch-fraction x current,
within-stop time x current: the −0.76 / −0.53 / +0.53 / +0.23 quartet) used the
retired count response; those coefficients are withdrawn until refit. What
survives on the fraction response is the `C`-rung interaction set below.

### Content `is_swr`-contrast ladder (arm + patch)

Content analog of the rate `is_swr`-contrast ladder, with the beta-bernoulli
covariates entered as `_x_is_current_arm` products (event scalars) or per-arm
main effects. Runner `scripts/content/fit_swr_contrast_content.py`; refit
2026-09-05 on the per-event fraction response with event-clustered SEs. The
value construct is beta-bernoulli
`reward_belief_mean_candidate_arm_preoutcome` — the single behavioral model
since the 2026-07-13 HMM retirement, so the old "construct-dependence" caveat is
moot.

**State modulation of the current-arm bias** (all `_x_is_current_arm` products,
animal-balanced median, animals-positive/5, session sign-rank FDR q):

| rung / term | pooled `all` | SWR | non-SWR |
| --- | --- | --- | --- |
| `C1_current` `is_current_arm` (no autocorrelation controls) | <!--v:part3.base_current_no_controls_all-->+0.78<!--/v--> (<!--v:part3.base_current_no_controls_all_sessions-->242<!--/v-->, <!--v:part3.base_current_no_controls_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.base_current_no_controls_all_q-->4.6e-37<!--/v-->) | <!--v:part3.base_current_no_controls_swr-->+0.84<!--/v--> (<!--v:part3.base_current_no_controls_swr_sessions-->242<!--/v-->, <!--v:part3.base_current_no_controls_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.base_current_no_controls_swr_q-->4.1e-34<!--/v-->) | <!--v:part3.base_current_no_controls_non_swr-->+0.64<!--/v--> (<!--v:part3.base_current_no_controls_non_swr_sessions-->241<!--/v-->, <!--v:part3.base_current_no_controls_non_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.base_current_no_controls_non_swr_q-->4.1e-35<!--/v-->) |
| `C4_armvalue` `reward_belief_mean_candidate_arm_preoutcome` | <!--v:part3.armvalue_all-->+0.078<!--/v--> (<!--v:part3.armvalue_all_sessions-->242<!--/v-->, <!--v:part3.armvalue_all_animals_positive-->4<!--/v-->/5, q <!--v:part3.armvalue_all_q-->0.11<!--/v-->) | <!--v:part3.armvalue_swr-->−0.11<!--/v--> (<!--v:part3.armvalue_swr_sessions-->239<!--/v-->, <!--v:part3.armvalue_swr_animals_positive-->2<!--/v-->/5, q <!--v:part3.armvalue_swr_q-->0.62<!--/v-->) | <!--v:part3.armvalue_non_swr-->+0.041<!--/v--> (<!--v:part3.armvalue_non_swr_sessions-->239<!--/v-->, <!--v:part3.armvalue_non_swr_animals_positive-->4<!--/v-->/5, q <!--v:part3.armvalue_non_swr_q-->0.053<!--/v-->) |
| `C6_rpe` `rpe_postoutcome_z` | <!--v:part3.rpe_current_all-->−0.12<!--/v--> (<!--v:part3.rpe_current_all_sessions-->242<!--/v-->, <!--v:part3.rpe_current_all_animals_positive-->0<!--/v-->/5, q <!--v:part3.rpe_current_all_q-->6.6e-8<!--/v-->) | <!--v:part3.rpe_current_swr-->−0.019<!--/v--> (<!--v:part3.rpe_current_swr_sessions-->240<!--/v-->, <!--v:part3.rpe_current_swr_animals_positive-->2<!--/v-->/5, q <!--v:part3.rpe_current_swr_q-->0.89<!--/v-->) | <!--v:part3.rpe_current_non_swr-->−0.16<!--/v--> (<!--v:part3.rpe_current_non_swr_sessions-->239<!--/v-->, <!--v:part3.rpe_current_non_swr_animals_positive-->0<!--/v-->/5, q <!--v:part3.rpe_current_non_swr_q-->7.8e-7<!--/v-->) |
| `C7_value` `relative_value_switch_minus_stay_preoutcome_z` | <!--v:part3.relative_value_current_all-->+0.096<!--/v--> (<!--v:part3.relative_value_current_all_sessions-->242<!--/v-->, <!--v:part3.relative_value_current_all_animals_positive-->4<!--/v-->/5, q <!--v:part3.relative_value_current_all_q-->2.5e-7<!--/v-->) | <!--v:part3.relative_value_current_swr-->+0.14<!--/v--> (<!--v:part3.relative_value_current_swr_sessions-->241<!--/v-->, <!--v:part3.relative_value_current_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.relative_value_current_swr_q-->2.1e-7<!--/v-->) | <!--v:part3.relative_value_current_non_swr-->+0.054<!--/v--> (<!--v:part3.relative_value_current_non_swr_sessions-->239<!--/v-->, <!--v:part3.relative_value_current_non_swr_animals_positive-->4<!--/v-->/5, q <!--v:part3.relative_value_current_non_swr_q-->0.051<!--/v-->) |
| `C8_uncert` `reward_belief_differential_entropy_leaf_mean_preoutcome_z` | <!--v:part3.entropy_current_all-->−0.040<!--/v--> (<!--v:part3.entropy_current_all_sessions-->242<!--/v-->, <!--v:part3.entropy_current_all_animals_positive-->2<!--/v-->/5, q <!--v:part3.entropy_current_all_q-->0.54<!--/v-->) | <!--v:part3.entropy_current_swr-->−0.14<!--/v--> (<!--v:part3.entropy_current_swr_sessions-->239<!--/v-->, <!--v:part3.entropy_current_swr_animals_positive-->0<!--/v-->/5, q <!--v:part3.entropy_current_swr_q-->0.023<!--/v-->) | <!--v:part3.entropy_current_non_swr-->+0.031<!--/v--> (<!--v:part3.entropy_current_non_swr_sessions-->238<!--/v-->, <!--v:part3.entropy_current_non_swr_animals_positive-->3<!--/v-->/5, q <!--v:part3.entropy_current_non_swr_q-->0.74<!--/v-->) |
| `C9_update` `reward_belief_mean_update_l1_all_ports_postoutcome_z` | <!--v:part3.update_current_all-->+0.24<!--/v--> (<!--v:part3.update_current_all_sessions-->242<!--/v-->, <!--v:part3.update_current_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.update_current_all_q-->2.0e-18<!--/v-->) | <!--v:part3.update_current_swr-->+0.22<!--/v--> (<!--v:part3.update_current_swr_sessions-->241<!--/v-->, <!--v:part3.update_current_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.update_current_swr_q-->1.3e-19<!--/v-->) | <!--v:part3.update_current_non_swr-->+0.14<!--/v--> (<!--v:part3.update_current_non_swr_sessions-->237<!--/v-->, <!--v:part3.update_current_non_swr_animals_positive-->4<!--/v-->/5, q <!--v:part3.update_current_non_swr_q-->4.6e-5<!--/v-->) |
| `C10_periswitch` `periswitch_index_trials_z` | <!--v:part3.periswitch_current_all-->+0.22<!--/v--> (<!--v:part3.periswitch_current_all_sessions-->221<!--/v-->, <!--v:part3.periswitch_current_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.periswitch_current_all_q-->6.3e-4<!--/v-->) | <!--v:part3.periswitch_current_swr-->+0.23<!--/v--> (<!--v:part3.periswitch_current_swr_sessions-->191<!--/v-->, <!--v:part3.periswitch_current_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.periswitch_current_swr_q-->3.3e-4<!--/v-->) | <!--v:part3.periswitch_current_non_swr-->+0.11<!--/v--> (<!--v:part3.periswitch_current_non_swr_sessions-->189<!--/v-->, <!--v:part3.periswitch_current_non_swr_animals_positive-->3<!--/v-->/5, q <!--v:part3.periswitch_current_non_swr_q-->0.60<!--/v-->) |
| `C11_block` `trials_since_block_change_z` | <!--v:part3.block_current_all-->−0.097<!--/v--> (<!--v:part3.block_current_all_sessions-->242<!--/v-->, <!--v:part3.block_current_all_animals_positive-->0<!--/v-->/5, q <!--v:part3.block_current_all_q-->1.5e-9<!--/v-->) | <!--v:part3.block_current_swr-->−0.15<!--/v--> (<!--v:part3.block_current_swr_sessions-->239<!--/v-->, <!--v:part3.block_current_swr_animals_positive-->0<!--/v-->/5, q <!--v:part3.block_current_swr_q-->9.7e-9<!--/v-->) | <!--v:part3.block_current_non_swr-->−0.012<!--/v--> (<!--v:part3.block_current_non_swr_sessions-->238<!--/v-->, <!--v:part3.block_current_non_swr_animals_positive-->2<!--/v-->/5, q <!--v:part3.block_current_non_swr_q-->0.051<!--/v-->) |
| `C12_switchrate` `patch_switch_fraction_last5_completed_trips_z` | <!--v:part3.switchrate_current_all-->+0.18<!--/v--> (<!--v:part3.switchrate_current_all_sessions-->242<!--/v-->, <!--v:part3.switchrate_current_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.switchrate_current_all_q-->5.9e-23<!--/v-->) | <!--v:part3.switchrate_current_swr-->+0.33<!--/v--> (<!--v:part3.switchrate_current_swr_sessions-->239<!--/v-->, <!--v:part3.switchrate_current_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.switchrate_current_swr_q-->3.6e-22<!--/v-->) | <!--v:part3.switchrate_current_non_swr-->+0.23<!--/v--> (<!--v:part3.switchrate_current_non_swr_sessions-->237<!--/v-->, <!--v:part3.switchrate_current_non_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.switchrate_current_non_swr_q-->7.3e-10<!--/v-->) |
| `C13_armuncert` `reward_belief_variance_candidate_arm_preoutcome_z` (per-arm main effect) | <!--v:part3.armuncert_all-->+0.31<!--/v--> (<!--v:part3.armuncert_all_sessions-->242<!--/v-->, <!--v:part3.armuncert_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_all_q-->2.7e-27<!--/v-->) | <!--v:part3.armuncert_swr-->+0.37<!--/v--> (<!--v:part3.armuncert_swr_sessions-->239<!--/v-->, <!--v:part3.armuncert_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_swr_q-->1.9e-27<!--/v-->) | <!--v:part3.armuncert_non_swr-->+0.18<!--/v--> (<!--v:part3.armuncert_non_swr_sessions-->239<!--/v-->, <!--v:part3.armuncert_non_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_non_swr_q-->1.8e-14<!--/v-->) |

Readings. **Exploration locks content onto the current arm** (`C12`, <!--v:part3.switchrate_current_swr_animals_positive-->5<!--/v-->/5 SWR and
<!--v:part3.switchrate_current_non_swr_animals_positive-->5<!--/v-->/5 non-SWR) and **near a patch switch content moves off it** (`C10`, pooled
<!--v:part3.periswitch_current_all_animals_positive-->5<!--/v-->/5, but the non-SWR split is a null at <!--v:part3.periswitch_current_non_swr-->+0.11<!--/v-->). **RPE pushes off current**
(`C6`, <!--v:part3.rpe_current_all_animals_positive-->0<!--/v-->/5 positive pooled and <!--v:part3.rpe_current_non_swr_animals_positive-->0<!--/v-->/5 in non-SWR). **Relative value ΔV sharpens onto
current** (`C7`) — but read the sign with care: ΔV is *Switch − Stay*, so a
positive coefficient means content sharpens onto the current arm when that arm's
value is **low**. **Value-update sharpens onto current** (`C9`, <!--v:part3.update_current_all_animals_positive-->5<!--/v-->/5 pooled,
<!--v:part3.update_current_swr_animals_positive-->5<!--/v-->/5 SWR, <!--v:part3.update_current_non_swr_animals_positive-->4<!--/v-->/5 non-SWR). The **scalar belief-entropy interaction is null** (`C8`) — the
per-arm uncertainty *main effect* (`C13`) is the content-uncertainty result; do
not conflate them. The **arm-value main effect is weak however the fit is specified**, and it does not
turn on the arm support (`C4_armvalue` and `CS5_armvalue` are both leaf-only): in the *split* `C4`
fits it is not significant (all <!--v:part3.armvalue_all-->+0.078<!--/v-->, <!--v:part3.armvalue_all_sessions-->242<!--/v--> sessions, <!--v:part3.armvalue_all_animals_positive-->4<!--/v-->/5, q <!--v:part3.armvalue_all_q-->0.11<!--/v-->; non-SWR <!--v:part3.armvalue_non_swr-->+0.041<!--/v-->, <!--v:part3.armvalue_non_swr_sessions-->239<!--/v-->, <!--v:part3.armvalue_non_swr_animals_positive-->4<!--/v-->/5;
SWR <!--v:part3.armvalue_swr-->−0.11<!--/v-->, <!--v:part3.armvalue_swr_sessions-->239<!--/v-->, <!--v:part3.armvalue_swr_animals_positive-->2<!--/v-->/5), and in the *pooled* `CS5` fit — one model over both event
classes with the `swr_x_` interaction, so the main term is the non-SWR reference level — it is
positive in the median but not across animals: `CS5_armvalue`'s main term
`reward_belief_mean_candidate_arm_preoutcome` reads **<!--v:part3.est_armvalue_fraction-->+0.26<!--/v-->** (fraction, <!--v:part3.est_armvalue_fraction_sessions-->242<!--/v-->
sessions, **<!--v:part3.est_armvalue_fraction_animals_positive-->3<!--/v-->/5 animals positive**, bootstrap [<!--v:part3.est_armvalue_fraction_low-->−0.26<!--/v-->, <!--v:part3.est_armvalue_fraction_high-->+0.61<!--/v-->]; count <!--v:part3.est_armvalue_count-->+0.70<!--/v-->,
<!--v:part3.est_armvalue_count_animals_positive-->4<!--/v-->/5; DM <!--v:part3.est_armvalue_dm-->+0.30<!--/v-->, <!--v:part3.est_armvalue_dm_animals_positive-->4<!--/v-->/5) and the leaf bundle's `Mv` reads **<!--v:part3.leaf_value-->+0.24<!--/v-->** (<!--v:part3.leaf_value_sessions-->242<!--/v--> sessions,
<!--v:part3.leaf_value_animals_positive-->3<!--/v-->/5). So the value main effect is positive in the median under every
specification but not consistent across animals; do not write that content is
value-selective, nor that it is not.

**Rate/content asymmetry (restated).** On *content*, several modulations
(uncertainty, periswitch, switchrate, value-update) are SWR-equal-or-stronger —
SWR content is *broader* at baseline yet *retunable* by state. The
matching statement on *rate* rested on the `is_swr`-contrast rate ladder and is
**PENDING** (see Layer 1).

### Per-arm uncertainty (the exploration axis)

Per-arm content covariate **`reward_belief_variance_candidate_arm_preoutcome`**
= the per-leaf Beta belief variance (the variance counterpart of
`reward_belief_mean_candidate_arm_preoutcome`, from the SAME Beta posterior).
Built into `event_arm_table` (leaf arm *j* to `betadist_var{j+1}`, trunk arms
0.0, standardized `_z`); rungs `Mu` / `Mvu` in the arm hierarchy and
`C13` / `CS6` in the `is_swr` ladder. Generative-model assumptions in
[behavior_models.md](behavior_models.md).

- **Content over-represents high-uncertainty arms.** On the leaf-only bundle
  `models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu`, `Mu`'s
  `reward_belief_variance_candidate_arm_preoutcome_z` is **<!--v:part3.leaf_uncertainty-->+0.31<!--/v-->** (session
  median over <!--v:part3.leaf_uncertainty_sessions-->242<!--/v--> sessions, **<!--v:part3.leaf_uncertainty_animals_positive-->5<!--/v-->/5 animals positive**, median event-clustered SE
  <!--v:part3.leaf_uncertainty_se-->0.18<!--/v-->). This is the exploration / gain axis (Mattar-Daw) the
  value/reward/space ladder was missing.
- **It earns its parameter where value does not.** Median per-session ΔAIC against the nested parent (positive
  = child wins): `M4 → Mu` **<!--v:part3.leaf_step_uncertainty-->+1.3<!--/v-->** (ΔAIC > 2 in <!--v:part3.leaf_step_uncertainty_clear-->112<!--/v--> of <!--v:part3.leaf_step_uncertainty_sessions-->242<!--/v--> sessions, <!--v:part3.leaf_step_uncertainty_animals-->5<!--/v-->/5
  animals child-better), against `M4 → Mv` **<!--v:part3.leaf_step_value-->−2.4<!--/v-->** (ΔAIC > 2 in <!--v:part3.leaf_step_value_clear-->22<!--/v-->/<!--v:part3.leaf_step_value_sessions-->242<!--/v-->, <!--v:part3.leaf_step_value_animals-->0<!--/v-->/5
  animals) and `Mu → Mvu` **<!--v:part3.leaf_step_both-->−1.5<!--/v-->** (ΔAIC > 2 in <!--v:part3.leaf_step_both_clear-->12<!--/v-->/<!--v:part3.leaf_step_both_sessions-->242<!--/v-->, <!--v:part3.leaf_step_both_animals-->0<!--/v-->/5). The session count
  is the number of sessions clearing ΔAIC > 2, not a bare sign count. On the
  same leaf design the value *coefficient* is positive in the median but not
  across animals — `Mv`
  `reward_belief_mean_candidate_arm_preoutcome` **<!--v:part3.leaf_value-->+0.24<!--/v-->** (<!--v:part3.leaf_value_sessions-->242<!--/v--> sessions, <!--v:part3.leaf_value_animals_positive-->3<!--/v-->/5)
  and the `CS5_armvalue` main term **<!--v:part3.est_armvalue_fraction-->+0.26<!--/v-->** (fraction, <!--v:part3.est_armvalue_fraction_sessions-->242<!--/v--> sessions, <!--v:part3.est_armvalue_fraction_animals_positive-->3<!--/v-->/5,
  bootstrap [<!--v:part3.est_armvalue_fraction_low-->−0.26<!--/v-->, <!--v:part3.est_armvalue_fraction_high-->+0.61<!--/v-->]).
- **Both populations over-represent uncertain arms, SWR more so.** The split
  `C13_armuncert` gives **SWR <!--v:part3.armuncert_swr-->+0.37<!--/v-->** (<!--v:part3.armuncert_swr_sessions-->239<!--/v--> sessions, <!--v:part3.armuncert_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_swr_q-->1.9e-27<!--/v-->) against **non-SWR
  <!--v:part3.armuncert_non_swr-->+0.18<!--/v-->** (<!--v:part3.armuncert_non_swr_sessions-->239<!--/v--> sessions, <!--v:part3.armuncert_non_swr_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_non_swr_q-->1.8e-14<!--/v-->), pooled `all` **<!--v:part3.armuncert_all-->+0.31<!--/v-->** (<!--v:part3.armuncert_all_sessions-->242<!--/v--> sessions, <!--v:part3.armuncert_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.armuncert_all_q-->2.7e-27<!--/v-->).
- **The pooled `swr_x` interaction is null as a direct effect and positive as a
  total effect, and the two estimators agree on each.** *Direct*
  `CS6_armuncert` — fraction **<!--v:part3.swr_armuncert_direct-->−0.058<!--/v-->** (<!--v:part3.swr_armuncert_direct_sessions-->242<!--/v--> sessions, <!--v:part3.swr_armuncert_direct_animals_positive-->2<!--/v-->/5, q <!--v:part3.swr_armuncert_direct_q-->0.88<!--/v-->) and DM
  **<!--v:part3.dm_swr_armuncert_direct-->−0.070<!--/v-->** (<!--v:part3.dm_swr_armuncert_direct_animals_positive-->2<!--/v-->/5, q <!--v:part3.dm_swr_armuncert_direct_q-->0.16<!--/v-->). *Total* `CS6nt_armuncert` — fraction **<!--v:part3.swr_armuncert_total-->+0.18<!--/v-->**
  (<!--v:part3.swr_armuncert_total_sessions-->242<!--/v--> sessions, **<!--v:part3.swr_armuncert_total_animals_positive-->5<!--/v-->/5**, q <!--v:part3.swr_armuncert_total_q-->1.9e-12<!--/v-->) and DM **<!--v:part3.dm_swr_armuncert_total-->+0.12<!--/v-->** (<!--v:part3.dm_swr_armuncert_total_animals_positive-->4<!--/v-->/5, q <!--v:part3.dm_swr_armuncert_total_q-->1.5e-10<!--/v-->). So the
  extra SWR weighting on uncertainty appears in the **total** contrast and
  vanishes once event length is held fixed. The earlier reading — that the
  pooled `CS6` interaction was a nine-arm fixed-effect artifact — is superseded:
  on the leaf-only fraction fit the direct interaction is simply a null.
- **PENDING (not re-derived on the fraction response):** the
  recency-orthogonalized uncertainty effect (+0.62), the recency VIF
  decomposition, the within-arm value-by-uncertainty horse-race (uncertainty
  +0.06 with value and recency near zero), the `Mvu` VIF diagnostics, and the
  per-arm realized-gain test. All were count-response fits; their numbers are
  withdrawn until refit. The *design* conclusions they supported — that the
  within-arm frame identifies value and uncertainty separately, and that the
  gain axis of content is epistemic uncertainty rather than realized
  value-update — are retained as hypotheses, not as measurements.

### SWR by arm value — the one estimator-dependent conclusion

`CS5_armvalue`'s `swr_x_reward_belief_mean_candidate_arm_preoutcome` is the only
term in the whole three-estimator comparison whose sign moves with the
likelihood. Animal-balanced median, animals-positive/5, animal-bootstrap
2.5–97.5, on the <!--v:part3.est_swr_armvalue_fraction_sessions-->238<!--/v-->-session cohort:

| estimand | count (retired) | fraction (shipped) | Dirichlet-multinomial |
| --- | --- | --- | --- |
| **direct** (`CS5_armvalue`, length-adjusted) | <!--v:part3.est_swr_armvalue_count-->−0.17<!--/v--> (<!--v:part3.est_swr_armvalue_count_animals_positive-->2<!--/v-->/5) [<!--v:part3.est_swr_armvalue_count_low-->−1.1<!--/v-->, <!--v:part3.est_swr_armvalue_count_high-->+0.18<!--/v-->] | **<!--v:part3.est_swr_armvalue_fraction-->−0.014<!--/v--> (<!--v:part3.est_swr_armvalue_fraction_animals_positive-->2<!--/v-->/5) [<!--v:part3.est_swr_armvalue_fraction_low-->−0.81<!--/v-->, <!--v:part3.est_swr_armvalue_fraction_high-->+0.20<!--/v-->]** | <!--v:part3.est_swr_armvalue_dm-->+0.046<!--/v--> (<!--v:part3.est_swr_armvalue_dm_animals_positive-->3<!--/v-->/5) [<!--v:part3.est_swr_armvalue_dm_low-->−0.73<!--/v-->, <!--v:part3.est_swr_armvalue_dm_high-->+0.10<!--/v-->] |
| **total** (`CS5nt_armvalue`) | — | **<!--v:part3.swr_armvalue_total-->−0.25<!--/v-->** (<!--v:part3.swr_armvalue_total_animals_positive-->0<!--/v-->/5, q <!--v:part3.swr_armvalue_total_q-->5.7e-6<!--/v-->) | **<!--v:part3.dm_swr_armvalue_total-->−0.46<!--/v-->** (<!--v:part3.dm_swr_armvalue_total_animals_positive-->0<!--/v-->/5, q <!--v:part3.dm_swr_armvalue_total_q-->3.1e-11<!--/v-->) |

The shipped audited fraction fit reads **<!--v:part3.swr_armvalue_direct-->−0.014<!--/v-->** (<!--v:part3.swr_armvalue_direct_sessions-->238<!--/v--> sessions, <!--v:part3.swr_armvalue_direct_animals_positive-->2<!--/v-->/5, q <!--v:part3.swr_armvalue_direct_q-->0.18<!--/v-->);
the DM's direct q is <!--v:part3.dm_swr_armvalue_direct_q-->0.031<!--/v-->. **By current outcome**, on the fraction response: the
**rewarded** stratum gives direct `CS5_armvalue` **<!--v:part3.swr_armvalue_direct_rewarded-->−0.053<!--/v-->** (<!--v:part3.swr_armvalue_direct_rewarded_sessions-->237<!--/v--> sessions, <!--v:part3.swr_armvalue_direct_rewarded_animals_positive-->2<!--/v-->/5
animals positive) and total `CS5nt_armvalue` **<!--v:part3.swr_armvalue_total_rewarded-->−0.33<!--/v-->** (<!--v:part3.swr_armvalue_total_rewarded_sessions-->239<!--/v--> sessions, <!--v:part3.swr_armvalue_total_rewarded_animals_positive-->0<!--/v-->/5) —
the same near-zero direct / negative total split as the pooled fit. The
**omitted** stratum is **not fit for these two rungs**, so there is no
omission-side value number. **The DIRECT effect is near zero under all three estimators and its sign
differs between them** — count and fraction negative, DM positive; every
bootstrap interval spans zero and no estimator has 0/5 animals positive —
**while the TOTAL (`nt`) effect is negative under both estimators that fit it.**
Report it as a three-way comparison; do not write that any estimator shows a
direct value avoidance.

The rung is also not worth its parameters on either honest scale. The
animal-balanced median per-session ΔAIC against `C2_base_leaf` — AIC(rung) −
AIC(parent), **negative = rung better**, the same convention as the rate ΔAIC in
Layer 1 — is **<!--v:part3.est_armvalue_rung_delta_aic_fraction-->+2.2<!--/v-->** (fraction) and **<!--v:part3.est_armvalue_rung_delta_aic_dm-->+2.0<!--/v-->** (DM), against **<!--v:part3.est_armvalue_rung_delta_aic_count-->−66<!--/v-->** on the retired
count scale. The count figure is the timebin-pseudoreplication artefact, not an
effect.

`CS5b_valuevsprevgoal`, the previous-goal control rung, is **refused as
unidentified**: <!--v:part3.dm_swr_prevgoal_sessions-->152<!--/v--> sessions with **<!--v:part3.dm_swr_prevgoal_separated-->104<!--/v--> flagged separated** under the DM (<!--v:part3.dm_swr_prevgoal-->−0.12<!--/v-->,
<!--v:part3.dm_swr_prevgoal_animals_positive-->1<!--/v-->/5, q <!--v:part3.dm_swr_prevgoal_q-->0.083<!--/v-->); the fraction fit has <!--v:part3.swr_prevgoal_sessions-->198<!--/v--> sessions (<!--v:part3.swr_prevgoal-->−0.061<!--/v-->, <!--v:part3.swr_prevgoal_animals_positive-->2<!--/v-->/5, q <!--v:part3.swr_prevgoal_q-->0.16<!--/v-->).

### Prediction models — does content predict the next choice?

The behavioral-relevance layer, and the weakest link. `replay_fraction` over a
behavioral baseline that carries `candidate_arm_distance_rank` (conditional
logit, session medians over the sessions the per-term gate keeps): non-SWR
**<!--v:part3.choice_non_swr-->+0.040<!--/v--> (p <!--v:part3.choice_non_swr_p-->0.84<!--/v-->)**, SWR <!--v:part3.choice_swr-->+0.041<!--/v--> (p <!--v:part3.choice_swr_p-->0.66<!--/v-->) — both
**null**. Content does **not** predict stay/leave (null across strata). Raw
`replay_fraction` is *negative* (<!--v:part3.choice_raw-->−0.65<!--/v-->, <!--v:part3.choice_raw_animals_positive-->0<!--/v-->/5 animals positive — the current-arm
bias: the animal leaves the arm it most replays), and over the behavior baseline
it reads <!--v:part3.choice_full-->+0.054<!--/v--> (<!--v:part3.choice_full_animals_positive-->4<!--/v-->/5 animals positive, <!--v:part3.choice_full_significant_fraction-->7 %<!--/v--> of sessions
individually significant, p <!--v:part3.choice_full_p-->0.38<!--/v-->). The fits are **in-sample only — no held-out CV** —
and the SWR split is thin (n <!--v:part3.choice_full_sessions-->180<!--/v--> → <!--v:part3.choice_non_swr_sessions-->85<!--/v-->). So: no choice
correlation, not a cross-validated prediction. The fits have **no provenance
sidecar**, so every number in this paragraph is pending.

### Spatial FORM of SWR vs non-SWR content: wells vs paths, and trajectory start/end

*(Content-occupancy and trajectory rollups. Neither is a content-ladder fit and
neither was part of the 2026-09 refit wave.)*

**Correction to the "non-SWR is current-focused" framing.** The arm-model
`is_current_arm`
coefficient is a *within-event relative* over-weighting (conditional Poisson conditions on
each event's total, so event-constants cancel and coefficients are compositional, not
absolute). The **content-occupancy** analysis (`scripts/content/analyze_content_occupancy.py`,
<!--v:part2.occupancy_epochs-->242<!--/v--> sessions) gives the absolute picture: **non-SWR content is mostly at reward wells
(~<!--v:part2.non_swr_content_at_well_fraction-->58 %<!--/v--> at wells: ~<!--v:part2.non_swr_content_remote_well_fraction-->45 %<!--/v--> remote reward wells + ~<!--v:part2.non_swr_content_own_well_fraction-->13 %<!--/v--> own well; ~<!--v:part2.non_swr_content_remote_path_fraction-->41 %<!--/v--> remote path); SWR content is
path-dominated (~<!--v:part2.swr_content_remote_path_fraction-->73 %<!--/v--> remote path vs <!--v:part2.swr_content_at_well_fraction-->26 %<!--/v--> at-well)** — time-bin-weighted shares, and non-SWR
content is at a well more often than SWR content in **<!--v:part2.animals_non_swr_more_at_well-->5<!--/v-->/5 animals**. So non-SWR represents **reward-well *locations*** (current *and
remote*), SWR represents **paths**.

- **How "at well" vs "path" is defined** (note the caveat): per non-local *immobile timebin*,
  take the **MAP** decode; `d_well` = track-graph distance to the nearest reward well (well =
  the animal's median immobile position on each leaf arm); **`d_well < 20 cm` = "at a well",
  `≥ 20 cm` = "path"** (crossed with `d_animal`, 20 cm, for remote vs own). So **"path" means
  ">20 cm from any reward port" — it includes mid-leaf-arm, NOT only trunk/inter-patch arms**,
  and it is MAP-based (not posterior mass). The clean, artifact-immune part is `remote_path`
  (far from animal *and* wells).
- **Retraction:** `reward_belief_variance_candidate_arm_preoutcome` is **0 on all
  trunk arms** (a per-*leaf* quantity), so the
  `C13`/`Mu` uncertainty effects are estimated **only on well/leaf arms** — we have **no
  evidence** on whether SWR *path* content relates to uncertainty. The earlier "SWR extends
  uncertainty to the paths toward those wells" was unsupported; testing it needs a path/trunk
  uncertainty covariate (not built).

**Trajectory start/end IS characterized** (per-event rollups from `analysis/replay/trajectory.py`,
in `event_table`: `start_arm`/`end_arm`, `start_position`/`end_position`, `net_displacement`,
`spatial_extent`, `step_distance_*`, `max_ahead`/`max_behind`/`ab_slope`, `n_reversals`,
`ends_at_next`). Part 2c reports the SWR split of the per-event properties as
medians: spatial extent <!--v:part2.extent_median_non_swr-->8.0<!--/v--> cm (non-SWR) against <!--v:part2.extent_median_swr-->50<!--/v--> cm (SWR),
largest step <!--v:part2.max_step_median_non_swr-->2.0<!--/v--> against <!--v:part2.max_step_median_swr-->14<!--/v--> cm, reversals <!--v:part2.reversals_median_non_swr-->0<!--/v--> against
<!--v:part2.reversals_median_swr-->3<!--/v--> → **SWR content = extended, multi-reversal trajectories; non-SWR = localized.**

### Distinction from Gillespie et al. 2021 (Neuron)

Gillespie et al. 2021 (same lab, same clusterless decoder, predecessor spatial
task) is the key replication risk for the content layer. Much of our **spatial**
content ladder *recovers* their results (a pipeline validation, to be framed as
replicate + extend, not as new): current-arm bias (their Fig 8), immediate-past
suppression, the **future/choice null** (their Fig 5), and reward-history
dependence (their Fig 3/6).

**The previous-goal comparison can no longer be made.** Gillespie report
previous-goal *enrichment* (their Fig 8). Our reward-memory family is **refused
as unidentified** — it has no usable rung in the per-session arm ladder (see
[design/model_history.md](design/model_history.md)) — and the previous-goal
control rung `CS5b_valuevsprevgoal` fits on only **<!--v:part3.dm_swr_prevgoal_sessions-->152<!--/v--> sessions with <!--v:part3.dm_swr_prevgoal_separated-->104<!--/v--> flagged
separated** under the DM (<!--v:part3.dm_swr_prevgoal-->−0.12<!--/v-->, <!--v:part3.dm_swr_prevgoal_animals_positive-->1<!--/v-->/5, q <!--v:part3.dm_swr_prevgoal_q-->0.083<!--/v-->; the fraction fit has <!--v:part3.swr_prevgoal_sessions-->198<!--/v-->
sessions, <!--v:part3.swr_prevgoal-->−0.061<!--/v-->, <!--v:part3.swr_prevgoal_animals_positive-->2<!--/v-->/5, q <!--v:part3.swr_prevgoal_q-->0.16<!--/v-->). The earlier avoidance coefficients (−1.485 at
the arm grain, −1.62 SWR-split, −1.49 patch-grain) and the −0.79
SWR-by-previous-goal interaction are **withdrawn**. Neither a divergence nor a
replication can be claimed from this rung as it stands.

**The graded-value discriminant is retired as stated.** It rested on
`CS5b_valuevsprevgoal` leaving the SWR by arm-value interaction unmoved at
−2.45 → −2.43. Both numbers came from the retired count response and the rung is
unidentified, so the discriminant is withdrawn. What remains is the arm-value
result itself: **estimator-dependent** in its direct form and negative in its
total form (<!--v:part3.swr_armvalue_total-->−0.25<!--/v--> fraction / <!--v:part3.dm_swr_armvalue_total-->−0.46<!--/v--> DM; see the CS5 section above).
Gillespie's binary-goal task structurally cannot yield a graded-value
coefficient either way, so the *conceptual* novelty stands even though the
discriminating fit does not.

**Uncertainty-to-content vs their visit-recency bias (their Fig 7)** remains a
distinction of *estimand* — theirs is between-arm binned recency, ours is
within-arm — but the numeric within-arm horse-race backing it was a
count-response fit and is **PENDING** re-derivation.

The remaining structurally-novel axes (no Gillespie analog): the **SWR/non-SWR
dissociation** and the **rate/intensity models** (they modeled content only).

## The intersection — what's new here

**The covariate atlas.** Rate entries come from the per-bin ladder's
reconstructed reward kernel; content entries are per-event fraction coefficients
with their estimand named.

| driver | RATE / coupling | CONTENT (current arm) |
| --- | --- | --- |
| immediate reward (kernel lag 1) | non-SWR **<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->×** vs with-SWR **<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v-->×** (SWR-coupled far more); paired in <!--v:part3.rate_paired_positive_fraction-->77 %<!--/v--> of <!--v:part3.rate_paired_sessions-->240<!--/v--> sessions, <!--v:part3.rate_paired_animals_positive-->5<!--/v-->/5 animals, p <!--v:part3.rate_paired_p-->1.1e-19<!--/v--> | the current-arm interaction is **PENDING** (count-response fit withdrawn) |
| recent reward (kernel lags 3–10) | **shared**: lowest RR <!--v:part3.rate_kernel_late_min_low-->0.89<!--/v-->–<!--v:part3.rate_kernel_late_min_high-->0.93<!--/v--> across the four populations | arm reward fraction non-SWR **<!--v:part3.armreward_non_swr-->+0.027<!--/v--> (null)** vs SWR **<!--v:part3.armreward_swr-->−0.26<!--/v-->**; SWR contrast **<!--v:part3.swr_armreward_direct-->−1.0<!--/v-->** direct / **<!--v:part3.swr_armreward_total-->−1.2<!--/v-->** total |
| arm value | not in the shipped rate ladder (**pending**) | main effect not significant (<!--v:part3.armvalue_all-->+0.078<!--/v--> split); the direct SWR interaction is **estimator-dependent** and near zero (fraction <!--v:part3.est_swr_armvalue_fraction-->−0.014<!--/v--> / count <!--v:part3.est_swr_armvalue_count-->−0.17<!--/v--> / DM <!--v:part3.est_swr_armvalue_dm-->+0.046<!--/v-->), the total is **<!--v:part3.swr_armvalue_total-->−0.25<!--/v--> / <!--v:part3.dm_swr_armvalue_total-->−0.46<!--/v-->** |
| arm uncertainty | not in the shipped rate ladder (**pending**) | both populations over-represent uncertain arms, **SWR <!--v:part3.armuncert_swr-->+0.37<!--/v--> > non-SWR <!--v:part3.armuncert_non_swr-->+0.18<!--/v-->**; pooled SWR interaction null direct, **<!--v:part3.swr_armuncert_total-->+0.18<!--/v-->** total |
| exploration (patch-switch fraction) | **pending** | **locks content onto the current arm** in both: <!--v:part3.switchrate_current_all-->+0.18<!--/v--> pooled, <!--v:part3.switchrate_current_swr-->+0.33<!--/v--> SWR, <!--v:part3.switchrate_current_non_swr-->+0.23<!--/v--> non-SWR (<!--v:part3.switchrate_current_all_animals_positive-->5<!--/v-->/5, <!--v:part3.switchrate_current_swr_animals_positive-->5<!--/v-->/5, <!--v:part3.switchrate_current_non_swr_animals_positive-->5<!--/v-->/5) |
| within-stop time | rate rises over ~1.5–2 s then plateaus in every population | the current-arm interaction is **pending** |

**The headline, restated: one half solid, one half estimand-dependent.**

> Reward gates with-SWR **occurrence**; the content half of the old double
> dissociation is not currently quotable as a coefficient.

- **Occurrence (solid).** The immediate reward component dissociates — kernel
  lag 1 RR **<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v-->** (SWR-coupled non-local) vs **<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->** (non-SWR), paired in
  **<!--v:part3.rate_paired_positive_fraction-->77 %<!--/v--> of <!--v:part3.rate_paired_sessions-->240<!--/v--> sessions, <!--v:part3.rate_paired_animals_positive-->5<!--/v-->/5 animals, p <!--v:part3.rate_paired_p-->1.1e-19<!--/v-->** — while the slower
  recent-reward component is **shared** by all four populations. So non-SWR
  events are a **less consummatory/reward-coupled mode** that obeys the same
  slower recent-reward baseline. The non-SWR stream's immediate response is weak
  (its lag-1 kernel is close to 1) but its immediate coefficient is still positive.
- **Content (restated).** SWR content is less current-locked (**<!--v:part3.swr_current_direct-->−0.071<!--/v-->** direct /
  **<!--v:part3.swr_current_total-->−0.41<!--/v-->** total) and less prev-locked (**<!--v:part3.swr_prev_direct-->−0.42<!--/v-->**), avoids recently rewarded
  arms (**<!--v:part3.swr_armreward_direct-->−1.0<!--/v-->** / **<!--v:part3.swr_armreward_total-->−1.2<!--/v-->**), and is **broader within the event**
  (`log_phi_is_swr` **<!--v:part3.dm_breadth-->+1.7<!--/v-->** total, **<!--v:part3.dm_breadth_length_adjusted-->+1.4<!--/v-->** length-adjusted, <!--v:part3.dm_breadth_animals_positive-->5<!--/v-->/5 animals,
  <!--v:part3.dm_breadth_positive_fraction-->100 %<!--/v--> of sessions). Whether it also avoids high-**value** arms is
  estimator-dependent in the direct form and negative in the total form.
- **Shared.** Exploration concentrates content on the current arm in both
  populations; both over-represent uncertain arms.
- **Withdrawn.** "Reward pushes non-SWR content off the current arm" and its
  with-SWR null — both from the retired count response.

## Confirmed (robust: multi-control and/or 5/5 animals)

1. **Rate baseline ordering and the dominant structure.** Non-local (all)
   **<!--v:part3.rate_baseline_non_local-->0.43<!--/v-->** > detected SWR **<!--v:part3.rate_baseline_swr-->0.27<!--/v-->** > the two non-local streams, no-SWR **<!--v:part3.rate_baseline_non_local_no_swr-->0.22<!--/v-->** and
   w/ SWR **<!--v:part3.rate_baseline_non_local_swr-->0.21<!--/v-->** events/s. The within-stop time course is what calibrates the
   process (KS-in-band <!--v:part3.rate_ladder_ks_band_const-->0.45<!--/v--> → <!--v:part3.rate_ladder_ks_band_time-->0.93<!--/v--> at `M5_time`), with self-history clustering
   and a smooth reward kernel on top.
2. **The immediate-reward dissociation, from two distinct measurements.** The
   reconstructed kernel puts lag 1 at RR **<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v-->** (non-local w/ SWR) against
   **<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->** (non-local, no SWR) — *how large* the effect is. Separately, the
   **paired within-session** test of the immediate basis coefficient between the
   two streams is positive in **<!--v:part3.rate_paired_positive_fraction-->77 %<!--/v--> of <!--v:part3.rate_paired_sessions-->240<!--/v--> paired sessions, <!--v:part3.rate_paired_animals_positive-->5<!--/v-->/5 animals, signed-rank p = <!--v:part3.rate_paired_p-->1.1e-19<!--/v-->** — *how consistently* they differ. Do not read the
   percentage as a property of the kernel values or vice versa. The slower
   recent-reward component (lags 3–10, lowest RR <!--v:part3.rate_kernel_late_min_low-->0.89<!--/v-->–<!--v:part3.rate_kernel_late_min_high-->0.93<!--/v-->) is **shared** across all
   four populations.
3. **Events are strongly non-Poisson** — refractory then bursty, peak RR
   **<!--v:part3.rate_burst_peak_low-->1.56<!--/v-->–<!--v:part3.rate_burst_peak_high-->2.67<!--/v-->** by population, detected SWRs the most tightly
   bursty and non-SWR the flattest.
4. **Content current-arm bias.** `C2_base` `is_current_arm` **<!--v:part3.base_current_all-->+0.73<!--/v-->** (<!--v:part3.base_current_all_sessions-->242<!--/v-->
   sessions, <!--v:part3.base_current_all_animals_positive-->5<!--/v-->/5, q <!--v:part3.base_current_all_q-->1.9e-34<!--/v-->), present in both populations (non-SWR <!--v:part3.base_current_non_swr-->+0.53<!--/v--> /
   SWR <!--v:part3.base_current_swr-->+0.88<!--/v-->) and sign-consistent across all three estimators (count <!--v:part3.est_current_count-->+0.67<!--/v--> /
   fraction <!--v:part3.est_current_fraction-->+0.73<!--/v--> / DM <!--v:part3.est_current_dm-->+0.73<!--/v-->).
5. **SWR content is less current- and prev-locked, and broader within the
   event.** `swr_x_is_current_arm` **<!--v:part3.swr_current_direct-->−0.071<!--/v-->** direct / **<!--v:part3.swr_current_total-->−0.41<!--/v-->** total;
   `swr_x_is_prev_arm` **<!--v:part3.swr_prev_direct-->−0.42<!--/v-->**; `log_phi_is_swr` **<!--v:part3.dm_breadth-->+1.7<!--/v-->** (<!--v:part3.dm_breadth_animals_positive-->5<!--/v-->/5 animals,
   <!--v:part3.dm_breadth_positive_fraction-->100 %<!--/v--> of sessions), **<!--v:part3.dm_breadth_length_adjusted-->+1.4<!--/v-->** length-adjusted.
6. **Non-SWR content is neutral to *recently rewarded* arms; with-SWR content
   avoids them.** Measured on `reward_fraction_candidate_arm_last10_visits` —
   the realized reward fraction over the last ten visits to that arm — **not**
   on the belief mean `reward_belief_mean_candidate_arm_preoutcome`, whose main
   effect is positive (item 7). `C3_armreward` non-SWR **<!--v:part3.armreward_non_swr-->+0.027<!--/v-->** (<!--v:part3.armreward_non_swr_sessions-->239<!--/v-->
   sessions, <!--v:part3.armreward_non_swr_animals_positive-->3<!--/v-->/5, q <!--v:part3.armreward_non_swr_q-->0.35<!--/v-->) against SWR **<!--v:part3.armreward_swr-->−0.26<!--/v-->** (<!--v:part3.armreward_swr_sessions-->241<!--/v--> sessions, **<!--v:part3.armreward_swr_animals_positive-->0<!--/v-->/5 animals
   positive**, q <!--v:part3.armreward_swr_q-->3.3e-4<!--/v-->); the pooled SWR interaction is
   **<!--v:part3.swr_armreward_direct-->−1.0<!--/v-->** direct (<!--v:part3.swr_armreward_direct_sessions-->242<!--/v--> sessions, **<!--v:part3.swr_armreward_direct_animals_positive-->0<!--/v-->/5** animals positive) / **<!--v:part3.swr_armreward_total-->−1.2<!--/v-->**
   total and negative under all three estimators.
7. **Content over-represents high-uncertainty arms.** `Mu`
   `reward_belief_variance_candidate_arm_preoutcome_z` **<!--v:part3.leaf_uncertainty-->+0.31<!--/v-->** (**session
   median** over <!--v:part3.leaf_uncertainty_sessions-->242<!--/v--> sessions, <!--v:part3.leaf_uncertainty_animals_positive-->5<!--/v-->/5 animals positive); split `C13_armuncert` SWR **<!--v:part3.armuncert_swr-->+0.37<!--/v-->** vs non-SWR **<!--v:part3.armuncert_non_swr-->+0.18<!--/v-->**
   (<!--v:part3.armuncert_swr_animals_positive-->5<!--/v-->/5 and <!--v:part3.armuncert_non_swr_animals_positive-->5<!--/v-->/5). The rung earns its parameter (`M4 → Mu` ΔAIC **<!--v:part3.leaf_step_uncertainty-->+1.3<!--/v-->**, ΔAIC > 2
   in <!--v:part3.leaf_step_uncertainty_clear-->112<!--/v-->/<!--v:part3.leaf_step_uncertainty_sessions-->242<!--/v--> sessions, <!--v:part3.leaf_step_uncertainty_animals-->5<!--/v-->/5 animals child-better) where the value rung does not
   (`M4 → Mv` **<!--v:part3.leaf_step_value-->−2.4<!--/v-->**, ΔAIC > 2 in <!--v:part3.leaf_step_value_clear-->22<!--/v-->/<!--v:part3.leaf_step_value_sessions-->242<!--/v-->, <!--v:part3.leaf_step_value_animals-->0<!--/v-->/5 animals). **Value's main effect
   on this same leaf design is positive in the median but not across animals** — `CS5_armvalue`'s
   `reward_belief_mean_candidate_arm_preoutcome` **<!--v:part3.est_armvalue_fraction-->+0.26<!--/v-->** (<!--v:part3.est_armvalue_fraction_sessions-->242<!--/v--> sessions, <!--v:part3.est_armvalue_fraction_animals_positive-->3<!--/v-->/5,
   bootstrap [<!--v:part3.est_armvalue_fraction_low-->−0.26<!--/v-->, <!--v:part3.est_armvalue_fraction_high-->+0.61<!--/v-->]), `Mv` **<!--v:part3.leaf_value-->+0.24<!--/v-->** (<!--v:part3.leaf_value_sessions-->242<!--/v--> sessions, <!--v:part3.leaf_value_animals_positive-->3<!--/v-->/5) — so the
   two axes are separated by model comparison.
8. **Exploration concentrates content on the current arm** in both populations
   (`C12_switchrate` **<!--v:part3.switchrate_current_all-->+0.18<!--/v-->** pooled, **<!--v:part3.switchrate_current_swr-->+0.33<!--/v-->** SWR, **<!--v:part3.switchrate_current_non_swr-->+0.23<!--/v-->** non-SWR,
   <!--v:part3.switchrate_current_all_animals_positive-->5<!--/v-->/5, <!--v:part3.switchrate_current_swr_animals_positive-->5<!--/v-->/5 and <!--v:part3.switchrate_current_non_swr_animals_positive-->5<!--/v-->/5 animals).

## Hypotheses (suggestive; need a further test)

- **Continuum, not two mechanisms.** Unimodal coupling distribution + graded SWR
  broadening of content. *Needs:* peri-onset **peak** distribution; the joint MPP.
- **Reward shifts SWR-coupling continuously** (binary dissociation = thresholded
  shadow). *Needs:* the joint occurrence+mark model.
- **Value-gating of with-SWR rate.** The covariates that carried this are not in
  the shipped rate ladder and their coefficients are **pending** re-derivation
  (Layer 1); the hypothesis is retained without a number.

## Unknown / pending — including the models we DON'T have

- **Clean exploit-vs-explore content model — the within-arm frame, not yet refit.**
  The design conclusion stands: within-arm (arm-demeaned, non-current-leaf) beta
  value `reward_belief_mean_candidate_arm_preoutcome` and beta uncertainty
  `reward_belief_variance_candidate_arm_preoutcome` are separately identifiable,
  and no HMM value is needed (HMM retired). The **numbers** — the within-arm
  horse-race coefficients and the `Mvu` conditioning diagnostics — were
  count-response fits and are **PENDING** re-derivation on the per-event fraction
  response before any of them can be quoted.
- **Full marked point process** (joint intensity + coupling + content; shared
  latent; power↔history independence). **Not built.** The three layers are treated as
  separable factors but never jointly fit.
- **Need / successor (prospective) content — structurally UN-buildable here.** The task
  cannot identify it: alternation makes `is_next_arm`/`is_prev_arm` collinear. No
  covariate separates prospective from retrospective on this track.
- **Graded ripple-power content model — NOT a committed rung.** A graded
  `ripple_power × current` term lives only in an exploratory script, not the ladder.
- **Content choice-mediation** (does rate × content predict the next choice). The choice correlation (#19d) is null over the behavioural baseline; stay/leave is
  null (#19e).
- **Per-arm realized-gain: tested under the retired HMM covariate and the
  retired count response, and negative.** Its coefficient is **withdrawn**; the
  reading it supported — that the gain axis of content is epistemic uncertainty
  rather than realized value-update — survives only as a hypothesis. The
  beta per-arm value-update re-run has not been done.
- **Pooled `swr_x` contrasts on arm-identity-tied covariates: read the split as
  well.** The earlier fixed-effect-artifact diagnosis of the pooled uncertainty
  interaction is superseded — on the leaf-only fraction fit the *direct*
  interaction is simply a null (<!--v:part3.swr_armuncert_direct-->−0.058<!--/v-->, <!--v:part3.swr_armuncert_direct_animals_positive-->2<!--/v-->/5) while the *total* twin is positive
  (<!--v:part3.swr_armuncert_total-->+0.18<!--/v-->, <!--v:part3.swr_armuncert_total_animals_positive-->5<!--/v-->/5). Report the split (`C13_armuncert`) alongside the pooled pair,
  and never a pooled contrast without its estimand.
- **Other content responses and held-out CV for the content hierarchy.** Three
  responses have now been compared on one cohort — retired count, shipped
  per-event fraction, and Dirichlet-multinomial
  ([design/content_estimator_comparison.md](design/content_estimator_comparison.md)) —
  but the arm hierarchy still has **no held-out cross-validation**; model
  comparison there rests on ΔAIC on the quasi-likelihood scale.

## Caveats & confounds (and how addressed)

- **Occupancy** (immobility happens at wells → current arm = reward well): the
  reward content effect uses the **within-well outcome contrast** (rewarded vs
  omitted at the same well, identifiable via the probabilistic schedule), which
  isolates the reward *outcome* from the well's location.
- **Dwell / within-stop time** (rewarded stops are <!--v:part3.stop_duration_mean_ratio-->3.5<!--/v-->× longer on the mean, <!--v:part3.stop_duration_mean_rewarded-->1.8<!--/v--> s
  vs <!--v:part3.stop_duration_mean_unrewarded-->0.52<!--/v--> s): `BASEc` carries
  `log1p_time_since_stop_start_s_z_x_is_current_arm`. Control is
  linear in `z(log1p t)`; a spline would be stricter.
- **Event length is post-treatment, not a nuisance.** The `BASEc` length products
  make a `swr_x_...` coefficient a **direct** effect; the `nt` twin gives the
  **total**. Neither is "the" answer — see the estimand bullet in *Provenance*.
- **Collinearity:** the trial-level reward-rate covariate (the mean of the reward
  lags, VIF infinite) was removed; the reward-history block is now a smooth
  three-feature basis whose coefficients are collinear **by construction**, which
  is why the kernel is reconstructed rather than read term by term.
- **Separation:** the whole last-rewarded family is **refused as unidentified**
  in the per-session fits, and `CS5b_valuevsprevgoal` fits on <!--v:part3.dm_swr_prevgoal_sessions-->152<!--/v--> sessions with
  <!--v:part3.dm_swr_prevgoal_separated-->104<!--/v--> flagged separated. Per-coefficient separation flags and a
  `|coef| x SD >= 10` magnitude backstop now decide which session fits enter an
  aggregate; a rung's excluded share must be stated with its coefficient.
- **Circularity** (content from the same decoder that defines events): the
  242-epoch spike-evidence run (Part 2b) puts close non-SWR events above the
  matched non-event baseline in <!--v:part2.non_swr_close_animals_above_baseline-->5<!--/v-->/5 animals, so the current-arm content is
  spike-supported. The close-event W3 (place-field-extent) worry needs the
  local-kernel re-decode regardless.

## Reproducibility (scripts & key commits)

- Rate: `scripts/rate/fit_per_bin_rate_models.py`,
  `scripts/rate/fit_per_stop_rate_models.py`, `scripts/rate/fit_uncertainty_rate.py`,
  `scripts/rate/fit_swr_contrast_rate.py`; summaries
  `scripts/rate/summarize_per_bin_rate_models.py`, plots
  `scripts/rate/plot_per_bin_rate_models.py`.
- Coupling: `scripts/coupling/fit_swr_power_model.py`.
- Content: `scripts/content/fit_replay_content_models.py`,
  `scripts/content/fit_swr_contrast_content.py`,
  `scripts/content/compare_content_estimators.py`;
  audits by `scripts/qa/compare_fits.py`.
- Contingency: `scripts/behavior/fetch_spatial_bandit_behavior.sh`,
  `scripts/_bandit_csv_contingency.py` (removed in d56e265; the contingency columns now come
  from `qvals_table`, built by `scripts/behavior/build_qvals_table.py`).
- Audits behind the current numbers:
  [2026-09-06-rate-refit-arrival-clock.md](../.claude/docs/reviews/2026-09-06-rate-refit-arrival-clock.md),
  [2026-09-05-content-refit-comparison.md](../.claude/docs/reviews/2026-09-05-content-refit-comparison.md),
  [2026-09-05-cs5-response-by-design-factorial.md](../.claude/docs/reviews/2026-09-05-cs5-response-by-design-factorial.md),
  [design/content_estimator_comparison.md](design/content_estimator_comparison.md).
