# Replay event-rate models: two families

> **Fitted Family-B effects** (baseline rates, reward kernel, self-history
> microstructure, ladder fit) are reported values in `paper/reported/behavior.json`,
> listed under question 3.3 in
> [../paper/reported/questions.md](../paper/reported/questions.md).

Two model families estimate **how often** non-local / SWR events occur (their
*rate*), as opposed to **what they represent** (their *content*). They answer
the same scientific question — does reward, recent reward history, or the
within-stop time course modulate replay occurrence, and does that modulation
differ between SWR-coupled and non-SWR non-local events — at two different
levels of confound control. Use **Family A** for the literature comparison and
the positive control; use **Family B** for the confound-free dissociation and
the point-process model-checking.

Both are fit on four outcome populations (tagged `outcome`): detected SWRs
(`swr`), all non-local (`non_local`), and the non-local SWR / no-SWR split
(`non_local_swr`, `non_local_no_swr`). The pooled `non_local` is the
one-population null tested against its split.

> **Cohort (2026-09-12, binding).** Both families are fit on the **well-anchored
> stop cohort** — `occupancy_state ∈ {at_well, mixed}`, **<!--v:behavior.stops_well_anchored-->93,560<!--/v-->** of the
> <!--v:behavior.stops_total-->120,320<!--/v--> canonical stops — which is the same population the content family is
> fit on at the event grain. The rule lives in one function and is selected with
> `--stop-cohort` on every rate runner
> ([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md)).
> The per-bin all-stops fit is kept as a **sensitivity bundle** (`_allstops`;
> the per-stop all-stops fit has no current output) and in-transit stops as their
> own (`_intransit`); an in-transit population is never
> pooled with a well-anchored one. **The in-transit bundle is thin** — only <!--v:behavior.rate_intransit_sessions-->52<!--/v--> of
> <!--v:behavior.rate_sessions-->242<!--/v--> sessions clear the per-bin sample gates on any population, and only <!--v:behavior.rate_intransit_sessions_non_local_swr-->10<!--/v--> on
> the SWR-coupled non-local stream — so an in-transit cell showing no effect is
> an absence of evidence, not evidence of absence, and its paired tests have no
> power. Every number below is from the well-anchored
> fit unless it is explicitly labelled all-stops. Every number below is read from `paper/reported/behavior.json`. The cohort comparison is
> [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md).

## Family A — per-stop rate (the literature-comparable layer)

- **Spec:** `MODEL_PER_STOP_RATE` (`R0` null, `R1` reward-history
  raised-cosine basis `reward_history_rcos_0..2` + `reward_history_incomplete`)
  in
  [`model_specs.py`](../src/continuum_swr_replay/analysis/behavior/model_specs.py).

> **Timing definition:** the basis is built from the outcomes physically known
> at bout start. The newest known outcome names the arrived trip after that
> trip's feedback boundary and the preceding outcome before it, so the kernel is
> past-only but is not indexed by a fixed "current" or "previous" trial.
> Each feature is the bump-weighted reward *fraction* over the lags its bump
> covers, so it lies in [0, 1] and `exp(coef)` is per unit of that fraction — a
> 0→1 contrast, not a single-trial indicator. `reward_history_rcos_0` weights
> the newest outcomes (lags 1–7, weights 1.00, 0.79, 0.54, …) and is the
> *immediate* component; `_1` spans all ten lags with its peak at lag 3; `_2`
> spans lags 2–10 and rises to its maximum at lag 10. Interpret them as one
> outcome-history kernel closed at bout start — read the per-lag effect off the
> reconstruction (`reward_kernel_basis.png`, and the `rate_kernel_*` values under question 3.3 in
  [../paper/reported/questions.md](../paper/reported/questions.md)),
> not off a single coefficient — and do not label the immediate component as
> uniformly consummatory or previous-trial reward.
> `reward_history_incomplete` is the **structural-missingness nuisance
> indicator**: 1 on the stops at the start of an epoch where fewer than two
> outcomes are yet known (the three features are then zero-filled), 0 elsewhere.
> The flagged stops form an unbroken prefix of every epoch: a share of
> <!--v:behavior.stops_reward_history_incomplete_fraction-->0.013<!--/v--> of stops and <!--v:behavior.onsets_reward_history_incomplete_fraction_non_local-->0.015<!--/v--> of non-local onsets.
> It is carried as a plain main effect wherever the basis is fit — never inside
> an `is_swr` expansion, never aggregated, never reported — and it is excluded
> from the separation judgment (`NUISANCE_TERMS`, below).
- **Cohort:** the well-anchored stops (`--stop-cohort at_well`, the default),
  then one frozen cohort per session for the whole ladder.
  `freeze_common_cohort` intersects the complete rows over the union of every
  `MODEL_PER_STOP_RATE` predictor with a positive `immobility_duration_s`
  offset, and `fit_per_stop_rate_model` then **raises** on any residual NaN
  instead of subsetting. `R0` and `R1` are therefore scored on identical stops,
  which is what makes their ΔAIC a model comparison rather than a cohort
  comparison. On the shipped fit that cohort costs nothing beyond the occupancy
  rule — **<!--v:behavior.per_stop_audit_stops_in-->93,560<!--/v-->** well-anchored stops in, <!--v:behavior.per_stop_audit_stops_own_drop-->5<!--/v--> dropped for a missing own
  predictor, `n_drop_common_cohort_other` <!--v:behavior.per_stop_audit_stops_common_drop-->0<!--/v-->, **<!--v:behavior.per_stop_audit_stops_modeled-->93,555<!--/v-->** modelled on `R0` and
  <!--v:behavior.per_stop_audit_stops_full-->93,215<!--/v--> on `R1` (<!--v:behavior.per_stop_audit_stops_full_session_excluded-->340<!--/v--> stops in gated-out sessions)
  (`models_per_stop_rate/all_models_cohort_audit.csv`).
- **Estimator:** one ordinary Poisson GLM per stop on the event *count*, with a
  `log(immobility_time)` offset — `exp(coef)` is the multiplicative change in
  events per **second of immobility**
  ([`per_stop_rate_model.py`](../src/continuum_swr_replay/analysis/behavior/rate/per_stop_rate_model.py)).
- **Runner / output:** `scripts/behavior/rate/fit_per_stop_rate_models.py` →
  `Processed-Data/replay_content/models_per_stop_rate/all_models.pkl`.
- **Cite it for:** the Ambrose 2016 / Sosa 2020 reward-rate comparison (same
  Poisson-events-per-second estimator) and the **detected-SWR positive
  control**. The reward history enters as one four-parameter block, so the
  citable quantity is the block's step, the median per-session `R1 − R0` ΔAIC.
  On the **well-anchored** cohort — the population the content family is also
  fit on — that step is **<!--v:behavior.per_stop_delta_aic_animal_non_local-->−4.3<!--/v-->** (non-local), **<!--v:behavior.per_stop_delta_aic_animal_non_local_swr-->−2.6<!--/v-->** (non-local w/ SWR),
  **<!--v:behavior.per_stop_delta_aic_animal_swr-->−2.7<!--/v-->** (detected SWR) and **<!--v:behavior.per_stop_delta_aic_animal_non_local_no_swr-->+0.88<!--/v-->** (non-local, no SWR) animal-balanced,
  and **<!--v:behavior.per_stop_delta_aic_session_non_local-->−4.2<!--/v-->** / **<!--v:behavior.per_stop_delta_aic_session_non_local_swr-->−3.1<!--/v-->** / **<!--v:behavior.per_stop_delta_aic_session_swr-->−4.3<!--/v-->** / **<!--v:behavior.per_stop_delta_aic_session_non_local_no_swr-->+0.88<!--/v-->** session-pooled, over <!--v:behavior.per_stop_sessions_non_local-->241<!--/v-->
  sessions each
  ([2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md),
  "Family A — per-stop reward step"). The block earns its parameters in three
  populations but not in the non-SWR one: **<!--v:behavior.per_stop_child_better_fraction_non_local-->74 %<!--/v-->** of sessions are
  child-better on the pooled non-local stream (<!--v:behavior.per_stop_animals_child_better_non_local-->5<!--/v-->/5 animals) against **<!--v:behavior.per_stop_child_better_fraction_non_local_no_swr-->46 %<!--/v-->**
  on the non-SWR one (<!--v:behavior.per_stop_animals_child_better_non_local_no_swr-->2<!--/v-->/5 animals), whose median step is on the wrong side of zero.
  The confound-free statement is Family B's either way. The retired outcome-flag
  rung's per-stop values are no longer Family-A results and are recorded only as
  retracted in the retired claims table (`paper/claims_table.md`, in git history).

This is the simple layer: one rate per stop, dwell handled as an exposure
offset. Its limitation is exactly that offset — see below.

## Family B — per-bin conditional intensity (the confound-free layer)

- **Spec:** `MODEL_PER_BIN_RATE`, a **point-process GLM
  conditional-intensity ladder** (Truccolo–Eden–Brown 2005) `M0_const … M9r`,
  plus an arrival-anchored sub-ladder (`M0a`/`M5a`/`M7ra`), in
  [`model_specs.py`](../src/continuum_swr_replay/analysis/behavior/model_specs.py).
- **Estimator:** one Poisson GLM per session on the **(stop × 15 ms bin)**
  design from `aggregate_to_timebins`, with a `log(exposure)` offset; the
  within-stop time course is modelled explicitly by a `bs(tc, df=4)` spline
  rather than averaged into a per-stop dwell offset
  ([`per_bin_rate_model.py`](../src/continuum_swr_replay/analysis/behavior/rate/per_bin_rate_model.py)).
- **At-risk bins (2026-11-03).** Only bins in which an event can start enter
  the fit: a bin inside an ongoing event of the outcome's detector is dropped
  (all Kay SWRs for detected SWR; all non-local events, with or without an
  SWR, for every non-local population), because one detector's events never
  overlap. This is the at-risk rule of counting-process models for recurrent
  episodes (Andersen and Gill 1982; Cook and Lawless 2007, §1.4.2 and §6.5).
  **Estimand:** every Family B rate and rate ratio is the rate at which events
  *start* while none is ongoing. Family A's is events per second of stop, which
  also moves when events get longer. The self-history control is counted from
  the previous event's **end** (`offhist_rcos_0..3`); counted from onsets it has
  almost no at-risk bins at short lags and cannot be estimated. On the earlier
  all-bins design the onset block's short-lag coefficient ran to −∞ (the bin
  after an onset never holds one), which is why many of those fits
  were flagged not converged; their reward kernels agree with the at-risk fits
  session by session.
- **The ladder** toggles three blocks independently, all at one bin width
  (`BIN_S = 0.015`) so the models are nested and AIC / CV / GOF are comparable:

  | model | adds |
  | --- | --- |
  | `M0_const` | homogeneous Poisson baseline (`count ~ 1`) |
  | `M1_rew` | reward-history basis (`reward_history_rcos_0..2` + `reward_history_incomplete`) |
  | `M5_time` | within-stop time course (spline) |
  | `M3r_histcos` | self-history (raised-cosine `offhist_rcos_0..3` over the 150 ms after the previous event's end) |
  | `M6r_time_histcos` | time + self-history (the reward block's nested null) |
  | `M7r_time_histcos_rew` | time + self-history + reward history (**the full additive model**) |
  | `M9r_rew_x_time` | + `reward_history_rcos_0 × bs(tc)` interaction (does reward modulate the time course?) |

  The table is in **spec order** (`MODEL_PER_BIN_RATE`), which is the order the
  summary table and the ladder figures use. `M6r_time_histcos` exists so the
  reward block has a null that already holds time AND self-history: the
  conditional reward step is `M7r − M6r`, and `M1_rew − M0_const` is its
  marginal counterpart.

  The reward block is everywhere the same four columns — the three basis
  features plus the `reward_history_incomplete` indicator described under
  Family A — so "adds reward history" always means one four-parameter block,
  never a per-lag dummy set. The reward×time interaction (`M9r`) is **built and
  fit**, not deferred.

  **Retired 2026-09-06: the five `onset_lag1..10` dummy rungs** (`M3_hist`,
  `M4_hist_rew`, `M6_time_hist`, `M7_time_hist_rew`, `M9_rew_x_time`). Ten
  collinear lag dummies separate on the refractory bins, which voids the whole
  rung's held-out metric, so those rungs could not enter the ladder's
  cross-validation at all. The raised-cosine rungs span the same window with
  four smooth parameters and are scored. Rationale and the consequence for the
  per-lag self-history readout: [design/model_history.md](design/model_history.md).
- **Cohort:** the well-anchored stops (`--stop-cohort at_well`, the default),
  then `freeze_per_bin_cohort` freezes one cohort per (session,
  population) over the union of every rung's design columns, keeping only bins
  complete for that union and with `exposure > 0`; `fit_per_bin_rate_model` and
  its cross-validation then **raise** on a NaN or a non-positive exposure rather
  than subsetting. Every rung of a session therefore reports the same `n_bins`
  and `n_stops`, and `compare_nested_models` NaNs any comparison whose two rungs
  disagree on `n_events` / `n_bins` / `n_stops`. On the shipped fit the union
  costs <!--v:behavior.rate_audit_stops_common_drop-->575<!--/v--> of the <!--v:behavior.rate_audit_stops_in-->93,560<!--/v--> well-anchored stops (plus <!--v:behavior.rate_audit_stops_own_drop-->5<!--/v--> with a missing own
  predictor and <!--v:behavior.rate_audit_stops_modeled_session_excluded-->38<!--/v--> in gated-out sessions), leaving **<!--v:behavior.rate_stops-->92,942<!--/v--> stops** in every rung of every population, over
  <!--v:behavior.rate_bins-->7,280,203<!--/v--> at-risk bins for the non-local populations and <!--v:behavior.rate_bins_swr-->7,359,637<!--/v--> for detected SWR
  (each detector's own events set its at-risk bins) (`models_per_bin_rate/per_bin_rate_cohort_audit.csv`;
  [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md),
  "Cohort size"). Those are the well-anchored stops carrying no
  arrival clock, which the arrival rungs put into the union. The freeze bites harder where
  rungs need different covariates — the `is_swr`-contrast ladder, whose audit
  reports the largest `n_drop_common_cohort_other`. There and in
  `MU_uncertainty` the value covariates' missing tail is charged to
  `n_drop_qvals_gap`, not to the union.

  The corpus is **<!--v:behavior.rate_sessions-->242<!--/v--> sessions**, and a session may still carry no fit for a
  population that has too few events in it. On the well-anchored fit the full
  additive rung `M7r_time_histcos_rew` is fit on **<!--v:behavior.rate_full_sessions_swr-->242<!--/v--> / <!--v:behavior.rate_full_sessions_non_local-->242<!--/v--> / <!--v:behavior.rate_full_sessions_non_local_swr-->242<!--/v--> / <!--v:behavior.rate_full_sessions_non_local_no_swr-->242<!--/v-->**
  sessions (detected SWR / non-local, all / non-local w/ SWR / non-local, no
  SWR) and `M5_time` on <!--v:behavior.rate_time_sessions_swr-->242<!--/v--> / <!--v:behavior.rate_time_sessions_non_local-->242<!--/v--> / <!--v:behavior.rate_time_sessions_non_local_swr-->242<!--/v--> / <!--v:behavior.rate_time_sessions_non_local_no_swr-->242<!--/v--> (fitted rows of
  `models_per_bin_rate/all_models.pkl`; the reward-quantity `n` per population is
  in
  [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md)).
  Quote the per-rung, per-population `n`, not a flat <!--v:behavior.rate_sessions-->242<!--/v-->.
- **Runner / output:** `scripts/behavior/rate/fit_per_bin_rate_models.py`
  (`--workers N` fans sessions over processes) →
  `Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`. Plot with
  `scripts/behavior/rate/plot_per_bin_rate_models.py`
  (→ `reward_kernel_basis.png`, `selfhistory_autocorr.png`) and
  `scripts/behavior/rate/plot_per_bin_rate_summary.py` (→ `per_bin_rate_ladder_summary.png`).
  Both draw the summary's statistics over the summary's sessions: the reward
  kernel and the single basis coefficients over the sessions the per-term gate
  keeps, the fitted time course of `M5_time` between its interior knots
  (`rate_fit_statistics.fitted_time_course`), AIC steps differenced within
  session, and the time courses' support over the well-anchored stops. The
  descriptive PSTH `scripts/exploratory/_timeresolved_psth.py`
  (→ `rate_timecourse_vs_pump.png`) reads the onset/dense tables directly.

  The four rate fitters — `scripts/behavior/rate/fit_per_bin_rate_models.py`,
  `scripts/behavior/rate/fit_per_stop_rate_models.py`, `scripts/behavior/rate/fit_swr_contrast_rate.py`
  and `scripts/behavior/rate/fit_uncertainty_rate.py` — plus the audited `is_swr`-contrast
  aggregation each write an `<artifact>.provenance.json`
  sidecar beside the artifact recording the git hash, the estimator, the command line, and
  the SHA-256 / mtime / covariate-schema version of each input table. Readers
  re-hash those inputs and **refuse a stale fit** — one whose input tables have
  changed since it was written — unless `--allow-stale` is passed, so a plot or
  a summary cannot silently describe a fit of an older table. Re-deriving the
  `is_swr`-contrast aggregate after an aggregation rule changes does not need a
  refit: `scripts/behavior/rate/fit_swr_contrast_rate.py --aggregate-only` rebuilds the
  audited table and its term diagnostics from the persisted per-session fits.
- **Cite it for:** the **dwell-, time-, reward-history- and clustering-controlled
  dissociation**. In `M7r_time_histcos_rew`, the reconstructed reward kernel at
  lag 1 dissociates — `non_local_swr` ≈ <!--v:behavior.rate_kernel_lag_one_non_local_swr-->1.44<!--/v-->, `swr` ≈ <!--v:behavior.rate_kernel_lag_one_swr-->1.39<!--/v-->, `non_local` ≈ <!--v:behavior.rate_kernel_lag_one_non_local-->1.18<!--/v-->,
  `non_local_no_swr` ≈ <!--v:behavior.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v--> — while the
  slower **recent-reward suppression** from lag 3 on (lowest per-population RR over lags 3–10: <!--v:behavior.rate_kernel_late_min_low-->0.89<!--/v-->–<!--v:behavior.rate_kernel_late_min_high-->0.93<!--/v-->) is
  **shared** across populations and returns to ~1 by lag 10. Separately, the
  lag-1 kernel is higher for `non_local_swr` than for `non_local_no_swr`
  **within session in <!--v:behavior.rate_paired_positive_fraction-->78 %<!--/v--> of <!--v:behavior.rate_paired_sessions-->197<!--/v--> paired sessions, <!--v:behavior.rate_paired_animals_positive-->5<!--/v-->/5
  animals, signed-rank p = <!--v:behavior.rate_paired_p-->3.2e-18<!--/v-->** (sessions treated as independent). Every
  kernel value is read over the sessions whose three reward terms pass the
  per-term separation gate. Those two are different measurements:
  the first is the size of the kernel at a lag, the second is the consistency of
  a between-population difference — quote them as separate clauses, never
  spliced. Both are properties of the *fitted kernel*, which is reconstructed
  from the three basis coefficients per session and drawn in
  `paper/figures/rate/reward_kernel_basis.png`; the smooth basis has no per-lag
  parameter, so read shape off that reconstruction rather than off any one
  coefficient. Its shape: a lag-1 rise that dissociates by SWR-coupling, then a
  shallow dip below 1 from lag 3 that is shared across populations and closes by
  lag 10. The lag≥2 suppression is a recent-reward effect; it is
  *consistent with* an RPE / relative-reward account (Kleinman & Foster 2024;
  Ambrose adaptive coding) but this GLM does not measure a prediction error, so
  the mechanism (RPE vs satiety vs reward-rate adaptation) is not established
  here. The self-history block is a control for the
  reward kernel, counted from event ends; its 150 ms window is too short and a
  single session too sparse to read history from it. On the pooled
  duration-aware fits (question 3.3) there is **no refractory period beyond the event itself** (15 ms after an event ends the rate ratio is <!--v:behavior.history_end_first_lag_rr_low-->1.68<!--/v-->–<!--v:behavior.history_end_first_lag_rr_high-->4.53<!--/v-->×), events **cluster right after the previous event ends** (<!--v:behavior.history_end_peak_low-->2.63<!--/v-->–<!--v:behavior.history_end_peak_high-->4.53<!--/v-->× at <!--v:behavior.history_end_peak_lag_ms_low-->15<!--/v-->–<!--v:behavior.history_end_peak_lag_ms_high-->45<!--/v--> ms after the end), and a **weaker onset-locked rhythm** remains (an animal counts when its peak's interval, widened for the lags searched, stays above 1): detected SWR <!--v:behavior.history_onset_peak_swr-->2.28<!--/v-->× at <!--v:behavior.history_onset_peak_lag_ms_swr-->90<!--/v--> ms (<!--v:behavior.history_onset_animals_swr-->5<!--/v-->/5 animals), non-local with SWR <!--v:behavior.history_onset_peak_non_local_swr-->1.83<!--/v-->× at <!--v:behavior.history_onset_peak_lag_ms_non_local_swr-->105<!--/v--> ms (<!--v:behavior.history_onset_animals_non_local_swr-->4<!--/v-->/5), non-local without SWR <!--v:behavior.history_onset_peak_non_local_no_swr-->1.34<!--/v-->× at <!--v:behavior.history_onset_peak_lag_ms_non_local_no_swr-->150<!--/v--> ms (<!--v:behavior.history_onset_animals_non_local_no_swr-->2<!--/v-->/5).

  **The reward effect is `rr_lag1`, not `rr_immediate`.** `rr_lag1` is the
  reconstructed kernel evaluated **at lag 1** — the blend of the three bumps that
  a reward one outcome back actually produces, and the number in the ordering
  above. `rr_immediate` is `exp` of the single bump-0 coefficient: the three
  coefficients are strongly collinear, and a 0 → 1 change in the bump-0 feature with the other two held
  fixed is no reward history that occurs, so it is not the effect of a reward at
  any lag and is not quoted as one (the basis coefficients are listed under
  question 3.3 in [../paper/reported/questions.md](../paper/reported/questions.md) for completeness). Say whether a quoted `rr_lag1` is the
  animal-balanced or the session-pooled summary — it is published
  both ways (session-pooled **<!--v:behavior.rate_kernel_lag_one_swr-->1.39<!--/v--> / <!--v:behavior.rate_kernel_lag_one_non_local-->1.18<!--/v--> / <!--v:behavior.rate_kernel_lag_one_non_local_swr-->1.44<!--/v--> / <!--v:behavior.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->**, animal-balanced
  **<!--v:behavior.rate_kernel_lag_one_animal_swr-->1.41<!--/v--> / <!--v:behavior.rate_kernel_lag_one_animal_non_local-->1.22<!--/v--> / <!--v:behavior.rate_kernel_lag_one_animal_non_local_swr-->1.47<!--/v--> / <!--v:behavior.rate_kernel_lag_one_animal_non_local_no_swr-->1.00<!--/v-->**, in the same row order).

  (Everything in this bullet is the well-anchored fit over <!--v:behavior.rate_sessions-->242<!--/v--> sessions — see
  the `rate_kernel_*` values under question 3.3 in
  [../paper/reported/questions.md](../paper/reported/questions.md) for the full kernel and
  [2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md)
  for what the cohort moved. The smooth reward basis has no per-lag parameter, so the
  lag-1 boost is smaller than the retired per-lag specification reported; the
  ordering and the dissociation are unchanged. The audited SWR-contrast
  ladder publishes **<!--v:behavior.rate_contrast_audited_rows-->32<!--/v--> aggregate rows** under the per-term separation rule —
  `G0`–`G7` across the three stratified populations plus `S1` and `S3`–`S9` on
  the contrast, split over the primary and post-outcome cohorts (6 primary + 2
  post-outcome rows per population). Under the retired fit-level flag it
  published none.)

### Stops are immobility fragments, not well visits

A "stop" is an immobility bout: the interval between two crossings of the
`head_speed < 4 cm/s` threshold (`bouts.py`). A single well visit is routinely
cut into several such bouts by small movements at the port, so **the modelled
unit is a fragment of a visit, not the visit**.

**Which fragments are modelled.** Both families fit the **well-anchored**
fragments: `occupancy_state` `at_well` **or** `mixed`, <!--v:behavior.stops_well_anchored-->93,560<!--/v--> of the <!--v:behavior.stops_total-->120,320<!--/v-->
canonical stops. A `mixed` bout is one whose first settling samples still read as
in-transit — it begins after the last well arrival and sits at the well — so
including it is what makes the rate cohort *contain* the content cohort instead
of crossing it. The <!--v:behavior.stops_in_transit-->25,768<!--/v--> in-transit bouts are a separate bundle and are
never pooled in ([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md)).


Consequence, and it is a scope condition on every Family-B number: `tc`, the 4 s
truncation, the `log(immobility_duration_s)` offset and
`event_table.time_since_stop_start_s` all measure time since a **speed-threshold
crossing**, not since arrival, since poke-in, or since reward. Events can sit in a fragment that opened well after the animal reached the well, so
"the rate is elevated early in the stop" is not by itself evidence of a
consummatory or arrival-locked process. The arrival-clock rungs below are the
direct test of that alternative.

### The arrival-clock variant of the time course

> Coverage: `M0a_const_arrival` fits <!--v:behavior.rate_arrival_const_sessions_swr-->242<!--/v--> / <!--v:behavior.rate_arrival_const_sessions_non_local-->242<!--/v--> / <!--v:behavior.rate_arrival_const_sessions_non_local_swr-->242<!--/v--> / <!--v:behavior.rate_arrival_const_sessions_non_local_no_swr-->242<!--/v--> sessions, `M5a_time_arrival` <!--v:behavior.rate_arrival_time_sessions_swr-->235<!--/v--> / <!--v:behavior.rate_arrival_time_sessions_non_local-->239<!--/v--> / <!--v:behavior.rate_arrival_time_sessions_non_local_swr-->222<!--/v--> / <!--v:behavior.rate_arrival_time_sessions_non_local_no_swr-->234<!--/v--> and
> `M7ra_time_arrival_histcos_rew` <!--v:behavior.rate_arrival_full_sessions_swr-->237<!--/v--> / <!--v:behavior.rate_arrival_full_sessions_non_local-->236<!--/v--> / <!--v:behavior.rate_arrival_full_sessions_non_local_swr-->221<!--/v--> / <!--v:behavior.rate_arrival_full_sessions_non_local_no_swr-->233<!--/v--> (detected SWR /
> non-local, all / non-local w/ SWR / non-local, no SWR; fits with a finite AIC
> in `models_per_bin_rate/all_models.pkl`), so the arrival
> sub-ladder's pooled comparisons are over the smaller of each pair, per
> population.

`M0a_const_arrival`, `M5a_time_arrival` and `M7ra_time_arrival_histcos_rew`
re-anchor the same spline block on `time_since_last_well_arrival_s` (bin centre
minus the arrival of the well occupied at bout start) instead of `tc`. Window
`[0, TMAX_ARRIVAL = 8]` s with interior knots (1, 2, 4, 6) s, chosen from the
at-well event distribution.

Cohort. The arrival column is part of the union `freeze_per_bin_cohort` freezes
over, so a bin with a NaN arrival clock (a stop with no completed trial) leaves
the common cohort for **every** rung, not only the arrival ones. The arrival
rungs then apply the window `0 <= time_since_last_well_arrival_s <= TMAX_ARRIVAL`
to that frozen cohort and report `n_bins_arrival_window` and
`retained_share_arrival_*` beside their `n_bins`.

**Comparison rule (binding).** An arrival rung is fit to a strict subset of the
frozen cohort's rows, and AIC, deviance and log-likelihood all scale with the
number of rows. An arrival rung is therefore compared **only** with another
arrival rung: `M5a_time_arrival` against `M0a_const_arrival` (which exists for
exactly this reason — the same homogeneous-Poisson formula as `M0_const`, refit
on the arrival window) and `M7ra_time_arrival_histcos_rew` against
`M5a_time_arrival`. `compare_nested_models` NaNs any comparison whose two rungs
disagree on `n_bins`, so a cross-clock difference cannot be taken by accident.
Every fitted row carries a `time_clock` column (`"stop"` / `"arrival"`) so the
constraint travels with the artifact. The two ladders are read **side by side**,
never differenced; the scale-free comparison the summary reports
(`rate_fit_statistics.clock_shape_statistics`) is the peak position of each
fitted time course and the KS distance between the two curves normalized on
their own windows. Both are read on each rung's
**supported interior** — **between the outermost interior knots**, so `tc` is
read on [1, 3] s (knots 1/2/3) and the arrival clock on [1, 6] s (knots
1/2/4/6) — because a cubic B-spline extrapolating past its outermost interior
knot sits over almost no exposure and can run away there. On the **well-anchored**
cohort, counting only argmaxes that are not on an interval edge
(<!--v:behavior.rate_time_peak_edge_count-->0<!--/v--> stop-clock and <!--v:behavior.rate_arrival_peak_edge_count-->0<!--/v--> arrival-clock argmaxes are), the two clocks put
the across-session median peak within about a second of each other: <!--v:behavior.rate_time_peak_s_low-->2.48<!--/v-->–<!--v:behavior.rate_time_peak_s_high-->2.95<!--/v--> s on `tc` against <!--v:behavior.rate_arrival_peak_s_low-->2.94<!--/v-->–<!--v:behavior.rate_arrival_peak_s_high-->5.21<!--/v--> s after arrival, with KS
distances between the normalized curves of <!--v:behavior.rate_clock_ks_non_local-->0.08<!--/v--> (non-local, all), <!--v:behavior.rate_clock_ks_non_local_no_swr-->0.08<!--/v-->
(non-local no-SWR), <!--v:behavior.rate_clock_ks_swr-->0.09<!--/v--> (detected SWR) and <!--v:behavior.rate_clock_ks_non_local_swr-->0.09<!--/v--> (non-local with SWR). Detected
SWR is no exception — its supported arrival-clock maximum is at <!--v:behavior.rate_arrival_peak_s_swr-->3.11<!--/v--> s. The all-stops sensitivity reads <!--v:behavior.rate_allstops_time_peak_s_low-->2.53<!--/v-->–<!--v:behavior.rate_allstops_time_peak_s_high-->3.00<!--/v--> s against
<!--v:behavior.rate_allstops_arrival_peak_s_low-->3.04<!--/v-->–<!--v:behavior.rate_allstops_arrival_peak_s_high-->5.79<!--/v--> s with KS distances <!--v:behavior.rate_allstops_clock_ks_low-->0.08<!--/v-->–<!--v:behavior.rate_allstops_clock_ks_high-->0.09<!--/v-->; the arrival-clock band is much wider there, so the cross-clock
agreement turns partly on the cohort
([2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md),
"Arrival clock vs stop clock, by cohort").

Read over the **full** [0, 8] s window, a median arrival-clock curve's argmax
can land on the 0 s grid point, where the window holds almost no exposure; that
is a boundary artefact, not an arrival-locked rate. **A 0.00 s peak must never be
read as arrival-locked.** The same caution applied to the non-local-no-SWR stop-clock
peak, which sits at <!--v:behavior.rate_time_peak_s_non_local_no_swr-->2.48<!--/v--> s on the well-anchored fit and <!--v:behavior.rate_allstops_time_peak_s_non_local_no_swr-->3.00<!--/v--> s on the all-stops fit;
an argmax at 3.00 s is the right edge of the supported interval: it bounds the peak rather than locating it.

Read the whole comparison as two descriptions of the same rate, not as one
clock beating the other — neither ladder is fit to the other's rows. The
arrival rungs are also heavily flagged: on the well-anchored fit <!--v:behavior.rate_arrival_separated_fraction_low-->56 %<!--/v--> to
<!--v:behavior.rate_arrival_separated_fraction_high-->64 %<!--/v--> of `M5a_time_arrival` fits (by population) carry a
separated non-nuisance term and are excluded from the shape summary, so a shape
summary rests on a minority of the corpus, not on <!--v:behavior.rate_sessions-->242<!--/v--> sessions. (The shape numbers rest on <!--v:behavior.rate_arrival_peak_sessions_swr-->89<!--/v--> / <!--v:behavior.rate_arrival_peak_sessions_non_local-->105<!--/v--> / <!--v:behavior.rate_arrival_peak_sessions_non_local_swr-->80<!--/v--> / <!--v:behavior.rate_arrival_peak_sessions_non_local_no_swr-->97<!--/v--> unflagged arrival-clock curves by population.)

On the all-stops fit, requiring the arrival clock costs the common
cohort **<!--v:behavior.rate_allstops_arrival_bins_dropped-->89,838<!--/v--> of <!--v:behavior.rate_allstops_arrival_bins_before-->7,943,566<!--/v--> bins (<!--v:behavior.rate_allstops_arrival_bins_dropped_fraction-->1 %<!--/v-->) and <!--v:behavior.rate_allstops_arrival_stops_dropped-->1,509<!--/v--> stops**; the window then
keeps **<!--v:behavior.rate_allstops_arrival_window_bins-->7,136,257<!--/v--> of <!--v:behavior.rate_allstops_frozen_bins-->7,853,728<!--/v--> bins (<!--v:behavior.rate_allstops_arrival_window_fraction-->91 %<!--/v-->)**. On the well-anchored cohort the
clock costs **<!--v:behavior.rate_audit_stops_common_drop-->575<!--/v--> of <!--v:behavior.rate_audit_stops_in-->93,560<!--/v--> stops** (above) and the window keeps a larger share
of every population's events (retention table above).

**The arrival clock used to be the clock of a well the animal had left.** On the
all-stops fit, part of the arrival window comes from `in_transit` bouts, whose
`last_well_arrival_time_s` is the arrival at the well the animal **left**. The arrival rungs still do
not condition on occupancy, but the well-anchored cohort holds no in-transit
bout, so on the canonical fit that population is gone and the clock measures
time at a well. The caveat survives only for the `_allstops` sensitivity
bundle.

**Retention is not balanced across populations.** The window keeps a different
fraction of each population's *events* — on the well-anchored cohort <!--v:behavior.rate_arrival_event_share_swr@pct-->95 %<!--/v-->
(detected SWR), <!--v:behavior.rate_arrival_event_share_non_local_swr@pct-->94 %<!--/v--> (non-local with SWR), <!--v:behavior.rate_arrival_event_share_non_local@pct-->91 %<!--/v--> (non-local, all) and
**<!--v:behavior.rate_arrival_event_share_non_local_no_swr@pct-->88 %<!--/v-->** (non-local without SWR) — so an arrival-ladder contrast between
populations is partly a contrast between windows. The imbalance is smaller than
it was on all stops (<!--v:behavior.rate_allstops_arrival_event_share_swr@pctnum-->91<!--/v--> / <!--v:behavior.rate_allstops_arrival_event_share_non_local_swr@pctnum-->91<!--/v--> / <!--v:behavior.rate_allstops_arrival_event_share_non_local@pctnum-->86<!--/v--> / **<!--v:behavior.rate_allstops_arrival_event_share_non_local_no_swr@pct-->81 %<!--/v-->**), because the bouts the
window discards most are the in-transit ones the cohort no longer holds (their
retention is <!--v:behavior.rate_intransit_arrival_event_share_low-->0.18<!--/v-->–<!--v:behavior.rate_intransit_arrival_event_share_high-->0.37<!--/v--> on the in-transit fit). Both the cohort audit and the runner log report it per
population (`result.attrs["arrival_clock_cohort"]`;
[2026-09-12-rate-cohort-atwell.md](../.claude/docs/reviews/2026-09-12-rate-cohort-atwell.md),
"Arrival-window retention").

**The arrival rungs are themselves rarely cross-validated.** On the
well-anchored fit `cv_bits_per_event` is NaN in <!--v:behavior.rate_arrival_time_cv_nan_fraction_non_local-->92 %<!--/v--> (`M5a_time_arrival`) and
<!--v:behavior.rate_arrival_full_cv_nan_fraction_non_local-->93 %<!--/v--> (`M7ra_time_arrival_histcos_rew`) of non-local fits — <!--v:behavior.rate_arrival_time_cv_nan_fraction_non_local_swr-->100 %<!--/v--> and <!--v:behavior.rate_arrival_full_cv_nan_fraction_non_local_swr-->100 %<!--/v--> on
the `non_local_swr` population — so their held-out numbers describe a small,
self-selected subset of sessions and must not be quoted as corpus quantities.
Read the arrival ladder on its fitted time course, not on its CV — and read it
**pooled**: `M5a_time_arrival` is flagged separated in
<!--v:behavior.rate_arrival_separated_fraction_low-->56 %<!--/v--> to <!--v:behavior.rate_arrival_separated_fraction_high-->64 %<!--/v--> of sessions (by population), so a
per-session arrival coefficient is usually not identified. The ladder-level AIC and the
across-session median curve are the readable quantities; individual
coefficients are not.

### Per-bin reward history is recomputed at the outcome boundary

The stop-level reward history is evaluated at **bout start**. The predicate
`stop_start < outcome_known_time_s <= stop_end` holds for <!--v:behavior.stops_crossing_outcome_fraction-->14 %<!--/v--> of all stops
(<!--v:behavior.stops_crossing_outcome-->16,353<!--/v--> of <!--v:behavior.stops_total-->120,320<!--/v-->), and the shift applies to the **<!--v:behavior.stops_outcome_shifted_fraction-->13 %<!--/v-->** (<!--v:behavior.stops_outcome_shifted-->16,066<!--/v--> stops)
that also keep a bin at or past the boundary. For those the per-bin design
shifts the lag vector at that boundary —
`lag_1 <- completed_trial_reward`, `lag_j <- lag_{j-1}` for bins whose left edge
is at or past it — and re-derives `reward_history_rcos_0..2` and
`reward_history_incomplete` from the shifted row with the **same** basis
functions the stop level uses. Bins before the boundary keep the bout-start
values. The one exclusion is a bout that genuinely contains a **later well
arrival** (the next stop of the session arrived inside its
`(stop_start, stop_end]`), because the completed trip changes partway through;
those are never shifted and are counted instead. On the canonical stop table <!--v:behavior.stops_spanning_arrival-->185<!--/v--> stops contain a later arrival and
<!--v:behavior.stops_spanning_arrival_crossing-->0<!--/v--> of them also contains an outcome boundary. See
[covariate_dictionary.md](covariate_dictionary.md) §F.4.

## MU — the uncertainty/value model (Family B extension)

`MU_uncertainty` (spec: `model_specs.MODEL_UNCERTAINTY_RATE`; runner:
`scripts/behavior/rate/fit_uncertainty_rate.py`, output
`models_per_bin_rate/uncertainty_rate.pkl`) is an **out-of-ladder** Family-B spec
that asks whether replay rate tracks the animal's *behavioral-model* value and
uncertainty beyond the time/reward/history controls. It is the M7r base
(time spline + reward kernel + raised-cosine self-history) plus a curated block:

- **nominal/history (per-unit):** `task_reward_probability_chosen_arm`, `task_reward_probability_patch_mean_gap_best_vs_second`,
  `patch_switch_fraction_last5_completed_trips`,
  `trials_since_block_change`.
- **model value (per-unit):** `rpe_postoutcome` = `reward − reward_belief_mean_chosen_arm_preoutcome` from the fitted
  RL model (`qvals_table.parquet`), represented canonically as
  `rpe_postoutcome`.
- **RL decision variables (z-scored → per-SD rate ratios):**
  `relative_value_switch_minus_stay_preoutcome` (ΔV = switch − stay value),
  `reward_belief_differential_entropy_leaf_mean_preoutcome` (the mean per-leaf
  Beta differential entropy — the belief-uncertainty covariate), and the two
  absolute value-update magnitudes
  `reward_belief_mean_update_l1_all_ports_postoutcome` and
  `reward_belief_mean_update_l1_alternative_patches_postoutcome`. Every one comes
  from the beta-Bernoulli behavioural model; the block carries no HMM covariate
  (the HMM was retired 2026-07-13). See
  [behavior_model_covariates.md](behavior_model_covariates.md).

Collinearity is controlled, and the exclusions are recorded in the runner
(`scripts/behavior/rate/fit_uncertainty_rate.py`): the model's chosen-port value is not
fit, being strongly correlated with ΔV; the nominal-RPE stand-in is not fit either
(it tracks `rpe_postoutcome`, the behavioural-model RPE the block fits
instead); `task_reward_probability_patch_mean_gap_best_vs_second` is kept
because it is nearly uncorrelated with ΔV and so carries independent nominal
information. Fit over the three populations
`swr` / `non_local_swr` / `non_local_no_swr` (the pooled `non_local` null is not
fit here).

## The result holds across both families

Both estimators now carry the *same* past-only reward-history block, and in both
the reward dependence is carried by the SWR-coupled stream and is weakest on the
non-SWR one:

- **Family A (per-stop):** on the well-anchored cohort the reward block improves
  the fit on three populations and not on the non-SWR one — median `R1 − R0`
  ΔAIC (animal-balanced / session-pooled)
  **<!--v:behavior.per_stop_delta_aic_animal_non_local-->−4.3<!--/v--> / <!--v:behavior.per_stop_delta_aic_session_non_local-->−4.2<!--/v-->** (non-local), **<!--v:behavior.per_stop_delta_aic_animal_non_local_swr-->−2.6<!--/v--> / <!--v:behavior.per_stop_delta_aic_session_non_local_swr-->−3.1<!--/v-->** (non-local w/ SWR),
  **<!--v:behavior.per_stop_delta_aic_animal_swr-->−2.7<!--/v--> / <!--v:behavior.per_stop_delta_aic_session_swr-->−4.3<!--/v-->** (detected SWR), **<!--v:behavior.per_stop_delta_aic_animal_non_local_no_swr-->+0.88<!--/v--> / <!--v:behavior.per_stop_delta_aic_session_non_local_no_swr-->+0.88<!--/v-->** (non-local, no SWR), over
  <!--v:behavior.per_stop_sessions_non_local-->241<!--/v--> sessions each, with <!--v:behavior.per_stop_child_better_fraction_non_local-->74 %<!--/v--> of sessions child-better on the pooled
  non-local stream against <!--v:behavior.per_stop_child_better_fraction_non_local_no_swr-->46 %<!--/v--> on the non-SWR one. The earlier per-stop
  headline — a large suppression of the rate after an omitted reward, in most
  sessions — came from an outcome-flag rung that is no longer fit: it was built
  from the trial *start*, so it could encode the not-yet-delivered outcome on
  approach stops, and the causal spec now uses the past-only basis instead (see
  the causal-hygiene note above). Neither its magnitude nor its session share
  survives into this ladder.
- **Family B (per-bin):** on the well-anchored cohort, the reconstructed
  kernel's lag-1 dissociation (**<!--v:behavior.rate_kernel_lag_one_non_local_swr-->1.44<!--/v-->** SWR-coupled vs **<!--v:behavior.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->** non-SWR) and the
  paired within-session comparison
  (**<!--v:behavior.rate_paired_positive_fraction-->78 %<!--/v--> of <!--v:behavior.rate_paired_sessions-->197<!--/v--> sessions, <!--v:behavior.rate_paired_animals_positive-->5<!--/v-->/5 animals, signed-rank p = <!--v:behavior.rate_paired_p-->3.2e-18<!--/v-->**),
  controlling for dwell, the within-stop time course, reward history, and event
  clustering. The paired test is weaker than on the all-stops sensitivity
  (<!--v:behavior.rate_allstops_paired_positive_fraction-->84 %<!--/v-->, p = <!--v:behavior.rate_allstops_paired_p-->2.3e-24<!--/v-->). The non-SWR lag-1 kernel cannot be told
  from 1 (p = <!--v:behavior.rate_kernel_lag_one_p_non_local_no_swr-->0.29<!--/v-->); how far that depends on the model's other terms is not
  yet computed, so "the non-SWR stream does not respond" is not a finding.
  A few kept fits did not converge (<!--v:behavior.rate_kernel_nonconverged_non_local_swr-->14<!--/v--> SWR-coupled, <!--v:behavior.rate_kernel_nonconverged_non_local_no_swr-->24<!--/v--> non-SWR), on the at-risk design described above.

Family A agrees in *direction* on a four-parameter block test; Family B is where
the effect is measured. Read the agreement as directional only: Family A's step
is an AIC comparison of a whole block, Family B's is a kernel value at one lag,
and the conditional Family-B version of Family A's test (`M7r − M6r`, the block
given time and self-history) is positive in all four populations.

## Point-process model checking (Family B)

Family B treats the event train as a point process and checks it as one via the
**time-rescaling theorem** (Brown et al. 2002): if the fitted conditional
intensity is correct, rescaling the inter-event intervals by the integrated
intensity yields a unit-rate Poisson process, so the rescaled ISIs should be
uniform (KS / Q-Q on the diagonal) and independent (flat rescaled-ISI
autocorrelation). `per_bin_rate_gof` computes these by reusing
`non_local_detector.model_checking.sorted_spikes` (`TimeRescaling`,
`point_process_residuals`, `plot_ks`, `plot_qq`,
`plot_rescaled_ISI_autocorrelation`) with `adjust_for_short_trials=True` (the
Wiener 2003 censoring correction for short/variable-length stops) — the
rescaling math is not re-implemented here. The per-fit scalars
(`ks_statistic`, `ks_within_band`, `max_isi_autocorr`, `dispersion`,
`mean_predicted`, `max_predicted`) are persisted for every fit in
`all_models.pkl`; `scripts/behavior/rate/plot_model_checking.py` summarizes them per rung
with `rate_fit_statistics.ladder_fit_statistics` for the
up-the-ladder story (`model_checking_ladder.png`) and re-fits a couple of
example sessions inline for the KS / Q-Q / ISI-autocorrelation / cumulative-
residual quartet (`model_checking_example_{rich,sparse}.png`).

Every rate/content/choice fit additionally carries `converged` / `separated`
status flags, the `|`-joined `separated_terms` naming the unidentified
coefficients, a `sd_<term>` per coefficient, and an `n_convergence_warnings`
count (from `analysis.behavior._fit_diagnostics`). Separation is judged **per
term**: the SE is non-finite or `> 1e3`, or `SE × the SD of its own predictor`
reaches `1.0` — the SE read on the effect scale, so the threshold is
`SE >= 1 / SD` — with `|coef| >= 10` standing in for a zero-SD column, which
cannot be judged on that scale. The structural nuisance terms `Intercept` and
`reward_history_incomplete` (`NUISANCE_TERMS`) are excluded from the judgment:
the indicator marks the handful of stops with fewer than two known outcomes,
which in most sessions carry no events at all, so its coefficient runs to `-inf`
while every scientific term beside it is ordinary. CV metrics carry per-fold
`cv_n_nonconverged_folds` / `cv_n_separated_folds`, and a CV metric is
invalidated (`nan`) if any fold is nonconverged or separated on a non-nuisance
term, rather than dropping the fold. The aggregation drops a session for a term
only when that term itself is named in its `separated_terms`; a product term
(`is_swr:<covariate>`, `swr_x_<covariate>`) is judged on its own coefficient and
does not inherit a factor's separation. A magnitude backstop still catches a
coefficient that ran away without tripping the flag, read on the same effect
scale: `|coef| × sd >= 10` using the fit's own persisted `sd_<term>`, so a steep
per-unit slope on a narrow covariate is kept and only legacy artifacts carrying
no `sd_<term>` fall back to the raw `|coef| >= 10`.

The fit-level `separated` share of the S rungs stays high — on the well-anchored
contrast fit, <!--v:behavior.rate_contrast_expanded_separated_low-->231<!--/v-->–<!--v:behavior.rate_contrast_expanded_separated_high-->239<!--/v--> of the
<!--v:behavior.rate_contrast_expanded_fits_low-->242<!--/v-->–<!--v:behavior.rate_contrast_expanded_fits_high-->242<!--/v--> fits of every rung that carries an `is_swr` expansion (`S2`–`S10`) —
because one term dominates it: `is_swr:reward_history_rcos_1` alone is flagged in
<!--v:behavior.rate_contrast_expanded_flagged_low-->223<!--/v-->–<!--v:behavior.rate_contrast_expanded_flagged_high-->239<!--/v--> fits per rung. The middle bump of the three-bump reward basis is nearly
collinear with its neighbours *inside* the expansion, where it is estimated on
the SWR-coupled bins alone; the same bump outside an expansion is ordinary
(`S1_rew`, which has no `is_swr` reward expansion, flags <!--v:behavior.rate_contrast_plain_flagged-->2<!--/v--> of <!--v:behavior.rate_contrast_plain_fits-->242<!--/v--> fits). The key
`is_swr:<covariate>` terms are barely affected: <!--v:behavior.rate_contrast_terms_losing_sessions-->1<!--/v--> of them lose any session to
the aggregation, at most <!--v:behavior.rate_contrast_most_dropped-->31<!--/v--> of <!--v:behavior.rate_contrast_most_dropped_of-->242<!--/v--> (`S3_periswitch`).
Read the fit-level share as "some coefficient in this fit is unidentified", never
as a verdict on the term being aggregated.

Judging separation per term is what restored the cross-validation metrics of
Family A: on the well-anchored fit <!--v:behavior.per_stop_cv_finite_fraction_full-->52 %<!--/v--> of the reward rung `R1`'s session-fits
carry a finite `cv_ll_per_stop_mean`. Quote the `R1` share, not one pooled with
`R0` — a rung that is always scored, so pooling flatters it.

What the ladder actually shows (<!--v:behavior.rate_sessions-->242<!--/v--> sessions; the fit numbers below are the
`outcome = non_local` values under question 3.3 in
[../paper/reported/questions.md](../paper/reported/questions.md)):

- **The within-stop time course is what calibrates the process.** `M0_const`
  fails the KS band on more than half of sessions (`ks_within_band` <!--v:behavior.rate_ladder_ks_band_const-->0.31<!--/v-->,
  non-Poisson: refractory + bursty), and the reward / self-history blocks
  *without* time do **not** fix it (`M1_rew` <!--v:behavior.rate_ladder_ks_band_rew-->0.26<!--/v-->, `M3r_histcos` <!--v:behavior.rate_ladder_ks_band_histcos-->0.30<!--/v-->). Adding
  the `bs(tc)` spline (`M5_time`) jumps `ks_within_band` to <!--v:behavior.rate_ladder_ks_band_time-->0.72<!--/v-->, `M6r` (+
  self-history) reaches <!--v:behavior.rate_ladder_ks_band_time_histcos-->0.94<!--/v--> and the full model `M7r_time_histcos_rew`
  <!--v:behavior.rate_ladder_ks_band_full-->0.95<!--/v-->. The dominant misfit of the constant model is
  the strong post-stop rate transient, not the fine-timescale clustering.
- **Dispersion stays far below 1 (underdispersed), not → 1.** At these sparse 15 ms
  bins the Poisson deviance is not a calibrated dispersion estimate, so the
  textbook "history block drives dispersion → 1" does not apply; read KS and the
  rescaled-ISI autocorrelation (neither discretization-sensitive) instead. The
  mean per-bin count is the §1 baseline rate times the 15 ms bin — ≈ <!--v:behavior.rate_bin_count_non_local_swr-->0.0033<!--/v-->
  (non-local w/ SWR) to ≈ <!--v:behavior.rate_bin_count_non_local-->0.0067<!--/v--> (non-local, all) — so the trapezoid rescaling is
  well-behaved on average; `mean_predicted` / `max_predicted` are reported so the
  high-rate-bin KS bias can be ruled out.
- **The time block is the only rung with both a large CV gain and a near-full CV
  cohort.** `M5_time` scores **<!--v:behavior.rate_ladder_cv_bits_time-->0.25<!--/v-->** held-out bits/event on **<!--v:behavior.rate_ladder_cv_sessions_time-->226<!--/v--> of <!--v:behavior.rate_ladder_sessions_time-->242<!--/v-->**
  non-local sessions. Its median AIC (<!--v:behavior.rate_ladder_aic_time-->2320<!--/v-->) is above
  `M7r_time_histcos_rew` (<!--v:behavior.rate_ladder_aic_full-->2303<!--/v-->) and `M6r_time_histcos` (<!--v:behavior.rate_ladder_aic_time_histcos-->2307<!--/v-->) — the AIC ordering
  and the held-out ordering are not the same ordering. The three rungs that carry
  time **and** the raised-cosine self-history block score <!--v:behavior.rate_ladder_cv_bits_time_histcos-->0.30<!--/v--> (`M6r_time_histcos`),
  <!--v:behavior.rate_ladder_cv_bits_full-->0.30<!--/v--> (`M7r_time_histcos_rew`) and <!--v:behavior.rate_ladder_cv_bits_reward_by_time-->0.24<!--/v--> (`M9r_rew_x_time`) bits/event but only
  on <!--v:behavior.rate_ladder_cv_sessions_time_histcos-->216<!--/v-->, <!--v:behavior.rate_ladder_cv_sessions_full-->213<!--/v--> and <!--v:behavior.rate_ladder_cv_sessions_reward_by_time-->128<!--/v--> sessions, so a median there is a selected-minority number and
  not a like-for-like win over `M5_time`. The fourth rung carrying that block,
  `M3r_histcos`, has no time spline and scores **<!--v:behavior.rate_ladder_cv_bits_histcos-->0.09<!--/v-->** over <!--v:behavior.rate_ladder_cv_sessions_histcos-->230<!--/v--> sessions — far
  below `M5_time` on a comparably selected cohort. The
  `onset_lag` dummy rungs could not be cross-validated — that
  is what retired them — and no longer exist. Use `M5_time`
  as the parsimonious predictive model and `M7r_time_histcos_rew` for the full
  additive coefficient read; the refractory / bursty *kernel shape* is now read
  off the raised-cosine reconstruction rather than per-lag dummies (see
  [design/model_history.md](design/model_history.md)).
- **The reward block is a small AIC gain once the rungs share rows (except on
  the non-SWR stream), and a loss once time and self-history are already in.**
  Read the block twice. *Marginally* (`M1_rew − M0_const`) the median per-session ΔAIC is
  **<!--v:behavior.rate_marginal_reward_delta_aic_non_local-->−8.9<!--/v-->** (non-local), **<!--v:behavior.rate_marginal_reward_delta_aic_non_local_swr-->−6.0<!--/v-->** (non-local w/ SWR), **<!--v:behavior.rate_marginal_reward_delta_aic_swr-->−7.6<!--/v-->** (detected SWR)
  and **<!--v:behavior.rate_marginal_reward_delta_aic_non_local_no_swr-->−0.53<!--/v-->** (non-local, no SWR): the block pays for itself against a
  constant rate on three populations, and on the non-SWR one it does not (the child
  wins in <!--v:behavior.rate_marginal_reward_child_better_fraction_non_local_no_swr-->52 %<!--/v--> of its sessions). *Conditionally* — the reward
  block given time **and** self-history, `M7r_time_histcos_rew − M6r_time_histcos`
  — every population moves the wrong way: **<!--v:behavior.rate_conditional_reward_delta_aic_swr-->+1.2<!--/v-->** (detected SWR), **<!--v:behavior.rate_conditional_reward_delta_aic_non_local-->+1.0<!--/v-->**
  (non-local), **<!--v:behavior.rate_conditional_reward_delta_aic_non_local_swr-->+1.8<!--/v-->** (non-local w/ SWR), **<!--v:behavior.rate_conditional_reward_delta_aic_non_local_no_swr-->+2.2<!--/v-->** (non-local, no SWR).
  The four-parameter smooth block does not pay for itself once the time
  course and self-history are in the model, even though the lag-1 kernel it
  builds is above 1 in most SWR-coupled sessions. The much larger reward step the retired
  eleven-parameter per-lag block used to report is not a comparison this one
  should be read against: those rungs were not scored on the same rows as their
  parent — the per-lag columns are NaN early in an epoch, so the reward rung was
  fit on fewer bins than the null it was compared against — so that step was
  never a like-for-like model comparison
  ([2026-09-06-rate-refit-arrival-clock.md](../.claude/docs/reviews/2026-09-06-rate-refit-arrival-clock.md)).
- **Within-session non-stationarity is not tested here.** The cumulative
  point-process residual within a session makes slow excursions; because each
  per-session intercept pins the residual at both ends (a Brownian bridge), the
  right test is the bridge sup statistic `max|Mₖ|/√n`. If pursued, the principled
  handling is a **state-space / SS-GLM** baseline drifting in wall-clock time
  (gap-aware across the inter-stop runs), whose latent trajectory could then be
  correlated against task variables — a separate spec, not built here.

## Caveats and what each family can/can't tell you

- **Cross-family CV is not comparable.** Family A scores 5-fold per-stop mean
  log-likelihood; Family B scores 10-fold (over stops) per-bin log-likelihood
  plus `cv_bits_per_event` vs `M0_const`. Compare CV **within** a family only.
- **Lag-dummy self-history was not cross-validatable and is retired.** The
  `onset_lag` dummies separate on the refractory bins in almost every session,
  which voids the whole rung's CV metric of `M3_hist` / `M4_hist_rew` /
  `M6_time_hist` / `M7_time_hist_rew` / `M9_rew_x_time`, so those five rungs
  were removed from the ladder on 2026-09-06 and the reward × time interaction
  re-parented onto `M7r_time_histcos_rew` as `M9r_rew_x_time`. The raised-cosine
  models are identified but still scored on a minority of sessions, so their CV
  is not comparable with `M5_time`'s; compare CV numbers only between rungs with
  comparable `n_cv`. The per-lag refractory / bursty kernel shape is now
  reconstructed from the four `hist_rcos_k` coefficients, which is smooth and
  cannot represent a single-lag notch.
- **Duration is a behavioral variable, not just exposure.** Stop durations are
  bimodal — reward creates the long consummatory mode — so the offset is not a
  nuisance to be divided out; the per-second / per-bin estimators isolate the
  neural rate while the duration structure itself carries reward information.
- **Family B truncates at `TMAX = 4 s`.** It models only the first 4 s of each
  stop, which holds **<!--v:behavior.onsets_within_tmax_fraction_non_local-->79 %<!--/v-->** of non-local onsets (median onset <!--v:behavior.onset_median_s_non_local-->2.4<!--/v--> s after
  the stop's speed-threshold crossing, with a long tail; <!--v:behavior.onsets_within_tmax_fraction_swr-->80 %<!--/v--> of SWR onsets
  and <!--v:behavior.onsets_within_tmax_fraction_mua-->79 %<!--/v--> of MUA onsets, `replay_content/event_onset_table.parquet`). Every
  Family-B conclusion is conditional on the first 4 s.
- **The post-outcome gate truncates the time course a second time, unevenly by
  outcome.** `MU_uncertainty` carries the current trial's realized outcome, so
  `fit_uncertainty_rate` keeps only bins whose left edge is at or after
  `outcome_known_time`. That boundary is pump onset on a rewarded trip but the
  conservative omission boundary (arrival + 0.30 s, capped at the collapsed
  task visit's final departure) on an
  omitted trip. The two outcome classes can therefore retain different portions
  of the stop and are audited separately before fitting.

  The consequence is a design property, not a bug in the gate. The spline's
  boundary knots stay at `(0, TMAX)` while no bin near `tc = 0` survives for
  omissions, so that region is extrapolated for one class and populated by the
  other. The basis is therefore **not orthogonal to outcome**, and the
  time-course and reward-kernel coefficients are partially confounded. Read
  `bs(tc)` coefficients from this model as conditional on the retained,
  outcome-dependent support — do not compare them to an ungated Family-B fit.

  Size the asymmetry before interpreting: the per-session gate audit
  (`*_post_outcome_gate_audit.csv`) reports `min_retained_tc_rewarded`,
  `min_retained_tc_omission`, their medians, and
  `min_retained_tc_omission_minus_rewarded`. A gap that is small relative to
  the first interior knot (1 s) leaves the spline comparable across classes; a
  gap approaching it does not.

## Future work (noted, not built here)

- The long immobility tail (raise `TMAX` or add a censoring term) and slow
  session drift belong in a **state-space model**, not as fixed covariates here.
- Arm / well / patch identity covariates (hard to interpret in this rate view).
- Cross-/ensemble-history between event types (SWR↔non-local).
- The raised-cosine self-history basis (`M3r_histcos`, `M7r_time_histcos_rew`)
  already recovers the over-fit history block; an L2-penalized / Firth fit of the
  per-lag dummies remains an alternative if the per-lag kernel itself needs
  regularizing.
