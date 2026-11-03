# Per-bin replay-rate ladder: fitted effects (Family B)

**Cohort: well-anchored stops.** Quantitative readout of the per-bin
conditional-intensity ladder (`MODEL_PER_BIN_RATE`, see
[rate_models.md](rate_models.md) for the methods), fit per session over the
**<!--v:part3.rate_sessions-->242<!--/v-->-session corpus** and persisted in
`Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`
(`BIN_S = 0.015`, `TMAX = 4 s`, 10 rungs — the seven stop-clock rungs read
below plus the three arrival-anchored ones). Every number here is read from
`paper/reported/part3.json` (`scripts/reported/summarize_part.py --part 3`),
which reads that artifact with the rules of
`scripts/rate/summarize_per_bin_rate_models.py` and `scripts/qa/compare_fits.py`.

The population is the **well-anchored** stops — `occupancy_state ∈ {at_well,
mixed}`, **<!--v:part3.stops_well_anchored-->93,560<!--/v-->** of the <!--v:part3.stops_total-->120,320<!--/v--> canonical stops, of which **<!--v:part3.rate_stops-->92,942<!--/v-->** (over
**<!--v:part3.rate_bins-->7,535,050<!--/v-->** bins) survive the common-cohort freeze — which is the same
population the content family is fit on at the event grain
([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md)). The
all-stops fit is kept as a sensitivity bundle (`all_models_allstops.pkl`) and
in-transit stops as their own (`all_models_intransit.pkl`).
**The in-transit bundle is thin** — only <!--v:part3.rate_intransit_sessions-->52<!--/v--> of <!--v:part3.rate_sessions-->242<!--/v--> sessions clear the
per-bin sample gates on any population — so an in-transit cell showing no effect
is an absence of evidence, not evidence of absence.

Not every population carries a fit in every session: the full additive rung
`M7r_time_histcos_rew` is fit on **<!--v:part3.rate_full_sessions_swr-->242<!--/v--> (detected SWR) / <!--v:part3.rate_full_sessions_non_local-->242<!--/v--> (non-local, all) /
<!--v:part3.rate_full_sessions_non_local_swr-->242<!--/v--> (non-local w/ SWR) / <!--v:part3.rate_full_sessions_non_local_no_swr-->240<!--/v--> (non-local, no SWR)** sessions. Per-section `n`
states which.

Three things are load-bearing for everything below.
(i) The reward-history features are **recomputed at the outcome boundary** for
the <!--v:part3.stops_outcome_shifted_fraction-->13 %<!--/v--> of stops that cross one, so a bin after the boundary carries the
outcome the animal has by then learned. (ii) The five `onset_lag1..10` dummy
rungs were **retired** — they could not be cross-validated — and the ladder now
runs `M0_const`, `M1_rew`, `M5_time`, `M3r_histcos`, `M6r_time_histcos`,
`M7r_time_histcos_rew`, `M9r_rew_x_time` on the stop clock. (iii) The cohort is
the well-anchored one. All three are described in [rate_models.md](rate_models.md).
**Figures:** the six-panel visual companion is
`paper/figures/rate/per_bin_rate_ladder_summary.png`
(`scripts/rate/plot_per_bin_rate_summary.py`); the kernel / GOF figures are
`paper/figures/rate/reward_kernel_basis.png`,
`selfhistory_autocorr.png`, `model_checking_ladder.png` and
`model_checking_example_{rich,sparse}.png`
(`scripts/rate/plot_per_bin_rate_models.py`,
`scripts/rate/plot_model_checking.py`). All were regenerated from this fit.

Effects are reported **per event-type population**: detected SWR (LFP ripples),
all non-local decoding events, and the non-local split by SWR overlap
(`non_local_swr` / `non_local_no_swr`). "RR" = rate ratio; the reward effect is
the reconstructed kernel (§2–3), not `exp` of any one basis coefficient.
Summaries are across-session medians, over the sessions whose terms pass the
per-term separation gate, with per-animal sign consistency (n/5) and a
signed-rank p that treats the sessions (run epochs, several per recording day)
as independent, so it overstates the evidence. Every effect is *conditional on the first 4 s of each stop*,
measured from a speed-threshold crossing.

## 1. Baseline rate (M0_const)

Events per second of immobility (`exp(intercept)`), median across <!--v:part3.rate_sessions-->242<!--/v--> sessions:

| population | rate (events/s) | IQR |
| --- | --- | --- |
| non-local (all) | <!--v:part3.rate_baseline_non_local-->0.43<!--/v--> | <!--v:part3.rate_baseline_lower_quartile_non_local-->0.32<!--/v-->–<!--v:part3.rate_baseline_upper_quartile_non_local-->0.56<!--/v--> |
| detected SWR | <!--v:part3.rate_baseline_swr-->0.27<!--/v--> | <!--v:part3.rate_baseline_lower_quartile_swr-->0.23<!--/v-->–<!--v:part3.rate_baseline_upper_quartile_swr-->0.31<!--/v--> |
| non-local, no SWR | <!--v:part3.rate_baseline_non_local_no_swr-->0.22<!--/v--> | <!--v:part3.rate_baseline_lower_quartile_non_local_no_swr-->0.15<!--/v-->–<!--v:part3.rate_baseline_upper_quartile_non_local_no_swr-->0.30<!--/v--> |
| non-local, w/ SWR | <!--v:part3.rate_baseline_non_local_swr-->0.21<!--/v--> | <!--v:part3.rate_baseline_lower_quartile_non_local_swr-->0.18<!--/v-->–<!--v:part3.rate_baseline_upper_quartile_non_local_swr-->0.26<!--/v--> |

The pairwise contrasts are median *within-session* rate ratios with a paired
Wilcoxon p:

| contrast | median rate ratio | p |
| --- | --- | --- |
| NL w/ SWR / NL no-SWR | <!--v:part3.rate_baseline_ratio_swr_coupled_to_no_swr-->0.98<!--/v-->× | <!--v:part3.rate_baseline_ratio_swr_coupled_to_no_swr_p-->0.30<!--/v--> |
| detected SWR / non-local (all) | <!--v:part3.rate_baseline_ratio_swr_to_non_local-->0.63<!--/v-->× | <!--v:part3.rate_baseline_ratio_swr_to_non_local_p-->6.3e-39<!--/v--> |
| NL w/ SWR / detected SWR | <!--v:part3.rate_baseline_ratio_swr_coupled_to_swr-->0.80<!--/v-->× | <!--v:part3.rate_baseline_ratio_swr_coupled_to_swr_p-->3.0e-41<!--/v--> |
| NL no-SWR / detected SWR | <!--v:part3.rate_baseline_ratio_no_swr_to_swr-->0.77<!--/v-->× | <!--v:part3.rate_baseline_ratio_no_swr_to_swr_p-->8.3e-11<!--/v--> |

The non-local split is roughly additive (<!--v:part3.rate_baseline_non_local_swr-->0.21<!--/v--> + <!--v:part3.rate_baseline_non_local_no_swr-->0.22<!--/v--> ≈ <!--v:part3.rate_baseline_non_local-->0.43<!--/v-->).
Read the coupling shares off the within-session ratios, not off the ratio of the
medians: the median session has **<!--v:part3.rate_baseline_ratio_swr_coupled_to_swr-->0.80<!--/v-->** SWR-coupled non-local events per detected SWR, and
SWR-coupled events are **<!--v:part3.rate_baseline_ratio_swr_coupled_to_non_local-->0.49<!--/v-->** of its non-local events
(`M0_const` intercepts in `all_models.pkl`). (Detected-SWR vs non-local is a
cross-modality comparison — LFP vs decoder — so read the within-non-local split
as the clean contrast.)

## 1b. Within-stop time course (the `bs(tc)` spline)

The dominant structure: replay rate rises over the first seconds of a stop
(panels A0/A1 of the summary figure). The spline uses **fixed shared
knots** (`SPLINE_KNOTS=(1,2,3)`, bounds `[0,4]`), so the fitted `λ(tc)` curve is
**reconstructable from the saved coefficients** (`exp(intercept + bs(tc)·β)`)
and the spline coefficients are comparable across sessions. Read the curve only
**between the outermost interior knots**, [1, 3] s: past the last interior knot
a cubic B-spline extrapolates over very little exposure and can run away. On
that interval the across-session median curve's argmax sits at <!--v:part3.rate_time_peak_s_swr-->2.62<!--/v--> s (detected SWR),
<!--v:part3.rate_time_peak_s_non_local-->2.96<!--/v--> s (non-local, all), <!--v:part3.rate_time_peak_s_non_local_swr-->2.48<!--/v--> s (non-local w/ SWR) and <!--v:part3.rate_time_peak_s_non_local_no_swr-->3.00<!--/v--> s (non-local, no
SWR), read over <!--v:part3.rate_time_peak_sessions_swr-->233<!--/v--> / <!--v:part3.rate_time_peak_sessions_non_local-->236<!--/v--> / <!--v:part3.rate_time_peak_sessions_non_local_swr-->192<!--/v--> / <!--v:part3.rate_time_peak_sessions_non_local_no_swr-->233<!--/v--> unflagged fits. <!--v:part3.rate_time_peak_edge_count-->1<!--/v--> of the four
argmaxes sit on the interval's 3 s edge (on the all-stops fit, <!--v:part3.rate_allstops_time_peak_edge_count-->1<!--/v-->): there the
median curve is still rising where the readout stops, so the argmax bounds the
peak rather than locating it.

**Duration caveat (important):** stop durations are very unequal and
*endogenous* — rewarded well-anchored stops are **<!--v:part3.stop_duration_mean_ratio-->3.6<!--/v-->× longer on the mean** (<!--v:part3.stop_duration_mean_rewarded-->2.2<!--/v--> s vs
<!--v:part3.stop_duration_mean_unrewarded-->0.60<!--/v--> s; <!--v:part3.stop_duration_median_ratio-->3.6<!--/v-->× on the median, <!--v:part3.stop_duration_median_rewarded-->1.0<!--/v--> s vs <!--v:part3.stop_duration_median_unrewarded-->0.28<!--/v--> s, `stop_table.parquet`) — so at
large `tc` only the long
(≈rewarded) stops survive: <!--v:part3.stop_survival_two_s-->25 %<!--/v--> of stops are still immobile at 2 s and <!--v:part3.stop_survival_four_s-->14 %<!--/v--> at
4 s. The exposure offset makes `λ(tc)` a proper rate, but the curve at large
`tc` reflects a **selected subpopulation**, not the same stops over time. The
figure marks this with a survival overlay and grey shading where <20 % survive.
The fitted panel (A1) uses `M7r_time_histcos_rew` (reward and
self-history controlled) so much of the rewarded-long-stop composition is
removed, but the principled fix is to model the stop *departure hazard* jointly
with the rate.

**The arrival-clock variant.** `tc` measures time since a speed-threshold
crossing, and most stops are fragments of a well visit rather than the visit
itself, so the sister rungs `M0a_const_arrival` / `M5a_time_arrival` /
`M7ra_time_arrival_histcos_rew` re-anchor the same spline on time since well
arrival over `[0, 8]` s. They are fit on a strict subset of this ladder's rows
and are therefore **never differenced against it** — read the two side by side.
On their supported interior, [1, 6] s, and counting only argmaxes not on an
interval edge, the two clocks put the median peak within about a second of each
other: on this cohort <!--v:part3.rate_time_peak_s_low-->2.48<!--/v-->–<!--v:part3.rate_time_peak_s_high-->2.96<!--/v--> s on `tc`
against <!--v:part3.rate_arrival_peak_s_low-->2.94<!--/v-->–<!--v:part3.rate_arrival_peak_s_high-->4.00<!--/v--> s after arrival, KS distance <!--v:part3.rate_clock_ks_low-->0.08<!--/v-->–<!--v:part3.rate_clock_ks_high-->0.09<!--/v--> between the normalized
curves. On all stops the bands are <!--v:part3.rate_allstops_time_peak_s_low-->2.57<!--/v-->–<!--v:part3.rate_allstops_time_peak_s_high-->2.98<!--/v--> s against <!--v:part3.rate_allstops_arrival_peak_s_low-->2.91<!--/v-->–<!--v:part3.rate_allstops_arrival_peak_s_high-->5.73<!--/v--> s, KS <!--v:part3.rate_allstops_clock_ks_low-->0.08<!--/v-->–<!--v:part3.rate_allstops_clock_ks_high-->0.09<!--/v-->: the
arrival-clock band is much wider there, so the agreement turns partly on the
cohort. The arrival
rungs are heavily flagged (<!--v:part3.rate_arrival_separated_fraction_low@pctnum-->58<!--/v-->–<!--v:part3.rate_arrival_separated_fraction_high-->63 %<!--/v--> of `M5a_time_arrival` fits carry a
separated non-nuisance term) and are rarely cross-validated, so they are read
pooled, on their fitted curve, never per session; see
[rate_models.md](rate_models.md). What the cohort did change is retention: the
arrival window keeps <!--v:part3.rate_arrival_event_share_non_local_no_swr@pctnum-->88<!--/v-->–<!--v:part3.rate_arrival_event_share_swr@pct-->95 %<!--/v--> of each population's event bins, against
<!--v:part3.rate_allstops_arrival_event_share_non_local_no_swr@pctnum-->81<!--/v-->–<!--v:part3.rate_allstops_arrival_event_share_swr@pct-->91 %<!--/v--> on all stops.

## 2. Reward at lag 1 (the reconstructed kernel)

The effect of a reward on the rate is the reward kernel reconstructed from the
three basis coefficients (§3). At lag 1, per population, over the sessions whose
three reward terms pass the per-term separation gate (the rule the content and
rate contrasts use); "animals" counts those whose median is above 1, and p is a
signed-rank test across sessions that treats them as independent:

| population | lag-1 RR | %>1 | animals | p | sessions | left out | not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| detected SWR | <!--v:part3.rate_kernel_lag_one_swr-->1.38<!--/v--> | <!--v:part3.rate_kernel_lag_one_positive_fraction_swr-->81 %<!--/v--> | <!--v:part3.rate_kernel_lag_one_animals_positive_swr-->5<!--/v-->/5 | <!--v:part3.rate_kernel_lag_one_p_swr-->1.9e-24<!--/v--> | <!--v:part3.rate_kernel_sessions_swr-->240<!--/v--> | <!--v:part3.rate_kernel_dropped_swr-->2<!--/v--> | <!--v:part3.rate_kernel_nonconverged_swr-->59<!--/v--> |
| non-local (all) | <!--v:part3.rate_kernel_lag_one_non_local-->1.18<!--/v--> | <!--v:part3.rate_kernel_lag_one_positive_fraction_non_local-->72 %<!--/v--> | <!--v:part3.rate_kernel_lag_one_animals_positive_non_local-->5<!--/v-->/5 | <!--v:part3.rate_kernel_lag_one_p_non_local-->3.8e-17<!--/v--> | <!--v:part3.rate_kernel_sessions_non_local-->241<!--/v--> | <!--v:part3.rate_kernel_dropped_non_local-->1<!--/v--> | <!--v:part3.rate_kernel_nonconverged_non_local-->39<!--/v--> |
| non-local, w/ SWR | **<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.50<!--/v-->** | <!--v:part3.rate_kernel_lag_one_positive_fraction_non_local_swr-->85 %<!--/v--> | <!--v:part3.rate_kernel_lag_one_animals_positive_non_local_swr-->5<!--/v-->/5 | <!--v:part3.rate_kernel_lag_one_p_non_local_swr-->3.2e-28<!--/v--> | <!--v:part3.rate_kernel_sessions_non_local_swr-->225<!--/v--> | <!--v:part3.rate_kernel_dropped_non_local_swr-->17<!--/v--> | <!--v:part3.rate_kernel_nonconverged_non_local_swr-->78<!--/v--> |
| non-local, no SWR | **<!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.00<!--/v-->** | <!--v:part3.rate_kernel_lag_one_positive_fraction_non_local_no_swr-->50 %<!--/v--> | <!--v:part3.rate_kernel_lag_one_animals_positive_non_local_no_swr-->4<!--/v-->/5 | <!--v:part3.rate_kernel_lag_one_p_non_local_no_swr-->0.55<!--/v--> | <!--v:part3.rate_kernel_sessions_non_local_no_swr-->218<!--/v--> | <!--v:part3.rate_kernel_dropped_non_local_no_swr-->22<!--/v--> | <!--v:part3.rate_kernel_nonconverged_non_local_no_swr-->62<!--/v--> |

Within session, the SWR-coupled lag-1 kernel is larger than the non-SWR one in
**<!--v:part3.rate_paired_positive_fraction-->79 %<!--/v--> of <!--v:part3.rate_paired_sessions-->207<!--/v--> paired sessions, <!--v:part3.rate_paired_animals_positive-->5<!--/v-->/5 animals, signed-rank p = <!--v:part3.rate_paired_p-->3.1e-19<!--/v-->**
(the sessions in which both streams pass the gate). The non-SWR lag-1 kernel
cannot be told from 1 here; on all stops it is
<!--v:part3.rate_allstops_kernel_lag_one_no_swr-->0.92<!--/v--> (p = <!--v:part3.rate_allstops_kernel_lag_one_p_no_swr-->0.029<!--/v-->, above 1 in <!--v:part3.rate_allstops_kernel_lag_one_animals_positive_no_swr-->0<!--/v-->/5 animals).
Kept fits that the optimizer flagged as not converged are counted, not left out:
why they do not converge is still open, and the values may move once it is
settled.

### Basis coefficients (not effect sizes)

Each reward feature is a bump-weighted reward *fraction* (`reward_history_rcos_0`
weights lags 1–7 at 1.00, 0.79, 0.54, …; `_1` spans all ten lags with its peak
at lag 3; `_2` rises to lag 10). The three coefficients are strongly collinear,
and a 0→1 change in one feature with the other two held fixed is no reward
history that occurs, so `exp(coef)` of one basis function is **not** the effect
of a reward at any lag. The coefficients are listed for completeness only; do
not quote them as rate ratios:

| population | bump 0 (M1_rew) | bump 0 (M7r) | bump 1 (M7r) | bump 2 (M7r) |
| --- | --- | --- | --- | --- |
| detected SWR | <!--v:part3.rate_raw_immediate_session_swr-->58<!--/v--> | <!--v:part3.rate_immediate_session_swr-->6.4<!--/v--> | <!--v:part3.rate_bump_one_session_swr-->0.021<!--/v--> | <!--v:part3.rate_bump_two_session_swr-->7.7<!--/v--> |
| non-local (all) | <!--v:part3.rate_raw_immediate_session_non_local-->30<!--/v--> | <!--v:part3.rate_immediate_session_non_local-->3.3<!--/v--> | <!--v:part3.rate_bump_one_session_non_local-->0.051<!--/v--> | <!--v:part3.rate_bump_two_session_non_local-->5.2<!--/v--> |
| non-local, w/ SWR | <!--v:part3.rate_raw_immediate_session_non_local_swr-->130<!--/v--> | <!--v:part3.rate_immediate_session_non_local_swr-->9.9<!--/v--> | <!--v:part3.rate_bump_one_session_non_local_swr-->0.0088<!--/v--> | <!--v:part3.rate_bump_two_session_non_local_swr-->15<!--/v--> |
| non-local, no SWR | <!--v:part3.rate_raw_immediate_session_non_local_no_swr-->12<!--/v--> | <!--v:part3.rate_immediate_session_non_local_no_swr-->1.4<!--/v--> | <!--v:part3.rate_bump_one_session_non_local_no_swr-->0.17<!--/v--> | <!--v:part3.rate_bump_two_session_non_local_no_swr-->2.4<!--/v--> |

(`exp` of the across-session median coefficient, each over the sessions where
that term passes the gate.) None of these three is a per-lag effect on its own;
they are one kernel.

## 3. Reward kernel, lags 1–10 (M7r, reconstructed from the basis)

The per-lag log-rate contribution is `Σ_k coef_k · B[k, lag] / Σ_lag B[k, lag]`
(row-normalized basis), evaluated **per session** and then summarized as the
median rate ratio across sessions — the same estimator
`scripts/rate/plot_per_bin_rate_models.py` draws in
`paper/figures/rate/reward_kernel_basis.png`, which is the figure to read (and
to cite) for the shape. Reconstructing from across-session median coefficients
instead would describe a session that does not exist, because the three basis
coefficients are collinear.

| population | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 | lag9 | lag10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| detected SWR | <!--v:part3.rate_kernel_lag_one_swr-->1.38<!--/v--> | <!--v:part3.rate_kernel_lag_two_swr-->1.04<!--/v--> | <!--v:part3.rate_kernel_lag_three_swr-->0.93<!--/v--> | <!--v:part3.rate_kernel_lag_four_swr-->0.90<!--/v--> | <!--v:part3.rate_kernel_lag_five_swr-->0.90<!--/v--> | <!--v:part3.rate_kernel_lag_six_swr-->0.92<!--/v--> | <!--v:part3.rate_kernel_lag_seven_swr-->0.94<!--/v--> | <!--v:part3.rate_kernel_lag_eight_swr-->0.97<!--/v--> | <!--v:part3.rate_kernel_lag_nine_swr-->1.00<!--/v--> | <!--v:part3.rate_kernel_lag_ten_swr-->1.04<!--/v--> |
| non-local (all) | <!--v:part3.rate_kernel_lag_one_non_local-->1.18<!--/v--> | <!--v:part3.rate_kernel_lag_two_non_local-->0.99<!--/v--> | <!--v:part3.rate_kernel_lag_three_non_local-->0.93<!--/v--> | <!--v:part3.rate_kernel_lag_four_non_local-->0.91<!--/v--> | <!--v:part3.rate_kernel_lag_five_non_local-->0.91<!--/v--> | <!--v:part3.rate_kernel_lag_six_non_local-->0.93<!--/v--> | <!--v:part3.rate_kernel_lag_seven_non_local-->0.95<!--/v--> | <!--v:part3.rate_kernel_lag_eight_non_local-->0.98<!--/v--> | <!--v:part3.rate_kernel_lag_nine_non_local-->1.01<!--/v--> | <!--v:part3.rate_kernel_lag_ten_non_local-->1.04<!--/v--> |
| non-local, w/ SWR | <!--v:part3.rate_kernel_lag_one_non_local_swr-->1.50<!--/v--> | <!--v:part3.rate_kernel_lag_two_non_local_swr-->1.07<!--/v--> | <!--v:part3.rate_kernel_lag_three_non_local_swr-->0.95<!--/v--> | <!--v:part3.rate_kernel_lag_four_non_local_swr-->0.90<!--/v--> | <!--v:part3.rate_kernel_lag_five_non_local_swr-->0.89<!--/v--> | <!--v:part3.rate_kernel_lag_six_non_local_swr-->0.91<!--/v--> | <!--v:part3.rate_kernel_lag_seven_non_local_swr-->0.93<!--/v--> | <!--v:part3.rate_kernel_lag_eight_non_local_swr-->0.98<!--/v--> | <!--v:part3.rate_kernel_lag_nine_non_local_swr-->1.02<!--/v--> | <!--v:part3.rate_kernel_lag_ten_non_local_swr-->1.07<!--/v--> |
| non-local, no SWR | <!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.00<!--/v--> | <!--v:part3.rate_kernel_lag_two_non_local_no_swr-->0.95<!--/v--> | <!--v:part3.rate_kernel_lag_three_non_local_no_swr-->0.93<!--/v--> | <!--v:part3.rate_kernel_lag_four_non_local_no_swr-->0.92<!--/v--> | <!--v:part3.rate_kernel_lag_five_non_local_no_swr-->0.94<!--/v--> | <!--v:part3.rate_kernel_lag_six_non_local_no_swr-->0.95<!--/v--> | <!--v:part3.rate_kernel_lag_seven_non_local_no_swr-->0.97<!--/v--> | <!--v:part3.rate_kernel_lag_eight_non_local_no_swr-->0.98<!--/v--> | <!--v:part3.rate_kernel_lag_nine_non_local_no_swr-->0.99<!--/v--> | <!--v:part3.rate_kernel_lag_ten_non_local_no_swr-->1.01<!--/v--> |

So the reward kernel is **a boost at lag 1 that dissociates by SWR coupling
(<!--v:part3.rate_kernel_lag_one_non_local_swr-->1.50<!--/v--> SWR-coupled vs <!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.00<!--/v--> non-SWR), then a mild suppression from lag 3 on
(the lowest per-population value over lags 3–10 is <!--v:part3.rate_kernel_late_min_low-->0.89<!--/v-->–<!--v:part3.rate_kernel_late_min_high-->0.92<!--/v-->, shared across all four populations) that returns to ~1 by lag
10**. The smooth basis has no per-lag parameter and cannot represent a
lag-specific spike, so read the shape off the reconstruction rather than off any
one coefficient. The lag≥2 suppression is a recent-reward effect; whether it is
an RPE / relative-reward signal or satiety / adaptation is **not** established
here — this GLM uses past binary outcomes, not a prediction error.

## 4. Self-history (within-stop autocorrelation)

`M7r_time_histcos_rew` carries a self-history block (four `offhist_rcos_k` raised-cosine coefficients over the ten 15 ms lags after the previous event's end) as a control for the reward kernel; <!--v:part3.rate_hist_separated_fraction_non_local-->28 %<!--/v-->–<!--v:part3.rate_hist_separated_fraction_non_local_swr-->42 %<!--/v--> of each population's fits flag one of those terms separated. The ladder is fit on the at-risk bins only (a bin inside an ongoing event is dropped; [rate_models.md](rate_models.md#family-b--per-bin-conditional-intensity-the-confound-free-layer)). The earlier all-bins design with history counted from onsets made a short-lag dip and a burst on simulated data alone ([review](../.claude/docs/reviews/2026-10-26-duration-aware-history-simulation.md)), and its short-lag coefficient had no finite estimate on real data.

On the duration-aware design instead (claim 19c: a bin is at risk only while no event is ongoing, and history is anchored on both the onset and the end of each event), there is **no refractory period beyond the event itself** (15 ms after an event ends the rate ratio is <!--v:part3.history_end_first_lag_rr_low-->1.68<!--/v-->–<!--v:part3.history_end_first_lag_rr_high-->4.53<!--/v-->×), events **cluster right after the previous event ends** (<!--v:part3.history_end_peak_low-->2.63<!--/v-->–<!--v:part3.history_end_peak_high-->4.53<!--/v-->× at <!--v:part3.history_end_peak_lag_ms_low-->15<!--/v-->–<!--v:part3.history_end_peak_lag_ms_high-->45<!--/v--> ms after the end), and a **weaker onset-locked rhythm** remains (an animal counts when its peak's interval, widened for the lags searched, stays above 1): detected SWR <!--v:part3.history_onset_peak_swr-->2.28<!--/v-->× at <!--v:part3.history_onset_peak_lag_ms_swr-->90<!--/v--> ms (<!--v:part3.history_onset_animals_swr-->5<!--/v-->/5 animals), non-local with SWR <!--v:part3.history_onset_peak_non_local_swr-->1.83<!--/v-->× at <!--v:part3.history_onset_peak_lag_ms_non_local_swr-->105<!--/v--> ms (<!--v:part3.history_onset_animals_non_local_swr-->4<!--/v-->/5), non-local without SWR <!--v:part3.history_onset_peak_non_local_no_swr-->1.34<!--/v-->× at <!--v:part3.history_onset_peak_lag_ms_non_local_no_swr-->150<!--/v--> ms (<!--v:part3.history_onset_animals_non_local_no_swr-->2<!--/v-->/5).

## 5. Behavioral-state covariates

**The shipped ladder carries none.** Every rung of `MODEL_PER_BIN_RATE` is built
from the time spline, the reward-history basis and the self-history basis only;
no rung carries a value, uncertainty, switch-rate or block covariate. Those are
fit in dedicated per-bin ladders, on their own frozen cohorts and with the
post-outcome gating they need:

- `MU_uncertainty` (`scripts/rate/fit_uncertainty_rate.py` →
  `models_per_bin_rate/uncertainty_rate.pkl`) — task value / uncertainty /
  switch-rate covariates per population.
- The `is_swr` contrast ladder (`scripts/rate/fit_swr_contrast_rate.py` →
  `models_per_bin_rate/swr_contrast_rate.pkl`) — the same covariates as an
  SWR-vs-non-SWR contrast.

Earlier revisions of this document carried a recent-reward-rate rung and a
patch-switch rung inside this ladder. They are not fit here any more, and their
numbers must not be carried forward: they were fit on a different reward block
and a different cohort.

## 6. Model ladder (fit summary, outcome = non-local)

Median AIC across sessions, median held-out bits/event from the per-session CV
(0 = the constant model), and the share of sessions whose time-rescaling KS
curve stays inside the band. `n` is the number of non-local sessions with a
finite-AIC fit; the arrival rungs are **not** in this table because they are fit
on a subset of the rows and their AIC is not on this scale.

`n_cv` is the number of those sessions whose CV metric is valid: a fold in which
any non-nuisance term is unidentified voids the whole metric, so a rung with a
small `n_cv` is summarized over a **selected minority** of sessions and its
bits/event is not comparable with a rung scored on the full corpus.

| model | median AIC | CV bits/event | n_cv | n | KS-in-band |
| --- | --- | --- | --- | --- | --- |
| M0_const | <!--v:part3.rate_ladder_aic_const-->2227<!--/v--> | <!--v:part3.rate_ladder_cv_bits_const-->0.00<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_const-->242<!--/v--> | <!--v:part3.rate_ladder_sessions_const-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_const-->0.45<!--/v--> |
| M1_rew | <!--v:part3.rate_ladder_aic_rew-->2221<!--/v--> | <!--v:part3.rate_ladder_cv_bits_rew-->0.02<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_rew-->238<!--/v--> | <!--v:part3.rate_ladder_sessions_rew-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_rew-->0.37<!--/v--> |
| M5_time | <!--v:part3.rate_ladder_aic_time-->2138<!--/v--> | <!--v:part3.rate_ladder_cv_bits_time-->0.24<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_time-->226<!--/v--> | <!--v:part3.rate_ladder_sessions_time-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_time-->0.93<!--/v--> |
| M3r_histcos | <!--v:part3.rate_ladder_aic_histcos-->2207<!--/v--> | <!--v:part3.rate_ladder_cv_bits_histcos-->0.05<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_histcos-->67<!--/v--> | <!--v:part3.rate_ladder_sessions_histcos-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_histcos-->0.37<!--/v--> |
| M6r_time_histcos | <!--v:part3.rate_ladder_aic_time_histcos-->2128<!--/v--> | <!--v:part3.rate_ladder_cv_bits_time_histcos-->0.22<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_time_histcos-->62<!--/v--> | <!--v:part3.rate_ladder_sessions_time_histcos-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_time_histcos-->0.96<!--/v--> |
| **M7r_time_histcos_rew** | <!--v:part3.rate_ladder_aic_full-->2128<!--/v--> | <!--v:part3.rate_ladder_cv_bits_full-->0.22<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_full-->61<!--/v--> | <!--v:part3.rate_ladder_sessions_full-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_full-->0.96<!--/v--> |
| M9r_rew_x_time | <!--v:part3.rate_ladder_aic_reward_by_time-->2133<!--/v--> | <!--v:part3.rate_ladder_cv_bits_reward_by_time-->0.18<!--/v--> | <!--v:part3.rate_ladder_cv_sessions_reward_by_time-->40<!--/v--> | <!--v:part3.rate_ladder_sessions_reward_by_time-->242<!--/v--> | <!--v:part3.rate_ladder_ks_band_reward_by_time-->0.95<!--/v--> |

- **The time spline is what calibrates the process** (KS-in-band jumps
  <!--v:part3.rate_ladder_ks_band_const-->0.45<!--/v--> → <!--v:part3.rate_ladder_ks_band_time-->0.93<!--/v--> at `M5_time`, and neither reward nor self-history fixes the
  constant model without it: <!--v:part3.rate_ladder_ks_band_rew-->0.37<!--/v--> and <!--v:part3.rate_ladder_ks_band_histcos-->0.37<!--/v-->).
- **`M5_time` is the best-generalizing model**, and the only rung with both a
  large held-out gain and a near-complete CV cohort (<!--v:part3.rate_ladder_cv_bits_time-->0.24<!--/v--> bits/event
  over <!--v:part3.rate_ladder_cv_sessions_time-->226<!--/v--> of <!--v:part3.rate_ladder_sessions_time-->242<!--/v--> sessions). The three rungs carrying time **and** the
  raised-cosine self-history block score <!--v:part3.rate_ladder_cv_bits_time_histcos-->0.22<!--/v--> (`M6r_time_histcos`, <!--v:part3.rate_ladder_cv_sessions_time_histcos-->62<!--/v--> sessions),
  <!--v:part3.rate_ladder_cv_bits_full-->0.22<!--/v--> (`M7r_time_histcos_rew`, <!--v:part3.rate_ladder_cv_sessions_full-->61<!--/v-->) and <!--v:part3.rate_ladder_cv_bits_reward_by_time-->0.18<!--/v--> (`M9r_rew_x_time`, <!--v:part3.rate_ladder_cv_sessions_reward_by_time-->40<!--/v-->), so their CV is
  **not** a like-for-like comparison against `M5_time`'s: a rung is not "better on
  held-out data" because a selected minority of sessions scored well.
  `M3r_histcos`, which carries that block without the time spline, scores only
  **<!--v:part3.rate_ladder_cv_bits_histcos-->0.05<!--/v-->** over <!--v:part3.rate_ladder_cv_sessions_histcos-->67<!--/v--> sessions — a comparably selected cohort, and far below
  `M5_time`. Note also that the AIC ordering differs from the held-out one:
  `M7r` (<!--v:part3.rate_ladder_aic_full-->2128<!--/v-->) and `M6r` (<!--v:part3.rate_ladder_aic_time_histcos-->2128<!--/v-->) sit below `M5_time` (<!--v:part3.rate_ladder_aic_time-->2138<!--/v-->) on median AIC. Use
  `M5_time` as the parsimonious predictive model and `M7r_time_histcos_rew`
  for the full additive coefficient read.
- **The reward block is a small gain marginally (except on the non-SWR stream)
  and a loss conditionally.** Median per-session `M1_rew − M0_const` ΔAIC is
  **<!--v:part3.rate_marginal_reward_delta_aic_swr-->−6.8<!--/v-->**, **<!--v:part3.rate_marginal_reward_delta_aic_non_local-->−7.8<!--/v-->**, **<!--v:part3.rate_marginal_reward_delta_aic_non_local_swr-->−5.4<!--/v-->**, **<!--v:part3.rate_marginal_reward_delta_aic_non_local_no_swr-->+0.089<!--/v-->**
  (detected SWR, non-local all, w/ SWR, no SWR; negative is better) — the child
  wins in <!--v:part3.rate_marginal_reward_child_better_fraction_swr-->79 %<!--/v--> / <!--v:part3.rate_marginal_reward_child_better_fraction_non_local-->84 %<!--/v--> / <!--v:part3.rate_marginal_reward_child_better_fraction_non_local_swr-->83 %<!--/v--> / <!--v:part3.rate_marginal_reward_child_better_fraction_non_local_no_swr-->49 %<!--/v--> of sessions, so
  the block pays for itself against a constant rate on three populations and not
  on the non-SWR one (on all stops that step is <!--v:part3.rate_allstops_marginal_reward_delta_aic_non_local_no_swr-->+0.65<!--/v--> and the child wins in
  <!--v:part3.rate_allstops_marginal_reward_child_better_fraction_non_local_no_swr-->44 %<!--/v-->). And
  `M7r_time_histcos_rew − M6r_time_histcos`, the same block given time **and**
  self-history, is positive everywhere:
  **<!--v:part3.rate_conditional_reward_delta_aic_swr-->+0.94<!--/v-->**, **<!--v:part3.rate_conditional_reward_delta_aic_non_local-->+1.0<!--/v-->**, **<!--v:part3.rate_conditional_reward_delta_aic_non_local_swr-->+1.6<!--/v-->**, **<!--v:part3.rate_conditional_reward_delta_aic_non_local_no_swr-->+2.4<!--/v-->** on the same four populations. The four-parameter smooth block does
  not pay for itself once the time course and self-history are in the model,
  even though the lag-1 kernel it builds is above 1 in most SWR-coupled
  sessions (§2): AIC weighs four parameters against a gain concentrated at lag 1.
- **`M9r_rew_x_time` (reward × time) costs CV and does not buy AIC** relative to
  its parent `M7r_time_histcos_rew` (<!--v:part3.rate_ladder_aic_reward_by_time-->2133<!--/v--> vs <!--v:part3.rate_ladder_aic_full-->2128<!--/v-->; <!--v:part3.rate_ladder_cv_bits_reward_by_time-->0.18<!--/v--> vs <!--v:part3.rate_ladder_cv_bits_full-->0.22<!--/v--> bits/event on
  a smaller CV cohort still), so there is no evidence here that recent reward
  reshapes the within-stop time course rather than scaling it.

See [rate_models.md](rate_models.md) for the GOF / dispersion caveats and the
retirement of the lag-dummy history rungs.
