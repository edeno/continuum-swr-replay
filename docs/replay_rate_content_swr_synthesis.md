# Replay rate × content × SWR-coupling: integrative synthesis

**Status:** living results note. Confirmed findings, open hypotheses, and unknowns
for how the **rate**, **content**, and **SWR-coupling** of non-local replay events
relate. Companion to the per-layer references: rate ladder
[per_bin_rate_results.md](per_bin_rate_results.md) / [rate_models.md](rate_models.md);
content claims registry [../paper/claims_table.md](../paper/claims_table.md).

> **Refit status (2026-09-12).** Both quantitative layers now share one
> population. The rate families and the `is_swr`-contrast content ladder were
> refit on the **well-anchored stop cohort** (`occupancy_state ∈ {at_well,
> mixed}`, 84,069 of 107,400 stops), which is the population the content family
> was already fit on at the event grain, so a rate claim and a content claim
> about non-SWR replay are now claims about the same events
> ([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md);
> [audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md),
> [audits/2026-09-12-content-cohort-comparison.md](audits/2026-09-12-content-cohort-comparison.md)).
> The rate numbers moved the most: the non-SWR stream's immediate reward
> component rises once the in-transit stops — which carried a disproportionate
> share of non-SWR onsets — leave the cohort, so the reward dissociation is
> weaker and is now a difference between two responding streams rather than
> response-vs-none. The content coefficients moved at the third decimal, and
> their per-SD units are now defined on the at-well fit cohort. The all-stops
> rate fits survive as a sensitivity bundle; an in-transit bundle exists but is
> thin (50 of 218 sessions) and is not evidence of absence.
>
> Before that, both quantitative layers were refit in the 2026-09
> fix wave, and the numbers below have been restated from those refits wherever
> the audits report them. The rate ladders were refit on **2026-09-06** with a
> smooth past-only reward-history basis, boundary-recomputed reward features,
> per-coefficient separation judging and shared-row rung scoring
> ([audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md));
> the content ladders were refit on **2026-09-05** onto the per-event allocation
> fraction with event-clustered standard errors
> ([audits/2026-09-05-content-refit-comparison.md](audits/2026-09-05-content-refit-comparison.md)).
> Any quantity this document still carries that was **not** re-derived from those
> refits is marked **PENDING** and must not be cited; where a claim rested on a
> rung that no longer exists, the claim is marked retired in place rather than
> deleted.

## Provenance & methods (read first)

- **Data:** event tables (`event_table.parquet`, `event_arm_table.parquet`,
  `stop_table.parquet`) on the canonical 218-session, 5-animal corpus. Rate
  models refit 2026-09-06; content models refit 2026-09-05.
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
  treated a median ~32 autocorrelated bins per event as independent draws, so
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
  `swr_x_reward_belief_mean_candidate_arm_preoutcome`), where count −0.398 (1/5
  animals) and DM −0.316 (0/5) are negative while the shipped fraction estimator
  is +0.237 (3/5), on the fully paired 216-session cohort. Every other key term
  agrees in sign across all three, and the magnitudes shrink monotonically from
  count to fraction to DM on the SWR interactions and the recency/novelty terms
  (`CS1` −0.461 → −0.373 → −0.297; `CS4` −1.449 → −1.198 → −0.841) while
  `is_current_arm` moves the other way (+0.626 → +0.681 → +0.737). Those three
  columns are one fitting run, made before the 2026-09-12 cohort refit, so the
  fraction column there is the pre-refit value of a coefficient that now reads
  +0.708; the comparison is between likelihoods and is read on its own rows. The DM also
  carries a parameter the conditional estimators do not — `log_phi_is_swr`, the
  within-event concentration contrast — which is the direct measurement of "SWR
  broadens content" (Layer 3). The fraction's event-clustered SE runs ~15–50 %
  above the DM's model SE (median ≈ 30 %), so the shipped estimator is if
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
bins, 218 sessions, the well-anchored stop cohort — 83,549 stops over 6,777,953
bins). Baseline rates (`M0_const`, events per second of
immobility, median across sessions): non-local (all) **0.407** > detected SWR
**0.289** > non-local no-SWR **0.225** > non-local w/ SWR **0.184**. The
non-local split is roughly additive, so **SWR-coupled non-local events are the
rarest type**. The dominant structure in every population is the within-stop
**time course** (rise over the first ~1.5–2 s then plateau; median peak 2.63 s
detected SWR / 2.82 s non-local / 2.31 s w/ SWR / 2.68 s no-SWR, read only on the
supported interval [1, 3] s), **self-history** clustering, and a smooth
**reward-history** kernel. The time spline is what calibrates the process
(KS-in-band 0.44 → 0.91 at `M5_time`). `M5_time` is the best-generalizing rung
(0.266 held-out bits/event over 208 of 218 sessions);
`M7r_time_histcos_rew` is the rung to read coefficients off (median AIC 2077,
KS-in-band 0.93), but its CV is scored on a selected 74 sessions and is **not** a
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
| detected SWR | 1.23 | 0.91–1.03 | 3.44 (76 % of sessions, 5/5, p 3.5e-19) |
| non-local (all) | 1.21 | 0.90–1.04 | 3.44 (81 %, 5/5, p 3.0e-21) |
| non-local, w/ SWR | **1.41** | 0.89–1.06 | **7.18** (85 %, 5/5, p 1.3e-26) |
| non-local, no SWR | **1.09** | 0.91–1.00 | **2.00** (66 %, 5/5, p 3.1e-06) |

The kernel lag-1 column is `rr_lag1` — the reconstructed kernel evaluated at lag
1 — and the last column is the across-session median of `exp` of the single
bump-0 coefficient. They are different quantities; the audited aggregate
summarizes the latter animal-balanced instead, where it reads 4.91 / 2.96 /
6.86 / 2.28 in the same row order. Never substitute one for another.

**The dissociation lives in the immediate component, and it is paired within
session:** the within-session comparison of `coef_reward_history_rcos_0` between
the two non-local streams is positive in **69.4 % of 216 paired sessions, 5/5
animals, signed-rank p = 1.1e-11**. After full control the non-SWR immediate
component is a real response of its own (RR 2.00, p = 3.1e-06, 5/5 animals) —
about 3.6× smaller than the SWR-coupled one, not absent. On all stops it read as
absent (RR 1.24, p = 0.22, 4/5, lag 1 at 0.99, paired in 79.2 %, p = 7.8e-22);
**that framing belonged to a population including in-transit stops and must not
be carried forward.** The dissociation survives on the shared cohort, weaker. The slower part of the kernel — a mild suppression from lag 3
on, RR ≈ 0.89–0.97, back to ~1 by lag 10 — is **shared** by all four
populations. Whether that slower part is an RPE / relative-reward signal or
satiety / adaptation is **not** established here: the GLM uses past binary
outcomes, not a prediction error.

**The reward block is a small AIC step marginally and a negative one
conditionally** (negative ΔAIC = child better). `M1_rew − M0_const` is **−3.23**
(detected SWR) / **−8.18** (non-local) / **−3.96** (w/ SWR) / **−0.75** (no
SWR); `M7r_time_histcos_rew − M6r_time_histcos`, the same block given time
**and** self-history, is **+1.58** / **+1.19** / **+2.41** / **+2.28** — every
population moves the wrong way. Family A (per-stop Poisson with a
`log(immobility)` offset, the literature-comparable layer) agrees in *direction*
only: median per-session `R1 − R0` ΔAIC **−1.41** (detected SWR) / **−5.12**
(non-local) / **−2.43** (w/ SWR) / **+0.30** (no SWR), 218 sessions each — the
block helps everywhere except the non-SWR stream. **Those Family-A values are
the all-stops fit**: the per-stop ladder was refit on the well-anchored cohort
but no audit of that refit has been published, so cite it as the sensitivity
until one is. **Retired with the outcome-flag
rung:** the per-stop omission RR 0.36 and recent-reward RR 0.62 headline came
from a rung built at the trial *start*, which could encode the not-yet-delivered
outcome on approach stops; it is no longer fit and neither its magnitude nor its
session share survives.

**Events are strongly non-Poisson.** The controlled self-history kernel is
refractory at short lags then bursty: peak RR **2.59** (detected SWR, 135 ms),
**2.02** (non-local, 120 ms), **2.43** (w/ SWR, 150 ms, dead time ≤45 ms),
**1.57** (no SWR, 105 ms). SWR-coupled non-local events are the most tightly
bursty stream and non-SWR events the flattest.

**Reward shifts the rate LEVEL, not the within-stop profile SHAPE.**
`M9r_rew_x_time` costs held-out bits and buys no AIC against its parent
`M7r_time_histcos_rew` (2083 vs 2077; 0.200 vs 0.244 bits/event, on a smaller CV
cohort still).

**Arrival clock.** Sister rungs (`M0a_const_arrival`, `M5a_time_arrival`,
`M7ra_time_arrival_histcos_rew`) re-anchor the same spline on time since well
arrival over [0, 8] s. They are fit on a strict subset of the rows and are
therefore **never differenced** against the stop-clock rungs. On their supported
interior the two clocks put the median peak at broadly the same delay
(2.2–3.0 s on the stop clock vs 3.0–3.6 s after arrival, KS distance
0.06–0.12), so the early-in-stop rise is not an artifact of the bout clock. They
retain a **different** share of each population's event bins — **0.943**
detected SWR, **0.929** w/ SWR, **0.906** non-local, **0.889** no SWR (on all
stops: 0.912 / 0.902 / 0.850 / 0.814) — so a
between-population contrast on that ladder is partly a contrast between windows;
and they are heavily flagged (42.9–63.6 % of `M5a_time_arrival` fits carry a
separated non-nuisance term), so read them pooled, on the fitted curve, never
per session. The peak positions and KS distances quoted just above are
all-stops values and have not been re-derived on this cohort.

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
[audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md)
("is_swr-contrast"). Note also that separation is heavy there: **64.7 %** of the
contrast cohort's post-outcome fits and **68.4 %** of its primary fits carry a
separated non-nuisance term, and `is_swr:reward_history_rcos_1` is flagged in
**74.8 %** of the fits that carry it — so any re-derived aggregate must state
its excluded share.

## Layer 2 — SWR-coupling (ripple power as a continuous mark)

`ripple_power_z` (event-mean consensus ripple power — the signal the Kay detector
thresholds, so `is_swr` is its thresholded version). Per-event regression
(`scripts/rate/fit_swr_power_model.py`). **Scope note:** this layer was **not**
part of the 2026-09 refit wave — its numbers are unchanged and have not been
re-audited against the current tables, so treat them as provisional beside the
restated rate and content numbers.

- **Distribution is ~unimodal** (Sarle bimodality 0.524; no gap at the detection
  threshold) → the SWR/non-SWR split looks like a **cut on a continuum**, not two
  modes. *Caveat:* event-mean smears any gap; the **peri-onset peak** (the
  detector's actual criterion) is the definitive test and is **not yet run**.
- Not duration-confounded (`corr(power, duration)` = 0.04).
- **Reward (+0.141/SD, 5/5, p 5.5e-17)** and **RPE (5/5)** raise SWR power;
  **exploration lowers it (5/5)**. Combined with Layer 1 (the reconstructed
  occurrence kernel puts the immediate reward effect at RR 1.23 for detected
  SWRs, 1.41 for SWR-coupled non-local events and 1.09 for non-SWR non-local
  events), reward acts on **coupling** more than on bare occurrence —
  i.e. the binary "with-SWR rate" dissociation is largely reward pushing events'
  ripple power up across the threshold. *Caveat:* the power-model reward covariate
  is the **current-well** reward (corr 0.799 with current `reward_outcome`,
  dwell-confounded), so "reward raises coupling" may be partly a **consummatory
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
| current-arm bias, pooled | `C2_base` `is_current_arm` | **+0.708** (218 sessions, 5/5, q < 0.0001) |
| current-arm bias, non-SWR | same, nonSWR population | **+0.725** (215 sessions, 5/5) |
| current-arm bias, SWR | same, SWR population | **+0.797** (199 sessions, 5/5) |
| SWR less current-locked (direct) | `CS1_current` `swr_x_is_current_arm` | **−0.3730** (218 sessions, 0/5, q < 0.0001) |
| SWR less current-locked (total) | `CS1nt_current` | **−0.5985** (218 sessions, 0/5, q < 0.0001) |
| SWR less prev-locked (direct) | `CS2_prev` `swr_x_is_prev_arm` | **−0.6457** (217 sessions, 1/5, q < 0.0001) |
| arm reward fraction, non-SWR | `C3_armreward` `reward_fraction_candidate_arm_last10_visits` | **+0.0515** (215 sessions, 3/5, q 0.335) — null |
| arm reward fraction, SWR | same, SWR population | **−0.2139** (197 sessions, 1/5, q 0.0058) |
| SWR avoids recently-rewarded arms (direct) | `CS4_armreward` `swr_x_reward_fraction_candidate_arm_last10_visits` | **−1.1975** (218 sessions, 0/5, q < 0.0001) |
| ... (total) | `CS4nt_armreward` | **−1.3346** (218 sessions, 0/5, q < 0.0001) |
| current-patch bias | `CP0_base` `is_current_patch` | **+1.1116** (208 sessions, 5/5, q < 0.0001) |
| SWR less patch-locked (direct / total) | `CPS1_current` / `CPS1nt_current` `swr_x_is_current_patch` | **−0.0708** (216 sessions, 2/5, q 0.635) / **−0.2615** (218 sessions, 0/5, q < 0.0001) |

**The population-split rows and the `swr_x_...` rows are different estimands,
not a contradiction.** A `SWR` or `nonSWR` row is a *separately baselined* fit on
that population's own, unpaired, differently sized session set (199 vs 215 on
`C2_base`), so it answers "is the bias present in this population". A
`swr_x_...` row is the *pooled single-fit* SWR contrast against the shared
non-SWR baseline, so it answers "does SWR change it". Both `C2_base` splits being
strongly positive while `swr_x_is_current_arm` is negative is therefore
consistent, not conflicting.

So **non-SWR content is reward-value-neutral while with-SWR content avoids
recently rewarded arms**, and SWR content is less locked to the current arm, the
previous arm and the current patch — with the caveat that the patch-level
current contrast survives only as a **total** effect. The current-arm bias is
not an estimator artifact: count **+0.626**, fraction **+0.681**, DM **+0.737**,
all 5/5, fraction bootstrap [+0.405, +1.496] — the three-estimator run, which
predates the cohort refit; the refit fraction value is **+0.708**.

**"SWR broadens content", measured directly rather than inferred from arm
coefficients.** The Dirichlet-multinomial adds a within-event **concentration**
parameter the conditional estimators do not have. `log_phi_is_swr` is **+1.799**
on `C2_base` (animal bootstrap [+1.447, +2.205]), positive for **5/5 animals and
100 % of sessions**, and positive on every rung of the pooled table. Large φ
means the event's timebins sit closer to independent draws from its own arm
distribution and its mass is spread over more arms, so an SWR event's φ is
≈ e^1.80 ≈ **6×** a non-SWR event's. **Caveat — φ is assumed length-invariant.**
The likelihood uses event length only as the multinomial index, and length is
post-treatment with respect to `is_swr`: SWR events run **1.73×** longer (median
45 vs 26 bins). The length-adjusted twin (`_dm_len`) puts the contrast at
**+1.502** on `C2_base` (5/5, q 2.7e-37) — ≈ **4.5×** at the same length — with
the length term itself at **+0.522 per SD**. About a sixth of the total runs
through event length and the finding survives. Quote the total (+1.80) or the
length-adjusted (+1.50) figure, but say which. This is a statement about
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
| `C1_current` `is_current_arm` (no autocorrelation controls) | +0.8732 (218, 5/5) | +0.8999 (218, 5/5) | +0.8939 (218, 5/5) |
| `C4_armvalue` `reward_belief_mean_candidate_arm_preoutcome` | +0.1275 (217, 3/5, q 0.276) — null | −0.2912 (191, 2/5, q 0.290) | +0.0465 (209, 3/5, q 0.353) |
| `C6_rpe` `rpe_postoutcome_z` | −0.0739 (218, 0/5, q < 0.0001) | −0.0717 (197, 1/5, q 0.780) | −0.1284 (216, 0/5, q < 0.0001) |
| `C7_value` `relative_value_switch_minus_stay_preoutcome_z` | +0.0775 (218, 4/5, q < 0.0001) | +0.1080 (197, 5/5) | +0.1055 (215, 4/5, q 0.0009) |
| `C8_uncert` `reward_belief_differential_entropy_leaf_mean_preoutcome_z` | −0.0380 (218, 2/5, q 0.343) — null | +0.0265 (195, 3/5, q 0.178) | −0.0503 (216, 2/5, q 0.353) |
| `C9_update` `reward_belief_mean_update_l1_all_ports_postoutcome_z` | +0.2289 (218, 5/5, q < 0.0001) | +0.2279 (197, 5/5) | +0.1981 (217, 5/5) |
| `C10_periswitch` `periswitch_index_trials_z` | +0.2307 (197, 5/5, q 0.0001) | — | +0.0209 (174, 4/5, q 0.078) |
| `C11_block` `trials_since_block_change_z` | −0.0605 (218, 0/5, q < 0.0001) | −0.0868 (196, 1/5) | −0.0615 (214, 1/5, q 0.007) |
| `C12_switchrate` `patch_switch_fraction_last5_completed_trips_z` | +0.2520 (217, 5/5, q < 0.0001) | +0.2582 (194, 5/5) | +0.2350 (213, 5/5) |
| `C13_armuncert` `reward_belief_variance_candidate_arm_preoutcome_z` (per-arm main effect) | +0.3763 (217, 5/5, q < 0.0001) | +0.3680 (186, 5/5) | +0.2949 (210, 5/5) |

Readings. **Exploration locks content onto the current arm** (`C12`, 5/5 in both
populations) and **near a patch switch content moves off it** (`C10`, pooled
5/5, but the non-SWR split is a null at +0.02). **RPE pushes off current**
(`C6`, 0/5 positive pooled and in non-SWR). **Relative value ΔV sharpens onto
current** (`C7`) — but read the sign with care: ΔV is *Switch − Stay*, so a
positive coefficient means content sharpens onto the current arm when that arm's
value is **low**. **Value-update sharpens onto current** (`C9`, 5/5
everywhere). The **scalar belief-entropy interaction is null** (`C8`) — the
per-arm uncertainty *main effect* (`C13`) is the content-uncertainty result; do
not conflate them. The **arm-value main effect depends on how the fit is specified**, not on the arm
support (`C4_armvalue` and `CS5_armvalue` are both leaf-only): in the *split* `C4`
fits it is a null (all +0.1275, 217 sessions, 3/5, q 0.276; non-SWR +0.0465, 209, 3/5;
SWR −0.2912, 191, 2/5), whereas in the *pooled* `CS5` fit — one model over both event
classes with the `swr_x_` interaction, so the main term is the non-SWR reference level — it is
**positive**: `CS5_armvalue`'s main term
`reward_belief_mean_candidate_arm_preoutcome` reads **+0.264** (fraction, 217
sessions, **5/5 animals positive**, bootstrap [+0.031, +0.818]; count +0.228,
5/5; DM +0.168, 4/5) and the leaf bundle's `Mv` reads **+0.2512** (215 sessions,
4/5). So the value story is a positive main effect in the pooled fit *plus* the
SWR interaction below — do not write that it lives entirely in the interaction,
and do not write that content is not value-selective.

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
  `reward_belief_variance_candidate_arm_preoutcome_z` is **+0.3509** (session
  median over 215 sessions, **5/5 animals positive**, median event-clustered SE
  0.182). This is the exploration / gain axis (Mattar-Daw) the
  value/reward/space ladder was missing.
- **It earns its parameter where value does not — but value's main effect is
  still positive.** Median per-session ΔAIC against the nested parent (positive
  = child wins): `M4 → Mu` **+1.80** (ΔAIC > 2 in 103 of 215 sessions, 5/5
  animals child-better), against `M4 → Mv` **−2.34** (ΔAIC > 2 in 20/215, 0/5
  animals) and `Mu → Mvu` **−1.61** (ΔAIC > 2 in 11/215, 0/5). The session count
  is the number of sessions clearing ΔAIC > 2, not a bare sign count. On the
  same leaf design the value *coefficient* is positive — `Mv`
  `reward_belief_mean_candidate_arm_preoutcome` **+0.2512** (215 sessions, 4/5)
  and the `CS5_armvalue` main term **+0.264** (fraction, 217 sessions, 5/5,
  bootstrap [+0.031, +0.818]) — so the contrast between the two axes is a model
  comparison, not a difference in sign.
- **Both populations over-represent uncertain arms, SWR more so.** The split
  `C13_armuncert` gives **SWR +0.3680** (186 sessions, 5/5) against **non-SWR
  +0.2949** (210 sessions, 5/5), pooled `all` **+0.3763** (217 sessions, 5/5),
  every q < 0.0001.
- **The pooled `swr_x` interaction is null as a direct effect and positive as a
  total effect, and the two estimators agree on each.** *Direct*
  `CS6_armuncert` — fraction **+0.0039** (217 sessions, 3/5, q 0.343) and DM
  **−0.027** (2/5, q 0.564). *Total* `CS6nt_armuncert` — fraction **+0.1807**
  (218 sessions, **5/5**, q < 0.0001) and DM **+0.146** (4/5, q 7.5e-12). So the
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
2.5–97.5, on the fully paired 216-session cohort:

| estimand | count (retired) | fraction (shipped) | Dirichlet-multinomial |
| --- | --- | --- | --- |
| **direct** (`CS5_armvalue`, length-adjusted) | −0.398 (1/5) [−0.904, +0.199] | **+0.237 (3/5) [−1.322, +0.539]** | −0.316 (0/5) [−1.136, −0.057] |
| **total** (`CS5nt_armvalue`) | — | **−0.6506** (1/5, q < 0.0001) | **−0.638** (0/5, q 7.9e-10) |

The shipped audited fraction fit reads **+0.2363** (216 sessions, 3/5, q 0.117);
the DM's direct q is 0.025. **By current outcome**, on the fraction response: the
**rewarded** stratum gives direct `CS5_armvalue` **+0.1303** (205 sessions, 3/5
animals positive) and total `CS5nt_armvalue` **−0.3514** (206 sessions, 0/5) —
the same direct-positive / total-negative split as the pooled fit. The
**omitted** stratum is **not fit for these two rungs**, so there is no
omission-side value number. **The DIRECT effect's sign is estimator-dependent** —
count and DM negative, the shipped fraction estimator positive; the three
bootstrap intervals overlap heavily, only the DM's excludes zero and only the DM
has 0/5 animals positive — **while the TOTAL (`nt`) effect is negative under
both estimators that fit it.** Report it as a three-way comparison; do not write
"the DM confirms value avoidance" without saying that the shipped estimator does
not.

The rung is also barely worth its parameters on either honest scale. Median
per-session ΔAIC against `C2_base_leaf` — **content convention: positive = child
better**, the opposite of the rate ΔAIC convention used in Layer 1 — is
**+2.524** (fraction) and **+2.252** (DM), against **−85.415** on the retired
count scale. The count figure is the timebin-pseudoreplication artefact, not an
effect.

**Where the count-response avoidance went** (response x design factorial,
session-median coefficient): count/`NT` **−0.980** (0/5 animals positive) →
count/full **−0.398** (1/5) → fraction/full **−0.114** (3/5). The event-length
adjustment and the response change each remove roughly half and act
approximately additively (additive prediction +0.03 against the observed −0.11);
`chimi` and `wilbur` are negative in every cell. Read the table as "the
count-response avoidance does not survive either change", **not** as evidence of
a positive effect.

`CS5b_valuevsprevgoal`, the previous-goal control rung, is **refused as
unidentified**: 133 sessions with **95 flagged separated** under the DM (−0.315,
2/5, q 0.297); the fraction fit has 134 sessions (+0.1855, 3/5, q 0.378).

### Prediction models — does content predict the next choice?

The behavioral-relevance layer, and the weakest link. `replay_fraction` over a
behavioral baseline (conditional logit): non-SWR **+0.34 (p 0.019)**, SWR
**null** — but the model **drops `dist_to_arm`** (quasi-separation: the animal
picks the closest arm 95–99 % of the time), so it is **not distance-controlled**
and the +0.34 may be current/nearest-arm bias. Content does **not** predict
stay/leave (null across strata). Raw `replay_fraction` is *negative* (−0.94,
5/5 — the current-arm bias: the animal leaves the arm it most replays), flipping
to **+0.38** over the behavior baseline (4/5 animals, but only 13 % of sessions
individually significant). The fits are **in-sample only — no held-out CV** —
and the SWR split is underpowered (n 214 → 183). So: a weak, non-SWR-carried,
sign-flips-on-control, in-sample-only choice correlation, not a cross-validated
prediction. These fits were **not** part of the 2026-09 refit wave and have not
been re-derived on the fraction response.

### Spatial FORM of SWR vs non-SWR content: wells vs paths, and trajectory start/end

*(Content-occupancy and trajectory rollups. Neither is a content-ladder fit and
neither was part of the 2026-09 refit wave.)*

**Correction to the "non-SWR is current-focused" framing.** The arm-model
`is_current_arm`
coefficient is a *within-event relative* over-weighting (conditional Poisson conditions on
each event's total, so event-constants cancel and coefficients are compositional, not
absolute). The **content-occupancy** analysis (`scripts/content/analyze_content_occupancy.py`,
218 sessions) gives the absolute picture: **non-SWR content is reward-well-dominated
(~67% at wells: ~37% remote reward wells + ~30% own well; ~33% path); SWR content is
path-dominated (~78% path vs 22% at-well)** — SWR-vs-non-SWR path fraction 0.78 vs 0.32,
**p=1.6e-37, 5/5 animals**. So non-SWR represents **reward-well *locations*** (current *and
remote*), SWR represents **paths**.

- **How "at well" vs "path" is defined** (note the caveat): per non-local *immobile timebin*,
  take the **MAP** decode; `d_well` = track-graph distance to the nearest reward well (well =
  the animal's median immobile position on each leaf arm); **`d_well < 20 cm` = "at a well",
  `≥ 20 cm` = "path"** (crossed with `d_animal`, 20 cm, for remote vs own). So **"path" means
  ">20 cm from any reward port" — it includes mid-leaf-arm, NOT only trunk/inter-patch arms**,
  and it is MAP-based (not posterior mass). The clean, artifact-immune part is `remote_path`
  (far from animal *and* wells). The strict trunk-vs-leaf version is the separate
  `swr_x_trunk` 3.55× enrichment.
- **Retraction:** `reward_belief_variance_candidate_arm_preoutcome` is **0 on all
  trunk arms** (a per-*leaf* quantity), so the
  `C13`/`Mu` uncertainty effects are estimated **only on well/leaf arms** — we have **no
  evidence** on whether SWR *path* content relates to uncertainty. The earlier "SWR extends
  uncertainty to the paths toward those wells" was unsupported; testing it needs a path/trunk
  uncertainty covariate (not built).

**Trajectory start/end IS characterized** (per-event rollups from `analysis/replay/trajectory.py`,
in `event_table`: `start_arm`/`end_arm`, `start_position`/`end_position`, `net_displacement`,
`spatial_extent`, `step_distance_*`, `max_ahead`/`max_behind`/`ab_slope`, `n_reversals`,
`ends_at_next`). A **quick SWR split (not yet a formal analysis)**:

| metric (mean) | non-SWR | SWR |
|---|---|---|
| spatial_extent (cm) | 26 | **85** |
| net_displacement (cm) | 17 | 48 |
| step_distance_max | 20 | 67 |
| n_reversals | 3.8 | 10.9 |
| max_ahead / max_behind | 18 / 114 | 14 / 145 |
| ends_at_next choice | 0.12 | 0.07 |

→ **SWR content = extended, multi-reversal trajectories; non-SWR = localized at wells.** Both
are **predominantly *behind* the animal** (max_behind ≫ max_ahead → retrospective), and content
**rarely ends at the next chosen arm** (7–12%) — consistent with not-choice-predictive. A
formal SWR-split trajectory characterization has **not** been run/reported yet.

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
control rung `CS5b_valuevsprevgoal` fits on only **133 sessions with 95 flagged
separated** under the DM (−0.315, 2/5, q 0.297; the fraction fit has 134
sessions, +0.1855, 3/5, q 0.378). The earlier avoidance coefficients (−1.485 at
the arm grain, −1.62 SWR-split, −1.49 patch-grain) and the −0.79
SWR-by-previous-goal interaction are **withdrawn**. Neither a divergence nor a
replication can be claimed from this rung as it stands.

**The graded-value discriminant is retired as stated.** It rested on
`CS5b_valuevsprevgoal` leaving the SWR by arm-value interaction unmoved at
−2.45 → −2.43. Both numbers came from the retired count response and the rung is
unidentified, so the discriminant is withdrawn. What remains is the arm-value
result itself: **estimator-dependent** in its direct form and negative in its
total form (−0.6506 fraction / −0.638 DM; see the CS5 section above).
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
| immediate reward (kernel lag 1) | non-SWR **1.09×** vs with-SWR **1.41×** (both respond; SWR-coupled more); paired in 69.4 % of 216 sessions, 5/5 animals, p 1.1e-11 | the current-arm interaction is **PENDING** (count-response fit withdrawn) |
| recent reward (kernel lags 3–10) | **shared**: RR ≈ 0.89–0.97 in all four populations | arm reward fraction non-SWR **+0.05 (null)** vs SWR **−0.23**; SWR contrast **−1.20** direct / **−1.33** total |
| arm value | not in the shipped rate ladder (**pending**) | main effect null (+0.13 split); the direct SWR interaction is **estimator-dependent** (fraction +0.24 / count −0.40 / DM −0.32), the total is **−0.65 / −0.64** |
| arm uncertainty | not in the shipped rate ladder (**pending**) | both populations over-represent uncertain arms, **SWR +0.37 > non-SWR +0.28**; pooled SWR interaction null direct, **+0.18** total |
| exploration (patch-switch fraction) | **pending** | **locks content onto the current arm** in both: +0.27 pooled, +0.26 SWR, +0.20 non-SWR (5/5 each) |
| within-stop time | rate rises over ~1.5–2 s then plateaus in every population | the current-arm interaction is **pending** |

**The headline, restated: one half solid, one half estimand-dependent.**

> Reward gates with-SWR **occurrence**; the content half of the old double
> dissociation is not currently quotable as a coefficient.

- **Occurrence (solid).** The immediate reward component dissociates — kernel
  lag 1 RR **1.41** (SWR-coupled non-local) vs **1.09** (non-SWR), paired in
  **69.4 % of 216 sessions, 5/5 animals, p 1.1e-11** — while the slower
  recent-reward component is **shared** by all four populations. So non-SWR
  events are a **less consummatory/reward-coupled mode** that obeys the same
  slower recent-reward baseline. Both streams respond to immediate reward on the
  shared cohort; the contrast is of degree.
- **Content (restated).** SWR content is less current-locked (**−0.37** direct /
  **−0.60** total) and less prev-locked (**−0.64**), avoids recently rewarded
  arms (**−1.20** / **−1.33**), and is **broader within the event**
  (`log_phi_is_swr` **+1.80** total, **+1.50** length-adjusted, 5/5 animals,
  100 % of sessions). Whether it also avoids high-**value** arms is
  estimator-dependent in the direct form and negative in the total form.
- **Shared.** Exploration concentrates content on the current arm in both
  populations; both over-represent uncertain arms.
- **Withdrawn.** "Reward pushes non-SWR content off the current arm" and its
  with-SWR null — both from the retired count response.

## Confirmed (robust: multi-control and/or 5/5 animals)

1. **Rate baseline ordering and the dominant structure.** Non-local (all)
   **0.407** > detected SWR **0.289** > non-local no-SWR **0.225** > non-local
   w/ SWR **0.184** events/s. The within-stop time course is what calibrates the
   process (KS-in-band 0.44 → 0.91 at `M5_time`), with self-history clustering
   and a smooth reward kernel on top.
2. **The immediate-reward dissociation, from two distinct measurements.** The
   reconstructed kernel puts lag 1 at RR **1.41** (non-local w/ SWR) against
   **1.09** (non-local, no SWR) — *how large* the effect is. Separately, the
   **paired within-session** test of the immediate basis coefficient between the
   two streams is positive in **69.4 % of 216 paired sessions, 5/5 animals,
   signed-rank p = 1.1e-11** — *how consistently* they differ. Do not read the
   percentage as a property of the kernel values or vice versa. The slower
   recent-reward component (lags 3–10, RR ≈ 0.89–0.97) is **shared** across all
   four populations.
3. **Events are strongly non-Poisson** — refractory then bursty, peak RR
   **1.57–2.59** by population, SWR-coupled non-local events the most tightly
   bursty and non-SWR the flattest.
4. **Content current-arm bias.** `C2_base` `is_current_arm` **+0.708** (218
   sessions, 5/5, q < 0.0001), present in both populations (non-SWR +0.725 /
   SWR +0.797) and sign-consistent across all three estimators (count +0.626 /
   fraction +0.681 / DM +0.737 on the pre-refit three-estimator run).
5. **SWR content is less current- and prev-locked, and broader within the
   event.** `swr_x_is_current_arm` **−0.3730** direct / **−0.5985** total;
   `swr_x_is_prev_arm` **−0.6457**; `log_phi_is_swr` **+1.799** (5/5 animals,
   100 % of sessions), **+1.502** length-adjusted.
6. **Non-SWR content is neutral to *recently rewarded* arms; with-SWR content
   avoids them.** Measured on `reward_fraction_candidate_arm_last10_visits` —
   the realized reward fraction over the last ten visits to that arm — **not**
   on the belief mean `reward_belief_mean_candidate_arm_preoutcome`, whose main
   effect is positive (item 7). `C3_armreward` non-SWR **+0.0515** (215
   sessions, 3/5, q 0.336) against SWR **−0.2139** (197 sessions, **4/5 animals
   negative**, 1/5 positive, q 0.0058); the pooled SWR interaction is
   **−1.1975** direct (218 sessions, **0/5** animals positive) / **−1.3346**
   total and negative under all three estimators. The 0/5 belongs to the pooled
   `CS4` interaction, not to the split `C3` SWR coefficient.
7. **Content over-represents high-uncertainty arms.** `Mu`
   `reward_belief_variance_candidate_arm_preoutcome_z` **+0.3509** (**session
   median** over 215 sessions, 5/5 animals positive); split `C13_armuncert` SWR **+0.3680** vs non-SWR **+0.2949**
   (5/5 each). The rung earns its parameter (`M4 → Mu` ΔAIC **+1.80**, ΔAIC > 2
   in 103/215 sessions, 5/5 animals child-better) where the value rung does not
   (`M4 → Mv` **−2.34**, ΔAIC > 2 in 20/215, 0/5 animals). **Value's main effect
   is nonetheless positive on this same leaf design** — `CS5_armvalue`'s
   `reward_belief_mean_candidate_arm_preoutcome` **+0.264** (217 sessions, 5/5,
   bootstrap [+0.031, +0.818]), `Mv` **+0.2512** (215 sessions, 4/5) — so the
   two axes are separated by model comparison, not by the sign of value.
8. **Exploration concentrates content on the current arm** in both populations
   (`C12_switchrate` **+0.2520** pooled, **+0.2582** SWR, **+0.2350** non-SWR,
   5/5 each).
9. **The reward-driven off-current content is spike-supported, not a decode /
   consummation artifact.** Reality engine (5 sessions): post-reward off-current
   `dll_per_spike` = 0.63, **99 % > 0** vs baseline ≈ 0; rewarded ≈ omitted
   (0.63 vs 0.65). *(The content coefficient this was built to check — the
   reward-by-current-arm interaction — is itself pending refit, so this stands
   as a reality check on the phenomenon, not on a current number.)*
10. **Off-current content's destination is the "novel" patch.** On patch-switch
    events the off-current content prefers the third (other) patch over the
    previous (non-SWR 0.24 vs 0.19; with-SWR 0.23 vs 0.16). Not a content-ladder
    fit and not part of the 2026-09 refit wave.

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
  cannot identify it: 84% alternation makes `is_next_arm`/`is_prev_arm` collinear
  (r=0.81). No
  covariate separates prospective from retrospective on this track.
- **Graded ripple-power content model — NOT a committed rung.** Graded
  `ripple_power × current` beats binary SWR (97% sessions, −291 AIC) but lives only in
  an exploratory script, not the ladder.
- **Content choice-mediation** (does rate × content predict the next choice). The
  prior choice correlation (#19d) is modest and **not distance-controlled** (`dist_to_arm`
  dropped for quasi-separation); stay/leave is null (#19e).
- **Per-arm realized-gain: tested under the retired HMM covariate and the
  retired count response, and negative.** Its coefficient is **withdrawn**; the
  reading it supported — that the gain axis of content is epistemic uncertainty
  rather than realized value-update — survives only as a hypothesis. The
  beta per-arm value-update re-run has not been done.
- **Pooled `swr_x` contrasts on arm-identity-tied covariates: read the split as
  well.** The earlier fixed-effect-artifact diagnosis of the pooled uncertainty
  interaction is superseded — on the leaf-only fraction fit the *direct*
  interaction is simply a null (+0.0039, 3/5) while the *total* twin is positive
  (+0.1807, 5/5). Report the split (`C13_armuncert`) alongside the pooled pair,
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
- **Dwell / within-stop time** (rewarded stops are 3.6× longer on the mean, 1.858 s
  vs 0.519 s): `BASEc` carries
  `log1p_time_since_stop_start_s_z_x_is_current_arm`, and the response x design
  factorial shows this term **changes nothing** on the CS5 coefficient — it is the
  event-length products, not the within-stop clock, that move it. Control is
  linear in `z(log1p t)`; a spline would be stricter.
- **Event length is post-treatment, not a nuisance.** The `BASEc` length products
  make a `swr_x_...` coefficient a **direct** effect; the `nt` twin gives the
  **total**. Neither is "the" answer — see the estimand bullet in *Provenance*.
- **Collinearity:** the trial-level reward-rate covariate (the mean of the reward
  lags, VIF infinite) was removed; the reward-history block is now a smooth
  three-feature basis whose coefficients are collinear **by construction**, which
  is why the kernel is reconstructed rather than read term by term.
- **Separation:** the whole last-rewarded family is **refused as unidentified**
  in the per-session fits, and `CS5b_valuevsprevgoal` fits on 133 sessions with
  95 flagged separated. Per-coefficient separation flags and a
  `|coef| x SD >= 10` magnitude backstop now decide which session fits enter an
  aggregate; a rung's excluded share must be stated with its coefficient.
- **Circularity** (content from the same decoder that defines events): both the
  baseline current-arm claim *and* the new reward-driven off-current content
  clear the spike-evidence engine (post-reward off-current `dll_per_spike` 0.63,
  99% > 0 vs baseline ≈ 0; **5 sessions, in-principle validation**). The
  **full-dataset** reality run is deliberately **gated on the planned full
  re-decode**, not run now: (i) the re-decode persists `log_likelihood` to
  `results.nc` so reality is a cheap direct read rather than a ~20 min/session
  recompute; (ii) the current decodes are stale (the position-trim drift), so a
  full run now would validate the wrong decodes; (iii) the close-event W3
  (place-field-extent) worry needs the local-kernel re-decode regardless. See the
  reality design spec (TODO §2, "gated on a first re-decoded session").

## Reproducibility (scripts & key commits)

- Rate: `scripts/rate/fit_per_bin_rate_models.py`,
  `scripts/rate/fit_per_stop_rate_models.py`, `scripts/rate/fit_uncertainty_rate.py`,
  `scripts/rate/fit_swr_contrast_rate.py`; summaries
  `scripts/rate/summarize_per_bin_rate_models.py`, plots
  `scripts/rate/plot_per_bin_rate_models.py`.
- Coupling: `scripts/rate/fit_swr_power_model.py`.
- Content: `scripts/content/fit_replay_content_models.py`,
  `scripts/content/fit_swr_contrast_content.py`,
  `scripts/content/compare_content_estimators.py`;
  audits by `scripts/qa/compare_fits.py`.
- Contingency: `scripts/behavior/fetch_spatial_bandit_behavior.sh`,
  `scripts/_bandit_csv_contingency.py`.
- Reality check on reward-modulated content: `scripts/qa/analyze_event_spike_evidence.py`
  (now emits per-event `start_time`) + `scripts/exploratory/_reality_reward_test.py`;
  content destination: `scripts/exploratory/_content_patch_destination.py`.
- Audits behind the current numbers:
  [audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md),
  [audits/2026-09-05-content-refit-comparison.md](audits/2026-09-05-content-refit-comparison.md),
  [audits/2026-09-05-cs5-response-by-design-factorial.md](audits/2026-09-05-cs5-response-by-design-factorial.md),
  [design/content_estimator_comparison.md](design/content_estimator_comparison.md).
