# Per-bin replay-rate ladder: fitted effects (Family B)

**Fit date: 2026-09-12. Cohort: well-anchored stops.** Quantitative readout of
the per-bin
conditional-intensity ladder (`MODEL_PER_BIN_RATE`, see
[rate_models.md](rate_models.md) for the methods), fit per session over the
**218-session corpus** and persisted in
`Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`
(`BIN_S = 0.015`, `TMAX = 4 s`, 10 rungs — the seven stop-clock rungs read
below plus the three arrival-anchored ones). Every number here is transcribed
from `uv run python scripts/rate/summarize_per_bin_rate_models.py` run against
that artifact, or read from the artifact directly where the section says so.

The population is the **well-anchored** stops — `occupancy_state ∈ {at_well,
mixed}`, **84,069** of the 107,400 canonical stops, of which **83,549** (over
**6,777,953** bins) survive the common-cohort freeze — which is the same
population the content family is fit on at the event grain
([design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md)). The
all-stops fit is kept as a sensitivity bundle (`all_models_allstops.pkl`) and
in-transit stops as their own (`all_models_intransit.pkl`); the three are placed
side by side in
[audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md).
**The in-transit column there is thin** — only 50 of 218 sessions clear the
per-bin sample gates on any population — so an in-transit cell showing no effect
is an absence of evidence, not evidence of absence.

Not every population carries a fit in every session: the full additive rung
`M7r_time_histcos_rew` is fit on **218 (detected SWR) / 217 (non-local, all) /
216 (non-local w/ SWR) / 218 (non-local, no SWR)** sessions. Per-section `n`
states which.

Three things are load-bearing for everything below.
(i) The reward-history features are **recomputed at the outcome boundary** for
the 13.4 % of stops that cross one, so a bin after the boundary carries the
outcome the animal has by then learned. (ii) The five `onset_lag1..10` dummy
rungs were **retired** — they could not be cross-validated — and the ladder now
runs `M0_const`, `M1_rew`, `M5_time`, `M3r_histcos`, `M6r_time_histcos`,
`M7r_time_histcos_rew`, `M9r_rew_x_time` on the stop clock. (iii) The cohort is
the well-anchored one, which is what most of the numbers below moved with: they
were previously measured over every immobility stop. All three are described
in [rate_models.md](rate_models.md).

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
(`non_local_swr` / `non_local_no_swr`). "RR" = rate ratio = `exp(coef)`;
summaries are across-session medians with per-animal sign consistency (n/5) and
a signed-rank p. Every effect is *conditional on the first 4 s of each stop*,
measured from a speed-threshold crossing.

## 1. Baseline rate (M0_const)

Events per second of immobility (`exp(intercept)`), median across 218 sessions:

| population | rate (events/s) | IQR |
| --- | --- | --- |
| non-local (all) | **0.407** | 0.319–0.534 |
| detected SWR | 0.289 | 0.223–0.346 |
| non-local, no SWR | 0.225 | 0.156–0.286 |
| non-local, w/ SWR | **0.184** | 0.137–0.249 |

Ordering: **non-local (all) > detected SWR > NL-no-SWR > NL-w/SWR.** The
pairwise contrasts are median *within-session* rate ratios with a paired
Wilcoxon p, and all four are significant:

| contrast | median rate ratio | p |
| --- | --- | --- |
| NL w/ SWR / NL no-SWR | 0.83× | 5.1e-09 |
| detected SWR / non-local (all) | 0.63× | 1.3e-24 |
| NL w/ SWR / detected SWR | 0.75× | 9.6e-37 |
| NL no-SWR / detected SWR | 0.81× | 2.4e-09 |

The non-local split is roughly additive (0.184 + 0.225 ≈ 0.407), so
**SWR-coupled non-local events are the rarest type**. Read the coupling shares
off the within-session ratios, not off the ratio of the medians: the median
session has **0.75** SWR-coupled non-local events per detected SWR, and
SWR-coupled events are **0.45** of its non-local events
(`M0_const` intercepts in `all_models.pkl`). (Detected-SWR vs non-local is a
cross-modality comparison — LFP vs decoder — so read the within-non-local split
as the clean contrast.)

## 1b. Within-stop time course (the `bs(tc)` spline)

The dominant structure: replay rate rises over the first ~1.5–2 s of a stop then
plateaus (panels A0/A1 of the summary figure). The spline uses **fixed shared
knots** (`SPLINE_KNOTS=(1,2,3)`, bounds `[0,4]`), so the fitted `λ(tc)` curve is
**reconstructable from the saved coefficients** (`exp(intercept + bs(tc)·β)`)
and the spline coefficients are comparable across sessions. Read the curve only
**between the outermost interior knots**, [1, 3] s: past the last interior knot
a cubic B-spline extrapolates over very little exposure and can run away. On
that interval the across-session median peak sits at 2.63 s (detected SWR),
2.82 s (non-local, all), 2.31 s (non-local w/ SWR) and 2.68 s (non-local, no
SWR), each read over 168–217 unflagged fits
([audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md),
"Burst peak of the smooth time course"). On the all-stops fit the non-SWR peak
sat exactly on the right edge of the readable interval (3.0 s) and could only be
read as "at or beyond 3 s"; on the well-anchored cohort it is inside the
interval and is a located maximum.

**Duration caveat (important):** stop durations are very unequal and
*endogenous* — rewarded stops are **3.6× longer on the mean** (1.858 s vs
0.519 s; 2.3× on the median, 0.518 s vs 0.228 s, `stop_table.parquet`) — so at
large `tc` only the long
(≈rewarded) stops survive: 20 % of stops are still immobile at 2 s and 12 % at
4 s. The exposure offset makes `λ(tc)` a proper rate, but the curve at large
`tc` reflects a **selected subpopulation**, not the same stops over time. The
figure marks this with a survival overlay and grey shading where <20 % survive
(~beyond 2 s). The fitted panel (A1) uses `M7r_time_histcos_rew` (reward and
self-history controlled) so much of the rewarded-long-stop composition is
removed, but the principled fix is to model the stop *departure hazard* jointly
with the rate.

**The arrival-clock variant.** `tc` measures time since a speed-threshold
crossing, and most stops are fragments of a well visit rather than the visit
itself, so the sister rungs `M0a_const_arrival` / `M5a_time_arrival` /
`M7ra_time_arrival_histcos_rew` re-anchor the same spline on time since well
arrival over `[0, 8]` s. They are fit on a strict subset of this ladder's rows
and are therefore **never differenced against it** — read the two side by side.
On their supported interior, [1, 6] s, the two clocks put the median peak at
broadly the same delay in every population: on this cohort 2.31–2.82 s on `tc`
against 2.80–3.80 s after arrival, KS distance 0.06–0.11 between the normalized
curves (on all stops, 2.2–3.0 s against 3.0–3.6 s, KS 0.06–0.12 — the statement
does not turn on the cohort; both columns are in
[audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md)).
The arrival
rungs are heavily flagged (42.9–63.6 % of `M5a_time_arrival` fits carry a
separated non-nuisance term) and are rarely cross-validated, so they are read
pooled, on their fitted curve, never per session; see
[rate_models.md](rate_models.md). What the cohort did change is retention: the
arrival window now keeps 88.9–94.3 % of each population's event bins, against
81.4–91.2 % on all stops.

## 2. Immediate reward component (basis bump 0)

`reward_history_rcos_0` is the bump-weighted reward *fraction* over roughly the
last seven known outcomes (weights 1.00, 0.79, 0.54, … at lags 1–7), so its
coefficient is **per unit of that fraction** — a 0→1 contrast, not a
single-trial indicator. The three basis coefficients are strongly collinear, so
read the *kernel reconstruction* in §3 for the per-lag effect and treat the
single-coefficient RRs below as a within-population contrast only.

| population | RR raw (M1_rew) | RR controlled (M7r) | %pos (M7r) | animals (M7r) | p (M7r) | n |
| --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 20.0 | 3.44 | 76 % | 5/5 | 3.5e-19 | 218 |
| non-local (all) | 32.4 | 3.44 | 81 % | 5/5 | 3.0e-21 | 217 |
| non-local, w/ SWR | **99.5** | **7.18** | 85 % | 5/5 | 1.3e-26 | 216 |
| non-local, no SWR | **18.8** | **2.00** | 66 % | 5/5 | 3.1e-06 | 218 |

Controlled = `M7r_time_histcos_rew` (within-stop time spline + raised-cosine
self-history + reward basis). The paired NL-w/SWR > NL-no-SWR dissociation is
**69.4 % of sessions, 5/5 animals, signed-rank p = 1.1e-11** over **216 paired
sessions**. The non-SWR immediate component is now a real response of its own
(RR 2.00, p = 3.1e-06, 5/5 animals) rather than the null it read as on all
stops (RR 1.24, p = 0.22, 4/5) — the dissociation is between two responding
streams, one about 3.6× the other, not between a response and its absence.

These are across-session medians. The audited aggregate summarizes the same
coefficient animal-balanced, where it reads **4.91 / 2.96 / 6.86 / 2.28** in the
same row order (`rr_immediate`, 5/5 animals each). Say which summary a quoted
number is.

The other two bumps (`M7r`, same fits) run the other way and are also
consistent — the middle bump is far below 1 and the slowest bump above it,
which is what produces the dip-then-recovery shape of §3:

| population | bump 1 RR | %pos | animals | bump 2 RR | %pos | animals |
| --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 0.05 | 21 % | 5/5 | 5.39 | 72 % | 5/5 |
| non-local (all) | 0.04 | 17 % | 5/5 | 5.19 | 80 % | 5/5 |
| non-local, w/ SWR | 0.01 | 15 % | 5/5 | 13.84 | 80 % | 5/5 |
| non-local, no SWR | 0.08 | 31 % | 5/5 | 3.29 | 67 % | 5/5 |

None of these three is a per-lag effect on its own; they are one kernel.

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
| detected SWR | 1.23 | 0.99 | 0.93 | 0.90 | 0.91 | 0.92 | 0.94 | 0.97 | 1.00 | 1.03 |
| non-local (all) | 1.21 | 0.99 | 0.92 | 0.90 | 0.91 | 0.92 | 0.94 | 0.98 | 1.00 | 1.04 |
| non-local, w/ SWR | **1.41** | 1.05 | 0.93 | 0.89 | 0.90 | 0.91 | 0.94 | 0.98 | 1.02 | 1.06 |
| non-local, no SWR | **1.09** | 0.95 | 0.92 | 0.91 | 0.92 | 0.93 | 0.95 | 0.97 | 0.98 | 1.00 |

So the reward kernel is **a boost at lag 1 that dissociates by SWR coupling
(1.41 SWR-coupled vs 1.09 non-SWR), then a mild suppression from lag 3 on
(RR ≈ 0.89–0.97, shared across all four populations) that returns to ~1 by lag
10**. On all stops the non-SWR lag-1 value was 0.99 — no boost at all — so the
qualitative statement "non-SWR replay is not reward-gated" belongs to that
population and not to this one. The smooth basis has no per-lag parameter and cannot represent a
lag-specific spike, so read the shape off the reconstruction rather than off any
one coefficient. The lag≥2 suppression is a recent-reward effect; whether it is
an RPE / relative-reward signal or satiety / adaptation is **not** established
here — this GLM uses past binary outcomes, not a prediction error.

## 4. Self-history (within-stop autocorrelation)

Reconstructed from the four `hist_rcos_k` coefficients of
`M7r_time_histcos_rew` the same way §3 reconstructs the reward kernel, over ten
15 ms lags. The shape is qualitatively the same for every population —
**refractory** (RR < 1) at short lags, **bursty** (RR > 1) at ~120–150 ms — but
the microstructure dissociates:

| population | refractory window (RR<1) | burst peak RR | @ lag | n |
| --- | --- | --- | --- | --- |
| detected SWR | ≤60 ms | **2.59** | 135 ms | 218 |
| non-local (all) | ≤60 ms | 2.02 | 120 ms | 217 |
| non-local, w/ SWR | **≤45 ms** | 2.43 | 150 ms | 216 |
| non-local, no SWR | ≤60 ms | **1.57** | 105 ms | 218 |

**SWR-coupled non-local events are the more tightly bursty stream** — the
shortest dead time (≤45 ms) and a 2.43× peak — while **non-SWR non-local events
are the flattest** (1.57× peak, and the earliest, at 105 ms). Events are
strongly non-Poisson in every population. Two cautions: the raised-cosine basis
is smooth, so a single-lag notch cannot appear in it and the tabulated
refractory window is a resolution-limited "≤", not a measured dead time; and
these are the *controlled* kernels (`M7r`), read after the time spline and the
reward block. The uncontrolled `M3r_histcos` kernels are larger throughout
(peaks 3.45 / 3.02 / 4.04 / 2.18× in the same row order), so do not mix the two.

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
curve stays inside the band. Every stop-clock rung is fit on all 218 non-local
sessions except `M7r_time_histcos_rew` (217); the arrival rungs are **not** in
this table because they are fit on a subset of the rows and their AIC is not on
this scale.

`n_cv` is the number of those sessions whose CV metric is valid: a fold in which
any non-nuisance term is unidentified voids the whole metric, so a rung with a
small `n_cv` is summarized over a **selected minority** of sessions and its
bits/event is not comparable with a rung scored on the full corpus. (`n_cv` is
read from `all_models.pkl`; the other three columns are the summarizer's.)

| model | median AIC | CV bits/event | n_cv | KS-in-band |
| --- | --- | --- | --- | --- |
| M0_const | 2172 | 0.000 | 218 | 0.44 |
| M1_rew | 2159 | 0.032 | 214 | 0.38 |
| M5_time | 2083 | **0.266** | 208 | 0.91 |
| M3r_histcos | 2165 | 0.044 | 75 | 0.33 |
| M6r_time_histcos | 2078 | 0.262 | 73 | 0.91 |
| **M7r_time_histcos_rew** | **2077** | 0.244 | 74 | **0.93** |
| M9r_rew_x_time | 2083 | 0.200 | 43 | 0.91 |

- **The time spline is what calibrates the process** (KS-in-band jumps
  0.44 → 0.91 at `M5_time`, and neither reward nor self-history fixes the
  constant model without it: 0.38 and 0.33).
- **`M5_time` is the best-generalizing model**, and the only rung with both a
  large held-out gain and a near-complete CV cohort (0.266 bits/event
  over 208 of 218 sessions). The three rungs carrying time **and** the
  raised-cosine self-history block — `M6r_time_histcos` (0.262/73),
  `M7r_time_histcos_rew` (0.244/74), `M9r_rew_x_time` (0.200/43) — score
  0.200–0.262 bits/event on 43–74 sessions each, so their CV is **not** a
  like-for-like comparison against `M5_time`'s: a rung is not "better on
  held-out data" because a selected third of sessions scored well.
  `M3r_histcos`, which carries that block without the time spline, scores only
  **0.044** over 75 sessions — a comparably selected cohort, and far below
  `M5_time`. Note also that the AIC ordering differs from the held-out one:
  `M7r` (2077) and `M6r` (2078) sit below `M5_time` (2083) on median AIC. Use
  `M5_time` as the parsimonious predictive model and `M7r_time_histcos_rew`
  for the full additive coefficient read.
- **The reward block is a small step marginally and a negative one
  conditionally.** Median per-session `M1_rew − M0_const` ΔAIC is **−3.23**
  (detected SWR), **−8.18** (non-local, all), **−3.96** (non-local w/ SWR) and
  **−0.75** (non-local, no SWR) — negative is better — so on this cohort the
  block pays for itself against a constant rate in all four populations, the
  non-SWR one only just (the child wins in 58.3 % of its sessions; on all stops
  that step was +0.14 and the child won in 48.2 %). But
  `M7r_time_histcos_rew − M6r_time_histcos`, the same block given time **and**
  self-history, is positive everywhere: **+1.58**, **+1.19**, **+2.41**,
  **+2.28** on the same four populations. The four-parameter smooth block does
  not pay for itself once the time course and self-history are in the model,
  even though its immediate component is a large and consistent effect (§2)
  ([audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md),
  "Nested reward steps").
- **`M9r_rew_x_time` (reward × time) costs CV and does not buy AIC** relative to
  its parent `M7r_time_histcos_rew` (2083 vs 2077; 0.200 vs 0.244 bits/event on
  a smaller CV cohort still), so there is no evidence here that recent reward
  reshapes the within-stop time course rather than scaling it.

See [rate_models.md](rate_models.md) for the GOF / dispersion caveats, the
retirement of the lag-dummy history rungs, and the preliminary within-session
drift.
