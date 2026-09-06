# Per-bin replay-rate ladder: fitted effects (Family B)

**Fit date: 2026-09-06.** Quantitative readout of the per-bin
conditional-intensity ladder (`MODEL_PER_BIN_RATE`, see
[rate_models.md](rate_models.md) for the methods), fit per session over the
**218-session corpus** and persisted in
`Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`
(`BIN_S = 0.015`, `TMAX = 4 s`, 10 rungs — the seven stop-clock rungs read
below plus the three arrival-anchored ones). Every number here is transcribed
from `uv run python scripts/rate/summarize_per_bin_rate_models.py` run against
that artifact, or read from the artifact directly where the section says so.

Not every population carries a fit in every session: the full additive rung
`M7r_time_histcos_rew` is fit on **218 (detected SWR) / 218 (non-local, all) /
216 (non-local w/ SWR) / 218 (non-local, no SWR)** sessions
([audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md),
"Sessions fitted per rung"). Per-section `n` states which.

Two things changed with this fit and are load-bearing for everything below.
(i) The reward-history features are **recomputed at the outcome boundary** for
the 13.4 % of stops that cross one, so a bin after the boundary carries the
outcome the animal has by then learned. (ii) The five `onset_lag1..10` dummy
rungs were **retired** — they could not be cross-validated — and the ladder now
runs `M0_const`, `M1_rew`, `M5_time`, `M3r_histcos`, `M6r_time_histcos`,
`M7r_time_histcos_rew`, `M9r_rew_x_time` on the stop clock. Both are described
in [rate_models.md](rate_models.md); the measured sizes are in the audit above.

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
| non-local (all) | **0.408** | 0.311–0.552 |
| detected SWR | 0.282 | 0.220–0.342 |
| non-local, no SWR | 0.234 | 0.162–0.308 |
| non-local, w/ SWR | **0.180** | 0.129–0.244 |

Ordering: **non-local (all) > detected SWR > NL-no-SWR > NL-w/SWR.** The
pairwise contrasts are median *within-session* rate ratios with a paired
Wilcoxon p, and all four are significant:

| contrast | median rate ratio | p |
| --- | --- | --- |
| NL w/ SWR / NL no-SWR | 0.75× | 5.6e-15 |
| detected SWR / non-local (all) | 0.62× | 3.0e-26 |
| NL w/ SWR / detected SWR | 0.74× | 7.6e-37 |
| NL no-SWR / detected SWR | 0.87× | 2.6e-05 |

The non-local split is roughly additive (0.180 + 0.234 ≈ 0.408), so
**SWR-coupled non-local events are the rarest type**. Read the coupling shares
off the within-session ratios, not off the ratio of the medians: the median
session has **0.74** SWR-coupled non-local events per detected SWR, and
SWR-coupled events are **0.43** of its non-local events
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
that interval the across-session median peak sits at 2.4 s (detected SWR),
2.8 s (non-local, all), 2.2 s (non-local w/ SWR) and 3.0 s (non-local, no
SWR) — the last exactly on the right edge of the readable interval, so treat it
as "at or beyond 3 s", not as a located maximum
([audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md),
"Time-course shape").

**Duration caveat (important):** stop durations are very unequal and
*endogenous* — rewarded stops are ~3.5× longer on the mean (2.3× on the median:
0.518 s vs 0.228 s, `stop_table.parquet`) — so at large `tc` only the long
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
broadly the same delay in every population (2.2–3.0 s on `tc` against
3.0–3.6 s after arrival, KS distance 0.06–0.12 between the normalized curves),
so the early-in-stop rise is not an artifact of the bout clock. The arrival
rungs are heavily flagged (37–66 % of their fits carry a separated
non-nuisance term) and are rarely cross-validated, so they are read pooled, on
their fitted curve, never per session; see [rate_models.md](rate_models.md).

## 2. Immediate reward component (basis bump 0)

`reward_history_rcos_0` is the bump-weighted reward *fraction* over roughly the
last seven known outcomes (weights 1.00, 0.79, 0.54, … at lags 1–7), so its
coefficient is **per unit of that fraction** — a 0→1 contrast, not a
single-trial indicator. The three basis coefficients are strongly collinear, so
read the *kernel reconstruction* in §3 for the per-lag effect and treat the
single-coefficient RRs below as a within-population contrast only.

| population | RR raw (M1_rew) | RR controlled (M7r) | %pos (M7r) | animals (M7r) | p (M7r) | n |
| --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 21.1 | 3.48 | 79 % | 5/5 | 6.1e-20 | 218 |
| non-local (all) | 22.3 | 2.85 | 72 % | 5/5 | 1.0e-14 | 218 |
| non-local, w/ SWR | **110.3** | **8.65** | 86 % | 5/5 | 2.9e-29 | 216 |
| non-local, no SWR | **7.9** | **1.24** | 56 % | 4/5 | 2.2e-01 | 218 |

Controlled = `M7r_time_histcos_rew` (within-stop time spline + raised-cosine
self-history + reward basis). The paired NL-w/SWR > NL-no-SWR dissociation is
**79.2 % of sessions, 5/5 animals, signed-rank p = 7.8e-22** over **216 paired
sessions**. After full control the non-SWR immediate component is
indistinguishable from no immediate reward response (RR 1.24, p = 0.22, and
only 4 of 5 animals positive), whereas the SWR-coupled response remains large.

The other two bumps (`M7r`, same fits) run the other way and are also
consistent — the middle bump is far below 1 and the slowest bump above it,
which is what produces the dip-then-recovery shape of §3:

| population | bump 1 RR | %pos | animals | bump 2 RR | %pos | animals |
| --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 0.06 | 20 % | 5/5 | 4.31 | 73 % | 5/5 |
| non-local (all) | 0.08 | 20 % | 5/5 | 3.30 | 79 % | 5/5 |
| non-local, w/ SWR | 0.01 | 12 % | 5/5 | 10.58 | 83 % | 5/5 |
| non-local, no SWR | 0.27 | 37 % | 4/5 | 2.12 | 61 % | 4/5 |

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
| detected SWR | 1.22 | 1.00 | 0.93 | 0.91 | 0.91 | 0.93 | 0.95 | 0.97 | 1.00 | 1.03 |
| non-local (all) | 1.14 | 0.98 | 0.92 | 0.91 | 0.92 | 0.93 | 0.96 | 0.98 | 1.00 | 1.03 |
| non-local, w/ SWR | **1.44** | 1.06 | 0.93 | 0.89 | 0.89 | 0.91 | 0.94 | 0.99 | 1.03 | 1.07 |
| non-local, no SWR | **0.99** | 0.92 | 0.92 | 0.92 | 0.93 | 0.95 | 0.96 | 0.99 | 0.99 | 1.00 |

So the reward kernel is **a boost at lag 1 that dissociates by SWR coupling
(1.44 SWR-coupled vs 0.99 non-SWR), then a mild suppression from lag 3 on
(RR ≈ 0.89–0.97, shared across all four populations) that returns to ~1 by lag
10**. The smooth basis has no per-lag parameter and cannot represent a
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
| detected SWR | ≤60 ms | **2.64** | 135 ms | 218 |
| non-local (all) | ≤60 ms | 2.08 | 135 ms | 218 |
| non-local, w/ SWR | **≤45 ms** | 2.50 | 150 ms | 216 |
| non-local, no SWR | ≤60 ms | **1.72** | 120 ms | 218 |

**SWR-coupled non-local events are the more tightly bursty stream** — the
shortest dead time (≤45 ms) and a 2.50× peak — while **non-SWR non-local events
are the flattest** (1.72× peak, and the earliest, at 120 ms). Events are
strongly non-Poisson in every population. Two cautions: the raised-cosine basis
is smooth, so a single-lag notch cannot appear in it and the tabulated
refractory window is a resolution-limited "≤", not a measured dead time; and
these are the *controlled* kernels (`M7r`), read after the time spline and the
reward block. The uncontrolled `M3r_histcos` kernels are larger throughout
(peaks 3.48 / 2.97 / 4.27 / 2.26× in the same row order), so do not mix the two.

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
curve stays inside the band. All seven stop-clock rungs are fit on all 218
non-local sessions; the arrival rungs are **not** in this table because they are
fit on a subset of the rows and their AIC is not on this scale.

`n_cv` is the number of those sessions whose CV metric is valid: a fold in which
any non-nuisance term is unidentified voids the whole metric, so a rung with a
small `n_cv` is summarized over a **selected minority** of sessions and its
bits/event is not comparable with a rung scored on the full corpus. (`n_cv` is
read from `all_models.pkl`; the other three columns are the summarizer's.)

| model | median AIC | CV bits/event | n_cv | KS-in-band |
| --- | --- | --- | --- | --- |
| M0_const | 2290 | 0.000 | 218 | 0.50 |
| M1_rew | 2281 | 0.025 | 216 | 0.46 |
| M5_time | 2223 | **0.225** | 217 | 0.85 |
| M3r_histcos | 2286 | 0.052 | 79 | 0.39 |
| M6r_time_histcos | **2221** | 0.229 | 82 | 0.83 |
| **M7r_time_histcos_rew** | **2220** | 0.220 | 84 | **0.87** |
| M9r_rew_x_time | 2228 | 0.186 | 64 | 0.86 |

- **The time spline is what calibrates the process** (KS-in-band jumps
  0.50 → 0.85 at `M5_time`, and neither reward nor self-history fixes the
  constant model without it: 0.46 and 0.39).
- **`M5_time` is the best-generalizing model**, and the only rung with both a
  large held-out gain and an essentially complete CV cohort (0.225 bits/event
  over 217 of 218 sessions). The three raised-cosine history rungs score
  0.19–0.23 bits/event but only on 64–84 sessions each, so their CV is **not**
  a like-for-like comparison against `M5_time`'s — a rung is not "better on
  held-out data" because a selected fifth of sessions scored well. Use
  `M5_time` as the parsimonious predictive model and `M7r_time_histcos_rew`
  for the full additive coefficient read.
- **The reward block is a small step marginally and a negative one
  conditionally.** Median per-session `M1_rew − M0_const` ΔAIC is **−3.01**
  (detected SWR), **−7.84** (non-local, all), **−5.42** (non-local w/ SWR) and
  **+0.14** (non-local, no SWR) — negative is better — so the block pays for
  itself against a constant rate in three of the four populations. But
  `M7r_time_histcos_rew − M6r_time_histcos`, the same block given time **and**
  self-history, is positive everywhere: **+1.64**, **+1.09**, **+2.17**,
  **+2.38** on the same four populations. The four-parameter smooth block does
  not pay for itself once the time course and self-history are in the model,
  even though its immediate component is a large and consistent effect (§2)
  ([audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md)).
- **`M9r_rew_x_time` (reward × time) costs AIC and CV** relative to its parent
  `M7r_time_histcos_rew` (2228 vs 2220; 0.186 vs 0.220 bits/event on a smaller
  CV cohort still), so there is no evidence here that recent reward reshapes
  the within-stop time course rather than scaling it.

See [rate_models.md](rate_models.md) for the GOF / dispersion caveats, the
retirement of the lag-dummy history rungs, and the preliminary within-session
drift.
