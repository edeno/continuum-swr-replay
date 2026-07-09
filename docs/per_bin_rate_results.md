# Per-bin replay-rate ladder: fitted effects (Family B)

Quantitative readout of the per-bin conditional-intensity ladder
(`MODEL_PER_BIN_RATE`, see [rate_models.md](rate_models.md) for the methods),
fit per session over **218 sessions** and persisted in
`models_per_bin_rate/all_models.pkl` (13 models, `BIN_S = 0.015`, `TMAX = 4 s`).
Numbers below are the **fixed-knot-spline** run (the time-course spline uses
fixed shared knots so its coefficients are cross-session comparable and the
fitted curve is reconstructable — see §1b); spline-model values shifted <2% from
the earlier data-driven-knot run, conclusions unchanged. Regenerate with
`uv run python scripts/rate/summarize_per_bin_rate_models.py`.

**Figures:** the six-panel visual companion is
`paper/figures/per_bin_rate_ladder_summary.png`
(`scripts/rate/plot_per_bin_rate_summary.py`); detailed kernel / GOF figures are
`reward_kernel_per_bin_rate.png`, `selfhistory_autocorr.png`,
`model_checking_ladder.png`, `model_checking_example_{rich,sparse}.png`
(`scripts/rate/plot_per_bin_rate_models.py`, `scripts/rate/plot_model_checking.py`).

Effects are reported **per event-type population**: detected SWR (LFP ripples,
`n_SWR`), all non-local decoding events, and the non-local split by SWR overlap
(`non_local_swr` / `non_local_no_swr`). "RR" = rate ratio = `exp(coef)`;
summaries are across-session medians with per-animal sign consistency (n/5) and a
signed-rank p. Every effect is *conditional on the first 4 s of each stop*.

## 1. Baseline rate (M0_const)

Events per second of immobility (`exp(intercept)`), all pairwise differences
significant (paired Wilcoxon p ≤ 1e-5):

| population | rate (events/s) | IQR |
| --- | --- | --- |
| non-local (all) | **0.405** | 0.314–0.550 |
| detected SWR | 0.285 | 0.223–0.340 |
| non-local, no SWR | 0.234 | 0.162–0.308 |
| non-local, w/ SWR | **0.181** | 0.128–0.244 |

Ordering: **non-local (all) > detected SWR > NL-no-SWR > NL-w/SWR.** The non-local
split is additive (0.181 + 0.234 ≈ 0.405), so **SWR-coupled non-local events are
the rarest type**; of the 0.285/s detected SWRs, ~0.181/s coincide with a
non-local event (~63%), and ~45% of non-local events coincide with an SWR.
(Detected-SWR vs non-local is a cross-modality comparison — LFP vs decoder — so
read the within-non-local split as the clean contrast.)

## 1b. Within-stop time course (M5/M7 spline)

The dominant structure: replay rate rises over the first ~1.5–2 s of a stop then
plateaus (panels A0/A1 of the summary figure). The spline uses **fixed shared
knots** (`SPLINE_KNOTS=(1,2,3)`, bounds `[0,4]`), so the fitted `λ(tc)` curve is
**reconstructable from the saved coefficients** (`exp(intercept + bs(tc)·β)`) and
the spline coefficients are comparable across sessions — neither was true with
the previous data-driven (per-session quantile) knots.

**Duration caveat (important):** stop durations are very unequal and *endogenous*
— rewarded stops are ~3.5× longer, so at large `tc` only the long (≈rewarded)
stops survive. The exposure offset makes `λ(tc)` a proper rate, but the curve at
large `tc` reflects a **selected subpopulation**, not the same stops over time.
The figure marks this: a survival overlay (fraction of stops still immobile) and
grey shading where <20% survive (~beyond 2 s). The fitted panel (A1) uses M7
(reward/history controlled) so much of the rewarded-long-stop composition is
removed, but the principled fix — modelling the stop *departure hazard* jointly
with the rate — is the [joint-replay-departure](../.claude/docs/plans/joint-replay-departure/PLAN.md)
plan.

## 2. Reward on the previous trial (lag-1)

Both non-local subtypes are reward-responsive, and the effect is **much smaller
for non-SWR** — raw and after full control:

| population | RR raw (M1) | RR controlled (M7) | animals (M7) |
| --- | --- | --- | --- |
| detected SWR | 2.58 | 1.41 | 5/5 |
| non-local (all) | 2.62 | 1.50 | 5/5 |
| non-local, w/ SWR | **6.31** | **3.93** | 5/5 |
| non-local, no SWR | **1.77** | **1.11** | 5/5 |

Controlled = M7 (within-stop time spline + self-history + reward kernel). The
paired NL-w/SWR > NL-no-SWR dissociation is **86% of sessions, 5/5 animals,
p = 1.2e-28**. After full control the non-SWR lag-1 effect is barely above 1
(1.11; significant across animals, p = 3.2e-3, but only 57% of sessions
positive) — i.e. most of the apparent non-SWR reward response is absorbed by the
within-stop time course / clustering, whereas the SWR-coupled response remains
large (3.93) — so non-SWR is *weakly reward-elevated*, not reward-independent.

## 3. Reward history kernel, lags 2–10 (M7)

Beyond the lag-1 boost, recent reward **suppresses** the rate, and this part is
**shared** across populations (RR ≈ 0.85–0.97 per lag, ~0.6× cumulative):

| population | lag1 | lag2 | lag3 | lag4 | lag5 | … | lag10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 1.45 | 0.95 | 0.87 | 0.91 | 0.95 | … | 0.97 |
| non-local, w/ SWR | 3.95 | 0.90 | 0.87 | 0.92 | 0.95 | … | 1.03 |
| non-local, no SWR | 1.12 | 0.86 | 0.90 | 0.93 | 0.94 | … | 0.95 |

So the reward kernel is **boost (lag-1, dissociates) then suppression (lags 2–10,
shared)**. The lag≥2 suppression is a recent-reward effect; whether it is an
RPE/relative-reward signal vs satiety/adaptation is not established here (the GLM
uses past binary outcomes, not a prediction error).

## 4. Self-history (within-stop autocorrelation)

Same qualitative shape for all types — **refractory** at short lags, **bursty**
at ~100 ms — but the **microstructure dissociates** (M7, controlled):

| population | refractory window (RR<1) | burst peak RR | @ lag |
| --- | --- | --- | --- |
| detected SWR | ≤60 ms | **2.41** | 135 ms |
| non-local (all) | ≤60 ms | 2.04 | 105 ms |
| non-local, w/ SWR | ≤75 ms | 2.09 | 105 ms |
| non-local, no SWR | **≤150 ms** | **1.54** | 135 ms |

**SWR-coupled events are tightly bursty** (short refractory, strong ~100 ms
clustering); **non-SWR non-local events are more refractory and weakly bursty**
(longest dead time ≤150 ms, flattest bump 1.54×). Read the *shape* from the
lag-dummy models (M3_hist / M7); use the raised-cosine models (M3r/M7r) for
prediction/CV — the dummy kernel overfits held-out (see §6).

## 5. Reward-rate and switch (exploratory uncertainty dissociations)

| covariate | detected SWR | NL w/ SWR | NL no-SWR |
| --- | --- | --- | --- |
| recent reward rate (M2r) | 0.91 (n.s.) | **1.28** (p=1e-5) | **0.89** (p=1e-2) |
| patch switch (M8) | 0.82 | **0.76** | **1.16** |

Both point the same way: **non-SWR non-local rate is elevated in low-reward /
exploratory states; SWR-coupled rate is elevated in high-reward / exploitative
states** (paired NL-w/SWR vs NL-no-SWR: reward-rate p=2e-6, switch p=6e-8). These
are model-free behavioral proxies; the task-grounded value/RPE/block test is the
`nonswr-uncertainty-rate` plan.

## 6. Model ladder (fit summary, outcome = non-local)

| model | median AIC | CV bits/event | KS-in-band |
| --- | --- | --- | --- |
| M0_const | 2320 | 0.000 | 0.50 |
| M1_rew1 | 2285 | +0.045 | 0.41 |
| M2_rewhist | 2156 | +0.026 | 0.38 |
| M3_hist (dummy) | 2312 | **−0.245** | 0.39 |
| M3r_histcos | 2308 | +0.033 | 0.40 |
| M5_time | 2249 | **+0.228** | 0.86 |
| M7_time_hist_rew (dummy) | 2107 | −0.108 | 0.91 |
| **M7r_time_histcos_rew** | **2105** | **+0.199** | 0.91 |
| M8_full_switch | 2107 | −0.167 | 0.91 |
| M9_rew_x_time | 2115 | −0.407 | 0.90 |

- **The time spline calibrates the process** (KS-in-band jumps 0.50→0.86 at M5).
- **The lag-dummy self-history overfits** (M3/M7 CV negative); **the raised-cosine
  basis fixes it** — `M7r` recovers a positive CV (+0.199 vs M7-dummy's −0.108) and
  has the **lowest AIC (2105)** in the ladder.
- **But M5_time is the best-generalizing model** (CV +0.228, the highest in the
  ladder). `M7r` has the lowest AIC but does **not** beat M5 on held-out CV
  (+0.199 < +0.228) — i.e. reward + raised-cosine history add inferential signal
  but **negligible held-out bits** over the within-stop time course (M7r − M5 =
  −0.010, only 46% of sessions improve). Use **M5_time as the parsimonious
  predictive model**; **M7r for the full additive coefficient read**.
- M8 (switch) and M9 (reward×time) add nothing on CV (and M9 separates on ~9% of
  sessions). See [rate_models.md](rate_models.md) for the GOF/dispersion caveats
  and the preliminary within-session drift.
