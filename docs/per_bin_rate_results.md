# Per-bin replay-rate ladder: fitted effects (Family B)

> **Header date: 2026-09-04 fit. Numbers below are stale as of the 2026-09-06
> refit and are not to be quoted until this file is rewritten.** The 2026-09-06
> refit changed the per-bin design (reward-history features recomputed at the
> outcome boundary for the 13.4 % of stops that cross it) and the ladder itself
> (the `onset_lag` dummy rungs retired, `M6r_time_histcos` added as the reward
> block's nested null, the reward×time interaction re-parented
> onto `M7r_time_histcos_rew`, and three arrival-anchored rungs added). Every
> section below reads a rung that either no longer exists or was refit on the
> new design. What changed, with the measured sizes, is in
> [audits/2026-09-06-rate-refit-arrival-clock.md](audits/2026-09-06-rate-refit-arrival-clock.md);
> the methods in [rate_models.md](rate_models.md) are current. Regenerate this
> file from the new artifact with
> `uv run python scripts/rate/summarize_per_bin_rate_models.py`.

Quantitative readout of the per-bin conditional-intensity ladder
(`MODEL_PER_BIN_RATE`, see [rate_models.md](rate_models.md) for the methods),
fit per session over **218 sessions** and persisted in
`models_per_bin_rate/all_models.pkl` (10 models on the current artifact,
`BIN_S = 0.015`, `TMAX = 4 s`).
Numbers below are the **reward-basis** run (2026-09-04): the reward block is the
three-bump raised-cosine basis over the last ten known outcomes
(`reward_history_rcos_0..2` plus the `reward_history_incomplete` nuisance
indicator), replacing the ten per-lag reward indicators, refit once more on
2026-09-04 with separation judged per coefficient rather than per fit. Every
coefficient, AIC and goodness-of-fit number is unchanged by that second refit;
what changed is which sessions carry a valid cross-validation metric (§6). The
time-course spline uses fixed shared knots so its coefficients are cross-session
comparable and the fitted curve is reconstructable (see §1b). Regenerate with
`uv run python scripts/rate/summarize_per_bin_rate_models.py`.

**Figures:** the six-panel visual companion is
`paper/figures/per_bin_rate_ladder_summary.png`
(`scripts/rate/plot_per_bin_rate_summary.py`); detailed kernel / GOF figures are
`reward_kernel_basis.png`, `selfhistory_autocorr.png`,
`model_checking_ladder.png`, `model_checking_example_{rich,sparse}.png`
(`scripts/rate/plot_per_bin_rate_models.py`, `scripts/rate/plot_model_checking.py`).

Effects are reported **per event-type population**: detected SWR (LFP ripples,
`n_swr_events`), all non-local decoding events, and the non-local split by SWR overlap
(`non_local_swr` / `non_local_no_swr`). "RR" = rate ratio = `exp(coef)`;
summaries are across-session medians with per-animal sign consistency (n/5) and a
signed-rank p. Every effect is *conditional on the first 4 s of each stop*.

## 1. Baseline rate (M0_const)

Events per second of immobility (`exp(intercept)`), all pairwise differences
significant (paired Wilcoxon p ≤ 3e-5):

| population | rate (events/s) | IQR |
| --- | --- | --- |
| non-local (all) | **0.406** | 0.314–0.551 |
| detected SWR | 0.283 | 0.223–0.340 |
| non-local, no SWR | 0.235 | 0.162–0.308 |
| non-local, w/ SWR | **0.181** | 0.128–0.244 |

Ordering: **non-local (all) > detected SWR > NL-no-SWR > NL-w/SWR.** The non-local
split is roughly additive (0.181 + 0.235 ≈ 0.406), so **SWR-coupled non-local events
are the rarest type**; of the 0.283/s detected SWRs, ~0.181/s coincide with a
non-local event (~64%), and ~45% of non-local events coincide with an SWR.
(Detected-SWR vs non-local is a cross-modality comparison — LFP vs decoder — so
read the within-non-local split as the clean contrast.)

## 1b. Within-stop time course (M5/M7 spline)

The dominant structure: replay rate rises over the first ~1.5–2 s of a stop then
plateaus (panels A0/A1 of the summary figure). The spline uses **fixed shared
knots** (`SPLINE_KNOTS=(1,2,3)`, bounds `[0,4]`), so the fitted `λ(tc)` curve is
**reconstructable from the saved coefficients** (`exp(intercept + bs(tc)·β)`) and
the spline coefficients are comparable across sessions.

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

## 2. Immediate reward component (basis bump 0)

`reward_history_rcos_0` is the bump-weighted reward *fraction* over roughly the
last seven outcomes (weights 1.00, 0.79, 0.54, … at lags 1–7), so its coefficient
is **per unit of that fraction** — a 0→1 contrast, not a single-trial indicator.
It is not comparable to the retired lag-1 indicator's RR, and the three basis
coefficients are strongly collinear (pairwise r 0.53–0.89), so read the *kernel
reconstruction* in §3 for the per-lag effect and treat the single-coefficient RRs
below as a within-population contrast only.

| population | RR raw (M1_rew) | RR controlled (M7) | %pos (M7) | animals (M7) |
| --- | --- | --- | --- | --- |
| detected SWR | 11.8 | 3.13 | 75% | 5/5 |
| non-local (all) | 10.6 | 1.82 | 63% | 5/5 |
| non-local, w/ SWR | **57.1** | **6.52** | 83% | 5/5 |
| non-local, no SWR | **4.0** | **0.82** | 44% | 5/5 |

Controlled = M7 (within-stop time spline + self-history + reward basis). The
paired NL-w/SWR > NL-no-SWR dissociation is **80% of sessions, 5/5 animals,
p = 2.5e-23** (n = 217 paired sessions). After full control the non-SWR immediate
component is at or below 1 (0.82, p = 0.11 across sessions), whereas the
SWR-coupled response remains large — the same dissociation the per-lag
specification reported, with the non-SWR side now indistinguishable from no
immediate reward response.

## 3. Reward kernel, lags 1–10 (M7, reconstructed from the basis)

The per-lag log-rate contribution is `Σ_k coef_k · B[k, lag] / Σ_lag B[k, lag]`
(row-normalized basis), evaluated **per session** and then summarized as the
median rate ratio across sessions — the same estimator
`plot_per_bin_rate_models.py` draws. (Reconstructing from across-session median
coefficients instead would describe a session that does not exist: the three
basis coefficients are strongly collinear, so their marginal medians are not a
fitted kernel. That variant reads ~0.05 higher at lag 1.)

| population | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 | lag7 | lag8 | lag9 | lag10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| detected SWR | 1.18 | 0.99 | 0.93 | 0.92 | 0.92 | 0.93 | 0.95 | 0.97 | 1.00 | 1.02 |
| non-local (all) | 1.06 | 0.95 | 0.92 | 0.92 | 0.93 | 0.94 | 0.96 | 0.98 | 1.00 | 1.02 |
| non-local, w/ SWR | **1.38** | 1.05 | 0.93 | 0.90 | 0.90 | 0.92 | 0.94 | 0.99 | 1.02 | 1.06 |
| non-local, no SWR | **0.91** | 0.89 | 0.91 | 0.94 | 0.95 | 0.97 | 0.97 | 0.99 | 0.99 | 1.00 |

So the reward kernel is **boost at lag 1 (dissociates: 1.38 SWR-coupled vs 0.91
non-SWR) then mild suppression from lag 3 on (RR ≈ 0.90–0.97, shared across
populations) returning to ~1 by lag 10**. The smooth basis cannot represent a
lag-specific spike, so the lag-1 boost is smaller here than the per-lag
specification reported; the *ordering and the dissociation are unchanged*. The
lag≥2 suppression is a recent-reward effect; whether it is an RPE/relative-reward
signal vs satiety/adaptation is not established here (the GLM uses past binary
outcomes, not a prediction error).

## 4. Self-history (within-stop autocorrelation)

Same qualitative shape for all types — **refractory** at short lags, **bursty**
at ~100–135 ms — but the **microstructure dissociates** (M7, controlled):

| population | refractory window (RR<1) | burst peak RR | @ lag |
| --- | --- | --- | --- |
| detected SWR | ≤60 ms | **2.77** | 135 ms |
| non-local (all) | ≤60 ms | 2.13 | 105 ms |
| non-local, w/ SWR | ≤75 ms | 2.41 | 135 ms |
| non-local, no SWR | **≤150 ms** | **1.82** | 135 ms |

**SWR-coupled events are tightly bursty** (short refractory, strong ~100 ms
clustering); **non-SWR non-local events are more refractory and weakly bursty**
(longest dead time ≤150 ms, flattest bump 1.82×). Read the *shape* from the
lag-dummy models (M3_hist / M7); use the raised-cosine models (M3r/M7r) for
prediction/CV (see §6).

## 5. Behavioral-state covariates

The exploratory recent-reward-rate (`M2r_rewrate`) and patch-switch
(`M8_full_switch`) rungs are no longer part of `MODEL_PER_BIN_RATE`. Those
covariates are now fit in the dedicated per-bin ladders, on their own frozen
cohorts and with the post-outcome gating they need:

- `MU_uncertainty` (`scripts/rate/fit_uncertainty_rate.py` →
  `models_per_bin_rate/uncertainty_rate.pkl`) — task value / uncertainty / switch-rate
  covariates per population.
- The `is_swr` contrast ladder (`scripts/rate/fit_swr_contrast_rate.py` →
  `models_per_bin_rate/swr_contrast_rate.pkl`) — the same covariates as an
  SWR-vs-non-SWR contrast.

Do **not** carry the retired M2r/M8 numbers forward from an earlier revision of
this document: those rungs were fit on a different reward block and a different
cohort.

## 6. Model ladder (fit summary, outcome = non-local)

Median AIC across sessions, held-out bits/event from the per-session CV (0 = the
constant model), and the share of sessions whose time-rescaling KS curve stays
inside the band:

`n_cv` is the number of sessions whose CV metric is valid: a fold in which any
non-nuisance term is unidentified voids the whole metric, so a rung with a small
`n_cv` is summarized over a **selected minority** of sessions and its bits/event
is not comparable with a rung scored on the full corpus.

| model | median AIC | CV bits/event | n_cv | KS-in-band |
| --- | --- | --- | --- | --- |
| M0_const | 2321 | 0.000 | 218 | 0.49 |
| M1_rew | 2305 | -0.050 | 214 | 0.46 |
| M3_hist (dummy) | 2312 | 0.061 | 2 | 0.40 |
| M3r_histcos | 2309 | 0.052 | 83 | 0.39 |
| M4_hist_rew | 2302 | 0.071 | 2 | 0.41 |
| M5_time | 2250 | **0.227** | 215 | 0.86 |
| M6_time_hist | 2253 | 0.079 | 2 | 0.85 |
| M7_time_hist_rew (dummy history) | 2253 | 0.094 | 2 | 0.88 |
| **M7r_time_histcos_rew** | **2244** | 0.130 | 86 | 0.87 |
| M9_rew_x_time | 2259 | 0.099 | 2 | 0.88 |

- **The time spline calibrates the process** (KS-in-band jumps 0.49 → 0.86 at M5).
- **The dummy-history rungs are barely cross-validated at all** (`n_cv` = 2 of
  218 for M3/M4/M6/M7/M9). The `onset_lag*` dummies are genuinely separated — a
  short lag perfectly predicts `count == 0` — in ~98% of sessions, and that
  separation is in the model's own self-history terms, not in a nuisance term.
  Their bits/event columns describe two sessions and must not be quoted as a
  corpus quantity.
- **The raised-cosine self-history basis has the lowest AIC (2244)** in the
  ladder. It is also the only history parameterization with a usable CV cohort
  (86 sessions vs 2), so the held-out comparison against `M7` is not a like-for-
  like one.
- **M5_time is the best-generalizing model** (CV 0.227 over 215 sessions, the
  highest in the ladder). `M7r` has the lowest AIC but does **not** beat M5 on
  held-out CV (0.130 over 86 sessions) — i.e. reward + raised-cosine history add
  inferential signal but **negligible held-out bits** over the within-stop time
  course. Use **M5_time as the parsimonious predictive model**; **M7r for the
  full additive coefficient read**.
- **The reward block is now a small AIC step.** Per session, `M1_rew − M0_const`
  has a median ΔAIC of −5.0 (non-local) to −1.6 (detected SWR), and
  `M7_time_hist_rew − M6_time_hist` is ≈ 0 to +2.8 — the four-parameter smooth
  reward block does not pay for itself once the time course and self-history are
  in the model, even though its immediate component is a large, consistent effect
  (§2). The retired eleven-parameter per-lag block bought ~100–160 AIC points on
  the same cohorts; see
  [audits/2026-09-04-rate-refit-comparison.md](audits/2026-09-04-rate-refit-comparison.md).
- M9 (reward×time) adds nothing on CV, on the same two-session cohort as the
  other dummy-history rungs. See [rate_models.md](rate_models.md) for the
  GOF/dispersion caveats and the preliminary within-session drift.
