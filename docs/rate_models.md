# Replay event-rate models: two families

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

## Family A — per-stop rate (the literature-comparable layer)

- **Spec:** `MODEL_PER_STOP_RATE` (`R0` null, `R1` previous-trial reward/switch,
  `R2` + recent reward rate, `Rlag` lagged reward kernel `reward_lag1..10` +
  `is_after_switch`) in
  [`model_specs.py`](../src/continuum_swr_replay/analysis/replay/model_specs.py).
- **Estimator:** one ordinary Poisson GLM per stop on the event *count*, with a
  `log(immobility_time)` offset — `exp(coef)` is the multiplicative change in
  events per **second of immobility**
  ([`per_stop_rate_model.py`](../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py)).
- **Runner / output:** `scripts/fit_per_stop_rate_models.py` →
  `Processed-Data/replay_content/models_per_stop_rate/all_models.pkl`.
- **Cite it for:** the Ambrose 2016 / Sosa 2020 reward-rate comparison (same
  Poisson-events-per-second estimator) and the **detected-SWR positive
  control** — detected SWRs reproduce the canonical reward-gating (omission and
  recent-reward suppression; see `#19c` in
  [`../paper/claims_table.md`](../paper/claims_table.md) for the values).

This is the simple layer: one rate per stop, dwell handled as an exposure
offset. Its limitation is exactly that offset — see below.

## Family B — per-bin conditional intensity (the confound-free layer)

- **Spec:** `MODEL_PER_BIN_RATE`, an 11-model **point-process GLM
  conditional-intensity ladder** (Truccolo–Eden–Brown 2005) `M0_const … M9` in
  [`model_specs.py`](../src/continuum_swr_replay/analysis/replay/model_specs.py).
- **Estimator:** one Poisson GLM per session on the **(stop × 15 ms bin)**
  design from `aggregate_to_timebins`, with a `log(exposure)` offset; the
  within-stop time course is modelled explicitly by a `bs(tc, df=4)` spline
  rather than averaged into a per-stop dwell offset
  ([`per_bin_rate_model.py`](../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py)).
- **The ladder** toggles four blocks independently, all at one bin width
  (`BIN_S = 0.015`) so the models are nested and AIC / CV / GOF are comparable:

  | model | adds |
  | --- | --- |
  | `M0_const` | homogeneous Poisson baseline (`count ~ 1`) |
  | `M1_rew1` | previous reward (`reward_lag1`) |
  | `M2_rewhist` | reward history kernel (`reward_lag1..10`) |
  | `M2r_rewrate` | continuous recent reward rate (alt encoding) |
  | `M3_hist` | self-history / autoregression (`onset_lag1..10`) |
  | `M4_hist_rew` | self-history + reward history |
  | `M5_time` | within-stop time course (spline) |
  | `M6_time_hist` | time + self-history |
  | `M7_time_hist_rew` | time + self-history + reward history (**the full additive model**) |
  | `M8_full_switch` | + `is_after_switch` |
  | `M9_rew_x_time` | + `reward_lag1 × bs(tc)` interaction (does reward modulate the time course?) |

  The reward×time interaction (`M9`) and the switch model (`M8`) are **built and
  fit**, not deferred.
- **Runner / output:** `scripts/fit_per_bin_rate_models.py`
  (`--workers N` fans sessions over processes) →
  `Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`. Plot with
  `scripts/plot_per_bin_rate_models.py`
  (→ `reward_kernel_per_bin_rate.png`, `selfhistory_autocorr.png`); the
  descriptive PSTH `scripts/_timeresolved_psth.py`
  (→ `rate_timecourse_vs_pump.png`) reads the onset/dense tables directly.
- **Cite it for:** the **dwell-, time-, reward-history- and clustering-controlled
  dissociation**. In `M7`, the immediate (lag-1) reward rate ratio dissociates —
  `non_local_swr ≈ 3.95` ≫ `non_local_no_swr ≈ 1.12` (`swr ≈ 1.45`), paired
  `non_local_swr > non_local_no_swr` in **86 % of sessions, 5/5 animals,
  p ≈ 1e-28** — while the slower reward-history (RPE) suppression at lags 2–10
  is **shared**. The self-history kernel is **refractory** (<1) below 60 ms and
  **bursty** (>1) at ~100 ms (events are non-Poisson).

  (These figures are from the most recent unified-15 ms-bin run; `#19c` in the
  claims table retains the original mixed-binning values — same dissociation,
  slightly different absolute coefficients, as expected from the re-binning.)

## The result holds across both families

Reward increases replay rate, and the SWR-coupled subset carries the strong
gating, in **both** estimators:

- **Family A (per-stop):** omission suppresses the non-local event rate in
  **99 % of sessions, 5/5 animals** (`is_after_omission RR ≈ 0.40`; stronger for
  `non_local_swr`, weaker for `non_local_no_swr`).
- **Family B (per-bin):** the lag-1 reward dissociation above, controlling for
  dwell, the within-stop time course, reward history, and event clustering.

The agreement across two estimators with different confound structure is the
robustness argument for `#19c` in [`../paper/claims_table.md`](../paper/claims_table.md).

## Point-process model-checking (Family B)

Family B treats the event train as a point process and checks it as one via the
time-rescaling theorem (`per_bin_rate_gof`, reusing
`non_local_detector.model_checking.sorted_spikes` — KS, rescaled-ISI
autocorrelation, point-process residuals). Adding the time + self-history blocks
walks the KS statistic into the band (`ks_within_band` fraction ≈ 0.50 for
`M0_const` → ≈ 0.91 for `M7`); the `M0_const` failure is the expected non-Poisson
signature that motivates the history block. Per-bin mean count is ≈0.005–0.015,
so the trapezoid rescaling is well-behaved on average; `mean_predicted` /
`max_predicted` are reported so the high-rate-bin KS bias can be ruled out, and
the KS read is cross-checked against the rescaled-ISI autocorrelation.

## Caveats and what each family can/can't tell you

- **Cross-family CV is not comparable.** Family A scores 5-fold per-stop mean
  log-likelihood; Family B scores 10-fold (over stops) per-bin log-likelihood
  plus `cv_bits_per_event` vs `M0_const`. Compare CV **within** a family only.
- **Self-history overfits out-of-sample.** In Family B the self-history block
  improves the in-sample KS but *hurts* held-out `cv_bits_per_event`
  (`M5_time ≈ +0.23` vs `M3_hist ≈ −0.25`): refractory near-separation produces
  extreme per-session coefficients that don't generalize. An L2-penalized
  (`fit_regularized`) or Firth history fit is the noted fallback (future work).
- **Duration is a behavioral variable, not just exposure.** Stop durations are
  bimodal — reward creates the long consummatory mode — so the offset is not a
  nuisance to be divided out; the per-second / per-bin estimators isolate the
  neural rate while the duration structure itself carries reward information.
- **Family B truncates at `TMAX = 4 s`.** It models only the first 4 s of each
  stop (~80 % of onsets; median onset ≈ 2.4 s with a long tail). Every Family-B
  conclusion is conditional on the first 4 s.

## Future work (noted, not built here)

- The long immobility tail (raise `TMAX` or add a censoring term) and slow
  session drift belong in a **state-space model**, not as fixed covariates here.
- Arm / well / patch identity covariates (hard to interpret in this rate view).
- Cross-/ensemble-history between event types (SWR↔non-local).
- An L2-penalized self-history fit to recover the over-fit history block.
