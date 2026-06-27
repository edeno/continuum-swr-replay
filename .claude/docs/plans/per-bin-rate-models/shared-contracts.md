# Shared contracts

Cross-phase contracts. Each is defined once here; phases link in by anchor and must not weaken these signatures/semantics.

Index:
- [attach_is_swr_to_onsets](#attach_is_swr_to_onsets) — Phase 1 defines, Phases 1/3 use.
- [aggregate_to_timebins](#aggregate_to_timebins) — Phase 1 defines, Phases 1/3 use.
- [POPULATIONS](#populations) — Phase 1 defines, Phases 1/3/4 use.
- [PerBinRateModelSpec + MODEL_PER_BIN_RATE](#perbinratemodelspec) — Phase 2 defines, Phases 3/4 use.
- [fit_per_bin_rate_model / cross_validate_per_bin_rate_model](#fit_per_bin_rate_model--cross_validate_per_bin_rate_model) — Phase 2 defines, Phase 3 uses.
- [per_bin_rate_gof (point-process goodness-of-fit)](#per_bin_rate_gof) — Phase 2 defines, Phase 5 uses.
- [models_per_bin_rate/all_models.pkl schema](#tidy-output-schema) — Phase 3 produces, Phase 4/5 read.

---

## attach_is_swr_to_onsets

Home: `src/continuum_swr_replay/analysis/replay/_helpers.py`. Replaces the within-stop time-rank match copy-pasted in `_reward_glm.py`, `_reward_glm_plot.py`, `_timeresolved_kernel_glm.py`, `_selfhistory_glm.py` (the `ev['rank'] = ... cumcount()` block near the top of each `main`).

```python
def attach_is_swr_to_onsets(
    onset_df: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add a canonical ``is_swr`` column to the non-local onset rows.

    Matches each non-local onset to its ``event_table`` event by within-stop
    time rank (both are ordered by time within a stop and are 1:1 — 65,913
    non-local onsets == 65,913 event-table rows). SWR/MUA onset rows get
    ``is_swr = NaN``. Returns a copy; does not mutate the input.
    """
    out = onset_df.copy()
    out["is_swr"] = np.nan
    ev = event_table[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = out[out["event_type"] == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    merged = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"],
                      how="left", suffixes=("", "_ev"))
    out.loc[nl.index, "is_swr"] = merged["is_swr_ev"].to_numpy()
    return out
```
Invariant: match rate ≥ 0.999 (smoke prints `is_swr.notna()` fraction on non-local rows; the scratch scripts saw 1.000). Do not weaken to a time-tolerance `merge_asof` — the rank match is exact.

---

## aggregate_to_timebins

Home: `_helpers.py`. Replaces `_selfhistory_glm.build_bins` and `_timeresolved_kernel_glm.onset_cells`. Produces the canonical **(stop × bin)** design; the `_reward_glm` per-(session, reward, tbin) view is a downstream `groupby` on this, not a second mode.

```python
def aggregate_to_timebins(
    stop_table: pd.DataFrame,
    onset_times: pd.DataFrame,   # subset of event_onset_table for ONE population
    *,
    bin_s: float,
    tmax: float,
    reward_lags: int = 0,         # how many reward_lag{j} cols to carry from stop_table
    self_history_lags: int = 0,   # how many onset_lag{j} self-history cols to build
    stop_covariates: tuple[str, ...] = (),  # extra per-stop cols carried from stop_table by stop_id
) -> pd.DataFrame:
    """One row per (stop, bin) up to ``tmax``, at ``bin_s`` resolution.

    Columns: animal, day, epoch, stop_id, binj, tc (=bin-center s, time-since-
    stop-onset), exposure (s of immobility in the bin), reward_lag1..reward_lags,
    each name in ``stop_covariates`` (e.g. recent_trial_reward_rate, is_after_switch),
    count (population onsets in the bin), onset_lag1..self_history_lags
    (same-population onset counts j bins back, within stop, reset at boundary).

    Bins are built from ``stop_table.immobility_time`` (continuous), independent
    of the 4 ms dense grid, so non-4-ms widths (e.g. 15 ms) are exact.
    ``exposure`` of the last bin is the partial remainder. Asserts ``count.max()
    <= 1`` at the chosen ``bin_s`` (events are pre-separated upstream; verified
    lossless at 15 ms) so the point-process binarization is safe — or logs the
    multi-event-bin fraction if a future re-binning breaks the assumption.
    """
```
Semantics nailed down:
- `nbin = min(ceil(immobility_time / bin_s), int(tmax / bin_s))`; bins built via `np.repeat`/`np.concatenate(np.arange(n))` (the `build_bins` idiom).
- `exposure = min(bin_s, immobility_time - binj*bin_s)`; drop rows with `exposure <= 1e-4`.
- `count`: `onset_times` (already filtered to one population) binned by `floor(t_since_stop_onset / bin_s)`, joined on `(stop_id, binj)`, fill 0.
- `onset_lag{j} = groupby('stop_id')['count'].shift(j).fillna(0)` after sorting by `(stop_id, binj)` — within-stop only.
- `reward_lag{j}`: carried from `stop_table` by `stop_id`.
- `stop_covariates`: each named column carried from `stop_table` by `stop_id`, same as the reward lags. Phase 3 passes `("recent_trial_reward_rate", "is_after_switch")` so `M2r_rewrate` and `M8_full_switch` have their predictors; both columns are present in `stop_table.pkl`. Without this, those two models' `formula()` would reference absent columns and `fit_per_bin_rate_model` would return `None` for every session — silently dropping them.

---

## POPULATIONS

Home: `_helpers.py` (a small selector) or `model_specs.py`. The four outcome tags, mirroring `fit_per_stop_rate_models.RESPONSE_COLS`:

```python
# given onset_df already passed through attach_is_swr_to_onsets:
POPULATIONS = {
    "swr":              lambda o: o[o.event_type == "SWR"],
    "non_local":        lambda o: o[o.event_type == "non_local"],
    "non_local_swr":    lambda o: o[(o.event_type == "non_local") & (o.is_swr == True)],   # noqa: E712
    "non_local_no_swr": lambda o: o[(o.event_type == "non_local") & (o.is_swr == False)],  # noqa: E712
}
```
Names match the per-stop `RESPONSE_COLS` keys exactly so the two families' tidy outputs join on `outcome`. `non_local` (pooled) = `non_local_swr ∪ non_local_no_swr`; it is the **one-population null hypothesis**, not merely a foil — if the pooled model's coefficients/GOF match both subtypes, they are one event type; if pooling masks a difference (as the reward-gating result suggests), they are distinct. The ladder fits all four so this is tested directly.

---

## PerBinRateModelSpec

Home: `model_specs.py`, appended after `MODEL_PER_STOP_RATE`/`Rlag` (after line 484). A *parallel* dataclass to `ModelSpec` (do not modify `ModelSpec`).

This is the **point-process GLM conditional-intensity ladder** (Truccolo–Eden–Brown
2005): `log λ = baseline + f(time-since-stop) + g(self-history) + Σβ·covariates`.
The models toggle the four blocks independently so each block's contribution is
readable *with and without the time course*. **All models share one bin width**
(`BIN_S = 0.015`) so the ladder is nested and AIC/CV/deviance/GOF are comparable —
fine bins are required anyway for the self-history term and for time-rescaling GOF.

```python
BIN_S = 0.015          # 15 ms — ladder-wide (refractory <60 ms, bursty ~100 ms; GOF needs fine bins)
TMAX = 4.0
SPLINE_DF = 4
N_HISTORY_LAGS = 10    # onset_lag1..10 = 150 ms self-history window

@dataclass(frozen=True)
class PerBinRateModelSpec:
    name: str
    time_spline: bool              # bs(tc, df=SPLINE_DF) — within-stop time course
    reward_lags: int               # 0 | 1 | N_REWARD_LAGS  (reward_lag1..k, past outcomes; omission == 1-reward, same regressor)
    reward_rate: bool              # recent_trial_reward_rate (continuous; ALT encoding of reward history)
    self_history_lags: int         # 0 | N_HISTORY_LAGS  (onset_lag1..L, within-stop autoregressive)
    switch: bool                   # is_after_switch
    reward_time_interaction: bool  # reward_lag1 : bs(tc)  (does reward modulate the time course?)
    description: str
    parent: str | None

    def formula(self) -> str:
        terms: list[str] = []
        if self.time_spline:
            terms.append(f"bs(tc, df={SPLINE_DF})")
        terms += [f"reward_lag{j}" for j in range(1, self.reward_lags + 1)]
        if self.reward_rate:
            terms.append("recent_trial_reward_rate")
        terms += [f"onset_lag{j}" for j in range(1, self.self_history_lags + 1)]
        if self.switch:
            terms.append("is_after_switch")
        if self.reward_time_interaction:
            terms.append(f"reward_lag1:bs(tc, df={SPLINE_DF})")
        return "count ~ " + (" + ".join(terms) if terms else "1")

# (time, rewL, rate, hist, switch, interact)
_N = N_REWARD_LAGS
MODEL_PER_BIN_RATE: dict[str, PerBinRateModelSpec] = {
  "M0_const":         PerBinRateModelSpec("M0_const",        False, 0,  False, 0, False, False, "homogeneous Poisson baseline", None),
  "M1_rew1":          PerBinRateModelSpec("M1_rew1",         False, 1,  False, 0, False, False, "+ previous reward (reward_lag1)", "M0_const"),
  "M2_rewhist":       PerBinRateModelSpec("M2_rewhist",      False, _N, False, 0, False, False, "+ reward history kernel (reward_lag1..10)", "M1_rew1"),
  "M2r_rewrate":      PerBinRateModelSpec("M2r_rewrate",     False, 0,  True,  0, False, False, "+ recent reward rate (continuous; alt to kernel)", "M0_const"),
  "M3_hist":          PerBinRateModelSpec("M3_hist",         False, 0,  False, N_HISTORY_LAGS, False, False, "+ self-history (autoregressive)", "M0_const"),
  "M4_hist_rew":      PerBinRateModelSpec("M4_hist_rew",     False, _N, False, N_HISTORY_LAGS, False, False, "+ self-history + reward history", "M3_hist"),
  "M5_time":          PerBinRateModelSpec("M5_time",         True,  0,  False, 0, False, False, "+ time-since-stop (spline)", "M0_const"),
  "M6_time_hist":     PerBinRateModelSpec("M6_time_hist",    True,  0,  False, N_HISTORY_LAGS, False, False, "+ time + self-history", "M5_time"),
  "M7_time_hist_rew": PerBinRateModelSpec("M7_time_hist_rew",True,  _N, False, N_HISTORY_LAGS, False, False, "+ time + self-history + reward history", "M6_time_hist"),
  "M8_full_switch":   PerBinRateModelSpec("M8_full_switch",  True,  _N, False, N_HISTORY_LAGS, True,  False, "+ switch (full additive)", "M7_time_hist_rew"),
  "M9_rew_x_time":    PerBinRateModelSpec("M9_rew_x_time",   True,  _N, False, N_HISTORY_LAGS, False, True,  "M7 + reward×time interaction", "M7_time_hist_rew"),
}
```
This is your nine-model ladder plus two variants: **M2r_rewrate** (continuous
`recent_trial_reward_rate` as an alternative reward encoding to the lag kernel) and
**M9_rew_x_time** (the reward×time interaction). Deferred by decision: arm/well/patch
identity (hard to interpret), slow session drift (→ the state-space model, not a fixed
covariate here), and cross-/ensemble-history (stretch goal). `formula()` + `BIN_S`/
`reward_lags`/`reward_rate`/`self_history_lags`/`switch`/`reward_time_interaction`
drive both `aggregate_to_timebins` and the fit, so the spec is the single source of
truth. Note `M0_const`'s formula is `count ~ 1` — with the partial-bin `exposure`
offset, `exp(intercept)` is the per-bin rate (no per-stop dwell offset needed).

---

## fit_per_bin_rate_model / cross_validate_per_bin_rate_model

Home: new `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py` (parallel to `per_stop_rate_model.py`). Mirrors `fit_per_stop_rate_model`'s return-dict shape (`coef_*`, `bse_*`, `pval_*`, `rr_*`, fit stats) so downstream code is uniform.

```python
def fit_per_bin_rate_model(
    design: pd.DataFrame,        # one session's aggregate_to_timebins output
    spec: PerBinRateModelSpec,
    *, min_bins: int = 200, min_events: int = 20,
) -> dict | None:
    """Per-session Poisson GLM: spec.formula() with offset=log(exposure).

    Returns coef_{term}/bse_/pval_/rr_ for every fitted term, plus n_bins,
    n_events, llf, aic, deviance, AND the per-bin fitted conditional intensity
    (predicted_count -> the GOF input). `dispersion = deviance / df_resid` is
    recorded for the overdispersion check (Poisson target ~1). None if too few
    bins/events, <2 reward levels (when reward terms present), or statsmodels
    raises (perfect separation at short self-history lags is caught -> None)."""
```
- Fit via `smf.glm(spec.formula(), data=design, family=Poisson(), offset=np.log(design['exposure']))`.
- `rr_{term} = exp(coef_{term})`. Patsy term names for `reward_lag{j}`/`onset_lag{j}` are the bare column names; the spline terms are `bs(tc, df=4)[k]` — record them but the scientific terms of interest are the reward/onset lags.
- Expose `predicted` (per-bin fitted mean count, `= results.predict()`) so the GOF contract can build the conditional intensity without refitting.

```python
def cross_validate_per_bin_rate_model(
    design, spec, *, n_folds=10, min_bins=200, seed=0
) -> dict | None:
    """10-fold over STOPS (not bins): hold out whole stops, fit on the rest,
    score held-out fit. Returns cv_ll_per_bin_mean/std, and cv_bits_per_event =
    SUM_heldout(ll_model - ll_M0_const) / (n_events * ln2) — the numerator is the
    SUMMED (not mean) held-out per-bin log-likelihood difference over the same
    held-out bins, with M0_const refit on each train fold; units = bits per event
    ('bits/event better than constant rate'). (Mirroring Family A's per-stop *mean*
    LL here would be wrong by a factor of n_bins.)
    Mirrors cross_validate_per_stop_rate_model's stop-based folds (per_stop_rate_model.py:160) so a
    stop's bins never split across folds. n_folds default is 10 (decision)."""
```

---

## per_bin_rate_gof

Home: `per_bin_rate_model.py`. **Reuses** `non_local_detector.model_checking.sorted_spikes`
(verified present: `TimeRescaling`, `point_process_residuals`, `plot_ks`, `plot_qq`,
`plot_rescaled_ISI_autocorrelation`) — do not re-implement time-rescaling. Treats the
event train exactly as a spike train: `is_spike = count > 0` per bin, `trial_id = stop_id`.

```python
def per_bin_rate_gof(design: pd.DataFrame, fitted: dict, *,
                     adjust_for_short_trials: bool = True) -> dict:
    """Point-process goodness-of-fit via the time-rescaling theorem (Brown et al.
    2002). `fitted['predicted']` is the per-bin conditional intensity. Returns
    ks_statistic, ks_within_band (ks < 1.36/sqrt(n_events)), max_isi_autocorr,
    dispersion (= deviance/df_resid), n_events. Plot helpers (KS / QQ / ISI-autocorr
    / point_process_residuals-vs-covariate) are emitted for example sessions only."""
    from non_local_detector.model_checking.sorted_spikes import TimeRescaling
    tr = TimeRescaling(
        conditional_intensity=fitted["predicted"],          # per-bin fitted mean count
        is_spike=design["count"].to_numpy() > 0,
        trial_id=design["stop_id"].to_numpy(),              # rescale within stop
        adjust_for_short_trials=adjust_for_short_trials,    # Wiener 2003 — handles short/censored stops
    )
    ...
```

- **Units verified:** `TimeRescaling` integrates `conditional_intensity` with `cumulative_trapezoid`
  at dx=1 (per sample), so the correct input is the per-bin **fitted mean count** (`results.predict()`),
  NOT a rate in Hz (passing Hz gives KS≈0.95 on true-Poisson data; mean count gives KS≈0.02).
- **Discretization precondition:** the trapezoid rescaling is only unbiased when per-bin mean count
  ≪ 1. Real per-bin means here are ≈0.005 (SWR) / ≈0.009 (non-local), so this holds on average — but
  in high-rate bins (immediately post-stop / bursty post-reward) local `mu` is larger, where the
  trapezoid adds a KS bias. So the `M0_const` KS failure must be **cross-checked against `dispersion`
  and the rescaled-ISI autocorrelation** (neither is discretization-sensitive); report mean AND **max**
  per-bin predicted count alongside KS so a reader can rule out the artifact.
- **`adjust_for_short_trials=True` is required** — our stops are short and variable-length;
  this is the censoring correction that makes time-rescaling valid here. (It corrects stops *shorter*
  than the window; it does NOT recover events beyond `TMAX` — the process is right-truncated at 4 s,
  see overview.)
- The **constant `M0_const` is expected to FAIL** the KS band (events are refractory+bursty,
  non-Poisson) — that failure is the diagnostic that motivates the self-history block;
  `M3_hist`/`M6_time_hist`/`Thist`-equivalents should walk the KS plot into the band.
- Overdispersion: `dispersion > 1` for models lacking the history term signals the clustering
  the autoregressive block should absorb.

---

## Tidy output schema

`Processed-Data/replay_content/models_per_bin_rate/all_models.pkl` — one row per `(outcome, model, animal, day, epoch)`, mirroring `models_per_stop_rate/all_models.pkl` so the two families concatenate/compare on `[outcome, model, animal, day, epoch]`.

Columns: `outcome` (one of POPULATIONS keys), `model` (one of the `MODEL_PER_BIN_RATE` keys M0_const…M9_rew_x_time), `animal, day, epoch`, then `coef_*`, `bse_*`, `pval_*`, `rr_*` for each fitted term, plus `n_bins, n_events, llf, aic, deviance, dispersion`, the CV fields `cv_ll_per_bin_mean, cv_ll_per_bin_std, cv_bits_per_event, cv_n_folds`, and the GOF fields `ks_statistic, ks_within_band, max_isi_autocorr`. Sessions that fail to fit contribute no row (same as `fit_per_stop_rate_hierarchy`).

Within-animal aggregation is **not** baked into the file — it is computed at read time (per-session medians + `wilcoxon` paired tests), exactly as `fit_per_stop_rate_models` leaves it. The plotters (Phase 4) do this.
