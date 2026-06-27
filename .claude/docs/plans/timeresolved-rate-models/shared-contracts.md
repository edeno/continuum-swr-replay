# Shared contracts

Cross-phase contracts. Each is defined once here; phases link in by anchor and must not weaken these signatures/semantics.

Index:
- [attach_is_swr_to_onsets](#attach_is_swr_to_onsets) — Phase 1 defines, Phases 1/3 use.
- [aggregate_to_timebins](#aggregate_to_timebins) — Phase 1 defines, Phases 1/3 use.
- [POPULATIONS](#populations) — Phase 1 defines, Phases 1/3/4 use.
- [TimeResolvedModelSpec + MODEL_TIMERESOLVED](#timeresolvedmodelspec) — Phase 2 defines, Phases 3/4 use.
- [fit_timeresolved_model / cross_validate_timeresolved_model](#fit_timeresolved_model--cross_validate_timeresolved_model) — Phase 2 defines, Phase 3 uses.
- [models_timeresolved/all_models.pkl schema](#tidy-output-schema) — Phase 3 produces, Phase 4 reads.

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
) -> pd.DataFrame:
    """One row per (stop, bin) up to ``tmax``, at ``bin_s`` resolution.

    Columns: animal, day, epoch, stop_id, binj, tc (=bin-center s, time-since-
    stop-onset), exposure (s of immobility in the bin), reward_lag1..reward_lags,
    count (population onsets in the bin), onset_lag1..self_history_lags
    (same-population onset counts j bins back, within stop, reset at boundary).

    Bins are built from ``stop_table.immobility_time`` (continuous), independent
    of the 4 ms dense grid, so non-4-ms widths (e.g. 15 ms) are exact.
    ``exposure`` of the last bin is the partial remainder.
    """
```
Semantics nailed down:
- `nbin = min(ceil(immobility_time / bin_s), int(tmax / bin_s))`; bins built via `np.repeat`/`np.concatenate(np.arange(n))` (the `build_bins` idiom).
- `exposure = min(bin_s, immobility_time - binj*bin_s)`; drop rows with `exposure <= 1e-4`.
- `count`: `onset_times` (already filtered to one population) binned by `floor(t_since_stop_onset / bin_s)`, joined on `(stop_id, binj)`, fill 0.
- `onset_lag{j} = groupby('stop_id')['count'].shift(j).fillna(0)` after sorting by `(stop_id, binj)` — within-stop only.
- `reward_lag{j}`: carried from `stop_table` by `stop_id`.

---

## POPULATIONS

Home: `_helpers.py` (a small selector) or `model_specs.py`. The four outcome tags, mirroring `fit_rate_models.RESPONSE_COLS`:

```python
# given onset_df already passed through attach_is_swr_to_onsets:
POPULATIONS = {
    "swr":              lambda o: o[o.event_type == "SWR"],
    "non_local":        lambda o: o[o.event_type == "non_local"],
    "non_local_swr":    lambda o: o[(o.event_type == "non_local") & (o.is_swr == True)],   # noqa: E712
    "non_local_no_swr": lambda o: o[(o.event_type == "non_local") & (o.is_swr == False)],  # noqa: E712
}
```
Names match the per-stop `RESPONSE_COLS` keys exactly so the two families' tidy outputs join on `outcome`.

---

## TimeResolvedModelSpec

Home: `model_specs.py`, appended after `MODEL_RATE`/`Rlag` (after line 484). A *parallel* dataclass to `ModelSpec` (do not modify `ModelSpec`).

```python
@dataclass(frozen=True)
class TimeResolvedModelSpec:
    name: str                      # "T0","T1","Tlag","Thist"
    reward_lags: int               # 0 (T0) | 1 (T1) | N_REWARD_LAGS (Tlag/Thist)
    self_history_lags: int         # 0 except Thist
    bin_s: float                   # 0.25 (T0/T1/Tlag) | 0.015 (Thist)
    tmax: float                    # 4.0
    spline_df: int                 # 4
    description: str
    parent: str | None

    def formula(self) -> str:
        terms = [f"bs(tc, df={self.spline_df})"]
        terms += [f"reward_lag{j}" for j in range(1, self.reward_lags + 1)]
        terms += [f"onset_lag{j}" for j in range(1, self.self_history_lags + 1)]
        return "count ~ " + " + ".join(terms)

MODEL_TIMERESOLVED: dict[str, TimeResolvedModelSpec] = {
    "T0":   TimeResolvedModelSpec("T0",   0,              0,  0.25, 4.0, 4, "quiescence time course (baseline)", None),
    "T1":   TimeResolvedModelSpec("T1",   1,              0,  0.25, 4.0, 4, "+ immediate reward (reward_lag1)", "T0"),
    "Tlag": TimeResolvedModelSpec("Tlag", N_REWARD_LAGS,  0,  0.25, 4.0, 4, "+ reward kernel (reward_lag1..10)", "T1"),
    "Thist":TimeResolvedModelSpec("Thist",N_REWARD_LAGS, 10,  0.015,4.0, 4, "+ self-history (onset_lag1..10, 15ms)", "Tlag"),
}
```
Notes: `Thist` uses 15 ms bins (refractoriness/burstiness) and 10 self-history lags = 150 ms; the others use 250 ms cells. `bin_s`/`reward_lags`/`self_history_lags` drive both `aggregate_to_timebins` and `formula()`, so spec is the single source of truth.

---

## fit_timeresolved_model / cross_validate_timeresolved_model

Home: new `src/continuum_swr_replay/analysis/replay/timeresolved_model.py` (parallel to `rate_model.py`). Mirrors `fit_rate_model`'s return-dict shape (`coef_*`, `bse_*`, `pval_*`, `rr_*`, fit stats) so downstream code is uniform.

```python
def fit_timeresolved_model(
    design: pd.DataFrame,        # one session's aggregate_to_timebins output
    spec: TimeResolvedModelSpec,
    *, min_bins: int = 200, min_events: int = 20,
) -> dict | None:
    """Per-session Poisson GLM: spec.formula() with offset=log(exposure).

    Returns coef_{term}/bse_/pval_/rr_ for every fitted term, plus n_bins,
    n_events, llf, aic, deviance. None if too few bins/events, <2 reward levels
    (when reward terms present), or statsmodels raises (perfect separation at
    short self-history lags is caught -> None for that session)."""
```
- Fit via `smf.glm(spec.formula(), data=design, family=Poisson(), offset=np.log(design['exposure']))`.
- `rr_{term} = exp(coef_{term})`. Patsy term names for `reward_lag{j}`/`onset_lag{j}` are the bare column names; the spline terms are `bs(tc, df=4)[k]` — record them but the scientific terms of interest are the reward/onset lags.

```python
def cross_validate_timeresolved_model(
    design, spec, *, n_folds=5, min_bins=200, seed=0
) -> dict | None:
    """K-fold over STOPS (not bins): hold out whole stops, fit on the rest,
    score mean held-out per-bin Poisson log-likelihood. Returns
    cv_ll_per_bin_mean/std/n_folds. Mirrors cross_validate_rate_model's
    stop-based folds (rate_model.py:160) so events in a stop never split
    across folds."""
```

---

## Tidy output schema

`Processed-Data/replay_content/models_timeresolved/all_models.pkl` — one row per `(outcome, model, animal, day, epoch)`, mirroring `models_rate/all_models.pkl` so the two families concatenate/compare on `[outcome, model, animal, day, epoch]`.

Columns: `outcome` (one of POPULATIONS keys), `model` (T0/T1/Tlag/Thist), `animal, day, epoch`, then `coef_*`, `bse_*`, `pval_*`, `rr_*` for each fitted term, plus `n_bins, n_events, llf, aic, deviance`, and `cv_ll_per_bin_mean, cv_ll_per_bin_std, cv_n_folds`. Sessions that fail to fit contribute no row (same as `fit_rate_hierarchy`).

Within-animal aggregation is **not** baked into the file — it is computed at read time (per-session medians + `wilcoxon` paired tests), exactly as `fit_rate_models` leaves it. The plotters (Phase 4) do this.
