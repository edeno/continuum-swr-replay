# Shared contracts

[← back to PLAN.md](PLAN.md)

1. [Reward-history kernel](#reward-history-kernel) — phases 1, 3, 4
2. [Common-cohort freeze for rate designs](#common-cohort-freeze-for-rate-designs) — phases 1, 3
3. [Fraction response](#fraction-response) — phases 2a, 2b
4. [Content estimator result schema](#content-estimator-result-schema) — phases 2a, 2b
5. [Fit provenance sidecar](#fit-provenance-sidecar) — phases 1, 2a, 2b, 3
6. [Standardization entry point](#standardization-entry-point) — phases 2a, 5a

## Reward-history kernel

The only reward-history block in any rate ladder is three normalized raised-cosine features over
the last ten known outcomes:

```
B = raised_cosine_history_basis(n_lags=10, n_basis=3)        # (3, 10); log-spaced bumps
lags[t, j-1] = reward (0/1) of the j-th most recent outcome known at query time t,
               NaN when fewer than j outcomes are known (lagged_trial_rewards)
reward_history_rcos_k[t] = Σ_j B[k, j-1] · lags[t, j-1] / Σ_j B[k, j-1]   over finite lags only
                          = NaN when no lag in bump k's support is known
```

Bump supports (weight > 0.05): bump 0 lags 1–7 (weights 1.00, 0.79, 0.54, …), bump 1 lags 1–10,
bump 2 lags 2–10. Consequences, not to be weakened:

- Features 0 and 1 exist from the first known outcome; feature 2 from the second. Nothing is
  dropped for this: a stop with fewer than two known outcomes has its undefined features set to
  0.0 and carries `reward_history_incomplete = 1` (else 0). That indicator enters every rate rung
  that carries the basis as a main effect only (never inside an `is_swr *` expansion), is a
  nuisance term, and is never interpreted or aggregated. The rate-ladder common cohort is every
  stop with positive immobility. The affected stops are an unbroken prefix of each epoch (median
  6 stops / 3 events per session, second outcome known ~15 s after session start).
- Each feature is a weighted reward rate in [0, 1]; coefficients are reported per unit, not per SD.
- Feature 0 is the immediate component (lag-1 weight 1.0); the claims that used
  `known_reward_outcome_lag_1` read from it. `is_swr:reward_history_rcos_0` and
  `reward_history_rcos_0:bs(tc)` replace the lag-1 interaction and the reward × time term.
- The ten lag dummies and the window-10 fraction remain persisted stop metadata and are never
  ladder predictors.
- The same function computes the stop-level values (bout start) and, in phase 3, the per-bin values
  after the outcome-boundary shift, from the same lag matrix.

## Common-cohort freeze for rate designs

One function, used by every rate runner before any rung is fit:

```python
# src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py

def freeze_per_bin_cohort(
    design: pd.DataFrame,
    specs: Iterable[PerBinRateModelSpec],
    *,
    extra_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Keep only bins complete for the UNION of every spec's design columns.

    Returns (frozen_design, report) with report =
    {"n_bins_before", "n_bins_after", "n_stops_before", "n_stops_after",
     "n_event_bins_before", "n_event_bins_after", "required_columns"}.
    Rows with exposure <= 0 are removed here as well. The returned frame carries
    ``attrs["per_bin_cohort_frozen"] = sorted(required_columns)``.
    """
```

Invariants (do not weaken):

- `fit_per_bin_rate_model` and `cross_validate_per_bin_rate_model` raise `ValueError` if any
  needed column contains NaN or any `exposure <= 0` remains, instead of dropping rows. They never
  subset the frame.
- Every rung of a ladder fit on one (session, population) design reports the same `n_bins` and
  `n_stops`. `compare_nested_models` NaNs the comparison when `n_bins` or `n_stops` differ, in
  addition to the existing `n_events` guard.
- The per-stop family has the same contract through `freeze_common_cohort` (already in
  `cohort_selection.py`) applied once per session over the union of `MODEL_PER_STOP_RATE`
  predictors with `offset_col="immobility_duration_s"`; `fit_per_stop_rate_model` raises on NaN
  instead of dropping.
- The cohort audit attributes union-drops to `n_drop_common_cohort_other` when the row is complete
  for the model's own predictors and to `n_drop_own_predictor_missing` otherwise.

## Fraction response

`arm_fraction` (event-arm table, already persisted at `tables_arm_patch.py:174-177`) and
`patch_fraction` (new, phase 2a) are the shipped content responses. Definition:

```
arm_fraction[e, a]   = arm_count_hard[e, a] / (n_nonlocal_bins[e] - n_undecoded_bins[e])
patch_fraction[e, p] = patch_count_hard[e, p] / (n_nonlocal_bins[e] - n_undecoded_bins[e])
```

Invariants:

- Sums to exactly 1.0 over the candidate set of every event with at least one real-arm bin; NaN
  for events with none (those events are dropped by `freeze_common_cohort(drop_zero_response=True)`).
- `fit_conditional_poisson` accepts it unchanged (its likelihood is linear in `y`); the analytic
  null becomes `-n_events * log(n_categories)`.
- `arm_count_hard` / `patch_count_hard` remain persisted and remain the response for the
  Dirichlet-multinomial fitter (phase 2b) and for the retired-count comparison. Runners select the
  response by `--response {fraction,hard_count}`; the default is `fraction`.

## Content estimator result schema

Every content fitter (`fit_conditional_poisson`, `fit_conditional_logit`,
`fit_dirichlet_multinomial`) returns a dict with these keys so runners, `model_comparison`, and
`contrast_aggregation` treat them uniformly:

| key | meaning |
| --- | --- |
| `coef_<term>`, `bse_<term>`, `pval_<term>` | point estimate, standard error, p per predictor |
| `bse_cluster_<term>` | event-clustered sandwich SE (conditional Poisson on fractions; phase 2a) — absent for DM, whose model-based SE is already event-level |
| `n_events`, `n_rows`, `n_params`, `llf`, `llf_null`, `aic`, `bic`, `pseudo_r2` | as today |
| `llf_includes_constants` | bool; `False` for statsmodels conditional fits, `True` for DM (see designs.md); cross-estimator AIC must equalize |
| `converged`, `separated`, `n_convergence_warnings` | as today |
| `coef_log_phi_<term>` / `bse_log_phi_<term>` | DM concentration coefficients (intercept and `is_swr`) |
| `estimator` | `"conditional_poisson_fraction"`, `"conditional_poisson_count"`, `"conditional_logit_dominant"`, `"dirichlet_multinomial"` |

`contrast_aggregation.aggregate_contrast` uses `bse_cluster_<term>` when present for
`typical_session_se`, else `bse_<term>`.

## Fit provenance sidecar

Every runner that writes a fitted artifact also writes `<artifact-stem>.provenance.json` next to
it:

```json
{
  "artifact": "all_models.pkl",
  "written_at": "2026-09-05T14:03:11-07:00",
  "git_hash": "…",
  "estimator": "conditional_poisson_fraction",
  "inputs": {
    "event_arm_table.parquet": {"sha256": "…", "mtime": "…", "covariate_schema_version": "covariate_contract_v5_dio_poke_trials"},
    "stop_table.parquet": {"sha256": "…", "mtime": "…", "covariate_schema_version": "…"}
  },
  "cli": ["fit_replay_content_models.py", "--hierarchy", "arm"]
}
```

Writer: `analysis/replay/fit_provenance.py:write_fit_provenance(artifact_path, inputs: dict[str, Path], estimator: str, cli: list[str])`.
Reader: `check_fit_provenance(artifact_path, inputs) -> list[str]` returns the names of inputs
whose current SHA-256 differs; every summarizer/plotter that reads a fit calls it and raises
unless `--allow-stale` is passed. Phase 1 introduces the module; later phases wire their runners.

## Standardization entry point

`analysis/replay/standardization.attach_standardized(df, cols, reference=None, *, support_mask=None, support_label="all_rows")`
is the only z-scoring code path. Fit-time z-scoring in runners calls it with
`support_label="fit_cohort_all_rows"` and persists the returned reference next to the fit output
(`<artifact-stem>.z_reference.json`). No runner defines its own `_z`.
