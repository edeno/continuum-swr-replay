# Designs

[← back to PLAN.md](PLAN.md)

1. [Event-clustered sandwich covariance for the conditional Poisson](#event-clustered-sandwich-covariance) — phase 2a
2. [Dirichlet-multinomial content fitter](#dirichlet-multinomial-content-fitter) — phase 2b
3. [Arrival clock and per-bin reward lags in the per-bin design](#arrival-clock-and-per-bin-reward-lags) — phase 3
4. [Output-directory keying for content runs](#output-directory-keying) — phase 1
5. [Reward-history basis features](#reward-history-basis-features) — phase 1

## Reward-history basis features

```python
# src/continuum_swr_replay/analysis/replay/covariates/history/reward_history.py

def reward_basis_from_lags(lags: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a (n_query, n_lags) lag matrix onto (n_basis, n_lags) bumps.

    Each output is the bump-weighted mean over the *available* (finite) lags:
    ``sum_j B[k, j] * lags[:, j] / sum_j B[k, j]`` over finite ``lags[:, j]``.
    NaN where no lag in the bump's support is finite.
    """
    finite = np.isfinite(lags)
    values = np.where(finite, lags, 0.0)                      # (n_query, n_lags)
    num = values @ basis.T                                    # (n_query, n_basis)
    den = finite.astype(float) @ basis.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def reward_history_basis_features(
    query_times: np.ndarray,
    port_experiences: pd.DataFrame,
    *,
    n_lags: int = REWARD_HISTORY_N_LAGS,
    n_basis: int = REWARD_HISTORY_N_BASIS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, incomplete) at each query time.

    features : (n_query, n_basis) normalized raised-cosine reward-history features,
        with undefined entries (no known lag in the bump's support) set to 0.0.
    incomplete : (n_query,) float, 1.0 where any feature was undefined (fewer
        than two known outcomes), else 0.0. Structural-missingness indicator;
        enters rate models as a nuisance main effect.
    """
    lags = lagged_trial_rewards(query_times, port_experiences, k=n_lags)
    raw = reward_basis_from_lags(lags, raised_cosine_history_basis(n_lags, n_basis))
    incomplete = np.isnan(raw).any(axis=1).astype(float)
    return np.nan_to_num(raw, nan=0.0), incomplete
```

Stop-table columns: `reward_history_rcos_0..2` and `reward_history_incomplete`. In the per-bin
recomputation after the outcome-boundary shift (section 3) the indicator is recomputed from the
shifted lag matrix the same way, so a bin after a stop's first-ever outcome can flip from
incomplete to defined within the stop.

`raised_cosine_history_basis` moves to `design_constants.py` in phase 5b; until then
`reward_history.py` imports it from `_helpers` (an import already present there for
`build_well_to_arm_map`). Interpretation and cohort consequences are in the
[contract](shared-contracts.md#reward-history-kernel).

## Event-clustered sandwich covariance

The conditional Poisson log-likelihood on rows `(e, a)` with response `y_ea`, design `x_ea`, and
event totals `L_e = Σ_a y_ea` is

```
ℓ(β) = Σ_e [ Σ_a y_ea x_eaᵀβ − L_e log Σ_a exp(x_eaᵀβ) ]
```

Per-event score and observed information:

```
p_ea  = softmax_a(x_eaᵀβ)
g_e   = Σ_a (y_ea − L_e p_ea) x_ea                         # (p,)
H_e   = L_e [ Σ_a p_ea x_ea x_eaᵀ − (Σ_a p_ea x_ea)(Σ_a p_ea x_ea)ᵀ ]   # (p, p)
V     = H⁻¹ (Σ_e g_e g_eᵀ) H⁻¹,   H = Σ_e H_e
```

With the fraction response, `L_e = 1` for every event, so `H` is the one-draw-per-event
information and the meat corrects for within-event mixing being less variable than a single draw.

```python
# src/continuum_swr_replay/analysis/replay/fitting.py

def event_clustered_covariance(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Sandwich covariance of a conditional-Poisson fit, clustered by event.

    Parameters
    ----------
    X : (n_rows, n_params) design, rows grouped by ``groups``.
    y : (n_rows,) response (counts or fractions).
    groups : (n_rows,) event id per row.
    params : (n_params,) fitted coefficients.
    """
    order = np.argsort(groups, kind="stable")
    X, y, groups = X[order], y[order], groups[order]
    _, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    p = X.shape[1]
    bread = np.zeros((p, p))
    meat = np.zeros((p, p))
    eta = X @ params
    for s, c in zip(starts, counts, strict=True):
        sl = slice(s, s + c)
        w = np.exp(eta[sl] - eta[sl].max())
        prob = w / w.sum()
        total = y[sl].sum()
        xbar = prob @ X[sl]
        score = (y[sl] - total * prob) @ X[sl]
        bread += total * ((X[sl].T * prob) @ X[sl] - np.outer(xbar, xbar))
        meat += np.outer(score, score)
    bread_inv = np.linalg.pinv(bread)
    return bread_inv @ meat @ bread_inv
```

`fit_conditional_poisson` adds `bse_cluster_<term> = sqrt(diag(V))` after the fit. Validation:
on `simulate_with_event_covariates` data with `L_e = 1` the clustered SE equals the model SE
within 5 %; on sticky-bin data (`simulate_autocorrelated_counts`, ρ = 0.95, response converted to
fractions) the clustered SE matches the empirical sampling SD across 200 replicates within 20 %.

## Dirichlet-multinomial content fitter

Model, per event `e` with candidate set `A_e` (9 arms, 3 patches, or 6 leaves):

```
p_e     = softmax_a(x_eaᵀβ)                       # same conditioning as the conditional Poisson
log φ_e = γ₀ + γᵀ z_e                             # z_e event-level modifiers, default (is_swr,)
c_e ~ DirichletMultinomial(L_e, φ_e · p_e)

log P(c_e) = lgamma(L_e+1) − Σ_a lgamma(c_ea+1)                     # multinomial coefficient
           + lgamma(φ_e) − lgamma(L_e+φ_e)
           + Σ_a [ lgamma(c_ea + φ_e p_ea) − lgamma(φ_e p_ea) ]
```

φ → ∞ recovers the count multinomial; small φ says bins within an event are near-copies. The
multinomial coefficient is included so `llf` is a full likelihood (`llf_includes_constants=True`);
to compare AIC against a statsmodels conditional fit, add `Σ_e [lgamma(L_e+1) − Σ_a lgamma(c_ea+1)]`
to the statsmodels `llf` first (helper `multinomial_log_constant(y, groups)` in the same module).

```python
# src/continuum_swr_replay/analysis/replay/dirichlet_multinomial.py
from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def _pack_events(long_df, predictors, response_col, event_col, concentration_cols):
    df = long_df.sort_values([event_col]).reset_index(drop=True)
    ids, starts, counts = np.unique(df[event_col].to_numpy(), return_index=True, return_counts=True)
    n_cat = int(counts[0])
    if not (counts == n_cat).all():
        raise ValueError("every event must carry the same number of candidate rows")
    X = df[predictors].to_numpy(dtype=float).reshape(len(ids), n_cat, len(predictors))
    Y = df[response_col].to_numpy(dtype=float).reshape(len(ids), n_cat)
    Z = df.groupby(event_col, sort=True)[list(concentration_cols)].first().to_numpy(dtype=float) if concentration_cols else np.zeros((len(ids), 0))
    return ids, X, Y, Z


def _make_negloglike(X, Y, Z):
    X, Y, Z = jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)
    L = Y.sum(axis=1)
    const = gammaln(L + 1.0) - gammaln(Y + 1.0).sum(axis=1)
    n_beta = X.shape[2]

    def negloglike(theta):
        beta, gamma = theta[:n_beta], theta[n_beta:]
        logp = jax.nn.log_softmax(jnp.einsum("eak,k->ea", X, beta), axis=1)
        log_phi = gamma[0] + Z @ gamma[1:]
        phi = jnp.exp(log_phi)
        alpha = phi[:, None] * jnp.exp(logp)
        ll = (
            const
            + gammaln(phi) - gammaln(L + phi)
            + (gammaln(Y + alpha) - gammaln(alpha)).sum(axis=1)
        )
        return -ll.sum()

    return jax.jit(negloglike), jax.jit(jax.grad(negloglike)), jax.jit(jax.hessian(negloglike))


def fit_dirichlet_multinomial(
    long_df: pd.DataFrame,
    predictors: list[str],
    *,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    concentration_cols: tuple[str, ...] = ("is_swr",),
    n_categories: int = 9,
    fixed_log_phi: float | None = None,
) -> dict | None:
    """Fit c_e ~ DM(L_e, exp(log_phi_e) * softmax(X_e beta)) for one session.

    ``fixed_log_phi`` pins the concentration intercept (animal-pooled fallback
    for small sessions); the ``is_swr`` modifier is still estimated.
    """
    ids, X, Y, Z = _pack_events(long_df, predictors, response_col, event_col, concentration_cols)
    if len(ids) < 2 * (len(predictors) + 1 + Z.shape[1]):
        return None
    f, g, h = _make_negloglike(X, Y, Z)
    n_beta, n_gamma = X.shape[2], 1 + Z.shape[1]
    theta0 = np.zeros(n_beta + n_gamma)
    theta0[n_beta] = 0.0 if fixed_log_phi is None else fixed_log_phi
    res = minimize(lambda t: float(f(t)), theta0, jac=lambda t: np.asarray(g(t)), method="L-BFGS-B")
    if not res.success:
        return None
    cov = np.linalg.pinv(np.asarray(h(res.x)))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = [*predictors, "log_phi_intercept", *[f"log_phi_{c}" for c in concentration_cols]]
    llf = -float(res.fun)
    n_params = len(names)
    out = {f"coef_{k}": float(v) for k, v in zip(names, res.x, strict=True)}
    out.update({f"bse_{k}": float(v) for k, v in zip(names, se, strict=True)})
    out.update({
        "n_events": int(len(ids)), "n_rows": int(len(ids) * X.shape[1]), "n_params": n_params,
        "llf": llf, "llf_includes_constants": True,
        "aic": 2 * n_params - 2 * llf, "bic": n_params * np.log(len(ids)) - 2 * llf,
        "converged": bool(res.success), "separated": bool(np.any(se[:n_beta] * X.reshape(-1, n_beta).std(axis=0) > 10)),
        "estimator": "dirichlet_multinomial",
    })
    return out
```

Notes for the executor: `p`-values come from a Wald z on `coef/bse`; separation uses the same
effect-scale rule as `_fit_diagnostics.fit_status`. The per-event `log P(c_e)` vector is also
exposed (`event_loglik(theta)`) for held-out CV by `GroupKFold` over events, mirroring
`model_checking.cross_validate_conditional_poisson`.

Simulator (phase 2b) in `simulation.py`:

```python
def simulate_dirichlet_multinomial_counts(beta_true, *, log_phi_intercept, log_phi_is_swr=0.0,
                                          n_events=300, L_e=32, seed=None):
    """Draw p_e ~ Dirichlet(phi_e * softmax(X_e beta)), then c_e ~ Multinomial(L_e, p_e)."""
```

built on `simulate_with_event_covariates`'s design sampler so the same predictor vocabulary is
available.

## Arrival clock and per-bin reward lags

Two new stop-table columns (phase 3, `tables_stop.py` after line 147):

- `last_well_arrival_time_s` — `_strict_first` of `dot["_goal_arrival_time"]` over the bout's rows.
- `outcome_known_time_s` — `_strict_first` of a new per-timebin `_outcome_known_time` written by
  `add_trial_context` next to `_goal_arrival_time` (`context.py:615-618`), equal to
  `outcome_times[completed_clipped]` where `has_completed & outcome_observed`, else NaN.

`aggregate_to_timebins` gains two derived per-bin columns when those stop columns are present:

```python
left_edge_abs = out["stop_start"] + out["binj"] * bin_s
out["time_since_last_well_arrival_s"] = left_edge_abs + 0.5 * bin_s - out["last_well_arrival_time_s"]
out["time_since_outcome_known_s"] = left_edge_abs + 0.5 * bin_s - out["outcome_known_time_s"]
```

and, when the spec carries the reward basis and the outcome boundary falls inside the stop
(`stop_start < outcome_known_time_s <= stop_end`), the per-bin lag matrix is shifted at the
boundary and the three basis features are recomputed from it:

```python
lag_cols = [f"known_reward_outcome_lag_{j}" for j in range(1, REWARD_HISTORY_N_LAGS + 1)]
post = left_edge_abs >= out["outcome_known_time_s"]
crosses = (out["stop_start"] < out["outcome_known_time_s"]) & post & out["occupancy_state"].eq("at_well")
lags = out[lag_cols].to_numpy(dtype=float)
shifted = lags.copy()
shifted[crosses, 1:] = lags[crosses, :-1]
shifted[crosses, 0] = out.loc[crosses, "completed_trial_reward"].to_numpy(dtype=float)
raw = reward_basis_from_lags(shifted, raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS))
out["reward_history_incomplete"] = np.isnan(raw).any(axis=1).astype(float)
for k, col in enumerate(REWARD_HISTORY_COLUMNS):
    out[col] = np.nan_to_num(raw[:, k], nan=0.0)
```

The ten lag columns are carried into the design only for this recomputation and dropped before
return. Stops whose bout spans the arrival (`occupancy_state == "mixed"`) are excluded from the
shift and counted in `out.attrs["n_stops_arrival_spanning"]`, because the completed trial itself
changes inside those bouts.

Arrival-anchored spline variant (`model_specs.py`; window and knots decided from the at-well event
distribution: median 3.7 s after arrival, IQR 2.7–5.1 s, 87.6 % within 8 s, balanced across SWR
classes):

```python
TMAX_ARRIVAL = 8.0
ARRIVAL_SPLINE_KNOTS = (1.0, 2.0, 4.0, 6.0)
_ARRIVAL_SPLINE = f"bs(time_since_last_well_arrival_s, knots={ARRIVAL_SPLINE_KNOTS}, lower_bound=0.0, upper_bound={TMAX_ARRIVAL})"
```

Column names follow the event table's existing vocabulary (`time_since_last_well_arrival_s`,
`time_since_last_reward_s` at `tables_event.py`); the per-bin columns are
`time_since_last_well_arrival_s` and `time_since_outcome_known_s`, and the stop columns are
`last_well_arrival_time_s` and `outcome_known_time_s`. `PerBinRateModelSpec` gains
`time_clock: str = "stop"`; `"arrival"` swaps `_SPLINE` for `_ARRIVAL_SPLINE` in `formula()` and
`_design_columns` adds the arrival column. The runner drops bins with the arrival time `< 0` or
`> TMAX_ARRIVAL` for arrival-clock specs only, after the common-cohort freeze, and records the
retained share by population (expected ≈ 88 %, ≈ 89 % SWR / 87 % non-SWR). New rungs:
`M5a_time_arrival` (parent `M0_const`) and `M7ra_time_arrival_histcos_rew` (parent `M5a_time_arrival`).

## Output-directory keying

`_resolve_run_config` (`fit_replay_content_models.py:1113-1196`): when `--models` names a strict
subset of `default_content_model_names(hierarchy)` and `--output-dir` is not given, the output
directory becomes `<base>_<models joined by '-'>` (e.g. `models_M4-Mv-Mu-Mvu`); when `--leaf-only`
is set, append `_leaf`. Writing a subset into the canonical `models/` or `models_patch/` raises
before any fit. `_publish_staged_directory` is otherwise unchanged.
