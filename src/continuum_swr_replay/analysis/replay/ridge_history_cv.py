"""Ridge-penalised, cross-validated self vs self+cross history comparison.

Within one session a non-local stream holds about one expected onset in the
other stream's 150 ms cross window, so the unpenalised cross rungs
(``Mselfcross_dur``) fail to converge in most sessions and in one animal's
pooled fit: a lag with no events has its maximum-likelihood coefficient at
``-inf``. This module replaces the AIC comparison with a held-out one:

- an animal's sessions are stacked with a session fixed effect, as
  :func:`duration_aware_history.fit_pooled_history` does;
- the history coefficients (self and cross, onset and end) carry a Gaussian
  prior ``lambda = 1 / sigma**2``; the intercept, the session effects, the time
  spline and the reward basis are unpenalised. The fit is the posterior mode,
  which is finite for every lag;
- folds are whole stops, assigned within each session, so every session is in
  every training set (its baseline is estimated) and no history window crosses
  a fold (history never crosses a stop);
- each model's ``sigma`` is chosen by inner cross-validation on the training
  folds of each outer fold, and the outer held-out Poisson log-likelihood of
  the self-only and the self+cross rung is compared fold by fold.

The fitter is Newton's method on the penalised log-likelihood with step
halving (the objective is strictly concave), in ``float64`` on the CPU.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy import linalg
from scipy.special import gammaln

from .duration_aware_history import HISTORY_BLOCKS
from .model_specs import PerBinRateModelSpec
from .per_bin_design import raised_cosine_history_basis

__all__ = [
    "FOLD_SEED",
    "MIN_SESSION_EVENTS",
    "N_FOLDS",
    "SIGMA_GRID",
    "RidgeFit",
    "assign_stop_folds",
    "compare_cross_history",
    "fit_ridge_poisson",
    "heldout_loglik",
    "nested_cv",
    "ridge_design",
    "ridge_kernels",
]

#: Prior SDs of a history coefficient (log rate ratio per bump) tried by the
#: inner cross-validation, strongest penalty first.
SIGMA_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
#: Outer folds of whole stops; the inner folds are the other outer folds.
N_FOLDS = 5
#: Seed of the stop-to-fold assignment.
FOLD_SEED = 20261030
#: A session with fewer response onsets than this is left out: an unpenalised
#: session effect with no events in a training set has its estimate at -inf.
MIN_SESSION_EVENTS = 10
#: Design columns of the history blocks (the penalised ones).
_HISTORY_COLUMN = re.compile(r"^(cross_)?(off)?hist_rcos_\d+$")


@dataclass(frozen=True)
class RidgeFit:
    """The posterior mode of one ridge Poisson fit.

    Attributes
    ----------
    params : np.ndarray, shape (n_features,)
    converged : bool
        The penalised log-likelihood changed by less than ``tol`` (relative).
    n_iter : int
    objective : float
        Penalised log-likelihood at ``params`` (without ``log y!``).
    """

    params: np.ndarray
    converged: bool
    n_iter: int
    objective: float


def ridge_design(
    design: pd.DataFrame, spec: PerBinRateModelSpec, *, group: str = "session"
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The rung's design matrix with a session effect, and its penalised columns.

    Returns
    -------
    X : np.ndarray, shape (n_bins, n_features)
        Column 0 is the intercept; ``C(group)`` is treatment coded.
    names : list of str
    penalized : np.ndarray of bool, shape (n_features,)
        True for the history columns (``*hist_rcos_k``).
    """
    rhs = spec.formula().split("~", 1)[1] + f" + C({group})"
    frame = patsy.dmatrix(rhs, design, return_type="dataframe")
    if len(frame) != len(design):
        raise ValueError(f"{spec.name}: patsy dropped {len(design) - len(frame)} rows")
    names = [str(c) for c in frame.columns]
    if names[0] != "Intercept":
        raise ValueError(f"{spec.name}: the first column is not the intercept")
    penalized = np.array([bool(_HISTORY_COLUMN.match(n)) for n in names])
    return frame.to_numpy(dtype=float), names, penalized


def _objective(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
    params: np.ndarray,
) -> float:
    eta = X @ params + offset
    return float(
        np.sum(weights * (y * eta - np.exp(eta))) - 0.5 * np.sum(penalty * params**2)
    )


def fit_ridge_poisson(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    penalty: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    start: np.ndarray | None = None,
    maxiter: int = 100,
    tol: float = 1e-10,
) -> RidgeFit:
    """Posterior mode of a log-link Poisson GLM under a Gaussian prior.

    Maximises ``sum_i w_i (y_i eta_i - exp(eta_i)) - (1/2) sum_j lambda_j
    beta_j**2`` with ``eta = X beta + offset``, by Newton steps with halving.

    Parameters
    ----------
    X : np.ndarray, shape (n_bins, n_features)
        Column 0 is the intercept.
    y, offset : np.ndarray, shape (n_bins,)
    penalty : np.ndarray, shape (n_features,)
        ``lambda_j = 1 / sigma_j**2``; 0 leaves a column free.
    weights : np.ndarray, shape (n_bins,), optional
        Row weights; 0/1 selects a training set without copying ``X``.
    start : np.ndarray, shape (n_features,), optional
        Warm start; by default the intercept-only rate.
    maxiter : int
    tol : float
        Relative change of the penalised log-likelihood that stops the fit.

    Returns
    -------
    RidgeFit
    """
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if start is None:
        params = np.zeros(X.shape[1])
        params[0] = np.log(np.sum(w * y) / np.sum(w * np.exp(offset)))
    else:
        params = np.asarray(start, dtype=float).copy()
    obj = _objective(X, y, offset, w, penalty, params)
    for it in range(1, maxiter + 1):
        mu = np.exp(X @ params + offset)
        grad = X.T @ (w * (y - mu)) - penalty * params
        root = X * np.sqrt(w * mu)[:, None]
        info = root.T @ root
        info[np.diag_indices_from(info)] += penalty
        step = linalg.solve(info, grad, assume_a="pos")
        t = 1.0
        while True:
            trial = params + t * step
            trial_obj = _objective(X, y, offset, w, penalty, trial)
            if trial_obj >= obj or t < 1e-10:
                break
            t /= 2
        change = trial_obj - obj
        params, obj = trial, trial_obj
        if abs(change) <= tol * (abs(obj) + 1.0):
            return RidgeFit(params, True, it, obj)
    return RidgeFit(params, False, maxiter, obj)


def heldout_loglik(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    params: np.ndarray,
    rows: np.ndarray,
) -> float:
    """Poisson log-likelihood of ``rows`` (boolean mask) at ``params``."""
    eta = X[rows] @ params + offset[rows]
    yr = y[rows]
    return float(np.sum(yr * eta - np.exp(eta) - gammaln(yr + 1.0)))


def assign_stop_folds(
    design: pd.DataFrame,
    *,
    group: str = "session",
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> np.ndarray:
    """Each row's fold: whole stops, dealt evenly within each session.

    Sessions in sorted order, each session's stops (sorted) permuted by one
    generator seeded with ``seed`` and dealt round-robin, so every session with
    at least ``n_folds`` stops has stops in every fold.

    Returns
    -------
    np.ndarray of int, shape (n_bins,)
    """
    rng = np.random.default_rng(seed)
    stops = (
        design[[group, "stop_id"]]
        .drop_duplicates()
        .sort_values([group, "stop_id"])
        .reset_index(drop=True)
    )
    sizes = stops.groupby(group, sort=True).size()
    stops["fold"] = np.concatenate([rng.permutation(n) % n_folds for n in sizes])
    return (
        design[[group, "stop_id"]]
        .merge(stops, on=[group, "stop_id"], how="left", validate="many_to_one")["fold"]
        .to_numpy()
    )


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    penalized: np.ndarray,
    folds: np.ndarray,
    *,
    sigma_grid: Sequence[float] = SIGMA_GRID,
    n_folds: int = N_FOLDS,
) -> dict:
    """Outer held-out log-likelihood with ``sigma`` chosen by inner CV.

    For outer fold ``k`` the inner folds are the other outer folds: each
    ``sigma`` is fit on the folds other than ``k`` and ``j`` and scored on
    ``j``; the ``sigma`` with the largest summed inner log-likelihood is the
    one fold ``k`` is scored at. Every ``sigma`` is also fit on all folds but
    ``k`` and scored on ``k`` (``outer_curve``), which picks the ``sigma`` of
    the full-data fit.

    Returns
    -------
    dict
        ``sigma`` (grid, strongest first), ``outer_ll`` (per fold, at the inner
        choice), ``chosen_sigma`` (per fold), ``outer_curve`` (folds x grid),
        ``n_fits``, ``n_nonconverged``.
    """
    grid = sorted(sigma_grid)
    lam = {s: np.where(penalized, 1.0 / s**2, 0.0) for s in grid}
    outer_ll = np.zeros(n_folds)
    chosen = np.zeros(n_folds)
    curve = np.zeros((n_folds, len(grid)))
    n_fits = n_bad = 0
    for k in range(n_folds):
        inner = np.zeros(len(grid))
        for j in range(n_folds):
            if j == k:
                continue
            train = ((folds != k) & (folds != j)).astype(float)
            start = None
            for i, s in enumerate(grid):
                fit = fit_ridge_poisson(
                    X, y, offset, lam[s], weights=train, start=start
                )
                n_fits += 1
                n_bad += not fit.converged
                start = fit.params
                inner[i] += heldout_loglik(X, y, offset, fit.params, folds == j)
        best = int(np.argmax(inner))
        train = (folds != k).astype(float)
        start = None
        for i, s in enumerate(grid):
            fit = fit_ridge_poisson(X, y, offset, lam[s], weights=train, start=start)
            n_fits += 1
            n_bad += not fit.converged
            start = fit.params
            curve[k, i] = heldout_loglik(X, y, offset, fit.params, folds == k)
        outer_ll[k] = curve[k, best]
        chosen[k] = grid[best]
    return {
        "sigma": grid,
        "outer_ll": outer_ll,
        "chosen_sigma": chosen,
        "outer_curve": curve,
        "n_fits": n_fits,
        "n_nonconverged": n_bad,
    }


def ridge_kernels(
    fit: RidgeFit,
    names: list[str],
    covariance: np.ndarray,
    spec: PerBinRateModelSpec,
) -> dict[str, dict[str, list[float]]]:
    """Per-lag log rate ratio and its posterior SD for each history block.

    ``covariance`` is the inverse penalised information at the mode (the
    Laplace posterior covariance), so the SD includes the prior's shrinkage.
    """
    out = {}
    for block, (prefix, _, rel) in HISTORY_BLOCKS.items():
        cols = [i for i, n in enumerate(names) if re.fullmatch(rf"{prefix}\d+", n)]
        if not cols:
            continue
        if rel == "self":
            n_lags, n_basis = spec.self_history_lags, spec.self_history_basis
        else:
            n_lags, n_basis = spec.cross_history_lags, spec.cross_history_basis
        if len(cols) != n_basis:
            raise ValueError(f"{block}: {len(cols)} columns, spec says {n_basis}")
        basis = raised_cosine_history_basis(n_lags, n_basis)
        beta = fit.params[cols]
        sigma = covariance[np.ix_(cols, cols)]
        out[block] = {
            "log_rr": (beta @ basis).tolist(),
            "sd": np.sqrt(np.einsum("kj,kl,lj->j", basis, sigma, basis)).tolist(),
            "coef": beta.tolist(),
        }
    return out


def compare_cross_history(
    design: pd.DataFrame,
    specs: Mapping[str, PerBinRateModelSpec],
    *,
    group: str = "session",
    sigma_grid: Sequence[float] = SIGMA_GRID,
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> dict:
    """Held-out comparison of a self-only and a self+cross rung on one design.

    Parameters
    ----------
    design : pd.DataFrame
        A stacked, frozen duration-aware design with a ``group`` column, the
        same rows for both rungs (``build_design(..., ladder=...)``).
    specs : mapping
        Two rungs, the self-only first and the self+cross second.

    Returns
    -------
    dict
        ``folds`` (one row per outer fold: events, each rung's held-out
        log-likelihood and chosen sigma, the gain and the gain per event),
        ``gain_per_event`` (summed over folds, nats per held-out onset),
        ``n_events``, ``n_bins``, ``n_sessions``, and per rung ``models``
        (the CV record, the full-data sigma, convergence, kernels, seconds).
    """
    (self_name, self_spec), (cross_name, cross_spec) = specs.items()
    folds = assign_stop_folds(design, group=group, n_folds=n_folds, seed=seed)
    y = design["count"].to_numpy(dtype=float)
    offset = np.log(design["exposure"].to_numpy(dtype=float))
    models: dict[str, dict] = {}
    for name, spec in ((self_name, self_spec), (cross_name, cross_spec)):
        t0 = time.perf_counter()
        X, names, penalized = ridge_design(design, spec, group=group)
        cv = nested_cv(
            X, y, offset, penalized, folds, sigma_grid=sigma_grid, n_folds=n_folds
        )
        full_sigma = cv["sigma"][int(np.argmax(cv["outer_curve"].sum(axis=0)))]
        lam = np.where(penalized, 1.0 / full_sigma**2, 0.0)
        full = fit_ridge_poisson(X, y, offset, lam)
        mu = np.exp(X @ full.params + offset)
        root = X * np.sqrt(mu)[:, None]
        info = root.T @ root
        info[np.diag_indices_from(info)] += lam
        covariance = linalg.inv(info, check_finite=True)
        models[name] = {
            "cv": {k: np.asarray(v).tolist() for k, v in cv.items()},
            "full_sigma": full_sigma,
            "full_converged": full.converged,
            "full_n_iter": full.n_iter,
            "n_features": X.shape[1],
            "n_penalized": int(penalized.sum()),
            "kernels": ridge_kernels(full, names, covariance, spec),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        del X, root
    events = np.array([y[folds == k].sum() for k in range(n_folds)])
    ll_self = np.asarray(models[self_name]["cv"]["outer_ll"])
    ll_cross = np.asarray(models[cross_name]["cv"]["outer_ll"])
    table = pd.DataFrame(
        {
            "fold": np.arange(n_folds),
            "n_events": events,
            f"ll_{self_name}": ll_self,
            f"ll_{cross_name}": ll_cross,
            f"sigma_{self_name}": models[self_name]["cv"]["chosen_sigma"],
            f"sigma_{cross_name}": models[cross_name]["cv"]["chosen_sigma"],
            "gain": ll_cross - ll_self,
        }
    )
    table["gain_per_event"] = table["gain"] / table["n_events"]
    return {
        "folds": table,
        "gain_per_event": float(table["gain"].sum() / events.sum()),
        "n_folds_positive": int((table["gain"] > 0).sum()),
        "n_events": int(y.sum()),
        "n_bins": int(len(y)),
        "n_sessions": int(design[group].nunique()),
        "models": models,
    }
