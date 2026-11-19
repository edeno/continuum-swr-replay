"""Ridge-penalized, cross-validated self vs self+cross history comparison.

Within one session a non-local stream holds about one expected onset in the
other stream's 150 ms cross window, so the unpenalized cross rungs
(``Mselfcross_dur``) fail to converge in most sessions and in one animal's
pooled fit: a lag with no events has its maximum-likelihood coefficient at
``-inf``. This module replaces the AIC comparison with a held-out one:

- an animal's sessions are stacked with a session fixed effect, as
  :func:`duration_aware_history.fit_pooled_history` does;
- the history coefficients (self and cross, onset and end) carry a Gaussian
  prior ``lambda = 1 / sigma**2``; the intercept, the session effects, the time
  spline and the reward basis are unpenalized. The fit is the posterior mode,
  which is finite for every lag;
- folds are whole stops, assigned within each session, so every session is in
  every training set (its baseline is estimated) and no history window crosses
  a fold (history never crosses a stop);
- each model's ``sigma`` is chosen by inner cross-validation on the training
  folds of each outer fold, and the outer held-out Poisson log-likelihood of
  the self-only and the self+cross rung is compared fold by fold.

The fitter is Newton's method on the penalized log-likelihood with step
halving (the objective is strictly concave), in ``float64``. The
cross-validation runs every (training set, sigma) fit of a rung as one batch in
JAX (:func:`fit_ridge_poisson_batch`), on whichever device the caller chose;
:func:`fit_ridge_poisson` is the one-fit NumPy reference it is tested against.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import patsy
from scipy import linalg
from scipy.special import gammaln

from ...behavior.history_basis import raised_cosine_history_basis
from ..model_specs import PerBinRateModelSpec
from .duration_aware_history import HISTORY_BLOCKS

__all__ = [
    "FOLD_SEED",
    "MIN_SESSION_EVENTS",
    "N_FOLDS",
    "RIDGE_CV_FILE",
    "SIGMA_GRID",
    "RidgeFit",
    "assign_stop_folds",
    "compare_cross_history",
    "fit_ridge_poisson",
    "fit_ridge_poisson_batch",
    "heldout_loglik",
    "nested_cv",
    "ridge_design",
    "ridge_kernels",
]

#: Prior SDs of a history coefficient (log rate ratio per bump) tried by the
#: inner cross-validation, strongest penalty first.
SIGMA_GRID: tuple[float, ...] = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
#: Outer folds of whole stops; the inner folds are the other outer folds.
N_FOLDS = 5
#: Seed of the stop-to-fold assignment.
FOLD_SEED = 20261030
#: A session with fewer response onsets than this is left out: an unpenalized
#: session effect with no events in a training set has its estimate at -inf.
MIN_SESSION_EVENTS = 10
#: The file the per-animal cross-validation results are written to, in
#: ``paths.CROSS_HISTORY_RIDGE_CV_DIR``; the behavior summary reads it.
RIDGE_CV_FILE = "cross_history_ridge_cv.pkl"
#: Elements of the largest ``(rows, n_fits, n_features)`` block a batched pass
#: holds (1 GB in float64).
_CHUNK_ELEMENTS = 125_000_000
#: Design columns of the history blocks (the penalized ones).
_HISTORY_COLUMN = re.compile(r"^(cross_)?(off)?hist_rcos_\d+$")


@dataclass(frozen=True)
class RidgeFit:
    """The posterior mode of one ridge Poisson fit.

    Attributes
    ----------
    params : np.ndarray, shape (n_features,)
    converged : bool
        The penalized log-likelihood changed by less than ``tol`` (relative).
    n_iter : int
    objective : float
        Penalized log-likelihood at ``params`` (without ``log y!``).
    """

    params: np.ndarray
    converged: bool
    n_iter: int
    objective: float


def ridge_design(
    design: pd.DataFrame, spec: PerBinRateModelSpec, *, group: str = "session"
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The rung's design matrix with a session effect, and its penalized columns.

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

    Maximizes ``sum_i w_i (y_i eta_i - exp(eta_i)) - (1/2) sum_j lambda_j
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
        Relative change of the penalized log-likelihood that stops the fit.

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


def _batch_shape(n_rows: int, n_fits: int, n_features: int) -> int:
    """Rows per chunk so a ``(rows, n_fits, n_features)`` block is ~1 GB."""
    return int(max(1024, min(n_rows, _CHUNK_ELEMENTS // (n_fits * n_features))))


def _chunks(array: np.ndarray, rows: int, fill: float) -> jnp.ndarray:
    """``array`` padded along axis 0 to a multiple of ``rows`` and split."""
    n = array.shape[0]
    pad = (-n) % rows
    padded = np.concatenate(
        [array, np.full((pad, *array.shape[1:]), fill, dtype=array.dtype)]
    )
    return jnp.asarray(padded.reshape(-1, rows, *array.shape[1:]))


def _weights(fold: jnp.ndarray, keep: jnp.ndarray) -> jnp.ndarray:
    """``(rows, n_fits)`` training weights of a chunk's rows (padding is -1)."""
    return (keep.T[jnp.clip(fold, 0)] & (fold >= 0)[:, None]).astype(jnp.float64)


def _batch_objective(data, keep, lam, beta):
    """Penalized training log-likelihood of each fit, shape ``(n_fits,)``."""

    def body(acc, chunk):
        X, y, o, f = chunk
        eta = X @ beta.T + o[:, None]
        w = _weights(f, keep)
        return acc + jnp.sum(w * (y[:, None] * eta - jnp.exp(eta)), axis=0), None

    ll, _ = jax.lax.scan(body, jnp.zeros(beta.shape[0]), data)
    return ll - 0.5 * jnp.sum(lam * beta**2, axis=1)


def _score_and_information(data, keep, lam, beta):
    """Penalized score ``(n_fits, p)`` and information ``(n_fits, p, p)``."""
    n_fits, p = beta.shape

    def body(acc, chunk):
        g, h = acc
        X, y, o, f = chunk
        mu = jnp.exp(X @ beta.T + o[:, None])
        w = _weights(f, keep)
        g = g + X.T @ (w * (y[:, None] - mu))
        h = h + jnp.einsum("np,nb,nq->bpq", X, w * mu, X)
        return (g, h), None

    init = (jnp.zeros((p, n_fits)), jnp.zeros((n_fits, p, p)))
    (g, h), _ = jax.lax.scan(body, init, data)
    eye = jnp.eye(p)
    return g.T - lam * beta, h + lam[:, :, None] * eye[None]


@partial(jax.jit, static_argnames=("maxiter", "max_halvings"))
def _newton_batch(data, keep, lam, tol, maxiter, max_halvings):
    """Batched Newton with per-fit step halving; see :func:`fit_ridge_poisson`."""
    X, y, o, f = data
    n_fits, p = lam.shape

    def totals(acc, chunk):
        Xc, yc, oc, fc = chunk
        w = _weights(fc, keep)
        return (acc[0] + w.T @ yc, acc[1] + w.T @ jnp.exp(oc)), None

    (sy, se), _ = jax.lax.scan(totals, (jnp.zeros(n_fits), jnp.zeros(n_fits)), data)
    beta0 = jnp.zeros((n_fits, p)).at[:, 0].set(jnp.log(sy / se))
    obj0 = _batch_objective(data, keep, lam, beta0)
    done0 = jnp.zeros(n_fits, dtype=bool)

    def cond(state):
        it, _, _, done, _ = state
        return (it < maxiter) & ~jnp.all(done)

    def body(state):
        it, beta, obj, done, n_iter = state
        grad, info = _score_and_information(data, keep, lam, beta)
        step = jnp.linalg.solve(info, grad[..., None])[..., 0]
        step = jnp.where(done[:, None], 0.0, step)

        def h_cond(h):
            k, _, _, ok = h
            return (k < max_halvings) & ~jnp.all(ok)

        def h_body(h):
            k, t, _, ok = h
            t = jnp.where(ok, t, t / 2)
            trial = _batch_objective(data, keep, lam, beta + t[:, None] * step)
            ok = ok | (jnp.isfinite(trial) & (trial >= obj))
            return k + 1, t, trial, ok

        t0 = jnp.ones(n_fits)
        trial0 = _batch_objective(data, keep, lam, beta + step)
        ok0 = done | (jnp.isfinite(trial0) & (trial0 >= obj))
        _, t, trial, ok = jax.lax.while_loop(h_cond, h_body, (0, t0, trial0, ok0))
        move = ok & ~done
        new_beta = jnp.where(move[:, None], beta + t[:, None] * step, beta)
        new_obj = jnp.where(move, trial, obj)
        conv = jnp.abs(new_obj - obj) <= tol * (jnp.abs(new_obj) + 1.0)
        n_iter = n_iter + (~done).astype(jnp.int32)
        return it + 1, new_beta, new_obj, done | conv, n_iter

    state = (0, beta0, obj0, done0, jnp.zeros(n_fits, dtype=jnp.int32))
    _, beta, obj, done, n_iter = jax.lax.while_loop(cond, body, state)
    return beta, obj, done, n_iter


@jax.jit
def _heldout_batch(data, beta, score_fold):
    """Each fit's Poisson log-likelihood on its own scoring fold."""

    def body(acc, chunk):
        X, y, o, f = chunk
        eta = X @ beta.T + o[:, None]
        ll = (
            y[:, None] * eta
            - jnp.exp(eta)
            - jax.scipy.special.gammaln(y + 1.0)[:, None]
        )
        return acc + jnp.sum(
            jnp.where(f[:, None] == score_fold[None, :], ll, 0.0), 0
        ), None

    out, _ = jax.lax.scan(body, jnp.zeros(beta.shape[0]), data)
    return out


@jax.jit
def _information_at(data, keep, lam, beta):
    return _score_and_information(data, keep, lam, beta)[1]


def fit_ridge_poisson_batch(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    folds: np.ndarray,
    excluded: np.ndarray,
    penalties: np.ndarray,
    *,
    score_fold: np.ndarray | None = None,
    maxiter: int = 100,
    tol: float = 1e-10,
    max_halvings: int = 40,
    covariance: bool = False,
) -> dict:
    """Many ridge Poisson fits of one design at once, in JAX ``float64``.

    The batched twin of :func:`fit_ridge_poisson` (same objective, start,
    halving and stopping rule, applied fit by fit): fit ``b`` trains on the
    rows whose fold ``excluded[b]`` does not mark, under ``penalties[b]``.
    Every fit reads the same design in one pass per Newton step, so a sigma
    grid times a set of training folds costs about one fit's passes.

    Parameters
    ----------
    X : np.ndarray, shape (n_bins, n_features)
    y, offset : np.ndarray, shape (n_bins,)
    folds : np.ndarray of int, shape (n_bins,)
    excluded : np.ndarray of bool, shape (n_fits, n_folds)
    penalties : np.ndarray, shape (n_fits, n_features)
    score_fold : np.ndarray of int, shape (n_fits,), optional
        Fold each fit is scored on (held-out log-likelihood, with ``log y!``).
    covariance : bool
        Also return the inverse penalized information at each mode.

    Returns
    -------
    dict
        ``params`` (n_fits, n_features), ``objective``, ``converged``,
        ``n_iter``, and ``heldout_ll`` / ``covariance`` when asked.
    """
    with jax.enable_x64(True):
        n_fits, n_features = penalties.shape
        rows = _batch_shape(len(y), n_fits, n_features)
        data = (
            _chunks(np.asarray(X, dtype=float), rows, 0.0),
            _chunks(np.asarray(y, dtype=float), rows, 0.0),
            _chunks(np.asarray(offset, dtype=float), rows, 0.0),
            _chunks(np.asarray(folds, dtype=np.int32), rows, -1),
        )
        keep = jnp.asarray(~np.asarray(excluded, dtype=bool))
        lam = jnp.asarray(penalties, dtype=jnp.float64)
        beta, obj, done, n_iter = _newton_batch(
            data, keep, lam, tol, maxiter, max_halvings
        )
        out = {
            "params": np.asarray(beta),
            "objective": np.asarray(obj),
            "converged": np.asarray(done),
            "n_iter": np.asarray(n_iter),
        }
        if score_fold is not None:
            out["heldout_ll"] = np.asarray(
                _heldout_batch(data, beta, jnp.asarray(score_fold, dtype=jnp.int32))
            )
        if covariance:
            info = np.asarray(_information_at(data, keep, lam, beta))
            out["covariance"] = np.linalg.inv(info)
        return out


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
    the full-data fit. All ``n_folds**2 * len(sigma_grid)`` fits run as one
    batch (:func:`fit_ridge_poisson_batch`).

    Returns
    -------
    dict
        ``sigma`` (grid, strongest first), ``outer_ll`` (per fold, at the inner
        choice), ``chosen_sigma`` (per fold), ``inner_curve`` and
        ``outer_curve`` (folds x grid), ``n_fits``, ``n_nonconverged``,
        ``max_n_iter``.
    """
    grid = sorted(sigma_grid)
    fits = [
        (k, j, i)
        for k in range(n_folds)
        for j in range(n_folds)
        for i in range(len(grid))
    ]
    excluded = np.zeros((len(fits), n_folds), dtype=bool)
    penalties = np.zeros((len(fits), X.shape[1]))
    score = np.zeros(len(fits), dtype=np.int32)
    for b, (k, j, i) in enumerate(fits):
        excluded[b, [k, j]] = True  # j == k: the outer training set
        penalties[b] = np.where(penalized, 1.0 / grid[i] ** 2, 0.0)
        score[b] = j
    res = fit_ridge_poisson_batch(
        X, y, offset, folds, excluded, penalties, score_fold=score
    )
    inner = np.zeros((n_folds, len(grid)))
    curve = np.zeros((n_folds, len(grid)))
    for b, (k, j, i) in enumerate(fits):
        if j == k:
            curve[k, i] = res["heldout_ll"][b]
        else:
            inner[k, i] += res["heldout_ll"][b]
    best = inner.argmax(axis=1)
    return {
        "sigma": grid,
        "outer_ll": curve[np.arange(n_folds), best],
        "chosen_sigma": np.asarray(grid)[best],
        "inner_curve": inner,
        "outer_curve": curve,
        "n_fits": len(fits),
        "n_nonconverged": int((~res["converged"]).sum()),
        "max_n_iter": int(res["n_iter"].max()),
    }


def ridge_kernels(
    fit: RidgeFit,
    names: list[str],
    covariance: np.ndarray,
    spec: PerBinRateModelSpec,
) -> dict[str, dict[str, list[float]]]:
    """Per-lag log rate ratio and its posterior SD for each history block.

    ``covariance`` is the inverse penalized information at the mode (the
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
        res = fit_ridge_poisson_batch(
            X,
            y,
            offset,
            folds,
            np.zeros((1, n_folds), dtype=bool),
            lam[None],
            covariance=True,
        )
        full = RidgeFit(
            res["params"][0],
            bool(res["converged"][0]),
            int(res["n_iter"][0]),
            float(res["objective"][0]),
        )
        covariance = res["covariance"][0]
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
        del X
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
