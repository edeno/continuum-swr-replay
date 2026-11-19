"""Dirichlet-multinomial fitter for replay content over candidate categories.

The conditional-Poisson content model treats the ``L_e`` timebins of an event as
independent multinomial draws over the candidate arms. They are not: a replay is
a smooth trajectory, so its bins are near-copies of one another and the model
information overstates how much an event carries.

This module fits the same softmax mean structure with an explicit within-event
concentration ``phi_e`` that measures that redundancy directly:

.. code-block:: text

    p_e     = softmax_a(x_ea' beta)          # same conditioning as conditional Poisson
    log phi_e = gamma_0 + gamma' z_e         # z_e event-level modifiers, default (is_swr,)
    c_e     ~ DirichletMultinomial(L_e, phi_e * p_e)

    log P(c_e) = lgamma(L_e + 1) - sum_a lgamma(c_ea + 1)          # multinomial coefficient
               + lgamma(phi_e) - lgamma(L_e + phi_e)
               + sum_a [ lgamma(c_ea + phi_e p_ea) - lgamma(phi_e p_ea) ]

``phi -> inf`` recovers the count multinomial (bins independent); small ``phi``
says the bins within an event are near-copies, and the standard errors widen
accordingly. Because ``log phi`` carries its own design, ``coef_log_phi_is_swr``
is a direct test of whether SWR events are more or less concentrated than
non-SWR events, rather than an assumption baked into the likelihood.

The multinomial coefficient is retained so ``llf`` is a full log-likelihood
(``llf_includes_constants=True``). To compare AIC or BIC against a statsmodels
conditional fit -- whose conditional likelihood omits that constant -- add
:func:`multinomial_log_constant` to the statsmodels ``llf`` first.

Optimization is L-BFGS-B on a JAX objective with exact gradients, and standard
errors come from the JAX Hessian at the optimum. The jitted objective is keyed on
array shape, so a session with a candidate count or event count not yet seen in
the process pays a one-off compile (~1 s); repeated fits at the same shape do not.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln as jax_gammaln
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    freeze_common_cohort,
)
from continuum_swr_replay.analysis.shared_statistics import (
    akaike_information_criterion,
    bayesian_information_criterion,
    mcfadden_pseudo_r2,
)
from continuum_swr_replay.tables.columns import require_columns

from .._fit_diagnostics import NUISANCE_TERMS, separated_terms
from ..fitting import within_group_full_rank

# Float64 is required, not a nicety: the log-likelihood differences the optimizer
# chases are O(1e-6) against an objective of order 1e4, and the Hessian that
# supplies the standard errors is differenced from the same quantity. In float32
# the gradient is noise at the optimum and the SEs come back garbage. So every
# JAX call here (event_loglik, _optimize, _standard_errors) runs inside
# ``jax.enable_x64(True)``, which is scoped to the call and to the calling thread,
# rather than switching float64 on for the whole process at import. The jitted
# functions are keyed on that setting, so they are traced in float64.

logger = logging.getLogger(__name__)

ESTIMATOR = "dirichlet_multinomial"

#: Name of the concentration intercept coefficient (``gamma_0``).
LOG_PHI_INTERCEPT = "log_phi_intercept"

# The concentration intercept is the DM analog of ``Intercept``: it sets the
# baseline within-event redundancy and is never an aggregated scientific effect.
# It is also structurally unjudgeable on the effect scale (its design column is a
# constant, so its SD is 0) and it is NaN by construction when pinned via
# ``fixed_log_phi``. The concentration *modifiers* (``log_phi_is_swr``) are
# scientific terms and stay in the judgment.
_DM_NUISANCE_TERMS = frozenset(NUISANCE_TERMS | {LOG_PHI_INTERCEPT})

_MAXITER = 1000

# The forkserver's preload list is process-wide and only read when the server
# first starts, so it is set once here rather than on every pool.
_forkserver_preloaded = False


# ── likelihood ──────────────────────────────────────────────────────────────


def _event_loglik_jax(
    theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray
) -> jnp.ndarray:
    """Per-event Dirichlet-multinomial log-likelihood (JAX, differentiable).

    Parameters
    ----------
    theta : jnp.ndarray, shape (n_predictors + 1 + n_concentration,)
        ``[beta, gamma_0, gamma_z]``.
    X : jnp.ndarray, shape (n_events, n_categories, n_predictors)
    Y : jnp.ndarray, shape (n_events, n_categories)
    Z : jnp.ndarray, shape (n_events, n_concentration)

    Returns
    -------
    jnp.ndarray, shape (n_events,)
    """
    n_beta = X.shape[2]
    beta, gamma = theta[:n_beta], theta[n_beta:]
    total = Y.sum(axis=1)
    const = jax_gammaln(total + 1.0) - jax_gammaln(Y + 1.0).sum(axis=1)
    log_p = jax.nn.log_softmax(jnp.einsum("eak,k->ea", X, beta), axis=1)
    phi = jnp.exp(gamma[0] + Z @ gamma[1:])
    # A candidate the mean structure has all but ruled out underflows
    # ``exp(log_p)`` to exactly 0, and ``lgamma(0)`` is ``+inf``: the event's
    # log-likelihood becomes NaN and the optimizer walks off the objective it
    # was following. Clipping at the smallest positive double keeps the
    # likelihood finite there and changes nothing anywhere else.
    alpha = jnp.clip(phi[:, None] * jnp.exp(log_p), min=jnp.finfo(jnp.float64).tiny)
    return (
        const
        + jax_gammaln(phi)
        - jax_gammaln(total + phi)
        + (jax_gammaln(Y + alpha) - jax_gammaln(alpha)).sum(axis=1)
    )


def _negloglike(
    theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray
) -> jnp.ndarray:
    return -_event_loglik_jax(theta, X, Y, Z).sum()


_event_loglik_jit = jax.jit(_event_loglik_jax)
_value_and_grad_jit = jax.jit(jax.value_and_grad(_negloglike))
_hessian_jit = jax.jit(jax.hessian(_negloglike))


def event_loglik(
    theta: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """Per-event log P(c_e) under the Dirichlet-multinomial model.

    Public because held-out scoring needs the per-event decomposition that the
    fitter's scalar objective sums away.

    Parameters
    ----------
    theta : array-like, shape (n_predictors + 1 + n_concentration,)
        Parameters in fitted order: ``beta`` for each predictor, then the
        concentration intercept, then one coefficient per concentration column.
    X : array-like, shape (n_events, n_categories, n_predictors)
        Packed category-varying design.
    Y : array-like, shape (n_events, n_categories)
        Packed response counts.
    Z : array-like, shape (n_events, n_concentration)
        Packed event-level concentration design (may have zero columns).

    Returns
    -------
    np.ndarray, shape (n_events,)
        Log-likelihood contribution of each event, including the multinomial
        coefficient.
    """
    theta = np.asarray(theta, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    expected = X.shape[2] + 1 + Z.shape[1]
    if theta.shape != (expected,):
        raise ValueError(
            f"theta has shape {theta.shape}; expected ({expected},) for "
            f"{X.shape[2]} predictor(s) and {Z.shape[1]} concentration column(s)"
        )
    if Y.shape != X.shape[:2] or Z.shape[0] != X.shape[0]:
        raise ValueError(
            f"shape mismatch: X {X.shape}, Y {Y.shape}, Z {Z.shape} must share "
            "the event axis (and X/Y the category axis)"
        )
    with jax.enable_x64(True):
        return np.asarray(
            _event_loglik_jit(
                jnp.asarray(theta), jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)
            )
        )


def multinomial_log_constant(y, groups) -> float:
    """Multinomial coefficient summed over events, ``sum_e log(L_e! / prod_a c_ea!)``.

    The statsmodels conditional likelihood drops this constant; the DM likelihood
    keeps it. Add this to a conditional fit's ``llf`` before comparing it (or its
    AIC/BIC) against a DM fit on the same response.

    Parameters
    ----------
    y : array-like, shape (n_rows,)
        Per-(event, category) counts.
    groups : array-like, shape (n_rows,)
        Event id per row; rows need not be sorted.

    Returns
    -------
    float
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if y.shape != groups.shape:
        raise ValueError(f"y shape {y.shape} != groups shape {groups.shape}")
    totals = pd.Series(y).groupby(groups).sum().to_numpy()
    return float(gammaln(totals + 1.0).sum() - gammaln(y + 1.0).sum())


# ── packing and validation ──────────────────────────────────────────────────


@dataclass(frozen=True)
class _Packed:
    """Event-major arrays plus the row bookkeeping the result dict reports."""

    ids: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    n_drop_zero_response: int
    n_drop_nan: int


def _pack_events(
    long_df: pd.DataFrame,
    predictors: list[str],
    response_col: str,
    event_col: str,
    concentration_cols: tuple[str, ...],
    n_categories: int,
) -> _Packed:
    """Reshape a long frame to (n_events, n_categories, ...) arrays, validating it.

    Raises rather than repairing: a response that is not a count, a ragged
    candidate set, an event-level predictor, or a concentration column that
    varies within an event all mean the caller built the wrong table, and a
    silently-repaired fit would be read as a real estimate.
    """
    concentration_cols = tuple(concentration_cols)
    required = list(
        dict.fromkeys([event_col, response_col, *predictors, *concentration_cols])
    )
    require_columns(long_df, required, "long_df")

    # Event-wise complete cases, as in ``fit_conditional_poisson``: dropping rows
    # would leave the event with a smaller candidate set than the model assumes.
    # ``n_drop_nan`` counts the events with a missing or non-finite value.
    df, selection = freeze_common_cohort(
        long_df[required], unit_col=event_col, required_cols=required
    )
    n_drop_nan = selection["n_missing_units"] + selection["n_nonfinite_units"]
    if n_drop_nan:
        logger.info(
            "fit_dirichlet_multinomial: dropping %d event(s) with a missing or "
            "non-finite required value",
            n_drop_nan,
        )

    df = df.sort_values([event_col], kind="stable").reset_index(drop=True)
    totals = df.groupby(event_col, sort=True)[response_col].transform("sum")

    zero_events = df.loc[totals == 0, event_col].unique()
    n_drop_zero_response = int(len(zero_events))
    if n_drop_zero_response:
        df = df[~df[event_col].isin(zero_events)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"no events left after dropping incomplete and zero-total events "
            f"({n_drop_nan} missing or non-finite, {n_drop_zero_response} zero-total)"
        )

    # Order matters: a fraction response also fails the integer check, and the
    # fraction message is the one that names the caller's actual mistake.
    event_totals = df.groupby(event_col, sort=True)[response_col].sum()
    if (event_totals <= 1.0).all():
        raise ValueError(
            f"{response_col!r} looks like a per-event fraction (every event total "
            "is <= 1); the Dirichlet-multinomial response must be timebin counts "
            "(e.g. arm_count_hard)"
        )

    y = df[response_col].to_numpy(dtype=float)
    if (y < 0).any():
        raise ValueError(f"{response_col!r} has negative values")
    if not np.allclose(y, np.round(y)):
        raise ValueError(
            f"{response_col!r} is not integer-valued; the Dirichlet-multinomial "
            "likelihood is defined on counts"
        )

    ids, counts = np.unique(df[event_col].to_numpy(), return_counts=True)
    n_cat = int(counts[0])
    if not (counts == n_cat).all():
        raise ValueError(
            "every event must carry the same number of candidate rows; found "
            f"{sorted(set(counts.tolist()))}"
        )
    if n_cat != n_categories:
        raise ValueError(
            f"packed candidate count {n_cat} != n_categories={n_categories}"
        )

    groups = df[event_col].to_numpy()
    X_rows = (
        df[predictors].to_numpy(dtype=float) if predictors else np.zeros((len(df), 0))
    )
    if predictors and not within_group_full_rank(X_rows, groups):
        raise ValueError(
            "within-event design is rank deficient: at least one predictor is "
            f"constant within every event (or collinear) in {predictors}"
        )

    if concentration_cols:
        by_event = df.groupby(event_col, sort=True)[list(concentration_cols)]
        n_unique = by_event.nunique()
        varying = [c for c in concentration_cols if (n_unique[c] > 1).any()]
        if varying:
            raise ValueError(
                f"concentration column(s) {varying} vary within an event; log phi "
                "is an event-level quantity"
            )
        Z = by_event.first().to_numpy(dtype=float)
    else:
        Z = np.zeros((len(ids), 0))

    return _Packed(
        ids=ids,
        X=X_rows.reshape(len(ids), n_cat, len(predictors)),
        Y=y.reshape(len(ids), n_cat),
        Z=Z,
        n_drop_zero_response=n_drop_zero_response,
        n_drop_nan=n_drop_nan,
    )


def _swr_event_counts(
    Z: np.ndarray, concentration_cols: tuple[str, ...]
) -> dict[str, float]:
    """Modeled events on each side of the ``is_swr`` concentration modifier.

    The ``log_phi_is_swr`` contrast is only as trustworthy as the smaller of the
    two event groups behind it, and neither ``n_events`` nor the effect-scale
    identifiability rule can see that split: a session with two SWR events can
    return a perfectly finite modifier that no downstream filter would question.
    Recording both counts on the row lets a reader (or an audit) find those
    sessions.

    Parameters
    ----------
    Z : np.ndarray, shape (n_events, n_concentration)
        Packed event-level concentration design.
    concentration_cols : tuple of str
        Column names, in ``Z`` order.

    Returns
    -------
    dict
        ``n_events_swr`` / ``n_events_nonswr``, both NaN when the concentration
        design carries no ``is_swr`` column (a within-SWR cohort, where the flag
        is constant and the modifier is not estimated at all).
    """
    if "is_swr" not in concentration_cols:
        return {"n_events_swr": np.nan, "n_events_nonswr": np.nan}
    column = Z[:, concentration_cols.index("is_swr")]
    n_swr = int(np.count_nonzero(column))
    return {
        "n_events_swr": float(n_swr),
        "n_events_nonswr": float(len(column) - n_swr),
    }


# ── optimization ────────────────────────────────────────────────────────────


def _optimize(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    fixed_log_phi: float | None,
    theta0: np.ndarray | None = None,
):
    """Maximize the DM likelihood; returns the scipy result.

    ``fixed_log_phi`` is imposed as an equality bound on the concentration
    intercept rather than by reparameterizing, so the objective, its gradient and
    its Hessian keep one shape (and one JIT compilation) across pinned and free
    fits.
    """
    n_beta = X.shape[2]
    n_theta = n_beta + 1 + Z.shape[1]
    if theta0 is None:
        theta0 = np.zeros(n_theta)
    theta0 = np.asarray(theta0, dtype=float).copy()
    bounds = None
    if fixed_log_phi is not None:
        theta0[n_beta] = float(fixed_log_phi)
        bounds = [(None, None)] * n_theta
        bounds[n_beta] = (float(fixed_log_phi), float(fixed_log_phi))

    with jax.enable_x64(True):
        Xj, Yj, Zj = jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)

        def fun(theta):
            value, grad = _value_and_grad_jit(jnp.asarray(theta), Xj, Yj, Zj)
            return float(value), np.asarray(grad, dtype=float)

        return minimize(
            fun,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": _MAXITER},
        )


def _standard_errors(
    theta: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    pinned_index: int | None,
) -> np.ndarray:
    """Hessian-based SEs; the pinned concentration intercept gets NaN.

    A pinned parameter is not estimated, so it has no sampling variance: its row
    and column are removed before inversion (leaving them in would let the pinned
    direction leak curvature into the free parameters' covariance).

    A flat direction of the observed information (a parameter the data cannot
    pin down at all -- ``phi`` on data that is exactly multinomial, say) gets
    ``inf``, not the ``0`` a pseudo-inverse would return for it. Zero would read
    as an infinitely precise estimate and would pass the separation check; ``inf``
    is what an unidentified parameter's standard error means, and it is what
    :func:`~._fit_diagnostics.separated_terms` flags.
    """
    with jax.enable_x64(True):
        hess = np.asarray(
            _hessian_jit(
                jnp.asarray(theta), jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)
            ),
            dtype=float,
        )
    se = np.full(theta.shape[0], np.nan)
    keep = np.arange(theta.shape[0])
    if pinned_index is not None:
        keep = np.delete(keep, pinned_index)
    sub = hess[np.ix_(keep, keep)]
    if not np.isfinite(sub).all():
        return se

    # Symmetric eigendecomposition rather than a bare pinv, so the flat
    # directions can be told apart from the well-determined ones.
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (sub + sub.T))
    tol = max(abs(eigenvalues).max(), 1.0) * sub.shape[0] * np.finfo(float).eps
    flat = eigenvalues <= tol
    inverse = np.where(flat, 0.0, 1.0 / np.where(flat, 1.0, eigenvalues))
    variance = (eigenvectors**2) @ inverse
    # Weight of each parameter on the flat subspace; non-negligible weight means
    # the parameter itself is not identified.
    flat_weight = (eigenvectors**2) @ flat.astype(float)
    se[keep] = np.where(
        flat_weight > 1e-8, np.inf, np.sqrt(np.clip(variance, 0.0, None))
    )
    return se


def _identification(
    theta: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    predictors: Sequence[str],
    concentration_cols: tuple[str, ...],
    pinned_index: int | None,
) -> tuple[np.ndarray, list[str], np.ndarray, tuple[str, ...]]:
    """A fit's standard errors and the terms it does not identify.

    Parameters
    ----------
    theta : np.ndarray, shape (n_beta + 1 + n_concentration,)
        Fitted ``beta``, the concentration intercept, then its modifiers.
    X : np.ndarray, shape (n_events, n_categories, n_beta)
    Y : np.ndarray, shape (n_events, n_categories)
    Z : np.ndarray, shape (n_events, n_concentration)
        The packed design, counts and concentration columns the fit used.
    predictors : sequence of str
        Names of the ``beta`` terms, in ``X``'s last-axis order.
    concentration_cols : tuple of str
        Names of the concentration modifiers, in ``Z``'s column order.
    pinned_index : int, optional
        Index of a pinned concentration intercept (see :func:`_standard_errors`).

    Returns
    -------
    se : np.ndarray, shape (n_theta,)
        Standard errors (:func:`_standard_errors`).
    names : list of str, length n_theta
        Term names in ``theta`` order.
    term_sd : np.ndarray, shape (n_theta,)
        Each term's design-column SD: the predictors', 0 for the concentration
        intercept, the modifiers'.
    separated : tuple of str
        The non-nuisance terms not identified, by the effect-scale rule of
        :func:`~._fit_diagnostics.separated_terms`.
    """
    n_beta = len(predictors)
    se = _standard_errors(theta, X, Y, Z, pinned_index=pinned_index)
    names = [
        *predictors,
        LOG_PHI_INTERCEPT,
        *[f"log_phi_{c}" for c in concentration_cols],
    ]
    predictor_sd = X.reshape(-1, n_beta).std(axis=0) if n_beta else np.zeros(0)
    # Effect-scale SDs for the concentration terms too: the intercept's design
    # column is a constant (SD 0, so it falls back to the absolute-coefficient
    # rule and is a nuisance term anyway), the modifiers' are their own columns.
    term_sd = np.concatenate([predictor_sd, np.zeros(1), Z.std(axis=0)])
    separated = separated_terms(
        theta,
        se,
        predictor_sd=term_sd,
        term_names=names,
        nuisance_terms=_DM_NUISANCE_TERMS,
    )
    return se, names, term_sd, separated


def _null_loglik(packed: _Packed, *, fixed_log_phi: float | None) -> float:
    """Log-likelihood of the DM model with ``beta = 0`` and the concentration free.

    The uniform-allocation analytic null used by the conditional-Poisson ladder is
    not available here: with a free ``phi`` the null is still a one- or
    two-parameter fit, so it is fit rather than assumed.
    """
    X_null = packed.X[:, :, :0]
    res = _optimize(X_null, packed.Y, packed.Z, fixed_log_phi=fixed_log_phi)
    return -float(res.fun)


# ── public fitters ──────────────────────────────────────────────────────────


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
    """Fit ``c_e ~ DM(L_e, exp(log_phi_e) * softmax(X_e beta))`` for one session.

    The mean structure is the conditional-Poisson softmax; what is added is an
    explicit within-event concentration whose log is linear in
    ``concentration_cols``. The resulting standard errors are event-level by
    construction, so no clustered sandwich is reported (and no
    ``bse_cluster_<term>`` key is emitted).

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data, one row per (event, candidate category), with
        ``n_categories`` rows per event.
    predictors : list[str]
        Category-varying predictor columns. No intercept: it is absorbed by the
        softmax, exactly as the conditional model conditions it out.
    response_col : str, optional
        Per-category integer timebin counts. A per-event fraction is rejected.
    event_col : str, optional
        Column with globally unique event ids.
    concentration_cols : tuple[str, ...], optional
        Event-level columns entering ``log phi_e``. Pass ``()`` for a single
        constant concentration.
    n_categories : int, optional
        Candidate categories per event (9 arms, 6 leaves, 3 patches). Checked
        against the packed shape.
    fixed_log_phi : float, optional
        Pin the concentration intercept at this value instead of estimating it
        (the concentration modifiers are still estimated). ``n_params`` drops by
        one and ``bse_log_phi_intercept`` is NaN.

    Returns
    -------
    dict | None
        The content-estimator result schema: ``coef_<term>``, ``bse_<term>``,
        ``pval_<term>`` for every beta and concentration term, ``sd_<term>`` for
        every beta term and ``sd_log_phi_<col>`` for every concentration
        modifier (NaN for the intercept, whose design column is constant), the
        events on each side of the ``is_swr`` modifier (``n_events_swr`` /
        ``n_events_nonswr``, NaN when it is absent), fit statistics (``llf``,
        ``llf_null``, ``aic``, ``bic``, ``pseudo_r2``), and diagnostics
        (``converged``, ``separated``, ``separated_terms``,
        ``n_convergence_warnings``). ``pseudo_r2`` is formed from two FULL
        log-likelihoods -- the multinomial constant is in both -- so it is NOT
        on the scale of a conditional fit's, and its null refits the
        concentration. ``None`` when the
        session carries too few events to identify the parameters, or when the
        optimizer returns a non-finite objective.

    Raises
    ------
    ValueError
        When the response is not integer counts, looks like a per-event fraction,
        or is negative; when events carry unequal candidate counts or a count
        that disagrees with ``n_categories``; when a predictor is constant within
        every event; or when a concentration column varies within an event.
    """
    predictors = list(predictors)
    concentration_cols = tuple(concentration_cols)
    packed = _pack_events(
        long_df, predictors, response_col, event_col, concentration_cols, n_categories
    )

    n_beta = len(predictors)
    n_events, n_cat = packed.Y.shape
    n_free = n_beta + (0 if fixed_log_phi is not None else 1) + packed.Z.shape[1]
    if n_events < 2 * n_free:
        logger.info(
            "fit_dirichlet_multinomial: %d events < 2*%d free parameters; not "
            "fitting %s",
            n_events,
            n_free,
            predictors,
        )
        return None

    res = _optimize(packed.X, packed.Y, packed.Z, fixed_log_phi=fixed_log_phi)
    llf = -float(res.fun)
    if not np.isfinite(llf) or not np.isfinite(res.x).all():
        logger.warning(
            "fit_dirichlet_multinomial: non-finite optimum for %s", predictors
        )
        return None

    se, names, term_sd, separated = _identification(
        res.x,
        packed.X,
        packed.Y,
        packed.Z,
        predictors=predictors,
        concentration_cols=concentration_cols,
        pinned_index=None if fixed_log_phi is None else n_beta,
    )
    predictor_sd = term_sd[:n_beta]

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(se > 0, res.x / se, np.nan)
    pvals = 2.0 * norm.sf(np.abs(z))

    llf_null = _null_loglik(packed, fixed_log_phi=fixed_log_phi)
    n_params = n_free
    # McFadden's ratio of two FULL log-likelihoods: both ``llf`` and ``llf_null``
    # carry the multinomial coefficient, which a statsmodels conditional fit's
    # likelihood omits. The constant is a large negative number that enters both
    # numerator and denominator, so this ratio is compressed toward zero
    # relative to a conditional fit's -- the two are not on one scale and must
    # not be tabulated against each other. Its null also differs: the DM null
    # refits the concentration, so the baseline is uniform-allocation-at-the-
    # fitted-redundancy rather than uniform-allocation-at-independent-bins.
    pseudo_r2 = mcfadden_pseudo_r2(llf, llf_null)

    return {
        **{f"coef_{k}": float(v) for k, v in zip(names, res.x, strict=True)},
        **{f"bse_{k}": float(v) for k, v in zip(names, se, strict=True)},
        **{f"pval_{k}": float(v) for k, v in zip(names, pvals, strict=True)},
        **{f"sd_{k}": float(v) for k, v in zip(predictors, predictor_sd, strict=True)},
        # Effect-scale SD for the concentration terms as well, so the shipped
        # aggregation judges ``log_phi_<col>`` on ``|coef| * sd`` -- the log-phi
        # swing the modifier actually produces -- instead of falling back to the
        # raw ``|coef| >= 10`` rule, which is not a scale any concentration term
        # is on. The intercept's design column is a constant with no spread to
        # measure, so its SD is NaN and it stays flag-only.
        f"sd_{LOG_PHI_INTERCEPT}": float("nan"),
        **{
            f"sd_log_phi_{c}": float(v)
            for c, v in zip(concentration_cols, packed.Z.std(axis=0), strict=True)
        },
        # Events on each side of the is_swr modifier (NaN where it is absent).
        **_swr_event_counts(packed.Z, concentration_cols),
        "estimator": ESTIMATOR,
        # The multinomial coefficient is in ``llf``; a statsmodels conditional
        # fit's is not (see ``multinomial_log_constant``).
        "llf_includes_constants": True,
        "n_events": int(n_events),
        "n_rows": int(n_events * n_cat),
        "n_categories": int(n_cat),
        "n_params": int(n_params),
        "llf": llf,
        "llf_null": float(llf_null),
        "aic": float(akaike_information_criterion(llf, n_params)),
        # BIC with n = n_events: the DM likelihood's independent units are events,
        # not timebins (which is the whole point of the concentration parameter).
        "bic": float(bayesian_information_criterion(llf, n_params, n_events)),
        "pseudo_r2": float(pseudo_r2),
        "converged": bool(res.success),
        "separated": len(separated) > 0,
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(separated),
        "n_convergence_warnings": int(not res.success),
        "n_drop_zero_response": packed.n_drop_zero_response,
        "n_drop_nan_events": packed.n_drop_nan,
    }


def cross_validate_dirichlet_multinomial(
    long_df: pd.DataFrame,
    predictors: list[str],
    *,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    concentration_cols: tuple[str, ...] = ("is_swr",),
    n_categories: int = 9,
    n_folds: int = 5,
    seed: int = 0,
) -> dict | None:
    """Held-out per-event log-likelihood of the DM model, by event-wise K-fold CV.

    Events are the independent unit, so the split is over events and the score is
    the held-out ``event_loglik`` -- the same quantity the fit maximizes, on data
    the fit did not see. The baseline is the DM null (``beta = 0``, concentration
    refit on the same training fold), so the reported bits per event are the
    predictive gain attributable to the mean structure alone and not to the
    concentration.

    Parameters
    ----------
    long_df, predictors, response_col, event_col, concentration_cols, n_categories
        As in :func:`fit_dirichlet_multinomial`.
    n_folds : int, optional
        Number of folds.
    seed : int, optional
        Seed for the shuffled fold assignment.

    Returns
    -------
    dict | None
        ``cv_ll_per_event_mean``, ``cv_ll_per_event_null_mean``,
        ``cv_bits_per_event_vs_null``, ``n_folds``, ``n_events_cv``,
        ``cv_n_nonconverged_folds`` and ``cv_n_separated_folds``. A fold whose
        full or null fit did not converge, or leaves a term unidentified, makes
        the three metrics NaN. ``None`` when there are too few events to split.
    """
    from sklearn.model_selection import GroupKFold

    predictors = list(predictors)
    concentration_cols = tuple(concentration_cols)
    packed = _pack_events(
        long_df, predictors, response_col, event_col, concentration_cols, n_categories
    )
    n_events = len(packed.ids)
    n_free = len(predictors) + 1 + packed.Z.shape[1]
    if n_events < max(n_folds * 2, 2 * n_free):
        return None

    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    event_index = np.arange(n_events)

    ll_full: list[float] = []
    ll_null: list[float] = []
    n_nonconverged = 0
    n_separated = 0
    for train, test in splitter.split(event_index, groups=packed.ids):
        res = _optimize(
            packed.X[train], packed.Y[train], packed.Z[train], fixed_log_phi=None
        )
        res_null = _optimize(
            packed.X[train][:, :, :0],
            packed.Y[train],
            packed.Z[train],
            fixed_log_phi=None,
        )
        n_nonconverged += int(not (res.success and res_null.success))
        *_, separated_full = _identification(
            res.x,
            packed.X[train],
            packed.Y[train],
            packed.Z[train],
            predictors=predictors,
            concentration_cols=concentration_cols,
            pinned_index=None,
        )
        *_, separated_null = _identification(
            res_null.x,
            packed.X[train][:, :, :0],
            packed.Y[train],
            packed.Z[train],
            predictors=[],
            concentration_cols=concentration_cols,
            pinned_index=None,
        )
        n_separated += int(bool(separated_full or separated_null))
        ll_full.extend(
            event_loglik(res.x, packed.X[test], packed.Y[test], packed.Z[test]).tolist()
        )
        ll_null.extend(
            event_loglik(
                res_null.x,
                packed.X[test][:, :, :0],
                packed.Y[test],
                packed.Z[test],
            ).tolist()
        )

    # Any nonconverged/separated fold invalidates the metric (NaN) rather than
    # dropping the fold, as in the conditional-Poisson and rate CVs; the counts
    # are reported alongside.
    invalid = (n_nonconverged > 0) or (n_separated > 0)
    mean_full = np.nan if invalid else float(np.mean(ll_full))
    mean_null = np.nan if invalid else float(np.mean(ll_null))
    return {
        "cv_ll_per_event_mean": mean_full,
        "cv_ll_per_event_null_mean": mean_null,
        "cv_bits_per_event_vs_null": (mean_full - mean_null) / np.log(2.0),
        "n_folds": int(n_folds),
        "n_events_cv": int(len(ll_full)),
        "cv_n_nonconverged_folds": int(n_nonconverged),
        "cv_n_separated_folds": int(n_separated),
    }


# ── per-session driving, with a pooled concentration for small sessions ─────


#: Default minimum modeled events for a session to estimate its own ``log phi``.
DEFAULT_MIN_EVENTS_SESSION_PHI = 40

#: Key under which :func:`animal_pooled_log_phi` stores the all-animal median.
POOLED_ALL_ANIMALS_KEY = "__all__"

#: Worker processes a runner spreads its per-session fits over by default.
DEFAULT_N_JOBS = 8

#: Skip classes that are the ESTIMATOR refusing a session rather than the cohort
#: rules excluding it. A refusal is rare and actionable, so a runner's cohort
#: audit names the session it happened to; ``fit_none`` (too few events for the
#: free fit's parameters) is routine and is summarized by class alone. The
#: classes are the part of a skip reason before the first ``":"``, so
#: ``dm_design:<message>`` matches ``"dm_design"``.
ESTIMATOR_SKIP_CLASSES = frozenset({"dm_design", "pinned_fit_none"})


def concentration_cols_for_cohort(frame: pd.DataFrame) -> tuple[str, ...]:
    """Event-level columns entering ``log phi`` for one cohort of rows.

    ``is_swr`` is the concentration modifier this estimator exists to measure --
    whether an SWR event's mass is spread over more arms than a non-SWR
    event's -- but only where the rows being fitted carry both kinds. Within an
    SWR subset (or an SWR / nonSWR population split) the flag is constant, its
    coefficient is not identified, and asking for it would return an infinite
    standard error rather than a concentration difference that is simply not in
    question there.

    Both content runners resolve the concentration design through this one
    function, so a subset fit cannot ask for a term the pooled fit's cohort
    defines and its own does not.

    Parameters
    ----------
    frame : pd.DataFrame
        The rows about to be fitted.

    Returns
    -------
    tuple of str
        ``("is_swr",)`` when the cohort carries both values, else ``()``.
    """
    if "is_swr" not in frame.columns:
        return ()
    values = set(frame["is_swr"].dropna().astype(bool).unique())
    return ("is_swr",) if values == {False, True} else ()


def animal_pooled_log_phi(
    rows: pd.DataFrame, *, min_events: int = DEFAULT_MIN_EVENTS_SESSION_PHI
) -> dict[str, float]:
    """Median ``coef_log_phi_intercept`` per animal, over its larger sessions.

    The concentration intercept is the one parameter a short session cannot
    pin down: with a handful of events the likelihood is nearly flat in it, and
    the optimizer walks it to whatever value makes the few observed allocations
    look certain -- which then propagates into that session's coefficient
    standard errors. Sessions with at least ``min_events`` modeled events do
    estimate it, and their median within an animal is what a short session of
    the same animal is pinned to.

    Parameters
    ----------
    rows : pd.DataFrame
        First-pass fits, one row per session, carrying ``animal``,
        ``n_events`` and ``coef_log_phi_intercept``.
    min_events : int, optional
        Minimum modeled events for a session to contribute.

    Returns
    -------
    dict of str to float
        Animal -> median intercept, plus :data:`POOLED_ALL_ANIMALS_KEY` ->
        median over every qualifying session of every animal. Empty when no
        session qualifies.

    Raises
    ------
    ValueError
        If a required column is absent: a pooled value read off an incomplete
        frame would silently pin every short session to a number derived from
        the wrong sessions.
    """
    require_columns(
        rows, ["animal", "n_events", "coef_log_phi_intercept"], "the first-pass fits"
    )
    if rows.empty:
        return {}
    qualifying = rows.loc[
        (pd.to_numeric(rows["n_events"], errors="coerce") >= min_events)
        & np.isfinite(pd.to_numeric(rows["coef_log_phi_intercept"], errors="coerce"))
    ]
    if qualifying.empty:
        return {}
    pooled = {
        str(animal): float(value)
        for animal, value in qualifying.groupby("animal")["coef_log_phi_intercept"]
        .median()
        .items()
    }
    pooled[POOLED_ALL_ANIMALS_KEY] = float(
        qualifying["coef_log_phi_intercept"].median()
    )
    return pooled


def resolve_pooled_log_phi(
    animal: object, pooled_log_phi: dict[str, float]
) -> tuple[float | None, str]:
    """The concentration intercept a short session of ``animal`` is pinned to.

    Returns ``(value, phi_source)``: the animal's own pooled median where it has
    one (``"animal_pooled"``), else the all-animal median
    (``"pooled_all_animals"``), else ``(None, "session")`` when nothing
    qualified and the session must stand on its own fit.
    """
    key = str(animal)
    if key in pooled_log_phi:
        return float(pooled_log_phi[key]), "animal_pooled"
    if POOLED_ALL_ANIMALS_KEY in pooled_log_phi:
        return float(pooled_log_phi[POOLED_ALL_ANIMALS_KEY]), "pooled_all_animals"
    return None, "session"


def _fit_one_session(task: tuple) -> tuple:
    """Worker entry point: fit one session, converting a bad design to a skip.

    Top-level (not a closure) because the process pool pickles it by reference.
    """
    key, frame, predictors, kwargs = task
    try:
        result = fit_dirichlet_multinomial(frame, list(predictors), **kwargs)
    except ValueError as exc:
        return key, None, f"dm_design:{exc}"
    if result is None:
        return key, None, "fit_none"
    return key, result, ""


def _run_session_fits(tasks: list[tuple], n_jobs: int) -> list[tuple]:
    """Map :func:`_fit_one_session` over ``tasks``, in input order.

    Each fit is run in a fresh subprocess. The JAX objective is jitted per
    ``(n_events, n_categories, n_predictors)`` shape and every session is a new
    shape, so a long in-process loop accumulates one compiled executable per
    session and grows without bound (measured at +729 MB over 30 shapes);
    ``max_tasks_per_child=1`` returns that memory after every fit. The
    forkserver context is deliberate: its server is fork+exec'd, so the workers
    never inherit a JAX runtime that the parent has already started, and each
    worker is then a cheap fork of a server that already carries this module.
    """
    global _forkserver_preloaded

    if not tasks:
        return []
    context = multiprocessing.get_context("forkserver")
    if not _forkserver_preloaded:
        # Preloaded so each worker is a cheap fork of a server that already
        # carries this module; without it every worker re-imports jax and
        # statsmodels (~5 s) before it can start fitting. This replaces the
        # default ``["__main__"]``: the worker needs this module and nothing
        # from the runner script, and importing a runner as ``__main__`` in the
        # server would re-execute its module body for no benefit.
        context.set_forkserver_preload([__name__])
        _forkserver_preloaded = True
    with ProcessPoolExecutor(
        max_workers=max(1, int(n_jobs)),
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        return list(pool.map(_fit_one_session, tasks))


def fit_sessions_dirichlet_multinomial(
    sessions: Sequence[tuple[Hashable, object, pd.DataFrame]],
    predictors: list[str],
    *,
    min_events: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
    n_jobs: int = 1,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    concentration_cols: tuple[str, ...] = ("is_swr",),
    n_categories: int = 9,
) -> tuple[list[tuple[Hashable, dict]], list[tuple[Hashable, str]]]:
    """Fit every session, pinning the short ones to a pooled concentration.

    Two passes. The first fits every session with a free concentration
    intercept; :func:`animal_pooled_log_phi` then summarizes the sessions that
    were long enough to estimate it, and the second pass refits the short ones
    (and retries the ones the free fit had too few events for) with that value
    pinned. Every returned row carries ``phi_source`` in ``{"session",
    "animal_pooled", "pooled_all_animals"}``.

    Parameters
    ----------
    sessions : sequence of (key, animal, frame)
        One entry per session: a sortable session key, the animal the pooled
        fallback is taken from, and that session's long-format rows.
    predictors : list of str
        Category-varying predictors for this rung.
    min_events : int, optional
        Sessions with fewer modeled events than this are pinned.
    n_jobs : int, optional
        Worker processes for the per-session fits.
    response_col, event_col, concentration_cols, n_categories
        As in :func:`fit_dirichlet_multinomial`.

    Returns
    -------
    list of (key, dict)
        The fitted rows, in sorted key order.
    list of (key, str)
        Sessions that produced no fit, in sorted key order, each with the
        reason: ``"fit_none"`` (too few events for the free fit's parameters, or
        a non-finite optimum), ``"dm_design:<message>"`` (a within-event design
        the fitter refused), or ``"pinned_fit_none"`` (the pinned refit of a
        short session failed where the free fit had succeeded).

    Notes
    -----
    The fitted cohort therefore depends on ``min_events``: a session just under
    the threshold is refit with its intercept pinned, and if that refit fails it
    is dropped even though its free fit had converged. Changing
    ``min_events`` can change which sessions a rung reports.
    """
    fit_kwargs = {
        "response_col": response_col,
        "event_col": event_col,
        "concentration_cols": tuple(concentration_cols),
        "n_categories": n_categories,
    }
    keep = list(
        dict.fromkeys([event_col, response_col, *predictors, *concentration_cols])
    )
    ordered = sorted(sessions, key=lambda item: item[0])
    animals = {key: animal for key, animal, _frame in ordered}
    frames = {
        key: frame[[column for column in keep if column in frame.columns]]
        for key, _animal, frame in ordered
    }

    first = _run_session_fits(
        [
            (key, frames[key], predictors, fit_kwargs)
            for key, _animal, _frame in ordered
        ],
        n_jobs,
    )

    fitted: dict[Hashable, dict] = {}
    skipped: dict[Hashable, str] = {}
    for key, result, reason in first:
        if result is None:
            skipped[key] = reason
        else:
            result["phi_source"] = "session"
            result["fixed_log_phi"] = np.nan
            fitted[key] = result

    pooled = animal_pooled_log_phi(
        pd.DataFrame(
            [
                {
                    "animal": animals[key],
                    "n_events": result["n_events"],
                    "coef_log_phi_intercept": result.get(
                        f"coef_{LOG_PHI_INTERCEPT}", np.nan
                    ),
                }
                for key, result in fitted.items()
            ],
            columns=["animal", "n_events", "coef_log_phi_intercept"],
        ),
        min_events=min_events,
    )

    # Short sessions, and the ones the free fit had too few events for: pinning
    # the intercept removes a free parameter, so a session the free fit refused
    # can still be fitted here.
    retry = [
        key
        for key, _animal, _frame in ordered
        if (key in fitted and int(fitted[key]["n_events"]) < min_events)
        or skipped.get(key) == "fit_none"
    ]
    if retry and not pooled:
        logger.warning(
            "no session reached %d events, so %d short session(s) keep their own "
            "concentration intercept",
            min_events,
            len(retry),
        )
    if retry and pooled:
        pinned = {key: resolve_pooled_log_phi(animals[key], pooled) for key in retry}
        second = _run_session_fits(
            [
                (key, frames[key], predictors, {**fit_kwargs, "fixed_log_phi": value})
                for key, (value, _source) in pinned.items()
            ],
            n_jobs,
        )
        for key, result, reason in second:
            value, source = pinned[key]
            if result is None:
                # Distinguished from a first-pass failure: this session HAD a
                # free fit and lost it to the pinning, which is a consequence of
                # the threshold rather than of the session's own data.
                had_free_fit = fitted.pop(key, None) is not None
                skipped[key] = (
                    "pinned_fit_none"
                    if had_free_fit and reason == "fit_none"
                    else reason
                )
                continue
            result["phi_source"] = source
            result["fixed_log_phi"] = float(value)
            fitted[key] = result
            skipped.pop(key, None)

    for key in fitted:
        fitted[key]["min_events_session_phi"] = int(min_events)
    n_pinned = sum(1 for row in fitted.values() if row["phi_source"] != "session")
    if n_pinned:
        by_animal = pd.Series(
            [
                str(animals[key])
                for key, row in fitted.items()
                if row["phi_source"] != "session"
            ]
        ).value_counts()
        logger.info(
            "pinned the concentration intercept for %d of %d session fits "
            "(fewer than %d events): %s",
            n_pinned,
            len(fitted),
            min_events,
            by_animal.to_dict(),
        )
    order = [key for key, _animal, _frame in ordered]
    return (
        [(key, fitted[key]) for key in order if key in fitted],
        [(key, skipped[key]) for key in order if key in skipped],
    )


__all__ = [
    "DEFAULT_MIN_EVENTS_SESSION_PHI",
    "DEFAULT_N_JOBS",
    "ESTIMATOR",
    "ESTIMATOR_SKIP_CLASSES",
    "LOG_PHI_INTERCEPT",
    "POOLED_ALL_ANIMALS_KEY",
    "animal_pooled_log_phi",
    "concentration_cols_for_cohort",
    "cross_validate_dirichlet_multinomial",
    "event_loglik",
    "fit_dirichlet_multinomial",
    "fit_sessions_dirichlet_multinomial",
    "multinomial_log_constant",
    "resolve_pooled_log_phi",
]
