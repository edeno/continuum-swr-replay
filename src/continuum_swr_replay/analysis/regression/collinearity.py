"""How well the other terms of a fit explain each term, on what the likelihood sees.

The design is weighted by the information at the fit, so that ``X~' X~`` is
the information ``H`` (:func:`weighted_design`):

- Poisson, Bernoulli: ``X~ = diag(sqrt(w)) X``, with ``w`` the family's
  weights at the fitted ``eta``; Gaussian: ``X~ = X`` (one scale for every
  row, which no measure below depends on);
- grouped softmax: per group ``g`` with response total ``n_g`` and
  probabilities ``p_g``, ``X~_g = sqrt(n_g) diag(sqrt(p_g)) (X_g - 1 p_g' X_g)``.

When the design has an intercept it is projected out of every other column
first, so a term's measures are about the variation the intercept does not
explain (else a 0/1 column whose mean is 0.85 would count as 85 % explained by
the intercept alone). Then, for each term ``i`` with columns ``Xi`` and the
other terms' columns ``X0``, QR of ``[X0, Xi]`` gives
``R = [[R00, R0i], [0, Rii]]`` and mgcv's three concurvity measures
(``concurvity(fit, full = TRUE)``; :func:`concurvity`):

    observed = ||R0i b||^2 / (||R0i b||^2 + ||Rii b||^2)     (b the term's fit)
    worst    = largest mu of  R0i' R0i v = mu (R0i' R0i + Rii' Rii) v
    estimate = ||R0i||_F^2 / (||R0i||_F^2 + ||Rii||_F^2)

For a term of one column the three are the R^2 of that column on the others,
and ``VIF = 1 / (1 - R^2)``. A term is separable when the others explain less
than ``MAX_EXPLAINED`` of it: ``r2`` for a term of one column, ``estimate``
for a term of several (:func:`separable`).

mgcv computes the measures on the unweighted model matrix, with the intercept
among the parametric columns ("para"). A smooth whose sum-to-zero constraint
was computed on the same rows is orthogonal to the intercept, so on unit
weights its measures are mgcv's.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.linalg

from .likelihoods import Gaussian, GroupedSoftmax, Likelihood

__all__ = [
    "MAX_EXPLAINED",
    "concurvity",
    "separable",
    "term_collinearity",
    "weighted_design",
]

#: A term is separable from the others when they explain less than this share
#: of it: R^2 0.8, a variance inflation factor of 5.
MAX_EXPLAINED = 0.8


def weighted_design(
    lik: Likelihood, X: np.ndarray, y: np.ndarray, aux: object, eta: np.ndarray
) -> np.ndarray:
    """The design weighted by the information at ``eta``.

    Parameters
    ----------
    lik : Likelihood
    X : np.ndarray, shape (n_rows, p)
    y : np.ndarray, shape (n_rows,)
        The response (the softmax's group totals weight its groups).
    aux : object
        The family's aux: the offset, the group ids; ignored for the Gaussian.
    eta : np.ndarray, shape (n_rows,)
        ``X @ beta`` at the fit.

    Returns
    -------
    np.ndarray, shape (n_rows, p)
        ``X~``, with ``X~' X~`` the information (the Gaussian's at unit scale).
    """
    X = np.asarray(X, dtype=float)
    if isinstance(lik, GroupedSoftmax):
        starts, sizes, p, total = lik._parts(eta, y, aux)
        means = np.add.reduceat(p[:, None] * X, starts, axis=0)
        centered = X - np.repeat(means, sizes, axis=0)
        return centered * np.sqrt(np.repeat(total, sizes) * p)[:, None]
    if isinstance(lik, Gaussian):
        return X
    weights = lik._weights(eta, aux)[0]
    return X * np.sqrt(weights)[:, None]


def concurvity(
    X: np.ndarray,
    terms: Mapping[str, np.ndarray],
    beta: np.ndarray,
    *,
    intercept: int | None,
) -> dict[str, dict[str, float]]:
    """mgcv's three concurvity measures of each term, on a given design.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, p)
        The (weighted) design.
    terms : mapping of str to np.ndarray of int
        Each term's columns; together with ``intercept`` they cover every
        column once.
    beta : np.ndarray, shape (p,)
        The fitted coefficients (``observed`` reads them).
    intercept : int or None
        The intercept's column, projected out of the others first; None
        without one.

    Returns
    -------
    dict of str to dict
        Term -> ``{"worst", "observed", "estimate"}``; each 0 for a model with
        one term.
    """
    columns = [np.asarray(c, dtype=int) for c in terms.values()]
    used = np.concatenate(columns) if columns else np.empty(0, int)
    expected = np.setdiff1d(
        np.arange(X.shape[1]), [] if intercept is None else [intercept]
    )
    if np.sort(used).tolist() != expected.tolist():
        raise ValueError("the terms and the intercept must cover every column once")
    if intercept is not None:
        q = X[:, intercept] / np.linalg.norm(X[:, intercept])
        X = X - np.outer(q, q @ X)
    R = np.linalg.qr(X[:, used], mode="r")  # every term's columns, once
    position = {int(c): k for k, c in enumerate(used)}
    out = {}
    for name, own in zip(terms, columns, strict=True):
        mine = [position[int(c)] for c in own]
        rest = [k for k in range(used.size) if k not in set(mine)]
        R_term = np.linalg.qr(R[:, rest + mine], mode="r")[:, len(rest) :]
        R0i, Rii = R_term[: len(rest)], R_term[len(rest) :]
        b = beta[own]
        explained, left = R0i @ b, Rii @ b
        cross = R0i.T @ R0i
        whole = cross + Rii.T @ Rii
        out[name] = {
            "worst": float(scipy.linalg.eigh(cross, whole, eigvals_only=True)[-1]),
            "observed": float(
                explained @ explained / (explained @ explained + left @ left)
            ),
            "estimate": float(np.sum(R0i**2) / (np.sum(R0i**2) + np.sum(Rii**2))),
        }
    return out


def term_collinearity(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    beta: np.ndarray,
    terms: Mapping[str, np.ndarray],
    *,
    intercept: int | None,
) -> dict[str, dict[str, float]]:
    """Each term's collinearity with the others at a fit (the record's
    ``collinearity``).

    The design is weighted by the information at ``beta``
    (:func:`weighted_design`); a column whose coefficient is NaN (inactive in
    the session) is left out, and a term with none left is not measured.

    Parameters
    ----------
    lik : Likelihood
    X : np.ndarray, shape (n_rows, p)
    y : np.ndarray, shape (n_rows,)
    aux : object
    beta : np.ndarray, shape (p,)
        The fit; NaN at its inactive columns.
    terms : mapping of str to np.ndarray of int
        Each term's columns; with ``intercept`` they cover every column once.
    intercept : int or None
        The intercept's column; None without one.

    Returns
    -------
    dict of str to dict
        Term -> ``{"r2", "vif"}`` for a term of one column, else
        ``{"observed", "worst", "estimate"}``.
    """
    active = np.isfinite(beta)
    if intercept is not None and not active[intercept]:
        raise ValueError("the intercept is inactive")
    index = np.cumsum(active) - 1  # a column's place among the active ones
    kept = {
        name: index[cols[active[cols]]]
        for name, cols in terms.items()
        if active[np.asarray(cols, dtype=int)].any()
    }
    Xa, ba = X[:, active], beta[active]
    weighted = weighted_design(lik, Xa, y, aux, Xa @ ba)
    measures = concurvity(
        weighted, kept, ba, intercept=None if intercept is None else index[intercept]
    )
    out = {}
    for name, cols in kept.items():
        m = measures[name]
        if cols.size == 1:
            r2 = m["estimate"]
            out[name] = {"r2": r2, "vif": 1.0 / (1.0 - r2)}
        else:
            out[name] = {key: m[key] for key in ("observed", "worst", "estimate")}
    return out


def separable(collinearity: Mapping[str, Mapping[str, float]]) -> dict[str, bool]:
    """Term -> True when the others explain less than ``MAX_EXPLAINED`` of it.

    Parameters
    ----------
    collinearity : mapping
        :func:`term_collinearity`'s output (a record's ``collinearity``):
        ``r2`` is read for a term of one column, ``estimate`` for a term of
        several.
    """
    return {
        name: (m["r2"] if "r2" in m else m["estimate"]) < MAX_EXPLAINED
        for name, m in collinearity.items()
    }
