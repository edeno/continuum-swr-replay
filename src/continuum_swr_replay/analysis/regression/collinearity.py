"""How well the other terms of a fit explain each term, on what the likelihood sees.

The design is weighted by the information at the fit, so that ``X~' X~`` is
the information ``H`` (:func:`weighted_design`):

- Poisson, Bernoulli: ``X~ = diag(sqrt(w)) X``, with ``w`` the family's
  weights at the fitted ``eta``; Gaussian: ``X~ = X`` (one scale for every
  row, which no measure below depends on);
- grouped softmax: per group ``g`` with response total ``n_g`` and
  probabilities ``p_g``, ``X~_g = sqrt(n_g) diag(sqrt(p_g)) (X_g - 1 p_g' X_g)``.

When the design has an intercept it is projected out of every other column
first (in a fit of several sessions, the baseline copies of each session,
``terms.py``, in its place; a day copy of a per-day level is not projected
out: it is an ordinary column of its category's term, one of the shared
columns of each session's block), so a term's measures are about the
variation the intercept does not explain (else a 0/1 column whose mean is
0.85 would count as 85 % explained by the intercept alone). Then, for each term ``i`` with columns ``Xi`` and the
other terms' columns ``X0``, QR of ``[X0, Xi]`` gives
``R = [[R00, R0i], [0, Rii]]`` and mgcv's three concurvity measures
(``concurvity(fit, full = TRUE)``; :func:`concurvity`):

    observed = ||R0i b||^2 / (||R0i b||^2 + ||Rii b||^2)     (b the term's fit)
    worst    = largest mu of  R0i' R0i v = mu (R0i' R0i + Rii' Rii) v
    estimate = ||R0i||_F^2 / (||R0i||_F^2 + ||Rii||_F^2)

``R0i`` is computed in an orthonormal basis of the other columns' range (a
column zero in a session, such as a smooth's basis function beyond the
session's values, adds no direction). For a term of one column the three are
the R^2 of that column on the others,
and ``VIF = 1 / (1 - R^2)``. A term is separable when the others explain less
than ``MAX_EXPLAINED`` of it: ``r2`` for a term of one column, ``estimate``
for a term of several (:func:`separable`).

mgcv computes the measures on the unweighted model matrix, with the intercept
among the parametric columns ("para"). A smooth whose sum-to-zero constraint
was computed on the same rows is orthogonal to the intercept on those rows, so
on unit weights its measures are mgcv's. A smooth centered on declared points
(``terms.Smooth.center_on``) is not orthogonal to the intercept on the fit's
rows: its concurvity, and the VIF of a slope beside it, then depend on where
its constraint puts the constant (the parametrization), not only on the data;
the same model centered on its rows gives other values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .likelihoods import GroupedSoftmax, Likelihood
from .reml import _positions
from .runtime import in_float64

__all__ = [
    "MAX_EXPLAINED",
    "NOT_VARYING",
    "RANK_TOL",
    "VARIES_TOL",
    "concurvity",
    "separable",
    "term_collinearity",
    "weighted_design",
]

#: A term is separable from the others when they explain less than this share
#: of it: R^2 0.8, a variance inflation factor of 5.
MAX_EXPLAINED = 0.8

#: A direction of a term's columns is seen by the data when its eigenvalue in
#: their Gram matrix is above RANK_TOL times the largest.
RANK_TOL = 1e-12

#: A term varies in a session when its columns keep more than VARIES_TOL of
#: their squared size once the intercept is projected out; below it what is
#: left is rounding.
VARIES_TOL = 1e-10

#: Why a term that does not vary is not measured.
NOT_VARYING = "does not vary in the session beyond the intercept"


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
    return lik.weighted_design(X, y, aux, eta)


def concurvity(
    X: np.ndarray,
    terms: Mapping[str, np.ndarray],
    beta: np.ndarray,
    *,
    intercept: Sequence[int] | None,
    size: np.ndarray | None = None,
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
    intercept : sequence of int or None
        The columns projected out of the others first: the intercept's, and
        in a fit of several sessions the baseline copies'. They are projected
        one after another, each on what the ones before it left, so the
        projection is onto their span. None (or empty) for none.
    size : np.ndarray, shape (p,), or None
        Each column's squared size before any projection, which the test of
        a term that does not vary reads; ``sum(X**2)`` when None. Given when
        ``X`` is already a reduced form of the design (:func:`term_collinearity`
        by session).

    Returns
    -------
    dict of str to dict
        Term -> ``{"worst", "observed", "estimate"}``; each 0 for a model with
        one term. A term whose columns, after the intercept is projected out,
        keep at most ``VARIES_TOL`` of their own squared size (a column
        constant in the session, or zero) is not measured: NaN, and
        ``"not_measured"`` gives the reason.
    """
    columns = [np.asarray(c, dtype=int) for c in terms.values()]
    used = np.concatenate(columns) if columns else np.empty(0, int)
    projected = [] if intercept is None else [int(c) for c in intercept]
    expected = np.setdiff1d(np.arange(X.shape[1]), projected)
    if np.sort(used).tolist() != expected.tolist():
        raise ValueError("the terms and the intercept must cover every column once")
    if size is None:  # each column's own size, before the projection
        size = np.sum(X**2, axis=0)
    for column in projected:
        q = X[:, column] / np.linalg.norm(X[:, column])
        X = X - np.outer(q, q @ X)
    R = np.linalg.qr(X[:, used], mode="r")  # R'R = X'X over the terms' columns
    position = {int(c): k for k, c in enumerate(used)}
    out = {}
    for name, own in zip(terms, columns, strict=True):
        mine = [position[int(c)] for c in own]
        rest = np.setdiff1d(np.arange(used.size), mine)
        A = R[:, mine]
        explained = _range(R[:, rest]).T @ A  # R0i, in a basis of X0's range
        b = beta[own]
        seen, total = explained @ b, A @ b
        cross, whole = explained.T @ explained, A.T @ A
        if not np.trace(whole) > VARIES_TOL * size[own].sum():
            out[name] = {
                "worst": np.nan,
                "observed": np.nan,
                "estimate": np.nan,
                "not_measured": NOT_VARYING,
            }
            continue
        out[name] = {
            "worst": _largest_ratio(cross, whole),
            "observed": _share(seen @ seen, total @ total),
            "estimate": _share(np.trace(cross), np.trace(whole)),
        }
    return out


def _range(M: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the range of ``M``, its singular values above
    ``RANK_TOL`` times the largest: the other terms of a session can hold
    columns that are zero there (a smooth's basis functions beyond the
    session's values), which add no direction."""
    if M.shape[1] == 0:
        return M[:, :0]
    U, singular, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, singular > np.sqrt(RANK_TOL) * max(singular[0], 0.0)]


def _share(part: float, whole: float) -> float:
    """``part / whole``; 0 when the whole is 0."""
    return float(part / whole) if whole > 0 else 0.0


def _largest_ratio(cross: np.ndarray, whole: np.ndarray) -> float:
    """The largest ``mu`` of ``cross v = mu whole v`` over the range of
    ``whole``: a term's columns that are zero in a session (a smooth's basis
    functions outside the session's values, which the penalty identifies)
    add directions the data do not see, and are left out."""
    values, vectors = np.linalg.eigh(whole)
    seen = values > RANK_TOL * max(values[-1], 0.0)
    if not seen.any():
        return 0.0
    W = vectors[:, seen] / np.sqrt(values[seen])
    return float(np.linalg.eigvalsh(W.T @ cross @ W)[-1])


@in_float64
def term_collinearity(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    beta: np.ndarray,
    terms: Mapping[str, np.ndarray],
    *,
    intercept: Sequence[int] | None,
    session_copies: Sequence[tuple[slice, Sequence[int]]] | None = None,
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
    intercept : sequence of int or None
        The columns projected out first (:func:`concurvity`): the
        intercept's and the active baseline copies'; None (or empty) for
        none. Each must be active.
    session_copies : sequence of (slice, sequence of int), or None
        In a fit of several sessions with baseline copies, each session's
        rows (contiguous) and its active baseline copy columns (a day copy
        of a per-day level is a column of its term, among the shared ones).
        Then each session's copies are projected out of its own rows (a copy
        is zero on the
        other sessions' rows, also once weighted, so the copies of different
        sessions are orthogonal): for each session, the weighted rows of its
        copies and of the other active columns, the copies' range (their
        singular directions above ``RANK_TOL``, as :func:`_range`) taken out
        of the others, and the R of what is left; the blocks are stacked, and
        :func:`concurvity` runs on them with the other columns of
        ``intercept`` (a shared intercept) and each column's size before the
        projection. The measures depend on the design only through ``X~' X~``,
        so they are those of the projection of every copy out of the whole
        design (to rounding), without its ``n x p`` copies per copy. One
        difference: a copy that its session's other copies explain wholly
        adds no direction here, while the projection one copy at a time
        divides what rounding leaves of it by its norm and so projects out an
        arbitrary direction (NaN for every term when the norm is exactly
        zero). None: every column of ``intercept`` is projected out of the
        whole design.

    Returns
    -------
    dict of str to dict
        Term -> ``{"r2", "vif"}`` for a term of one column (``vif`` infinite
        at ``r2`` 1), else ``{"observed", "worst", "estimate"}``; a term that
        does not vary in the session has NaN there and ``"not_measured"``
        with the reason.
    """
    active = np.isfinite(beta)
    projected = [] if intercept is None else [int(c) for c in intercept]
    if not active[projected].all():
        raise ValueError("the intercept is inactive")
    index = _positions(active)
    kept = {
        name: index[cols[active[cols]]]
        for name, cols in terms.items()
        if active[np.asarray(cols, dtype=int)].any()
    }
    if session_copies is None:
        Xa, ba = X[:, active], beta[active]
        weighted = weighted_design(lik, Xa, y, aux, Xa @ ba)
        measures = concurvity(
            weighted, kept, ba, intercept=[int(index[c]) for c in projected]
        )
    else:
        measures = _concurvity_by_session(
            lik, X, y, aux, beta, terms, projected, session_copies
        )
    out = {}
    for name, cols in kept.items():
        m = measures[name]
        if cols.size == 1:
            r2 = m["estimate"]
            vif = np.inf if 1.0 - r2 <= RANK_TOL else 1.0 / (1.0 - r2)
            out[name] = {"r2": r2, "vif": float(np.nan if np.isnan(r2) else vif)}
        else:
            out[name] = {key: m[key] for key in ("observed", "worst", "estimate")}
        if "not_measured" in m:
            out[name]["not_measured"] = m["not_measured"]
    return out


def _concurvity_by_session(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    beta: np.ndarray,
    terms: Mapping[str, np.ndarray],
    projected: list[int],
    session_copies: Sequence[tuple[slice, Sequence[int]]],
) -> dict[str, dict[str, float]]:
    """:func:`concurvity` of the active term columns with each session's
    copies projected out of its own rows (:func:`term_collinearity`,
    ``session_copies``)."""
    active = np.isfinite(beta)
    copies = {int(c) for _, own in session_copies for c in own}
    shared = [int(j) for j in np.flatnonzero(active) if int(j) not in copies]
    place = {c: k for k, c in enumerate(shared)}
    size = np.zeros(len(shared))
    blocks = []
    for rows, own in session_copies:
        if rows.stop <= rows.start:
            continue
        columns = [*own, *shared]
        Xs = X[rows][:, columns]
        part = None if aux is None else np.asarray(aux)[rows]
        if isinstance(lik, GroupedSoftmax):
            part = part - part[0]  # the session's groups, numbered from 0
        Ws = weighted_design(lik, Xs, y[rows], part, Xs @ beta[columns])
        left = Ws[:, len(own) :]
        size += np.sum(left**2, axis=0)
        basis = _range(Ws[:, : len(own)])
        left = left - basis @ (basis.T @ left)
        blocks.append(np.linalg.qr(left, mode="r"))
    kept = {
        name: np.array([place[int(c)] for c in cols if active[c]])
        for name, cols in terms.items()
        if active[np.asarray(cols, dtype=int)].any()
    }
    return concurvity(
        np.vstack(blocks),
        kept,
        beta[shared],
        intercept=[place[c] for c in projected if c not in copies],
        size=size,
    )


def separable(collinearity: Mapping[str, Mapping[str, float]]) -> dict[str, bool]:
    """Term -> True when the others explain less than ``MAX_EXPLAINED`` of it
    (False for a term not measured).

    Parameters
    ----------
    collinearity : mapping
        :func:`term_collinearity`'s output (a record's ``collinearity``):
        ``r2`` is read for a term of one column, ``estimate`` for a term of
        several.
    """
    return {
        name: bool((m["r2"] if "r2" in m else m["estimate"]) < MAX_EXPLAINED)
        for name, m in collinearity.items()
    }
