"""Log-likelihoods of the shared fitter and their derivatives on the eta scale.

Each family is a small stateless class with the same five methods. ``eta`` is
the linear predictor ``X @ beta``, shape ``(n_rows,)``; ``aux`` is the family's
extra data (below), passed to every method so that one object serves every
session and, for the Gaussian, every scale.

- ``check(y, aux)``: raise ``ValueError``, naming the problem, when the
  response or ``aux`` is outside what the family takes (the fit calls it once,
  at its entry, after checking that ``y`` is finite).
- ``loglik(eta, y, aux)``: the log-likelihood, a float.
- ``score_eta(eta, y, aux)``: ``dl/deta``, shape ``(n_rows,)``; the coefficient
  gradient is ``X.T @ score_eta``.
- ``information(X, eta, y, aux)``: ``-d2l/dbeta2``, shape ``(p, p)``.
- ``dinformation(X, eta, y, aux, d)``: the derivative of ``information`` when
  ``eta`` moves along ``d`` (shape ``(n_rows,)``), shape ``(p, p)``.
- ``d2information(X, eta, y, aux, d1, d2)``: the second derivative of
  ``information`` along ``d1`` and ``d2``, shape ``(p, p)``.

The information is the observed information. For the three diagonal families it
is ``X' diag(w) X`` with weights ``w(eta)`` that do not depend on ``y``; their
directional derivatives are ``X' diag(w' d) X`` and ``X' diag(w'' d1 d2) X``.

=================  =====================================  ============  ==========  ==========
family             aux                                    w             w'          w''
=================  =====================================  ============  ==========  ==========
Poisson, log link  offset = log(exposure), (n_rows,)      mu            mu          mu
Bernoulli, logit   None                                   mu (1 - mu)   w (1-2 mu)  w (1-6 w)
Gaussian           phi, the scale (float)                 1 / phi       0           0
=================  =====================================  ============  ==========  ==========

The grouped softmax (one group per event, one row per candidate) takes the
group id of each row as ``aux``, sorted so that a group's rows are contiguous.
Constants are left out where they do not depend on ``beta``: the Poisson
``-log y!`` and the multinomial coefficient. The Gaussian keeps its
normalizing constant, which depends on the scale.

A family's ``quadratic`` is True when its log-likelihood is quadratic in
``eta`` (the Gaussian): one full Newton step from any start is then the exact
optimum, and the fit takes it without a cap on its length.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.special import expit

__all__ = [
    "Bernoulli",
    "Gaussian",
    "GroupedSoftmax",
    "Likelihood",
    "Poisson",
    "group_starts",
]


class Likelihood(Protocol):
    """The interface the Newton fit and REML read."""

    name: str
    quadratic: bool

    def check(self, y: np.ndarray, aux: object) -> None: ...

    def identifying_design(self, X: np.ndarray, aux: object) -> np.ndarray: ...

    def loglik(self, eta: np.ndarray, y: np.ndarray, aux: object) -> float: ...

    def score_eta(self, eta: np.ndarray, y: np.ndarray, aux: object) -> np.ndarray: ...

    def information(
        self, X: np.ndarray, eta: np.ndarray, y: np.ndarray, aux: object
    ) -> np.ndarray: ...

    def dinformation(
        self, X: np.ndarray, eta: np.ndarray, y: np.ndarray, aux: object, d: np.ndarray
    ) -> np.ndarray: ...

    def d2information(
        self,
        X: np.ndarray,
        eta: np.ndarray,
        y: np.ndarray,
        aux: object,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray: ...


def _weighted_gram(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``X' diag(weights) X``, shape ``(p, p)``."""
    return (X.T * weights) @ X


class _DiagonalFamily:
    """A family whose information is ``X' diag(w(eta)) X``."""

    def identifying_design(self, X: np.ndarray, aux: object) -> np.ndarray:
        """The design whose column space the likelihood identifies: ``X``."""
        return X

    def _weights(
        self, eta: np.ndarray, aux: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(w, dw/deta, d2w/deta2)``, each shape ``(n_rows,)``."""
        raise NotImplementedError

    def information(
        self, X: np.ndarray, eta: np.ndarray, y: np.ndarray, aux: object
    ) -> np.ndarray:
        return _weighted_gram(X, self._weights(eta, aux)[0])

    def dinformation(
        self, X: np.ndarray, eta: np.ndarray, y: np.ndarray, aux: object, d: np.ndarray
    ) -> np.ndarray:
        return _weighted_gram(X, self._weights(eta, aux)[1] * d)

    def d2information(
        self,
        X: np.ndarray,
        eta: np.ndarray,
        y: np.ndarray,
        aux: object,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray:
        return _weighted_gram(X, self._weights(eta, aux)[2] * d1 * d2)


class Poisson(_DiagonalFamily):
    """Poisson counts with a log link and an offset ``aux = log(exposure)``.

    ``l = sum y (eta + o) - exp(eta + o)``.
    """

    name = "poisson"
    quadratic = False

    def check(self, y: np.ndarray, aux: np.ndarray) -> None:
        """Counts at least zero; an offset per row, finite (a row of zero
        exposure, offset minus infinity, carries no likelihood: drop it)."""
        if np.any(y < 0):
            raise ValueError(f"Poisson response below zero on {np.sum(y < 0)} rows")
        offset = np.asarray(aux, dtype=float)
        if offset.shape != y.shape:
            raise ValueError(f"offset of shape {offset.shape} for {y.size} rows")
        if np.any(offset == -np.inf):
            raise ValueError(
                f"zero exposure (offset -inf) on {np.sum(offset == -np.inf)} rows: "
                "drop those rows, which carry no likelihood"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError(
                f"offset not finite on {np.sum(~np.isfinite(offset))} rows"
            )

    def loglik(self, eta: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        log_mu = eta + aux
        return float(y @ log_mu - np.exp(log_mu).sum())

    def score_eta(self, eta: np.ndarray, y: np.ndarray, aux: np.ndarray) -> np.ndarray:
        return y - np.exp(eta + aux)

    def _weights(
        self, eta: np.ndarray, aux: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.exp(eta + aux)
        return mu, mu, mu


class Bernoulli(_DiagonalFamily):
    """0/1 responses with a logit link; ``aux`` is unused.

    ``l = sum y eta - log(1 + exp(eta))``.
    """

    name = "bernoulli"
    quadratic = False

    def check(self, y: np.ndarray, aux: object) -> None:
        """0/1 outcomes only; the family does not take proportions."""
        outside = (y != 0) & (y != 1)
        if np.any(outside):
            raise ValueError(
                f"Bernoulli response outside {{0, 1}} on {np.sum(outside)} rows "
                "(it takes 0/1 outcomes, not proportions)"
            )

    def loglik(self, eta: np.ndarray, y: np.ndarray, aux: object) -> float:
        return float(y @ eta - np.logaddexp(0.0, eta).sum())

    def score_eta(self, eta: np.ndarray, y: np.ndarray, aux: object) -> np.ndarray:
        return y - expit(eta)

    def _weights(
        self, eta: np.ndarray, aux: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = expit(eta)
        w = mu * (1.0 - mu)
        return w, w * (1.0 - 2.0 * mu), w * (1.0 - 6.0 * w)


class Gaussian(_DiagonalFamily):
    """Normal responses with an identity link and scale ``aux = phi``.

    ``l = -sum (y - eta)^2 / (2 phi) - n/2 log(2 pi phi)``.
    """

    name = "gaussian"
    quadratic = True

    def check(self, y: np.ndarray, aux: float) -> None:
        """A finite, positive scale."""
        if not (np.ndim(aux) == 0 and np.isfinite(aux) and aux > 0):
            raise ValueError(
                f"the Gaussian scale must be finite and positive, not {aux}"
            )

    def loglik(self, eta: np.ndarray, y: np.ndarray, aux: float) -> float:
        residual = y - eta
        return float(
            -(residual @ residual) / (2.0 * aux)
            - 0.5 * y.size * np.log(2.0 * np.pi * aux)
        )

    def score_eta(self, eta: np.ndarray, y: np.ndarray, aux: float) -> np.ndarray:
        return (y - eta) / aux

    def _weights(
        self, eta: np.ndarray, aux: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero = np.zeros_like(eta)
        return np.full_like(eta, 1.0 / aux), zero, zero


def group_starts(groups: np.ndarray) -> np.ndarray:
    """Index of each group's first row, shape ``(n_groups,)``.

    Raises
    ------
    ValueError
        If ``groups`` is empty or not sorted (a group's rows must be
        contiguous).
    """
    groups = np.asarray(groups)
    if groups.size == 0:
        raise ValueError("no rows")
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError("group ids must be sorted so each group's rows are contiguous")
    return np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])


class GroupedSoftmax:
    """A softmax over each group's rows; ``aux`` is the sorted group id per row.

    For group ``g`` with response ``y_gk`` (a fraction summing to one, or a
    count), ``n_g = sum_k y_gk`` and ``p_g = softmax(eta_g)``::

        l = sum_g sum_k y_gk log p_gk
        dl/deta_gk = y_gk - n_g p_gk
        information = sum_g n_g X_g' G_g X_g,   G = diag(p) - p p'

    Within a group, with ``v = G d = p (d - p'd)``::

        D_d G      = diag(v) - v p' - p v'
        D_d2 D_d1 G = diag(q) - q p' - p q' - v1 v2' - v2 v1',   q = (D_d2 G) d1

    It is the conditional Poisson likelihood of a fixed group total. A constant
    within a group cancels, so a group-level column is not identified and must
    not be in ``X``.
    """

    name = "grouped_softmax"
    quadratic = False

    def check(self, y: np.ndarray, aux: np.ndarray) -> None:
        """Finite, sorted group ids, one per row; shares (or counts) at least
        zero; every group with a positive total."""
        groups = np.asarray(aux)
        if groups.shape != y.shape:
            raise ValueError(f"group ids of shape {groups.shape} for {y.size} rows")
        if groups.dtype.kind == "f" and not np.all(np.isfinite(groups)):
            raise ValueError(
                f"group ids not finite on {np.sum(~np.isfinite(groups))} rows"
            )
        if np.any(y < 0):
            raise ValueError(f"softmax shares negative on {np.sum(y < 0)} rows")
        empty = np.add.reduceat(y, group_starts(groups)) <= 0
        if np.any(empty):
            raise ValueError(
                f"{np.sum(empty)} group(s) whose responses sum to zero: drop them, "
                "they carry no likelihood"
            )

    def identifying_design(self, X: np.ndarray, aux: np.ndarray) -> np.ndarray:
        """``X`` less each group's column means, shape ``(n_rows, p)``: the
        likelihood sees only variation within a group, so a combination of
        columns constant within every group is not identified."""
        starts = group_starts(aux)
        sizes = np.diff(starts, append=X.shape[0])
        means = np.add.reduceat(X, starts, axis=0) / sizes[:, None]
        return X - np.repeat(means, sizes, axis=0)

    @staticmethod
    def _parts(
        eta: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(starts, sizes, p, total)``: ``p`` per row, shape ``(n_rows,)``;
        the group total ``n_g`` per group, shape ``(n_groups,)``."""
        starts = group_starts(groups)
        sizes = np.diff(starts, append=eta.size)
        top = np.maximum.reduceat(eta, starts)
        weight = np.exp(eta - np.repeat(top, sizes))
        p = weight / np.repeat(np.add.reduceat(weight, starts), sizes)
        return starts, sizes, p, np.add.reduceat(y, starts)

    def loglik(self, eta: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        starts = group_starts(aux)
        sizes = np.diff(starts, append=eta.size)
        top = np.maximum.reduceat(eta, starts)
        shifted = eta - np.repeat(top, sizes)
        log_norm = np.log(np.add.reduceat(np.exp(shifted), starts))
        return float(y @ shifted - np.add.reduceat(y, starts) @ log_norm)

    def score_eta(self, eta: np.ndarray, y: np.ndarray, aux: np.ndarray) -> np.ndarray:
        _, sizes, p, total = self._parts(eta, y, aux)
        return y - np.repeat(total, sizes) * p

    @staticmethod
    def _sandwich(
        X: np.ndarray,
        starts: np.ndarray,
        sizes: np.ndarray,
        total: np.ndarray,
        diagonal: np.ndarray,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """``sum_g n_g X_g' (diag(a) - sum_(u, v) u v') X_g`` for a per-row
        ``a`` and per-row vector pairs ``(u, v)``, shape ``(p, p)``."""
        out = _weighted_gram(X, np.repeat(total, sizes) * diagonal)
        for u, v in pairs:
            Xu = np.add.reduceat(u[:, None] * X, starts)
            Xv = np.add.reduceat(v[:, None] * X, starts)
            out -= (Xu.T * total) @ Xv
        return out

    def information(
        self, X: np.ndarray, eta: np.ndarray, y: np.ndarray, aux: np.ndarray
    ) -> np.ndarray:
        starts, sizes, p, total = self._parts(eta, y, aux)
        return self._sandwich(X, starts, sizes, total, p, [(p, p)])

    def dinformation(
        self,
        X: np.ndarray,
        eta: np.ndarray,
        y: np.ndarray,
        aux: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        starts, sizes, p, total = self._parts(eta, y, aux)
        v = self._g_times(p, d, starts, sizes)
        return self._sandwich(X, starts, sizes, total, v, [(v, p), (p, v)])

    def d2information(
        self,
        X: np.ndarray,
        eta: np.ndarray,
        y: np.ndarray,
        aux: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray:
        starts, sizes, p, total = self._parts(eta, y, aux)
        v1 = self._g_times(p, d1, starts, sizes)
        v2 = self._g_times(p, d2, starts, sizes)
        # q = (D_d2 G) d1 = v2 * d1 - v2 (p'd1) - p (v2'd1), per group.
        p_d1 = np.repeat(np.add.reduceat(p * d1, starts), sizes)
        v2_d1 = np.repeat(np.add.reduceat(v2 * d1, starts), sizes)
        q = v2 * (d1 - p_d1) - p * v2_d1
        return self._sandwich(
            X, starts, sizes, total, q, [(q, p), (p, q), (v1, v2), (v2, v1)]
        )

    @staticmethod
    def _g_times(
        p: np.ndarray, d: np.ndarray, starts: np.ndarray, sizes: np.ndarray
    ) -> np.ndarray:
        """``G d = p (d - p'd)`` per group, shape ``(n_rows,)``."""
        return p * (d - np.repeat(np.add.reduceat(p * d, starts), sizes))
