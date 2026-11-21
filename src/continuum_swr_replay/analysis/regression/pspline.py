"""Cubic P-splines: a B-spline basis on fixed, equally spaced knots.

A smooth term ``f(x) = B(x) beta`` uses ``n_basis`` B-splines of ``degree``
whose inner knots split a fixed range ``[lo, hi]`` into equal intervals, the
same in every session. Outside the range ``x`` is clamped to it, so the curve
is flat beyond either end. The penalty is the squared second-order difference
of the coefficients, ``P = D2' D2``; on equally spaced knots its null space is
the coefficients linear in their index, which draw straight lines in ``x``.

One sum-to-zero constraint makes a smooth identifiable beside an intercept:
``Z`` spans the coefficients whose curve sums to zero over a given set of rows
(the pooled rows of a model, computed once and used for every session). The
constrained design is ``B Z`` and its penalty ``Z' P Z``. This is mgcv's
construction for ``bs = "ps"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "ConstrainedSmooth",
    "PSplineBasis",
    "difference_penalty",
    "penalty_rank",
    "sum_to_zero_constraint",
]


class ConstrainedSmooth(NamedTuple):
    """A smooth term with its sum-to-zero constraint (:meth:`PSplineBasis.constrained`).

    Attributes
    ----------
    design : np.ndarray, shape (n_rows, n_basis - 1)
        ``B Z``, the constrained design at the rows given.
    penalty : np.ndarray, shape (n_basis - 1, n_basis - 1)
        ``Z' P Z``, the difference penalty of the constrained coefficients.
    constraint : np.ndarray, shape (n_basis, n_basis - 1)
        ``Z`` (:func:`sum_to_zero_constraint`), to build the design at other
        rows as ``basis.design(x) @ Z``.
    """

    design: np.ndarray
    penalty: np.ndarray
    constraint: np.ndarray


@dataclass(frozen=True)
class PSplineBasis:
    """B-splines on equally spaced knots over ``[lo, hi]``.

    Attributes
    ----------
    lo, hi : float
        The range; inner knots at ``lo`` and ``hi``.
    n_basis : int
        Number of basis functions; the range holds ``n_basis - degree``
        intervals.
    degree : int
        Polynomial degree (3, cubic).
    """

    lo: float
    hi: float
    n_basis: int
    degree: int = 3

    def __post_init__(self) -> None:
        if not self.hi > self.lo:
            raise ValueError(f"empty range [{self.lo}, {self.hi}]")
        if self.n_basis <= self.degree:
            raise ValueError(f"n_basis={self.n_basis} must exceed degree={self.degree}")

    @classmethod
    def from_spacing(
        cls, lo: float, hi: float, spacing: float, degree: int = 3
    ) -> PSplineBasis:
        """The basis whose knots are ``spacing`` apart over ``[lo, hi]``.

        Raises
        ------
        ValueError
            If ``spacing`` does not divide the range into a whole number of
            intervals.
        """
        n_intervals = (hi - lo) / spacing
        if not np.isclose(n_intervals, round(n_intervals)):
            raise ValueError(
                f"spacing {spacing} does not split [{lo}, {hi}] into a whole "
                "number of intervals"
            )
        return cls(lo, hi, int(round(n_intervals)) + degree, degree)

    @property
    def knots(self) -> np.ndarray:
        """The knot vector, shape ``(n_basis + degree + 1,)``."""
        n_intervals = self.n_basis - self.degree
        width = (self.hi - self.lo) / n_intervals
        return self.lo + width * np.arange(-self.degree, n_intervals + self.degree + 1)

    def design(self, x: np.ndarray) -> np.ndarray:
        """The basis at ``x``, clamped to ``[lo, hi]``.

        Parameters
        ----------
        x : np.ndarray, shape (n_rows,)

        Returns
        -------
        np.ndarray, shape (n_rows, n_basis)
            Each row sums to one.
        """
        clamped = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        if clamped.size == 0:
            return np.empty((0, self.n_basis))
        return BSpline.design_matrix(
            clamped, self.knots, self.degree, extrapolate=False
        ).toarray()

    def penalty(self, order: int = 2) -> np.ndarray:
        """The difference penalty of the coefficients, shape ``(n_basis, n_basis)``."""
        return difference_penalty(self.n_basis, order)

    def constrained(self, x: np.ndarray, order: int = 2) -> ConstrainedSmooth:
        """The smooth at ``x`` with the sum-to-zero constraint computed on ``x``.

        Parameters
        ----------
        x : np.ndarray, shape (n_rows,)
            The rows the constraint is computed on (the pooled rows of a model).
        order : int
            The order of the difference penalty.

        Returns
        -------
        ConstrainedSmooth
            ``B Z``, ``Z' P Z`` and ``Z``, with ``B`` the basis at ``x``.
        """
        B = self.design(x)
        Z = sum_to_zero_constraint(B)
        return ConstrainedSmooth(B @ Z, Z.T @ self.penalty(order) @ Z, Z)


def difference_penalty(n_basis: int, order: int = 2) -> np.ndarray:
    """``D' D`` with ``D`` the ``order``-th difference matrix.

    Returns
    -------
    np.ndarray, shape (n_basis, n_basis)
        Rank ``n_basis - order``; its null space is the polynomials of degree
        below ``order`` in the coefficient index.
    """
    D = np.diff(np.eye(n_basis), n=order, axis=0)
    return D.T @ D


def sum_to_zero_constraint(B: np.ndarray) -> np.ndarray:
    """The coefficients whose curve sums to zero over the rows of ``B``.

    With ``c = sum_rows B``, ``Z`` is an orthonormal basis of the null space of
    ``c'``, from a complete QR of ``c``.

    Parameters
    ----------
    B : np.ndarray, shape (n_rows, n_basis)
        The basis at the rows the constraint is computed on.

    Returns
    -------
    np.ndarray, shape (n_basis, n_basis - 1)
    """
    c = B.sum(axis=0)
    Q, _ = np.linalg.qr(c[:, None], mode="complete")
    return Q[:, 1:]


def penalty_rank(P: np.ndarray) -> int:
    """Numerical rank of a symmetric penalty matrix."""
    return int(np.linalg.matrix_rank(P, hermitian=True))
