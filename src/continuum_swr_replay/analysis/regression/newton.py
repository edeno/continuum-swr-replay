"""Penalized Newton fit at fixed penalty strengths, in float64.

Minimizes ``F(beta) = -l(beta) + beta' S beta / 2`` with ``S`` the summed,
embedded penalty ``sum_j lam_j S_j`` (zero for an unpenalized fit). Each step
solves ``A step = g`` by Cholesky, with ``g`` the gradient of ``-F`` and
``A = H + S`` (``H`` the information), and the fit stops when the Newton
decrement ``g' A^-1 g / 2`` -- the gain in ``-F`` the step predicts -- is below
the tolerance. A step is shortened so no coefficient moves more than
``max_step``, then halved until ``F`` does not rise; a step whose ``F`` is not
finite is rejected the same way. ``eta`` is never clipped.

Once the decrement is below the tolerance the fit still takes that last full
step, and reports ``A`` and the decrement where it lands. Stopping before it
would leave a coefficient off by up to about ``sqrt(2 tol / lambda_min)``
(``lambda_min`` the smallest eigenvalue of ``A``): 4e-5 on a content fit of 80
events. After it the error is of the order of the square of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .constants import (
    MAX_STEP_HALVINGS,
    MAX_STEP_LENGTH,
    NEWTON_DECREMENT_TOL,
    NEWTON_MAX_STEPS,
)
from .likelihoods import Likelihood

__all__ = ["NewtonFit", "penalized_newton"]


@dataclass(frozen=True)
class NewtonFit:
    """The end state of :func:`penalized_newton`.

    Attributes
    ----------
    beta : np.ndarray, shape (p,)
        The coefficients.
    A : np.ndarray, shape (p, p)
        ``H + S`` at ``beta``; its inverse is the Bayesian covariance.
    converged : bool
        True when the decrement at ``beta`` is below the tolerance.
    newton_decrement : float
        ``g' A^-1 g / 2`` at ``beta``; NaN when ``A`` is not positive definite.
    n_iter : int
        Newton steps taken.
    loglik : float
        The log-likelihood at ``beta`` (without the penalty).
    """

    beta: np.ndarray
    A: np.ndarray
    converged: bool
    newton_decrement: float
    n_iter: int
    loglik: float


def penalized_newton(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    beta0: np.ndarray,
    *,
    tol: float = NEWTON_DECREMENT_TOL,
    max_iter: int = NEWTON_MAX_STEPS,
    max_halvings: int = MAX_STEP_HALVINGS,
    max_step: float = MAX_STEP_LENGTH,
) -> NewtonFit:
    """Fit ``beta`` by penalized Newton from ``beta0``.

    Parameters
    ----------
    lik : Likelihood
        The family (:mod:`.likelihoods`).
    X : np.ndarray, shape (n_rows, p)
        Design.
    y : np.ndarray, shape (n_rows,)
        Response.
    aux : object
        The family's extra data (offset, scale, group ids).
    S : np.ndarray, shape (p, p)
        The summed penalty at the chosen strengths; zeros for none.
    beta0 : np.ndarray, shape (p,)
        Start.
    tol : float
        Stop when the Newton decrement is below this.
    max_iter : int
        Most Newton steps.
    max_halvings : int
        Most halvings of one step.
    max_step : float
        Longest step, as the largest change of any coefficient.

    Returns
    -------
    NewtonFit
        Not converged, without raising, when ``A`` is not positive definite
        (an unpenalized design the data cannot identify), when no halving of a
        step lowers ``F``, or after ``max_iter`` steps.
    """
    beta = np.array(beta0, dtype=float)

    def objective(b: np.ndarray) -> tuple[float, float]:
        loglik = lik.loglik(X @ b, y, aux)
        return -loglik + 0.5 * b @ S @ b, loglik

    F, loglik = objective(beta)
    for n_iter in range(max_iter + 1):
        eta = X @ beta
        g = X.T @ lik.score_eta(eta, y, aux) - S @ beta
        A = lik.information(X, eta, y, aux) + S
        try:
            factor = scipy.linalg.cho_factor(A, lower=True)
        except np.linalg.LinAlgError:
            return NewtonFit(beta, A, False, np.nan, n_iter, loglik)
        step = scipy.linalg.cho_solve(factor, g)
        decrement = 0.5 * float(g @ step)
        if decrement < tol:
            return _last_step(
                lik,
                X,
                y,
                aux,
                S,
                NewtonFit(beta, A, True, decrement, n_iter, loglik),
                step,
            )
        if n_iter == max_iter:
            break
        scale = min(1.0, max_step / np.max(np.abs(step)))
        for _ in range(max_halvings):
            candidate = beta + scale * step
            F_new, loglik_new = objective(candidate)
            if np.isfinite(F_new) and F_new <= F:
                break
            scale *= 0.5
        else:
            return NewtonFit(beta, A, False, decrement, n_iter, loglik)
        beta, F, loglik = candidate, F_new, loglik_new
    return NewtonFit(beta, A, False, decrement, n_iter, loglik)


def _last_step(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    fit: NewtonFit,
    step: np.ndarray,
) -> NewtonFit:
    """The converged ``fit`` moved by its full Newton ``step``.

    ``fit`` is returned unchanged when the step does not lower ``F`` (its gain
    is below the objective's rounding) or lands where ``A`` is not positive
    definite.
    """
    beta = fit.beta + step
    eta = X @ beta
    loglik = lik.loglik(eta, y, aux)
    F_old = -fit.loglik + 0.5 * fit.beta @ S @ fit.beta
    F_new = -loglik + 0.5 * beta @ S @ beta
    if not (np.isfinite(F_new) and F_new <= F_old):
        return fit
    g = X.T @ lik.score_eta(eta, y, aux) - S @ beta
    A = lik.information(X, eta, y, aux) + S
    try:
        factor = scipy.linalg.cho_factor(A, lower=True)
    except np.linalg.LinAlgError:
        return fit
    decrement = 0.5 * float(g @ scipy.linalg.cho_solve(factor, g))
    return NewtonFit(beta, A, True, decrement, fit.n_iter + 1, loglik)
