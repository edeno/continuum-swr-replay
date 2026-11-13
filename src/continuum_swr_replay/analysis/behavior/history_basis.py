"""The raised-cosine history basis and the projection of a lag matrix onto it.

Pure array primitives with no package dependency, shared by the reward-history
covariates (``covariates.history.reward_history``) and the per-bin rate design
(``replay_models.per_bin_design``), which builds the self- and cross-history
blocks from the same basis.
"""

from __future__ import annotations

import numpy as np

__all__ = ["raised_cosine_history_basis", "reward_basis_from_lags"]


def raised_cosine_history_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Log-spaced raised-cosine self-history basis (Pillow et al. 2008).

    Smooth, overlapping bumps over within-stop lags ``1..n_lags`` used to encode
    the self-history block. Projecting the ``onset_lag{j}`` counts onto these
    ``n_basis`` bumps replaces ``n_lags`` collinear lag dummies that otherwise
    perfectly separate on the refractory bins (coefficients -> -inf) and wreck
    held-out prediction.

    Parameters
    ----------
    n_lags : int
        Number of within-stop lags spanned (the history window in bins).
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        ``B[k, j]`` is the weight of lag ``j+1`` in basis function ``k``.
    """
    x = np.log(np.arange(1, n_lags + 1))
    centers = np.linspace(x.min(), x.max(), n_basis)
    spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
    B = np.zeros((n_basis, n_lags))
    for k, c in enumerate(centers):
        arg = np.clip((x - c) * np.pi / (2 * spacing), -np.pi, np.pi)
        B[k] = 0.5 * (1 + np.cos(arg))
    return B


def reward_basis_from_lags(lags: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a lag matrix onto raised-cosine bumps, normalized per bump.

    Each output is the bump-weighted mean over the *available* (finite) lags,
    ``sum_j B[k, j] * lags[:, j] / sum_j B[k, j]``, so a query with only part of
    the history is still on the reward-fraction scale rather than being shrunk
    toward zero by the missing lags.

    Parameters
    ----------
    lags : np.ndarray, shape (n_query, n_lags)
        Per-lag reward outcomes (0/1), ``NaN`` where the lag is not known.
    basis : np.ndarray, shape (n_basis, n_lags)
        Bump weights; ``basis[k, j]`` is the weight of lag ``j + 1`` in bump
        ``k``.

    Returns
    -------
    np.ndarray, shape (n_query, n_basis)
        Normalized projections; ``NaN`` where no lag in the bump's support is
        finite.
    """
    finite = np.isfinite(lags)
    values = np.where(finite, lags, 0.0)  # (n_query, n_lags)
    num = values @ basis.T  # (n_query, n_basis)
    den = finite.astype(float) @ basis.T  # (n_query, n_basis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out
