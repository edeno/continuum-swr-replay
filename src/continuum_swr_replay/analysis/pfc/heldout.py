"""Blocked held-out deviance, and the null a content-free regressor gives.

The peri-event models are compared by how well they predict spikes they were
not fit on, not by a likelihood ratio. Two things make the usual asymptotics
unusable here: the fits are penalised (the ratio's reference distribution is
not a chi-square at any stated degrees of freedom), and spike counts in
neighbouring 2 ms bins are not independent, so an in-sample deviance drop is
worth much less than its nominal degrees of freedom suggest.

**Folds are blocks, not bins.** A random split of bins would put a bin 2 ms
either side of a held-out bin in the training set, and a model could score well
by interpolating the local rate rather than by predicting the event response.
The fit mask is already a union of contiguous immobility runs, so each *whole*
run is assigned to one fold, greedily to the lightest fold so the folds carry
comparable numbers of bins. The assignment is seeded and recorded.

**The score is the Poisson deviance on the held-out fold**, from a fit on the
other ``K - 1``. The penalty is part of the estimator, so it is applied to the
training fit exactly as in the full-data fit; the score itself is unpenalised,
because it is a prediction error and not an objective.

A held-out improvement has no reference distribution of its own -- it is not a
likelihood ratio -- so what an improvement of a given size means is a separate
question, answered by the null the matched control's re-draws give.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class HeldoutDeviance(NamedTuple):
    """A model's out-of-fold Poisson deviance, per fold and summed.

    Attributes
    ----------
    per_fold : NDArray, shape (n_folds, n_units)
        Deviance of the held-out fold under a fit on the other folds. ``nan``
        for a unit x fold whose training fit produced no usable parameters.
    total : NDArray, shape (n_units,)
        Sum over folds -- the model's held-out deviance. ``nan`` if any fold
        is, so a unit is never scored on a partial set of folds.
    """

    per_fold: NDArray
    total: NDArray


def _contiguous_runs(mask: NDArray) -> NDArray:
    """Half-open ``[start, stop)`` bounds of each contiguous True run."""
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2).astype(np.int64)


def assign_blocked_folds(
    fit_mask: pd.Series | NDArray, n_folds: int, seed: int
) -> NDArray:
    """Assign whole immobility runs to folds, balanced by bin count.

    A run is the unit of assignment because neighbouring bins of one run are
    not independent: splitting a run across folds would let a model score on
    held-out bins it has effectively seen. Runs are visited in a seeded random
    order and each goes to the fold holding the fewest bins so far, so no fold
    can end more than one run's length above any other -- a few percent of a
    fold on a session with hundreds of bouts. The order is random rather than
    longest-first (which balances better still) because the assignment has to
    be a draw: the null the improvements are read against is built from the
    same folds, and a rule with no randomness in it would give every session
    with the same bout lengths the same split.

    Parameters
    ----------
    fit_mask : pandas.Series of bool or NDArray of bool, shape (n_time,)
        The bins the models are fit on. Contiguous True runs are the blocks.
    n_folds : int
        Number of folds; at least 2, and no more than the number of runs.
    seed : int
        Seeds the shuffle. The same mask and seed always give the same folds.

    Returns
    -------
    NDArray of int, shape (n_fit_bins,)
        Fold index in ``[0, n_folds)`` for each fit bin, in time order -- the
        row order of a design restricted to ``fit_mask``.

    Raises
    ------
    ValueError
        If ``n_folds`` is below 2 or above the number of contiguous runs.
    """
    mask = np.asarray(fit_mask, dtype=bool)
    runs = _contiguous_runs(mask)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if len(runs) < n_folds:
        raise ValueError(
            f"the fit mask has {len(runs)} contiguous runs, fewer than the "
            f"{n_folds} folds asked for; a fold would be empty"
        )

    lengths = runs[:, 1] - runs[:, 0]
    order = np.random.default_rng(seed).permutation(len(runs))

    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    fold_of_run = np.empty(len(runs), dtype=np.int64)
    for run_index in order:
        fold = int(np.argmin(fold_sizes))
        fold_of_run[run_index] = fold
        fold_sizes[fold] += int(lengths[run_index])

    folds = np.empty(int(mask.sum()), dtype=np.int64)
    position = 0
    for run_index, length in enumerate(lengths):
        folds[position : position + int(length)] = fold_of_run[run_index]
        position += int(length)
    return folds


def poisson_deviance(counts: NDArray, fitted_mean: NDArray) -> NDArray:
    """Poisson deviance of held-out counts under fitted means.

    ``2 sum_i [y_i log(y_i / mu_i) - (y_i - mu_i)]``, with the ``y log y`` term
    read as zero where ``y`` is zero.

    Parameters
    ----------
    counts : NDArray, shape (n_time, n_units)
        Observed counts on the scored bins.
    fitted_mean : NDArray, shape (n_time, n_units)
        Predicted means on the same bins.

    Returns
    -------
    NDArray, shape (n_units,)
    """
    counts = np.asarray(counts, dtype=float)
    fitted_mean = np.asarray(fitted_mean, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(counts > 0, counts / fitted_mean, 1.0)
        terms = counts * np.log(ratio) - (counts - fitted_mean)
    return 2.0 * terms.sum(axis=0)


def heldout_deviance(
    design: NDArray,
    counts: NDArray,
    penalty: NDArray | None,
    folds: NDArray,
) -> HeldoutDeviance:
    """Out-of-fold Poisson deviance of one design, for every unit at once.

    Parameters
    ----------
    design : NDArray, shape (n_fit_bins, n_features)
        Design restricted to the fit mask, in the same row order as ``folds``.
    counts : NDArray, shape (n_fit_bins, n_units) or (n_fit_bins,)
        Spike counts on those bins.
    penalty : NDArray, shape (n_features,), or None
        The estimator's penalty, applied to each training fit exactly as to the
        full-data fit. The held-out score itself is unpenalised: it is a
        prediction error, not the objective.
    folds : NDArray of int, shape (n_fit_bins,)
        Fold index per bin, from :func:`assign_blocked_folds`.

    Returns
    -------
    HeldoutDeviance

    Raises
    ------
    ValueError
        If the design, the counts and the folds disagree on the number of rows.
    """
    from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

    design = np.asarray(design, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    folds = np.asarray(folds, dtype=np.int64)
    if not (design.shape[0] == counts.shape[0] == folds.shape[0]):
        raise ValueError(
            f"design has {design.shape[0]} rows, counts {counts.shape[0]} and "
            f"folds {folds.shape[0]}; they must agree"
        )

    n_folds = int(folds.max()) + 1
    per_fold = np.full((n_folds, counts.shape[1]), np.nan)
    for fold in range(n_folds):
        held_out = folds == fold
        fitted = fit_poisson_glms(
            design[~held_out], counts[~held_out], penalty=penalty
        )
        fitted_mean = np.exp(design[held_out] @ fitted.params.T)
        per_fold[fold] = poisson_deviance(counts[held_out], fitted_mean)
    return HeldoutDeviance(per_fold=per_fold, total=per_fold.sum(axis=0))
