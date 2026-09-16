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

**The null comes from re-drawing the matched control.** A held-out improvement
has no null distribution to read it against: it is not a likelihood ratio, and
its sampling distribution depends on the autocorrelation of the spike train and
on how the events happen to fall in the folds. The matched control already
defines "a duration-matched non-event regressor drawn from the fitted bouts";
drawing it ``R`` times with different seeds gives, per unit, ``R`` held-out
improvements from a regressor built by the same rules with no event content.
The observed improvement is read against that distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.glm import (
    DEFAULT_MATCHED_CONTROL_ATTEMPTS,
    convolve_event_basis,
    event_onset_counts,
    make_matched_control_times,
)


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


class ControlDrawNull(NamedTuple):
    """Held-out improvements from ``R`` draws of the matched non-event control.

    Attributes
    ----------
    improvement : NDArray, shape (n_draws, n_units)
        ``D_ho(constant) - D_ho(control_r)`` -- what a content-free regressor
        buys, on the same folds and under the same penalty as a real class.
    engagement : NDArray, shape (n_draws, n_units)
        ``D_ho(control) - D_ho(control + control_r)``: the engagement
        statistic's null, with the tested regressor replaced by a content-free
        one and the baseline left as it is. Empty when no baseline control was
        given.
    constant_deviance : NDArray, shape (n_units,)
        ``D_ho(constant)``, the reference every improvement is taken against.
    n_placed : NDArray of int, shape (n_draws,)
        Control windows each draw placed. A draw that placed far fewer than the
        others is a thinner regressor and its improvement is not comparable.
    """

    improvement: NDArray
    engagement: NDArray
    constant_deviance: NDArray
    n_placed: NDArray


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


def empirical_pvalue(observed: NDArray, null: NDArray) -> NDArray:
    """Fraction of null draws at least as large as the observed statistic.

    Parameters
    ----------
    observed : NDArray, shape (n_units,)
        The observed statistic per unit.
    null : NDArray, shape (n_draws, n_units)
        The same statistic from draws with no event content.

    Returns
    -------
    NDArray, shape (n_units,)
        ``mean(null >= observed)`` over the draws that produced a finite value.
        The resolution is one draw, so zero is attainable and a unit with no
        finite draw is ``nan``.
    """
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    usable = np.isfinite(null)
    n_usable = usable.sum(axis=0)
    at_least = (usable & (null >= observed[None, :])).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pvalue = np.where(n_usable > 0, at_least / n_usable, np.nan)
    return np.where(np.isfinite(observed), pvalue, np.nan)


def control_design(
    control_times: pd.DataFrame,
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    *,
    extra_times: pd.DataFrame | None = None,
) -> NDArray:
    """Intercept plus the convolved basis of one (or two) event sets, on the fit bins.

    Parameters
    ----------
    control_times : pandas.DataFrame
        Windows whose ``start_time`` onsets drive the first basis block.
    fit_mask : pandas.Series of bool, shape (n_time,)
        Indexed by the session's sample times; the design is built on the full
        axis and returned on the True bins, as the fitted designs are.
    spline_basis : NDArray, shape (n_lags, n_basis)
    lags : NDArray, shape (n_lags,)
        Lags in samples, positive meaning after onset.
    extra_times : pandas.DataFrame, optional
        A second event set, entered as its own basis block after the first.

    Returns
    -------
    NDArray, shape (n_fit_bins, 1 + n_basis * n_blocks)
    """
    time = np.asarray(fit_mask.index, dtype=float)
    rows = np.asarray(fit_mask, dtype=bool)
    blocks = [np.ones((len(time), 1))]
    for times in (control_times, extra_times):
        if times is None:
            continue
        onsets = event_onset_counts(times, time)
        blocks.append(convolve_event_basis(onsets, spline_basis, lags))
    return np.concatenate(blocks, axis=1)[rows]


def control_draw_null(
    event_times: Mapping[str, pd.DataFrame],
    fit_mask: pd.Series,
    counts: NDArray,
    folds: NDArray,
    *,
    rng_seeds: Sequence[int],
    spline_basis: NDArray,
    lags: NDArray,
    penalty_sigma: float | None = None,
    baseline_control: pd.DataFrame | None = None,
    max_attempts: int = DEFAULT_MATCHED_CONTROL_ATTEMPTS,
) -> ControlDrawNull:
    """Held-out improvements from repeated draws of the matched control.

    Each draw re-runs :func:`~continuum_swr_replay.analysis.pfc.glm.make_matched_control_times`
    with a different seed and the same rules, builds the same design a real
    event class gets, and scores it on the same folds under the same penalty.
    The result is, per unit, what a regressor with no event content buys --
    the distribution an observed improvement is read against.

    Parameters
    ----------
    event_times : Mapping[str, pandas.DataFrame]
        The session's event classes, as the matched-control draw needs them.
    fit_mask : pandas.Series of bool, shape (n_time,)
        The bins the models are fit on, indexed by sample time in seconds.
    counts : NDArray, shape (n_time, n_units)
        Spike counts on the full time axis, one column per unit.
    folds : NDArray of int, shape (n_fit_bins,)
        Fold index per fit bin, from :func:`assign_blocked_folds`.
    rng_seeds : Sequence[int]
        One seed per draw. Recorded by the caller; the number of seeds is ``R``.
    spline_basis : NDArray, shape (n_lags, n_basis)
    lags : NDArray, shape (n_lags,)
        The fit window's lags in samples.
    penalty_sigma : float or None
        Prior standard deviation of a basis coefficient, on the log-rate scale.
        ``None`` scores unpenalised fits.
    baseline_control : pandas.DataFrame, optional
        The session's shipped matched control. Given, each draw also scores
        ``baseline_control + control_r`` against ``baseline_control`` alone --
        the engagement statistic's null, with the tested regressor replaced by
        a content-free one drawn by the same rules. The draws are independent
        of the shipped control and may overlap it, exactly as two real classes
        never do; a caller comparing the two should say so.
    max_attempts : int
        Placement attempts per source event, passed to the draw.

    Returns
    -------
    ControlDrawNull
    """
    from continuum_swr_replay.analysis.pfc.batched_glm import make_ridge_penalty

    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    fit_rows = np.asarray(fit_mask, dtype=bool)
    fit_counts = counts[fit_rows]

    def penalty_for(design: NDArray) -> NDArray | None:
        if penalty_sigma is None:
            return None
        return make_ridge_penalty(design.shape[1], penalty_sigma)

    constant = np.ones((int(fit_rows.sum()), 1))
    constant_deviance = heldout_deviance(constant, fit_counts, None, folds).total

    baseline_deviance = None
    if baseline_control is not None:
        baseline_design = control_design(
            baseline_control, fit_mask, spline_basis, lags
        )
        baseline_deviance = heldout_deviance(
            baseline_design, fit_counts, penalty_for(baseline_design), folds
        ).total

    improvements = []
    engagements = []
    n_placed = []
    for seed in rng_seeds:
        draw = make_matched_control_times(
            event_times,
            fit_mask,
            np.random.default_rng(seed),
            max_attempts=max_attempts,
        )
        n_placed.append(len(draw.times))
        design = control_design(draw.times, fit_mask, spline_basis, lags)
        deviance = heldout_deviance(
            design, fit_counts, penalty_for(design), folds
        ).total
        improvements.append(constant_deviance - deviance)
        if baseline_deviance is not None:
            joint = control_design(
                baseline_control, fit_mask, spline_basis, lags, extra_times=draw.times
            )
            joint_deviance = heldout_deviance(
                joint, fit_counts, penalty_for(joint), folds
            ).total
            engagements.append(baseline_deviance - joint_deviance)

    return ControlDrawNull(
        improvement=np.asarray(improvements),
        engagement=(
            np.asarray(engagements)
            if engagements
            else np.empty((0, counts.shape[1]))
        ),
        constant_deviance=constant_deviance,
        n_placed=np.asarray(n_placed, dtype=int),
    )
