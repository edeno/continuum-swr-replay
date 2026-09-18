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
The fit mask is already a union of contiguous immobility runs. Each whole run
is kept together, and runs close enough for the event kernel's supports to
overlap are joined before assignment; the resulting blocks go greedily to the
lightest fold so the folds carry comparable numbers of bins. The assignment is
seeded and recorded.

**The score is the Poisson deviance on the held-out fold**, from a fit on the
other ``K - 1``. The penalty is part of the estimator, so it is applied to the
training fit exactly as in the full-data fit; the score itself is unpenalised,
because it is a prediction error and not an objective.

**The null comes from re-drawing the matched control, once per event class.**
A held-out improvement has no null distribution to read it against: it is not a
likelihood ratio, and its sampling distribution depends on the autocorrelation
of the spike train and on how the events happen to fall in the folds. The
matched control already defines "a duration-matched non-event regressor drawn
from the fitted bouts"; drawing it ``R`` times with different seeds gives, per
unit, ``R`` held-out improvements from a regressor built by the same rules with
no event content.

That null has to be matched to the class it stands in for. An improvement grows
with the number of onsets a regressor carries and with how long each one is, so
a null drawn from all non-local events -- hundreds of them -- would be a
*harder* comparison for a class with a third as many, and an easier one for a
class with more. Each class therefore gets its own draws: ``n_k`` windows
carrying class ``k``'s own durations, drawn by the same rules from the same
bouts and avoiding the same real events, and class ``k``'s improvement is read
against class ``k``'s draws.

Some real events have no legal non-event window in their own fitted bout. They
cannot enter a matched randomisation test: comparing an all-event observed
regressor with a thinner null changes the estimand. A dedicated, deterministic
placement therefore selects one fixed jointly matchable subset per class,
before spikes are inspected. Both the observed test statistic and every null
draw use exactly that subset. The all-event held-out deviances and differences
remain separate descriptive outputs; the sidecar reports how many events the
matched test excluded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    PreparedHeldoutResponse,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.glm import (
    DEFAULT_MATCHED_CONTROL_ATTEMPTS,
    EVENT_CLASSES,
    MATCHED_CONTROL_SOURCE_CLASS,
    _merge_windows,
    _overlaps_any,
    convolve_event_basis,
    event_onset_counts,
    make_matched_control_times,
)

#: Smallest Monte Carlo sample used for a reported empirical p-value. Below
#: this floor the grid is too coarse even for the descriptive ``p < .05``
#: preview the bundles report. Production requests a larger sample, but this
#: floor also governs deliberately shortened diagnostics and failed fits.
MIN_USABLE_NULL_DRAWS = 20

#: Canonical production Monte Carlo sample. With the plus-one correction this
#: gives a minimum p-value of 1/1001 and about 50 grid points at or below 0.05.
#: Kept with the p-value implementation so runners and figures cannot silently
#: advertise different null sizes.
DEFAULT_CONTROL_DRAWS = 1_000

#: Provenance name for the finite-sample correction used by
#: :func:`empirical_pvalue`. Readers use this to refuse older held-out bundles
#: whose identically named columns used the invalid ``b / R`` estimate.
EMPIRICAL_PVALUE_METHOD = "plus_one"

#: Provenance name for the event population entering the matched-control test.
#: Bundles produced before this policy compared the all-event observed
#: statistic with a sometimes thinner control draw, so readers must not pool
#: them with the corrected fixed-subset estimand.
CONTROL_NULL_EVENT_POLICY = "fixed_jointly_matchable_subset"


def required_usable_null_draws(n_requested: int) -> int:
    """Finite draws required to report a p-value from a requested null.

    Production requires the complete requested Monte Carlo sample, so an
    ``R=1000`` label always means a denominator of 1,000 rather than merely up
    to 1,000. The fixed floor still keeps deliberately shortened diagnostics
    from reporting very coarse p-values.
    """
    n_requested = int(n_requested)
    if n_requested < 0:
        raise ValueError(f"n_requested must be non-negative, got {n_requested}")
    if n_requested == 0:
        return 0
    return max(MIN_USABLE_NULL_DRAWS, n_requested)


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
    improvement : dict[str, NDArray], each of shape (n_draws, n_units)
        Per event class, ``D_ho(constant) - D_ho(control_r)`` -- what a
        regressor with that class's own onset count and durations, and no event
        content, buys on the same folds under the same penalty. Read class
        ``k``'s observed improvement against ``improvement[k]``.
    observed_improvement : dict[str, NDArray], each of shape (n_units,)
        Observed class improvement computed from the same fixed matchable
        source events used by that class's null draws. This can differ from the
        separately reported all-event held-out difference when an event has no
        legal within-bout match.
    engagement : NDArray, shape (n_draws, n_units)
        ``D_ho(control) - D_ho(control + control_r)``: the engagement
        statistic's null, with the tested regressor replaced by a content-free
        one matched to it and the baseline left as it is. Empty when no
        baseline control was given.
    observed_engagement : NDArray, shape (n_units,)
        Engagement improvement with the tested real regressor restricted to
        the same fixed source-event subset as ``engagement``. Empty when no
        baseline control was given.
    constant_deviance : NDArray, shape (n_units,)
        ``D_ho(constant)``, the reference every improvement is taken against.
    n_placed : dict[str, NDArray of int], each of shape (n_draws,)
        Windows each draw placed, per class, against ``n_tested`` below. A draw
        that placed far fewer is a thinner regressor and its improvement is not
        comparable.
    n_events : dict[str, int]
        Total real onsets of each class.
    n_tested : dict[str, int]
        Real onsets in the class's fixed jointly matchable subset -- what each
        observed test statistic and every null draw is matched to.
    tested_source_indices : dict[str, NDArray of int]
        Positional indices of the real events in each fixed tested subset.
    engagement_overlap : NDArray, shape (n_draws,)
        Fraction of each engagement draw's windows that overlap the shipped
        baseline control's. The draws are independent of that control, so they
        may land on it, which two real event classes never do; a large fraction
        would mean the null's added regressor duplicates the baseline it is
        entered beside and buys less than a content-free regressor should.
        Empty when no baseline control was given.
    """

    improvement: dict[str, NDArray]
    observed_improvement: dict[str, NDArray]
    engagement: NDArray
    observed_engagement: NDArray
    constant_deviance: NDArray
    n_placed: dict[str, NDArray]
    n_events: dict[str, int]
    n_tested: dict[str, int]
    tested_source_indices: dict[str, NDArray]
    engagement_overlap: NDArray


def _contiguous_runs(mask: NDArray) -> NDArray:
    """Half-open ``[start, stop)`` bounds of each contiguous True run."""
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2).astype(np.int64)


def assign_blocked_folds(
    fit_mask: pd.Series | NDArray,
    n_folds: int,
    seed: int,
    *,
    support_radius_bins: int = 0,
) -> NDArray:
    """Assign support-separated immobility blocks to folds, balanced by size.

    A run is the unit of assignment because neighbouring bins of one run are
    not independent: splitting a run across folds would let a model score on
    held-out bins it has effectively seen. When ``support_radius_bins`` is
    positive, runs separated by no more than twice that radius are first
    joined into one block. This prevents a symmetric peri-event kernel from
    covering training bins in one run and held-out bins in another across a
    short False gap. Blocks are visited in a seeded random
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
    support_radius_bins : int
        Half-width of the widest time-local regressor, in bins. Nearby runs
        whose expanded supports overlap are assigned together. Zero preserves
        the ordinary contiguous-run definition.

    Returns
    -------
    NDArray of int, shape (n_fit_bins,)
        Fold index in ``[0, n_folds)`` for each fit bin, in time order -- the
        row order of a design restricted to ``fit_mask``.

    Raises
    ------
    ValueError
        If ``n_folds`` is below 2 or above the number of support-separated
        blocks, or if ``support_radius_bins`` is negative.
    """
    mask = np.asarray(fit_mask, dtype=bool)
    runs = _contiguous_runs(mask)
    if support_radius_bins < 0:
        raise ValueError(
            f"support_radius_bins must be non-negative, got {support_radius_bins}"
        )
    if len(runs) and support_radius_bins:
        joined: list[list[int]] = [[int(runs[0, 0]), int(runs[0, 1])]]
        maximum_gap = 2 * int(support_radius_bins)
        for start, stop in runs[1:]:
            if int(start) - joined[-1][1] <= maximum_gap:
                joined[-1][1] = int(stop)
            else:
                joined.append([int(start), int(stop)])
        blocks = np.asarray(joined, dtype=np.int64)
    else:
        blocks = runs
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if len(blocks) < n_folds:
        raise ValueError(
            f"the fit mask has {len(blocks)} support-separated blocks, fewer "
            f"than the {n_folds} folds asked for; a fold would be empty"
        )

    lengths = np.asarray(
        [int(mask[start:stop].sum()) for start, stop in blocks], dtype=np.int64
    )
    order = np.random.default_rng(seed).permutation(len(blocks))

    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    fold_of_block = np.empty(len(blocks), dtype=np.int64)
    for block_index in order:
        fold = int(np.argmin(fold_sizes))
        fold_of_block[block_index] = fold
        fold_sizes[fold] += int(lengths[block_index])

    folds = np.empty(int(mask.sum()), dtype=np.int64)
    position = 0
    for block_index, length in enumerate(lengths):
        stop = position + int(length)
        folds[position:stop] = fold_of_block[block_index]
        position = stop
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

    This is the reference implementation the batch's scorer is held to: one
    plain fold-by-fold loop over
    :func:`~continuum_swr_replay.analysis.pfc.batched_glm.fit_poisson_glms`,
    with no batching, no device residency and no shape padding. The
    production path
    (:func:`~continuum_swr_replay.analysis.pfc.batched_glm.score_poisson_designs_heldout`)
    must reproduce it to 1e-8 on a production-shaped design, on
    session-shaped folds and on fits that hit the iteration cap; where the two
    disagree, this one is right.

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
    from continuum_swr_replay.analysis.pfc.batched_glm import (
        fit_poisson_glms,
        poisson_deviance_from_params,
    )

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
    labels = np.unique(folds)
    if labels.size < 2 or not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError(
            "folds must use every consecutive label from 0 and contain at "
            f"least two folds; got {labels.tolist()}"
        )

    n_folds = int(labels.size)
    per_fold = np.full((n_folds, counts.shape[1]), np.nan)
    for fold in range(n_folds):
        held_out = folds == fold
        fitted = fit_poisson_glms(
            design[~held_out],
            counts[~held_out],
            penalty=penalty,
            compute_covariance=False,
        )
        score = poisson_deviance_from_params(
            design[held_out], counts[held_out], fitted.params
        )
        usable = np.asarray(fitted.converged, dtype=bool) & np.all(
            np.isfinite(fitted.params), axis=1
        )
        usable &= np.isfinite(score)
        per_fold[fold] = np.where(usable, score, np.nan)
    return HeldoutDeviance(per_fold=per_fold, total=per_fold.sum(axis=0))


def empirical_pvalue(
    observed: NDArray,
    null: NDArray,
    *,
    min_usable_draws: int = MIN_USABLE_NULL_DRAWS,
) -> NDArray:
    """Finite-sample-valid Monte Carlo p-value for an observed statistic.

    Parameters
    ----------
    observed : NDArray, shape (n_units,)
        The observed statistic per unit.
    null : NDArray, shape (n_draws, n_units)
        The same statistic from draws with no event content.
    min_usable_draws : int
        Minimum finite draws required per unit. Fewer draws return ``nan``;
        production passes the complete requested sample from
        :func:`required_usable_null_draws`.

    Returns
    -------
    NDArray, shape (n_units,)
        ``(1 + count(null >= observed)) / (1 + n_usable)``. The plus-one
        correction treats the observed statistic as one exchangeable draw and
        prevents an impossible zero p-value. A unit below the usable-draw floor
        is ``nan``.
    """
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError(
            "observed must have shape (n_units,) and null "
            f"(n_draws, n_units); got {observed.shape} and {null.shape}"
        )
    if min_usable_draws < 1:
        raise ValueError(f"min_usable_draws must be at least 1, got {min_usable_draws}")
    usable = np.isfinite(null)
    n_usable = usable.sum(axis=0)
    at_least = (usable & (null >= observed[None, :])).sum(axis=0)
    pvalue = np.where(
        n_usable >= min_usable_draws,
        (at_least + 1) / (n_usable + 1),
        np.nan,
    )
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


def null_draw_seeds(control_seed: int, n_draws: int) -> NDArray:
    """Seeds for the null draws, derived from the session's own control seed.

    The null has to be a different draw from the control the models were fit
    with -- a draw that reproduced it would score the shipped regressor against
    itself and return a p-value of 1 by construction. Deriving the seeds from
    the session's seed (rather than from a counter) keeps a session's null
    reproducible on its own, and hashing seed and draw index together makes a
    collision with the session seed, or between draws, something that is
    checked rather than assumed.

    Parameters
    ----------
    control_seed : int
        The session's matched-control seed, from
        :func:`~continuum_swr_replay.analysis.pfc.glm.matched_control_seed`.
    n_draws : int
        Number of draws, ``R``.

    Returns
    -------
    NDArray of int, shape (n_draws,)
        Distinct, and none equal to ``control_seed``.

    Raises
    ------
    ValueError
        If ``n_draws`` is negative, or if the derived seeds are not distinct
        and different from ``control_seed`` (which would make a draw a repeat
        of another, or of the shipped control).
    """
    if n_draws < 0:
        raise ValueError(f"n_draws must be non-negative, got {n_draws}")
    seeds = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{int(control_seed)}|{draw}".encode()).digest()[:8],
                "little",
            )
            for draw in range(n_draws)
        ],
        dtype=np.uint64,
    )
    if len(set(seeds.tolist())) != n_draws or int(control_seed) in seeds.tolist():
        raise ValueError(
            "the derived null seeds are not distinct from each other and from "
            "the session's control seed"
        )
    return seeds


def _matchable_subset_seed(control_seed: int, class_name: str) -> int:
    """A class-specific seed for selecting the fixed tested event subset."""
    return int.from_bytes(
        hashlib.sha256(
            f"{int(control_seed)}|matchable_subset|{class_name}".encode()
        ).digest()[:8],
        "little",
    )


def overlapping_fraction(
    windows: pd.DataFrame, starts: NDArray, ends: NDArray
) -> float:
    """Fraction of ``windows`` that overlap any of ``[starts, ends)``.

    The null's draws are independent of the shipped matched control, so a draw
    may place a window where the control already sits -- something two real
    event classes never do, because each class's windows avoid the others'.
    Blocking it would mean drawing the null by different rules from the control
    it stands in for, which is the worse trade; measuring it is the cost of
    that choice, and the caller reports it.

    Parameters
    ----------
    windows : pandas.DataFrame
        ``start_time`` / ``end_time`` of the drawn windows.
    starts, ends : NDArray, shape (n_windows,)
        Ascending, non-overlapping bounds to test against, as
        :func:`~continuum_swr_replay.analysis.pfc.glm._merge_windows` returns
        them.

    Returns
    -------
    float
        ``nan`` when no window was drawn.
    """
    if not len(windows):
        return float("nan")
    hits = sum(
        _overlaps_any(float(start), float(end), starts, ends)
        for start, end in zip(
            windows["start_time"].to_numpy(dtype=float),
            windows["end_time"].to_numpy(dtype=float),
            strict=True,
        )
    )
    return float(hits / len(windows))


def control_draw_null(
    event_times: Mapping[str, pd.DataFrame],
    fit_mask: pd.Series,
    counts: NDArray | None,
    folds: NDArray | None,
    *,
    rng_seeds: Sequence[int],
    subset_seed: int = 0,
    spline_basis: NDArray,
    lags: NDArray,
    penalty_sigma: float | None = None,
    classes: Sequence[str] = EVENT_CLASSES,
    baseline_control: pd.DataFrame | None = None,
    engagement_class: str = "non_local_no_SWR",
    max_attempts: int = DEFAULT_MATCHED_CONTROL_ATTEMPTS,
    model_deviance: Mapping[str, NDArray] | None = None,
    response: PreparedHeldoutResponse | None = None,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
) -> ControlDrawNull:
    """Held-out improvements from repeated matched-control draws, per class.

    For each event class, a dedicated seed first selects one fixed subset of
    real events for which a joint within-bout control placement exists. The
    selection uses event timing and duration only, never spikes. The observed
    statistic and every null draw then use exactly that subset. This is
    necessary when a real event has no legal non-event window in its own bout:
    an all-event observed regressor and a thinner control regressor are not the
    same statistic.

    For each event class and each draw seed, this then re-runs
    :func:`~continuum_swr_replay.analysis.pfc.glm.make_matched_control_times`
    with **that class's tested events as the source**, so the draw carries the
    tested onset count and durations, drawn from the same fitted bouts and
    avoiding all real events (every class's windows, including the ones being
    stood in for). The draw is turned into the design a real class gets and
    scored on the same folds under the same penalty, giving per unit what a
    regressor of that size and shape with no event content buys.

    Parameters
    ----------
    event_times : Mapping[str, pandas.DataFrame]
        The session's event classes, as the matched-control draw needs them:
        the four of
        :data:`~continuum_swr_replay.analysis.pfc.glm.EVENT_CLASSES`, each with
        ``start_time`` / ``end_time``.
    fit_mask : pandas.Series of bool, shape (n_time,)
        The bins the models are fit on, indexed by sample time in seconds.
    counts : NDArray, shape (n_time, n_units), optional
        Spike counts on the full time axis, one column per unit. Supply this
        together with ``folds`` only when ``response`` is omitted.
    folds : NDArray of int, shape (n_fit_bins,), optional
        Fold index per fit bin, from :func:`assign_blocked_folds`. Supply this
        together with ``counts`` only when ``response`` is omitted.
    rng_seeds : Sequence[int]
        One seed per draw, ``R`` of them; see :func:`null_draw_seeds`. Each
        class's draw ``r`` is seeded from the same value, so the classes share
        a draw index but not a window -- their source events differ.
    subset_seed : int
        Session seed from which a distinct selection seed is derived for each
        class. This selection is independent of ``rng_seeds`` and is recorded
        by the caller as part of the control-draw provenance.
    spline_basis : NDArray, shape (n_lags, n_basis)
    lags : NDArray, shape (n_lags,)
        The fit window's lags in samples.
    penalty_sigma : float or None
        Prior standard deviation of a basis coefficient, on the log-rate scale.
        ``None`` scores unpenalised fits.
    classes : Sequence[str]
        Event classes to build a null for.
    baseline_control : pandas.DataFrame, optional
        The session's shipped matched control. Given, each draw also scores
        ``baseline_control + control_r`` against ``baseline_control`` alone --
        the engagement statistic's null, with the tested regressor replaced by
        a draw matched to ``engagement_class`` (the class the real statistic
        tests) and the baseline left as it is. The draws are independent of the
        shipped control and may overlap it, which two real classes never do; a
        caller comparing the two should say so.
    engagement_class : str
        Class the engagement null's added regressor is matched to.
    max_attempts : int
        Whole-draw placement attempts for each null replicate. The fixed
        subset selection always uses the production default so setting this to
        zero can test incomplete-draw handling without selecting an empty
        estimand.
    model_deviance : mapping, optional
        Already computed all-event held-out deviances, keyed by model name.
        When a tested subset contains every event, these values avoid refitting
        the identical observed model. A proper subset is always rescored.
    response : PreparedHeldoutResponse, optional
        Device-resident fit-bin counts and folds. Supplying the response used
        for the session's all-event scores avoids retransferring it for every
        null draw. When supplied, ``counts`` and ``folds`` must both be
        ``None``: accepting redundant same-shaped inputs would let a response
        prepared from another dataset or fold split silently win. When
        omitted, the response is prepared once here from the raw inputs.
    design_batch_size : int
        Maximum number of equal-width null designs generated and scored
        together. Draws are discarded after each batch, so memory use stays
        bounded as ``len(rng_seeds)`` grows.

    Returns
    -------
    ControlDrawNull

    Raises
    ------
    ValueError
        If ``rng_seeds`` holds a repeat -- two draws from one seed are one
        draw counted twice.
    KeyError
        If a requested class is not in ``event_times``.
    """
    from continuum_swr_replay.analysis.pfc.batched_glm import make_ridge_penalty

    seeds = [int(seed) for seed in rng_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("rng_seeds must be distinct; a repeated seed repeats a draw")

    fit_rows = np.asarray(fit_mask, dtype=bool)
    n_fit_rows = int(fit_rows.sum())
    if design_batch_size < 1:
        raise ValueError(f"design_batch_size must be positive, got {design_batch_size}")
    if response is None:
        if counts is None or folds is None:
            raise ValueError(
                "supply either raw counts and folds or response; raw mode needs both"
            )
        counts_array = np.asarray(counts, dtype=float)
        if counts_array.ndim == 1:
            counts_array = counts_array[:, None]
        response = prepare_heldout_response(counts_array[fit_rows], folds)
    elif counts is not None or folds is not None:
        raise ValueError(
            "supply either raw counts and folds or response, not both; redundant "
            "same-shaped inputs cannot establish response identity"
        )
    elif response.n_time != n_fit_rows:
        raise ValueError(
            "the prepared response does not match the null's fit mask: "
            f"response has {response.n_time} rows, mask has {n_fit_rows}"
        )
    n_units = response.n_units

    def penalty_for(design: NDArray) -> NDArray | None:
        if penalty_sigma is None:
            return None
        return make_ridge_penalty(design.shape[1], penalty_sigma)

    def score_design_batch(designs: NDArray) -> NDArray:
        """Score one bounded host batch and return only per-design totals."""
        designs = np.asarray(designs, dtype=float)
        if designs.ndim != 3 or designs.shape[0] == 0:
            raise ValueError(
                "a score batch must have shape "
                f"(n_designs, n_time, n_features), got {designs.shape}"
            )
        return score_poisson_designs_heldout(
            designs,
            response,
            penalty_for(designs[0]),
            design_batch_size=design_batch_size,
        ).total

    selection_classes = list(classes)
    if baseline_control is not None and engagement_class not in selection_classes:
        selection_classes.append(engagement_class)

    n_events = {name: int(len(event_times[name])) for name in selection_classes}
    tested_source_indices: dict[str, NDArray] = {}
    tested_events: dict[str, pd.DataFrame] = {}
    for class_name in selection_classes:
        source = {
            **event_times,
            MATCHED_CONTROL_SOURCE_CLASS: event_times[class_name],
        }
        selection = make_matched_control_times(
            source,
            fit_mask,
            np.random.default_rng(_matchable_subset_seed(subset_seed, class_name)),
            # Keep the estimand independent of a caller deliberately reducing
            # attempts to exercise incomplete-null handling.
            max_attempts=DEFAULT_MATCHED_CONTROL_ATTEMPTS,
        )
        tested_source_indices[class_name] = selection.source_indices
        tested_events[class_name] = (
            event_times[class_name]
            .iloc[selection.source_indices]
            .reset_index(drop=True)
        )
    n_tested = {name: int(len(events)) for name, events in tested_events.items()}

    def draw_for(class_name: str, seed: int) -> pd.DataFrame:
        """One control draw matched to the class's fixed tested subset.

        The source slot is swapped for this class's selected events; the
        blocked windows are still the union over every original class, which
        covers the real ``non_local`` set either way (it is the union of its
        two subclasses).
        """
        source = {
            **event_times,
            MATCHED_CONTROL_SOURCE_CLASS: tested_events[class_name],
        }
        return make_matched_control_times(
            source,
            fit_mask,
            np.random.default_rng(seed),
            max_attempts=max_attempts,
        ).times

    if model_deviance is not None and "constant" in model_deviance:
        constant_deviance = np.asarray(model_deviance["constant"], dtype=float)
    else:
        constant = np.ones((int(fit_rows.sum()), 1))
        constant_deviance = score_design_batch(constant[None, :, :])[0]

    observed_improvement: dict[str, NDArray] = {}
    for class_name in classes:
        if n_tested[class_name] == 0:
            observed_improvement[class_name] = np.full(n_units, np.nan)
            continue
        if (
            n_tested[class_name] == n_events[class_name]
            and model_deviance is not None
            and class_name in model_deviance
        ):
            observed_improvement[class_name] = constant_deviance - np.asarray(
                model_deviance[class_name], dtype=float
            )
            continue
        observed_design = control_design(
            tested_events[class_name], fit_mask, spline_basis, lags
        )
        observed_deviance = score_design_batch(observed_design[None, :, :])[0]
        observed_improvement[class_name] = constant_deviance - observed_deviance

    engagement = np.empty((0, n_units))
    engagement_overlap = np.empty(0)
    observed_engagement = np.empty(0)
    baseline_design = None
    baseline_deviance = None
    baseline_starts = np.empty(0)
    baseline_ends = np.empty(0)
    if baseline_control is not None:
        baseline_design = control_design(baseline_control, fit_mask, spline_basis, lags)
        if model_deviance is not None and "matched_control" in model_deviance:
            baseline_deviance = np.asarray(
                model_deviance["matched_control"], dtype=float
            )
        else:
            baseline_deviance = score_design_batch(baseline_design[None, :, :])[0]
        baseline_starts, baseline_ends = _merge_windows(
            baseline_control["start_time"].to_numpy(dtype=float),
            baseline_control["end_time"].to_numpy(dtype=float),
        )
        full_engagement_name = f"matched_control + {engagement_class}"
        if n_tested[engagement_class] == 0:
            observed_engagement = np.full(n_units, np.nan)
        elif (
            n_tested[engagement_class] == n_events[engagement_class]
            and model_deviance is not None
            and full_engagement_name in model_deviance
        ):
            observed_engagement = baseline_deviance - np.asarray(
                model_deviance[full_engagement_name], dtype=float
            )
        else:
            tested_design = control_design(
                tested_events[engagement_class], fit_mask, spline_basis, lags
            )
            observed_joint = np.concatenate(
                [baseline_design, tested_design[:, 1:]], axis=1
            )
            observed_joint_deviance = score_design_batch(observed_joint[None, :, :])[0]
            observed_engagement = baseline_deviance - observed_joint_deviance
        engagement = np.full((len(seeds), n_units), np.nan)
        engagement_overlap = np.full(len(seeds), np.nan)

    def score_draw_stream(
        class_name: str, *, score_improvement: bool
    ) -> tuple[NDArray, NDArray]:
        """Build, score and release fixed-size batches for one event class."""
        per_draw = np.full((len(seeds), n_units), np.nan)
        placed = np.zeros(len(seeds), dtype=int)
        score_engagement = (
            baseline_control is not None and class_name == engagement_class
        )

        for batch_start in range(0, len(seeds), design_batch_size):
            batch_stop = min(batch_start + design_batch_size, len(seeds))
            exact_indices: list[int] = []
            exact_designs: list[NDArray] = []
            for draw_index in range(batch_start, batch_stop):
                times = draw_for(class_name, seeds[draw_index])
                placed[draw_index] = len(times)
                if score_engagement:
                    engagement_overlap[draw_index] = overlapping_fraction(
                        times, baseline_starts, baseline_ends
                    )
                if len(times) != n_tested[class_name] or n_tested[class_name] == 0:
                    continue
                exact_indices.append(draw_index)
                exact_designs.append(
                    control_design(times, fit_mask, spline_basis, lags)
                )

            if not exact_designs:
                continue
            design_batch = np.stack(exact_designs)
            del exact_designs
            if score_improvement:
                draw_deviance = score_design_batch(design_batch)
                per_draw[exact_indices] = constant_deviance - draw_deviance

            if score_engagement:
                if baseline_design is None or baseline_deviance is None:
                    raise RuntimeError("engagement scoring needs a baseline control")
                # Both designs have the same intercept. Reuse the baseline and
                # draw blocks, but materialize only this small batch of joints.
                joint_batch = np.empty(
                    (
                        design_batch.shape[0],
                        design_batch.shape[1],
                        baseline_design.shape[1] + design_batch.shape[2] - 1,
                    ),
                    dtype=float,
                )
                joint_batch[:, :, : baseline_design.shape[1]] = baseline_design
                joint_batch[:, :, baseline_design.shape[1] :] = design_batch[:, :, 1:]
                joint_deviance = score_design_batch(joint_batch)
                engagement[exact_indices] = baseline_deviance - joint_deviance

        return per_draw, placed

    improvement: dict[str, NDArray] = {}
    n_placed: dict[str, NDArray] = {}
    for class_name in classes:
        per_draw, placed = score_draw_stream(class_name, score_improvement=True)
        improvement[class_name] = per_draw
        n_placed[class_name] = placed

    # A custom ``classes`` subset may omit the engagement class. It still gets
    # its own streamed designs, but the production/default path above reuses
    # the design while scoring that class and never retains an R-sized cache.
    if baseline_control is not None and engagement_class not in classes:
        score_draw_stream(engagement_class, score_improvement=False)

    return ControlDrawNull(
        improvement=improvement,
        observed_improvement=observed_improvement,
        engagement=engagement,
        observed_engagement=observed_engagement,
        constant_deviance=constant_deviance,
        n_placed=n_placed,
        n_events=n_events,
        n_tested=n_tested,
        tested_source_indices=tested_source_indices,
        engagement_overlap=engagement_overlap,
    )
