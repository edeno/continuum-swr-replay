"""The common spike-evidence diagnostic, read on the decode grid for every inventory.

Each grid row's spikes are scored by the decoder's prior-excluded clusterless
log-likelihood ``L_k(x)`` over the real place bins: the library's
``predict_clusterless_kde_log_likelihood`` (``is_local=False``) on the encoding
model the decode saved. A row's **evidence** is

    max over remote usable bins x of L_k(x)  -  L_k(the animal's bin)

A bin is remote when its track-graph distance from the animal's bin is above
``remote_cm``, and usable when a CPU rebuild of the likelihood is the decode
there (not ``subnormal_occupancy_bin_mask``). Above zero, some remote place
explains the row's spikes better than the animal's own place does. A silent
row is scored too: the ground-process term is likelihood as well. A row has no
evidence (NaN) when the animal's position is not finite, its bin is flagged,
or no usable bin is remote.

Row ``k`` holds the spikes with ``t_k <= s < t_{k + 1}``, as in
``clusterless_epoch_log_likelihood``, whatever rows are computed together. The
grid's last sample holds no span of spikes and is never read.

The likelihood reads no position (``is_local=False``) and no posterior, so the
evidence of an interval depends only on its rows' spikes and the animal's bins.
"""

from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    _drop_the_unreachable_row,
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence.terms import (
    interior_bin_mask,
    nearest_interior_bins,
    subnormal_occupancy_bin_mask,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    RowTraces,
    _over_rows,
    interval_rows,
)

__all__ = [
    "BLOCK_ROWS",
    "animal_bins",
    "epoch_row_evidence",
    "interval_evidence",
    "log_likelihood_blocks",
    "remote_evidence",
]

BLOCK_ROWS = 25_000
"""Rows per library call at most (50 s at 500 Hz); bounds the memory of a long run."""


def _row_blocks(rows: NDArray[np.bool_], block_rows: int) -> list[tuple[int, int]]:
    """``(first, stop)`` of each block: the runs of ``rows`` cut every ``block_rows``."""
    edges = np.diff(np.r_[0, rows.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        (first, min(first + block_rows, stop))
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
        for first in range(start, stop, block_rows)
    ]


def log_likelihood_blocks(
    time: NDArray,
    rows: NDArray,
    linear_position: NDArray,
    spike_times: Sequence[NDArray],
    spike_waveform_features: Sequence[NDArray],
    encargs: dict,
    *,
    block_rows: int = BLOCK_ROWS,
) -> Iterator[tuple[int, NDArray]]:
    """The log-likelihood of the requested rows, one block at a time.

    Each block is one library call on its rows padded by one sample before and
    two after (fewer at the grid's ends), the pads dropped. Row ``k`` then
    holds exactly the spikes in ``[t_k, t_{k + 1})``, as in one call on the
    whole grid, wherever the blocks are cut.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The decode grid's sample times in seconds.
    rows : NDArray of bool, shape (n_time,)
        The rows to compute.
    linear_position : NDArray, shape (n_time,)
        Passed to the library, which does not read it for this likelihood.
    spike_times : sequence of NDArray, each shape (n_spikes,)
        Spike times in seconds, one array per tetrode.
    spike_waveform_features : sequence of NDArray, each shape (n_spikes, n_features)
        Aligned with ``spike_times``.
    encargs : dict
        The likelihood's arguments, ``encoding_inputs(...)["encargs"]``.
    block_rows : int, optional
        Rows per call at most.

    Yields
    ------
    first_row : int
        The block's first row.
    log_likelihood : NDArray, shape (n_block_rows, n_interior_bins)
        Rows ``first_row`` onward, on the real place bins.

    Raises
    ------
    ValueError
        The grid's last sample is requested, or a row's likelihood is not
        finite.
    """
    from non_local_detector.likelihoods.clusterless_kde import (
        predict_clusterless_kde_log_likelihood,
    )

    time = np.asarray(time, dtype=np.float64)
    rows = np.asarray(rows, dtype=bool)
    if rows.shape != time.shape:
        raise ValueError(f"rows has shape {rows.shape}; the grid {time.shape}.")
    if rows[-1]:
        raise ValueError(
            "The grid's last sample holds no span of spikes and is never read."
        )
    position = np.asarray(linear_position, dtype=np.float64).reshape(-1, 1)
    for first, stop in _row_blocks(rows, block_rows):
        low, high = max(first - 1, 0), min(stop + 2, time.size)
        grid = time[low:high]
        # the library's own cut, applied first so only the block's spikes are sent
        in_grid = [(t >= grid[0]) & (t <= grid[-1]) for t in spike_times]
        log_likelihood = predict_clusterless_kde_log_likelihood(
            cast(Any, grid),
            cast(Any, grid),
            cast(Any, position[low:high]),
            [np.asarray(t)[k] for t, k in zip(spike_times, in_grid, strict=True)],
            [
                np.asarray(f)[k]
                for f, k in zip(spike_waveform_features, in_grid, strict=True)
            ],
            **encargs,
            is_local=False,
            disable_progress_bar=True,
        )
        # the call's last row holds no span; the pads hold other rows' spikes
        reachable = _drop_the_unreachable_row(
            np.asarray(log_likelihood), grid, encargs
        ).log_likelihood
        yield first, reachable[first - low : stop - low]


def animal_bins(environment: Any, linear_position: NDArray) -> NDArray[np.int64]:
    """The real place bin nearest the animal on the track graph, -1 where unknown.

    Parameters
    ----------
    environment : non_local_detector.environment.Environment
        Fitted.
    linear_position : NDArray, shape (n_rows,)
        The animal's linear position in cm, NaN where not tracked.

    Returns
    -------
    animal_bin : NDArray[np.int64], shape (n_rows,)
        Index into the real place bins; -1 where the position is not finite.
    """
    position = np.asarray(linear_position, dtype=np.float64)
    known = np.isfinite(position)
    bins = np.full(position.shape, -1, dtype=np.int64)
    if known.any():
        bins[known] = nearest_interior_bins(environment, position[known])
    return bins


def remote_evidence(
    log_likelihood: NDArray,
    animal_bin: NDArray,
    remote: NDArray,
    usable: NDArray,
) -> NDArray[np.float64]:
    """Per row, the best remote usable bin's log-likelihood less the animal's.

    Parameters
    ----------
    log_likelihood : NDArray, shape (n_rows, n_bins)
    animal_bin : NDArray of int, shape (n_rows,)
        The animal's bin, -1 where its position is not finite.
    remote : NDArray of bool, shape (n_bins, n_bins)
        ``remote[i, j]``: bin ``j`` is remote from bin ``i``.
    usable : NDArray of bool, shape (n_bins,)
        The bins whose likelihood is the decode's.

    Returns
    -------
    evidence : NDArray, shape (n_rows,)
        NaN where the animal's bin is unknown or not usable, or no usable bin
        is remote from it.
    """
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    animal_bin = np.asarray(animal_bin, dtype=np.int64)
    usable = np.asarray(usable, dtype=bool)
    known = animal_bin >= 0
    at = np.where(known, animal_bin, 0)  # -1 would index the last bin
    candidates = np.asarray(remote, dtype=bool)[at] & usable
    best = np.where(candidates, log_likelihood, -np.inf).max(axis=1)
    at_animal = log_likelihood[np.arange(at.size), at]
    valid = known & usable[at] & candidates.any(axis=1)
    evidence: NDArray[np.float64] = np.where(valid, best - at_animal, np.nan)
    return evidence


def epoch_row_evidence(
    time: NDArray,
    rows: NDArray,
    linear_position: NDArray,
    spike_times: Sequence[NDArray],
    spike_waveform_features: Sequence[NDArray],
    encargs: dict,
    *,
    remote_cm: float,
    block_rows: int = BLOCK_ROWS,
) -> NDArray[np.float64]:
    """The evidence of each requested row of one epoch.

    Parameters
    ----------
    time, rows, linear_position, spike_times, spike_waveform_features, encargs
        As :func:`log_likelihood_blocks`; ``linear_position`` (cm) also gives
        the animal's bin.
    remote_cm : float
        A bin is remote when its track distance from the animal's is above this.
    block_rows : int, optional

    Returns
    -------
    evidence : NDArray, shape (n_time,)
        :func:`remote_evidence` on the requested rows; NaN on every other row.

    Raises
    ------
    ValueError
        The track-interior bins are not exactly the non-junction bins
        (``interior_bin_mask``), or the likelihood's columns are not the real
        bins.
    """
    environment = encargs["environment"]
    n_real = int(interior_bin_mask(environment).sum())
    usable = ~subnormal_occupancy_bin_mask(encargs["occupancy"], environment)
    remote = interior_bin_distances_cm(environment) > remote_cm
    rows = np.asarray(rows, dtype=bool)
    position = np.asarray(linear_position, dtype=np.float64)
    bins = np.full(rows.shape, -1, dtype=np.int64)
    bins[rows] = animal_bins(environment, position[rows])
    evidence = np.full(rows.shape, np.nan)
    for first, block in log_likelihood_blocks(
        time,
        rows,
        position,
        spike_times,
        spike_waveform_features,
        encargs,
        block_rows=block_rows,
    ):
        if block.shape[1] != n_real:
            raise ValueError(
                f"The likelihood has {block.shape[1]} columns; {n_real} real bins."
            )
        span = slice(first, first + block.shape[0])
        evidence[span] = remote_evidence(block, bins[span], remote, usable)
    return evidence


def interval_evidence(
    traces: RowTraces, evidence: NDArray, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Each interval's evidence over its rows, and how many rows were scored.

    Parameters
    ----------
    traces : RowTraces
        The epoch's grid.
    evidence : NDArray, shape (n_time,)
        :func:`epoch_row_evidence`, computed on every row of ``intervals``.
    intervals : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of each interval, in seconds.

    Returns
    -------
    scores : pd.DataFrame
        On ``intervals``' index: ``evidence_dll``, the mean over the rows with
        evidence (NaN when none has); ``evidence_n_valid_rows`` (Int64, NA
        with no row); ``evidence_valid_fraction``, valid rows over ``n_rows``
        (NaN with no row); and ``evidence_unscorable``, ``"no_row"``,
        ``"no_valid_row"`` or ``""``.

    Raises
    ------
    ValueError
        ``evidence`` is not on the grid, or an interval's rows reach the grid's
        last sample.
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.shape != traces.time.shape:
        raise ValueError(
            f"evidence has shape {evidence.shape}; the grid {traces.time.shape}."
        )
    first, n_rows = interval_rows(
        traces.time,
        intervals["event_start_s"].to_numpy(dtype=np.float64),
        intervals["event_end_s"].to_numpy(dtype=np.float64),
    )
    has_rows = n_rows > 0
    if (has_rows & (first + n_rows >= traces.time.size)).any():
        raise ValueError(
            "An interval reaches the grid's last sample, which holds no span of "
            "spikes and is never read."
        )
    valid = np.isfinite(evidence)
    n_valid = np.zeros(first.shape, dtype=np.int64)
    n_valid[has_rows] = _over_rows(np.add, valid.astype(np.int64), first, n_rows)
    total = np.zeros(first.shape)
    total[has_rows] = _over_rows(np.add, np.where(valid, evidence, 0.0), first, n_rows)
    scored = n_valid > 0
    dll = np.full(first.shape, np.nan)
    dll[scored] = total[scored] / n_valid[scored]
    fraction = np.full(first.shape, np.nan)
    fraction[has_rows] = n_valid[has_rows] / n_rows[has_rows]
    unscorable = np.where(~has_rows, "no_row", np.where(scored, "", "no_valid_row"))
    return pd.DataFrame(
        {
            "evidence_dll": dll,
            "evidence_n_valid_rows": pd.arrays.IntegerArray(n_valid, ~has_rows),
            "evidence_valid_fraction": fraction,
            "evidence_unscorable": unscorable.astype(object),
        },
        index=intervals.index,
    )
