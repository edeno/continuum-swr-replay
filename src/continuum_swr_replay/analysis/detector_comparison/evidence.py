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
``clusterless_epoch_log_likelihood``, whatever rows are computed together: the
spikes a row holds never depend on the blocking (:data:`BLOCK_ROWS`). Its
values can, at the float32 rounding of the library's sums (at most about 5e-5
on log-likelihoods of order 100). The
grid's last sample holds no span of spikes and is never a row (the row rule of
``features``).

The likelihood reads no position (``is_local=False``) and no posterior, so the
evidence of an interval depends only on its rows' spikes and the animal's bins.
"""

from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    _drop_the_unreachable_row,
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    RowTraces,
    _grid_rows,
    _over_rows,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    interior_bin_mask,
    nearest_interior_bins,
    subnormal_occupancy_bin_mask,
)
from continuum_swr_replay.intervals import contiguous_true_runs

__all__ = [
    "BLOCK_ROWS",
    "animal_bins",
    "epoch_row_evidence",
    "interval_evidence",
    "log_likelihood_blocks",
    "remote_evidence",
]

BLOCK_ROWS = 25_000
"""Rows of one grid block (50 s at 500 Hz).

The grid is cut into blocks of this many rows, from row 0, and the likelihood
is computed one block at a time, so its memory is bounded by the block. Every
library call covers ``BLOCK_ROWS + 3`` samples, so they all share one time
shape: the library runs JAX op by op and compiles anew for each new shape
(about 7 s a call on a real epoch, against 0.7 s per 1,000 rows of spikes).
"""


def _block_spans(
    n_time: int, rows: NDArray[np.bool_], block_rows: int
) -> list[tuple[int, int, int, int]]:
    """``(first, stop, low, high)`` of each grid block holding a requested row.

    Rows ``first`` to ``stop - 1`` are the block's; samples ``low`` to
    ``high - 1`` are its call, ``block_rows + 3`` of them (the whole grid when
    shorter): one sample before the block and two after, the call moved
    inward at the grid's ends.
    """
    n_rows = n_time - 1  # the last sample is no row
    n_call = min(block_rows + 3, n_time)
    spans = []
    for first in range(0, n_rows, block_rows):
        stop = min(first + block_rows, n_rows)
        if rows[first:stop].any():
            low = min(max(first - 1, 0), n_time - n_call)
            spans.append((first, stop, low, low + n_call))
    return spans


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
    """The log-likelihood of the requested rows, one grid block at a time.

    The grid is cut into blocks of ``block_rows`` rows from row 0, and each
    block holding a requested row is one library call. The call covers the
    block padded by one sample before and two after, moved inward at the
    grid's ends so every call has ``block_rows + 3`` samples; the pads are
    dropped. Only the spikes of the block's requested rows are sent, so
    the other rows of a call are wrong and are never returned. Row ``k`` holds
    exactly the spikes in ``[t_k, t_{k + 1})`` (the grid's last row also the
    spike on its final sample), as in one call on the whole grid.

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
        Rows per block, :data:`BLOCK_ROWS` by default. It bounds a call's
        memory and fixes its time shape (``block_rows + 3`` samples); it
        changes no row's spikes, only the float rounding of its values.

    Yields
    ------
    first_row : int
        The first row of a run of requested rows within one block.
    log_likelihood : NDArray, shape (n_run_rows, n_interior_bins)
        The run's rows, on the real place bins.

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
    # each spike's row, as the decode bins it: t_k <= s < t_{k+1}, the final
    # sample in the last row; -1 off the grid
    spike_rows = []
    for t in spike_times:
        t = np.asarray(t, dtype=np.float64)
        row = np.searchsorted(time, t, side="right") - 1
        row[t == time[-1]] = time.size - 2
        row[(t < time[0]) | (t > time[-1])] = -1
        spike_rows.append(row)
    for first, stop, low, high in _block_spans(time.size, rows, block_rows):
        in_block = rows.copy()
        in_block[:first] = False
        in_block[stop:] = False
        sent = [(row >= 0) & in_block[np.maximum(row, 0)] for row in spike_rows]
        log_likelihood = predict_clusterless_kde_log_likelihood(
            cast(Any, time[low:high]),
            cast(Any, time[low:high]),
            cast(Any, position[low:high]),
            [np.asarray(t)[k] for t, k in zip(spike_times, sent, strict=True)],
            [
                np.asarray(f)[k]
                for f, k in zip(spike_waveform_features, sent, strict=True)
            ],
            **encargs,
            is_local=False,
            disable_progress_bar=True,
        )
        # the call's last row holds no span; it and the pads are dropped
        reachable = _drop_the_unreachable_row(
            np.asarray(log_likelihood), time[low:high], encargs
        ).log_likelihood
        for start, end in contiguous_true_runs(in_block[first:stop]).tolist():
            yield first + start, reachable[first + start - low : first + end - low]


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
        As :func:`log_likelihood_blocks`.

    Returns
    -------
    evidence : NDArray, shape (n_time,)
        :func:`remote_evidence` on the requested rows; NaN on every other row.
        A row not requested reads NaN as an invalid row does, so a caller must
        request every row it scores.

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
        ``"no_valid_row"`` or ``""``. The rows are ``features``': the grid's
        last sample is never one, so an interval holding only it is
        ``"no_row"``.

    Raises
    ------
    ValueError
        ``evidence`` is not on the grid.
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.shape != traces.time.shape:
        raise ValueError(
            f"evidence has shape {evidence.shape}; the grid {traces.time.shape}."
        )
    first, n_rows = _grid_rows(traces.time, intervals)
    has_rows = n_rows > 0
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
