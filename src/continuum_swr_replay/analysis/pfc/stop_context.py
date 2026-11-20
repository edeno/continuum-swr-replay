"""Stop-level context of the mPFC response to replay, pooled across sessions.

Does the mPFC response to a replay event differ with the stop's reward outcome,
or before a trip that leaves the patch? Both are properties of the *stop*, and
replay is concentrated after reward and away from switches, so a slope within
one session is rarely estimable. This module estimates the context effect once,
from all sessions together.

The response
------------
For unit ``u`` and replay event ``e`` (one class at a time) in stop ``s``:

- ``y_ue``: spikes in the post-onset window ``[0, +200) ms`` (fitted bins only).
- ``B_us``, ``T_s``: spikes and seconds in the stop's *event-free* fitted time,
  the fitted bins of the stop outside every event's ``[-250, +250] ms`` window
  (any class: non-local events, SWRs and the events the analyzed-time rule
  dropped, in the cohort or not).

The model
---------
``E[y_ue] = b_us * w_e * exp(alpha_u + beta . x_s + gamma * overlap_e)`` with
``b_us`` the unit's rate in the stop's event-free time, ``w_e`` the window's
fitted seconds (0.2 s when complete), ``x_s`` the stop's context
(``rewarded``, ``next_trip_switches_patch``) and ``overlap_e`` the number of
other events whose onset falls in ``e``'s post window. ``exp(beta)`` is the
multiplicative change of the post-onset response relative to the stop's own
baseline.

Two estimators of the same mean model are provided:

- ``"conditional"`` (the main one): ``b_us`` is a free parameter per (unit,
  stop) cell, estimated jointly from ``B_us ~ Poisson(b_us T_s)``. Conditioning
  on the cell's total count removes it exactly (Poisson fixed effects), so the
  estimate does not inherit the noise of a plugged-in baseline.
- ``"plugin"``: the offset ``log(B_us / T_s * w_e)`` is plugged in. It is
  biased when baselines are low counts, because ``E[log B] != log E[B]`` and
  the bias differs between short and long stops; it is kept as a check.

Both are fit by Newton's method with the per-unit intercepts ``alpha_u``
eliminated through their block-diagonal Hessian (no unit dummies are built),
and the uncertainty is a sandwich clustered by recording day (NWB file): mPFC
units are sorted once per day and recur in every epoch of that day, so the
epochs of a day are not independent. With about ten days the sandwich takes the
CR1 small-sample factor and the intervals a t quantile with ``G - 1`` degrees
of freedom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from continuum_swr_replay.analysis.pfc.event_sets import DEFAULT_BASIS_HALF_WINDOW_S
from continuum_swr_replay.analysis.pfc.load_population import CALLED_ALPHA
from continuum_swr_replay.analysis.pfc.population_statistics import called_units
from continuum_swr_replay.intervals import (
    get_spikecount_per_time_bin,
    merge_intervals,
    overlaps_open,
    sample_step,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

logger = logging.getLogger(__name__)

__all__ = [
    "CONTEXT_TERMS",
    "ESTIMATORS",
    "OVERLAP_TERM",
    "POST_WINDOW_S",
    "SIGNS",
    "STOP_CONTEXT_CLASSES",
    "UNIT_SETS",
    "StopContextFit",
    "assemble_stop_context_table",
    "count_other_onsets",
    "fit_stop_context_glm",
    "model_frame",
    "near_onset_mask",
    "onset_bin_index",
    "prepare_session_stop_context",
    "select_units",
    "stop_context_design",
    "summarize_stop_context",
]

#: Post-onset response window, ``[0, POST_WINDOW_S)`` after event onset.
POST_WINDOW_S = 0.2
#: Half-width of the window around every event onset that event-free baseline
#: time avoids; the canonical kernels' fit window.
AVOID_HALF_WINDOW_S = DEFAULT_BASIS_HALF_WINDOW_S
#: The replay classes analyzed.
STOP_CONTEXT_CLASSES = ("non_local_SWR", "non_local_no_SWR")
#: Stop-level context terms, in design order.
CONTEXT_TERMS = ("rewarded", "next_trip_switches_patch")
#: Event-level nuisance term.
OVERLAP_TERM = "overlap"
#: Largest absolute log-scale estimate taken as finite (exp(10) ~ 22,000);
#: beyond it a term is separated and has no estimate.
MAX_ABS_COEF = 10.0
#: The two estimators of :func:`fit_stop_context_glm`.
ESTIMATORS = ("conditional", "plugin")
#: The response signs a fit's units are chosen by (:func:`select_units`).
SIGNS = ("increase", "decrease")
#: The unit sets of :func:`select_units`: the units called for the class at
#: ``CALLED_ALPHA``, or every unit of the sign.
UNIT_SETS = ("called", "all")

#: Columns of the per-session event-level table, in order.
TABLE_COLUMNS = (
    "nwb_file_name",
    "epoch",
    "animal",
    "unit",
    "event_class",
    "event_id",
    "stop_idx",
    "onset_s",
    "y",
    "post_exposure_s",
    "baseline_count",
    "baseline_time_s",
    "stop_fit_count",
    "stop_fit_time_s",
    "rewarded",
    "next_trip_switches_patch",
    "overlap",
    "unit_fit_rate_hz",
)


# ---------------------------------------------------------------------------
# Session assembly: pure helpers
# ---------------------------------------------------------------------------


def onset_bin_index(onsets: NDArray, time: NDArray) -> NDArray:
    """Time bin holding each onset, the spike-binning convention.

    Parameters
    ----------
    onsets : NDArray, shape (n_events,)
        Onset times in seconds.
    time : NDArray, shape (n_time,)
        Bin start times in seconds, ascending.

    Returns
    -------
    NDArray of int, shape (n_events,)
        ``i`` with ``time[i] <= onset < time[i + 1]``, the bin
        :func:`~continuum_swr_replay.intervals.get_spikecount_per_time_bin`
        puts a spike at that time in, and the bin the canonical design places
        the onset in (:func:`~continuum_swr_replay.analysis.pfc.design.event_onset_counts`).

    Raises
    ------
    ValueError
        If an onset lies outside ``[time[0], time[-1]]``.
    """
    onsets = np.asarray(onsets, dtype=float)
    if len(onsets) and (onsets.min() < time[0] or onsets.max() > time[-1]):
        raise ValueError("an onset lies outside the session's time axis")
    return np.clip(np.searchsorted(time, onsets, side="right") - 1, 0, len(time) - 1)


def near_onset_mask(onset_bins: NDArray, n_time: int, half_window_bins: int) -> NDArray:
    """Bins within ``half_window_bins`` of any onset bin, inclusive.

    Parameters
    ----------
    onset_bins : NDArray of int, shape (n_onsets,)
        Onset bin indices.
    n_time : int
        Number of time bins.
    half_window_bins : int
        Half-width in bins; bin ``b`` is near onset ``o`` when
        ``|b - o| <= half_window_bins``.

    Returns
    -------
    NDArray of bool, shape (n_time,)
    """
    edges = np.zeros(n_time + 1, dtype=np.int64)
    onset_bins = np.asarray(onset_bins, dtype=np.int64)
    np.add.at(edges, np.clip(onset_bins - half_window_bins, 0, n_time), 1)
    np.add.at(edges, np.clip(onset_bins + half_window_bins + 1, 0, n_time), -1)
    return np.cumsum(edges[:-1]) > 0


def count_other_onsets(
    event_bins: NDArray,
    *,
    non_local_bins: NDArray,
    non_local_start: NDArray,
    non_local_end: NDArray,
    swr_bins: NDArray,
    swr_start: NDArray,
    swr_end: NDArray,
    dropped_bins: NDArray,
    n_post_bins: int,
) -> NDArray:
    """Other events whose onset falls in each event's post window.

    The post window of an event with onset bin ``b`` is bins
    ``[b, b + n_post_bins)``. Each other event is counted once, at its onset:
    every non-local event but the event itself, every SWR that overlaps no
    non-local event (an SWR that overlaps a replay event is part of that
    event; the open overlap test, which equals the canonical ``is_swr`` rule
    for intervals of positive duration), and every event the analyzed-time
    rule dropped.

    Parameters
    ----------
    event_bins : NDArray of int, shape (n_events,)
        Onset bins of the events to count around. Each must be one of
        ``non_local_bins``.
    non_local_bins, non_local_start, non_local_end : NDArray, shape (n_non_local,)
        Onset bin, start and end (s) of every non-local event of the session.
    swr_bins, swr_start, swr_end : NDArray, shape (n_swr,)
        Onset bin, start and end (s) of every SWR of the session.
    dropped_bins : NDArray of int, shape (n_dropped,)
        Onset bins of the events the analyzed-time rule dropped.
    n_post_bins : int
        Post-window length in bins.

    Returns
    -------
    NDArray of int, shape (n_events,)
    """
    event_bins = np.asarray(event_bins, dtype=np.int64)
    if not len(swr_bins) == len(swr_start) == len(swr_end):
        raise ValueError(
            f"{len(swr_bins)} SWR onset bins for {len(swr_start)} SWR intervals; "
            "filter the intervals to the time axis before binning their onsets"
        )

    def in_window(onsets: NDArray) -> NDArray:
        onsets = np.sort(np.asarray(onsets, dtype=np.int64))
        return np.searchsorted(onsets, event_bins + n_post_bins, side="left") - (
            np.searchsorted(onsets, event_bins, side="left")
        )

    replay = merge_intervals(
        np.asarray(non_local_start, dtype=float), np.asarray(non_local_end, dtype=float)
    )
    swr_with_replay = overlaps_open(
        np.asarray(swr_start, dtype=float),
        np.asarray(swr_end, dtype=float),
        replay[:, 0],
        replay[:, 1],
    )
    # The event's own onset is one of the non-local onsets in its window.
    count = (
        in_window(non_local_bins)
        - 1
        + in_window(np.asarray(swr_bins)[~swr_with_replay])
        + in_window(dropped_bins)
    )
    if (count < 0).any():
        raise ValueError("an event's own onset is missing from the non-local onsets")
    return count


def _stop_of_times(times: NDArray, stops: pd.DataFrame) -> NDArray:
    """Row of ``stops`` (closed ``[stop_start, stop_end]``) holding each time, or -1."""
    starts = stops["stop_start"].to_numpy(dtype=float)
    ends = stops["stop_end"].to_numpy(dtype=float)
    if np.any(np.diff(starts) < 0):
        raise ValueError("stops must be sorted by stop_start")
    row = np.searchsorted(starts, times, side="right") - 1
    clipped = np.clip(row, 0, None)
    inside = (row >= 0) & (times <= ends[clipped]) if len(starts) else row >= 0
    return np.where(inside, row, -1)


def assemble_stop_context_table(
    *,
    time: NDArray,
    fit_mask: NDArray,
    spike_times: list[NDArray],
    events: dict[str, pd.DataFrame],
    all_non_local: pd.DataFrame,
    swr_intervals: pd.DataFrame,
    dropped_events: pd.DataFrame,
    stops: pd.DataFrame,
    sampling_frequency: float,
    post_window_s: float = POST_WINDOW_S,
    avoid_half_window_s: float = AVOID_HALF_WINDOW_S,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One session's event-level response table (no session key columns).

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Bin start times in seconds (the aligned 500 Hz position axis).
    fit_mask : NDArray of bool, shape (n_time,)
        The canonical fitted bins (immobile, in a cohort stop, clear of
        excluded time).
    spike_times : list of NDArray
        One array of spike times (s) per unit, in the bundle's unit order.
    events : dict of str to pd.DataFrame
        Per class in :data:`STOP_CONTEXT_CLASSES`, the cohort's events:
        ``event_id``, ``start_time``, ``end_time``.
    all_non_local : pd.DataFrame
        Every non-local event of the session, in the cohort or not:
        ``start_time``, ``end_time``.
    swr_intervals : pd.DataFrame
        Every speed-gated SWR of the session: ``start_time``, ``end_time``.
    dropped_events : pd.DataFrame
        The events the analyzed-time rule dropped: ``start_time``, ``end_time``.
    stops : pd.DataFrame, shape (n_stops, ...)
        The cohort stops: ``stop_idx``, ``stop_start``, ``stop_end`` (s,
        closed), ``is_rewarded_outcome`` (0/1/NaN) and
        ``next_trip_switches_patch`` (0/1/NaN).
    sampling_frequency : float
        Bins per second.
    post_window_s : float
        Post-onset window length.
    avoid_half_window_s : float
        Half-width of the window around every onset that baseline time avoids.

    Returns
    -------
    table : pd.DataFrame, shape (n_units * n_events, ...)
        One row per unit and event of the two classes: ``unit`` (bundle row),
        ``event_class``, ``event_id``, ``stop_idx``, ``onset_s``, ``y`` (spikes
        in the fitted bins of the post window), ``post_exposure_s`` (fitted
        seconds in it), ``baseline_count`` / ``baseline_time_s`` (spikes and
        seconds in the stop's event-free fitted time), ``stop_fit_count`` /
        ``stop_fit_time_s`` (in all the stop's fitted time), ``rewarded``,
        ``next_trip_switches_patch``, ``overlap`` and ``unit_fit_rate_hz`` (the
        unit's mean rate over all fitted bins at the measured bin width
        ``median(diff(time))``, as the bundles' ``immobility_avg_firing_rate``
        is computed). Seconds elsewhere are bins / ``sampling_frequency``; the
        two differ by a constant factor (about 5e-5) that cancels in every rate
        ratio.
    report : dict
        Counts: units, stops, fitted bins, event-free bins, events with no
        fitted post time and with only part of it, events per class, and the
        onsets on the time axis that baseline time avoids (non-local, SWR,
        dropped).

    Raises
    ------
    ValueError
        If an analyzed event, or a fitted bin, lies outside every cohort stop.
    """
    time = np.asarray(time, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    n_time = len(time)
    if fit.shape != (n_time,):
        raise ValueError(f"fit_mask has shape {fit.shape}, expected ({n_time},)")
    n_post = int(round(post_window_s * sampling_frequency))
    half = int(round(avoid_half_window_s * sampling_frequency))
    stops = stops.sort_values("stop_start").reset_index(drop=True)
    n_stops = len(stops)

    # Fitted bins per stop, and the event-free ones.
    stop_of_bin = np.full(n_time, -1)
    stop_of_bin[fit] = _stop_of_times(time[fit], stops)
    if (stop_of_bin[fit] < 0).any():
        raise ValueError("a fitted bin lies outside every cohort stop")

    def on_axis(frame: pd.DataFrame) -> pd.DataFrame:
        """The events whose onset lies on the time axis (the ones binned)."""
        start = frame["start_time"].to_numpy(dtype=float)
        return frame.loc[(start >= time[0]) & (start <= time[-1])]

    def onsets(frame: pd.DataFrame) -> NDArray:
        return onset_bin_index(frame["start_time"].to_numpy(dtype=float), time)

    # Filtered once, so each onset bin and its interval stay paired.
    swr_on_axis = on_axis(swr_intervals)
    non_local_bins = onsets(on_axis(all_non_local))
    swr_bins = onsets(swr_on_axis)
    dropped_bins = onsets(on_axis(dropped_events))
    near_event = near_onset_mask(
        np.concatenate([non_local_bins, swr_bins, dropped_bins]), n_time, half
    )
    free = fit & ~near_event
    stop_fit_bins = np.bincount(stop_of_bin[fit], minlength=n_stops)
    stop_free_bins = np.bincount(stop_of_bin[free], minlength=n_stops)

    # Events of the analyzed classes, with their stop and post window.
    frames = []
    for event_class in STOP_CONTEXT_CLASSES:
        frame = events[event_class].loc[:, ["event_id", "start_time", "end_time"]]
        frames.append(frame.assign(event_class=event_class))
    event_frame = pd.concat(frames, ignore_index=True)
    start = event_frame["start_time"].to_numpy(dtype=float)
    event_bins = onset_bin_index(start, time)
    stop_row = _stop_of_times(start, stops)
    if (stop_row < 0).any():
        raise ValueError(
            f"{int((stop_row < 0).sum())} analyzed event(s) lie outside every "
            "cohort stop"
        )
    post_index = event_bins[:, None] + np.arange(n_post)[None, :]
    post_valid = post_index < n_time
    post_fit = np.where(post_valid, fit[np.clip(post_index, 0, n_time - 1)], False)
    post_exposure = post_fit.sum(axis=1) / sampling_frequency
    overlap = count_other_onsets(
        event_bins,
        non_local_bins=non_local_bins,
        non_local_start=all_non_local["start_time"].to_numpy(dtype=float),
        non_local_end=all_non_local["end_time"].to_numpy(dtype=float),
        swr_bins=swr_bins,
        swr_start=swr_on_axis["start_time"].to_numpy(dtype=float),
        swr_end=swr_on_axis["end_time"].to_numpy(dtype=float),
        dropped_bins=dropped_bins,
        n_post_bins=n_post,
    )

    n_units = len(spike_times)
    n_events = len(event_frame)
    y = np.zeros((n_units, n_events), dtype=np.int64)
    baseline = np.zeros((n_units, n_stops), dtype=np.int64)
    stop_count = np.zeros((n_units, n_stops), dtype=np.int64)
    rate = np.zeros(n_units, dtype=float)
    bin_width_s = sample_step(time)
    for unit, unit_spikes in enumerate(spike_times):
        counts = get_spikecount_per_time_bin(np.asarray(unit_spikes), time)
        fitted = np.where(fit, counts, 0)
        cumulative = np.concatenate([[0], np.cumsum(fitted)])
        last = np.clip(event_bins + n_post, 0, n_time)
        y[unit] = cumulative[last] - cumulative[event_bins]
        baseline[unit] = np.bincount(
            stop_of_bin[free], weights=counts[free], minlength=n_stops
        ).astype(np.int64)
        stop_count[unit] = np.bincount(
            stop_of_bin[fit], weights=counts[fit], minlength=n_stops
        ).astype(np.int64)
        rate[unit] = counts[fit].mean() / bin_width_s if fit.any() else np.nan

    unit_index = np.repeat(np.arange(n_units), n_events)
    event_index = np.tile(np.arange(n_events), n_units)
    stop_of_row = stop_row[event_index]
    table = pd.DataFrame(
        {
            "unit": unit_index,
            "event_class": event_frame["event_class"].to_numpy()[event_index],
            "event_id": event_frame["event_id"].to_numpy()[event_index],
            "stop_idx": stops["stop_idx"].to_numpy()[stop_of_row],
            "onset_s": start[event_index],
            "y": y.ravel(),
            "post_exposure_s": post_exposure[event_index],
            "baseline_count": baseline[unit_index, stop_of_row],
            "baseline_time_s": stop_free_bins[stop_of_row] / sampling_frequency,
            "stop_fit_count": stop_count[unit_index, stop_of_row],
            "stop_fit_time_s": stop_fit_bins[stop_of_row] / sampling_frequency,
            "rewarded": pd.to_numeric(
                stops["is_rewarded_outcome"], errors="coerce"
            ).to_numpy(dtype=float)[stop_of_row],
            "next_trip_switches_patch": pd.to_numeric(
                stops["next_trip_switches_patch"], errors="coerce"
            ).to_numpy(dtype=float)[stop_of_row],
            "overlap": overlap[event_index],
            "unit_fit_rate_hz": rate[unit_index],
        }
    )
    report = {
        "n_units": n_units,
        "n_stops": n_stops,
        "n_fit_bins": int(fit.sum()),
        "n_event_free_fit_bins": int(free.sum()),
        "n_events_no_post_fit_time": int((post_exposure == 0).sum()),
        "n_events_partial_post_fit_time": int(
            ((post_exposure > 0) & (post_exposure < n_post / sampling_frequency)).sum()
        ),
        **{
            f"n_events_{event_class}": int(len(events[event_class]))
            for event_class in STOP_CONTEXT_CLASSES
        },
        "n_avoided_onsets": {
            "non_local": int(len(non_local_bins)),
            "swr": int(len(swr_bins)),
            "dropped": int(len(dropped_bins)),
        },
    }
    return table, report


# ---------------------------------------------------------------------------
# Session assembly: canonical inputs
# ---------------------------------------------------------------------------


def prepare_session_stop_context(
    nwb_file_name: str,
    epoch_name: str,
    *,
    brain_area: str = "mPFC",
    sampling_frequency: float = SAMPLING_FREQUENCY,
    bundle_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble one session's table from the canonical Figure 5 inputs.

    The session is assembled by the steps
    :func:`~continuum_swr_replay.analysis.pfc.marked_glm.prepare_marked_session_context`
    takes (decode/position alignment, the canonical fit mask with the
    analyzed-time rule, the canonical event sets on the at-well cohort), so
    events, fitted bins and spike binning equal Figure 5's. The stop context is
    the cohort stop's ``is_rewarded_outcome`` and the forward
    ``next_trip_switches_patch`` derived over every stop of the session.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch within that file, e.g. ``"02_r1"``.
    brain_area : str
        Sorted-unit area whose spikes are counted.
    sampling_frequency : float
        Sampling rate of the time axis, in Hz.
    bundle_path : str or Path, optional
        The session's canonical modulation bundle; when given, its unit count
        and ``immobility_avg_firing_rate`` must match the units counted here.

    Returns
    -------
    table : pd.DataFrame
        :func:`assemble_stop_context_table` with the session key columns.
    report : dict
        Its counts, the cohort report, the assembly counts and the SHA-256
        digests of every input read.

    Raises
    ------
    ValueError
        If the bundle's units do not match the counted units.
    """
    # Deferred: these open Spyglass / DataJoint resources at import time.
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_canonical_events,
        load_session_swr_intervals,
    )
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.design import symmetric_lags
    from continuum_swr_replay.analysis.pfc.marked_model_family import (
        load_session_stop_patch_switch_marks,
    )
    from continuum_swr_replay.analysis.pfc.session import (
        _load_aligned_session,
        _session_events,
        _session_fit_population,
    )
    from continuum_swr_replay.decoding.identifiers import (
        format_epoch_identifier,
        parse_nwb_filename,
    )
    from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
    from continuum_swr_replay.runs.io import sha256_file

    stop_cohort = "at_well"
    session_logger = logging.getLogger(
        format_epoch_identifier(nwb_file_name, epoch_name, brain_area, "_stop_context")
    )
    cohort_stops, cohort_report = load_session_stop_cohort(
        nwb_file_name, epoch_name, cohort=stop_cohort
    )
    switch_marks, stop_table_sha256 = load_session_stop_patch_switch_marks(
        nwb_file_name, epoch_name
    )
    missing_marks = set(cohort_stops["stop_idx"]) - set(switch_marks.index)
    if missing_marks:
        raise ValueError(f"{len(missing_marks)} cohort stops have no switch marks")
    stops = cohort_stops.assign(
        next_trip_switches_patch=switch_marks.loc[
            cohort_stops["stop_idx"], "next_trip_switches_patch"
        ].to_numpy(dtype=float)
    )

    results, data, position_info, alignment = _load_aligned_session(
        nwb_file_name, epoch_name, session_logger
    )
    time = position_info.index.to_numpy(dtype=float)
    fit_lags = symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, sampling_frequency)
    data, excluded, _, fit_mask, bin_counts = _session_fit_population(
        data,
        nwb_file_name,
        epoch_name,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        logger=session_logger,
    )
    event_times, event_counts, event_provenance, dropped, _, _ = _session_events(
        data,
        results,
        nwb_file_name,
        epoch_name,
        time=time,
        fit_mask=fit_mask,
        fit_lags=fit_lags,
        sampling_frequency=sampling_frequency,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
        logger=session_logger,
    )
    # Baseline time avoids every event of the session, in the cohort or not.
    all_non_local, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    all_swr, _ = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
    )
    spike_times = list(data.get("spike_times", {}).get(brain_area, []))
    table, report = assemble_stop_context_table(
        time=time,
        fit_mask=np.asarray(fit_mask, dtype=bool),
        spike_times=spike_times,
        events=event_times,
        all_non_local=all_non_local,
        swr_intervals=all_swr,
        dropped_events=dropped,
        stops=stops,
        sampling_frequency=sampling_frequency,
    )
    animal, _ = parse_nwb_filename(nwb_file_name)
    table.insert(0, "animal", animal)
    table.insert(0, "epoch", epoch_name)
    table.insert(0, "nwb_file_name", nwb_file_name)

    join: dict[str, Any] = {}
    if bundle_path is not None:
        bundle = pd.read_pickle(bundle_path)
        rates = (
            table.drop_duplicates("unit").set_index("unit")["unit_fit_rate_hz"]
            if len(table)
            else pd.Series(dtype=float)
        )
        if len(bundle) != len(spike_times):
            raise ValueError(
                f"bundle {bundle_path} has {len(bundle)} units; "
                f"{len(spike_times)} were counted"
            )
        if len(rates):
            difference = np.abs(
                bundle["immobility_avg_firing_rate"].to_numpy(dtype=float)[
                    rates.index.to_numpy()
                ]
                - rates.to_numpy()
            )
            if not np.all(difference <= 1e-6 * np.maximum(1.0, rates.to_numpy())):
                raise ValueError(
                    f"bundle {bundle_path}: immobility_avg_firing_rate differs from "
                    f"the counted rate by up to {np.nanmax(difference):.3g} Hz"
                )
            join["bundle_rate_max_abs_difference_hz"] = float(np.max(difference))
        join["bundle_path"] = str(bundle_path)
        join["bundle_sha256"] = sha256_file(bundle_path)
        join["bundle_n_units"] = int(len(bundle))

    report = {
        **report,
        **join,
        "cohort": cohort_report,
        "bin_counts": bin_counts,
        "event_counts": event_counts,
        "alignment": alignment,
        "n_excluded_intervals": int(len(excluded)),
        "input_provenance": {
            **event_provenance,
            "stop_table_sha256": stop_table_sha256,
        },
    }
    return table, report


# ---------------------------------------------------------------------------
# Model data
# ---------------------------------------------------------------------------


def select_units(
    units: pd.DataFrame, event_class: str, sign: str, unit_set: str
) -> np.ndarray:
    """Which units one fit uses: their peak for ``event_class`` has ``sign``
    and, for ``unit_set == "called"``, they are called for ``event_class`` at
    :data:`~continuum_swr_replay.analysis.pfc.load_population.CALLED_ALPHA`
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_units`).
    Another class's peak or p never enters.

    Parameters
    ----------
    units : pandas.DataFrame
        One row per unit, with ``<event_class>: peak_log_modulation`` and the
        class's held-out p column.
    event_class : str
        One of :data:`STOP_CONTEXT_CLASSES`.
    sign : str
        One of :data:`SIGNS`; a NaN or zero peak has neither sign.
    unit_set : str
        One of :data:`UNIT_SETS`.

    Returns
    -------
    np.ndarray of bool, shape (n_units,)
    """
    peak = units[f"{event_class}: peak_log_modulation"].to_numpy(dtype=float)
    has_sign = peak > 0 if sign == "increase" else peak < 0
    if unit_set == "all":
        return has_sign
    return has_sign & called_units(units, event_class, CALLED_ALPHA)


def model_frame(
    table: pd.DataFrame,
    *,
    event_class: str,
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows of one class for the selected units, with every drop counted.

    Parameters
    ----------
    table : pd.DataFrame
        Concatenated session tables (:func:`assemble_stop_context_table` plus
        session keys).
    event_class : str
        One of :data:`STOP_CONTEXT_CLASSES`.
    units : pd.DataFrame
        The selected units: ``nwb_file_name``, ``epoch``, ``unit``.

    Returns
    -------
    frame : pd.DataFrame
        Rows kept: known context, positive event-free time, fitted post time,
        and a (unit, stop) cell with at least one spike in the stop's fitted
        time. Adds ``session`` (``"<nwb_file_name> <epoch>"``) and ``day``
        (``nwb_file_name``, the recording day the standard errors cluster on).
    drops : dict
        ``n_stops_with_events`` (stops with an event of the class, for the
        selected units), of them ``n_stops_unknown_reward``,
        ``n_stops_unknown_switch`` and ``n_stops_no_event_free_time``; then
        ``n_events_no_post_fit_time`` and ``n_unit_stop_cells_silent`` (the
        unit had no spike in the stop's fitted time).
    """
    rows = table.loc[table["event_class"].eq(event_class)].merge(
        units.loc[:, ["nwb_file_name", "epoch", "unit"]],
        on=["nwb_file_name", "epoch", "unit"],
        how="inner",
        validate="many_to_one",
    )
    rows = rows.assign(
        session=rows["nwb_file_name"] + " " + rows["epoch"], day=rows["nwb_file_name"]
    )
    stop_key = ["session", "stop_idx"]
    stops = rows.drop_duplicates(stop_key)
    unknown_reward = ~np.isfinite(stops["rewarded"].to_numpy(dtype=float))
    unknown_switch = ~np.isfinite(
        stops["next_trip_switches_patch"].to_numpy(dtype=float)
    )
    no_free = stops["baseline_time_s"].to_numpy(dtype=float) <= 0
    drops = {
        "n_stops_with_events": int(len(stops)),
        "n_stops_unknown_reward": int(unknown_reward.sum()),
        "n_stops_unknown_switch": int(unknown_switch.sum()),
        "n_stops_no_event_free_time": int(no_free.sum()),
    }
    keep = (
        np.isfinite(rows["rewarded"].to_numpy(dtype=float))
        & np.isfinite(rows["next_trip_switches_patch"].to_numpy(dtype=float))
        & (rows["baseline_time_s"].to_numpy(dtype=float) > 0)
    )
    rows = rows.loc[keep]
    no_post = rows["post_exposure_s"].to_numpy(dtype=float) <= 0
    drops["n_events_no_post_fit_time"] = int(
        rows.loc[no_post].drop_duplicates(["session", "event_id"]).shape[0]
    )
    rows = rows.loc[~no_post]
    silent = rows["stop_fit_count"].to_numpy() == 0
    drops["n_unit_stop_cells_silent"] = int(
        rows.loc[silent].drop_duplicates(["session", "unit", "stop_idx"]).shape[0]
    )
    return rows.loc[~silent].reset_index(drop=True), drops


@dataclass(frozen=True)
class StopContextDesign:
    """Arrays one fit needs, with units and (unit, stop) cells coded.

    Attributes
    ----------
    y : NDArray, shape (n_rows,)
        Post-window spike counts.
    exposure : NDArray, shape (n_rows,)
        Post-window fitted seconds.
    X : NDArray, shape (n_rows, n_terms)
        Terms, in ``names`` order.
    names : tuple of str
        Term names.
    unit : NDArray of int, shape (n_rows,)
        Unit code of each row.
    cell : NDArray of int, shape (n_rows,)
        (unit, stop) cell code of each row.
    cell_unit : NDArray of int, shape (n_cells,)
        Unit code of each cell.
    baseline_count, baseline_time : NDArray, shape (n_cells,)
        Event-free spikes and seconds of each cell.
    cluster : NDArray of int, shape (n_rows,)
        Cluster (recording day) code of each row; every unit lies in one.
    """

    y: NDArray
    exposure: NDArray
    X: NDArray
    names: tuple[str, ...]
    unit: NDArray
    cell: NDArray
    cell_unit: NDArray
    baseline_count: NDArray
    baseline_time: NDArray
    cluster: NDArray


def stop_context_design(
    frame: pd.DataFrame,
    terms: tuple[str, ...] = (*CONTEXT_TERMS, OVERLAP_TERM),
    *,
    cluster: str = "day",
) -> StopContextDesign:
    """Code a :func:`model_frame` for :func:`fit_stop_context_glm`.

    Parameters
    ----------
    frame : pd.DataFrame
        Model rows: ``session``, ``unit``, ``stop_idx``, ``y``,
        ``post_exposure_s``, ``baseline_count``, ``baseline_time_s``, the
        ``cluster`` column and every ``terms`` entry.
    terms : tuple of str
        Columns used as terms.
    cluster : str
        The column the standard errors cluster on; each session must lie in
        one cluster. ``"day"`` (the recording day) by default.

    Returns
    -------
    StopContextDesign
        Units and (unit, stop) cells are coded within session, as before.

    Raises
    ------
    ValueError
        If a session lies in more than one cluster.
    """
    pairs = frame[["session", cluster]].drop_duplicates()
    if pairs["session"].duplicated().any():
        raise ValueError(f"a session lies in more than one {cluster!r} cluster")
    session = pd.factorize(frame["session"])[0]
    unit = pd.factorize(pd.MultiIndex.from_arrays([session, frame["unit"]]))[0]
    cell, cell_keys = pd.factorize(
        pd.MultiIndex.from_arrays([unit, frame["stop_idx"].to_numpy()])
    )
    first = np.unique(cell, return_index=True)[1]
    return StopContextDesign(
        y=frame["y"].to_numpy(dtype=float),
        exposure=frame["post_exposure_s"].to_numpy(dtype=float),
        X=frame.loc[:, list(terms)].to_numpy(dtype=float),
        names=tuple(terms),
        unit=unit,
        cell=cell,
        cell_unit=unit[first],
        baseline_count=frame["baseline_count"].to_numpy(dtype=float)[first],
        baseline_time=frame["baseline_time_s"].to_numpy(dtype=float)[first],
        cluster=pd.factorize(frame[cluster])[0],
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopContextFit:
    """One fit of :func:`fit_stop_context_glm`.

    Attributes
    ----------
    names : tuple of str
        Term names.
    coef : NDArray, shape (n_terms,)
        Log-scale estimates; NaN for a term the data do not identify.
    se_cluster : NDArray, shape (n_terms,)
        Sandwich standard errors clustered by the design's clusters (recording
        days), with the CR1 small-sample factor
        ``G / (G - 1) * (N - 1) / (N - K)`` (:func:`fit_stop_context_glm`).
    se_model : NDArray, shape (n_terms,)
        Model-based (inverse information) standard errors.
    converged : bool
        Whether Newton's method met its tolerance.
    n_iter : int
        Iterations taken.
    counts : dict
        Rows, units, cells and clusters used, and units left out as
        uninformative.
    """

    names: tuple[str, ...]
    coef: NDArray
    se_cluster: NDArray
    se_model: NDArray
    converged: bool
    n_iter: int
    counts: dict[str, int]


def _cell_sums(values: NDArray, cell: NDArray, n_cells: int) -> NDArray:
    """Sum rows (shape (n_rows,) or (n_rows, k)) within cells."""
    if values.ndim == 1:
        return np.bincount(cell, weights=values, minlength=n_cells)
    return np.column_stack(
        [
            np.bincount(cell, weights=values[:, j], minlength=n_cells)
            for j in range(values.shape[1])
        ]
    )


def _conditional_terms(
    alpha: NDArray, theta: NDArray, d: StopContextDesign, n_cells: int, n_units: int
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Log-likelihood, scores and Hessian blocks of the conditional model.

    Within a (unit, stop) cell ``c`` with total ``N_c = B_c + sum_e y_e``, the
    counts are multinomial over the baseline (weight ``T_c``) and the events
    (weight ``w_e exp(eta_e)``), ``eta_e = alpha_u + x_e . theta``.
    """
    eta = alpha[d.unit] + d.X @ theta
    weight = d.exposure * np.exp(eta)
    denominator = d.baseline_time + np.bincount(
        d.cell, weights=weight, minlength=n_cells
    )
    total = d.baseline_count + np.bincount(d.cell, weights=d.y, minlength=n_cells)
    p = weight / denominator[d.cell]
    loglik = float(np.sum(d.y * np.log(weight)) - np.sum(total * np.log(denominator)))
    residual = d.y - total[d.cell] * p
    p_cell = np.bincount(d.cell, weights=p, minlength=n_cells)
    q_cell = _cell_sums(p[:, None] * d.X, d.cell, n_cells)
    k = d.X.shape[1]
    outer = (d.X[:, :, None] * d.X[:, None, :]).reshape(len(d.y), k * k)
    r_cell = _cell_sums(p[:, None] * outer, d.cell, n_cells).reshape(n_cells, k, k)
    a_unit = np.bincount(
        d.cell_unit, weights=total * p_cell * (1 - p_cell), minlength=n_units
    )
    b_unit = _cell_sums((total * (1 - p_cell))[:, None] * q_cell, d.cell_unit, n_units)
    c_theta = np.einsum(
        "c,cij->ij", total, r_cell - q_cell[:, :, None] * q_cell[:, None, :]
    )
    return loglik, residual, a_unit, b_unit, c_theta


def _plugin_terms(
    alpha: NDArray, theta: NDArray, d: StopContextDesign, n_cells: int, n_units: int
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Log-likelihood, scores and Hessian blocks with the baseline plugged in."""
    offset = np.log(d.baseline_count[d.cell] / d.baseline_time[d.cell] * d.exposure)
    eta = offset + alpha[d.unit] + d.X @ theta
    mu = np.exp(eta)
    loglik = float(np.sum(d.y * eta - mu))
    residual = d.y - mu
    a_unit = np.bincount(d.unit, weights=mu, minlength=n_units)
    b_unit = _cell_sums(mu[:, None] * d.X, d.unit, n_units)
    c_theta = (d.X * mu[:, None]).T @ d.X
    return loglik, residual, a_unit, b_unit, c_theta


def fit_stop_context_glm(
    design: StopContextDesign,
    *,
    estimator: str = "conditional",
    max_iter: int = 100,
    tol: float = 1e-9,
) -> StopContextFit:
    """Fit the stop-context Poisson model with per-unit intercepts.

    The unit intercepts ``alpha_u`` enter only through a diagonal Hessian
    block, so each Newton step solves the ``n_terms`` system of the Schur
    complement and back-substitutes ``alpha``; no unit dummy is built. A unit
    with no spike in any post window (``alpha_u = -inf``) or, for the
    conditional estimator, none in any of its cells' event-free time
    (``alpha_u = +inf``) carries no information on the terms and is left out
    (counted). For the plug-in estimator a cell with no event-free spike has
    no finite offset and is left out (counted).

    Parameters
    ----------
    design : StopContextDesign
        From :func:`stop_context_design`.
    estimator : {"conditional", "plugin"}
        See the module docstring.
    max_iter : int
        Newton iterations allowed.
    tol : float
        Convergence tolerance on the largest absolute step.

    Returns
    -------
    StopContextFit
        A term the kept rows do not identify (no variation within any unit)
        is reported as NaN. A fit that does not converge, or whose estimate
        runs past :data:`MAX_ABS_COEF` (a separated term), reports every term
        as NaN with ``converged`` False. With one cluster ``se_cluster`` is
        NaN.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}; got {estimator!r}")
    d = design
    counts: dict[str, int] = {}
    keep_rows = np.ones(len(d.y), dtype=bool)
    if estimator == "plugin":
        zero_baseline = d.baseline_count[d.cell] <= 0
        counts["n_cells_zero_baseline"] = int(len(np.unique(d.cell[zero_baseline])))
        keep_rows &= ~zero_baseline
    n_units_all = int(d.unit.max()) + 1 if len(d.unit) else 0
    unit_y = np.bincount(
        d.unit[keep_rows], weights=d.y[keep_rows], minlength=n_units_all
    )
    cell_kept = np.zeros(len(d.cell_unit), dtype=bool)
    cell_kept[d.cell[keep_rows]] = True
    unit_b = np.bincount(
        d.cell_unit[cell_kept],
        weights=d.baseline_count[cell_kept],
        minlength=n_units_all,
    )
    has_rows = np.bincount(d.unit[keep_rows], minlength=n_units_all) > 0
    counts["n_units_no_post_spikes"] = int((has_rows & (unit_y == 0)).sum())
    counts["n_units_no_baseline_spikes"] = int(
        (has_rows & (unit_y > 0) & (unit_b == 0)).sum()
    )
    keep_rows &= ((unit_y > 0) & (unit_b > 0))[d.unit]
    d = _subset_design(d, keep_rows)
    n_units = int(d.unit.max()) + 1 if len(d.unit) else 0
    n_cells = len(d.cell_unit)
    names = d.names
    # A term with no variation within any unit is absorbed by alpha.
    identified = np.array(
        [_varies_within_units(d.X[:, j], d.unit) for j in range(d.X.shape[1])],
        dtype=bool,
    )
    d = replace(d, X=d.X[:, identified])
    counts.update(
        n_rows=int(len(d.y)),
        n_units=n_units,
        n_cells=n_cells,
        n_clusters=int(len(np.unique(d.cluster))),
        n_spikes_post=int(d.y.sum()),
    )
    coef = np.full(len(names), np.nan)
    se_cluster = np.full(len(names), np.nan)
    se_model = np.full(len(names), np.nan)
    if n_units == 0 or not identified.any():
        return StopContextFit(names, coef, se_cluster, se_model, False, 0, counts)
    terms = _conditional_terms if estimator == "conditional" else _plugin_terms
    theta = np.zeros(d.X.shape[1])
    alpha = _initial_alpha(d, n_units)
    loglik, residual, a_unit, b_unit, c_theta = terms(alpha, theta, d, n_cells, n_units)
    converged = False
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        grad_alpha = np.bincount(d.unit, weights=residual, minlength=n_units)
        grad_theta = d.X.T @ residual
        schur = c_theta - (b_unit / a_unit[:, None]).T @ b_unit
        try:
            step_theta = np.linalg.solve(
                schur, grad_theta - b_unit.T @ (grad_alpha / a_unit)
            )
        except np.linalg.LinAlgError:
            # A term separated (its estimate runs off to infinity): no estimate.
            break
        step_alpha = (grad_alpha - b_unit @ step_theta) / a_unit
        # Step halving keeps the log-likelihood from decreasing.
        scale = 1.0
        while True:
            new = terms(
                alpha + scale * step_alpha,
                theta + scale * step_theta,
                d,
                n_cells,
                n_units,
            )
            if new[0] >= loglik - 1e-12 * abs(loglik) or scale < 1e-6:
                break
            scale /= 2
        alpha = alpha + scale * step_alpha
        theta = theta + scale * step_theta
        loglik, residual, a_unit, b_unit, c_theta = new
        largest = max(
            np.abs(scale * step_theta).max(), np.abs(scale * step_alpha).max()
        )
        if largest < tol:
            converged = True
            break
    # A fit that did not converge (a separated term) or whose estimate is at
    # a boundary has no estimate; it is reported as NaN, never as a number.
    if not converged or np.abs(theta).max() > MAX_ABS_COEF:
        return StopContextFit(names, coef, se_cluster, se_model, False, n_iter, counts)
    schur = c_theta - (b_unit / a_unit[:, None]).T @ b_unit
    bread = np.linalg.inv(schur)
    # Each unit lies in one cluster and its intercept's score is zero at the
    # optimum, so the cluster scores of the terms are the whole meat.
    scores = _cell_sums(residual[:, None] * d.X, d.cluster, counts["n_clusters"])
    n_clusters = counts["n_clusters"]
    meat = scores.T @ scores * _cr1_factor(n_clusters, len(d.y), d.X.shape[1])
    cov = bread @ meat @ bread
    coef[identified] = theta
    # One cluster's score is zero at the optimum: no clustered SE.
    se_cluster[identified] = np.sqrt(np.diag(cov)) if n_clusters > 1 else np.nan
    se_model[identified] = np.sqrt(np.diag(bread))
    return StopContextFit(
        names=names,
        coef=coef,
        se_cluster=se_cluster,
        se_model=se_model,
        converged=converged,
        n_iter=n_iter,
        counts=counts,
    )


def _cr1_factor(n_clusters: int, n_rows: int, n_terms: int) -> float:
    """The CR1 small-sample factor ``G / (G - 1) * (N - 1) / (N - K)``.

    ``K`` counts the context and overlap terms only. The unit intercepts (and,
    for the conditional estimator, the (unit, stop) baselines) are absorbed
    and nested within clusters, since a unit is sorted once per recording day;
    nested fixed effects are not counted in ``K`` (Cameron and Miller 2015,
    *J. Human Resources* 50:317, section IV.B). With ``N`` in the hundreds of
    thousands the second factor is 1 to many digits; ``G / (G - 1)`` is the
    part that matters with about ten clusters.
    """
    if n_clusters < 2:
        return 1.0
    return n_clusters / (n_clusters - 1) * (n_rows - 1) / max(n_rows - n_terms, 1)


def _varies_within_units(values: NDArray, unit: NDArray) -> bool:
    """Whether ``values`` takes more than one value within some unit."""
    if not len(values):
        return False
    lowest = np.full(unit.max() + 1, np.inf)
    highest = np.full(unit.max() + 1, -np.inf)
    np.minimum.at(lowest, unit, values)
    np.maximum.at(highest, unit, values)
    return bool(np.any(highest > lowest))


def _subset_design(d: StopContextDesign, keep: NDArray) -> StopContextDesign:
    """Rows ``keep`` of a design, with units, cells and clusters re-coded."""
    unit, _ = pd.factorize(d.unit[keep])
    cell_old = d.cell[keep]
    cell, cell_values = pd.factorize(cell_old)
    cluster, _ = pd.factorize(d.cluster[keep])
    first = np.unique(cell, return_index=True)[1]
    return StopContextDesign(
        y=d.y[keep],
        exposure=d.exposure[keep],
        X=d.X[keep],
        names=d.names,
        unit=unit,
        cell=cell,
        cell_unit=unit[first],
        baseline_count=d.baseline_count[np.asarray(cell_values)],
        baseline_time=d.baseline_time[np.asarray(cell_values)],
        cluster=cluster,
    )


def _initial_alpha(d: StopContextDesign, n_units: int) -> NDArray:
    """Each unit's log ratio of post-window to baseline-expected spikes."""
    rate = d.baseline_count / d.baseline_time
    expected = d.exposure * rate[d.cell]
    observed = np.bincount(d.unit, weights=d.y, minlength=n_units)
    predicted = np.bincount(d.unit, weights=expected, minlength=n_units)
    return np.log(np.maximum(observed, 0.5) / np.maximum(predicted, 1e-12))


# ---------------------------------------------------------------------------
# Summary across sessions and animals
# ---------------------------------------------------------------------------


def summarize_stop_context(
    frame: pd.DataFrame,
    *,
    estimator: str = "conditional",
    terms: tuple[str, ...] = (*CONTEXT_TERMS, OVERLAP_TERM),
    cluster: str = "day",
) -> pd.DataFrame:
    """Pooled fit, per-animal fits and leave-one-animal-out fits of one frame.

    Parameters
    ----------
    frame : pd.DataFrame
        From :func:`model_frame` (one class, one sign, one unit set), with
        ``animal``.
    estimator : {"conditional", "plugin"}
        Passed to :func:`fit_stop_context_glm`.
    terms : tuple of str
        Terms of the model.
    cluster : str
        The column the standard errors cluster on (:func:`stop_context_design`).

    Returns
    -------
    pd.DataFrame, one row per term
        ``term``, ``coef``, ``exp_coef``, ``se_cluster`` (clustered by
        recording day), ``ci_df`` (``G - 1``), ``ci_low``, ``ci_high``
        (``exp(coef -/+ t_{0.975, G-1} se_cluster)``), ``se_model``,
        ``converged``, ``exp_coef_<animal>`` (per-animal fits),
        ``n_animals``, ``n_animals_estimated``, ``n_animals_pooled_sign``,
        ``n_loao_estimated`` (leave-one-animal-out fits with an estimate),
        ``loao_exp_coef_min`` / ``_max`` (their range),
        ``loao_exp_coef_<animal>`` (fit without that animal), and the pooled
        fit's counts (``n_rows``, ``n_units``, ``n_cells``, ``n_clusters``, the
        uninformative units) plus ``n_events``, ``n_stops``, ``n_sessions``,
        ``n_days``.
    """
    animals = sorted(frame["animal"].unique())
    pooled = fit_stop_context_glm(
        stop_context_design(frame, terms, cluster=cluster), estimator=estimator
    )
    per_animal = {
        animal: fit_stop_context_glm(
            stop_context_design(
                frame.loc[frame["animal"].eq(animal)], terms, cluster=cluster
            ),
            estimator=estimator,
        )
        for animal in animals
    }
    leave_out = (
        {
            animal: fit_stop_context_glm(
                stop_context_design(
                    frame.loc[~frame["animal"].eq(animal)], terms, cluster=cluster
                ),
                estimator=estimator,
            )
            for animal in animals
        }
        if len(animals) > 1
        else {}
    )
    df = pooled.counts["n_clusters"] - 1
    quantile = stats.t.ppf(0.975, df) if df > 0 else np.nan
    rows = []
    for index, term in enumerate(pooled.names):
        coef = pooled.coef[index]
        se = pooled.se_cluster[index]
        row: dict[str, Any] = {
            "term": term,
            "coef": coef,
            "exp_coef": np.exp(coef),
            "se_cluster": se,
            "ci_df": df,
            "ci_low": np.exp(coef - quantile * se),
            "ci_high": np.exp(coef + quantile * se),
            "se_model": pooled.se_model[index],
            "converged": pooled.converged
            and all(
                fit.converged for fit in (*per_animal.values(), *leave_out.values())
            ),
        }
        animal_coef = np.array([per_animal[a].coef[index] for a in animals])
        for animal, value in zip(animals, animal_coef, strict=True):
            row[f"exp_coef_{animal}"] = np.exp(value)
        finite = np.isfinite(animal_coef)
        row["n_animals"] = len(animals)
        row["n_animals_estimated"] = int(finite.sum())
        row["n_animals_pooled_sign"] = int(
            (np.sign(animal_coef[finite]) == np.sign(coef)).sum()
        )
        loao = np.array([leave_out[a].coef[index] for a in leave_out])
        # A failed leave-out would hide inside nanmin/nanmax: count them.
        row["n_loao_estimated"] = int(np.isfinite(loao).sum())
        row["loao_exp_coef_min"] = (
            float(np.exp(np.nanmin(loao))) if np.isfinite(loao).any() else np.nan
        )
        row["loao_exp_coef_max"] = (
            float(np.exp(np.nanmax(loao))) if np.isfinite(loao).any() else np.nan
        )
        for animal in leave_out:
            row[f"loao_exp_coef_{animal}"] = np.exp(leave_out[animal].coef[index])
        row.update(pooled.counts)
        row["n_events"] = int(frame.drop_duplicates(["session", "event_id"]).shape[0])
        row["n_stops"] = int(frame.drop_duplicates(["session", "stop_idx"]).shape[0])
        row["n_sessions"] = int(frame["session"].nunique())
        row["n_days"] = int(frame[cluster].nunique())
        rows.append(row)
    return pd.DataFrame(rows)
