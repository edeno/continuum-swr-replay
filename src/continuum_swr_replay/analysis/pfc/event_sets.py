"""Fit masks, event sets and basis coverage for one PFC session.

The immobility bins a session's GLMs are fit on, and the four event sets
(SWR, replay, and the two replay/SWR classes) its design matrices are
keyed on, restricted to the shared stop cohort.

An extended event set adds a fifth class, ``SWR_no_non_local``: the kept,
cohort-restricted SWRs that overlap no canonical non-local event of the
session (whether or not that event is in the cohort) and no window the
analysed-time rule dropped (:func:`swr_without_non_local`). It is a subset of
``SWR``, so every window a matched control must avoid is unchanged by it.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.events.cohort_selection import (
    events_in_stops,
    stop_interval_mask,
)
from continuum_swr_replay.intervals import (
    interval_mask,
    merge_intervals,
    overlaps_open,
    pad_intervals,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

logger = logging.getLogger(__name__)


def build_session_fit_mask(
    position_info: pd.DataFrame,
    *,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    immobility_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    excluded_intervals: pd.DataFrame | None = None,
    basis_half_window_s: float | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Bins the immobility GLMs are fit on, for one session.

    Three restrictions compose. Immobility is the long-standing one. The stop
    cohort keeps only the time the animal spends stopped at a well, which is
    the population the hippocampal rate and content ladders are read on, so a
    PFC claim and a replay claim describe the same stops. Excluded time (the
    artifact mask widened by 0.3 s) is not analysed: a bin whose peri-event
    basis window (``+/- basis_half_window_s``) touches it is not fit, so no
    fitted bin's design reads excluded time.

    Parameters
    ----------
    position_info : pandas.DataFrame
        Time-indexed position table carrying ``head_speed``. The index is
        absolute seconds, the same clock the stop table's bounds are on.
    cohort_stops : pandas.DataFrame, optional
        Stops kept by
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.
    stop_cohort : str
        Cohort name, recorded in the returned counts. ``"all_stops"`` means *no
        stop restriction at all* -- every immobile bin in the epoch is fit,
        reproducing the pre-cohort mask exactly.
    immobility_threshold : float
        Speed threshold for immobility (cm/s).
    excluded_intervals : pandas.DataFrame, optional
        The session's excluded time (``analysed_time.excluded_intervals``):
        ``start_time`` and ``end_time``, absolute seconds, closed. None
        excludes nothing.
    basis_half_window_s : float, optional
        Half-width of the peri-event basis, in seconds;
        :data:`DEFAULT_BASIS_HALF_WINDOW_S` when None.

    Returns
    -------
    is_immobile : pandas.Series of bool, shape (n_time,)
        Immobility alone; still the definition of the *running* comparison.
    fit_mask : pandas.Series of bool, shape (n_time,)
        ``is_immobile & in_cohort_stop`` (``is_immobile`` for ``all_stops``),
        less the bins whose basis window touches excluded time.
    counts : dict
        ``stop_cohort``, ``n_bins_total``, ``n_bins_immobile``, ``n_bins_fit``,
        and ``n_bins_excluded``: the bins immobility and the stop cohort would
        fit that are not fit because their basis window touches excluded time
        (0 without it).

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    is_immobile = position_info.head_speed < immobility_threshold
    if stop_cohort == "all_stops":
        fit_mask = is_immobile
    else:
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        in_cohort = stop_interval_mask(
            position_info.index.to_numpy(dtype=float), cohort_stops
        )
        fit_mask = is_immobile & pd.Series(in_cohort, index=position_info.index)
    n_bins_excluded = 0
    if excluded_intervals is not None and len(excluded_intervals):
        half_window = (
            DEFAULT_BASIS_HALF_WINDOW_S
            if basis_half_window_s is None
            else basis_half_window_s
        )
        reached = interval_mask(
            position_info.index.to_numpy(dtype=float),
            pad_intervals(excluded_intervals, half_window),
        )
        n_bins_excluded = int((fit_mask.to_numpy() & reached).sum())
        fit_mask = fit_mask & pd.Series(~reached, index=position_info.index)
    counts = {
        "stop_cohort": stop_cohort,
        "n_bins_total": int(len(is_immobile)),
        "n_bins_immobile": int(is_immobile.sum()),
        "n_bins_fit": int(fit_mask.sum()),
        "n_bins_excluded": n_bins_excluded,
    }
    return is_immobile, fit_mask, counts


#: Half-width of the peri-event spline basis, in seconds. The fit window is
#: +/-125 samples at 500 Hz; a dropped event within this of a fitted bin is an
#: event whose basis would have covered bins the model does fit.
DEFAULT_BASIS_HALF_WINDOW_S = 0.25


def count_events_covering_fit_bins(
    onsets: NDArray,
    times: NDArray,
    fit_mask: NDArray,
    *,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> int:
    """How many of ``onsets`` have a fitted bin within the basis half-window.

    The cohort restriction drops events by onset, but the regressor basis spans
    +/-250 ms around each event, so an event dropped just before a stop's
    leading edge would have covered bins that *are* fit. Those bins are then fit
    with that event's contribution set to zero. This counts the events for which
    that is true, so a run can state the size of the effect rather than assume
    it away.

    Parameters
    ----------
    onsets : numpy.ndarray, shape (n_events,)
        Onset times of the dropped events, in seconds.
    times : numpy.ndarray, shape (n_time,)
        Sample times, ascending, on the same clock.
    fit_mask : numpy.ndarray of bool, shape (n_time,)
        The bins the models are fit on.
    basis_half_window_s : float
        Half-width of the peri-event basis, in seconds.

    Returns
    -------
    int
        Number of onsets with at least one fitted bin in
        ``[onset - basis_half_window_s, onset + basis_half_window_s]``.
    """
    onsets = np.asarray(onsets, dtype=float)
    if onsets.size == 0:
        return 0
    # Prefix sums make the per-event window test a pair of searchsorted lookups
    # rather than a scan over the time axis.
    fitted_before = np.concatenate(
        [[0], np.cumsum(np.asarray(fit_mask, dtype=bool), dtype=np.int64)]
    )
    first = np.searchsorted(times, onsets - basis_half_window_s, side="left")
    last = np.searchsorted(times, onsets + basis_half_window_s, side="right")
    return int((fitted_before[last] - fitted_before[first] > 0).sum())


def _basis_coverage_counts(
    dropped_non_local_onsets: NDArray,
    dropped_ripple_onsets: NDArray,
    *,
    kept_non_local: int,
    kept_ripple: int,
    time: NDArray,
    fit_mask: pd.Series | None,
    basis_half_window_s: float,
) -> dict[str, Any]:
    """The leading-edge coverage diagnostic, as four report fields.

    Named per event class and expressed both as a count and as a fraction of the
    events that were kept, because the count alone is not interpretable without
    the cohort's size.
    """
    if fit_mask is None:
        return {
            "n_nonlocal_dropped_overlapping_fit": None,
            "n_ripple_dropped_overlapping_fit": None,
            "n_nonlocal_dropped_overlapping_fit_frac_of_kept": None,
            "n_ripple_dropped_overlapping_fit_frac_of_kept": None,
        }
    mask = np.asarray(fit_mask, dtype=bool)
    n_non_local = count_events_covering_fit_bins(
        dropped_non_local_onsets,
        time,
        mask,
        basis_half_window_s=basis_half_window_s,
    )
    n_ripple = count_events_covering_fit_bins(
        dropped_ripple_onsets, time, mask, basis_half_window_s=basis_half_window_s
    )
    return {
        "n_nonlocal_dropped_overlapping_fit": n_non_local,
        "n_ripple_dropped_overlapping_fit": n_ripple,
        "n_nonlocal_dropped_overlapping_fit_frac_of_kept": (
            n_non_local / kept_non_local if kept_non_local else None
        ),
        "n_ripple_dropped_overlapping_fit_frac_of_kept": (
            n_ripple / kept_ripple if kept_ripple else None
        ),
    }


def _restrict_to_frame(
    events: pd.DataFrame, position_time: pd.Index, *, what: str
) -> tuple[pd.DataFrame, int]:
    """Drop events not wholly inside the aligned position frame, and count them.

    The canonical events were built on the decode's own time axis. A session
    whose decode and position axes were trimmed to their shared run could in
    principle carry an event in a dropped tail; such an event has no bins to be
    modelled on, so keeping it would put an onset outside the design matrix.
    The 27 trimmed sessions were checked and none has one, so a non-zero count
    here is a data change worth seeing rather than routine attrition.
    """
    if events.empty:
        return events, 0
    first, last = float(position_time[0]), float(position_time[-1])
    inside = (events.start_time.to_numpy(dtype=float) >= first) & (
        events.end_time.to_numpy(dtype=float) <= last
    )
    n_outside = int((~inside).sum())
    if n_outside:
        logger.warning(
            "%d of %d %s lie outside the aligned position frame [%f, %f] and "
            "were dropped; the decode and position axes disagree over more "
            "than an empty tail",
            n_outside,
            len(events),
            what,
            first,
            last,
        )
    return events.loc[inside].reset_index(drop=True), n_outside


def build_session_event_sets(
    data: Mapping[str, Any],
    results: Any,
    *,
    canonical_events: pd.DataFrame,
    swr_intervals: pd.DataFrame,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    fit_mask: pd.Series | None = None,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build the four event sets the PFC design matrices are keyed on.

    Both populations are canonical and are read, not re-detected here: the
    non-local events are rows of the shipped event table and the SWR intervals
    are the SWR source's speed-gated ``SWR_event`` labels (the direct Kay
    inventory, :mod:`continuum_swr_replay.analysis.events.swr_source`), both
    loaded by :mod:`continuum_swr_replay.analysis.pfc.canonical_events`. The SWR
    split is
    the event table's own ``is_swr`` column -- strict positive-duration overlap
    with those same labels -- so a Figure 4 event and a content- or rate-family
    event with the same ``event_id`` are the same event, classified the same
    way.

    Events are restricted to the stop cohort by onset, BEFORE the split is
    read, so ``non_local_SWR`` and ``non_local_no_SWR`` partition exactly the
    events that are kept.

    Parameters
    ----------
    data : Mapping
        Session data from ``data_loaders.load_data``; only ``position_info`` is
        read.
    results : xarray.Dataset
        HPC decoding results, used only to align the position frame.
    canonical_events : pandas.DataFrame
        The session's rows of the canonical event table, from
        :func:`~continuum_swr_replay.analysis.pfc.canonical_events.load_session_canonical_events`:
        ``event_id``, ``start_time``, ``end_time``, ``is_swr``,
        ``occupancy_state``. Passed in rather than loaded here so this function
        stays testable without the shipped tables.
    swr_intervals : pandas.DataFrame
        The session's speed-gated SWR intervals, from
        :func:`~continuum_swr_replay.analysis.pfc.canonical_events.load_session_swr_intervals`:
        ``start_time``, ``end_time``.
    cohort_stops : pandas.DataFrame, optional
        Stops kept for this session; required unless ``stop_cohort`` is
        ``"all_stops"``.
    stop_cohort : str
        Cohort name; ``"all_stops"`` applies no restriction.
    fit_mask : pandas.Series of bool, optional
        The bins the models will be fit on. Used only to measure how many
        dropped events would have had basis coverage over fitted bins; when
        omitted those counts are ``None``.
    basis_half_window_s : float
        Half-width of the peri-event basis used for that measurement.

    Returns
    -------
    event_times : dict of str to pandas.DataFrame
        ``non_local_no_SWR``, ``non_local_SWR``, ``non_local``, ``SWR``.
    counts : dict
        ``stop_cohort`` and the event counts before and after the restriction
        (``n_nonlocal_before``/``n_nonlocal_after``,
        ``n_ripple_before``/``n_ripple_after``), the split
        (``n_nonlocal_swr``, ``n_nonlocal_no_swr``), the leading-edge
        coverage diagnostic ``n_nonlocal_dropped_overlapping_fit`` /
        ``n_ripple_dropped_overlapping_fit`` with their fractions of the kept
        counts (``*_frac_of_kept``), and how many canonical events and SWR
        intervals fell outside the aligned position frame
        (``n_events_outside_frame``, ``n_swr_intervals_outside_frame``).

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    # Deferred import: this helper opens Spyglass / DataJoint at import time.
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )

    # Idempotent when the caller already aligned the pair (run_session does):
    # the shared run is then the whole of both axes and this is the identity.
    _, position_info, _ = align_results_to_position(results, data["position_info"])
    position_time = position_info.index

    non_local_times, n_events_outside_frame = _restrict_to_frame(
        canonical_events, position_time, what="canonical events"
    )
    swr_intervals, n_swr_outside_frame = _restrict_to_frame(
        swr_intervals, position_time, what="SWR intervals"
    )

    counts: dict[str, Any] = {
        "stop_cohort": stop_cohort,
        "n_events_outside_frame": n_events_outside_frame,
        "n_swr_intervals_outside_frame": n_swr_outside_frame,
        "n_nonlocal_before": int(len(non_local_times)),
        "n_ripple_before": int(len(swr_intervals)),
    }
    dropped_non_local_onsets = np.empty(0, dtype=float)
    dropped_ripple_onsets = np.empty(0, dtype=float)
    if stop_cohort != "all_stops":
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        keep_non_local = events_in_stops(
            non_local_times.start_time.to_numpy(dtype=float), cohort_stops
        )
        keep_ripple = events_in_stops(
            swr_intervals.start_time.to_numpy(dtype=float), cohort_stops
        )
        dropped_non_local_onsets = non_local_times.start_time.to_numpy(dtype=float)[
            ~keep_non_local
        ]
        dropped_ripple_onsets = swr_intervals.start_time.to_numpy(dtype=float)[
            ~keep_ripple
        ]
        non_local_times = non_local_times.loc[keep_non_local]
        swr_intervals = swr_intervals.loc[keep_ripple]
    counts["n_nonlocal_after"] = int(len(non_local_times))
    counts["n_ripple_after"] = int(len(swr_intervals))
    counts.update(
        _basis_coverage_counts(
            dropped_non_local_onsets,
            dropped_ripple_onsets,
            kept_non_local=counts["n_nonlocal_after"],
            kept_ripple=counts["n_ripple_after"],
            time=position_time.to_numpy(dtype=float),
            fit_mask=fit_mask,
            basis_half_window_s=basis_half_window_s,
        )
    )

    # The split is read from the canonical table, not recomputed against the
    # interval list: the column already encodes strict positive-duration
    # overlap with exactly these speed-gated labels, and recomputing it here
    # would let the PFC path drift from the content and rate families again.
    is_swr = non_local_times.is_swr.to_numpy(dtype=bool)
    non_local_swr_times = non_local_times.loc[is_swr].reset_index(drop=True)
    non_local_no_swr_times = non_local_times.loc[~is_swr].reset_index(drop=True)
    counts["n_nonlocal_swr"] = int(len(non_local_swr_times))
    counts["n_nonlocal_no_swr"] = int(len(non_local_no_swr_times))

    # Both populations use the same whole-event immobility definition, so the
    # SWR-controlled decomposition cannot reintroduce a movement-spanning SWR
    # as a secondary event source.
    event_times = {
        "non_local_no_SWR": non_local_no_swr_times,
        "non_local_SWR": non_local_swr_times,
        "non_local": non_local_times,
        "SWR": swr_intervals,
    }
    return event_times, counts


#: The event classes the design matrices are keyed on. A matched control may
#: not overlap a window of any of them -- including ``SWR``, whose windows are
#: not all covered by the non-local classes.
EVENT_CLASSES = ("non_local_no_SWR", "non_local_SWR", "non_local", "SWR")

#: The extended set's fifth class: SWRs that carry no non-local event.
SWR_NO_NON_LOCAL = "SWR_no_non_local"

#: The event classes of the extended event set, in design order.
EXTENDED_EVENT_CLASSES = (*EVENT_CLASSES, SWR_NO_NON_LOCAL)

#: The event sets a run can be cut with: the canonical four classes, or those
#: plus ``SWR_no_non_local``.
EVENT_SETS = ("canonical", "with_swr_without_replay")


def swr_without_non_local(
    swr_intervals: pd.DataFrame,
    non_local_events: pd.DataFrame,
    unknown_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """The SWRs that overlap no non-local event.

    Overlap is the open test (:func:`~continuum_swr_replay.intervals.overlaps_open`),
    which agrees with the strict positive-duration overlap that defines the
    canonical ``is_swr`` for intervals of positive duration. An SWR that
    overlaps no non-local event but does overlap a window the analysed-time
    rule dropped has an unknown class -- the dropped window may have held a
    non-local event -- and is left out, and counted.

    Parameters
    ----------
    swr_intervals : pandas.DataFrame
        The kept, cohort-restricted SWRs: ``start_time``, ``end_time``,
        seconds.
    non_local_events : pandas.DataFrame
        Every canonical non-local event of the session, in the cohort or not:
        ``start_time``, ``end_time``.
    unknown_windows : pandas.DataFrame
        The windows the analysed-time rule dropped (``start_time``,
        ``end_time``), from
        :func:`~continuum_swr_replay.analysis.pfc.matched_control.artifact_dropped_events`.

    Returns
    -------
    swr_no_non_local : pandas.DataFrame
        The SWRs of ``swr_intervals`` that overlap neither, in their order,
        with a fresh index.
    counts : dict
        ``n_swr_with_non_local``, ``n_swr_no_non_local`` and
        ``n_swr_no_non_local_unknown``; they sum to ``len(swr_intervals)``.
    """

    def windows(frame: pd.DataFrame) -> NDArray:
        return merge_intervals(
            frame["start_time"].to_numpy(dtype=float),
            frame["end_time"].to_numpy(dtype=float),
        )

    starts = swr_intervals["start_time"].to_numpy(dtype=float)
    ends = swr_intervals["end_time"].to_numpy(dtype=float)
    replay = windows(non_local_events)
    dropped = windows(unknown_windows)
    overlaps_replay = overlaps_open(starts, ends, replay[:, 0], replay[:, 1])
    unknown = ~overlaps_replay & overlaps_open(
        starts, ends, dropped[:, 0], dropped[:, 1]
    )
    keep = ~overlaps_replay & ~unknown
    return swr_intervals.loc[keep].reset_index(drop=True), {
        "n_swr_with_non_local": int(overlaps_replay.sum()),
        "n_swr_no_non_local": int(keep.sum()),
        "n_swr_no_non_local_unknown": int(unknown.sum()),
    }
