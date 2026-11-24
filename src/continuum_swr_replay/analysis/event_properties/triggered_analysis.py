"""Event-triggered analysis: signal windows around events, and their averages."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import sem

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals,
)
from continuum_swr_replay.analysis.event_definition.no_spike_periods import (
    level_outside_periods,
    no_spike_periods,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.event_analysis import (
    get_event_start_end_times,
)
from continuum_swr_replay.analysis.shared_statistics import animal_medians
from continuum_swr_replay.intervals import overlaps_intervals
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.tables.columns import require_columns

logger = logging.getLogger(__name__)


def get_signal_windows(
    data: pd.DataFrame,
    event_times: pd.DataFrame,
    signal_cols: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    event_id_col: str | None = None,
) -> xr.DataArray:
    """Extract signal windows around event times.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with signal columns and grouping columns
        Must have columns: animal, day, epoch, {signal_cols}
    event_times : pd.DataFrame
        DataFrame with event times and metadata
        Must have columns: animal, day, epoch, {time_col}
    signal_cols : list[str]
        Names of signal columns to extract from data
    time_col : str
        Column in event_times with event times (typically "start_time" or "end_time")
    window_sec : float
        Total window duration in seconds (centered on event time)
    sampling_frequency : float
        Sampling rate in Hz
    event_id_col : str | None
        Column name in event_times for event IDs (for coordinate labeling)
        If None, sequential integers are used

    Returns
    -------
    xr.DataArray
        Shape: (n_events, n_timepoints, n_signals)
        Dimensions: ("event", "time", "signal")
        Coordinates:
            - event: MultiIndex from event_times if event_id_col provided, else integers
            - time: Time relative to event in seconds (negative = before, positive = after)
            - signal: Signal names from signal_cols

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid

    Notes
    -----
    - Excluded time (the artifact mask widened by 0.3 s) is not analyzed: an
      event whose window touches the epoch's excluded time
      (``analyzed_time.excluded_intervals``; closed) is dropped whole, and its row
      is not returned.
    - Windows are centered on event times
    - Missing data filled with NaN (occurs for events near data boundaries)
    - Events outside the data time range will have partial or fully NaN windows
    - Empty sessions (no events for a given animal/day/epoch) are logged and skipped
    - If ALL sessions have no events, returns an empty DataArray with shape (0, window_len, n_signals)
    - The empty DataArray maintains proper coordinates for time and signal dimensions

    Examples
    --------
    >>> signal_windows = get_signal_windows(
    ...     data_over_time,
    ...     non_local_times,
    ...     signal_cols=["theta_power_zscore", "SWR_zscore", "head_speed"],
    ...     time_col="start_time",
    ...     window_sec=2.0,
    ... )
    >>> # Average across events
    >>> mean_triggered = signal_windows.mean(dim="event")
    >>> # Plot
    >>> mean_triggered.sel(signal="SWR_zscore").plot()

    """
    # Validate inputs
    if sampling_frequency <= 0:
        raise ValueError(
            f"sampling_frequency must be positive, got {sampling_frequency}"
        )
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")

    require_columns(data, ["animal", "day", "epoch", *signal_cols], "data")
    required_event_cols = ["animal", "day", "epoch", time_col]
    if event_id_col is not None:
        required_event_cols.append(event_id_col)
    require_columns(event_times, required_event_cols, "event_times")

    # Compute window parameters
    half_window_sec = window_sec / 2.0
    dt = 1 / sampling_frequency
    half_window_samples = int(round(half_window_sec / dt))
    window_len = 2 * half_window_samples + 1
    offsets = np.arange(-half_window_samples, half_window_samples + 1)

    all_windows = []
    event_id_list: list[
        tuple[Any, Any, Any, Any] | int
    ] = []  # Mixed type for flexibility

    for (animal, day, epoch), group in data.groupby(["animal", "day", "epoch"]):
        # Get events for this session
        session_events = event_times.loc[
            (event_times.animal == animal)
            & (event_times.day == day)
            & (event_times.epoch == epoch)
        ]

        if len(session_events) == 0:
            logger.debug(f"No events found for {animal}/{day}/{epoch}, skipping")
            continue

        excluded = excluded_intervals(swr_epoch_id_of_session(animal, day, epoch))
        if len(excluded):
            centers = session_events[time_col].to_numpy(dtype=float)
            touches = overlaps_intervals(
                centers - half_window_sec, centers + half_window_sec, excluded
            )
            if touches.any():
                logger.info(
                    "%d of %d windows for %s/%s/%s touch excluded time and are dropped",
                    int(touches.sum()),
                    len(touches),
                    animal,
                    day,
                    epoch,
                )
                session_events = session_events.loc[~touches]
            if len(session_events) == 0:
                continue

        times = session_events[time_col].to_numpy()
        time_index = group.index.values
        time_min, time_max = time_index[0], time_index[-1]

        # Find the *nearest* sample to each event time. ``searchsorted``
        # alone returns the insertion point (the next sample), which
        # for an event at 0.0025 s on a 2 ms grid centers the window
        # on 0.004 s instead of 0.002 s. Compare distances to the
        # neighbors bracketing the insertion point and pick the
        # closer one.
        right = np.searchsorted(time_index, times)
        right_clipped = np.clip(right, 0, len(time_index) - 1)
        left_clipped = np.clip(right - 1, 0, len(time_index) - 1)
        left_d = np.abs(times - time_index[left_clipped])
        right_d = np.abs(times - time_index[right_clipped])
        center_idxs = np.where(left_d <= right_d, left_clipped, right_clipped)

        # Check if events are within extractable range (accounting for window size)
        # Events exactly at boundaries or outside need special handling
        extractable = (times >= time_min + half_window_sec) & (
            times <= time_max - half_window_sec
        )

        # Warn about events with incomplete windows
        if (~extractable).any():
            n_partial = (~extractable).sum()
            logger.warning(
                f"{n_partial} events for {animal}/{day}/{epoch} "
                f"outside extractable range [{time_min + half_window_sec:.2f}, "
                f"{time_max - half_window_sec:.2f}], will have partial windows"
            )

        # Create window indices
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        # Check bounds - both index bounds AND temporal bounds
        # For events way outside the data range, even "valid" indices might be wrong
        mask = (all_window_idxs >= 0) & (all_window_idxs < len(group))

        # Also check if the actual time difference is reasonable
        # This catches events that are far outside the data range
        for event_idx in range(len(times)):
            center_time = time_index[center_idxs[event_idx]]
            event_time = times[event_idx]
            # If event is more than half window away from closest data point, mask all
            if abs(event_time - center_time) > half_window_sec:
                mask[event_idx, :] = False

        # Initialize windows with NaN
        n_events_session = len(times)
        n_signals = len(signal_cols)
        windows = np.full((n_events_session, window_len, n_signals), np.nan)

        # Fill valid data
        valid_rows, valid_cols = np.where(mask)
        valid_indices = all_window_idxs[mask]

        for i, signal_col in enumerate(signal_cols):
            windows[valid_rows, valid_cols, i] = group[signal_col].to_numpy()[
                valid_indices
            ]

        all_windows.append(windows)

        # Track event IDs
        if event_id_col is not None:
            # Store tuples for MultiIndex
            for _, row in session_events.iterrows():
                event_id_list.append(
                    (row["animal"], row["day"], row["epoch"], row[event_id_col])
                )
        else:
            # Store sequential integers - convert range to list for type compatibility
            event_id_list.extend(
                list(range(len(event_id_list), len(event_id_list) + n_events_session))
            )

    if not all_windows:
        # Return empty array with proper structure
        logger.warning("No events found in any session, returning empty array")
        return xr.DataArray(
            np.empty((0, window_len, len(signal_cols))),
            dims=("event", "time", "signal"),
            coords={
                "event": [],
                "time": offsets * dt,
                "signal": signal_cols,
            },
        )

    # Concatenate all sessions
    all_windows_arr = np.vstack(all_windows)
    time_axis = offsets * dt

    # Create proper index for events
    if event_id_col is not None:
        # Use pandas MultiIndex from tuples
        event_index = pd.MultiIndex.from_tuples(
            event_id_list, names=["animal", "day", "epoch", event_id_col]
        )
    else:
        # Use RangeIndex for consistency with project patterns (see event_definition/nonlocal_event_rule.py)
        event_index = pd.RangeIndex(start=0, stop=len(event_id_list), name="event")

    return xr.DataArray(
        all_windows_arr,
        dims=("event", "time", "signal"),
        coords={
            "event": event_index,
            "time": time_axis,
            "signal": signal_cols,
        },
    )


def compute_triggered_average_curves(
    data_over_time: pd.DataFrame,
    event_types: dict,
    signals: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
) -> dict:
    """Reduce ``data_over_time`` to event-triggered curves: mean, SEM and quartiles.

    Splits the work that :func:`plot_triggered_averages` used to do inline
    into a small, cacheable product: for each signal, the time axis and the
    across-event mean and SEM for non-SWR (no-overlap) vs SWR-coincident
    (overlap) immobile non-local events. The returned dict is a few KB,
    letting callers cache it instead of the multi-GB ``data_over_time``.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated per-timebin dataset.
    event_types : dict
        Output of :func:`get_event_types`.
    signals : list of str
        Signal columns to reduce, in order.
    time_col : str
        Alignment column ('start_time' or 'end_time').
    window_sec : float
        Half-window around each event, in seconds.

    Returns
    -------
    dict
        ``{signal: {"time", "no_overlap_mean", "no_overlap_sem",
        "overlap_mean", "overlap_sem", and for each class the animal-balanced
        quartiles "{class}_q25", "{class}_median", "{class}_q75"}}`` with
        numpy-array values. The mean and SEM are over every event; each
        quartile is the mean over animals of each animal's quartile over its
        events (:func:`animal_balanced_quartiles`), so the median is the mean
        of :func:`animal_window_medians`' lines. A lag where no event has a
        value is NaN.

    """
    signal_types = list(signals)
    windows = triggered_windows(
        data_over_time, event_types, signal_types, time_col, window_sec
    )
    windows_no, windows_ov = windows["no_overlap"], windows["overlap"]
    no_mean_all = windows_no.mean(dim="event")
    ov_mean_all = windows_ov.mean(dim="event")

    curves: dict = {}
    for signal_type in signal_types:
        no_mean = no_mean_all.sel(signal=signal_type)
        quartiles = {
            f"{name}_{label}": values
            for name, windows in (("no_overlap", windows_no), ("overlap", windows_ov))
            for label, values in animal_balanced_quartiles(
                windows.sel(signal=signal_type).values,
                windows.indexes["event"].get_level_values("animal"),
            ).items()
        }
        curves[signal_type] = {
            **quartiles,
            "time": no_mean["time"].values,
            "no_overlap_mean": no_mean.values,
            "no_overlap_sem": sem(
                windows_no.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
            "overlap_mean": ov_mean_all.sel(signal=signal_type).values,
            "overlap_sem": sem(
                windows_ov.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
        }
    return curves


def triggered_windows(
    data_over_time: pd.DataFrame,
    event_types: dict,
    signals: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
) -> dict[str, xr.DataArray]:
    """The signal windows around the immobile non-local events, by SWR overlap:
    what :func:`compute_triggered_average_curves` reduces.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Per-timebin decode frames (one epoch's or the aggregated ones).
    event_types : dict
        Output of :func:`get_event_types` of the same frames.
    signals : list of str
        Signal columns, in order.
    time_col : str
        Alignment column (``"start_time"`` or ``"end_time"``).
    window_sec : float
        The window around each event, in seconds (half on each side).

    Returns
    -------
    dict of str to xr.DataArray
        ``"no_overlap"`` (events with no SWR) and ``"overlap"`` (events that
        overlap one): :func:`get_signal_windows`' windows, dims ``(event,
        time, signal)``, the event index carrying ``animal``, ``day``,
        ``epoch`` and ``non_local_event``.
    """
    return {
        key: get_signal_windows(
            data=data_over_time,
            event_times=get_event_start_end_times(
                event_types[name], event_col="non_local_event"
            ),
            signal_cols=list(signals),
            time_col=time_col,
            window_sec=window_sec,
            sampling_frequency=SAMPLING_FREQUENCY,
            event_id_col="non_local_event",
        )
        for key, name in (
            ("no_overlap", "non_local_event_immobile_no_overlap_SWR_event"),
            ("overlap", "non_local_event_immobile_overlap_SWR_event"),
        )
    }


def animal_window_medians(windows: xr.DataArray, signal: str) -> pd.DataFrame:
    """Each animal's median over its events of one signal's windows, at each
    lag (:func:`animal_quartiles`' median).

    Parameters
    ----------
    windows : xr.DataArray, dims (event, time, signal)
        :func:`get_signal_windows`' windows, the event index carrying
        ``animal``.
    signal : str
        The signal.

    Returns
    -------
    pd.DataFrame, shape (n_lags, n_animals)
        Indexed by lag (s), a column per animal (sorted): the median over the
        animal's events with a value at that lag (NaN where none has one).
    """
    return animal_quartiles(
        windows.sel(signal=signal).values,
        windows.indexes["event"].get_level_values("animal"),
    )["median"].set_axis(windows["time"].values)


#: Figure 2g's signals around event starts, in its rows' order, and its window
#: (s, in all: half on each side), which Figure 2's supplement draws per animal.
TRIGGERED_SIGNALS = ("SWR_zscore", "MUA_zscore", "theta_power_zscore", "head_speed")
TRIGGERED_WINDOW_S = 0.3
WINDOW_KINDS = {"overlap": True, "no_overlap": False}
"""The keys of :func:`triggered_windows`' windows and each one's ``is_swr``, in
the figures' drawing order (the SWR events first, under the no-SWR events)."""

#: The ``get_event_types`` entries :func:`compute_triggered_average_curves` reads:
#: the immobile non-local events that do and do not overlap an SWR.
TRIGGERING_EVENT_TYPES = (
    "non_local_event_immobile_no_overlap_SWR_event",
    "non_local_event_immobile_overlap_SWR_event",
)


def triggering_events_in(event_types: dict, events: pd.DataFrame) -> dict:
    """``event_types`` with its triggering non-local events limited to ``events``.

    The decode frames label every immobile non-local event; the triggered
    averages are over the events of the event table's selection
    (``cohort_selection.select_visit_context``). An event of the frames is kept
    when its session and label are those of a row of ``events``.

    Parameters
    ----------
    event_types : dict of str to pd.DataFrame
        ``get_event_types`` of the aggregated frames, whose rows carry
        ``animal``, ``day``, ``epoch`` and ``non_local_event``.
    events : pd.DataFrame
        The selected events, with ``animal``, ``session_date``, ``epoch`` and
        ``non_local_event`` (the frame label of each).

    Returns
    -------
    dict
        A copy of ``event_types`` whose :data:`TRIGGERING_EVENT_TYPES` hold the
        rows of the selected events only; the other entries are unchanged.
    """
    require_columns(
        events,
        ["animal", "session_date", "epoch", "non_local_event"],
        "the selected events",
    )
    keys = pd.MultiIndex.from_arrays(
        [
            events["animal"].astype(str),
            events["session_date"].astype(np.int64),
            events["epoch"].astype(str),
            events["non_local_event"].astype(np.int64),
        ]
    )
    kept = dict(event_types)
    for name in TRIGGERING_EVENT_TYPES:
        frame = event_types[name]
        rows = pd.MultiIndex.from_arrays(
            [
                frame["animal"].astype(str),
                frame["day"].astype(np.int64),
                frame["epoch"].astype(str),
                frame["non_local_event"].astype(np.int64),
            ]
        )
        kept[name] = frame.loc[rows.isin(keys)]
    return kept


QUARTILES = (("q25", 0.25), ("median", 0.5), ("q75", 0.75))
"""The quartiles a triggered trace draws, by name."""


def animal_quartiles(
    windows: np.ndarray, animals: np.ndarray | pd.Index
) -> dict[str, pd.DataFrame]:
    """Each animal's quartiles over its events of event-triggered windows, at
    each lag.

    Parameters
    ----------
    windows : np.ndarray, shape (n_events, n_lags)
        One signal's windows (``get_signal_windows``), NaN where a window has
        no value.
    animals : array_like, shape (n_events,)
        Each window's animal.

    Returns
    -------
    dict of str to pd.DataFrame
        ``q25``, ``median`` and ``q75``, each shape (n_lags, n_animals), a
        column per animal (sorted): the quartile over the animal's events with
        a value at that lag (NaN where none has one).
    """
    animals = np.asarray(animals)
    names = sorted(set(animals))
    return {
        label: pd.DataFrame(
            {
                name: np.nanquantile(windows[animals == name], q, axis=0)
                for name in names
            }
        )
        for label, q in QUARTILES
    }


def animal_balanced_quartiles(
    windows: np.ndarray, animals: np.ndarray | pd.Index
) -> dict[str, np.ndarray]:
    """The animal-balanced quartiles of event-triggered windows, at each lag:
    the mean over animals of each animal's quartile over its events
    (:func:`animal_quartiles`). The band between ``q25`` and ``q75`` stays a
    spread over events, with every animal weighted the same.

    Parameters
    ----------
    windows : np.ndarray, shape (n_events, n_lags)
        One signal's windows, NaN where a window has no value.
    animals : array_like, shape (n_events,)
        Each window's animal.

    Returns
    -------
    dict
        ``q25``, ``median`` and ``q75``, each shape (n_lags,): the mean over
        the animals with a value at that lag; NaN where no animal has one.
    """
    return {
        label: table.mean(axis=1).to_numpy()
        for label, table in animal_quartiles(windows, animals).items()
    }


NO_SPIKE_SIGNAL = "MUA_zscore"
"""The signal drawn around the start of No-Spike periods."""
NO_SPIKE_KIND = "no_spike"
"""The stored windows' kind of a No-Spike period's (:func:`epoch_trigger_windows`)."""
WINDOW_ROW_KEYS = ("kind", "signal", "non_local_event")
"""The stored windows' columns before their lags (:func:`epoch_trigger_windows`)."""


def lag_column(offset: int) -> str:
    """The stored windows' column of a lag of ``offset`` samples."""
    return f"lag{offset:+04d}"


def frame_event_counts(event_types: dict) -> dict[str, int]:
    """The decode frames' immobile non-local events, by SWR overlap.

    Parameters
    ----------
    event_types : dict
        ``get_event_types`` of one or more decode frames.

    Returns
    -------
    dict
        ``swr``, ``non_swr`` and ``total``: the events (one per session and
        label) that overlap an SWR, that do not, and all of them.
    """

    def n_events(frame: pd.DataFrame) -> int:
        return len(
            frame[["animal", "day", "epoch", "non_local_event"]].drop_duplicates()
        )

    return {
        "swr": n_events(event_types["non_local_event_immobile_overlap_SWR_event"]),
        "non_swr": n_events(
            event_types["non_local_event_immobile_no_overlap_SWR_event"]
        ),
        "total": n_events(event_types["non_local_event_immobile"]),
    }


def _window_rows(
    windows: np.ndarray, kind: str, signal: str, labels: np.ndarray
) -> pd.DataFrame:
    """Stored rows of one kind's and signal's windows ``(n, n_lags)``."""
    half = windows.shape[1] // 2
    rows = pd.DataFrame(
        windows,
        columns=[lag_column(offset) for offset in range(-half, half + 1)],
    )
    rows.insert(0, "non_local_event", np.asarray(labels, dtype=np.int64))
    rows.insert(0, "signal", signal)
    rows.insert(0, "kind", kind)
    return rows


def epoch_trigger_windows(
    frame: pd.DataFrame,
    event_types: dict,
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
    window_sec: float = TRIGGERED_WINDOW_S,
) -> tuple[pd.DataFrame, dict]:
    """One epoch's windows around event starts, as Figure 2g and its supplement
    draw them, and its record.

    Every immobile non-local event's window of each of :data:`TRIGGERED_SIGNALS`
    around its start, by SWR overlap (:func:`triggered_windows`; kind
    ``"overlap"`` or ``"no_overlap"``, ``non_local_event`` its frame label), and
    each No-Spike period's window of :data:`NO_SPIKE_SIGNAL` around its start
    (``no_spike_periods.no_spike_periods``: those that start inside a well
    visit; kind :data:`NO_SPIKE_KIND`, ``non_local_event`` the period's place
    in time order). A window that touches excluded time is left out
    (:func:`get_signal_windows`).

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch's decode frame, read with the SWR source.
    event_types : dict
        ``nonlocal_events.get_event_types`` of ``frame``.
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        The epoch's well visits.
    window_sec : float, optional
        The window, in all (s).

    Returns
    -------
    windows : pd.DataFrame
        One row per window and signal: :data:`WINDOW_ROW_KEYS`, then one column
        per lag (:func:`lag_column`; the lag in s is the offset over
        ``SAMPLING_FREQUENCY``).
    record : dict
        ``n_events``: the frame's events by SWR overlap
        (:func:`frame_event_counts`, every event, the ones outside a visit
        too); ``no_spike_level``: the MUA level of immobile time outside the
        periods (``no_spike_periods.level_outside_periods``);
        ``n_no_spike_periods``: the periods.
    """
    rows = []
    for kind, windows in triggered_windows(
        frame, event_types, list(TRIGGERED_SIGNALS), window_sec=window_sec
    ).items():
        labels = windows.indexes["event"].get_level_values("non_local_event")
        rows += [
            _window_rows(windows.sel(signal=signal).values, kind, signal, labels)
            for signal in TRIGGERED_SIGNALS
        ]
    periods = no_spike_periods(frame, arrival_times, departure_times)
    if not periods.empty:
        session = frame[["animal", "day", "epoch"]].iloc[0]
        windows = get_signal_windows(
            frame,
            periods.assign(**session.to_dict(), period=np.arange(len(periods))),
            [NO_SPIKE_SIGNAL],
            window_sec=window_sec,
            event_id_col="period",
        )
        rows.append(
            _window_rows(
                windows.sel(signal=NO_SPIKE_SIGNAL).values,
                NO_SPIKE_KIND,
                NO_SPIKE_SIGNAL,
                windows.indexes["event"].get_level_values("period"),
            )
        )
    record = {
        "n_events": frame_event_counts(event_types),
        "no_spike_level": level_outside_periods(
            frame, periods, arrival_times, departure_times, NO_SPIKE_SIGNAL
        ),
        "n_no_spike_periods": len(periods),
    }
    return pd.concat(rows, ignore_index=True), record


def _stored(
    windows: pd.DataFrame, kind: str, signal: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One kind's and signal's stored windows ``(n, n_lags)``, their animals and
    the lags (s)."""
    mine = windows[(windows["kind"] == kind) & (windows["signal"] == signal)]
    lags = [column for column in windows.columns if column.startswith("lag")]
    offsets = np.array([int(column.removeprefix("lag")) for column in lags])
    return (
        mine[lags].to_numpy(),
        mine["animal"].to_numpy(),
        offsets * (1 / SAMPLING_FREQUENCY),
    )


def stored_windows_of(windows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """The stored windows of the selected events: an event's rows are kept when
    its session and label are those of a row of ``events`` (as
    :func:`triggering_events_in` keeps them); the No-Spike periods' rows are
    kept whole.

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`epoch_trigger_windows`' rows of every epoch, with ``animal``,
        ``session_date`` and ``epoch``.
    events : pd.DataFrame
        The selected events, with ``animal``, ``session_date``, ``epoch`` and
        ``non_local_event``.
    """
    keys = ["animal", "session_date", "epoch", "non_local_event"]
    require_columns(events, keys, "the selected events")

    def index(table: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [
                table["animal"].astype(str),
                table["session_date"].astype(np.int64),
                table["epoch"].astype(str),
                table["non_local_event"].astype(np.int64),
            ]
        )

    kept = (windows["kind"] == NO_SPIKE_KIND).to_numpy() | index(windows).isin(
        index(events)
    )
    return windows.loc[kept]


def stored_triggered_curves(windows: pd.DataFrame) -> dict:
    """Figure 2g's curves from the stored windows: for each of
    :data:`TRIGGERED_SIGNALS`, ``time`` (s) and, for each SWR-overlap kind,
    its animal-balanced quartiles (:func:`animal_balanced_quartiles`), as
    ``"{kind}_q25"``, ``"{kind}_median"`` and ``"{kind}_q75"``.

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`stored_windows_of`' rows.
    """
    curves: dict = {}
    for signal in TRIGGERED_SIGNALS:
        curves[signal] = {}
        for kind in ("no_overlap", "overlap"):
            values, animals, time = _stored(windows, kind, signal)
            for label, quartile in animal_balanced_quartiles(values, animals).items():
                curves[signal][f"{kind}_{label}"] = quartile
            curves[signal]["time"] = time
    return curves


def stored_animal_window_medians(windows: pd.DataFrame) -> dict:
    """Figure 2's supplement's traces from the stored windows: by SWR-overlap
    kind, each signal's per-animal median windows
    (``pd.DataFrame (n_lags, n_animals)``, as :func:`animal_window_medians`
    gives them), with ``time`` (s).

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`stored_windows_of`' rows.
    """
    traces: dict = {}
    for kind in ("overlap", "no_overlap"):
        traces[kind] = {}
        for signal in TRIGGERED_SIGNALS:
            values, animals, time = _stored(windows, kind, signal)
            traces[kind][signal] = animal_quartiles(values, animals)["median"].set_axis(
                time
            )
        traces["time"] = time
    return traces


def stored_no_spike_trace(windows: pd.DataFrame, epochs: pd.DataFrame) -> dict:
    """The No-Spike panel's trace from the stored windows: ``time`` (s), and
    ``q25``, ``median`` and ``q75`` animal-balanced over the periods
    (:func:`animal_balanced_quartiles`); ``level``: the mean over animals of
    each animal's median over its epochs of each epoch's MUA level outside its
    periods.

    Parameters
    ----------
    windows : pd.DataFrame
        The stored windows (their No-Spike rows are read).
    epochs : pd.DataFrame
        One row per epoch: ``animal`` and ``no_spike_level``.
    """
    values, animals, time = _stored(windows, NO_SPIKE_KIND, NO_SPIKE_SIGNAL)
    return {
        "time": time,
        **animal_balanced_quartiles(values, animals),
        "level": float(
            animal_medians(epochs["no_spike_level"], epochs["animal"]).mean()
        ),
    }
