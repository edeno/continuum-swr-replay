"""No-Spike periods: when the decoder holds the No-Spike state.

The decoder enters the No-Spike state when the decoded tetrodes fall silent, so
what the tetrodes' activity does around these periods describes the state; it
does not test it.

A period is found by the non-local events' rule applied to the No-Spike state's
probability (``nonlocal_event_rule.detect_nonlocal_events``, which takes any
state's probability): a run of samples with the probability strictly above
:data:`NO_SPIKE_THRESHOLD` that lasts the events' minimum
(``parameters.NON_LOCAL_MINIMUM_DURATION_S``, 7 samples on the decode grid),
during which the animal is immobile throughout. As the other events are, a period
is kept only when it lies wholly outside excluded time (the decode frame's
``in_excluded_time``) and starts inside a well visit
(``well_visits.in_well_visit``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    detect_nonlocal_events,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.tables.columns import require_columns

NO_SPIKE_PROBABILITY = "No-Spike_prob"
"""The decode frame's column of the No-Spike state's probability."""

NO_SPIKE_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD
"""The probability a period stays strictly above: the non-local events'
threshold, 0.8, which the events summary's "No-Spike above 0.8" also uses."""

FRAME_COLUMNS = (NO_SPIKE_PROBABILITY, "head_speed", "in_excluded_time")
"""The decode frame's columns :func:`no_spike_periods` reads."""


def no_spike_periods(
    frame: pd.DataFrame,
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
    threshold: float = NO_SPIKE_THRESHOLD,
) -> pd.DataFrame:
    """One epoch's No-Spike periods (the module's rule).

    Parameters
    ----------
    frame : pd.DataFrame, shape (n_time, ...)
        One epoch's decode frame, indexed by time (s), with
        :data:`FRAME_COLUMNS`.
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        The epoch's well visits (``prepare_data.well_visit_times``).
    threshold : float, optional
        The probability a period stays strictly above.

    Returns
    -------
    pd.DataFrame
        One row per period: ``start_time`` and ``end_time`` (its first and last
        sample, s) and ``duration`` (s), in time order.
    """
    require_columns(frame, FRAME_COLUMNS, "the decode frame")
    probability = xr.DataArray(frame[NO_SPIKE_PROBABILITY].to_numpy(), dims="time")
    periods, labels = detect_nonlocal_events(
        probability, frame["head_speed"].to_numpy(), frame.index, threshold
    )
    excluded = (
        pd.Series(frame["in_excluded_time"].to_numpy(dtype=bool), index=labels.values)
        .groupby(level=0)
        .any()
    )
    keep = ~excluded.reindex(periods["non_local_event"]).to_numpy(dtype=bool)
    keep &= in_well_visit(
        periods["start_time"].to_numpy(dtype=float), arrival_times, departure_times
    )
    return periods.loc[keep, ["start_time", "end_time", "duration"]].reset_index(
        drop=True
    )


def level_outside_periods(
    frame: pd.DataFrame,
    periods: pd.DataFrame,
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
    column: str,
) -> float:
    """The median of ``column`` over one epoch's immobile analyzed time inside a
    well visit, outside its No-Spike periods: the level the signal holds when
    the decoder is not in the state.

    Parameters
    ----------
    frame : pd.DataFrame
        One epoch's decode frame, indexed by time (s), with ``head_speed``,
        ``in_excluded_time`` and ``column``.
    periods : pd.DataFrame
        :func:`no_spike_periods` of the epoch.
    arrival_times, departure_times : np.ndarray
        The epoch's well visits.
    column : str

    Returns
    -------
    float
        NaN when no sample qualifies.
    """
    time = frame.index.to_numpy(dtype=float)
    chosen = (
        immobile_analyzed_bins(frame)
        & in_well_visit(time, arrival_times, departure_times)
        & ~interval_mask(time, periods)
    )
    values = frame[column].to_numpy(dtype=float)[chosen]
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
