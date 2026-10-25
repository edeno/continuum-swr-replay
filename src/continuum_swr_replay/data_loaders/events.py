"""DIO event data loading.

This module loads digital I/O (DIO) event data from Spyglass/NWB files. DIO events
capture hardware-triggered signals from the behavioral apparatus, including beam
breaks (well visits), reward pump activations, and environmental lights.

These events are essential for constructing the task structure: identifying when the
animal visits wells, receives rewards, and transitions between behavioral states.

Main Functions
--------------
load_dios
    Primary entry point that loads all DIO event types (beam breaks, pump events,
    light events) for a given session and time range. Returns organized event data
    as a TypedDict.

Helper Functions
----------------
convert_dio_events_to_start_stop_times
    Converts raw DIO binary signals to discrete events with start/end times.
    Handles edge cases where events occur at data boundaries.

Event Types
-----------
Beam Breaks (Poke events):
    Triggered when animal's nose breaks infrared beams at well locations. Used to
    identify well visits and construct trial structure. Event names match pattern
    "Poke%".

Pump Events:
    Triggered when reward pump activates to deliver liquid reward. Used to identify
    rewarded trials. Event names match pattern "Pump%".

Light Events:
    Environmental lighting changes during the task. Event names match pattern
    "Light%".

Data Structure
--------------
All events are returned as pandas DataFrames with:
    - start_time: Event onset time (seconds)
    - end_time: Event offset time (seconds)
    - event_name: String identifier for the event type
    - event_number: Sequential numbering within each event type
    - Multi-index: (event_name, event_number)

Notes
-----
- All times are in seconds
- Events are filtered to valid_times range (typically epoch boundaries)
- DIO data is binary (0/1) and converted to discrete intervals
- Queries use Spyglass DIOEvents table with SQL-like LIKE pattern matching
- Edge case: if event starts at last time point, end_time is clamped to last index

See Also
--------
continuum_swr_replay.data_loaders.trials : Trial construction from DIO events
continuum_swr_replay.data_types : DIOEventData TypedDict

Examples
--------
>>> dio_data = load_dios("chimi20200212_.nwb", (100.0, 1000.0))
>>> beam_breaks = dio_data["beam_breaks"]
>>> pump_events = dio_data["pump_events"]

"""

import numpy as np
import pandas as pd

from ..data_types import DIOEventData
from .utils import quiet_spyglass


def convert_dio_events_to_start_stop_times(
    event: pd.Series | pd.DataFrame, event_name: str
) -> pd.DataFrame:
    """Convert DIO events to start/stop time format.

    Parameters
    ----------
    event : pandas.Series or pandas.DataFrame
        One DIO channel with a time index
    event_name : str
        Name of the event

    Returns
    -------
    pd.DataFrame
        DataFrame with start_time, end_time, event_name, and event_number columns

    """
    values = np.asarray(event)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("one DIO channel must be a one-dimensional signal")
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("DIO channel values must be finite binary 0/1 samples")
    timestamps = pd.to_numeric(pd.Index(event.index), errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(timestamps).all() or (
        len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("DIO timestamps must be finite and strictly increasing")

    active = numeric.astype(bool)
    start_indices = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    falling_indices = np.flatnonzero(~active & np.r_[False, active[:-1]])
    end_indices = np.empty(len(start_indices), dtype=np.int64)
    for event_number, start_index in enumerate(start_indices):
        later_falls = falling_indices[falling_indices > start_index]
        end_indices[event_number] = (
            int(later_falls[0]) if later_falls.size else len(active) - 1
        )

    event_times = pd.DataFrame(
        {
            "start_time": timestamps[start_indices],
            "end_time": timestamps[end_indices],
            "event_name": event_name,
            "event_number": np.arange(len(start_indices)) + 1,
        },
    )
    return event_times.set_index(["event_name", "event_number"])


def _load_dio_event_type(
    nwb_file_name: str, pattern: str, valid_times: tuple[float, float]
) -> pd.DataFrame:
    """Load and process a specific type of DIO event.

    Helper function to eliminate code duplication when loading different
    DIO event types (beam breaks, pump events, light events).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    pattern : str
        SQL LIKE pattern for event names (e.g., "Poke%", "Pump%", "Light%")
    valid_times : tuple[float, float]
        (start_time, end_time) tuple defining valid time range in seconds

    Returns
    -------
    pd.DataFrame
        Processed DIO events with start/end times and event numbers

    See Also
    --------
    load_dios : Main function that uses this helper for all event types
    convert_dio_events_to_start_stop_times : Converts raw events to intervals

    """
    # Lazy import: Spyglass requires DataJoint connection / VPN at
    # import time. Keeping this inside the function lets pure helpers
    # like ``convert_dio_events_to_start_stop_times`` import cleanly
    # offline.
    from spyglass.common import DIOEvents

    quiet_spyglass()

    events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents
            & {"nwb_file_name": nwb_file_name}
            & f'dio_event_name LIKE "{pattern}"'
        ).fetch_nwb()
    }
    # Short-circuit when no rows match the LIKE pattern: a session
    # without any ``Light%`` (or ``Pump%`` / ``Poke%``) events would
    # otherwise raise ``ValueError: No objects to concatenate`` from
    # ``pd.concat([])`` and fail the entire ``load_dios`` call. Return
    # a schema-complete empty frame so downstream consumers see "no
    # events of this type" rather than a crash.
    if not events:
        return pd.DataFrame(
            columns=["start_time", "end_time"],
            index=pd.MultiIndex.from_arrays(
                [
                    pd.Index([], dtype=str, name="event_name"),
                    pd.Index([], dtype=int, name="event_number"),
                ]
            ),
        )
    return pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in events.items()
        ],
        axis=0,
    )


def load_dios(nwb_file_name: str, valid_times: tuple[float, float]) -> DIOEventData:
    """Load DIO events (beam breaks, pump events, lights).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    valid_times : tuple[float, float]
        (start_time, end_time) tuple defining valid time range in seconds

    Returns
    -------
    DIOEventData
        TypedDict containing:
        - beam_breaks: pd.DataFrame with poke events
        - pump_events: pd.DataFrame with reward pump events
        - light_events: pd.DataFrame with light events

    """
    beam_breaks = _load_dio_event_type(nwb_file_name, "Poke%", valid_times)
    pump_events = _load_dio_event_type(nwb_file_name, "Pump%", valid_times)
    light_events = _load_dio_event_type(nwb_file_name, "Light%", valid_times)

    return {
        "beam_breaks": beam_breaks,
        "pump_events": pump_events,
        "light_events": light_events,
    }
