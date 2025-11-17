"""DIO event data loading."""

import warnings
from typing import TypedDict

import numpy as np
import pandas as pd
from spyglass.common import DIOEvents

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


class DIOEventData(TypedDict):
    """Return type for load_dios function.

    Attributes
    ----------
    beam_breaks : pd.DataFrame
        DataFrame with beam break events (poke-in/poke-out)
    pump_events : pd.DataFrame
        DataFrame with reward pump events
    light_events : pd.DataFrame
        DataFrame with light on/off events
    """

    beam_breaks: pd.DataFrame
    pump_events: pd.DataFrame
    light_events: pd.DataFrame


def convert_dio_events_to_start_stop_times(
    event: pd.DataFrame, event_name: str
) -> pd.DataFrame:
    """Convert DIO events to start/stop time format.

    Parameters
    ----------
    event : pd.DataFrame
        Event data with time index
    event_name : str
        Name of the event

    Returns
    -------
    pd.DataFrame
        DataFrame with start_time, end_time, event_name, and event_number columns
    """
    start_indices = event.astype(bool).to_numpy().nonzero()[0]
    end_indices = start_indices + 1
    end_indices[end_indices >= len(event)] = len(event) - 1

    event_times = pd.DataFrame(
        {
            "start_time": event.iloc[start_indices].index,
            "end_time": event.iloc[end_indices].index,
            "event_name": event_name,
            "event_number": np.arange(len(start_indices)) + 1,
        },
    )
    return event_times.set_index(["event_name", "event_number"])


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
    beam_breaks = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents & {"nwb_file_name": nwb_file_name} & 'dio_event_name LIKE "Poke%"'
        ).fetch_nwb()
    }
    beam_breaks = pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in beam_breaks.items()
        ],
        axis=0,
    )

    pump_events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents & {"nwb_file_name": nwb_file_name} & 'dio_event_name LIKE "Pump%"'
        ).fetch_nwb()
    }
    pump_events = pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in pump_events.items()
        ],
        axis=0,
    )

    light_events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents
            & {"nwb_file_name": nwb_file_name}
            & 'dio_event_name LIKE "Light%"'
        ).fetch_nwb()
    }
    light_events = pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in light_events.items()
        ],
        axis=0,
    )

    return {
        "beam_breaks": beam_breaks,
        "pump_events": pump_events,
        "light_events": light_events,
    }
