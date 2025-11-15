"""Utility functions for data loading."""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from spyglass.common import IntervalList, PositionIntervalMap, Session

from ..parameters import TrainingType
from .constants import NWB_FILES

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def parse_nwb_filename(filename: str) -> tuple[str, int]:
    """Parse NWB filename to extract animal name and date.

    Parameters
    ----------
    filename : str
        NWB filename in format: {animal}{YYYYMMDD}_.nwb
        Example: "chimi20200212_.nwb"

    Returns
    -------
    tuple[str, int]
        Animal name and date as (animal, date)
        Example: ("chimi", 20200212)

    Raises
    ------
    ValueError
        If filename doesn't match expected format

    Examples
    --------
    >>> parse_nwb_filename("chimi20200212_.nwb")
    ('chimi', 20200212)

    >>> parse_nwb_filename("peanut20201125_.nwb")
    ('peanut', 20201125)
    """
    # Pattern: {animal_name}{8_digits}_.nwb
    # Animal name is one or more letters, date is exactly 8 digits
    pattern = r"^([a-zA-Z]+)(\d{8})_\.nwb$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Invalid NWB filename format: {filename}. "
            f"Expected format: {{animal}}{{YYYYMMDD}}_.nwb"
        )

    animal = match.group(1)
    date = int(match.group(2))

    return animal, date


def get_epoch_info() -> pd.DataFrame:
    """Get information about all available epochs across all animals.

    Returns
    -------
    pd.DataFrame
        Epoch information with columns:
        - animal, date, epoch, epoch_tag
        - exposure (session number for that animal)
        - start_time, end_time
    """
    epoch_info = (
        pd.DataFrame(
            (
                Session
                & [
                    {
                        "nwb_file_name": nwbf,
                        "session_description": "Spatial bandit task (regular)",
                    }
                    for nwbf in NWB_FILES
                ]
            )
            * PositionIntervalMap
        )
        .set_index(["nwb_file_name", "interval_list_name"])
        .sort_index()
    )

    # only use run epochs
    epoch_info = epoch_info[
        epoch_info.index.get_level_values("interval_list_name").str.match(
            r"^[0-9]+_r[0-9]+$"
        )
    ]

    animal, date = zip(
        *[
            parse_nwb_filename(file_name)
            for file_name in epoch_info.index.get_level_values(
                "nwb_file_name"
            ).to_numpy()
        ],
        strict=True,
    )
    epoch_info["animal"] = animal
    epoch_info["date"] = date

    epoch, epoch_tag = zip(
        *epoch_info.index.get_level_values("interval_list_name").str.split("_"),
        strict=True,
    )
    epoch_info["epoch"] = np.asarray(epoch, dtype=int)
    epoch_info["epoch_tag"] = epoch_tag

    def _count_exposure(df: pd.DataFrame) -> pd.DataFrame:
        df["exposure"] = np.arange(len(df)) + 1
        return df

    epoch_info = (
        epoch_info.sort_values(["animal", "date", "epoch"])
        .groupby("animal", group_keys=False)
        .apply(_count_exposure)
    )

    interval_times = np.stack(
        (
            IntervalList
            & epoch_info.reset_index()
            .loc[:, ("nwb_file_name", "interval_list_name")]
            .to_dict(orient="records")
        ).fetch("valid_times")
    ).squeeze()

    epoch_info["start_time"] = pd.to_datetime(interval_times[:, 0], unit="s")
    epoch_info["end_time"] = pd.to_datetime(interval_times[:, 1], unit="s")

    return epoch_info


def get_labels(times: pd.DataFrame, time: pd.Index) -> pd.DataFrame:
    """Create event labels for each time point.

    Parameters
    ----------
    times : pd.DataFrame
        Event times with start_time and end_time columns
    time : pd.Index
        Time index to label

    Returns
    -------
    pd.DataFrame
        Event labels (0 for no event, 1+ for event number)
    """
    event_labels = pd.DataFrame(
        0, index=time, columns=["event_number"], dtype=int
    )
    # This means the label will be +1 because label count starts at 0
    for i, row in enumerate(times.itertuples()):
        event_labels.loc[row.start_time : row.end_time, :] = i + 1

    return event_labels


def get_training_timepoints(
    data: dict[str, Any], training_type: str | TrainingType = TrainingType.NO_RIPPLE
) -> np.ndarray:
    """Get boolean mask of time points to use for training.

    Parameters
    ----------
    data : dict
        Data dictionary with position_info and ripple_times
    training_type : str or TrainingType, optional
        Type of training data selection. Default is TrainingType.NO_RIPPLE.
        Accepts either TrainingType enum or equivalent string:
        - TrainingType.ALL or "all": use all time points
        - TrainingType.NO_RIPPLE or "no_ripple": exclude ripple times

    Returns
    -------
    np.ndarray
        Boolean mask of time points to use for training

    Examples
    --------
    >>> # Using enum (type-safe, recommended)
    >>> mask = get_training_timepoints(data, TrainingType.ALL)
    >>> # Using string (backward compatible)
    >>> mask = get_training_timepoints(data, "all")
    """
    # Convert string to enum if needed (backward compatibility)
    if isinstance(training_type, str):
        try:
            training_type = TrainingType(training_type)
        except ValueError:
            raise ValueError(
                f"Unknown training_type: {training_type}. "
                f"Must be TrainingType.ALL, TrainingType.NO_RIPPLE, 'all', or 'no_ripple'."
            ) from None

    if training_type == TrainingType.ALL:
        n_time = len(data["position_info"].index)
        return np.ones((n_time,), dtype=bool)

    elif training_type == TrainingType.NO_RIPPLE:
        ripple_labels = get_labels(
            data["ripple_times"], data["position_info"].index.to_numpy()
        )
        is_ripple = ripple_labels > 0
        not_ripple = ~np.asarray(is_ripple).squeeze()

        return not_ripple
    else:
        raise ValueError(
            f"Unknown training_type: {training_type}. "
            f"Must be TrainingType.ALL or TrainingType.NO_RIPPLE."
        )
