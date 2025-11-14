"""Utility functions for data loading."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from spyglass.common import IntervalList, PositionIntervalMap, Session

from .constants import NWB_FILES

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


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
            (
                file_name.replace("_.nwb", "")[:-8],
                int(file_name.replace("_.nwb", "")[-8:]),
            )
            for file_name in epoch_info.index.get_level_values(
                "nwb_file_name"
            ).to_numpy()
        ], strict=False
    )
    epoch_info["animal"] = animal
    epoch_info["date"] = date

    epoch, epoch_tag = zip(
        *epoch_info.index.get_level_values("interval_list_name").str.split("_"), strict=False
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
        np.zeros_like(time, dtype=int), index=time, columns=["event_number"]
    )
    # This means the label will be +1 because label count starts at 0
    for i, row in enumerate(times.itertuples()):
        event_labels.loc[row.start_time : row.end_time, :] = i + 1

    return event_labels


def get_training_timepoints(
    data: dict[str, Any], training_type: str = "no_ripple"
) -> np.ndarray:
    """Get boolean mask of time points to use for training.

    Parameters
    ----------
    data : dict
        Data dictionary with position_info and ripple_times
    training_type : str, optional
        Type of training data selection:
        - "all": use all time points
        - "no_ripple": exclude ripple times

    Returns
    -------
    np.ndarray
        Boolean mask of time points to use for training
    """
    if training_type == "all":
        n_time = len(data["position_info"].index)
        return np.ones((n_time,), dtype=bool)

    elif training_type == "no_ripple":
        ripple_labels = get_labels(
            data["ripple_times"], data["position_info"].index.to_numpy()
        )
        is_ripple = ripple_labels > 0
        not_ripple = ~np.asarray(is_ripple).squeeze()

        return not_ripple
    else:
        raise ValueError(
            f"Unknown training_type: {training_type}. Must be 'all' or 'no_ripple'."
        )
