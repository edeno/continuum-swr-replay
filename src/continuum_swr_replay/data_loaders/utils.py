"""Utility functions for data loading.

This module provides helper functions for data loading, session management, and
time-based data filtering. These utilities support the main data loading pipeline
by handling epoch discovery, event labeling, and training data
selection for decoding models.

Main Functions
--------------
get_epoch_info
    Queries Spyglass database for all available epochs across animals. Results are
    cached for efficiency. This is the primary entry point for discovering available
    data sessions.
get_training_timepoints
    Creates boolean masks to select time points for decoder training, with options
    to exclude ripple times or use all data.
get_labels
    Efficiently labels time points with event IDs using vectorized binary search.

The epoch identifier and NWB file name parsers (``parse_epoch_identifier``,
``format_epoch_identifier``, ``parse_nwb_filename``) are in
``continuum_swr_replay.decoding.identifiers``.

Features
--------
- Cached database queries: get_epoch_info uses @cache to avoid redundant queries
- Efficient time labeling: get_labels uses np.searchsorted for O(n log n) complexity
- Type-safe training selection: Supports both TrainingType enum and string inputs

Notes
-----
- All database queries use Spyglass DataJoint tables (Session, IntervalList, etc.)
- ``quiet_spyglass`` hides pynwb/hdmf/datajoint warnings and Spyglass logging;
  every Spyglass-reading loader calls it once it has imported Spyglass
- Cached results persist for the entire Python session
- Time indices are always in seconds

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Main data loading orchestration
continuum_swr_replay.data_loaders.constants : NWB_FILES list and task constants
continuum_swr_replay.decoding.identifiers : TrainingType enum definition
continuum_swr_replay.data_types : SessionData TypedDict

Examples
--------
>>> # Get all available epochs
>>> epoch_info = get_epoch_info()
>>> epoch_info[["animal", "date", "epoch"]].head()

>>> # Get training mask excluding ripples
>>> from continuum_swr_replay.decoding.identifiers import TrainingType
>>> mask = get_training_timepoints(data, TrainingType.NO_RIPPLE)

"""

from __future__ import annotations

import logging
import warnings
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# ``spyglass.common`` is imported inside :func:`get_epoch_info` rather
# than at module top level. Importing ``spyglass.common`` opens a
# DataJoint connection at import time, which would re-couple every
# offline caller of ``utils`` (notably ``get_labels``, used by
# ``analysis.events.nonlocal_detection`` and friends) to the live
# database. Keeping it function-local means leaf imports stay offline
# and only ``get_epoch_info`` actually requires DataJoint.

if TYPE_CHECKING:
    from ..data_types import SessionData

from ..decoding.identifiers import (
    RUN_EPOCH_PATTERN,
    TrainingType,
    parse_epoch_name,
    parse_nwb_filename,
)
from ..intervals import interval_mask
from .constants import (
    FIRST_EXPOSURE_DESCRIPTIONS,
    NWB_FILES,
    REGULAR_SESSION_DESCRIPTION,
)


@cache
def quiet_spyglass() -> None:
    """Hide pynwb, hdmf and DataJoint UserWarnings and Spyglass logging below ERROR.

    Every loader that reads through Spyglass calls this right after it imports
    Spyglass, so a process is quieted when it first reads, not when a loader
    module is imported. It runs once per process; later calls return at once.

    Call it only after a Spyglass import: the first import of any Spyglass
    module runs ``spyglass.utils.logging``, which sets the ``spyglass`` logger's
    level from DataJoint's ``loglevel`` and would undo a level set before it.
    """
    for module in ("pynwb", "hdmf", "datajoint"):
        warnings.filterwarnings("ignore", category=UserWarning, module=module)
    logging.getLogger("spyglass.utils").setLevel(logging.ERROR)
    logging.getLogger("spyglass").setLevel(logging.ERROR)


def session_restriction() -> list[dict[str, str]]:
    """The ``Session`` rows ``get_epoch_info`` admits, as a DataJoint restriction.

    Every ``NWB_FILES`` session described as the regular spatial-bandit task,
    then each first-exposure session of ``FIRST_EXPOSURE_DESCRIPTIONS`` under its
    own description (wilbur20210326's "... - exploration session").

    Returns
    -------
    list of dict
        ``{"nwb_file_name", "session_description"}`` per admitted session.
    """
    return [
        {"nwb_file_name": name, "session_description": REGULAR_SESSION_DESCRIPTION}
        for name in NWB_FILES
    ] + [
        {"nwb_file_name": name, "session_description": description}
        for name, description in FIRST_EXPOSURE_DESCRIPTIONS.items()
    ]


@cache
def get_epoch_info() -> pd.DataFrame:
    """Get information about all available epochs across all animals.

    This function queries the Spyglass database, which is expensive. Results
    are cached for the lifetime of the Python process to avoid redundant
    database queries.

    It lists every run epoch (``^[0-9]+_r[0-9]+$``) with a
    ``PositionIntervalMap`` row of the sessions ``session_restriction`` admits:
    the regular sessions of ``NWB_FILES`` and, since 2026-10-05, the listed
    first-exposure sessions. That is what Spyglass holds, not the cohort: a stage
    that enumerates epochs to analyse reads
    ``epoch_register.analysed_epochs()``, which keeps the epochs the epoch
    register analyses and raises on one it does not list.

    Returns
    -------
    pd.DataFrame
        Epoch information with columns:
        - animal, date, epoch, epoch_tag
        - start_time, end_time

        in (animal, date, epoch) order. An animal's first-exposure epoch is
        the epoch register's to say (``epoch_register.first_exposure_epochs``).

    Notes
    -----
    The cache persists for the entire Python session. If the database is
    modified externally during runtime, the cached result will be stale.
    This is acceptable for typical analysis workflows where the database
    is static.

    """
    # Deferred import: ``spyglass.common`` opens a DataJoint connection
    # at import time, which we don't want to incur for callers that
    # only need ``get_labels`` / constants from this module.
    from spyglass.common import IntervalList, PositionIntervalMap, Session

    quiet_spyglass()

    epoch_info = (
        pd.DataFrame((Session & session_restriction()) * PositionIntervalMap)
        .set_index(["nwb_file_name", "interval_list_name"])
        .sort_index()
    )

    # only use run epochs
    epoch_info = epoch_info[
        epoch_info.index.get_level_values("interval_list_name").str.match(
            RUN_EPOCH_PATTERN.pattern
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
        *map(
            parse_epoch_name,
            epoch_info.index.get_level_values("interval_list_name"),
        ),
        strict=True,
    )
    epoch_info["epoch"] = np.asarray(epoch, dtype=int)
    epoch_info["epoch_tag"] = epoch_tag

    epoch_info = epoch_info.sort_values(["animal", "date", "epoch"])

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

    Loops over events; each event's index range is found by binary search
    (``np.searchsorted``) and labelled with one slice assignment, so the cost
    is O(n_events * log(n_times)) searches plus one write per labelled time
    point, rather than an O(n_times) boolean mask per event.

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

    Notes
    -----
    Time points within [start_time, end_time] (inclusive) are labeled with
    sequential integers starting from 1. Time points outside all events are
    labeled 0.

    """
    event_labels = pd.DataFrame(0, index=time, columns=["event_number"], dtype=int)

    time_array = time.to_numpy()

    # Per-event binary search for the index range, then one slice write.
    for i, (start, end) in enumerate(
        zip(times["start_time"], times["end_time"], strict=True), start=1
    ):
        start_idx = np.searchsorted(time_array, start, side="left")
        end_idx = np.searchsorted(time_array, end, side="right")

        # Use iloc for integer-based indexing (loc would be label-based)
        event_labels.iloc[start_idx:end_idx, 0] = i

    return event_labels


def get_training_timepoints(
    data: SessionData, training_type: str | TrainingType = TrainingType.NO_RIPPLE
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
        - TrainingType.NO_RIPPLE or "no_ripple": exclude ripple times, each
          ripple closed (``intervals.interval_mask``)

    Returns
    -------
    np.ndarray
        Boolean mask of time points to use for training, shape (n_time,)

    Raises
    ------
    ValueError
        For an unknown `training_type`, or, for "no_ripple", a ripple with a
        NaN bound or a start after its end.

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
        # closed: a time on a ripple's start or end is a ripple time
        return ~interval_mask(data["position_info"].index, data["ripple_times"])
    else:
        raise ValueError(
            f"Unknown training_type: {training_type}. "
            f"Must be TrainingType.ALL or TrainingType.NO_RIPPLE."
        )
