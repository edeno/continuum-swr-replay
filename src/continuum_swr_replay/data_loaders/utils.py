"""Utility functions for data loading.

This module provides helper functions for data loading, session management, and
time-based data filtering. These utilities support the main data loading pipeline
by handling NWB file parsing, epoch discovery, event labeling, and training data
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
parse_nwb_filename
    Extracts animal name and date from NWB filenames.
parse_epoch_identifier
    Splits an epoch identifier ``{animal}{date}_{epoch}_{brain_area}`` and its
    decode suffixes into an :class:`EpochKey`.
format_epoch_identifier
    Builds an epoch identifier, the inverse of ``parse_epoch_identifier``.

Features
--------
- Cached database queries: get_epoch_info uses @cache to avoid redundant queries
- Efficient time labeling: get_labels uses np.searchsorted for O(n log n) complexity
- Type-safe training selection: Supports both TrainingType enum and string inputs
- Robust parsing: Validates NWB filename format with regex

Notes
-----
- All database queries use Spyglass DataJoint tables (Session, IntervalList, etc.)
- Warning filters suppress verbose pynwb/hdmf/datajoint logging
- Cached results persist for the entire Python session
- Time indices are always in seconds

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Main data loading orchestration
continuum_swr_replay.data_loaders.constants : NWB_FILES list and task constants
continuum_swr_replay.parameters : TrainingType enum definition
continuum_swr_replay.types : SessionData TypedDict

Examples
--------
>>> # Get all available epochs
>>> epoch_info = get_epoch_info()
>>> epoch_info[["animal", "date", "epoch"]].head()

>>> # Parse filename
>>> animal, date = parse_nwb_filename("chimi20200212_.nwb")

>>> # Get training mask excluding ripples
>>> from continuum_swr_replay.parameters import TrainingType
>>> mask = get_training_timepoints(data, TrainingType.NO_RIPPLE)

"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# ``spyglass.common`` is imported inside :func:`get_epoch_info` rather
# than at module top level. Importing ``spyglass.common`` opens a
# DataJoint connection at import time, which would re-couple every
# offline caller of ``utils`` (notably ``get_labels``, used by
# ``analysis.events.detection`` and friends) to the live
# database. Keeping it function-local means leaf imports stay offline
# and only ``get_epoch_info`` actually requires DataJoint.

if TYPE_CHECKING:
    from ..types import SessionData

from ..parameters import TrainingType
from .constants import (
    FIRST_EXPOSURE_DESCRIPTIONS,
    NWB_FILES,
    REGULAR_SESSION_DESCRIPTION,
)

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

    >>> parse_nwb_filename("j1620210706_.nwb")
    ('j16', 20210706)

    """
    # Pattern: {animal_name}{8_digits}_.nwb
    # Animal name is one or more alphanumeric chars (non-greedy), date is exactly 8 digits
    # The date is always YYYYMMDD format, so we match the last 8 digits before _.nwb
    pattern = r"^([a-zA-Z0-9]+?)(\d{8})_\.nwb$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Invalid NWB filename format: {filename}. "
            f"Expected format: {{animal}}{{YYYYMMDD}}_.nwb"
        )

    animal = match.group(1)
    date = int(match.group(2))

    return animal, date


DEFAULT_SWR_DEFINITION = "Kay"
"""A decode run with the default ripple detector adds no suffix to its identifier."""

OTHER_RIPPLE_DETECTORS = ("Karlsson", "Yu")
"""The ripple detectors whose name a decode would append; Kay's it does not.

No decode carries ``_Yu`` yet; it is listed so the day one does, the parser
reads it as a ripple detector and not as a name no decode writes."""

DEFAULT_TRAINING_TYPE = "no_ripple"
"""The training type a decode does not append to its identifier."""


@dataclass(frozen=True)
class EpochKey:
    """What an epoch identifier says about the decode it names."""

    epoch_id: str
    nwb_file_name: str
    epoch_name: str
    brain_area: str
    swr_definition: str
    contfrag: bool
    non_stationary: bool
    training_type: str

    @property
    def animal(self) -> str:
        """The animal, as :func:`parse_nwb_filename` reads ``nwb_file_name``."""
        return parse_nwb_filename(self.nwb_file_name)[0]

    @property
    def date(self) -> int:
        """The recording date ``YYYYMMDD``, as :func:`parse_nwb_filename` reads it."""
        return parse_nwb_filename(self.nwb_file_name)[1]

    @property
    def is_canonical(self) -> bool:
        """The decode the comparison uses: Kay's SWRs, no_ripple, stationary."""
        return (
            not self.contfrag
            and self.swr_definition == DEFAULT_SWR_DEFINITION
            and not self.non_stationary
            and self.training_type == DEFAULT_TRAINING_TYPE
        )


def parse_epoch_identifier(epoch_id: str) -> EpochKey:
    """Split ``{animal}{date}_{epoch}_{brain_area}`` and any decode suffixes.

    Suffixes come in the order ``run_decoding_bandit.py`` and
    ``run_decoding_contfrag.py`` append them: ``_contfrag``, a ripple detector
    other than Kay, ``_non_stationary``, and a training type other than
    ``no_ripple``. Anything else is refused rather than read as, say, a
    ripple detector called ``all``.

    Parameters
    ----------
    epoch_id : str
        Such as ``"j1620210707_02_r1_HPC"`` or ``"chimi20200212_02_r1_mPFC_contfrag"``.

    Returns
    -------
    EpochKey
        ``animal`` and ``date`` are read from the NWB stem only when asked for,
        so a stem :func:`parse_nwb_filename` refuses is refused there.

    Raises
    ------
    ValueError
        If `epoch_id` has fewer than four ``_``-separated parts, an empty one,
        or a suffix no decode appends.

    Examples
    --------
    >>> key = parse_epoch_identifier("j1620210707_02_r1_HPC_contfrag")
    >>> key.animal, key.date, key.epoch_name, key.brain_area, key.contfrag
    ('j16', 20210707, '02_r1', 'HPC', True)
    """
    parts = epoch_id.split("_")
    if len(parts) < 4 or not all(parts):
        raise ValueError(
            f"{epoch_id!r} is not {{animal}}{{date}}_{{epoch}}_{{brain_area}} with "
            f"optional decode suffixes."
        )
    stem, number, replicate, brain_area, *rest = parts
    contfrag = rest[:1] == ["contfrag"]
    rest = rest[1:] if contfrag else rest
    swr_definition = DEFAULT_SWR_DEFINITION
    if rest and rest[0] in OTHER_RIPPLE_DETECTORS:
        swr_definition, rest = rest[0], rest[1:]
    non_stationary = rest[:2] == ["non", "stationary"]
    rest = rest[2:] if non_stationary else rest
    training_type = DEFAULT_TRAINING_TYPE
    if rest[:1] == ["all"]:
        training_type, rest = "all", rest[1:]
    if rest:
        raise ValueError(
            f"{epoch_id!r} ends in {'_'.join(rest)!r}, which no decode appends."
        )
    return EpochKey(
        epoch_id=epoch_id,
        nwb_file_name=f"{stem}_.nwb",
        epoch_name=f"{number}_{replicate}",
        brain_area=brain_area,
        swr_definition=swr_definition,
        contfrag=contfrag,
        non_stationary=non_stationary,
        training_type=training_type,
    )


def format_epoch_identifier(
    nwb_file_name: str, epoch: str, brain_area: str, suffix: str = ""
) -> str:
    """Build ``{animal}{date}_{epoch}_{brain_area}{suffix}``, the epoch identifier.

    The inverse of :func:`parse_epoch_identifier`. Only a trailing ``_.nwb`` is
    removed from `nwb_file_name`.

    Parameters
    ----------
    nwb_file_name : str
        Such as ``"chimi20200212_.nwb"``.
    epoch : str
        Such as ``"02_r1"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    suffix : str, optional
        Decode suffixes, each with its leading ``_``, in the order
        :func:`parse_epoch_identifier` reads them, such as ``"_contfrag"`` or
        ``"_non_stationary"``. None by default.

    Returns
    -------
    str

    Examples
    --------
    >>> format_epoch_identifier("chimi20200212_.nwb", "02_r1", "mPFC", "_contfrag")
    'chimi20200212_02_r1_mPFC_contfrag'
    """
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch}_{brain_area}{suffix}"


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
        - exposure (session number for that animal)
        - start_time, end_time

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

    epoch_info = (
        pd.DataFrame((Session & session_restriction()) * PositionIntervalMap)
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

    Uses vectorized binary search (np.searchsorted) for efficient labeling
    of large datasets. Complexity is O(n_events * log(n_times)) instead of
    O(n_events * n_times) with explicit boolean masking.

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

    # Vectorized labeling using binary search (O(n_events * log(n_times)))
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
        ripple_labels = get_labels(data["ripple_times"], data["position_info"].index)
        is_ripple = ripple_labels > 0
        not_ripple = ~np.asarray(is_ripple).squeeze()

        return not_ripple
    else:
        raise ValueError(
            f"Unknown training_type: {training_type}. "
            f"Must be TrainingType.ALL or TrainingType.NO_RIPPLE."
        )
