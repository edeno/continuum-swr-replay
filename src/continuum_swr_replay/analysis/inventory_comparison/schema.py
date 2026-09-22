"""One table per detector, in one schema (spec 03, "Common schema").

Each inventory is one row per event, over the same epochs and speed trace.
The span of an event is ``event_end_s - event_start_s`` for every detector,
because the native durations follow different conventions: our canonical
table adds one sample width to the span, the memoryless detector does not.
The native duration is kept beside the span and never stands in for it.
Keys are namespaced by detector, ``"<detector>:<native key>"``, so two
inventories never share one.

Detector-specific statistics stay in each detector's own table; this schema
carries only what every detector has.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    event_speeds_are_immobile,
)

__all__ = [
    "INVENTORY_DTYPES",
    "SessionIdentity",
    "empty_inventory",
    "from_classified_nonlocal_table",
    "from_memoryless_table",
    "inventory_from_intervals",
    "validate_inventory",
]

INVENTORY_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "brain_area": object,
        "detector": object,
        "event_index": np.int64,
        "event_key": object,
        "run_id": object,
        "event_start_s": np.float64,
        "event_end_s": np.float64,
        "interval_span_s": np.float64,
        "event_midpoint_s": np.float64,
        "native_duration_s": np.float64,
        "animal_max_speed_cm_s": np.float64,
        "animal_mean_speed_cm_s": np.float64,
        "whole_event_speed_eligible": np.bool_,
    }
)
"""Every column, in order, with the dtype it always has, even when empty."""

STATE_SPACE = "state_space"
MEMORYLESS = "memoryless"


@dataclass(frozen=True)
class SessionIdentity:
    """The epoch an inventory's events come from."""

    animal: str
    day: int
    epoch: str
    brain_area: str


def empty_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in INVENTORY_DTYPES.items()}
    )


def validate_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Check the schema's invariants and return the table in schema order.

    Raises
    ------
    ValueError
        A column is missing, an event has no positive duration, the span or
        midpoint does not follow from the bounds, or a key repeats.
    """
    missing = [column for column in INVENTORY_DTYPES if column not in inventory]
    if missing:
        raise ValueError(f"The inventory lacks {missing}.")
    table = inventory[list(INVENTORY_DTYPES)].astype(dict(INVENTORY_DTYPES))
    start, end = table.event_start_s.to_numpy(), table.event_end_s.to_numpy()
    if not (
        np.isfinite(start).all() and np.isfinite(end).all() and (end > start).all()
    ):
        raise ValueError(
            "Every event needs finite bounds and a positive duration; an event of "
            "no duration can never overlap another under the strict rule."
        )
    if not np.array_equal(table.interval_span_s.to_numpy(), end - start):
        raise ValueError("interval_span_s must be event_end_s - event_start_s.")
    if not np.allclose(table.event_midpoint_s.to_numpy(), (start + end) / 2):
        raise ValueError("event_midpoint_s must be halfway between the bounds.")
    if not table.event_key.is_unique:
        raise ValueError("An event key repeats; keys are namespaced by detector.")
    return table


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Order by session, detector and start; index; compute the shared timing."""
    if frame.empty:
        return empty_inventory()
    frame = frame.sort_values(
        ["animal", "day", "epoch", "detector", "event_start_s"], kind="stable"
    ).reset_index(drop=True)
    frame["event_index"] = frame.groupby(
        ["animal", "day", "epoch", "detector"], sort=False
    ).cumcount()
    frame["interval_span_s"] = frame.event_end_s - frame.event_start_s
    frame["event_midpoint_s"] = (frame.event_start_s + frame.event_end_s) / 2
    return validate_inventory(frame)


def inventory_from_intervals(
    starts: NDArray,
    ends: NDArray,
    *,
    native_keys: NDArray,
    detector: str,
    session: SessionIdentity,
    run_id: str,
    speed_time_s: NDArray,
    speed_cm_per_s: NDArray,
    native_duration_s: NDArray | None = None,
) -> pd.DataFrame:
    """An inventory from bare intervals, such as SWR or MUA event labels.

    Speeds are read over every sample of the speed trace from the event's
    start to its end inclusive, as the event tables read them.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_events,)
    native_keys : ndarray, shape (n_events,)
        The detector's own event identifiers; must not repeat.
    detector : str
    session : SessionIdentity
    run_id : str
    speed_time_s, speed_cm_per_s : ndarray, shape (n_samples,)
    native_duration_s : ndarray, shape (n_events,), optional
        The detector's own duration; the span when not given.

    Returns
    -------
    inventory : DataFrame
        Columns and dtypes as ``INVENTORY_DTYPES``.

    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    keys = np.asarray(native_keys)
    if not starts.shape == ends.shape == keys.shape:
        raise ValueError("starts, ends and native_keys must be the same length.")
    if len(np.unique(keys)) != keys.size:
        raise ValueError(f"A native key of {detector} repeats.")
    time = np.asarray(speed_time_s, dtype=np.float64)
    speed = np.asarray(speed_cm_per_s, dtype=np.float64)
    first = np.searchsorted(time, starts, side="left")
    last = np.searchsorted(time, ends, side="right")
    samples = [speed[a:b] for a, b in zip(first, last, strict=True)]
    frame = pd.DataFrame(
        {
            "animal": session.animal,
            "day": session.day,
            "epoch": session.epoch,
            "brain_area": session.brain_area,
            "detector": detector,
            "event_key": [f"{detector}:{key}" for key in keys],
            "run_id": run_id,
            "event_start_s": starts,
            "event_end_s": ends,
            "native_duration_s": ends - starts
            if native_duration_s is None
            else np.asarray(native_duration_s, dtype=np.float64),
            "animal_max_speed_cm_s": [s.max() if s.size else np.nan for s in samples],
            "animal_mean_speed_cm_s": [s.mean() if s.size else np.nan for s in samples],
            "whole_event_speed_eligible": [
                event_speeds_are_immobile(s) for s in samples
            ],
        }
    )
    return _finish(frame)


def from_classified_nonlocal_table(
    classified: pd.DataFrame, *, brain_area: str, run_id: str
) -> pd.DataFrame:
    """Our state-space cohort, from ``build_classified_nonlocal_event_table``."""
    frame = pd.DataFrame(
        {
            "animal": classified.animal,
            "day": classified.day,
            "epoch": classified.epoch,
            "brain_area": brain_area,
            "detector": STATE_SPACE,
            "event_key": [f"{STATE_SPACE}:{key}" for key in classified.event_id],
            "run_id": run_id,
            "event_start_s": classified.event_start,
            "event_end_s": classified.event_end,
            "native_duration_s": classified.event_duration,
            "animal_max_speed_cm_s": classified.animal_max_speed_cm_s,
            "animal_mean_speed_cm_s": classified.animal_mean_speed_cm_s,
            "whole_event_speed_eligible": classified.whole_event_speed_eligible,
        }
    )
    return _finish(frame)


def from_memoryless_table(table: pd.DataFrame) -> pd.DataFrame:
    """The memoryless cohort, from ``build_memoryless_event_table``."""
    frame = pd.DataFrame(
        {
            "animal": table.animal,
            "day": table.day,
            "epoch": table.epoch,
            "brain_area": table.brain_area,
            "detector": MEMORYLESS,
            "event_key": [f"{MEMORYLESS}:{key}" for key in table.ms_event_id],
            "run_id": table.run_id,
            "event_start_s": table.event_start_s,
            "event_end_s": table.event_end_s,
            "native_duration_s": table.event_duration_s,
            "animal_max_speed_cm_s": table.animal_max_speed_cm_s,
            "animal_mean_speed_cm_s": table.animal_mean_speed_cm_s,
            "whole_event_speed_eligible": table.whole_event_speed_eligible,
        }
    )
    return _finish(frame)
