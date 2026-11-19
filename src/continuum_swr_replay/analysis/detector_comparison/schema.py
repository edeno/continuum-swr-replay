"""One table per detector, in one schema.

Each inventory is one row per event, over the same epochs and speed trace.
The span of an event is ``event_end_s - event_start_s`` for every detector,
because the native durations follow different conventions: our canonical
table adds one sample width to the span, the memoryless detector does not.
The native duration is kept beside the span and never stands in for it.
Keys name the detector and the epoch, ``"<detector>:<animal><day>_<epoch>:<native
key>"``, so no two events of any comparison share one: an SWR detector's native
keys restart in every epoch, and a join on a key that repeated across epochs
would match events that never overlapped.

Detector-specific statistics stay in each detector's own table; this schema
carries only what every detector has.
"""

from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.immobility import (
    event_speeds_are_immobile,
)
from continuum_swr_replay.data_loaders.constants import BRAIN_AREAS
from continuum_swr_replay.decoding.identifiers import nwb_file_name, session_name
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "INVENTORY_DTYPES",
    "MEMORYLESS_COHORTS",
    "SessionIdentity",
    "combine_inventories",
    "empty_inventory",
    "event_keys",
    "from_classified_nonlocal_table",
    "from_memoryless_table",
    "inventory_from_intervals",
    "native_event_keys",
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
MEMORYLESS_CRITERIA = "memoryless_criteria"
"""The memoryless detector's ``replay_criteria`` cohort, compared as an
inventory of its own."""

MEMORYLESS_COHORTS: Mapping[str, str] = MappingProxyType(
    {"published": "shuffle_pass", "replay_criteria": "is_replay"}
)
"""Which rows of the memoryless table make a cohort: every row of that table
is a candidate, accepted or not, so an adapter must say which it means.
``published`` is the paper's definition (duration, dispersion and the
significance test); ``replay_criteria`` leaves the significance test out."""


@dataclass(frozen=True)
class SessionIdentity:
    """The epoch an inventory's events come from.

    Attributes
    ----------
    animal, epoch : str
    day : int
        The recording date as an integer, as the session columns carry it.
    brain_area : str
        One of the project's names; capitalization matters.

    """

    animal: str
    day: int
    epoch: str
    brain_area: str

    def __post_init__(self) -> None:
        for name in ("animal", "epoch", "brain_area"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
        if isinstance(self.day, bool) or not isinstance(self.day, int | np.integer):
            raise TypeError(f"day must be an integer, got {self.day!r}.")
        if self.day <= 0:
            raise ValueError(f"day must be positive, got {self.day}.")
        if self.brain_area not in BRAIN_AREAS:
            raise ValueError(
                f"brain_area must be one of {BRAIN_AREAS}, got {self.brain_area!r}."
            )


def empty_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in INVENTORY_DTYPES.items()}
    )


def validate_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Check the schema's invariants and return the table in schema order.

    Raises
    ------
    ValueError
        A column is missing; a session column, the eligibility flag or a
        bound is missing or not finite; an event has no positive duration;
        the span or midpoint does not follow from the bounds; a key repeats
        or does not name its own detector and epoch; a detector name holds
        the key separator; one epoch carries two brain areas; or two events
        of one detector in one epoch overlap.
    """
    missing = [column for column in INVENTORY_DTYPES if column not in inventory]
    if missing:
        raise ValueError(f"The inventory lacks {missing}.")
    session_columns = list(SESSION_COLUMNS_DAY)
    if inventory[[*session_columns, "detector"]].isna().any().any():
        raise ValueError(
            "Every event needs its animal, day, epoch and detector; a missing one "
            "would drop the event from every grouped count without a word."
        )
    if inventory.whole_event_speed_eligible.isna().any():
        raise ValueError("whole_event_speed_eligible must be True or False, not NA.")
    table = inventory[list(INVENTORY_DTYPES)].astype(dict(INVENTORY_DTYPES))
    if table.detector.str.contains(":").any():
        raise ValueError("A detector name must not hold ':', the key separator.")
    start, end = table.event_start_s.to_numpy(), table.event_end_s.to_numpy()
    if not (
        np.isfinite(start).all() and np.isfinite(end).all() and (end > start).all()
    ):
        raise ValueError(
            "Every event needs finite bounds and a positive duration; an event of "
            "no duration can never overlap another under the strict rule."
        )
    native = table.native_duration_s.to_numpy()
    if not (np.isfinite(native).all() and (native > 0).all()):
        raise ValueError("native_duration_s must be finite and positive.")
    if not np.array_equal(table.interval_span_s.to_numpy(), end - start):
        raise ValueError("interval_span_s must be event_end_s - event_start_s.")
    if not np.allclose(table.event_midpoint_s.to_numpy(), (start + end) / 2):
        raise ValueError("event_midpoint_s must be halfway between the bounds.")
    if not table.event_key.is_unique:
        raise ValueError("An event key repeats; keys name the detector and epoch.")
    expected_prefix = (
        table.detector
        + ":"
        + table.animal
        + table.day.astype(str)
        + "_"
        + table.epoch
        + ":"
    )
    mismatched = [
        (key, prefix)
        for key, prefix in zip(table.event_key, expected_prefix, strict=True)
        if not key.startswith(prefix)
    ]
    if mismatched:
        raise ValueError(
            f"An event key does not name its own detector and epoch: {mismatched[0]}."
        )
    areas = table.groupby(session_columns, sort=False).brain_area.nunique()
    if (areas > 1).any():
        raise ValueError(
            "One epoch carries more than one brain area; the comparison keys on "
            "the epoch and would pair events across areas."
        )
    for key, group in table.groupby([*session_columns, "detector"], sort=False):
        *session, detector = key
        ordered = group.sort_values("event_start_s")
        if (
            ordered.event_start_s.to_numpy()[1:] < ordered.event_end_s.to_numpy()[:-1]
        ).any():
            raise ValueError(
                f"Two events of {detector} in {session} overlap; one detector's "
                "events must be disjoint for the recount and the counts to hold."
            )
    return table


def combine_inventories(
    inventories: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every inventory in one table, split by whole-event speed eligibility.

    Only speed-eligible events are compared: an event during movement cannot
    mark a still one as matched, as the canonical SWR labels are speed-gated
    before they mark a non-local event.

    Parameters
    ----------
    inventories : iterable of DataFrame
        Each in the common schema.

    Returns
    -------
    eligible : DataFrame
        The speed-eligible events, in the common schema.
    moving : DataFrame
        The rest, kept so native counts can be reported beside eligible ones.

    Raises
    ------
    ValueError
        A key repeats across the inventories.

    """
    combined = validate_inventory(
        pd.concat([empty_inventory(), *inventories], ignore_index=True)
    )
    eligible = combined.whole_event_speed_eligible
    return (
        combined[eligible].reset_index(drop=True),
        combined[~eligible].reset_index(drop=True),
    )


def event_keys(
    detector: str, animal: object, day: object, epoch: object, native: Sized
) -> list[str]:
    """``"<detector>:<animal><day>_<epoch>:<native key>"`` for each event.

    ``animal``, ``day`` and ``epoch`` are one value or one per event; the middle
    part is ``identifiers.session_name``. :func:`native_event_keys` reads the
    native key back.

    Parameters
    ----------
    detector : str
        The inventory's detector; it must not hold ``":"``.
    animal, day, epoch : object or array-like
        One value, or one per event.
    native : sized, shape (n_events,)
        Each event's key in its detector's own output.

    Returns
    -------
    list of str, length n_events

    Raises
    ------
    ValueError
        `detector` holds ``":"``.
    """
    if ":" in detector:
        raise ValueError(f"A detector name must not hold ':', got {detector!r}.")
    n_events = len(native)
    columns = [
        np.broadcast_to(np.asarray(values, dtype=object), (n_events,))
        for values in (animal, day, epoch, native)
    ]
    return [
        f"{detector}:{session_name(nwb_file_name(a, d), e)}:{k}"
        for a, d, e, k in zip(*columns, strict=True)
    ]


def native_event_keys(event_keys: pd.Series) -> pd.Series:
    """The detector's own key of each ``event_key``: what follows its session.

    An event key is ``"<detector>:<animal><day>_<epoch>:<native key>"``; neither
    the detector nor the session holds a ``:``, so the native key is everything
    after the second one.

    Parameters
    ----------
    event_keys : pd.Series of str

    Returns
    -------
    pd.Series of str
        The native keys, on the index of `event_keys`.

    Raises
    ------
    ValueError
        If a key has fewer than three ``:``-separated parts.
    """
    parts = event_keys.str.split(":", n=2)
    if not (parts.str.len() == 3).all():
        bad = event_keys[parts.str.len() != 3].iloc[0]
        raise ValueError(f"{bad!r} is not <detector>:<session>:<native key>.")
    return parts.str[2]


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
    if time.shape != speed.shape or np.any(np.diff(time) <= 0):
        raise ValueError(
            "speed_time_s must be strictly increasing and match speed_cm_per_s."
        )
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
            "event_key": event_keys(
                detector, session.animal, session.day, session.epoch, keys
            ),
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
    """Our state-space cohort, from ``build_classified_nonlocal_event_table``.

    Every row of that table is a detected non-local event, so every row is
    an event of the cohort; whole-event speed eligibility travels as a flag
    and is applied to every inventory alike by :func:`combine_inventories`.
    """
    require_columns(
        classified,
        (
            "animal",
            "day",
            "epoch",
            "event_id",
            "event_start",
            "event_end",
            "event_duration",
            "animal_max_speed_cm_s",
            "animal_mean_speed_cm_s",
            "whole_event_speed_eligible",
        ),
        "the classified non-local event table",
    )
    frame = pd.DataFrame(
        {
            "animal": classified.animal,
            "day": classified.day,
            "epoch": classified.epoch,
            "brain_area": brain_area,
            "detector": STATE_SPACE,
            "event_key": event_keys(
                STATE_SPACE,
                classified.animal,
                classified.day,
                classified.epoch,
                classified.event_id,
            ),
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


def from_memoryless_table(table: pd.DataFrame, *, cohort: str) -> pd.DataFrame:
    """One memoryless cohort, from ``build_memoryless_event_table``.

    That table is one row per candidate sequence, accepted or not, so this
    adapter takes only the rows the named cohort selects; passing every row
    would inflate the memoryless inventory with the detector's own rejections
    and every count and denominator built on it.

    Parameters
    ----------
    table : DataFrame
        From ``build_memoryless_event_table``.
    cohort : str
        A key of :data:`MEMORYLESS_COHORTS`: ``"published"`` for the paper's
        replay definition, ``"replay_criteria"`` for duration and dispersion
        alone. ``"published"`` needs every candidate meeting the replay
        criteria to have been shuffled. A candidate with an
        ``artifact_drop_reason`` (excluded time reaches it) is in neither.

    Raises
    ------
    ValueError
        If the cohort is unknown, a column is missing, a published cohort was
        not shuffled, an artifact run is configured and the table has no
        ``artifact_drop_reason`` (it was made before the rule), or the table's
        ``swr_definition`` is not the configured source
        (``swr_source.swr_source_name()``): another artifact run (no run
        included) would make its ``artifact_drop_reason`` read another mask,
        and another inventory run its SWR labels another inventory's.

    """
    if cohort not in MEMORYLESS_COHORTS:
        raise ValueError(
            f"cohort must be one of {list(MEMORYLESS_COHORTS)}, got {cohort!r}."
        )
    flag = MEMORYLESS_COHORTS[cohort]
    require_columns(
        table,
        (
            "animal",
            "day",
            "epoch",
            "brain_area",
            "run_id",
            "ms_event_id",
            "event_start_s",
            "event_end_s",
            "event_duration_s",
            "animal_max_speed_cm_s",
            "animal_mean_speed_cm_s",
            "whole_event_speed_eligible",
            "is_replay",
            "shuffle_pass",
            "n_shuffles",
            "swr_definition",
        ),
        "the memoryless event table",
    )
    untested = table.is_replay.to_numpy(dtype=bool) & (table.n_shuffles.to_numpy() == 0)
    if cohort == "published" and untested.any():
        raise ValueError(
            f"{int(untested.sum())} candidates meeting the replay criteria were never "
            "shuffled, so the published cohort cannot be read from this table: a "
            "run made without the shuffle passes nothing and would read as a "
            "detector that found no replay."
        )
    selected = table[flag].to_numpy(dtype=bool)
    if "artifact_drop_reason" in table.columns:
        # a candidate in excluded time is not analyzed: in no cohort
        # not in place: the flag's array can be the caller's column itself
        selected = selected & table["artifact_drop_reason"].isna().to_numpy()
    elif swr_source.ARTIFACT_INTERVALS_RUN_ID is not None:
        raise ValueError(
            "The memoryless table has no artifact_drop_reason column, but the "
            f"artifact run {swr_source.ARTIFACT_INTERVALS_RUN_ID!r} is "
            "configured: a table made before the analyzed-time rule would read "
            "its candidates in excluded time as kept. Rerun the detector."
        )
    configured = swr_source.swr_source_name()
    built_from = {str(name) for name in table["swr_definition"].unique()}
    if any(name != configured for name in built_from):
        raise ValueError(
            f"The memoryless table was built from SWR source(s) {sorted(built_from)} "
            f"(its swr_definition), not the configured {configured!r}: its SWR "
            "labels or its artifact_drop_reason read another inventory or mask. "
            "Rerun the detector under the configured source."
        )
    table = table[selected]
    frame = pd.DataFrame(
        {
            "animal": table.animal,
            "day": table.day,
            "epoch": table.epoch,
            "brain_area": table.brain_area,
            "detector": MEMORYLESS,
            "event_key": event_keys(
                MEMORYLESS, table.animal, table.day, table.epoch, table.ms_event_id
            ),
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


def typed_table(
    frames: Iterable[pd.DataFrame], dtypes: Mapping[str, object]
) -> pd.DataFrame:
    """The frames in one table with the columns and dtypes given."""
    rows = [frame for frame in frames if len(frame)]
    table = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            {name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()}
        )
    )
    return table[list(dtypes)].astype(dict(dtypes)).reset_index(drop=True)
