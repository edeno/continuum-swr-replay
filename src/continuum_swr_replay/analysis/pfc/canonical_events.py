"""The PFC analyses' entry point to the canonical non-local events and SWRs.

Every family in this paper must describe the same events. The hippocampal
content and rate ladders read the shipped event table
(``Processed-Data/replay_content/event_table.parquet``) and the speed-gated
``SWR_event`` labels of the SWR source
(:mod:`~continuum_swr_replay.analysis.hippocampus.swr_source`, the direct Kay
inventory) on the cached ``*_HPC_80_data_over_time.pkl`` frame's grid; this
module is the one place the PFC path loads those, so no PFC script re-detects
non-local events or re-splits them on a separately detected ripple population.

Two definitions are fixed here and nowhere else on the PFC path:

- **A non-local event** is a row of the shipped event table. That table is
  :func:`~continuum_swr_replay.analysis.hippocampus.event_table.build_classified_nonlocal_event_table`'s
  output on the canonical cached inputs, so an event carries the same
  ``event_id`` the content and rate tables key on.
- **``is_swr``** is that table's column: strict positive-duration overlap with
  the SWR source's speed-gated ``SWR_event`` labels. It is read, never
  recomputed.

The SWR interval table returned here is the *same* population that column was
built from -- ``SWR_event`` labels gated by the canonical whole-event
immobility rule (every sampled speed finite and strictly ``<`` the threshold)
and turned into sampled bounds by
:func:`~continuum_swr_replay.analysis.hippocampus.event_immobility.intervals_from_labels`.

The session key is the canonical ``(animal, session_date, epoch)`` triple,
which the PFC runners hold as ``nwb_file_name`` / ``epoch_name`` instead; the
translation is done here once, exactly as
:mod:`continuum_swr_replay.analysis.pfc.cohort` does it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table, resolved_table_path, sha256_file

__all__ = [
    "DEFAULT_EVENT_TABLE_PATH",
    "IS_SWR_DEFINITION",
    "canonical_event_provenance",
    "canonical_event_input_hashes",
    "load_session_canonical_events",
    "load_session_swr_intervals",
    "session_data_over_time_path",
    "session_swr_events_file",
]

#: The canonical per-event table every family's events are read from.
DEFAULT_EVENT_TABLE_PATH = PROCESSED_DATA_DIR / "replay_content" / "event_table.parquet"

#: Provenance labels for the two definitions this module fixes. Written to each
#: run's sidecar so an artifact states which event definition it was fit on.
IS_SWR_DEFINITION = "canonical_event_table"

#: Columns of the shipped event table this path reads, and the names the GLM
#: uses for the two timing columns.
_EVENT_COLUMN_RENAMES = {"event_start_s": "start_time", "event_end_s": "end_time"}
_EVENT_COLUMNS = (
    "event_id",
    "event_start_s",
    "event_end_s",
    "is_swr",
    "occupancy_state",
)


def load_session_canonical_events(
    nwb_file_name: str,
    epoch_name: str,
    *,
    event_table_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load one session's non-local events from the canonical event table.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"wilbur20210329_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.
    event_table_path : str or Path, optional
        Event table to read; defaults to :data:`DEFAULT_EVENT_TABLE_PATH`.

    Returns
    -------
    events : pandas.DataFrame
        The session's events, sorted by ``start_time``, with a fresh integer
        index. Columns ``event_id``, ``start_time``, ``end_time``, ``is_swr``,
        ``occupancy_state``. ``start_time`` / ``end_time`` are absolute seconds
        on ``position_info.index``'s clock.
    sha256 : str
        Hash of the table that was read, for the run's provenance sidecar.

    Raises
    ------
    FileNotFoundError
        If no event table exists at ``event_table_path``.
    KeyError
        If the table lacks the session or event columns.
    SwrSourceError
        If the table was not built from the configured SWR source
        (``swr_source.require_swr_source``). Its ``is_swr`` column is combined
        with :func:`load_session_swr_intervals`' live source intervals, so the
        two must be one source.
    ValueError
        If the session has no events at all -- a missing session is a data
        problem, not an empty cohort, and re-detecting events for it would
        reintroduce exactly the per-path event definition this module exists
        to remove.
    """
    path = (
        DEFAULT_EVENT_TABLE_PATH if event_table_path is None else Path(event_table_path)
    )
    animal, session_date = parse_nwb_filename(nwb_file_name)

    event_table = read_table(path)
    swr_source.require_swr_source(event_table, what=f"the event table at {path}")
    missing = [
        column
        for column in (*SESSION_COLUMNS, *_EVENT_COLUMNS)
        if column not in event_table.columns
    ]
    if missing:
        raise KeyError(f"event table at {path} is missing column(s) {missing}")

    is_session = (
        (event_table["animal"] == animal)
        & (event_table["session_date"] == session_date)
        & (event_table["epoch"] == epoch_name)
    )
    events = event_table.loc[is_session, list(_EVENT_COLUMNS)]
    if events.empty:
        raise ValueError(
            f"no events for ({animal}, {session_date}, {epoch_name}) in {path}; "
            "the session is absent from the canonical event table, so its "
            "non-local events cannot be resolved"
        )

    events = (
        events.rename(columns=_EVENT_COLUMN_RENAMES)
        .sort_values("start_time")
        .reset_index(drop=True)
    )
    # Hash the file ``read_table`` resolved, not the parquet name: a directory
    # still holding the pickle form would otherwise be hashed at a path that
    # does not exist.
    return events, sha256_file(resolved_table_path(path))


def session_data_over_time_path(
    nwb_file_name: str,
    epoch_name: str,
    *,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    data_dir: str | Path | None = None,
) -> Path:
    """Path of one session's cached per-bin decode frame.

    The threshold tag in the name is the one the decoder wrote
    (``int(non_local_threshold * 100)``), so a run reads the labels that were
    cut at the threshold it asks for rather than whichever file exists.
    """
    animal, session_date = parse_nwb_filename(nwb_file_name)
    directory = PROCESSED_DATA_DIR if data_dir is None else Path(data_dir)
    threshold_int = int(non_local_threshold * 100)
    return directory / (
        f"{animal}{session_date}_{epoch_name}_HPC_{threshold_int}_data_over_time.pkl"
    )


def load_session_swr_intervals(
    nwb_file_name: str,
    epoch_name: str,
    *,
    dot_path: str | Path | None = None,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[pd.DataFrame, str]:
    """Load one session's SWR intervals: the SWR source's, on the cached frame's grid.

    The cached decode frame is read through the SWR source
    (:func:`~continuum_swr_replay.analysis.hippocampus.swr_source.read_dot_with_swr_source`),
    so its ``SWR_event`` labels are the direct Kay inventory's, not the
    decode's stored ripples. The intervals are then built exactly as
    :func:`~continuum_swr_replay.analysis.hippocampus.event_table.build_classified_nonlocal_event_table`
    builds them -- the labels gated atomically by the canonical whole-event
    immobility rule, then reduced to their sampled bounds -- so the population
    here and the population the canonical ``is_swr`` column was computed against
    are the same one.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"wilbur20210329_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.
    dot_path : str or Path, optional
        Cached decode frame to read; defaults to
        :func:`session_data_over_time_path`.
    non_local_threshold : float
        Threshold tag of the cached frame, used only to name it.
    speed_threshold : float
        Immobility threshold in cm/s. The rule is strictly ``<``.

    Returns
    -------
    swr_intervals : pandas.DataFrame
        Columns ``start_time`` and ``end_time`` in absolute seconds, sorted by
        ``start_time``, with a fresh integer index. Empty (correct columns,
        zero rows) if the session has no speed-eligible SWR.
    sha256 : str
        Hash of the frame that was read, for the run's provenance sidecar. The
        SWR source's own files are hashed by
        :func:`canonical_event_input_hashes`.

    Raises
    ------
    FileNotFoundError
        If the cached frame does not exist.
    ValueError
        If the frame lacks ``head_speed`` or the session columns.
    SwrSourceError
        If the SWR source cannot serve the session.
    """
    path = (
        session_data_over_time_path(
            nwb_file_name, epoch_name, non_local_threshold=non_local_threshold
        )
        if dot_path is None
        else Path(dot_path)
    )
    if not path.exists():
        raise FileNotFoundError(
            f"no cached decode frame at {path}; the session's SWR labels are "
            "the canonical SWR population and cannot be re-detected here"
        )

    dot = swr_source.read_dot_with_swr_source(path)
    gated = filter_event_labels_by_speed(
        dot, ["SWR_event"], speed_threshold=speed_threshold
    )
    intervals = intervals_from_labels(gated, "SWR_event")
    intervals = (
        intervals.loc[:, ["start_time", "end_time"]]
        .sort_values("start_time")
        .reset_index(drop=True)
        .astype(float)
    )
    return intervals, sha256_file(path)


def canonical_event_provenance(
    *,
    event_table_path: str | Path,
    event_table_sha256: str | None,
    data_over_time_path: str | Path,
    data_over_time_sha256: str | None,
    swr_events_path: str | Path | None,
    swr_events_sha256: str | None,
) -> dict[str, str | None]:
    """The sidecar block naming the event definition and the inputs it came from.

    Defined once so the runner's skip check and the run's own sidecar cannot
    disagree about the key names: a mismatch there would silently skip a
    session whose events had changed. ``swr_interval_source`` names the SWR
    source (``swr_source_name()``) and ``swr_events_*`` the inventory file the SWR
    intervals were read from.
    """
    return {
        "is_swr_definition": IS_SWR_DEFINITION,
        "immobility_inequality": "<",
        "canonical_event_table_path": str(event_table_path),
        "canonical_event_table_sha256": event_table_sha256,
        "swr_interval_source": swr_source.swr_source_name(),
        "swr_events_path": None if swr_events_path is None else str(swr_events_path),
        "swr_events_sha256": swr_events_sha256,
        "data_over_time_path": str(data_over_time_path),
        "data_over_time_sha256": data_over_time_sha256,
    }


def session_swr_events_file(
    nwb_file_name: str, epoch_name: str
) -> dict[str, str] | dict[str, None]:
    """The SWR source's events file for a session, checked against its manifest.

    ``{"path", "sha256"}``, both ``None`` when the source cannot serve the
    session: that matches no bundle, so the session is refit and the loader
    then fails loudly.
    """
    try:
        return swr_source.swr_source_files(
            swr_source.swr_epoch_id(nwb_file_name, epoch_name)
        )["events"]
    except swr_source.SwrSourceError:
        return {"path": None, "sha256": None}


def canonical_event_input_hashes(
    nwb_file_name: str,
    epoch_name: str,
    *,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    event_table_path: str | Path | None = None,
) -> dict[str, str | None]:
    """:func:`canonical_event_provenance` for a session, hashing from disk.

    No input is loaded, so a caller deciding whether a session needs refitting
    pays three hash passes rather than a table read. A missing input (or an
    epoch the SWR source cannot serve) hashes to ``None``, which matches no
    bundle -- the session is then refit (and fails loudly in the loader) rather
    than skipped.
    """
    table = (
        DEFAULT_EVENT_TABLE_PATH if event_table_path is None else Path(event_table_path)
    )
    resolved_table = resolved_table_path(table)
    dot = session_data_over_time_path(
        nwb_file_name, epoch_name, non_local_threshold=non_local_threshold
    )
    swr_events = session_swr_events_file(nwb_file_name, epoch_name)
    return canonical_event_provenance(
        event_table_path=resolved_table,
        event_table_sha256=(
            sha256_file(resolved_table) if resolved_table.exists() else None
        ),
        data_over_time_path=dot,
        data_over_time_sha256=sha256_file(dot) if dot.exists() else None,
        swr_events_path=swr_events["path"],
        swr_events_sha256=swr_events["sha256"],
    )
