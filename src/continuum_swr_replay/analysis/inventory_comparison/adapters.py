"""Each detector's recorded outputs for one epoch, on the common schema.

One adapter per inventory reads what its detector recorded for an epoch and
returns three tables (:class:`InventoryTables`): the inventory, with its
moving events kept and flagged; what the detector observed; and the events the
analysed-time rule sets aside, with their reasons. The accounting modules do
no IO; the readers of recorded outputs are here.

Every adapter reads one epoch's canonical decode frame, read once through the
SWR source (``analysed_time.read_dot_with_swr_source``), and its drops, read once
from the same pickle (``analysed_time.read_artifact_event_drops``): the frame as
saved, never ``artifact_event_drops`` of the read frame, whose dropped labels
are already gone. The frame's ``head_speed`` on its grid is the one speed
trace, and the grid's first and last time are the epoch's bounds.

An event is set aside, and leaves the inventory, when the one rule drops it
for excluded time (``artifact``, ``near_artifact``) or for overlapping an SWR
or MUA event dropped for excluded time (``overlaps_dropped_event``). That
second pass reads every event at its bounds sampled on the decode grid, as
``analysed_time`` reads the paper's SWRs, so Kay's kept events are exactly the
ones ``analysed_time.with_swr_source`` keeps. A moving event stays in the inventory with ``whole_event_speed_eligible`` False. So a
detector's native count is its inventory's rows plus its set-aside rows.

A run whose record for the epoch cannot be used (no manifest, the epoch not in
it, a status other than ``ok`` or ``partial``, the detector not ``ok``, a file
missing or changed since the run, events not numbered 1..n, two LFP events
or memoryless candidates sharing a sample of the decode grid, a memoryless
record that does not describe its table or the frame's grid) observed none of
the epoch: its adapter returns an empty inventory and a ``missing_run_table``
row saying why.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.events import analysed_time, swr_source
from continuum_swr_replay.analysis.events.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.eligibility import (
    EXCLUDED_TIME_REASONS,
    DropReason,
    event_drop_reasons,
    overlaps_dropped_events,
)
from continuum_swr_replay.analysis.events.immobility import (
    EventDefinitionError,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.hippocampus.detection_inputs import (
    carey_output_paths,
    memoryless_output_paths,
)
from continuum_swr_replay.analysis.inventory_comparison.availability import (
    missing_run_table,
    observation_table,
    sample_runs,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    SET_ASIDE_DTYPES,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    MEMORYLESS_COHORTS,
    STATE_SPACE,
    SessionIdentity,
    _event_keys,
    empty_inventory,
    from_classified_nonlocal_table,
    from_memoryless_table,
    inventory_from_intervals,
    validate_inventory,
)
from continuum_swr_replay.intervals import intersect, interval_mask
from continuum_swr_replay.runs.io import read_manifest, sha256_file
from continuum_swr_replay.tables.event_ids import hash_event_id
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "NATIVE_IS_ELIGIBLE",
    "SWR_DETECTORS",
    "InventoryTables",
    "carey_inventory",
    "memoryless_inventory",
    "mua_inventory",
    "state_space_inventory",
    "swr_detector_inventory",
]

SWR_DETECTORS = ("kay", "karlsson", "yu", "zugaro")
"""The SWR runner's inventories, one events file each per epoch."""

CAREY = "carey"
MUA = "mua"
BRAIN_AREA = "HPC"
"""Every inventory compared here is CA1's: the decode, its MUA, and the LFP."""

NATIVE_IS_ELIGIBLE: Mapping[str, str] = MappingProxyType(
    {
        detector: (
            f"The decode labels only speed-eligible {kind} events "
            "(build_speed_eligible_event_cohort in get_data_over_time), so the "
            "native count is the eligible count (before the events set aside "
            "for excluded time): its moving events were never stored, and none "
            "are recounted from the marks."
        )
        for detector, kind in ((MUA, "MUA"), (STATE_SPACE, "non-local"))
    }
)
"""The inventories whose native count equals their eligible count, and why;
for the counts' manifest and the report."""

SET_ASIDE_REASONS = EXCLUDED_TIME_REASONS | {DropReason.OVERLAPS_DROPPED_EVENT.value}
"""The rule's reasons that take an event out of its inventory."""


class InventoryTables(NamedTuple):
    """One detector's tables for one epoch."""

    inventory: pd.DataFrame
    """Columns and dtypes as ``schema.INVENTORY_DTYPES``."""
    observations: pd.DataFrame
    """Columns and dtypes as ``availability.OBSERVATION_DTYPES``."""
    set_aside: pd.DataFrame
    """Columns and dtypes as ``overlap.SET_ASIDE_DTYPES``."""


class _Unavailable(Exception):
    """The run's record for the epoch cannot be used; the message says why."""


class _Epoch(NamedTuple):
    session: SessionIdentity
    epoch_id: str
    time: NDArray
    speed: NDArray
    bounds: tuple[float, float]


def _epoch(dot: pd.DataFrame) -> _Epoch:
    """The one epoch a frame read through the SWR source holds."""
    analysed_time.require_swr_source(dot, what="the decode frame")
    sessions = dot[list(SESSION_COLUMNS_DAY)].drop_duplicates()
    if len(sessions) != 1:
        raise ValueError(f"An adapter reads one epoch at a time, not {len(sessions)}.")
    animal, day, epoch = sessions.iloc[0]
    time = dot.index.to_numpy(dtype=np.float64)
    return _Epoch(
        session=SessionIdentity(
            animal=str(animal), day=int(day), epoch=str(epoch), brain_area=BRAIN_AREA
        ),
        epoch_id=swr_source.swr_epoch_id_of_session(animal, day, epoch),
        time=time,
        speed=dot["head_speed"].to_numpy(dtype=np.float64),
        bounds=(float(time[0]), float(time[-1])),
    )


def _set_aside(
    epoch: _Epoch,
    detector: str,
    native_keys: NDArray,
    starts: NDArray,
    ends: NDArray,
    reasons: NDArray,
) -> pd.DataFrame:
    session = epoch.session
    frame = pd.DataFrame(
        {
            "animal": session.animal,
            "day": session.day,
            "epoch": session.epoch,
            "detector": detector,
            "event_key": _event_keys(
                detector, session.animal, session.day, session.epoch, native_keys
            ),
            "interval_start_s": np.asarray(starts, dtype=np.float64),
            "interval_end_s": np.asarray(ends, dtype=np.float64),
            "reason": np.asarray(reasons, dtype=object),
        }
    )
    return frame[list(SET_ASIDE_DTYPES)].astype(dict(SET_ASIDE_DTYPES))


def _no_set_aside() -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in SET_ASIDE_DTYPES.items()}
    )


def _observations(
    epoch: _Epoch, detector: str, run_id: str, observed: NDArray, reason: str
) -> pd.DataFrame:
    """The observation table over the decode grid's span; what the detector
    observed outside it is outside the epoch every detector is compared on."""
    within = intersect(
        np.asarray(observed, dtype=np.float64).reshape(-1, 2), np.array([epoch.bounds])
    )
    return observation_table(
        epoch.session,
        detector=detector,
        run_id=run_id,
        epoch_bounds=epoch.bounds,
        observed=within,
        unavailable_reason=reason,
    )


def _unavailable(
    epoch: _Epoch, detector: str, run_id: str, reason: str
) -> InventoryTables:
    return InventoryTables(
        inventory=empty_inventory(),
        observations=missing_run_table(
            epoch.session,
            detector=detector,
            run_id=run_id,
            epoch_bounds=epoch.bounds,
            reason=reason,
        ),
        set_aside=_no_set_aside(),
    )


def _run_files(
    run_dir: Path,
    epoch_id: str,
    files: Mapping[str, str],
    *,
    detector: str | None = None,
    hashed: bool = True,
) -> tuple[dict[str, Path], dict]:
    """The epoch's files in a run folder, checked against its manifest.

    The check ``swr_source.swr_source_files`` makes for Kay, for any run
    folder: the epoch's record must be ``ok`` or ``partial``, ``detector`` (if
    given) must be ``ok`` in its ``detectors``, and each file must exist with
    the sha256 the manifest recorded for it; an output with no sha256 is a
    mismatch. The memoryless runner records each output's path and no sha256,
    so its files (``hashed=False``) are checked for presence only.

    Parameters
    ----------
    run_dir : Path
    epoch_id : str
    files : mapping
        Manifest output name -> the file's name in ``run_dir``.
    detector : str, optional
        The detector whose status the record must give as ``ok``.
    hashed : bool
        Whether the manifest records each output's sha256.

    Returns
    -------
    paths : dict
        Output name -> path.
    record : dict
        The epoch's manifest record.

    Raises
    ------
    _Unavailable
        The record cannot be used, with why.
    ValueError
        The manifest is another run's: the folder is not the run named.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise _Unavailable(f"no manifest at {manifest_path}")
    manifest = read_manifest(run_dir)
    if manifest.get("run_id") != run_dir.name:
        raise ValueError(
            f"{manifest_path} is run {manifest.get('run_id')!r}, not the folder's "
            f"{run_dir.name!r}."
        )
    record = manifest.get("epochs", {}).get(epoch_id)
    if record is None:
        raise _Unavailable(f"{epoch_id} is not in run {run_dir.name}")
    status = record.get("status")
    if status not in swr_source.USABLE_STATUSES:
        why = record.get("error") or record.get("reason") or "no reason recorded"
        raise _Unavailable(f"{status}: {str(why)[:200]}")
    if detector is not None:
        detector_status = record.get("detectors", {}).get(detector)
        if detector_status != "ok":
            raise _Unavailable(f"{detector} {detector_status!r} in a {status!r} record")
    paths = {}
    for name, file_name in files.items():
        path = run_dir / file_name
        if not path.exists():
            raise _Unavailable(f"{file_name} is missing")
        if hashed:
            output = record.get("outputs", {}).get(name)
            expected = output.get("sha256") if isinstance(output, Mapping) else None
            if sha256_file(path) != expected:
                raise _Unavailable(
                    f"{file_name} has sha256 other than the manifest's "
                    f"({str(expected)[:12]})"
                )
        paths[name] = path
    return paths, record


def _numbered_events(path: Path) -> pd.DataFrame:
    """An LFP detector's events, indexed ``event_number`` 1..n, as
    ``swr_source`` requires of Kay's.

    Raises
    ------
    _Unavailable
        The table is not so indexed.
    """
    events = pd.read_parquet(path)
    if events.index.name != "event_number" or not np.array_equal(
        events.index.to_numpy(), np.arange(1, len(events) + 1)
    ):
        raise _Unavailable(f"{path.name} is not indexed by event_number 1..n")
    return events


def _sampled_bounds(intervals: pd.DataFrame, epoch: _Epoch) -> NDArray:
    """Each interval's first and last sample on the decode grid
    (``analysed_time.sampled_bounds``), as the rule's second pass reads them.

    Returns
    -------
    bounds : ndarray, shape (n_intervals, 2)
        NaN for an interval the speed rule rejects, which the pass never reads.
    """
    _, bounds = analysed_time.sampled_bounds(
        intervals[["start_time", "end_time"]], epoch.time, epoch.speed
    )
    return bounds


def _lfp_inventory(
    detector: str,
    run_dir: Path,
    events: pd.DataFrame,
    observed: NDArray,
    epoch: _Epoch,
    drops: pd.DataFrame,
) -> InventoryTables:
    """An LFP detector's events under the one rule: excluded time with the
    artifact mask and the speed rule (``event_drop_reasons``), then its second
    pass on the grid-sampled bounds of each kept event and of the SWR and MUA
    events dropped for excluded time, as ``analysed_time`` makes it.

    Two events sharing a sample of the grid cannot both be labelled on it, so
    the second pass cannot read them: the detector is then unavailable for the
    epoch, with the error as its reason."""
    starts = events["start_time"].to_numpy(dtype=np.float64)
    ends = events["end_time"].to_numpy(dtype=np.float64)
    keys = events.index.to_numpy()
    reasons = event_drop_reasons(
        events[["start_time", "end_time"]],
        speed=epoch.speed,
        time=epoch.time,
        artifacts=swr_source.artifact_intervals(epoch.epoch_id),
    )
    try:
        sampled = _sampled_bounds(events, epoch)
    except EventDefinitionError as error:
        return _unavailable(
            epoch,
            detector,
            run_dir.name,
            f"two events share a decode-grid sample: {str(error)[:200]}",
        )
    kept = np.flatnonzero(pd.isna(reasons) & np.isfinite(sampled[:, 0]))
    overlapping = overlaps_dropped_events(
        sampled[kept, 0], sampled[kept, 1], analysed_time.dropped_on_grid(drops)
    )
    reasons[kept[overlapping]] = DropReason.OVERLAPS_DROPPED_EVENT.value
    aside = np.isin(reasons.astype(str), list(SET_ASIDE_REASONS))
    inventory = inventory_from_intervals(
        starts[~aside],
        ends[~aside],
        native_keys=keys[~aside],
        detector=detector,
        session=epoch.session,
        run_id=run_dir.name,
        speed_time_s=epoch.time,
        speed_cm_per_s=epoch.speed,
        native_duration_s=events["duration"].to_numpy(dtype=np.float64)[~aside],
    )
    return InventoryTables(
        inventory=inventory,
        observations=_observations(
            epoch, detector, run_dir.name, observed, "lfp_not_observed"
        ),
        set_aside=_set_aside(
            epoch, detector, keys[aside], starts[aside], ends[aside], reasons[aside]
        ),
    )


def swr_detector_inventory(
    run_dir: Path, detector: str, *, dot: pd.DataFrame, drops: pd.DataFrame
) -> InventoryTables:
    """One SWR detector's inventory from the SWR runner's run folder.

    Parameters
    ----------
    run_dir : Path
        The run folder, ``.../replay_detection_comparison/swr/<run id>``.
    detector : str
        One of :data:`SWR_DETECTORS`. Its events are
        ``{epoch_id}_{detector}_events.parquet`` (native key ``event_number``,
        native duration the table's ``duration``); its observed time is the
        ``swr_observed`` file's ``stored_band`` scope, ``zugaro`` for Zugaro.
    dot : DataFrame
        The epoch's decode frame, read through the SWR source.
    drops : DataFrame
        ``analysed_time.read_artifact_event_drops`` of the decode pickle, the
        frame as saved; never ``artifact_event_drops`` of the read frame,
        whose dropped labels are already gone.

    Returns
    -------
    InventoryTables
        Kay's eligible events are exactly the paper's SWRs, the ones
        ``analysed_time.with_swr_source`` keeps in the same frame.

    """
    if detector not in SWR_DETECTORS:
        raise ValueError(f"detector must be one of {SWR_DETECTORS}, got {detector!r}.")
    epoch = _epoch(dot)
    try:
        files, _ = _run_files(
            run_dir,
            epoch.epoch_id,
            {
                name: path.name
                for name, path in swr_source.swr_inventory_paths(
                    epoch.epoch_id, run_dir, (detector,)
                ).items()
            },
            detector=detector,
        )
        events = _numbered_events(files[detector])
    except _Unavailable as why:
        return _unavailable(epoch, detector, run_dir.name, str(why))
    scope = "zugaro" if detector == "zugaro" else swr_source.OBSERVED_SCOPE
    observed = pd.read_parquet(files["swr_observed"])
    stretches = observed.loc[observed["scope"] == scope, ["start_s", "end_s"]]
    return _lfp_inventory(detector, run_dir, events, stretches.to_numpy(), epoch, drops)


def carey_inventory(
    run_dir: Path, *, dot: pd.DataFrame, drops: pd.DataFrame
) -> InventoryTables:
    """Carey's candidate events from the Carey runner's run folder.

    ``{epoch_id}_carey_events.parquet`` and its observed stretches,
    ``{epoch_id}_carey_observed.parquet`` (scope ``carey``), under the same rule as the SWR
    inventories (:func:`swr_detector_inventory`, whose parameters these are).
    """
    epoch = _epoch(dot)
    try:
        files, _ = _run_files(
            run_dir,
            epoch.epoch_id,
            {
                name: path.name
                for name, path in carey_output_paths(epoch.epoch_id, run_dir).items()
            },
        )
        events = _numbered_events(files["events"])
    except _Unavailable as why:
        return _unavailable(epoch, CAREY, run_dir.name, str(why))
    observed = pd.read_parquet(files["observed"])
    stretches = observed.loc[observed["scope"] == CAREY, ["start_s", "end_s"]]
    return _lfp_inventory(CAREY, run_dir, events, stretches.to_numpy(), epoch, drops)


def _label_set_aside(
    epoch: _Epoch, detector: str, drops: pd.DataFrame, event_col: str
) -> pd.DataFrame:
    """The frame's ``event_col`` events the rule dropped, keyed by label."""
    rows = drops[(drops["event_col"] == event_col) & drops["drop_reason"].notna()]
    return _set_aside(
        epoch,
        detector,
        rows["label"].to_numpy(),
        rows["start_time"].to_numpy(),
        rows["end_time"].to_numpy(),
        rows["drop_reason"].to_numpy(),
    )


def _outside_mask(epoch: _Epoch) -> NDArray:
    mask = interval_mask(epoch.time, swr_source.artifact_intervals(epoch.epoch_id))
    outside: NDArray = ~mask
    return outside


def mua_inventory(
    dot: pd.DataFrame, *, drops: pd.DataFrame, run_id: str
) -> InventoryTables:
    """The decode frame's MUA events.

    The frame's ``MUA_event`` labels at their sampled bounds (native key the
    label); the frame read through the source holds only the kept ones, and
    the ones the rule dropped are set aside from ``drops``. Observed: the
    decode grid's runs of samples outside the artifact mask.

    Parameters
    ----------
    dot : DataFrame
        The epoch's decode frame, read through the SWR source.
    drops : DataFrame
        ``analysed_time.read_artifact_event_drops`` of the decode pickle, the
        frame as saved; never ``artifact_event_drops`` of the read frame,
        whose dropped labels are already gone.
    run_id : str
        Names the decode the labels come from.

    """
    epoch = _epoch(dot)
    events = intervals_from_labels(dot, "MUA_event")
    inventory = inventory_from_intervals(
        events["start_time"].to_numpy(),
        events["end_time"].to_numpy(),
        native_keys=events["MUA_event"].to_numpy(),
        detector=MUA,
        session=epoch.session,
        run_id=run_id,
        speed_time_s=epoch.time,
        speed_cm_per_s=epoch.speed,
    )
    return InventoryTables(
        inventory=inventory,
        observations=_observations(
            epoch,
            MUA,
            run_id,
            sample_runs(epoch.time, _outside_mask(epoch)),
            "in_artifact_mask",
        ),
        set_aside=_label_set_aside(epoch, MUA, drops, "MUA_event"),
    )


def state_space_inventory(
    dot: pd.DataFrame, *, drops: pd.DataFrame, run_id: str
) -> InventoryTables:
    """The state-space non-local events, from the canonical classified table.

    ``build_classified_nonlocal_event_table`` on the frame; the events the rule
    dropped are set aside from ``drops``, keyed by the ``event_id`` that table
    gives an event. Observed: the decode grid's runs of samples where the
    non-local probability is finite, outside the artifact mask. Parameters as
    :func:`mua_inventory`.
    """
    epoch = _epoch(dot)
    inventory = from_classified_nonlocal_table(
        build_classified_nonlocal_event_table(dot),
        brain_area=BRAIN_AREA,
        run_id=run_id,
    )
    rows = drops[
        (drops["event_col"] == "non_local_event") & drops["drop_reason"].notna()
    ]
    # the event_id the classified table hashes from the same values
    event_ids = hash_event_id(
        pd.DataFrame.from_records(
            [
                {
                    "animal": epoch.session.animal,
                    "day": dot["day"].iloc[0],
                    "epoch": epoch.session.epoch,
                    "non_local_event": int(label),
                }
                for label in rows["label"]
            ],
            columns=["animal", "day", "epoch", "non_local_event"],
        )
    )
    probability = dot["Non-Local_Continuous_prob"].to_numpy(dtype=np.float64) + dot[
        "Non-Local_Fragmented_prob"
    ].to_numpy(dtype=np.float64)
    return InventoryTables(
        inventory=inventory,
        observations=_observations(
            epoch,
            STATE_SPACE,
            run_id,
            sample_runs(epoch.time, np.isfinite(probability) & _outside_mask(epoch)),
            "no_posterior_or_in_artifact_mask",
        ),
        set_aside=_set_aside(
            epoch,
            STATE_SPACE,
            event_ids,
            rows["start_time"].to_numpy(),
            rows["end_time"].to_numpy(),
            rows["drop_reason"].to_numpy(),
        ),
    )


def _check_memoryless_record(record: dict, table: pd.DataFrame, epoch: _Epoch) -> None:
    """Raise :class:`_Unavailable` unless the record describes the table it
    names and was run on the frame's grid."""
    grid = record.get("time_grid", {})
    recorded = {
        "n_candidates": record.get("n_candidates"),
        "n_replay": record.get("n_replay"),
        "n_shuffle_pass": record.get("n_shuffle_pass"),
        "time_grid start_s": grid.get("start_s"),
        "time_grid end_s": grid.get("end_s"),
        "time_grid n_samples": grid.get("n_samples"),
    }
    found = {
        "n_candidates": len(table),
        "n_replay": int(table["is_replay"].sum()),
        "n_shuffle_pass": int(table["shuffle_pass"].sum()),
        "time_grid start_s": epoch.bounds[0],
        "time_grid end_s": epoch.bounds[1],
        "time_grid n_samples": epoch.time.size,
    }
    for name, value in found.items():
        if recorded[name] != value:
            raise _Unavailable(
                f"the record's {name} is {recorded[name]!r}, the table or frame "
                f"gives {value!r}"
            )


def _memoryless_second_pass(
    table: pd.DataFrame, flag: str, epoch: _Epoch, drops: pd.DataFrame
) -> NDArray:
    """Each candidate's ``artifact_drop_reason``, its second pass read on the
    grid as the LFP inventories' is.

    The stored excluded-time reasons stand. The stored
    ``overlaps_dropped_event`` is not read: the cohort's candidates outside
    excluded time are checked at their bounds sampled on the decode grid
    against the SWR and MUA events dropped for excluded time at theirs
    (``analysed_time.dropped_on_grid``).

    Raises
    ------
    EventDefinitionError
        Two of the checked candidates share a sample of the grid.
    """
    stored = table["artifact_drop_reason"].to_numpy(dtype=object)
    reasons: NDArray = stored.copy()
    reasons[~np.isin(stored.astype(str), list(EXCLUDED_TIME_REASONS))] = None
    checked = np.flatnonzero(table[flag].to_numpy(dtype=bool) & pd.isna(reasons))
    sampled = _sampled_bounds(
        pd.DataFrame(
            {
                "start_time": table["event_start_s"].to_numpy(dtype=np.float64),
                "end_time": table["event_end_s"].to_numpy(dtype=np.float64),
            }
        ).iloc[checked],
        epoch,
    )
    on_grid = np.isfinite(sampled[:, 0])
    overlapping = overlaps_dropped_events(
        sampled[on_grid, 0], sampled[on_grid, 1], analysed_time.dropped_on_grid(drops)
    )
    reasons[checked[on_grid][overlapping]] = DropReason.OVERLAPS_DROPPED_EVENT.value
    return reasons


def memoryless_inventory(
    run_dir: Path,
    *,
    cohort: str,
    detector: str,
    dot: pd.DataFrame,
    drops: pd.DataFrame,
) -> InventoryTables:
    """One memoryless cohort from the memoryless runner's run folder.

    Parameters
    ----------
    run_dir : Path
        The run folder, ``.../replay_detection_comparison/memoryless/<run id>``.
    cohort : str
        A key of ``schema.MEMORYLESS_COHORTS``, as for ``from_memoryless_table``,
        whose checks (an unshuffled table for ``published``, a table made
        under another SWR source or before the rule) surface here.
    detector : str
        The inventory's name in every returned table, such as ``memoryless``
        for ``published`` and ``memoryless_criteria`` for ``replay_criteria``.
    dot : DataFrame
        The epoch's decode frame, read through the SWR source (for its
        session, bounds and speed).
    drops : DataFrame
        ``analysed_time.read_artifact_event_drops`` of the decode pickle, as for
        :func:`swr_detector_inventory`.

    Returns
    -------
    InventoryTables
        The cohort's candidates with no ``artifact_drop_reason``; those the
        cohort selects that have one are set aside with it. The stored
        excluded-time reasons stand; ``overlaps_dropped_event`` is read again
        on the grid-sampled bounds, as for the LFP inventories; two candidates
        sharing a grid sample make the run read unavailable. Observed: the
        observed file's ``decoded_windows`` stretches. The run records no
        sha256 of its outputs, so the record's ``n_candidates``, ``n_replay``
        and ``n_shuffle_pass`` must describe the table, and its ``time_grid``
        the frame's grid; otherwise the run reads unavailable.

    """
    epoch = _epoch(dot)
    try:
        files, record = _run_files(
            run_dir,
            epoch.epoch_id,
            {
                name: memoryless_output_paths(epoch.epoch_id, run_dir)[name].name
                for name in ("events", "observed")
            },
            hashed=False,
        )
        table = pd.read_parquet(files["events"])
        _check_memoryless_record(record, table, epoch)
    except _Unavailable as why:
        return _unavailable(epoch, detector, run_dir.name, str(why))
    # an unknown cohort is refused by from_memoryless_table
    if cohort in MEMORYLESS_COHORTS and "artifact_drop_reason" in table.columns:
        try:
            reasons = _memoryless_second_pass(
                table, MEMORYLESS_COHORTS[cohort], epoch, drops
            )
        except EventDefinitionError as error:
            return _unavailable(
                epoch,
                detector,
                run_dir.name,
                f"two candidates share a decode-grid sample: {str(error)[:200]}",
            )
        table = table.assign(artifact_drop_reason=reasons)
    named = from_memoryless_table(table, cohort=cohort)
    named["detector"] = detector
    named["event_key"] = detector + named["event_key"].str.removeprefix(MEMORYLESS)
    if "artifact_drop_reason" in table.columns:
        # the candidates the cohort selects that the rule dropped
        rows = table[
            table[MEMORYLESS_COHORTS[cohort]].to_numpy(dtype=bool)
            & table["artifact_drop_reason"].notna().to_numpy()
        ]
        set_aside = _set_aside(
            epoch,
            detector,
            rows["ms_event_id"].to_numpy(),
            rows["event_start_s"].to_numpy(),
            rows["event_end_s"].to_numpy(),
            rows["artifact_drop_reason"].to_numpy(),
        )
    else:
        set_aside = _no_set_aside()
    observed = pd.read_parquet(files["observed"])
    windows = observed.loc[observed["kind"] == "decoded_windows", ["start_s", "end_s"]]
    return InventoryTables(
        inventory=validate_inventory(named),
        observations=_observations(
            epoch, detector, run_dir.name, windows.to_numpy(), "no_decoded_window"
        ),
        set_aside=set_aside,
    )
