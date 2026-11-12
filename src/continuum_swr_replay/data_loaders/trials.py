"""Trial and well visit construction from task events.

This module constructs structured trial and well visit DataFrames from raw task events
(beam breaks, pump activations) and position data. It implements the logic for
segmenting continuous behavior into discrete trials and identifying key behavioral
events like well visits, runs, rewards, and patch changes.

Main Functions
--------------
make_well_visits_df
    Primary entry point that constructs the complete well visits DataFrame from
    beam breaks, pump events, position data, and track graph. Combines poke and
    run intervals into a unified behavioral segmentation.
make_trials_df_from_well_visits
    Aggregates well visits into trials by grouping consecutive well visit events
    and computing trial-level summaries (reward status, timing, etc.).
make_port_experiences_df
    Builds the complementary physical-poke experience stream: the origin poke
    followed by one destination experience for every complete movement trial.

Helper Functions
----------------
get_poke_in_poke_out_times
    Merges consecutive beam break events into poke in/out intervals using a
    maximum inter-event interval threshold.
get_run_intervals
    Extracts movement intervals between consecutive pokes.
get_start_end_node_id
    Determines which wells (graph nodes) the animal is at for each interval
    by finding the nearest node to the projected position.

Data Structure
--------------
Well Visits DataFrame contains:
    - start_time, end_time, duration: Interval timing
    - from_well, to_well: Start and end well node IDs (0-5)
    - from_patch, to_patch: Start and end patch IDs (1-3)
    - is_run: True if movement interval (from_well != to_well)
    - is_reward: True if reward was delivered during this interval
    - reward_on_time: First pump onset in the interval, NaN for no pump
    - is_patch_change: True if movement crossed patch boundaries
    - trial_number: Trial index assignment

Trials DataFrame contains:
    - movement_start_time: final poke-out at the origin port
    - destination_first_poke_in_time: first physical poke-in at the destination
    - destination_first_poke_out_time: end of its first photogate bout
    - end_time: final destination poke-out, after any same-port re-pokes
    - duration: final-exit to final-exit task-trial duration
    - reward_on_time: First pump onset on rewarded trials, else NaN
    - outcome_known_time: reward_on_time for rewards; destination first poke-in
      + OMISSION_KNOWN_LATENCY_SEC, capped at the final same-port poke-out, for
      omissions
    - from_well, to_well: Starting and ending well IDs
    - from_patch, to_patch: Starting and ending patch IDs
    - is_reward: True if trial resulted in reward
    - is_patch_change: True if trial crossed patch boundaries

    Only movements bounded by two recorded physical pokes enter ``trials``.
    Epoch edges are not pokes and therefore cannot create a destination,
    omission, or completed trial.

Port Experiences DataFrame contains:
    - experience_id: Zero-based chronological port-experience index
    - movement_trial_number: Source movement trial, missing for the initial port
    - port, patch: Experienced reward port and its patch
    - arrival_time / first_poke_in_time: first poke-in at the port
    - first_poke_out_time: end of the first physical photogate bout
    - departure_time / final_poke_out_time: final poke-out before a new port
    - n_physical_poke_bouts: number of same-port photogate bouts collapsed
    - reward_on_time, outcome_known_time, is_reward: Outcome and its clock
    - is_initial_experience: True only for the epoch's starting-port experience
    - is_boundary_derived: always False in the canonical physical-poke stream;
      retained as an explicit fail-closed provenance field
    - outcome_observed: always True in the canonical physical-poke stream

Notes
-----
- Touching DIO fragments are merged; a positive low-state gap preserves a
  distinct physical poke bout
- Pump onsets are assigned to exactly one physical poke and then propagated to
  the movement trial that arrives at that poke
- Consecutive same-port physical pokes form one task experience -- the direct
  analog of a per-visit row in Comrie's StateScript parse. A movement trial is
  our derived pairing of consecutive experiences; it begins only at the final
  same-port exit
- Epoch boundaries are excluded; only recorded physical poke intervals define visits
- Physical poke identity comes from the Poke1--Poke6 DIO channel. Position and
  graph geometry remain available for independent quality control but do not
  define the canonical trial endpoints.

See Also
--------
continuum_swr_replay.data_loaders.events : DIO event loading
continuum_swr_replay.data_loaders.constants : WELL_PATCH_MAPPING
continuum_swr_replay.data_types : WellNodeData TypedDict

Examples
--------
>>> well_visits = make_well_visits_df(beam_breaks, pump_events)
>>> trials = make_trials_df_from_well_visits(well_visits)
>>> trials[["from_well", "to_well", "is_reward"]].head()
>>> experiences = make_port_experiences_df(well_visits, trials)
>>> experiences[["port", "is_reward", "is_initial_experience"]].head()

"""

import logging

import networkx as nx
import numpy as np
import pandas as pd

from ..data_types import WellNodeData
from .constants import POKE_EVENT_TO_WELL, PUMP_EVENT_TO_WELL, WELL_PATCH_MAPPING
from .trial_clocks import outcome_known_time_from_task_visit

logger = logging.getLogger(__name__)

PUMP_AUDIT_ATTRS = (
    "gap_recovered_pump_count",
    "unmatched_pump_event_count",
    "unmatched_pump_events",
    "unmatched_pump_event_times",
)
"""Required session-level pump-assignment audit fields on port experiences."""


def _nonnegative_audit_count(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return count


def validate_port_experience_pump_audit(port_experiences: pd.DataFrame) -> None:
    """Validate the authoritative post-recovery pump-assignment audit.

    These attributes describe session-level DIO events, so they remain defined
    even when the scientific port-experience table has zero rows.  A cache or
    downstream table lacking them predates whole-visit gap recovery and must
    not be interpreted as current input.
    """
    missing = [name for name in PUMP_AUDIT_ATTRS if name not in port_experiences.attrs]
    if missing:
        raise KeyError(f"port experiences are missing pump-audit attrs: {missing}")

    _nonnegative_audit_count(
        port_experiences.attrs["gap_recovered_pump_count"],
        name="gap_recovered_pump_count",
    )
    unmatched_count = _nonnegative_audit_count(
        port_experiences.attrs["unmatched_pump_event_count"],
        name="unmatched_pump_event_count",
    )
    events = port_experiences.attrs["unmatched_pump_events"]
    times = port_experiences.attrs["unmatched_pump_event_times"]
    if not isinstance(events, tuple) or not isinstance(times, tuple):
        raise ValueError("unmatched pump events and times must be immutable tuples")
    if unmatched_count != len(events) or unmatched_count != len(times):
        raise ValueError(
            "unmatched_pump_event_count must equal the event and time tuple lengths"
        )

    event_times: list[float] = []
    for event in events:
        if not isinstance(event, tuple) or len(event) != 2:
            raise ValueError("each unmatched pump event must be a (channel, time) pair")
        channel, event_time = event
        if str(channel) not in PUMP_EVENT_TO_WELL:
            raise ValueError(f"unmatched pump event has invalid channel: {channel!r}")
        try:
            numeric_time = float(event_time)
        except (TypeError, ValueError) as exc:
            raise ValueError("unmatched pump event times must be numeric") from exc
        if not np.isfinite(numeric_time):
            raise ValueError("unmatched pump event times must be finite")
        event_times.append(numeric_time)
    try:
        numeric_times = tuple(float(value) for value in times)
    except (TypeError, ValueError) as exc:
        raise ValueError("unmatched pump event times must be numeric") from exc
    if not np.isfinite(numeric_times).all() or tuple(event_times) != numeric_times:
        raise ValueError(
            "unmatched_pump_event_times must exactly match unmatched_pump_events"
        )


def _stamp_post_recovery_pump_audit(
    frame: pd.DataFrame,
    *,
    unmatched: list[tuple[str, float]],
    recovered_indices: set[int],
) -> None:
    """Stamp authoritative pump audit attrs after whole-visit recovery."""
    orphan_pumps = tuple(
        pump for index, pump in enumerate(unmatched) if index not in recovered_indices
    )
    frame.attrs["gap_recovered_pump_count"] = len(recovered_indices)
    frame.attrs["unmatched_pump_event_count"] = len(orphan_pumps)
    frame.attrs["unmatched_pump_events"] = orphan_pumps
    frame.attrs["unmatched_pump_event_times"] = tuple(
        float(time) for _, time in orphan_pumps
    )
    validate_port_experience_pump_audit(frame)


def get_poke_in_poke_out_times(
    beam_breaks: pd.DataFrame, max_inter_event_interval: float = 0.0
) -> pd.DataFrame:
    """Merge beam break events into poke in/out intervals.

    Parameters
    ----------
    beam_breaks : pd.DataFrame
        Beam break events with start_time and end_time
    max_inter_event_interval : float, optional
        Maximum gap between fragments to merge. The canonical default is zero:
        only touching fragments from one sustained DIO-high interval merge;
        any observed low-state gap preserves a distinct physical re-poke.

    Returns
    -------
    pd.DataFrame
        Merged poke events with start_time, end_time, and duration

    """
    if not np.isfinite(max_inter_event_interval) or max_inter_event_interval < 0.0:
        raise ValueError("max_inter_event_interval must be finite and nonnegative")
    # Handle empty DataFrame
    if len(beam_breaks) == 0:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration", "event_name"]
        )

    merged_events = []
    for event_name, df in beam_breaks.groupby("event_name"):
        # ``shift(-1)`` below assumes the rows are in chronological
        # order. Spyglass's ``load_dios`` returns sorted events, but
        # this helper is also called directly on caller-constructed
        # frames and the inter-event-interval / label cumsum would
        # silently produce negative-duration pokes if a row landed
        # out of order.
        df = df.sort_values("start_time").reset_index(drop=True)
        interevent_interval = (df.start_time.shift(-1) - df.end_time).fillna(0)
        labels = (
            (interevent_interval > max_inter_event_interval)
            .shift(1, fill_value=False)
            .cumsum()
        )
        new_events = df.groupby(labels).agg({"start_time": "first", "end_time": "last"})
        new_events["event_name"] = event_name
        merged_events.append(new_events)

    return (
        pd.concat(merged_events)
        .sort_values("start_time")
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )


def get_run_intervals(poke_times: pd.DataFrame) -> pd.DataFrame:
    """Extract run intervals between pokes.

    Parameters
    ----------
    poke_times : pd.DataFrame
        Poke events with start_time and end_time

    Returns
    -------
    pd.DataFrame
        Run intervals with start_time, end_time, and duration

    """
    runs = (
        pd.DataFrame(
            {
                "start_time": poke_times["end_time"].to_numpy()[:-1],
                "end_time": poke_times["start_time"].shift(-1).to_numpy()[:-1],
            }
        )
        .dropna()
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )
    if (runs["duration"] < 0.0).any():
        raise ValueError(
            "physical poke intervals overlap in time; cannot construct a "
            "poke-to-poke movement interval"
        )
    return runs


def _poke_wells_from_event_names(event_names: pd.Series) -> np.ndarray:
    """Translate authoritative physical poke channels to reward-well IDs."""
    names = event_names.astype(str)
    wells = names.map(POKE_EVENT_TO_WELL)
    if wells.isna().any():
        invalid = tuple(sorted(names.loc[wells.isna()].unique())[:10])
        raise ValueError(
            "beam-break event_name must identify a canonical physical poke "
            f"channel {tuple(POKE_EVENT_TO_WELL)}; invalid={invalid}"
        )
    return wells.to_numpy(dtype=np.int64)


def _event_names(frame: pd.DataFrame, *, kind: str) -> pd.Series:
    """Return DIO channel names from either a column or named index level."""
    if "event_name" in frame.columns:
        return frame["event_name"].astype(str).reset_index(drop=True)
    if "event_name" in frame.index.names:
        return pd.Series(
            frame.index.get_level_values("event_name").astype(str),
            dtype=str,
        )
    if frame.empty:
        return pd.Series(dtype=str)
    raise KeyError(
        f"nonempty {kind} events require an event_name column or index level"
    )


def make_physical_poke_intervals(
    beam_breaks: pd.DataFrame,
    pump_events: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct physical photogate bouts and attach validated pump onsets.

    This is the acquisition-grain table.  Rows are physical beam-break bouts,
    including repeated pokes at the same port.  They are not movement trials
    and are not collapsed task experiences.  A pump is credited only when its
    onset lies in exactly one bout and its hardware channel identifies the same
    reward port.
    """
    pokes = get_poke_in_poke_out_times(beam_breaks).sort_values(
        "start_time", kind="stable"
    )
    pokes = pokes.reset_index(drop=True)
    pokes["physical_poke_id"] = np.arange(len(pokes), dtype=np.int64)
    if pokes.empty:
        pokes["port"] = pd.Series(dtype="int64")
        pokes["is_reward"] = pd.Series(dtype=bool)
        pokes["reward_on_time"] = pd.Series(dtype=float)
    else:
        pokes["port"] = _poke_wells_from_event_names(pokes["event_name"])
        pokes["is_reward"] = False
        pokes["reward_on_time"] = np.nan
    poke_starts = pokes["start_time"].to_numpy(dtype=float)
    poke_ends = pokes["end_time"].to_numpy(dtype=float)
    if (
        not np.isfinite(poke_starts).all()
        or not np.isfinite(poke_ends).all()
        or (poke_ends < poke_starts).any()
        or (len(pokes) > 1 and (poke_starts[1:] < poke_ends[:-1]).any())
    ):
        raise ValueError(
            "physical poke intervals must be finite, ordered, and disjoint"
        )

    pump_names = _event_names(pump_events, kind="pump")
    if "start_time" not in pump_events.columns:
        if not pump_events.empty:
            raise KeyError("pump events require start_time")
        pump_times = np.empty(0, dtype=float)
    else:
        pump_times = pd.to_numeric(pump_events["start_time"], errors="coerce").to_numpy(
            dtype=float
        )
    if not np.isfinite(pump_times).all():
        raise ValueError("pump event start_time values must be finite")
    if len(pump_names) != len(pump_times):
        raise ValueError("pump channel names do not align with pump onset times")
    pump_wells = pump_names.map(PUMP_EVENT_TO_WELL)
    if pump_wells.isna().any():
        invalid = tuple(sorted(pump_names.loc[pump_wells.isna()].unique())[:10])
        raise ValueError(
            "pump event_name must identify a canonical reward-pump channel "
            f"{tuple(PUMP_EVENT_TO_WELL)}; invalid={invalid}"
        )

    poke_ports = pokes["port"].to_numpy(dtype=np.int64)
    unmatched: list[tuple[str, float]] = []
    order = np.argsort(pump_times, kind="stable")
    for pump_index in order:
        pump_time = float(pump_times[pump_index])
        pump_name = str(pump_names.iloc[pump_index])
        pump_well = int(pump_wells.iloc[pump_index])
        matches = (poke_starts <= pump_time) & (pump_time <= poke_ends)
        n_matches = int(matches.sum())
        if n_matches == 0:
            unmatched.append((pump_name, pump_time))
            continue
        if n_matches > 1:
            raise ValueError(
                "a reward-pump onset overlaps multiple physical poke intervals; "
                f"pump_time={pump_time}, matches={n_matches}"
            )
        poke_index = int(np.flatnonzero(matches)[0])
        if poke_ports[poke_index] != pump_well:
            raise ValueError(
                "pump channel disagrees with the containing physical poke; "
                f"pump={pump_name}, poke={pokes.loc[poke_index, 'event_name']}, "
                f"pump_time={pump_time}"
            )
        if np.isfinite(float(pokes.loc[poke_index, "reward_on_time"])):
            raise ValueError(
                "one physical poke contains multiple reward-pump onsets; "
                f"physical_poke_id={pokes.loc[poke_index, 'physical_poke_id']}"
            )
        pokes.loc[poke_index, "is_reward"] = True
        pokes.loc[poke_index, "reward_on_time"] = pump_time

    # Per-bout stage: a pump that matches no single beam-on interval is
    # recorded here but not yet credited. Some of these fall in a beam-release
    # gap between two bouts of the SAME visit -- inside Comrie's whole-visit
    # reward window -- and are recovered when pokes are collapsed to port
    # experiences. The loud warning fires there, on the post-recovery count of
    # genuine (transit) orphans.
    pokes.attrs["unmatched_pump_event_count"] = len(unmatched)
    pokes.attrs["unmatched_pump_events"] = tuple(unmatched)
    pokes.attrs["unmatched_pump_event_times"] = tuple(time for _, time in unmatched)
    return pokes


def _collapse_physical_pokes_to_port_experiences(
    physical_pokes: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse consecutive same-port re-pokes using Comrie's task rule."""
    columns = [
        "experience_id",
        "movement_trial_number",
        "is_initial_experience",
        "is_boundary_derived",
        "outcome_observed",
        "port",
        "patch",
        "arrival_time",
        "departure_time",
        "first_poke_in_time",
        "first_poke_out_time",
        "final_poke_out_time",
        "n_physical_poke_bouts",
        "reward_on_time",
        "outcome_known_time",
        "is_reward",
    ]
    if physical_pokes.empty:
        experiences = pd.DataFrame(columns=columns)
        experiences.attrs.update(physical_pokes.attrs)
        unmatched = list(physical_pokes.attrs.get("unmatched_pump_events", ()))
        _stamp_post_recovery_pump_audit(
            experiences,
            unmatched=unmatched,
            recovered_indices=set(),
        )
        return experiences
    required = {
        "physical_poke_id",
        "port",
        "start_time",
        "end_time",
        "is_reward",
        "reward_on_time",
    }
    missing = sorted(required - set(physical_pokes.columns))
    if missing:
        raise KeyError(f"physical poke table is missing columns: {missing}")
    pokes = physical_pokes.sort_values("start_time", kind="stable").reset_index(
        drop=True
    )
    starts = pokes["start_time"].to_numpy(dtype=float)
    ends = pokes["end_time"].to_numpy(dtype=float)
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or (ends < starts).any()
        or (len(pokes) > 1 and (starts[1:] < ends[:-1]).any())
    ):
        raise ValueError(
            "physical poke intervals must be finite, ordered, and disjoint"
        )
    ports = pokes["port"].to_numpy(dtype=np.int64)
    experience_ids = np.r_[0, np.cumsum(ports[1:] != ports[:-1])].astype(np.int64)
    pokes["experience_id"] = experience_ids

    # Pump onsets that fell in a beam-release gap between a visit's bouts are
    # unmatched at the per-bout stage but still lie inside Comrie's whole-visit
    # reward window [first_poke_in, final_poke_out]. Recover the same-well ones
    # so a collapsed visit is credited exactly as her parser does -- a real
    # case (e.g. wilbur, where a brief beam release can put the pump on a
    # return poke). Genuine transit orphans (different well, or outside every
    # visit) stay unmatched, which her parser also drops.
    #
    # This recovery does NOT license a long reward latency. The binding rule is
    # OMISSION_KNOWN_LATENCY_SEC = 0.30 s, and
    # ``outcome_known_times_from_task_visits`` raises if any recovered pump
    # lands later than that after its visit's first poke-in. Measured over the
    # 218-session corpus (2026-09-03), latency from arrival is p99 91 ms with a
    # maximum of 236 ms, so the guard has headroom over every observed pump.
    unmatched = list(physical_pokes.attrs.get("unmatched_pump_events", ()))
    unmatched_well = [PUMP_EVENT_TO_WELL.get(str(name)) for name, _ in unmatched]
    recovered_pumps: set[int] = set()

    records: list[dict[str, object]] = []
    for experience_id, group in pokes.groupby("experience_id", sort=True):
        arrival = float(group["start_time"].iloc[0])
        departure = float(group["end_time"].iloc[-1])
        first_poke_out = float(group["end_time"].iloc[0])
        port = int(group["port"].iloc[0])
        # A collapsed same-port visit is rewarded if a pump fell inside ANY of
        # its physical pokes -- and, via the gap recovery below, anywhere in the
        # whole visit window -- not only the first bout. ``reward_on`` is that
        # pump's onset (the actual delivery instant, which gates replay access
        # to the outcome). The ``size > 1`` guard still rejects the genuine
        # anomaly of two distinct pumps inside one visit.
        bout_reward_on = pd.to_numeric(
            group["reward_on_time"], errors="coerce"
        ).to_numpy(dtype=float)
        reward_times = list(bout_reward_on[np.isfinite(bout_reward_on)])
        for pump_index, (_, pump_time) in enumerate(unmatched):
            if (
                pump_index not in recovered_pumps
                and unmatched_well[pump_index] == port
                and arrival <= float(pump_time) <= departure
            ):
                reward_times.append(float(pump_time))
                recovered_pumps.add(pump_index)
        reward_array = np.sort(np.asarray(reward_times, dtype=float))
        if reward_array.size > 1:
            raise ValueError(
                "one task port experience contains multiple rewarded pokes; "
                f"experience_id={experience_id}"
            )
        reward_on = float(reward_array[0]) if reward_array.size else np.nan
        is_reward = bool(reward_array.size)
        outcome_observed = is_reward or bool(
            (
                group["end_time"].to_numpy(dtype=float)
                > group["start_time"].to_numpy(dtype=float)
            ).any()
        )
        outcome_known = (
            outcome_known_time_from_task_visit(
                rewarded=is_reward,
                first_poke_in_time=arrival,
                final_poke_out_time=departure,
                reward_on_time=reward_on,
            )
            if outcome_observed
            else np.nan
        )
        records.append(
            {
                "experience_id": int(experience_id),
                "movement_trial_number": (
                    pd.NA if experience_id == 0 else int(experience_id)
                ),
                "is_initial_experience": experience_id == 0,
                "is_boundary_derived": False,
                "outcome_observed": outcome_observed,
                "port": port,
                "patch": WELL_PATCH_MAPPING[port],
                "arrival_time": arrival,
                "departure_time": departure,
                "first_poke_in_time": arrival,
                "first_poke_out_time": first_poke_out,
                "final_poke_out_time": departure,
                "n_physical_poke_bouts": int(len(group)),
                "reward_on_time": reward_on,
                "outcome_known_time": outcome_known,
                "is_reward": is_reward,
            }
        )
    experiences = pd.DataFrame.from_records(records, columns=columns)
    observed = experiences["outcome_observed"].to_numpy(dtype=bool)
    if observed.any():
        first_observed = int(np.flatnonzero(observed)[0])
        last_observed = int(np.flatnonzero(observed)[-1])
        if not observed[first_observed : last_observed + 1].all():
            raise ValueError(
                "an unobserved photogate experience occurs inside the task stream"
            )
        experiences.attrs["leading_censored_physical_experiences"] = first_observed
        experiences.attrs["trailing_censored_physical_experiences"] = (
            len(experiences) - last_observed - 1
        )
        experiences = experiences.iloc[first_observed : last_observed + 1].reset_index(
            drop=True
        )
        experiences["experience_id"] = np.arange(len(experiences), dtype=np.int64)
        experiences["is_initial_experience"] = experiences["experience_id"].eq(0)
        experiences["movement_trial_number"] = pd.array(
            [pd.NA, *range(1, len(experiences))], dtype="Int64"
        )
    else:
        experiences = experiences.iloc[0:0].copy()
    experiences["movement_trial_number"] = pd.array(
        experiences["movement_trial_number"], dtype="Int64"
    )
    # Preserve acquisition diagnostics first, then overwrite the pump fields
    # with the authoritative post-recovery partition. The inverse order would
    # make a recovered gap pump appear both recovered and unmatched.
    experiences.attrs.update(physical_pokes.attrs)
    _stamp_post_recovery_pump_audit(
        experiences,
        unmatched=unmatched,
        recovered_indices=recovered_pumps,
    )
    orphan_pumps = list(experiences.attrs["unmatched_pump_events"])
    if orphan_pumps:
        # A pump credited to no reward-port visit cannot enter reward history --
        # the same outcome as Comrie's parser. Surface it loudly rather than
        # leaving it in attrs no caller reads: a cluster in one session points
        # at a missed beam break or spurious pulse that would corrupt history.
        logger.warning(
            "%d reward-pump onset(s) fell outside every physical poke visit and "
            "were dropped from reward history (%d recovered from same-well gaps; "
            "orphan channels/times: %s)",
            len(orphan_pumps),
            len(recovered_pumps),
            tuple((name, round(time, 3)) for name, time in orphan_pumps[:10]),
        )
    return experiences


def get_start_end_node_id(
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
    times: pd.DataFrame,
    well_nodes: list[int] | None = None,
) -> WellNodeData:
    """Determine start and end well nodes for each interval.

    Finds the nearest well node to the animal's projected position at
    the start and end of each interval. Only considers actual well nodes
    (not internal junction nodes) to prevent misassignment.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with projected position columns
    track_graph : nx.Graph
        Track graph with node positions
    times : pd.DataFrame
        Times with start_time and end_time columns
    well_nodes : list[int] | None
        Node IDs to consider as wells. If None, uses the well nodes of
        WELL_PATCH_MAPPING, which must all be in the graph.

    Returns
    -------
    WellNodeData
        TypedDict with 'from_well' and 'to_well' node ID arrays

    """
    projected_cols = ["projected_x_position", "projected_y_position"]

    # We want the position row at-or-before each interval boundary.
    # ``side="right"`` returns the *insertion index after* an exact match,
    # so ``searchsorted([0, 1], 0, side="right") == 1`` — without the
    # ``- 1`` correction we would sample the *next* row, mislabeling
    # ``from_well``/``to_well`` (e.g. for an interval [0, 1] over a
    # position frame at t=0 (well 0) and t=1 (well 1), both endpoints
    # would resolve to row 1 → from_well = to_well = 1). Subtract 1
    # then clamp to the valid index range.
    max_idx = len(position_info) - 1
    start_indices = np.clip(
        position_info.index.searchsorted(times.start_time.values, side="right") - 1,
        0,
        max_idx,
    )
    end_indices = np.clip(
        position_info.index.searchsorted(times.end_time.values, side="right") - 1,
        0,
        max_idx,
    )

    start_time_pos = position_info.iloc[start_indices][projected_cols].to_numpy()
    end_time_pos = position_info.iloc[end_indices][projected_cols].to_numpy()

    node_attrs = nx.get_node_attributes(track_graph, "pos")

    candidate_ids = sorted(WELL_PATCH_MAPPING if well_nodes is None else well_nodes)

    candidate_pos = np.array([node_attrs[n] for n in candidate_ids])
    candidate_arr = np.array(candidate_ids)

    start_node_ids = candidate_arr[
        np.argmin(
            np.linalg.norm(
                start_time_pos[:, None, :] - candidate_pos[None, :, :], axis=2
            ),
            axis=1,
        )
    ]
    end_node_ids = candidate_arr[
        np.argmin(
            np.linalg.norm(
                end_time_pos[:, None, :] - candidate_pos[None, :, :], axis=2
            ),
            axis=1,
        )
    ]

    return {
        "from_well": start_node_ids,
        "to_well": end_node_ids,
    }


def make_well_visits_df(
    beam_breaks: pd.DataFrame,
    pump_events: pd.DataFrame,
) -> pd.DataFrame:
    """Construct well visits DataFrame from beam breaks and pump events.

    Parameters
    ----------
    beam_breaks : pd.DataFrame
        Beam break events
    pump_events : pd.DataFrame
        Reward pump events

    Returns
    -------
    pd.DataFrame
        Well visits with trial structure and reward information

    """
    # Scientific visits are defined by recorded beam breaks. Earlier versions
    # inserted zero-duration epoch-edge rows, mapped them to the nearest well,
    # and then allowed those synthetic rows to create plausible movement
    # trials. An epoch boundary is neither a poke nor evidence of occupancy, so
    # it must not enter the scientific interval stream.
    poke_times = make_physical_poke_intervals(beam_breaks, pump_events)
    poke_times["is_observed_poke"] = True
    poke_times = poke_times.sort_values("start_time").reset_index(drop=True)
    if poke_times.empty:
        empty = pd.DataFrame(
            columns=(
                "from_well",
                "to_well",
                "start_time",
                "end_time",
                "duration",
                "is_observed_poke",
                "is_run",
                "is_reward",
                "reward_on_time",
                "from_patch",
                "to_patch",
                "is_patch_change",
                "trial_number",
            )
        )
        empty.attrs.update(poke_times.attrs)
        return empty
    # The DIO channel is the authoritative port identity. Position is deliberately
    # not used here: during a long beam break the tracked point can cross onto an
    # adjoining segment, but that does not turn a physical poke into a run.
    poke_wells = poke_times["port"].to_numpy(dtype=np.int64)
    poke_experience_ids = np.r_[0, np.cumsum(poke_wells[1:] != poke_wells[:-1])].astype(
        np.int64
    )
    poke_times["port_experience_id"] = poke_experience_ids
    poke_times["from_well"] = poke_wells
    poke_times["to_well"] = poke_wells
    poke_times = poke_times.set_index(["from_well", "to_well"])

    run_times = get_run_intervals(poke_times)
    run_times["is_observed_poke"] = False
    # A movement interval connects consecutive physical pokes. Its endpoints
    # therefore come from those DIO channels, not from noisy tracked-position
    # samples at the interval boundaries.
    run_times["from_well"] = poke_wells[:-1]
    run_times["to_well"] = poke_wells[1:]
    run_times["port_experience_id"] = poke_experience_ids[:-1]
    run_times = run_times.set_index(["from_well", "to_well"])

    well_visits = (
        pd.concat((poke_times, run_times)).sort_values("start_time").reset_index()
    )
    well_visits["is_run"] = ~well_visits["is_observed_poke"].astype(bool) & well_visits[
        "from_well"
    ].ne(well_visits["to_well"])

    well_visits["is_reward"] = well_visits["is_reward"].eq(True)
    well_visits["is_reward"] = (
        well_visits["is_run"] & well_visits["is_reward"].shift(-1)
    ) | well_visits["is_reward"]
    well_visits.attrs.update(poke_times.attrs)
    well_visits["from_patch"] = well_visits["from_well"].map(WELL_PATCH_MAPPING)
    well_visits["to_patch"] = well_visits["to_well"].map(WELL_PATCH_MAPPING)
    well_visits["is_patch_change"] = (
        well_visits["from_patch"] != well_visits["to_patch"]
    )

    trial_number = np.full((len(well_visits),), np.nan)
    n_runs = int(well_visits.is_run.sum())
    if n_runs:
        trial_number[well_visits.is_run] = np.arange(1, n_runs + 1)
        trial_number = (
            pd.DataFrame(trial_number).ffill().bfill().astype(int).to_numpy().squeeze()
        )
        well_visits["trial_number"] = trial_number
    else:
        well_visits["trial_number"] = pd.array(
            [pd.NA] * len(well_visits), dtype="Int64"
        )

    return well_visits


def make_trials_df_from_well_visits(well_visits: pd.DataFrame) -> pd.DataFrame:
    """Create movement trials from physical poke intervals.

    Consecutive physical pokes at the same port are one task experience.  A
    movement trial starts at the *final* poke-out of one experience and ends at
    the final poke-out of the next distinct-port experience.  The destination
    becomes known at that experience's first poke-in.  The per-visit boundaries
    (first poke-in / final poke-out) match Comrie's StateScript parser, whose
    canonical unit is the per-visit row -- our ``port_experiences`` stream is
    that direct analog. The final-exit-to-final-exit movement trial is our
    derived pairing of consecutive experiences, used by the manuscript.

    Parameters
    ----------
    well_visits : pd.DataFrame
        Well visits data

    Returns
    -------
    pd.DataFrame
        Trials with aggregated information (reward, timing, start/end wells)

    """
    required = {
        "start_time",
        "end_time",
        "from_well",
        "is_observed_poke",
        "is_reward",
        "reward_on_time",
    }
    missing = sorted(required - set(well_visits.columns))
    if missing:
        raise KeyError(f"well_visits is missing physical-poke columns: {missing}")
    physical = well_visits.loc[well_visits["is_observed_poke"].astype(bool)].copy()
    physical = physical.sort_values("start_time", kind="stable").reset_index(drop=True)
    if "physical_poke_id" not in physical:
        physical["physical_poke_id"] = np.arange(len(physical), dtype=np.int64)
    physical["port"] = pd.to_numeric(physical["from_well"], errors="raise").astype(
        np.int64
    )
    physical.attrs.update(well_visits.attrs)
    experiences = _collapse_physical_pokes_to_port_experiences(physical)

    columns = [
        "movement_start_time",
        "destination_first_poke_in_time",
        "destination_first_poke_out_time",
        "end_time",
        "duration",
        "from_well",
        "to_well",
        "from_patch",
        "to_patch",
        "is_patch_change",
        "reward_on_time",
        "outcome_known_time",
        "is_reward",
    ]
    if len(experiences) < 2:
        trials = pd.DataFrame(columns=columns)
        trials.index = pd.Index([], dtype="int64", name="trial_number")
    else:
        origin = experiences.iloc[:-1].reset_index(drop=True)
        destination = experiences.iloc[1:].reset_index(drop=True)
        movement_start = origin["final_poke_out_time"].to_numpy(dtype=float)
        arrival = destination["first_poke_in_time"].to_numpy(dtype=float)
        trial_end = destination["final_poke_out_time"].to_numpy(dtype=float)
        if (arrival < movement_start).any() or (trial_end < arrival).any():
            raise ValueError("movement-trial clocks are not chronologically ordered")
        trials = pd.DataFrame(
            {
                "movement_start_time": movement_start,
                "destination_first_poke_in_time": arrival,
                "destination_first_poke_out_time": destination[
                    "first_poke_out_time"
                ].to_numpy(dtype=float),
                "end_time": trial_end,
                "duration": trial_end - movement_start,
                "from_well": origin["port"].to_numpy(dtype=np.int64),
                "to_well": destination["port"].to_numpy(dtype=np.int64),
                "from_patch": origin["patch"].to_numpy(dtype=np.int64),
                "to_patch": destination["patch"].to_numpy(dtype=np.int64),
                "reward_on_time": destination["reward_on_time"].to_numpy(dtype=float),
                "outcome_known_time": destination["outcome_known_time"].to_numpy(
                    dtype=float
                ),
                "is_reward": destination["is_reward"].to_numpy(dtype=bool),
            },
            index=pd.Index(
                np.arange(1, len(experiences), dtype=np.int64),
                name="trial_number",
            ),
        )
        trials["is_patch_change"] = trials["from_patch"].ne(trials["to_patch"])
        trials = trials[columns]
    # Pump-assignment counts are session-level provenance rather than trial-row
    # variables. The authoritative post-recovery audit therefore lives on the
    # cached ``port_experiences.attrs`` rather than being duplicated here.
    return trials


def make_port_experiences_df(
    well_visits: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Return the task-experience stream derived from physical photogate bouts.

    Consecutive same-port re-pokes are collapsed to one experience, matching
    Comrie's StateScript parser. Arrival time is the first poke-in and
    departure time is the final poke-out. The first row is the origin of
    the first movement; every later row is one movement-trial destination.
    """
    columns = [
        "experience_id",
        "movement_trial_number",
        "is_initial_experience",
        "is_boundary_derived",
        "outcome_observed",
        "port",
        "patch",
        "arrival_time",
        "departure_time",
        "first_poke_in_time",
        "first_poke_out_time",
        "final_poke_out_time",
        "n_physical_poke_bouts",
        "reward_on_time",
        "outcome_known_time",
        "is_reward",
    ]
    if (
        "is_observed_poke" in well_visits
        and not well_visits["is_observed_poke"].astype(bool).any()
        and not trials.empty
    ):
        raise ValueError("first movement has no preceding origin-port occupancy")
    required_visits = {
        "start_time",
        "end_time",
        "from_well",
        "is_observed_poke",
        "is_reward",
        "reward_on_time",
    }
    missing_visits = sorted(required_visits - set(well_visits.columns))
    if missing_visits:
        raise KeyError(
            f"port experiences require physical-poke columns; missing={missing_visits}"
        )
    physical = well_visits.loc[well_visits["is_observed_poke"].astype(bool)].copy()
    physical = physical.sort_values("start_time", kind="stable").reset_index(drop=True)
    if "physical_poke_id" not in physical:
        physical["physical_poke_id"] = np.arange(len(physical), dtype=np.int64)
    physical["port"] = pd.to_numeric(physical["from_well"], errors="raise").astype(
        np.int64
    )
    physical.attrs.update(well_visits.attrs)
    experiences = _collapse_physical_pokes_to_port_experiences(physical)

    if trials.empty:
        if len(experiences) > 1:
            raise ValueError(
                "physical pokes contain distinct task experiences but the movement "
                "trial table is empty"
            )
        empty = pd.DataFrame(columns=columns)
        empty.attrs.update(experiences.attrs)
        return empty

    required_trials = {
        "movement_start_time",
        "destination_first_poke_in_time",
        "destination_first_poke_out_time",
        "end_time",
        "from_well",
        "to_well",
        "from_patch",
        "to_patch",
        "is_patch_change",
        "reward_on_time",
        "outcome_known_time",
        "is_reward",
    }
    missing_trials = sorted(required_trials - set(trials.columns))
    if missing_trials:
        raise KeyError(
            "port experiences require canonical movement-trial columns; "
            f"missing={missing_trials}"
        )
    ordered_trials = trials.sort_values("destination_first_poke_in_time", kind="stable")
    if len(experiences) != len(ordered_trials) + 1:
        raise ValueError(
            "physical port experiences must equal movement trials plus one; "
            f"experiences={len(experiences)}, trials={len(ordered_trials)}"
        )

    origin_ports = experiences["port"].to_numpy(dtype=np.int64)[:-1]
    destination_ports = experiences["port"].to_numpy(dtype=np.int64)[1:]
    if not np.array_equal(
        origin_ports, ordered_trials["from_well"].to_numpy(dtype=np.int64)
    ) or not np.array_equal(
        destination_ports, ordered_trials["to_well"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(
            "movement trials do not match the consecutive physical port experiences"
        )
    if not np.array_equal(
        experiences["patch"].to_numpy(dtype=np.int64)[:-1],
        ordered_trials["from_patch"].to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        experiences["patch"].to_numpy(dtype=np.int64)[1:],
        ordered_trials["to_patch"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("movement-trial patches disagree with port experiences")

    exact_clock_pairs = (
        (
            experiences["departure_time"].to_numpy(dtype=float)[:-1],
            ordered_trials["movement_start_time"].to_numpy(dtype=float),
            "origin final poke-out",
        ),
        (
            experiences["arrival_time"].to_numpy(dtype=float)[1:],
            ordered_trials["destination_first_poke_in_time"].to_numpy(dtype=float),
            "destination first poke-in",
        ),
        (
            experiences["first_poke_out_time"].to_numpy(dtype=float)[1:],
            ordered_trials["destination_first_poke_out_time"].to_numpy(dtype=float),
            "destination first poke-out",
        ),
        (
            experiences["departure_time"].to_numpy(dtype=float)[1:],
            ordered_trials["end_time"].to_numpy(dtype=float),
            "destination final poke-out",
        ),
        (
            experiences["outcome_known_time"].to_numpy(dtype=float)[1:],
            ordered_trials["outcome_known_time"].to_numpy(dtype=float),
            "outcome availability",
        ),
    )
    for observed, expected, label in exact_clock_pairs:
        if not np.allclose(observed, expected, rtol=0.0, atol=0.0):
            raise ValueError(f"{label} does not align between experiences and trials")
    if not np.allclose(
        experiences["reward_on_time"].to_numpy(dtype=float)[1:],
        ordered_trials["reward_on_time"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError("reward onset does not align between experiences and trials")
    if not np.array_equal(
        experiences["is_reward"].to_numpy(dtype=bool)[1:],
        ordered_trials["is_reward"].to_numpy(dtype=bool),
    ):
        raise ValueError("reward labels do not align between experiences and trials")
    expected_switch = (
        ordered_trials["from_patch"].ne(ordered_trials["to_patch"]).to_numpy(dtype=bool)
    )
    if not np.array_equal(
        expected_switch,
        ordered_trials["is_patch_change"].to_numpy(dtype=bool),
    ):
        raise ValueError("movement-trial patch-switch labels are inconsistent")

    experiences = experiences.copy()
    experiences.loc[1:, "movement_trial_number"] = pd.array(
        ordered_trials.index.to_numpy(), dtype="Int64"
    )
    experiences["movement_trial_number"] = pd.array(
        experiences["movement_trial_number"], dtype="Int64"
    )
    result = experiences[columns].copy()
    result.attrs.update(experiences.attrs)
    validate_port_experience_pump_audit(result)
    return result


def photogate_engagement_at_times(
    times: np.ndarray,
    physical_pokes_or_visits: pd.DataFrame,
) -> pd.DataFrame:
    """Sample literal photogate engagement without inferring task occupancy.

    Task port context spans first poke-in through final poke-out across same-port
    re-pokes. This helper answers the narrower acquisition question: whether a
    beam was broken at each exact time. Intervals are treated as half-open
    ``[start_time, end_time)`` so touching bouts cannot both own one sample.
    """
    query = np.asarray(times, dtype=float)
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("photogate query times must be a finite one-dimensional array")
    frame = physical_pokes_or_visits
    if "is_observed_poke" in frame:
        frame = frame.loc[frame["is_observed_poke"].astype(bool)]
    port_column = "port" if "port" in frame else "from_well"
    required = {"start_time", "end_time", port_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"physical poke intervals are missing columns: {missing}")
    pokes = frame.sort_values("start_time", kind="stable")
    starts = pd.to_numeric(pokes["start_time"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(pokes["end_time"], errors="coerce").to_numpy(dtype=float)
    ports = pd.to_numeric(pokes[port_column], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or not np.isfinite(ports).all()
        or (ends < starts).any()
        or (len(pokes) > 1 and (starts[1:] < ends[:-1]).any())
    ):
        raise ValueError(
            "physical poke intervals must be finite, ordered, and disjoint"
        )

    engaged = np.zeros(len(query), dtype=bool)
    sampled_port = np.full(len(query), np.nan)
    for start, end, port in zip(starts, ends, ports, strict=True):
        mask = (query >= start) & (query < end)
        if (engaged & mask).any():
            raise RuntimeError("one time sample belongs to multiple photogate bouts")
        engaged |= mask
        sampled_port[mask] = port
    return pd.DataFrame(
        {
            "is_photogate_engaged": engaged,
            "photogate_port": sampled_port,
        }
    )


def align_port_experiences_to_task_records(
    port_experiences: pd.DataFrame,
    task_records: pd.DataFrame,
    *,
    max_time_difference_s: float | None = None,
) -> pd.DataFrame:
    """Audit physical task experiences against Comrie-style task records.

    The task relation is authoritative for the task-scoped port/reward
    sequence; DIO is authoritative for physical timing. A unique contiguous
    port/reward match is required. Extra physical experiences are permitted
    only before or after that block and are reported explicitly rather than
    silently deleted. Optional time tolerance makes timing disagreement a hard
    gate while retaining exact differences in the returned audit metadata.
    """
    required_experience = {"port", "is_reward", "arrival_time", "departure_time"}
    missing_experience = sorted(required_experience - set(port_experiences.columns))
    if missing_experience:
        raise KeyError(
            f"port experiences are missing alignment columns: {missing_experience}"
        )
    if "leaf" in task_records:
        task_ports = (
            pd.to_numeric(task_records["leaf"], errors="coerce").to_numpy(dtype=float)
            - 1.0
        )
    elif "port" in task_records:
        task_ports = pd.to_numeric(task_records["port"], errors="coerce").to_numpy(
            dtype=float
        )
    else:
        raise KeyError("task records require leaf (1-6) or port (0-5)")
    if "reward" not in task_records:
        raise KeyError("task records require reward")
    task_reward_raw = pd.to_numeric(task_records["reward"], errors="coerce").to_numpy(
        dtype=float
    )
    experience_ports = pd.to_numeric(
        port_experiences["port"], errors="coerce"
    ).to_numpy(dtype=float)
    experience_reward_raw = pd.to_numeric(
        port_experiences["is_reward"], errors="coerce"
    ).to_numpy(dtype=float)
    for values, name in (
        (task_ports, "task port"),
        (experience_ports, "experience port"),
    ):
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or not np.isin(values, tuple(WELL_PATCH_MAPPING)).all()
        ):
            raise ValueError(f"{name} values must be integer reward ports 0-5")
    for values, name in (
        (task_reward_raw, "task reward"),
        (experience_reward_raw, "experience reward"),
    ):
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{name} values must be binary")

    n_task = len(task_records)
    n_physical = len(port_experiences)
    if n_task == 0:
        raise ValueError("task record alignment requires at least one task row")
    if n_task > n_physical:
        raise ValueError(
            "task records contain more experiences than the physical DIO stream"
        )
    candidate_offsets = [
        offset
        for offset in range(n_physical - n_task + 1)
        if np.array_equal(experience_ports[offset : offset + n_task], task_ports)
        and np.array_equal(
            experience_reward_raw[offset : offset + n_task], task_reward_raw
        )
    ]
    if not candidate_offsets:
        raise ValueError(
            "task records do not match any contiguous port/reward sequence in "
            "the physical experiences"
        )

    has_task_times = {"poke_in_ts", "poke_out_ts"}.issubset(task_records.columns)
    if len(candidate_offsets) > 1 and not has_task_times:
        raise ValueError(
            "multiple physical subsequences match the task records; task poke "
            "times are required to disambiguate"
        )
    time_differences_by_offset: dict[int, np.ndarray] = {}
    if has_task_times:
        task_in = pd.to_numeric(task_records["poke_in_ts"], errors="coerce").to_numpy(
            dtype=float
        )
        task_out = pd.to_numeric(task_records["poke_out_ts"], errors="coerce").to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(task_in).all()
            or not np.isfinite(task_out).all()
            or (task_out < task_in).any()
        ):
            raise ValueError("task poke clocks must be finite and ordered")
        physical_in = port_experiences["arrival_time"].to_numpy(dtype=float)
        physical_out = port_experiences["departure_time"].to_numpy(dtype=float)
        for offset in candidate_offsets:
            time_differences_by_offset[offset] = np.r_[
                physical_in[offset : offset + n_task] - task_in,
                physical_out[offset : offset + n_task] - task_out,
            ]
        scores = {
            offset: float(np.max(np.abs(differences)))
            for offset, differences in time_differences_by_offset.items()
        }
        best_score = min(scores.values())
        best = [offset for offset, score in scores.items() if score == best_score]
        if len(best) != 1:
            raise ValueError("task timing does not uniquely disambiguate the match")
        offset = best[0]
        differences = time_differences_by_offset[offset]
    else:
        offset = candidate_offsets[0]
        differences = np.array([], dtype=float)

    max_abs_difference = (
        float(np.max(np.abs(differences))) if differences.size else np.nan
    )
    if max_time_difference_s is not None:
        if not np.isfinite(max_time_difference_s) or max_time_difference_s < 0.0:
            raise ValueError("max_time_difference_s must be finite and nonnegative")
        if max_abs_difference > max_time_difference_s:
            raise ValueError(
                "physical and task poke clocks exceed the declared tolerance; "
                f"max_abs_difference_s={max_abs_difference}, "
                f"tolerance_s={max_time_difference_s}"
            )

    aligned = port_experiences.copy()
    in_task = np.zeros(n_physical, dtype=bool)
    in_task[offset : offset + n_task] = True
    aligned["is_in_task_record"] = in_task
    task_number = np.full(n_physical, np.nan)
    if "trial_number_by_epoch" in task_records:
        task_ids = pd.to_numeric(
            task_records["trial_number_by_epoch"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        task_ids = np.arange(n_task, dtype=float)
    if (
        not np.isfinite(task_ids).all()
        or not np.equal(task_ids, np.floor(task_ids)).all()
        or len(np.unique(task_ids)) != n_task
    ):
        raise ValueError("task record IDs must be unique integers")
    task_number[offset : offset + n_task] = task_ids
    aligned["task_record_number"] = pd.array(task_number, dtype="Int64")
    aligned.attrs.update(port_experiences.attrs)
    aligned.attrs["leading_physical_experiences"] = offset
    aligned.attrs["trailing_physical_experiences"] = n_physical - offset - n_task
    aligned.attrs["matched_task_experiences"] = n_task
    aligned.attrs["max_abs_poke_time_difference_s"] = max_abs_difference
    return aligned
