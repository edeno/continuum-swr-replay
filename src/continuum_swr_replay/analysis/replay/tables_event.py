"""Canonical event-grain builders: one row per non-local event."""

from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.immobility import (
    require_canonical_speed_threshold,
)
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    event_strict_overlap_keys,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
)
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.event_ids import (
    hash_event_id as _hash_event_id,
)
from continuum_swr_replay.tables.event_ids import (
    hash_stop_id as _hash_stop_id,
)

from ._helpers import (
    align_posterior_to_dot as _align_posterior_to_dot,
)
from ._helpers import (
    arm_to_patch as _arm_to_patch,
)
from ._helpers import (
    arms_of_wells as _arms_of_wells,
)
from ._helpers import (
    first_row_floats as _first_row_floats,
)
from ._helpers import (
    safe_mode as _safe_mode,
)
from ._helpers import (
    single_arm_from_well as _single_arm_from_well,
)
from ._helpers import (
    stay_indicator as _stay_indicator,
)
from ._tables_common import (
    _EVENT_SEQUENCE_TABLE_BUILD_COLUMNS,
    _EVENT_TABLE_BUILD_COLUMNS,
    _in_event_mask,
    _prepare_dot,
)
from .bouts import (
    assign_bout as _assign_bout,
)
from .bouts import (
    immobility_bout_intervals as _immobility_bout_intervals,
)
from .bouts import (
    nearest_well_distance_cm as _nearest_well_distance_cm,
)
from .bouts import (
    time_since_stop as _time_since_stop,
)
from .covariates import (
    add_trial_context,
    summarize_behavioral_location,
)
from .summaries import (
    arm_map_from_posterior,
    n_nonlocal_bins,
    nonlocal_weight_sum,
    posterior_to_arm_marginal,
)
from .trajectory import (
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
    compute_trajectory_rollups,
)
from .uncertainty import (
    posterior_entropy_mean,
    posterior_hpd_widths,
    posterior_peak_mean,
)

# State-probability column mapping for the 4-state Non-Local model:
# per-event rollup column -> data_over_time per-bin column.
_NONLOCAL_STATE_COLS = {
    "local_prob_mean": "Local_prob",
    "nospike_prob_mean": "No-Spike_prob",
    "continuous_prob_mean": "Non-Local_Continuous_prob",
    "fragmented_prob_mean": "Non-Local_Fragmented_prob",
}

#: Trial-context columns each event reads at its first timebin, as floats.
_EVENT_FIRST_ROW_COLUMNS = (
    "prev_well",
    "next_well",
    "last_rewarded_well",
    "completed_trial_id",
    "did_last_completed_trip_switch_patch",
    "is_rewarded_outcome",
)

# ── Shared prep for the event / event-sequence builders ──


class _SessionKey(NamedTuple):
    """The (animal, day, epoch) identity of the session being built.

    Its fields mirror the canonical day-grain column tuple
    ``session_keys.SESSION_COLUMNS_DAY``; ``test_session_key_matches_columns_day``
    pins that correspondence so a future column rename cannot drift them apart.
    ``day`` is emitted to the output frame as the ``session_date`` column.
    """

    animal: str
    day: int
    epoch: str


def _annotate_dot_for_events(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    speed_threshold: float,
) -> tuple[pd.DataFrame, _SessionKey, set, np.ndarray, np.ndarray]:
    """Annotate the full dot frame and derive session identity + immobility bouts.

    Runs before event filtering so bout starts are correct even when an event
    begins partway through a bout, and per-timebin arm mapping is available.
    Returns ``(dot, session_key, mua_overlap_keys, bout_starts, bout_ends)``.
    """
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = _prepare_dot(dot, env)
    mua_overlap_keys = (
        event_strict_overlap_keys(dot, "non_local_event", "MUA_event")
        if "MUA_event" in dot.columns
        else set()
    )
    dot = add_trial_context(dot, trials, port_experiences)
    dot["nearest_well_distance_cm"] = _nearest_well_distance_cm(dot, track_graph)

    # Per-session immobility bouts — computed from the full frame so
    # bout starts are correct even when an event begins partway through
    # a bout. Both start and end timestamps are needed so events can be
    # mapped onto a stable ``stop_id`` (the hash of the bout index).
    bout_starts, bout_ends = _immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )
    session_key = _SessionKey(animal=animal, day=day, epoch=epoch)
    return dot, session_key, mua_overlap_keys, bout_starts, bout_ends


def _aggregate_event_features(
    nl: pd.DataFrame,
    event_id_series: np.ndarray,
    session_key: _SessionKey,
    mua_overlap_keys: set,
    track_graph: nx.Graph,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-event aggregates over the in-event timebins.

    Runs the hard-count summaries, the per-event metadata loop, and the
    sorted-array alignment. Returns ``(per_event, nonlocal_prob_arr)`` -- the
    ``nonlocal_prob`` array is reused by the posterior and trajectory rollups.
    """
    animal, day, epoch = session_key
    # Per-event summaries via the shared summary helpers.
    # ``nl`` is already filtered to ``nonlocal_prob > nonlocal_threshold``
    # (plus immobility and non-zero event label), so the hard-count
    # helper must not apply the threshold a second time — passing 0.0
    # means "count every row you were given". Keeps the two filter
    # definitions from silently diverging if one is ever tuned without
    # the other. ``nonlocal_weight_sum`` sums the raw w_t over the same
    # pre-filtered rows; this matches the legacy ``prepare_session``
    # aggregation contract.
    nonlocal_prob_arr = nl["nonlocal_prob"].to_numpy()
    ids_sorted, n_bins = n_nonlocal_bins(
        nonlocal_prob_arr, event_id_series, threshold=0.0
    )
    _, weight_sum = nonlocal_weight_sum(nonlocal_prob_arr, event_id_series)

    # Per-event metadata (groupby once, iterate sorted ids).
    groups = nl.groupby("non_local_event", sort=True)
    event_records: list[dict] = []

    # Trial-derived context at each event's first timebin, NaN kept, so an
    # event that begins before any trial has no trial context (as in the
    # stop tables).
    first = _first_row_floats(groups, _EVENT_FIRST_ROW_COLUMNS)
    prev_arms = _arms_of_wells(first["prev_well"], track_graph)
    next_arms = _arms_of_wells(first["next_well"], track_graph)
    last_rewarded_arms = _arms_of_wells(first["last_rewarded_well"], track_graph)

    for i, (ev_id, g) in enumerate(groups):
        times = g.index.to_numpy()
        arm_series = g["decoded_arm"]
        real_arm_mask = arm_series.isin(ALL_ARMS)
        arms_real = arm_series[real_arm_mask]
        # ``n_junction_bins`` is a misnomer kept for schema stability: it
        # counts bins whose ``decoded_arm`` is not a real arm, and
        # ``map_estimate_to_arm`` excludes junction bins from its
        # nearest-neighbour lookup, returning NaN only where
        # ``map_estimate`` itself is NaN. So this is an *undecoded*-bin
        # count, not a count of MAPs that landed on a junction.
        n_junction = int((~real_arm_mask).sum())

        # Keep behavioral location and physical segment as distinct facts.
        # The old trunk->destination-well fallback silently made an in-transit
        # event look as though it occurred at the future goal. The whole-window
        # location rollup is the model-facing definition; the modal physical
        # segment remains available for topology/QC without changing it.
        ts = (
            g["track_segment_id"]
            if "track_segment_id" in g.columns
            else pd.Series(dtype=float)
        )
        physical_arm = _safe_mode(ts)
        location = summarize_behavioral_location(
            g[["occupancy_state", "occupancy_well_id"]]
        )
        location_well = float(location["occupancy_well_id"])
        location_arm = _single_arm_from_well(location_well, track_graph)
        current_arm = location_arm
        prev_arm = prev_arms[i]
        next_arm = next_arms[i]
        last_rewarded_arm = last_rewarded_arms[i]

        context_time = float(times[0])
        last_port_entry = g["_goal_arrival_time"].iloc[0]
        time_since_last_port_entry = (
            context_time - float(last_port_entry)
            if pd.notna(last_port_entry)
            else np.nan
        )
        if np.isfinite(time_since_last_port_entry) and time_since_last_port_entry < 0:
            raise RuntimeError(
                "past-only trial context produced a negative time since port entry"
            )
        distance = g["nearest_well_distance_cm"].to_numpy(dtype=float)
        finite_distance = distance[np.isfinite(distance)]

        event_records.append(
            {
                "non_local_event": int(ev_id),
                "is_mua": (animal, day, epoch, int(ev_id)) in mua_overlap_keys,
                "occupancy_state": location["occupancy_state"],
                "at_well": location["at_well"],
                "occupancy_well_id": location_well,
                "modal_observed_track_segment_id": physical_arm,
                "occupancy_well_arm": current_arm,
                "prev_arm": prev_arm,
                "next_arm": next_arm,
                "last_rewarded_arm": last_rewarded_arm,
                "current_patch": _arm_to_patch(current_arm),
                "prev_patch": _arm_to_patch(prev_arm),
                "next_patch": _arm_to_patch(next_arm),
                # Completion-order ID of the trip completed by the event
                # start. Two visits to the same well inside one long
                # immobility bout are separated only by this ID, so
                # downstream visit-grain grouping cannot tell them apart
                # without it.
                "completed_trial_id": first["completed_trial_id"][i],
                "did_last_completed_trip_switch_patch": first[
                    "did_last_completed_trip_switch_patch"
                ][i],
                # The native trial reward/outcome indicators share one source
                # array (see _helpers); persist one canonical
                # ``is_rewarded_outcome``. The ``is_omitted_outcome``
                # complement and the stay flag are derived downstream, so
                # neither is persisted here.
                "is_rewarded_outcome": first["is_rewarded_outcome"][i],
                "is_stay": _stay_indicator(next_arm, current_arm),
                "n_undecoded_bins": n_junction,
                "n_unique_arms": int(arms_real.nunique()),
                "time_since_last_well_arrival_s": time_since_last_port_entry,
                "time_since_last_reward_s": float(
                    context_time - g["_last_reward_time"].iloc[0]
                )
                if pd.notna(g["_last_reward_time"].iloc[0])
                else np.nan,
                "nearest_well_distance_cm_min": (
                    float(finite_distance.min()) if finite_distance.size else np.nan
                ),
                "nearest_well_distance_cm_mean": (
                    float(finite_distance.mean()) if finite_distance.size else np.nan
                ),
                "nearest_well_distance_cm_max": (
                    float(finite_distance.max()) if finite_distance.size else np.nan
                ),
            }
        )

    per_event = pd.DataFrame.from_records(event_records)
    per_event["at_well"] = pd.array(per_event["at_well"], dtype="boolean")

    # Align summary arrays (sorted by event id) with per_event.
    # summaries.* returned unique_ids in sorted order; groupby(sort=True)
    # yields events in the same order — assert to catch drift.
    if not np.array_equal(
        per_event["non_local_event"].to_numpy(),
        ids_sorted.astype(per_event["non_local_event"].dtype),
    ):
        raise RuntimeError(
            "Event-id sort mismatch between groupby result and per_event table"
        )
    per_event["n_nonlocal_bins"] = n_bins
    per_event["nonlocal_probability_sum"] = weight_sum
    return per_event, nonlocal_prob_arr


def _attach_time_since_stop(
    per_event: pd.DataFrame, nl: pd.DataFrame, bout_starts: np.ndarray
) -> pd.DataFrame:
    """Attach ``time_since_stop_start_s`` (median of per-timebin values)."""
    per_timebin_tss = _time_since_stop(nl.index.to_numpy(), bout_starts)
    tss_by_event = (
        pd.Series(per_timebin_tss, index=nl.index)
        .groupby(nl["non_local_event"].to_numpy(), sort=True)
        .median()
        .to_numpy()
    )
    per_event["time_since_stop_start_s"] = tss_by_event
    return per_event


def _attach_posterior_rollups(
    per_event: pd.DataFrame,
    conditional_posterior,
    dot: pd.DataFrame,
    env,
    in_event_mask: pd.Series,
    nonlocal_prob_arr: np.ndarray,
    event_id_series: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Attach the conditional-posterior-dependent columns.

    Absent ``conditional_posterior``, emit the canonical columns as NaN so the
    schema is stable across call sites. Returns ``(per_event, arm_posterior)``
    where ``arm_posterior`` (or ``None``) feeds the trajectory rollups.
    """
    posterior_cols = (
        "posterior_peak_mean",
        "posterior_entropy_mean",
        "posterior_hpd_arm_count_mean_coverage_50pct",
        "posterior_hpd_arm_count_mean_coverage_80pct",
        "posterior_hpd_arm_count_mean_coverage_95pct",
    )
    arm_posterior = None  # set below when conditional_posterior is supplied
    if conditional_posterior is None:
        for c in posterior_cols:
            per_event[c] = np.nan
    else:
        edge_id = np.asarray(env.place_bin_centers_nodes_df_["edge_id"])
        post_vals = _align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event_mask.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(
            post_in_event, edge_id, n_arms=len(ALL_ARMS)
        )

        _, peak = posterior_peak_mean(arm_posterior, nonlocal_prob_arr, event_id_series)
        _, entropy = posterior_entropy_mean(
            arm_posterior, nonlocal_prob_arr, event_id_series
        )
        _, hpd = posterior_hpd_widths(
            arm_posterior,
            nonlocal_prob_arr,
            event_id_series,
            coverages=(0.5, 0.8, 0.95),
        )

        per_event["posterior_peak_mean"] = peak
        per_event["posterior_entropy_mean"] = entropy
        per_event["posterior_hpd_arm_count_mean_coverage_50pct"] = hpd[0.5]
        per_event["posterior_hpd_arm_count_mean_coverage_80pct"] = hpd[0.8]
        per_event["posterior_hpd_arm_count_mean_coverage_95pct"] = hpd[0.95]
    return per_event, arm_posterior


def _attach_trajectory_rollups(
    per_event: pd.DataFrame,
    nl: pd.DataFrame,
    event_id_series: np.ndarray,
    env,
    track_graph: nx.Graph,
    place_bin_size: float,
    nonlocal_prob_arr: np.ndarray,
    arm_posterior,
    nonlocal_threshold: float,
) -> pd.DataFrame:
    """Attach the trajectory rollups.

    These require the augmented data_over_time columns (step_distance, raw
    ripple/MUA/theta, run_speed_baseline). When ``dot`` predates the
    augmentation, emit the columns as NaN so the schema stays stable.
    """
    trajectory_cols = (*SHARED_TRAJECTORY_COLUMNS, *NONLOCAL_ONLY_COLUMNS)
    if {"step_distance", "run_speed_baseline"}.issubset(nl.columns):
        rollups = compute_trajectory_rollups(
            nl=nl,
            event_id=event_id_series,
            env=env,
            track_graph=track_graph,
            state_prob_cols=_NONLOCAL_STATE_COLS,
            continuous_col="Non-Local_Continuous_prob",
            fragmented_col="Non-Local_Fragmented_prob",
            place_bin_size=place_bin_size,
            nonlocal_prob=nonlocal_prob_arr,
            arm_posterior=arm_posterior,
            non_local_threshold=nonlocal_threshold,
        )
        for col_name in trajectory_cols:
            per_event[col_name] = rollups[col_name]
    else:
        for col_name in trajectory_cols:
            per_event[col_name] = np.nan
    return per_event


def _attach_identifiers(
    per_event: pd.DataFrame, session_key: _SessionKey
) -> pd.DataFrame:
    """Insert the session identifiers and the stable ``event_id`` hash."""
    animal, day, epoch = session_key
    per_event.insert(0, "animal", animal)
    per_event.insert(1, "session_date", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_id"] = _hash_event_id(per_event)
    return per_event


def _attach_stop_ids(
    per_event: pd.DataFrame,
    session_key: _SessionKey,
    bout_starts: np.ndarray,
    bout_ends: np.ndarray,
) -> pd.DataFrame:
    """Map each event to its immobility bout (``stop_idx`` / ``stop_id``)."""
    animal, day, epoch = session_key
    # Each event's start belongs to exactly one immobility bout (the in-event
    # filter guarantees immobility, so ``_assign_bout`` should find a valid
    # bout for every row).
    stop_idx = _assign_bout(
        per_event["event_start_s"].to_numpy(), bout_starts, bout_ends
    )
    n_outside = int((stop_idx < 0).sum())
    if n_outside > 0:
        raise ValueError(
            f"{n_outside}/{len(stop_idx)} events fall outside all immobility bouts"
        )
    per_event["stop_idx"] = stop_idx
    per_event["stop_id"] = _hash_stop_id(animal, day, epoch, stop_idx)
    return per_event


def _merge_classified_event_facts(
    per_event: pd.DataFrame,
    dot: pd.DataFrame,
    speed_threshold: float,
    nonlocal_threshold: float,
) -> pd.DataFrame:
    """Attach the canonical event timing and the SWR classification.

    Both come from the same second pass over the event labels
    (:func:`build_classified_nonlocal_event_table`), which covers the identical
    complete window the aggregate pass does -- ``_in_event_mask`` validates the
    speed/probability contract without clipping. Timing is therefore *derived
    once*, here, rather than recomputed by the aggregate pass and checked for
    agreement afterwards. Merged on ``event_id``; every model-cohort event must
    have exactly one classified row.
    """
    classified = build_classified_nonlocal_event_table(
        dot,
        speed_threshold=speed_threshold,
        nonlocal_threshold=nonlocal_threshold,
    )
    classified = classified.rename(
        columns={
            "event_start": "event_start_s",
            "event_end": "event_end_s",
            "event_duration": "event_duration_s",
            "event_midpoint_time": "event_midpoint_s",
        }
    )
    classified_cols = [
        "event_id",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "event_midpoint_s",
        "animal_max_speed_cm_s",
        "animal_mean_speed_cm_s",
        "animal_start_speed_cm_s",
        "animal_end_speed_cm_s",
        "whole_event_speed_eligible",
        "endpoints_speed_below_thresh",
        "peak_nonlocal_prob",
        "is_swr",
        "overlap_class",
        "overlap_subclass",
        "n_swrs_overlapping",
        "matched_swr_event",
        "matched_swr_start",
        "matched_swr_end",
    ]
    per_event = per_event.merge(
        classified[classified_cols],
        on="event_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    # ``is_swr`` is a non-nullable bool in the classified table, so a NaN
    # here can only mean an unmatched left key (a legacy event with no
    # classified row). Correct *matching* relies on both passes hashing
    # the same (animal, day, epoch, non_local_event) tuple via
    # ``hash_event_id`` -- this guard catches unmatched rows, not a
    # hypothetical mismatch from changing the key in only one pass. An
    # unmatched row would also leave the event with no timing at all.
    n_unclassified = int(per_event["is_swr"].isna().sum())
    if n_unclassified > 0:
        raise RuntimeError(
            f"{n_unclassified}/{len(per_event)} replay events have no row in "
            "the classified event table; both passes cover the identical "
            "validated label window, so an unmatched key means the two "
            "passes hashed different (animal, day, epoch, non_local_event) "
            "tuples."
        )
    return per_event


def build_event_table(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    conditional_posterior=None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    place_bin_size: float = PLACE_BIN_SIZE_1D,
) -> pd.DataFrame:
    """One row per non-local event with canonical event-level columns.

    Consumes the pre-saved ``{epoch}_HPC_{thr}_data_over_time.pkl`` frame
    (``dot``), read through the SWR source
    (``analysed_time.read_dot_with_swr_source``) so its ``SWR_event`` is the
    paper's SWRs, plus the session's ``env`` / ``trials`` /
    ``port_experiences`` / ``track_graph``
    and emits the event-level slice of the canonical schema.

    The table is composed from two passes over the same complete, validated
    event-label window:

    - **Event aggregates** (arms, patches, context flags, content, posterior
      summaries) use every bin in an accepted label. ``_in_event_mask``
      validates that all bins meet the saved speed and probability contracts;
      it never clips individual bins.
    - **Event timing and SWR classification** come from
      :func:`continuum_swr_replay.analysis.events.classified_table.build_classified_nonlocal_event_table`,
      and are merged on ``event_id``: the canonical ``event_*_s`` window,
      ``is_swr`` (strict positive-duration SWR overlap), ``overlap_class`` /
      ``overlap_subclass`` / ``n_swrs_overlapping`` / ``matched_swr_*``, speed
      summaries, and ``peak_nonlocal_prob``.

    Both passes cover the identical validated label window, so the timing is
    derived once -- in the classified pass -- rather than computed twice and
    checked for agreement. ``stop_idx`` / ``stop_id`` are then assigned from
    that single ``event_start_s``.

    ``conditional_posterior`` is optional. When supplied (e.g., from
    :func:`posterior_analysis.load_conditional_posterior`), five additional columns
    are populated: ``posterior_peak_mean``, ``posterior_entropy_mean``, and
    ``posterior_hpd_arm_count_mean_coverage_{50,80,95}pct``. When not supplied
    those columns are ``NaN`` so the schema is stable across call sites.
    (``nonlocal_probability_sum`` is always emitted, independent of the
    posterior.)

    Intentionally absent from the current output: the stop-level
    ``immobility_time`` aggregation (``build_stop_table``);
    ``block_number`` and ``trials_since_block_change`` (requires a
    contingency-block loader not yet wired).

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed data-over-time frame produced by
        :func:`continuum_swr_replay.analysis.hippocampus.data_aggregation.get_data_over_time`.
        Must carry ``animal``, ``day``, ``epoch``, ``non_local_event``,
        ``SWR_event``, ``MUA_event``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``, ``map_estimate``, ``head_speed``.
    env : Environment
        Decoder environment for ``map_estimate_to_arm``.
    trials : pd.DataFrame
        Spyglass trial frame with ``movement_start_time``, ``end_time``,
        ``to_well``, ``from_well``, ``is_reward``, ``is_patch_change``.
    port_experiences : pd.DataFrame
        Complete reward-port occupancy stream: the starting port followed by
        one destination per movement trial. Visit covariates use every spatial
        occupancy; reward covariates use only physically observed outcomes.
        Movement context continues to use ``trials``.
    track_graph : networkx.Graph
        Session track graph; edges carry ``edge_id``.
    speed_threshold : float
        Head-speed threshold for immobility (cm/s). Only the project-wide
        constant is accepted: the read gates the SWR and MUA labels at it.
    nonlocal_threshold : float
        Non-local state probability threshold. Default matches the
        project-wide constant.

    Returns
    -------
    pd.DataFrame
        One row per event with the columns documented below. Empty
        frame (correct columns, zero rows) if the session has no
        qualifying events.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``.

    Columns
    -------
    Identifiers
        ``animal``, ``session_date``, ``epoch``, ``non_local_event``, ``event_id``
    Timing (canonical complete event window)
        ``event_start_s``, ``event_end_s``, ``event_duration_s``,
        ``event_midpoint_s``, ``time_since_stop_start_s``
    Behavioral location and arms
        ``occupancy_state``, ``at_well``, ``occupancy_well_id``,
        ``occupancy_well_arm``, ``modal_observed_track_segment_id``,
        ``prev_arm``, ``next_arm`` (post-hoc), ``last_rewarded_arm``
    Patches
        ``current_patch``, ``prev_patch``, ``next_patch`` (post-hoc)
    Context flags
        ``did_last_completed_trip_switch_patch``, ``is_rewarded_outcome``,
        and ``is_stay``
    Event-type
        ``is_swr`` (strict positive-duration SWR overlap; see
        :mod:`continuum_swr_replay.analysis.events.classified_table`),
        ``is_mua`` (strict positive-duration MUA overlap)
    SWR overlap (from the classified table)
        ``overlap_class``, ``overlap_subclass``, ``n_swrs_overlapping``,
        ``matched_swr_event``, ``matched_swr_start``, ``matched_swr_end``
    Per-event speed / peak (full label window)
        ``max_speed``, ``mean_speed``, ``start_speed``, ``end_speed``,
        ``whole_event_speed_eligible``, the legacy endpoint diagnostic,
        ``peak_nonlocal_prob``
    Stop linkage
        ``stop_idx``, ``stop_id``
    Content summaries
        ``n_nonlocal_bins``, ``nonlocal_probability_sum``,
        ``n_undecoded_bins``, ``n_unique_arms``.
        ``n_undecoded_bins`` counts in-event timebins whose ``decoded_arm``
        is not a real arm. ``map_estimate_to_arm`` excludes those bins
        from its lookup and returns NaN only where ``map_estimate`` is
        NaN, so this is an *undecoded*-bin count, not a count of MAPs
        that landed on a junction.

    Notes
    -----
    ``event_id`` is a 63-bit pandas hash of
    ``(animal, day, epoch, non_local_event)`` — stable across runs and
    across row-ordering. Collisions are negligible at project scale
    (~50k events) but would surface as events with >1 unique
    ``non_local_event`` under the same ``event_id``; callers that need
    paranoia can group on the tuple instead.

    ``location_state`` is rolled up over the complete event window. An event
    is ``at_well`` only when every sampled bin is at the same well; canonical
    tables retain ``in_transit``, ``mixed``, and ``unknown`` events so model
    cohorts can filter them explicitly. ``physical_arm`` is the modal observed
    track segment, including trunks.

    ``occupancy_well_arm`` and ``modal_observed_track_segment_id`` answer
    different questions (see ``behavioral_location_from_port_experiences`` for the full
    note): ``occupancy_well_arm`` is a **task state** -- which well is being
    visited, from trip intervals, NaN mid-trip -- while
    ``modal_observed_track_segment_id`` is a **position** -- which graph edge
    the tracking falls on, always defined. They disagree on 46 of 59,374
    at-well events, where the animal is inside its well-visit interval but a
    mean 48.4 cm from the well. Neither substitutes for the other.

    ``in_transit`` labels the trip interval, **not** motion: the modelled cohort
    is speed-gated immobility throughout, and off-well events have a median
    speed of 1.35 cm/s. Those events are animals standing still partway through
    a trip. They are excluded from the content models because the well-visit
    construct is undefined there, not because the animal was running.
    """
    require_canonical_speed_threshold(speed_threshold, "build_event_table")
    dot, session_key, mua_overlap_keys, bout_starts, bout_ends = (
        _annotate_dot_for_events(
            dot, env, trials, port_experiences, track_graph, speed_threshold
        )
    )

    # Filter to the in-event timebins the rest of the builder aggregates over.
    in_event_mask = _in_event_mask(dot, speed_threshold, nonlocal_threshold)
    nl = dot.loc[in_event_mask]

    # Empty-output short-circuit — return a correctly-typed empty frame
    # so downstream concat does not blow up on column mismatch.
    column_order = list(_EVENT_TABLE_BUILD_COLUMNS)
    if len(nl) == 0:
        return stamp_covariate_schema(pd.DataFrame(columns=column_order), table="event")

    event_id_series = nl["non_local_event"].to_numpy()

    per_event, nonlocal_prob_arr = _aggregate_event_features(
        nl, event_id_series, session_key, mua_overlap_keys, track_graph
    )

    per_event = _attach_time_since_stop(per_event, nl, bout_starts)

    per_event, arm_posterior = _attach_posterior_rollups(
        per_event,
        conditional_posterior,
        dot,
        env,
        in_event_mask,
        nonlocal_prob_arr,
        event_id_series,
    )

    per_event = _attach_trajectory_rollups(
        per_event,
        nl,
        event_id_series,
        env,
        track_graph,
        place_bin_size,
        nonlocal_prob_arr,
        arm_posterior,
        nonlocal_threshold,
    )

    per_event = _attach_identifiers(per_event, session_key)

    per_event = _merge_classified_event_facts(
        per_event, dot, speed_threshold, nonlocal_threshold
    )

    per_event = _attach_stop_ids(per_event, session_key, bout_starts, bout_ends)

    return stamp_covariate_schema(per_event[column_order], table="event")


def build_event_sequence_table(
    dot: pd.DataFrame,
    env,
    conditional_posterior,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per in-event timebin with the arm-marginal posterior.

    The per-timebin canonical view of what the decoder said each timebin
    represented, inside non-local events. Its ``event_id`` hash matches
    :func:`build_event_table` row-for-row, so the two can be joined on
    ``event_id`` without loss.

    Parameters
    ----------
    dot : pd.DataFrame
        Same as :func:`build_event_table`.
    env : Environment
        Same as :func:`build_event_table`; ``place_bin_centers_nodes_df_``
        supplies the ``edge_id`` vector that drives the arm-posterior
        collapse (junction bins excluded and renormalized in
        :func:`summaries.posterior_to_arm_marginal`).
    conditional_posterior : xr.DataArray
        Shape ``(time, position)`` — from
        :func:`posterior_analysis.load_conditional_posterior`. Must have the
        same time length as ``dot``.
    speed_threshold, nonlocal_threshold : float
        Same defaults as :func:`build_event_table` (``speed_threshold``
        accepts only the canonical value there and here). Keeping the two
        filters in lockstep is important: ``event_sequence_table`` and
        ``event_table`` must cover exactly the same in-event rows, or
        joins across them drift.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``animal``, ``session_date``, ``epoch``, ``non_local_event``,
          ``event_id`` (identifiers; matches ``event_table``)
        - ``t_in_event`` — 0-indexed position within the event
        - ``time`` — raw timebin timestamp in seconds
        - ``nonlocal_prob`` — ``w_t`` at the timebin
        - ``decoded_arm_map`` — MAP arm from the arm-marginal posterior
        - ``arm_posterior_0`` … ``arm_posterior_8`` — per-arm
          probabilities (rows sum to 1 up to fp tolerance)

    Notes
    -----
    The arm-marginal posterior used here is the same one that feeds
    :func:`build_event_table`'s posterior-dependent columns — no
    arithmetic difference can leak between the two tables.
    """
    require_canonical_speed_threshold(speed_threshold, "build_event_sequence_table")
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = _prepare_dot(dot, env)
    in_event_mask = _in_event_mask(dot, speed_threshold, nonlocal_threshold)

    column_order = list(_EVENT_SEQUENCE_TABLE_BUILD_COLUMNS)
    if not in_event_mask.any():
        return stamp_covariate_schema(
            pd.DataFrame(columns=column_order), table="event_sequence"
        )

    nl = dot.loc[in_event_mask]
    edge_id = np.asarray(env.place_bin_centers_nodes_df_["edge_id"])
    post_vals = _align_posterior_to_dot(conditional_posterior, dot)
    post_in_event = post_vals[in_event_mask.to_numpy()]
    arm_posterior = posterior_to_arm_marginal(
        post_in_event, edge_id, n_arms=len(ALL_ARMS)
    )
    decoded_arm_map = arm_map_from_posterior(arm_posterior)

    # t_in_event: 0-indexed position within each event, in timebin order.
    # ``groupby(...).cumcount()`` preserves the original row order within
    # each group, which matches the sort-by-time order of ``nl`` since
    # ``dot`` is time-sorted.
    t_in_event = nl.groupby("non_local_event", sort=False).cumcount().to_numpy()

    out = pd.DataFrame(
        {
            "animal": animal,
            "session_date": day,
            "epoch": epoch,
            "non_local_event": nl["non_local_event"].to_numpy(),
            "t_in_event": t_in_event,
            "time": nl.index.to_numpy(),
            "nonlocal_prob": nl["nonlocal_prob"].to_numpy(),
            "decoded_arm_map": decoded_arm_map,
        }
    )
    for arm in ALL_ARMS:
        out[f"arm_posterior_{arm}"] = arm_posterior[:, arm]

    out["event_id"] = _hash_event_id(out)
    return stamp_covariate_schema(out[column_order], table="event_sequence")
