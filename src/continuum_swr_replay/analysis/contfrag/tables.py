"""Canonical per-session tables for ContFrag replay content analysis.

This module produces event tables from externally-defined events (SWR or
high-MUA times), using the ContFrag classifier's decoded posterior for
content analysis. The key difference from the NonLocal tables
(``analysis.replay.tables_*``) is that events come from external detectors,
not from model-detected non-local probability thresholding.

The ContFrag model has 2 states (Continuous, Fragmented), both of which
decode meaningful spatial content. Event-level summaries include the
state composition during each event (continuous_prob_mean, dominant_state).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.analysis.hippocampus.event_detection import (
    event_strict_overlap_keys,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    require_canonical_speed_threshold,
    whole_event_immobility_mask,
)
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema
from continuum_swr_replay.tables.event_ids import hash_stop_id

from ...data_loaders.constants import ALL_ARMS
from ...parameters import PLACE_BIN_SIZE_1D
from ..hippocampus.swr_source import excluded_intervals_of
from ..replay._helpers import (
    align_count_matrix,
    align_posterior_to_dot,
    arm_to_patch,
    assert_event_label_alignment,
    assert_single_session_alignment,
    safe_mode,
    single_arm_from_well,
    stay_indicator,
    stay_indicator_from_patches,
    strict_first,
)
from ..replay.bouts import (
    assign_bout,
    bout_exposure_s,
    immobility_bout_intervals,
    nearest_well_distance_cm,
    summarize_bout_location,
    time_since_stop,
)
from ..replay.covariates import (
    add_trial_context,
    recent_trial_reward_rate,
    reward_history_basis_features,
    summarize_behavioral_location,
)
from ..replay.covariates.candidate_arm import (
    attach_candidate_arm_covariates,
    broadcast_candidate_arms,
)
from ..replay.design_constants import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from ..replay.spatial_mapping import map_estimate_to_arm
from ..replay.summaries import (
    arm_count_hard,
    arm_count_soft,
    arm_fraction_from_event_total,
    arm_posterior_mass,
    posterior_to_arm_marginal,
)
from ..replay.table_contracts import validate_event_arm_table
from ..replay.trajectory import (
    SHARED_TRAJECTORY_COLUMNS,
    compute_trajectory_rollups,
)
from ..replay.uncertainty import (
    posterior_entropy_mean,
    posterior_hpd_widths,
    posterior_peak_mean,
)

# ContFrag state-probability column mapping (2-state model).
_CONTFRAG_STATE_COLS = {
    "continuous_prob_mean": "Continuous_prob",
    "fragmented_prob_mean": "Fragmented_prob",
}
# ``continuous_prob_mean`` / ``fragmented_prob_mean`` are produced by
# ``_summarize_event_state`` (pre-augmentation-safe), so the orchestrator
# supplies every shared column *except* those two.
_CONTFRAG_TRAJECTORY_COLUMNS = tuple(
    c
    for c in SHARED_TRAJECTORY_COLUMNS
    if c not in ("continuous_prob_mean", "fragmented_prob_mean")
)


def _prepare_contfrag_dot(dot: pd.DataFrame, env) -> pd.DataFrame:
    """Annotate ``dot`` with ``decoded_arm`` for ContFrag data.

    Unlike the NonLocal version, there is no ``nonlocal_prob`` column
    to compute — both states always decode position.
    """
    dot = dot.copy()
    dot["decoded_arm"] = map_estimate_to_arm(dot["map_estimate"].to_numpy(), env)
    return dot


_EVENT_SOURCE_TO_COL = {"SWR": "SWR_event", "MUA": "MUA_event"}


def _resolve_event_col(event_source: str) -> str:
    """Map ``event_source`` to its data-over-time column name.

    Strict dispatch: anything other than ``"SWR"`` / ``"MUA"`` raises
    ``ValueError``. The previous inline form
    ``"SWR_event" if event_source == "SWR" else "MUA_event"`` silently
    routed any typo (``"swr"``, ``"Swr"``, ``"ripple"``…) to the MUA
    column, so direct calls into these public helpers could build a
    table for the wrong event source without warning.
    """
    try:
        return _EVENT_SOURCE_TO_COL[event_source]
    except KeyError as exc:
        raise ValueError(
            f"event_source must be 'SWR' or 'MUA', got {event_source!r}"
        ) from exc


def _in_contfrag_event_mask(
    dot: pd.DataFrame,
    speed_threshold: float,
    event_source: str = "SWR",
) -> pd.Series:
    """Boolean mask for in-event timebins using external event detectors.

    Thin wrapper over
    :func:`~continuum_swr_replay.analysis.hippocampus.event_immobility.whole_event_immobility_mask`,
    which owns the repo-wide definition. What was once a ContFrag-local
    choice — all-immobile rather than per-timebin or endpoints, so that
    movement-spanning events cannot straddle two immobility bouts and
    break ``Σ stop.n_selected_events == len(event_table)`` — is now the shared rule
    for every event source. The degenerate multi-second ``MUA_event``
    labels in ``docs/known_issues/degenerate_mua_events.md`` are the
    motivating case.

    On a frame read through ``swr_source.with_swr_source`` this is a no-op;
    it excludes nothing further. The read holds every label to the rule (an
    SWR that moves is dropped, a moving MUA label is refused), and the
    historical ~0.5% SWR / ~0.7% MUA exclusion happens upstream at cohort
    construction.
    """
    event_col = _resolve_event_col(event_source)
    return whole_event_immobility_mask(
        dot,
        event_col,
        speed_threshold=speed_threshold,
    )


def _hash_contfrag_event_id(
    events: pd.DataFrame,
    event_source: str = "SWR",
    brain_area: str = "HPC",
) -> np.ndarray:
    """63-bit stable hash of ``(animal, session_date, epoch, event_label)``.

    ``event_source`` and ``brain_area`` are mixed into the hash so a
    SWR/HPC event with label 7 and a MUA/HPC event with label 7 (or
    SWR/mPFC) cannot collide. Without this, concatenating tables
    across sources or brain areas produces silent ID collisions on
    every join keyed by ``event_id``.
    """
    event_col = _resolve_event_col(event_source)
    # Read the session-date column by value; ``pd.util.hash_pandas_object``
    # ignores column labels, so the id is identical whether the frame spells
    # it ``session_date`` (canonical) or ``day`` (raw intermediate).
    date_col = "session_date" if "session_date" in events.columns else "day"
    keys = events[["animal", date_col, "epoch", event_col]].copy()
    keys["__event_source"] = event_source
    keys["__brain_area"] = brain_area
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def _summarize_event_state(
    cont_prob: np.ndarray, frag_prob: np.ndarray
) -> tuple[float, float, str | None, float]:
    """Per-event Continuous/Fragmented summary with NaN-safe semantics.

    Returns ``(cont_mean, frag_mean, dominant, cont_frac)``.

    Both means are the per-state ``nanmean`` (NaN if that state has no
    finite bin). ``dominant`` and ``cont_frac`` are well-defined only
    when *both* states have at least one finite bin — otherwise the
    comparison ``cont_prob > frag_prob`` and the ``cont_mean >= frag_mean``
    test silently route NaN-side missingness into a confident label.
    Surface ``None`` / NaN in those degenerate cases instead.
    """
    cont_finite = bool(np.isfinite(cont_prob).any())
    frag_finite = bool(np.isfinite(frag_prob).any())
    # Reported per-state means use each state's own finite bins —
    # they're independent estimates of the state's average probability
    # across the event and are meant to be read on their own.
    cont_mean = float(np.nanmean(cont_prob)) if cont_finite else float("nan")
    frag_mean = float(np.nanmean(frag_prob)) if frag_finite else float("nan")
    # Both ``dominant`` and ``cont_frac`` need a comparable view of
    # the two states. Compute them from PAIRED-FINITE bins only —
    # bins where both states have a finite probability — so the
    # reported label and bin-wise fraction agree. The previous form
    # built ``dominant`` from the per-state nanmeans (which include
    # unpaired bins) and ``cont_frac`` from paired bins, so on inputs
    # like ``cont=[0.9, 0.1]``, ``frag=[0.8, NaN]`` we got
    # ``dominant='Fragmented'`` (per-state means: 0.5 vs 0.8) while
    # the only comparable bin had cont > frag, ``cont_frac=1.0``. NaN
    # comparisons silently return False, so paired-mean masking is
    # also what stops partial-NaN bins from being counted as
    # "Fragmented" wins.
    paired = np.isfinite(cont_prob) & np.isfinite(frag_prob)
    if paired.any():
        paired_cont_mean = float(cont_prob[paired].mean())
        paired_frag_mean = float(frag_prob[paired].mean())
        dominant: str | None = (
            "Continuous" if paired_cont_mean >= paired_frag_mean else "Fragmented"
        )
        cont_frac = float((cont_prob[paired] > frag_prob[paired]).mean())
    else:
        dominant = None
        cont_frac = float("nan")
    return cont_mean, frag_mean, dominant, cont_frac


def build_contfrag_event_table(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    event_source: str = "SWR",
    brain_area: str = "HPC",
    conditional_posterior=None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    place_bin_size: float = PLACE_BIN_SIZE_1D,
) -> pd.DataFrame:
    """One row per externally-defined event with canonical event-level columns.

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed data-over-time frame from
        ``get_contfrag_data_over_time()``.
    env : Environment
        Decoder environment for ``map_estimate_to_arm``.
    trials : pd.DataFrame
        Trial frame with ``start_time``, ``end_time``, ``to_well``,
        ``from_well``, ``is_reward``, ``is_patch_change``.
    port_experiences : pd.DataFrame
        Complete starting-port-plus-destinations occupancy stream. Visit
        covariates use every occupancy; reward covariates use only observed
        outcomes.
    track_graph : nx.Graph
        Session track graph.
    event_source : {"SWR", "MUA"}
        Which external event detector to use for event boundaries.
    conditional_posterior : xr.DataArray, optional
        Combined posterior for posterior-dependent columns.
    speed_threshold : float
        Head-speed threshold for immobility (cm/s). Only the canonical value
        is accepted: the read gates the other detector's labels at it.

    Returns
    -------
    pd.DataFrame
        One row per event. Includes all standard event-table columns
        plus ContFrag-specific: ``continuous_prob_mean``,
        ``fragmented_prob_mean``, ``dominant_state`` (ties go to
        Continuous), ``continuous_fraction``.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``.
    """
    require_canonical_speed_threshold(speed_threshold, "build_contfrag_event_table")
    if "animal" not in dot.columns:
        raise KeyError(
            "dot must carry 'animal' column (set by get_contfrag_data_over_time)"
        )
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    event_col = _resolve_event_col(event_source)

    dot = _prepare_contfrag_dot(dot, env)
    other_event_col = "MUA_event" if event_col == "SWR_event" else "SWR_event"
    other_overlap_keys = (
        event_strict_overlap_keys(dot, event_col, other_event_col)
        if other_event_col in dot.columns
        else set()
    )
    dot = add_trial_context(dot, trials, port_experiences)
    dot["nearest_well_distance_cm"] = nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )

    in_event = _in_contfrag_event_mask(dot, speed_threshold, event_source)
    ev = dot.loc[in_event]

    column_order = _contfrag_event_table_build_columns(event_source)
    if len(ev) == 0:
        return stamp_covariate_schema(pd.DataFrame(columns=column_order), table="event")

    event_id_series = ev[event_col].to_numpy()

    groups = ev.groupby(event_col, sort=True)
    event_records: list[dict] = []

    for ev_id, g in groups:
        times = g.index.to_numpy()
        arm_series = g["decoded_arm"]
        real_arm_mask = arm_series.isin(ALL_ARMS)
        arms_real = arm_series[real_arm_mask]
        n_junction = int((~real_arm_mask).sum())

        # Keep whole-window behavioral location separate from the observed
        # physical segment. A trunk/in-transit event must not be rewritten as
        # the trip's destination well.
        ts = (
            g["track_segment_id"]
            if "track_segment_id" in g.columns
            else pd.Series(dtype=float)
        )
        physical_arm = safe_mode(ts)
        location = summarize_behavioral_location(
            g[["occupancy_state", "occupancy_well_id"]]
        )
        location_well = float(location["occupancy_well_id"])
        location_arm = single_arm_from_well(location_well, track_graph)
        current_arm = location_arm
        # Trial-derived context columns are NaN at every dot timebin
        # before the first trial start (see ``add_trial_context``).
        # A NaN-skipping "first" would skip leading NaNs and return the next
        # trial's context, leaking future state into events whose first
        # timebin had no trial active. The stop-table builder already
        # uses the strict semantics for the same hazard — match it
        # here so trial-derived event-level fields propagate NaN when
        # the event begins before any trial.
        prev_arm = single_arm_from_well(strict_first(g["prev_well"]), track_graph)
        next_arm = single_arm_from_well(strict_first(g["next_well"]), track_graph)
        last_rewarded_arm = single_arm_from_well(
            strict_first(g["last_rewarded_well"]), track_graph
        )

        is_after_switch = strict_first(g["did_last_completed_trip_switch_patch"])
        # reward_outcome and is_after_reward are the same source array; persist
        # one canonical is_rewarded_outcome. Stay/omission complements derived
        # downstream, not persisted.
        is_rewarded_outcome = strict_first(g["is_rewarded_outcome"])

        # ContFrag-specific: state composition during event. Defer to
        # ``_summarize_event_state`` for NaN-safe semantics — the
        # comparison ``cont_prob > frag_prob`` and ``cont_mean >=
        # frag_mean`` silently route NaN-side missingness into a
        # confident "Fragmented / 0% Continuous" label, which has
        # tripped both the all-NaN and the asymmetric (one state's
        # probabilities all NaN) cases. The helper requires both
        # states to have at least one finite bin before producing a
        # label / fraction.
        cont_mean, frag_mean, dominant, cont_frac = _summarize_event_state(
            g["Continuous_prob"].to_numpy(),
            g["Fragmented_prob"].to_numpy(),
        )

        midpoint_idx = len(times) // 2
        midpoint_time = float(times[midpoint_idx])
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

        record = {
            event_col: int(ev_id),
            "event_start_s": float(times[0]),
            "event_end_s": float(times[-1]),
            "event_duration_s": float(
                times[-1] - times[0] + 1.0 / DEFAULT_SAMPLING_FREQUENCY_HZ
            ),
            "is_swr": event_col == "SWR_event"
            or (animal, day, epoch, int(ev_id)) in other_overlap_keys,
            "is_mua": event_col == "MUA_event"
            or (animal, day, epoch, int(ev_id)) in other_overlap_keys,
            "continuous_prob_mean": cont_mean,
            "fragmented_prob_mean": frag_mean,
            "dominant_state": dominant,
            "continuous_fraction": cont_frac,
            "occupancy_state": location["occupancy_state"],
            "at_well": location["at_well"],
            "occupancy_well_id": location_well,
            "modal_observed_track_segment_id": physical_arm,
            "occupancy_well_arm": current_arm,
            "prev_arm": prev_arm,
            "next_arm": next_arm,
            "last_rewarded_arm": last_rewarded_arm,
            "current_patch": arm_to_patch(current_arm),
            "prev_patch": arm_to_patch(prev_arm),
            "next_patch": arm_to_patch(next_arm),
            "did_last_completed_trip_switch_patch": is_after_switch,
            "is_rewarded_outcome": is_rewarded_outcome,
            "is_stay": stay_indicator(next_arm, current_arm),
            "n_timebins": len(g),
            # Counts bins with no real-arm ``decoded_arm``; since
            # ``map_estimate_to_arm`` never returns a junction, that means
            # undecoded (NaN ``map_estimate``) bins, not junction MAPs.
            "n_undecoded_bins": n_junction,
            "n_unique_arms": int(arms_real.nunique()),
            "event_midpoint_s": midpoint_time,
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
        event_records.append(record)

    per_event = pd.DataFrame.from_records(event_records)
    per_event["at_well"] = pd.array(per_event["at_well"], dtype="boolean")

    # time_since_stop per event
    per_timebin_tss = time_since_stop(ev.index.to_numpy(), bout_starts)
    tss_by_event = (
        pd.Series(per_timebin_tss, index=ev.index)
        .groupby(ev[event_col].to_numpy(), sort=True)
        .median()
        .to_numpy()
    )
    per_event["time_since_stop_start_s"] = tss_by_event

    # Posterior-dependent columns
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
        post_vals = align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(
            post_in_event, edge_id, n_arms=len(ALL_ARMS)
        )
        # For ContFrag, weight = 1.0 for all timebins (both states decode)
        weights = np.ones(len(post_in_event))
        _, peak = posterior_peak_mean(arm_posterior, weights, event_id_series)
        _, entropy = posterior_entropy_mean(arm_posterior, weights, event_id_series)
        _, hpd = posterior_hpd_widths(
            arm_posterior, weights, event_id_series, coverages=(0.5, 0.8, 0.95)
        )
        per_event["posterior_peak_mean"] = peak
        per_event["posterior_entropy_mean"] = entropy
        per_event["posterior_hpd_arm_count_mean_coverage_50pct"] = hpd[0.5]
        per_event["posterior_hpd_arm_count_mean_coverage_80pct"] = hpd[0.8]
        per_event["posterior_hpd_arm_count_mean_coverage_95pct"] = hpd[0.95]

    # Trajectory rollups (shared with the Non-Local table). Gated on the
    # augmented data_over_time columns; continuous/fragmented_prob_mean come
    # from ``_summarize_event_state`` above, so they are excluded here.
    if {"step_distance", "run_speed_baseline"}.issubset(ev.columns):
        rollups = compute_trajectory_rollups(
            nl=ev,
            event_id=event_id_series,
            env=env,
            track_graph=track_graph,
            state_prob_cols=_CONTFRAG_STATE_COLS,
            continuous_col="Continuous_prob",
            fragmented_col="Fragmented_prob",
            place_bin_size=place_bin_size,
            nonlocal_prob=None,
            arm_posterior=arm_posterior,
        )
        for col_name in _CONTFRAG_TRAJECTORY_COLUMNS:
            per_event[col_name] = rollups[col_name]
    else:
        for col_name in _CONTFRAG_TRAJECTORY_COLUMNS:
            per_event[col_name] = np.nan

    # Identifiers + stable hashes
    per_event.insert(0, "animal", animal)
    per_event.insert(1, "session_date", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_source"] = event_source
    per_event["brain_area"] = brain_area
    per_event["event_id"] = _hash_contfrag_event_id(per_event, event_source, brain_area)

    # stop_id
    stop_idx = assign_bout(
        per_event["event_start_s"].to_numpy(), bout_starts, bout_ends
    )
    # The whole-event immobility filter guarantees every retained event sits
    # inside a bout, so a -1 here means that invariant broke upstream. Hashing
    # it would mint a stop_id matching no stop row -- an event silently orphaned
    # from every stop-keyed join. The NonLocal builder raises; match it.
    n_outside = int((stop_idx < 0).sum())
    if n_outside > 0:
        raise ValueError(
            f"{n_outside}/{len(stop_idx)} {event_source} events fall outside "
            "all immobility bouts"
        )
    per_event["stop_idx"] = stop_idx
    per_event["stop_id"] = hash_stop_id(
        animal,
        day,
        epoch,
        stop_idx,
        brain_area=brain_area,
        event_source=event_source,
    )

    return stamp_covariate_schema(per_event[column_order], table="event")


def build_contfrag_event_arm_table(
    event_table: pd.DataFrame,
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    event_source: str = "SWR",
    brain_area: str = "HPC",  # noqa: ARG001 — passed for API symmetry; event_id is inherited
    conditional_posterior=None,
    contingency: pd.DataFrame | None = None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """One row per (event, arm) for externally-defined events.

    Parameters
    ----------
    event_table : pd.DataFrame
        Output of ``build_contfrag_event_table()``.
    dot, env, trials, port_experiences, track_graph
        Same as ``build_contfrag_event_table()``.
    event_source : {"SWR", "MUA"}
        Which external event detector defines event boundaries.
    conditional_posterior : xr.DataArray, optional
        Combined posterior for arm_posterior_mass.
    speed_threshold : float
        Head-speed threshold for immobility; only the canonical value.

    Returns
    -------
    pd.DataFrame
        Columns: see ``_contfrag_event_arm_table_columns()``.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``.
    """
    require_canonical_speed_threshold(speed_threshold, "build_contfrag_event_arm_table")
    event_col = _resolve_event_col(event_source)
    canonical_column_order = _contfrag_event_arm_table_build_columns(event_source)
    if len(event_table) == 0:
        return stamp_covariate_schema(
            pd.DataFrame(columns=canonical_column_order), table="event_arm"
        )

    n_arms = len(ALL_ARMS)
    n_events = len(event_table)

    dot = _prepare_contfrag_dot(dot, env)
    in_event = _in_contfrag_event_mask(dot, speed_threshold, event_source)
    ev = dot.loc[in_event]

    # Arm counts — weight = 1.0 for all timebins (both states decode position)
    arm_series = ev["decoded_arm"]
    real_mask = arm_series.isin(ALL_ARMS).to_numpy()
    arm_map_real = arm_series.to_numpy()[real_mask].astype(np.int64)
    event_id_real = ev[event_col].to_numpy()[real_mask]
    weights_real = np.ones(real_mask.sum())
    weights_all = np.ones(len(ev))
    event_id_all = ev[event_col].to_numpy()

    assert_single_session_alignment(
        dot,
        event_table,
        source_name=f"ContFrag {event_source} dot",
        target_session_cols=("animal", "session_date", "epoch"),
    )
    assert_event_label_alignment(
        event_id_all,
        event_table[event_col].to_numpy(),
        source_name=f"ContFrag {event_source} qualified bins",
    )

    ids_real, counts_hard = arm_count_hard(
        weights_real, arm_map_real, event_id_real, n_arms=n_arms, threshold=0.0
    )
    _, counts_soft = arm_count_soft(
        weights_real, arm_map_real, event_id_real, n_arms=n_arms
    )

    ev_ids = event_table[event_col].to_numpy().astype(np.int64)
    hard_aligned = align_count_matrix(counts_hard, ids_real, ev_ids, n_arms)
    soft_aligned = align_count_matrix(counts_soft, ids_real, ev_ids, n_arms)

    if conditional_posterior is not None:
        edge_id = np.asarray(env.place_bin_centers_nodes_df_["edge_id"])
        post_vals = align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(post_in_event, edge_id, n_arms=n_arms)
        ids_all, mass = arm_posterior_mass(weights_all, arm_posterior, event_id_all)
        mass_aligned = align_count_matrix(mass, ids_all, ev_ids, n_arms)
    else:
        mass_aligned = np.full((n_events, n_arms), np.nan)

    # Broadcast event-level fields to 9-per-event
    long = broadcast_candidate_arms(event_table)
    long["arm_count_hard"] = hard_aligned.flatten()
    long["arm_count_soft"] = soft_aligned.flatten()
    long["arm_posterior_mass"] = mass_aligned.flatten()

    # arm_fraction. Use the count of real-arm bins as the denominator
    # so the per-arm fractions sum to 1 over an event's real bins,
    # matching the non-local table convention. The numerator
    # (``arm_count_hard``) only counts bins whose decoded arm is in
    # ``ALL_ARMS``; using ``n_timebins`` as the denominator (which
    # also includes junction / missing-arm bins) silently shrinks the
    # fraction sum below 1 — the user verified a 5-real / 5-junction
    # event sums to 0.5.
    n_real_long = long["n_timebins"].to_numpy() - long["n_undecoded_bins"].to_numpy()
    long["arm_fraction"] = arm_fraction_from_event_total(
        long["arm_count_hard"].to_numpy(),
        n_real_long,
    )
    long["arm_fraction_posterior"] = arm_fraction_from_event_total(
        long["arm_posterior_mass"].to_numpy(),
        long["n_timebins"].to_numpy(dtype=float),
    )

    # ContFrag stop IDs include event-source/brain-area provenance, whereas the
    # canonical qvals table is keyed by the plain Replay bout identity.
    contingency_stop_id = hash_stop_id(
        long["animal"].to_numpy(),
        long["session_date"].to_numpy(),
        long["epoch"].to_numpy(),
        long["stop_idx"].to_numpy(),
    )
    long = attach_candidate_arm_covariates(
        long,
        event_table=event_table,
        port_experiences=port_experiences,
        track_graph=track_graph,
        contingency_stop_id=contingency_stop_id,
        contingency=contingency,
        include_belief_variance=False,
    )

    # ContFrag-specific interactions
    cont_mean = long["continuous_prob_mean"].to_numpy()
    long["continuous_replay_score_mean_x_is_current_arm"] = (
        cont_mean * long["is_current_arm"].to_numpy()
    )
    long["continuous_replay_score_mean_x_is_prev_arm"] = (
        cont_mean * long["is_prev_arm"].to_numpy()
    )
    long["continuous_replay_score_mean_x_is_next_arm"] = (
        cont_mean * long["is_next_arm"].to_numpy()
    )
    long["continuous_x_last_rewarded"] = (
        cont_mean * long["is_last_rewarded_arm"].to_numpy()
    )

    validate_event_arm_table(long, event_table=event_table)
    canonical = stamp_covariate_schema(long, table="event_arm")
    return canonical[canonical_column_order]


def build_contfrag_stop_table(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    event_source: str = "SWR",
    brain_area: str = "HPC",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """One row per immobility bout, including zero-event bouts.

    The ``stop_id`` hash matches ``build_contfrag_event_table``'s
    ``stop_id`` row-for-row, so the two tables join cleanly.
    ``immobility_duration_s`` is the stop's span less the excluded time
    inside it, and the stop's location, segment and distance summaries read
    only its rows outside excluded time, as in the NonLocal stop table (one
    helper, ``bouts.summarize_bout_location``).

    The table carries every column the shared per-stop rate ladder
    (``MODEL_PER_STOP_RATE``) reads: the reward-history basis
    ``reward_history_rcos_0..2`` and its flag ``reward_history_incomplete``,
    computed at bout start by ``reward_history_basis_features`` exactly as in
    the NonLocal stop table. The NonLocal table's other columns
    (``known_reward_outcome_lag_*``, the stop clocks, the per-detector counts
    ``n_swr_events``, ``n_mua_events``, ``n_nonlocal_swr_events`` and
    ``n_nonlocal_non_swr_events``, and ``completed_trial_reward``) are not built
    here; no ContFrag consumer reads them.

    Parameters
    ----------
    dot : pd.DataFrame
        ContFrag data-over-time frame.
    trials : pd.DataFrame
        Trial frame from Spyglass.
    port_experiences : pd.DataFrame
        Complete starting-port-plus-destinations occupancy stream. Reward
        covariates use only its physically observed outcomes.
    track_graph : nx.Graph
        Session track graph.
    event_source : {"SWR", "MUA"}
        Which external detector defines events.
    speed_threshold : float
        Head-speed threshold for immobility; only the canonical value.

    Returns
    -------
    pd.DataFrame
        Columns: see ``_CONTFRAG_STOP_TABLE_COLUMNS``.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``.
    """
    require_canonical_speed_threshold(speed_threshold, "build_contfrag_stop_table")
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    event_col = _resolve_event_col(event_source)

    # add_trial_context writes its columns into the frame it is given.
    dot = dot.copy(deep=False)
    dot = add_trial_context(dot, trials, port_experiences)
    dot["nearest_well_distance_cm"] = nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )
    column_order = list(_CONTFRAG_STOP_TABLE_BUILD_COLUMNS)
    if bout_starts.size == 0:
        return stamp_covariate_schema(pd.DataFrame(columns=column_order), table="stop")

    n_bouts = bout_starts.size
    times = dot.index.to_numpy()
    row_bout_idx = assign_bout(times, bout_starts, bout_ends)

    in_bout = row_bout_idx >= 0
    bout_rows = dot.loc[in_bout].assign(_bout_idx=row_bout_idx[in_bout])
    grouped = bout_rows.groupby("_bout_idx", sort=True)

    per_bout = pd.DataFrame({"stop_idx": np.arange(n_bouts, dtype=np.int64)})
    per_bout["stop_start"] = bout_starts
    per_bout["stop_end"] = bout_ends
    # The stop's exposure: its span less the excluded time inside it. The
    # location summaries below read the same intervals.
    excluded = excluded_intervals_of(dot)
    per_bout["immobility_duration_s"] = bout_exposure_s(
        bout_starts, bout_ends, excluded
    )
    per_bout["reward_fraction_last10_known_trials"] = recent_trial_reward_rate(
        bout_starts, port_experiences
    )
    # The reward-history basis the per-stop rate ladder reads (R1), built at
    # bout start by the NonLocal stop table's helper so the two tables agree.
    reward_history, reward_history_incomplete = reward_history_basis_features(
        bout_starts, port_experiences
    )
    for j, column in enumerate(REWARD_HISTORY_COLUMNS):
        per_bout[column] = reward_history[:, j]
    per_bout[REWARD_HISTORY_INCOMPLETE] = reward_history_incomplete

    per_bout["is_swr_stop"] = (
        grouped["SWR_event"]
        .apply(lambda s: (s > 0).any())
        .reindex(np.arange(n_bouts), fill_value=False)
        .to_numpy()
    )
    if "MUA_event" in dot.columns:
        per_bout["is_mua_stop"] = (
            grouped["MUA_event"]
            .apply(lambda s: (s > 0).any())
            .reindex(np.arange(n_bouts), fill_value=False)
            .to_numpy()
        )
    else:
        per_bout["is_mua_stop"] = False

    # n_selected_events: count each included event once, in the bout of its
    # event_start — mirroring build_contfrag_event_table
    # (``stop_idx = assign_bout(event_start)``) and using the SAME
    # all-immobile inclusion mask, so the population matches the event
    # table and ``Σ stop.n_selected_events == len(event_table)`` holds by
    # construction. (All-immobile events cannot straddle bouts, but
    # counting by event_start keeps the two tables aligned regardless.)
    in_event = _in_contfrag_event_mask(dot, speed_threshold, event_source)
    event_rows = dot.loc[in_event]
    if len(event_rows) == 0:
        n_events_per_bout = np.zeros(n_bouts, dtype=np.int64)
    else:
        event_start_per_event = (
            event_rows.reset_index(names="_event_time")
            .groupby(event_col)["_event_time"]
            .min()
            .to_numpy()
        )
        event_bout = assign_bout(event_start_per_event, bout_starts, bout_ends)
        n_events_per_bout = (
            pd.Series(event_bout[event_bout >= 0])
            .value_counts()
            .reindex(np.arange(n_bouts), fill_value=0)
            .astype(np.int64)
            .to_numpy()
        )
    per_bout["has_replay"] = n_events_per_bout > 0
    per_bout["n_selected_events"] = n_events_per_bout

    # Trial context at bout start. ``skipna=False`` is critical:
    # the default ``first()`` skips leading NaNs and surfaces the
    # first valid mid-bout row, leaking trial context into stops
    # that span a trial boundary (the post-boundary trial's
    # a later finite ``is_rewarded_outcome`` would attach to a bout whose
    # actual start had no observable outcome).
    bout_first_row = grouped.first(skipna=False)
    for col in (
        "did_last_completed_trip_switch_patch",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    ):
        per_bout[col] = bout_first_row[col].reindex(np.arange(n_bouts)).to_numpy()

    # reward_outcome / is_after_reward are one source array; persist one
    # canonical is_rewarded_outcome (stay/omission derived downstream).
    per_bout["is_rewarded_outcome"] = (
        bout_first_row["is_rewarded_outcome"].reindex(np.arange(n_bouts)).to_numpy()
    )

    per_bout["completed_trial_id"] = pd.array(
        bout_first_row["completed_trial_id"].reindex(np.arange(n_bouts)),
        dtype="Int64",
    )

    # Behavioral location, modal physical segment and distances over the
    # bout's rows outside excluded time, as in the NonLocal stop table (one
    # helper). The at-well occupancy arm (occupancy_well_arm) is NaN off a
    # single well.
    location = summarize_bout_location(bout_rows, n_bouts, track_graph, excluded)
    for column in location.columns:
        per_bout[column] = location[column]
    current_arm_vals = location["occupancy_well_arm"].to_numpy()

    # The contextual wells at the bout's first row: the state at the stop's
    # start, not a summary over its bins, so read whether or not that row is
    # in excluded time.
    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    completed_trial_arm_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        prev_arm_vals[bi] = single_arm_from_well(
            strict_first(g["prev_well"]), track_graph
        )
        next_arm_vals[bi] = single_arm_from_well(
            strict_first(g["next_well"]), track_graph
        )
        last_rew_vals[bi] = single_arm_from_well(
            strict_first(g["last_rewarded_well"]), track_graph
        )
        completed_trial_arm_vals[bi] = single_arm_from_well(
            strict_first(g["last_visited_well"]), track_graph
        )

    per_bout["prev_arm"] = prev_arm_vals
    per_bout["next_arm"] = next_arm_vals
    per_bout["last_rewarded_arm"] = last_rew_vals
    per_bout["completed_trial_arm"] = completed_trial_arm_vals
    cur_patch = np.array([arm_to_patch(a) for a in current_arm_vals])
    nxt_patch = np.array([arm_to_patch(a) for a in next_arm_vals])
    per_bout["current_patch"] = cur_patch
    per_bout["prev_patch"] = [arm_to_patch(a) for a in prev_arm_vals]
    per_bout["next_patch"] = nxt_patch
    per_bout["is_stay"] = stay_indicator_from_patches(nxt_patch, cur_patch)

    per_bout.insert(0, "animal", animal)
    per_bout.insert(1, "session_date", day)
    per_bout.insert(2, "epoch", epoch)
    per_bout["event_source"] = event_source
    per_bout["brain_area"] = brain_area
    per_bout["stop_id"] = hash_stop_id(
        animal,
        day,
        epoch,
        per_bout["stop_idx"].to_numpy(),
        brain_area=brain_area,
        event_source=event_source,
    )

    return stamp_covariate_schema(per_bout[column_order], table="stop")


# ── Column definitions ──


def _contfrag_event_table_build_columns(event_source: str = "SWR") -> list[str]:
    """Internal pre-migration column order for a ContFrag event table."""
    event_col = _resolve_event_col(event_source)
    return [
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
        event_col,
        "event_id",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "event_midpoint_s",
        "time_since_stop_start_s",
        "time_since_last_well_arrival_s",
        "time_since_last_reward_s",
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "modal_observed_track_segment_id",
        "nearest_well_distance_cm_min",
        "nearest_well_distance_cm_mean",
        "nearest_well_distance_cm_max",
        "is_swr",
        "is_mua",
        "continuous_prob_mean",
        "fragmented_prob_mean",
        "dominant_state",
        "continuous_fraction",
        "did_last_completed_trip_switch_patch",
        "is_rewarded_outcome",
        "is_stay",
        "occupancy_well_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "n_timebins",
        "posterior_peak_mean",
        "posterior_entropy_mean",
        "posterior_hpd_arm_count_mean_coverage_50pct",
        "posterior_hpd_arm_count_mean_coverage_80pct",
        "posterior_hpd_arm_count_mean_coverage_95pct",
        "n_undecoded_bins",
        "n_unique_arms",
        "stop_idx",
        "stop_id",
        # Trajectory rollups shared with the Non-Local table (continuous /
        # fragmented_prob_mean are above, from _summarize_event_state).
        *_CONTFRAG_TRAJECTORY_COLUMNS,
    ]


def _contfrag_event_table_columns(event_source: str = "SWR") -> list[str]:
    """Canonical persisted column order for a ContFrag event table."""
    return list(
        stamp_covariate_schema(
            pd.DataFrame(columns=_contfrag_event_table_build_columns(event_source)),
            table="event",
        ).columns
    )


#: Canonical column order for ContFrag stop table.
_CONTFRAG_STOP_TABLE_BUILD_COLUMNS: tuple[str, ...] = (
    "animal",
    "session_date",
    "epoch",
    "event_source",
    "brain_area",
    "stop_idx",
    "stop_id",
    "stop_start",
    "stop_end",
    "immobility_duration_s",
    "reward_fraction_last10_known_trials",
    "is_swr_stop",
    "is_mua_stop",
    "has_replay",
    "n_selected_events",
    "did_last_completed_trip_switch_patch",
    "is_rewarded_outcome",
    "completed_trial_id",
    "completed_trial_arm",
    "patch_switch_fraction_last5_completed_trips",
    "leaky_q_value_diff",
    "is_stay",
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "occupancy_well_arm",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
    # After the trial context, as in the NonLocal stop table's order.
    *REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)

_CONTFRAG_STOP_TABLE_COLUMNS: tuple[str, ...] = tuple(
    stamp_covariate_schema(
        pd.DataFrame(columns=_CONTFRAG_STOP_TABLE_BUILD_COLUMNS), table="stop"
    ).columns
)


def _contfrag_event_arm_table_build_columns(event_source: str = "SWR") -> list[str]:
    """Internal pre-migration column order for a ContFrag event-arm table."""
    event_col = _resolve_event_col(event_source)
    return [
        # Identifiers
        "animal",
        "session_date",
        "epoch",
        "event_source",
        "brain_area",
        event_col,
        "event_id",
        "stop_idx",
        "stop_id",
        # Event-level context
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "time_since_stop_start_s",
        "time_since_last_well_arrival_s",
        "time_since_last_reward_s",
        "occupancy_state",
        "at_well",
        "occupancy_well_id",
        "modal_observed_track_segment_id",
        "nearest_well_distance_cm_min",
        "nearest_well_distance_cm_mean",
        "nearest_well_distance_cm_max",
        "is_swr",
        "is_mua",
        "continuous_prob_mean",
        "fragmented_prob_mean",
        "dominant_state",
        "continuous_fraction",
        "did_last_completed_trip_switch_patch",
        "is_rewarded_outcome",
        "is_stay",
        "occupancy_well_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "n_timebins",
        # Per-arm identity
        "candidate_arm",
        "arm_patch_id",
        # Content responses
        "arm_count_hard",
        "arm_count_soft",
        "arm_posterior_mass",
        "arm_fraction",
        "arm_fraction_posterior",
        # Binary indicators
        "is_current_arm",
        "is_prev_arm",
        "is_next_arm",
        "is_last_rewarded_arm",
        "is_last_rewarded_noncurrent_arm",
        "is_same_patch_as_current",
        # Spatial
        "distance_to_candidate_arm_cm",
        "candidate_arm_distance_rank",
        # Reward / value
        "visit_recency_trials",
        "visit_recency_trials_event_leaf_centered",
        "visit_recency_trials_event_leaf_centered_spline_1",
        "visit_recency_trials_event_leaf_centered_spline_2",
        "visit_recency_trials_event_leaf_centered_spline_3",
        "is_candidate_arm_never_visited",
        "reward_count_candidate_arm_last10_known_trials",
        "reward_fraction_candidate_arm_last10_visits",
        "task_reward_probability_candidate_arm",
        "reward_belief_mean_candidate_arm_preoutcome",
        # Interactions
        "is_current_arm_x_is_last_rewarded_arm",
        "time_since_stop_start_s_x_is_current_arm",
        "time_since_stop_start_s_x_is_prev_arm",
        "time_since_stop_start_s_x_is_next_arm",
        "swr_x_is_current_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_next_arm",
        "swr_x_is_last_rewarded_arm",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
        "swr_x_visit_recency_trials_event_leaf_centered",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_1",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_2",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_3",
        "swr_x_is_candidate_arm_never_visited",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "mua_x_is_current_arm",
        "mua_x_is_prev_arm",
        "mua_x_is_next_arm",
        "mua_x_is_last_rewarded_arm",
        "is_omitted_outcome_x_is_last_rewarded_arm",
        # ContFrag-specific interactions
        "continuous_replay_score_mean_x_is_current_arm",
        "continuous_replay_score_mean_x_is_prev_arm",
        "continuous_replay_score_mean_x_is_next_arm",
        "continuous_x_last_rewarded",
    ]


def _contfrag_event_arm_table_columns(event_source: str = "SWR") -> list[str]:
    """Canonical persisted column order for a ContFrag event-arm table."""
    return list(
        stamp_covariate_schema(
            pd.DataFrame(columns=_contfrag_event_arm_table_build_columns(event_source)),
            table="event_arm",
        ).columns
    )
