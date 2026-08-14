"""Canonical per-session tables for ContFrag replay content analysis.

This module produces event tables from externally-defined events (SWR or
high-MUA times), using the ContFrag classifier's decoded posterior for
content analysis. The key difference from the NonLocal tables
(``analysis.replay.tables``) is that events come from external detectors,
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
)
from continuum_swr_replay.analysis.hippocampus.event_detection import (
    event_strict_overlap_keys,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
    whole_event_immobility_mask,
)

from ...data_loaders.constants import ALL_ARMS
from ...parameters import PLACE_BIN_SIZE_1D
from ..replay._helpers import (
    add_trial_context,
    align_count_matrix,
    align_posterior_to_dot,
    arm_to_patch,
    assign_bout,
    attach_contingency_value_block,
    build_well_to_arm_map,
    eq_indicator,
    eq_indicator_or_nan,
    hash_stop_id,
    immobility_bout_intervals,
    nearest_well_distance_cm,
    recent_trial_reward_rate,
    safe_mode,
    single_arm_from_well,
    strict_first,
    summarize_behavioral_location,
    time_since_stop,
)
from ..replay.covariates import (
    center_visit_recency_within_event,
    cumulative_reward_all_events,
    recent_reward_rate_all_events,
    visit_recency_and_never_visited_all_events,
)
from ..replay.spatial import arm_dist_rank, dist_to_arm_from_well
from ..replay.spatial_mapping import arm_to_well, map_estimate_to_arm
from ..replay.summaries import (
    arm_count_hard,
    arm_count_soft,
    arm_posterior_mass,
    posterior_to_arm_marginal,
)
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
    break ``Σ stop.n_events == len(event_table)`` — is now the shared rule
    for every event source. The degenerate multi-second ``MUA_event``
    labels in ``docs/known_issues/degenerate_mua_events.md`` are the
    motivating case.

    On a frame that already went through ``_filter_contfrag_event_labels``
    or a speed-eligible cohort builder this is a no-op; it excludes
    nothing further. The historical ~0.5% SWR / ~0.7% MUA exclusion
    happens upstream at cohort construction.
    """
    event_col = _resolve_event_col(event_source)
    return whole_event_immobility_mask(
        dot,
        event_col,
        speed_threshold=speed_threshold,
    )


def _filter_contfrag_event_labels(
    dot: pd.DataFrame,
    speed_threshold: float,
) -> pd.DataFrame:
    """Apply the same whole-event rule to both external detector labels."""
    event_cols = [column for column in ("SWR_event", "MUA_event") if column in dot]
    return filter_event_labels_by_speed(
        dot,
        event_cols,
        speed_threshold=speed_threshold,
    )


def _hash_contfrag_event_id(
    events: pd.DataFrame,
    event_source: str = "SWR",
    brain_area: str = "HPC",
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, event_label)``.

    ``event_source`` and ``brain_area`` are mixed into the hash so a
    SWR/HPC event with label 7 and a MUA/HPC event with label 7 (or
    SWR/mPFC) cannot collide. Without this, concatenating tables
    across sources or brain areas produces silent ID collisions on
    every join keyed by ``event_id``.
    """
    event_col = _resolve_event_col(event_source)
    keys = events[["animal", "day", "epoch", event_col]].copy()
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
    track_graph : nx.Graph
        Session track graph.
    event_source : {"SWR", "MUA"}
        Which external event detector to use for event boundaries.
    conditional_posterior : xr.DataArray, optional
        Combined posterior for posterior-dependent columns.
    speed_threshold : float
        Head-speed threshold for immobility (cm/s).

    Returns
    -------
    pd.DataFrame
        One row per event. Includes all standard event-table columns
        plus ContFrag-specific: ``continuous_prob_mean``,
        ``fragmented_prob_mean``, ``dominant_state`` (ties go to
        Continuous), ``continuous_fraction``.
    """
    if "animal" not in dot.columns:
        raise KeyError(
            "dot must carry 'animal' column (set by get_contfrag_data_over_time)"
        )
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    event_col = _resolve_event_col(event_source)

    dot = _prepare_contfrag_dot(dot, env)
    dot = _filter_contfrag_event_labels(dot, speed_threshold)
    other_event_col = "MUA_event" if event_col == "SWR_event" else "SWR_event"
    other_overlap_keys = (
        event_strict_overlap_keys(dot, event_col, other_event_col)
        if other_event_col in dot.columns
        else set()
    )
    dot = add_trial_context(dot, trials)
    dot["nearest_well_distance_cm"] = nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )

    in_event = _in_contfrag_event_mask(dot, speed_threshold, event_source)
    ev = dot.loc[in_event]

    column_order = _contfrag_event_table_columns(event_source)
    if len(ev) == 0:
        return pd.DataFrame(columns=column_order)

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
        location = summarize_behavioral_location(g[["location_state", "location_well"]])
        location_well = float(location["location_well"])
        location_arm = single_arm_from_well(location_well, track_graph)
        current_arm = location_arm
        # Trial-derived context columns are NaN at every dot timebin
        # before the first trial start (see ``add_trial_context``).
        # ``safe_first`` would skip leading NaNs and return the next
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

        is_after_switch = strict_first(g["is_after_switch"])
        is_after_omission = strict_first(g["is_after_omission"])
        is_after_reward = strict_first(g["is_after_reward"])
        reward_outcome = strict_first(g["reward_outcome"])
        is_after_stay = (
            1.0 - is_after_switch if not np.isnan(is_after_switch) else np.nan
        )

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
            "event_start": float(times[0]),
            "event_end": float(times[-1]),
            "event_duration": float(times[-1] - times[0]),
            "is_swr": event_col == "SWR_event"
            or (animal, day, epoch, int(ev_id)) in other_overlap_keys,
            "is_mua": event_col == "MUA_event"
            or (animal, day, epoch, int(ev_id)) in other_overlap_keys,
            "continuous_prob_mean": cont_mean,
            "fragmented_prob_mean": frag_mean,
            "dominant_state": dominant,
            "continuous_fraction": cont_frac,
            "location_state": location["location_state"],
            "at_well": location["at_well"],
            "location_well": location_well,
            "location_arm": location_arm,
            "physical_arm": physical_arm,
            "current_arm": current_arm,
            "prev_arm": prev_arm,
            "next_arm": next_arm,
            "last_rewarded_arm": last_rewarded_arm,
            "current_patch": arm_to_patch(current_arm),
            "prev_patch": arm_to_patch(prev_arm),
            "next_patch": arm_to_patch(next_arm),
            "is_after_switch": is_after_switch,
            "is_after_omission": is_after_omission,
            "is_after_reward": is_after_reward,
            "is_after_stay": is_after_stay,
            "reward_outcome": reward_outcome,
            "is_stay": float(arm_to_patch(next_arm) == arm_to_patch(current_arm))
            if not (
                np.isnan(arm_to_patch(next_arm)) or np.isnan(arm_to_patch(current_arm))
            )
            else np.nan,
            "n_timebins": len(g),
            # Counts bins with no real-arm ``decoded_arm``; since
            # ``map_estimate_to_arm`` never returns a junction, that means
            # undecoded (NaN ``map_estimate``) bins, not junction MAPs.
            "n_junction_bins": n_junction,
            "n_unique_arms": int(arms_real.nunique()),
            "event_midpoint_time": midpoint_time,
            "time_since_port_entry": time_since_last_port_entry,
            "time_since_last_port_entry": time_since_last_port_entry,
            "time_since_reward": float(context_time - g["_last_reward_time"].iloc[0])
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
    per_event["time_since_stop"] = tss_by_event

    # Posterior-dependent columns
    posterior_cols = (
        "posterior_peak_mean",
        "posterior_entropy_mean",
        "posterior_hpd_width_50",
        "posterior_hpd_width_80",
        "posterior_hpd_width_95",
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
        per_event["posterior_hpd_width_50"] = hpd[0.5]
        per_event["posterior_hpd_width_80"] = hpd[0.8]
        per_event["posterior_hpd_width_95"] = hpd[0.95]

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
    per_event.insert(1, "day", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_source"] = event_source
    per_event["brain_area"] = brain_area
    per_event["event_id"] = _hash_contfrag_event_id(per_event, event_source, brain_area)

    # stop_id
    stop_idx = assign_bout(per_event["event_start"].to_numpy(), bout_starts, bout_ends)
    per_event["stop_idx"] = stop_idx
    per_event["stop_id"] = hash_stop_id(
        animal,
        day,
        epoch,
        stop_idx,
        brain_area=brain_area,
        event_source=event_source,
    )

    return per_event[column_order]


def build_contfrag_event_arm_table(
    event_table: pd.DataFrame,
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
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
    dot, env, trials, track_graph
        Same as ``build_contfrag_event_table()``.
    event_source : {"SWR", "MUA"}
        Which external event detector defines event boundaries.
    conditional_posterior : xr.DataArray, optional
        Combined posterior for arm_posterior_mass.
    speed_threshold : float
        Head-speed threshold for immobility.

    Returns
    -------
    pd.DataFrame
        Columns: see ``_contfrag_event_arm_table_columns()``.
    """
    event_col = _resolve_event_col(event_source)
    column_order = _contfrag_event_arm_table_columns(event_source)
    if len(event_table) == 0:
        return pd.DataFrame(columns=column_order)

    n_arms = len(ALL_ARMS)
    n_events = len(event_table)

    dot = _prepare_contfrag_dot(dot, env)
    dot = _filter_contfrag_event_labels(dot, speed_threshold)
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
    long = event_table.loc[event_table.index.repeat(n_arms)].reset_index(drop=True)
    long["arm"] = np.tile(ALL_ARMS, n_events)
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
    n_real_long = long["n_timebins"].to_numpy() - long["n_junction_bins"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            n_real_long > 0,
            long["arm_count_hard"].to_numpy() / np.maximum(n_real_long, 1),
            np.nan,
        )
    long["arm_fraction"] = frac

    # Binary indicators. is_current / is_same_patch propagate NaN where
    # the animal's location is unobserved (event window not wholly at
    # one well); the rest treat an absent reference as a true zero.
    # Mirrors replay/tables.py -- see eq_indicator_or_nan.
    arm_arr = long["arm"].to_numpy()
    long["is_current"] = eq_indicator_or_nan(arm_arr, long["current_arm"].to_numpy())
    long["is_prev"] = eq_indicator(arm_arr, long["prev_arm"].to_numpy())
    long["is_next"] = eq_indicator(arm_arr, long["next_arm"].to_numpy())
    long["is_last_rewarded"] = eq_indicator(
        arm_arr, long["last_rewarded_arm"].to_numpy()
    )
    # M5/M6 identifiable reward-memory predictor: last-rewarded arm
    # restricted to events where it is NOT the current arm. Mirrors
    # replay/tables.py — see model_specs.py for the rationale.
    long["is_last_rewarded_elsewhere"] = long["is_last_rewarded"].to_numpy() * (
        1 - long["is_current"].to_numpy()
    )
    long["arm_patch_id"] = np.array([arm_to_patch(a) for a in arm_arr])
    long["is_same_patch"] = eq_indicator_or_nan(
        long["arm_patch_id"].to_numpy(), long["current_patch"].to_numpy()
    )

    # Per-event spatial predictors. Distances still depend on the event's
    # current location, but trial-history covariates are computed in batch
    # below so the full trial frame is not rescanned for every event.
    dist_vals = np.full(n_events * n_arms, np.nan)
    rank_vals = np.full(n_events * n_arms, np.nan)

    for pos, (_, ev_row) in enumerate(event_table.iterrows()):
        base = pos * n_arms
        current_arm_val = ev_row["current_arm"]
        current_well = (
            arm_to_well(current_arm_val, track_graph)
            if not np.isnan(current_arm_val)
            else np.nan
        )
        dists = dist_to_arm_from_well(current_well, track_graph)
        dist_arr = np.array([dists[a] for a in ALL_ARMS])
        ranks = arm_dist_rank(dist_arr)

        for j in range(n_arms):
            idx = base + j
            dist_vals[idx] = dist_arr[j]
            rank_vals[idx] = ranks[j]

    # Match the event-level context cutoff: every model-facing history
    # covariate is frozen at event start. A mixed event crossing goal arrival
    # must not combine pre-arrival outcome context with post-arrival history.
    event_times = event_table["event_start"].to_numpy(dtype=float)
    vr_2d = np.full((n_events, n_arms), np.nan)
    anv_2d = np.full((n_events, n_arms), np.nan)
    cr_2d = np.full((n_events, n_arms), np.nan)
    rrr_2d = np.full((n_events, n_arms), np.nan)
    query_mask = ~np.isnan(event_times)
    if len(trials) > 0 and query_mask.any():
        well_to_arm = build_well_to_arm_map(track_graph)
        query_times = event_times[query_mask]
        vr_2d[query_mask], anv_2d[query_mask] = (
            visit_recency_and_never_visited_all_events(query_times, trials, well_to_arm)
        )
        cr_2d[query_mask] = cumulative_reward_all_events(
            query_times, trials, well_to_arm
        )
        # Canonical per-arm reward rate: rewarded fraction of the last ten
        # visits to that arm (covariate_dictionary.md D1).
        rrr_2d[query_mask] = recent_reward_rate_all_events(
            query_times, trials, well_to_arm, k=10
        )

    vr_vals = vr_2d.reshape(-1)
    anv_vals = anv_2d.reshape(-1)
    cr_vals = cr_2d.reshape(-1)
    rrr_vals = rrr_2d.reshape(-1)

    long["dist_to_arm"] = dist_vals
    long["arm_dist_rank"] = rank_vals
    long["visit_recency"] = vr_vals
    long["arm_never_visited"] = anv_vals
    # Time-local relative recency among the six leaf arms. Centering within
    # each event prevents future events from changing earlier model rows.
    long["visit_recency_demeaned"] = center_visit_recency_within_event(
        vr_vals.reshape(n_events, n_arms), anv_vals.reshape(n_events, n_arms)
    ).reshape(-1)
    long["cumulative_reward"] = cr_vals
    long["reward_rate_arm_k10"] = rrr_vals

    # p_reward_arm / q_arm from the per-stop contingency artifact (shared with
    # replay/tables.py via attach_contingency_value_block). ContFrag's own
    # stop_id mixes in brain_area/event_source so SWR/MUA and per-area tables
    # never collide on it; the contingency artifact, however, is keyed by the
    # plain NonLocal-pipeline hash of (animal, day, epoch, stop_idx) shared
    # across pipelines — so a join on ContFrag's stop_id matches nothing.
    # Recompute the plain key from the natural keys (always present on the
    # broadcast event_table) for the merge.
    cont_key = hash_stop_id(
        long["animal"].to_numpy(),
        long["day"].to_numpy(),
        long["epoch"].to_numpy(),
        long["stop_idx"].to_numpy(),
    )
    long["p_reward_arm"], long["q_arm"] = attach_contingency_value_block(
        long["arm"].to_numpy(), cont_key, contingency
    )

    # Interactions
    tss = long["time_since_stop"].to_numpy()
    is_swr_i = long["is_swr"].astype(int).to_numpy()
    is_mua_i = long["is_mua"].astype(int).to_numpy()
    is_after_switch_arr = long["is_after_switch"].to_numpy()
    is_after_omission_arr = long["is_after_omission"].to_numpy()
    ic = long["is_current"].to_numpy()
    ip = long["is_prev"].to_numpy()
    inx = long["is_next"].to_numpy()
    ilr = long["is_last_rewarded"].to_numpy()

    long["current_x_last_rewarded"] = ic * ilr
    long["time_x_current"] = tss * ic
    long["time_x_prev"] = tss * ip
    long["time_x_next"] = tss * inx
    long["swr_x_current"] = is_swr_i * ic
    long["swr_x_prev"] = is_swr_i * ip
    long["swr_x_next"] = is_swr_i * inx
    long["swr_x_last_rewarded"] = is_swr_i * ilr
    # M7 — SWR × value/memory interactions (mirrors replay/tables.py).
    long["swr_x_reward_rate_arm_k10"] = (
        is_swr_i * long["reward_rate_arm_k10"].to_numpy()
    )
    long["swr_x_visit_recency_demeaned"] = (
        is_swr_i * long["visit_recency_demeaned"].to_numpy()
    )
    long["swr_x_arm_never_visited"] = is_swr_i * long["arm_never_visited"].to_numpy()
    long["swr_x_is_last_rewarded_elsewhere"] = (
        is_swr_i * long["is_last_rewarded_elsewhere"].to_numpy()
    )
    long["mua_x_current"] = is_mua_i * ic
    long["mua_x_prev"] = is_mua_i * ip
    long["mua_x_next"] = is_mua_i * inx
    long["mua_x_last_rewarded"] = is_mua_i * ilr
    long["switch_x_current"] = is_after_switch_arr * ic
    long["switch_x_prev"] = is_after_switch_arr * ip
    long["omission_x_last_rewarded"] = is_after_omission_arr * ilr

    # ContFrag-specific interactions
    cont_mean = long["continuous_prob_mean"].to_numpy()
    long["continuous_x_current"] = cont_mean * ic
    long["continuous_x_prev"] = cont_mean * ip
    long["continuous_x_next"] = cont_mean * inx
    long["continuous_x_last_rewarded"] = cont_mean * ilr

    return long[column_order]


def build_contfrag_stop_table(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    event_source: str = "SWR",
    brain_area: str = "HPC",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """One row per immobility bout, including zero-event bouts.

    The ``stop_id`` hash matches ``build_contfrag_event_table``'s
    ``stop_id`` row-for-row, so the two tables join cleanly.

    Parameters
    ----------
    dot : pd.DataFrame
        ContFrag data-over-time frame.
    trials : pd.DataFrame
        Trial frame from Spyglass.
    track_graph : nx.Graph
        Session track graph.
    event_source : {"SWR", "MUA"}
        Which external detector defines events.
    speed_threshold : float
        Head-speed threshold for immobility.

    Returns
    -------
    pd.DataFrame
        Columns: see ``_CONTFRAG_STOP_TABLE_COLUMNS``.
    """
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    event_col = _resolve_event_col(event_source)

    dot = _filter_contfrag_event_labels(dot, speed_threshold)
    dot = add_trial_context(dot, trials)
    dot["nearest_well_distance_cm"] = nearest_well_distance_cm(dot, track_graph)

    bout_starts, bout_ends = immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )
    column_order = list(_CONTFRAG_STOP_TABLE_COLUMNS)
    if bout_starts.size == 0:
        return pd.DataFrame(columns=column_order)

    n_bouts = bout_starts.size
    times = dot.index.to_numpy()
    row_bout_idx = assign_bout(times, bout_starts, bout_ends)

    in_bout = row_bout_idx >= 0
    bout_rows = dot.loc[in_bout].assign(_bout_idx=row_bout_idx[in_bout])
    grouped = bout_rows.groupby("_bout_idx", sort=True)

    per_bout = pd.DataFrame({"stop_idx": np.arange(n_bouts, dtype=np.int64)})
    per_bout["stop_start"] = bout_starts
    per_bout["stop_end"] = bout_ends
    per_bout["immobility_time"] = bout_ends - bout_starts
    per_bout["reward_rate_trial_k10"] = recent_trial_reward_rate(bout_starts, trials)

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

    # n_events: count each included event once, in the bout of its
    # event_start — mirroring build_contfrag_event_table
    # (``stop_idx = assign_bout(event_start)``) and using the SAME
    # all-immobile inclusion mask, so the population matches the event
    # table and ``Σ stop.n_events == len(event_table)`` holds by
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
    per_bout["n_events"] = n_events_per_bout

    # Trial context at bout start. ``skipna=False`` is critical:
    # the default ``first()`` skips leading NaNs and surfaces the
    # first valid mid-bout row, leaking trial context into stops
    # that span a trial boundary (the post-boundary trial's
    # ``is_after_reward`` / ``reward_outcome`` would attach to a
    # bout whose actual start had no trial active).
    bout_first_row = grouped.first(skipna=False)
    for col in (
        "is_after_switch",
        "is_after_omission",
        "is_after_reward",
        "is_after_stay",
        "reward_outcome",
        "recent_switch_rate",
        "leaky_q_value_diff",
    ):
        if col == "is_after_stay":
            stay = np.where(
                np.isnan(bout_first_row["is_after_switch"].to_numpy()),
                np.nan,
                1.0 - bout_first_row["is_after_switch"].to_numpy(),
            )
            per_bout["is_after_stay"] = (
                pd.Series(stay).reindex(np.arange(n_bouts)).to_numpy()
            )
        else:
            per_bout[col] = bout_first_row[col].reindex(np.arange(n_bouts)).to_numpy()

    per_bout["completed_trial_id"] = pd.array(
        bout_first_row["completed_trial_id"].reindex(np.arange(n_bouts)),
        dtype="Int64",
    )

    # Whole-window behavioral location plus modal physical segment. The legacy
    # current_arm name is a compatibility alias of strict location_arm.
    location_state_vals: list[str] = []
    at_well_vals: list[object] = []
    location_well_vals = np.full(n_bouts, np.nan)
    location_arm_vals = np.full(n_bouts, np.nan)
    physical_arm_vals = np.full(n_bouts, np.nan)
    nearest_min_vals = np.full(n_bouts, np.nan)
    nearest_mean_vals = np.full(n_bouts, np.nan)
    nearest_max_vals = np.full(n_bouts, np.nan)
    current_arm_vals = np.full(n_bouts, np.nan)
    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    completed_trial_arm_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        physical_arm = (
            safe_mode(g["track_segment_id"]) if "track_segment_id" in g else np.nan
        )
        location = summarize_behavioral_location(g[["location_state", "location_well"]])
        location_well = float(location["location_well"])
        location_arm = single_arm_from_well(location_well, track_graph)
        location_state_vals.append(str(location["location_state"]))
        at_well_vals.append(location["at_well"])
        location_well_vals[bi] = location_well
        location_arm_vals[bi] = location_arm
        physical_arm_vals[bi] = physical_arm
        current_arm_vals[bi] = location_arm
        distance = g["nearest_well_distance_cm"].to_numpy(dtype=float)
        finite_distance = distance[np.isfinite(distance)]
        if finite_distance.size:
            nearest_min_vals[bi] = finite_distance.min()
            nearest_mean_vals[bi] = finite_distance.mean()
            nearest_max_vals[bi] = finite_distance.max()
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

    per_bout["location_state"] = location_state_vals
    per_bout["at_well"] = pd.array(at_well_vals, dtype="boolean")
    per_bout["location_well"] = location_well_vals
    per_bout["location_arm"] = location_arm_vals
    per_bout["physical_arm"] = physical_arm_vals
    per_bout["nearest_well_distance_cm_min"] = nearest_min_vals
    per_bout["nearest_well_distance_cm_mean"] = nearest_mean_vals
    per_bout["nearest_well_distance_cm_max"] = nearest_max_vals
    per_bout["current_arm"] = current_arm_vals
    per_bout["prev_arm"] = prev_arm_vals
    per_bout["next_arm"] = next_arm_vals
    per_bout["last_rewarded_arm"] = last_rew_vals
    per_bout["completed_trial_arm"] = completed_trial_arm_vals
    cur_patch = np.array([arm_to_patch(a) for a in current_arm_vals])
    nxt_patch = np.array([arm_to_patch(a) for a in next_arm_vals])
    per_bout["current_patch"] = cur_patch
    per_bout["prev_patch"] = [arm_to_patch(a) for a in prev_arm_vals]
    per_bout["next_patch"] = nxt_patch
    both_valid = ~np.isnan(cur_patch.astype(float)) & ~np.isnan(nxt_patch.astype(float))
    per_bout["is_stay"] = np.where(
        both_valid, (cur_patch == nxt_patch).astype(float), np.nan
    )

    per_bout.insert(0, "animal", animal)
    per_bout.insert(1, "day", day)
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

    return per_bout[column_order]


# ── Column definitions ──


def _contfrag_event_table_columns(event_source: str = "SWR") -> list[str]:
    """Canonical column order for ContFrag event table."""
    event_col = _resolve_event_col(event_source)
    return [
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
        event_col,
        "event_id",
        "event_start",
        "event_end",
        "event_duration",
        "event_midpoint_time",
        "time_since_stop",
        "time_since_port_entry",
        "time_since_last_port_entry",
        "time_since_reward",
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
        "physical_arm",
        "nearest_well_distance_cm_min",
        "nearest_well_distance_cm_mean",
        "nearest_well_distance_cm_max",
        "is_swr",
        "is_mua",
        "continuous_prob_mean",
        "fragmented_prob_mean",
        "dominant_state",
        "continuous_fraction",
        "is_after_switch",
        "is_after_omission",
        "is_after_reward",
        "is_after_stay",
        "reward_outcome",
        "is_stay",
        "current_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "n_timebins",
        "posterior_peak_mean",
        "posterior_entropy_mean",
        "posterior_hpd_width_50",
        "posterior_hpd_width_80",
        "posterior_hpd_width_95",
        "n_junction_bins",
        "n_unique_arms",
        "stop_idx",
        "stop_id",
        # Trajectory rollups shared with the Non-Local table (continuous /
        # fragmented_prob_mean are above, from _summarize_event_state).
        *_CONTFRAG_TRAJECTORY_COLUMNS,
    ]


#: Canonical column order for ContFrag stop table.
_CONTFRAG_STOP_TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "day",
    "epoch",
    "event_source",
    "brain_area",
    "stop_idx",
    "stop_id",
    "stop_start",
    "stop_end",
    "immobility_time",
    "reward_rate_trial_k10",
    "is_swr_stop",
    "is_mua_stop",
    "has_replay",
    "n_events",
    "is_after_switch",
    "is_after_omission",
    "is_after_reward",
    "is_after_stay",
    "reward_outcome",
    "completed_trial_id",
    "completed_trial_arm",
    "recent_switch_rate",
    "leaky_q_value_diff",
    "is_stay",
    "location_state",
    "at_well",
    "location_well",
    "location_arm",
    "physical_arm",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "current_arm",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
)


def _contfrag_event_arm_table_columns(event_source: str = "SWR") -> list[str]:
    """Canonical column order for ContFrag event-arm table."""
    event_col = _resolve_event_col(event_source)
    return [
        # Identifiers
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
        event_col,
        "event_id",
        "stop_idx",
        "stop_id",
        # Event-level context
        "event_start",
        "event_end",
        "event_duration",
        "time_since_stop",
        "time_since_port_entry",
        "time_since_last_port_entry",
        "time_since_reward",
        "location_state",
        "at_well",
        "location_well",
        "location_arm",
        "physical_arm",
        "nearest_well_distance_cm_min",
        "nearest_well_distance_cm_mean",
        "nearest_well_distance_cm_max",
        "is_swr",
        "is_mua",
        "continuous_prob_mean",
        "fragmented_prob_mean",
        "dominant_state",
        "continuous_fraction",
        "is_after_switch",
        "is_after_omission",
        "is_after_reward",
        "is_after_stay",
        "reward_outcome",
        "is_stay",
        "current_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "n_timebins",
        # Per-arm identity
        "arm",
        "arm_patch_id",
        # Content responses
        "arm_count_hard",
        "arm_count_soft",
        "arm_posterior_mass",
        "arm_fraction",
        # Binary indicators
        "is_current",
        "is_prev",
        "is_next",
        "is_last_rewarded",
        "is_last_rewarded_elsewhere",
        "is_same_patch",
        # Spatial
        "dist_to_arm",
        "arm_dist_rank",
        # Reward / value
        "visit_recency",
        "visit_recency_demeaned",
        "arm_never_visited",
        "cumulative_reward",
        "reward_rate_arm_k10",
        "p_reward_arm",
        "q_arm",
        # Interactions
        "current_x_last_rewarded",
        "time_x_current",
        "time_x_prev",
        "time_x_next",
        "swr_x_current",
        "swr_x_prev",
        "swr_x_next",
        "swr_x_last_rewarded",
        "swr_x_reward_rate_arm_k10",
        "swr_x_visit_recency_demeaned",
        "swr_x_arm_never_visited",
        "swr_x_is_last_rewarded_elsewhere",
        "mua_x_current",
        "mua_x_prev",
        "mua_x_next",
        "mua_x_last_rewarded",
        "switch_x_current",
        "switch_x_prev",
        "omission_x_last_rewarded",
        # ContFrag-specific interactions
        "continuous_x_current",
        "continuous_x_prev",
        "continuous_x_next",
        "continuous_x_last_rewarded",
    ]
