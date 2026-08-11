"""Canonical per-session tables for replay content analysis.

This module produces the four canonical per-session tables:

- ``event_table`` — one row per non-local event
  (:func:`build_event_table`)
- ``event_arm_table`` — one row per event × arm
  (:func:`build_event_arm_table`)
- ``event_sequence_table`` — one row per in-event timebin
  (:func:`build_event_sequence_table`)
- ``stop_table`` — one row per stopping period, including zero-event
  stops (:func:`build_stop_table`)

Intermediate columns added by ``_add_trial_context`` (``prev_well``,
``current_well``, ``next_well``, ``last_rewarded_well``) use short
well-suffixed names; per-arm scalars in the canonical tables
(``prev_arm``, ``current_arm``, ``next_arm``, ``last_rewarded_arm``)
come from ``_single_arm_from_well(well_id, track_graph)``.

Three symbols are intentionally exempt from the canonical
rename's grep-zero rule:

- ``SWR_event`` and ``MUA_event`` — integer event-label columns
  written into ``data_over_time.pkl`` by the decoder. They carry
  more information than the booleans derived from them (``is_mua`` =
  any ``MUA_event > 0`` within the event; ``SWR_event`` likewise feeds
  the classified table's strict-overlap ``is_swr`` — see below) and are
  consumed by ``quality_metrics`` / ``event_analysis`` via
  groupby-by-ripple aggregation. Renaming would force a decoder re-run
  across all session pickles.
- ``last_rewarded_well`` — the intermediate column written by
  ``_add_trial_context``. Kept in symmetry with the canonical
  output scalar ``last_rewarded_arm`` so the well↔arm pairing
  (``prev_well``/``prev_arm``, ``current_well``/``current_arm``,
  etc.) reads consistently.

``is_swr`` is no longer the inclusive ``any SWR_event > 0 within event``
rule. It now means strict positive-duration overlap with any SWR, taken
together with the full ``overlap_class`` / ``overlap_subclass`` /
``matched_swr_*`` / ``label_event_*`` / per-event speed / peak-probability
columns from the canonical classified event table in
:mod:`continuum_swr_replay.analysis.hippocampus.event_table`. The legacy
``event_start`` / ``event_end`` / ``event_duration`` /
``event_midpoint_time`` columns remain the immobile-high-prob filtered
survivor window; the full label-window boundary is carried additively as
``label_event_*``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
)
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D

from ._helpers import (
    add_trial_context as _add_trial_context,
)
from ._helpers import (
    align_count_matrix as _align_count_matrix,
)
from ._helpers import (
    align_posterior_to_dot as _align_posterior_to_dot,
)
from ._helpers import (
    arm_to_patch as _arm_to_patch,
)
from ._helpers import (
    assign_bout as _assign_bout,
)
from ._helpers import (
    attach_arm_uncertainty as _attach_arm_uncertainty,
)
from ._helpers import (
    attach_contingency_value_block as _attach_contingency_value_block,
)
from ._helpers import (
    build_well_to_arm_map as _build_well_to_arm_map,
)
from ._helpers import (
    cumulative_reward_all_events as _cumulative_reward_all_events,
)
from ._helpers import (
    eq_indicator as _eq_indicator,
)
from ._helpers import (
    hash_event_id as _hash_event_id,
)
from ._helpers import (
    hash_stop_id as _hash_stop_id,
)
from ._helpers import (
    immobility_bout_intervals as _immobility_bout_intervals,
)
from ._helpers import (
    lagged_trial_rewards as _lagged_trial_rewards,
)
from ._helpers import (
    recent_reward_rate_all_events as _recent_reward_rate_all_events,
)
from ._helpers import (
    recent_trial_reward_rate as _recent_trial_reward_rate,
)
from ._helpers import (
    safe_mode as _safe_mode,
)
from ._helpers import (
    single_arm_from_well as _single_arm_from_well,
)
from ._helpers import (
    strict_first as _strict_first,
)
from ._helpers import (
    time_since_stop as _time_since_stop,
)
from ._helpers import (
    visit_recency_all_events as _visit_recency_all_events,
)
from .model_specs import N_REWARD_LAGS
from .spatial import arm_dist_rank, dist_to_arm_from_well
from .spatial_mapping import arm_to_well, map_estimate_to_arm
from .summaries import (
    arm_count_hard,
    arm_count_soft,
    arm_map_from_posterior,
    arm_posterior_mass,
    n_nonlocal_bins,
    nonlocal_weight_sum,
    posterior_mass_total,
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

# Continuous per-arm covariates standardized to pooled-GLOBAL unit SD as ``<col>_z``.
# The build fills NaN placeholders per session; the real global z is computed once in
# ``prepare_data.main`` after the all-session concat (see analysis.replay.standardization).
EVENT_ARM_STANDARDIZED_COLS = (
    "q_arm", "reward_rate_arm_k10", "p_reward_arm", "arm_uncertainty"
)

# State-probability column mapping for the 4-state Non-Local model:
# per-event rollup column -> data_over_time per-bin column.
_NONLOCAL_STATE_COLS = {
    "local_prob_mean": "Local_prob",
    "nospike_prob_mean": "No-Spike_prob",
    "continuous_prob_mean": "Non-Local_Continuous_prob",
    "fragmented_prob_mean": "Non-Local_Fragmented_prob",
}

# ── Shared prep for the event / event-sequence builders ──


def _prepare_dot(dot: pd.DataFrame, env) -> pd.DataFrame:
    """Return a copy of ``dot`` annotated with ``decoded_arm`` and ``nonlocal_prob``.

    ``decoded_arm`` comes from :func:`map_estimate_to_arm` (junction bins
    excluded). ``nonlocal_prob`` is the sum of the two non-local state
    posteriors — this is the ``w_t`` that drives event filtering and
    per-event weight sums.
    """
    dot = dot.copy()
    dot["decoded_arm"] = map_estimate_to_arm(dot["map_estimate"].to_numpy(), env)
    dot["nonlocal_prob"] = (
        dot["Non-Local_Continuous_prob"].to_numpy()
        + dot["Non-Local_Fragmented_prob"].to_numpy()
    )
    return dot


def _in_event_mask(
    dot: pd.DataFrame,
    speed_threshold: float,
    nonlocal_threshold: float,
) -> pd.Series:
    """Boolean mask over ``dot`` for in-event timebins.

    A row is in-event iff the decoder assigned a non-local event label,
    the animal was immobile, and the smoothed non-local posterior
    cleared the threshold. Matches the legacy ``prepare_session``
    filter.
    """
    return (
        (dot["non_local_event"] > 0)
        & (dot["head_speed"] < speed_threshold)
        & (dot["nonlocal_prob"] > nonlocal_threshold)
    )


# ── Public API ──


def build_event_table(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    conditional_posterior=None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    place_bin_size: float = PLACE_BIN_SIZE_1D,
) -> pd.DataFrame:
    """One row per non-local event with canonical event-level columns.

    Consumes the pre-saved ``{epoch}_HPC_{thr}_data_over_time.pkl`` frame
    (``dot``) plus the session's ``env`` / ``trials`` / ``track_graph``
    and emits the event-level slice of the canonical schema.

    The table is composed from two passes over ``dot``:

    - **Legacy aggregates** (timing, arms, patches, context flags, content
      and posterior summaries, ``stop_id``) are computed over the filtered
      sub-window — rows where the event label is set, the animal is
      immobile, and the smoothed non-local posterior clears the threshold
      (``_in_event_mask``). These columns, including ``event_start`` /
      ``event_end`` / ``event_duration`` / ``event_midpoint_time``, are the
      filtered-survivor window and are unchanged from prior versions.
    - **SWR classification and additive full-window summaries** come from
      :func:`continuum_swr_replay.analysis.hippocampus.event_table.build_classified_nonlocal_event_table`,
      computed over the *full* event-label window (no immobility / prob
      filter), and merged in on ``event_id``: ``is_swr`` (strict
      positive-duration SWR overlap), ``overlap_class`` /
      ``overlap_subclass`` / ``n_swrs_overlapping`` / ``matched_swr_*``,
      the per-event speed summaries, ``peak_nonlocal_prob``, and the full
      label-window boundary as ``label_event_*``.

    The replay population is unchanged: an event appears here iff at least
    one of its timebins survived ``_in_event_mask`` (the filtered subset is
    a strict subset of the full label window, so every retained event has a
    classified row).

    ``conditional_posterior`` is optional. When supplied (e.g., from
    :func:`posterior_io.load_conditional_posterior`), six additional
    columns are populated: ``posterior_mass_total``,
    ``posterior_peak_mean``, ``posterior_entropy_mean``, and
    ``posterior_hpd_width_{50,80,95}``. When not supplied those columns
    are ``NaN`` so the schema is stable across call sites.

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
        Spyglass trial frame with ``start_time``, ``end_time``,
        ``to_well``, ``from_well``, ``is_reward``, ``is_patch_change``.
    track_graph : networkx.Graph
        Session track graph; edges carry ``edge_id``.
    speed_threshold : float
        Head-speed threshold for immobility (cm/s). Default matches the
        project-wide constant.
    nonlocal_threshold : float
        Non-local state probability threshold. Default matches the
        project-wide constant.

    Returns
    -------
    pd.DataFrame
        One row per event with the columns documented below. Empty
        frame (correct columns, zero rows) if the session has no
        qualifying events.

    Columns
    -------
    Identifiers
        ``animal``, ``day``, ``epoch``, ``non_local_event``, ``event_id``
    Timing (filtered survivor window)
        ``event_start``, ``event_end``, ``event_duration``,
        ``time_since_stop``
    Timing (full label window, additive)
        ``label_event_start``, ``label_event_end``,
        ``label_event_duration``, ``label_event_midpoint_time``
    Arms (scalar per event)
        ``current_arm``, ``prev_arm``, ``next_arm`` (post-hoc),
        ``last_rewarded_arm``
    Patches
        ``current_patch``, ``prev_patch``, ``next_patch`` (post-hoc)
    Context flags
        ``is_after_switch``, ``is_after_reward``, ``is_after_omission``,
        ``is_after_stay``, ``reward_outcome``
    Event-type
        ``is_swr`` (strict positive-duration SWR overlap; see
        :mod:`continuum_swr_replay.analysis.hippocampus.event_table`),
        ``is_mua`` (any ``MUA_event > 0`` within event)
    SWR overlap (from the classified table)
        ``overlap_class``, ``overlap_subclass``, ``n_swrs_overlapping``,
        ``matched_swr_event``, ``matched_swr_start``, ``matched_swr_end``
    Per-event speed / peak (full label window)
        ``max_speed``, ``mean_speed``, ``start_speed``, ``end_speed``,
        ``endpoints_speed_below_thresh``, ``peak_nonlocal_prob``
    Stop linkage
        ``stop_idx``, ``stop_id``
    Content summaries
        ``n_nonlocal_bins``, ``nonlocal_weight_sum``,
        ``n_junction_bins``, ``n_unique_arms``. Despite the name,
        ``n_junction_bins`` counts in-event timebins whose ``decoded_arm``
        is not a real arm. ``map_estimate_to_arm`` excludes junction bins
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

    ``current_arm`` is the modal ``decoded_arm`` of the animal's physical
    position during the event (from the decoder's map estimate on
    ``head_position``-derived track-segment labeling). Falls back to
    ``well_to_arm_set(current_well)`` when the modal position
    lands on a trunk arm — matches the pre-refactor semantic in
    ``data_preparation._add_per_arm_predictors_single_session``.
    """
    # --- 1. Annotate the full dot frame (needed for bout detection and
    #     per-timebin arm mapping prior to event filtering) ---
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = _prepare_dot(dot, env)
    dot = _add_trial_context(dot, trials)

    # Per-session immobility bouts — computed from the full frame so
    # bout starts are correct even when an event begins partway through
    # a bout. Both start and end timestamps are needed so events can be
    # mapped onto a stable ``stop_id`` (the hash of the bout index).
    bout_starts, bout_ends = _immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )

    # --- 2. Filter to in-event timebins ---
    in_event_mask = _in_event_mask(dot, speed_threshold, nonlocal_threshold)
    nl = dot.loc[in_event_mask]

    # Empty-output short-circuit — return a correctly-typed empty frame
    # so downstream concat does not blow up on column mismatch.
    column_order = list(_EVENT_TABLE_COLUMNS)
    if len(nl) == 0:
        return pd.DataFrame(columns=column_order)

    event_id_series = nl["non_local_event"].to_numpy()

    # --- 3. Per-event summaries via the shared summary helpers ---
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

    # --- 4. Per-event metadata (groupby once, iterate sorted ids) ---
    groups = nl.groupby("non_local_event", sort=True)
    event_records: list[dict] = []

    for ev_id, g in groups:
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

        # current_arm: modal track_segment_id (the arm the animal is
        # physically on during the event). Accept only leaf arms — the
        # ones that touch a well. ``arm_to_well`` returns NaN for trunk
        # arms (6/7/8 in the bandit graph) because they connect two
        # junctions and no well, which is the same condition the legacy
        # ``_add_per_arm_predictors_single_session`` uses to trigger the
        # ``current_well`` fallback. Driving the check off the
        # graph — rather than hard-coding {0..5} — keeps the fallback
        # correct on any track graph.
        ts = (
            g["track_segment_id"]
            if "track_segment_id" in g.columns
            else pd.Series(dtype=float)
        )
        ts_modal = _safe_mode(ts)
        current_arm = np.nan
        if not np.isnan(ts_modal):
            well_of_modal = arm_to_well(ts_modal, track_graph)
            if not np.isnan(well_of_modal):
                current_arm = float(ts_modal)
        # Trial-derived context columns are NaN at every dot timebin
        # before the first trial start (see ``add_trial_context``).
        # ``_safe_first`` would silently skip those leading NaNs and
        # return the next trial's context, leaking future state into
        # events whose first timebin had no trial active. The
        # stop-table builder already uses the strict semantics for the
        # same hazard — match it here so trial-derived event-level
        # fields propagate NaN when the event begins before any trial.
        if np.isnan(current_arm):
            current_arm = _single_arm_from_well(
                _strict_first(g["current_well"]), track_graph
            )

        prev_arm = _single_arm_from_well(_strict_first(g["prev_well"]), track_graph)
        next_arm = _single_arm_from_well(_strict_first(g["next_well"]), track_graph)
        last_rewarded_arm = _single_arm_from_well(
            _strict_first(g["last_rewarded_well"]), track_graph
        )

        is_after_switch = _strict_first(g["is_after_switch"])
        is_after_omission = _strict_first(g["is_after_omission"])
        is_after_reward = _strict_first(g["is_after_reward"])
        reward_outcome = _strict_first(g["reward_outcome"])
        is_after_stay = (
            1.0 - is_after_switch if not np.isnan(is_after_switch) else np.nan
        )

        event_records.append(
            {
                "non_local_event": int(ev_id),
                "event_start": float(times[0]),
                "event_end": float(times[-1]),
                "event_duration": float(times[-1] - times[0]),
                "is_mua": bool((g["MUA_event"] > 0).any())
                if "MUA_event" in g.columns
                else False,
                "current_arm": current_arm,
                "prev_arm": prev_arm,
                "next_arm": next_arm,
                "last_rewarded_arm": last_rewarded_arm,
                "current_patch": _arm_to_patch(current_arm),
                "prev_patch": _arm_to_patch(prev_arm),
                "next_patch": _arm_to_patch(next_arm),
                "is_after_switch": is_after_switch,
                "is_after_omission": is_after_omission,
                "is_after_reward": is_after_reward,
                "is_after_stay": is_after_stay,
                "reward_outcome": reward_outcome,
                "is_stay": float(_arm_to_patch(next_arm) == _arm_to_patch(current_arm))
                if not (
                    np.isnan(_arm_to_patch(next_arm))
                    or np.isnan(_arm_to_patch(current_arm))
                )
                else np.nan,
                "n_junction_bins": n_junction,
                "n_unique_arms": int(arms_real.nunique()),
                "event_midpoint_time": float(times[len(times) // 2]),
                "time_since_port_entry": float(
                    times[len(times) // 2] - _strict_first(g["_goal_arrival_time"])
                )
                if not np.isnan(_strict_first(g["_goal_arrival_time"]))
                else np.nan,
                "time_since_reward": float(
                    times[len(times) // 2] - _strict_first(g["_last_reward_time"])
                )
                if not np.isnan(_strict_first(g["_last_reward_time"]))
                else np.nan,
            }
        )

    per_event = pd.DataFrame.from_records(event_records)

    # --- 5. Align summary arrays (sorted by event id) with per_event ---
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
    per_event["nonlocal_weight_sum"] = weight_sum

    # --- 6. time_since_stop per event (median of per-timebin values) ---
    per_timebin_tss = _time_since_stop(nl.index.to_numpy(), bout_starts)
    tss_by_event = (
        pd.Series(per_timebin_tss, index=nl.index)
        .groupby(nl["non_local_event"].to_numpy(), sort=True)
        .median()
        .to_numpy()
    )
    per_event["time_since_stop"] = tss_by_event

    # --- 7. Conditional-posterior-dependent columns ---
    # Absent ``conditional_posterior`` — emit the canonical columns as
    # NaN so the schema is stable across call sites. When supplied, we
    # marginalize to arms inside ``nl`` (same in-event filter as the
    # rest of the builder) and feed the helpers in
    # :mod:`summaries` / :mod:`uncertainty`.
    posterior_cols = (
        "posterior_mass_total",
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
        post_vals = _align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event_mask.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(
            post_in_event, edge_id, n_arms=len(ALL_ARMS)
        )

        _, mass_total = posterior_mass_total(nonlocal_prob_arr, event_id_series)
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

        per_event["posterior_mass_total"] = mass_total
        per_event["posterior_peak_mean"] = peak
        per_event["posterior_entropy_mean"] = entropy
        per_event["posterior_hpd_width_50"] = hpd[0.5]
        per_event["posterior_hpd_width_80"] = hpd[0.8]
        per_event["posterior_hpd_width_95"] = hpd[0.95]

    # --- 7b. Trajectory rollups ---
    # These require the augmented data_over_time columns (step_distance, raw
    # ripple/MUA/theta, run_speed_baseline). When ``dot`` predates the
    # augmentation, emit the columns as NaN so the schema stays stable.
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

    # --- 8. Identifiers + stable hashes ---
    per_event.insert(0, "animal", animal)
    per_event.insert(1, "day", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_id"] = _hash_event_id(per_event)

    # stop_id: each event's event_start belongs to exactly one
    # immobility bout (the in-event filter guarantees immobility, so
    # ``_assign_bout`` should find a valid bout for every row).
    stop_idx = _assign_bout(per_event["event_start"].to_numpy(), bout_starts, bout_ends)
    n_outside = int((stop_idx < 0).sum())
    if n_outside > 0:
        raise ValueError(
            f"{n_outside}/{len(stop_idx)} events fall outside all immobility bouts"
        )
    per_event["stop_idx"] = stop_idx
    per_event["stop_id"] = _hash_stop_id(animal, day, epoch, stop_idx)

    # --- 9. Attach SWR classification + additive full-window summaries ---
    # The groupby in section 4 above produced the legacy
    # filtered-survivor-window columns (timing + aggregates) unchanged.
    # build_classified_nonlocal_event_table is the canonical classified
    # table, computed over the *full* event-label window (no immobility /
    # probability filter). Merge the two on ``event_id``: left-join from
    # the filtered legacy population (sort=False preserves its row order
    # and count) so the replay table keeps its existing ``event_id`` set,
    # row count, and order. The filtered population is a strict subset of
    # the full label window, so every legacy event must have a classified
    # row — assert it.
    classified = build_classified_nonlocal_event_table(
        dot, speed_threshold=speed_threshold
    )
    classified = classified.rename(
        columns={
            "event_start": "label_event_start",
            "event_end": "label_event_end",
            "event_duration": "label_event_duration",
            "event_midpoint_time": "label_event_midpoint_time",
        }
    )
    classified_cols = [
        "event_id",
        "label_event_start",
        "label_event_end",
        "label_event_duration",
        "label_event_midpoint_time",
        "max_speed",
        "mean_speed",
        "start_speed",
        "end_speed",
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
    # ``hash_event_id`` — this guard catches unmatched rows, not a
    # hypothetical mismatch from changing the key in only one pass.
    n_unclassified = int(per_event["is_swr"].isna().sum())
    if n_unclassified > 0:
        raise RuntimeError(
            f"{n_unclassified}/{len(per_event)} replay events have no row in "
            "the classified event table; the filtered replay population must "
            "be a subset of the full event-label window."
        )

    return per_event[column_order]


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
        :func:`posterior_io.load_conditional_posterior`. Must have the
        same time length as ``dot``.
    speed_threshold, nonlocal_threshold : float
        Same defaults as :func:`build_event_table`. Keeping the two
        filters in lockstep is important: ``event_sequence_table`` and
        ``event_table`` must cover exactly the same in-event rows, or
        joins across them drift.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``animal``, ``day``, ``epoch``, ``non_local_event``,
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
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = _prepare_dot(dot, env)
    in_event_mask = _in_event_mask(dot, speed_threshold, nonlocal_threshold)

    column_order = list(_EVENT_SEQUENCE_TABLE_COLUMNS)
    if not in_event_mask.any():
        return pd.DataFrame(columns=column_order)

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
            "day": day,
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
    return out[column_order]


def build_stop_table(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per immobility bout, **including zero-event bouts**.

    The stop-level canonical view of the session. Every contiguous
    immobility bout (defined by ``head_speed < speed_threshold``) becomes a
    row, regardless of whether any non-local event fired during it. The
    hurdle occurrence model requires the zero-event rows.

    The ``stop_id`` column is a 63-bit hash of
    ``(animal, day, epoch, stop_idx)`` and matches
    :func:`build_event_table`'s ``stop_id`` row-for-row, so the two
    tables join cleanly.

    Deferred to later chunks: aggregated replay-summary columns
    (Σ ``arm_count_hard`` per arm, etc.) that require
    ``event_arm_table``.

    Note
    ----
    Single-sample bouts — when the animal dips below the speed
    threshold for exactly one timebin — yield ``immobility_time == 0``
    rows. They are kept as-is here; ``stop_table`` is the raw
    canonical frame and downstream consumers decide whether to filter
    them (e.g., the hurdle logistic, which uses
    ``log(immobility_time)``, will need to exclude or clip them).


    Parameters
    ----------
    dot, trials, track_graph, speed_threshold, nonlocal_threshold
        Same roles as in :func:`build_event_table`. ``env`` is not
        needed here — the stop table never touches the decoded arm of
        replay timebins.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`_STOP_TABLE_COLUMNS`.
    """
    if "animal" not in dot.columns:
        raise KeyError("dot must carry 'animal' column (set by get_data_over_time)")
    animal = str(dot["animal"].iloc[0])
    day = int(dot["day"].iloc[0])
    epoch = str(dot["epoch"].iloc[0])

    dot = dot.copy()
    dot["nonlocal_prob"] = (
        dot["Non-Local_Continuous_prob"].to_numpy()
        + dot["Non-Local_Fragmented_prob"].to_numpy()
    )
    dot = _add_trial_context(dot, trials)

    bout_starts, bout_ends = _immobility_bout_intervals(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )
    column_order = list(_STOP_TABLE_COLUMNS)
    if bout_starts.size == 0:
        return pd.DataFrame(columns=column_order)

    n_bouts = bout_starts.size
    times = dot.index.to_numpy()
    row_bout_idx = _assign_bout(times, bout_starts, bout_ends)  # -1 for mobile rows

    # Per-bout aggregations — use pandas groupby on the in-bout rows.
    in_bout = row_bout_idx >= 0
    bout_rows = dot.loc[in_bout].assign(_bout_idx=row_bout_idx[in_bout])
    grouped = bout_rows.groupby("_bout_idx", sort=True)

    per_bout = pd.DataFrame({"stop_idx": np.arange(n_bouts, dtype=np.int64)})
    per_bout["stop_start"] = bout_starts
    per_bout["stop_end"] = bout_ends
    per_bout["immobility_time"] = bout_ends - bout_starts
    per_bout["reward_rate_trial_k10"] = _recent_trial_reward_rate(
        bout_starts, trials
    )
    # Per-lag reward outcomes (reward_lag1..N): the immediate previous trial
    # (lag1, dwell-confounded) through earlier trials (lag>=2, dwell-clean) —
    # the kernel parameterization fit by the Rlag rate model.
    lag_rewards = _lagged_trial_rewards(bout_starts, trials, N_REWARD_LAGS)
    for j in range(N_REWARD_LAGS):
        per_bout[f"reward_lag{j + 1}"] = lag_rewards[:, j]

    per_bout["is_swr_stop"] = (
        grouped["SWR_event"]
        .apply(lambda s: (s > 0).any())
        .reindex(np.arange(n_bouts), fill_value=False)
        .to_numpy()
    )
    # n_SWR: number of distinct detected ripples (SWR_event labels) in the
    # bout — the raw-ripple analogue of n_non_local, for SWR-rate models.
    # (is_swr_stop == n_SWR > 0.)
    per_bout["n_SWR"] = (
        grouped["SWR_event"]
        .apply(lambda s: int(s[s > 0].nunique()))
        .reindex(np.arange(n_bouts), fill_value=0)
        .astype(np.int64)
        .to_numpy()
    )
    if "MUA_event" in dot.columns:
        per_bout["is_mua_stop"] = (
            grouped["MUA_event"]
            .apply(lambda s: (s > 0).any())
            .reindex(np.arange(n_bouts), fill_value=False)
            .to_numpy()
        )
        per_bout["n_MUA"] = (
            grouped["MUA_event"]
            .apply(lambda s: int(s[s > 0].nunique()))
            .reindex(np.arange(n_bouts), fill_value=0)
            .astype(np.int64)
            .to_numpy()
        )
    else:
        per_bout["is_mua_stop"] = False
        per_bout["n_MUA"] = 0

    # n_events: unique non_local_event > 0 within the bout AND with
    # nonlocal_prob above threshold AND speed below threshold — all
    # three clauses of ``_in_event_mask``. The speed clause is
    # typically redundant (bout membership already implies low speed)
    # but kept for consistency with ``build_event_table`` so
    # ``n_events`` in ``stop_table`` exactly equals the number of
    # events in ``event_table`` for that stop, even when a single
    # noisy sample pushes speed above threshold mid-bout.
    event_mask = (
        (bout_rows["non_local_event"] > 0)
        & (bout_rows["head_speed"] < speed_threshold)
        & (bout_rows["nonlocal_prob"] > nonlocal_threshold)
    )
    event_rows = bout_rows.loc[event_mask]
    n_events_per_bout = (
        event_rows.groupby("_bout_idx")["non_local_event"]
        .nunique()
        .reindex(np.arange(n_bouts), fill_value=0)
        .astype(np.int64)
        .to_numpy()
    )
    per_bout["has_replay"] = n_events_per_bout > 0
    per_bout["n_events"] = n_events_per_bout
    # Same count under the repo-wide name (cf. data_aggregation.py's
    # n_non_local / n_SWR / n_MUA). Both names ship: ``n_events`` is the
    # response column the per-stop rate models fit and the only count name
    # the ContFrag stop table carries, so it cannot simply be dropped.
    # Renaming is a ~50-site migration across src, tests and scripts, not a
    # cleanup; until then a test pins the two columns identical.
    per_bout["n_non_local"] = n_events_per_bout

    # Trial-context at bout start: first in-bout row's trial-derived
    # values. ``skipna=False`` is critical — the default ``first()``
    # silently skips NaNs and surfaces the first valid row anywhere
    # in the bout, which leaks future trial context into stops whose
    # actual bout start had no trial active yet (e.g. a bout
    # spanning a trial boundary inherits the post-boundary trial's
    # ``is_after_reward`` / ``reward_outcome``).
    bout_first_row = grouped.first(skipna=False)
    for col in (
        "is_after_switch",
        "is_after_omission",
        "is_after_reward",
        "is_after_stay",
        "reward_outcome",
    ):
        if col == "is_after_stay":
            # Not produced by _add_trial_context; derive from is_after_switch.
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

    # Arm / patch fields. ``current_arm`` uses the same modal-segment
    # + trunk-fallback rule as event-level current_arm; at a well-stop
    # the modal segment is a leaf arm. ``_strict_first`` keeps the
    # first row's NaN-or-finite value as-is so a bout that started
    # before any trial doesn't inherit the next trial's context.
    current_arm_vals = np.full(n_bouts, np.nan)
    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        ts_modal = (
            _safe_mode(g["track_segment_id"]) if "track_segment_id" in g else np.nan
        )
        ca = np.nan
        if not np.isnan(ts_modal):
            well_of_modal = arm_to_well(ts_modal, track_graph)
            if not np.isnan(well_of_modal):
                ca = float(ts_modal)
        if np.isnan(ca):
            ca = _single_arm_from_well(_strict_first(g["current_well"]), track_graph)
        current_arm_vals[bi] = ca
        prev_arm_vals[bi] = _single_arm_from_well(
            _strict_first(g["prev_well"]), track_graph
        )
        next_arm_vals[bi] = _single_arm_from_well(
            _strict_first(g["next_well"]), track_graph
        )
        last_rew_vals[bi] = _single_arm_from_well(
            _strict_first(g["last_rewarded_well"]), track_graph
        )

    per_bout["current_arm"] = current_arm_vals
    per_bout["prev_arm"] = prev_arm_vals
    per_bout["next_arm"] = next_arm_vals
    per_bout["last_rewarded_arm"] = last_rew_vals
    cur_patch = np.array([_arm_to_patch(a) for a in current_arm_vals])
    nxt_patch = np.array([_arm_to_patch(a) for a in next_arm_vals])
    per_bout["current_patch"] = cur_patch
    per_bout["prev_patch"] = [_arm_to_patch(a) for a in prev_arm_vals]
    per_bout["next_patch"] = nxt_patch
    # is_stay: will the next trip be within the same patch? Post-hoc.
    both_valid = ~np.isnan(cur_patch.astype(float)) & ~np.isnan(nxt_patch.astype(float))
    per_bout["is_stay"] = np.where(
        both_valid, (cur_patch == nxt_patch).astype(float), np.nan
    )

    per_bout.insert(0, "animal", animal)
    per_bout.insert(1, "day", day)
    per_bout.insert(2, "epoch", epoch)
    per_bout["stop_id"] = _hash_stop_id(
        animal, day, epoch, per_bout["stop_idx"].to_numpy()
    )

    return per_bout[column_order]


def add_nonlocal_swr_split(
    stop_table: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add per-stop ``n_non_local_SWR`` / ``n_non_local_no_SWR`` columns.

    Splits each stop's non-local event count by the **canonical** event-level
    ``is_swr`` (strict positive-duration SWR overlap, from ``event_table``)
    aggregated on ``stop_id`` — so the split matches ``event_table`` exactly
    rather than introducing a second inclusive-overlap convention. Stops with
    no events get 0/0, and ``n_non_local_SWR + n_non_local_no_SWR`` equals
    ``n_non_local`` for every stop.

    This is an assembly-level join (like the retrospective-metrics merge): it
    needs both canonical tables and so lives outside :func:`build_stop_table`,
    which is deliberately event-table-independent.
    """
    # Idempotent: drop any prior split so re-calling never produces _x/_y dups.
    out = stop_table.drop(
        columns=["n_non_local_SWR", "n_non_local_no_SWR"], errors="ignore"
    ).copy()
    total = out.get("n_non_local", out.get("n_events"))
    if (
        len(event_table) == 0
        or "stop_id" not in event_table.columns
        or "is_swr" not in event_table.columns
    ):
        out["n_non_local_SWR"] = 0
        out["n_non_local_no_SWR"] = (
            pd.Series(0, index=out.index) if total is None else total
        ).astype(np.int64)
        return out

    ev = event_table.dropna(subset=["stop_id"]).copy()
    # A non-local event whose stop_id is absent from stop_table would be silently
    # dropped by the left-merge below, leaving that stop a phantom 0/0 split. That
    # only happens if the two tables were built from different decodes/passes --
    # fail loud rather than corrupt the SWR split feeding the rate models.
    missing_stops = set(ev["stop_id"].unique()) - set(out["stop_id"])
    if missing_stops:
        raise ValueError(
            f"add_nonlocal_swr_split: {len(missing_stops)} stop_id(s) in "
            "event_table are absent from stop_table (e.g. "
            f"{next(iter(missing_stops))!r}); their non-local events would be "
            "silently dropped. The two tables must be built from the same decodes."
        )
    ev["_swr"] = ev["is_swr"].fillna(False).astype(bool)
    counts = (
        ev.groupby(["stop_id", "_swr"])
        .size()
        .unstack("_swr", fill_value=0)
        .rename(columns={True: "n_non_local_SWR", False: "n_non_local_no_SWR"})
    )
    for col in ("n_non_local_SWR", "n_non_local_no_SWR"):
        if col not in counts.columns:
            counts[col] = 0
    out = out.merge(
        counts[["n_non_local_SWR", "n_non_local_no_SWR"]],
        left_on="stop_id",
        right_index=True,
        how="left",
    )
    out[["n_non_local_SWR", "n_non_local_no_SWR"]] = (
        out[["n_non_local_SWR", "n_non_local_no_SWR"]].fillna(0).astype(np.int64)
    )
    return out


def build_event_arm_table(
    event_table: pd.DataFrame,
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    conditional_posterior=None,
    contingency: pd.DataFrame | None = None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per (event, arm) — nine rows per event.

    Assumes ``event_table`` was produced by :func:`build_event_table`
    on the same ``dot`` / ``trials`` / ``track_graph``. Broadcasts
    event-level fields (including ``stop_id``, context flags, and
    event-level posterior columns) across arms and augments each
    (event, arm) row with the three arm-count responses, the binary
    arm indicators, spatial and reward predictors, and the interaction
    terms.

    ``p_reward_arm`` (per-arm nominal reward probability) and ``q_arm``
    (per-arm model value / belief, Q) are populated from the contingency
    block when one is supplied, and emitted as NaN otherwise.

    Parameters
    ----------
    event_table : pd.DataFrame
        Output of :func:`build_event_table`.
    dot, env, trials, track_graph
        Same as :func:`build_event_table`. ``env`` is needed for
        ``map_estimate_to_arm`` and, when ``conditional_posterior`` is
        passed, for the ``place_bin_centers_nodes_df_`` edge_id vector.
    conditional_posterior : xr.DataArray, optional
        When supplied, the ``arm_posterior_mass`` column is populated
        via :func:`summaries.arm_posterior_mass`; otherwise it is NaN.
    speed_threshold, nonlocal_threshold : float
        Kept in lockstep with :func:`build_event_table`. Any drift
        would invalidate the ``event_id`` join.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`_EVENT_ARM_TABLE_COLUMNS`. Empty (correct
        schema) if ``event_table`` is empty.
    """
    column_order = list(_EVENT_ARM_TABLE_COLUMNS)
    if len(event_table) == 0:
        return pd.DataFrame(columns=column_order)

    n_arms = len(ALL_ARMS)
    n_events = len(event_table)

    dot = _prepare_dot(dot, env)
    in_event = _in_event_mask(dot, speed_threshold, nonlocal_threshold)
    nl = dot.loc[in_event]

    # Arm-count responses via the shared helpers. Junction bins (NaN
    # ``decoded_arm``) are excluded from the MAP-arm counts — they can
    # still contribute to the posterior-mass response via
    # :func:`posterior_to_arm_marginal`, which zeroes junction bins
    # and renormalizes per timebin.
    arm_series = nl["decoded_arm"]
    real_mask = arm_series.isin(ALL_ARMS).to_numpy()
    arm_map_real = arm_series.to_numpy()[real_mask].astype(np.int64)
    event_id_real = nl["non_local_event"].to_numpy()[real_mask]
    nonlocal_prob_real = nl["nonlocal_prob"].to_numpy()[real_mask]
    nonlocal_prob_all = nl["nonlocal_prob"].to_numpy()
    event_id_all = nl["non_local_event"].to_numpy()

    ids_real, counts_hard = arm_count_hard(
        nonlocal_prob_real, arm_map_real, event_id_real, n_arms=n_arms, threshold=0.0
    )
    _, counts_soft = arm_count_soft(
        nonlocal_prob_real, arm_map_real, event_id_real, n_arms=n_arms
    )

    # Align count matrices with event_table's event order. ``ids_real``
    # comes from np.unique (sorted), so missing events (those whose
    # every in-event timebin was on a junction bin — extremely rare)
    # need zero rows filled in.
    ev_ids = event_table["non_local_event"].to_numpy().astype(np.int64)
    hard_aligned = _align_count_matrix(counts_hard, ids_real, ev_ids, n_arms)
    soft_aligned = _align_count_matrix(counts_soft, ids_real, ev_ids, n_arms)

    if conditional_posterior is not None:
        edge_id = np.asarray(env.place_bin_centers_nodes_df_["edge_id"])
        post_vals = _align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(post_in_event, edge_id, n_arms=n_arms)
        ids_all, mass = arm_posterior_mass(
            nonlocal_prob_all, arm_posterior, event_id_all
        )
        mass_aligned = _align_count_matrix(mass, ids_all, ev_ids, n_arms)
    else:
        mass_aligned = np.full((n_events, n_arms), np.nan)

    # Broadcast event-level fields to 9-per-event, then attach per-arm
    # columns. ``np.repeat`` on the numpy representation avoids copying
    # the whole frame's index repeatedly.
    long = event_table.loc[event_table.index.repeat(n_arms)].reset_index(drop=True)
    long["arm"] = np.tile(ALL_ARMS, n_events)
    long["arm_count_hard"] = hard_aligned.flatten()
    long["arm_count_soft"] = soft_aligned.flatten()
    long["arm_posterior_mass"] = mass_aligned.flatten()

    # arm_fraction: numerator and denominator must cover the same set
    # of timebins. ``arm_count_hard`` excludes NaN-``decoded_arm``
    # timebins (counted as ``n_junction_bins``); the denominator must
    # exclude them too, or the fraction deflates below 1 whenever any
    # MAP failed to attribute to a real arm. NaN when no real-arm
    # bins exist (denominator <= 0).
    n_real_bins_long = (
        long["n_nonlocal_bins"].to_numpy() - long["n_junction_bins"].to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            n_real_bins_long > 0,
            long["arm_count_hard"].to_numpy() / np.maximum(n_real_bins_long, 1),
            np.nan,
        )
    long["arm_fraction"] = frac

    # arm_fraction_posterior: posterior-mass-based fraction. Uses
    # nonlocal_weight_sum as the denominator (= posterior_mass_total
    # under renormalization). NaN when posterior is unavailable or
    # nonlocal_weight_sum == 0.
    nws_long = long["nonlocal_weight_sum"].to_numpy()
    apm = long["arm_posterior_mass"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_post = np.where(
            (nws_long > 0) & np.isfinite(apm),
            apm / np.maximum(nws_long, 1e-10),
            np.nan,
        )
    long["arm_fraction_posterior"] = frac_post

    # Binary indicators: scalar equality against event-level arm cols.
    # On real bandit graph each well has degree 1 so is_prev / is_next
    # / is_last_rewarded are 1 on exactly one arm per event. If a
    # reference arm is NaN (session boundary, no last-rewarded, etc.),
    # the indicator is 0 for every arm of that event.
    arm_arr = long["arm"].to_numpy()
    long["is_current"] = _eq_indicator(arm_arr, long["current_arm"].to_numpy())
    long["is_prev"] = _eq_indicator(arm_arr, long["prev_arm"].to_numpy())
    long["is_next"] = _eq_indicator(arm_arr, long["next_arm"].to_numpy())
    long["is_last_rewarded"] = _eq_indicator(
        arm_arr, long["last_rewarded_arm"].to_numpy()
    )
    # is_last_rewarded_elsewhere: the last-rewarded arm, but only on
    # events where it is NOT the arm the animal currently occupies.
    # ``is_last_rewarded`` is ~98% identical to ``is_current`` (the
    # animal is usually rewarded where it now sits), so the two are
    # near-collinear in a joint model. ``is_current`` (M1) already
    # carries the rewarded-at-current-arm case; this indicator is the
    # identifiable residual — reward memory at a non-current arm — and
    # has disjoint support from ``is_current`` by construction.
    long["is_last_rewarded_elsewhere"] = (
        long["is_last_rewarded"].to_numpy() * (1 - long["is_current"].to_numpy())
    ).astype(int)
    long["arm_patch_id"] = np.array([_arm_to_patch(a) for a in arm_arr])
    long["is_same_patch"] = _eq_indicator(
        long["arm_patch_id"].to_numpy(), long["current_patch"].to_numpy()
    )

    # Per-event per-arm predictors: dist_to_arm, arm_dist_rank,
    # visit_recency, cumulative_reward, reward_rate_arm_k10.
    # Vectorized: precompute well→arm map once, then batch all events.
    well_to_arm_map = _build_well_to_arm_map(track_graph)

    # -- dist_to_arm and arm_dist_rank (per unique current_well) --
    current_arms = event_table["current_arm"].to_numpy()
    # Compute distances for each unique current_arm value
    unique_arms = np.unique(current_arms[np.isfinite(current_arms)])
    dist_cache: dict[int, np.ndarray] = {}
    rank_cache: dict[int, np.ndarray] = {}
    for ua in unique_arms:
        cw = arm_to_well(ua, track_graph)
        dists = dist_to_arm_from_well(cw, track_graph)
        d = np.array([dists[a] for a in ALL_ARMS])
        dist_cache[int(ua)] = d
        rank_cache[int(ua)] = arm_dist_rank(d)

    nan_row = np.full(n_arms, np.nan)
    dist_2d = np.array(
        [
            dist_cache.get(int(a), nan_row) if np.isfinite(a) else nan_row
            for a in current_arms
        ]
    )
    rank_2d = np.array(
        [
            rank_cache.get(int(a), nan_row) if np.isfinite(a) else nan_row
            for a in current_arms
        ]
    )

    # -- visit_recency, cumulative_reward, reward_rate_arm_k10 (batch) --
    event_times = event_table["event_midpoint_time"].to_numpy()
    if len(trials) > 0:
        vr_2d = _visit_recency_all_events(event_times, trials, well_to_arm_map)
        cr_2d = _cumulative_reward_all_events(event_times, trials, well_to_arm_map)
        # canonical per-arm reward rate: rewarded fraction of the last 10 visits
        # to that arm (covariate_dictionary.md D1; was 5-visit).
        rrr_2d = _recent_reward_rate_all_events(
            event_times, trials, well_to_arm_map, k=10
        )
    else:
        vr_2d = np.full((n_events, n_arms), np.nan)
        cr_2d = np.full((n_events, n_arms), np.nan)
        rrr_2d = np.full((n_events, n_arms), np.nan)

    # Events with NaN midpoint time get NaN for trial-derived columns
    nan_mask = np.isnan(event_times)
    if nan_mask.any():
        vr_2d[nan_mask] = np.nan
        cr_2d[nan_mask] = np.nan
        rrr_2d[nan_mask] = np.nan

    long["dist_to_arm"] = dist_2d.flatten()
    long["arm_dist_rank"] = rank_2d.flatten()
    long["visit_recency"] = vr_2d.flatten()
    # visit_recency with each arm's mean subtracted. Raw visit_recency
    # has a large static per-arm component that the arm fixed effects
    # absorb; the demeaned column is the part orthogonal to arm_fe and
    # is what M4+ uses (see model_specs.py).
    long["visit_recency_demeaned"] = long["visit_recency"] - long.groupby("arm")[
        "visit_recency"
    ].transform("mean")
    long["cumulative_reward"] = cr_2d.flatten()
    long["reward_rate_arm_k10"] = rrr_2d.flatten()

    # -- p_reward_arm / q_arm: per-arm nominal value + per-arm model value (Q)
    # from the per-stop contingency artifact (covariate_dictionary.md D2/D3).
    # The NonLocal event-arm table's own stop_id matches the contingency key.
    long["p_reward_arm"], long["q_arm"] = _attach_contingency_value_block(
        long["arm"].to_numpy(), long["stop_id"].to_numpy(), contingency
    )
    # arm_uncertainty: per-arm beta-model belief variance (betadist_var), the
    # variance counterpart of the beta value q_arm from the SAME posterior — so a
    # value+uncertainty model stays construct-coherent (one behavioral model).
    long["arm_uncertainty"] = _attach_arm_uncertainty(
        long["arm"].to_numpy(), long["stop_id"].to_numpy(), contingency
    )

    # Interactions: event-level × arm-level products.
    tss = long["time_since_stop"].to_numpy()
    is_swr_i = long["is_swr"].astype(int).to_numpy()
    is_mua_i = long["is_mua"].astype(int).to_numpy()
    is_after_switch = long["is_after_switch"].to_numpy()
    is_after_omission = long["is_after_omission"].to_numpy()
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
    # M7 — SWR × value/memory interactions. Previously omitted: the
    # raw swr_x_last_rewarded was r=1.0 with swr_x_current under the
    # old is_last_rewarded predictor; the _elsewhere fix makes the
    # reward-memory interaction identifiable. Sparse (~2% of events)
    # but honest.
    long["swr_x_reward_rate_arm_k10"] = (
        is_swr_i * long["reward_rate_arm_k10"].to_numpy()
    )
    long["swr_x_visit_recency_demeaned"] = (
        is_swr_i * long["visit_recency_demeaned"].to_numpy()
    )
    long["swr_x_is_last_rewarded_elsewhere"] = (
        is_swr_i * long["is_last_rewarded_elsewhere"].to_numpy()
    )
    long["swr_x_arm_uncertainty"] = is_swr_i * long["arm_uncertainty"].to_numpy()
    long["mua_x_current"] = is_mua_i * ic
    long["mua_x_prev"] = is_mua_i * ip
    long["mua_x_next"] = is_mua_i * inx
    long["mua_x_last_rewarded"] = is_mua_i * ilr
    long["switch_x_current"] = is_after_switch * ic
    long["switch_x_prev"] = is_after_switch * ip
    long["omission_x_last_rewarded"] = is_after_omission * ilr

    # Standardized (<col>_z) placeholders: the real pooled-GLOBAL z is computed
    # post-concat in prepare_data.main (a per-session build cannot know the global
    # mean/SD). Fill NaN here so the canonical column order carries the names.
    for c in EVENT_ARM_STANDARDIZED_COLS:
        long[f"{c}_z"] = np.nan

    return long[column_order]


def build_event_patch_table(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse event-arm table to event-patch level (3 rows per event).

    Sums ``arm_count_hard`` across arms within each patch, and
    aggregates per-arm predictors to patch level. The result is ready
    for patch-level conditional Poisson fitting (``MODEL_PATCH``).

    Parameters
    ----------
    event_table : pd.DataFrame
        Canonical event table (for ``last_rewarded_arm`` → patch mapping).
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).

    Returns
    -------
    pd.DataFrame
        3 rows per event (one per patch), with columns:
        ``event_id``, ``patch``, ``patch_count_hard``,
        ``is_current_patch``, ``is_prev_patch``,
        ``is_last_rewarded_patch``, ``is_last_rewarded_patch_elsewhere``,
        ``cumulative_reward_patch``, ``visit_recency_patch``,
        ``switch_x_current_patch``, ``swr_x_current_patch``,
        ``swr_x_last_rewarded_patch_elsewhere``, ``swr_x_prev_patch``.
    """
    # All groupbys / merges below key on ``event_id`` only. If a
    # caller passes session-local IDs (event_id=1 in two sessions),
    # the per-(event_id, patch) groupby silently fuses the two
    # sessions' arm counts and the merges to ``event_table`` then
    # produce a Cartesian product (a tiny two-session repro produced
    # 48 rows instead of the expected 6 session-patch rows). Mirror
    # the precondition already enforced in ``predict_arm_fractions``,
    # ``build_choice_dataset``, and ``stop_patch_replay``.
    from .choice_prediction import _require_session_unique_ids

    _require_session_unique_ids(
        event_arm_table, "event_id", helper_name="build_event_patch_table"
    )
    _require_session_unique_ids(
        event_table, "event_id", helper_name="build_event_patch_table"
    )

    eat = event_arm_table.copy()
    eat["patch"] = eat["arm"].map(TRACK_SEGMENT_TO_PATCH)

    # ``cumulative_reward`` is sometimes NaN (no reward history known
    # yet for a session). The default pandas sum has ``skipna=True``,
    # which collapses an all-NaN group to ``0.0`` — turning unknown
    # reward history into a real zero. ``min_count=1`` preserves NaN
    # when no non-null observations exist in the group, which is what
    # the downstream P3 patch model needs to drop those rows from its
    # complete-case fit instead of conditioning on a fabricated zero.
    agg_spec: dict = {
        "patch_count_hard": ("arm_count_hard", "sum"),
        "is_current_patch": ("is_same_patch", "max"),
        "cumulative_reward_patch": (
            "cumulative_reward",
            lambda s: s.sum(min_count=1),
        ),
        "visit_recency_patch": ("visit_recency", "min"),
    }
    # Value aggregates: patch mean over the patch's LEAF arms only (trunk arms
    # 6,7,8 are not reward sites) per the aggregator rule. Mask trunk to NaN so
    # the skipna mean averages the leaves; a patch whose leaves are all NaN
    # (unmapped stop / never-visited) stays NaN and drops at fit time.
    leaf = eat["arm"] < 6
    for col, out_col in (
        ("reward_rate_arm_k10", "reward_rate_arm_k10_patch"),
        ("p_reward_arm", "p_reward_arm_patch"),
        ("q_arm", "q_arm_patch"),
    ):
        if col in eat.columns:
            eat[f"_leaf_{col}"] = eat[col].where(leaf)
            agg_spec[out_col] = (f"_leaf_{col}", "mean")
    patch_eat = eat.groupby(["event_id", "patch"]).agg(**agg_spec).reset_index()

    # is_last_rewarded_patch: is the last-rewarded arm in this patch?
    last_rew = event_table[["event_id", "last_rewarded_arm"]].copy()
    last_rew["last_rewarded_patch"] = last_rew["last_rewarded_arm"].map(
        lambda a: float(TRACK_SEGMENT_TO_PATCH.get(int(a), np.nan))
        if pd.notna(a)
        else np.nan
    )
    patch_eat = patch_eat.merge(
        last_rew[["event_id", "last_rewarded_patch"]],
        on="event_id",
        how="left",
    )
    patch_eat["is_last_rewarded_patch"] = (
        patch_eat["patch"] == patch_eat["last_rewarded_patch"]
    ).astype(int)
    patch_eat = patch_eat.drop(columns=["last_rewarded_patch"])

    # is_last_rewarded_patch_elsewhere: the last-rewarded patch, but
    # only where it is NOT the patch the animal currently occupies.
    # ``is_last_rewarded_patch`` overlaps ``is_current_patch`` heavily
    # (the animal is usually rewarded in its current patch); the patch
    # hierarchy uses this disjoint-support residual at P2/P3, mirroring
    # the arm-level ``is_last_rewarded_elsewhere``. See model_specs.py.
    patch_eat["is_last_rewarded_patch_elsewhere"] = (
        patch_eat["is_last_rewarded_patch"].to_numpy()
        * (1 - patch_eat["is_current_patch"].to_numpy())
    ).astype(int)

    # is_prev_patch: is this the patch the animal just came from?
    prev_p = event_table[["event_id", "prev_patch"]].copy()
    patch_eat = patch_eat.merge(prev_p, on="event_id", how="left")
    patch_eat["is_prev_patch"] = (patch_eat["patch"] == patch_eat["prev_patch"]).astype(
        int
    )
    patch_eat = patch_eat.drop(columns=["prev_patch"])

    # switch_x_current_patch: is_after_switch × is_current_patch
    switch_col = event_table[["event_id", "is_after_switch"]].copy()
    patch_eat = patch_eat.merge(switch_col, on="event_id", how="left")
    patch_eat["switch_x_current_patch"] = (
        patch_eat["is_after_switch"] * patch_eat["is_current_patch"]
    )
    patch_eat = patch_eat.drop(columns=["is_after_switch"])

    # SWR × patch interactions (P4, PS3). is_swr is event-level — bring
    # it in from event_table once and reuse for the three products. The
    # column is RETAINED in the output so the patch-level fitter's
    # ``--swr-subset`` filter (which gates on an ``is_swr`` column) works
    # for ``--hierarchy patch`` as it does for arm fits.
    # Tolerant of test fixtures that omit is_swr: default to all-False.
    if "is_swr" in event_table.columns:
        swr_col = event_table[["event_id", "is_swr"]].copy()
        patch_eat = patch_eat.merge(swr_col, on="event_id", how="left")
        # Every patch-table event_id comes from event_table, so the merge
        # must be total. Fail loudly rather than fabricate is_swr=False for
        # an unmatched id (which would silently mislabel SWR status).
        n_unmatched = int(patch_eat["is_swr"].isna().sum())
        if n_unmatched > 0:
            raise ValueError(
                f"build_event_patch_table: {n_unmatched} patch rows have an "
                "event_id absent from event_table; is_swr cannot be assigned."
            )
        is_swr_i = patch_eat["is_swr"].astype(int).to_numpy()
    else:
        is_swr_i = np.zeros(len(patch_eat), dtype=int)
    patch_eat["swr_x_current_patch"] = (
        is_swr_i * patch_eat["is_current_patch"].to_numpy()
    )
    patch_eat["swr_x_last_rewarded_patch_elsewhere"] = (
        is_swr_i * patch_eat["is_last_rewarded_patch_elsewhere"].to_numpy()
    )
    patch_eat["swr_x_prev_patch"] = is_swr_i * patch_eat["is_prev_patch"].to_numpy()

    # Session keys are required by the patch-level fitter, which groups
    # by ["animal", "day", "epoch"] before fitting per session. Without
    # them, ``fit_replay_content_models.py --hierarchy patch`` raises
    # KeyError before fitting; loading the joined table would otherwise
    # silently mix events across sessions.
    session_cols = ["animal", "day", "epoch"]
    if all(col in event_table.columns for col in session_cols):
        patch_eat = patch_eat.merge(
            event_table[["event_id", *session_cols]],
            on="event_id",
            how="left",
        )

    return patch_eat


#: Canonical column order for ``event_table``.
#:
#: Kept central so the empty-session short-circuit returns the right shape and
#: so downstream consumers can rely on column order.
_EVENT_TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "event_id",
    "event_start",
    "event_end",
    "event_duration",
    "event_midpoint_time",
    "time_since_stop",
    "time_since_port_entry",
    "time_since_reward",
    "is_swr",
    "is_mua",
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
    "n_nonlocal_bins",
    "nonlocal_weight_sum",
    "posterior_mass_total",
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_width_50",
    "posterior_hpd_width_80",
    "posterior_hpd_width_95",
    "n_junction_bins",
    "n_unique_arms",
    "stop_idx",
    "stop_id",
    # Additive full-window timing + SWR classification from
    # build_classified_nonlocal_event_table (full label window).
    # Legacy event_start/end/duration/midpoint above remain the
    # filtered survivor-window values; these label_event_* carry the
    # full event-label boundary.
    "label_event_start",
    "label_event_end",
    "label_event_duration",
    "label_event_midpoint_time",
    "max_speed",
    "mean_speed",
    "start_speed",
    "end_speed",
    "endpoints_speed_below_thresh",
    "peak_nonlocal_prob",
    "overlap_class",
    "overlap_subclass",
    "n_swrs_overlapping",
    "matched_swr_event",
    "matched_swr_start",
    "matched_swr_end",
    # Trajectory rollups (shared with ContFrag) + Non-Local-only state means.
    *SHARED_TRAJECTORY_COLUMNS,
    *NONLOCAL_ONLY_COLUMNS,
)


#: Canonical column order for ``stop_table``.
_STOP_TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "day",
    "epoch",
    "stop_idx",
    "stop_id",
    "stop_start",
    "stop_end",
    "immobility_time",
    "reward_rate_trial_k10",
    "is_swr_stop",
    "n_SWR",
    "is_mua_stop",
    "n_MUA",
    "has_replay",
    "n_events",
    "n_non_local",
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
    *[f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)],
)


#: Canonical column order for ``event_arm_table``.
_EVENT_ARM_TABLE_COLUMNS: tuple[str, ...] = (
    # Identifiers (broadcast from event_table)
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "event_id",
    "stop_idx",
    "stop_id",
    # Event-level context (broadcast)
    "event_start",
    "event_end",
    "event_duration",
    "time_since_stop",
    "time_since_port_entry",
    "time_since_reward",
    "is_swr",
    "is_mua",
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
    "n_nonlocal_bins",
    # Per-arm identity
    "arm",
    "arm_patch_id",
    # Content responses
    "arm_count_hard",
    "arm_count_soft",
    "arm_posterior_mass",
    "arm_fraction",
    "arm_fraction_posterior",
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
    "cumulative_reward",
    "reward_rate_arm_k10",
    "p_reward_arm",
    "q_arm",
    "arm_uncertainty",
    # Interactions (+ collinearity control)
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
    "swr_x_is_last_rewarded_elsewhere",
    "swr_x_arm_uncertainty",
    "mua_x_current",
    "mua_x_prev",
    "mua_x_next",
    "mua_x_last_rewarded",
    "switch_x_current",
    "switch_x_prev",
    "omission_x_last_rewarded",
    # Standardized (pooled-global) continuous covariates, filled post-concat in
    # prepare_data.main (see EVENT_ARM_STANDARDIZED_COLS / analysis.replay.standardization).
    *[f"{c}_z" for c in EVENT_ARM_STANDARDIZED_COLS],
)


#: Canonical column order for ``event_sequence_table``.
_EVENT_SEQUENCE_TABLE_COLUMNS: tuple[str, ...] = (
    "animal",
    "day",
    "epoch",
    "non_local_event",
    "event_id",
    "t_in_event",
    "time",
    "nonlocal_prob",
    "decoded_arm_map",
    *(f"arm_posterior_{a}" for a in ALL_ARMS),
)
