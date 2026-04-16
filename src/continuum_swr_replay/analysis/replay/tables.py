"""Canonical per-session tables for replay content analysis.

This module produces the four canonical tables specified in
``plans/2026_04_14_replay_content_analysis_plan.md`` Section 2:

- ``event_table`` — one row per non-local event
- ``event_arm_table`` — one row per event × arm (not yet implemented)
- ``event_sequence_table`` — one row per in-event timebin (not yet implemented)
- ``stop_table`` — one row per stopping period, including zero-event
  stops (not yet implemented)

This file is being built incrementally (plan Phase B). Chunk 1 lands
``build_event_table`` only; the other three tables, the conditional-
posterior-dependent columns, MUA wiring, and invariant tests come in
subsequent commits. Deferred columns are listed explicitly in
``build_event_table``'s docstring.

Canonical column names follow the plan's Section 1. Intermediate
columns added by ``_add_trial_context`` (``prev_well``,
``current_well``, ``next_well``, ``last_rewarded_well``) use short
well-suffixed names; per-arm scalars in the canonical tables
(``prev_arm``, ``current_arm``, ``next_arm``, ``last_rewarded_arm``)
come from ``_single_arm_from_well(well_id, track_graph)``.

Three symbols are intentionally exempt from the Phase B canonical
rename's grep-zero rule:

- ``SWR_event`` and ``MUA_event`` — integer event-label columns
  written into ``data_over_time.pkl`` by the decoder. They carry
  more information than the booleans derived from them (``is_swr`` =
  ``SWR_event > 0``; same for MUA) and are consumed by
  ``quality_metrics`` / ``event_analysis`` via groupby-by-ripple
  aggregation. Renaming would force a decoder re-run across all
  session pickles.
- ``last_rewarded_well`` — the intermediate column written by
  ``_add_trial_context``. Kept in symmetry with the canonical
  output scalar ``last_rewarded_arm`` so the well↔arm pairing
  (``prev_well``/``prev_arm``, ``current_well``/``current_arm``,
  etc.) reads consistently.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .spatial import arm_dist_rank, dist_to_arm_from_well
from .spatial_mapping import arm_to_well, map_estimate_to_arm, well_to_arm_set
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
from .uncertainty import (
    posterior_entropy_mean,
    posterior_hpd_widths,
    posterior_peak_mean,
)

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


# ── Trial-derived context (port of legacy add_trial_context) ──


def _add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-timebin trial-derived columns to ``dot`` (in place, returned).

    In-progress trip semantics: at each timepoint ``t``, the "current
    trip" is the latest trial whose ``start_time`` is at or before ``t``.
    This is the trip the animal is currently completing (or has just
    completed, if ``t`` falls in the gap between two trips).

    Direct port of the legacy ``data_preparation.add_trial_context`` so
    coefficient values at Phase B gate match pre-refactor numbers
    exactly. Kept private; ``build_event_table`` owns the behavior.

    Columns added
    -------------
    - ``prev_well``, ``current_well``
    - ``next_well``
    - ``last_rewarded_well``
    - ``is_after_switch`` — current trip's ``is_patch_change``
    - ``is_after_omission`` — ``~current trip's is_reward``
    - ``is_after_reward`` — current trip's ``is_reward``
    - ``reward_outcome`` — current trip's ``is_reward`` as float

    Note
    ----
    ``reward_outcome`` and ``is_after_reward`` are identical by
    construction on this task (``reward_outcome`` = ``is_reward``;
    ``is_after_reward`` = ``is_reward`` because the current trip just
    completed). Both are kept to satisfy the plan's canonical schema,
    which names them separately; ``is_after_omission`` is the
    complement. ``is_after_stay`` is added in :func:`build_event_table`,
    not here (it needs no trial-level computation).
    """
    dot_times = dot.index.values
    n_trials = len(trials)

    # Empty-trials short-circuit: populate every trial-derived column
    # with NaN so the schema is complete and ``build_event_table`` can
    # still run end-to-end (e.g., for a session with no valid trials
    # or for unit-test fixtures that stub trial context out entirely).
    if n_trials == 0:
        for col in (
            "prev_well",
            "current_well",
            "next_well",
            "last_rewarded_well",
            "reward_outcome",
            "is_after_reward",
            "is_after_omission",
            "is_after_switch",
        ):
            dot[col] = np.nan
        return dot

    trial_start_times = trials["start_time"].values
    trial_end_times = trials["end_time"].values
    to_wells = trials["to_well"].values
    from_wells = trials["from_well"].values
    is_reward = trials["is_reward"].values
    is_patch_change = trials["is_patch_change"].values

    current_idx = np.searchsorted(trial_start_times, dot_times, side="right") - 1
    valid = current_idx >= 0

    current_clipped = np.clip(current_idx, 0, n_trials - 1)
    next_idx = current_idx + 1
    next_valid = valid & (next_idx < n_trials)
    next_clipped = np.clip(next_idx, 0, n_trials - 1)

    dot["prev_well"] = np.where(
        valid, from_wells[current_clipped], np.nan
    )
    dot["current_well"] = np.where(
        valid, to_wells[current_clipped], np.nan
    )
    dot["next_well"] = np.where(
        next_valid, to_wells[next_clipped], np.nan
    )

    rewarded_end_times = trial_end_times[is_reward]
    rewarded_wells = to_wells[is_reward]
    if len(rewarded_end_times) > 0:
        rew_idx = np.searchsorted(rewarded_end_times, dot_times, side="right") - 1
        rew_idx_clipped = np.clip(rew_idx, 0, None)
        dot["last_rewarded_well"] = np.where(
            rew_idx >= 0, rewarded_wells[rew_idx_clipped], np.nan
        )
    else:
        dot["last_rewarded_well"] = np.nan

    current_reward = np.where(
        valid, is_reward[current_clipped].astype(float), np.nan
    )
    dot["reward_outcome"] = current_reward
    dot["is_after_reward"] = current_reward
    dot["is_after_omission"] = np.where(valid, 1.0 - current_reward, np.nan)
    dot["is_after_switch"] = np.where(
        valid, is_patch_change[current_clipped].astype(float), np.nan
    )

    return dot


# ── Immobility-bout helpers (port of legacy compute_time_since_stop) ──


def _immobility_bout_starts(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> np.ndarray:
    """Timestamps at which speed crossed from above to below the threshold.

    Includes the first sample if the session starts immobile. Direct
    port of the legacy bout-detection logic in
    ``predictors.compute_time_since_stop``.
    """
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    imm_times = dot.index.to_numpy()
    transitions = np.where(~imm[:-1] & imm[1:])[0] + 1
    if len(imm) > 0 and imm[0]:
        transitions = np.concatenate([[0], transitions])
    return imm_times[transitions]


def _immobility_bout_intervals(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end timestamps for every contiguous immobility bout.

    Generalization of :func:`_immobility_bout_starts` that also emits
    bout ends — the first immobile sample's timestamp through the last
    consecutive immobile sample's timestamp, inclusive. An interval of
    length 1 (a single-sample bout) gets identical start and end.

    A session that never drops below the speed threshold returns two
    empty arrays.

    Returns
    -------
    starts : np.ndarray, shape (n_bouts,)
    ends : np.ndarray, shape (n_bouts,)
    """
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    times = dot.index.to_numpy()
    if imm.size == 0 or not imm.any():
        return np.array([]), np.array([])

    # Pad with False at both ends so transitions capture edge bouts.
    # Rising edges: first True of a run. Falling edges: first False
    # after the run — shift back by 1 to get the last True index.
    padded = np.concatenate(([False], imm, [False]))
    diff = np.diff(padded.astype(np.int8))
    start_idx = np.where(diff == 1)[0]          # index into imm of run starts
    end_idx = np.where(diff == -1)[0] - 1       # index into imm of run last-True
    return times[start_idx], times[end_idx]


def _time_since_stop(
    event_times: np.ndarray,
    bout_start_times: np.ndarray,
) -> np.ndarray:
    """Seconds from most recent immobility-bout start to each event time.

    Clamps negative values (event before any detected bout) to 0. Returns
    an empty array if ``bout_start_times`` is empty.
    """
    event_times = np.asarray(event_times, dtype=float)
    if bout_start_times.size == 0:
        return np.full_like(event_times, np.nan)
    idx = np.searchsorted(bout_start_times, event_times, side="right") - 1
    idx = np.clip(idx, 0, bout_start_times.size - 1)
    return np.clip(event_times - bout_start_times[idx], 0, None)


# ── Scalar extractions from per-arm sets ──


def _single_arm_from_well(well: float, track_graph: nx.Graph) -> float:
    """Deterministic scalar arm for a well, ``NaN`` if the well is NaN or isolated.

    On the real bandit graph each well is degree-1, so
    :func:`well_to_arm_set` returns a 1-element set. On fixtures with
    higher-degree wells, returns the minimum arm index for determinism.
    """
    if well is None or (isinstance(well, float) and np.isnan(well)):
        return np.nan
    arms = well_to_arm_set(well, track_graph)
    if len(arms) == 0:
        return np.nan
    return float(min(arms))


def _arm_to_patch(arm: float) -> float:
    """Patch ID for an arm, ``NaN`` propagating."""
    if arm is None or (isinstance(arm, float) and np.isnan(arm)):
        return np.nan
    return float(TRACK_SEGMENT_TO_PATCH.get(int(arm), np.nan))


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


def _align_posterior_to_dot(
    conditional_posterior,
    dot: pd.DataFrame,
) -> np.ndarray:
    """Return a ``(T_dot, n_position_bins)`` posterior array aligned to ``dot.index``.

    The decoder writes ``results.nc`` with a ``time`` coordinate that is
    the same time axis produced by
    :func:`data_aggregation.get_data_over_time` — so in practice the two
    are already 1-to-1. We still assert lengths match to catch silent
    misalignment if the aggregator and decoder ever diverge.

    Parameters
    ----------
    conditional_posterior : xr.DataArray
        Shape ``(time, position)`` from
        :func:`posterior_io.load_conditional_posterior`.
    dot : pd.DataFrame
        Time-indexed data-over-time frame.
    """
    post_vals = np.asarray(conditional_posterior.values)
    if post_vals.shape[0] != len(dot):
        raise ValueError(
            f"conditional_posterior time axis has {post_vals.shape[0]} "
            f"entries but dot has {len(dot)}; the two must be aligned "
            "1-to-1 (both come from the same decoder run)."
        )
    # Length-match alone is not enough — bin-centered vs. edge-indexed
    # time would slide every row by half a sample without tripping the
    # length check. Explicitly verify the two time axes agree within a
    # microsecond (500 Hz sampling = 2 ms between bins, so 1 µs is
    # well below any legitimate numerical noise).
    if "time" in conditional_posterior.coords:
        post_times = np.asarray(conditional_posterior["time"].values, dtype=float)
        drift = float(np.abs(post_times - dot.index.to_numpy(dtype=float)).max())
        if drift > 1e-6:
            raise ValueError(
                f"conditional_posterior time axis drifts from dot.index "
                f"by up to {drift:.3g}s — the two must share the same "
                "time origin, not just the same length."
            )
    return post_vals


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
) -> pd.DataFrame:
    """One row per non-local event with canonical event-level columns.

    Consumes the pre-saved ``{epoch}_HPC_{thr}_data_over_time.pkl`` frame
    (``dot``) plus the session's ``env`` / ``trials`` / ``track_graph``
    and emits the event-level slice of the plan's canonical schema.

    ``conditional_posterior`` is optional. When supplied (e.g., from
    :func:`posterior_io.load_conditional_posterior`), six additional
    columns are populated: ``posterior_mass_total``,
    ``posterior_peak_mean``, ``posterior_entropy_mean``, and
    ``posterior_hpd_width_{50,80,95}``. When not supplied those columns
    are ``NaN`` so the schema is stable across call sites.

    Still deferred to later chunks (intentionally absent from the
    current output): ``stop_id`` and the stop-level ``immobility_time``
    aggregation (chunk 3, ``build_stop_table``); ``block_number`` and
    ``trials_since_block_change`` (requires a contingency-block loader
    not yet wired).

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
    Timing
        ``event_start``, ``event_end``, ``event_duration``,
        ``time_since_stop``
    Arms (scalar per event)
        ``current_arm``, ``prev_arm``, ``next_arm`` (post-hoc),
        ``last_rewarded_arm``
    Patches
        ``current_patch``, ``prev_patch``, ``next_patch`` (post-hoc)
    Context flags
        ``is_after_switch``, ``is_after_reward``, ``is_after_omission``,
        ``is_after_stay``, ``reward_outcome``
    Event-type
        ``is_swr`` (any ``SWR_event > 0`` within event),
        ``is_mua`` (any ``MUA_event > 0`` within event)
    Content summaries
        ``n_nonlocal_bins``, ``nonlocal_weight_sum``,
        ``n_junction_bins``, ``n_unique_arms``

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
    column_order = _event_table_columns()
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
        ts = g["track_segment_id"] if "track_segment_id" in g.columns else pd.Series(dtype=float)
        ts_modal = _safe_mode(ts)
        current_arm = np.nan
        if not np.isnan(ts_modal):
            well_of_modal = arm_to_well(ts_modal, track_graph)
            if not np.isnan(well_of_modal):
                current_arm = float(ts_modal)
        if np.isnan(current_arm):
            current_arm = _single_arm_from_well(
                _safe_first(g["current_well"]), track_graph
            )

        prev_arm = _single_arm_from_well(
            _safe_first(g["prev_well"]), track_graph
        )
        next_arm = _single_arm_from_well(
            _safe_first(g["next_well"]), track_graph
        )
        last_rewarded_arm = _single_arm_from_well(
            _safe_first(g["last_rewarded_well"]), track_graph
        )

        is_after_switch = _safe_first(g["is_after_switch"])
        is_after_omission = _safe_first(g["is_after_omission"])
        is_after_reward = _safe_first(g["is_after_reward"])
        reward_outcome = _safe_first(g["reward_outcome"])
        is_after_stay = (
            1.0 - is_after_switch if not np.isnan(is_after_switch) else np.nan
        )

        event_records.append(
            {
                "non_local_event": int(ev_id),
                "event_start": float(times[0]),
                "event_end": float(times[-1]),
                "event_duration": float(times[-1] - times[0]),
                "is_swr": bool((g["SWR_event"] > 0).any()),
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
                "is_stay": float(
                    _arm_to_patch(next_arm) == _arm_to_patch(current_arm)
                )
                if not (np.isnan(_arm_to_patch(next_arm)) or np.isnan(_arm_to_patch(current_arm)))
                else np.nan,
                "n_junction_bins": n_junction,
                "n_unique_arms": int(arms_real.nunique()),
                # Legacy ``prepare_session`` uses the middle-index
                # timestamp, not the mean — preserve that choice for
                # Phase B gate parity even though the two coincide when
                # timebins are uniformly sampled.
                "event_midpoint_time": float(times[len(times) // 2]),
            }
        )

    per_event = pd.DataFrame.from_records(event_records)

    # --- 5. Align summary arrays (sorted by event id) with per_event ---
    # summaries.* returned unique_ids in sorted order; groupby(sort=True)
    # yields events in the same order — assert to catch drift.
    np.testing.assert_array_equal(
        per_event["non_local_event"].to_numpy(),
        ids_sorted.astype(per_event["non_local_event"].dtype),
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
        _, peak = posterior_peak_mean(
            arm_posterior, nonlocal_prob_arr, event_id_series
        )
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

    # --- 8. Identifiers + stable hashes ---
    per_event.insert(0, "animal", animal)
    per_event.insert(1, "day", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_id"] = _hash_event_id(per_event)

    # stop_id: each event's event_start belongs to exactly one
    # immobility bout (the in-event filter guarantees immobility, so
    # ``_assign_bout`` should find a valid bout for every row).
    stop_idx = _assign_bout(
        per_event["event_start"].to_numpy(), bout_starts, bout_ends
    )
    per_event["stop_idx"] = stop_idx
    per_event["stop_id"] = _hash_stop_id(animal, day, epoch, stop_idx)

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

    This is the plan's Section 2.3 table — the per-timebin canonical
    view of what the decoder said each timebin represented, inside
    non-local events. Its ``event_id`` hash matches
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

    column_order = _event_sequence_table_columns()
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

    This is the plan's Section 2.4 table — the stop-level canonical
    view of the session. Every contiguous immobility bout (defined by
    ``head_speed < speed_threshold``) becomes a row, regardless of
    whether any non-local event fired during it. Stage 4 Part 1
    (hurdle occurrence model) requires the zero-event rows.

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
    them (e.g., Stage 4's hurdle logistic, which uses
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
        Columns: see :func:`_stop_table_columns`.
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
    column_order = _stop_table_columns()
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

    per_bout["is_swr_stop"] = (
        grouped["SWR_event"].apply(lambda s: (s > 0).any()).reindex(
            np.arange(n_bouts), fill_value=False
        ).to_numpy()
    )
    if "MUA_event" in dot.columns:
        per_bout["is_mua_stop"] = (
            grouped["MUA_event"].apply(lambda s: (s > 0).any()).reindex(
                np.arange(n_bouts), fill_value=False
            ).to_numpy()
        )
    else:
        per_bout["is_mua_stop"] = False

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

    # Trial-context at bout start: first in-bout row's trial-derived
    # values. For zero-event bouts these are still meaningful (the
    # animal's behavioral state at the time of the stop).
    first_row = grouped.first()
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
                np.isnan(first_row["is_after_switch"].to_numpy()),
                np.nan,
                1.0 - first_row["is_after_switch"].to_numpy(),
            )
            per_bout["is_after_stay"] = pd.Series(stay).reindex(
                np.arange(n_bouts)
            ).to_numpy()
        else:
            per_bout[col] = (
                first_row[col]
                .reindex(np.arange(n_bouts))
                .to_numpy()
            )

    # Arm / patch fields. ``current_arm`` uses the same modal-segment +
    # trunk-fallback rule as event-level current_arm; at a well-stop
    # the modal segment is a leaf arm.
    current_arm_vals = np.full(n_bouts, np.nan)
    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        ts_modal = _safe_mode(g["track_segment_id"]) if "track_segment_id" in g else np.nan
        ca = np.nan
        if not np.isnan(ts_modal):
            well_of_modal = arm_to_well(ts_modal, track_graph)
            if not np.isnan(well_of_modal):
                ca = float(ts_modal)
        if np.isnan(ca):
            ca = _single_arm_from_well(
                _safe_first(g["current_well"]), track_graph
            )
        current_arm_vals[bi] = ca
        prev_arm_vals[bi] = _single_arm_from_well(
            _safe_first(g["prev_well"]), track_graph
        )
        next_arm_vals[bi] = _single_arm_from_well(
            _safe_first(g["next_well"]), track_graph
        )
        last_rew_vals[bi] = _single_arm_from_well(
            _safe_first(g["last_rewarded_well"]), track_graph
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
    per_bout["is_stay"] = np.where(both_valid, (cur_patch == nxt_patch).astype(float), np.nan)

    per_bout.insert(0, "animal", animal)
    per_bout.insert(1, "day", day)
    per_bout.insert(2, "epoch", epoch)
    per_bout["stop_id"] = _hash_stop_id(
        animal, day, epoch, per_bout["stop_idx"].to_numpy()
    )

    return per_bout[column_order]


def build_event_arm_table(
    event_table: pd.DataFrame,
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    conditional_posterior=None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """Plan Section 2.2: one row per (event, arm) — nine rows per event.

    Assumes ``event_table`` was produced by :func:`build_event_table`
    on the same ``dot`` / ``trials`` / ``track_graph``. Broadcasts
    event-level fields (including ``stop_id``, context flags, and
    event-level posterior columns) across arms and augments each
    (event, arm) row with the three arm-count responses, the binary
    arm indicators, spatial and reward predictors, and the interaction
    terms from plan §1.7.

    ``nominal_reward_prob`` is emitted as NaN — it requires the
    contingency-block loader that Phase B intentionally defers.

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
        Columns: see :func:`_event_arm_table_columns`. Empty (correct
        schema) if ``event_table`` is empty.
    """
    column_order = _event_arm_table_columns()
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
        arm_posterior = posterior_to_arm_marginal(
            post_in_event, edge_id, n_arms=n_arms
        )
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

    # arm_fraction: defined only when n_nonlocal_bins > 0.
    n_bins_long = long["n_nonlocal_bins"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            n_bins_long > 0,
            long["arm_count_hard"].to_numpy() / np.maximum(n_bins_long, 1),
            np.nan,
        )
    long["arm_fraction"] = frac

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
    long["arm_patch_id"] = np.array([_arm_to_patch(a) for a in arm_arr])
    long["is_same_patch"] = _eq_indicator(
        long["arm_patch_id"].to_numpy(), long["current_patch"].to_numpy()
    )

    # Per-event per-arm predictors: dist_to_arm, arm_dist_rank,
    # visit_recency, cumulative_reward, recent_reward_rate. Iterate
    # events (cost = n_events × n_arms × trials — dominated by the
    # trials scan; still acceptable at project scale).
    dist_vals = np.full(n_events * n_arms, np.nan)
    rank_vals = np.full(n_events * n_arms, np.nan)
    vr_vals = np.full(n_events * n_arms, np.nan)
    cr_vals = np.full(n_events * n_arms, np.nan)
    rrr_vals = np.full(n_events * n_arms, np.nan)

    # Use ``enumerate`` — not ``i`` from ``iterrows`` — so this loop
    # works regardless of what ``event_table``'s index looks like.
    # After a cross-session ``pd.concat`` (Phase B4) the index is no
    # longer a zero-based ``RangeIndex``; using the label as a
    # positional offset would silently scatter writes into the wrong
    # output slots or raise ``IndexError``.
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

        event_time = ev_row["event_midpoint_time"]
        if np.isnan(event_time) or len(trials) == 0:
            vr = dict.fromkeys(ALL_ARMS, np.nan)
            cr = dict.fromkeys(ALL_ARMS, np.nan)
            rrr = dict.fromkeys(ALL_ARMS, np.nan)
        else:
            vr = _visit_recency_per_arm(event_time, trials, track_graph)
            cr = _cumulative_reward_per_arm(event_time, trials, track_graph)
            rrr = _recent_reward_rate_per_arm(event_time, trials, track_graph)

        for j, a in enumerate(ALL_ARMS):
            idx = base + j
            dist_vals[idx] = dist_arr[j]
            rank_vals[idx] = ranks[j]
            vr_vals[idx] = vr[a]
            cr_vals[idx] = cr[a]
            rrr_vals[idx] = rrr[a]

    long["dist_to_arm"] = dist_vals
    long["arm_dist_rank"] = rank_vals
    long["visit_recency"] = vr_vals
    long["cumulative_reward"] = cr_vals
    long["recent_reward_rate"] = rrr_vals
    long["nominal_reward_prob"] = np.nan  # deferred; requires block loader

    # Interactions (plan §1.7). Event-level × arm-level products.
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
    long["mua_x_current"] = is_mua_i * ic
    long["mua_x_prev"] = is_mua_i * ip
    long["mua_x_next"] = is_mua_i * inx
    long["switch_x_current"] = is_after_switch * ic
    long["switch_x_prev"] = is_after_switch * ip
    long["omission_x_last_rewarded"] = is_after_omission * ilr

    return long[column_order]


# ── Helpers ──


def _safe_mode(series: pd.Series) -> float:
    """Mode of a series (first value on tie), ``NaN`` if empty or all NaN."""
    if len(series) == 0:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    mode = valid.mode()
    if len(mode) == 0:
        return np.nan
    return float(mode.iloc[0])


def _safe_first(series: pd.Series) -> float:
    """First non-NaN value of a series, ``NaN`` if empty or all NaN."""
    if len(series) == 0:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[0])


def _hash_event_id(events: pd.DataFrame) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, non_local_event)``."""
    keys = events[["animal", "day", "epoch", "non_local_event"]]
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def _hash_stop_id(
    animal: str,
    day: int,
    epoch: str,
    stop_idx: np.ndarray,
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, stop_idx)`` per bout."""
    keys = pd.DataFrame(
        {
            "animal": animal,
            "day": day,
            "epoch": epoch,
            "stop_idx": np.asarray(stop_idx, dtype=np.int64),
        }
    )
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def _align_count_matrix(
    counts: np.ndarray,
    ids_sorted: np.ndarray,
    target_ids: np.ndarray,
    n_arms: int,
) -> np.ndarray:
    """Reindex a ``(n_events_src, n_arms)`` count matrix to ``target_ids``.

    Missing source events (e.g., every in-event timebin on a junction
    bin) fill as zeros (appropriate for count responses) or NaN (for
    posterior-mass). Callers pass a float/int matrix and the fill is
    inferred from the dtype.
    """
    mapping = {int(i): k for k, i in enumerate(ids_sorted)}
    fill = np.nan if np.issubdtype(counts.dtype, np.floating) else 0
    out = np.full((target_ids.size, n_arms), fill, dtype=counts.dtype)
    for k, tid in enumerate(target_ids):
        src_idx = mapping.get(int(tid))
        if src_idx is not None:
            out[k] = counts[src_idx]
    return out


def _eq_indicator(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``int(values == reference)`` with NaN-in-reference = 0."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(int)
    out[np.isnan(reference)] = 0
    return out


def _visit_recency_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    n_default: int | None = None,
) -> dict[int, float]:
    """Trials since each arm was last visited (port of legacy helper).

    Direct port of ``predictors.compute_visit_recency_for_event``
    with one correctness tightening: when there are no past trials at
    ``event_time`` (session start or empty-trials stub), every arm
    returns ``NaN`` rather than ``n_default = n_trials + 1``. The
    legacy fallback coerced "never visited" to a finite "1 trial ago"
    for empty sessions — misleading for downstream GLMs that interpret
    ``visit_recency`` as a quantitative predictor.
    """
    past = trials[trials["end_time"] <= event_time]
    if len(past) == 0:
        return dict.fromkeys(ALL_ARMS, np.nan)
    n_trials = len(trials)
    if n_default is None:
        n_default = n_trials + 1
    result = {arm: float(n_default) for arm in ALL_ARMS}
    for trial_offset, trial in enumerate(reversed(list(past.itertuples()))):
        trials_ago = trial_offset + 1
        for well_col in ("to_well", "from_well"):
            well_id = getattr(trial, well_col)
            if np.isnan(well_id):
                continue
            arms = well_to_arm_set(well_id, track_graph)
            for arm in arms:
                if result[arm] == float(n_default):
                    result[arm] = float(trials_ago)
        if all(v != float(n_default) for v in result.values()):
            break
    return result


def _cumulative_reward_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    window: int = 10,
) -> dict[int, float]:
    """Reward count at each arm over the last ``window`` trials.

    Port of ``predictors.compute_cumulative_reward_for_event``.
    """
    past = trials[trials["end_time"] <= event_time].tail(window)
    result = dict.fromkeys(ALL_ARMS, 0.0)
    for trial in past.itertuples():
        if not trial.is_reward:
            continue
        well_id = trial.to_well
        if np.isnan(well_id):
            continue
        arms = well_to_arm_set(well_id, track_graph)
        for arm in arms:
            result[arm] += 1.0
    return result


def _recent_reward_rate_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    k: int = 5,
) -> dict[int, float]:
    """Fraction of the last ``k`` visits to each arm that were rewarded.

    New (not in legacy). A "visit to arm *a*" is a completed trial
    whose ``to_well`` touches arm *a*. Rate = fraction of the last
    *k* such trials that were rewarded. Returns ``NaN`` for arms with
    no past visits.
    """
    past = trials[trials["end_time"] <= event_time]
    if len(past) == 0:
        return dict.fromkeys(ALL_ARMS, np.nan)

    to_wells = past["to_well"].to_numpy()
    rewards = past["is_reward"].to_numpy()

    result: dict[int, float] = {}
    for arm in ALL_ARMS:
        # Cache-friendly per-trial check: is the arm adjacent to
        # to_well? Computing well_to_arm_set each row is cheap on this
        # graph (9 edges) but cached nevertheless via the inner loop.
        matches = np.array(
            [
                (not np.isnan(w)) and (arm in well_to_arm_set(w, track_graph))
                for w in to_wells
            ],
            dtype=bool,
        )
        arm_rewards = rewards[matches][-k:]
        result[arm] = (
            float(np.asarray(arm_rewards, dtype=float).mean())
            if arm_rewards.size > 0
            else np.nan
        )
    return result


def _assign_bout(
    event_times: np.ndarray,
    bout_starts: np.ndarray,
    bout_ends: np.ndarray,
) -> np.ndarray:
    """Return the bout index containing each timestamp, or ``-1`` if none.

    ``bout_starts`` and ``bout_ends`` must be sorted and the same length
    (inclusive intervals). For each ``t`` in ``event_times`` the result
    is the unique ``i`` with ``bout_starts[i] <= t <= bout_ends[i]``,
    or ``-1`` if no bout contains ``t`` (e.g., the timestamp fell in a
    mobile gap between bouts).
    """
    event_times = np.asarray(event_times, dtype=float)
    if bout_starts.size == 0:
        return np.full(event_times.shape, -1, dtype=np.int64)
    idx = np.searchsorted(bout_starts, event_times, side="right") - 1
    clipped = np.clip(idx, 0, bout_starts.size - 1)
    in_bout = (idx >= 0) & (event_times <= bout_ends[clipped])
    return np.where(in_bout, idx, -1).astype(np.int64)


def _event_table_columns() -> list[str]:
    """Canonical column order for ``event_table``.

    Kept central so the empty-session short-circuit returns the right
    shape and so downstream consumers can rely on column order.
    """
    return [
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
    ]


def _stop_table_columns() -> list[str]:
    """Canonical column order for ``stop_table``."""
    return [
        "animal",
        "day",
        "epoch",
        "stop_idx",
        "stop_id",
        "stop_start",
        "stop_end",
        "immobility_time",
        "is_swr_stop",
        "is_mua_stop",
        "has_replay",
        "n_events",
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
    ]


def _event_arm_table_columns() -> list[str]:
    """Canonical column order for ``event_arm_table``."""
    return [
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
        # Binary indicators
        "is_current",
        "is_prev",
        "is_next",
        "is_last_rewarded",
        "is_same_patch",
        # Spatial
        "dist_to_arm",
        "arm_dist_rank",
        # Reward / value
        "visit_recency",
        "cumulative_reward",
        "recent_reward_rate",
        "nominal_reward_prob",
        # Interactions (plan §1.7 + collinearity control)
        "current_x_last_rewarded",
        "time_x_current",
        "time_x_prev",
        "time_x_next",
        "swr_x_current",
        "swr_x_prev",
        "swr_x_next",
        "swr_x_last_rewarded",
        "mua_x_current",
        "mua_x_prev",
        "mua_x_next",
        "switch_x_current",
        "switch_x_prev",
        "omission_x_last_rewarded",
    ]


def _event_sequence_table_columns() -> list[str]:
    """Canonical column order for ``event_sequence_table``."""
    cols = [
        "animal",
        "day",
        "epoch",
        "non_local_event",
        "event_id",
        "t_in_event",
        "time",
        "nonlocal_prob",
        "decoded_arm_map",
    ]
    cols.extend(f"arm_posterior_{a}" for a in ALL_ARMS)
    return cols
