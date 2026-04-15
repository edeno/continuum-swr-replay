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

Canonical column names follow the plan's Section 1; the legacy names
(``non_local_event`` alongside ``event_id``, ``is_swr_flag``, ``replay_arm``,
``prev_trial_from_well`` etc.) still appear in other modules but are
slated for rename in a later Phase B commit.
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

from .spatial_mapping import map_estimate_to_arm, well_to_arm_set
from .summaries import n_nonlocal_bins, nonlocal_weight_sum

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
    - ``prev_trial_from_well``, ``prev_trial_to_well``
    - ``next_trial_to_well``
    - ``last_rewarded_well``
    - ``is_after_switch`` — current trip's ``is_patch_change``
    - ``is_after_omission`` — ``~current trip's is_reward``
    - ``is_after_reward`` — current trip's ``is_reward``
    - ``reward_outcome`` — current trip's ``is_reward`` as float
    """
    dot_times = dot.index.values
    n_trials = len(trials)
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

    dot["prev_trial_from_well"] = np.where(
        valid, from_wells[current_clipped], np.nan
    )
    dot["prev_trial_to_well"] = np.where(
        valid, to_wells[current_clipped], np.nan
    )
    dot["next_trial_to_well"] = np.where(
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
        return np.zeros_like(event_times)
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


# ── Public API ──


def build_event_table(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per non-local event with canonical event-level columns.

    Consumes the pre-saved ``{epoch}_HPC_{thr}_data_over_time.pkl`` frame
    (``dot``) plus the session's ``env`` / ``trials`` / ``track_graph``
    and emits the event-level slice of the plan's canonical schema.

    Chunk 1 scope: columns that can be derived from ``dot`` alone — no
    ``results.nc`` loading, no stop-level aggregation, no per-timebin
    conditional posterior. Subsequent commits add
    ``posterior_mass_total`` / ``posterior_peak_mean`` / ``posterior_entropy_mean``
    / ``posterior_hpd_width_{50,80,95}`` (chunk 2, needs ``results.nc``),
    ``stop_id`` / stop-level ``immobility_time`` (chunk 3,
    ``build_stop_table``), and ``block_number`` / ``trials_since_block_change``
    (requires contingency-block loader not yet wired).

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

    ``current_arm`` is the modal ``replay_arm`` of the animal's physical
    position during the event (from the decoder's map estimate on
    ``head_position``-derived track-segment labeling). Falls back to
    ``well_to_arm_set(prev_trial_to_well)`` when the modal position
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

    dot = dot.copy()
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"].to_numpy(), env)
    dot["nonlocal_prob"] = (
        dot["Non-Local_Continuous_prob"].to_numpy()
        + dot["Non-Local_Fragmented_prob"].to_numpy()
    )
    dot = _add_trial_context(dot, trials)

    # Per-session immobility bouts — computed from the full frame so
    # bout starts are correct even when an event begins partway through
    # a bout.
    bout_starts = _immobility_bout_starts(
        dot, speed_col="head_speed", speed_threshold=speed_threshold
    )

    # --- 2. Filter to in-event timebins ---
    in_event_mask = (
        (dot["non_local_event"] > 0)
        & (dot["head_speed"] < speed_threshold)
        & (dot["nonlocal_prob"] > nonlocal_threshold)
    )
    nl = dot.loc[in_event_mask]

    # Empty-output short-circuit — return a correctly-typed empty frame
    # so downstream concat does not blow up on column mismatch.
    column_order = _event_table_columns()
    if len(nl) == 0:
        return pd.DataFrame(columns=column_order)

    event_id_series = nl["non_local_event"].to_numpy()

    # --- 3. Per-event summaries via the shared summary helpers ---
    ids_sorted, n_bins = n_nonlocal_bins(
        nl["nonlocal_prob"].to_numpy(),
        event_id_series,
        threshold=nonlocal_threshold,
    )
    _, weight_sum = nonlocal_weight_sum(
        nl["nonlocal_prob"].to_numpy(), event_id_series
    )

    # --- 4. Per-event metadata (groupby once, iterate sorted ids) ---
    groups = nl.groupby("non_local_event", sort=True)
    event_records: list[dict] = []

    for ev_id, g in groups:
        times = g.index.to_numpy()
        arm_series = g["replay_arm"]
        real_arm_mask = arm_series.isin(ALL_ARMS)
        arms_real = arm_series[real_arm_mask]
        n_junction = int((~real_arm_mask).sum())

        # current_arm: modal track_segment_id (animal's physical position)
        ts = g["track_segment_id"] if "track_segment_id" in g.columns else pd.Series(dtype=float)
        ts_modal = _safe_mode(ts)
        # Fall back to prev_trial_to_well's arm if the modal segment is
        # not a leaf arm (trunk arms 6/7/8 cannot be an animal's well).
        if np.isnan(ts_modal) or int(ts_modal) not in ALL_ARMS:
            current_arm = _single_arm_from_well(
                _safe_first(g["prev_trial_to_well"]), track_graph
            )
        else:
            current_arm = float(ts_modal)

        prev_arm = _single_arm_from_well(
            _safe_first(g["prev_trial_from_well"]), track_graph
        )
        next_arm = _single_arm_from_well(
            _safe_first(g["next_trial_to_well"]), track_graph
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
                "n_junction_bins": n_junction,
                "n_unique_arms": int(arms_real.nunique()),
                "event_midpoint_time": float(np.mean(times)),
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

    # --- 7. Identifiers + stable hash ---
    per_event.insert(0, "animal", animal)
    per_event.insert(1, "day", day)
    per_event.insert(2, "epoch", epoch)
    per_event["event_id"] = _hash_event_id(per_event)

    return per_event[column_order]


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
        "current_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
        "n_nonlocal_bins",
        "nonlocal_weight_sum",
        "n_junction_bins",
        "n_unique_arms",
    ]
