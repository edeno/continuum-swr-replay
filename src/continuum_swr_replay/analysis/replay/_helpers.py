"""Shared helpers for event-table builders.

These utilities are used by both the NonLocal (``tables.py``) and
ContFrag (``contfrag/tables.py``) table builders. They handle
trial-context annotation, immobility-bout detection, spatial
arm/well/patch mappings, and generic DataFrame helpers.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .model_specs import ARM_FE_PREDICTORS
from .spatial_mapping import well_to_arm_set

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


# ── Trial-derived context ──


def add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-timebin trial-derived columns to ``dot`` (in place, returned).

    In-progress trip semantics: at each timepoint ``t``, the "current
    trip" is the latest trial whose ``start_time`` is at or before ``t``.
    This is the trip the animal is currently completing (or has just
    completed, if ``t`` falls in the gap between two trips).

    Columns added
    -------------
    - ``prev_well``, ``current_well``
    - ``next_well``
    - ``last_rewarded_well``
    - ``is_after_switch`` — current trip's ``is_patch_change``
    - ``is_after_omission`` — ``~current trip's is_reward``
    - ``is_after_reward`` — current trip's ``is_reward``
    - ``reward_outcome`` — current trip's ``is_reward`` as float
    """
    dot_times = dot.index.values
    n_trials = len(trials)

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
            "_goal_arrival_time",
            "_last_reward_time",
        ):
            dot[col] = np.nan
        return dot

    # ``np.searchsorted`` below assumes ``trial_start_times`` /
    # ``reward_timestamps`` are non-decreasing. The local data-loader
    # path produces sorted trials, but Spyglass-loaded or manually
    # passed frames are not protected — sort defensively so
    # ``current_well`` / ``last_rewarded_well`` never silently drift
    # from the true chronological order.
    trials = trials.sort_values("start_time").reset_index(drop=True)

    trial_start_times = trials["start_time"].values
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

    dot["prev_well"] = np.where(valid, from_wells[current_clipped], np.nan)
    dot["current_well"] = np.where(valid, to_wells[current_clipped], np.nan)
    dot["next_well"] = np.where(next_valid, to_wells[next_clipped], np.nan)

    # Use the same per-row goal-arrival cutoff convention as
    # ``_trial_cutoff_times``: ``goal_arrival_time`` when finite,
    # ``end_time`` when NaN, ``end_time`` everywhere when the column
    # is absent. The history helpers
    # (``visit_recency_all_events`` etc.) call ``_trial_cutoff_times``
    # for the same alignment — this consolidation makes the per-row
    # NaN fallback match between ``add_trial_context``'s reward /
    # goal-arrival writes and the history helpers' cutoffs.
    arrival_times = _trial_cutoff_times(trials)
    reward_timestamps = arrival_times[is_reward]
    rewarded_wells = to_wells[is_reward]
    if len(reward_timestamps) > 0:
        rew_idx = np.searchsorted(reward_timestamps, dot_times, side="right") - 1
        rew_idx_clipped = np.clip(rew_idx, 0, None)
        dot["last_rewarded_well"] = np.where(
            rew_idx >= 0, rewarded_wells[rew_idx_clipped], np.nan
        )
    else:
        dot["last_rewarded_well"] = np.nan

    dot["_goal_arrival_time"] = np.where(
        valid, arrival_times[current_clipped], np.nan
    )

    if len(reward_timestamps) > 0:
        dot["_last_reward_time"] = np.where(
            rew_idx >= 0, reward_timestamps[rew_idx_clipped], np.nan
        )
    else:
        dot["_last_reward_time"] = np.nan

    current_reward = np.where(valid, is_reward[current_clipped].astype(float), np.nan)
    dot["reward_outcome"] = current_reward
    dot["is_after_reward"] = current_reward
    dot["is_after_omission"] = np.where(valid, 1.0 - current_reward, np.nan)
    dot["is_after_switch"] = np.where(
        valid, is_patch_change[current_clipped].astype(float), np.nan
    )

    return dot


# ── Immobility-bout helpers ──


def immobility_bout_starts(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> np.ndarray:
    """Timestamps at which speed crossed from above to below the threshold."""
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    imm_times = dot.index.to_numpy()
    transitions = np.where(~imm[:-1] & imm[1:])[0] + 1
    if len(imm) > 0 and imm[0]:
        transitions = np.concatenate([[0], transitions])
    return imm_times[transitions]


def immobility_bout_intervals(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end timestamps for every contiguous immobility bout.

    Returns
    -------
    starts : np.ndarray, shape (n_bouts,)
    ends : np.ndarray, shape (n_bouts,)
    """
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    times = dot.index.to_numpy()
    if imm.size == 0 or not imm.any():
        return np.array([]), np.array([])

    padded = np.concatenate(([False], imm, [False]))
    diff = np.diff(padded.astype(np.int8))
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0] - 1
    return times[start_idx], times[end_idx]


def time_since_stop(
    event_times: np.ndarray,
    bout_start_times: np.ndarray,
) -> np.ndarray:
    """Seconds from most recent immobility-bout start to each event time."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_start_times.size == 0:
        return np.full_like(event_times, np.nan)
    idx = np.searchsorted(bout_start_times, event_times, side="right") - 1
    idx = np.clip(idx, 0, bout_start_times.size - 1)
    return np.clip(event_times - bout_start_times[idx], 0, None)


# ── Scalar extractions from per-arm sets ──


def single_arm_from_well(well: float, track_graph: nx.Graph) -> float:
    """Deterministic scalar arm for a well, ``NaN`` if the well is NaN or isolated."""
    if well is None or (isinstance(well, float) and np.isnan(well)):
        return np.nan
    arms = well_to_arm_set(well, track_graph)
    if len(arms) == 0:
        return np.nan
    return float(min(arms))


def arm_to_patch(arm: float) -> float:
    """Patch ID for an arm, ``NaN`` propagating."""
    if arm is None or (isinstance(arm, float) and np.isnan(arm)):
        return np.nan
    return float(TRACK_SEGMENT_TO_PATCH.get(int(arm), np.nan))


# ── Posterior alignment ──


def align_posterior_to_dot(
    conditional_posterior,
    dot: pd.DataFrame,
) -> np.ndarray:
    """Return a ``(T_dot, n_position_bins)`` posterior array aligned to ``dot.index``."""
    post_vals = np.asarray(conditional_posterior.values)
    if post_vals.shape[0] != len(dot):
        raise ValueError(
            f"conditional_posterior time axis has {post_vals.shape[0]} "
            f"entries but dot has {len(dot)}; the two must be aligned "
            "1-to-1 (both come from the same decoder run)."
        )
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


# ── Generic helpers ──


def safe_mode(series: pd.Series) -> float:
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


def safe_first(series: pd.Series) -> float:
    """First non-NaN value of a series, ``NaN`` if empty or all NaN."""
    if len(series) == 0:
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[0])


def strict_first(series: pd.Series) -> float:
    """First row's value as-is — NaN propagates.

    Use this when extracting event-level / bout-level context columns
    that come from ``add_trial_context``: those columns are NaN at every
    timebin before the first trial start. ``safe_first`` would silently
    skip leading NaNs and inherit the *next* trial's context, leaking
    future state into events that started before any trial existed.
    ``strict_first`` returns the first row's value as-is (NaN stays
    NaN), so before-trial events surface as missing context instead of
    fabricated post-hoc values.
    """
    if len(series) == 0:
        return np.nan
    v = series.iloc[0]
    return float(v) if pd.notna(v) else np.nan


def strict_mean(series: pd.Series) -> float:
    """``Series.mean`` with ``skipna=False``.

    pandas' default ``"mean"`` aggregator silently drops NaN, so a
    groupby reduction over a column that contains a mix of finite and
    NaN values emits the mean of the surviving rows. For Stage-5/6
    stop-level / arm-level aggregations that's a per-event survivorship
    bias: a stop with one finite event and one NaN event reports the
    finite event's value while ``n_events`` still counts both. Use
    this aggregator instead so any NaN propagates to the output.
    """
    return float(series.mean(skipna=False))


def strict_sum(series: pd.Series) -> float:
    """``Series.sum`` with ``skipna=False``."""
    return float(series.sum(skipna=False))


def hash_event_id(
    events: pd.DataFrame,
    event_col: str = "non_local_event",
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, event_col)``."""
    keys = events[["animal", "day", "epoch", event_col]]
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def validate_event_ids(event_table: pd.DataFrame) -> None:
    """Raise if ``event_id`` has hash collisions after cross-session concat.

    Call this after ``pd.concat`` across sessions to guard against the
    (unlikely but non-zero) birthday-paradox collision in 63-bit hashes.
    """
    n_unique = event_table["event_id"].nunique()
    n_total = len(event_table)
    if n_unique != n_total:
        raise RuntimeError(
            f"event_id hash collision: {n_total} rows but {n_unique} unique IDs"
        )


def hash_stop_id(
    animal: str,
    day: int,
    epoch: str,
    stop_idx: np.ndarray,
    *,
    brain_area: str | None = None,
    event_source: str | None = None,
) -> np.ndarray:
    """63-bit stable hash of ``(animal, day, epoch, stop_idx)`` per bout.

    Optional ``brain_area`` / ``event_source`` keywords are mixed into
    the hash when supplied. ContFrag callers pass them so SWR/MUA and
    HPC/mPFC/OFC stop tables never collide on ``stop_id``. The
    NonLocal pipeline omits both (single-area, single-source by
    construction) and keeps its existing hashes intact.
    """
    keys = pd.DataFrame(
        {
            "animal": animal,
            "day": day,
            "epoch": epoch,
            "stop_idx": np.asarray(stop_idx, dtype=np.int64),
        }
    )
    if brain_area is not None:
        keys["__brain_area"] = brain_area
    if event_source is not None:
        keys["__event_source"] = event_source
    return (pd.util.hash_pandas_object(keys, index=False) % (2**63)).to_numpy()


def assign_bout(
    event_times: np.ndarray,
    bout_starts: np.ndarray,
    bout_ends: np.ndarray,
) -> np.ndarray:
    """Return the bout index containing each timestamp, or ``-1`` if none."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_starts.size == 0:
        return np.full(event_times.shape, -1, dtype=np.int64)
    idx = np.searchsorted(bout_starts, event_times, side="right") - 1
    clipped = np.clip(idx, 0, bout_starts.size - 1)
    in_bout = (idx >= 0) & (event_times <= bout_ends[clipped])
    return np.where(in_bout, idx, -1).astype(np.int64)


def eq_indicator(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``int(values == reference)`` with NaN-in-reference = 0."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(int)
    out[np.isnan(reference)] = 0
    return out


def align_count_matrix(
    counts: np.ndarray,
    ids_sorted: np.ndarray,
    target_ids: np.ndarray,
    n_arms: int,
) -> np.ndarray:
    """Reindex a ``(n_events_src, n_arms)`` count matrix to ``target_ids``."""
    mapping = {int(i): k for k, i in enumerate(ids_sorted)}
    fill = np.nan if np.issubdtype(counts.dtype, np.floating) else 0
    out = np.full((target_ids.size, n_arms), fill, dtype=counts.dtype)
    for k, tid in enumerate(target_ids):
        src_idx = mapping.get(int(tid))
        if src_idx is not None:
            out[k] = counts[src_idx]
    return out


def add_arm_fixed_effects(
    df: pd.DataFrame,
    arm_col: str = "arm",
) -> pd.DataFrame:
    """Add the arm fixed-effect dummy columns named by ``ARM_FE_PREDICTORS``.

    ``arm_fe_k`` is ``1`` where ``df[arm_col] == k`` and ``0`` otherwise.
    The dropped reference arm (the ninth) is absent from
    ``ARM_FE_PREDICTORS`` — including all nine dummies would be
    collinear with the conditional-Poisson per-event normalization.

    The columns are added in place and ``df`` is returned (matching
    the in-place contract of :func:`add_trial_context`). Existing
    ``arm_fe_*`` columns are overwritten.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format event-arm frame. Must carry ``arm_col``.
    arm_col : str
        Integer arm-id column (``"arm"`` in the canonical event-arm
        table, ``"arm_id"`` in simulator output).

    Returns
    -------
    pd.DataFrame
        ``df`` with the dummy columns added.
    """
    if arm_col not in df.columns:
        raise KeyError(f"add_arm_fixed_effects: missing arm column {arm_col!r}")
    arm = df[arm_col].to_numpy()
    for k, col in enumerate(ARM_FE_PREDICTORS):
        df[col] = (arm == k).astype(np.int8)
    return df


# ── Per-arm reward/visit history helpers ──


def build_well_to_arm_map(track_graph: nx.Graph) -> dict[int, int]:
    """Precompute well -> arm mapping from the track graph.

    On the real bandit graph each well (degree-1 node) maps to exactly
    one arm (edge).  Building this once avoids repeated graph traversals
    inside the per-event helpers.
    """
    mapping: dict[int, int] = {}
    for u, v, data in track_graph.edges(data=True):
        eid = data.get("edge_id")
        if eid is None:
            continue
        for node in (u, v):
            if track_graph.degree(node) == 1:
                mapping[node] = eid
    return mapping


# ── Vectorized batch helpers (operate on all events at once) ──


def _trial_cutoff_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial cutoff timestamp for "is this trial past relative to an event?".

    Uses ``goal_arrival_time`` (the moment the animal arrives at and
    is rewarded at the goal well) when available, falling back to
    ``end_time`` per-row when ``goal_arrival_time`` is NaN, and
    per-frame when the column is absent. This matches the convention
    already used by :func:`add_trial_context` for ``last_rewarded_well``
    and the ``_goal_arrival_time`` / ``_last_reward_time`` columns.

    Why this matters: between ``goal_arrival_time`` and ``end_time``
    of the same trial the animal is sitting at the well in the
    post-arrival stop. Cutting at ``end_time`` makes a replay event
    during that stop see ``last_rewarded_well`` set to the just-
    completed well (because that uses ``goal_arrival_time``) while
    ``visit_recency`` / ``cumulative_reward`` / ``recent_reward_rate``
    haven't counted the same arrival yet — value history that lags
    the reward-context columns by the post-arrival window. Using
    ``goal_arrival_time`` uniformly aligns all four trial-derived
    columns onto the same temporal cutoff.
    """
    if "goal_arrival_time" in trials.columns:
        cutoff = trials["goal_arrival_time"].to_numpy(dtype=float)
        end = trials["end_time"].to_numpy(dtype=float)
        missing = ~np.isfinite(cutoff)
        return np.where(missing, end, cutoff)
    return trials["end_time"].to_numpy(dtype=float)


def visit_recency_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    n_default: int | None = None,
) -> np.ndarray:
    """Trials-ago since each arm was last visited, for all events.

    Returns ``(n_events, n_arms)`` array with the same semantics as
    :func:`visit_recency_per_arm` but computed in a single vectorized
    pass.
    """
    n_arms = len(ALL_ARMS)
    arm_idx = {a: j for j, a in enumerate(ALL_ARMS)}
    n_events = len(event_times)
    n_trials_total = len(trials)

    if n_default is None:
        n_default = n_trials_total + 1
    default_f = float(n_default)

    if n_trials_total == 0:
        return np.full((n_events, n_arms), np.nan)

    # ``np.searchsorted`` below assumes the cutoff times are non-decreasing.
    # Sort defensively so chronological-order assumptions hold for any
    # caller (Spyglass-loaded frames are not guaranteed to be sorted).
    # Use ``goal_arrival_time`` (with ``end_time`` fallback) as the
    # cutoff so this helper aligns with ``add_trial_context`` —
    # see ``_trial_cutoff_times`` for the rationale.
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()

    # For each trial, record which arm was visited (to_well = destination).
    # Uses to_well only, consistent with cumulative_reward_all_events.
    # A "visit" is arriving at a well, not departing from one.
    arm_visit_indices: list[np.ndarray] = [np.empty(0, dtype=np.intp) for _ in ALL_ARMS]
    for ti in range(n_trials_total):
        w = to_wells[ti]
        if np.isnan(w):
            continue
        arm_id = well_to_arm.get(int(w))
        if arm_id is not None and arm_id in arm_idx:
            j = arm_idx[arm_id]
            arm_visit_indices[j] = np.append(arm_visit_indices[j], ti)

    # cutoff[i] = number of trials with cutoff_time <= event_times[i]
    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")

    out = np.full((n_events, n_arms), default_f)

    # Events with no past trials -> NaN
    no_past = cutoffs == 0
    out[no_past] = np.nan

    for j in range(n_arms):
        visits = arm_visit_indices[j]
        if visits.size == 0:
            continue
        # For each event, find how many of this arm's visits are before cutoff
        n_vis = np.searchsorted(visits, cutoffs, side="left")
        # The last visit before cutoff is at visits[n_vis - 1]
        has_visit = (~no_past) & (n_vis > 0)
        last_visit_trial_idx = np.where(
            has_visit, visits[np.clip(n_vis - 1, 0, visits.size - 1)], -1
        )
        # trials_ago = cutoff - last_visit_trial_idx
        trials_ago = np.where(has_visit, cutoffs - last_visit_trial_idx, default_f)
        out[has_visit, j] = trials_ago[has_visit].astype(float)

    return out


def cumulative_reward_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    window: int = 10,
) -> np.ndarray:
    """Reward count per arm over the last ``window`` trials, for all events.

    Returns ``(n_events, n_arms)`` array.
    """
    n_arms = len(ALL_ARMS)
    arm_idx = {a: j for j, a in enumerate(ALL_ARMS)}
    n_events = len(event_times)
    n_trials_total = len(trials)

    if n_trials_total == 0:
        # No completed trials → unknown history, not zero reward
        # (matches the sibling helpers).
        return np.full((n_events, n_arms), np.nan)

    # Sort defensively by the goal-arrival cutoff (see
    # ``visit_recency_all_events`` and ``_trial_cutoff_times``).
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()
    is_reward = trials["is_reward"].to_numpy().astype(bool)

    # Build per-arm cumulative reward arrays (cumsum over trial order)
    # reward_at[j][ti] = 1 if trial ti visits arm j and is rewarded
    reward_per_arm = np.zeros((n_trials_total, n_arms), dtype=float)
    for ti in range(n_trials_total):
        if not is_reward[ti]:
            continue
        w = to_wells[ti]
        if np.isnan(w):
            continue
        arm_id = well_to_arm.get(int(w))
        if arm_id is not None and arm_id in arm_idx:
            reward_per_arm[ti, arm_idx[arm_id]] = 1.0

    # cumsum with a leading zero row for easy slicing
    cum = np.zeros((n_trials_total + 1, n_arms), dtype=float)
    np.cumsum(reward_per_arm, axis=0, out=cum[1:])

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")
    window_starts = np.maximum(cutoffs - window, 0)

    out = cum[cutoffs] - cum[window_starts]

    # No-past-trials → NaN, matching the sibling helpers' contract.
    no_past = cutoffs == 0
    if no_past.any():
        out[no_past] = np.nan
    return out


def recent_reward_rate_all_events(
    event_times: np.ndarray,
    trials: pd.DataFrame,
    well_to_arm: dict[int, int],
    k: int = 5,
) -> np.ndarray:
    """Fraction of last ``k`` visits to each arm that were rewarded, for all events.

    Returns ``(n_events, n_arms)`` array.

    Trunk arms (graph edges that touch no well — never a trial
    destination, so never a reward site) are filled with ``0.0``
    rather than ``NaN``: their reward rate is structurally zero, not
    unknown. This matters because the conditional-Poisson fitter
    drops an entire event if *any* of its nine arm rows is NaN, and
    every event has three trunk-arm rows — leaving them NaN would
    annihilate the M3+ event set. A leaf arm that simply has not been
    visited yet in a session *with* prior trials stays ``NaN`` (a
    genuine unknown); the no-trial-history rows stay ``NaN`` too,
    matching the sibling helpers' contract.
    """
    n_arms = len(ALL_ARMS)
    n_events = len(event_times)
    n_trials_total = len(trials)
    # Arms reachable as a trial destination = leaf arms (values of the
    # well→arm map). Anything else is a trunk arm.
    reachable_arms = set(well_to_arm.values())

    if n_trials_total == 0:
        return np.full((n_events, n_arms), np.nan)

    # Sort defensively by the goal-arrival cutoff (see
    # ``visit_recency_all_events`` and ``_trial_cutoff_times``).
    trials = trials.assign(_cutoff=_trial_cutoff_times(trials))
    trials = trials.sort_values("_cutoff").reset_index(drop=True)
    cutoff_times = trials["_cutoff"].to_numpy()
    to_wells = trials["to_well"].to_numpy()
    is_reward = trials["is_reward"].to_numpy().astype(float)

    # Per-arm: arrays of visit trial indices and corresponding rewards
    arm_visits: list[np.ndarray] = []
    arm_rewards: list[np.ndarray] = []
    for arm in ALL_ARMS:
        vis_idx = []
        vis_rew = []
        for ti in range(n_trials_total):
            w = to_wells[ti]
            if np.isnan(w):
                continue
            aid = well_to_arm.get(int(w))
            if aid == arm:
                vis_idx.append(ti)
                vis_rew.append(is_reward[ti])
        arm_visits.append(np.array(vis_idx, dtype=np.intp))
        arm_rewards.append(np.array(vis_rew, dtype=float))

    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")

    # Events with no past trials -> NaN
    no_past = cutoffs == 0
    out = np.full((n_events, n_arms), np.nan)

    for j in range(n_arms):
        # Trunk arm: reward rate is structurally 0 wherever there is
        # any trial history. ``no_past`` rows stay NaN.
        if ALL_ARMS[j] not in reachable_arms:
            out[~no_past, j] = 0.0
            continue
        visits = arm_visits[j]
        rewards = arm_rewards[j]
        if visits.size == 0:
            continue
        # n_vis_before[i] = number of visits to this arm before cutoff
        n_vis_before = np.searchsorted(visits, cutoffs, side="left")
        for i in range(n_events):
            if no_past[i]:
                continue
            nv = n_vis_before[i]
            if nv == 0:
                continue
            start = max(0, nv - k)
            out[i, j] = rewards[start:nv].mean()

    return out


def recent_trial_reward_rate(
    query_times: np.ndarray,
    trials: pd.DataFrame,
    k: int = 10,
) -> np.ndarray:
    """Rewarded fraction of the most recent ``k`` trials before each query time.

    The arm-agnostic counterpart of :func:`recent_reward_rate_all_events`:
    the denominator is trials, not visits to one arm. A trial is "before"
    a query time if its goal-arrival cutoff (see :func:`_trial_cutoff_times`)
    is at or before that time.

    Parameters
    ----------
    query_times : np.ndarray, shape (n_query,)
        Times (s) at which to evaluate the recent reward rate — e.g.
        immobility-bout start times for the stop-level rate model.
    trials : pd.DataFrame
        Must carry ``is_reward`` plus the columns ``_trial_cutoff_times``
        consumes.
    k : int
        Window length in trials.

    Returns
    -------
    np.ndarray, shape (n_query,)
        Rewarded fraction of the last ``k`` trials; ``NaN`` for query
        times that precede every trial cutoff.
    """
    query_times = np.asarray(query_times, dtype=float)
    if len(trials) == 0:
        return np.full(query_times.size, np.nan)

    # Sort by the goal-arrival cutoff so ``searchsorted`` is valid; see
    # ``_trial_cutoff_times`` for why this cutoff (not ``end_time``).
    cutoff_times = np.sort(_trial_cutoff_times(trials))
    is_reward = (
        trials.assign(_cutoff=_trial_cutoff_times(trials))
        .sort_values("_cutoff")["is_reward"]
        .to_numpy()
        .astype(float)
    )
    cum = np.concatenate([[0.0], np.cumsum(is_reward)])

    counts = np.searchsorted(cutoff_times, query_times, side="right")
    window_starts = np.maximum(counts - k, 0)
    n_in_window = counts - window_starts
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (cum[counts] - cum[window_starts]) / n_in_window
    rate[counts == 0] = np.nan
    return rate


# ── Scalar per-event wrappers (preserve existing API) ──


def visit_recency_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    n_default: int | None = None,
) -> dict[int, float]:
    """Trials since each arm was last visited."""
    w2a = build_well_to_arm_map(track_graph)
    row = visit_recency_all_events(
        np.array([event_time]), trials, w2a, n_default=n_default
    )
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def cumulative_reward_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    window: int = 10,
) -> dict[int, float]:
    """Reward count at each arm over the last ``window`` trials."""
    w2a = build_well_to_arm_map(track_graph)
    row = cumulative_reward_all_events(
        np.array([event_time]), trials, w2a, window=window
    )
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}


def recent_reward_rate_per_arm(
    event_time: float,
    trials: pd.DataFrame,
    track_graph: nx.Graph,
    k: int = 5,
) -> dict[int, float]:
    """Fraction of the last ``k`` visits to each arm that were rewarded."""
    w2a = build_well_to_arm_map(track_graph)
    row = recent_reward_rate_all_events(np.array([event_time]), trials, w2a, k=k)
    return {arm: float(row[0, j]) for j, arm in enumerate(ALL_ARMS)}
