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

    dot["_goal_arrival_time"] = np.where(valid, arrival_times[current_clipped], np.nan)

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
    ``visit_recency`` / ``cumulative_reward`` / ``reward_rate_arm_k10``
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


_SESSION_COLS = ["animal", "day", "epoch"]


# ── Per-bin event-rate aggregation ──


def attach_is_swr_to_onsets(
    onset_df: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add a canonical ``is_swr`` column to the non-local onset rows.

    Matches each non-local onset to its ``event_table`` event by within-stop
    time rank (both are ordered by time within a stop and are 1:1 -- the
    non-local onset count equals the event-table row count). SWR/MUA onset
    rows get ``is_swr = NaN``. Returns a copy; does not mutate the input.

    Parameters
    ----------
    onset_df : pd.DataFrame
        Event-onset rows. Must carry ``stop_id``, ``event_type``, and
        ``onset_time``.
    event_table : pd.DataFrame
        Per-event table. Must carry ``stop_id``, ``event_start``, ``is_swr``.

    Returns
    -------
    pd.DataFrame
        Copy of ``onset_df`` with an added ``is_swr`` column (the matched
        boolean for ``event_type == "non_local"`` rows, ``NaN`` elsewhere).
    """
    out = onset_df.copy()
    # object dtype holds the matched booleans alongside NaN for SWR/MUA rows
    # without the float coercion pandas now warns about.
    out["is_swr"] = pd.Series(np.nan, index=out.index, dtype="object")
    ev = event_table[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = out[out["event_type"] == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    merged = nl.merge(
        ev[["stop_id", "rank", "is_swr"]],
        on=["stop_id", "rank"],
        how="left",
        suffixes=("", "_ev"),
        # duplicate (stop_id, rank) keys would expand rows and misalign the
        # positional assignment below; the cumcount ranks are unique by
        # construction, so this is cheap insurance against that silent failure.
        validate="one_to_one",
    )
    # The rank-join only assigns the correct is_swr if the onset table and
    # event_table carry the same per-stop non-local events. A handful of
    # boundary onsets legitimately fail to match (~0.1%), but a gross mismatch
    # means the two were built from divergent non-local/speed thresholds or
    # decodes -- fail loud rather than silently mislabel is_swr downstream.
    matched = merged["is_swr_ev"].notna()
    if len(merged) and matched.mean() < 0.99:
        raise ValueError(
            "attach_is_swr_to_onsets: only "
            f"{matched.mean():.1%} of non-local onsets matched an event_table "
            "row by (stop_id, within-stop rank). The onset table and "
            "event_table must be built from the same non-local/speed thresholds "
            "and decodes."
        )
    out.loc[nl.index, "is_swr"] = merged["is_swr_ev"].to_numpy()
    return out


def raised_cosine_history_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Log-spaced raised-cosine self-history basis (Pillow et al. 2008).

    Smooth, overlapping bumps over within-stop lags ``1..n_lags`` used to encode
    the self-history block. Projecting the ``onset_lag{j}`` counts onto these
    ``n_basis`` bumps replaces ``n_lags`` collinear lag dummies that otherwise
    perfectly separate on the refractory bins (coefficients -> -inf) and wreck
    held-out prediction.

    Parameters
    ----------
    n_lags : int
        Number of within-stop lags spanned (the history window in bins).
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        ``B[k, j]`` is the weight of lag ``j+1`` in basis function ``k``.
    """
    x = np.log(np.arange(1, n_lags + 1))
    centers = np.linspace(x.min(), x.max(), n_basis)
    spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
    B = np.zeros((n_basis, n_lags))
    for k, c in enumerate(centers):
        arg = np.clip((x - c) * np.pi / (2 * spacing), -np.pi, np.pi)
        B[k] = 0.5 * (1 + np.cos(arg))
    return B


def aggregate_to_timebins(
    stop_table: pd.DataFrame,
    onset_times: pd.DataFrame,
    *,
    bin_s: float,
    tmax: float,
    reward_lags: int = 0,
    self_history_lags: int = 0,
    self_history_basis: int = 0,
    cross_onset_times: pd.DataFrame | None = None,
    cross_history_lags: int = 0,
    cross_history_basis: int = 0,
    stop_covariates: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One row per (stop, bin) up to ``tmax``, at ``bin_s`` resolution.

    Builds the canonical (stop x bin) point-process design used by the per-bin
    event-rate models. Bins are built from ``stop_table.immobility_time``
    (continuous), independent of any dense 4 ms grid, so non-4-ms widths (e.g.
    15 ms) are exact. The last bin's ``exposure`` is the partial remainder.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table (unique ``stop_id``). Must carry ``animal``, ``day``,
        ``epoch``, ``stop_id``, ``immobility_time``, plus any ``reward_lag{j}``
        and ``stop_covariates`` columns requested.
    onset_times : pd.DataFrame
        Event-onset rows for ONE population (already filtered). Must carry
        ``stop_id`` and ``t_since_stop_onset``.
    bin_s : float
        Bin width in seconds.
    tmax : float
        Right-truncation; bins beyond ``tmax`` are dropped.
    reward_lags : int
        Number of ``reward_lag{j}`` columns to carry from ``stop_table``.
    self_history_lags : int
        Number of ``onset_lag{j}`` self-history columns to build (same-
        population onset counts ``j`` bins back, within stop, reset at the
        stop boundary).
    self_history_basis : int
        If > 0, also add ``hist_rcos_0..self_history_basis-1`` columns: the
        ``onset_lag`` counts projected onto a log-spaced raised-cosine basis
        (see :func:`raised_cosine_history_basis`). Requires ``self_history_lags
        > 0`` (the basis spans that window). Used by the ``*_histcos`` models,
        which generalize where the lag dummies overfit the refractory bins.
    stop_covariates : tuple[str, ...]
        Extra per-stop columns carried from ``stop_table`` by ``stop_id``.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal``, ``day``, ``epoch``, ``stop_id``, ``binj``,
        ``tc`` (bin-center s = time since stop onset), ``exposure`` (s of
        immobility in the bin), ``reward_lag1..reward_lags``, each name in
        ``stop_covariates``, ``count`` (population onsets in the bin),
        ``onset_lag1..self_history_lags``, and (if ``self_history_basis``)
        ``hist_rcos_0..self_history_basis-1``.
    """
    if self_history_basis and not self_history_lags:
        raise ValueError("self_history_basis requires self_history_lags > 0")
    if cross_history_basis and not cross_history_lags:
        raise ValueError("cross_history_basis requires cross_history_lags > 0")
    if cross_history_lags and cross_onset_times is None:
        raise ValueError("cross_history_lags requires cross_onset_times")
    reward_cols = [f"reward_lag{j}" for j in range(1, reward_lags + 1)]
    carry = reward_cols + list(stop_covariates)

    s = stop_table[stop_table["immobility_time"] > 0]
    # ceil, not int: when tmax is not a multiple of bin_s, int() drops the final
    # partial window [floor(tmax/bin_s)*bin_s, tmax) so onsets there are lost.
    nmax = int(np.ceil(tmax / bin_s))
    nbin = np.minimum(np.ceil(s["immobility_time"].to_numpy() / bin_s), nmax).astype(
        int
    )
    idx = np.repeat(np.arange(len(s)), nbin)
    binj = (
        np.concatenate([np.arange(n) for n in nbin])
        if len(s)
        else np.empty(0, dtype=int)
    )
    cols = [*_SESSION_COLS, "stop_id", "immobility_time", *carry]
    out = s.iloc[idx][cols].reset_index(drop=True)
    out["binj"] = binj
    out["tc"] = (binj + 0.5) * bin_s
    # bin width, clipped by both the stop's immobility end and tmax (the final
    # bin of a stop longer than tmax is exposed only up to tmax, not bin_s).
    out["exposure"] = np.minimum(
        bin_s, np.minimum(out["immobility_time"], tmax) - binj * bin_s
    )
    out = out[out["exposure"] > 1e-4]

    src = onset_times[onset_times["t_since_stop_onset"] < tmax]
    # clip into the design's bin range so an onset in the final partial window
    # [(_nmax-1)*bin_s, tmax) lands in the last bin rather than a phantom bin
    # that the merge below would silently drop.
    src_binj = np.clip(
        (src["t_since_stop_onset"].to_numpy() / bin_s).astype(int), 0, nmax - 1
    )
    cnt = (
        pd.DataFrame({"stop_id": src["stop_id"].to_numpy(), "binj": src_binj})
        .groupby(["stop_id", "binj"])
        .size()
        .rename("count")
    )
    out = out.merge(cnt, on=["stop_id", "binj"], how="left")
    out["count"] = out["count"].fillna(0)

    if self_history_lags:
        out = out.sort_values(["stop_id", "binj"])
        grouped = out.groupby("stop_id")["count"]
        for j in range(1, self_history_lags + 1):
            out[f"onset_lag{j}"] = grouped.shift(j).fillna(0)

    if self_history_basis:
        B = raised_cosine_history_basis(self_history_lags, self_history_basis)
        lag_cols = [f"onset_lag{j}" for j in range(1, self_history_lags + 1)]
        proj = out[lag_cols].to_numpy() @ B.T
        for k in range(self_history_basis):
            out[f"hist_rcos_{k}"] = proj[:, k]

    # Cross-population history: lagged onset counts of a different event stream
    # (e.g. with-SWR non-local onsets as predictors of the non-SWR rate). Binned
    # on the SAME grid as the response, shifted within stop (lag >= 1 -> only the
    # cross stream's past predicts the current bin), reset at the stop boundary.
    if cross_history_lags:
        xsrc = cross_onset_times[cross_onset_times["t_since_stop_onset"] < tmax]
        xbinj = np.clip(
            (xsrc["t_since_stop_onset"].to_numpy() / bin_s).astype(int), 0, nmax - 1
        )
        xcnt = (
            pd.DataFrame({"stop_id": xsrc["stop_id"].to_numpy(), "binj": xbinj})
            .groupby(["stop_id", "binj"])
            .size()
            .rename("cross_count")
        )
        out = out.merge(xcnt, on=["stop_id", "binj"], how="left")
        out["cross_count"] = out["cross_count"].fillna(0)
        out = out.sort_values(["stop_id", "binj"])
        xgrouped = out.groupby("stop_id")["cross_count"]
        for j in range(1, cross_history_lags + 1):
            out[f"cross_lag{j}"] = xgrouped.shift(j).fillna(0)
        if cross_history_basis:
            Bx = raised_cosine_history_basis(cross_history_lags, cross_history_basis)
            xlag_cols = [f"cross_lag{j}" for j in range(1, cross_history_lags + 1)]
            xproj = out[xlag_cols].to_numpy() @ Bx.T
            for k in range(cross_history_basis):
                out[f"cross_hist_rcos_{k}"] = xproj[:, k]
        out = out.drop(columns=["cross_count"])

    out = out.drop(columns=["immobility_time"])
    return out.reset_index(drop=True)


# given onset_df already passed through attach_is_swr_to_onsets:
POPULATIONS = {
    "swr": lambda o: o[o.event_type == "SWR"],
    "non_local": lambda o: o[o.event_type == "non_local"],
    "non_local_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == True)  # noqa: E712
    ],
    "non_local_no_swr": lambda o: o[
        (o.event_type == "non_local") & (o.is_swr == False)  # noqa: E712
    ],
}


# ── Scalar per-event wrappers (preserve existing API) ──


# Behavioral covariate definitions were extracted to covariates.py; they are
# re-exported here so existing ``from ..._helpers import <name>`` call sites keep
# resolving. covariates.py imports _trial_cutoff_times / build_well_to_arm_map from
# this module, so _helpers must finish loading first (it does: the package __init__
# imports _helpers before covariates).
from .covariates import (  # noqa: E402,F401
    UNCERTAINTY_COVARIATES,
    _leaky_q_value_diff,
    _omission_streak,
    add_uncertainty_covariates,
    attach_arm_uncertainty,
    attach_contingency_value_block,
    cumulative_reward_all_events,
    cumulative_reward_per_arm,
    lagged_trial_rewards,
    recent_reward_rate_all_events,
    recent_reward_rate_per_arm,
    recent_trial_reward_rate,
    visit_recency_all_events,
    visit_recency_per_arm,
)
