"""Shared helpers for event-table builders.

These utilities are used by both the NonLocal (``tables.py``) and
ContFrag (``contfrag/tables.py``) table builders. Trial clocks, behavioral
context, and behavioral-history covariates live in :mod:`.covariates`.
"""

from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.session_keys import SESSION_COLUMNS

from .model_specs import (
    ARM_FE_PREDICTORS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from .spatial_mapping import well_to_arm_set

logger = logging.getLogger(__name__)
_SESSION_COLS = list(SESSION_COLUMNS)

#: Per-lag known-outcome columns the reward-history basis is built over.
_REWARD_LAG_COLUMNS = tuple(
    f"known_reward_outcome_lag_{j}" for j in range(1, REWARD_HISTORY_N_LAGS + 1)
)
#: Stop columns needed to place a bin on the arrival / outcome-availability clocks.
_ARRIVAL_CLOCK_STOP_COLUMNS = (
    "stop_start",
    "last_well_arrival_time_s",
    "outcome_known_time_s",
)
#: Extra stop columns needed to re-derive the reward history at the outcome boundary.
_OUTCOME_SHIFT_STOP_COLUMNS = (
    "stop_start",
    "outcome_known_time_s",
    "completed_trial_reward",
    *_REWARD_LAG_COLUMNS,
)
#: Stop columns read only at STOP level, to find the bouts that span an arrival.
#: They are never carried per bin (the per-bin design has its own clocks).
_ARRIVAL_SPAN_STOP_COLUMNS = (
    "stop_start",
    "stop_end",
    "last_well_arrival_time_s",
)


def nearest_well_distance_cm(
    dot: pd.DataFrame,
    track_graph: nx.Graph,
) -> np.ndarray:
    """Euclidean distance from each projected position to the nearest well.

    Wells are the degree-one nodes of ``track_graph``.  A malformed track
    graph raises: an all-NaN column is indistinguishable from "the animal
    was never near a well", and would silently empty any downstream
    near-a-well cohort for the whole session.  Missing position *data*
    (absent coordinate columns, individual non-finite rows) is a genuine
    data gap and still yields NaN.

    Parameters
    ----------
    dot : pd.DataFrame
        Timebin frame; uses ``projected_x_position`` / ``projected_y_position``.
    track_graph : nx.Graph
        Track graph whose degree-one nodes are wells, each with a ``pos``
        attribute holding a finite ``(x, y)`` pair.

    Returns
    -------
    np.ndarray, shape (len(dot),)
        Distance in cm, NaN where the position is unavailable.

    Raises
    ------
    ValueError
        If the graph has no degree-one nodes, or any well lacks a finite
        two-element ``pos``.
    """
    n_rows = len(dot)
    coordinate_columns = ("projected_x_position", "projected_y_position")
    if not set(coordinate_columns).issubset(dot.columns):
        return np.full(n_rows, np.nan)

    well_nodes = [node for node in track_graph.nodes if track_graph.degree(node) == 1]
    if not well_nodes:
        raise ValueError(
            "track_graph has no degree-one nodes, so no wells can be located"
        )
    node_positions = nx.get_node_attributes(track_graph, "pos")
    missing_pos = [node for node in well_nodes if node not in node_positions]
    if missing_pos:
        raise ValueError(
            f"track_graph well nodes lack a 'pos' attribute: {missing_pos}"
        )

    try:
        well_positions = np.asarray(
            [node_positions[node] for node in well_nodes], dtype=float
        )
    except (TypeError, ValueError) as err:
        raise ValueError(
            "track_graph well 'pos' attributes are not numeric (x, y) pairs"
        ) from err
    if (
        well_positions.shape != (len(well_nodes), 2)
        or not np.isfinite(well_positions).all()
    ):
        raise ValueError(
            "track_graph well 'pos' attributes must be finite two-element "
            f"(x, y) pairs; got shape {well_positions.shape}"
        )

    positions = dot.loc[:, coordinate_columns].to_numpy(dtype=float)
    output = np.full(n_rows, np.nan)
    valid = np.isfinite(positions).all(axis=1)
    if valid.any():
        displacement = positions[valid, None, :] - well_positions[None, :, :]
        output[valid] = np.linalg.norm(displacement, axis=2).min(axis=1)
    return output


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
    """63-bit stable hash of ``(animal, <session date>, epoch, event_col)``.

    The session-date column is read by value, so the hash is identical whether
    the frame spells it ``session_date`` (canonical, repository-owned tables) or
    ``day`` (the intermediate classified event table, not covariate-stamped).
    ``pd.util.hash_pandas_object`` ignores column labels, so the two spellings
    hash to the same id and still match across the event/classified merge.
    """
    date_col = "session_date" if "session_date" in events.columns else "day"
    keys = events[["animal", date_col, "epoch", event_col]]
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
    """Elementwise ``int(values == reference)`` with NaN-in-reference = 0.

    Use this only where an absent reference genuinely means "no arm
    qualifies" -- ``is_prev`` on the first trip of a session, or
    ``is_last_rewarded`` before the first reward.  Where an absent
    reference instead means "unobserved", use
    :func:`eq_indicator_or_nan` so the row is excluded by complete-case
    fitting rather than silently labelled a zero.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(int)
    out[np.isnan(reference)] = 0
    return out


def eq_indicator_or_nan(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise ``values == reference`` as 1.0/0.0, NaN where reference is NaN.

    The missing-data counterpart of :func:`eq_indicator`, for indicators
    whose reference can be *unobserved* rather than merely absent.
    ``occupancy_well_arm`` is NaN whenever an event window is not wholly at one
    well (``in_transit``, ``mixed``, ``unknown``); the animal still
    occupies some arm, we just cannot say which.  Encoding that as 0 on
    every arm would assert "not the current arm" for all nine, which is
    a fabricated observation.  Propagating NaN instead lets complete-case
    cohort selection drop the event through its logged path.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        Per-row candidate values (e.g. arm or patch IDs).
    reference : np.ndarray, shape (n_rows,)
        Per-row reference value, broadcast alongside ``values``.

    Returns
    -------
    np.ndarray, shape (n_rows,), dtype float
        1.0 where equal, 0.0 where unequal, NaN where ``reference`` is NaN.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    out = (values == reference).astype(float)
    out[np.isnan(reference)] = np.nan
    return out


def align_count_matrix(
    counts: np.ndarray,
    ids_sorted: np.ndarray,
    target_ids: np.ndarray,
    n_arms: int,
) -> np.ndarray:
    """Reindex a ``(n_events_src, n_arms)`` count matrix to ``target_ids``.

    An id in ``target_ids`` with no row in ``ids_sorted`` is an event that
    contributed no counted rows -- every one of its in-event timebins failed to
    attribute to a real arm. Its output row is filled, and **the fill value
    depends on the input dtype**: integer counts fill ``0`` (a real count of
    zero) and float counts fill ``NaN`` (unknown). That is deliberate --
    ``arm_count_hard`` genuinely observed zero real-arm bins, while
    ``arm_fraction``'s 0/0 must stay undefined rather than become 0 or +inf --
    but it is subtle enough to be worth stating, since the two paths differ
    only by ``counts.dtype``.

    On the shipped corpus no event triggers the fill (``n_junction_bins`` is 0
    for all 65,913), so this path is exercised only by tests.
    """
    mapping = {int(i): k for k, i in enumerate(ids_sorted)}
    fill = np.nan if np.issubdtype(counts.dtype, np.floating) else 0
    out = np.full((target_ids.size, n_arms), fill, dtype=counts.dtype)
    for k, tid in enumerate(target_ids):
        src_idx = mapping.get(int(tid))
        if src_idx is not None:
            out[k] = counts[src_idx]
    return out


def assert_event_label_alignment(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    source_name: str,
) -> None:
    """Require one exact event-label set before response matrices are aligned.

    ``align_count_matrix`` may legitimately fill a hard-count row when every bin
    of a known event maps to a junction. It must not, however, conceal an event
    missing from the source frame or an extra event from another/stale build.
    Callers therefore pass the labels from *all* qualified source bins here,
    before selecting only real-arm bins for counting.
    """
    source = np.asarray(source_ids, dtype=float)
    target = np.asarray(target_ids, dtype=float)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError(
            f"event-label alignment failed for {source_name}: labels must be finite"
        )
    if not np.equal(source, np.floor(source)).all() or not np.equal(
        target, np.floor(target)
    ).all():
        raise ValueError(
            f"event-label alignment failed for {source_name}: labels must be integral"
        )
    if len(np.unique(target)) != len(target):
        raise ValueError(
            "event-label alignment failed: target event table contains duplicate labels"
        )
    source_set = set(source.astype(np.int64))
    target_set = set(target.astype(np.int64))
    if source_set != target_set:
        missing = sorted(target_set - source_set)[:10]
        extra = sorted(source_set - target_set)[:10]
        raise ValueError(
            f"event-label alignment failed for {source_name}: missing target labels "
            f"{missing}, extra source labels {extra}. The source frame and event "
            "table must come from the same session and build."
        )


def assert_single_session_alignment(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_name: str,
    session_cols: tuple[str, ...] = ("animal", "day", "epoch"),
    target_session_cols: tuple[str, ...] | None = None,
) -> None:
    """Require source and target frames to name the same single session.

    Event labels are session-local integers. Exact label-set equality alone
    cannot detect a stale source frame from another session whose labels happen
    to overlap, so builders validate the natural session key before aligning
    response matrices.
    """
    target_cols = target_session_cols or session_cols
    if len(target_cols) != len(session_cols):
        raise ValueError("source and target session keys must have equal arity")
    missing_source = [column for column in session_cols if column not in source]
    missing_target = [column for column in target_cols if column not in target]
    if missing_source or missing_target:
        raise KeyError(
            f"session alignment for {source_name} requires columns "
            f"source={session_cols}, target={target_cols}; "
            f"missing from source={missing_source}, "
            f"target={missing_target}"
        )
    if source.loc[:, list(session_cols)].isna().any().any() or target.loc[
        :, list(target_cols)
    ].isna().any().any():
        raise ValueError(
            f"session alignment for {source_name} requires nonmissing natural keys"
        )

    source_keys = source.loc[:, list(session_cols)].drop_duplicates()
    target_keys = target.loc[:, list(target_cols)].drop_duplicates()
    if len(source_keys) != 1 or len(target_keys) != 1:
        raise ValueError(
            f"session alignment for {source_name} requires exactly one session "
            f"per frame; found source={len(source_keys)}, target={len(target_keys)}"
        )
    source_key = tuple(source_keys.iloc[0].tolist())
    target_key = tuple(target_keys.iloc[0].tolist())
    if source_key != target_key:
        raise ValueError(
            f"session alignment failed for {source_name}: source={source_key}, "
            f"target={target_key}"
        )


def add_arm_fixed_effects(
    df: pd.DataFrame,
    arm_col: str = "candidate_arm",
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
        Integer arm-id column (``"candidate_arm"`` in the canonical event-arm
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


def attach_is_swr_to_onsets(
    onset_df: pd.DataFrame, event_table: pd.DataFrame
) -> pd.DataFrame:
    """Add a canonical ``is_swr`` column to the non-local onset rows.

    First freezes the intersection of natural session keys carried by both
    tables, then matches each non-local onset to its ``event_table`` event by
        exact ``(session, stop_id, onset_time == event_start_s)`` identity. Within
    every retained session the sources must be a complete one-to-one partition.
    Sessions present in only one artifact are excluded as an explicit
    regeneration/build-coverage gap rather than blocking valid shared sessions.
    SWR/MUA onset rows get ``is_swr = NaN``.

    Parameters
    ----------
    onset_df : pd.DataFrame
        Event-onset rows. Must carry ``stop_id``, ``event_type``, and
        ``onset_time``.
    event_table : pd.DataFrame
        Per-event table. Must carry ``stop_id``, ``event_start_s``, ``is_swr``.

    Returns
    -------
    pd.DataFrame
        Shared-session copy of ``onset_df`` with an added ``is_swr`` column (the
        matched boolean for ``event_type == "non_local"`` rows, ``NaN``
        elsewhere). ``attrs["event_session_alignment"]`` records shared and
        excluded session keys and row counts.
    """
    session_presence = {
        "onset": [column in onset_df for column in _SESSION_COLS],
        "event": [column in event_table for column in _SESSION_COLS],
    }
    for name, presence in session_presence.items():
        if any(presence) and not all(presence):
            missing = [
                column
                for column, present in zip(_SESSION_COLS, presence, strict=True)
                if not present
            ]
            raise KeyError(
                f"{name} table has a partial natural session key; missing {missing}"
            )
    has_session_keys = all(session_presence["onset"]) and all(
        session_presence["event"]
    )
    if all(session_presence["onset"]) != all(session_presence["event"]):
        raise KeyError(
            "onset and event tables must either both carry natural session keys "
            f"{tuple(_SESSION_COLS)} or both be single-session fixtures"
        )

    if has_session_keys:
        if onset_df[_SESSION_COLS].isna().any().any() or event_table[
            _SESSION_COLS
        ].isna().any().any():
            raise ValueError("onset/event natural session keys must not be missing")
        onset_sessions = pd.MultiIndex.from_frame(
            onset_df[_SESSION_COLS].drop_duplicates()
        )
        event_sessions = pd.MultiIndex.from_frame(
            event_table[_SESSION_COLS].drop_duplicates()
        )
        shared_sessions = onset_sessions.intersection(event_sessions, sort=False)
        onset_only = onset_sessions.difference(event_sessions, sort=False)
        event_only = event_sessions.difference(onset_sessions, sort=False)
        if len(shared_sessions) == 0 and (len(onset_df) or len(event_table)):
            raise ValueError(
                "onset and event tables share no natural session keys; they are "
                "from incompatible cohorts or builds"
            )
        onset_keys = pd.MultiIndex.from_frame(onset_df[_SESSION_COLS])
        event_keys = pd.MultiIndex.from_frame(event_table[_SESSION_COLS])
        out = onset_df.loc[onset_keys.isin(shared_sessions)].copy()
        event_shared = event_table.loc[event_keys.isin(shared_sessions)].copy()
        merge_keys = [*_SESSION_COLS, "stop_id"]
    else:
        # Small direct unit fixtures historically omit natural keys. Treat both
        # complete frames as one implicit session and retain strict whole-frame
        # equality; never mix a keyed frame with an unkeyed one above.
        out = onset_df.copy()
        event_shared = event_table.copy()
        shared_sessions = ["<implicit-single-session>"] if len(out) or len(event_shared) else []
        onset_only = []
        event_only = []
        merge_keys = ["stop_id"]

    alignment_audit = {
        "n_onset_sessions": int(len(shared_sessions) + len(onset_only)),
        "n_event_sessions": int(len(shared_sessions) + len(event_only)),
        "n_shared_sessions": int(len(shared_sessions)),
        "onset_only_sessions": tuple(map(tuple, onset_only)),
        "event_only_sessions": tuple(map(tuple, event_only)),
        "n_onset_rows_input": int(len(onset_df)),
        "n_event_rows_input": int(len(event_table)),
        "n_onset_rows_retained": int(len(out)),
        "n_event_rows_retained": int(len(event_shared)),
    }
    if len(onset_only) or len(event_only):
        message = (
            "attach_is_swr_to_onsets excluded non-shared artifact sessions; "
            f"onset_only={alignment_audit['onset_only_sessions']}, "
            f"event_only={alignment_audit['event_only_sessions']}. Regenerate "
            "the missing sessions before full-corpus inference."
        )
        # The warning is the API-level signal, but every rate runner installs
        # ``filterwarnings("ignore", category=RuntimeWarning)`` at import, so a
        # warning alone would drop these sessions with no trace in the log. The
        # cohort actually fitted is a scientific fact, not a nuisance.
        logger.error(
            "%s Fitting %d of %d onset sessions (%d of %d onset rows).",
            message,
            alignment_audit["n_shared_sessions"],
            alignment_audit["n_onset_sessions"],
            alignment_audit["n_onset_rows_retained"],
            alignment_audit["n_onset_rows_input"],
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    # object dtype holds the matched booleans alongside NaN for SWR/MUA rows
    # without the float coercion pandas now warns about.
    out["is_swr"] = pd.Series(np.nan, index=out.index, dtype="object")
    # Join on the onset time itself, not on within-stop ordinal rank. A
    # non-local onset IS its event's start, so the two are exactly equal by
    # construction. Ranking instead makes one missing onset shift every later
    # rank in that stop, relabelling events that were present and correct --
    # and because the shifted ranks still all find *a* row, the match-rate
    # guard below stays silent through it. Matching on time contains the
    # failure to the row that is actually missing.
    ev = event_shared[[*merge_keys, "event_start_s", "is_swr"]].copy()
    nl = out[out["event_type"] == "non_local"].copy()
    if has_session_keys:
        onset_counts = nl.groupby(_SESSION_COLS, sort=False).size()
        event_counts = ev.groupby(_SESSION_COLS, sort=False).size()
        counts = pd.concat(
            [onset_counts.rename("onsets"), event_counts.rename("events")], axis=1
        ).fillna(0)
        bad_counts = counts[counts["onsets"] != counts["events"]]
    else:
        bad_counts = pd.DataFrame() if len(nl) == len(ev) else pd.DataFrame(
            {"onsets": [len(nl)], "events": [len(ev)]}, index=["implicit"]
        )
    if len(bad_counts):
        raise ValueError(
            "attach_is_swr_to_onsets requires an exact non-local onset/event "
            "partition within every shared session; mismatched counts: "
            f"{bad_counts.astype(int).to_dict(orient='index')}"
        )
    nl["_onset_order"] = np.arange(len(nl))
    merged = nl.merge(
        ev.rename(columns={"event_start_s": "onset_time"}),
        on=[*merge_keys, "onset_time"],
        how="left",
        suffixes=("", "_ev"),
        # Duplicate (stop_id, onset_time) keys would expand rows and misalign
        # the positional assignment below.
        validate="one_to_one",
    )
    # Any mismatch means the onset table and event_table were built from
    # divergent non-local/speed thresholds or decodes -- fail loud rather than
    # silently leave is_swr unlabelled downstream. On the shipped artifacts the
    # time join matches exactly; unmatched rows would make SWR and non-SWR fail
    # to partition the pooled non-local population.
    merged = merged.sort_values("_onset_order", kind="stable")
    matched = merged["is_swr_ev"].notna()
    if len(merged) and not matched.all():
        raise ValueError(
            "attach_is_swr_to_onsets requires every non-local onset to be exactly "
            f"matched; {int((~matched).sum())}/{len(matched)} were unmatched by "
            "(session, stop_id, onset_time). The onset table and "
            "event_table must be built from the same non-local/speed thresholds "
            "and decodes."
        )
    out.loc[nl.index, "is_swr"] = merged["is_swr_ev"].to_numpy()
    out.attrs["event_session_alignment"] = alignment_audit
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


def reward_basis_from_lags(lags: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a lag matrix onto raised-cosine bumps, normalized per bump.

    Each output is the bump-weighted mean over the *available* (finite) lags,
    ``sum_j B[k, j] * lags[:, j] / sum_j B[k, j]``, so a query with only part of
    the history is still on the reward-fraction scale rather than being shrunk
    toward zero by the missing lags.

    Parameters
    ----------
    lags : np.ndarray, shape (n_query, n_lags)
        Per-lag reward outcomes (0/1), ``NaN`` where the lag is not known.
    basis : np.ndarray, shape (n_basis, n_lags)
        Bump weights; ``basis[k, j]`` is the weight of lag ``j + 1`` in bump
        ``k``.

    Returns
    -------
    np.ndarray, shape (n_query, n_basis)
        Normalized projections; ``NaN`` where no lag in the bump's support is
        finite.
    """
    finite = np.isfinite(lags)
    values = np.where(finite, lags, 0.0)  # (n_query, n_lags)
    num = values @ basis.T  # (n_query, n_basis)
    den = finite.astype(float) @ basis.T  # (n_query, n_basis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def reward_kernel_weights(n_lags: int, n_basis: int) -> np.ndarray:
    """Per-lag weight each reward-history basis coefficient contributes.

    The persisted features are bump-weighted reward *fractions* (each bump is
    normalized by the sum of its weights over the lags it spans), so an outcome
    at lag ``j`` contributes ``sum_k coef_k * B[k, j] / sum_j' B[k, j']`` to the
    log rate. This returns that row-normalized basis.

    Parameters
    ----------
    n_lags : int
        Number of reward lags the basis spans.
    n_basis : int
        Number of raised-cosine bumps.

    Returns
    -------
    np.ndarray, shape (n_basis, n_lags)
        Row-normalized raised-cosine basis.
    """
    B = raised_cosine_history_basis(n_lags, n_basis)
    return B / B.sum(axis=1, keepdims=True)


def reconstruct_reward_kernels(
    coefs: np.ndarray, n_lags: int, n_basis: int
) -> np.ndarray:
    """Per-session reward kernels over lags ``1..n_lags``, on the log-rate scale.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_sessions, n_basis)
        Fitted basis coefficients, one row per session. Rows with any NaN are
        dropped, so a session that did not fit the block contributes nothing.
    n_lags, n_basis : int
        Shape of the basis the coefficients were fit against.

    Returns
    -------
    np.ndarray, shape (n_fitted_sessions, n_lags)
        Log-rate contribution of a reward at each lag, per session.
    """
    coefs = np.atleast_2d(np.asarray(coefs, dtype=float))
    coefs = coefs[~np.isnan(coefs).any(axis=1)]
    return coefs @ reward_kernel_weights(n_lags, n_basis)


def reconstruct_self_history_kernels(
    coefs: np.ndarray, n_lags: int, n_basis: int
) -> np.ndarray:
    """Per-session self-history kernels over lags ``1..n_lags``, log-rate scale.

    The reward and self-history blocks share the raised-cosine bumps but not the
    projection. Reward features are bump-weighted *fractions* over the available
    lags, so their kernel is read off the ROW-NORMALIZED basis
    (:func:`reward_kernel_weights`); ``hist_rcos_k`` is the plain projection of
    the ``onset_lag{j}`` counts onto the raw bumps, so an onset ``j`` bins back
    contributes ``sum_k coef_k * B[k, j]`` with ``B`` un-normalized. Using the
    reward weights here would divide every lag by the bump's span.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_sessions, n_basis)
        Fitted ``coef_hist_rcos_k`` values, one row per session. Rows with any
        NaN are dropped, so a session that did not fit the block contributes
        nothing.
    n_lags, n_basis : int
        Shape of the basis the coefficients were fit against.

    Returns
    -------
    np.ndarray, shape (n_fitted_sessions, n_lags)
        Log-rate contribution of an onset at each lag, per session.
    """
    coefs = np.atleast_2d(np.asarray(coefs, dtype=float))
    coefs = coefs[~np.isnan(coefs).any(axis=1)]
    return coefs @ raised_cosine_history_basis(n_lags, n_basis)


def _arrival_spanning_stop_ids(stops: pd.DataFrame) -> np.ndarray:
    """``stop_id`` of the bouts that contain a later well arrival.

    A stop spans an arrival iff the NEXT stop of the same session -- ordering
    by ``stop_start`` within ``animal, session_date, epoch`` -- reports a
    ``last_well_arrival_time_s`` strictly inside ``(stop_start, stop_end]`` of
    this bout. Well arrivals are monotone within a session, so only the next
    stop's arrival can fall inside this one and the test is exact. A stop with
    no successor, or whose successor has no arrival clock, does not span.

    Parameters
    ----------
    stops : pd.DataFrame
        Per-stop frame (unique ``stop_id``) carrying ``animal``,
        ``session_date``, ``epoch``, ``stop_start``, ``stop_end`` and
        ``last_well_arrival_time_s``.

    Returns
    -------
    np.ndarray
        The spanning stops' ``stop_id`` values (possibly empty).
    """
    ordered = stops.sort_values([*_SESSION_COLS, "stop_start"], kind="stable")
    next_arrival = ordered.groupby(_SESSION_COLS, sort=False, observed=True)[
        "last_well_arrival_time_s"
    ].shift(-1)
    spans = (next_arrival > ordered["stop_start"]) & (
        next_arrival <= ordered["stop_end"]
    )
    return ordered.loc[spans.fillna(False).to_numpy(), "stop_id"].to_numpy()


def aggregate_to_timebins(
    stop_table: pd.DataFrame,
    onset_times: pd.DataFrame,
    *,
    bin_s: float,
    tmax: float,
    reward_basis: bool = False,
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
        Per-stop table (unique ``stop_id``). Must carry ``animal``,
        ``session_date``, ``epoch``, ``stop_id``, ``immobility_duration_s``,
        plus the reward-history basis columns (when ``reward_basis``) and any
        ``stop_covariates`` columns requested.
    onset_times : pd.DataFrame
        Event-onset rows for ONE population (already filtered). Must carry
        ``stop_id`` and ``t_since_stop_onset``.
    bin_s : float
        Bin width in seconds.
    tmax : float
        Right-truncation; bins beyond ``tmax`` are dropped.
    reward_basis : bool
        Carry the reward-history basis features ``REWARD_HISTORY_COLUMNS`` and
        the structural-missingness indicator ``REWARD_HISTORY_INCOMPLETE`` from
        ``stop_table``.
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
        Columns: ``animal``, ``session_date``, ``epoch``, ``stop_id``, ``binj``,
        ``tc`` (bin-center s = time since stop onset), ``exposure`` (s of
        immobility in the bin), the reward-history basis columns (if
        ``reward_basis``), each name in
        ``stop_covariates``, ``count`` (population onsets in the bin),
        ``onset_lag1..self_history_lags``, and (if ``self_history_basis``)
        ``hist_rcos_0..self_history_basis-1``. When ``stop_table`` carries
        ``stop_start``, ``last_well_arrival_time_s`` and
        ``outcome_known_time_s``, two further per-bin clocks are emitted:
        ``time_since_last_well_arrival_s`` and ``time_since_outcome_known_s``
        (bin centre minus the absolute clock; ``NaN`` where the clock is
        ``NaN``, and negative for bins that precede the outcome boundary).

    Notes on the outcome boundary
    -----------------------------
    A stop's reward history is evaluated at bout start, so a stop that is still
    in progress when the current trip's reward or omission becomes observable
    would carry a history missing that outcome for its later bins. When
    ``reward_basis`` is set and ``stop_table`` additionally carries
    ``stop_end``, ``last_well_arrival_time_s``, ``completed_trial_reward`` and
    the ``known_reward_outcome_lag_{1..10}`` columns, bins whose LEFT EDGE is at
    or past ``outcome_known_time_s`` (for a stop whose boundary falls inside it)
    get the lag vector shifted by one --
    ``[completed_trial_reward, lag_1, ..., lag_9]`` -- and the basis features and
    ``reward_history_incomplete`` re-derived from it by the same functions the
    stop level uses. Bins before the boundary keep the bout-start values. The
    one exception is a bout that spans a well arrival -- the next stop of the
    session arrived inside ``(stop_start, stop_end]`` (see
    :func:`_arrival_spanning_stop_ids`) -- which changes which trip is the
    completed one partway through, so it is never shifted;
    ``out.attrs["n_stops_arrival_spanning"]`` counts how many such stops have at
    least one bin at or after the boundary (a boundary that falls inside a stop
    but beyond its last retained bin leaves no bin to shift and is not counted).

    The absolute stop clocks are carried only to derive these per-bin columns
    and are dropped before return unless explicitly requested through
    ``stop_covariates``; an absolute session clock is never a model predictor.

    Raises
    ------
    ValueError
        If the binned counts do not reconcile with the onsets that fall inside
        the stops' exposed windows (``t_since_stop_onset < min(tmax,
        immobility_duration_s)``) AND land in a bin the exposure filter keeps --
        e.g. an onset at or past the end of its stop's immobility, which is
        binned into the stop's last bin but earns no exposure there.

    Notes
    -----
    A stop whose immobility ends a sliver past a bin edge has a final bin with
    ``exposure <= 1e-4``, which is dropped from the design; an onset in that
    sliver is therefore not counted anywhere. That is a real (tiny) loss rather
    than an inconsistency, so it does not raise -- it is counted in
    ``out.attrs["n_onsets_in_dropped_bins"]`` so it stays visible.
    """
    if self_history_basis and not self_history_lags:
        raise ValueError("self_history_basis requires self_history_lags > 0")
    if cross_history_basis and not cross_history_lags:
        raise ValueError("cross_history_basis requires cross_history_lags > 0")
    if cross_history_lags and cross_onset_times is None:
        raise ValueError("cross_history_lags requires cross_onset_times")
    reward_cols = (
        [*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE] if reward_basis else []
    )
    # Optional per-bin clocks / outcome-boundary shift: both are gated on the
    # stop frame actually carrying their inputs, so a stop table built before
    # the arrival clocks existed produces exactly the design it produced before.
    emit_clocks = all(c in stop_table.columns for c in _ARRIVAL_CLOCK_STOP_COLUMNS)
    shift_at_outcome = reward_basis and all(
        c in stop_table.columns
        for c in (*_OUTCOME_SHIFT_STOP_COLUMNS, *_ARRIVAL_SPAN_STOP_COLUMNS)
    )
    internal_cols = list(
        dict.fromkeys(
            (_ARRIVAL_CLOCK_STOP_COLUMNS if emit_clocks else ())
            + (_OUTCOME_SHIFT_STOP_COLUMNS if shift_at_outcome else ())
        )
    )
    carry = list(dict.fromkeys([*reward_cols, *internal_cols, *stop_covariates]))
    # Carried only to derive the per-bin columns; the absolute session clocks in
    # particular must never reach a model as predictors (see covariate_timing).
    drop_internal = [c for c in internal_cols if c not in stop_covariates]

    s = stop_table[stop_table["immobility_duration_s"] > 0]
    # ceil, not int: when tmax is not a multiple of bin_s, int() drops the final
    # partial window [floor(tmax/bin_s)*bin_s, tmax) so onsets there are lost.
    nmax = int(np.ceil(tmax / bin_s))
    nbin = np.minimum(np.ceil(s["immobility_duration_s"].to_numpy() / bin_s), nmax).astype(
        int
    )
    idx = np.repeat(np.arange(len(s)), nbin)
    binj = (
        np.concatenate([np.arange(n) for n in nbin])
        if len(s)
        else np.empty(0, dtype=int)
    )
    cols = [*_SESSION_COLS, "stop_id", "immobility_duration_s", *carry]
    out = s.iloc[idx][cols].reset_index(drop=True)
    out["binj"] = binj
    out["tc"] = (binj + 0.5) * bin_s
    if emit_clocks:
        # Bin CENTRE on two absolute session clocks, as elapsed seconds. The
        # arrival clock is >= 0 for an at-well stop (the animal arrived before
        # the bout began); the outcome clock is NEGATIVE for bins that precede
        # the reward/omission becoming observable.
        bin_center_abs = out["stop_start"].to_numpy(dtype=float) + (binj + 0.5) * bin_s
        out["time_since_last_well_arrival_s"] = bin_center_abs - out[
            "last_well_arrival_time_s"
        ].to_numpy(dtype=float)
        out["time_since_outcome_known_s"] = bin_center_abs - out[
            "outcome_known_time_s"
        ].to_numpy(dtype=float)
    # bin width, clipped by both the stop's immobility end and tmax (the final
    # bin of a stop longer than tmax is exposed only up to tmax, not bin_s).
    out["exposure"] = np.minimum(
        bin_s, np.minimum(out["immobility_duration_s"], tmax) - binj * bin_s
    )
    out = out[out["exposure"] > 1e-4]

    src = onset_times[onset_times["t_since_stop_onset"] < tmax]
    # floor with a tolerance, not int(): int() truncates the quotient, and an
    # onset exactly on a bin edge whose binary-float quotient lands just below
    # the integer (0.3 / 0.1 -> 2.9999999999999996) would fall one bin early,
    # shifting the estimated time course backwards. The clip keeps an onset in
    # the final partial window [(nmax-1)*bin_s, tmax) in the last bin rather
    # than a phantom bin that the merge below would silently drop.
    src_binj = np.clip(
        np.floor(src["t_since_stop_onset"].to_numpy() / bin_s + 1e-9).astype(int),
        0,
        nmax - 1,
    )
    cnt = (
        pd.DataFrame({"stop_id": src["stop_id"].to_numpy(), "binj": src_binj})
        .groupby(["stop_id", "binj"])
        .size()
        .rename("count")
    )
    out = out.merge(cnt, on=["stop_id", "binj"], how="left")
    out["count"] = out["count"].fillna(0)
    # Every onset inside a stop's exposed window whose bin survives the exposure
    # filter must land on exactly one row of the design; a mismatch means the
    # response and the exposure disagree, which a Poisson fit cannot detect.
    # The one legitimate loss is an onset in a final sliver bin that the filter
    # removed (immobility ending just past a bin edge): it is counted, not
    # raised on, so the design stays consistent and the loss stays visible.
    expected = src[["stop_id", "t_since_stop_onset"]].merge(
        s[["stop_id", "immobility_duration_s"]], on="stop_id", how="inner"
    )
    t_expected = expected["t_since_stop_onset"].to_numpy(dtype=float)
    immobility = expected["immobility_duration_s"].to_numpy(dtype=float)
    stop_end = np.minimum(immobility, tmax)
    in_window = t_expected < stop_end
    expected_binj = np.clip(
        np.floor(t_expected / bin_s + 1e-9).astype(int), 0, nmax - 1
    )
    bin_kept = (stop_end - expected_binj * bin_s) > 1e-4
    n_expected = int((in_window & bin_kept).sum())
    n_onsets_in_dropped_bins = int((in_window & ~bin_kept).sum())
    n_binned = int(out["count"].sum())
    if n_binned != n_expected:
        raise ValueError(
            f"aggregate_to_timebins binned {n_binned} onset(s) but {n_expected} "
            "fall inside a stop's exposed window "
            "(t_since_stop_onset < min(tmax, immobility_duration_s)) and in a "
            "retained bin; the (stop x bin) design's counts and exposure disagree"
        )

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
            np.floor(xsrc["t_since_stop_onset"].to_numpy() / bin_s + 1e-9).astype(int),
            0,
            nmax - 1,
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

    n_stops_arrival_spanning: int | None = None
    if shift_at_outcome:
        # The reward history a stop carries is evaluated at BOUT START, so bins
        # after the current trip's reward/omission became observable would
        # otherwise be modelled with a history that omits the outcome the animal
        # already has. Shift the lag vector by one at the boundary and re-derive
        # the basis features with the same functions the stop level uses, so the
        # two levels stay one kernel rather than two.
        stop_start = out["stop_start"].to_numpy(dtype=float)
        boundary = out["outcome_known_time_s"].to_numpy(dtype=float)
        left_edge_abs = stop_start + out["binj"].to_numpy(dtype=float) * bin_s
        # NaN boundary -> False, so a stop with no known outcome is untouched.
        # ``stop_start < boundary`` excludes stops already past their outcome at
        # bout start (their bout-start history already contains it); the bins of
        # a stop whose boundary sits beyond its last bin never satisfy ``post``.
        inside = (stop_start < boundary) & (left_edge_abs >= boundary)
        # A bout that spans a well arrival changes WHICH trip is the completed
        # one partway through, so ``completed_trial_reward`` is not the outcome
        # that becomes known at the boundary; those stops are left on their
        # bout-start history and counted instead of silently mis-shifted. A bout
        # spans an arrival iff the NEXT stop of the same session (ordered by
        # ``stop_start``) arrived at its well inside ``(stop_start, stop_end]``;
        # arrivals are monotone in the session, so the next stop's is the only
        # one that can fall there and the test is exact.
        spans_arrival = out["stop_id"].isin(_arrival_spanning_stop_ids(s)).to_numpy()
        crosses = inside & ~spans_arrival
        n_stops_arrival_spanning = int(
            pd.unique(out["stop_id"].to_numpy()[inside & spans_arrival]).size
        )

        lags = out[list(_REWARD_LAG_COLUMNS)].to_numpy(dtype=float)
        shifted = lags.copy()
        shifted[crosses, 1:] = lags[crosses, :-1]
        shifted[crosses, 0] = out["completed_trial_reward"].to_numpy(dtype=float)[
            crosses
        ]
        raw = reward_basis_from_lags(
            shifted,
            raised_cosine_history_basis(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS),
        )
        out[REWARD_HISTORY_INCOMPLETE] = np.isnan(raw).any(axis=1).astype(float)
        for k, col in enumerate(REWARD_HISTORY_COLUMNS):
            out[col] = np.nan_to_num(raw[:, k], nan=0.0)

    out = out.drop(columns=["immobility_duration_s", *drop_internal]).reset_index(
        drop=True
    )
    out.attrs["n_onsets_in_dropped_bins"] = n_onsets_in_dropped_bins
    if n_stops_arrival_spanning is not None:
        out.attrs["n_stops_arrival_spanning"] = n_stops_arrival_spanning
    return out


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
