"""Small shared primitives for the event-table builders.

What is left here is the residue that has no better home: NaN-strict series
reducers, elementwise indicator builders, the well/arm/patch scalar lookups,
and the alignment asserts every builder runs before it joins two frames on
event labels. They are used by both the NonLocal (``tables_*.py``) and
ContFrag (``contfrag/tables.py``) builders.

Neighbours that used to live here: immobility bouts in :mod:`.bouts`, the
``event_id`` / ``stop_id`` hashes in :mod:`.event_ids`, and the (stop x bin)
design with its history bases in :mod:`.per_bin_design`. Trial clocks,
behavioral context, and behavioral-history covariates live in
:mod:`.covariates`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .design_constants import ARM_FE_PREDICTORS
from .spatial_mapping import well_to_arm_set

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


# ── Stay / switch ──


def stay_indicator_from_patches(next_patch, current_patch):
    """1.0 when the next trip stays in the current patch, 0.0 when it switches.

    ``NaN`` propagates: an unknown patch on either side is not evidence of a
    stay *or* a switch, and encoding it as 0 would tell a model the animal
    switched. Accepts scalars or equal-length arrays and returns the matching
    shape.

    Parameters
    ----------
    next_patch, current_patch : float or np.ndarray
        Patch IDs, already mapped from arms by :func:`arm_to_patch`.

    Returns
    -------
    float or np.ndarray
        1.0 / 0.0 / ``NaN``, elementwise.
    """
    nxt = np.asarray(next_patch, dtype=float)
    cur = np.asarray(current_patch, dtype=float)
    out = np.where(np.isnan(nxt) | np.isnan(cur), np.nan, (nxt == cur).astype(float))
    return float(out) if out.ndim == 0 else out


def stay_indicator(next_arm: float, current_arm: float) -> float:
    """``is_stay`` for one row, from the two arms rather than their patches.

    The arm-level entry point to :func:`stay_indicator_from_patches`; the
    stay/switch distinction is a *patch* comparison, so both arms are mapped
    through :func:`arm_to_patch` first. Two arms of the same patch are a stay.
    """
    return stay_indicator_from_patches(
        arm_to_patch(next_arm), arm_to_patch(current_arm)
    )


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
    if (
        not np.equal(source, np.floor(source)).all()
        or not np.equal(target, np.floor(target)).all()
    ):
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
    if (
        source.loc[:, list(session_cols)].isna().any().any()
        or target.loc[:, list(target_cols)].isna().any().any()
    ):
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
