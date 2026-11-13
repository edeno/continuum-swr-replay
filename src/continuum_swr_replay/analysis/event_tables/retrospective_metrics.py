"""Per-event retrospective metrics: does an event's replay content align with
the arm the animal visited next (or just left)?

Five event-level metrics, merged into the event table by
:func:`compute_retrospective_metrics` during the table build:

- ``chosen_arm_fraction``: per-event ``arm_fraction`` at ``arm == next_arm``.
- ``ends_at_next``: does the last in-event MAP arm equal ``next_arm``?
- ``contains_path_to_next``: does the event's MAP-arm trace cover the
  geometric path from ``occupancy_well_arm`` to ``next_arm``?
- ``chosen_patch_fraction``: replay mass assigned to ``next_patch``.
- ``ends_at_next_patch``: does the final MAP arm belong to ``next_patch``?

The ``prev_arm`` mirrors (``chosen_arm_fraction_prev``, ``ends_at_prev``,
``contains_path_from_prev``) are computed the same way. The shuffle null and
the per-session alignment statistics built on these metrics are in
:mod:`continuum_swr_replay.analysis.replay_models.choice.retrospective`.

Bandit-track topology
---------------------
Wells are arms 0–5 (each well is degree-1 in the track graph and is
thus identified with a single leaf arm). Trunk arms 6, 7, 8 connect
patch junctions to the center node; they are traversed in transit but
the animal never stops there.

The geometric path from well ``i`` to well ``j`` is:

- ``{i}`` if ``i == j``
- ``{i, j}`` if both wells share a patch (trunks not crossed)
- ``{i, j, trunk(p_i), trunk(p_j)}`` otherwise, where
  ``trunk(p) = p + 5`` (patch 1 → arm 6, patch 2 → arm 7, patch 3 → arm 8).

This module never depends on a NetworkX track graph; the topology is
encoded by ``WELL_PATCH_MAPPING`` from
:mod:`continuum_swr_replay.data_loaders.constants`, so tests and
production code share the same source of truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)
from continuum_swr_replay.tables.session_keys import present_session_columns

from ._common import strict_sum

__all__ = [
    "arm_path_well_to_well",
    "chosen_arm_fraction",
    "chosen_arm_fraction_prev",
    "chosen_patch_fraction",
    "compute_retrospective_metrics",
    "contains_path_from_prev",
    "contains_path_to_next",
    "ends_at_next",
    "ends_at_next_patch",
    "ends_at_prev",
]


# Patch (1/2/3) → trunk arm id. Each patch junction connects to the
# center node via exactly one trunk arm, and the trunk-arm IDs are 5
# offsets above the patch IDs by construction. This is the only place a
# magic offset appears; everything else routes through this mapping.
_PATCH_TO_TRUNK_ARM: dict[int, int] = {1: 6, 2: 7, 3: 8}

# Wells are arms 0–5 (degree-1 leaves of the track graph).
_VALID_WELL_ARMS = frozenset(WELL_PATCH_MAPPING.keys())


def _empty_metric_frame(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Empty metric frame preserving available session keys."""
    return pd.DataFrame(columns=[*present_session_columns(df), "event_id", metric_col])


def arm_path_well_to_well(current_arm: int, next_arm: int) -> set[int]:
    """Set of arm IDs traversed on the geometric path between two wells.

    Parameters
    ----------
    current_arm, next_arm : int
        Well-arm IDs in 0..5. Trunk arms (6,7,8) and out-of-range
        values raise ``ValueError`` — the animal cannot stop at a
        non-well, so they should never appear as endpoints.

    Returns
    -------
    set[int]
        Arm IDs traversed on the path. Always at least 1, at most 4.

    Examples
    --------
    >>> sorted(arm_path_well_to_well(0, 0))   # same arm
    [0]
    >>> sorted(arm_path_well_to_well(0, 1))   # same patch
    [0, 1]
    >>> sorted(arm_path_well_to_well(0, 2))   # cross patch
    [0, 2, 6, 7]
    """
    if int(current_arm) not in _VALID_WELL_ARMS:
        raise ValueError(
            f"current_arm={current_arm} is not a well arm (must be in 0..5)"
        )
    if int(next_arm) not in _VALID_WELL_ARMS:
        raise ValueError(f"next_arm={next_arm} is not a well arm (must be in 0..5)")
    i, j = int(current_arm), int(next_arm)
    if i == j:
        return {i}
    p_i = WELL_PATCH_MAPPING[i]
    p_j = WELL_PATCH_MAPPING[j]
    if p_i == p_j:
        return {i, j}
    return {i, j, _PATCH_TO_TRUNK_ARM[p_i], _PATCH_TO_TRUNK_ARM[p_j]}


# ─────────────────────────────────────────────────────────────────────
# Per-event metrics
# ─────────────────────────────────────────────────────────────────────


def _arm_fraction_at(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    *,
    target_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Per-event ``arm_fraction`` at ``arm == event_table[target_col]``.

    Events with ``target_col`` NaN are dropped. Shared core of
    :func:`chosen_arm_fraction` (``target_col="next_arm"``, prospective) and
    :func:`chosen_arm_fraction_prev` (``target_col="prev_arm"``, retrospective).
    """
    if event_table.empty:
        return _empty_metric_frame(event_table, out_col)
    ev = event_table.dropna(subset=[target_col]).copy()
    ev[target_col] = ev[target_col].astype(int)

    sess_keys = present_session_columns(event_arm_table)
    ea = event_arm_table[[*sess_keys, "event_id", "candidate_arm", "arm_fraction"]]

    merged = ev.merge(
        ea,
        left_on=[*sess_keys, "event_id", target_col],
        right_on=[*sess_keys, "event_id", "candidate_arm"],
        how="left",
    )
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    out[out_col] = merged["arm_fraction"].astype(float)
    return out


def chosen_arm_fraction(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event ``arm_fraction`` at ``arm == next_arm`` (prospective).

    Events with ``next_arm`` NaN are dropped (the animal never left,
    or the session ended before another stop occurred).

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``chosen_arm_fraction``. Session-key
        columns (``animal``/``day``/``epoch``) are passed through if
        present in ``event_table``.
    """
    return _arm_fraction_at(
        event_table,
        event_arm_table,
        target_col="next_arm",
        out_col="chosen_arm_fraction",
    )


def chosen_arm_fraction_prev(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event ``arm_fraction`` at ``arm == prev_arm`` (retrospective).

    The ``prev_arm`` mirror of :func:`chosen_arm_fraction`: how much of the
    replay content sits on the arm the animal just left.
    """
    return _arm_fraction_at(
        event_table,
        event_arm_table,
        target_col="prev_arm",
        out_col="chosen_arm_fraction_prev",
    )


def _ends_at_arm(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    target_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Per-event boolean: last in-event MAP arm equals ``target_col``.

    The "last" bin is the maximum ``t_in_event`` per event. Events with NaN
    ``target_col`` are dropped; events with no in-event sequence row, or a NaN
    final ``decoded_arm_map`` (junction-bin artifact), record ``False`` so the
    downstream ``np.nanmean`` counts them in the denominator. Shared core of
    :func:`ends_at_next` and :func:`ends_at_prev`.
    """
    if event_table.empty:
        return _empty_metric_frame(event_table, out_col)
    ev = event_table.dropna(subset=[target_col]).copy()
    ev[target_col] = ev[target_col].astype(int)

    sess_keys = present_session_columns(event_sequence_table)
    last_bin = (
        event_sequence_table.sort_values([*sess_keys, "event_id", "t_in_event"])
        .groupby([*sess_keys, "event_id"], as_index=False)
        .tail(1)[[*sess_keys, "event_id", "decoded_arm_map"]]
        .rename(columns={"decoded_arm_map": "_last_arm"})
    )

    merged = ev.merge(last_bin, on=[*sess_keys, "event_id"], how="left")
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    eq = merged["_last_arm"].astype("Int64") == merged[target_col].astype("Int64")
    out[out_col] = eq.fillna(False).astype(bool)
    return out


def ends_at_next(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: last in-event MAP arm equals ``next_arm`` (prospective).

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``ends_at_next`` (bool). Session-key
        columns are passed through if present.
    """
    return _ends_at_arm(
        event_table,
        event_sequence_table,
        target_col="next_arm",
        out_col="ends_at_next",
    )


def ends_at_prev(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: last in-event MAP arm equals ``prev_arm`` (retrospective).

    The ``prev_arm`` mirror of :func:`ends_at_next`: whether the replay
    trajectory ends at the arm the animal just left.
    """
    return _ends_at_arm(
        event_table,
        event_sequence_table,
        target_col="prev_arm",
        out_col="ends_at_prev",
    )


def _contains_path(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Per-event boolean: MAP-arm trace covers the path ``source_col`` → ``target_col``.

    "Covers" is deliberately a set-membership criterion: every arm in
    :func:`arm_path_well_to_well(source, target)` must appear at least once in
    the event's ``decoded_arm_map``. Arm order, adjacency, repeat counts, and
    intervening off-path arms do not affect the result. Events with NaN
    ``source_col`` or ``target_col`` are dropped. Shared core of
    :func:`contains_path_to_next` (current→next, prospective) and
    :func:`contains_path_from_prev` (prev→current, retrospective).
    """
    if event_table.empty:
        return _empty_metric_frame(event_table, out_col)
    ev = event_table.dropna(subset=[source_col, target_col]).copy()
    ev[source_col] = ev[source_col].astype(int)
    ev[target_col] = ev[target_col].astype(int)

    sess_keys = present_session_columns(event_sequence_table)
    # Empty ``event_sequence_table`` is a legitimate input
    # (sessions can have events but no in-event timebins yet, e.g.
    # before posterior reconstruction lands). The previous chained
    # ``Series.groupby(...).agg(...).rename(columns=...)`` produced
    # a Series for an empty input, and ``Series.rename`` rejects the
    # ``columns=`` kwarg with TypeError — which the runner script
    # then swallows under a broad except, silently dropping the whole
    # session. Use named aggregation so the result is
    # always a DataFrame, even when empty.
    if event_sequence_table.empty:
        arm_sets = pd.DataFrame(columns=[*sess_keys, "event_id", "_decoded_arm_set"])
    else:
        arm_sets = event_sequence_table.groupby(
            [*sess_keys, "event_id"], as_index=False
        ).agg(
            _decoded_arm_set=(
                "decoded_arm_map",
                lambda s: frozenset(int(x) for x in s.dropna().unique()),
            )
        )

    merged = ev.merge(arm_sets, on=[*sess_keys, "event_id"], how="left")
    contains = np.array(
        [
            (
                False
                if not isinstance(decoded, frozenset)
                else arm_path_well_to_well(int(c), int(n)).issubset(decoded)
            )
            for c, n, decoded in zip(
                merged[source_col].to_numpy(),
                merged[target_col].to_numpy(),
                merged["_decoded_arm_set"].to_numpy(),
                strict=True,
            )
        ],
        dtype=bool,
    )
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    out[out_col] = contains
    return out


def contains_path_to_next(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: MAP-arm trace covers ``occupancy_well_arm`` → ``next_arm``.

    "Covers" means every arm in
    ``arm_path_well_to_well(occupancy_well_arm, next_arm)`` appears in the
    event's ``decoded_arm_map``. This ignores ordering, continuity, repeats,
    and off-path arms. Events with undefined occupancy or ``next_arm`` are
    dropped.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``contains_path_to_next`` (bool).
    """
    return _contains_path(
        event_table,
        event_sequence_table,
        source_col="occupancy_well_arm",
        target_col="next_arm",
        out_col="contains_path_to_next",
    )


def contains_path_from_prev(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: MAP-arm trace covers ``prev_arm`` → ``occupancy_well_arm``.

    The retrospective mirror of :func:`contains_path_to_next`: whether the
    replay covers the route the animal just took (the just-left arm back to
    where it currently is).
    """
    return _contains_path(
        event_table,
        event_sequence_table,
        source_col="prev_arm",
        target_col="occupancy_well_arm",
        out_col="contains_path_from_prev",
    )


def chosen_patch_fraction(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event fraction of replay mass on arms in ``next_patch``.

    Computed as the sum of ``arm_fraction`` over arms whose patch
    label equals ``next_patch``. With three patches in the bandit task,
    chance under uniform replay is 3/9 = 1/3 (each patch claims 3
    arms out of 9), much higher than the arm-level chance of 1/9.

    The patch-level metric is robust to within-patch alternation: in
    the bandit task ``next_arm == prev_arm`` ~90% of the time, but
    within-patch alternation does NOT change ``next_patch``, so this
    metric is informative on the cross-patch trials that the arm-level
    metric is most polluted by.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``chosen_patch_fraction``. Session-key
        columns passed through.
    """
    if event_table.empty or "next_patch" not in event_table.columns:
        return _empty_metric_frame(event_table, "chosen_patch_fraction")
    ev = event_table.dropna(subset=["next_patch"]).copy()
    ev["next_patch"] = ev["next_patch"].astype(int)

    sess_keys = present_session_columns(event_arm_table)
    ea = event_arm_table[
        [*sess_keys, "event_id", "candidate_arm", "arm_fraction"]
    ].copy()
    # Map arm → patch via TRACK_SEGMENT_TO_PATCH.
    ea["arm_patch"] = ea["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)

    merged = ev.merge(ea, on=[*sess_keys, "event_id"], how="left")
    in_target_patch = merged["arm_patch"] == merged["next_patch"]
    grp_cols = [*sess_keys, "event_id"]
    # Sum arm_fraction over target-patch arms only. Two failure
    # modes the previous form fabricated 0.0 for:
    #   1. event_arm_table contains the event's arms but the
    #      target-patch arm_fractions are all NaN — handled by
    #      ``skipna=False`` below.
    #   2. event_arm_table has no rows for the target patch at all
    #      (empty event_arm_table or only non-target rows merged
    #      in). The old ``.where(in_target_patch, 0.0)`` then made
    #      every retained row zero and ``sum`` returned 0.0.
    #      Mask non-target rows to NaN instead so their absence is
    #      visible in the sum.
    #
    # We then explicitly require at least one target-patch row per
    # event: if there are no target arms at all for an event, the
    # output is NaN, not 0.0. Rows that are non-target but PRESENT
    # contribute NaN; pandas' ``skipna=False`` sum then propagates
    # NaN — but we *want* zero for legitimate "all-zero target arms,
    # no contribution" cases. So we keep non-target as 0.0 only when
    # ``in_target_patch`` has at least one True per group.
    merged["_mass"] = merged["arm_fraction"].astype(float).where(in_target_patch, 0.0)
    has_target_rows = (
        merged.assign(_in=in_target_patch)
        .groupby(grp_cols, as_index=False)["_in"]
        .any()
        .rename(columns={"_in": "_has_target"})
    )
    out = (
        merged.groupby(grp_cols, as_index=False)["_mass"]
        .agg(strict_sum)
        .rename(columns={"_mass": "chosen_patch_fraction"})
        .merge(has_target_rows, on=grp_cols, how="left")
    )
    # No target-patch rows merged in → return NaN (cannot tell
    # "no replay on chosen patch" from "we have no arm data").
    out.loc[~out["_has_target"].fillna(False), "chosen_patch_fraction"] = np.nan
    out = out.drop(columns=["_has_target"])
    return out


def ends_at_next_patch(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: last in-event MAP arm's patch equals ``next_patch``.

    The "last" bin is identified by the maximum ``t_in_event`` per
    event. The MAP arm is mapped to a patch via
    :data:`TRACK_SEGMENT_TO_PATCH`. Bins whose MAP arm has no patch
    label (NaN, junction-bin artifact) yield NaN and are excluded
    from the equality test (the row is recorded as False).

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``ends_at_next_patch`` (bool).
        Session-key columns passed through.
    """
    if event_table.empty or "next_patch" not in event_table.columns:
        return _empty_metric_frame(event_table, "ends_at_next_patch")
    ev = event_table.dropna(subset=["next_patch"]).copy()
    ev["next_patch"] = ev["next_patch"].astype(int)

    sess_keys = present_session_columns(event_sequence_table)
    last_bin = (
        event_sequence_table.sort_values([*sess_keys, "event_id", "t_in_event"])
        .groupby([*sess_keys, "event_id"], as_index=False)
        .tail(1)[[*sess_keys, "event_id", "decoded_arm_map"]]
        .rename(columns={"decoded_arm_map": "_last_arm"})
    )
    last_bin["_last_patch"] = last_bin["_last_arm"].map(TRACK_SEGMENT_TO_PATCH)

    merged = ev.merge(last_bin, on=[*sess_keys, "event_id"], how="left")
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    # Compute equality on nullable Int64, then fill ``pd.NA`` with
    # False and cast to plain ``bool`` so this column matches the
    # docstring contract ("recorded as False" when the last MAP arm
    # has no patch label) and so ``_per_session_metric_means``'s
    # ``astype(float)`` → ``np.nanmean`` reduction counts those rows
    # in the denominator. Without the fillna, ``<NA>`` becomes NaN
    # under ``astype(float)`` and gets silently dropped, inflating
    # the per-session mean.
    eq = merged["_last_patch"].astype("Int64") == merged["next_patch"].astype("Int64")
    out["ends_at_next_patch"] = eq.fillna(False).astype(bool)
    return out


def compute_retrospective_metrics(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    include_prev: bool = True,
) -> pd.DataFrame:
    """Join the retrospective metrics into a single per-event table.

    Parameters
    ----------
    include_prev : bool, default True
        Also compute the ``prev_arm`` retrospective mirror:
        ``chosen_arm_fraction_prev``, ``ends_at_prev``,
        ``contains_path_from_prev``. Requires a ``prev_arm`` column.

    Returns
    -------
    pd.DataFrame
        One row per event with valid ``occupancy_well_arm`` / ``next_arm`` /
        ``next_patch``. Always includes ``chosen_arm_fraction``,
        ``ends_at_next``, ``contains_path_to_next``. Adds
        ``chosen_patch_fraction``, ``ends_at_next_patch`` when the event
        table has ``next_patch``, and the ``prev_arm`` mirror when
        ``include_prev=True``.
    """
    sess_keys = present_session_columns(event_table)
    cf = chosen_arm_fraction(event_table, event_arm_table)
    en = ends_at_next(event_table, event_sequence_table)
    cp = contains_path_to_next(event_table, event_sequence_table)
    out = cf.merge(en, on=[*sess_keys, "event_id"], how="outer").merge(
        cp, on=[*sess_keys, "event_id"], how="outer"
    )
    if "next_patch" in event_table.columns:
        cpf = chosen_patch_fraction(event_table, event_arm_table)
        enp = ends_at_next_patch(event_table, event_sequence_table)
        if not cpf.empty:
            out = out.merge(cpf, on=[*sess_keys, "event_id"], how="outer")
        if not enp.empty:
            out = out.merge(enp, on=[*sess_keys, "event_id"], how="outer")
    if include_prev and "prev_arm" in event_table.columns:
        for prev in (
            chosen_arm_fraction_prev(event_table, event_arm_table),
            ends_at_prev(event_table, event_sequence_table),
            contains_path_from_prev(event_table, event_sequence_table),
        ):
            if not prev.empty:
                out = out.merge(prev, on=[*sess_keys, "event_id"], how="outer")
    return out
