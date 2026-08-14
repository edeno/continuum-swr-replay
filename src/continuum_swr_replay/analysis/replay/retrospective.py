"""Retrospective alignment of replay content with the next choice.

Where the prospective choice models ask whether replay content predicts
the next choice (adding `arm_fraction` to a behavior-only baseline), these
metrics ask the weaker, post-hoc question: does the replay content during
a stop align with whichever arm the animal actually visited next?

Five event-level metrics are computed across three comparison families:

- ``chosen_arm_fraction``: per-event ``arm_fraction`` at ``arm == next_arm``.
- ``ends_at_next``: does the last in-event MAP arm equal ``next_arm``?
- ``contains_path_to_next``: does the event's MAP-arm trace cover the
  geometric path from ``current_arm`` to ``next_arm``?
- ``chosen_patch_fraction``: replay mass assigned to ``next_patch``.
- ``ends_at_next_patch``: does the final MAP arm belong to ``next_patch``?

The null is :func:`next_arm_circular_shift` (in :mod:`shuffles`), with one
future label per contiguous current-arm run within ``stop_id``. Arm metrics
condition on current arm, patch metrics on current patch, and path coverage on
current arm, exact route length, and a coarse event-duration bin. Each family
has its own observable cohort and denominator. Non-identifiable strata are
excluded and reported rather than treated as evidence of no effect.

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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)

from ._helpers import strict_sum
from .shuffles import next_arm_circular_shift

__all__ = [
    "arm_path_well_to_well",
    "at_well_event_cohort",
    "chosen_arm_fraction",
    "chosen_arm_fraction_prev",
    "chosen_patch_fraction",
    "compute_retrospective_metrics",
    "contains_path_from_prev",
    "contains_path_to_next",
    "ends_at_next",
    "ends_at_next_patch",
    "ends_at_prev",
    "freeze_retrospective_cohort",
    "retrospective_alignment_with_shuffle",
]


# Patch → set of arm IDs that fall in that patch (leaf + trunk).
# Built once from TRACK_SEGMENT_TO_PATCH so the topology has a single
# source of truth.
_PATCH_TO_ARM_SET: dict[int, frozenset[int]] = {
    p: frozenset(arm for arm, patch in TRACK_SEGMENT_TO_PATCH.items() if patch == p)
    for p in set(TRACK_SEGMENT_TO_PATCH.values())
}


# Patch (1/2/3) → trunk arm id. Each patch junction connects to the
# center node via exactly one trunk arm, and the trunk-arm IDs are 5
# offsets above the patch IDs by construction. This is the only place a
# magic offset appears; everything else routes through this mapping.
_PATCH_TO_TRUNK_ARM: dict[int, int] = {1: 6, 2: 7, 3: 8}

# Wells are arms 0–5 (degree-1 leaves of the track graph).
_VALID_WELL_ARMS = frozenset(WELL_PATCH_MAPPING.keys())


def _session_keys(df: pd.DataFrame) -> list[str]:
    """Session-key columns present in ``df``."""
    return [c for c in ("animal", "day", "epoch") if c in df.columns]


def _visit_context_ids(event_table: pd.DataFrame) -> pd.Series:
    """One identifier per behavioral visit: ``(stop, completed trip, arm)``.

    ``stop_id`` identifies an immobility bout, not a visit. A bout can cross a
    trial boundary, and the animal can leave and return before it ends, so a
    bout may contain several visits with different future choices.

    ``completed_trial_id`` is what separates them. Keying on contiguous
    ``current_arm`` runs alone would merge two visits to the *same* well within
    one bout -- the intervening trip is invisible once transit events are
    filtered out -- and the only symptom would be a hard failure on the
    one-label-per-unit check. The completion-order ID advances across that
    trip, so the two visits stay distinct.

    Raises
    ------
    KeyError
        If ``completed_trial_id`` is absent, which marks a table built before
        the event schema carried per-event trial context.
    """
    required = ("stop_id", "completed_trial_id", "current_arm")
    missing = [column for column in required if column not in event_table.columns]
    if missing:
        raise KeyError(
            "Retrospective visit contexts require event-grain column(s) "
            f"{missing}; regenerate event_table.pkl with the current builder"
        )
    if event_table.empty:
        return pd.Series(index=event_table.index, dtype=object)
    return pd.Series(
        list(
            zip(
                event_table["stop_id"],
                event_table["completed_trial_id"],
                event_table["current_arm"],
                strict=True,
            )
        ),
        index=event_table.index,
        dtype=object,
    )


def at_well_event_cohort(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter all three event tables to one explicit at-well event cohort.

    Canonical tables use the nullable ``at_well`` label. Legacy inputs without
    that column are rejected: their ``current_arm`` may contain the retired
    destination-well fallback for in-transit events. The same event keys are
    applied to arm and sequence rows, preventing metric-specific denominators.
    """
    if event_table.empty:
        return (
            event_table.copy(),
            event_arm_table.iloc[0:0].copy(),
            event_sequence_table.iloc[0:0].copy(),
        )
    missing = [
        column for column in ("at_well", "current_arm") if column not in event_table
    ]
    if missing:
        raise KeyError(
            "at-well retrospective cohort requires canonical location column(s) "
            f"{missing}; regenerate event tables"
        )
    current = pd.to_numeric(event_table["current_arm"], errors="coerce")
    keep = pd.Series(np.isfinite(current), index=event_table.index)
    keep &= event_table["at_well"].eq(True).fillna(False)  # noqa: E712
    events = event_table.loc[keep].copy()
    key_cols = [*_session_keys(event_table), "event_id"]
    keys = events[key_cols].drop_duplicates()

    def _restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        missing = [column for column in key_cols if column not in frame]
        if missing:
            raise KeyError(
                f"Event-family table is missing at-well cohort key column(s): {missing}"
            )
        return frame.merge(keys, on=key_cols, how="inner", validate="many_to_one")

    return events, _restrict(event_arm_table), _restrict(event_sequence_table)


def _empty_metric_frame(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Empty metric frame preserving available session keys."""
    return pd.DataFrame(columns=[*_session_keys(df), "event_id", metric_col])


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

    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_arm_table.columns]
    ea = event_arm_table[[*sess_keys, "event_id", "arm", "arm_fraction"]]

    merged = ev.merge(
        ea,
        left_on=[*sess_keys, "event_id", target_col],
        right_on=[*sess_keys, "event_id", "arm"],
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

    sess_keys = [
        c for c in ("animal", "day", "epoch") if c in event_sequence_table.columns
    ]
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

    sess_keys = [
        c for c in ("animal", "day", "epoch") if c in event_sequence_table.columns
    ]
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
    """Per-event boolean: MAP-arm trace covers ``current_arm`` → ``next_arm``.

    "Covers" means every arm in ``arm_path_well_to_well(current_arm, next_arm)``
    appears in the event's ``decoded_arm_map``. This ignores ordering,
    continuity, repeats, and off-path arms. Events with NaN
    ``current_arm``/``next_arm`` are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``contains_path_to_next`` (bool).
    """
    return _contains_path(
        event_table,
        event_sequence_table,
        source_col="current_arm",
        target_col="next_arm",
        out_col="contains_path_to_next",
    )


def contains_path_from_prev(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: MAP-arm trace covers ``prev_arm`` → ``current_arm``.

    The retrospective mirror of :func:`contains_path_to_next`: whether the
    replay covers the route the animal just took (the just-left arm back to
    where it currently is).
    """
    return _contains_path(
        event_table,
        event_sequence_table,
        source_col="prev_arm",
        target_col="current_arm",
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

    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_arm_table.columns]
    ea = event_arm_table[[*sess_keys, "event_id", "arm", "arm_fraction"]].copy()
    # Map arm → patch via TRACK_SEGMENT_TO_PATCH.
    ea["arm_patch"] = ea["arm"].map(TRACK_SEGMENT_TO_PATCH)

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

    sess_keys = [
        c for c in ("animal", "day", "epoch") if c in event_sequence_table.columns
    ]
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
    include_patch: bool = True,
    include_prev: bool = True,
) -> pd.DataFrame:
    """Join the retrospective metrics into a single per-event table.

    Parameters
    ----------
    include_patch : bool, default True
        Also compute ``chosen_patch_fraction`` and
        ``ends_at_next_patch``. Set False for the arm-only legacy
        behavior (matches the original three-metric API).
    include_prev : bool, default True
        Also compute the ``prev_arm`` retrospective mirror:
        ``chosen_arm_fraction_prev``, ``ends_at_prev``,
        ``contains_path_from_prev``. Requires a ``prev_arm`` column.

    Returns
    -------
    pd.DataFrame
        One row per event with valid ``current_arm`` / ``next_arm`` /
        ``next_patch``. Always includes ``chosen_arm_fraction``,
        ``ends_at_next``, ``contains_path_to_next``. Adds
        ``chosen_patch_fraction``, ``ends_at_next_patch`` when
        ``include_patch=True``, and the ``prev_arm`` mirror when
        ``include_prev=True``.
    """
    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_table.columns]
    cf = chosen_arm_fraction(event_table, event_arm_table)
    en = ends_at_next(event_table, event_sequence_table)
    cp = contains_path_to_next(event_table, event_sequence_table)
    out = cf.merge(en, on=[*sess_keys, "event_id"], how="outer").merge(
        cp, on=[*sess_keys, "event_id"], how="outer"
    )
    if include_patch and "next_patch" in event_table.columns:
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


# ─────────────────────────────────────────────────────────────────────
# Shuffle null
# ─────────────────────────────────────────────────────────────────────


_DEFAULT_METRICS = (
    "chosen_arm_fraction",
    "ends_at_next",
    "contains_path_to_next",
    "chosen_patch_fraction",
    "ends_at_next_patch",
)

_ARM_TARGET_METRICS = frozenset({"chosen_arm_fraction", "ends_at_next"})
_PATCH_TARGET_METRICS = frozenset({"chosen_patch_fraction", "ends_at_next_patch"})
_PATH_METRICS = frozenset({"contains_path_to_next"})
_METRIC_FAMILIES = {
    "arm": _ARM_TARGET_METRICS,
    "patch": _PATCH_TARGET_METRICS,
    "path": _PATH_METRICS,
}
_SHUFFLE_VISIT_UNIT = "_retrospective_visit_unit"
_SHUFFLE_CURRENT_ARM = "_retrospective_current_arm"
_SHUFFLE_CURRENT_PATCH = "_retrospective_current_patch"
_SHUFFLE_ROUTE_LENGTH = "_retrospective_route_length"
_SHUFFLE_EVENT_LENGTH_BIN = "_retrospective_event_length_bin"
_SHUFFLE_HELPER_COLUMNS = (
    _SHUFFLE_VISIT_UNIT,
    _SHUFFLE_CURRENT_ARM,
    _SHUFFLE_CURRENT_PATCH,
    _SHUFFLE_ROUTE_LENGTH,
    _SHUFFLE_EVENT_LENGTH_BIN,
)


@dataclass(frozen=True)
class _FrozenRetrospectiveCohort:
    """Common observable cohort plus its calibrated-shuffle definition."""

    event_table: pd.DataFrame
    event_arm_table: pd.DataFrame
    event_sequence_table: pd.DataFrame
    audit: dict[str, int]
    family: str
    strata_cols: tuple[str, ...]
    variation_cols: tuple[str, ...]
    label_cols: tuple[str, ...]


def _validate_requested_metrics(metrics: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and de-duplicate the public metric selection."""
    requested = tuple(dict.fromkeys(metrics))
    if not requested:
        raise ValueError("At least one retrospective metric must be requested")
    unknown = sorted(set(requested).difference(_DEFAULT_METRICS))
    if unknown:
        raise ValueError(f"Unknown retrospective metric(s): {unknown}")
    return requested


def _requested_metrics_by_family(
    metrics: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Partition requested metrics without imposing a cross-family cohort."""
    requested = _validate_requested_metrics(metrics)
    return {
        family: tuple(metric for metric in requested if metric in family_metrics)
        for family, family_metrics in _METRIC_FAMILIES.items()
        if any(metric in family_metrics for metric in requested)
    }


def _compute_requested_metrics(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Compute only selected metrics on one already-frozen event cohort."""
    metrics = _validate_requested_metrics(metrics)
    key_cols = [*_session_keys(event_table), "event_id"]
    out = event_table[key_cols].copy()
    calculators = {
        "chosen_arm_fraction": lambda: chosen_arm_fraction(
            event_table, event_arm_table
        ),
        "ends_at_next": lambda: ends_at_next(event_table, event_sequence_table),
        "contains_path_to_next": lambda: contains_path_to_next(
            event_table, event_sequence_table
        ),
        "chosen_patch_fraction": lambda: chosen_patch_fraction(
            event_table, event_arm_table
        ),
        "ends_at_next_patch": lambda: ends_at_next_patch(
            event_table, event_sequence_table
        ),
    }
    for metric in metrics:
        calculated = calculators[metric]()
        out = out.merge(
            calculated,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    return out


def _per_session_metric_means(
    metrics_df: pd.DataFrame,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Reduce per-event metrics to the session-level statistics
    these metrics report."""
    out: dict[str, float] = {}
    selected = (
        tuple(col for col in _DEFAULT_METRICS if col in metrics_df)
        if metrics is None
        else metrics
    )
    for col in selected:
        out[col] = float(np.nanmean(metrics_df[col].astype(float)))
    return out


def _observable_event_keys(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Keys observable for every requested metric and every label shuffle.

    Endpoint metrics deliberately do *not* require a finite final MAP arm.
    Their scientific question is whether the event ends at the target, so an
    undecoded final bin is a negative endpoint rather than missing data.  Path
    coverage still requires at least one finite decoded arm; otherwise there
    is no decoded sequence on which to evaluate set coverage.
    """
    key_cols = [*_session_keys(event_table), "event_id"]
    keys = event_table[key_cols].drop_duplicates()
    if any(
        metric in {"chosen_arm_fraction", "chosen_patch_fraction"} for metric in metrics
    ):
        if not {"arm", "arm_fraction"}.issubset(event_arm_table):
            raise KeyError(
                "Arm-fraction retrospective metrics require arm and arm_fraction"
            )
        arm = event_arm_table.copy()
        arm["_finite_fraction"] = np.isfinite(
            pd.to_numeric(arm["arm_fraction"], errors="coerce")
        )
        arm_observed = arm.groupby(key_cols, as_index=False).agg(
            _n_arm_rows=("arm", "size"),
            _arm_set=("arm", lambda values: frozenset(values)),
            _all_finite=("_finite_fraction", "all"),
        )
        arm_observed = arm_observed[
            arm_observed["_n_arm_rows"].eq(len(TRACK_SEGMENT_TO_PATCH))
            & arm_observed["_arm_set"].eq(frozenset(TRACK_SEGMENT_TO_PATCH))
            & arm_observed["_all_finite"]
        ][key_cols]
        keys = keys.merge(arm_observed, on=key_cols, how="inner")

    endpoint_metrics = {"ends_at_next", "ends_at_next_patch"}
    path_metrics = {"contains_path_to_next"}
    requested_sequence_metrics = set(metrics) & (endpoint_metrics | path_metrics)
    if requested_sequence_metrics:
        if "decoded_arm_map" not in event_sequence_table:
            raise KeyError("Sequence retrospective metrics require decoded_arm_map")

    if set(metrics) & path_metrics:
        sequence = event_sequence_table.assign(
            _valid_decoded=np.isfinite(
                pd.to_numeric(event_sequence_table["decoded_arm_map"], errors="coerce")
            )
        )
        grouped_sequence = sequence.groupby(key_cols, sort=False)
        sequence_observed = grouped_sequence.agg(
            _any_valid_decoded=("_valid_decoded", "any")
        ).reset_index()
        sequence_observed = sequence_observed[sequence_observed["_any_valid_decoded"]]
        sequence_observed = sequence_observed[key_cols]
        keys = keys.merge(sequence_observed, on=key_cols, how="inner")
    return keys


def _freeze_retrospective_cohort_with_audit(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> _FrozenRetrospectiveCohort:
    """Freeze one metric family's cohort and calibrated shuffle strata."""
    metrics = _validate_requested_metrics(metrics)
    matching_families = [
        family
        for family, family_metrics in _METRIC_FAMILIES.items()
        if set(metrics).issubset(family_metrics)
    ]
    if len(matching_families) != 1:
        raise ValueError(
            "A frozen retrospective cohort can contain only one metric family; "
            f"got {metrics}"
        )
    family = matching_families[0]
    n_events_input = int(len(event_table))
    required_event_columns = {
        "current_arm",
        "next_arm",
        "stop_id",
        "event_midpoint_time",
    }
    if family == "patch":
        required_event_columns.update({"current_patch", "next_patch"})
    if family == "path":
        required_event_columns.add("n_nonlocal_bins")
    missing = sorted(required_event_columns.difference(event_table.columns))
    if missing:
        raise KeyError(
            "Calibrated retrospective shuffle requires canonical event "
            f"column(s) {missing}; regenerate event tables"
        )
    session_cols = _session_keys(event_table)
    if session_cols and event_table[session_cols].drop_duplicates().shape[0] > 1:
        raise ValueError(
            "retrospective alignment requires one session at a time; "
            "cross-session label exchange is invalid"
        )

    events, event_arm, event_sequence = at_well_event_cohort(
        event_table, event_arm_table, event_sequence_table
    )
    n_events_after_at_well = int(len(events))
    numeric_current = pd.to_numeric(events["current_arm"], errors="coerce")
    invalid_current = ~numeric_current.isin(_VALID_WELL_ARMS)
    if invalid_current.any():
        raise ValueError(
            "Retrospective endpoints must be integer well arms 0..5; found "
            f"{int(invalid_current.sum())} invalid current_arm values"
        )
    events["current_arm"] = numeric_current.astype(int)
    if events["stop_id"].isna().any():
        raise ValueError("Retrospective shuffle unit stop_id contains NaN")
    midpoint = pd.to_numeric(events["event_midpoint_time"], errors="coerce")
    if not np.isfinite(midpoint).all():
        raise ValueError("Retrospective event_midpoint_time must be finite")
    events["event_midpoint_time"] = midpoint.astype(float)
    # Derive visits before removing undefined future labels. Otherwise an
    # intervening visit with no next choice could disappear and collapse a
    # later return to the same arm into the earlier visit.
    events[_SHUFFLE_VISIT_UNIT] = _visit_context_ids(events)
    visit_groups = events.groupby(_SHUFFLE_VISIT_UNIT, sort=False, dropna=False)
    available_next_labels = [
        column for column in ("next_arm", "next_patch") if column in events
    ]
    for column in available_next_labels:
        n_unique = visit_groups[column].nunique(dropna=False)
        inconsistent = n_unique[n_unique > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Retrospective shuffle requires one future label per "
                "contiguous visit context; unit "
                f"{inconsistent.index[0]!r} has multiple {column!r} values"
            )

    numeric_next = pd.to_numeric(events["next_arm"], errors="coerce")
    valid_next = np.isfinite(numeric_next)
    events = events.loc[valid_next].copy()
    numeric_next = numeric_next.loc[valid_next]
    invalid_next = ~numeric_next.isin(_VALID_WELL_ARMS)
    if invalid_next.any():
        raise ValueError(
            "Retrospective endpoints must be integer well arms 0..5; found "
            f"{int(invalid_next.sum())} invalid next_arm values"
        )
    events["next_arm"] = numeric_next.astype(int)
    n_events_after_valid_next = int(len(events))

    key_cols = [*_session_keys(events), "event_id"]

    def _restrict_family(
        frame: pd.DataFrame,
        keys: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        missing_keys = sorted(set(key_cols).difference(frame.columns))
        if missing_keys:
            raise KeyError(
                "Event-family table is missing retrospective cohort key "
                f"column(s): {missing_keys}"
            )
        return frame.merge(keys, on=key_cols, how="inner", validate="many_to_one")

    event_keys = events[key_cols].drop_duplicates()
    event_arm = _restrict_family(event_arm, event_keys)
    event_sequence = _restrict_family(event_sequence, event_keys)

    if family == "patch":
        numeric_current_patch = pd.to_numeric(events["current_patch"], errors="coerce")
        numeric_patch = pd.to_numeric(events["next_patch"], errors="coerce")
        expected_current_patch = (
            events["current_arm"].map(WELL_PATCH_MAPPING).astype(float)
        )
        expected_patch = events["next_arm"].map(WELL_PATCH_MAPPING).astype(float)
        invalid_current_patch = ~np.isfinite(
            numeric_current_patch
        ) | numeric_current_patch.ne(expected_current_patch)
        invalid_next_patch = ~np.isfinite(numeric_patch) | numeric_patch.ne(
            expected_patch
        )
        if invalid_current_patch.any() or invalid_next_patch.any():
            raise ValueError(
                "Patch retrospective metrics require finite patch labels "
                "consistent with their arms; found "
                f"{int(invalid_current_patch.sum())} invalid current_patch and "
                f"{int(invalid_next_patch.sum())} invalid next_patch event(s)"
            )
        events["current_patch"] = numeric_current_patch.astype(int)
        events["next_patch"] = numeric_patch.astype(int)

    observable_keys = _observable_event_keys(
        events,
        event_arm,
        event_sequence,
        metrics,
    )
    events = events.merge(
        observable_keys, on=key_cols, how="inner", validate="one_to_one"
    )
    event_arm = _restrict_family(event_arm, observable_keys)
    event_sequence = _restrict_family(event_sequence, observable_keys)
    n_events_after_observability = int(len(events))

    events[_SHUFFLE_CURRENT_ARM] = events["current_arm"].astype(int)

    if family == "arm":
        strata_cols = [_SHUFFLE_CURRENT_ARM]
        variation_cols = ["next_arm"]
        label_cols = ["next_arm"]
    elif family == "patch":
        events[_SHUFFLE_CURRENT_PATCH] = events["current_patch"].astype(int)
        strata_cols = [_SHUFFLE_CURRENT_PATCH]
        variation_cols = ["next_patch"]
        # Shift the paired future-arm label with its patch label even though
        # patch metrics use only next_patch.
        label_cols = ["next_arm", "next_patch"]
    else:
        events[_SHUFFLE_ROUTE_LENGTH] = np.fromiter(
            (
                len(arm_path_well_to_well(current, target))
                for current, target in zip(
                    events["current_arm"], events["next_arm"], strict=True
                )
            ),
            dtype=np.int64,
            count=len(events),
        )
        lengths = pd.to_numeric(events["n_nonlocal_bins"], errors="coerce")
        invalid_length = (
            ~np.isfinite(lengths) | lengths.le(0) | lengths.ne(np.floor(lengths))
        )
        if invalid_length.any():
            raise ValueError(
                "Path retrospective metrics require positive integer "
                f"n_nonlocal_bins; found {int(invalid_length.sum())} invalid event(s)"
            )
        # Labels move at visit-context grain, so event duration also needs a
        # visit-grain summary. Octave bins of the within-visit median preserve
        # broad duration opportunity without making almost every visit unique.
        visit_median_length = lengths.groupby(events[_SHUFFLE_VISIT_UNIT]).transform(
            "median"
        )
        events[_SHUFFLE_EVENT_LENGTH_BIN] = np.floor(
            np.log2(visit_median_length)
        ).astype(int)
        strata_cols = [
            _SHUFFLE_CURRENT_ARM,
            _SHUFFLE_ROUTE_LENGTH,
            _SHUFFLE_EVENT_LENGTH_BIN,
        ]
        variation_cols = ["next_arm"]
        label_cols = ["next_arm"]

    grouped_units = events.groupby(_SHUFFLE_VISIT_UNIT, sort=False, dropna=False)
    unit_constant_cols = list(strata_cols)
    for column in dict.fromkeys(unit_constant_cols):
        n_unique = grouped_units[column].nunique(dropna=False)
        inconsistent = n_unique[n_unique > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Retrospective shuffle requires constant strata within each "
                "contiguous visit context; unit "
                f"{inconsistent.index[0]!r} has multiple {column!r} values"
            )

    unit_rows = events.drop_duplicates(_SHUFFLE_VISIT_UNIT, keep="first")[
        [_SHUFFLE_VISIT_UNIT, *strata_cols, *label_cols]
    ].copy()
    stratum_group = unit_rows.groupby(strata_cols, sort=False, dropna=False)
    n_units_in_stratum = stratum_group[_SHUFFLE_VISIT_UNIT].transform("size")
    singleton = n_units_in_stratum.lt(2)
    # Every relevant future label must vary. A stratum that is constant in
    # even one requested target cannot identify that target's null contrast.
    any_constant = pd.Series(False, index=unit_rows.index)
    for column in variation_cols:
        any_constant |= stratum_group[column].transform("nunique").lt(2)
    deterministic = n_units_in_stratum.ge(2) & any_constant
    movable = ~(singleton | deterministic)
    retained_units = unit_rows.loc[movable, _SHUFFLE_VISIT_UNIT]
    retained_unit_set = set(retained_units)

    n_strata_input = int(unit_rows[strata_cols].drop_duplicates().shape[0])
    n_strata_singleton = int(
        unit_rows.loc[singleton, strata_cols].drop_duplicates().shape[0]
    )
    n_strata_deterministic = int(
        unit_rows.loc[deterministic, strata_cols].drop_duplicates().shape[0]
    )
    n_units_input = int(len(unit_rows))
    n_units_singleton = int(singleton.sum())
    n_units_deterministic = int(deterministic.sum())

    events = events.loc[events[_SHUFFLE_VISIT_UNIT].isin(retained_unit_set)].copy()
    retained_keys = events[key_cols].drop_duplicates()
    event_arm = _restrict_family(event_arm, retained_keys)
    event_sequence = _restrict_family(event_sequence, retained_keys)
    audit = {
        "n_events_input": n_events_input,
        "n_events_after_at_well": n_events_after_at_well,
        "n_events_after_valid_next": n_events_after_valid_next,
        "n_events_after_common_observability": n_events_after_observability,
        "n_events_excluded_immovable": (
            n_events_after_observability - int(len(events))
        ),
        "n_shuffle_units_input": n_units_input,
        "n_shuffle_units": int(len(retained_units)),
        "n_shuffle_units_excluded_singleton": n_units_singleton,
        "n_shuffle_units_excluded_deterministic": n_units_deterministic,
        "n_shuffle_strata_input": n_strata_input,
        "n_shuffle_strata": n_strata_input
        - n_strata_singleton
        - n_strata_deterministic,
        "n_shuffle_strata_excluded_singleton": n_strata_singleton,
        "n_shuffle_strata_excluded_deterministic": n_strata_deterministic,
    }
    return _FrozenRetrospectiveCohort(
        event_table=events,
        event_arm_table=event_arm,
        event_sequence_table=event_sequence,
        audit=audit,
        family=family,
        strata_cols=tuple(strata_cols),
        variation_cols=tuple(variation_cols),
        label_cols=tuple(label_cols),
    )


def freeze_retrospective_cohort(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the union of requested families' identifiable cohorts.

    Each metric family is frozen independently. The returned event keys are
    their union, not a denominator for inference; use the per-metric counts from
    :func:`retrospective_alignment_with_shuffle` for scientific reporting.

    Returns
    -------
    tuple of pd.DataFrame
        The three tables restricted to the union of frozen family cohorts.
    """
    family_metrics = _requested_metrics_by_family(metrics)
    frozen_families = [
        _freeze_retrospective_cohort_with_audit(
            event_table,
            event_arm_table,
            event_sequence_table,
            metrics=selected,
        )
        for selected in family_metrics.values()
    ]
    key_cols = [*_session_keys(event_table), "event_id"]
    union_events = (
        pd.concat(
            [frozen.event_table for frozen in frozen_families],
            ignore_index=True,
        )
        .drop_duplicates(key_cols, keep="first")
        .drop(columns=_SHUFFLE_HELPER_COLUMNS, errors="ignore")
    )
    union_keys = union_events[key_cols].drop_duplicates()

    def _restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        return frame.merge(
            union_keys,
            on=key_cols,
            how="inner",
            validate="many_to_one",
        )

    return union_events, _restrict(event_arm_table), _restrict(event_sequence_table)


def retrospective_alignment_with_shuffle(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    n_shuffles: int = 100,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int | bool | str]:
    """Real vs. shuffled alignment for one session.

    Each metric family gets its own visit-context cohort and calibrated null:
    arm metrics condition on ``current_arm``; patch metrics condition on
    ``current_patch``; path coverage conditions on ``current_arm``, exact route
    length, and coarse duration. The statistic is real minus shuffled.

    Metrics share a cohort only within their family. A family with no movable
    stratum reports zero events and NaN statistics without removing observable
    events from another family.

    Parameters
    ----------
    event_table, event_arm_table, event_sequence_table : pd.DataFrame
        Must come from a SINGLE session. Cross-session shuffling
        would cross-contaminate behavioral statistics.
    n_shuffles : int, default 100
        Number of circular-shift draws.
    metrics : tuple of str
        Subset of :data:`_DEFAULT_METRICS`. Useful in tests for
        skipping the sequence-table-dependent metrics.
    rng : numpy Generator, optional

    Returns
    -------
    dict
        Keys: ``real_mean_<metric>``, ``shuffled_mean_<metric>``,
        ``real_minus_shuffled_<metric>``, ``shuffled_std_<metric>``,
        ``n_events_<metric>`` and family-specific retained/excluded visit and
        stratum audits. A legacy ``n_events`` key is included only when one
        metric family was requested; multi-family output has no misleading
        global denominator. Provenance fields record the visit-unit and cohort
        policy; path output explicitly records its movable-strata scope.

    Raises
    ------
    KeyError
        If a required column is absent.
    RuntimeError
        If a metric's observed count disagrees with the frozen cohort
        size, which would mean the common-cohort guarantee was broken.
    """
    if rng is None:
        rng = np.random.default_rng()
    metrics = _validate_requested_metrics(metrics)
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be nonnegative")
    family_metrics = _requested_metrics_by_family(metrics)
    out: dict[str, float | int | bool | str] = {
        "n_shuffles": int(n_shuffles),
        "cohort_policy": "metric_family_specific",
        "shuffle_unit_definition": "contiguous_current_arm_run_within_stop_id",
    }
    if "path" in family_metrics:
        out["path_null_scope"] = "movable_current_arm_route_length_duration_strata"
        out["path_null_effectively_cross_patch"] = True

    for family, selected_metrics in family_metrics.items():
        frozen = _freeze_retrospective_cohort_with_audit(
            event_table,
            event_arm_table,
            event_sequence_table,
            metrics=selected_metrics,
        )
        family_suffix = f"{family}_family"
        n_events = int(len(frozen.event_table))
        for audit_name, value in frozen.audit.items():
            out[f"{audit_name}_{family_suffix}"] = value
        out[f"n_events_{family_suffix}"] = n_events

        if len(family_metrics) == 1:
            out.update(frozen.audit)
            out["n_events"] = n_events

        if frozen.audit["n_shuffle_units"] < 2:
            for metric in selected_metrics:
                out[f"n_events_{metric}"] = 0
                out[f"real_mean_{metric}"] = float("nan")
                out[f"shuffled_mean_{metric}"] = float("nan")
                out[f"shuffled_std_{metric}"] = float("nan")
                out[f"real_minus_shuffled_{metric}"] = float("nan")
            continue

        real = _compute_requested_metrics(
            frozen.event_table,
            frozen.event_arm_table,
            frozen.event_sequence_table,
            selected_metrics,
        )
        real_counts = {
            metric: int(real[metric].notna().sum()) for metric in selected_metrics
        }
        for metric, count in real_counts.items():
            if count != n_events:
                raise RuntimeError(
                    f"Retrospective metric {metric} observed {count} events but "
                    f"its frozen {family} cohort contains {n_events}"
                )
        real_means = _per_session_metric_means(real, selected_metrics)
        shuffled_acc: dict[str, list[float]] = {
            metric: [] for metric in selected_metrics
        }
        for _ in range(n_shuffles):
            shuffled_ev = next_arm_circular_shift(
                frozen.event_table,
                columns=frozen.label_cols,
                unit_col=_SHUFFLE_VISIT_UNIT,
                strata_cols=frozen.strata_cols,
                variation_cols=frozen.variation_cols,
                require_movement=True,
                rng=rng,
            )
            shuffled_metrics = _compute_requested_metrics(
                shuffled_ev,
                frozen.event_arm_table,
                frozen.event_sequence_table,
                selected_metrics,
            )
            shuffled_means = _per_session_metric_means(
                shuffled_metrics, selected_metrics
            )
            for metric in selected_metrics:
                shuffled_count = int(shuffled_metrics[metric].notna().sum())
                if shuffled_count != n_events:
                    raise RuntimeError(
                        f"Shuffled retrospective metric {metric} observed "
                        f"{shuffled_count} events but its frozen {family} "
                        f"cohort contains {n_events}"
                    )
                shuffled_acc[metric].append(shuffled_means[metric])

        for metric in selected_metrics:
            out[f"real_mean_{metric}"] = real_means[metric]
            out[f"n_events_{metric}"] = real_counts[metric]
            if shuffled_acc[metric]:
                shuffled_mean = float(np.mean(shuffled_acc[metric]))
                out[f"shuffled_mean_{metric}"] = shuffled_mean
                out[f"shuffled_std_{metric}"] = float(np.std(shuffled_acc[metric]))
                out[f"real_minus_shuffled_{metric}"] = (
                    real_means[metric] - shuffled_mean
                )
            else:
                out[f"shuffled_mean_{metric}"] = float("nan")
                out[f"shuffled_std_{metric}"] = float("nan")
                out[f"real_minus_shuffled_{metric}"] = float("nan")
    return out
