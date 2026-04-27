"""Stage 5B: retrospective alignment of replay content with the next choice.

Where Stage 5A asked whether replay content predicts the next choice
(adding `arm_fraction` to a behavior-only baseline), Stage 5B asks the
weaker, post-hoc question: does the replay content during a stop align
with whichever arm the animal actually visited next?

Three event-level metrics are computed:

- ``chosen_arm_fraction``: per-event ``arm_fraction`` at ``arm == next_arm``.
- ``ends_at_next``: does the last in-event MAP arm equal ``next_arm``?
- ``contains_path_to_next``: does the event's MAP-arm trace cover the
  geometric path from ``current_arm`` to ``next_arm``?

The natural null is :func:`next_arm_circular_shift` (in
:mod:`shuffles`), which shifts ``next_arm`` across events in temporal
order, preserving the marginal distribution of ``next`` and slow drifts
in patch preference but breaking the per-event alignment.

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

from .shuffles import next_arm_circular_shift

__all__ = [
    "arm_path_well_to_well",
    "chosen_arm_fraction",
    "chosen_patch_fraction",
    "compute_retrospective_metrics",
    "contains_path_to_next",
    "ends_at_next",
    "ends_at_next_patch",
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
# offsets above the patch IDs by construction (CLAUDE.md "Spatial
# Bandit Task Structure"). This is the only place a magic offset
# appears; everything else routes through this mapping.
_PATCH_TO_TRUNK_ARM: dict[int, int] = {1: 6, 2: 7, 3: 8}

# Wells are arms 0–5 (degree-1 leaves of the track graph).
_VALID_WELL_ARMS = frozenset(WELL_PATCH_MAPPING.keys())


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


def chosen_arm_fraction(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event ``arm_fraction`` at ``arm == next_arm``.

    Events with ``next_arm`` NaN are dropped (the animal never left,
    or the session ended before another stop occurred).

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``chosen_arm_fraction``. Session-key
        columns (``animal``/``day``/``epoch``) are passed through if
        present in ``event_table``.
    """
    if event_table.empty:
        return pd.DataFrame(columns=["event_id", "chosen_arm_fraction"])
    ev = event_table.dropna(subset=["next_arm"]).copy()
    ev["next_arm"] = ev["next_arm"].astype(int)

    # Join (event_id, next_arm) onto event_arm_table to pick the chosen
    # arm's fraction. We keep `arm_fraction` only.
    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_arm_table.columns]
    ea = event_arm_table[[*sess_keys, "event_id", "arm", "arm_fraction"]]

    merged = ev.merge(
        ea,
        left_on=[*sess_keys, "event_id", "next_arm"],
        right_on=[*sess_keys, "event_id", "arm"],
        how="left",
    )
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    out["chosen_arm_fraction"] = merged["arm_fraction"].astype(float)
    return out


def ends_at_next(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: last in-event MAP arm equals ``next_arm``.

    The "last" bin is identified by the maximum ``t_in_event`` per
    event. Events with NaN ``next_arm`` are dropped (no choice to
    align to).

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``ends_at_next`` (bool). Session-key
        columns are passed through if present.
    """
    if event_table.empty:
        return pd.DataFrame(columns=["event_id", "ends_at_next"])
    ev = event_table.dropna(subset=["next_arm"]).copy()
    ev["next_arm"] = ev["next_arm"].astype(int)

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
    out["ends_at_next"] = merged["_last_arm"].astype("Int64") == merged[
        "next_arm"
    ].astype("Int64")
    return out


def contains_path_to_next(
    event_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event boolean: MAP-arm trace covers the geometric path
    from ``current_arm`` to ``next_arm``.

    "Covers" means every arm in
    :func:`arm_path_well_to_well(current_arm, next_arm)` appears at
    least once in the event's ``decoded_arm_map``. Events with NaN
    ``current_arm`` or ``next_arm`` are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: ``event_id``, ``contains_path_to_next`` (bool).
        Session-key columns are passed through if present.
    """
    if event_table.empty:
        return pd.DataFrame(columns=["event_id", "contains_path_to_next"])
    ev = event_table.dropna(subset=["current_arm", "next_arm"]).copy()
    ev["current_arm"] = ev["current_arm"].astype(int)
    ev["next_arm"] = ev["next_arm"].astype(int)

    sess_keys = [
        c for c in ("animal", "day", "epoch") if c in event_sequence_table.columns
    ]
    arm_sets = (
        event_sequence_table.groupby([*sess_keys, "event_id"], as_index=False)[
            "decoded_arm_map"
        ]
        .agg(lambda s: frozenset(int(x) for x in s.dropna().unique()))
        .rename(columns={"decoded_arm_map": "_decoded_arm_set"})
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
                merged["current_arm"].to_numpy(),
                merged["next_arm"].to_numpy(),
                merged["_decoded_arm_set"].to_numpy(),
                strict=True,
            )
        ],
        dtype=bool,
    )
    out_cols = [*sess_keys, "event_id"]
    out = merged[out_cols].copy()
    out["contains_path_to_next"] = contains
    return out


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
        return pd.DataFrame(columns=["event_id", "chosen_patch_fraction"])
    ev = event_table.dropna(subset=["next_patch"]).copy()
    ev["next_patch"] = ev["next_patch"].astype(int)

    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_arm_table.columns]
    ea = event_arm_table[[*sess_keys, "event_id", "arm", "arm_fraction"]].copy()
    # Map arm → patch via TRACK_SEGMENT_TO_PATCH.
    ea["arm_patch"] = ea["arm"].map(TRACK_SEGMENT_TO_PATCH)

    merged = ev.merge(ea, on=[*sess_keys, "event_id"], how="left")
    in_target_patch = merged["arm_patch"] == merged["next_patch"]
    merged["_mass"] = merged["arm_fraction"].astype(float).where(in_target_patch, 0.0)

    grp_cols = [*sess_keys, "event_id"]
    out = (
        merged.groupby(grp_cols, as_index=False)["_mass"]
        .sum()
        .rename(columns={"_mass": "chosen_patch_fraction"})
    )
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
        return pd.DataFrame(columns=["event_id", "ends_at_next_patch"])
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
    out["ends_at_next_patch"] = (
        merged["_last_patch"].astype("Int64") == merged["next_patch"].astype("Int64")
    )
    return out


def compute_retrospective_metrics(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    include_patch: bool = True,
) -> pd.DataFrame:
    """Join the Stage 5B metrics into a single per-event table.

    Parameters
    ----------
    include_patch : bool, default True
        Also compute ``chosen_patch_fraction`` and
        ``ends_at_next_patch``. Set False for the arm-only legacy
        behavior (matches the original three-metric API).

    Returns
    -------
    pd.DataFrame
        One row per event with valid ``current_arm`` / ``next_arm`` /
        ``next_patch``. Always includes ``chosen_arm_fraction``,
        ``ends_at_next``, ``contains_path_to_next``. Adds
        ``chosen_patch_fraction``, ``ends_at_next_patch`` when
        ``include_patch=True``.
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


def _per_session_metric_means(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Reduce per-event metrics to the session-level statistics
    Stage 5B reports."""
    out: dict[str, float] = {}
    for col in _DEFAULT_METRICS:
        if col in metrics_df.columns:
            out[col] = float(np.nanmean(metrics_df[col].astype(float)))
    return out


def retrospective_alignment_with_shuffle(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    n_shuffles: int = 100,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Real vs. shuffled alignment for one session.

    Each shuffle calls :func:`next_arm_circular_shift` on
    ``event_table`` and recomputes the requested metrics. The
    statistic of interest is real minus shuffled for each metric.

    Parameters
    ----------
    event_table, event_arm_table, event_sequence_table : pd.DataFrame
        Must come from a SINGLE session. Cross-session shuffling
        would cross-contaminate behavioral statistics.
    n_shuffles : int, default 100
        Number of circular-shift draws.
    metrics : tuple of str
        Subset of ``("chosen_arm_fraction", "ends_at_next",
        "contains_path_to_next")``. Useful in tests for skipping the
        sequence-table-dependent metrics.
    rng : numpy Generator, optional

    Returns
    -------
    dict
        Keys: ``real_mean_<metric>``, ``shuffled_mean_<metric>``,
        ``real_minus_shuffled_<metric>``, ``shuffled_std_<metric>``,
        ``n_events``, ``n_shuffles``.
    """
    if rng is None:
        rng = np.random.default_rng()

    sess_keys = [c for c in ("animal", "day", "epoch") if c in event_table.columns]
    if sess_keys:
        n_sess = event_table[sess_keys].drop_duplicates().shape[0]
        if n_sess > 1:
            raise ValueError(
                f"retrospective_alignment_with_shuffle received {n_sess} "
                "sessions; pass one session at a time."
            )

    real = compute_retrospective_metrics(
        event_table, event_arm_table, event_sequence_table
    )
    real_means = _per_session_metric_means(real)

    shuffled_acc: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_shuffles):
        shuffled_ev = next_arm_circular_shift(event_table, rng=rng)
        sm = compute_retrospective_metrics(
            shuffled_ev, event_arm_table, event_sequence_table
        )
        sm_means = _per_session_metric_means(sm)
        for m in metrics:
            if m in sm_means:
                shuffled_acc[m].append(sm_means[m])

    out: dict[str, float] = {
        "n_events": int(len(real)),
        "n_shuffles": int(n_shuffles),
    }
    for m in metrics:
        if m not in real_means:
            continue
        out[f"real_mean_{m}"] = real_means[m]
        if shuffled_acc[m]:
            out[f"shuffled_mean_{m}"] = float(np.mean(shuffled_acc[m]))
            out[f"shuffled_std_{m}"] = float(np.std(shuffled_acc[m]))
            out[f"real_minus_shuffled_{m}"] = real_means[m] - float(
                np.mean(shuffled_acc[m])
            )
        else:
            out[f"shuffled_mean_{m}"] = float("nan")
            out[f"shuffled_std_{m}"] = float("nan")
            out[f"real_minus_shuffled_{m}"] = float("nan")
    return out
