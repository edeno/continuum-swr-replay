"""Per-event replay-trajectory rollups, shared by both event-table builders.

Each family is a pure function over per-timebin arrays already filtered to
in-event rows and grouped by ``event_id``. Every family returns
``(unique_sorted_event_ids, {column_name: per_event_array})`` so callers assign
columns the same way the posterior rollups in :mod:`uncertainty` do. The
``unique_sorted_event_ids`` equals ``np.unique(event_id)``, matching the
``groupby(sort=True)`` order the table builders assert against.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from ...data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from ..hippocampus.posterior_analysis import map_estimate_to_map_nodes
from .spatial import arm_adjacency, path_continuity
from .summaries import _group_by_event

RollupResult = tuple[np.ndarray, dict[str, np.ndarray]]

#: trunk arms join the three patch junctions to the centre of the track
_TRUNK_ARMS = frozenset({6, 7, 8})

#: per-event columns produced identically for both scenarios (parity target).
SHARED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    # trajectory_dynamics
    "replay_start_position_cm", "replay_end_position_cm", "start_arm", "end_arm", "net_displacement_cm",
    "path_length_cm", "tortuosity", "replay_speed_median_cm_s", "replay_speed_p90_cm_s",
    "replay_speed_continuous_median_cm_s", "step_distance_median_cm", "step_distance_p90_cm",
    "step_distance_max_cm", "spatial_extent_cm", "n_reversals", "pause_fraction",
    "n_patches_visited", "crosses_trunk", "terminates_at_well", "compression_factor",
    # ahead_behind_summary
    "mean_abs_ahead_behind_cm", "max_ahead_cm", "max_behind_cm", "max_abs_ahead_behind_cm",
    "remote_fraction", "ab_slope_cm_s", "is_outbound",
    # uncertainty_extra
    "hpd_coverage_cm", "multimodal_fraction",
    # path_continuity
    "path_continuity_fraction",
    # neural signal means (z-scored; the raw traces require re-loading data
    # that has drifted from the decode axis, so the stored per-session
    # z-scores are used directly)
    "ripple_power_z", "mua_rate_z", "theta_z",
    # state means shared by both scenarios
    "continuous_prob_mean", "fragmented_prob_mean",
)

#: per-event columns specific to the 4-state Non-Local model.
NONLOCAL_ONLY_COLUMNS: tuple[str, ...] = (
    "local_prob_mean", "nospike_prob_mean", "nonlocal_fraction", "nospike_fraction",
)

#: non-state, non-signal data_over_time columns the rollup families read
#: directly (``decoded_arm`` / ``track_segment_id`` are added by the table
#: builder's ``_prepare_dot`` before the rollups run).
_REQUIRED_DOT_COLUMNS: tuple[str, ...] = (
    "map_estimate", "step_distance", "map_speed", "decoded_arm",
    "run_speed_baseline", "ahead_behind_distance", "hpd", "track_segment_id",
)

#: per-session z-scored neural signals already in data_over_time (both
#: scenarios), mapped to their per-event rollup column names. Raw traces are
#: not rolled up — re-loading them returns a time axis trimmed since the
#: decodes (post-trial padding removed by ``get_position_info``), whereas the
#: stored z-scores cover the full decode axis.
_SIGNAL_COLUMNS: dict[str, str] = {
    "ripple_power_z": "SWR_zscore",
    "mua_rate_z": "MUA_zscore",
    "theta_z": "theta_power_zscore",
}


def _p90(values: np.ndarray) -> float:
    """90th percentile ignoring NaN."""
    return float(np.nanpercentile(values, 90))


def _nan_safe(col: str, reducer):
    """Per-event reducer over ``g[col]`` that returns NaN on an all-NaN group.

    Avoids the ``All-NaN slice`` RuntimeWarning that ``nanmedian``/``nanmax``/
    ``nanpercentile`` emit for an event whose bins are all junction
    placeholders (NaN ``step_distance``), matching the guarded speed stats.
    """
    def _fn(g: pd.DataFrame) -> float:
        v = g[col].to_numpy()
        return float(reducer(v)) if np.isfinite(v).any() else np.nan

    return _fn


def brain_state_means(
    *,
    event_id: np.ndarray,
    state_probs: dict[str, np.ndarray],
    extra_means: dict[str, np.ndarray],
    nonlocal_prob: np.ndarray | None = None,
    non_local_threshold: float = 0.8,
) -> RollupResult:
    """Per-event means of state posteriors and raw/z-scored neural signals.

    Parameters
    ----------
    event_id : np.ndarray, shape (T,)
    nonlocal_prob : np.ndarray, shape (T,)
        Per-timebin non-local probability ``w_t`` (Continuous + Fragmented).
    state_probs : dict[str, np.ndarray]
        Per-timebin state posteriors to average, keyed by output column name
        (e.g. ``local_prob_mean`` -> ``Local_prob`` column). The Non-Local
        table passes all four states; ContFrag passes Continuous/Fragmented.
    extra_means : dict[str, np.ndarray]
        Other per-timebin signals to average per event, keyed by output column
        name (e.g. ``ripple_power``, ``ripple_power_z``, ``mua_rate`` ...).

    Returns
    -------
    (event_ids, {column: per_event_mean})
        Plus ``nonlocal_fraction`` (fraction of bins with ``w_t`` above the
        threshold) and ``nospike_fraction`` when a ``nospike_prob_mean`` state
        is supplied.
    """
    unique, inv, n_events = _group_by_event(event_id)

    def _mean(values: np.ndarray) -> np.ndarray:
        # NaN-aware per-event mean: a raw LFP/MUA signal can carry NaN where a
        # coverage gap left a bin unmatched, so average only the finite bins
        # per event (an event entirely in a gap returns NaN).
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        total = np.bincount(inv[finite], weights=v[finite], minlength=n_events)
        cnt = np.bincount(inv[finite], minlength=n_events).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cnt > 0, total / cnt, np.nan)

    out = {col: _mean(v) for col, v in {**state_probs, **extra_means}.items()}

    # ``nonlocal_fraction`` only applies to the 4-state Non-Local model;
    # the 2-state ContFrag model has no ``w_t`` (both states are non-local).
    if nonlocal_prob is not None:
        out["nonlocal_fraction"] = _mean(np.asarray(nonlocal_prob, float) > non_local_threshold)
    return unique, out


def ahead_behind_summary(
    *,
    event_id: np.ndarray,
    ahead_behind_distance: np.ndarray,
    decoded_arm: np.ndarray,
    current_arm: np.ndarray,
    time: np.ndarray,
) -> RollupResult:
    """Reach, remoteness, and geometric direction from ahead/behind distance.

    Columns: ``mean_abs_ahead_behind``, ``max_ahead``, ``max_behind``,
    ``max_abs_ahead_behind``, ``remote_fraction``, ``ab_slope``,
    ``is_outbound``.
    """
    df = pd.DataFrame(
        {
            "event_id": np.asarray(event_id),
            "ab": np.asarray(ahead_behind_distance, float),
            "decoded_arm": np.asarray(decoded_arm, float),
            "current_arm": np.asarray(current_arm, float),
            "time": np.asarray(time, float),
        }
    )
    grouped = df.groupby("event_id", sort=True)

    def _slope(g: pd.DataFrame) -> float:
        m = np.isfinite(g["ab"]) & np.isfinite(g["time"])
        if m.sum() < 2:
            return np.nan
        return float(np.polyfit(g["time"][m], g["ab"][m], 1)[0])

    def _remote(g: pd.DataFrame) -> float:
        m = np.isfinite(g["decoded_arm"])
        if not m.any():
            return np.nan
        return float((g["decoded_arm"][m] != g["current_arm"][m]).mean())

    ab = grouped["ab"]
    out = {
        "mean_abs_ahead_behind_cm": ab.apply(lambda s: np.nanmean(np.abs(s))).to_numpy(),
        "max_ahead_cm": np.clip(ab.max().to_numpy(), 0.0, None),
        "max_behind_cm": np.clip(-ab.min().to_numpy(), 0.0, None),
        "max_abs_ahead_behind_cm": ab.apply(lambda s: np.nanmax(np.abs(s))).to_numpy(),
        "remote_fraction": grouped.apply(_remote, include_groups=False).to_numpy(),
        "ab_slope_cm_s": grouped.apply(_slope, include_groups=False).to_numpy(),
    }
    # NaN (not False) where ab_slope is undefined (<2 finite points), so an
    # event with unknown direction is not silently counted as inbound.
    slope = out["ab_slope_cm_s"]
    out["is_outbound"] = np.where(np.isfinite(slope), (slope > 0).astype(float), np.nan)
    return np.array(list(grouped.groups.keys())), out


def _hpd_coverage_mean(event_id: np.ndarray, hpd: np.ndarray) -> np.ndarray:
    """Per-event mean of the per-timebin spatial HPD coverage (NaN-aware)."""
    unique, inv, n_events = _group_by_event(event_id)
    hpd = np.asarray(hpd, dtype=float)
    valid = np.isfinite(hpd)
    num = np.bincount(inv[valid], weights=hpd[valid], minlength=n_events)
    den = np.bincount(inv[valid], minlength=n_events).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _is_multimodal(arm_indices: np.ndarray, adjacency: dict[int, set[int]]) -> bool:
    """True if the HPD arm set spans more than one connected component.

    Two occupied arms are in the same component when a path of pairwise
    graph-adjacent *occupied* arms links them. A contiguous chain A-B-C (A and C
    not directly adjacent but joined through the occupied arm B) is therefore one
    component and **unimodal**; only genuinely separated arms are multimodal.

    This replaces the old all-pairs adjacency test, which flagged any HPD set
    with a non-directly-adjacent pair as multimodal and so mislabeled connected
    chains.
    """
    arms = {int(a) for a in arm_indices}
    if len(arms) <= 1:
        return False
    # Breadth-first over occupied, mutually adjacent arms from an arbitrary
    # start; multimodal iff it cannot reach every occupied arm (>1 component).
    start = next(iter(arms))
    seen = {start}
    stack = [start]
    while stack:
        a = stack.pop()
        for b in adjacency.get(a, set()):
            if b in arms and b not in seen:
                seen.add(b)
                stack.append(b)
    return seen != arms


def _count_ab_reversals(ahead_behind: np.ndarray, delta: float) -> int:
    """Count direction reversals in a signed ahead/behind distance series.

    A reversal is a change of travel direction along the track graph, measured
    on the signed ahead/behind distance (``sign x graph-geodesic-distance`` from
    the animal to the decoded position; positive ahead, negative behind). This
    is branch-safe: moving outbound across a track junction does not create a
    spurious reversal (geodesic distance keeps growing), and a trajectory that
    passes *through* the animal (behind -> ahead) is monotone in the signed
    coordinate, not a turn — unlike sign changes of the discontinuous
    ``linear_position``, which fake a reversal at every branch crossing.

    Reversals are counted with a hysteresis dead-band ``delta``: a direction
    change registers only once the series retraces more than ``delta`` from its
    running extreme since the last confirmed turn, so bin-to-bin decoder jitter
    does not inflate the count. The first established direction is not a
    reversal. ``delta = 0`` reduces to a strict sign-change count of the
    increments (flat steps ignored).

    Parameters
    ----------
    ahead_behind : np.ndarray, shape (n_time,)
        Time-ordered signed ahead/behind distance (cm). NaN bins (junctions /
        undefined) are dropped before counting.
    delta : float
        Dead-band amplitude in cm (``>= 0``).

    Returns
    -------
    int
        Number of direction reversals.
    """
    x = np.asarray(ahead_behind, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0
    n_reversals = 0
    direction = 0  # +1 rising, -1 falling, 0 not yet established
    ref = float(x[0])  # running extreme since the last confirmed turn
    for v in x[1:]:
        v = float(v)
        if direction == 1:
            if v > ref:
                ref = v
            elif v < ref - delta:
                n_reversals += 1
                direction = -1
                ref = v
        elif direction == -1:
            if v < ref:
                ref = v
            elif v > ref + delta:
                n_reversals += 1
                direction = 1
                ref = v
        else:  # direction not yet established
            if v > ref + delta:
                direction = 1
                ref = v
            elif v < ref - delta:
                direction = -1
                ref = v
    return n_reversals


def uncertainty_extra(
    *,
    event_id: np.ndarray,
    hpd: np.ndarray,
    arm_posterior: np.ndarray,
    arm_adjacency: dict[int, set[int]],
    coverage: float = 0.95,
) -> RollupResult:
    """Per-event decode-uncertainty extras.

    Columns:

    - ``hpd_coverage_cm`` — per-event mean of the per-timebin spatial HPD
      coverage column (cm) already in ``data_over_time``.
    - ``multimodal_fraction`` — fraction of in-event timebins whose ``coverage``
      HPD arm set spans **more than one connected component** on the track graph
      (threshold-free in the sense that adjacency is binary track topology). A
      contiguous chain of arms is one component (unimodal).

    Parameters
    ----------
    arm_posterior : np.ndarray, shape (T, n_arms)
        Arm-marginal posterior (rows sum to 1); columns ordered by arm id.
    arm_adjacency : dict[int, set[int]]
        Arm -> set of graph-adjacent arms, from ``spatial.arm_adjacency``.
    """
    return _group_by_event(event_id)[0], {
        "hpd_coverage_cm": _hpd_coverage_mean(event_id, hpd),
        "multimodal_fraction": _multimodal_fraction(
            event_id=event_id,
            arm_posterior=arm_posterior,
            arm_adjacency=arm_adjacency,
            coverage=coverage,
        ),
    }


def _multimodal_fraction(
    *,
    event_id: np.ndarray,
    arm_posterior: np.ndarray,
    arm_adjacency: dict[int, set[int]],
    coverage: float = 0.95,
) -> np.ndarray:
    """Per-event fraction of timebins whose HPD arm set spans >1 component.

    Split out from :func:`uncertainty_extra` because it is the *only* rollup
    that needs the arm posterior -- ``hpd_coverage_cm`` reads the per-bin
    ``hpd`` column alone. Keeping them fused forced every caller without a
    posterior into a parallel branch that recomputed ``hpd_coverage_cm``.
    """
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
    )

    _, inv, n_events = _group_by_event(event_id)
    arm_posterior = np.asarray(arm_posterior, dtype=float)
    arm_da = xr.DataArray(
        arm_posterior,
        dims=("time", "position"),
        coords={"position": np.arange(arm_posterior.shape[1])},
    )
    threshold = get_highest_posterior_threshold(arm_da, coverage=coverage)
    in_set = arm_posterior >= threshold[:, None]
    multimodal = np.array(
        [_is_multimodal(np.flatnonzero(row), arm_adjacency) for row in in_set],
        dtype=float,
    )
    counts = np.bincount(inv, minlength=n_events).astype(float)
    mm_total = np.bincount(inv, weights=multimodal, minlength=n_events)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0, mm_total / counts, np.nan)


def trajectory_dynamics(
    *,
    event_id: np.ndarray,
    map_estimate: np.ndarray,
    step_distance: np.ndarray,
    map_speed: np.ndarray,
    decoded_arm: np.ndarray,
    continuous_prob: np.ndarray,
    fragmented_prob: np.ndarray,
    run_speed_baseline: np.ndarray,
    ahead_behind_distance: np.ndarray,
    env,
    place_bin_size: float,
    well_radius_cm: float | None = None,
) -> RollupResult:
    """Trajectory shape/speed/extent rollups.

    Columns: ``start_position``, ``end_position``, ``start_arm``, ``end_arm``,
    ``net_displacement``, ``path_length``, ``tortuosity``,
    ``replay_speed_median``, ``replay_speed_p90``,
    ``replay_speed_continuous_median``, ``step_distance_median``,
    ``step_distance_p90``, ``step_distance_max``, ``spatial_extent``,
    ``n_reversals``, ``pause_fraction``, ``n_patches_visited``,
    ``crosses_trunk``, ``terminates_at_well``, ``compression_factor``.
    """
    nodes_df = env.place_bin_centers_nodes_df_
    node_ids, has_valid = map_estimate_to_map_nodes(map_estimate, nodes_df)
    graph = env.track_graph_with_bin_centers_edges_
    # All-pairs shortest-path distances over the bin-centre graph,
    # precomputed by the environment at fit time. Recomputing them here
    # would be the same Dijkstra over the same graph, per session, with a
    # chance of drifting from the distances every other consumer of
    # ``env`` sees.
    dist = env.distance_between_nodes_
    # Well nodes are the degree-1 terminals of the bin-centre track graph (the
    # six well ends). ``well_radius`` defaults to four place bins.
    well_nodes = [n for n, deg in graph.degree() if deg == 1]
    well_radius = well_radius_cm if well_radius_cm is not None else 4.0 * place_bin_size

    df = pd.DataFrame(
        {
            "event_id": np.asarray(event_id),
            "map_estimate": np.asarray(map_estimate, float),
            "step_distance": np.asarray(step_distance, float),
            "map_speed": np.asarray(map_speed, float),
            "decoded_arm": np.asarray(decoded_arm, float),
            "continuous_prob": np.asarray(continuous_prob, float),
            "fragmented_prob": np.asarray(fragmented_prob, float),
            "run_speed_baseline": np.asarray(run_speed_baseline, float),
            "ahead_behind": np.asarray(ahead_behind_distance, float),
            "node": node_ids,
            "has_valid": has_valid,
        }
    )
    grouped = df.groupby("event_id", sort=True)
    ids = np.array(list(grouped.groups.keys()))

    def _per_event(fn) -> np.ndarray:
        return grouped.apply(fn, include_groups=False).to_numpy()

    def _valid_nodes(g: pd.DataFrame) -> list:
        return [n for n, ok in zip(g["node"], g["has_valid"], strict=True) if ok]

    def _start_pos(g):
        v = g["map_estimate"][np.isfinite(g["map_estimate"])]
        return float(v.iloc[0]) if len(v) else np.nan

    def _end_pos(g):
        v = g["map_estimate"][np.isfinite(g["map_estimate"])]
        return float(v.iloc[-1]) if len(v) else np.nan

    def _start_arm(g):
        v = g["decoded_arm"][np.isfinite(g["decoded_arm"])]
        return float(v.iloc[0]) if len(v) else np.nan

    def _end_arm(g):
        v = g["decoded_arm"][np.isfinite(g["decoded_arm"])]
        return float(v.iloc[-1]) if len(v) else np.nan

    def _net_disp(g):
        nodes = _valid_nodes(g)
        if len(nodes) < 2:
            return 0.0 if nodes else np.nan
        return float(dist[nodes[0]][nodes[-1]])

    def _spatial_extent(g):
        nodes = set(_valid_nodes(g))
        if len(nodes) < 2:
            return 0.0 if nodes else np.nan
        return float(max(dist[a][b] for a in nodes for b in nodes))

    # Direction reversals are counted on the signed ahead/behind distance
    # (graph-geodesic, branch-safe) rather than the discontinuous
    # ``linear_position`` (whose branch crossings faked reversals). The
    # hysteresis dead-band is one place bin — the same spatial scale
    # ``pause_fraction`` uses — so decoder jitter within a bin is not a turn.
    def _n_reversals(g):
        return _count_ab_reversals(g["ahead_behind"].to_numpy(), place_bin_size)

    def _n_patches(g):
        arms = g["decoded_arm"][np.isfinite(g["decoded_arm"])]
        patches = {TRACK_SEGMENT_TO_PATCH[int(a)] for a in arms}
        return len(patches)

    def _crosses_trunk(g):
        arms = {int(a) for a in g["decoded_arm"][np.isfinite(g["decoded_arm"])]}
        return bool(arms & _TRUNK_ARMS) and _n_patches(g) > 1

    def _cont_speed_median(g):
        mask = g["continuous_prob"] > g["fragmented_prob"]
        v = g["map_speed"][mask]
        return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan

    def _pause_fraction(g):
        # NaN-aware: fraction of *valid* steps within one place bin (junction
        # steps are NaN). An all-NaN event returns NaN, like the speed stats.
        sd = g["step_distance"].to_numpy()
        finite = np.isfinite(sd)
        return float((sd[finite] <= place_bin_size).mean()) if finite.any() else np.nan

    # "Terminates at a well" requires the final decoded position to be within
    # ``well_radius`` graph-cm of a degree-1 well node — not merely that the last
    # decoded arm is a leaf arm (which also matches mid-arm / patch-junction-end
    # endpoints).
    def _terminates_at_well(g):
        a = _end_arm(g)
        if not (np.isfinite(a) and 0 <= a < 6):
            return False
        nodes = _valid_nodes(g)
        if not nodes:
            return False
        end_dist = dist.get(nodes[-1], {})
        nearest_well = min(
            (end_dist[w] for w in well_nodes if w in end_dist), default=np.inf
        )
        return bool(nearest_well <= well_radius)

    # CAVEAT: ``step_distance`` is the session-level per-bin MAP displacement
    # (distance into bin t from bin t-1). The first in-event bin's step
    # therefore crosses the event boundary (its t-1 lies just *before* the
    # event), so ``path_length`` and ``step_distance_max``/``_p90`` include one
    # boundary step per event. Events are mostly contiguous so this is one step
    # out of ~10-25; ``path_length`` is defined as Σ step_distance.
    path_length = _per_event(lambda g: float(np.nansum(g["step_distance"])))
    net_disp = _per_event(_net_disp)
    with np.errstate(divide="ignore", invalid="ignore"):
        tortuosity = np.where(net_disp > 0, path_length / net_disp, np.nan)

    out = {
        "replay_start_position_cm": _per_event(_start_pos),
        "replay_end_position_cm": _per_event(_end_pos),
        "start_arm": _per_event(_start_arm),
        "end_arm": _per_event(_end_arm),
        "net_displacement_cm": net_disp,
        "path_length_cm": path_length,
        "tortuosity": tortuosity,
        "replay_speed_median_cm_s": _per_event(_nan_safe("map_speed", np.nanmedian)),
        "replay_speed_p90_cm_s": _per_event(_nan_safe("map_speed", _p90)),
        "replay_speed_continuous_median_cm_s": _per_event(_cont_speed_median),
        "step_distance_median_cm": _per_event(_nan_safe("step_distance", np.nanmedian)),
        "step_distance_p90_cm": _per_event(_nan_safe("step_distance", _p90)),
        "step_distance_max_cm": _per_event(_nan_safe("step_distance", np.nanmax)),
        "spatial_extent_cm": _per_event(_spatial_extent),
        "n_reversals": _per_event(_n_reversals),
        "pause_fraction": _per_event(_pause_fraction),
        "n_patches_visited": _per_event(_n_patches),
        "crosses_trunk": _per_event(_crosses_trunk),
        "terminates_at_well": _per_event(_terminates_at_well),
    }
    baseline = _per_event(lambda g: float(g["run_speed_baseline"].iloc[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["compression_factor"] = np.where(
            baseline > 0, out["replay_speed_median_cm_s"] / baseline, np.nan
        )
    return ids, out


def compute_trajectory_rollups(
    *,
    nl: pd.DataFrame,
    event_id: np.ndarray,
    env,
    track_graph: nx.Graph,
    state_prob_cols: dict[str, str],
    continuous_col: str,
    fragmented_col: str,
    place_bin_size: float,
    nonlocal_prob: np.ndarray | None = None,
    arm_posterior: np.ndarray | None = None,
    non_local_threshold: float = 0.8,
) -> dict[str, np.ndarray]:
    """Orchestrate all trajectory rollups into one ``{column: per_event_array}``.

    Shared by both event-table builders. The truly-shared families
    (``trajectory_dynamics``, ``ahead_behind_summary``, ``uncertainty_extra``,
    ``path_continuity``, neural-signal means, continuous/fragmented state means)
    produce :data:`SHARED_TRAJECTORY_COLUMNS` identically across scenarios; the
    extra state means / fractions in :data:`NONLOCAL_ONLY_COLUMNS` are emitted
    only when ``state_prob_cols`` includes the 4-state model's columns. All
    arrays are keyed to ``np.unique(event_id)``.

    Parameters
    ----------
    nl : pd.DataFrame
        In-event timebins, time-indexed, carrying the augmented
        ``data_over_time`` columns plus ``decoded_arm`` / ``track_segment_id``.
    state_prob_cols : dict[str, str]
        Output column name -> ``nl`` state-probability column.
    continuous_col, fragmented_col : str
        ``nl`` columns for the continuous / fragmented state probability.
    nonlocal_prob : np.ndarray | None
        Per-bin ``w_t`` for ``nonlocal_fraction`` (Non-Local only; ``None``
        for ContFrag).
    arm_posterior : np.ndarray | None
        Arm-marginal posterior for ``multimodal_fraction``; when ``None`` the
        HPD-set-based columns are ``NaN``.
    """
    # Fail loud with the full missing set if the data_over_time schema and the
    # rollups have drifted apart (e.g. after a re-decode that changes builder
    # columns) — clearer than a KeyError deep inside one family.
    required = {
        *_REQUIRED_DOT_COLUMNS,
        *_SIGNAL_COLUMNS.values(),
        *state_prob_cols.values(),
        continuous_col,
        fragmented_col,
    }
    missing = sorted(required - set(nl.columns))
    if missing:
        raise KeyError(
            "compute_trajectory_rollups: in-event frame is missing required "
            f"data_over_time columns {missing}"
        )

    def col(name: str) -> np.ndarray:
        return nl[name].to_numpy()

    out: dict[str, np.ndarray] = {}

    _, dyn = trajectory_dynamics(
        event_id=event_id,
        map_estimate=col("map_estimate"),
        step_distance=col("step_distance"),
        map_speed=col("map_speed"),
        decoded_arm=col("decoded_arm"),
        continuous_prob=col(continuous_col),
        fragmented_prob=col(fragmented_col),
        run_speed_baseline=col("run_speed_baseline"),
        ahead_behind_distance=col("ahead_behind_distance"),
        env=env,
        place_bin_size=place_bin_size,
    )
    out.update(dyn)

    _, ab = ahead_behind_summary(
        event_id=event_id,
        ahead_behind_distance=col("ahead_behind_distance"),
        decoded_arm=col("decoded_arm"),
        current_arm=col("track_segment_id"),
        time=nl.index.to_numpy(),
    )
    out.update(ab)

    _, bs = brain_state_means(
        event_id=event_id,
        state_probs={o: col(c) for o, c in state_prob_cols.items()},
        extra_means={o: col(c) for o, c in _SIGNAL_COLUMNS.items()},
        nonlocal_prob=nonlocal_prob,
        non_local_threshold=non_local_threshold,
    )
    out.update(bs)

    # nospike_fraction: fraction of bins where No-Spike is the modal state
    # (4-state Non-Local model only).
    if "nospike_prob_mean" in state_prob_cols:
        names = list(state_prob_cols.values())
        stacked = np.column_stack([col(c) for c in names])
        modal = np.asarray(names)[np.argmax(stacked, axis=1)]
        is_nospike = (modal == state_prob_cols["nospike_prob_mean"]).astype(float)
        unique, inv, n_events = _group_by_event(event_id)
        counts = np.bincount(inv, minlength=n_events).astype(float)
        out["nospike_fraction"] = (
            np.bincount(inv, weights=is_nospike, minlength=n_events) / counts
        )

    # hpd_coverage_cm reads the per-bin ``hpd`` column and nothing else, so it is
    # computed unconditionally. multimodal_fraction is the only rollup that needs
    # the arm posterior, and is NaN when the caller has none (the builders accept
    # conditional_posterior=None and promise a stable schema).
    out["hpd_coverage_cm"] = _hpd_coverage_mean(event_id, col("hpd"))
    out["multimodal_fraction"] = (
        _multimodal_fraction(
            event_id=event_id,
            arm_posterior=arm_posterior,
            arm_adjacency=arm_adjacency(track_graph),
        )
        if arm_posterior is not None
        else np.full(len(np.unique(event_id)), np.nan)
    )

    adj = arm_adjacency(track_graph)
    pc = (
        pd.DataFrame({"event_id": np.asarray(event_id), "decoded_arm": col("decoded_arm")})
        .groupby("event_id", sort=True)["decoded_arm"]
        .apply(lambda s: path_continuity(s.to_numpy(), adj))
        .to_numpy()
    )
    out["path_continuity_fraction"] = pc

    return out
