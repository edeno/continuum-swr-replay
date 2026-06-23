"""Per-event replay-trajectory rollups, shared by both event-table builders.

Each family is a pure function over per-timebin arrays already filtered to
in-event rows and grouped by ``event_id``. Every family returns
``(unique_sorted_event_ids, {column_name: per_event_array})`` so callers assign
columns the same way the posterior rollups in :mod:`uncertainty` do. The
``unique_sorted_event_ids`` equals ``np.unique(event_id)``, matching the
``groupby(sort=True)`` order the table builders assert against.

Definitions follow the design spec
``docs/superpowers/specs/2026-06-23-replay-trajectory-metrics-design.md``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

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
    "start_position", "end_position", "start_arm", "end_arm", "net_displacement",
    "path_length", "tortuosity", "replay_speed_median", "replay_speed_p90",
    "replay_speed_continuous_median", "step_distance_median", "step_distance_p90",
    "step_distance_max", "spatial_extent", "n_reversals", "pause_fraction",
    "n_patches_visited", "crosses_trunk", "terminates_at_well", "compression_factor",
    # ahead_behind_summary
    "mean_abs_ahead_behind", "max_ahead", "max_behind", "max_abs_ahead_behind",
    "remote_fraction", "ab_slope", "is_outbound",
    # uncertainty_extra
    "hpd_coverage_cm", "multimodal_fraction",
    # path_continuity
    "path_continuity",
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


def _graph_distance_lookup(track_graph: nx.Graph) -> dict:
    """All-pairs shortest-path distance dict over the bin-centre track graph."""
    return dict(nx.shortest_path_length(track_graph, weight="distance"))


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
        "mean_abs_ahead_behind": ab.apply(lambda s: np.nanmean(np.abs(s))).to_numpy(),
        "max_ahead": np.clip(ab.max().to_numpy(), 0.0, None),
        "max_behind": np.clip(-ab.min().to_numpy(), 0.0, None),
        "max_abs_ahead_behind": ab.apply(lambda s: np.nanmax(np.abs(s))).to_numpy(),
        "remote_fraction": grouped.apply(_remote, include_groups=False).to_numpy(),
        "ab_slope": grouped.apply(_slope, include_groups=False).to_numpy(),
    }
    out["is_outbound"] = out["ab_slope"] > 0
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
    """True if the HPD arm set contains two non-graph-adjacent arms."""
    arms = [int(a) for a in arm_indices]
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            if b not in adjacency.get(a, set()):
                return True
    return False


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
      HPD arm set spans **non-graph-adjacent** arms (threshold-free in the
      sense that adjacency is binary track topology).

    Parameters
    ----------
    arm_posterior : np.ndarray, shape (T, n_arms)
        Arm-marginal posterior (rows sum to 1); columns ordered by arm id.
    arm_adjacency : dict[int, set[int]]
        Arm -> set of graph-adjacent arms, from ``spatial.arm_adjacency``.
    """
    import xarray as xr
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
    )

    unique, inv, n_events = _group_by_event(event_id)
    hpd_coverage_cm = _hpd_coverage_mean(event_id, hpd)

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
        multimodal_fraction = np.where(counts > 0, mm_total / counts, np.nan)

    return unique, {
        "hpd_coverage_cm": hpd_coverage_cm,
        "multimodal_fraction": multimodal_fraction,
    }


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
    env,
    place_bin_size: float,
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
    dist = _graph_distance_lookup(env.track_graph_with_bin_centers_edges_)
    node_lin = dict(
        zip(nodes_df["node_id"].to_numpy(), nodes_df["linear_position"].to_numpy(), strict=True)
    )

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

    def _n_reversals(g):
        nodes = _valid_nodes(g)
        if len(nodes) < 3:
            return 0
        lin = np.array([node_lin[n] for n in nodes])
        signs = np.sign(np.diff(lin))
        signs = signs[signs != 0]
        return int(np.sum(np.diff(signs) != 0)) if len(signs) >= 2 else 0

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

    path_length = _per_event(lambda g: float(np.nansum(g["step_distance"])))
    net_disp = _per_event(_net_disp)
    with np.errstate(divide="ignore", invalid="ignore"):
        tortuosity = np.where(net_disp > 0, path_length / net_disp, np.nan)

    out = {
        "start_position": _per_event(_start_pos),
        "end_position": _per_event(_end_pos),
        "start_arm": _per_event(_start_arm),
        "end_arm": _per_event(_end_arm),
        "net_displacement": net_disp,
        "path_length": path_length,
        "tortuosity": tortuosity,
        "replay_speed_median": _per_event(lambda g: float(np.nanmedian(g["map_speed"]))),
        "replay_speed_p90": _per_event(
            lambda g: float(np.nanpercentile(g["map_speed"], 90))
            if np.isfinite(g["map_speed"]).any() else np.nan
        ),
        "replay_speed_continuous_median": _per_event(_cont_speed_median),
        "step_distance_median": _per_event(lambda g: float(np.nanmedian(g["step_distance"]))),
        "step_distance_p90": _per_event(
            lambda g: float(np.nanpercentile(g["step_distance"], 90))
        ),
        "step_distance_max": _per_event(lambda g: float(np.nanmax(g["step_distance"]))),
        "spatial_extent": _per_event(_spatial_extent),
        "n_reversals": _per_event(_n_reversals),
        "pause_fraction": _per_event(
            lambda g: float((g["step_distance"] <= place_bin_size).mean())
        ),
        "n_patches_visited": _per_event(_n_patches),
        "crosses_trunk": _per_event(_crosses_trunk),
        "terminates_at_well": _per_event(
            lambda g: bool(_end_arm(g) in range(6)) if np.isfinite(_end_arm(g)) else False
        ),
    }
    baseline = _per_event(lambda g: float(g["run_speed_baseline"].iloc[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["compression_factor"] = np.where(
            baseline > 0, out["replay_speed_median"] / baseline, np.nan
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

    if arm_posterior is not None:
        _, unc = uncertainty_extra(
            event_id=event_id,
            hpd=col("hpd"),
            arm_posterior=arm_posterior,
            arm_adjacency=arm_adjacency(track_graph),
        )
        out.update(unc)
    else:
        # hpd_coverage_cm needs only the per-bin hpd column; multimodal_fraction
        # needs the arm posterior, so it is NaN when the posterior is absent.
        out["hpd_coverage_cm"] = _hpd_coverage_mean(event_id, col("hpd"))
        out["multimodal_fraction"] = np.full(len(np.unique(event_id)), np.nan)

    adj = arm_adjacency(track_graph)
    pc = (
        pd.DataFrame({"event_id": np.asarray(event_id), "decoded_arm": col("decoded_arm")})
        .groupby("event_id", sort=True)["decoded_arm"]
        .apply(lambda s: path_continuity(s.to_numpy(), adj))
        .to_numpy()
    )
    out["path_continuity"] = pc

    return out
