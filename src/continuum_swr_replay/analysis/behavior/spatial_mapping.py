"""Map decoded replay positions to spatial units (arms, patches).

Functions for converting MAP-estimated linear positions to track segment IDs
(arms, 0-8) and patch IDs (1-3), and for finding arms adjacent to wells.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)


def _nearest_bin_index(bin_positions: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the bin nearest each position, the first one on a tie.

    The same index as ``np.abs(bin_positions[None, :] - x[:, None]).argmin(1)``
    without its ``(n_x, n_bins)`` temporary: on strictly increasing bins the
    nearest one is a ``searchsorted`` neighbor of ``x``. Other bins take the
    dense path.

    Parameters
    ----------
    bin_positions : np.ndarray, shape (n_bins,)
        Bin centers, at least one.
    x : np.ndarray, shape (n_x,)
        Positions, none NaN.

    Returns
    -------
    np.ndarray of int, shape (n_x,)
    """
    n_bins = len(bin_positions)
    if n_bins < 2 or not np.all(np.diff(bin_positions) > 0):
        return np.abs(bin_positions[None, :] - x[:, None]).argmin(axis=1)
    right = np.searchsorted(bin_positions, x, side="left")
    left = right - 1
    left_clipped = np.maximum(left, 0)
    right_clipped = np.minimum(right, n_bins - 1)
    d_left = np.abs(bin_positions[left_clipped] - x)
    d_right = np.abs(bin_positions[right_clipped] - x)
    # equidistant -> the left bin, as argmin takes the first minimum
    take_left = (right == n_bins) | ((left >= 0) & (d_left <= d_right))
    idx = np.where(take_left, left_clipped, right_clipped)
    # The rounded distances fall monotonically toward x on each side, so the
    # only other bin that can tie the chosen one is the one just left of it
    # (far outside the bins, or at infinity); argmin would take that one.
    previous = np.maximum(idx - 1, 0)
    tied = (idx > 0) & (
        np.abs(bin_positions[previous] - x) == np.where(take_left, d_left, d_right)
    )
    if tied.any():
        idx[tied] = np.abs(bin_positions[None, :] - x[tied, None]).argmin(axis=1)
    return idx


def map_estimate_to_patch(map_estimate: np.ndarray, env) -> np.ndarray:
    """Map decoded linear position to patch ID (1, 2, or 3).

    The arm of each MAP estimate (:func:`map_estimate_to_arm`, the nearest
    real track place bin) mapped to its patch via TRACK_SEGMENT_TO_PATCH.

    Parameters
    ----------
    map_estimate : array-like
        MAP position estimates in linear cm.
    env : Environment
        Fitted environment object with place_bin_centers_nodes_df_.

    Returns
    -------
    np.ndarray
        Patch IDs (1, 2, or 3). NaN where map_estimate is NaN.
    """
    arms = map_estimate_to_arm(map_estimate, env)
    result = np.full(len(arms), np.nan)
    valid = ~np.isnan(arms)
    result[valid] = [
        TRACK_SEGMENT_TO_PATCH.get(int(arm), np.nan) for arm in arms[valid]
    ]
    return result


def map_estimate_to_arm(map_estimate: np.ndarray, env) -> np.ndarray:
    """Map decoded linear position to arm/track segment ID (0-8).

    Junction bins (edge_id=-1) are excluded from the nearest-neighbor
    lookup because they are interpolation placeholders at edge gaps in
    the linearization, not real spatial positions. Every valid MAP
    estimate therefore maps to a real arm.

    Parameters
    ----------
    map_estimate : array-like
        MAP position estimates in linear cm.
    env : Environment
        Fitted environment object with place_bin_centers_nodes_df_.

    Returns
    -------
    np.ndarray
        Arm IDs (0-8). NaN only where map_estimate is NaN.
    """
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    lp = np.asarray(bin_df["linear_position"])
    ei = np.asarray(bin_df["edge_id"])

    # Exclude junction bins (edge_id == -1) from nearest-neighbor lookup
    real_mask = ei >= 0
    lp_real = lp[real_mask]
    ei_real = ei[real_mask]

    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)
    if valid.any() and len(lp_real) > 0:
        idx = _nearest_bin_index(lp_real, map_est[valid])
        result[valid] = ei_real[idx].astype(float)
    return result


def arm_to_well(arm_id: float, track_graph, well_ids: set[int] | None = None) -> float:
    """Return the well node at the terminal end of a leaf arm.

    For leaf arms (well ↔ patch junction), returns the well node ID.
    For trunk arms (patch junction ↔ center junction) or any arm that
    does not touch a well, returns NaN.

    Parameters
    ----------
    arm_id : float
        Arm/edge ID. NaN returns NaN.
    track_graph : networkx.Graph
        Track graph with edge_id attributes.
    well_ids : set[int] | None
        Set of node IDs that are wells. Defaults to {0, 1, 2, 3, 4, 5}.

    Returns
    -------
    float
        Well node ID (0-5), or NaN if the arm is not a leaf arm.
    """
    if np.isnan(arm_id):
        return np.nan
    if well_ids is None:
        well_ids = {0, 1, 2, 3, 4, 5}
    arm_id_int = int(arm_id)
    for u, v, data in track_graph.edges(data=True):
        if data.get("edge_id") != arm_id_int:
            continue
        if u in well_ids:
            return float(u)
        if v in well_ids:
            return float(v)
        return np.nan
    return np.nan


def well_to_arm_set(well_id: float, track_graph) -> set[int]:
    """Get the set of arm (edge) IDs adjacent to a well node.

    On the real bandit track graph each well is a terminal (degree-1)
    node connecting to exactly one "leaf" arm, which in turn reaches a
    patch junction and then the center junction via a "trunk" arm.
    So ``well_to_arm_set`` returns a set of size 1 per valid well,
    and the predictors derived from wells (``is_prev``, ``is_next``,
    ``is_last_rewarded``) each mark exactly one of the nine arms.

    This function is written generically for any track graph, so it
    will return larger sets for hypothetical graphs where wells have
    higher degree (e.g., minimal test fixtures with direct well-to-well
    edges).

    Parameters
    ----------
    well_id : float
        Well node ID (0-5). NaN returns empty set.
    track_graph : networkx.Graph
        Track graph with edge_id attributes.

    Returns
    -------
    set[int]
        Edge IDs (arm IDs) adjacent to the well (size 1 on the real graph).
    """
    if np.isnan(well_id):
        return set()
    well_id = int(well_id)
    arms = set()
    for u, v, data in track_graph.edges(data=True):
        if u == well_id or v == well_id:
            eid = data.get("edge_id")
            if eid is not None:
                arms.add(eid)
    if len(arms) == 0:
        logger.warning("No arms found for well %d", well_id)
    return arms


# ── Well, arm and patch lookups on the track graph ──


def single_arm_from_well(well: float, track_graph: nx.Graph) -> float:
    """Deterministic scalar arm for a well, ``NaN`` if the well is NaN or isolated."""
    if well is None or (isinstance(well, float) and np.isnan(well)):
        return np.nan
    arms = well_to_arm_set(well, track_graph)
    if len(arms) == 0:
        return np.nan
    return float(min(arms))


def arms_of_wells(wells: np.ndarray, track_graph: nx.Graph) -> np.ndarray:
    """:func:`single_arm_from_well` of each well, ``NaN`` for a ``NaN`` well.

    Parameters
    ----------
    wells : np.ndarray, shape (n,)
        Well node IDs as floats.
    track_graph : nx.Graph
        Session track graph.

    Returns
    -------
    np.ndarray, shape (n,)
        Arm IDs as floats.
    """
    arm_of = {
        well: single_arm_from_well(well, track_graph)
        for well in np.unique(wells[~np.isnan(wells)])
    }
    return pd.Series(wells, dtype=float).map(arm_of).to_numpy(dtype=float)


def arm_to_patch(arm: float) -> float:
    """Patch ID for an arm, ``NaN`` propagating."""
    if arm is None or (isinstance(arm, float) and np.isnan(arm)):
        return np.nan
    return float(TRACK_SEGMENT_TO_PATCH.get(int(arm), np.nan))


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


# ── Stay / switch ──


def stay_indicator_from_patches(
    next_patch: float | np.ndarray, current_patch: float | np.ndarray
) -> float | np.ndarray:
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
