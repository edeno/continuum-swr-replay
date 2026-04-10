"""Map decoded replay positions to spatial units (arms, patches).

Functions for converting MAP-estimated linear positions to track segment IDs
(arms, 0-8) and patch IDs (1-3), and for finding arms adjacent to wells.
"""

from __future__ import annotations

import logging

import numpy as np

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)


def map_estimate_to_patch(map_estimate: np.ndarray, env) -> np.ndarray:
    """Map decoded linear position to patch ID (1, 2, or 3).

    Finds the nearest place bin for each MAP estimate, looks up its edge_id,
    and maps to a patch via TRACK_SEGMENT_TO_PATCH. Junction bins (edge_id=-1)
    map to NaN.

    Parameters
    ----------
    map_estimate : array-like
        MAP position estimates in linear cm.
    env : Environment
        Fitted environment object with place_bin_centers_nodes_df_.

    Returns
    -------
    np.ndarray
        Patch IDs (1, 2, or 3). NaN where map_estimate is NaN or at junctions.
    """
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    lp = np.asarray(bin_df["linear_position"])
    ei = np.asarray(bin_df["edge_id"])
    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)
    if valid.any():
        idx = np.abs(lp[None, :] - map_est[valid, None]).argmin(axis=1)
        result[valid] = [TRACK_SEGMENT_TO_PATCH.get(int(ei[j]), np.nan) for j in idx]
    return result


def map_estimate_to_arm(map_estimate: np.ndarray, env) -> np.ndarray:
    """Map decoded linear position to arm/track segment ID (0-8).

    Parameters
    ----------
    map_estimate : array-like
        MAP position estimates in linear cm.
    env : Environment
        Fitted environment object with place_bin_centers_nodes_df_.

    Returns
    -------
    np.ndarray
        Arm IDs (0-8). NaN where map_estimate is NaN or at junctions (edge=-1).
    """
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    lp = np.asarray(bin_df["linear_position"])
    ei = np.asarray(bin_df["edge_id"])
    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)
    if valid.any():
        idx = np.abs(lp[None, :] - map_est[valid, None]).argmin(axis=1)
        edges = ei[idx]
        result[valid] = [float(e) if e >= 0 else np.nan for e in edges]
    return result


def well_to_arm_set(well_id: float, track_graph) -> set[int]:
    """Get the set of arm (edge) IDs adjacent to a well node.

    Each well connects to 2 arms. This means predictors derived from wells
    (is_prev, is_next, is_last_rewarded) mark 2 of 9 arms as 1, while
    is_current (from track_segment_id) marks only 1 arm.

    Parameters
    ----------
    well_id : float
        Well node ID (0-5). NaN returns empty set.
    track_graph : networkx.Graph
        Track graph with edge_id attributes.

    Returns
    -------
    set[int]
        Edge IDs (arm IDs) adjacent to the well.
    """
    if np.isnan(well_id):
        return set()
    well_id = int(well_id)
    arms = set()
    for u, v, data in track_graph.edges(data=True):
        if u == well_id or v == well_id:
            arms.add(data["edge_id"])
    if len(arms) == 0:
        logger.warning("No arms found for well %d", well_id)
    return arms
