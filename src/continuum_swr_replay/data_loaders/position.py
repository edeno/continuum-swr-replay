"""Position data loading and linearization.

This module handles loading position data from Spyglass database and performing
spatial linearization along track graphs. Position data includes raw 2D coordinates,
projected positions on the track, and 1D linear position along graph edges.

Functions
---------
get_interpolated_position_info
    Interpolate position data to new time points and compute linear position.
get_position_info
    Load and process position data for a given session and epoch.
get_epoch_position_info
    An epoch's position as ``load_data`` uses it, shared with the SWR runner.

Notes
-----
Position tracking uses DLC (DeepLabCut) when available, falling back to older
tracking methods. All position data is linearized to 1D along the track graph
for decoding analysis.

See Also
--------
load_data : Main data loading orchestrator that uses these functions
trials : Trial construction that depends on position data

"""

from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from track_linearization import get_linearized_position

from ..data_types import PositionInfoDict
from .constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: NDArray[np.float64],
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
    position_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate position info to new time points and add linearization.

    Combines the original position data with new time points via linear
    interpolation, then computes linearized position along the track graph.
    This ensures position data is available at all desired time points
    (e.g., spike times, LFP timestamps).

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with time index (seconds) and columns for x, y coordinates.
    time : np.ndarray, shape (n_times,)
        New time points (seconds) for interpolation. These will be the final
        time points in the output.
    track_graph : nx.Graph
        NetworkX graph representing the track structure with nodes as wells
        and edges as track segments.
    edge_order : list of tuple of (int, int)
        Ordered list of graph edges (node pairs) defining the linearization path.
    edge_spacing : list of float
        Spacing values along each edge for linearization.
    position_columns : list of str, optional
        Names of columns containing x and y position coordinates.
        Default is ["head_position_x", "head_position_y"].

    Returns
    -------
    pd.DataFrame
        Interpolated position DataFrame with additional columns:
        - Original position columns (interpolated)
        - linear_position : 1D position along track (cm)
        - projected_x_position, projected_y_position : Position projected onto track
        - track_segment_id : ID of current track segment

    Notes
    -----
    The function merges original and new time points, interpolates linearly,
    then extracts only the requested time points. This avoids extrapolation
    artifacts at boundaries.

    See Also
    --------
    get_position_info : Main position loading function that calls this
    track_linearization.get_linearized_position : Underlying linearization function

    """
    if position_columns is None:
        position_columns = ["head_position_x", "head_position_y"]

    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index, time))),
        name="time",
    )

    interpolated_position_info = (
        position_info.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=time)
    )

    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat(
        (
            interpolated_position_info,
            linear_position_info,
        ),
        axis=1,
    )


def _track_graph_name(nwb_file_name: str, linearized_graph_names: list[str]) -> str:
    """The track graph an interval's position is linearized on.

    The linearization key leaves the track graph free, so an interval can be
    linearized on several graphs: senor20201030_ 02_r1 has the session-named
    graph and an unrelated "6 arm" graph.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name; its part before the first ``_`` names the session's graph.
    linearized_graph_names : list of str
        ``track_graph_name`` of every ``IntervalLinearizedPosition`` row of the
        interval.

    Returns
    -------
    str
        The one linearized graph; otherwise the session-named graph, when there
        is no linearization or it is among several.

    Raises
    ------
    ValueError
        Several linearized graphs, none of them the session-named graph.
    """
    session_graph_name = nwb_file_name.split("_")[0]
    if len(linearized_graph_names) == 1:
        return linearized_graph_names[0]
    if not linearized_graph_names or session_graph_name in linearized_graph_names:
        return session_graph_name
    raise ValueError(
        f"{nwb_file_name} is linearized on several track graphs "
        f"{sorted(linearized_graph_names)}, none of them {session_graph_name!r}"
    )


def get_position_info(
    nwb_file_name: str, epoch_name: str, pos_name: str
) -> PositionInfoDict:
    """Load position data for a given session and epoch.

    Loads raw position tracking data from Spyglass database, performs quality
    control (drops NaN values), restricts to valid trial times, and computes
    linearized position along the track graph. Attempts to use DLC (DeepLabCut)
    tracking when available, falling back to older tracking methods.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Epoch identifier in format "NN_rM" (e.g., "02_r1" for epoch 2, replicate 1).
    pos_name : str
        Position interval name from Spyglass PositionIntervalMap table.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'position_info' : pd.DataFrame
            Position data with time index and columns including:
            head_position_x, head_position_y (raw coordinates),
            linear_position (1D position along track),
            projected_x_position, projected_y_position (projected onto track),
            track_segment_id (current track segment),
            head_speed, head_orientation (if available),
            patch_id (current patch ID, 1-3).
        - 'linear_edge_order' : list of tuple of (int, int)
            Ordered edges for linearization.
        - 'linear_edge_spacing' : list of float
            Spacing along each edge.
        - 'track_graph' : nx.Graph
            NetworkX graph of the track with node and edge attributes.

    Notes
    -----
    The function first attempts to load DLC-tracked position from PositionOutput
    table. If unavailable, falls back to IntervalPositionInfo. Position data is
    then restricted to " noPrePostTrialTimes" intervals if available (excludes
    pre-trial and post-trial periods).

    Track graph is the one the interval was linearized on. With no
    linearization, or with several of which one is the session-named graph
    (first part of the NWB filename), the session-named graph is used; several
    without it raise ``ValueError``, as do several DLC position entries for the
    epoch.

    See Also
    --------
    get_interpolated_position_info : Interpolates position to new time points
    load_data : Main orchestrator that calls this function

    Examples
    --------
    >>> pos_data = get_position_info("chimi20200212_.nwb", "02_r1", "02_r1")
    >>> pos_data['position_info'].columns
    Index(['head_position_x', 'head_position_y', 'head_speed', ...])
    >>> pos_data['track_graph'].number_of_nodes()
    6  # Six wells in the environment

    """
    # Lazy imports: Spyglass / DataJoint require a database connection
    # at import time. Keeping these inside the function lets pure
    # helpers (e.g. ``get_interpolated_position_info``) import cleanly
    # offline.
    from spyglass.common import IntervalList
    from spyglass.common.common_position import IntervalPositionInfo
    from spyglass.linearization.v0.main import (
        IntervalLinearizedPosition,
        TrackGraph,
    )
    from spyglass.position import PositionOutput

    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": "default_decoding",
    }

    linearization_key = {
        "position_info_param_name": "default_decoding",
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "linearization_param_name": "default",
    }

    track_graph_name = _track_graph_name(
        nwb_file_name,
        list(
            (IntervalLinearizedPosition() & linearization_key).fetch("track_graph_name")
        ),
    )

    track_graph = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).get_networkx_track_graph()
    track_graph_params = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).fetch1()
    linear_edge_order = track_graph_params["linear_edge_order"]
    linear_edge_spacing = track_graph_params["linear_edge_spacing"]

    epoch = int(epoch_name.split("_")[0])
    dlc_merge_ids = (
        PositionOutput().merge_restrict()
        & {
            "nwb_file_name": nwb_file_name,
            "source": "DLCPosV1",
            "epoch": epoch,
        }
    ).fetch("merge_id")
    if len(dlc_merge_ids) > 1:
        raise ValueError(
            f"{len(dlc_merge_ids)} DLCPosV1 position entries for {nwb_file_name} "
            f"epoch {epoch}; expected at most one"
        )
    if len(dlc_merge_ids) == 1:
        position_info = (
            (
                (PositionOutput() & {"merge_id": str(dlc_merge_ids[0])})
                .fetch1_dataframe()
                .dropna()
            )
            .drop(columns="video_frame_ind")
            .add_prefix("head_")
        )
        time = (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna().index
    else:
        position_info = (
            (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna()
        )
        time = position_info.index

    # nwb_file_name and interval_list_name are IntervalList's whole primary key,
    # so this is zero rows or one.
    valid_interval_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": epoch_name + " noPrePostTrialTimes",
    }
    if len(IntervalList & valid_interval_key) == 0:
        logger.debug(
            "No noPrePostTrialTimes interval for %s %s", nwb_file_name, epoch_name
        )
    else:
        valid_interval_times = (IntervalList & valid_interval_key).fetch1("valid_times")
        # Slice both axes to the valid interval — position_info is the
        # raw DLC frame, ``time`` is the interpolation grid. They come
        # from independent Spyglass tables, so each must be trimmed
        # explicitly. Without this, ``get_interpolated_position_info``
        # extrapolates across the pre/post-trial gap, leaving NaN rows
        # that downstream callers then have to ``dropna`` away.
        t_start, t_end = valid_interval_times[0][0], valid_interval_times[-1][1]
        position_info = position_info.loc[t_start:t_end]
        time = time[(time >= t_start) & (time <= t_end)]

    position_info = get_interpolated_position_info(
        position_info,
        time,
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )
    position_info["patch_id"] = position_info["track_segment_id"].map(
        TRACK_SEGMENT_TO_PATCH
    )

    return {
        "position_info": position_info,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
        "track_graph": track_graph,
    }


def get_merge_position_info(pos_merge_id: str) -> pd.DataFrame:
    """A ``PositionOutput`` entry's position, read from the source it names.

    ``(PositionOutput & {"merge_id": ...}).fetch1_dataframe()`` asks the merge
    table which part holds the merge id and refuses when two do ("Found 2
    potential parents"). j1620210710's ``02_r1`` and ``04_r2`` entries are in
    both the CommonPos and the TrodesPosV1 part: their keys hold the same
    values under different names, which give the same merge id. The master row
    names one source, CommonPos. ``get_position_info`` (``load_data``) reads a
    CommonPos epoch's position from ``IntervalPositionInfo`` itself, and so does
    this: the part of the master's source gives its parent's key, and the
    parent's ``fetch1_dataframe`` the position. For an entry in one part it is
    the frame the merge table's own ``fetch1_dataframe`` returns. Read-only.

    Parameters
    ----------
    pos_merge_id : str
        A ``PositionOutput`` merge id, such as a ripple detection key's
        ``pos_merge_id``.

    Returns
    -------
    pd.DataFrame
        The source table's ``fetch1_dataframe``, indexed by time in seconds.
    """
    from spyglass.position import PositionOutput

    output = PositionOutput()
    restriction = {"merge_id": pos_merge_id}
    source = (output & restriction).fetch1("source")
    parent_key = (getattr(output, source) & restriction).fetch1()
    parent = output.merge_get_parent_class(source)
    frame: pd.DataFrame = (parent & parent_key).fetch1_dataframe()
    return frame


def get_epoch_position_info(nwb_file_name: str, epoch_name: str) -> PositionInfoDict:
    """Load an epoch's position as ``load_data`` uses it.

    Looks up the epoch's position interval in ``PositionIntervalMap``, loads it
    with ``get_position_info``, then drops rows with NaN ``head_position_x`` or
    ``head_position_y``. ``load_data`` and the SWR runner both call this, so
    they share speed and span.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Epoch identifier in format "NN_rM" (e.g., "02_r1").

    Returns
    -------
    PositionInfoDict
        ``get_position_info``'s output, its ``position_info`` without the
        NaN-position rows.

    """
    from spyglass.common import PositionIntervalMap

    position_interval_name = (
        PositionIntervalMap
        & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
    ).fetch1("position_interval_name")
    position_data = get_position_info(nwb_file_name, epoch_name, position_interval_name)

    # Defense in depth: get_position_info now trims the interpolation
    # grid to the valid interval, so this dropna should be a no-op in
    # the typical case. Kept as a safety net against per-frame DLC
    # tracking gaps that may leave NaN x/y inside the valid window.
    n_before = len(position_data["position_info"])
    position_data["position_info"] = position_data["position_info"].dropna(
        subset=["head_position_x", "head_position_y"]
    )
    n_dropped = n_before - len(position_data["position_info"])
    if n_dropped > 0:
        logger.info(
            f"Dropped {n_dropped} time points with NaN position "
            f"({100 * n_dropped / n_before:.1f}%)"
        )

    return position_data
