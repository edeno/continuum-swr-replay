"""The two event speeds on simulated replays whose speed is known.

The recording and the model are the memoryless comparison's
(``memoryless_sequence.sweep.build_recording``: Figure 1's simulated run, the
state-space model fitted as a real decode is). A replay is drawn into it
spike by spike (``simulation.draw_replay``), a window around it is decoded, and
``event_speed.event_speeds`` reads the decode over the replay's own span. Both
speeds are within 30 % of the speed a traveling replay was drawn with and under
30 cm/s for one held still.
"""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from non_local_detector.simulate.sorted_spikes_simulation import TRACK_HEIGHT

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    N_NEURONS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    draw_replay,
    place_field_means,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import (
    build_recording,
    decode_window,
    usable_intervals,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_replays import (
    drawn_replay,
    planned_cells,
)
from continuum_swr_replay.analysis.event_properties.event_speed import event_speeds
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

pytestmark = [pytest.mark.slow, pytest.mark.integration]

CONTINUOUS = "Non-Local Continuous"
MIDPOINT_CM = 90.0
# (path, cm; duration, s): held still, then 100, 225, 375 and 750 cm/s
REPLAYS = ((0.0, 0.20), (20.0, 0.20), (45.0, 0.20), (45.0, 0.12), (60.0, 0.08))


@pytest.fixture(scope="module")
def recording() -> dict:
    return build_recording(seed=0)


def linear_environment(position_cm: np.ndarray) -> SimpleNamespace:
    """A straight track with one graph node per place bin, as ``event_speeds``
    reads an environment."""
    graph = nx.Graph()
    for k, gap in enumerate(np.diff(position_cm)):
        graph.add_edge(k, k + 1, distance=float(gap))
    return SimpleNamespace(
        place_bin_centers_nodes_df_=pd.DataFrame(
            {"node_id": np.arange(position_cm.size), "linear_position": position_cm}
        ),
        distance_between_nodes_=dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="distance")
        ),
        track_graph_with_bin_centers_edges_=graph,
    )


def measured(recording: dict, cell: dict) -> pd.Series:
    """The speeds of one drawn replay, over its own span."""
    means = place_field_means(N_NEURONS, TRACK_HEIGHT)
    replay = drawn_replay(
        cell["path_cm"],
        cell["midpoint_cm"],
        cell["duration_s"],
        cell["direction"],
        cell["at_s"],
    )
    injected, (start_s, end_s) = draw_replay(recording["spike_times"], means, replay)
    time, position, _ = decode_window(recording, (start_s, end_s))
    results = recording["detector"].predict(
        position_time=time, position=position, spike_times=injected, time=time
    )
    posterior = results["acausal_posterior"]
    state = np.asarray(posterior.coords["state"].values)
    place = np.asarray(posterior.coords["position"].values)[state == CONTINUOUS]
    interior = np.isfinite(place)
    continuous = np.nan_to_num(np.asarray(posterior.values)[:, state == CONTINUOUS])[
        :, interior
    ]
    non_local = (
        continuous
        + np.nan_to_num(
            np.asarray(posterior.values)[:, state == "Non-Local Fragmented"]
        )[:, interior]
    )
    a, b = np.searchsorted(time, [start_s, end_s])
    rows = np.arange(a, b + 1)
    speeds = event_speeds(
        np.ones(rows.size, dtype=int),
        rows,
        place[interior][non_local[rows].argmax(axis=1)],
        continuous[rows],
        linear_environment(place[interior]),
        1 / SAMPLING_FREQUENCY,
    )
    return speeds.iloc[0]


@pytest.mark.parametrize(("path_cm", "duration_s"), REPLAYS)
def test_speeds_of_a_drawn_replay(recording, path_cm, duration_s):
    cells = planned_cells(usable_intervals(recording), n_seeds=1)
    (cell,) = [
        c
        for c in cells
        if (c["path_cm"], c["duration_s"], c["midpoint_cm"], c["direction"])
        == (path_cm, duration_s, MIDPOINT_CM, 1)
    ]
    speeds = measured(recording, cell)
    for name in ("map_line_fit_speed_cm_s", "transport_slope_speed_cm_s"):
        if path_cm == 0:
            assert speeds[name] < 30.0, name
        else:
            assert speeds[name] == pytest.approx(path_cm / duration_s, rel=0.3), name
