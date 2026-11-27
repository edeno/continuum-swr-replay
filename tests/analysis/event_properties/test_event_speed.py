"""The two event speeds on distributions whose speed is known."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from continuum_swr_replay.analysis.event_properties import event_speed

BIN_S = 0.002
BIN_CM = 2.0


def line_track(n_bins: int = 60) -> SimpleNamespace:
    """A straight track of 2 cm place bins, one graph node per bin."""
    graph = nx.path_graph(n_bins)
    nx.set_edge_attributes(graph, BIN_CM, "distance")
    return environment(graph, np.arange(n_bins), np.arange(n_bins) * BIN_CM)


def y_track(n_arm: int = 8) -> SimpleNamespace:
    """Three arms of 2 cm bins that meet at a junction node holding no bin, with
    one zero-length edge, as the environment's graph has at a bin edge."""
    graph = nx.Graph()
    junction = 3 * n_arm
    for arm in range(3):
        nodes = [arm * n_arm + k for k in range(n_arm)]
        nx.add_path(graph, [junction, *nodes], distance=BIN_CM)
    graph.add_edge(n_arm - 1, junction + 1, distance=0.0)
    position = np.concatenate(
        [100.0 * arm + np.arange(n_arm) * BIN_CM for arm in range(3)]
    )
    return environment(graph, np.arange(3 * n_arm), position)


def environment(
    graph: nx.Graph, nodes: np.ndarray, position: np.ndarray
) -> SimpleNamespace:
    """What ``event_speeds`` reads of an environment, with one junction
    placeholder row (node -1) that no distribution has a column for."""
    table = pd.DataFrame({"node_id": nodes, "linear_position": position})
    placeholder = pd.DataFrame(
        {"node_id": [-1], "linear_position": [position.max() + 50]}
    )
    return SimpleNamespace(
        place_bin_centers_nodes_df_=pd.concat([table, placeholder], ignore_index=True),
        distance_between_nodes_=dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="distance")
        ),
        track_graph_with_bin_centers_edges_=graph,
    )


def bumps(center_cm: np.ndarray, n_bins: int = 60, width_cm: float = 4.0) -> np.ndarray:
    """A Gaussian bump over a straight track's bins at each center; rows sum to 1."""
    position = np.arange(n_bins) * BIN_CM
    mass = np.exp(
        -0.5 * ((position[None, :] - np.asarray(center_cm)[:, None]) / width_cm) ** 2
    )
    return mass / mass.sum(axis=1, keepdims=True)


def transport_by_linear_program(first, second, between) -> float:
    """The earth mover's distance as the transport problem itself."""
    n = first.size
    equal = np.zeros((2 * n, n * n))
    for k in range(n):
        equal[k, k * n : (k + 1) * n] = 1
        equal[n + k, k::n] = 1
    solved = linprog(
        between.ravel(),
        A_eq=equal[:-1],
        b_eq=np.r_[first, second][:-1],
        bounds=(0, None),
    )
    assert solved.success
    return float(solved.fun)


@pytest.mark.parametrize(
    ("x", "y", "slope", "intercept"),
    [
        ([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 2.0, 1.0),  # the free fit
        ([0.0, 1.0, 2.0], [5.0, 3.0, 1.0], 0.0, 3.0),  # falling: slope 0, the mean
        ([1.0, 2.0, 3.0], [1.0, 3.0, 5.0], 11 / 7, 0.0),  # free intercept -1: through 0
    ],
)
def test_nonnegative_line(x, y, slope, intercept):
    fitted = event_speed.nonnegative_line(np.array(x), np.array(y))
    assert fitted == pytest.approx((slope, intercept))


def test_nonnegative_line_refuses_one_x_or_a_negative_value():
    with pytest.raises(ValueError, match="two distinct"):
        event_speed.nonnegative_line(np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="negative"):
        event_speed.nonnegative_line(np.array([0.0, 1.0]), np.array([-1.0, 2.0]))


def test_map_line_fit_speed():
    elapsed = np.arange(6) * BIN_S
    assert event_speed.map_line_fit_speed(1000.0 * elapsed, elapsed) == pytest.approx(
        1000.0
    )
    assert event_speed.map_line_fit_speed(np.zeros(6), elapsed) == 0.0
    # out 4 cm and back: the free slope is 0 and the fitted one is not negative
    assert event_speed.map_line_fit_speed(
        np.array([0, 2, 4, 4, 2, 0.0]), elapsed
    ) == pytest.approx(0.0)
    # an undecoded bin is left out; two decoded bins are too few
    with_gap = np.array([0.0, np.nan, 4.0, 6.0, 8.0, 10.0])
    assert event_speed.map_line_fit_speed(with_gap, elapsed) == pytest.approx(1000.0)
    assert np.isnan(
        event_speed.map_line_fit_speed(np.array([0.0, 2.0, np.nan]), elapsed[:3])
    )


@pytest.mark.parametrize("track", [line_track(12), y_track(4)])
def test_transport_distance_is_the_transport_problems_optimum(track):
    nodes = track.place_bin_centers_nodes_df_["node_id"].to_numpy()[:-1]
    length, beyond = event_speed.transport_edges(
        track.track_graph_with_bin_centers_edges_, nodes
    )
    between = np.array(
        [[track.distance_between_nodes_[a][b] for b in nodes] for a in nodes]
    )
    rng = np.random.default_rng(0)
    for _ in range(5):
        first, second = rng.dirichlet(np.full(nodes.size, 0.3), size=2)
        ours = event_speed.transport_distance(
            first[None], second[None], length, beyond
        )[0]
        assert ours == pytest.approx(
            transport_by_linear_program(first, second, between), abs=1e-6
        )


def test_transport_distance_of_two_points_is_their_track_distance():
    track = y_track(4)
    nodes = track.place_bin_centers_nodes_df_["node_id"].to_numpy()[:-1]
    length, beyond = event_speed.transport_edges(
        track.track_graph_with_bin_centers_edges_, nodes
    )
    first, second = np.zeros((2, nodes.size))
    first[3] = 1.0  # the end of arm 0: 4 bins from the junction
    second[4 + 1] = 1.0  # arm 1, 2 bins from the junction
    distance = event_speed.transport_distance(
        first[None], second[None], length, beyond
    )[0]
    assert distance == pytest.approx(6 * BIN_CM)


def test_transport_edges_refuses_a_loop():
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 1.0, "distance")
    with pytest.raises(ValueError, match="not a tree"):
        event_speed.transport_edges(graph, np.arange(4))


def test_event_speeds_returns_the_speed_of_a_moving_bump_and_zero_for_a_still_one():
    """Event 1 travels at 1,000 cm/s (one bin per time bin) for 60 ms, event 2
    stays at 40 cm, event 3 travels but only its first half is Continuous, and
    event 4 is 7 bins (the shortest an event can be): lags of 5 and 6 bins only,
    one fewer than a fit needs."""
    track = line_track()
    n = 30
    centers = {
        1: 20.0 + 2.0 * np.arange(n),
        2: np.full(n, 40.0),
        3: 20.0 + 2.0 * np.arange(n),
        4: 20.0 + 2.0 * np.arange(7),
    }
    probability = {k: np.ones(v.size) for k, v in centers.items()}
    probability[3][15:] = 0.2
    event_id = np.concatenate([np.full(v.size, k) for k, v in centers.items()])
    center = np.concatenate(list(centers.values()))
    row = np.concatenate([1000 * k + np.arange(v.size) for k, v in centers.items()])
    continuous = bumps(center) * np.concatenate(list(probability.values()))[:, None]
    speeds = event_speed.event_speeds(
        event_id, row, center, continuous, track, BIN_S
    ).set_index("event_id")

    assert speeds.n_bins.tolist() == [30, 30, 30, 7]
    assert speeds.map_line_fit_speed_cm_s.tolist() == pytest.approx(
        [1000.0, 0.0, 1000.0, 1000.0]
    )
    assert speeds.transport_slope_speed_cm_s[[1, 2, 3]].tolist() == pytest.approx(
        [1000.0, 0.0, 1000.0]
    )
    assert speeds.transport_still_cm[[1, 2]].tolist() == pytest.approx(
        [0.0, 0.0], abs=1e-9
    )
    assert speeds.continuous_bin_share.tolist() == pytest.approx([1.0, 1.0, 0.5, 1.0])
    assert speeds.transport_n_lags.tolist() == [6, 6, 6, 2]
    assert np.isnan(speeds.transport_slope_speed_cm_s[4])


def test_transport_slope_sets_aside_a_difference_that_does_not_grow_with_the_lag():
    """A bump that alternates between two widths around a center moving at 500
    cm/s. Pairs an even lag apart have the same width and differ by the travel
    alone: the slope is the speed and the intercept 0. Pairs an odd lag apart
    differ by the widths as well: the intercept takes most of that, and the
    slope is below the speed (the two differences do not add), never above."""
    track = line_track()
    n = 40
    center = 20.0 + 1.0 * np.arange(n)
    wide = np.arange(n) % 2 == 0
    mass = np.where(
        wide[:, None], bumps(center, width_cm=8.0), bumps(center, width_cm=3.0)
    )
    length, beyond = event_speed.transport_edges(
        track.track_graph_with_bin_centers_edges_, np.arange(60)
    )
    every = np.ones(n, dtype=bool)

    def fitted(lag_bins):
        return event_speed.transport_slope_speed(
            mass, np.arange(n), every, length, beyond, BIN_S, lag_bins=lag_bins
        )

    even_speed, even_still, n_lags = fitted((6, 8, 10, 12))
    assert (even_speed, n_lags) == (pytest.approx(500.0, rel=0.02), 4)
    assert even_still == pytest.approx(0.0, abs=0.1)
    odd_speed, odd_still, _ = fitted((5, 7, 9, 11))
    assert 400.0 < odd_speed < 500.0
    assert odd_still > 0.5
    # one lag alone would read the width difference as speed
    one_lag = event_speed.transport_distance(mass[:-5], mass[5:], length, beyond).mean()
    assert one_lag / (5 * BIN_S) > 550.0


def test_a_split_label_pairs_bins_by_their_place_on_the_grid():
    """An event whose bins are rows 0-11 and 20-31: a pair is 5 bins apart on
    the grid, never across the gap as neighbors."""
    track = line_track()
    row = np.r_[np.arange(12), 20 + np.arange(12)]
    center = 20.0 + 2.0 * row
    speeds = event_speed.event_speeds(
        np.ones(24, dtype=int), row, center, bumps(center), track, BIN_S
    )
    assert speeds.transport_slope_speed_cm_s[0] == pytest.approx(1000.0)
    assert speeds.map_line_fit_speed_cm_s[0] == pytest.approx(1000.0)


def test_event_speeds_refuses_columns_for_other_bins():
    track = line_track()
    with pytest.raises(ValueError, match="real place bins"):
        event_speed.event_speeds(
            np.ones(3), np.arange(3), np.zeros(3), np.ones((3, 61)), track, BIN_S
        )
