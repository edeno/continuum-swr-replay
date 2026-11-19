"""Tests of content_along_arms: arm positions and density along the arms."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    arm_positions,
    check_arm_orientation,
    content_density_along_arms,
    distance_from_animal_shares,
    distance_to_nearest_well,
    leaf_and_trunk_profiles,
    log2_density_ratio,
)


@pytest.fixture
def bin_nodes() -> pd.DataFrame:
    """Two arms: edge 6 with bin centers 1..9 cm, a junction placeholder at 15,
    edge 0 with centers 21..41 cm."""
    return pd.DataFrame(
        {
            "edge_id": [6] * 5 + [-1] + [0] * 6,
            "linear_position": [1, 3, 5, 7, 9, 15, 21, 25, 29, 33, 37, 41],
        }
    )


def test_a_position_takes_its_nearest_real_bins_arm_and_fraction(bin_nodes):
    positions = arm_positions(np.array([1.0, 5.0, 9.0, 31.0, 41.0]), bin_nodes)
    assert positions["arm"].tolist() == [6, 6, 6, 0, 0]
    np.testing.assert_allclose(positions["fraction"], [0, 0.5, 1, 0.5, 1])
    np.testing.assert_allclose(positions["arm_length_cm"], [8, 8, 8, 20, 20])


def test_a_junction_placeholder_is_never_an_arm(bin_nodes):
    # 15 cm is the placeholder's center: nearest real bins are 9 (edge 6) and 21.
    positions = arm_positions(np.array([14.0, 16.0]), bin_nodes)
    assert positions["arm"].tolist() == [6, 0]
    np.testing.assert_allclose(positions["fraction"], [1, 0])  # clipped


def test_a_missing_position_is_refused(bin_nodes):
    with pytest.raises(ValueError, match="not finite"):
        arm_positions(np.array([1.0, np.nan]), bin_nodes)


def test_arms_must_run_outward_to_the_graphs_end_points():
    # Center 9, junction 6, wells 0 and 1: the wells are the degree-1 nodes.
    graph = nx.Graph([(9, 6), (6, 0), (6, 1)])
    check_arm_orientation([(9, 6), (6, 0), (6, 1)], graph)
    with pytest.raises(ValueError, match="trunk"):
        check_arm_orientation([(6, 9), (6, 0)], graph)
    with pytest.raises(ValueError, match="leaf"):
        check_arm_orientation([(9, 6), (0, 6)], graph)
    # Node 0 is a well only because it is an end point: given another arm it is
    # a junction, and an arm ending there does not end at a well.
    with pytest.raises(ValueError, match="leaf"):
        check_arm_orientation([(9, 6), (6, 0)], nx.Graph([(9, 6), (6, 0), (0, 2)]))


def _bins(animal, is_swr, arm, fraction, length=10.0):
    return pd.DataFrame(
        {
            "animal": animal,
            "is_swr": is_swr,
            "arm": arm,
            "fraction": fraction,
            "arm_length_cm": length,
        }
    )


def test_density_is_share_per_cm_and_the_animals_weigh_the_same():
    # Arm 0 (leaf) and arm 6 (trunk), 10 cm each, 2 segments of 5 cm.
    # Animal a: 3 bins at arm 0's well end, 1 on arm 6's first half.
    # Animal b: 1 bin at arm 0's well end only (its one bin weighs as a's four).
    bins = pd.concat(
        [
            _bins("a", False, [0, 0, 0, 6], [0.9, 0.9, 1.0, 0.1]),
            _bins("b", False, [0], [0.8]),
        ],
        ignore_index=True,
    )
    density = content_density_along_arms(bins, n_segments=2)["density_per_cm"]
    # a: 0.75 / 5 cm = 0.15 at (0, 1); b: 1 / 5 = 0.2; mean 0.175.
    assert density[(False, 0, 1)] == pytest.approx(0.175)
    assert density[(False, 6, 0)] == pytest.approx((0.25 / 5 + 0) / 2)
    assert density[(False, 0, 0)] == 0
    # Each animal integrates to 1, so the mean does too.
    assert (density * 5).sum() == pytest.approx(1)


def test_profiles_average_the_arms_of_each_type():
    bins = _bins("a", True, [0, 1, 6, 6], [0.9, 0.9, 0.1, 0.1])
    density = content_density_along_arms(bins, n_segments=2)
    profiles = leaf_and_trunk_profiles(density)["density_per_cm"]
    # leaf arms 0 and 1 each hold 1/4 at segment 1: 0.05 per cm, mean 0.05.
    assert profiles[(True, "leaf", 1)] == pytest.approx(0.05)
    assert profiles[(True, "trunk", 0)] == pytest.approx(0.1)


def test_the_ratio_is_log2_no_swr_over_swr_and_undefined_without_content():
    # Arm 0, 2 segments of 5 cm. No SWR: 1 bin in each half; SWR: both bins in
    # the far half. Densities: no SWR 0.1, 0.1; SWR 0, 0.2.
    bins = pd.concat(
        [_bins("a", False, [0, 0], [0.1, 0.9]), _bins("a", True, [0, 0], [0.9, 0.9])],
        ignore_index=True,
    )
    ratio = log2_density_ratio(content_density_along_arms(bins, n_segments=2))
    assert np.isnan(ratio[(0, 0)])
    assert ratio[(0, 1)] == pytest.approx(-1.0)


def test_distance_to_the_nearest_well_goes_out_through_either_end_of_the_arm():
    # Center 9 -- trunk (edge 6, 30 cm) -- junction 6 -- leaf arms to wells 0
    # (edge 0, 50 cm) and 1 (edge 1, 20 cm). The center joins two more trunks
    # whose wells are 200 cm away, so it is a junction, not an end point.
    graph = nx.Graph()
    graph.add_edge(9, 6, edge_id=6, distance=30.0)
    graph.add_edge(6, 0, edge_id=0, distance=50.0)
    graph.add_edge(6, 1, edge_id=1, distance=20.0)
    for trunk, junction, well in ((7, 7, 2), (8, 8, 4)):
        graph.add_edge(9, junction, edge_id=trunk, distance=100.0)
        graph.add_edge(junction, well, edge_id=well, distance=100.0)
    order = [(9, 6), (6, 0), (6, 1), (9, 7), (7, 2), (9, 8), (8, 4)]
    distance = distance_to_nearest_well(
        np.array([0, 0, 0, 6, 1]), np.array([1.0, 0.2, 0.0, 0.0, 0.5]), graph, order
    )
    # On arm 0 at its end: 0. At 0.2 (10 cm from the junction): 40 cm to well 0,
    # 10 + 20 = 30 cm to well 1 through the junction. At the junction: 20 cm.
    # The center (start of the trunk): 30 + 20 = 50 cm. Halfway down arm 1: 10.
    np.testing.assert_allclose(distance, [0.0, 30.0, 20.0, 50.0, 10.0])


def test_distance_shares_split_at_a_well_and_weigh_animals_the_same():
    bins = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b"],
            "is_swr": False,
            "distance_to_animal_cm": [5.0, 15.0, 15.0, 25.0, 25.0],
            "distance_to_well_cm": [1.0, 50.0, 2.0, 3.0, 40.0],
        }
    )
    shares = distance_from_animal_shares(bins, np.array([0, 10, 20, 30]), 20.0)
    no_swr = shares.loc[False]
    # a: [0, 10) one at a well (1/4); [10, 20) one at a well, one elsewhere;
    # [20, 30) one at a well. b: its one bin is elsewhere in [20, 30).
    np.testing.assert_allclose(no_swr["at_well"], [0.125, 0.125, 0.125])
    np.testing.assert_allclose(no_swr["elsewhere"], [0.0, 0.125, 0.5])
    assert no_swr.to_numpy().sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside"):
        distance_from_animal_shares(bins, np.array([0, 10, 20]), 20.0)
