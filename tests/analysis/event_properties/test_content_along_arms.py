"""Tests of content_along_arms: arm positions and density along the arms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    arm_positions,
    check_arm_orientation,
    content_density_along_arms,
    leaf_and_trunk_profiles,
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


def test_arms_must_run_outward():
    check_arm_orientation([(9, 6), (6, 0), (6, 1)])
    with pytest.raises(ValueError, match="trunk"):
        check_arm_orientation([(6, 9), (6, 0)])
    with pytest.raises(ValueError, match="leaf"):
        check_arm_orientation([(9, 6), (0, 6)])


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
