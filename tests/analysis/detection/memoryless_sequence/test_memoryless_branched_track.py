"""The detector on a branched track, where linear position is not distance.

Two bins that are adjacent in linearized coordinate can sit on different arms,
and two bins far apart in that coordinate can be neighbors through a junction.
The detector measures jumps and dispersion on the track graph for that
reason. These tests build a posterior that walks a chosen path over a
real Y-shaped ``Environment`` and check what the detector makes of it, with no
encoding model in the way.
"""

import networkx as nx
import numpy as np
import pytest
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.decoding.encoding_model import interior_bin_distances_cm

ARM_CM = 60.0
GAP_CM = 20.0
"""Linearization gap between arms. Wide enough that crossing a junction looks
like a large step in linear coordinate while being a small one on the graph."""

CONFIG = MemorylessSequenceConfig()
WINDOWS_PER_SECOND = 1.0 / CONFIG.stride_s


def y_track() -> nx.Graph:
    """Three arms of equal length meeting at one center node."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 3), (1, 3), (2, 3)])
    nx.set_edge_attributes(
        graph, {(0, 3): ARM_CM, (1, 3): ARM_CM, (2, 3): ARM_CM}, "distance"
    )
    nx.set_edge_attributes(graph, {(0, 3): 0, (1, 3): 1, (2, 3): 2}, "edge_id")
    nx.set_node_attributes(
        graph,
        {0: (0.0, 0.0), 1: (2 * ARM_CM, 0.0), 2: (ARM_CM, ARM_CM), 3: (ARM_CM, 0.0)},
        "pos",
    )
    return graph


@pytest.fixture(scope="module")
def track():
    """A fitted Y environment, its interior bins, and their graph distances."""
    covered = np.concatenate(
        [
            np.linspace(0.0, ARM_CM, 400),
            np.linspace(ARM_CM + GAP_CM, 2 * ARM_CM + GAP_CM, 400),
            np.linspace(2 * (ARM_CM + GAP_CM), 3 * ARM_CM + 2 * GAP_CM, 400),
        ]
    )
    environment = Environment(
        place_bin_size=CONFIG.place_bin_size_cm,
        track_graph=y_track(),
        edge_order=[(0, 3), (3, 1), (3, 2)],
        edge_spacing=GAP_CM,
    ).fit_place_grid(covered[:, None], infer_track_interior=True)

    interior = np.asarray(environment.is_track_interior_).ravel()
    return {
        "environment": environment,
        "linear_cm": np.asarray(environment.place_bin_centers_).ravel()[interior],
        "distance_cm": interior_bin_distances_cm(environment),
        "n_placeholders": int(
            (environment.place_bin_centers_nodes_df_.edge_id == -1).sum()
        ),
    }


def arm_bins(track, arm: int) -> np.ndarray:
    """Interior bin indices on one arm, ordered from the center outward."""
    linear_cm = track["linear_cm"]
    span = ARM_CM + GAP_CM
    on_arm = (linear_cm >= arm * span - 1e-9) & (linear_cm <= arm * span + ARM_CM)
    bins = np.flatnonzero(on_arm)
    # arm 0 is laid out tip-to-center, the others center-to-tip
    return bins[::-1] if arm == 0 else bins


def walk(track, bins: np.ndarray, duration_s: float) -> tuple:
    """A posterior that visits each bin in turn, and its window midpoints."""
    n_windows = max(2, round(duration_s * WINDOWS_PER_SECOND))
    visited = bins[
        np.clip(
            np.round(np.linspace(0, bins.size - 1, n_windows)).astype(int),
            0,
            bins.size - 1,
        )
    ]
    posterior = np.zeros((n_windows, track["distance_cm"].shape[0]))
    posterior[np.arange(n_windows), visited] = 1.0
    return posterior, np.arange(n_windows) * CONFIG.stride_s, visited


def detect(track, posterior, midpoint_s):
    return detect_candidate_sequences(
        posterior,
        midpoint_s,
        track["distance_cm"],
        np.zeros(posterior.shape[0]),
        CONFIG,
    )


class TestGeometry:
    def test_the_junction_placeholders_are_outside_the_interior_bins(self, track):
        assert track["n_placeholders"] == 2
        assert track["distance_cm"].shape[0] == track["linear_cm"].size

    def test_arms_are_laid_out_end_to_end_in_linear_coordinate(self, track):
        for arm in (0, 1, 2):
            assert arm_bins(track, arm).size >= ARM_CM / CONFIG.place_bin_size_cm - 1

    def test_bins_near_the_junction_are_close_on_the_graph_and_far_in_linear(
        self, track
    ):
        near_center = [arm_bins(track, arm)[0] for arm in (0, 1, 2)]
        linear_cm, distance_cm = track["linear_cm"], track["distance_cm"]

        for first, second in ((0, 1), (1, 2), (0, 2)):
            graph_gap = distance_cm[near_center[first], near_center[second]]
            linear_gap = abs(
                linear_cm[near_center[first]] - linear_cm[near_center[second]]
            )
            assert graph_gap < CONFIG.jump_threshold_cm
            assert linear_gap > CONFIG.jump_threshold_cm

    def test_arm_tips_are_closer_on_the_graph_than_in_linear(self, track):
        distance_cm, linear_cm = track["distance_cm"], track["linear_cm"]
        tip_0, tip_2 = arm_bins(track, 0)[-1], arm_bins(track, 2)[-1]

        assert distance_cm[tip_0, tip_2] == pytest.approx(2 * ARM_CM, abs=4.0)
        assert (
            abs(linear_cm[tip_0] - linear_cm[tip_2]) > 1.5 * distance_cm[tip_0, tip_2]
        )


class TestWithinOneArm:
    def test_a_long_traverse_is_replay(self, track):
        posterior, midpoint_s, _ = walk(track, arm_bins(track, 0), 0.25)

        found = detect(track, posterior, midpoint_s)

        assert found.is_replay.tolist() == [True]
        assert found.dispersion_cm[0] > CONFIG.dispersion_threshold_cm

    def test_a_short_traverse_is_not(self, track):
        """Under the dispersion criterion a replay must travel."""
        bins = arm_bins(track, 0)[:10]  # 20 cm of track
        posterior, midpoint_s, _ = walk(track, bins, 0.25)

        found = detect(track, posterior, midpoint_s)

        assert not found.is_replay.any()
        assert found.dispersion_cm.max() < CONFIG.dispersion_threshold_cm

    def test_retracing_a_path_does_not_change_dispersion(self, track):
        """Dispersion measures spread about a center, not distance traveled.
        An out-and-back replay covers twice the ground and scores the same, so
        dispersion and path length are not interchangeable summaries."""
        out = arm_bins(track, 0)
        there_and_back, midpoint_s, _ = walk(
            track, np.concatenate([out, out[::-1]]), 0.40
        )
        one_way, one_way_midpoint_s, _ = walk(track, out, 0.25)

        both = detect(track, there_and_back, midpoint_s)
        single = detect(track, one_way, one_way_midpoint_s)

        assert both.runs.shape == (1, 2)
        assert both.dispersion_cm[0] == pytest.approx(single.dispersion_cm[0], rel=0.05)


class TestAcrossAJunction:
    def test_a_replay_through_the_center_stays_one_sequence(self, track):
        """The step between arms is small on the graph, so nothing is cut."""
        path = np.concatenate([arm_bins(track, 0)[:14][::-1], arm_bins(track, 1)[:14]])
        posterior, midpoint_s, _ = walk(track, path, 0.25)

        found = detect(track, posterior, midpoint_s)

        assert found.runs.shape == (1, 2)
        assert found.is_replay.tolist() == [True]

    def test_the_same_path_would_be_cut_by_a_linear_metric(self, track):
        """What graph distance is for: in linearized coordinate the junction is a long
        step, so a detector differencing that coordinate would reject it."""
        path = np.concatenate([arm_bins(track, 0)[:14][::-1], arm_bins(track, 1)[:14]])
        posterior, midpoint_s, visited = walk(track, path, 0.25)
        linear_cm = track["linear_cm"][visited]

        found = detect(track, posterior, midpoint_s)

        assert np.nanmax(found.decisions.jump_cm) < CONFIG.jump_threshold_cm
        assert np.abs(np.diff(linear_cm)).max() > CONFIG.jump_threshold_cm


class TestSwitchingArms:
    def test_a_teleport_between_arms_cuts_the_sequence(self, track):
        """No journey through the center, so the graph step is large."""
        before = arm_bins(track, 0)[20:34]
        after = arm_bins(track, 2)[20:34]
        posterior, midpoint_s, _ = walk(track, np.concatenate([before, after]), 0.40)

        found = detect(track, posterior, midpoint_s)

        jumps = found.decisions.jump_cm
        assert np.nanmax(jumps) > CONFIG.jump_threshold_cm
        assert not found.decisions.gates.jump_below_threshold.all()

    def test_neither_half_of_a_switched_replay_survives_on_its_own(self, track):
        """Each half travels too little to clear the dispersion criterion, so
        a replay that changes arm is reported as nothing at all."""
        before = arm_bins(track, 0)[20:34]
        after = arm_bins(track, 2)[20:34]
        posterior, midpoint_s, _ = walk(track, np.concatenate([before, after]), 0.40)

        found = detect(track, posterior, midpoint_s)

        assert not found.is_replay.any()

    def test_the_cut_is_the_jump_criterion_and_not_the_spread(self, track):
        before = arm_bins(track, 0)[20:34]
        after = arm_bins(track, 2)[20:34]
        posterior, midpoint_s, _ = walk(track, np.concatenate([before, after]), 0.40)

        found = detect(track, posterior, midpoint_s)

        assert found.decisions.gates.spread_below_threshold.all()
        assert found.decisions.gates.has_center.all()
