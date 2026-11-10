"""Tests for make_task_variables end-to-end integration.

Tests the full task variable computation pipeline including
trial number assignment, path progress, distance to goal,
turn direction, and omission/patch change indicators.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.task_variables import make_task_variables

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def linear_track():
    """Simple 3-node linear track: 0 --10cm-- 1 --10cm-- 2."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    nx.set_edge_attributes(graph, {(0, 1): 10.0, (1, 2): 10.0}, "distance")
    graph.nodes[0]["pos"] = (0, 0)
    graph.nodes[1]["pos"] = (10, 0)
    graph.nodes[2]["pos"] = (20, 0)
    graph.edges[(0, 1)]["edge_id"] = 0
    graph.edges[(1, 2)]["edge_id"] = 1
    return graph


@pytest.fixture
def position_info():
    """Position data: 4 seconds at 500 Hz, moving linearly."""
    time = np.arange(0.0, 4.0, 0.002)
    n = len(time)
    return pd.DataFrame(
        {
            "projected_x_position": np.linspace(0, 20, n),
            "projected_y_position": np.zeros(n),
            "track_segment_id": np.where(np.linspace(0, 20, n) < 10, 0, 1).astype(int),
            "linear_position": np.linspace(0, 20, n),
            "head_speed": np.full(n, 5.0),
        },
        index=time,
    )


@pytest.fixture
def trials_df():
    """Two trials covering the position data time range."""
    return pd.DataFrame(
        {
            "movement_start_time": [0.0, 2.0],
            "end_time": [2.0, 4.0],
            "destination_first_poke_in_time": [1.5, 3.5],
            "from_well": [0, 1],
            "to_well": [1, 2],
            "is_reward": [True, False],
            "is_patch_change": [False, True],
        }
    )


# ============================================================================
# Tests: Output Structure
# ============================================================================


class TestMakeTaskVariablesOutput:
    """Test that make_task_variables returns correct structure."""

    def test_index_matches_position_info(self, position_info, trials_df, linear_track):
        """Output index should match position_info index."""
        result = make_task_variables(position_info, trials_df, linear_track)
        np.testing.assert_array_equal(
            result.index.to_numpy(), position_info.index.to_numpy()
        )

    def test_has_required_columns(self, position_info, trials_df, linear_track):
        """Output should contain all required columns."""
        result = make_task_variables(position_info, trials_df, linear_track)

        required = [
            "trial_number",
            "time_to_goal",
            "start_node_id",
            "goal_node_id",
            "dist_to_goal",
            "path_progress",
            "turn_direction",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_has_omission_columns(self, position_info, trials_df, linear_track):
        """Output should have is_no_reward_N columns for default shifts."""
        result = make_task_variables(position_info, trials_df, linear_track)

        for shift in range(1, 6):
            col = f"is_no_reward_{shift}"
            assert col in result.columns, f"Missing column: {col}"

    def test_has_patch_change_columns(self, position_info, trials_df, linear_track):
        """Output should have is_patch_changeN columns for default shifts."""
        result = make_task_variables(position_info, trials_df, linear_track)

        for shift in range(-5, 6):
            col = f"is_patch_change{shift}"
            assert col in result.columns, f"Missing column: {col}"

    def test_custom_omission_shifts(self, position_info, trials_df, linear_track):
        """Custom omission shifts should produce corresponding columns."""
        result = make_task_variables(
            position_info, trials_df, linear_track, omission_shifts=[1, 2]
        )

        assert "is_no_reward_1" in result.columns
        assert "is_no_reward_2" in result.columns
        assert "is_no_reward_3" not in result.columns

    def test_custom_patch_change_shifts(self, position_info, trials_df, linear_track):
        """Custom patch change shifts should produce corresponding columns."""
        result = make_task_variables(
            position_info, trials_df, linear_track, patch_change_shifts=[-1, 0, 1]
        )

        assert "is_patch_change-1" in result.columns
        assert "is_patch_change0" in result.columns
        assert "is_patch_change1" in result.columns
        assert "is_patch_change-2" not in result.columns


# ============================================================================
# Tests: Variable Values
# ============================================================================


class TestMakeTaskVariablesValues:
    """Test correctness of computed task variable values."""

    def test_trial_number_assigned_during_trials(
        self, position_info, trials_df, linear_track
    ):
        """Trial number should be assigned for time points within trials."""
        result = make_task_variables(position_info, trials_df, linear_track)

        # Check trial 0: time [0.0, 2.0)
        trial_0_mask = (position_info.index >= 0.0) & (position_info.index < 2.0)
        trial_0_nums = result.loc[trial_0_mask, "trial_number"]
        assert (trial_0_nums == 0).all()

        # Check trial 1: time [2.0, 4.0)
        # The last sample at exactly 4.0 is excluded because end_time is exclusive
        trial_1_mask = (position_info.index >= 2.0) & (position_info.index < 4.0)
        trial_1_nums = result.loc[trial_1_mask, "trial_number"]
        assert (trial_1_nums == 1).all()

    def test_start_and_goal_node_ids(self, position_info, trials_df, linear_track):
        """Start and goal node IDs should match trial definitions."""
        result = make_task_variables(position_info, trials_df, linear_track)

        # Trial 0: from_well=0, to_well=1
        trial_0_mask = (position_info.index >= 0.0) & (position_info.index < 2.0)
        assert (result.loc[trial_0_mask, "start_node_id"] == 0).all()
        assert (result.loc[trial_0_mask, "goal_node_id"] == 1).all()

        # Trial 1: from_well=1, to_well=2
        trial_1_mask = (position_info.index >= 2.0) & (position_info.index < 4.0)
        assert (result.loc[trial_1_mask, "start_node_id"] == 1).all()
        assert (result.loc[trial_1_mask, "goal_node_id"] == 2).all()

    def test_time_to_goal_increases_before_arrival(
        self, position_info, trials_df, linear_track
    ):
        """Time to goal should increase from trial start to goal arrival."""
        result = make_task_variables(position_info, trials_df, linear_track)

        # Trial 0: start=0.0, goal_arrival=1.5
        pre_goal = result.loc[0.0:1.498, "time_to_goal"]
        assert np.all(np.diff(pre_goal.values) >= 0), "time_to_goal should increase"
        # Position index aligns with trial start at t=0.0 exactly; the
        # variable is *time since start*, so the first sample must be
        # exactly zero — a tolerance window would mask trial-anchor drift.
        assert pre_goal.iloc[0] == 0.0

    def test_time_to_goal_zero_after_arrival(
        self, position_info, trials_df, linear_track
    ):
        """Time to goal should be 0 after goal arrival."""
        result = make_task_variables(position_info, trials_df, linear_track)

        # Trial 0: goal_arrival=1.5, end=2.0
        post_goal = result.loc[1.5:1.998, "time_to_goal"]
        assert (post_goal == 0.0).all()

    def test_path_progress_in_bounds(self, position_info, trials_df, linear_track):
        """Path progress should be in [0, 1]."""
        result = make_task_variables(position_info, trials_df, linear_track)

        progress = result["path_progress"].dropna()
        assert progress.min() >= 0.0
        assert progress.max() <= 1.0

    def test_path_progress_zero_after_goal_arrival(
        self, position_info, trials_df, linear_track
    ):
        """Path progress should be 0 after goal arrival."""
        result = make_task_variables(position_info, trials_df, linear_track)

        post_goal = result.loc[1.5:1.998, "path_progress"]
        assert (post_goal == 0.0).all()

    def test_dist_to_goal_non_negative(self, position_info, trials_df, linear_track):
        """Distance to goal should be non-negative."""
        result = make_task_variables(position_info, trials_df, linear_track)

        dist = result["dist_to_goal"].dropna()
        assert (dist >= 0).all()

    def test_ffill_fills_inter_trial_gaps(self):
        """Forward fill should propagate values into gaps between trials."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        nx.set_edge_attributes(graph, {(0, 1): 10.0}, "distance")
        graph.nodes[0]["pos"] = (0, 0)
        graph.nodes[1]["pos"] = (10, 0)
        graph.edges[(0, 1)]["edge_id"] = 0

        # Gap between trials: [1.0, 2.0)
        time = np.arange(0.0, 3.0, 0.002)
        n = len(time)
        pos = pd.DataFrame(
            {
                "projected_x_position": np.linspace(0, 10, n),
                "projected_y_position": np.zeros(n),
                "track_segment_id": np.zeros(n, dtype=int),
                "linear_position": np.linspace(0, 10, n),
                "head_speed": np.full(n, 5.0),
            },
            index=time,
        )
        trials = pd.DataFrame(
            {
                "movement_start_time": [0.0, 2.0],
                "end_time": [1.0, 3.0],
                "destination_first_poke_in_time": [0.8, 2.8],
                "from_well": [0, 0],
                "to_well": [1, 1],
                "is_reward": [True, True],
                "is_patch_change": [False, False],
            }
        )

        result = make_task_variables(pos, trials, graph)

        # After ffill, the gap between trials should have values from trial 0
        gap_values = result.loc[1.0:1.998, "trial_number"]
        assert not gap_values.isna().any(), "Gap should be forward-filled"


# ============================================================================
# Tests: Omission and Patch Change Indicators
# ============================================================================


class TestOmissionAndPatchChangeIndicators:
    """Test reward omission and patch change indicator columns."""

    def test_is_no_reward_reflects_previous_trial(
        self, position_info, trials_df, linear_track
    ):
        """is_no_reward_1 should reflect reward status of previous trial."""
        result = make_task_variables(
            position_info, trials_df, linear_track, omission_shifts=[1]
        )

        # Trial 0: is_reward=True, so trial 1's is_no_reward_1 should be False
        trial_1_mask = (position_info.index >= 2.0) & (position_info.index < 4.0)
        assert not result.loc[trial_1_mask, "is_no_reward_1"].any()

    def test_patch_change_indicator_set_correctly(
        self, position_info, trials_df, linear_track
    ):
        """is_patch_change0 should reflect current trial's patch change status."""
        result = make_task_variables(
            position_info, trials_df, linear_track, patch_change_shifts=[0]
        )

        # Trial 0: is_patch_change=False
        trial_0_mask = (position_info.index >= 0.0) & (position_info.index < 2.0)
        assert not result.loc[trial_0_mask, "is_patch_change0"].any()

        # Trial 1: is_patch_change=True
        trial_1_mask = (position_info.index >= 2.0) & (position_info.index < 4.0)
        assert result.loc[trial_1_mask, "is_patch_change0"].all()


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestMakeTaskVariablesEdgeCases:
    """Test edge cases for make_task_variables."""

    def test_single_trial(self, linear_track):
        """Should work with just one trial."""
        time = np.arange(0.0, 2.0, 0.002)
        n = len(time)
        pos = pd.DataFrame(
            {
                "projected_x_position": np.linspace(0, 10, n),
                "projected_y_position": np.zeros(n),
                "track_segment_id": np.zeros(n, dtype=int),
                "linear_position": np.linspace(0, 10, n),
                "head_speed": np.full(n, 5.0),
            },
            index=time,
        )
        trials = pd.DataFrame(
            {
                "movement_start_time": [0.0],
                "end_time": [2.0],
                "destination_first_poke_in_time": [1.5],
                "from_well": [0],
                "to_well": [1],
                "is_reward": [True],
                "is_patch_change": [False],
            }
        )

        result = make_task_variables(pos, trials, linear_track)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n
        assert "trial_number" in result.columns

    def test_empty_trials(self, position_info, linear_track):
        """Empty trials DataFrame should produce all-NaN task variables."""
        empty_trials = pd.DataFrame(
            {
                "movement_start_time": pd.Series(dtype=float),
                "end_time": pd.Series(dtype=float),
                "destination_first_poke_in_time": pd.Series(dtype=float),
                "from_well": pd.Series(dtype=int),
                "to_well": pd.Series(dtype=int),
                "is_reward": pd.Series(dtype=bool),
                "is_patch_change": pd.Series(dtype=bool),
            }
        )

        result = make_task_variables(position_info, empty_trials, linear_track)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(position_info)
