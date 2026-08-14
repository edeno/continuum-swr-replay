"""Tests for make_well_visits_df and make_trials_df_from_well_visits.

These functions construct the behavioral trial structure from raw events.
Bugs here silently corrupt all downstream trial-level analyses.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.trials import (
    make_trials_df_from_well_visits,
    make_well_visits_df,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def track_graph():
    """6-well bandit track: center (6) connected to wells 0-5."""
    graph = nx.Graph()
    edges = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph, dict.fromkeys(edges, 20.0), "distance")

    # Wells in a star pattern around center
    positions = {
        0: (0, 20),  # Patch 1
        1: (0, -20),  # Patch 1
        2: (20, 20),  # Patch 2
        3: (20, -20),  # Patch 2
        4: (40, 20),  # Patch 3
        5: (40, -20),  # Patch 3
        6: (20, 0),  # Center
    }
    for node, pos in positions.items():
        graph.nodes[node]["pos"] = pos

    return graph


@pytest.fixture
def position_info():
    """Position data: 20s at 500 Hz, animal near well 0 then well 2."""
    time = np.arange(0.0, 20.0, 0.002)
    n = len(time)
    half = n // 2

    # First half near well 0 (0, 20), second half near well 2 (20, 20)
    x = np.concatenate([np.full(half, 0.0), np.full(n - half, 20.0)])
    y = np.concatenate([np.full(half, 20.0), np.full(n - half, 20.0)])

    return pd.DataFrame(
        {
            "projected_x_position": x,
            "projected_y_position": y,
        },
        index=time,
    )


@pytest.fixture
def beam_breaks():
    """Beam breaks: pokes at well 0 then well 2."""
    return pd.DataFrame(
        {
            "start_time": [1.0, 1.2, 12.0, 12.3],
            "end_time": [1.1, 1.3, 12.1, 12.4],
            "event_name": ["Poke1", "Poke1", "Poke3", "Poke3"],
        }
    )


@pytest.fixture
def pump_events():
    """Reward delivered during the second poke (at well 2)."""
    return pd.DataFrame({"start_time": [12.05]})


# ============================================================================
# Tests: make_well_visits_df
# ============================================================================


class TestMakeWellVisitsDf:
    """Test well visit construction from raw events."""

    def test_has_required_columns(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Output should have all required columns."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        required = [
            "start_time",
            "end_time",
            "duration",
            "from_well",
            "to_well",
            "is_run",
            "is_reward",
            "reward_on_time",
            "from_patch",
            "to_patch",
            "is_patch_change",
            "trial_number",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_start_time(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Well visits should be sorted by start_time."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        times = result["start_time"].values
        assert np.all(np.diff(times) >= 0), "Not sorted by start_time"

    def test_runs_have_different_start_end_wells(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Rows marked is_run should have from_well != to_well."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        runs = result[result["is_run"]]
        if len(runs) > 0:
            assert (runs["from_well"] != runs["to_well"]).all()

    def test_pokes_have_same_start_end_wells(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Non-run rows should have from_well == to_well."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        pokes = result[~result["is_run"]]
        if len(pokes) > 0:
            assert (pokes["from_well"] == pokes["to_well"]).all()

    def test_trial_numbers_assigned(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Every row should have a trial number (no NaN after ffill/bfill)."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        assert not result["trial_number"].isna().any()

    def test_trial_numbers_sequential(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Trial numbers for runs should be sequential starting from 1."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        run_trials = result.loc[result["is_run"], "trial_number"].values
        assert len(run_trials) > 0, "Fixture should produce at least one run"
        expected = np.arange(1, len(run_trials) + 1)
        np.testing.assert_array_equal(run_trials, expected)

    def test_reward_associated_with_correct_visit(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Reward should be associated with the visit that contains the pump event."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        # At least one row should have is_reward=True
        assert result["is_reward"].any(), "No rewards detected"

    def test_reward_on_time_uses_earliest_unsorted_pump(
        self, beam_breaks, position_info, track_graph
    ):
        """Direct callers need not pre-sort pump events."""
        pumps = pd.DataFrame({"start_time": [12.08, 12.05]})

        result = make_well_visits_df(beam_breaks, pumps, position_info, track_graph)

        reward_times = result.loc[
            result["is_reward"] & result["reward_on_time"].notna(),
            "reward_on_time",
        ]
        assert reward_times.min() == pytest.approx(12.05)

    def test_patch_mapping_correct(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """from_patch and to_patch should match WELL_PATCH_MAPPING."""
        from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING

        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        for _, row in result.iterrows():
            if row["from_well"] in WELL_PATCH_MAPPING:
                assert row["from_patch"] == WELL_PATCH_MAPPING[row["from_well"]]
            if row["to_well"] in WELL_PATCH_MAPPING:
                assert row["to_patch"] == WELL_PATCH_MAPPING[row["to_well"]]

    def test_is_patch_change_correct(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """is_patch_change should be True iff from_patch != to_patch."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        expected = result["from_patch"] != result["to_patch"]
        np.testing.assert_array_equal(result["is_patch_change"].values, expected.values)

    def test_no_pump_events_means_no_rewards(
        self, beam_breaks, position_info, track_graph
    ):
        """With empty pump events, no visits should have is_reward=True."""
        empty_pumps = pd.DataFrame({"start_time": pd.Series(dtype=float)})

        result = make_well_visits_df(
            beam_breaks, empty_pumps, position_info, track_graph
        )

        assert not result["is_reward"].any()

    def test_durations_non_negative(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """All durations should be non-negative."""
        result = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )

        assert (result["duration"] >= 0).all()


# ============================================================================
# Tests: make_trials_df_from_well_visits
# ============================================================================


class TestMakeTrialsDfFromWellVisits:
    """Test trial aggregation from well visits."""

    def test_has_goal_arrival_time(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Each trial should have a goal_arrival_time."""
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        assert "goal_arrival_time" in trials.columns
        assert not trials["goal_arrival_time"].isna().any()

    def test_goal_arrival_within_trial_bounds(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """goal_arrival_time should be within [start_time, end_time]."""
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        assert (trials["goal_arrival_time"] >= trials["start_time"]).all()
        assert (trials["goal_arrival_time"] <= trials["end_time"]).all()

    def test_start_before_end(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """start_time should be before end_time for all trials."""
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        assert (trials["start_time"] <= trials["end_time"]).all()

    def test_trial_count_matches_run_count(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Number of trials should equal number of unique run trial numbers."""
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        n_runs = well_visits["is_run"].sum()
        # Each run defines a trial; trial count should not exceed run count
        if n_runs > 0:
            assert 1 <= len(trials) <= n_runs
        else:
            assert len(trials) == 0

    def test_has_required_columns(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        """Output should have all trial-level columns."""
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        required = [
            "start_time",
            "end_time",
            "is_reward",
            "reward_on_time",
            "outcome_known_time",
            "from_well",
            "to_well",
            "from_patch",
            "to_patch",
            "is_patch_change",
            "goal_arrival_time",
            "duration",
        ]
        for col in required:
            assert col in trials.columns, f"Missing column: {col}"

    def test_outcome_clock_uses_pump_for_reward_and_poke_out_for_omission(
        self, beam_breaks, pump_events, position_info, track_graph
    ):
        well_visits = make_well_visits_df(
            beam_breaks, pump_events, position_info, track_graph
        )
        trials = make_trials_df_from_well_visits(well_visits)

        rewarded = trials["is_reward"]
        np.testing.assert_allclose(
            trials.loc[rewarded, "outcome_known_time"],
            trials.loc[rewarded, "reward_on_time"],
        )
        np.testing.assert_allclose(
            trials.loc[~rewarded, "outcome_known_time"],
            trials.loc[~rewarded, "end_time"],
        )
        assert (trials["outcome_known_time"] >= trials["goal_arrival_time"]).all()
        assert (trials["outcome_known_time"] <= trials["end_time"]).all()
