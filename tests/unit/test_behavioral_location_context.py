"""Behavioral-location and past-only trial-context contracts."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay._helpers import (
    add_trial_context,
    behavioral_location_at_times,
    nearest_well_distance_cm,
    summarize_behavioral_location,
)


def _trials() -> pd.DataFrame:
    """Two valid trips, deliberately returned out of time order."""
    return pd.DataFrame(
        {
            "start_time": [4.0, 1.0],
            "end_time": [7.0, 3.5],
            "goal_arrival_time": [6.0, 3.0],
            "reward_on_time": [np.nan, 3.25],
            "outcome_known_time": [7.0, 3.25],
            "from_well": [1, 0],
            "to_well": [2, 1],
            "is_reward": [False, True],
            "is_patch_change": [True, False],
        }
    )


def test_behavioral_location_uses_arrival_and_departure_boundaries():
    times = np.array([0.5, 1.0, 2.999, 3.0, 3.999, 4.0, 5.999, 6.0, 8.0])

    out = behavioral_location_at_times(times, _trials())

    assert out["location_state"].tolist() == [
        "unknown",
        "in_transit",
        "in_transit",
        "at_well",
        "at_well",
        "in_transit",
        "in_transit",
        "at_well",
        "at_well",
    ]
    assert out["location_well"].tolist() == [
        pd.NA,
        pd.NA,
        pd.NA,
        1.0,
        1.0,
        pd.NA,
        pd.NA,
        2.0,
        2.0,
    ]
    assert out["at_well"].tolist() == [
        pd.NA,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def test_trial_context_is_past_only_while_next_well_is_explicitly_future():
    # Mid-trip 1, arrival, reward onset, departure, mid-trip 2, arrival,
    # omission boundary.
    times = np.array([2.0, 3.0, 3.249, 3.25, 4.0, 5.0, 6.0, 7.0])
    out = add_trial_context(pd.DataFrame(index=times), _trials())

    # The in-progress first trip exposes no destination/outcome through a
    # past-context field. Its destination is present only in ``next_well``.
    assert pd.isna(out.loc[2.0, "last_visited_well"])
    assert pd.isna(out.loc[2.0, "current_well"])
    assert pd.isna(out.loc[2.0, "prev_well"])
    assert pd.isna(out.loc[2.0, "reward_outcome"])
    assert pd.isna(out.loc[2.0, "is_after_switch"])
    assert pd.isna(out.loc[2.0, "completed_trial_id"])
    assert pd.isna(out.loc[2.0, "recent_switch_rate"])
    assert pd.isna(out.loc[2.0, "leaky_q_value_diff"])
    assert out.loc[2.0, "next_well"] == 1.0

    # Arrival is inclusive for destination/switch history, but the current
    # reward stays unavailable until its separate pump-onset clock.
    for time in (3.0, 3.249):
        assert out.loc[time, "last_visited_well"] == 1.0
        assert out.loc[time, "current_well"] == 1.0
        assert out.loc[time, "prev_well"] == 0.0
        assert pd.isna(out.loc[time, "reward_outcome"])
        assert pd.isna(out.loc[time, "is_after_reward"])
        assert pd.isna(out.loc[time, "is_after_omission"])
        assert pd.isna(out.loc[time, "last_rewarded_well"])
        assert out.loc[time, "is_after_switch"] == 0.0
        assert out.loc[time, "completed_trial_id"] == 0
        assert out.loc[time, "recent_switch_rate"] == 0.0
        assert out.loc[time, "leaky_q_value_diff"] == 0.0
        assert out.loc[time, "_goal_arrival_time"] == 3.0
        assert out.loc[time, "next_well"] == 2.0

    # Pump onset is inclusive for reward-bearing context.
    for time in (3.25, 4.0, 5.0):
        assert out.loc[time, "reward_outcome"] == 1.0
        assert out.loc[time, "is_after_reward"] == 1.0
        assert out.loc[time, "is_after_omission"] == 0.0
        assert out.loc[time, "last_rewarded_well"] == 1.0
        assert out.loc[time, "_last_reward_time"] == 3.25

    # Exact departure changes location, but cannot change completed context.
    assert out.loc[3.0, "location_state"] == "at_well"
    assert out.loc[4.0, "location_state"] == "in_transit"

    # Trial 2's destination is visible at arrival, while its omission remains
    # unknown until the conservative trip-end/poke-out boundary.
    assert out.loc[6.0, "last_visited_well"] == 2.0
    assert out.loc[6.0, "current_well"] == 2.0
    assert out.loc[6.0, "prev_well"] == 1.0
    assert pd.isna(out.loc[6.0, "reward_outcome"])
    assert pd.isna(out.loc[6.0, "is_after_omission"])
    assert out.loc[6.0, "is_after_switch"] == 1.0
    assert out.loc[6.0, "completed_trial_id"] == 1
    assert out.loc[6.0, "recent_switch_rate"] == pytest.approx(0.5)
    assert out.loc[6.0, "leaky_q_value_diff"] == pytest.approx(0.15)
    assert pd.isna(out.loc[6.0, "next_well"])
    assert out.loc[7.0, "reward_outcome"] == 0.0
    assert out.loc[7.0, "is_after_reward"] == 0.0
    assert out.loc[7.0, "is_after_omission"] == 1.0
    assert out.loc[7.0, "last_rewarded_well"] == 1.0

    assert str(out["completed_trial_id"].dtype) == "Int64"


def test_appending_future_trials_cannot_change_past_context_or_location():
    first = _trials().sort_values("goal_arrival_time").iloc[[0]].copy()
    appended = _trials()
    dot_time = np.array([3.5])

    short = add_trial_context(pd.DataFrame(index=dot_time), first)
    long = add_trial_context(pd.DataFrame(index=dot_time), appended)

    # ``next_well`` is intentionally forward-looking and can become known when
    # later data are appended. Every past-only field must be invariant.
    past_only = [
        "prev_well",
        "current_well",
        "last_visited_well",
        "last_rewarded_well",
        "reward_outcome",
        "is_after_reward",
        "is_after_omission",
        "is_after_switch",
        "completed_trial_id",
        "recent_switch_rate",
        "leaky_q_value_diff",
        "_goal_arrival_time",
        "_last_reward_time",
        "location_state",
        "location_well",
        "at_well",
    ]
    for column in past_only:
        pd.testing.assert_series_equal(
            short[column].reset_index(drop=True),
            long[column].reset_index(drop=True),
            check_names=False,
        )
    assert pd.isna(short["next_well"].iloc[0])
    assert long["next_well"].iloc[0] == 2.0


def test_empty_trials_have_explicit_unknown_location_and_context():
    times = np.array([1.0, 2.0])
    out = add_trial_context(pd.DataFrame(index=times), pd.DataFrame())

    assert out["location_state"].tolist() == ["unknown", "unknown"]
    assert out["at_well"].isna().all()
    assert out["location_well"].isna().all()
    assert out["last_visited_well"].isna().all()
    assert out["next_well"].isna().all()
    assert out["completed_trial_id"].isna().all()
    assert out["recent_switch_rate"].isna().all()
    assert out["leaky_q_value_diff"].isna().all()
    assert str(out["completed_trial_id"].dtype) == "Int64"


def test_full_trial_history_includes_a_completed_visit_with_no_sampled_stop():
    """A reduced stop sequence may jump 0→2, but history must include trial 1."""
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 2.0, 4.0],
            "goal_arrival_time": [1.0, 3.0, 5.0],
            "end_time": [1.5, 3.5, 5.5],
            "from_well": [1, 0, 0],
            "to_well": [0, 0, 1],
            "is_reward": [True, False, True],
            "is_patch_change": [False, True, True],
        }
    )

    # Deliberately sample no timestamp after trial 1 and before trial 2. This
    # mimics a completed visit with no immobility bout in the reduced stop table.
    out = add_trial_context(pd.DataFrame(index=[1.2, 5.2]), trials)

    assert out["completed_trial_id"].tolist() == [0, 2]
    np.testing.assert_allclose(out["recent_switch_rate"], [0.0, 2 / 3])
    # Trial 0 raises Q(well 0) to .65; the missing-stop omission on trial 1
    # updates it to .455. Thus trial 2's pre-update top-two margin is again 0.
    np.testing.assert_allclose(out["leaky_q_value_diff"], [0.0, 0.0])


@pytest.mark.parametrize(
    ("trials", "match"),
    [
        (
            pd.DataFrame(
                {
                    "start_time": [0.0, 2.0],
                    "end_time": [3.0, 4.0],
                    "from_well": [0, 1],
                    "to_well": [1, 2],
                    "is_reward": [1, 0],
                    "is_patch_change": [0, 0],
                }
            ),
            "overlap",
        ),
        (
            pd.DataFrame(
                {
                    "start_time": [2.0],
                    "end_time": [1.0],
                    "from_well": [0],
                    "to_well": [1],
                    "is_reward": [1],
                    "is_patch_change": [0],
                }
            ),
            "before its start",
        ),
    ],
)
def test_invalid_trial_intervals_fail_loudly(trials: pd.DataFrame, match: str):
    with pytest.raises(ValueError, match=match):
        behavioral_location_at_times(np.array([1.0]), trials)
    with pytest.raises(ValueError, match=match):
        add_trial_context(pd.DataFrame(index=[1.0]), trials)


@pytest.mark.parametrize("bad_well", [1.7, 99.0, np.nan])
def test_invalid_well_ids_fail_before_integer_mapping(bad_well):
    trials = _trials()
    trials["to_well"] = trials["to_well"].astype(float)
    trials.loc[0, "to_well"] = bad_well

    with pytest.raises(ValueError, match="to_well"):
        behavioral_location_at_times(np.array([1.0]), trials)
    with pytest.raises(ValueError, match="to_well"):
        add_trial_context(pd.DataFrame(index=[1.0]), trials)


@pytest.mark.parametrize(
    ("states", "wells", "expected"),
    [
        (["at_well", "at_well"], [2.0, 2.0], ("at_well", True, 2.0)),
        (["in_transit", "in_transit"], [np.nan, np.nan], ("in_transit", False, np.nan)),
        (["unknown", "unknown"], [np.nan, np.nan], ("unknown", pd.NA, np.nan)),
        (["at_well", "in_transit"], [2.0, np.nan], ("mixed", False, np.nan)),
        (["at_well", "at_well"], [1.0, 2.0], ("mixed", False, np.nan)),
    ],
)
def test_location_rollup_uses_the_entire_window(states, wells, expected):
    got = summarize_behavioral_location(states, wells)

    assert got["location_state"] == expected[0]
    if expected[1] is pd.NA:
        assert pd.isna(got["at_well"])
    else:
        assert got["at_well"] is expected[1]
    if np.isnan(expected[2]):
        assert np.isnan(got["location_well"])
    else:
        assert got["location_well"] == expected[2]


def test_location_rollup_accepts_location_frame_and_handles_empty_window():
    frame = behavioral_location_at_times(np.array([3.0, 3.5]), _trials())
    assert summarize_behavioral_location(frame) == {
        "location_state": "at_well",
        "at_well": True,
        "location_well": 1.0,
    }

    mixed_frame = behavioral_location_at_times(np.array([3.5, 4.0]), _trials())
    assert summarize_behavioral_location(mixed_frame) == {
        "location_state": "mixed",
        "at_well": False,
        "location_well": np.nan,
    }

    empty = summarize_behavioral_location([], [])
    assert empty["location_state"] == "unknown"
    assert pd.isna(empty["at_well"])
    assert np.isnan(empty["location_well"])


def _track_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(5.0, 0.0))
    graph.add_edges_from([(0, 2), (2, 1)])
    return graph


def test_nearest_well_distance_uses_degree_one_node_geometry():
    dot = pd.DataFrame(
        {
            "projected_x_position": [3.0, 8.0, np.nan],
            "projected_y_position": [4.0, 0.0, 0.0],
        }
    )

    np.testing.assert_allclose(
        nearest_well_distance_cm(dot, _track_graph()),
        np.array([5.0, 2.0, np.nan]),
        equal_nan=True,
    )


def test_nearest_well_distance_is_nan_when_geometry_is_unavailable():
    dot = pd.DataFrame({"projected_x_position": [1.0]})
    graph = nx.Graph()
    graph.add_edge(0, 1)

    assert np.isnan(nearest_well_distance_cm(dot, graph)).all()
