"""Behavioral-location and past-only trial-context contracts."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.bouts import (
    nearest_well_distance_cm,
)
from continuum_swr_replay.analysis.replay.covariates import (
    add_trial_context as _add_trial_context,
)
from continuum_swr_replay.analysis.replay.covariates import (
    behavioral_location_from_port_experiences,
    summarize_behavioral_location,
)
from continuum_swr_replay.data_loaders.constants import (
    OMISSION_KNOWN_LATENCY_SEC,
)
from tests.unit._port_experience_fixtures import port_experiences_from_trials


def add_trial_context(dot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Exercise the production API with a complete experience stream."""
    return _add_trial_context(
        dot,
        trials,
        port_experiences_from_trials(trials, prepend_initial=True),
    )


def _trials() -> pd.DataFrame:
    """Two valid trips, deliberately returned out of time order."""
    return pd.DataFrame(
        {
            "movement_start_time": [4.0, 1.0],
            "end_time": [7.0, 3.5],
            "destination_first_poke_in_time": [6.0, 3.0],
            "reward_on_time": [np.nan, 3.25],
            # Omission: arrival + the omission latency, capped at end_time.
            # Rewarded: pump onset. Derived from the constant so this fixture
            # cannot drift away from the production rule.
            "outcome_known_time": [
                min(6.0 + OMISSION_KNOWN_LATENCY_SEC, 7.0),
                3.25,
            ],
            "from_well": [1, 0],
            "to_well": [2, 1],
            "is_reward": [False, True],
            "is_patch_change": [True, False],
        }
    )


def test_behavioral_location_uses_arrival_and_departure_boundaries():
    times = np.array([0.5, 1.0, 2.999, 3.0, 3.999, 4.0, 5.999, 6.0, 8.0])

    out = behavioral_location_from_port_experiences(
        times, port_experiences_from_trials(_trials(), prepend_initial=True)
    )

    assert out["occupancy_state"].tolist() == [
        "at_well",
        "in_transit",
        "in_transit",
        "at_well",
        "in_transit",
        "in_transit",
        "in_transit",
        "at_well",
        "in_transit",
    ]
    assert out["occupancy_well_id"].tolist() == [
        0.0,
        pd.NA,
        pd.NA,
        1.0,
        pd.NA,
        pd.NA,
        pd.NA,
        2.0,
        pd.NA,
    ]
    assert out["at_well"].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_trial_context_is_past_only_while_next_well_is_explicitly_future():
    # Mid-trip 1, arrival, reward onset, departure, mid-trip 2, arrival,
    # omission boundary.
    times = np.array([2.0, 3.0, 3.249, 3.25, 4.0, 5.0, 6.0, 7.0])
    out = add_trial_context(pd.DataFrame(index=times), _trials())

    # The in-progress first trip exposes no destination/outcome through a
    # past-context field. Its destination is present only in ``next_well``.
    assert pd.isna(out.loc[2.0, "last_visited_well"])
    assert pd.isna(out.loc[2.0, "prev_well"])
    assert pd.isna(out.loc[2.0, "is_rewarded_outcome"])
    assert pd.isna(out.loc[2.0, "did_last_completed_trip_switch_patch"])
    assert pd.isna(out.loc[2.0, "completed_trial_id"])
    assert pd.isna(out.loc[2.0, "patch_switch_fraction_last5_completed_trips"])
    assert pd.isna(out.loc[2.0, "leaky_q_value_diff"])
    assert out.loc[2.0, "next_well"] == 1.0

    # Arrival is inclusive for destination/switch history, but the current
    # reward stays unavailable until its separate pump-onset clock.
    for time in (3.0, 3.249):
        assert out.loc[time, "last_visited_well"] == 1.0
        assert out.loc[time, "prev_well"] == 0.0
        assert pd.isna(out.loc[time, "is_rewarded_outcome"])
        assert pd.isna(out.loc[time, "last_rewarded_well"])
        assert out.loc[time, "did_last_completed_trip_switch_patch"] == 0.0
        assert out.loc[time, "completed_trial_id"] == 0
        assert out.loc[time, "patch_switch_fraction_last5_completed_trips"] == 0.0
        assert out.loc[time, "leaky_q_value_diff"] == 0.0
        assert out.loc[time, "_goal_arrival_time"] == 3.0
        assert out.loc[time, "next_well"] == 2.0

    # Pump onset is inclusive for reward-bearing context.
    for time in (3.25, 4.0, 5.0):
        assert out.loc[time, "is_rewarded_outcome"] == 1.0
        assert (1.0 - out.loc[time, "is_rewarded_outcome"]) == 0.0
        assert out.loc[time, "last_rewarded_well"] == 1.0
        assert out.loc[time, "_last_reward_time"] == 3.25

    # Exact departure changes location, but cannot change completed context.
    assert out.loc[3.0, "occupancy_state"] == "at_well"
    assert out.loc[4.0, "occupancy_state"] == "in_transit"

    # Trial 2's destination is visible at arrival, while its omission remains
    # unknown until the conservative trip-end/poke-out boundary.
    assert out.loc[6.0, "last_visited_well"] == 2.0
    assert out.loc[6.0, "prev_well"] == 1.0
    assert pd.isna(out.loc[6.0, "is_rewarded_outcome"])
    assert out.loc[6.0, "did_last_completed_trip_switch_patch"] == 1.0
    assert out.loc[6.0, "completed_trial_id"] == 1
    assert out.loc[6.0, "patch_switch_fraction_last5_completed_trips"] == pytest.approx(
        0.5
    )
    assert out.loc[6.0, "leaky_q_value_diff"] == pytest.approx(0.15)
    assert pd.isna(out.loc[6.0, "next_well"])
    assert out.loc[7.0, "is_rewarded_outcome"] == 0.0
    assert (1.0 - out.loc[7.0, "is_rewarded_outcome"]) == 1.0
    assert out.loc[7.0, "last_rewarded_well"] == 1.0

    assert str(out["completed_trial_id"].dtype) == "Int64"


def test_appending_future_trials_cannot_change_past_context_or_location():
    first = _trials().sort_values("destination_first_poke_in_time").iloc[[0]].copy()
    appended = _trials()
    dot_time = np.array([3.5])

    short = add_trial_context(pd.DataFrame(index=dot_time), first)
    long = add_trial_context(pd.DataFrame(index=dot_time), appended)

    # ``next_well`` is intentionally forward-looking and can become known when
    # later data are appended. Every past-only field must be invariant.
    past_only = [
        "prev_well",
        "last_visited_well",
        "last_rewarded_well",
        "is_rewarded_outcome",
        "did_last_completed_trip_switch_patch",
        "completed_trial_id",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
        "_goal_arrival_time",
        "_outcome_known_time",
        "_last_reward_time",
        "occupancy_state",
        "occupancy_well_id",
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


def test_initial_port_dwell_is_at_well_until_physical_departure():
    trials = _trials()
    experiences = port_experiences_from_trials(
        trials, prepend_initial=True, initial_reward=False
    )
    arrival = float(experiences.loc[0, "arrival_time"])
    departure = float(experiences.loc[0, "departure_time"])
    destination_arrival = float(experiences.loc[1, "arrival_time"])
    times = np.array(
        [arrival - 0.1, arrival, departure - 1e-6, departure, destination_arrival]
    )

    out = behavioral_location_from_port_experiences(times, experiences)

    assert out["occupancy_state"].tolist() == [
        "unknown",
        "at_well",
        "at_well",
        "in_transit",
        "at_well",
    ]
    assert out.loc[arrival, "occupancy_well_id"] == experiences.loc[0, "port"]


def _break_reward_alignment(frame):
    """Flip the first observed destination's reward, keeping the reward clock
    internally consistent, so only the cross-source reward check can trip."""
    out = frame.copy()
    now_reward = not bool(out.loc[1, "is_reward"])
    out.loc[1, "is_reward"] = now_reward
    out.loc[1, "reward_on_time"] = (
        out.loc[1, "outcome_known_time"] if now_reward else np.nan
    )
    return out


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda frame: frame.assign(port=frame["port"].mask(frame.index == 0, 1)),
            "initial port",
        ),
        (
            lambda frame: frame.assign(
                movement_trial_number=frame["movement_trial_number"].mask(
                    frame.index == 1, 99
                )
            ),
            "movement trial",
        ),
        (
            lambda frame: frame.assign(
                departure_time=frame["departure_time"].mask(
                    frame.index == 0, frame.loc[0, "departure_time"] + 0.1
                )
            ),
            "departure",
        ),
        (
            # An observed destination whose reward disagrees with the movement
            # trial must fail (bit-exact reward check at observed positions).
            # Flip is_reward AND its reward_on_time together so the experiences
            # frame stays internally consistent (omission => reward_on_time NaN),
            # isolating the CROSS-SOURCE reward mismatch rather than the internal
            # reward-clock invariant.
            _break_reward_alignment,
            "rewards do not align",
        ),
        (
            # An observed destination whose outcome-known clock disagrees with the
            # trial must fail (clock check at observed positions).
            lambda frame: frame.assign(
                outcome_known_time=frame["outcome_known_time"].mask(
                    frame.index == 1, frame.loc[1, "outcome_known_time"] + 0.5
                )
            ),
            "outcome_known_time does not align",
        ),
    ],
)
def test_trial_experience_cross_source_misalignment_fails(mutate, match):
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)

    with pytest.raises(ValueError, match=match):
        _add_trial_context(
            pd.DataFrame(index=[3.0]), trials, mutate(experiences.copy())
        )


def test_boundary_only_terminal_destination_is_rejected_as_noncanonical():
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    terminal_index = experiences["experience_id"].idxmax()
    terminal_arrival = float(experiences.loc[terminal_index, "arrival_time"])
    experiences.loc[terminal_index, "is_boundary_derived"] = True
    experiences.loc[terminal_index, "outcome_observed"] = False
    experiences.loc[terminal_index, "reward_on_time"] = np.nan
    experiences.loc[terminal_index, "outcome_known_time"] = np.nan
    experiences.loc[terminal_index, "is_reward"] = pd.NA

    with pytest.raises(ValueError, match="physical poke"):
        _add_trial_context(pd.DataFrame(index=[terminal_arrival]), trials, experiences)


def test_empty_trials_have_explicit_unknown_location_and_context():
    times = np.array([1.0, 2.0])
    out = add_trial_context(pd.DataFrame(index=times), pd.DataFrame())

    assert out["occupancy_state"].tolist() == ["unknown", "unknown"]
    assert out["at_well"].isna().all()
    assert out["occupancy_well_id"].isna().all()
    assert out["last_visited_well"].isna().all()
    assert out["next_well"].isna().all()
    assert out["completed_trial_id"].isna().all()
    assert out["patch_switch_fraction_last5_completed_trips"].isna().all()
    assert out["leaky_q_value_diff"].isna().all()
    assert out["_goal_arrival_time"].isna().all()
    assert out["_outcome_known_time"].isna().all()
    assert str(out["completed_trial_id"].dtype) == "Int64"


def test_empty_trials_reject_nonempty_port_experience_stream():
    experiences = port_experiences_from_trials(_trials(), prepend_initial=True)
    with pytest.raises(ValueError, match="must be empty"):
        _add_trial_context(pd.DataFrame(index=[1.0]), pd.DataFrame(), experiences)


def test_full_trial_history_includes_a_completed_visit_with_no_sampled_stop():
    """A reduced stop sequence may jump 0→2, but history must include trial 1."""
    trials = pd.DataFrame(
        {
            "movement_start_time": [0.0, 2.0, 4.0],
            "destination_first_poke_in_time": [1.0, 3.0, 5.0],
            "end_time": [1.5, 3.5, 5.5],
            "from_well": [1, 0, 2],
            "to_well": [0, 2, 4],
            "is_reward": [True, True, True],
            "is_patch_change": [False, True, True],
        }
    )

    # Deliberately sample no timestamp after trial 1 and before trial 2. This
    # mimics a completed visit with no immobility bout in the reduced stop table.
    out = add_trial_context(pd.DataFrame(index=[1.2, 5.2]), trials)

    assert out["completed_trial_id"].tolist() == [0, 2]
    np.testing.assert_allclose(
        out["patch_switch_fraction_last5_completed_trips"], [0.0, 2 / 3]
    )
    # Trial 0 raises Q(well 0) to .65; the missing-stop reward on trial 1
    # raises Q(well 2) to .65. Thus trial 2's pre-update top-two margin is 0;
    # skipping the hidden visit would incorrectly leave a .15 margin.
    np.testing.assert_allclose(out["leaky_q_value_diff"], [0.0, 0.0])


def test_trial_context_rejects_a_discontinuous_poke_chain():
    trials = pd.DataFrame(
        {
            "movement_start_time": [0.0, 2.0],
            "destination_first_poke_in_time": [1.0, 3.0],
            "end_time": [1.5, 3.5],
            "from_well": [0, 2],
            "to_well": [1, 3],
            "is_reward": [False, False],
            "is_patch_change": [False, False],
        }
    )
    with pytest.raises(ValueError, match="continuous physical-poke chain"):
        add_trial_context(pd.DataFrame(index=[3.2]), trials)


@pytest.mark.parametrize(
    ("trials", "match"),
    [
        (
            pd.DataFrame(
                {
                    "movement_start_time": [0.0, 2.0],
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
                    "movement_start_time": [2.0],
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
        add_trial_context(pd.DataFrame(index=[1.0]), trials)


@pytest.mark.parametrize("bad_well", [1.7, 99.0, np.nan])
def test_invalid_well_ids_fail_before_integer_mapping(bad_well):
    trials = _trials()
    trials["to_well"] = trials["to_well"].astype(float)
    trials.loc[0, "to_well"] = bad_well

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

    assert got["occupancy_state"] == expected[0]
    if expected[1] is pd.NA:
        assert pd.isna(got["at_well"])
    else:
        assert got["at_well"] is expected[1]
    if np.isnan(expected[2]):
        assert np.isnan(got["occupancy_well_id"])
    else:
        assert got["occupancy_well_id"] == expected[2]


def test_location_rollup_accepts_location_frame_and_handles_empty_window():
    experiences = port_experiences_from_trials(_trials(), prepend_initial=True)
    frame = behavioral_location_from_port_experiences(
        np.array([3.0, 3.25]), experiences
    )
    assert summarize_behavioral_location(frame) == {
        "occupancy_state": "at_well",
        "at_well": True,
        "occupancy_well_id": 1.0,
    }

    mixed_frame = behavioral_location_from_port_experiences(
        np.array([3.25, 3.5]), experiences
    )
    assert summarize_behavioral_location(mixed_frame) == {
        "occupancy_state": "mixed",
        "at_well": False,
        "occupancy_well_id": np.nan,
    }

    empty = summarize_behavioral_location([], [])
    assert empty["occupancy_state"] == "unknown"
    assert pd.isna(empty["at_well"])
    assert np.isnan(empty["occupancy_well_id"])


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


def test_outcome_known_time_names_the_boundary_from_arrival_onward():
    """``_outcome_known_time`` is a clock, not an outcome-gated value.

    It is unknown until a trip has completed, and from that arrival onward it
    carries the trip's own outcome-known time -- including at bins that
    precede the boundary, where the gated ``is_rewarded_outcome`` is still
    missing. Expected values come from the fixture's own trial clocks so the
    contract cannot be restated as literal seconds.
    """
    trials = _trials()
    ordered = trials.sort_values("destination_first_poke_in_time").reset_index(
        drop=True
    )
    arrival = ordered["destination_first_poke_in_time"].to_numpy(dtype=float)
    outcome = ordered["outcome_known_time"].to_numpy(dtype=float)
    before_first_arrival = float(ordered["movement_start_time"].iloc[0])
    between = 0.5 * (outcome[0] + arrival[1])

    times = np.array(
        [before_first_arrival, arrival[0], outcome[0], between, arrival[1], outcome[1]]
    )
    out = add_trial_context(pd.DataFrame(index=times), trials)

    # No trip has completed yet: the boundary is genuinely unknown.
    assert pd.isna(out.loc[before_first_arrival, "_outcome_known_time"])

    # Arrival is inclusive, and the boundary is known before it is reached
    # even though the outcome itself is not yet observable there.
    assert out.loc[arrival[0], "_outcome_known_time"] == pytest.approx(outcome[0])
    assert pd.isna(out.loc[arrival[0], "is_rewarded_outcome"])
    for time in (outcome[0], between):
        assert out.loc[time, "_outcome_known_time"] == pytest.approx(outcome[0])
    assert out.loc[outcome[0], "is_rewarded_outcome"] == 1.0

    # The next completed trip replaces the boundary with its own.
    for time in (arrival[1], outcome[1]):
        assert out.loc[time, "_outcome_known_time"] == pytest.approx(outcome[1])
    assert out.loc[arrival[1], "_goal_arrival_time"] == pytest.approx(arrival[1])
