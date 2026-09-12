"""One cohort rule, two grains: the rate and content families select the same stops.

The rate ladders are fit on a stop table and the content ladders on a long
(event x candidate) frame, so each family asks the cohort question at its own
grain. These tests pin the two answers together: both grains read
``occupancy_state`` through the same rule, and for a synthetic session they
select the same stops. They also pin the rule as a no-op where the content
runners previously reached the at-well cohort by other means, so adopting it
could not have moved a fitted number.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.replay.cohort_selection import (
    select_event_cohort,
    select_stop_cohort,
)
from scripts.content import fit_replay_content_models as content_mod
from scripts.content import fit_swr_contrast_content as contrast_mod


def _synthetic_session() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five stops of one session, and the events that occurred inside them.

    Off-well events carry ``is_current_arm = NaN``, as the canonical arm table
    does: the animal's current arm is undefined when the event window is not
    wholly at one well.
    """
    stops = pd.DataFrame(
        {
            "animal": ["a"] * 5,
            "session_date": [1] * 5,
            "epoch": ["02_r1"] * 5,
            "stop_id": ["s0", "s1", "s2", "s3", "s4"],
            "occupancy_state": [
                "at_well",
                "in_transit",
                "at_well",
                "mixed",
                "in_transit",
            ],
            "immobility_time": [1.0, 2.0, 1.5, 3.0, 2.5],
        }
    )
    # One event per stop, two candidate arms each.
    events = stops.loc[stops.index.repeat(2)].copy()
    events["event_id"] = [f"e{i}" for i in range(5) for _ in range(2)]
    events["candidate_arm"] = [0, 1] * 5
    events["is_current_arm"] = [
        (1.0 - arm) if state == "at_well" else float("nan")
        for state, arm in zip(
            events["occupancy_state"], events["candidate_arm"], strict=True
        )
    ]
    return stops, events.reset_index(drop=True)


def test_both_grains_select_the_same_stops_for_one_session():
    stops, events = _synthetic_session()

    selected_stops, stop_report = select_stop_cohort(stops, cohort="at_well")
    selected_events, event_report = select_event_cohort(events, cohort="at_well")

    assert set(selected_stops["stop_id"]) == {"s0", "s2"}
    # The events the content family keeps sit in exactly the stops the rate
    # family keeps -- that equality is the whole point of the shared rule.
    assert set(selected_events["stop_id"]) == set(selected_stops["stop_id"])
    assert stop_report["rule"] == event_report["rule"]


def test_the_two_grains_agree_on_every_cohort():
    stops, events = _synthetic_session()

    for cohort in ("at_well", "in_transit", "all_stops"):
        selected_stops, _ = select_stop_cohort(stops, cohort=cohort)
        selected_events, _ = select_event_cohort(events, cohort=cohort)
        # Every stop with an event; the synthetic session gives each stop one.
        assert set(selected_events["stop_id"]) == set(selected_stops["stop_id"]), cohort


def test_shared_rule_reproduces_the_content_runner_at_well_expression():
    """The adopted rule selects the event set the inline expression did."""
    _, events = _synthetic_session()

    selected, _ = select_event_cohort(events, cohort="at_well")
    legacy = events.loc[events["occupancy_state"].eq("at_well")]

    assert set(selected["event_id"]) == set(legacy["event_id"])
    assert len(selected) == len(legacy)


def test_at_well_selection_matches_the_location_predictor_complete_case():
    """A rung carrying ``is_current_arm`` already lost the off-well events.

    Its complete-case filter and the cohort rule select the same events, which
    is why routing the content runners through the shared rule leaves every
    location-aware rung's numbers untouched.
    """
    _, events = _synthetic_session()

    by_cohort, _ = select_event_cohort(events, cohort="at_well")
    complete_case = events.loc[
        ~events["event_id"].isin(
            events.loc[events["is_current_arm"].isna(), "event_id"]
        )
    ]

    assert set(by_cohort["event_id"]) == set(complete_case["event_id"])


def test_content_runners_route_through_the_shared_rule():
    """Both content runners hold the shared function, not a private copy."""
    assert content_mod.select_event_cohort is select_event_cohort
    assert contrast_mod.select_event_cohort is select_event_cohort
