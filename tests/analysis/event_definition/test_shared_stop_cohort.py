"""One cohort rule, two grains: the content cohort nests inside the rate cohort.

The rate ladders are fit on a stop table and the content ladders on a long
(event x candidate) frame, and ``occupancy_state`` does not say the same thing
at the two grains -- a stop is ``mixed`` as soon as its sampled states differ,
while an event is ``at_well`` only when its whole window is at one well. The
two predicates therefore differ on purpose, and what has to hold is not that
they agree label-for-label but that the content cohort NESTS inside the rate
cohort: every event the content family fits lies in a stop the rate family
fits. These tests pin that, the documented residual (an off-well event inside a
well-anchored stop is in the rate cohort and not the content one), and that the
runners call these functions rather than their own copies.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    STOP_COHORT_RULES,
    STOP_COHORTS,
    select_event_cohort,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.replay_models import contrast_aggregation
from continuum_swr_replay.analysis.replay_models.content import cohort as content_cohort
from continuum_swr_replay.analysis.replay_models.model_specs import MODEL_HIERARCHY
from scripts.replay_models.rate import fit_per_bin_rate_models as per_bin_mod
from scripts.replay_models.rate import fit_per_stop_rate_models as per_stop_mod
from scripts.replay_models.rate import fit_swr_contrast_rate as contrast_rate_mod
from scripts.replay_models.rate import fit_uncertainty_rate as uncertainty_mod

RATE_RUNNERS = (per_bin_mod, per_stop_mod, contrast_rate_mod, uncertainty_mod)


def _synthetic_session() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One session whose event labels are NOT copied from their stop's label.

    Four stops, deliberately built so the two grains disagree where the real
    tables disagree:

    ``s_well``    ``at_well`` stop holding two ``at_well`` events;
    ``s_mixed``   ``mixed`` stop holding two ``at_well`` events **and** one
                  ``in_transit`` event -- the settling-bout case that makes the
                  stop label ``mixed`` while most of its events are at a well;
    ``s_transit`` ``in_transit`` stop holding one ``in_transit`` event;
    ``s_unknown`` ``unknown`` stop holding no event.

    Off-well events carry ``is_current_arm = NaN``, as the canonical arm table
    does: the current arm is undefined when the window is not wholly at a well.
    """
    stops = pd.DataFrame(
        {
            "animal": ["a"] * 4,
            "session_date": [1] * 4,
            "epoch": ["02_r1"] * 4,
            "stop_id": ["s_well", "s_mixed", "s_transit", "s_unknown"],
            "occupancy_state": ["at_well", "mixed", "in_transit", "unknown"],
            "immobility_time": [1.0, 3.0, 2.0, 0.5],
        }
    )
    event_spec = [
        ("e_well_0", "s_well", "at_well"),
        ("e_well_1", "s_well", "at_well"),
        ("e_mixed_0", "s_mixed", "at_well"),
        ("e_mixed_1", "s_mixed", "at_well"),
        ("e_mixed_transit", "s_mixed", "in_transit"),
        ("e_transit", "s_transit", "in_transit"),
    ]
    rows = []
    for event_id, stop_id, state in event_spec:
        for arm in (0, 1):
            rows.append(
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "event_id": event_id,
                    "stop_id": stop_id,
                    "occupancy_state": state,
                    "candidate_arm": arm,
                    "is_current_arm": (
                        (1.0 - arm) if state == "at_well" else float("nan")
                    ),
                }
            )
    return stops, pd.DataFrame(rows)


AT_WELL_EVENTS = {"e_well_0", "e_well_1", "e_mixed_0", "e_mixed_1"}
WELL_ANCHORED_STOPS = {"s_well", "s_mixed"}


def test_stop_rule_keeps_the_well_anchored_stops():
    """A ``mixed`` stop is a well-anchored bout, so the rate cohort keeps it."""
    stops, _ = _synthetic_session()

    selected, report = select_stop_cohort(stops, cohort="at_well")

    assert set(selected["stop_id"]) == WELL_ANCHORED_STOPS
    assert report["counts_after"] == {"at_well": 1, "mixed": 1}
    assert report["rule"] == STOP_COHORT_RULES[("at_well", "stop")]


def test_event_rule_keeps_only_the_wholly_at_well_events():
    _, events = _synthetic_session()

    selected, report = select_event_cohort(events, cohort="at_well")

    assert set(selected["event_id"]) == AT_WELL_EVENTS
    # Two of the four live in a stop whose own label is 'mixed'.
    assert report["rule"] == STOP_COHORT_RULES[("at_well", "event")]
    assert report["rule"] != STOP_COHORT_RULES[("at_well", "stop")]


def test_the_content_cohort_nests_inside_the_rate_cohort():
    """Every event the content family fits lies in a stop the rate family fits."""
    stops, events = _synthetic_session()

    selected_stops, _ = select_stop_cohort(stops, cohort="at_well")
    selected_events, _ = select_event_cohort(events, cohort="at_well")

    assert set(selected_events["stop_id"]) <= set(selected_stops["stop_id"])
    # And the mixed stop is exactly what makes the naive stop predicate wrong:
    # dropping it would strand two of the four content events.
    naive = stops.loc[stops["occupancy_state"].eq("at_well"), "stop_id"]
    assert not set(selected_events["stop_id"]) <= set(naive)


def test_the_documented_residual_is_in_the_rate_cohort_only():
    """An off-well event inside a well-anchored stop is rate-only, by design."""
    stops, events = _synthetic_session()

    selected_stops, _ = select_stop_cohort(stops, cohort="at_well")
    selected_events, _ = select_event_cohort(events, cohort="at_well")

    residual = events.loc[
        events["stop_id"].isin(set(selected_stops["stop_id"]))
        & ~events["event_id"].isin(set(selected_events["event_id"])),
        "event_id",
    ].unique()

    assert set(residual) == {"e_mixed_transit"}


def test_in_transit_is_the_same_predicate_at_both_grains():
    stops, events = _synthetic_session()

    selected_stops, stop_report = select_stop_cohort(stops, cohort="in_transit")
    selected_events, event_report = select_event_cohort(events, cohort="in_transit")

    assert set(selected_stops["stop_id"]) == {"s_transit"}
    assert set(selected_events["event_id"]) == {"e_mixed_transit", "e_transit"}
    assert stop_report["rule"] == event_report["rule"]


def test_all_stops_restricts_nothing_at_either_grain():
    stops, events = _synthetic_session()

    selected_stops, _ = select_stop_cohort(stops, cohort="all_stops")
    selected_events, _ = select_event_cohort(events, cohort="all_stops")

    assert len(selected_stops) == len(stops)
    assert set(selected_events["event_id"]) == set(events["event_id"])


def test_at_well_events_are_the_location_predictor_complete_case():
    """A rung carrying ``is_current_arm`` already lost the off-well events.

    Its complete-case filter and the event-grain rule select the same events,
    which is why routing the content runners through the shared rule leaves
    every location-aware rung's numbers untouched.
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
    """Both content cohorts hold the shared function, not a private copy."""
    assert content_cohort.select_event_cohort is select_event_cohort
    assert contrast_aggregation.select_event_cohort is select_event_cohort


def test_content_runner_cohort_step_refuses_a_table_without_the_column():
    """A stale table cannot quietly fit the whole corpus.

    The content cohort calls its cohort step unconditionally, so a frame missing
    ``occupancy_state`` raises there. The canonical-schema gate upstream refuses
    such a table first; this is the second line of defence, and the reason the
    selection carries no ``column in frame`` guard.
    """
    frame = pd.DataFrame({"event_id": ["e0"], "candidate_arm": [0]})

    with pytest.raises(ValueError, match="occupancy_state"):
        content_cohort.select_event_cohort(frame, cohort="at_well")

    with pytest.raises(ValueError, match="occupancy_state"):
        content_cohort.prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=["M0"],
            hierarchy_kind="arm",
            table_label="synthetic",
        )


@pytest.mark.parametrize("module", RATE_RUNNERS, ids=lambda m: m.__name__)
def test_rate_runners_route_through_the_shared_rule(module):
    """Every rate runner holds the shared selection and suffix, not a copy."""
    assert module.select_stop_cohort is select_stop_cohort
    assert module.stop_cohort_suffix is stop_cohort_suffix


@pytest.mark.parametrize("module", RATE_RUNNERS, ids=lambda m: m.__name__)
def test_rate_runners_offer_every_cohort_on_the_command_line(module, capsys):
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--stop-cohort" in help_text
    for cohort in STOP_COHORTS:
        assert cohort in help_text


@pytest.mark.parametrize("module", RATE_RUNNERS, ids=lambda m: m.__name__)
def test_rate_runners_refuse_an_unknown_cohort(module, capsys):
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--stop-cohort", "at_the_well"])

    assert excinfo.value.code == 2
    assert "at_the_well" in capsys.readouterr().err


def test_cohort_suffixes_keep_the_canonical_names_unsuffixed():
    """The at-well bundle keeps the names every reader hard-codes."""
    assert stop_cohort_suffix("at_well") == ""
    assert stop_cohort_suffix("at_well", smoke=True) == "_smoke"
    assert stop_cohort_suffix("all_stops") == "_allstops"
    assert stop_cohort_suffix("all_stops", smoke=True) == "_allstops_smoke"
    assert stop_cohort_suffix("in_transit") == "_intransit"
