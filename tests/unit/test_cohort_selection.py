"""Tests for whole-unit common-cohort selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.cohort_selection import (
    STOP_COHORT_RULES,
    freeze_common_cohort,
    select_event_cohort,
    select_stop_cohort,
)


def test_freeze_common_cohort_drops_whole_unit_for_one_missing_row():
    frame = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3],
            "response": [1, 0, 1, 0, 0, 1],
            "predictor": [0.1, 0.2, 0.3, float("nan"), 0.5, 0.6],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="event_id",
        required_cols=("response", "predictor"),
    )

    assert tuple(cohort["event_id"].unique()) == (1, 3)
    assert report["missing_unit_ids"] == (2,)
    assert report["n_missing_units"] == 1
    assert report["n_units_before"] == 3
    assert report["n_units_after"] == 2
    assert report["n_rows_after"] == 4
    pd.testing.assert_frame_equal(
        frame,
        pd.DataFrame(
            {
                "event_id": [1, 1, 2, 2, 3, 3],
                "response": [1, 0, 1, 0, 0, 1],
                "predictor": [0.1, 0.2, 0.3, float("nan"), 0.5, 0.6],
            }
        ),
    )


def test_freeze_common_cohort_drops_whole_unit_for_nonpositive_offset():
    frame = pd.DataFrame(
        {
            "stop_id": [1, 2, 2, 3, 4],
            "n_events": [0, 1, 1, 2, 3],
            "immobility_duration_s": [1.0, 1.0, 0.0, -0.5, 2.0],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="stop_id",
        required_cols=("n_events",),
        offset_col="immobility_duration_s",
    )

    assert tuple(cohort["stop_id"]) == (1, 4)
    assert report["nonpositive_offset_unit_ids"] == (2, 3)
    assert report["n_nonpositive_offset_units"] == 2


def test_freeze_common_cohort_drops_whole_unit_for_zero_response_sum():
    frame = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3],
            "response": [1, 0, 0, 0, 2, 0],
            "predictor": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="event_id",
        required_cols=("predictor",),
        response_col="response",
        drop_zero_response=True,
    )

    assert tuple(cohort["event_id"].unique()) == (1, 3)
    assert report["zero_response_unit_ids"] == (2,)
    assert report["n_zero_response_units"] == 1


def test_freeze_common_cohort_drops_units_without_exactly_one_choice():
    frame = pd.DataFrame(
        {
            "stop_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "chosen": [1, 0, 0, 1, 1, 0, 0, 0, 0],
            "predictor": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="stop_id",
        required_cols=("predictor",),
        choice_response_col="chosen",
    )

    assert tuple(cohort["stop_id"].unique()) == (1,)
    assert report["invalid_choice_unit_ids"] == (2, 3)
    assert report["n_invalid_choice_units"] == 2


@pytest.mark.parametrize("invalid_values", ([2, -1, 0], [0.5, 0.5, 0]))
def test_freeze_common_cohort_rejects_nonbinary_choice_values(invalid_values):
    frame = pd.DataFrame(
        {
            "stop_id": [1, 1, 1, 2, 2, 2],
            "chosen": [1, 0, 0, *invalid_values],
            "predictor": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="stop_id",
        required_cols=("predictor",),
        choice_response_col="chosen",
    )

    assert tuple(cohort["stop_id"].unique()) == (1,)
    assert report["invalid_choice_unit_ids"] == (2,)


def test_freeze_common_cohort_ignores_unobserved_categorical_unit_ids():
    frame = pd.DataFrame(
        {
            "event_id": pd.Categorical([1, 1], categories=[1, 2]),
            "response": [1, 0],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="event_id",
        required_cols=("response",),
        response_col="response",
        drop_zero_response=True,
    )

    assert tuple(cohort["event_id"].unique()) == (1,)
    assert report["zero_response_unit_ids"] == ()
    assert report["n_units_after"] == 1


def test_freeze_common_cohort_fails_loudly_for_missing_columns():
    frame = pd.DataFrame({"event_id": [1, 2], "response": [1, 1]})

    with pytest.raises(KeyError, match=r"missing required column\(s\).*predictor"):
        freeze_common_cohort(
            frame,
            unit_col="event_id",
            required_cols=("response", "predictor"),
        )


def test_freeze_common_cohort_requires_response_when_dropping_zero_units():
    frame = pd.DataFrame({"event_id": [1], "response": [1]})

    with pytest.raises(ValueError, match="response_col is required"):
        freeze_common_cohort(
            frame,
            unit_col="event_id",
            required_cols=("response",),
            drop_zero_response=True,
        )


def test_freeze_common_cohort_rejects_missing_unit_identifier():
    frame = pd.DataFrame({"event_id": [1.0, float("nan")], "response": [1, 1]})

    with pytest.raises(ValueError, match="event_id.*missing unit identifiers"):
        freeze_common_cohort(
            frame,
            unit_col="event_id",
            required_cols=("response",),
        )


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_freeze_common_cohort_drops_whole_unit_for_nonfinite_numeric_value(bad):
    frame = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "response": [1, 0, 1, 0],
            "predictor": [0.1, bad, 0.2, 0.3],
        }
    )

    cohort, report = freeze_common_cohort(
        frame,
        unit_col="event_id",
        required_cols=("response", "predictor"),
    )

    assert tuple(cohort["event_id"].unique()) == (2,)
    assert report["nonfinite_unit_ids"] == (1,)
    assert report["n_nonfinite_units"] == 1


def _occupancy_stops() -> pd.DataFrame:
    """Six stops spanning every occupancy state the canonical table carries."""
    return pd.DataFrame(
        {
            "stop_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
            "occupancy_state": [
                "at_well",
                "at_well",
                "in_transit",
                "mixed",
                "unknown",
                None,
            ],
            "immobility_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.mark.parametrize(
    ("cohort", "expected"),
    [
        # at_well is WELL-ANCHORED at the stop grain: a 'mixed' stop (s3) is an
        # at-well bout whose first settling samples read as in-transit.
        ("at_well", ["s0", "s1", "s3"]),
        ("in_transit", ["s2"]),
        ("all_stops", ["s0", "s1", "s2", "s3", "s4", "s5"]),
    ],
)
def test_select_stop_cohort_keeps_exactly_the_named_population(cohort, expected):
    stops = _occupancy_stops()

    selected, _ = select_stop_cohort(stops, cohort=cohort)

    assert list(selected["stop_id"]) == expected
    # The input is never mutated and the selection is re-indexed from zero.
    assert len(stops) == 6
    assert list(selected.index) == list(range(len(expected)))


def test_select_stop_cohort_report_states_the_rule_and_the_counts():
    stops = _occupancy_stops()

    _, report = select_stop_cohort(stops, cohort="at_well")

    assert report["cohort"] == "at_well"
    assert report["rule"] == STOP_COHORT_RULES[("at_well", "stop")]
    assert report["n_stops_before"] == 6
    assert report["n_stops_after"] == 3
    assert report["n_stops_dropped"] == 3
    assert report["counts_before"] == {
        "at_well": 2,
        "in_transit": 1,
        "mixed": 1,
        "unknown": 1,
        "missing": 1,
    }
    assert report["counts_after"] == {"at_well": 2, "mixed": 1}


def test_select_stop_cohort_all_stops_keeps_the_missing_state():
    """``all_stops`` states no restriction, so a NaN state is not a silent drop."""
    stops = _occupancy_stops()

    selected, report = select_stop_cohort(stops, cohort="all_stops")

    assert len(selected) == 6
    assert report["counts_after"]["missing"] == 1


def test_select_stop_cohort_raises_without_occupancy_state():
    stops = _occupancy_stops().drop(columns=["occupancy_state"])

    with pytest.raises(KeyError, match="occupancy_state"):
        select_stop_cohort(stops, cohort="at_well")


def test_select_stop_cohort_rejects_an_unknown_cohort():
    with pytest.raises(ValueError, match="unknown cohort"):
        select_stop_cohort(_occupancy_stops(), cohort="at_the_well")


def test_select_event_cohort_selects_whole_events_and_counts_them():
    long_df = pd.DataFrame(
        {
            "event_id": ["e0", "e0", "e1", "e1", "e2", "e2"],
            "candidate_arm": [0, 1, 0, 1, 0, 1],
            "occupancy_state": [
                "at_well",
                "at_well",
                "in_transit",
                "in_transit",
                "at_well",
                "at_well",
            ],
        }
    )

    selected, report = select_event_cohort(long_df, cohort="at_well")

    assert set(selected["event_id"]) == {"e0", "e2"}
    # Every candidate row of a retained event survives (the candidate set is
    # never shrunk by the cohort rule).
    assert len(selected) == 4
    assert report["n_events_before"] == 3
    assert report["n_events_after"] == 2
    assert report["n_events_dropped"] == 1
    assert report["n_rows_before"] == 6
    assert report["n_rows_after"] == 4
    # The tallies count events, not the candidate rows each event spans.
    assert report["counts_before"] == {"at_well": 2, "in_transit": 1}
    assert report["counts_after"] == {"at_well": 2}


def test_select_event_cohort_raises_without_the_event_column():
    long_df = pd.DataFrame({"occupancy_state": ["at_well"], "x": [1.0]})

    with pytest.raises(KeyError, match="event_id"):
        select_event_cohort(long_df, cohort="at_well")
