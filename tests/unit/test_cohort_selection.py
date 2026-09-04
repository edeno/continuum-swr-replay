"""Tests for whole-unit common-cohort selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.cohort_selection import (
    freeze_common_cohort,
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
