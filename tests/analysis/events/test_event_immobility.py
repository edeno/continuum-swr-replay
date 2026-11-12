"""Tests for the shared whole-event immobility contract."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
    event_speeds_are_immobile,
    filter_event_intervals_by_speed,
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
    whole_event_immobility_mask,
)


@pytest.mark.parametrize("bad_speed", [4.0, np.nan, np.inf, -np.inf, 10.0])
def test_interval_filter_requires_every_speed_finite_and_below_four(bad_speed):
    """One bad interior sample excludes the complete interval, not one bin."""
    time = np.arange(7, dtype=float)
    speed = np.array([1.0, 3.999, 2.0, 0.0, 1.0, bad_speed, 2.0])
    events = pd.DataFrame(
        {
            "start_time": [0.0, 4.0],
            "end_time": [2.0, 6.0],
            "detector_score": [11.0, 22.0],
        },
        index=pd.Index([7, 9], name="detector_event"),
    )

    filtered = filter_event_intervals_by_speed(events, speed=speed, time=time)

    assert filtered.index.tolist() == [7]
    assert filtered["detector_score"].tolist() == [11.0]


def test_interval_filter_uses_inclusive_event_boundaries():
    """A speed sample exactly at an interval endpoint belongs to the event."""
    events = pd.DataFrame({"start_time": [1.0], "end_time": [2.0]})
    time = np.arange(4, dtype=float)
    speed = np.array([0.0, 1.0, 4.0, 0.0])

    assert filter_event_intervals_by_speed(events, speed=speed, time=time).empty


def test_interval_filter_excludes_events_covering_no_sample():
    """An event with no speed sample is unverifiable, so it is excluded.

    This is the conservative reading of the whole-event rule: immobility
    that cannot be measured is not asserted. It bites for intervals
    falling in a position-tracking gap or past the position tail, which
    are dropped from ``n_SWR`` / ``n_MUA`` without a diagnostic — see the
    caveat in the module docstring.
    """
    # Samples exist at t ∈ {0, 1, 2, 10, 11}; both events miss all of them.
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    speed = np.zeros_like(time)  # immobile wherever it is measured
    events = pd.DataFrame(
        {"start_time": [3.0, 100.0], "end_time": [4.0, 101.0]},
    )

    assert filter_event_intervals_by_speed(events, speed=speed, time=time).empty


def test_empty_speed_array_is_not_immobile():
    """Zero samples must not vacuously satisfy the all-samples rule."""
    assert event_speeds_are_immobile(np.array([], dtype=float)) is False


def test_cohort_preserves_source_ids_when_an_earlier_event_is_rejected():
    """Filtering must leave ID gaps rather than renumber surviving events."""
    events = pd.DataFrame({"start_time": [0.0, 3.0], "end_time": [1.0, 4.0]})
    time = pd.Index(np.arange(5, dtype=float), name="time")
    speed = np.array([4.0, 1.0, 0.0, 1.0, 1.0])

    intervals, labels = build_speed_eligible_event_cohort(
        events,
        speed=speed,
        time=time,
        event_col="SWR_event",
    )

    assert labels.tolist() == [0, 0, 0, 2, 2]
    assert intervals["SWR_event"].tolist() == [2]
    assert intervals[["start_time", "end_time", "duration"]].iloc[0].tolist() == [
        3.0,
        4.0,
        1.0,
    ]


def test_cohort_can_number_only_eligible_nonlocal_intervals():
    """Non-Local IDs retain the historical post-selection numbering rule."""
    events = pd.DataFrame({"start_time": [0.0, 3.0], "end_time": [1.0, 4.0]})
    time = pd.Index(np.arange(5, dtype=float), name="time")
    speed = np.array([4.0, 1.0, 0.0, 1.0, 1.0])

    intervals, labels = build_speed_eligible_event_cohort(
        events,
        speed=speed,
        time=time,
        event_col="non_local_event",
        preserve_source_ids=False,
    )

    assert labels.tolist() == [0, 0, 0, 1, 1]
    assert intervals["non_local_event"].tolist() == [1]


def test_cohort_uses_one_axis_for_speed_gating_and_label_writing():
    """Sub-tolerance axis offsets cannot create unchecked labeled bins."""
    events = pd.DataFrame({"start_time": [0.0], "end_time": [0.004]})
    position_time = pd.Index([0.0000005, 0.0040005], name="time")
    speed = np.array([1.0, 5.0])

    intervals, labels = build_speed_eligible_event_cohort(
        events,
        speed=speed,
        time=position_time,
        event_col="non_local_event",
    )

    assert labels.tolist() == [1, 0]
    assert intervals["start_time"].tolist() == [position_time[0]]
    assert intervals["end_time"].tolist() == [position_time[0]]
    assert (speed[labels.to_numpy() > 0] < 4.0).all()


def test_cohort_rejects_same_source_intervals_sharing_a_sample():
    """One detector cannot assign the same sampled bin to two event IDs.

    Label overwrite followed by filtering used to let a rejected later event
    erase the overlapping tail of an otherwise eligible earlier event.
    """
    events = pd.DataFrame({"start_time": [0.0, 1.0], "end_time": [2.0, 3.0]})
    time = pd.Index(np.arange(4, dtype=float), name="time")
    speed = np.array([1.0, 1.0, 1.0, 5.0])

    with pytest.raises(EventDefinitionError, match="same-source.*overlap"):
        build_speed_eligible_event_cohort(
            events,
            speed=speed,
            time=time,
            event_col="SWR_event",
        )


@pytest.mark.parametrize("event_col", ["non_local_event", "SWR_event", "MUA_event"])
@pytest.mark.parametrize("bad_speed", [4.0, np.nan, np.inf, -np.inf, 10.0])
def test_labeled_event_mask_is_atomic_for_every_event_source(event_col, bad_speed):
    """The three event sources obey the same strict, all-finite rule."""
    data = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "01_r1",
            event_col: [1, 1, 1, 0, 2, 2, 2],
            "head_speed": [1.0, 3.999, 2.0, 0.0, 1.0, bad_speed, 2.0],
        }
    )

    mask = whole_event_immobility_mask(data, event_col)

    assert mask.tolist() == [True, True, True, False, False, False, False]


def test_labeled_event_mask_scopes_reused_labels_by_session():
    """A bad event in one session must not reject the same label in another."""
    data = pd.DataFrame(
        {
            "animal": ["a"] * 3 + ["b"] * 3,
            "day": 1,
            "epoch": "01_r1",
            "SWR_event": [1, 1, 1, 1, 1, 1],
            "head_speed": [1.0, 2.0, 3.0, 1.0, np.nan, 2.0],
        },
        index=[0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
    )

    mask = whole_event_immobility_mask(data, "SWR_event")

    assert mask.tolist() == [True, True, True, False, False, False]


def test_filtering_labels_is_atomic_and_does_not_mutate_input():
    data = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "01_r1",
            "SWR_event": [1, 1, 1, 0, 2, 2],
            "MUA_event": [0, 3, 3, 0, 4, 4],
            "head_speed": [1.0, 5.0, 1.0, 0.0, 1.0, 2.0],
        }
    )
    original = data.copy()

    filtered = filter_event_labels_by_speed(data, ["SWR_event", "MUA_event"])

    assert filtered["SWR_event"].tolist() == [0, 0, 0, 0, 2, 2]
    assert filtered["MUA_event"].tolist() == [0, 0, 0, 0, 4, 4]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("bad_probability", [0.8, 0.5, np.nan, np.inf, -np.inf])
def test_nonlocal_probability_contract_rejects_one_bad_labeled_bin(
    bad_probability,
):
    data = pd.DataFrame(
        {
            "non_local_event": [0, 1, 1, 1, 0],
            "continuous": [bad_probability, 0.9, bad_probability, 0.9, np.nan],
            "fragmented": 0.0,
        }
    )

    with pytest.raises(EventDefinitionError, match="probability-threshold contract"):
        validate_nonlocal_event_probability_contract(
            data,
            probability_cols=("continuous", "fragmented"),
        )


def test_nonlocal_probability_contract_ignores_unlabeled_bins():
    data = pd.DataFrame(
        {
            "non_local_event": [0, 1, 1, 0],
            "continuous": [np.nan, 0.81, 0.9, np.inf],
            "fragmented": 0.0,
        }
    )

    validate_nonlocal_event_probability_contract(
        data,
        probability_cols=("continuous", "fragmented"),
    )


def test_nonlocal_probability_columns_are_required_only_for_active_labels():
    validate_nonlocal_event_probability_contract(
        pd.DataFrame({"non_local_event": [0, 0]})
    )

    with pytest.raises(KeyError, match="Non-Local_Continuous_prob"):
        validate_nonlocal_event_probability_contract(
            pd.DataFrame({"non_local_event": [0, 1]})
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_speed_threshold_raises(threshold):
    data = pd.DataFrame(
        {
            "animal": ["a"],
            "day": [1],
            "epoch": ["01_r1"],
            "SWR_event": [1],
            "head_speed": [1.0],
        }
    )

    with pytest.raises(ValueError, match="speed_threshold must be finite and positive"):
        whole_event_immobility_mask(data, "SWR_event", speed_threshold=threshold)


def test_speed_filter_logs_how_many_intervals_it_removed(caplog):
    """The drop is recorded, so an eligible count is never reported alone."""
    time = np.array([0.0, 1.0, 2.0, 3.0])
    speed = np.array([0.0, 30.0, 0.0, 0.0])
    events = pd.DataFrame({"start_time": [0.0, 2.0], "end_time": [1.0, 3.0]})

    with caplog.at_level(
        logging.INFO,
        logger="continuum_swr_replay.analysis.events.immobility",
    ):
        kept = filter_event_intervals_by_speed(events, speed=speed, time=time)

    assert len(kept) == 1
    assert "removed 1 of 2 intervals" in caplog.text
