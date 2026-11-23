"""The events summary's builder functions: each statistic reaches the key that names it.

Each test calls one builder function on a hand-made input whose statistics all
differ, so a value filed under a neighboring key (the other class, the other
bound, the other detector) fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    SPIKE_EVIDENCE_METRICS,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_animals
from continuum_swr_replay.reported import events as builder
from continuum_swr_replay.reported.values import Value
from tests.analysis.detector_comparison.test_inventory_nonlocal_comparison import (
    SELECTED,
    VISITS,
    _exposure,
    _write_run,
)
from tests.support._restriction_tables import time_sums

NAN = np.nan


def _by_key(values: list[Value]) -> dict[str, Value]:
    by_key = {value.key: value for value in values}
    assert len(by_key) == len(values)
    return by_key


def _numbers(values: list[Value]) -> dict[str, float]:
    return {key: value.value for key, value in _by_key(values).items()}


def _events() -> pd.DataFrame:
    """Thirteen events of three animals in four epochs, as
    (animal, epoch, overlap_class, start offset, end offset): 7 non-SWR, 2
    SWR-contained and 4 SWR-partial. Animal a is 3/4 non-SWR, b 2/5, c 2/4."""
    rows = [
        ("a", "e1", "non_swr", NAN, NAN),
        ("a", "e1", "non_swr", NAN, NAN),
        ("a", "e2", "non_swr", NAN, NAN),
        ("a", "e2", "swr_contained", 0.01, -0.01),
        ("b", "e1", "non_swr", NAN, NAN),
        ("b", "e1", "non_swr", NAN, NAN),
        ("b", "e1", "swr_partial", 0.02, -0.02),
        ("b", "e1", "swr_partial", 0.03, -0.03),
        ("b", "e1", "swr_partial", 0.05, -0.04),
        ("c", "e1", "non_swr", NAN, NAN),
        ("c", "e1", "non_swr", NAN, NAN),
        ("c", "e1", "swr_contained", 0.06, -0.05),
        ("c", "e1", "swr_partial", 0.30, 0.90),
    ]
    frame = pd.DataFrame(
        rows, columns=["animal", "epoch", "overlap_class", "start", "end"]
    )
    return pd.DataFrame(
        {
            "animal": frame["animal"],
            "session_date": "d1",
            "epoch": frame["epoch"],
            "overlap_class": frame["overlap_class"],
            "whole_event_speed_eligible": True,
            "is_swr": frame["overlap_class"] != "non_swr",
            # an overlapping event's SWR spans [10, 11]; the others have none
            "matched_swr_start": np.where(frame["start"].notna(), 10.0, NAN),
            "matched_swr_end": np.where(frame["end"].notna(), 11.0, NAN),
            "event_start_s": 10.0 + frame["start"].fillna(0.0),
            "event_end_s": 11.0 + frame["end"].fillna(0.0),
        }
    )


def test_event_cohort_values_name_each_class_and_each_animal_count():
    numbers = _numbers(builder.event_cohort_values(_events(), "s"))
    assert numbers == {
        "nonlocal_events": 13,
        "nonlocal_events_non_swr": 7,
        "nonlocal_events_swr_contained": 2,
        "nonlocal_events_swr_partial": 4,
        "non_swr_fraction": pytest.approx(7 / 13),
        "swr_contained_fraction": pytest.approx(2 / 13),
        "swr_partial_fraction": pytest.approx(4 / 13),
        # Figure 2d's bars: the mean over animals of each animal's shares
        "non_swr_fraction_animal_mean": pytest.approx(0.55),
        "swr_contained_fraction_animal_mean": pytest.approx(1 / 6),
        "swr_partial_fraction_animal_mean": pytest.approx(0.85 / 3),
        "non_swr_fraction_animal_min": pytest.approx(0.4),
        "non_swr_fraction_animal_max": pytest.approx(0.75),
        "animals": 3,
        "epochs": 4,
        # a and c (2 non-SWR against 1 and 1); in c they are half, not more
        "animals_non_swr_plurality": 2,
        "animals_non_swr_majority": 1,
    }


def _quality_sessions() -> pd.DataFrame:
    """Four sessions: errors 1, 3, 9 and none (median 3, mean 13/3); every state
    fraction distinct, so a fraction filed under another state or movement
    fails. Session d has no immobile bin."""
    return pd.DataFrame(
        {
            "median_error_cm": [1.0, 9.0, 3.0, NAN],
            "moving_Local_prob": [0.9, 0.8, 0.7, 0.95],
            "moving_No-Spike_prob": [0.05, 0.1, 0.2, 0.0],
            "moving_Non-Local_Continuous_prob": [0.03, 0.06, 0.07, 0.01],
            "moving_Non-Local_Fragmented_prob": [0.02, 0.04, 0.03, 0.04],
            "immobile_Local_prob": [0.5, 0.3, 0.4, NAN],
            "immobile_No-Spike_prob": [0.2, 0.3, 0.25, NAN],
            "immobile_Non-Local_Continuous_prob": [0.21, 0.22, 0.23, NAN],
            "immobile_Non-Local_Fragmented_prob": [0.09, 0.18, 0.12, NAN],
        }
    )


def test_decoding_error_values_are_the_median_over_sessions_with_an_error():
    numbers = _numbers(builder.decoding_error_values(_quality_sessions(), "s"))
    assert numbers == {"decoding_error_median_cm": 3.0, "decoding_error_sessions": 3}


def test_state_fraction_values_name_each_movement_and_state():
    numbers = _numbers(
        builder.state_fraction_values(_quality_sessions(), "s")
        + builder.immobile_state_fraction_values(_quality_sessions(), "s")
    )
    assert numbers == {
        # the median over the four sessions: the mean of the middle two
        "state_fraction_moving_local": pytest.approx(0.85),
        "state_fraction_moving_no_spike": pytest.approx(0.075),
        "state_fraction_moving_continuous": pytest.approx(0.045),
        "state_fraction_moving_fragmented": pytest.approx(0.035),
        # over the three sessions with an immobile bin
        "state_fraction_immobile_local": pytest.approx(0.4),
        "state_fraction_immobile_no_spike": pytest.approx(0.25),
        "state_fraction_immobile_continuous": pytest.approx(0.22),
        "state_fraction_immobile_fragmented": pytest.approx(0.12),
    }
    values = _by_key(
        builder.state_fraction_values(_quality_sessions(), "s")
        + builder.immobile_state_fraction_values(_quality_sessions(), "s")
    )
    for key, value in values.items():
        if "_immobile_" in key:
            assert value.figures == builder.TAGS["state_fractions_immobile"].figures
            assert "inside a well visit" in value.description
        else:
            assert value.figures == builder.TAGS["state_fractions"].figures


def test_calibration_values_are_each_bins_mean_over_animals_of_session_means():
    """Animal a has two sessions, with distances 10 - k and 20 - k in bin k:
    its value is 15 - k; animal b has one, 5 - k/2. The value is the mean over
    animals, not over the three sessions pooled, with the smallest and largest
    animal."""
    centers = (np.arange(10) + 0.5) / 10
    rows = [
        {"animal": animal, "day": 1, "epoch": epoch, "prob_bin_center": c,
         "median_error_cm": value(k)}
        for animal, epoch, value in (
            ("a", 1, lambda k: 10.0 - k),
            ("a", 2, lambda k: 20.0 - k),
            ("b", 1, lambda k: 5.0 - k / 2),
        )
        for k, c in enumerate(centers)
    ]  # fmt: skip
    calibration = pd.DataFrame(rows).sample(frac=1.0, random_state=0)

    values = _by_key(builder.calibration_values(calibration, "s"))

    for k, word in enumerate(builder.CALIBRATION_BINS):
        key = f"calibration_error_bin_{word}"
        a, b = 15.0 - k, 5.0 - k / 2
        assert values[key].value == pytest.approx((a + b) / 2)
        assert values[f"{key}_animal_min"].value == pytest.approx(min(a, b))
        assert values[f"{key}_animal_max"].value == pytest.approx(max(a, b))
    description = values["calibration_error_bin_one"].description
    assert "centered on 0.05 (bin 1 of 10)" in description
    assert "mean over animals" in description and "calibrat" not in description
    assert {v.figures for v in values.values()} == {("S2d",)}
    with pytest.raises(ValueError, match="has 9 bins, not 10"):
        builder.calibration_values(
            calibration[calibration["prob_bin_center"] != centers[3]], "s"
        )


def test_nonlocal_probability_values_are_the_three_shares_with_the_animals_range():
    """The shares at most 0.2, between 0.2 and 0.8, and above 0.8 of the small
    table of the analysis test, each with its smallest and largest animal,
    tagged to question 2a.6 and the supplement's panel."""
    from tests.analysis.decode_quality.test_nonlocal_probability import _stored

    counts = _stored(
        {
            ("a", "1"): {0.0: 10},
            ("a", "2"): {0.1: 5, 0.9: 5},
            ("b", "1"): {0.5: 1, 0.98: 3},
        }
    ).assign(epoch_id="e")
    values = _by_key(builder.nonlocal_probability_values(counts, "s"))
    assert values["nonlocal_probability_low_share"].value == pytest.approx(0.375)
    assert values["nonlocal_probability_middle_share"].value == pytest.approx(0.125)
    assert values["nonlocal_probability_high_share"].value == pytest.approx(0.5)
    assert values["nonlocal_probability_high_share_animal_min"].value == 0.25
    assert values["nonlocal_probability_high_share_animal_max"].value == 0.75
    assert len(values) == 9
    assert {v.figures for v in values.values()} == {("S2c",)}
    assert {v.questions for v in values.values()} == {("2a.6",)}


def test_boundary_offset_values_are_in_milliseconds_start_and_end_apart():
    values = _by_key(builder.boundary_offset_values(_events(), "s"))
    assert values["swr_overlapping_events"].value == 6
    # starts 10, 20, 30, 50, 60, 300 ms; ends -50, -40, -30, -20, -10, 900 ms
    assert values["start_offset_median_ms"].value == pytest.approx(40.0)
    assert values["end_offset_median_ms"].value == pytest.approx(-25.0)
    assert values["start_offset_median_ms"].unit == "ms"
    # starts beyond 25 ms: 30, 50, 60, 300; beyond 50: 60, 300 (of six)
    assert values["start_offset_beyond_twenty_five_ms_fraction"].value == 4 / 6
    assert values["start_offset_beyond_fifty_ms_fraction"].value == 2 / 6
    # ends beyond 25 ms: -50, -40, -30, 900; beyond 50: 900
    assert values["end_offset_beyond_twenty_five_ms_fraction"].value == 4 / 6
    assert values["end_offset_beyond_fifty_ms_fraction"].value == 1 / 6


def test_boundary_timing_values_are_the_nine_shares_with_the_animals_range():
    """The six SWR-overlapping events: a (together, together); b (together,
    together), (later, earlier) twice; c (later, earlier), (later, later)."""
    values = _by_key(builder.boundary_timing_values(_events(), "s"))
    assert len(values) == 9 * 4
    shares = {k: v.value for k, v in values.items() if k.endswith("_share")}
    assert sum(shares.values()) == pytest.approx(1.0)
    means = {k: v.value for k, v in values.items() if k.endswith("_animal_mean")}
    assert sum(means.values()) == pytest.approx(1.0)
    # later-earlier, the mean over animals: (0 + 2/3 + 1/2) / 3
    assert values[
        "timing_start_later_end_earlier_share_animal_mean"
    ].value == pytest.approx(7 / 18)
    assert values["timing_start_together_end_together_share"].value == 2 / 6
    assert values["timing_start_later_end_earlier_share"].value == 3 / 6
    assert values["timing_start_later_end_later_share"].value == 1 / 6
    # later-earlier: a 0, b 2/3, c 1/2
    assert values["timing_start_later_end_earlier_share_animal_min"].value == 0.0
    assert values[
        "timing_start_later_end_earlier_share_animal_max"
    ].value == pytest.approx(2 / 3)
    # the panel draws the animals' mean and range; the pooled share no figure
    for key, value in values.items():
        assert value.figures == (() if key.endswith("_share") else ("S2f",))
    assert {v.questions for v in values.values()} == {("2b.4",)}
    assert "more than 25 ms later than the matched SWR's start" in (
        values["timing_start_later_end_earlier_share"].description
    )


def _other_detector_events() -> pd.DataFrame:
    """Kay: three events of animal a overlap no non-local event (two mostly
    Local, one mostly No-Spike) and one of animal b overlaps one. The Widloski
    detector has three events, the other detectors two."""
    kay = [
        ("kay", "a", False, 0.8, 0.1, 0.0, 0.1),
        ("kay", "a", False, 0.9, 0.0, 0.0, 0.1),
        ("kay", "a", False, 0.2, 0.6, 0.0, 0.2),
        ("kay", "b", True, 0.0, 0.1, 0.9, 0.0),
    ]
    others = []
    for detector in builder.DETECTORS.values():
        if detector == "kay":
            continue
        others += [
            (detector, "a", True, 0.0, 0.0, 1.0, 0.0),
            (detector, "a", False, 0.8, 0.1, 0.0, 0.1),
        ]
    others.append(("memoryless", "a", False, 0.8, 0.1, 0.0, 0.1))
    return pd.DataFrame(
        kay + others, columns=["detector", "animal", "matched", *STATE_SHARE_COLUMNS]
    ).assign(day="d1", epoch="e1")


def test_nonlocal_found_values_are_figure_2es_right_bars():
    """For every detector Figure 2e draws, the share of the selected non-local
    events its events overlap, pooled over events, with the lowest and highest
    animal (an animal's share the mean over its epochs), from the function the
    figure draws with; tagged with question 2b.3 and panel 2e."""
    from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison

    detectors = (*nonlocal_comparison.FIGURE_DETECTORS, *builder.TEXT_FOUND_DETECTORS)
    rows = []
    for k, detector in enumerate(detectors):
        # animal a: 1 of 2 events matched in epoch 1, 1 of 1 in epoch 2 (0.75);
        # animal b: k % 2 of 1
        rows += [
            ("a", 1, detector, True), ("a", 1, detector, False),
            ("a", 2, detector, True), ("b", 1, detector, bool(k % 2)),
        ]  # fmt: skip
    memberships = pd.DataFrame(
        rows, columns=["animal", "day", "other_detector", "matched"]
    ).assign(epoch="02_r1")
    values = _by_key(builder.nonlocal_found_values(memberships, "s"))
    words = {detector: word for word, detector in builder.DETECTORS.items()}
    expected = {}
    for k, detector in enumerate(detectors):
        key = f"nonlocal_events_found_by_{words[detector]}_fraction"
        expected[key] = (2 + k % 2) / 4
        expected[f"{key}_animal_mean"] = (0.75 + k % 2) / 2
        expected[f"{key}_animal_min"] = min(0.75, k % 2)
        expected[f"{key}_animal_max"] = max(0.75, k % 2)
    assert {k: v.value for k, v in values.items()} == pytest.approx(expected)
    assert "nonlocal_events_found_by_widloski_fraction" in values
    # the criteria-only Widloski inventory is a value for the text, untagged,
    # and so is every pooled share: the panel draws the animals' mean
    text = "nonlocal_events_found_by_widloski_criteria_fraction"
    for key, value in values.items():
        assert value.questions == ("2b.3",)
        untagged = key.startswith(text) or key.endswith("_fraction")
        assert value.figures == (() if untagged else ("2e",))
        assert value.kind == "fraction"
        assert value.function == "nonlocal_found_values"


def test_detections_with_nonlocal_values_are_figure_2es_left_statistic():
    """The statistic of Figure 2e's left bars: for each detector the panel
    draws, the mean over animals of each animal's share of its events that
    overlap a non-local event and the lowest and highest animal, tagged to
    the panel; for the criteria-only Widloski inventory, also pooled, for the
    text."""
    from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison

    detectors = (*nonlocal_comparison.FIGURE_DETECTORS, "memoryless_criteria")
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b"] * len(detectors),
            "day": [1, 1, 2, 1] * len(detectors),
            "epoch": "02_r1",
            "detector": np.repeat(detectors, 4),
            "matched": [True, False, True, False] * len(detectors),
        }
    )
    values = _by_key(builder.detections_with_nonlocal_values(events, "s"))
    key = "widloski_criteria_with_nonlocal_fraction"
    # a: the mean of its epochs' 1/2 and 1; b: 0
    assert {k: v.value for k, v in values.items() if k.startswith(key)} == (
        pytest.approx(
            {
                key: 0.5,
                f"{key}_animal_mean": 0.375,
                f"{key}_animal_min": 0.0,
                f"{key}_animal_max": 0.75,
            }
        )
    )
    assert values["kay_with_nonlocal_fraction_animal_mean"].value == 0.375
    assert "kay_with_nonlocal_fraction" not in values
    for name, value in values.items():
        assert value.figures == (() if name.startswith(key) else ("2e",))


def test_other_detector_values_name_each_detector_and_each_state():
    values = _by_key(builder.other_detector_values(_other_detector_events(), "s"))
    numbers = {key: value.value for key, value in values.items()}
    kay = {
        key.removeprefix("kay_"): number
        for key, number in numbers.items()
        if key.startswith("kay_")
    }
    assert kay == {
        "events": 4,
        "no_nonlocal_fraction": 0.75,
        "no_nonlocal_fraction_animal_min": 0.0,
        "no_nonlocal_fraction_animal_max": 1.0,
        "no_nonlocal_fraction_animal_mean": 0.5,
        "mostly_local_fraction": 0.5,
        "mostly_local_fraction_animal_mean": pytest.approx(1 / 3),
        "no_nonlocal_mostly_local_fraction": pytest.approx(2 / 3),
        "no_nonlocal_mostly_no_spike_fraction": pytest.approx(1 / 3),
        "no_nonlocal_mixed_fraction": 0.0,
        "overlap_local_share_median": 0.0,
        "overlap_no_spike_share_median": 0.1,
        "overlap_nonlocal_share_median": 0.9,
        "overlap_uncertain_share_median": 0.0,
        "overlap_any_local_fraction": 0.0,
        "overlap_any_no_spike_fraction": 1.0,
        "overlap_all_nonlocal_fraction": 0.0,
    }
    # the Widloski detector's two inventories, under their words
    assert numbers["widloski_events"] == 3
    assert numbers["widloski_no_nonlocal_fraction"] == pytest.approx(2 / 3)
    assert numbers["widloski_criteria_events"] == 2
    assert numbers["widloski_criteria_no_nonlocal_fraction"] == 0.5
    assert numbers["yu_overlap_all_nonlocal_fraction"] == 1.0
    # only the main figure's detector has the per-animal range and its figure
    assert "yu_no_nonlocal_fraction_animal_min" not in numbers
    assert values["kay_events"].figures == ("2",)
    assert values["yu_events"].figures == ()
    # Figure 2c draws the main detector's animal means; its pooled shares no
    # figure
    assert values["kay_mostly_local_fraction_animal_mean"].figures == ("2c",)
    assert values["kay_no_nonlocal_fraction_animal_mean"].figures == ("2c",)
    assert values["kay_mostly_local_fraction"].figures == ()
    assert values["kay_no_nonlocal_fraction"].figures == ()


def test_left_out_values_name_each_covering_and_each_rate(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    numbers = _numbers(
        builder.left_out_values(
            tmp_path / "run", tmp_path / "inventory", SELECTED, VISITS, _exposure(), "s"
        )
    )
    assert numbers == {
        "rest_hours": 1.0,
        "nonlocal_time_s": 2.0,
        "kay_time_s": 0.5,
        "nonlocal_time_outside_kay_s": 1.5,
        "nonlocal_time_outside_kay_fraction": 0.75,
        "nonlocal_time_outside_swr_s": 1.25,
        "nonlocal_time_outside_swr_fraction": 0.625,
        "nonlocal_time_outside_trigger_s": 1.0,
        "nonlocal_time_outside_trigger_fraction": 0.5,
        "nonlocal_time_outside_widloski_s": 1.75,
        "nonlocal_time_outside_widloski_fraction": 0.875,
        "nonlocal_events_outside_kay": 1,
        "nonlocal_events_outside_kay_fraction": 0.5,
        "nonlocal_events_outside_kay_per_hour": 1.0,
        "nonlocal_events_outside_widloski": 2,
        "nonlocal_events_outside_widloski_fraction": 1.0,
        "nonlocal_events_outside_widloski_per_hour": 2.0,
        "nonlocal_events_outside_trigger": 4,
        "nonlocal_events_outside_trigger_fraction": pytest.approx(4 / 7),
        "nonlocal_events_outside_trigger_per_hour": 4.0,
    }


def _signal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One animal, one epoch, two non-SWR and two SWR events. Signal number k
    (theta 0, ripple 1, MUA 2, slow gamma 3, fast gamma 4) has non-SWR values
    10k + 1 and 10k + 2, SWR values 10k + 3 and 10k + 7, a running level of
    10k + 4 and a quiet-rest level of 10k + 1.8. In windows as long as each
    class's events the levels are higher, the more so for the longer SWR
    events: rest 10k + 1.2 (non-SWR) and 10k + 5.5 (SWR), running 10k + 4.5 and
    10k + 4.8."""
    events = pd.DataFrame(
        {
            "animal": "a",
            "session_date": "d1",
            "epoch": "e1",
            "is_swr": [False, False, True, True],
        }
    )
    reference = pd.DataFrame({"animal": ["a"], "session_date": "d1", "epoch": "e1"})
    for k, (signal, column) in enumerate(builder._SIGNAL_EVENT_COLUMNS.items()):
        events[column] = 10.0 * k + np.array([1.0, 2.0, 3.0, 7.0])
        reference[f"{signal}_running_median"] = 10.0 * k + 4.0
        reference[f"{signal}_rest_median"] = 10.0 * k + 1.8
        reference[f"{signal}_rest_matched_non_swr"] = 10.0 * k + 1.2
        reference[f"{signal}_rest_matched_swr"] = 10.0 * k + 5.5
        reference[f"{signal}_running_matched_non_swr"] = 10.0 * k + 4.5
        reference[f"{signal}_running_matched_swr"] = 10.0 * k + 4.8
    return events, reference


def test_brain_state_values_name_each_signal_class_and_reference():
    values = _by_key(
        builder.brain_state_values(*_signal_tables(), "s")
        + builder.gamma_z_values(*_signal_tables(), "s")
    )
    numbers = {key: value.value for key, value in values.items()}
    for k, signal in enumerate(builder._SIGNAL_EVENT_COLUMNS):
        own = {
            key.removeprefix(f"{signal}_"): number
            for key, number in numbers.items()
            if key.startswith(f"{signal}_")
        }
        expected = {
            "running_reference": 10 * k + 4.0,
            "rest_reference": 10 * k + 1.8,
            "median_non_swr": 10 * k + 1.5,
            "median_swr": 10 * k + 5.0,
            # non-SWR: 1.5 is below running (4) and not above rest (1.8)
            "animals_below_running_non_swr": 1,
            "animals_above_rest_non_swr": 0,
            # SWR: 5 is not below running and is above rest
            "animals_below_running_swr": 0,
            "animals_above_rest_swr": 1,
        }
        if signal == "theta":
            # of the SWR events, 7 reaches the running level and 3 does not
            expected |= {"like_fraction_non_swr": 0.0, "like_fraction_swr": 0.5}
        if signal.endswith("gamma"):
            # Gamma is compared with windows as long as each class's events.
            expected = {
                "median_non_swr": 10 * k + 1.5,
                "median_swr": 10 * k + 5.0,
                "rest_matched_reference_non_swr": 10 * k + 1.2,
                "rest_matched_reference_swr": 10 * k + 5.5,
                "running_matched_reference_non_swr": 10 * k + 4.5,
                "running_matched_reference_swr": 10 * k + 4.8,
                # non-SWR: 1.5 is above its windows' rest (1.2), below running (4.5)
                "animals_above_rest_non_swr": 1,
                "animals_below_running_non_swr": 1,
                # SWR: 5 is not above its windows' rest (5.5), though it is above
                # the median of single rest bins (1.8); nor below running (4.8)
                "animals_above_rest_swr": 0,
                "animals_below_running_swr": 0,
            }
        assert own == pytest.approx(expected), signal
    assert values["ripple_median_swr"].figures == ("2g",)
    assert values["mua_median_swr"].figures == ("2g",)
    assert values["theta_median_swr"].figures == ("2g",)


def test_gamma_db_values_name_each_band_and_class():
    reference = pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "slow_gamma_db_non_swr": [1.0, 3.0, -1.0],
            "slow_gamma_db_swr": [2.0, 4.0, 6.0],
            "fast_gamma_db_non_swr": [-3.0, -1.0, -2.0],
            "fast_gamma_db_swr": [0.5, 0.7, -0.2],
        }
    )
    assert _numbers(builder.gamma_db_values(reference, "s")) == {
        "slow_gamma_db_median_non_swr": 1.0,
        "slow_gamma_db_animals_above_rest_non_swr": 1,
        "slow_gamma_db_median_swr": 4.0,
        "slow_gamma_db_animals_above_rest_swr": 2,
        "fast_gamma_db_median_non_swr": -2.0,
        "fast_gamma_db_animals_above_rest_non_swr": 0,
        "fast_gamma_db_median_swr": 0.5,
        "fast_gamma_db_animals_above_rest_swr": 1,
    }


def _spike_metrics() -> pd.DataFrame:
    rows = [
        # group, animal, epoch, spikes, tetrodes, dll per spike, GPI distance
        ("baseline", "a", "a1", 10, 2, 0.0, 5.0),
        ("baseline", "a", "a2", 15, 2, 0.2, 5.0),
        ("baseline", "b", "b1", 20, 4, 1.0, 7.0),
        ("SWR-remote", "a", "a1", 30, 6, 2.0, 50.0),
        ("SWR-remote", "a", "a2", 35, 6, 2.2, 55.0),
        ("SWR-remote", "b", "b1", 40, 8, 3.0, 70.0),
        ("nonSWR-remote", "a", "a1", 12, 3, 0.5, 40.0),
        ("nonSWR-remote", "b", "b1", 14, 5, 0.5, 44.0),
        ("nonSWR-close", "a", "a1", 11, 1, -1.0, 4.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "group",
            "animal",
            "epoch_id",
            "n_spikes",
            "n_tetrodes",
            "dll_per_spike",
            "dist_to_gpi_peak",
        ],
    )


def test_spike_evidence_values_name_each_group_and_each_count():
    values = _by_key(builder.spike_evidence_values(_spike_metrics(), "s"))
    numbers = {key: value.value for key, value in values.items()}
    assert numbers == {
        "evidence_epochs": 3,
        "evidence_animals": 2,
        "baseline_scored": 3,
        "baseline_spikes_median": 15.0,
        "baseline_tetrodes_median": 2.0,
        "baseline_dll_per_spike_median": 0.2,
        "baseline_gpi_distance_median_cm": 5.0,
        "swr_remote_scored": 3,
        "swr_remote_spikes_median": 35.0,
        "swr_remote_tetrodes_median": 6.0,
        "swr_remote_dll_per_spike_median": 2.2,
        "swr_remote_gpi_distance_median_cm": 55.0,
        "swr_remote_epochs_above_baseline": 3,
        "swr_remote_epochs_scored": 3,
        "swr_remote_animals_above_baseline": 2,
        "non_swr_remote_scored": 2,
        "non_swr_remote_spikes_median": 13.0,
        "non_swr_remote_tetrodes_median": 4.0,
        "non_swr_remote_dll_per_spike_median": 0.5,
        "non_swr_remote_gpi_distance_median_cm": 42.0,
        # a1: 0.5 against 0.0; b1: 0.5 against 1.0. a: 0.5 against 0.1
        "non_swr_remote_epochs_above_baseline": 1,
        "non_swr_remote_epochs_scored": 2,
        "non_swr_remote_animals_above_baseline": 1,
        "non_swr_close_scored": 1,
        "non_swr_close_spikes_median": 11.0,
        "non_swr_close_tetrodes_median": 1.0,
        "non_swr_close_dll_per_spike_median": -1.0,
        "non_swr_close_gpi_distance_median_cm": 4.0,
        "non_swr_close_epochs_above_baseline": 0,
        "non_swr_close_epochs_scored": 1,
        "non_swr_close_animals_above_baseline": 0,
    }
    # the quiet windows are question 2c.3's negative control; close and remote
    # events answer it alike
    assert values["baseline_dll_per_spike_median"].questions == ("2c.3",)
    assert values["non_swr_close_dll_per_spike_median"].questions == ("2c.3",)
    assert values["non_swr_remote_dll_per_spike_median"].questions == ("2c.3",)
    assert values["non_swr_close_spikes_median"].questions == ("2d.9",)
    assert values["non_swr_close_gpi_distance_median_cm"].questions == ("2c.4",)


def test_heldout_by_swr_values_read_the_immobile_per_spike_rows():
    stats = pd.DataFrame(
        [
            ("Non-Local Continuous | SWR", "immobile", "runcv_nl_anim_mark_ps")
            + (0.6, 0.4, 0.8, "5/5"),
            ("Non-Local Continuous | no SWR", "immobile", "runcv_nl_anim_mark_ps")
            + (0.2, 0.1, 0.3, "4/6"),
            # another regime and another metric of the same states
            ("Non-Local Continuous | SWR", "moving", "runcv_nl_anim_mark_ps")
            + (9.0, 9.0, 9.0, "1/1"),
            ("Non-Local Continuous | SWR", "immobile", "runcv_nl_anim_mark")
            + (19.0, 19.0, 19.0, "1/1"),
        ],
        columns=[
            "state",
            "regime",
            "metric",
            "animal_estimate",
            "animal_ci_lo",
            "animal_ci_hi",
            "animals_pos",
        ],
    )
    assert _numbers(builder.heldout_by_swr_values(stats, "s")) == {
        "heldout_swr_per_spike": 0.6,
        "heldout_swr_per_spike_ci_low": 0.4,
        "heldout_swr_per_spike_ci_high": 0.8,
        "heldout_swr_animals_positive": 5,
        "heldout_swr_animals": 5,
        "heldout_non_swr_per_spike": 0.2,
        "heldout_non_swr_per_spike_ci_low": 0.1,
        "heldout_non_swr_per_spike_ci_high": 0.3,
        "heldout_non_swr_animals_positive": 4,
        "heldout_non_swr_animals": 6,
    }


def test_class_distribution_values_count_the_animals_with_one_mode():
    animals = list(analyzed_animals())
    distribution = pd.DataFrame(
        {
            "animal": [*animals, "pooled", animals[0]],
            "power": ["peak"] * len(animals) + ["peak", "mean"],
            "dip_p": [0.5, 0.6, 0.01, 0.9, 0.04, 0.9, 0.9],
            "main_mode_z": [1.0, 2.0, 0.9, 1.5, 1.2, 9.0, -9.0],
        }
    )
    assert _numbers(builder.class_distribution_values(distribution, "s")) == {
        "ripple_peak_animals_unimodal": 3,
        "ripple_peak_mode_animal_min": 0.9,
        "ripple_peak_mode_animal_max": 2.0,
    }
    with pytest.raises(ValueError, match="one row for each of 5 animals"):
        builder.class_distribution_values(distribution.iloc[1:], "s")


def _signal_counts() -> pd.DataFrame:
    """Each analyzed animal (i = 0 to 4), one epoch: by ripple power, 100 bins
    below 2 z of which i + 1 non-local and 100 above of which 9 - i; by MUA,
    100 bins below 2 z of which 2 non-local and 10 above of which i + 1."""
    rows = []
    for i, animal in enumerate(analyzed_animals()):
        for signal, cells in (
            ("ripple_power", ((-np.inf, 2.0, 100, i + 1), (2.0, np.inf, 100, 9 - i))),
            ("mua", ((-np.inf, 2.0, 100, 2), (2.0, np.inf, 10, i + 1))),
        ):
            for low, high, n_bins, n_nonlocal in cells:
                rows += [
                    (f"{animal}_e", animal, signal, low, high, 0.0, 0.2, n_bins - n_nonlocal),
                    (f"{animal}_e", animal, signal, low, high, 0.8, 1.0, n_nonlocal),
                ]  # fmt: skip
    return pd.DataFrame(
        rows,
        columns=[
            "epoch_id", "animal", "signal", "signal_low", "signal_high",
            "probability_low", "probability_high", "n_bins",
        ],
    )  # fmt: skip


def test_nonlocal_power_values_give_the_share_of_nonlocal_time_below_the_threshold():
    assert _numbers(builder.nonlocal_power_values(_signal_counts(), "s")) == {
        "nonlocal_time_below_swr_threshold_fraction": pytest.approx(0.3),
        "nonlocal_time_below_swr_threshold_fraction_animal_min": pytest.approx(0.1),
        "nonlocal_time_below_swr_threshold_fraction_animal_max": pytest.approx(0.5),
    }


def test_nonlocal_mua_values_give_the_share_above_the_level_and_below_it():
    """Above 2 z of MUA, (i + 1) / 10 of the bins are non-local; of the
    non-local bins, 2 / (2 + i + 1) have MUA below 2 z. Tagged with the MUA
    panel of Figure 2's supplement."""
    values = _by_key(builder.nonlocal_mua_values(_signal_counts(), "s"))
    below = [2 / (3 + i) for i in range(5)]
    assert {k: v.value for k, v in values.items()} == {
        "nonlocal_share_mua_above_level": pytest.approx(0.3),
        "nonlocal_share_mua_above_level_animal_min": pytest.approx(0.1),
        "nonlocal_share_mua_above_level_animal_max": pytest.approx(0.5),
        "nonlocal_time_below_mua_level_fraction": pytest.approx(np.mean(below)),
        "nonlocal_time_below_mua_level_fraction_animal_min": pytest.approx(min(below)),
        "nonlocal_time_below_mua_level_fraction_animal_max": pytest.approx(max(below)),
        "mua_level_z": 2.0,
    }
    assert {v.figures for v in values.values()} == {
        builder.TAGS["nonlocal_mua"].figures
    }
    assert "2 z" in values["nonlocal_share_mua_above_level"].description


def test_rule_sweep_values_name_each_setting_and_value_and_the_criterion_check():
    """Every swept value's share with no SWR (pooled, lowest and highest animal)
    and events (against the rule's and counted), from
    ``rule_sweep.sweep_statistics``; the
    rule's own share; the criterion's agreement. Question 2a.5, the sweep
    panel."""
    from continuum_swr_replay.analysis.event_properties import rule_sweep
    from tests.analysis.event_properties.test_rule_sweep import _counts

    values = _by_key(builder.rule_sweep_values(_counts(), "s"))
    stats = rule_sweep.sweep_statistics(_counts()).set_index(["setting", "value"])
    key = "rule_sweep_probability_threshold_point_nine"
    assert values[f"{key}_no_swr_fraction"].value == pytest.approx(
        stats.loc[("probability_threshold", 0.9), "no_swr_fraction"]
    )
    assert values[f"{key}_events_vs_rule"].value == pytest.approx(30 / 40)
    assert values[f"{key}_events"].value == 30
    assert values[
        "rule_sweep_minimum_samples_twenty_five_no_swr_fraction_animal_max"
    ].value == (pytest.approx(10 / 16))
    assert values[
        "rule_sweep_swr_threshold_two_no_swr_fraction"
    ].value == pytest.approx(0.5)
    assert values["rule_sweep_swr_criterion_agreement"].value == pytest.approx(37 / 40)
    n_values = sum(len(v) for v in rule_sweep.SETTINGS.values())
    assert values[
        "rule_sweep_minimum_samples_twenty_five_no_swr_fraction_animal_mean"
    ].value == pytest.approx((0.5 + 10 / 16) / 2)
    # per value: the share, the animals' mean, its two animal bounds, events
    # against the rule's and the events
    assert len(values) == 6 * n_values + 1
    assert not any(c.isdigit() for k in values for c in k)
    for key, value in values.items():
        assert value.questions == ("2a.5",)
        pooled = key.endswith("_no_swr_fraction")
        assert value.figures == (() if pooled else builder.TAGS["rule_sweep"].figures)


def _state_calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The in-visit calibration's bins and trends: each state's estimate k + bin
    index (k = 0, 1, 2 for Local, Continuous, Fragmented), intervals 0.5 either
    side, 100 windows a bin over 242 epochs; slopes 1, 2, 3 with intervals 0.4
    either side."""
    from continuum_swr_replay.analysis.event_evidence.state_evidence import (
        IN_VISIT_CALIBRATION_METRICS,
        PROBABILITY_BIN_LABELS,
    )

    bins, trends = [], []
    for k, (state, (metric,)) in enumerate(IN_VISIT_CALIBRATION_METRICS.items()):
        for j, label in enumerate(PROBABILITY_BIN_LABELS):
            bins.append(
                {
                    "regime": "immobile", "state": state, "metric": metric,
                    "pbin": label, "n": 100, "n_epochs": 242,
                    "animal_estimate": k + j, "animal_ci_lo": k + j - 0.5,
                    "animal_ci_hi": k + j + 0.5,
                }
            )  # fmt: skip
        trends.append(
            {
                "regime": "immobile", "state": state, "metric": metric,
                "slope_animal_estimate": k + 1.0,
                "slope_animal_ci_lo": k + 0.6, "slope_animal_ci_hi": k + 1.4,
            }
        )  # fmt: skip
    return pd.DataFrame(bins), pd.DataFrame(trends)


def test_state_calibration_values_name_each_state_and_state_the_caveats():
    bins, trends = _state_calibration()
    values = _by_key(builder.state_calibration_values(bins, trends, "s"))
    numbers = {k: v.value for k, v in values.items()}
    assert numbers["state_calibration_local_top_bin_per_spike"] == 4
    assert numbers[
        "state_calibration_continuous_top_bin_per_spike_ci_low"
    ] == pytest.approx(4.5)
    assert numbers["state_calibration_fragmented_bottom_bin_per_spike"] == 2
    assert numbers["state_calibration_fragmented_slope_per_spike"] == 3.0
    assert numbers["state_calibration_local_slope_per_spike_ci_high"] == pytest.approx(
        1.4
    )
    assert numbers["state_calibration_windows"] == 500
    assert not any(c.isdigit() for k in values for c in k)
    description = values["state_calibration_continuous_top_bin_per_spike"].description
    for caveat in ("every tetrode", "only the remote place", "other time blocks"):
        assert caveat in description
    for value in values.values():
        assert value.questions == ("2a.7",)
        assert value.figures == builder.TAGS["state_calibration"].figures


def step_fits() -> pd.DataFrame:
    """Step fits of every analyzed animal at the main settings: 1 animal with a
    step in duration, 3 in spatial extent and in replay speed, 2 in spike
    evidence, none in the other properties; spike evidence's shares are
    0.5, -0.1, 0.3, 0.2, 0.9 (median 0.3, 4 positive)."""
    animals = list(analyzed_animals())
    with_step = {"duration": 1, "extent": 3, "speed": 3, "evidence": 2}
    evidence_shares = [0.5, -0.1, 0.3, 0.2, 0.9]
    rows = []
    for prop in [*builder.OTHER_STEP_PROPERTIES, "evidence"]:
        for i, animal in enumerate(animals):
            share = evidence_shares[i] if prop == "evidence" else -1.0
            rows.append((prop, animal, i < with_step.get(prop, 0), share))
    return pd.DataFrame(
        rows, columns=["property", "animal", "has_step", "step_share_overall"]
    ).assign(
        n_knots=builder.STEP_KNOTS,
        power="criterion",
        window_z=builder.STEP_WINDOW_Z,
        covariate="none",
    )


def test_class_step_values_name_the_properties_with_the_most_steps():
    animals = list(analyzed_animals())
    steps = step_fits()
    values = _by_key(builder.class_step_values(steps, "s"))
    assert {key: value.value for key, value in values.items()} == {
        "step_other_properties_criterion_animals_max": 3,
    }
    description = values["step_other_properties_criterion_animals_max"].description
    assert description.endswith("attained by spatial extent, replay speed")
    # spike evidence's step has its own builder: it alone reads the spike-evidence run
    evidence = _by_key(builder.class_step_evidence_values(steps, "s"))
    assert {key: value.value for key, value in evidence.items()} == {
        "step_evidence_criterion_animals": 2,
        "step_evidence_criterion_animals_same_sign": 4,
        "step_evidence_criterion_share_median": 0.3,
    }
    assert {value.function for value in evidence.values()} == {
        "class_step_evidence_values"
    }
    for build in (builder.class_step_values, builder.class_step_evidence_values):
        with pytest.raises(ValueError, match="one fit for each of 5 animals"):
            build(steps[steps.animal != animals[0]], "s")


def _property_events() -> pd.DataFrame:
    """One animal, two non-SWR events and one SWR event. Property number i has
    non-SWR values i + 1 and i + 3 and the SWR value i + 10."""
    events = pd.DataFrame(
        {
            "animal": "a",
            "session_date": "d1",
            "epoch": "e1",
            "is_swr": [False, False, True],
        }
    )
    for i, (_, column, *_) in enumerate(builder._PROPERTIES):
        events[column] = i + np.array([1.0, 3.0, 10.0])
    return events


def test_property_values_name_each_property_class_and_direction():
    values = _by_key(
        builder.property_values(_property_events(), "s")
        + builder.start_distance_values(_property_events(), "s")
    )
    assert len(values) == 3 * len(builder._PROPERTIES)
    for i, (word, _, scale, unit, _, direction, _) in enumerate(builder._PROPERTIES):
        assert values[f"{word}_median_non_swr"].value == scale * (i + 2.0), word
        assert values[f"{word}_median_swr"].value == scale * (i + 10.0), word
        assert values[f"{word}_median_swr"].unit == unit
        # the non-SWR median is the lower one in every property
        assert values[f"{word}_animals_in_direction"].value == (
            1 if direction == "lower" else 0
        ), word
    # durations are reported in milliseconds
    assert values["duration_median_non_swr"].value == 2000.0
    assert values["peak_animals_in_direction"].value == 0
    assert values["extent_animals_in_direction"].value == 1


def test_content_distance_values_are_each_class_share_by_distance_band():
    """Per class and band (under 20 cm, 20 to 100 cm, 100 cm or more): the
    share pooled over events with no figure, and the mean over animals of each
    animal's share with the lowest and highest animal, tagged to the
    supplement's distance panel; and the two cuts."""
    from tests.analysis.event_properties.test_event_statistics import (
        _distance_events,
    )

    values = _by_key(builder.content_distance_values(_distance_events(), "s"))
    assert len(values) == 2 * 3 * 4 + 2
    key = "content_distance_under_non_swr_fraction"
    assert values[key].value == pytest.approx(1 / 6)
    assert values[f"{key}_animal_mean"].value == pytest.approx(1 / 12)
    assert values[f"{key}_animal_max"].value == pytest.approx(1 / 6)
    assert values["content_distance_beyond_swr_fraction_animal_mean"].value == (
        pytest.approx(3 / 8)
    )
    assert values["content_distance_between_swr_fraction_animal_min"].value == 0.0
    assert values["content_distance_near_cm"].value == 20.0
    assert values["content_distance_remote_cm"].value == 100.0
    for name, value in values.items():
        assert value.questions == ("2d.1",)
        pooled = name.endswith("_fraction")
        assert value.figures == (() if pooled else ("S3c",)), name
    assert "under 20 cm from the animal" in values[key].description


def test_hpd_tight_values_are_the_share_below_the_cut():
    """The share of each class's events below the cut with its interval, the
    animals whose no-SWR share is higher and the cut, from the panel's
    function; tagged to the panel, while the pooled hpd_median_* values carry
    no figure tag."""
    # three animals of one epoch each, alike: no-SWR 10 and 300 cm (half
    # tight), SWR 10, 300, 400, 400 (a quarter)
    events = pd.DataFrame(
        {
            "animal": np.repeat(["a", "b", "c"], 6),
            "session_date": 1,
            "epoch": "02_r1",
            "is_swr": np.tile([False, False, True, True, True, True], 3),
            "hpd_coverage_cm": np.tile([10.0, 300.0, 10.0, 300.0, 400.0, 400.0], 3),
        }
    )
    values = _by_key(builder.hpd_tight_values(events, "s"))
    numbers = {k: v.value for k, v in values.items()}
    assert numbers["hpd_tight_share_non_swr"] == pytest.approx(0.5)
    assert numbers["hpd_tight_share_swr"] == pytest.approx(0.25)
    assert numbers["hpd_tight_share_animals_non_swr_higher"] == 3
    assert numbers["hpd_tight_cut_cm"] == 50.0
    assert set(numbers) == {
        f"hpd_tight_share_{w}{end}"
        for w in ("non_swr", "swr")
        for end in ("", "_ci_low", "_ci_high")
    } | {"hpd_tight_share_animals_non_swr_higher", "hpd_tight_cut_cm"}
    assert "95 % of the decoded probability" in (
        values["hpd_tight_share_swr"].description
    )
    for value in values.values():
        assert value.questions == ("2d.5",)
        assert value.figures == builder.TAGS["hpd"].figures
    (hpd,) = [p for p in builder._PROPERTIES if p[0] == "hpd"]
    assert hpd[6].figures == ()


def test_property_interval_values_name_each_bound():
    rows = {}
    for i, (word, *_) in enumerate(builder._PROPERTIES):
        rows[word] = {
            "difference_low": i + 0.1,
            "difference_high": i + 0.2,
            "matched_difference": i + 0.3,
            "matched_difference_low": i + 0.4,
            "matched_difference_high": i + 0.5,
            "matched_difference_nan_replicates": 0,
            "n_replicates": 1000,
            "matched_replicates_skipping_strata": 0,
            "n_animals": 5,
        }
    intervals = pd.DataFrame.from_dict(rows, orient="index")
    numbers = _numbers(builder.property_interval_values(intervals, "s"))
    # extent is the second property
    assert {k: v for k, v in numbers.items() if k.startswith("extent_")} == {
        "extent_difference_low": 1.1,
        "extent_difference_high": 1.2,
        "extent_matched_difference": 1.3,
        "extent_matched_difference_low": 1.4,
        "extent_matched_difference_high": 1.5,
    }
    # duration is not matched on itself
    assert {k: v for k, v in numbers.items() if k.startswith("duration_")} == {
        "duration_difference_low": 0.1,
        "duration_difference_high": 0.2,
    }


def test_rest_theta_values_name_each_class():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "a", "b"],
            "session_date": "d1",
            "epoch": "e1",
            "is_swr": [False, False, False, True, True],
            "theta_z": [2.0, 0.0, 0.0, 5.0, 5.0],
            "theta_running_median": 1.0,
            "rest_windows": 10,
            "rest_theta_like_share": [0.2, 0.4, 0.1, 0.3, 0.9],
        }
    )
    assert _numbers(builder.rest_theta_values(events, "s")) == {
        "theta_like_fraction_rest_matched_non_swr": pytest.approx(0.7 / 3),
        # a: half its events against 0.3 of its windows; b: none against 0.1
        "theta_like_animals_above_rest_matched_non_swr": 1,
        "theta_like_fraction_rest_matched_swr": pytest.approx(0.6),
        "theta_like_animals_above_rest_matched_swr": 2,
    }


def test_figure_3_values_are_tagged_to_the_panel_that_shows_them():
    """Spatial extent is panel c. The arm count answers the same question and is
    in no panel. The share of content at a well is quoted on panel f; the
    shares that depend on where the animal is (own and remote wells, remote
    paths, near the animal) are in no panel."""
    values = _by_key(builder.property_values(_property_events(), "s"))
    assert values["extent_median_swr"].figures == ("3c",)
    assert values["arms_median_swr"].figures == ()
    assert values["arms_median_swr"].questions == values["extent_median_swr"].questions
    assert builder.TAGS["content_occupancy"].figures == ("3f",)
    assert builder.TAGS["content_by_animal"].figures == ()
    assert (
        builder.TAGS["content_by_animal"].questions
        == builder.TAGS["content_occupancy"].questions
    )


def test_content_occupancy_values_name_each_group_and_place():
    # Remote: over 20 cm from the animal; at a well: under 20 cm from one.
    # The track: a leaf arm (0) and a trunk arm (6), each of 25 2-cm place bins.
    def rows(animal, epoch, is_swr, d_animal, d_well, arm=0):
        return pd.DataFrame(
            {
                "event_id": 0,
                "animal": animal,
                "epoch_id": epoch,
                "is_swr": is_swr,
                "distance_to_animal_cm": d_animal,
                "distance_to_well_cm": d_well,
                "arm": arm,
                "arm_length_cm": 48.0,
                "n_place_bins": 25,
            }
        )

    bins = pd.concat(
        [
            # a, no SWR: e1 a remote well; e2 its own well and a remote path:
            # epoch shares remote well 1, 0; own well 0, 0.5; remote path 0, 0.5.
            rows("a", "e1", False, [100.0], [5.0]),
            rows("a", "e2", False, [5.0, 100.0], [5.0, 50.0]),
            # a, SWR: a remote path. b, no SWR: a remote well; b, SWR: near the
            # animal off the wells.
            rows("a", "e1", True, [100.0], [50.0], arm=6),
            rows("b", "e3", False, [100.0], [5.0]),
            rows("b", "e3", True, [5.0], [50.0], arm=6),
        ],
        ignore_index=True,
    )
    values = builder.content_occupancy_values(bins, "s")
    shown = {value.key for value in values if value.figures}
    assert shown == {
        "occupancy_epochs",
        "non_swr_content_at_well_fraction",
        "swr_content_at_well_fraction",
        "animals_non_swr_more_at_well",
        "even_content_at_well_fraction",
    }
    assert _numbers(values) == {
        "occupancy_epochs": 3,
        # no SWR: a (0.5, 0.25, 0.25, 0), b (1, 0, 0, 0); the animals' mean
        "non_swr_content_remote_well_fraction": pytest.approx(0.75),
        "non_swr_content_own_well_fraction": pytest.approx(0.125),
        "non_swr_content_at_well_fraction": pytest.approx(0.875),
        "non_swr_content_remote_path_fraction": pytest.approx(0.125),
        "non_swr_content_near_off_fraction": pytest.approx(0.0),
        # SWR: a on a remote path, b near the animal off the wells
        "swr_content_remote_well_fraction": pytest.approx(0.0),
        "swr_content_own_well_fraction": pytest.approx(0.0),
        "swr_content_at_well_fraction": pytest.approx(0.0),
        "swr_content_remote_path_fraction": pytest.approx(0.5),
        "swr_content_near_off_fraction": pytest.approx(0.5),
        "animals_non_swr_more_at_well": 2,
        # 20 cm of the 50 cm leaf arm, of each animal's 100 cm track
        "even_content_at_well_fraction": pytest.approx(0.2),
    }


def test_the_values_read_from_another_output_have_a_builder_of_their_own():
    """The summary pends the values of a stale output by their builder: the
    gamma pass's are ``gamma_z_values``', the start-distance output's
    ``start_distance_values``'; the event table's are in neither."""
    gamma = _by_key(builder.gamma_z_values(*_signal_tables(), "s"))
    assert len(gamma) == 20
    assert {key.split("_")[0] for key in gamma} == {"slow", "fast"}
    assert {v.function for v in gamma.values()} == {"gamma_z_values"}
    brain_state = _by_key(builder.brain_state_values(*_signal_tables(), "s"))
    assert {key.split("_")[0] for key in brain_state} == {"theta", "ripple", "mua"}

    start = _by_key(builder.start_distance_values(_property_events(), "s"))
    assert set(start) == {
        "start_distance_median_non_swr",
        "start_distance_median_swr",
        "start_distance_animals_in_direction",
    }
    assert {v.function for v in start.values()} == {"start_distance_values"}
    properties = builder.property_values(_property_events(), "s")
    assert not {v.key for v in properties} & set(start)


def test_spike_evidence_metrics_read_every_epoch_and_refuse_a_missing_one(tmp_path):
    for epoch, group in (("a1", "baseline"), ("b1", "SWR-remote")):
        (tmp_path / epoch).mkdir()
        pd.DataFrame({"epoch_id": [epoch, epoch], "group": group}).to_parquet(
            tmp_path / epoch / SPIKE_EVIDENCE_METRICS
        )
    metrics = builder.spike_evidence_metrics(tmp_path, n_epochs=2)
    assert list(metrics["epoch_id"]) == ["a1", "a1", "b1", "b1"]
    with pytest.raises(ValueError, match="holds 2 epochs' metrics, not 3"):
        builder.spike_evidence_metrics(tmp_path, n_epochs=3)
    # a table that names another epoch than its folder's
    pd.DataFrame({"epoch_id": ["a1"], "group": "baseline"}).to_parquet(
        tmp_path / "b1" / SPIKE_EVIDENCE_METRICS
    )
    with pytest.raises(ValueError, match="holds epochs \\['a1'\\]"):
        builder.spike_evidence_metrics(tmp_path, n_epochs=2)


def _windows(rows) -> pd.DataFrame:
    """One epoch per animal: (animal, group, kind, participation, spike rate)."""
    return pd.DataFrame(
        rows, columns=["animal", "group", "kind", "participation", "spike_rate_hz"]
    ).assign(epoch_id=lambda t: t["animal"] + "1")


def test_participation_values_name_each_group_window_and_measure():
    table = _windows(
        [
            ("a", "no SWR", "event", 0.6, 300.0),
            ("a", "no SWR", "control", 0.3, 100.0),
            ("b", "no SWR", "event", 0.4, 200.0),
            ("b", "no SWR", "control", 0.5, 100.0),
            ("a", "SWR", "event", 0.8, 900.0),
            ("a", "SWR", "control", 0.3, 100.0),
            ("b", "SWR", "event", 0.6, 700.0),
            ("b", "SWR", "control", 0.4, 100.0),
            # an event with no control
            ("b", "SWR", "event", 0.6, 700.0),
        ]
    )
    numbers = _numbers(builder.participation_values(table, "s"))
    assert numbers["non_swr_event_participation"] == pytest.approx(0.5)
    assert numbers["non_swr_control_participation"] == pytest.approx(0.4)
    assert numbers["non_swr_participation_animals_above_rest"] == 1
    assert numbers["non_swr_spike_rate_hz_animals_above_rest"] == 2
    assert numbers["swr_event_spike_rate_hz"] == pytest.approx(800.0)
    assert numbers["swr_participation_animals"] == 2
    assert numbers["participation_events_without_control"] == 1


def test_likelihood_advantage_values_are_the_animal_mean_and_its_interval():
    table = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c", "a", "b", "c"],
            "epoch_id": ["a1", "a1", "b1", "b1", "c1", "a1", "b1", "c1"],
            "group": ["no SWR"] * 5 + ["SWR"] * 3,
            "D_spike": [0.2, 0.4, 0.5, np.nan, 0.9, 1.0, 1.2, 1.4],
        }
    )
    numbers = _numbers(builder.likelihood_advantage_values(table, "s"))
    # no SWR: a 0.3, b 0.5 (its NaN unscored), c 0.9
    assert numbers["non_swr_likelihood_advantage_per_spike"] == pytest.approx(
        (0.3 + 0.5 + 0.9) / 3
    )
    assert numbers["non_swr_likelihood_advantage_events"] == 5
    assert numbers["non_swr_likelihood_advantage_events_scored"] == 4
    assert numbers["swr_likelihood_advantage_per_spike"] == pytest.approx(1.2)
    assert (
        numbers["swr_likelihood_advantage_per_spike_ci_low"]
        < 1.2
        < numbers["swr_likelihood_advantage_per_spike_ci_high"]
    )


def test_state_composition_values_name_each_group_and_state():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "session_date": ["d1"] * 4,
            "epoch": ["02_r1"] * 4,
            "is_swr": [False, True, False, True],
            "continuous_prob_mean": [0.9, 0.6, 0.7, 0.8],
            "fragmented_prob_mean": [0.05, 0.3, 0.2, 0.1],
        }
    )
    values = _by_key(builder.state_composition_values(events, "s"))
    assert values["non_swr_continuous_share"].value == pytest.approx(0.8)
    assert values["swr_fragmented_share"].value == pytest.approx(0.2)
    assert values["swr_fragmented_share"].figures == ("3b",)
    assert len(values) == 4


def test_tetrode_holdout_values_name_each_estimate_and_group():
    rows = []
    for animal, epoch, gain in (("a", "a1", 0.4), ("b", "b1", 0.6), ("c", "c1", 0.8)):
        for group in ("no SWR", "SWR"):
            for kind, value in (("event", gain), ("control", 0.0)):
                rows.append(
                    {
                        "animal": animal,
                        "epoch_id": epoch,
                        "group": group,
                        "kind": kind,
                        "gain_per_spike": value,
                        "spiking_gain_per_spike": value + 0.1,
                        "silent_gain_per_spike": -0.1,
                        "agreement_per_spike": value / 2,
                    }
                )
    values = builder.tetrode_holdout_values(pd.DataFrame(rows), "s")
    numbers = _numbers(values)
    by_key = _by_key(values)
    # Figure 2i and the supplement's panel S2f draw the events' gain, S2f
    # alone its parts in the bins with and without a spike; no figure draws
    # the controls' gain or the agreement
    agreement = {k for k in by_key if "agreement" in k or "control" in k}
    assert {by_key[k].figures for k in agreement} == {()}
    parts = {k for k in by_key if "spiking" in k or "silent" in k}
    assert {by_key[k].figures for k in parts} == {("S2h",)}
    assert {v.figures for k, v in by_key.items() if k not in agreement | parts} == {
        ("2j", "S2h")
    }
    assert {v.questions for v in values} == {("2c.3",)}
    assert numbers["non_swr_heldout_gain_per_spike"] == pytest.approx(0.6)
    assert numbers["swr_heldout_spiking_gain_per_spike"] == pytest.approx(0.7)
    assert numbers["swr_heldout_silent_gain_per_spike"] == pytest.approx(-0.1)
    silent = by_key["non_swr_heldout_silent_gain_per_spike"].description
    assert "silent" in silent and "need not add" in silent
    assert (
        "need not add" in by_key["non_swr_heldout_spiking_gain_per_spike"].description
    )
    assert numbers["non_swr_heldout_control_gain_per_spike"] == pytest.approx(0.0)
    assert numbers["swr_heldout_events_scored"] == 3
    # the agreement: the same statistic over agreement_per_spike
    assert numbers["non_swr_heldout_agreement_per_spike"] == pytest.approx(0.3)
    assert numbers["swr_heldout_agreement_per_spike_ci_low"] < 0.3
    assert numbers["swr_heldout_control_agreement_per_spike"] == pytest.approx(0.0)
    description = by_key["non_swr_heldout_agreement_per_spike"].description
    assert "no place preference" in description
    assert "not a zero reference" not in description
    # every held-out tetrode value says the windows' time is not held out
    assert all(
        "choice of time is not held out" in v.description
        for k, v in by_key.items()
        if k.endswith("_per_spike")
    )
    # the controls share the animal's own place: no zero reference for the
    # agreement, one for the gain
    control = by_key["swr_heldout_control_agreement_per_spike"].description
    assert "animal's own place" in control and "not a zero reference" in control
    assert "not a zero reference" not in (
        by_key["swr_heldout_control_gain_per_spike"].description
    )
    # six estimates, three values each, two groups; two counts
    assert len(numbers) == 6 * 3 * 2 + 2


_WIDTHS = (10.0, 30.0, 70.0, 150.0, 250.0)
"""One event of each group per animal in each width bin of Figure 2h."""
_BIN_AGREEMENT = (0.4, 0.2, 0.1, 0.0, -0.1)
"""Each width bin's agreement per spike."""


def _extra_windows() -> pd.DataFrame:
    """The held-out tetrode run with its extra scores: animals a, b, c (k = 1,
    2, 3), one epoch each, for each group one event in each width bin
    (:data:`_WIDTHS`, ids 100 on) and its control. In an event: the gain 0.1k
    (SWR 0.5 more), 0.7 of it from the rate term; the agreement by bin
    (:data:`_BIN_AGREEMENT`), half from the rate term, its reference -0.2
    (none in the widest bin of animal a's no-SWR events, no donor); mark
    information 0.3 and spike information 0.5 (0.2 and 0.4 in the controls)."""
    rows = []
    event_id = 100
    for k, animal in enumerate("abc", start=1):
        for group in ("no SWR", "SWR"):
            for width, agreement in zip(_WIDTHS, _BIN_AGREEMENT, strict=True):
                gain = 0.1 * k + (0.5 if group == "SWR" else 0.0)
                null = (
                    np.nan if (animal, group, width) == ("a", "no SWR", 250.0) else -0.2
                )
                for kind, mark, spike in (("event", 0.3, 0.5), ("control", 0.2, 0.4)):
                    rows.append(
                        {
                            "animal": animal, "epoch_id": f"{animal}1",
                            "event_id": event_id, "group": group, "kind": kind,
                            "hpd_cm": width, "gain_per_spike": gain,
                            "gain_rate_per_spike": 0.7 * gain,
                            "gain_mark_per_spike": 0.3 * gain,
                            "agreement_per_spike": agreement,
                            "agreement_rate_per_spike": agreement / 2,
                            "agreement_mark_per_spike": agreement / 2,
                            "agreement_null_per_spike": null,
                            "hpd_null_cm": np.nan if np.isnan(null) else width,
                            "agreement_minus_null_per_spike": agreement - null,
                            "mark_information_per_spike": mark,
                            "spike_information_per_spike": spike,
                        }
                    )  # fmt: skip
                event_id += 1
    return pd.DataFrame(rows)


def test_shared_content_values_are_the_agreement_its_reference_and_shares():
    values = _by_key(builder.shared_content_values(_extra_windows(), "s"))
    numbers = {key: value.value for key, value in values.items()}
    key = "non_swr_shared_content_{}_{}"
    for bin_word, agreement in zip(
        (
            "below_twenty_five_cm", "twenty_five_to_fifty_cm", "fifty_to_hundred_cm",
            "hundred_to_two_hundred_cm", "two_hundred_cm_or_more",
        ),
        _BIN_AGREEMENT,
        strict=True,
    ):  # fmt: skip
        assert numbers[key.format(bin_word, "agreement_per_spike")] == pytest.approx(
            agreement
        )
        assert numbers[key.format(bin_word, "reference_per_spike")] == pytest.approx(
            -0.2
        )
        assert numbers[
            key.format(bin_word, "agreement_minus_reference_per_spike")
        ] == pytest.approx(agreement + 0.2)
        # one event in five per animal and epoch
        assert numbers[key.format(bin_word, "share_of_events_animal_mean")] == (
            pytest.approx(0.2)
        )
        assert numbers[key.format(bin_word, "share_of_events")] == pytest.approx(0.2)
    widest = "two_hundred_cm_or_more"
    assert numbers[key.format(widest, "events")] == 3
    assert numbers[key.format(widest, "width_matched_events")] == 2
    assert (
        numbers[
            key.format("below_twenty_five_cm", "agreement_minus_reference_per_spike")
            + "_animals_positive"
        ]
        == 3
    )
    # the supplement's panel S2g draws the difference and the animals' shares,
    # with the counts, and the agreement and the reference; the pooled share
    # is for the text
    for name, value in values.items():
        assert value.questions == ("2c.3",)
        if name.endswith("_share_of_events"):
            assert value.figures == (), name
        else:
            assert value.figures == ("S2i",), name
    # five bins, two groups: agreement and reference (three values each), the
    # difference (four), the share positive (three), the share pooled and by
    # animal (four), the epochs, the events and the reference's events
    assert len(values) == 5 * 2 * (3 + 3 + 4 + 3 + 4 + 3)
    assert (
        "95 % HPD length"
        in values[key.format(widest, "agreement_per_spike")].description
    )


def test_tetrode_holdout_extra_values_split_the_evidence():
    values = _by_key(builder.tetrode_holdout_extra_values(_extra_windows(), "s"))
    numbers = {key: value.value for key, value in values.items()}
    # no SWR: gains 0.1, 0.2, 0.3; SWR 0.6, 0.7, 0.8
    assert numbers["non_swr_heldout_gain_rate_per_spike"] == pytest.approx(0.14)
    assert numbers["swr_heldout_gain_mark_per_spike"] == pytest.approx(0.21)
    # each animal's median event over the bins: agreement 0.1
    assert numbers["swr_heldout_agreement_rate_per_spike"] == pytest.approx(0.05)
    assert numbers["swr_heldout_agreement_reference_per_spike"] == pytest.approx(-0.2)
    assert numbers["swr_heldout_agreement_minus_reference_per_spike"] == (
        pytest.approx(0.3)
    )
    assert (
        numbers["swr_heldout_agreement_minus_reference_per_spike_animals_positive"] == 3
    )
    assert numbers["non_swr_heldout_mark_information_per_spike"] == pytest.approx(0.3)
    assert numbers["non_swr_heldout_control_mark_information_per_spike"] == (
        pytest.approx(0.2)
    )
    assert numbers["non_swr_heldout_mark_information_minus_control_per_spike"] == (
        pytest.approx(0.1)
    )
    assert numbers["swr_heldout_spike_information_minus_control_per_spike"] == (
        pytest.approx(0.1)
    )
    assert numbers["swr_heldout_control_spike_information_per_spike"] == (
        pytest.approx(0.4)
    )
    # six measures of the events (three values; the difference four), two of
    # the controls (three) and two paired (four), two groups
    assert len(values) == 2 * (5 * 3 + 4 + 2 * 3 + 2 * 3 + 2 * 4)
    # Figure 2i draws the agreement minus the reference (over another event's
    # place); no figure draws the rest
    drawn = {k for k in values if "_heldout_agreement_minus_reference_per_spike" in k}
    assert len(drawn) == 2 * 4
    assert {values[k].figures for k in drawn} == {("2j",)}
    assert {v.figures for k, v in values.items() if k not in drawn} == {()}
    assert {v.questions for v in values.values()} == {("2c.3",)}
    # none of the canonical run's keys is made again
    assert not set(values) & {
        f"{w}_heldout_{n}_per_spike"
        for w in ("swr", "non_swr")
        for n in ("gain", "spiking_gain", "silent_gain", "control_gain", "agreement",
                  "control_agreement")
    }  # fmt: skip


def test_low_rate_values_split_the_events_by_their_place_rate():
    """The narrowest event of each group and animal is decoded at a low-rate
    place (rank 0.05), the others at rank 0.5; of the canonical events, the two
    no-SWR events of animal a's first epoch are low-rate events."""
    windows = _extra_windows()
    events = _events().assign(event_id=range(13))
    ranked = windows[windows["kind"] == "event"]
    place_rate = pd.concat(
        [
            ranked[["event_id", "animal", "epoch_id", "group"]].assign(
                start_place_rate_rank=np.where(ranked["hpd_cm"] == 10.0, 0.05, 0.5),
                end_place_rate_rank=np.where(ranked["hpd_cm"] == 10.0, 0.0, 0.5),
            ),
            pd.DataFrame(
                {
                    "event_id": range(13),
                    "animal": events["animal"],
                    "epoch_id": events["animal"] + events["epoch"],
                    "group": np.where(events["is_swr"], "SWR", "no SWR"),
                    "start_place_rate_rank": [0.05, 0.05] + [0.5] * 11,
                    "end_place_rate_rank": [0.05, 0.05] + [0.5] * 11,
                }
            ),
        ],
        ignore_index=True,
    )
    values = _by_key(builder.low_rate_values(place_rate, events, windows, "s"))
    numbers = {key: value.value for key, value in values.items()}
    shares = builder.low_rate_places.low_rate_share_statistics(place_rate)
    assert numbers["non_swr_low_rate_share_animal_mean"] == pytest.approx(
        shares.loc["no SWR", "share_animal_mean"]
    )
    # animal a: e1 2 of 2, e2 0 of 1, a1 1 of 5: mean 0.4; b and c lower
    assert numbers["non_swr_low_rate_share_animal_max"] == pytest.approx(0.4)
    assert numbers["non_swr_low_rate_events"] == 5
    # SWR: each animal 1 of 5 in its window epoch, none in its other epoch
    assert numbers["swr_low_rate_share_animal_mean"] == pytest.approx(0.1)
    assert numbers["swr_place_rate_first_tenth_share"] == pytest.approx(0.1)
    assert numbers["swr_place_rate_sixth_tenth_share"] == pytest.approx(0.9)
    assert numbers["swr_place_rate_tenth_tenth_share"] == pytest.approx(0.0)
    # without a's two low-rate events: a 1/2, b 2/5, c 2/4
    assert numbers["non_swr_fraction_animal_mean_without_low_rate"] == pytest.approx(
        (0.5 + 0.4 + 0.5) / 3
    )
    assert numbers["nonlocal_events_without_low_rate"] == 11
    # the agreement minus the reference is not split by place group
    assert not [k for k in values if "_heldout_agreement" in k]
    assert numbers["swr_low_rate_heldout_gain_mark_per_spike"] == pytest.approx(0.21)
    assert numbers["swr_other_place_heldout_mark_information_per_spike"] == (
        pytest.approx(0.3)
    )
    assert numbers["non_swr_low_rate_heldout_events_scored"] == 3
    # no figure draws these
    for name, value in values.items():
        assert value.questions == ("2c.3",)
        assert value.figures == (), name
    # two groups: five share values, ten tenths; the fraction and its events;
    # three measures x two places x two groups x four, and the events scored
    assert len(values) == 2 * (5 + 10) + 2 + 3 * 2 * 2 * 4 + 2 * 2


def _restriction_windows() -> pd.DataFrame:
    """Animals a, b, c (k = 1, 2, 3), one epoch each, one window of 2 spikes per
    kind and group, with ``none`` 0 and per spike: no-SWR events ``full - swr``
    2k; SWR events ``swr - none`` 3k, ``full - none`` 2.5k, ``full - swr``
    -0.5k; no-SWR controls ``full - swr`` -0.1k, SWR controls 0.1k. The
    two-sided scores are twice the filter scores. Animal a has one more no-SWR
    control, without spikes.

    Outside SWRs, over half a second and 2 spikes (inside a well visit too):
    ``full - swr`` per second 4k in no-SWR events and k in SWR events (0 in
    controls), so per spike k and k / 4; the two-sided scores twice those. Each
    window is its own event's (a control's event id is its event's); the
    summary selects the windows of the events that start inside a well visit
    before the builders see them (``tests/reported/test_events_summary_planted.py``)."""
    sums = {
        ("event", "no SWR"): (0.0, 4.0),
        ("event", "SWR"): (6.0, 5.0),
        ("control", "no SWR"): (0.0, -0.2),
        ("control", "SWR"): (0.0, 0.2),
    }
    rows = []
    for k, animal in enumerate("abc", start=1):
        for (kind, group), (swr, full) in sums.items():
            rows.append((animal, kind, group, 2, k * swr, k * full, f"{animal}{group}"))
    rows.append(("a", "control", "no SWR", 0, 0.0, 1.0, "aextra"))
    table = pd.DataFrame(
        rows,
        columns=["animal", "kind", "group", "n_spikes", "swr", "full", "event_id"],
    )
    k = table["animal"].map({"a": 1.0, "b": 2.0, "c": 3.0})
    per_s = np.select(
        [
            (table["kind"] == "event") & (table["group"] == "no SWR"),
            (table["kind"] == "event") & (table["group"] == "SWR"),
        ],
        [4 * k, k],
        0.0,
    )
    outside = {
        "none": 0.0,
        "swr": 0.0,
        "full": 0.5 * per_s,
        "none_two_sided": 0.0,
        "swr_two_sided": 0.0,
        "full_two_sided": per_s,
    }
    return table.assign(
        epoch_id=table["animal"] + "1",
        n_bins_excluded=0,
        duration_s=1.0,
        none=0.0,
        none_two_sided=0.0,
        swr_two_sided=2 * table["swr"],
        full_two_sided=2 * table["full"],
        n_bins_outside_swr=1,
        duration_outside_swr_s=0.5,
        **{f"{model}_outside_swr": value for model, value in outside.items()},
        n_spikes_outside_swr=2,
        n_bins_outside_swr_in_visit=1,
        n_spikes_outside_swr_in_visit=2,
        **{f"{model}_outside_swr_in_visit": value for model, value in outside.items()},
    )


def test_restriction_window_values_name_each_window_contrast_and_score():
    windows = _restriction_windows()
    groups = builder.restriction_window_values(windows, "s")
    # of the window values, only the counts after the visit read the visits
    assert [v.key for v in groups.after_visit] == [
        f"restriction_canonical_{word}_events_outside_swr_after_visit"
        for word in ("non_swr", "swr")
    ]
    values = _by_key([*groups.windows, *groups.after_visit])
    # every key names the canonical events
    assert all(key.startswith("restriction_canonical_") for key in values)
    numbers = {key: value.value for key, value in values.items()}
    assert numbers["restriction_canonical_non_swr_events_full_minus_swr_per_spike"] == (
        pytest.approx(4.0)
    )
    assert numbers["restriction_canonical_swr_events_swr_minus_none_per_spike"] == (
        pytest.approx(6.0)
    )
    assert numbers["restriction_canonical_swr_events_full_minus_none_per_spike"] == (
        pytest.approx(5.0)
    )
    assert numbers["restriction_canonical_swr_events_full_minus_swr_per_spike"] == (
        pytest.approx(-1.0)
    )
    assert numbers[
        "restriction_canonical_non_swr_controls_full_minus_swr_per_spike"
    ] == (pytest.approx(-0.2))
    assert numbers["restriction_canonical_swr_controls_full_minus_swr_per_spike"] == (
        pytest.approx(0.2)
    )
    assert numbers[
        "restriction_canonical_non_swr_events_full_minus_swr_two_sided_per_spike"
    ] == pytest.approx(8.0)
    assert (
        numbers[
            "restriction_canonical_swr_events_full_minus_swr_two_sided_per_spike_ci_low"
        ]
        < -2.0
        < numbers[
            "restriction_canonical_swr_events_full_minus_swr_two_sided_per_spike_ci_high"
        ]
    )
    assert (
        numbers[
            "restriction_canonical_swr_events_full_minus_swr_per_spike_animals_positive"
        ]
        == 0
    )
    assert (
        numbers[
            "restriction_canonical_swr_controls_full_minus_swr_per_spike_animals_positive"
        ]
        == 3
    )
    assert numbers["restriction_canonical_non_swr_controls"] == 4
    assert numbers["restriction_canonical_non_swr_controls_scored"] == 3
    assert numbers["restriction_canonical_swr_events_scored"] == 3
    assert numbers["restriction_canonical_animals"] == 3
    # outside SWRs, per spike: no SWR 1, 2, 3; SWR 0.25, 0.5, 0.75
    key = "restriction_canonical_{}_events_outside_swr_full_minus_swr{}_per_spike"
    assert numbers[key.format("non_swr", "")] == pytest.approx(2.0)
    assert numbers[key.format("swr", "")] == pytest.approx(0.5)
    assert numbers[key.format("swr", "_two_sided")] == pytest.approx(1.0)
    assert values[key.format("swr", "")].unit == "nats per spike"
    # per second: no SWR 4, 8, 12; SWR 1, 2, 3
    key = "restriction_canonical_{}_events_outside_swr_full_minus_swr{}_per_s"
    assert numbers[key.format("non_swr", "")] == pytest.approx(8.0)
    assert numbers[key.format("swr", "")] == pytest.approx(2.0)
    assert numbers[key.format("swr", "_two_sided")] == pytest.approx(4.0)
    assert numbers[key.format("swr", "") + "_animals_positive"] == 3
    assert values[key.format("swr", "")].unit == "nats per s"
    assert numbers["restriction_canonical_swr_events_outside_swr_scored_per_s"] == 3
    assert numbers["restriction_canonical_swr_events_outside_swr_scored_per_spike"] == 3
    assert numbers["restriction_canonical_swr_events_outside_swr_without_spikes"] == 0
    assert numbers["restriction_canonical_non_swr_events_outside_swr_after_visit"] == 0
    # two families x six comparisons x (estimate, two bounds, animals positive),
    # two counts for each kind and group, and the animals; outside SWRs, two
    # families x two units x two groups x four, the two groups' events scored
    # per unit, and the two groups' events without spikes and after the visit
    assert len(numbers) == 2 * 6 * 4 + 2 * 4 + 1 + 2 * 2 * 2 * 4 + 2 * 2 + 2 * 2
    assert {v.questions for v in values.values()} == {("2c.3",)}
    # no figure draws these
    assert {v.figures for v in values.values()} == {()}
    # the summary says what they are over (``over``), once
    assert not any(
        "start inside a well visit" in v.description for v in values.values()
    )


def test_restriction_heldout_event_values_are_the_canonical_statistic():
    """The canonical windows' events, read as held-out events: the same
    statistic as the canonical values outside SWRs. Figure 2j's third axis
    draws the no-SWR events' gain, with their counts; the SWR events' values
    are for the text."""
    windows = _restriction_windows()
    events = windows[windows["kind"] == "event"]
    values = _by_key(builder.restriction_heldout_event_values(events, "s"))
    numbers = {key: value.value for key, value in values.items()}
    key = "restriction_heldout_{}_events_outside_swr_full_minus_swr_per_spike"
    # outside SWRs, per spike: no SWR 1, 2, 3; SWR 0.25, 0.5, 0.75
    assert numbers[key.format("non_swr")] == pytest.approx(2.0)
    assert numbers[key.format("swr")] == pytest.approx(0.5)
    assert numbers[key.format("swr") + "_ci_low"] < 0.5
    assert numbers[key.format("swr") + "_animals_positive"] == 3
    assert values[key.format("swr")].unit == "nats per spike"
    assert numbers["restriction_heldout_non_swr_events"] == 3
    assert numbers["restriction_heldout_swr_events_outside_swr_scored_per_spike"] == 3
    assert numbers["restriction_heldout_swr_events_outside_swr_without_spikes"] == 0
    # per group: the estimate, two bounds, animals positive, the events, those
    # scored and those without spikes
    assert len(numbers) == 2 * 7
    assert {k for k, v in values.items() if v.figures == ("2k",)} == {
        k for k in values if k.startswith("restriction_heldout_non_swr_")
    }
    assert {v.figures for v in values.values()} == {(), ("2k",)}
    assert {v.questions for v in values.values()} == {("2c.3",)}
    canonical = _numbers(builder.restriction_window_values(windows, "s").windows)
    for word in ("non_swr", "swr"):
        assert numbers[key.format(word)] == pytest.approx(
            canonical[
                f"restriction_canonical_{word}_events_outside_swr_full_minus_swr_per_spike"
            ]
        )


def test_restriction_agreement_values_are_the_shares_found_again():
    """Canonical events: animal a 2 of 2 no-SWR found and 0 of 1 SWR; animal b
    1 of 1 SWR. Held-out events: animal a 1 of 2 no-SWR with a canonical
    partner, animal b 1 of 1 SWR."""
    canonical = pd.DataFrame(
        {
            "epoch_id": ["a1", "a1", "a1", "b1"],
            "animal": ["a", "a", "a", "b"],
            "group": ["no SWR", "no SWR", "SWR", "SWR"],
            "shares_bin_with_heldout": [True, True, False, True],
        }
    )
    heldout = pd.DataFrame(
        {
            "epoch_id": ["a1", "a1", "b1"],
            "animal": ["a", "a", "b"],
            "group": ["no SWR", "no SWR", "SWR"],
            "shares_bin_with_canonical": [True, False, True],
        }
    )
    values = _by_key(builder.restriction_agreement_values(heldout, canonical, "s"))
    numbers = {key: value.value for key, value in values.items()}
    assert numbers["restriction_canonical_events_found_again"] == pytest.approx(0.75)
    assert numbers["restriction_canonical_non_swr_events_found_again"] == 1.0
    assert numbers["restriction_canonical_swr_events_found_again"] == 0.5
    assert numbers["restriction_canonical_events_found_again_animal_min"] == (
        pytest.approx(2 / 3)
    )
    assert numbers["restriction_canonical_events_found_again_animal_max"] == 1.0
    assert numbers["restriction_heldout_events_with_canonical"] == pytest.approx(2 / 3)
    assert numbers["restriction_heldout_non_swr_events_with_canonical"] == 0.5
    assert numbers["restriction_heldout_swr_events_with_canonical"] == 1.0
    assert numbers["restriction_heldout_events"] == 3
    assert values["restriction_canonical_events_found_again"].kind == "fraction"
    # two sets x three groups x (share, lowest and highest animal), the events
    assert len(numbers) == 2 * 3 * 3 + 1
    assert {v.figures for v in values.values()} == {()}
    assert {v.questions for v in values.values()} == {("2c.3",)}


def _outside_swr_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Animals a (epochs a1, a2), b (b1, b2) and c (c1, c2), 10 bins of 1 s
    outside SWRs inside a well visit each, 5 spikes, ``full - swr`` there of 20,
    40, 60, 80, 50 and 70 (two-sided twice that); 4 of the bins and 2 of the
    spikes in event windows, with a third of the gain. Of that, the selected
    no-SWR events hold 10, 16, 30, 32, 25 and 28 inside the visit (more outside
    it, which never counts), and the selected SWR events a fifth of the gain in
    their time outside the SWR. Time outside a visit and SWR bins never count;
    the summary selects the windows (an event that starts outside a visit,
    planted here, is not handed to the builder). Returns the time sums, the
    time steps and the selected windows."""
    rows, windows = [], []
    for epoch, gain, in_events in (
        ("a1", 20.0, 10.0),
        ("a2", 40.0, 16.0),
        ("b1", 60.0, 30.0),
        ("b2", 80.0, 32.0),
        ("c1", 50.0, 25.0),
        ("c2", 70.0, 28.0),
    ):
        rows += [
            (epoch, "immobile_outside_swr", True, True, 4, 2, 0.0, gain / 3),
            (epoch, "immobile_outside_swr", True, False, 6, 3, 0.0, 2 * gain / 3),
            (epoch, "immobile_outside_swr", False, False, 7, 7, 0.0, 1000.0),
            (epoch, "immobile_in_swr", True, True, 5, 5, 0.0, 1000.0),
        ]
        windows += [
            (epoch, "event", "no SWR", in_events, in_events + 50, f"{epoch}n"),
            (epoch, "event", "SWR", gain / 5, gain / 5, f"{epoch}s"),
            (epoch, "event", "no SWR", 1000.0, 1000.0, f"{epoch}away"),
        ]
    windows = pd.DataFrame(
        windows,
        columns=["epoch_id", "kind", "group", "in_visit", "outside", "event_id"],
    )
    columns = dict.fromkeys(
        (
            *builder.restriction_windows.OUTSIDE_SWR_COLUMNS,
            *builder.restriction_windows.BIN_FACT_COLUMNS,
        ),
        0.0,
    )
    windows = windows.assign(
        n_bins_excluded=0, animal=windows["epoch_id"].str[0], **columns
    )
    # 2 spikes outside SWRs in each window, so that the windows have a value
    # per spike there (Figure 2j's third axis)
    windows["n_spikes_outside_swr"] = 2
    for suffix, scale in (("", 1.0), ("_two_sided", 2.0)):
        windows[f"full{suffix}_outside_swr"] = scale * windows["outside"]
        windows[f"full{suffix}_outside_swr_in_visit"] = scale * windows["in_visit"]
    selected = windows[~windows["event_id"].str.endswith("away")]
    bin_s = pd.Series(1.0, index=["a1", "a2", "b1", "b2", "c1", "c2"])
    return time_sums(rows), bin_s, selected


def test_restriction_outside_swr_values_are_the_gain_and_its_share_in_a_visit():
    """The gain outside SWRs and its shares, on :func:`_outside_swr_inputs`."""
    sums, bin_s, selected = _outside_swr_inputs()
    groups = builder.restriction_outside_swr_values(sums, bin_s, selected, "s")
    values = _by_key([*groups.with_events, *groups.outside_events])
    numbers = {key: value.value for key, value in values.items()}
    key = "restriction_outside_swr_full_minus_swr"
    # the values over the time no held-out event holds are a group of their own
    assert {v.key for v in groups.outside_events} == {
        f"{key}{family}_outside_heldout_events_{unit}{end}"
        for family in ("", "_two_sided")
        for unit in ("per_spike", "per_s")
        for end in ("", "_ci_low", "_ci_high", "_animals_positive")
    }
    # a: median of 2 and 4 = 3; b: of 6 and 8 = 7; c: of 5 and 7 = 6
    assert numbers[f"{key}_per_s"] == pytest.approx(16 / 3)
    assert numbers[f"{key}_per_s_ci_low"] < 16 / 3 < numbers[f"{key}_per_s_ci_high"]
    assert numbers[f"{key}_two_sided_per_s"] == pytest.approx(32 / 3)
    # per spike, twice that: 5 spikes over 10 s
    assert numbers[f"{key}_per_spike"] == pytest.approx(32 / 3)
    assert values[f"{key}_per_spike"].unit == "nats per spike"
    assert numbers[f"{key}_per_s_animals_positive"] == 3
    assert numbers[f"{key}_epochs_positive"] == 6
    assert numbers[f"{key}_epochs"] == 6
    # shares 0.5 and 0.4 in every animal: each animal's median 0.45
    assert numbers[f"{key}_share_in_heldout_non_swr_events"] == pytest.approx(0.45)
    assert numbers[f"{key}_two_sided_share_in_heldout_non_swr_events"] == pytest.approx(
        0.45
    )
    assert values[f"{key}_share_in_heldout_non_swr_events"].kind == "fraction"
    # the rest: two thirds of the gain (animal medians 30, 70, 60 nats) over 6 s
    # and 3 spikes
    rest = 2 / 3 * 160 / 3
    assert numbers[f"{key}_outside_heldout_events_per_s"] == pytest.approx(rest / 6)
    assert numbers[f"{key}_outside_heldout_events_per_spike"] == pytest.approx(rest / 3)
    assert numbers[f"{key}_two_sided_outside_heldout_events_per_s"] == pytest.approx(
        2 * rest / 6
    )
    assert numbers[f"{key}_outside_heldout_events_per_s_animals_positive"] == 3
    # the SWR events' time outside the SWR: a fifth of each epoch's gain
    assert numbers[f"{key}_share_in_heldout_swr_events_outside_swr"] == pytest.approx(
        0.2
    )
    assert numbers[f"{key}_two_sided_share_in_heldout_swr_events_outside_swr"] == (
        pytest.approx(0.2)
    )
    # elsewhere: 1 - 0.5 - 0.2 and 1 - 0.4 - 0.2 in every animal, median 0.35
    assert numbers[f"{key}_share_outside_heldout_events"] == pytest.approx(0.35)
    assert numbers[f"{key}_two_sided_share_outside_heldout_events"] == (
        pytest.approx(0.35)
    )
    # a quantity: below 0 where the full decoder does worse
    assert values[f"{key}_share_outside_heldout_events"].kind == "quantity"
    assert (
        numbers[f"{key}_share_outside_heldout_events_ci_low"]
        <= 0.35
        <= numbers[f"{key}_share_outside_heldout_events_ci_high"]
    )
    # per family: per spike and per second, the gain and the rest (four each),
    # the epochs (two), and three shares (three each)
    assert len(numbers) == 2 * (2 * 2 * 4 + 2 + 3 * 3)
    # Figure 2j draws the filter family's gain per spike, and per spike
    # outside every held-out event; the shares are for the text, with no figure
    drawn = {
        f"{key}{where}_per_spike{end}"
        for where in ("", "_outside_heldout_events")
        for end in ("", "_ci_low", "_ci_high", "_animals_positive")
    }
    assert {k for k, v in values.items() if v.figures == ("2k",)} == drawn
    assert {v.figures for k, v in values.items() if k not in drawn} == {()}


def _immobile_time_sums() -> pd.DataFrame:
    """Animals a, b, c, one epoch each, 10 immobile bins of 1 s inside a well
    visit (5 in SWRs) with 20 spikes; ``swr - none`` and ``full - none`` over
    them 20 and 30, 24 and 32, 21 and 30 (two-sided twice that); moving bins
    and time outside a visit do not count."""
    rows = []
    for epoch, swr, full in (
        ("a1", 20.0, 30.0),
        ("b1", 24.0, 32.0),
        ("c1", 21.0, 30.0),
    ):
        rows += [
            (epoch, "immobile_in_swr", True, True, 5, 10, swr, full - 5.0),
            (epoch, "immobile_outside_swr", True, False, 5, 10, 0.0, 5.0),
            (epoch, "immobile_outside_swr", False, False, 5, 10, 0.0, 500.0),
            (epoch, "moving", True, False, 50, 10, 1000.0, 0.0),
        ]
    return time_sums(rows)


def test_restriction_immobile_values_are_each_decoders_gain_and_the_share():
    bin_s = pd.Series(1.0, index=["a1", "b1", "c1"])
    values = _by_key(
        builder.restriction_immobile_values(_immobile_time_sums(), bin_s, "s")
    )
    numbers = {key: value.value for key, value in values.items()}
    assert numbers["restriction_immobile_swr_minus_none_per_s"] == pytest.approx(
        6.5 / 3
    )
    assert numbers["restriction_immobile_full_minus_none_per_s"] == pytest.approx(
        9.2 / 3
    )
    assert numbers["restriction_immobile_full_minus_swr_per_s"] == pytest.approx(0.9)
    assert numbers["restriction_immobile_full_minus_none_two_sided_per_s"] == (
        pytest.approx(18.4 / 3)
    )
    # per spike, half that: 20 spikes over 10 s
    assert numbers["restriction_immobile_full_minus_swr_per_spike"] == (
        pytest.approx(0.45)
    )
    assert numbers["restriction_immobile_swr_minus_none_per_spike"] == (
        pytest.approx(6.5 / 6)
    )
    assert values["restriction_immobile_swr_minus_none_per_spike"].unit == (
        "nats per spike"
    )
    assert numbers["restriction_immobile_full_minus_swr_per_s_animals_positive"] == 3
    assert numbers["restriction_immobile_epochs_full_above_swr_only"] == 3
    assert numbers["restriction_immobile_epochs"] == 3
    share = (2 / 3 + 3 / 4 + 0.7) / 3
    assert numbers["restriction_immobile_share_captured_by_swr_only"] == (
        pytest.approx(share)
    )
    assert values["restriction_immobile_share_captured_by_swr_only"].kind == "fraction"
    # per family: three gains x two units x four, the epochs above, the share
    # and its bounds; and the epochs
    assert len(numbers) == 2 * (3 * 2 * 4 + 1 + 3) + 1
    # no figure draws them: Figure 2j compares the full and the SWR-only
    # decoders inside and outside SWRs, not over all immobile time
    assert {v.figures for v in values.values()} == {()}
    assert {v.questions for v in values.values()} == {("2c.3",)}
    # the summary says what they are over (``over``), once
    assert not any("inside a well visit" in v.description for v in values.values())


def test_restriction_figure_statistics_are_the_tagged_values():
    """Figure 2j's statistics are the rows the drawn values are made from."""
    sums, bin_s, windows = _outside_swr_inputs()
    statistics = builder.restriction_totals.restriction_figure_statistics(
        windows, sums, bin_s
    )
    outside = statistics.outside
    groups = builder.restriction_outside_swr_values(sums, bin_s, windows, "s")
    numbers = _numbers(groups.with_events)
    key = "restriction_outside_swr_full_minus_swr"
    for measure, name in (
        (builder.RESTRICTION_FIGURE_OUTSIDE_SWR, "per_spike"),
        ("share_in_no_swr_events", "share_in_heldout_non_swr_events"),
        ("share_in_swr_events_outside_swr", "share_in_heldout_swr_events_outside_swr"),
        ("share_outside_events", "share_outside_heldout_events"),
    ):
        np.testing.assert_allclose(
            outside.loc[("filter", measure), "estimate"], numbers[f"{key}_{name}"]
        )
    numbers = _numbers(groups.outside_events)
    np.testing.assert_allclose(
        outside.loc[("filter", builder.RESTRICTION_FIGURE_OUTSIDE_EVENTS), "estimate"],
        numbers[f"{key}_outside_heldout_events_per_spike"],
    )
    numbers = _numbers(builder.restriction_inside_swr_values(sums, bin_s, "s"))
    np.testing.assert_allclose(
        statistics.inside.loc[
            ("filter", builder.RESTRICTION_FIGURE_INSIDE_SWR), "estimate"
        ],
        numbers["restriction_inside_swr_full_minus_swr_per_spike"],
    )
    # the events' gain, on windows that carry their spikes outside SWRs
    events = _restriction_windows()
    events = events[events["kind"] == "event"]
    statistics = builder.restriction_totals.restriction_figure_statistics(
        events, sums, bin_s
    )
    numbers = _numbers(builder.restriction_heldout_event_values(events, "s"))
    np.testing.assert_allclose(
        statistics.events.loc[builder.RESTRICTION_FIGURE_EVENT_GROUP, "estimate"],
        numbers[
            "restriction_heldout_non_swr_events_outside_swr_full_minus_swr_per_spike"
        ],
    )


def test_restriction_inside_swr_values_are_the_gain_in_swrs_inside_a_visit():
    """In SWRs: full - swr is 5, 3 and 4 nats over 10 spikes and 5 s per
    epoch; per spike 0.5, 0.3, 0.4 and per second twice that. The same
    statistic, interval and epoch counts as the gain outside SWRs, both score
    families, question 2c.3; the filter family's gain per spike (and its
    interval and animals) is drawn on Figure 2j."""
    bin_s = pd.Series(1.0, index=["a1", "b1", "c1"])
    values = _by_key(
        builder.restriction_inside_swr_values(_immobile_time_sums(), bin_s, "s")
    )
    numbers = {key: value.value for key, value in values.items()}
    key = "restriction_inside_swr_full_minus_swr"
    assert numbers[f"{key}_per_spike"] == pytest.approx(0.4)
    assert numbers[f"{key}_per_s"] == pytest.approx(0.8)
    assert numbers[f"{key}_two_sided_per_spike"] == pytest.approx(0.8)
    assert numbers[f"{key}_per_spike_animals_positive"] == 3
    assert numbers[f"{key}_epochs_positive"] == 3
    assert numbers[f"{key}_epochs"] == 3
    assert values[f"{key}_per_spike"].unit == "nats per spike"
    assert values[f"{key}_per_s"].unit == "nats per s"
    assert numbers[f"{key}_per_spike_ci_low"] <= 0.4
    # per family: two units x four, and the epochs (two)
    assert len(numbers) == 2 * (2 * 4 + 2)
    drawn = {
        f"{key}_per_spike{end}"
        for end in ("", "_ci_low", "_ci_high", "_animals_positive")
    }
    for name, value in values.items():
        assert value.questions == ("2c.3",)
        assert value.figures == (("2k",) if name in drawn else ())
    assert "inside SWRs" in values[f"{key}_per_spike"].description


def _swr_allowed_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """:func:`_immobile_time_sums` with the fourth decoder: ``swr_allowed`` is
    ``swr`` plus 1 nat in SWRs and ``swr`` outside them (two-sided twice that).
    So per epoch (a, b, c): inside SWRs (10 spikes, 5 s) ``full -
    swr_allowed`` 4, 2, 3 and ``swr_allowed - swr`` 1; outside SWRs (10
    spikes, 5 s) ``full - swr_allowed`` 5 and ``swr_allowed - swr`` 0; and the
    held-out events of :func:`_restriction_windows` with ``swr_allowed`` a
    quarter of ``full`` outside SWRs, so ``full - swr_allowed`` there is three
    quarters of ``full - swr``. Returns the time sums, the time steps and the
    events."""
    sums = _immobile_time_sums()
    in_swr = (sums["category"] == "immobile_in_swr").to_numpy()
    sums["swr_allowed"] = sums["swr"] + np.where(in_swr, 1.0, 0.0)
    sums["swr_allowed_two_sided"] = 2 * sums["swr_allowed"]
    windows = _restriction_windows()
    events = windows[windows["kind"] == "event"].copy()
    events["swr_allowed"] = events["swr"]
    events["swr_allowed_two_sided"] = events["swr_two_sided"]
    for suffix in ("", "_two_sided"):
        for where in ("_outside_swr", "_outside_swr_in_visit"):
            events[f"swr_allowed{suffix}{where}"] = (
                0.25 * events[f"full{suffix}{where}"]
            )
    return sums, pd.Series(1.0, index=["a1", "b1", "c1"]), events


def test_restriction_swr_allowed_values_compare_the_fourth_decoder():
    """On :func:`_swr_allowed_inputs`: each contrast of the fourth decoder over
    each time, per spike and per second, both families, with the statistic,
    interval and epoch counts of the gain inside SWRs; the SWR-only and full
    decoders of the same pass over all immobile time; and the full minus the
    SWR-allowed decoder in the held-out no-SWR events. Figure 2's supplement
    (S2m) draws the filter family's gains per spike of
    ``SWR_ALLOWED_FIGURE_*`` and the held-out events' value with its counts."""
    sums, bin_s, events = _swr_allowed_inputs()
    values = _by_key(builder.restriction_swr_allowed_values(events, sums, bin_s, "s"))
    numbers = {key: value.value for key, value in values.items()}
    # inside SWRs: per spike 0.4, 0.2, 0.3 and 0.1; per second twice that
    key = "restriction_inside_swr_{}{}_{}"
    assert numbers[key.format("full_minus_swr_allowed", "", "per_spike")] == (
        pytest.approx(0.3)
    )
    assert numbers[key.format("full_minus_swr_allowed", "", "per_s")] == (
        pytest.approx(0.6)
    )
    assert numbers[key.format("swr_allowed_minus_swr", "", "per_spike")] == (
        pytest.approx(0.1)
    )
    assert numbers[key.format("swr_allowed_minus_swr", "_two_sided", "per_spike")] == (
        pytest.approx(0.2)
    )
    # swr_allowed - none in SWRs: 21, 25, 22 over 10 spikes
    assert numbers[key.format("swr_allowed_minus_none", "", "per_spike")] == (
        pytest.approx(6.8 / 3)
    )
    # outside SWRs: the SWR-allowed decoder is the SWR-only decoder there
    key = "restriction_outside_swr_{}_per_spike"
    assert numbers[key.format("full_minus_swr_allowed")] == pytest.approx(0.5)
    assert numbers[key.format("swr_allowed_minus_swr")] == pytest.approx(0.0)
    assert numbers[key.format("swr_allowed_minus_swr") + "_animals_positive"] == 0
    # all immobile time (20 spikes, 10 s): full - swr_allowed 9, 7, 8 nats
    key = "restriction_immobile_{}_per_spike"
    assert numbers[key.format("full_minus_swr_allowed")] == pytest.approx(0.4)
    assert numbers[key.format("swr_allowed_minus_swr")] == pytest.approx(0.05)
    assert numbers[key.format("swr_allowed_minus_none")] == pytest.approx(6.8 / 6)
    assert numbers["restriction_immobile_full_minus_swr_allowed_epochs"] == 3
    assert numbers["restriction_immobile_full_minus_swr_allowed_epochs_positive"] == 3
    assert values[key.format("full_minus_swr_allowed")].unit == "nats per spike"
    assert values["restriction_immobile_full_minus_swr_allowed_per_s"].unit == (
        "nats per s"
    )
    # the other two decoders of the same pass equal the canonical values on the
    # same time sums
    canonical = _numbers(builder.restriction_immobile_values(sums, bin_s, "s"))
    for contrast in ("swr_minus_none", "full_minus_none"):
        for family in ("", "_two_sided"):
            for unit in ("per_spike", "per_s"):
                assert numbers[
                    f"restriction_swr_allowed_run_immobile_{contrast}{family}_{unit}"
                ] == pytest.approx(
                    canonical[f"restriction_immobile_{contrast}{family}_{unit}"]
                )
    # the held-out no-SWR events outside SWRs: three quarters of 1, 2, 3
    key = "restriction_heldout_non_swr_events_outside_swr_full_minus_swr_allowed"
    assert numbers[f"{key}_per_spike"] == pytest.approx(1.5)
    assert numbers[f"{key}_per_spike_animals_positive"] == 3
    assert numbers["restriction_heldout_swr_allowed_run_non_swr_events"] == 3
    assert (
        numbers[
            "restriction_heldout_swr_allowed_run_non_swr_events_outside_swr_scored"
            "_per_spike"
        ]
        == 3
    )
    # eleven contrasts over a time, two families: per spike four values, per
    # second six; the held-out events' four and two counts
    assert len(numbers) == 11 * 2 * (4 + 6) + 4 + 2
    ends = ("", "_ci_low", "_ci_high", "_animals_positive")
    drawn = (
        {
            f"restriction_{time}_{contrast}_per_spike{end}"
            for time in ("inside_swr", "outside_swr", "immobile")
            for contrast in ("full_minus_swr_allowed", "swr_allowed_minus_swr")
            for end in ends
        }
        | {
            f"restriction{run}_immobile_{contrast}_per_spike{end}"
            for run, contrast in (
                ("_swr_allowed_run", "swr_minus_none"),
                ("", "swr_allowed_minus_none"),
                ("_swr_allowed_run", "full_minus_none"),
            )
            for end in ends
        }
        | {f"{key}_per_spike{end}" for end in ends}
        | {
            "restriction_heldout_swr_allowed_run_non_swr_events",
            "restriction_heldout_swr_allowed_run_non_swr_events_outside_swr_scored"
            "_per_spike",
        }
    )
    for name, value in values.items():
        assert value.questions == ("2c.3",)
        assert value.figures == (("S2n",) if name in drawn else ()), name
    # no key of the canonical builders is made again
    canonical_keys = set(canonical) | set(
        _numbers(builder.restriction_inside_swr_values(sums, bin_s, "s"))
    )
    assert not canonical_keys & set(numbers)
    assert "SWR-allowed decoder" in values[f"{key}_per_spike"].description


def test_swr_allowed_statistics_are_the_tagged_values():
    """Figure 2's supplement, panel m, draws the rows the S2m values are made
    from."""
    sums, bin_s, events = _swr_allowed_inputs()
    statistics = builder.restriction_totals.swr_allowed_statistics(events, sums, bin_s)
    numbers = _numbers(builder.restriction_swr_allowed_values(events, sums, bin_s, "s"))
    for time in ("inside_swr", "outside_swr", "immobile"):
        for measure in builder.SWR_ALLOWED_FIGURE_DIFFERENCES:
            assert statistics.times.loc[
                (time, "filter", measure), "estimate"
            ] == pytest.approx(numbers[f"restriction_{time}_{measure}"])
    for measure in builder.SWR_ALLOWED_FIGURE_DECODERS:
        run = "" if measure.startswith("swr_allowed") else "_swr_allowed_run"
        assert statistics.times.loc[
            ("immobile", "filter", measure), "estimate"
        ] == pytest.approx(numbers[f"restriction{run}_immobile_{measure}"])
    assert statistics.events.loc[
        builder.RESTRICTION_FIGURE_EVENT_GROUP, "estimate"
    ] == pytest.approx(
        numbers[
            "restriction_heldout_non_swr_events_outside_swr_full_minus_swr_allowed"
            "_per_spike"
        ]
    )


def _part_time_sums() -> pd.DataFrame:
    """Animals a, b, c, one epoch each, bins of 1 s outside SWRs inside a well
    visit, one spike a bin, ``full - swr`` per bin by part: 4 bins within
    100 ms of the scoring mask at 0 nats, 3 bins beyond 100 ms at k nats each
    (k = 1, 2, 3 for a, b, c), 2 beyond 250 ms at 2k, 1 beyond 500 ms at 4k,
    all sampled under 10 s but the last, sampled 60 s or more; and in each
    epoch 1 bin where SWR detection was not observed (b's has 2), at 100 nats,
    which never counts."""
    rows, observed, margin, sampling = [], [], [], []
    for k, epoch in enumerate(("a1", "b1", "c1"), start=1):
        for n_bins, nats, beyond, sampled, seen in (
            (4, 0.0, 0.0, 0.0, True),
            (3, 3.0 * k, 0.1, 0.0, True),
            (2, 4.0 * k, 0.25, 0.0, True),
            (1, 4.0 * k, 0.5, 60.0, True),
            (2 if epoch == "b1" else 1, 100.0, 0.5, 60.0, False),
        ):
            rows.append(
                (epoch, "immobile_outside_swr", True, False, n_bins, n_bins, 0.0, nats)
            )
            observed.append(seen)
            margin.append(beyond)
            sampling.append(sampled)
    return time_sums(
        rows,
        swr_observed=observed,
        beyond_swr_margin_s=margin,
        place_sampling_at_least_s=sampling,
    )


def test_restriction_outside_swr_part_values_are_the_gain_over_each_part():
    bin_s = pd.Series(1.0, index=["a1", "b1", "c1"])
    values = _by_key(
        builder.restriction_outside_swr_part_values(_part_time_sums(), bin_s, "s")
    )
    numbers = {key: value.value for key, value in values.items()}
    stem = "restriction_outside_swr"
    assert numbers[f"{stem}_not_observed_bins"] == 4
    assert numbers[f"{stem}_not_observed_epochs"] == 3
    # beyond 100 ms: bins of 3k, 2k, 4k over 6 bins, so 11k / 6 per spike;
    # animals 11/6, 22/6, 33/6, mean 22/6
    key = f"{stem}_beyond_hundred_ms_full_minus_swr"
    assert numbers[f"{key}_per_spike"] == pytest.approx(22 / 6)
    assert numbers[f"{key}_per_s"] == pytest.approx(22 / 6)
    assert numbers[f"{key}_two_sided_per_spike"] == pytest.approx(44 / 6)
    assert numbers[f"{key}_epochs_positive"] == 3
    assert numbers[f"{key}_epochs"] == 3
    assert numbers[f"{stem}_beyond_hundred_ms_share_of_bins"] == pytest.approx(0.6)
    # beyond 250 ms: 4k and 4k over 3 bins (mean 16 / 3); beyond 500 ms: 4k
    # over 1 bin
    assert numbers[f"{stem}_beyond_quarter_s_full_minus_swr_per_spike"] == (
        pytest.approx(16 / 3)
    )
    assert numbers[f"{stem}_beyond_half_s_full_minus_swr_per_spike"] == (
        pytest.approx(8.0)
    )
    assert numbers[f"{stem}_beyond_half_s_share_of_bins"] == pytest.approx(0.1)
    # sampled at least 10 s and at least 60 s: the last bin only
    assert numbers[f"{stem}_sampled_ten_s_full_minus_swr_per_spike"] == (
        pytest.approx(8.0)
    )
    assert numbers[f"{stem}_sampled_minute_full_minus_swr_per_s"] == (
        pytest.approx(8.0)
    )
    assert values[f"{key}_per_spike"].unit == "nats per spike"
    assert values[f"{stem}_beyond_hundred_ms_share_of_bins"].kind == "fraction"
    assert "100 ms" in values[f"{key}_per_spike"].description
    share = values[f"{stem}_sampled_minute_share_of_bins"].description
    assert "60 s" in share and "no interval is given" in share
    assert "lowest animal's value" in share
    lowest = values[f"{stem}_sampled_minute_share_of_bins_animal_min"].description
    assert "lowest of the animals' values of the share" in lowest and "60 s" in lowest
    assert "SWR detection was observed" in values[f"{key}_per_spike"].description
    assert numbers[f"{stem}_beyond_hundred_ms_share_of_bins_animal_min"] == (
        pytest.approx(0.6)
    )
    # per part: two families x (two units x four + two epoch counts), and the
    # share with the animals' lowest; and the two counts left out
    assert len(numbers) == 5 * (2 * (2 * 4 + 2) + 2) + 2
    # Figure 2's supplement draws the gain per spike away from the SWRs in the
    # filter family (panel S2e); the gain at well-sampled places is for the text
    drawn = {
        k
        for k in values
        if "_beyond_" in k
        and "_full_minus_swr_per_spike" in k
        and "_two_sided" not in k
    }
    assert len(drawn) == 3 * 4
    assert {
        k
        for k, v in values.items()
        if v.figures == builder.TAGS["nonlocal_restriction_distance"].figures
    } == drawn
    assert {v.figures for k, v in values.items() if k not in drawn} == {()}
    assert {v.questions for v in values.values()} == {("2c.3",)}


def test_the_gain_outside_swrs_leaves_out_the_bins_without_ripple_signal():
    """The headline's bins are those observed: 10 per epoch, 11k nats."""
    bin_s = pd.Series(1.0, index=["a1", "b1", "c1"])
    groups = builder.restriction_outside_swr_values(
        _part_time_sums(), bin_s, _restriction_windows().iloc[:0], "s"
    )
    numbers = _numbers(groups.with_events)
    assert numbers["restriction_outside_swr_full_minus_swr_per_spike"] == (
        pytest.approx(22 / 10)
    )
    headline = _by_key(groups.with_events)[
        "restriction_outside_swr_full_minus_swr_per_spike"
    ]
    assert "where SWR detection was observed" in headline.description


def test_resolved_content_values_name_the_share_with_no_swr_by_width():
    """The analysis test's events (``test_resolved_content``): among the
    resolved events 5/12 have no SWR, among the wide ones 3/4."""
    from tests.analysis.event_properties.test_resolved_content import (
        _events_and_windows,
    )

    values = _by_key(builder.resolved_content_values(*_events_and_windows(), "s"))
    numbers = {key: value.value for key, value in values.items()}
    assert numbers["non_swr_fraction_animal_mean_resolved"] == pytest.approx(5 / 12)
    assert numbers["non_swr_fraction_animal_min_resolved"] == pytest.approx(1 / 3)
    assert numbers["non_swr_fraction_animal_mean_wide"] == pytest.approx(3 / 4)
    assert numbers["nonlocal_events_resolved"] == 6
    assert numbers["nonlocal_events_wide"] == 3
    assert numbers["resolved_share_animal_mean"] == pytest.approx(31 / 48)
    assert numbers["nonlocal_events_without_width"] == 1
    assert len(values) == 2 * 4 + 4
    assert {v.questions for v in values.values()} == {("2b.2",)}
    assert {v.figures for v in values.values()} == {()}
    assert "200 cm" in values["non_swr_fraction_animal_mean_resolved"].description
