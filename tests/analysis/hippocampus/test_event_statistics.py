"""The statistics of the immobile non-local events, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus import event_statistics as stats


def _events(rows: list[tuple]) -> pd.DataFrame:
    """Events as (animal, epoch, overlap_class, start, end, swr_start, swr_end)."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "animal",
            "epoch",
            "overlap_class",
            "event_start_s",
            "event_end_s",
            "matched_swr_start",
            "matched_swr_end",
        ],
    )
    return frame.assign(
        session_date="d1",
        whole_event_speed_eligible=True,
        is_swr=frame["overlap_class"] != "non_swr",
    )


EVENTS = _events(
    [
        # animal a: 3 of 4 events are non-SWR
        ("a", "e1", "non_swr", 0.0, 0.1, np.nan, np.nan),
        ("a", "e1", "non_swr", 1.0, 1.1, np.nan, np.nan),
        ("a", "e2", "non_swr", 2.0, 2.1, np.nan, np.nan),
        ("a", "e2", "swr_contained", 3.02, 3.08, 3.0, 3.1),
        # animal b: non-SWR is the largest class but not a majority
        ("b", "e1", "non_swr", 0.0, 0.1, np.nan, np.nan),
        ("b", "e1", "non_swr", 1.0, 1.1, np.nan, np.nan),
        ("b", "e1", "swr_contained", 2.01, 2.05, 2.0, 2.1),
        ("b", "e1", "swr_partial", 2.95, 3.20, 3.0, 3.1),
        ("b", "e1", "swr_partial", 4.04, 4.30, 4.0, 4.1),
    ]
)


def test_overlap_class_statistics_count_events_animals_and_epochs():
    result = stats.overlap_class_statistics(EVENTS)
    assert result["n_events"] == 9
    assert (
        result["n_non_swr"],
        result["n_swr_contained"],
        result["n_swr_partial"],
    ) == (
        5,
        2,
        2,
    )
    assert result["non_swr_fraction"] == pytest.approx(5 / 9)
    assert result["non_swr_fraction_animal_min"] == pytest.approx(2 / 5)
    assert result["non_swr_fraction_animal_max"] == pytest.approx(3 / 4)
    assert (result["n_animals"], result["n_epochs"]) == (2, 3)
    # b's two non-SWR events tie its two partial ones, so only a has a plurality
    assert result["n_animals_non_swr_plurality"] == 1
    assert result["n_animals_non_swr_majority"] == 1


def test_overlap_class_statistics_count_an_absent_class_as_a_share_of_zero():
    # animal c has SWR-contained events only: its non-SWR share is 0, the lowest
    only_swr = _events(
        [
            ("c", "e1", "swr_contained", 0.02, 0.08, 0.0, 0.1),
            ("c", "e1", "swr_contained", 1.02, 1.08, 1.0, 1.1),
        ]
    )
    result = stats.overlap_class_statistics(pd.concat([EVENTS, only_swr]))
    assert result["non_swr_fraction_animal_min"] == 0.0
    assert result["non_swr_fraction_animal_max"] == pytest.approx(3 / 4)
    assert result["n_animals"] == 3
    assert result["n_animals_non_swr_plurality"] == 1
    assert result["n_animals_non_swr_majority"] == 1


@pytest.mark.parametrize("overlap_class", [None, "swr"])
def test_overlap_class_statistics_refuse_an_event_of_no_class(overlap_class):
    unclassed = EVENTS.copy()
    unclassed.loc[0, "overlap_class"] = overlap_class
    with pytest.raises(ValueError, match="1 events have an overlap_class that is"):
        stats.overlap_class_statistics(unclassed)


def test_overlap_class_statistics_refuse_an_event_that_is_not_immobile():
    moving = EVENTS.assign(whole_event_speed_eligible=[False] + [True] * 8)
    with pytest.raises(ValueError, match="not whole-event immobile"):
        stats.overlap_class_statistics(moving)


def test_boundary_offsets_are_event_bound_minus_swr_bound():
    offsets = stats.boundary_offsets(EVENTS)
    assert len(offsets) == 4
    np.testing.assert_allclose(offsets["start_offset_s"], [0.02, 0.01, -0.05, 0.04])
    np.testing.assert_allclose(offsets["end_offset_s"], [-0.02, -0.05, 0.10, 0.20])
    result = stats.boundary_offset_statistics(EVENTS)
    assert result["n_overlapping"] == 4
    assert result["start_offset_median_s"] == pytest.approx(0.015)
    assert result["end_offset_median_s"] == pytest.approx(0.04)


def test_boundary_offsets_refuse_an_overlapping_event_without_its_swr():
    broken = EVENTS.copy()
    broken.loc[3, "matched_swr_end"] = np.nan
    with pytest.raises(ValueError, match="no matched SWR bounds"):
        stats.boundary_offsets(broken)


def _signal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Animal a has three events of each class in epoch e1 and one in e2, so the
    median over its events of the epoch levels (e1's) is not the median of its
    two epochs' levels (their mean)."""
    reference = pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "session_date": "d1",
            "epoch": ["e1", "e2", "e1"],
            "theta_running_median": [1.0, 2.0, 1.0],
            "theta_rest_median": [-1.0, 0.0, -0.5],
        }
    )
    rows = [
        ("a", "e1", False, 1.2),
        ("a", "e1", False, 1.4),
        ("a", "e1", False, 0.8),
        ("a", "e2", False, 1.3),
        ("b", "e1", False, -1.0),
        ("b", "e1", False, 5.0),
        ("b", "e1", False, 0.0),
        ("a", "e1", True, -0.7),
        ("a", "e1", True, -0.8),
        ("a", "e1", True, -0.6),
        ("a", "e2", True, -0.75),
        ("b", "e1", True, 0.0),
        ("b", "e1", True, np.nan),
    ]
    events = pd.DataFrame(rows, columns=["animal", "epoch", "is_swr", "theta_z"])
    return events.assign(session_date="d1"), reference


def test_reference_medians_are_medians_over_epochs():
    _, reference = _signal_tables()
    levels = stats.reference_medians(reference, ["theta"])
    assert levels.loc["theta", "running"] == 1.0
    assert levels.loc["theta", "rest"] == -0.5


def test_event_signal_statistics_compare_each_event_with_its_own_epoch():
    events, reference = _signal_tables()
    result = stats.event_signal_statistics(events, reference, {"theta": "theta_z"})
    non_swr = result.loc[("theta", False)]
    # -1, 0, 0.8, 1.2, 1.3, 1.4, 5: the median is 1.2 (the mean is 1.24)
    assert non_swr["median"] == pytest.approx(1.2)
    # a: event median 1.25 against the running levels 1, 1, 1, 2 of its events'
    # epochs, median 1.0: not below (it is below 1.5, the median of a's two
    # epochs). b: 0.0 against 1.0: below.
    assert non_swr["n_animals_below_running"] == 1
    # a: 1.25 against rest -1.0; b: 0.0 against -0.5
    assert non_swr["n_animals_above_rest"] == 2
    # 1.2, 1.4 and 5.0 reach their epoch's 1.0; 0.8, -1.0 and 0.0 do not, and
    # 1.3 does not reach e2's 2.0
    assert non_swr["reaches_running_fraction"] == pytest.approx(3 / 7)
    # the SWR event with no theta value is left out
    swr = result.loc[("theta", True)]
    # -0.8, -0.75, -0.7, -0.6, 0.0
    assert swr["median"] == pytest.approx(-0.7)
    assert swr["n_animals_below_running"] == 2
    # a: event median -0.725 against the rest levels -1, -1, -1, 0 of its events'
    # epochs, median -1.0: above (it is not above -0.5, the median of a's two
    # epochs). b: 0.0 against -0.5: above.
    assert swr["n_animals_above_rest"] == 2
    assert swr["reaches_running_fraction"] == 0.0


def test_event_signal_statistics_refuse_an_epoch_without_a_reference():
    events, reference = _signal_tables()
    with pytest.raises(ValueError, match="no running reference"):
        stats.event_signal_statistics(events, reference.iloc[:2], {"theta": "theta_z"})


def test_power_change_statistics_count_animals_above_rest():
    reference = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "slow_gamma_db_non_swr": [1.0, 3.0, -1.0, -2.0],
            "slow_gamma_db_swr": [0.5, 0.5, 0.5, 0.5],
        }
    )
    result = stats.power_change_statistics(reference, ["slow_gamma"])
    assert result.loc[("slow_gamma", False), "median"] == 0.0
    assert result.loc[("slow_gamma", False), "n_animals_above_rest"] == 1
    assert result.loc[("slow_gamma", True), "n_animals_above_rest"] == 2
    reference.loc[0, "slow_gamma_db_swr"] = np.nan
    with pytest.raises(ValueError, match="1 epochs have no slow_gamma_db_swr"):
        stats.power_change_statistics(reference, ["slow_gamma"])


def test_rest_matched_theta_statistics_weigh_windows_by_event():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "is_swr": [False, False, False, True],
            "theta_z": [2.0, 0.0, 0.0, 5.0],
            "theta_running_median": 1.0,
            "rest_windows": 10,
            "rest_theta_like_share": [0.2, 0.4, 0.1, 0.3],
        }
    )
    result = stats.rest_matched_theta_statistics(events)
    assert result.loc[False, "rest_theta_like_fraction"] == pytest.approx(0.7 / 3)
    # a: half its events are theta-like against 0.3 of its windows; b: none against 0.1
    assert result.loc[False, "n_animals_above_rest"] == 1
    assert result.loc[True, "n_animals_above_rest"] == 1
    with pytest.raises(ValueError, match="1 events have no duration-matched"):
        stats.rest_matched_theta_statistics(events.assign(rest_windows=[10, 0, 10, 10]))


def _spike_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["baseline"] * 4 + ["remote"] * 4,
            "animal": ["a", "a", "b", "b"] * 2,
            "epoch_id": ["a1", "a2", "b1", "b1"] * 2,
            "n_spikes": [10, 20, 30, 40, 1, 2, 3, 4],
            "n_tetrodes": [1, 1, 2, 2, 3, 3, 4, 4],
            "dll_per_spike": [0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 4.0, 6.0],
            "dist_to_gpi_peak": [5.0, 5.0, 5.0, 5.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


def test_spike_evidence_statistics_refuse_an_epoch_without_its_baseline():
    metrics = _spike_metrics()
    # epoch a2 has a remote event and no quiet window
    without = metrics.drop(index=1)
    with pytest.raises(ValueError, match="1 epochs have no 'baseline' dll_per_spike"):
        stats.spike_evidence_statistics(without, "baseline")
    # its only quiet window has no value
    metrics.loc[1, "dll_per_spike"] = np.nan
    with pytest.raises(ValueError, match="1 epochs have no 'baseline' dll_per_spike"):
        stats.spike_evidence_statistics(metrics, "baseline")
    # a group name the run does not hold
    with pytest.raises(ValueError, match="3 epochs have no 'quiet' dll_per_spike"):
        stats.spike_evidence_statistics(_spike_metrics(), "quiet")


def test_spike_evidence_statistics_compare_each_epoch_and_animal_with_its_baseline():
    metrics = _spike_metrics()
    result = stats.spike_evidence_statistics(metrics, "baseline")
    remote = result.loc["remote"]
    assert remote["n_scored"] == 4
    assert remote["spikes_median"] == 2.5
    assert remote["dll_per_spike_median"] == 2.5
    assert remote["gpi_distance_median_cm"] == 25.0
    assert remote["n_epochs_scored"] == 3
    # a1 and a2 equal their baselines (0.0, 1.0), which is not above; b1: 5.0 > 3.0
    assert remote["n_epochs_above_baseline"] == 1
    # a: 0.5 against 0.5 (not above); b: 5.0 against 3.0
    assert remote["n_animals_above_baseline"] == 1
    assert np.isnan(result.loc["baseline", "n_epochs_above_baseline"])


def _steps() -> pd.DataFrame:
    rows = []
    for prop, with_step, shares in (
        ("evidence", [True, True, False], [0.5, -0.1, 0.2]),
        ("duration", [True, False, False], [0.0, 0.0, 0.0]),
        ("speed", [True, False, False], [0.0, 0.0, 0.0]),
    ):
        for animal, (step, share) in enumerate(zip(with_step, shares, strict=True)):
            rows.append((prop, animal, 8, "criterion", 1.5, "none", step, share))
    # another setting, which the main statistics leave out
    rows.append(("evidence", 0, 4, "criterion", 1.5, "none", True, 9.0))
    return pd.DataFrame(
        rows,
        columns=[
            "property",
            "animal",
            "n_knots",
            "power",
            "window_z",
            "covariate",
            "has_step",
            "step_share_overall",
        ],
    )


def test_threshold_step_statistics_use_the_main_setting_only():
    result = stats.threshold_step_statistics(
        _steps(), n_knots=8, window_z=1.5, n_animals=3
    )
    assert result.loc["evidence", "n_animals_with_step"] == 2
    assert result.loc["evidence", "n_animals_same_sign"] == 2
    assert result.loc["evidence", "step_share_median"] == pytest.approx(0.2)
    assert stats.most_steps(result, ["duration", "speed"]) == (1, ["duration", "speed"])
    with pytest.raises(ValueError, match="not every property has 5 animals"):
        stats.threshold_step_statistics(_steps(), n_knots=8, window_z=1.5, n_animals=5)


def test_peak_ripple_mode_statistics_leave_out_the_pooled_row_and_other_powers():
    distribution = pd.DataFrame(
        {
            "animal": ["a", "b", "pooled", "a"],
            "power": ["peak", "peak", "peak", "mean"],
            "dip_p": [0.5, 0.01, 0.9, 0.9],
            "main_mode_z": [1.0, 2.0, 9.0, 9.0],
        }
    )
    result = stats.peak_ripple_mode_statistics(distribution)
    assert result == {
        "n_animals_unimodal": 1,
        "mode_animal_min": 1.0,
        "mode_animal_max": 2.0,
    }


def test_class_medians_count_animals_in_the_stated_direction():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b", "b", "b", "c", "c"],
            "is_swr": [False, False, True, False, True, True, False, True],
            "extent": [1.0, 3.0, 5.0, 8.0, 4.0, np.nan, 1.0, 9.0],
        }
    )
    lower = stats.class_medians(events, "extent", "lower")
    assert lower["median_non_swr"] == 2.0
    assert lower["median_swr"] == 5.0
    # a: 2 < 5; b: 8 > 4; c: 1 < 9
    assert lower["n_animals_in_direction"] == 2
    higher = stats.class_medians(events, "extent", "higher")
    assert higher["n_animals_in_direction"] == 1


def test_content_occupancy_shares_weigh_sessions_by_their_time_bins():
    occupancy = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b", "b"],
            "group": ["SWR", "SWR", "nonSWR", "SWR", "nonSWR"],
            "n": [10, 30, 20, 10, 10],
            "remote_well": [0.5, 0.1, 0.4, 0.2, 0.1],
            "own_well": [0.1, 0.1, 0.2, 0.3, 0.1],
            "remote_path": [0.2, 0.4, 0.2, 0.3, 0.4],
            "near_off": [0.2, 0.4, 0.2, 0.2, 0.4],
        }
    )
    shares = stats.content_occupancy_shares(occupancy)
    assert shares.loc["SWR", "remote_well"] == pytest.approx((5 + 3 + 2) / 50)
    assert shares.loc["SWR", "at_well"] == pytest.approx((6 + 6 + 5) / 50)
    assert shares.loc["nonSWR", "near_off"] == pytest.approx((4 + 4) / 30)
    assert list(shares.columns) == list(stats.OCCUPANCY_PLACES)
    # a: non-SWR 0.6 at a well against SWR 0.3; b: 0.2 against 0.5
    assert stats.n_animals_non_swr_more_at_well(occupancy) == 1
