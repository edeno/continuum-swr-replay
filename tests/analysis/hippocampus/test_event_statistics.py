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
    assert (result["n_swr_contained"], result["n_swr_partial"]) == (4, 2)
    assert result["non_swr_fraction"] == pytest.approx(5 / 11)
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
    result = stats.boundary_offset_statistics(offsets)
    assert result["n_overlapping"] == 4
    assert result["start_offset_median_s"] == pytest.approx(0.015)
    assert result["end_offset_median_s"] == pytest.approx(0.04)
    # events none of which overlaps an SWR: no row, and the two columns
    none = stats.boundary_offsets(EVENTS[EVENTS["overlap_class"] == "non_swr"])
    assert none.empty
    assert list(none.columns) == ["start_offset_s", "end_offset_s"]


def test_boundary_offsets_refuse_an_overlapping_event_without_its_swr():
    broken = EVENTS.copy()
    broken.loc[3, "matched_swr_end"] = np.nan
    with pytest.raises(ValueError, match="no matched SWR bounds"):
        stats.boundary_offsets(broken)


def _signal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Animal a has three events of each class in epoch e1 and one in e2, so the
    median over its events of the epoch levels (e1's) is neither their mean nor
    the median of its two epochs' levels, and its median event value lies
    between the three."""
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
        ("a", "e1", False, 1.1),
        ("a", "e1", False, 1.3),
        ("a", "e1", False, 0.8),
        ("a", "e2", False, 1.2),
        ("b", "e1", False, -1.0),
        ("b", "e1", False, 5.0),
        ("b", "e1", False, 1.0),
        ("a", "e1", True, -0.8),
        ("a", "e1", True, -0.9),
        ("a", "e1", True, -0.7),
        ("a", "e2", True, -0.85),
        ("b", "e1", True, 0.0),
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
    # -1, 0.8, 1, 1.1, 1.2, 1.3, 5: the median is 1.1 (the mean is 1.34)
    assert non_swr["median"] == pytest.approx(1.1)
    # a: event median 1.15 against the running levels 1, 1, 1, 2 of its events'
    # epochs, median 1.0: not below (it is below their mean, 1.25, and below
    # 1.5, the median of a's two epochs). b: 1.0 against 1.0: equal, which is
    # not below.
    assert non_swr["n_animals_below_running"] == 0
    # a: 1.15 against rest -1.0; b: 1.0 against -0.5
    assert non_swr["n_animals_above_rest"] == 2
    # 1.1, 1.3 and 5.0 reach their epoch's 1.0, and b's 1.0 equals it, which
    # reaches it; 0.8 and -1.0 do not, and 1.2 does not reach e2's 2.0
    assert non_swr["reaches_running_fraction"] == pytest.approx(4 / 7)
    swr = result.loc[("theta", True)]
    # -0.9, -0.85, -0.8, -0.7, 0.0
    assert swr["median"] == pytest.approx(-0.8)
    assert swr["n_animals_below_running"] == 2
    # a: event median -0.825 against the rest levels -1, -1, -1, 0 of its events'
    # epochs, median -1.0: above (it is not above their mean, -0.75, nor -0.5,
    # the median of a's two epochs). b: 0.0 against -0.5: above.
    assert swr["n_animals_above_rest"] == 2
    assert swr["reaches_running_fraction"] == 0.0


def test_event_signal_statistics_refuse_an_epoch_without_a_reference():
    events, reference = _signal_tables()
    signals = {"theta": "theta_z", "ripple": "ripple_power_z"}
    events = events.assign(ripple_power_z=1.0)
    reference = reference.assign(ripple_running_median=2.0, ripple_rest_median=0.0)
    assert len(stats.event_signal_statistics(events, reference, signals)) == 4
    # b's epoch is not in the reference at all: its 4 events have no level
    with pytest.raises(ValueError, match=r"theta_running_median +4"):
        stats.event_signal_statistics(events, reference.iloc[:2], signals)
    # the rest level, and a signal other than theta, are checked as well: the
    # first epoch holds 6 events
    for column in (
        "theta_rest_median",
        "ripple_running_median",
        "ripple_rest_median",
    ):
        broken = reference.copy()
        broken.loc[0, column] = np.nan
        with pytest.raises(ValueError, match=rf"{column} +6"):
            stats.event_signal_statistics(events, broken, signals)


def test_event_signal_statistics_refuse_an_event_without_a_value():
    events, reference = _signal_tables()
    events.loc[0, "theta_z"] = np.nan
    with pytest.raises(ValueError, match=r"theta_z +1"):
        stats.event_signal_statistics(events, reference, {"theta": "theta_z"})


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


def test_rest_matched_theta_statistics_average_events_not_windows():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "is_swr": [False, False, False, True],
            # a's first event equals the running level, which is theta-like
            "theta_z": [1.0, 0.0, 0.0, 5.0],
            "theta_running_median": 1.0,
            "rest_windows": [10, 5, 10, 10],
            "rest_theta_like_share": [0.2, 0.4, 0.1, 0.3],
        }
    )
    result = stats.rest_matched_theta_statistics(events)
    # the mean of the three events' shares; over their 25 windows it would be 0.2
    assert result.loc[False, "rest_theta_like_fraction"] == pytest.approx(0.7 / 3)
    assert result.loc[True, "rest_theta_like_fraction"] == pytest.approx(0.3)
    # a: half its events are theta-like against 0.3 of its windows; b: none against 0.1
    assert result.loc[False, "n_animals_above_rest"] == 1
    assert result.loc[True, "n_animals_above_rest"] == 1
    with pytest.raises(ValueError, match="1 events have no duration-matched"):
        stats.rest_matched_theta_statistics(events.assign(rest_windows=[10, 0, 10, 10]))
    # a missing value is refused, not read as "not theta-like" or left out of a mean
    for column in (
        "theta_z",
        "theta_running_median",
        "rest_windows",
        "rest_theta_like_share",
    ):
        broken = events.astype({"rest_windows": float})
        broken.loc[1, column] = np.nan
        with pytest.raises(ValueError, match=rf"{column} +1"):
            stats.rest_matched_theta_statistics(broken)


def _spike_metrics() -> pd.DataFrame:
    """Six quiet windows and six remote events in five epochs of three animals;
    no median equals its mean, and more epochs than animals are above baseline."""
    return pd.DataFrame(
        {
            "group": ["baseline"] * 6 + ["remote"] * 6,
            "animal": ["a", "a", "b", "b", "c", "c"] * 2,
            "epoch_id": ["a1", "a2", "b1", "b1", "c1", "c2"] * 2,
            "n_spikes": [10, 20, 30, 40, 50, 90] + [1, 2, 3, 4, 5, 30],
            "n_tetrodes": [1, 1, 2, 2, 2, 5] + [3, 3, 4, 4, 5, 8],
            "dll_per_spike": [0.0, 1.0, 2.0, 4.0, 0.0, 10.0]
            + [0.0, 1.0, 4.0, 6.0, 1.0, 3.0],
            "dist_to_gpi_peak": [5.0] * 6 + [10.0, 20.0, 30.0, 40.0, 50.0, 150.0],
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
    with pytest.raises(ValueError, match="5 epochs have no 'quiet' dll_per_spike"):
        stats.spike_evidence_statistics(_spike_metrics(), "quiet")


def test_spike_evidence_statistics_compare_each_epoch_and_animal_with_its_baseline():
    metrics = _spike_metrics()
    result = stats.spike_evidence_statistics(metrics, "baseline")
    remote = result.loc["remote"]
    assert remote["n_scored"] == 6
    assert remote["spikes_median"] == 3.5  # the mean is 7.5
    assert remote["tetrodes_median"] == 4.0  # the mean is 4.5
    assert remote["dll_per_spike_median"] == 2.0  # the mean is 2.5
    assert remote["gpi_distance_median_cm"] == 35.0  # the mean is 50
    assert remote["n_epochs_scored"] == 5
    # a1 and a2 equal their baselines (0.0, 1.0), which is not above; b1: 5.0
    # against 3.0; c1: 1.0 against 0.0; c2: 3.0 against 10.0
    assert remote["n_epochs_above_baseline"] == 2
    # a: 0.5 against 0.5 (not above); b: 5.0 against 3.0; c: 2.0 against 5.0
    assert remote["n_animals_above_baseline"] == 1
    baseline = result.loc["baseline"]
    assert baseline["n_scored"] == 6
    assert baseline["spikes_median"] == 35.0
    assert baseline["tetrodes_median"] == 2.0
    assert baseline["dll_per_spike_median"] == 1.5
    assert baseline["gpi_distance_median_cm"] == 5.0
    assert baseline["n_epochs_scored"] == 5
    assert np.isnan(baseline["n_epochs_above_baseline"])
    assert np.isnan(baseline["n_animals_above_baseline"])


STEP_COLUMNS = [
    "property",
    "animal",
    "n_knots",
    "power",
    "window_z",
    "covariate",
    "has_step",
    "step_share_overall",
]


def _steps() -> pd.DataFrame:
    rows = []
    for prop, with_step, shares in (
        ("evidence", [True, True, False], [0.5, -0.1, 0.3]),
        ("duration", [True, False, False], [0.0, 0.0, 0.0]),
        ("speed", [True, False, False], [0.0, 0.0, 0.0]),
    ):
        for animal, (step, share) in enumerate(zip(with_step, shares, strict=True)):
            rows.append((prop, animal, 8, "criterion", 1.5, "none", step, share))
    # other settings, which the main statistics leave out: each would add a step
    rows += [
        ("evidence", 2, 4, "criterion", 1.5, "none", True, 9.0),
        ("evidence", 2, 8, "event", 1.5, "none", True, 9.0),
        ("evidence", 2, 8, "criterion", 1.0, "none", True, 9.0),
        ("evidence", 2, 8, "criterion", 1.5, "log_duration", True, 9.0),
    ]
    return pd.DataFrame(rows, columns=STEP_COLUMNS)


def test_threshold_step_statistics_use_the_main_setting_only():
    result = stats.threshold_step_statistics(
        _steps(), n_knots=8, window_z=1.5, n_animals=3
    )
    assert result.loc["evidence", "n_animals_with_step"] == 2
    assert result.loc["evidence", "n_animals_same_sign"] == 2
    # the median of 0.5, -0.1 and 0.3 (their mean is 0.23)
    assert result.loc["evidence", "step_share_median"] == pytest.approx(0.3)
    assert result.loc["duration", "n_animals_same_sign"] == 0
    assert stats.most_steps(result, ["duration", "speed"]) == (1, ["duration", "speed"])
    assert stats.most_steps(result, ["evidence", "speed"]) == (2, ["evidence"])
    with pytest.raises(ValueError, match="one fit for each of 5 animals"):
        stats.threshold_step_statistics(_steps(), n_knots=8, window_z=1.5, n_animals=5)


def test_threshold_step_statistics_refuse_an_animal_fitted_twice():
    steps = _steps()
    # duration: animal 0 twice and animal 2 not at all, still three fits
    steps.loc[(steps.property == "duration") & (steps.animal == 2), "animal"] = 0
    with pytest.raises(ValueError, match="one fit for each of 3 animals"):
        stats.threshold_step_statistics(steps, n_knots=8, window_z=1.5, n_animals=3)


def _distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["a", "b", "c", "pooled", "a"],
            "power": ["peak", "peak", "peak", "peak", "mean"],
            "dip_p": [0.5, 0.01, 0.05, 0.9, 0.9],
            "main_mode_z": [1.0, 2.0, 1.5, 9.0, 0.0],
        }
    )


def test_peak_ripple_mode_statistics_leave_out_the_pooled_row_and_other_powers():
    result = stats.peak_ripple_mode_statistics(_distribution(), n_animals=3)
    # c's p equals the level, which does not reject one mode
    assert result == {
        "n_animals_unimodal": 2,
        "mode_animal_min": 1.0,
        "mode_animal_max": 2.0,
    }


def test_peak_ripple_mode_statistics_refuse_a_missing_or_repeated_animal_or_p():
    with pytest.raises(ValueError, match="one row for each of 4 animals"):
        stats.peak_ripple_mode_statistics(_distribution(), n_animals=4)
    repeated = _distribution()
    repeated.loc[2, "animal"] = "a"
    with pytest.raises(ValueError, match="one row for each of 3 animals"):
        stats.peak_ripple_mode_statistics(repeated, n_animals=3)
    for column in ("dip_p", "main_mode_z"):
        missing = _distribution()
        missing.loc[1, column] = np.nan
        with pytest.raises(ValueError, match=f"no {column}"):
            stats.peak_ripple_mode_statistics(missing, n_animals=3)


def test_class_medians_count_animals_in_the_stated_direction():
    events = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b", "b", "b", "c", "c", "d", "d"],
            "is_swr": [False, False, True, False, True, True, False, True, False, True],
            "extent": [1.0, 3.0, 5.0, 8.0, 4.0, np.nan, 1.0, 9.0, 7.0, 7.0],
        }
    )
    lower = stats.class_medians(events, "extent", "lower")
    # non-SWR: 1, 1, 3, 7, 8; SWR: 4, 5, 7, 9
    assert lower["median_non_swr"] == 3.0
    assert lower["median_swr"] == 6.0
    # a: 2 < 5; b: 8 > 4; c: 1 < 9; d: 7 against 7, in neither direction
    assert lower["n_animals_in_direction"] == 2
    higher = stats.class_medians(events, "extent", "higher")
    assert higher["n_animals_in_direction"] == 1
    with pytest.raises(ValueError, match="direction 'above'"):
        stats.class_medians(events, "extent", "above")


def test_content_occupancy_shares_weigh_sessions_by_their_time_bins():
    occupancy = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b", "b", "c", "c"],
            "group": ["SWR", "SWR", "nonSWR", "SWR", "nonSWR", "SWR", "nonSWR"],
            "n": [10, 30, 20, 10, 10, 10, 10],
            "remote_well": [0.5, 0.1, 0.4, 0.2, 0.1, 0.1, 0.5],
            "own_well": [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.2],
            "remote_path": [0.2, 0.4, 0.2, 0.3, 0.4, 0.4, 0.2],
            "near_off": [0.2, 0.4, 0.2, 0.2, 0.4, 0.4, 0.1],
        }
    )
    shares = stats.content_occupancy_shares(occupancy)
    assert shares.loc["SWR", "remote_well"] == pytest.approx((5 + 3 + 2 + 1) / 60)
    assert shares.loc["SWR", "at_well"] == pytest.approx((6 + 6 + 5 + 2) / 60)
    assert shares.loc["nonSWR", "near_off"] == pytest.approx((4 + 4 + 1) / 40)
    assert list(shares.columns) == list(stats.OCCUPANCY_PLACES)
    # a: non-SWR 0.6 at a well against SWR 0.3; b: 0.2 against 0.5; c: 0.7
    # against 0.2
    assert stats.n_animals_non_swr_more_at_well(occupancy) == 2


def test_content_occupancy_shares_refuse_a_session_with_time_bins_and_no_share():
    occupancy = pd.DataFrame(
        {
            "animal": ["a", "a", "a"],
            "group": ["SWR", "SWR", "SWR"],
            "n": [10, 30, 0],
            "remote_well": [0.5, np.nan, np.nan],
            "own_well": [0.1, 0.1, np.nan],
            "remote_path": [0.2, 0.4, np.nan],
            "near_off": [0.2, 0.4, np.nan],
        }
    )
    with pytest.raises(ValueError, match="1 sessions with time bins have no share"):
        stats.content_occupancy_shares(occupancy)
    # a session with no time bin has no share to give, and weighs nothing
    occupancy.loc[1, "remote_well"] = 0.1
    shares = stats.content_occupancy_shares(occupancy)
    assert shares.loc["SWR", "remote_well"] == pytest.approx((5 + 3) / 40)
