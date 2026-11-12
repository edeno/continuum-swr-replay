"""The events summary's builder functions: each statistic reaches the key that names it.

Each test calls one builder function on a hand-made input whose statistics all
differ, so a value filed under a neighbouring key (the other class, the other
bound, the other detector) fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.reported import events as builder
from continuum_swr_replay.reported.values import Value
from tests.analysis.inventory_comparison.test_inventory_nonlocal_comparison import _write_run

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
        "non_swr_fraction_animal_min": pytest.approx(0.4),
        "non_swr_fraction_animal_max": pytest.approx(0.75),
        "animals": 3,
        "epochs": 4,
        # a and c (2 non-SWR against 1 and 1); in c they are half, not more
        "animals_non_swr_plurality": 2,
        "animals_non_swr_majority": 1,
    }


def test_boundary_offset_values_are_in_milliseconds_start_and_end_apart():
    values = _by_key(builder.boundary_offset_values(_events(), "s"))
    assert values["swr_overlapping_events"].value == 6
    # starts 10, 20, 30, 50, 60, 300 ms; ends -50, -40, -30, -20, -10, 900 ms
    assert values["start_offset_median_ms"].value == pytest.approx(40.0)
    assert values["end_offset_median_ms"].value == pytest.approx(-25.0)
    assert values["start_offset_median_ms"].unit == "ms"


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
    )


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


def test_left_out_values_name_each_covering_and_each_rate(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "report")
    numbers = _numbers(
        builder.left_out_values(tmp_path / "run", tmp_path / "report", "s")
    )
    assert numbers == {
        "rest_hours": 2.0,
        "nonlocal_time_s": 2.0,
        "kay_time_s": 1.5,
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
        "nonlocal_events_outside_kay_per_hour": 0.5,
        "nonlocal_events_outside_widloski": 2,
        "nonlocal_events_outside_widloski_fraction": pytest.approx(2 / 3),
        "nonlocal_events_outside_widloski_per_hour": 1.0,
        "nonlocal_events_outside_trigger": 4,
        "nonlocal_events_outside_trigger_fraction": 0.5,
        "nonlocal_events_outside_trigger_per_hour": 2.0,
    }


def _signal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One animal, one epoch, two non-SWR and two SWR events. Signal number k
    (theta 0, ripple 1, MUA 2, slow gamma 3, fast gamma 4) has non-SWR values
    10k + 1 and 10k + 2, SWR values 10k + 3 and 10k + 7, a running level of
    10k + 4 and a quiet-rest level of 10k + 1.8."""
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
        assert own == pytest.approx(expected), signal
    assert values["ripple_median_swr"].figures == ("3b",)
    assert values["theta_median_swr"].figures == ()


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
    # the remote groups' values, and no other, also answer the remote question
    with_remote_question = {key for key, v in values.items() if "2c.7" in v.questions}
    assert with_remote_question == {
        key for key in values if key.startswith(("swr_remote_", "non_swr_remote_"))
    }
    assert values["baseline_dll_per_spike_median"].questions == ("2c.3", "2c.5")
    assert values["non_swr_close_dll_per_spike_median"].questions == ("2c.3",)
    assert values["non_swr_close_spikes_median"].questions == ("2c.2",)
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
    animals = list(analysed_animals())
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


def test_class_step_values_name_the_properties_with_the_most_steps():
    animals = list(analysed_animals())
    with_step = {"duration": 1, "extent": 3, "speed": 3, "evidence": 2}
    evidence_shares = [0.5, -0.1, 0.3, 0.2, 0.9]
    rows = []
    for prop in [*builder._OTHER_STEP_PROPERTIES, "evidence"]:
        for i, animal in enumerate(animals):
            share = evidence_shares[i] if prop == "evidence" else -1.0
            rows.append((prop, animal, i < with_step.get(prop, 0), share))
    steps = pd.DataFrame(
        rows, columns=["property", "animal", "has_step", "step_share_overall"]
    ).assign(
        n_knots=builder.STEP_KNOTS,
        power="criterion",
        window_z=builder.STEP_WINDOW_Z,
        covariate="none",
    )
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
    events = pd.DataFrame({"animal": "a", "is_swr": [False, False, True]})
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


def test_content_occupancy_values_name_each_group_and_place():
    occupancy = pd.DataFrame(
        {
            "epoch_id": ["e1", "e2", "e1", "e3", "e3", "e4", "e4"],
            "animal": ["a", "a", "a", "b", "b", "c", "c"],
            "group": ["SWR", "SWR", "nonSWR", "SWR", "nonSWR", "SWR", "nonSWR"],
            "n": [10, 30, 20, 10, 10, 10, 10],
            "remote_well": [0.5, 0.1, 0.4, 0.2, 0.1, 0.1, 0.5],
            "own_well": [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.2],
            "remote_path": [0.2, 0.4, 0.2, 0.3, 0.4, 0.4, 0.2],
            "near_off": [0.2, 0.4, 0.2, 0.2, 0.4, 0.4, 0.1],
        }
    )
    assert _numbers(builder.content_occupancy_values(occupancy, "s")) == {
        "occupancy_epochs": 4,
        "swr_content_remote_well_fraction": pytest.approx(11 / 60),
        "swr_content_own_well_fraction": pytest.approx(8 / 60),
        "swr_content_at_well_fraction": pytest.approx(19 / 60),
        "swr_content_remote_path_fraction": pytest.approx(21 / 60),
        "swr_content_near_off_fraction": pytest.approx(20 / 60),
        "non_swr_content_remote_well_fraction": pytest.approx(14 / 40),
        "non_swr_content_own_well_fraction": pytest.approx(7 / 40),
        "non_swr_content_at_well_fraction": pytest.approx(21 / 40),
        "non_swr_content_remote_path_fraction": pytest.approx(10 / 40),
        "non_swr_content_near_off_fraction": pytest.approx(9 / 40),
        # a: non-SWR 0.6 at a well against SWR 0.3; b: 0.2 against 0.5; c: 0.7
        # against 0.2
        "animals_non_swr_more_at_well": 2,
    }


def test_the_values_read_from_another_output_have_a_builder_of_their_own():
    """The summary pends the values of a stale output by their builder: the
    gamma pass's are ``gamma_z_values``', the start-distance output's
    ``start_distance_values``'; the event table's are in neither."""
    gamma = _by_key(builder.gamma_z_values(*_signal_tables(), "s"))
    assert len(gamma) == 16
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
            tmp_path / epoch / builder.SPIKE_EVIDENCE_METRICS
        )
    metrics = builder.spike_evidence_metrics(tmp_path, n_epochs=2)
    assert list(metrics["epoch_id"]) == ["a1", "a1", "b1", "b1"]
    with pytest.raises(ValueError, match="holds 2 epochs' metrics, not 3"):
        builder.spike_evidence_metrics(tmp_path, n_epochs=3)
    # a table that names another epoch than its folder's
    pd.DataFrame({"epoch_id": ["a1"], "group": "baseline"}).to_parquet(
        tmp_path / "b1" / builder.SPIKE_EVIDENCE_METRICS
    )
    with pytest.raises(ValueError, match="holds epochs \\['a1'\\]"):
        builder.spike_evidence_metrics(tmp_path, n_epochs=2)
