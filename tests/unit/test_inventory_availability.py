"""What each detector observed (spec 03, "Observation availability")."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.inventory_comparison.availability import (
    OBSERVATION_DTYPES,
    exposure,
    missing_run_table,
    observation_table,
    sample_runs,
    still_intervals,
    with_window_support,
)
from continuum_swr_replay.analysis.inventory_comparison.intervals import total_duration
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity

SESSION = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
FS = 500.0


def grid(n=1000, start=100.0):
    return start + np.arange(n) / FS


class TestSampleRuns:
    def test_a_complete_trace_is_one_run_from_first_to_last_sample(self):
        time = grid()

        np.testing.assert_array_equal(
            sample_runs(time, np.ones(time.size, bool)), [[time[0], time[-1]]]
        )

    def test_missing_samples_split_the_run_and_are_not_interpolated_over(self):
        time = grid()
        valid = np.ones(time.size, bool)
        valid[400:450] = False

        runs = sample_runs(time, valid)

        np.testing.assert_array_equal(
            runs, [[time[0], time[399]], [time[450], time[-1]]]
        )

    def test_a_gap_in_the_timestamps_splits_the_run(self):
        """Missing timestamps remove coverage just as NaN samples do."""
        time = np.delete(grid(), np.arange(400, 450))

        runs = sample_runs(time, np.ones(time.size, bool))

        assert runs.shape == (2, 2)
        assert runs[0, 1] == time[399] and runs[1, 0] == time[400]

    def test_no_valid_sample_is_no_coverage(self):
        assert sample_runs(grid(), np.zeros(1000, bool)).shape == (0, 2)


class TestWindowSupport:
    def test_a_window_overlapping_a_data_gap_has_no_midpoint_coverage(self):
        """An 80 ms window is evaluable at its midpoint only with its whole
        support; within 40 ms of a gap or an edge it is not."""
        time = grid()
        valid = np.ones(time.size, bool)
        valid[400:450] = False

        observed = with_window_support(sample_runs(time, valid), half_window_s=0.040)

        np.testing.assert_allclose(
            observed,
            [
                [time[0] + 0.040, time[399] - 0.040],
                [time[450] + 0.040, time[-1] - 0.040],
            ],
        )

    def test_a_run_shorter_than_one_window_observes_nothing(self):
        time = grid(30)  # 58 ms

        observed = with_window_support(
            sample_runs(time, np.ones(30, bool)), half_window_s=0.040
        )

        assert observed.shape == (0, 2)


class TestExposure:
    def test_is_observed_time_the_animal_was_still(self):
        time = grid()
        speed = np.zeros(time.size)
        speed[200:300] = 9.0
        speed[600] = np.nan  # an unknown speed is not a still one

        still = still_intervals(time, speed)
        exposed = exposure(sample_runs(time, np.ones(time.size, bool)), still)

        assert exposed.shape == (3, 2)
        assert total_duration(exposed) == pytest.approx(
            (199 + (599 - 300) + (999 - 601)) / FS
        )


class TestObservationTable:
    def test_partitions_the_epoch_into_observed_and_unavailable_stretches(self):
        time = grid()
        valid = np.ones(time.size, bool)
        valid[400:450] = False
        observed = sample_runs(time, valid)

        table = observation_table(
            SESSION,
            detector="Kay",
            run_id="run",
            epoch_bounds=(time[0], time[-1]),
            observed=observed,
            unavailable_reason="missing LFP",
        )

        assert list(table.columns) == list(OBSERVATION_DTYPES)
        assert table.available.tolist() == [True, False, True]
        assert table.reason.tolist() == ["", "missing LFP", ""]
        # the stretches tile the epoch with nothing left over
        np.testing.assert_array_equal(
            table.start_s.to_numpy()[1:], table.end_s.to_numpy()[:-1]
        )
        assert table.start_s.iloc[0] == time[0] and table.end_s.iloc[-1] == time[-1]

    def test_a_run_that_found_nothing_is_still_observed(self):
        """Complete data with zero detections is an observation, not a gap."""
        time = grid()

        table = observation_table(
            SESSION,
            detector="Kay",
            run_id="run",
            epoch_bounds=(time[0], time[-1]),
            observed=sample_runs(time, np.ones(time.size, bool)),
            unavailable_reason="missing LFP",
        )

        assert table.available.tolist() == [True]

    def test_a_missing_run_is_unavailable_throughout(self):
        time = grid()

        table = missing_run_table(
            SESSION,
            detector="Yu",
            run_id="run",
            epoch_bounds=(time[0], time[-1]),
            reason="no Yu inventory for this epoch",
        )

        assert table.available.tolist() == [False]
        assert table.reason.tolist() == ["no Yu inventory for this epoch"]

    def test_observed_time_outside_the_epoch_is_refused(self):
        with pytest.raises(ValueError, match="epoch"):
            observation_table(
                SESSION,
                detector="Kay",
                run_id="run",
                epoch_bounds=(100.0, 101.0),
                observed=np.array([[99.0, 100.5]]),
                unavailable_reason="missing LFP",
            )
