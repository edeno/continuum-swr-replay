"""What each detector observed."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    window_midpoints_s,
)
from continuum_swr_replay.analysis.inventory_comparison.availability import (
    OBSERVATION_DTYPES,
    coverage_from_observations,
    exposure,
    missing_run_table,
    observation_table,
    sample_runs,
    still_intervals,
    window_supported,
)
from continuum_swr_replay.analysis.inventory_comparison.intervals import (
    contains,
    intersect,
    subtract,
    total_duration,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    event_matches,
    overlap_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    combine_inventories,
    inventory_from_intervals,
)

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

    def test_a_non_finite_timestamp_splits_the_run_rather_than_erasing_it(self):
        time = np.array([0.0, 0.002, np.nan, 0.006, 0.008])

        runs = sample_runs(time, np.ones(time.size, bool))

        np.testing.assert_array_equal(runs, [[0.0, 0.002], [0.006, 0.008]])


WINDOW = MemorylessSequenceConfig().window_n_samples


class TestWindowSupport:
    def test_a_window_overlapping_a_data_gap_has_no_midpoint_coverage(self):
        """A window over samples [k, k + 40) is stamped at k + 20, so a sample
        is covered only with 20 valid samples before it and 19 after."""
        time = grid()
        valid = np.ones(time.size, bool)
        valid[400:450] = False

        observed = sample_runs(time, window_supported(time, valid, WINDOW))

        np.testing.assert_array_equal(
            observed, [[time[20], time[380]], [time[470], time[980]]]
        )

    def test_a_gap_in_the_timestamps_inside_a_window_removes_its_support(self):
        time = np.delete(grid(), np.arange(400, 450))

        observed = sample_runs(
            time, window_supported(time, np.ones(time.size, bool), WINDOW)
        )

        np.testing.assert_array_equal(
            observed, [[time[20], time[380]], [time[420], time[-20]]]
        )

    @pytest.mark.parametrize("base_s", [1234.5678, 1_581_500_000.123456])
    def test_covers_every_midpoint_the_detector_stamps_on_a_real_clock(self, base_s):
        """On a clock 1 ppm off nominal, a fixed trim in seconds misses the
        first or last window; counting samples cannot. At the 1.6e9 s of an
        NWB timestamp the float step is 2.4e-7 s, still far inside a sample."""
        time = base_s + np.arange(5000) / FS * (1 + 1e-6)

        observed = sample_runs(
            time, window_supported(time, np.ones(time.size, bool), WINDOW)
        )
        midpoints = window_midpoints_s(time, MemorylessSequenceConfig())

        assert contains(observed, midpoints, midpoints).all()
        np.testing.assert_array_equal(observed, [[midpoints[0], time[-20]]])

    def test_a_run_shorter_than_one_window_observes_nothing(self):
        time = grid(30)  # 58 ms

        supported = window_supported(time, np.ones(30, bool), WINDOW)

        assert not supported.any()


class TestExposure:
    def test_is_observed_time_the_animal_was_still(self):
        time = grid()
        speed = np.zeros(time.size)
        speed[200:300] = 9.0
        speed[600] = np.nan  # an unknown speed is not a still one

        still = still_intervals(time, speed)
        exposed = exposure(sample_runs(time, np.ones(time.size, bool)), still)

        # each still run reaches the moving sample beside it, not the unknown one
        assert exposed.shape == (3, 2)
        assert total_duration(exposed) == pytest.approx(
            (200 + (599 - 299) + (999 - 601)) / FS
        )


class TestStillRunsReachTheNeighbouringSamples:
    """An event is speed-eligible when every sample inside it is still, so a
    still run's exposure reaches the samples on either side of it."""

    def test_a_still_run_reaches_the_moving_samples_beside_it(self):
        time = grid()
        speed = np.zeros(time.size)
        speed[300:600] = 9.0

        np.testing.assert_array_equal(
            still_intervals(time, speed),
            [[time[0], time[300]], [time[599], time[999]]],
        )

    def test_a_lone_still_sample_spans_from_one_moving_neighbour_to_the_other(self):
        time = grid()
        speed = np.full(time.size, 9.0)
        speed[500] = 0.0

        np.testing.assert_array_equal(
            still_intervals(time, speed), [[time[499], time[501]]]
        )

    def test_two_runs_one_moving_sample_apart_are_one_interval(self):
        """Both reach the moving sample; events holding it stay ineligible."""
        time = grid()
        speed = np.zeros(time.size)
        speed[500] = 9.0

        np.testing.assert_array_equal(
            still_intervals(time, speed), [[time[0], time[999]]]
        )

    @pytest.mark.parametrize("break_kind", ["gap", "nan speed", "nan time"])
    def test_a_run_does_not_reach_across_a_break(self, break_kind):
        """Still, then moving, then still, with a break at each change."""
        time = grid()
        speed = np.zeros(time.size)
        speed[301:600] = 9.0
        expected = [[time[0], time[299]], [time[601], time[999]]]
        breaks = [300, 600]
        if break_kind == "gap":
            keep = np.ones(time.size, bool)
            keep[breaks] = False
            time, speed = time[keep], speed[keep]
        elif break_kind == "nan speed":
            speed[breaks] = np.nan
        else:
            time[breaks] = np.nan

        np.testing.assert_array_equal(still_intervals(time, speed), expected)

    @staticmethod
    def _matches(time, speed, start_s, end_s):
        """One event of detector ``a`` against a detector ``b`` that found
        nothing, both observing the whole trace."""
        inventory, moving = combine_inventories(
            [
                inventory_from_intervals(
                    np.array([start_s]),
                    np.array([end_s]),
                    native_keys=np.arange(1),
                    detector="a",
                    session=SESSION,
                    run_id="run",
                    speed_time_s=time,
                    speed_cm_per_s=speed,
                )
            ]
        )
        observed = np.array([[time[0], time[-1]]])
        still = still_intervals(time, speed)
        coverage = {
            ("sim", 20200101, "02_r1", d): Coverage(
                observed=observed, exposure=exposure(observed, still)
            )
            for d in ("a", "b")
        }
        return moving, event_matches(inventory, overlap_pairs(inventory), coverage)

    def test_an_event_ending_after_the_last_still_sample_is_unmatched(self):
        """Its only samples are still, so it is compared, not left unknown."""
        time = grid()
        speed = np.where(np.arange(time.size) < 500, 0.0, 9.0)

        # 1 ms is half a sample interval: the event ends between two samples
        moving, matches = self._matches(time, speed, time[495], time[499] + 0.001)

        assert moving.empty
        row = matches.set_index("detector").loc["a"]
        assert row.within_shared_exposure
        assert row.membership == "unmatched"

    def test_an_event_starting_before_the_first_still_sample_is_unmatched(self):
        time = grid()
        speed = np.where(np.arange(time.size) < 500, 9.0, 0.0)

        # 1 ms is half a sample interval: the event starts between two samples
        moving, matches = self._matches(time, speed, time[499] + 0.001, time[505])

        assert moving.empty
        row = matches.set_index("detector").loc["a"]
        assert row.within_shared_exposure
        assert row.membership == "unmatched"

    def test_an_event_with_a_moving_sample_is_set_aside_not_counted(self):
        """The reach takes in the moving sample's time, never its events."""
        time = grid()
        speed = np.where(np.arange(time.size) < 500, 0.0, 9.0)

        moving, matches = self._matches(time, speed, time[495], time[500])

        assert len(moving) == 1
        assert matches.empty


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


SESSION = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")


class TestBoundsAndConversion:
    def test_inverted_epoch_bounds_are_refused(self):
        with pytest.raises(ValueError, match="epoch_bounds"):
            missing_run_table(
                SESSION, detector="x", run_id="r", epoch_bounds=(10.0, 0.0), reason="no"
            )
        with pytest.raises(ValueError, match="epoch_bounds"):
            observation_table(
                SESSION,
                detector="x",
                run_id="r",
                epoch_bounds=(10.0, 0.0),
                observed=np.empty((0, 2)),
                unavailable_reason="no",
            )

    def test_a_single_sample_covers_no_time_without_a_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runs = sample_runs(np.array([1.0]), np.array([True]))
            runs_two = sample_runs(np.array([1.0, np.nan]), np.array([True, True]))

        assert runs.shape == (0, 2)
        assert runs_two.shape == (0, 2)

    def test_coverage_comes_from_the_observation_tables_themselves(self):
        """One origin for both representations of what a detector observed;
        a missing run keeps its place on the roster with empty coverage."""
        still = {("sim", 20200101, "02_r1"): np.array([[0.0, 30.0], [40.0, 100.0]])}
        observations = pd.concat(
            [
                observation_table(
                    SESSION,
                    detector="x",
                    run_id="r",
                    epoch_bounds=(0.0, 100.0),
                    observed=np.array([[0.0, 50.0], [60.0, 100.0]]),
                    unavailable_reason="lfp missing",
                ),
                missing_run_table(
                    SESSION,
                    detector="y",
                    run_id="r",
                    epoch_bounds=(0.0, 100.0),
                    reason="no run",
                ),
            ],
            ignore_index=True,
        )

        coverage = coverage_from_observations(observations, still)

        x = coverage[("sim", 20200101, "02_r1", "x")]
        y = coverage[("sim", 20200101, "02_r1", "y")]
        np.testing.assert_allclose(x.observed, [[0.0, 50.0], [60.0, 100.0]])
        np.testing.assert_allclose(
            x.exposure, [[0.0, 30.0], [40.0, 50.0], [60.0, 100.0]]
        )
        assert y.observed.shape == (0, 2) and y.exposure.shape == (0, 2)
        assert sorted(k[3] for k in coverage) == ["x", "y"]


class TestExcludedTime:
    """Excluded time (the artifact mask widened by 0.3 s) is no detector's
    observed time or exposure, whatever its observation table says."""

    EPOCH_ID = "sim20200101_02_r1_HPC"
    CORES = [("S", 20.0, 21.0), ("spk", 70.0, 70.0)]
    """Padded into the mask [19, 22] and [69.9, 70.1]."""
    EXCLUDED = np.array([[18.7, 22.3], [69.6, 70.4]])
    STILL = np.array([[0.0, 30.0], [40.0, 100.0]])

    @pytest.fixture
    def masked(self, tmp_path, monkeypatch):
        from continuum_swr_replay.analysis.hippocampus import swr_source
        from tests.unit._swr_inventory_fixtures import write_artifact_run

        directory = write_artifact_run(
            tmp_path / "artifacts", {self.EPOCH_ID: self.CORES}
        )
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_ROOT", directory.parent)
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", directory.name)
        np.testing.assert_allclose(
            swr_source.excluded_intervals(self.EPOCH_ID).to_numpy(), self.EXCLUDED
        )

    def _observations(self) -> tuple[pd.DataFrame, dict]:
        stretches = {
            # the memoryless runner's decoded windows keep the reach
            "memoryless": np.array([[0.0, 18.9], [22.1, 100.0]]),
            "kay": np.array([[0.0, 50.0], [60.0, 100.0]]),
            "state_space": np.array([[0.0, 100.0]]),
        }
        tables = [
            observation_table(
                SESSION,
                detector=detector,
                run_id="r",
                epoch_bounds=(0.0, 100.0),
                observed=runs,
                unavailable_reason="not decoded",
            )
            for detector, runs in stretches.items()
        ]
        tables.append(
            missing_run_table(
                SESSION,
                detector="missing",
                run_id="r",
                epoch_bounds=(0.0, 100.0),
                reason="no run",
            )
        )
        return pd.concat(tables, ignore_index=True), stretches

    def test_every_detectors_exposure_is_its_stretches_less_excluded_time(self, masked):
        from continuum_swr_replay.analysis.hippocampus import swr_source

        excluded = swr_source.excluded_intervals(self.EPOCH_ID).to_numpy()
        observations, stretches = self._observations()

        coverage = coverage_from_observations(
            observations, {("sim", 20200101, "02_r1"): self.STILL}
        )

        for detector, runs in stretches.items():
            entry = coverage[("sim", 20200101, "02_r1", detector)]
            analysed = subtract(runs, excluded)
            np.testing.assert_array_equal(entry.observed, analysed)
            np.testing.assert_array_equal(
                entry.exposure, exposure(analysed, self.STILL)
            )
            assert total_duration(intersect(entry.exposure, excluded)) == 0.0
        empty = coverage[("sim", 20200101, "02_r1", "missing")]
        assert empty.observed.shape == (0, 2) and empty.exposure.shape == (0, 2)
        # the reach around the S block is gone from the memoryless exposure too
        np.testing.assert_allclose(
            coverage[("sim", 20200101, "02_r1", "memoryless")].exposure,
            [[0.0, 18.7], [22.3, 30.0], [40.0, 69.6], [70.4, 100.0]],
        )

    def test_with_no_artifact_run_the_stretches_are_the_observed_time(self):
        observations, stretches = self._observations()

        coverage = coverage_from_observations(
            observations, {("sim", 20200101, "02_r1"): self.STILL}
        )

        for detector, runs in stretches.items():
            entry = coverage[("sim", 20200101, "02_r1", detector)]
            np.testing.assert_allclose(entry.observed, runs)
            np.testing.assert_allclose(entry.exposure, exposure(runs, self.STILL))
