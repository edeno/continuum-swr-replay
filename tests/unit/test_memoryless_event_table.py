"""The memoryless event table: one row per candidate sequence.

The recording is built by hand so each column has one right answer. Three
sequences come out of the real detector:

1. windows 10-49 travel bins 0-39 inside an SWR: replay, contained, eligible;
2. windows 70-132 travel bins 50-111 in two pieces bridged across a blurred
   three-window gap; a speed sample above threshold falls between window
   midpoints inside the gap, so the window gate misses it and the whole-event
   rule must not;
3. windows 153-192 hold still at bin 150: not replay, and outside the stretch
   where SWR labels were observed, though an SWR overlaps it.

Each blurred filler is centered on the last bin of the sequence before it.
Window k's jump is to window k + 1, so a filler centered elsewhere would reject
the sequence's last window and end it one window early. The fillers between
sequences last 80 ms, beyond the 50 ms merge gap.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.event_table import (
    MEMORYLESS_EVENT_TABLE_DTYPES,
    SequenceIdentity,
    build_memoryless_event_table,
    build_shuffle_draw_table,
    canonical_swr_intervals,
    observed_label_intervals,
    shuffle_eligible,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    ShuffleResult,
)

CONFIG = MemorylessSequenceConfig()
N_BINS = 200
BIN_CM = 2.0
START_S = 10.0
"""The first window's midpoint; nonzero so a time and an index cannot be confused."""

IDENTITY = SequenceIdentity(
    animal="sim", day=1, epoch="02_r1", brain_area="HPC", run_id="test-run"
)


def blurred(center_bin: float, sd_cm: float = 25.0) -> np.ndarray:
    """A posterior too broad for the spread gate, centered where it is put."""
    centers = np.arange(N_BINS) * BIN_CM
    weights = np.exp(-0.5 * ((centers - center_bin * BIN_CM) / sd_cm) ** 2)
    return weights / weights.sum()


def delta(bin_index: int) -> np.ndarray:
    row = np.zeros(N_BINS)
    row[bin_index] = 1.0
    return row


def build_posterior() -> np.ndarray:
    rows = [blurred(100)] * 10  # 0-9
    rows += [delta(b) for b in range(0, 40)]  # 10-49: sequence 1
    rows += [blurred(39)] * 20  # 50-69
    rows += [delta(b) for b in range(50, 80)]  # 70-99: sequence 2, first piece
    rows += [blurred(81)] * 3  # 100-102: the merged gap
    rows += [delta(b) for b in range(82, 112)]  # 103-132: second piece
    rows += [blurred(111)] * 20  # 133-152
    rows += [delta(150)] * 40  # 153-192: sequence 3, still
    rows += [blurred(150)] * 27  # 193-219
    return np.array(rows)


@pytest.fixture(scope="module")
def recording():
    posterior = build_posterior()
    n_windows = posterior.shape[0]
    midpoint_s = START_S + CONFIG.stride_s * np.arange(n_windows)
    centers = np.arange(N_BINS) * BIN_CM
    distance = np.abs(centers[:, None] - centers[None, :])
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, distance, np.zeros(n_windows), CONFIG
    )

    # a native 2 ms speed trace over the whole decode, still but for one sample
    # between the midpoints of windows 100 and 101, inside sequence 2's gap
    speed_time_s = np.arange(
        START_S - 0.04, midpoint_s[-1] + 0.04, 1 / CONFIG.sampling_frequency
    )
    speed = np.zeros(speed_time_s.size)
    between = midpoint_s[100] + CONFIG.stride_s / 2
    speed[np.argmin(np.abs(speed_time_s - between))] = 9.0

    swr_intervals = pd.DataFrame(
        {
            "SWR_event": [1, 2],
            "start_time": [midpoint_s[5], midpoint_s[180]],
            "end_time": [midpoint_s[60], midpoint_s[210]],
        }
    )
    # labels observed over the first two sequences only
    observed = np.array([[START_S - 0.04, midpoint_s[145]]])
    return {
        "candidates": candidates,
        "midpoint_s": midpoint_s,
        "distance": distance,
        "speed_time_s": speed_time_s,
        "speed": speed,
        "swr_intervals": swr_intervals,
        "observed": observed,
    }


def passing_result() -> ShuffleResult:
    return ShuffleResult(
        observed_spread_cm=1.0,
        observed_jump_cm=1.0,
        null_spread_cm=np.full(100, 5.0),
        null_jump_cm=np.full(100, 5.0),
    )


def build(recording, **overrides) -> pd.DataFrame:
    arguments = {
        "speed_time_s": recording["speed_time_s"],
        "speed_cm_per_s": recording["speed"],
        "bin_distance_cm": recording["distance"],
        "swr_intervals": recording["swr_intervals"],
        "swr_observed_intervals": recording["observed"],
        "swr_definition": "Kay",
        "shuffle_results": {0: passing_result()},
    } | overrides
    return build_memoryless_event_table(recording["candidates"], IDENTITY, **arguments)


class TestTheRecording:
    """The fixture yields the three sequences the module docstring describes."""

    def test_three_sequences_with_the_intended_bounds(self, recording):
        np.testing.assert_array_equal(
            recording["candidates"].runs, [[10, 49], [70, 132], [153, 192]]
        )
        assert recording["candidates"].is_replay.tolist() == [True, True, False]
        assert [len(piece) for piece in recording["candidates"].pieces] == [1, 2, 1]


class TestColumns:
    def test_one_row_per_candidate_in_order_with_the_declared_dtypes(self, recording):
        table = build(recording)

        assert len(table) == 3
        assert list(table.columns) == list(MEMORYLESS_EVENT_TABLE_DTYPES)
        for column, dtype in MEMORYLESS_EVENT_TABLE_DTYPES.items():
            assert table[column].dtype == dtype, column
        assert table.sequence_label.tolist() == [1, 2, 3]

    def test_identity_is_on_every_row(self, recording):
        table = build(recording)

        assert (table.animal == "sim").all()
        assert (table.day == 1).all()
        assert (table.epoch == "02_r1").all()
        assert (table.brain_area == "HPC").all()
        assert (table.run_id == "test-run").all()
        assert (table.swr_definition == "Kay").all()

    def test_carries_its_own_key_and_never_the_state_space_one(self, recording):
        """D6: a cohort-specific event_id would mis-join silently."""
        table = build(recording)

        assert "event_id" not in table.columns
        assert table.ms_event_id.is_unique

    def test_the_key_is_stable_and_depends_on_the_run(self, recording):
        first = build(recording)
        again = build(recording)
        other_run = build_memoryless_event_table(
            recording["candidates"],
            SequenceIdentity(
                animal="sim", day=1, epoch="02_r1", brain_area="HPC", run_id="other"
            ),
            speed_time_s=recording["speed_time_s"],
            speed_cm_per_s=recording["speed"],
            bin_distance_cm=recording["distance"],
            swr_intervals=recording["swr_intervals"],
            swr_observed_intervals=recording["observed"],
            swr_definition="Kay",
            shuffle_results={0: passing_result()},
        )

        np.testing.assert_array_equal(first.ms_event_id, again.ms_event_id)
        assert not set(first.ms_event_id) & set(other_run.ms_event_id)


class TestTiming:
    def test_bounds_are_the_first_and_last_window_midpoints(self, recording):
        table = build(recording)
        midpoint = recording["midpoint_s"]

        np.testing.assert_allclose(table.event_start_s, midpoint[[10, 70, 153]])
        np.testing.assert_allclose(table.event_end_s, midpoint[[49, 132, 192]])
        np.testing.assert_allclose(
            table.event_duration_s, table.event_end_s - table.event_start_s
        )
        np.testing.assert_allclose(
            table.event_midpoint_s, (table.event_start_s + table.event_end_s) / 2
        )

    def test_the_decoding_support_is_separate_and_wider_by_half_a_window(
        self, recording
    ):
        """The window width is never added to the event itself."""
        table = build(recording)
        half = CONFIG.window_duration_s / 2

        np.testing.assert_allclose(table.support_start_s, table.event_start_s - half)
        np.testing.assert_allclose(table.support_end_s, table.event_end_s + half)


class TestDetectorStatistics:
    def test_counts_every_window_and_every_piece(self, recording):
        table = build(recording)

        assert table.n_windows.tolist() == [40, 63, 40]
        assert table.n_pieces.tolist() == [1, 2, 1]

    def test_jumps_are_within_pieces_and_the_bridge_is_reported_apart(self, recording):
        """Bins 79 and 82 face each other across the gap: a 6 cm bridge that
        must not be counted as a consecutive-window jump."""
        table = build(recording)
        merged = table.iloc[1]

        assert merged.max_jump_cm == pytest.approx(2.0)
        assert merged.mean_jump_cm == pytest.approx(2.0)
        assert merged.max_bridge_cm == pytest.approx(6.0)
        assert np.isnan(table.iloc[0].max_bridge_cm)

    def test_path_distance_follows_the_retained_centers_across_the_bridge(
        self, recording
    ):
        table = build(recording)

        # 29 steps of 2 cm in each piece, and the 6 cm bridge between them
        assert table.iloc[1].path_distance_cm == pytest.approx(29 * 2 + 6 + 29 * 2)
        assert table.iloc[2].path_distance_cm == pytest.approx(0.0)

    def test_spread_and_dispersion_come_from_the_retained_windows(self, recording):
        table = build(recording)

        np.testing.assert_allclose(table.mean_posterior_spread_cm, 0.0, atol=1e-9)
        np.testing.assert_allclose(
            table.dispersion_cm, recording["candidates"].dispersion_cm
        )
        assert table.is_replay.tolist() == [True, True, False]


class TestSpeed:
    def test_every_native_sample_between_the_endpoints_counts(self, recording):
        """The window gate read midpoint speeds and missed the sample; the
        whole-event rule reads every sample, merged gaps included."""
        table = build(recording)

        assert table.whole_event_speed_eligible.tolist() == [True, False, True]
        assert table.animal_max_speed_cm_s.tolist() == pytest.approx([0.0, 9.0, 0.0])
        assert table.iloc[1].animal_mean_speed_cm_s > 0.0

    def test_an_event_with_no_speed_sample_is_not_eligible(self, recording):
        table = build(
            recording,
            speed_time_s=recording["speed_time_s"][:10],
            speed_cm_per_s=recording["speed"][:10],
            shuffle_results={},  # nothing is eligible, so nothing is shuffled
        )

        assert not table.whole_event_speed_eligible.any()
        assert table.animal_max_speed_cm_s.isna().all()

    def test_speed_arrays_of_different_lengths_are_refused(self, recording):
        with pytest.raises(ValueError, match="speed"):
            build(recording, speed_cm_per_s=recording["speed"][:-1])


class TestSwrLabels:
    def test_contained_non_swr_and_unknown(self, recording):
        table = build(recording)

        assert table.overlap_class.tolist() == ["swr_contained", "non_swr", "unknown"]
        assert table.is_swr.tolist()[:2] == [True, False]
        assert pd.isna(table.is_swr.iloc[2])

    def test_an_unobserved_event_never_counts_as_non_swr(self, recording):
        """Sequence 3 overlaps an SWR, but its labels were not observed."""
        table = build(recording)
        unobserved = table.iloc[2]

        assert unobserved.overlap_class == "unknown"
        assert unobserved.overlap_subclass == "unknown"
        assert pd.isna(unobserved.n_swrs_overlapping)
        assert pd.isna(unobserved.matched_swr_event)

    def test_observed_everywhere_gives_the_partial_overlap(self, recording):
        whole = np.array([[START_S - 0.04, recording["midpoint_s"][-1] + 0.04]])

        table = build(recording, swr_observed_intervals=whole)

        assert table.overlap_class.tolist() == [
            "swr_contained",
            "non_swr",
            "swr_partial",
        ]
        # a partial overlap is still SWR-coincident
        assert table.is_swr.tolist() == [True, False, True]
        assert table.matched_swr_event.tolist()[0] == 1
        assert table.matched_swr_event.tolist()[2] == 2
        assert table.n_swrs_overlapping.tolist() == [1, 0, 1]

    def test_the_matched_swr_bounds_are_the_swr_itself(self, recording):
        table = build(recording)
        swr = recording["swr_intervals"].iloc[0]

        assert table.iloc[0].matched_swr_start == pytest.approx(swr.start_time)
        assert table.iloc[0].matched_swr_end == pytest.approx(swr.end_time)
        assert np.isnan(table.iloc[1].matched_swr_start)

    def test_no_swr_labels_at_all_is_unknown_everywhere(self, recording):
        table = build(recording, swr_intervals=None, swr_observed_intervals=None)

        assert (table.overlap_class == "unknown").all()
        assert table.is_swr.isna().all()

    def test_labels_without_saying_where_they_were_observed_are_refused(
        self, recording
    ):
        """Assuming full coverage would turn missing LFP into non-SWR."""
        with pytest.raises(ValueError, match="observed"):
            build(recording, swr_observed_intervals=None)


class TestShuffleColumns:
    def test_eligibility_needs_the_replay_criteria_and_whole_event_stillness(
        self, recording
    ):
        eligible = shuffle_eligible(
            recording["candidates"],
            recording["speed_time_s"],
            recording["speed"],
        )

        assert eligible.tolist() == [0]

    def test_a_tested_candidate_carries_its_statistics(self, recording):
        table = build(recording)
        tested = table.iloc[0]

        assert tested.n_shuffles == 100
        assert tested.shuffle_testable
        assert tested.shuffle_pass
        assert tested.p_spread == pytest.approx(1 / 101)
        assert tested.shuffle_observed_spread_cm == pytest.approx(1.0)

    def test_an_untested_candidate_is_marked_by_zero_draws(self, recording):
        """Not tested is not the same as tested and untestable."""
        table = build(recording)
        untested = table.iloc[1:]

        assert (untested.n_shuffles == 0).all()
        assert not untested.shuffle_pass.any()
        assert not untested.shuffle_testable.any()
        assert untested.p_spread.isna().all()

    def test_shuffle_pass_alone_selects_the_published_definition(self, recording):
        table = build(recording)

        cohort = table[table.shuffle_pass]
        assert (cohort.is_replay & cohort.whole_event_speed_eligible).all()

    def test_no_shuffle_run_leaves_every_candidate_untested(self, recording):
        table = build(recording, shuffle_results=None)

        assert (table.n_shuffles == 0).all()
        assert not table.shuffle_pass.any()

    def test_a_result_for_an_ineligible_candidate_is_refused(self, recording):
        with pytest.raises(ValueError, match="eligible"):
            build(recording, shuffle_results={0: passing_result(), 1: passing_result()})

    def test_an_eligible_candidate_without_a_result_is_refused(self, recording):
        with pytest.raises(ValueError, match="eligible"):
            build(recording, shuffle_results={})


class TestShuffleDrawTable:
    def test_one_row_per_draw_keyed_on_the_event(self, recording):
        table = build(recording)

        draws = build_shuffle_draw_table(table, {0: passing_result()})

        assert len(draws) == 100
        assert (draws.ms_event_id == table.ms_event_id.iloc[0]).all()
        assert draws.draw.tolist() == list(range(1, 101))
        np.testing.assert_allclose(draws.null_spread_cm, 5.0)
        np.testing.assert_allclose(draws.null_jump_cm, 5.0)

    def test_no_results_gives_an_empty_table_with_its_columns(self, recording):
        draws = build_shuffle_draw_table(build(recording), {})

        assert draws.empty
        assert list(draws.columns) == [
            "ms_event_id",
            "draw",
            "null_spread_cm",
            "null_jump_cm",
        ]


class TestEmptyEpoch:
    def test_no_candidates_gives_the_declared_dtypes(self, recording):
        posterior = np.tile(blurred(60), (40, 1))
        midpoint_s = START_S + CONFIG.stride_s * np.arange(40)
        none = detect_candidate_sequences(
            posterior, midpoint_s, recording["distance"], np.zeros(40), CONFIG
        )

        table = build_memoryless_event_table(
            none,
            IDENTITY,
            speed_time_s=recording["speed_time_s"],
            speed_cm_per_s=recording["speed"],
            bin_distance_cm=recording["distance"],
            swr_intervals=recording["swr_intervals"],
            swr_observed_intervals=recording["observed"],
            swr_definition="Kay",
            shuffle_results={},
        )

        assert table.empty
        for column, dtype in MEMORYLESS_EVENT_TABLE_DTYPES.items():
            assert table[column].dtype == dtype, column

    def test_concatenating_an_empty_epoch_keeps_every_dtype(self, recording):
        """The canonical table's empty return is all-object; this one is not."""
        full = build(recording)
        empty = full.iloc[:0]

        combined = pd.concat([empty, full], ignore_index=True)

        for column, dtype in MEMORYLESS_EVENT_TABLE_DTYPES.items():
            assert combined[column].dtype == dtype, column


def data_over_time(n: int = 60, **columns) -> pd.DataFrame:
    """A minimal time-indexed frame with the columns the SWR helpers read."""
    frame = pd.DataFrame(
        {
            "animal": "sim",
            "day": 1,
            "epoch": "02_r1",
            "head_speed": np.zeros(n),
            "SWR_event": np.zeros(n, dtype=int),
        },
        index=pd.Index(START_S + np.arange(n) / 500.0, name="time"),
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestCanonicalSwrIntervals:
    def test_bounds_are_the_first_and_last_labeled_sample(self):
        labels = np.zeros(60, dtype=int)
        labels[10:20] = 1
        frame = data_over_time(SWR_event=labels)

        intervals = canonical_swr_intervals(frame, speed_threshold=4.0)

        assert intervals.SWR_event.tolist() == [1]
        assert intervals.start_time.iloc[0] == pytest.approx(frame.index[10])
        assert intervals.end_time.iloc[0] == pytest.approx(frame.index[19])

    def test_an_swr_during_movement_is_dropped_whole(self):
        """The same speed qualification the canonical table applies."""
        labels = np.zeros(60, dtype=int)
        labels[10:20], labels[30:40] = 1, 2
        speed = np.zeros(60)
        speed[35] = 9.0
        frame = data_over_time(SWR_event=labels, head_speed=speed)

        intervals = canonical_swr_intervals(frame, speed_threshold=4.0)

        assert intervals.SWR_event.tolist() == [1]

    def test_a_time_column_is_the_time_axis(self):
        labels = np.zeros(60, dtype=int)
        labels[10:20] = 1
        frame = data_over_time(SWR_event=labels).reset_index()

        intervals = canonical_swr_intervals(frame, speed_threshold=4.0)

        assert intervals.start_time.iloc[0] == pytest.approx(frame.time.iloc[10])

    def test_more_than_one_epoch_is_refused(self):
        frame = pd.concat([data_over_time(), data_over_time().assign(epoch="04_r2")])

        with pytest.raises(ValueError, match="one epoch"):
            canonical_swr_intervals(frame, speed_threshold=4.0)


class TestObservedLabelIntervals:
    def test_a_complete_label_column_is_observed_throughout(self):
        frame = data_over_time()

        observed = observed_label_intervals(frame, "SWR_event")

        np.testing.assert_allclose(observed, [[frame.index[0], frame.index[-1]]])

    def test_missing_labels_split_the_coverage(self):
        labels = np.zeros(60)
        labels[20:30] = np.nan
        frame = data_over_time(SWR_event=labels)

        observed = observed_label_intervals(frame, "SWR_event")

        np.testing.assert_allclose(
            observed,
            [[frame.index[0], frame.index[19]], [frame.index[30], frame.index[59]]],
        )

    def test_no_labels_at_all_is_no_coverage(self):
        frame = data_over_time(SWR_event=np.full(60, np.nan))

        observed = observed_label_intervals(frame, "SWR_event")

        assert observed.shape == (0, 2)
