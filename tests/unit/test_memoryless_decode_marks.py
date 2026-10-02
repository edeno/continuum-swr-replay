"""Decode-time HPC marks: the 0820d39d coincident rule and the decode's span."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.decode_marks import (
    decode_time_coincident_keep,
    decode_time_marks,
)
from continuum_swr_replay.data_loaders.spikes import detect_coincident_spikes

FAR = 1.0
"""Seconds between unrelated spikes, far beyond the 40 us closeness threshold."""


def _kept_times(spike_times):
    keep = decode_time_coincident_keep(spike_times)
    return [np.asarray(t)[k] for t, k in zip(spike_times, keep, strict=True)]


class TestDecodeTimeCoincidentRule:
    def test_two_tetrode_cluster_on_four_is_kept_unlike_the_corrected_rule(self):
        """Only the trailing spike counts at 0820d39d: 1/4 is not over 0.33.

        The corrected rule counts both (2/4) and removes them; this is the
        difference that made the reloaded marks miss the decode's.
        """
        spike_times = [
            np.array([1.0, 1.0 + FAR]),
            np.array([1.00001, 3.0]),
            np.array([5.0]),
            np.array([7.0]),
        ]

        kept = _kept_times(spike_times)
        corrected, _ = detect_coincident_spikes(spike_times)

        assert [len(t) for t in kept] == [2, 2, 1, 1]
        assert [len(t) for t in corrected] == [1, 1, 1, 1]

    def test_three_tetrode_cluster_on_four_is_removed_with_its_first_spike(self):
        spike_times = [
            np.array([1.0, 2.0]),
            np.array([1.00001, 3.0]),
            np.array([1.00002, 4.0]),
            np.array([5.0]),
        ]

        kept = _kept_times(spike_times)

        np.testing.assert_array_equal(kept[0], [2.0])
        np.testing.assert_array_equal(kept[1], [3.0])
        np.testing.assert_array_equal(kept[2], [4.0])
        np.testing.assert_array_equal(kept[3], [5.0])

    def test_repeat_spikes_on_one_tetrode_count_once(self):
        """Three close spikes, two of them on one tetrode, span 1/4 labeled."""
        spike_times = [
            np.array([1.0]),
            np.array([1.00001, 1.00002]),
            np.array([5.0]),
            np.array([7.0]),
        ]

        assert [len(t) for t in _kept_times(spike_times)] == [1, 2, 1, 1]

    def test_no_spikes_keeps_nothing_for_every_electrode(self):
        keep = decode_time_coincident_keep([np.array([]), np.array([])])

        assert [k.size for k in keep] == [0, 0]

    def test_isolated_spikes_are_all_kept(self):
        rng = np.random.default_rng(0)
        spike_times = [np.sort(rng.uniform(0, 100, 50)) for _ in range(5)]
        # nudge any accidental near-coincidences apart
        spike_times = [t + 0.01 * i for i, t in enumerate(spike_times)]

        keep = decode_time_coincident_keep(spike_times)

        for times, k in zip(spike_times, keep, strict=True):
            np.testing.assert_array_equal(k, np.arange(times.size))


class TestDecodeTimeMarks:
    def test_features_follow_their_spikes(self):
        spike_times = [np.array([1.0, 2.0, 3.0])]
        features = [np.array([[10.0], [20.0], [30.0]])]

        times, marks = decode_time_marks(spike_times, features, (0.0, 10.0))

        np.testing.assert_array_equal(times[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(marks[0], [[10.0], [20.0], [30.0]])

    def test_span_is_inclusive_at_both_ends(self):
        spike_times = [np.array([0.5, 1.0, 2.0, 3.0, 3.5])]
        features = [np.arange(5, dtype=float)[:, None]]

        times, marks = decode_time_marks(spike_times, features, (1.0, 3.0))

        np.testing.assert_array_equal(times[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(marks[0][:, 0], [1.0, 2.0, 3.0])

    def test_electrodes_with_no_spikes_in_span_are_dropped(self):
        spike_times = [np.array([1.0]), np.array([50.0]), np.array([2.0])]
        features = [np.ones((1, 4)), 2 * np.ones((1, 4)), 3 * np.ones((1, 4))]

        times, marks = decode_time_marks(spike_times, features, (0.0, 10.0))

        assert len(times) == len(marks) == 2
        assert marks[1][0, 0] == pytest.approx(3.0)

    def test_coincident_removal_precedes_the_span(self):
        """A cluster is judged on every electrode, then the span is applied."""
        spike_times = [
            np.array([1.0, 2.0]),
            np.array([1.00001, 3.0]),
            np.array([1.00002, 4.0]),
            np.array([5.0]),
        ]
        features = [np.zeros((t.size, 1)) for t in spike_times]

        times, _ = decode_time_marks(spike_times, features, (0.0, 4.5))

        assert [t.tolist() for t in times] == [[2.0], [3.0], [4.0]]
