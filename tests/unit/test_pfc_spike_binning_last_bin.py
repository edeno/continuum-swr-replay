"""Regression test for ``get_spikecount_per_time_bin`` last-bin reachability.

The previous ``np.digitize(spike_times, time[1:-1])`` indexing produced
output indices ``0..N-2`` against a ``minlength=N`` bincount, leaving
the final output bin unreachable. With ``time=[0,1,2,3]``, spikes at
2.5 and 3.0 both landed in bin 2 and bin 3 stayed at zero. The fix is
``np.digitize(spike_times, time[1:])`` which spreads spikes across
all N output bins.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin


class TestPfcSpikeBinningLastBinReachable:
    def test_spike_at_last_edge_is_counted(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([3.0])
        counts = get_spikecount_per_time_bin(spike_times, time)
        assert counts.shape == (4,)
        assert counts[-1] == 1, (
            f"expected count 1 in last bin, got {counts.tolist()}"
        )

    def test_two_spikes_split_between_penultimate_and_last(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([2.5, 3.0])
        counts = get_spikecount_per_time_bin(spike_times, time)
        assert counts[-2] == 1, f"penultimate bin empty: counts={counts.tolist()}"
        assert counts[-1] == 1, f"last bin empty: counts={counts.tolist()}"

    def test_total_count_preserved(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([0.5, 1.5, 2.5, 3.0])
        counts = get_spikecount_per_time_bin(spike_times, time)
        assert int(counts.sum()) == len(spike_times)
