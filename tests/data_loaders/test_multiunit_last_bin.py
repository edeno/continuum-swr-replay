"""Regression test for ``get_multiunit_hse`` last-bin reachability.

The previous ``np.digitize(t, bin_time[1:-1])`` indexing left the
final output bin unreachable: with ``bin_time = [0, 1, 2, 3]``,
spikes at 2.5 and 3.0 both landed in bin 2 and bin 3 stayed at zero.
The fix is ``np.digitize(t, bin_time[1:])`` which spreads spikes
across all N output bins.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.data_loaders.lfp import get_multiunit_hse


class TestMultiunitLastBinReachable:
    def test_spike_at_last_edge_is_counted(self):
        # A single unit, one spike at the last bin edge.
        spike_times = [np.array([3.0])]
        bin_time = np.array([0.0, 1.0, 2.0, 3.0])
        speed = np.zeros(bin_time.shape[0])

        _, multiunit_firing_rate, _ = get_multiunit_hse(spike_times, speed, bin_time)

        # The final output bin (index -1) must contain a non-zero rate.
        last_bin_rate = multiunit_firing_rate["firing_rate"].iloc[-1]
        assert last_bin_rate > 0, (
            f"expected non-zero rate in final bin, got {last_bin_rate}; "
            "the previous ``bin_time[1:-1]`` digitize produced "
            "[0, 0, 0, 0] for this input."
        )

    def test_two_spikes_split_between_penultimate_and_last(self):
        # Two spikes; one at 2.5 (interior of penultimate bin) and one
        # at 3.0 (the last edge). They must end up in different bins.
        spike_times = [np.array([2.5, 3.0])]
        bin_time = np.array([0.0, 1.0, 2.0, 3.0])
        speed = np.zeros(bin_time.shape[0])

        _, multiunit_firing_rate, _ = get_multiunit_hse(spike_times, speed, bin_time)

        rates = multiunit_firing_rate["firing_rate"].to_numpy()
        # Penultimate (bin 2) and last (bin 3) both non-zero.
        assert rates[-2] > 0, f"penultimate bin empty: rates={rates.tolist()}"
        assert rates[-1] > 0, f"last bin empty: rates={rates.tolist()}"
