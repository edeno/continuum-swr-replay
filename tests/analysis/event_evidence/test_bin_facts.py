"""Tests for the per-bin facts: spikes binned as the decoder bins them, and the
well-visit flag, on grids small enough to check by hand."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.event_evidence import bin_facts as bf
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    spikes_in_bins,
)

TIME = np.array([0.0, 1.0, 2.0, 3.0])


def test_spikes_go_to_the_decoders_bins():
    """digitize(t, time[1:-1]): [0, 1) -> 0, [1, 2) -> 1, [2, 3] -> 2; the last
    bin is never reached and spikes outside [0, 3] are dropped."""
    spikes = [np.array([-0.1, 0.0, 0.5, 1.0]), np.array([2.0, 2.5, 3.0, 3.1])]
    counts = bf.decoder_bin_spike_counts(spikes, TIME)
    np.testing.assert_array_equal(counts, [2, 1, 3, 0])


def test_counts_match_spikes_in_bins():
    """The same binning as ``state_evidence.terms.spikes_in_bins``: the spikes
    it keeps in a bin are the bin's count."""
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 1.0, 0.002)
    spikes = [np.sort(rng.uniform(-0.05, 1.05, 400)) for _ in range(3)]
    marks = [np.zeros((s.size, 4)) for s in spikes]
    counts = bf.decoder_bin_spike_counts(spikes, time)
    for b in (0, 17, time.size - 2, time.size - 1):
        keep = np.zeros(time.size, dtype=bool)
        keep[b] = True
        kept, _ = spikes_in_bins(spikes, marks, time, keep)
        assert counts[b] == sum(k.size for k in kept)


def test_time_must_ascend():
    with pytest.raises(ValueError, match="ascending"):
        bf.decoder_bin_spike_counts([np.array([0.5])], np.array([0.0, 0.0, 1.0]))


def test_bin_facts_flags_bins_inside_a_visit():
    """Visits [0.5, 2.0) and [2.5, 3.0), as ``prepare_data.well_visit_times``
    gives them: bins at 1 and 2.5 are inside (arrival inclusive), 2 and 3 are
    not (departure exclusive)."""
    time = np.array([0.0, 1.0, 2.0, 2.5, 3.0])
    visits = (np.array([0.5, 2.5]), np.array([2.0, 3.0]))
    table = bf.bin_facts(time, [np.array([0.2, 1.5, 2.6])], visits)
    assert list(table.columns) == list(bf.BIN_FACTS_COLUMNS)
    np.testing.assert_array_equal(table["n_spikes"], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(
        table["in_well_visit"], [False, True, False, True, False]
    )
    assert table["n_spikes"].dtype == np.int32


def test_bin_facts_without_visits():
    table = bf.bin_facts(TIME, [np.array([0.5])], (np.array([]), np.array([])))
    assert not table["in_well_visit"].any()
