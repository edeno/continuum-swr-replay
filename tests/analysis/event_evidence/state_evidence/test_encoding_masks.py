"""The running-only encodings' training bins, the analyzed regimes and the scored spikes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    EVENT_MARGIN_BINS,
    analyzed_regimes,
    running_only_masks,
    spikes_in_bins,
    training_spike_bins,
)


def test_running_only_masks_refuse_an_event_spike_in_the_library_span():
    n = 200
    time = np.arange(n) * 0.002
    speed = np.full(n, 20.0)
    position = np.c_[np.linspace(0, 100, n), np.zeros(n)]
    folds = np.zeros(n, dtype=np.int8)
    folds[100:] = 1
    in_event = np.zeros(n, dtype=bool)
    in_event[50] = True

    none = np.zeros(n, dtype=bool)
    masks, record = running_only_masks(
        time, speed, position, in_event, folds, is_missing=none
    )

    # the event bin and two bins either side are left out, and the padding
    # stops short of it
    assert not masks["run"][48:53].any() and masks["run"][47] and masks["run"][53]
    assert record["run"]["event_bins_in_library_span"] == 0
    assert record["running_bins_left_out_near_events"] == 5
    # no position around the event: the library joins the runs across it
    position[48:53] = np.nan
    with pytest.raises(ValueError, match="1 non-local event bins are in the run"):
        running_only_masks(time, speed, position, in_event, folds, is_missing=none)
    # a fold with no running bin
    position[48:53] = 0.0
    with pytest.raises(ValueError, match="runcv1 encoding has no training bin"):
        running_only_masks(
            time, speed, position, in_event, np.zeros(n, np.int8), is_missing=none
        )
    # the decode's missing bins are in no running-only fit
    missing = none.copy()
    missing[150:160] = True
    masks, record = running_only_masks(
        time, speed, position, in_event, folds, is_missing=missing
    )
    assert not masks["run"][150:160].any() and masks["run"][149] and masks["run"][160]
    assert record["missing_bins_at_running_speed"] == 10


def test_float_rounding_of_the_bin_width_cannot_carry_the_span_into_an_event():
    """At epoch timestamps a bin can be an ulp narrower than the first one.

    The library pads a run by the first two bins' spacing, so with one bin of
    margin its span can reach a sliver into the event bin (one j16 pilot epoch
    did); a margin of :data:`EVENT_MARGIN_BINS` (two) cannot.
    """
    n = 20_000
    time = 1625682923.263429 + np.arange(n) * 0.002
    speed = np.full(n, 20.0)
    position = np.c_[np.linspace(0, 100, n), np.zeros(n)]
    in_event = np.zeros(n, dtype=bool)
    in_event[np.arange(100, n - 100, 97)] = True
    one_bin = ~(in_event | np.r_[in_event[1:], False] | np.r_[False, in_event[:-1]])
    assert (in_event & training_spike_bins(time, position, one_bin)).any()

    _, record = running_only_masks(
        time,
        speed,
        position,
        in_event,
        np.arange(n) // 5000 % 2,
        is_missing=np.zeros(n, dtype=bool),
    )

    for name in ("run", "runcv0", "runcv1"):
        assert record[name]["event_bins_in_library_span"] == 0


def test_the_margin_is_two_bins():
    assert EVENT_MARGIN_BINS == 2


def test_analyzed_regimes_leave_out_excluded_time_and_unknown_positions():
    time = np.arange(8) * 0.002
    speed = np.array([1.0, 1.0, 20.0, 20.0, 5.0, 1.0, 20.0, np.nan])
    linear = np.array([0.0, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    # bins 5 and 6 (0.010 and 0.012 s): the bounds are in
    excluded = pd.DataFrame({"start_time": [0.010], "end_time": [0.012]})

    regime, in_excluded = analyzed_regimes(time, speed, linear, excluded)

    assert regime.tolist() == ["immobile", "", "running", "running", "", "", "", ""]
    assert in_excluded.tolist() == [False] * 5 + [True, True, False]


def test_spikes_in_bins_keeps_the_spikes_the_decoder_bins_into_a_kept_bin():
    # the decoder bins a spike in [time[0], time[-1]] by np.digitize on
    # time[1:-1]: [0, 1) -> 0, [1, 2) -> 1, [2, 3] -> 2
    time = np.array([0.0, 1.0, 2.0, 3.0])
    keep = np.array([True, False, True, False])
    spikes = [np.array([-0.5, 0.0, 0.9, 1.0, 2.5, 3.0, 3.5]), np.array([1.5])]
    marks = [10.0 * np.arange(7)[:, None], np.array([[7.0]])]

    kept_times, kept_marks = spikes_in_bins(spikes, marks, time, keep)

    # before time[0] and after time[-1] are outside; bins 0 and 2 are kept
    np.testing.assert_array_equal(kept_times[0], [0.0, 0.9, 2.5, 3.0])
    np.testing.assert_array_equal(kept_marks[0], [[10.0], [20.0], [40.0], [50.0]])
    assert kept_times[1].size == 0 and kept_marks[1].shape == (0, 1)
