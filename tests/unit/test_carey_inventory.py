"""Carey's candidate detector on one epoch's shared inputs.

The settings are the package defaults; a tetrode's multiunit train is one
unit, binned onto the LFP's samples by the MUA rule; the artifact mask is
left out of the joint score's scaling, and the events it overlaps are dropped.
"""

from __future__ import annotations

import inspect
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from ripple_detection import Carey_candidate_detector, filter_ripple_band
from ripple_detection.simulate import simulate_session, simulate_time

from continuum_swr_replay.analysis.hippocampus import swr_detectors
from continuum_swr_replay.analysis.hippocampus.swr_detectors import (
    CAREY_SETTINGS,
    DETECTOR_SETTINGS,
    carey_observed_stretches,
    run_carey_detector,
    tetrode_spike_counts,
)

FS = 1000.0
"""Hz: the LFP grid Carey runs on."""

INJECTED_RIPPLE_TIMES = [5.0, 10.0, 15.0, 20.0, 25.0]
"""Seconds: the five ripple-plus-burst events injected into 30 s."""

N_TETRODES = 20
"""Simulated multiunit trains, one per tetrode."""

TETRODE_RATE_HZ = (10.0, 40.0)
"""Each simulated tetrode's baseline multiunit rate, drawn in this range.

At these rates the 30 s session gives exactly the five injected events and no
spurious one (the package's default 0.5-5 Hz units at 20 units give 13)."""

INPUT_PARAMETERS = frozenset(
    {"theta_lfp", "theta_band", "theta_threshold", "ripple_score", "normalization_mask"}
)
"""Keyword parameters that are inputs, not settings: no theta channel is given,
so its band and threshold do not apply; the ripple score is formed from the
band; the normalization mask comes from the artifact mask."""


def _spike_times(multiunit: NDArray, time: NDArray) -> list[NDArray]:
    """Each column's counts as spike times, every spike on its sample's time."""
    return [np.repeat(time, column.astype(int)) for column in multiunit.T]


@cache
def _simulated_epoch() -> SimpleNamespace:
    """A 30 s immobile session at 1000 Hz, as ``run_carey_detector`` takes it.

    Cached, so the arrays are shared: copy before changing them.
    """
    time = simulate_time(30_000, FS)
    session = simulate_session(
        time,
        INJECTED_RIPPLE_TIMES,
        n_units=N_TETRODES,
        baseline_rate=TETRODE_RATE_HZ,
        rng=0,
    )
    return SimpleNamespace(
        time=time,
        band=filter_ripple_band(session.lfps, sampling_frequency=FS),
        spike_times=_spike_times(session.multiunit, time),
        multiunit=session.multiunit,
        speed=session.speed,
        windows=session.ripple_windows,
    )


def _no_mask(n_time: int) -> NDArray[np.bool_]:
    return np.zeros(n_time, dtype=bool)


def _overlapping(events: pd.DataFrame, start_s: float, end_s: float) -> NDArray:
    """Whether each event shares any time with ``[start_s, end_s]``."""
    return (events["start_time"].to_numpy() <= end_s) & (
        events["end_time"].to_numpy() >= start_s
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_are_the_package_defaults():
    """Every setting is the detector's own default, and every setting is set."""
    parameters = inspect.signature(Carey_candidate_detector).parameters
    defaults = {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in INPUT_PARAMETERS
    }
    assert dict(CAREY_SETTINGS) == defaults


def test_settings_are_read_only_and_apart_from_the_four():
    with pytest.raises(TypeError):
        CAREY_SETTINGS["low_threshold"] = 2.0  # type: ignore[index]
    assert "carey" not in DETECTOR_SETTINGS


# ---------------------------------------------------------------------------
# Binning each tetrode's spikes onto the LFP grid
# ---------------------------------------------------------------------------


def test_spikes_are_binned_by_the_mua_rule_and_not_in_a_gap():
    """Bin ``i`` holds ``time[i] <= s < time[i + 1]``; ``time[-1]`` is in the last.

    The step from 3 to 10 is a gap (median step 1): a spike within one median
    step of 3 is in sample 3's bin, one beyond it is in the gap and not
    counted. Spikes outside the grid are not counted either.
    """
    time = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    first = np.array([-0.5, 0.0, 3.0, 3.5, 4.0, 7.0, 12.0, 12.5])
    second = np.array([2.5, 10.0])

    counts = tetrode_spike_counts([first, second], time)

    np.testing.assert_array_equal(
        counts,
        [[1, 0], [0, 0], [0, 1], [2, 0], [0, 1], [0, 0], [1, 0]],
    )


def test_spikes_on_their_samples_reproduce_the_counts():
    """Column order is the marks' order, and no spike is lost on a regular grid."""
    epoch = _simulated_epoch()

    counts = tetrode_spike_counts(epoch.spike_times, epoch.time)

    assert counts.shape == (epoch.time.size, N_TETRODES)
    np.testing.assert_array_equal(counts, epoch.multiunit)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_finds_the_injected_events_each_with_five_active_tetrodes():
    epoch = _simulated_epoch()
    counts = tetrode_spike_counts(epoch.spike_times, epoch.time)

    events = run_carey_detector(
        epoch.time,
        epoch.band,
        counts,
        epoch.speed,
        FS,
        unobserved=_no_mask(epoch.time.size),
    )

    assert isinstance(events, pd.DataFrame)
    assert len(events) == len(INJECTED_RIPPLE_TIMES)
    for start_s, end_s in epoch.windows:
        assert _overlapping(events, start_s, end_s).sum() == 1
    assert (events["n_active_units"] >= CAREY_SETTINGS["minimum_active_units"]).all()
    assert events.index.name == "event_number"
    np.testing.assert_array_equal(events.index, np.arange(1, len(events) + 1))


def test_the_mask_leaves_the_scaling_and_drops_what_it_overlaps(monkeypatch):
    """The mask's complement is the normalization sample; an event it touches is
    dropped and the rest are renumbered 1..n."""
    epoch = _simulated_epoch()
    counts = tetrode_spike_counts(epoch.spike_times, epoch.time)
    unobserved = (epoch.time >= 14.9) & (epoch.time <= 15.1)
    calls = []

    def spy(*args, **settings):
        calls.append(settings)
        return Carey_candidate_detector(*args, **settings)

    monkeypatch.setattr(swr_detectors, "Carey_candidate_detector", spy)

    events = run_carey_detector(
        epoch.time, epoch.band, counts, epoch.speed, FS, unobserved=unobserved
    )

    (settings,) = calls
    np.testing.assert_array_equal(settings.pop("normalization_mask"), ~unobserved)
    assert settings == CAREY_SETTINGS
    assert isinstance(events, pd.DataFrame)
    assert len(events) == len(INJECTED_RIPPLE_TIMES) - 1
    assert events.attrs["n_dropped_overlapping_unobserved"] == 1
    assert not _overlapping(events, 14.9, 15.1).any()
    np.testing.assert_array_equal(events.index, np.arange(1, len(events) + 1))
    assert events.index.name == "event_number"


def test_no_event_spans_a_nan_stretch_or_a_gap_and_neither_is_observed():
    """NaN band over the 10 s event, a 20 ms timestamp gap in the 20 s one."""
    epoch = _simulated_epoch()
    keep = (epoch.time < 19.990) | (epoch.time > 20.010)
    time = epoch.time[keep]
    band = epoch.band[keep].copy()
    nan_rows = (time >= 9.9) & (time <= 10.1)
    band[nan_rows] = np.nan
    counts = tetrode_spike_counts(epoch.spike_times, time)
    unobserved = _no_mask(time.size)

    events = run_carey_detector(
        time, band, counts, epoch.speed[keep], FS, unobserved=unobserved
    )
    stretches = carey_observed_stretches(time, band, counts, unobserved)

    assert isinstance(events, pd.DataFrame)
    assert not _overlapping(events, 9.9, 10.1).any()
    spans_gap = (events["start_time"] < 19.990) & (events["end_time"] > 20.010)
    assert not spans_gap.any()
    starts_s, ends_s = time[stretches[:, 0]], time[stretches[:, 1] - 1]
    assert not ((starts_s <= 10.1) & (ends_s >= 9.9)).any()
    assert not ((starts_s < 19.990) & (ends_s > 20.010)).any()
    np.testing.assert_allclose(
        np.column_stack([starts_s, ends_s]),
        [[0.0, 9.899], [10.101, 19.989], [20.011, 29.999]],
    )
    # a spike in the gap is not counted on the sample before it
    before_gap = np.flatnonzero(time == epoch.time[19_989])[0]
    np.testing.assert_array_equal(counts[before_gap], epoch.multiunit[19_989])


def test_observed_stretches_drop_short_blocks_and_the_mask():
    """A block shorter than the minimum duration is unobserved; so is the mask.

    The mask splits a block without shortening what is left below the minimum:
    Carey splits its blocks only at NaN rows and gaps, so a short piece left by
    the mask is still observed.
    """
    time = np.arange(1_000) / FS
    band = np.ones((1_000, 2))
    band[100] = np.nan  # rows 0-99 stay, a 100-sample block
    band[110] = np.nan  # rows 101-109: 9 samples, under the 20 the minimum spans
    counts = np.zeros((1_000, 5), dtype=int)
    unobserved = np.zeros(1_000, dtype=bool)
    unobserved[500:990] = True  # leaves rows 990-999, 10 samples

    stretches = carey_observed_stretches(time, band, counts, unobserved)

    np.testing.assert_array_equal(stretches, [[0, 100], [111, 500], [990, 1_000]])


# ---------------------------------------------------------------------------
# The package's errors are returned, not an empty table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_tetrodes", "has_spikes", "message"),
    [
        (4, True, "minimum_active_units is 5 but multiunit has 4"),
        (N_TETRODES, False, "multiunit contains no spikes"),
    ],
)
def test_too_few_tetrodes_or_no_spikes_is_the_package_error(
    n_tetrodes, has_spikes, message
):
    epoch = _simulated_epoch()
    counts = tetrode_spike_counts(epoch.spike_times[:n_tetrodes], epoch.time)
    if not has_spikes:
        counts = np.zeros_like(counts)

    result = run_carey_detector(
        epoch.time,
        epoch.band,
        counts,
        epoch.speed,
        FS,
        unobserved=_no_mask(epoch.time.size),
    )

    assert isinstance(result, ValueError)
    assert message in str(result)
