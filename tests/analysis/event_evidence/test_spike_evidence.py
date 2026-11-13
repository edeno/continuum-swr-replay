"""The spike-evidence windows and metrics, on synthetic frames.

The stored likelihood stand-in: ``log_likelihood`` (time x state bins) whose
Non-Local Continuous columns hold the non-local likelihood at every place bin,
the junction placeholders the environment does not count as interior stored as
NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.event_evidence import spike_evidence

N_PLACE = 12
INTERIOR = np.ones(N_PLACE, dtype=bool)
INTERIOR[[4, 9]] = False
"""Two junction placeholders: stored as NaN, not interior."""
CENTERS = 2.0 * np.arange(N_PLACE)
REAL = CENTERS[INTERIOR]
N_TIME = 60
TIME = 1000.0 + 0.002 * np.arange(N_TIME)


class StraightTrack:
    """Place bins every 2 cm; two of them are not interior."""

    place_bin_centers_ = CENTERS[:, None]
    is_track_interior_ = INTERIOR

    def get_distances_to_interior_bins(self, positions):
        return np.abs(np.asarray(positions, dtype=float)[:, :1] - REAL[None, :])


def results(log_likelihood_nl: np.ndarray) -> xr.Dataset:
    """A ``results.nc`` stand-in: Local (all place bins), No-Spike (one), and the
    two non-local states (all place bins) holding ``log_likelihood_nl``."""
    nl = np.where(INTERIOR[None, :], log_likelihood_nl, np.nan)
    local = np.zeros((N_TIME, N_PLACE))
    no_spike = np.zeros((N_TIME, 1))
    ll = np.concatenate([local, no_spike, nl, nl], axis=1).astype(np.float32)
    state = (
        ["Local"] * N_PLACE
        + ["No-Spike"]
        + ["Non-Local Continuous"] * N_PLACE
        + ["Non-Local Fragmented"] * N_PLACE
    )
    position = np.r_[CENTERS, np.nan, CENTERS, CENTERS]
    return xr.Dataset(
        {"log_likelihood": (("time", "state_bins"), ll)},
        coords={
            "time": TIME,
            "state": ("state_bins", state),
            "position": ("state_bins", position),
        },
    )


def test_the_columns_are_the_non_local_interior_bins_in_order():
    cols = spike_evidence.stored_likelihood_columns(
        results(np.zeros((N_TIME, N_PLACE))), StraightTrack()
    )
    nl_start = N_PLACE + 1
    assert cols.tolist() == (nl_start + np.flatnonzero(INTERIOR)).tolist()


def test_bins_that_are_not_the_environment_s_refuse():
    ds = results(np.zeros((N_TIME, N_PLACE)))
    ds = ds.assign_coords(position=("state_bins", ds.position.values + 1.0))
    with pytest.raises(ValueError, match="not the environment's place bins"):
        spike_evidence.stored_likelihood_columns(ds, StraightTrack())


def test_an_event_s_rows_and_columns_are_read():
    values = np.arange(N_TIME * N_PLACE, dtype=float).reshape(N_TIME, N_PLACE)
    ds = results(values)
    cols = spike_evidence.stored_likelihood_columns(ds, StraightTrack())
    L = spike_evidence.stored_event_likelihood(ds["log_likelihood"], cols, 5, 8)
    np.testing.assert_array_equal(L, values[5:9][:, INTERIOR])
    assert np.isfinite(L).all()


def test_spikes_are_counted_over_the_half_open_span():
    spikes = [np.array([1.0, 1.5, 2.0]), np.array([2.0]), np.array([0.5])]
    assert spike_evidence.window_spike_counts(spikes, 1.0, 2.0) == (2, 1)
    assert spike_evidence.window_spike_counts(spikes, 1.0, 2.5) == (4, 2)


# --- the windows --------------------------------------------------------------------


def _session_frame() -> pd.DataFrame:
    """100 bins. Non-local events (probability 0.9) of 4, 3, 3, 3, 2 and 3
    bins; everything else is clearly local (probability 0), except bin 49
    (0.2, not below the threshold); bin 35 runs at 4 cm/s."""
    n = 100
    label = np.zeros(n, dtype=int)
    ab = np.zeros(n)
    swr = np.zeros(n, dtype=int)
    for k, (first, last, distance) in enumerate(
        [
            (2, 5, 150.0),  # SWR-remote: the SWR covers bins 3-4
            (8, 10, -120.0),  # nonSWR-remote
            (12, 14, 10.0),  # nonSWR-close
            (16, 18, 60.0),  # 40-100 cm: left out
            (20, 21, 150.0),  # two bins: too short
            (23, 25, 10.0),  # SWR-close: left out
        ],
        start=1,
    ):
        label[first : last + 1] = k
        ab[first : last + 1] = distance
    ab[3] = np.nan  # a bin with no distance does not count in the mean
    swr[3:5] = 1
    swr[23:26] = 2
    probability = np.where(label > 0, 0.9, 0.0)
    probability[49] = 0.2
    speed = np.ones(n)
    speed[35] = 4.0
    return pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "non_local_event": label,
            "SWR_event": swr,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": probability / 2,
            "Non-Local_Fragmented_prob": probability / 2,
            "head_speed": speed,
            "ahead_behind_distance": ab,
            "in_excluded_time": np.zeros(n, dtype=bool),
        },
        index=pd.Index(0.002 * np.arange(n), name="time"),
    )


def test_the_events_are_grouped_by_swr_overlap_and_distance_and_the_quiet_windows_are_analysed_time():
    frame = _session_frame()
    # bins 30 and 31 are excluded time
    frame.iloc[30:32, frame.columns.get_loc("in_excluded_time")] = True

    windows = spike_evidence.build_session_events(frame)

    expected = pd.DataFrame(
        {
            "s": [2, 8, 12, 26, 32, 36],
            "e": [5, 10, 14, 29, 34, 48],
            "mean_abs_ab": [150.0, 120.0, 10.0, 0.0, 0.0, 0.0],
            "is_swr": [True, False, False, False, False, False],
            "n_bins": [4, 3, 3, 4, 3, 13],
            "group": [
                "SWR-remote",
                "nonSWR-remote",
                "nonSWR-close",
                "baseline",
                "baseline",
                "baseline",
            ],
        }
    )
    # the quiet runs 0-1, 6-7 and the single bins are too short; 50-99 (50
    # bins) is too long
    pd.testing.assert_frame_equal(windows, expected)


def test_an_event_that_is_not_one_run_refuses(monkeypatch):
    frame = _session_frame()
    frame.iloc[40, frame.columns.get_loc("non_local_event")] = 2
    monkeypatch.setattr(
        spike_evidence,
        "validate_dot_event_contract",
        lambda _dot: None,  # the contract check would refuse it first
    )
    with pytest.raises(ValueError, match="not one contiguous run"):
        spike_evidence.build_session_events(frame)


# --- the metrics --------------------------------------------------------------------


def _likelihood(n_bins: int) -> np.ndarray:
    """2 nats at the last real bin (the MAP), 0.5 at the first (the animal),
    -1 elsewhere."""
    L = np.full((n_bins, REAL.size), -1.0)
    L[:, -1] = 2.0
    L[:, 0] = 0.5
    return L


def test_the_metrics_compare_the_map_with_the_animal_over_the_finite_bins():
    # bins 10-13; bin 11 has no MAP, so three bins score 1.5 nats each
    mape = np.full(N_TIME, REAL[-1])
    mape[11] = np.nan
    lin = np.full(N_TIME, REAL[0])

    metrics = spike_evidence.event_metrics(
        10, 13, _likelihood(4), 2, mape, lin, StraightTrack(), REAL, REAL[2]
    )

    assert metrics == pytest.approx(
        {
            "dll_animal": 1.5,
            "dll_per_spike": 4.5 / 2,
            "n_spikes": 2,
            "map_mean": REAL[-1],
            "dist_to_gpi_peak": REAL[-1] - REAL[2],
        }
    )


def test_a_window_with_no_spike_or_fewer_than_two_finite_bins_is_not_scored():
    mape = np.full(N_TIME, REAL[-1])
    lin = np.full(N_TIME, REAL[0])
    args = (StraightTrack(), REAL, REAL[2])
    assert (
        spike_evidence.event_metrics(10, 13, _likelihood(4), 0, mape, lin, *args)
        is None
    )
    lin[11:14] = np.nan
    assert (
        spike_evidence.event_metrics(10, 13, _likelihood(4), 5, mape, lin, *args)
        is None
    )
