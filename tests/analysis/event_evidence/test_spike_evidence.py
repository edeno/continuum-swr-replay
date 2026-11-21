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


def test_the_events_are_grouped_by_swr_overlap_and_distance_and_the_quiet_windows_are_analyzed_time():
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


# --- the analyses' subset -----------------------------------------------------

EPOCH = "chimi20200225_02_r1_HPC"
"""The epoch of the hand-built event table (``chimi``, 20200225, ``02_r1``)."""

BIG_ID = 2**63 + 5
"""An ``event_id`` above 2**53, which a float would round."""


def _event_table() -> pd.DataFrame:
    """Three canonical events of :data:`EPOCH`, one of them never scored."""
    return pd.DataFrame(
        {
            "animal": "chimi",
            "session_date": 20200225,
            "epoch": "02_r1",
            "event_start_s": [10.002, 20.004, 30.0],
            "event_id": np.array([BIG_ID, 7, 8], dtype=np.uint64),
        }
    )


def _scored() -> pd.DataFrame:
    """Two scored events of the table and three quiet windows."""
    return pd.DataFrame(
        {
            "epoch_id": EPOCH,
            "group": ["SWR-remote", "nonSWR-close", "baseline", "baseline", "baseline"],
            "start_time": [20.004, 10.002, 5.0, 12.0, 25.0],
            "dll_per_spike": [1.0, 2.0, 0.1, 0.2, 9.0],
        }
    )


def test_each_scored_event_takes_its_event_id_by_epoch_and_start():
    out = spike_evidence.with_event_ids(_scored(), _event_table())
    assert out["event_id"].dtype == "UInt64"
    assert out["event_id"].iloc[0] == 7
    assert out["event_id"].iloc[1] == BIG_ID  # exact, not rounded
    assert out["event_id"].iloc[2:].isna().all()  # quiet windows are no events
    assert out["dll_per_spike"].tolist() == _scored()["dll_per_spike"].tolist()


def test_a_scored_event_without_its_event_refuses():
    scored = _scored()
    scored.loc[0, "start_time"] = 20.006  # one sample later: no event starts there
    with pytest.raises(ValueError, match="1 scored events match no event"):
        spike_evidence.with_event_ids(scored, _event_table())


def test_two_events_with_one_epoch_and_start_refuse():
    events = _event_table()
    events.loc[2, "event_start_s"] = 10.002
    with pytest.raises(ValueError, match="share an epoch and start"):
        spike_evidence.with_event_ids(_scored(), events)


def test_the_subset_is_the_selected_events_and_the_quiet_windows_in_a_visit():
    metrics = spike_evidence.with_event_ids(_scored(), _event_table())
    # one visit, [8, 15): the quiet windows at 5 and 25 s start outside it
    visits = {EPOCH: (np.array([8.0]), np.array([15.0]))}
    keep = spike_evidence.in_analyzed_subset(metrics, [BIG_ID, 8], visits)
    # the event 7 is not selected, though nothing about its time is checked here
    assert keep.tolist() == [False, True, False, True, False]


def test_a_quiet_window_outside_every_visit_does_not_move_the_statistics():
    """The planted window (dll 9 at 25 s) would raise the baseline median."""
    metrics = spike_evidence.with_event_ids(_scored(), _event_table())
    visits = {EPOCH: (np.array([8.0]), np.array([15.0]))}
    kept = metrics[spike_evidence.in_analyzed_subset(metrics, [BIG_ID, 7], visits)]
    assert kept.loc[kept["group"] == "baseline", "dll_per_spike"].tolist() == [0.2]


def test_a_quiet_window_at_a_visit_s_final_poke_out_is_outside():
    metrics = spike_evidence.with_event_ids(_scored(), _event_table())
    visits = {EPOCH: (np.array([5.0]), np.array([12.0]))}
    keep = spike_evidence.in_analyzed_subset(metrics, [], visits)
    assert keep.tolist() == [False, False, True, False, False]


def test_an_epoch_without_visits_refuses():
    metrics = spike_evidence.with_event_ids(_scored(), _event_table())
    with pytest.raises(ValueError, match="no well visits for 1 epochs"):
        spike_evidence.in_analyzed_subset(metrics, [7], {})
