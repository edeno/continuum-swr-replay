"""The spike-evidence ("reality") engine scores events with the decode's stored
log-likelihood.

The HPC decode stores ``log_likelihood`` (time x state bins) in ``results.nc``.
Its Non-Local Continuous columns hold the non-local likelihood at every place
bin, including the junction placeholders the environment does not count as
interior (stored as NaN). Row ``k`` holds the spikes in ``[t_k, t_{k+1})``, so
an event of bins ``s..e`` counts its spikes over ``[t_s, t_{e+1})``. Every input
here is synthetic: a straight track with two non-interior bins and a stand-in
encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.event_evidence.analyze_event_spike_evidence as spike_evidence
from continuum_swr_replay.decoding.files import decode_paths

EPOCH_ID = "aa20200101_02_r1_HPC"
N_PLACE = 12
INTERIOR = np.ones(N_PLACE, dtype=bool)
INTERIOR[[4, 9]] = False
"""Two junction placeholders: stored as NaN, not interior."""
CENTERS = 2.0 * np.arange(N_PLACE)
REAL = CENTERS[INTERIOR]
N_TIME = 60
TIME = 1000.0 + 0.002 * np.arange(N_TIME)
ANIMAL_CM = REAL[0]
MAP_CM = REAL[-1]


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


@pytest.fixture
def stored_session(monkeypatch, tmp_path):
    """``analyze_session`` with the stored likelihood, on stand-ins.

    The non-local likelihood is +1 nat at the MAP bin and 0 elsewhere, so each
    scored bin's dLL is exactly 1. One event's MAP is in a flagged
    (subnormal-occupancy) bin: the stored path scores it.
    """
    ll = np.zeros((N_TIME, N_PLACE))
    ll[:, np.flatnonzero(INTERIOR)[-1]] = 1.0
    ds = results(ll)
    path = decode_paths(EPOCH_ID, tmp_path, threshold_percent=80).results
    ds.to_netcdf(path, engine="h5netcdf")  # as the decode writes it
    dot = pd.DataFrame(
        {
            "map_estimate": np.full(N_TIME, MAP_CM),
            "linear_position": np.full(N_TIME, ANIMAL_CM),
        },
        index=pd.Index(TIME, name="time"),
    )
    events = pd.DataFrame(
        {
            "s": [10, 30],
            "e": [14, 34],
            "n_bins": 5,
            "mean_abs_ab": 150.0,
            "is_swr": False,
            "group": "nonSWR-remote",
        }
    )
    # spikes in the events' bins, one in the last bin of the first event
    # (t_14 <= x < t_15) and one just after it (t_15)
    spike_times = [
        np.array([TIME[10], TIME[12], TIME[14] + 0.001, TIME[15]]),
        np.array([TIME[31]]),
    ]
    occupancy = np.full(INTERIOR.sum(), 3e-2, dtype=np.float32)
    occupancy[-1] = 1e-30  # the MAP bin is flagged
    patched = {
        "processed_data_dir": lambda: tmp_path,
        "read_dot_with_swr_source": lambda _path, **_kwargs: dot,
        "validate_dot_event_contract": lambda _dot: None,
        "build_session_events": lambda _dot: events.copy(),
        "load_encoding": lambda _e: {
            "env": StraightTrack(),
            "encargs": {"occupancy": occupancy},
            "pos": REAL,
            "gpi": np.ones(INTERIOR.sum()),
        },
        "load_marks": lambda *_a: (spike_times, [None, None]),
        "event_likelihood": lambda *_a: pytest.fail("the rebuild must not run"),
    }
    for name, value in patched.items():
        monkeypatch.setattr(spike_evidence, name, value)
    return spike_evidence.analyze_session(EPOCH_ID, tmp_path, n_sample=10)


def test_the_stored_path_scores_every_event_with_the_decode_s_likelihood(
    stored_session,
):
    metrics, context = stored_session
    metrics = metrics.sort_values("start_time").reset_index(drop=True)
    assert len(metrics) == 2
    np.testing.assert_allclose(metrics["dll_animal"], [1.0, 1.0])
    # spikes over [t_s, t_{e+1}): the last bin's spike counts, t_15's does not
    assert metrics["n_spikes"].tolist() == [3, 1]
    assert metrics["n_tetrodes"].tolist() == [1, 1]
    np.testing.assert_allclose(metrics["dll_per_spike"], [5 / 3, 5 / 1])
    # the flagged MAP bin is counted, not left out
    assert context["left_out"]["n_events_left_out"] == 0
    assert context["left_out"]["n_events_map_in_flagged_bins"] == 2
