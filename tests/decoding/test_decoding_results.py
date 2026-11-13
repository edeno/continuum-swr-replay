"""The one ``results.nc`` reader (``decoding.results.open_results``).

The decode runners save ``results.nc`` after ``reset_index("state_bins")``;
the reader rebuilds the (``state``, ``position``) multi-index and keeps the
file closable. Offline: a synthetic file written as the runners write it.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    get_conditional_non_local_posterior,
    load_conditional_posterior,
)
from continuum_swr_replay.decoding.results import (
    ENCODING_KEY,
    load_saved_detector,
    nonlocal_probability,
    open_results,
)
from tests.support._memoryless_clusterless_epoch import _StandInDetector

STATES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]


def _decoded() -> xr.Dataset:
    """A decode's results as the detector returns them: ``state_bins`` indexed,
    No-Spike's one bin at a NaN position."""
    centers = np.array([1.0, 3.0, 5.0])
    state = ["Local"] * 3 + ["No-Spike"] + ["Non-Local Continuous"] * 3
    state += ["Non-Local Fragmented"] * 3
    position = np.concatenate([centers, [np.nan], centers, centers])
    rng = np.random.default_rng(0)
    posterior = rng.random((4, position.size))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return xr.Dataset(
        {"acausal_posterior": (("time", "state_bins"), posterior)},
        coords={
            "time": [0.0, 0.002, 0.004, 0.006],
            "state": ("state_bins", state),
            "position": ("state_bins", position),
        },
        attrs={"provenance": "{}"},
    ).set_index(state_bins=["state", "position"])


def _saved(tmp_path: Path) -> Path:
    path = tmp_path / "x_results.nc"
    _decoded().reset_index("state_bins").to_netcdf(path, engine="h5netcdf")
    return path


def test_the_state_bins_multi_index_is_rebuilt(tmp_path):
    results = open_results(_saved(tmp_path))

    index = results.indexes["state_bins"]
    assert isinstance(index, pd.MultiIndex)
    assert list(index.names) == ["state", "position"]
    assert index.equals(_decoded().indexes["state_bins"])
    np.testing.assert_array_equal(
        results.acausal_posterior.sel(state="Non-Local Continuous").position,
        [1.0, 3.0, 5.0],
    )
    xr.testing.assert_identical(
        results.reset_index("state_bins"), _decoded().reset_index("state_bins")
    )


def test_the_conditional_posterior_reads_through_it(tmp_path):
    xr.testing.assert_identical(
        load_conditional_posterior(_saved(tmp_path)),
        get_conditional_non_local_posterior(_decoded()),
    )


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")
def test_closing_it_closes_the_file(tmp_path):
    """``set_index`` returns a Dataset that would not close the file; the
    reader hands it the open file's ``close``."""
    path = _saved(tmp_path)

    def open_handles() -> int:
        fds = Path("/proc/self/fd")
        return sum(
            1 for fd in fds.iterdir() if os.path.realpath(fd) == str(path.resolve())
        )

    with open_results(path) as results:
        results.acausal_posterior.load()
        assert open_handles() >= 1
    assert open_handles() == 0


def test_the_non_local_probability_sums_the_two_non_local_states_row_by_row():
    frame = pd.DataFrame(
        {
            "Local_prob": np.array([0.5, 0.0, 0.25], dtype=np.float32),
            "Non-Local_Continuous_prob": np.array([0.25, 0.5, np.nan], np.float32),
            "Non-Local_Fragmented_prob": np.array([0.25, 0.5, 0.75], np.float32),
        },
        index=pd.Index([2.0, 1.0, 3.0], name="time"),
    )
    total = nonlocal_probability(frame)
    pd.testing.assert_series_equal(
        total,
        pd.Series(np.array([0.5, 1.0, np.nan], np.float32), index=frame.index),
    )


class _Environment:
    """The attributes the loader reads of an encoding environment."""

    def __init__(self, centers, interior) -> None:
        self.place_bin_centers_ = centers
        self.is_track_interior_ = interior


def test_a_saved_detector_comes_back_with_its_grid_marked_fitted(tmp_path):
    """The flag is restored only for a grid that is there; a missing file, a
    missing encoding group and a missing grid are each refused."""
    path = tmp_path / "e_detector.pkl"
    with pytest.raises(FileNotFoundError, match="No detector"):
        load_saved_detector(path)

    joblib.dump(_StandInDetector({("other", 7): {}}), path)
    with pytest.raises(KeyError, match=r"other.*7"):
        load_saved_detector(path)

    environment = _Environment(np.arange(3.0), np.ones(3, dtype=bool))
    joblib.dump(_StandInDetector({ENCODING_KEY: {"environment": environment}}), path)
    loaded = load_saved_detector(path)
    assert loaded.encoding_model_[ENCODING_KEY]["environment"]._is_fitted is True

    environment.place_bin_centers_ = None
    joblib.dump(_StandInDetector({ENCODING_KEY: {"environment": environment}}), path)
    with pytest.raises(ValueError, match="place grid"):
        load_saved_detector(path)
