"""The one ``results.nc`` reader (``decoding.results.open_results``).

The decode runners save ``results.nc`` after ``reset_index("state_bins")``;
the reader rebuilds the (``state``, ``position``) multi-index and keeps the
file closable. Offline: a synthetic file written as the runners write it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    get_conditional_non_local_posterior,
    load_conditional_posterior,
)
from continuum_swr_replay.decoding.results import open_results

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
