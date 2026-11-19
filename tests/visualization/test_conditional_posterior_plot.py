"""The conditional-posterior panel draws the package's conditional posterior."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from continuum_swr_replay.visualization.plots import (  # noqa: E402
    plot_conditional_non_local_posterior,
)

TIME = np.array([10.0, 10.002, 10.004])
POSITION = np.array([0.0, 2.0, 4.0, 6.0])


def _results() -> xr.Dataset:
    # (state, time, position). At 10.002 s every bit of mass is Local: the
    # non-local mass is zero there.
    continuous = [[0.2, 0.1, 0.1, 0.0], [0.0] * 4, [0.0, 0.0, 0.3, 0.0]]
    fragmented = [[0.0, 0.1, 0.1, 0.0], [0.0] * 4, [0.0, 0.1, 0.0, 0.0]]
    local = [[0.1, 0.1, 0.2, 0.0], [0.25] * 4, [0.2, 0.2, 0.2, 0.0]]
    posterior = xr.DataArray(
        np.array([local, continuous, fragmented]),
        coords={
            "state": ["Local", "Non-Local Continuous", "Non-Local Fragmented"],
            "time": TIME,
            "position": POSITION,
        },
        dims=["state", "time", "position"],
    )
    return xr.Dataset({"acausal_posterior": posterior})


def test_the_panel_draws_the_package_posterior_forward_filled_and_off_track_gray():
    env = SimpleNamespace(
        place_bin_centers_=POSITION,
        is_track_interior_=np.array([True, True, True, False]),
    )
    data = {
        "position_info": pd.DataFrame(
            {"linear_position": [1.0, 2.0, 3.0]}, index=pd.Index(TIME, name="time")
        )
    }
    fig, ax = plt.subplots()
    plot_conditional_non_local_posterior(
        data, slice(TIME[0], TIME[-1]), _results(), env, ax
    )
    drawn = np.ma.filled(ax.collections[0].get_array(), np.nan).reshape(
        len(POSITION), len(TIME)
    )
    plt.close(fig)
    third = 1.0 / 3.0
    expected = np.array(
        [
            # each time bin's non-local mass normalized over position
            [third, third, third, np.nan],
            # no non-local mass: the bin before it, not 0 / 0
            [third, third, third, np.nan],
            [0.0, 0.25, 0.75, np.nan],
        ]
    ).T
    np.testing.assert_allclose(drawn, expected)
