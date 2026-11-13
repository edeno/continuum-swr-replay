"""The non-local probability is one sum, on a decode's output and on its frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    nonlocal_probability,
    nonlocal_state_probability,
)

STATES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]


def _state_probabilities() -> xr.DataArray:
    values = np.array(
        [
            [0.7, 0.1, 0.15, 0.05],
            [0.1, 0.0, 0.5, 0.4],
            [np.nan, np.nan, np.nan, np.nan],
            [0.2, 0.2, np.nan, 0.6],
        ]
    )
    return xr.DataArray(
        values,
        dims=("time", "states"),
        coords={"time": [0.0, 0.002, 0.004, 0.006], "states": STATES},
    )


def test_the_decode_sum_is_continuous_plus_fragmented_and_nan_where_either_is():
    probability = nonlocal_state_probability(_state_probabilities())
    assert probability.dims == ("time",)
    np.testing.assert_array_equal(
        probability.values, [0.15 + 0.05, 0.5 + 0.4, np.nan, np.nan]
    )


def test_the_frame_sum_equals_the_decode_sum():
    states = _state_probabilities()
    frame = pd.DataFrame(
        {
            "Non-Local_Continuous_prob": states.sel(states="Non-Local Continuous"),
            "Non-Local_Fragmented_prob": states.sel(states="Non-Local Fragmented"),
        },
        index=states.time.values,
    )
    np.testing.assert_array_equal(
        nonlocal_probability(frame).to_numpy(),
        nonlocal_state_probability(states).values,
    )


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
