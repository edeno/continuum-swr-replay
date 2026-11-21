"""Which times lie inside a well visit (``event_definition/well_visits.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.covariates.context import (
    behavioral_location_from_port_experiences,
)
from continuum_swr_replay.analysis.event_definition.well_visits import (
    in_well_visit,
    well_visit_index,
)

#: Two visits: [10, 20) and [40, 50) s.
_ARRIVALS = np.array([10.0, 40.0])
_DEPARTURES = np.array([20.0, 50.0])


def test_poke_in_is_inside_and_final_poke_out_is_outside():
    times = np.array([9.999, 10.0, 19.999, 20.0, 30.0, 40.0, 49.999, 50.0, 60.0])
    index = well_visit_index(times, _ARRIVALS, _DEPARTURES)
    np.testing.assert_array_equal(index, [-1, 0, 0, -1, -1, 1, 1, -1, -1])
    np.testing.assert_array_equal(
        in_well_visit(times, _ARRIVALS, _DEPARTURES), index >= 0
    )


def test_times_need_no_order_and_no_visits_means_no_time_inside():
    times = np.array([45.0, 5.0, 15.0])
    np.testing.assert_array_equal(
        well_visit_index(times, _ARRIVALS, _DEPARTURES), [1, -1, 0]
    )
    assert not in_well_visit(times, np.array([]), np.array([])).any()


def test_refuses_visits_out_of_order_or_of_unequal_length():
    with pytest.raises(ValueError, match="ascending"):
        well_visit_index(np.array([1.0]), _ARRIVALS[::-1], _DEPARTURES[::-1])
    with pytest.raises(ValueError, match="one length"):
        well_visit_index(np.array([1.0]), _ARRIVALS, _DEPARTURES[:1])


def test_equals_the_per_sample_state_of_the_table_builders():
    """The builders' per-sample ``at_well`` is this rule, on a random session."""
    rng = np.random.default_rng(0)
    gaps = rng.uniform(0.5, 6.0, size=80)
    edges = 100.0 + np.cumsum(gaps)
    arrivals, departures = edges[0::2], edges[1::2]
    # Consecutive visits are at different wells (re-pokes at one well are one visit).
    ports = (np.arange(arrivals.size) * 5) % 6
    experiences = pd.DataFrame(
        {
            "experience_id": np.arange(arrivals.size),
            "movement_trial_number": pd.array(
                [pd.NA, *range(arrivals.size - 1)], dtype="Int64"
            ),
            "is_initial_experience": [True] + [False] * (arrivals.size - 1),
            "is_boundary_derived": False,
            "outcome_observed": True,
            "port": ports,
            "patch": ports // 2 + 1,
            "arrival_time": arrivals,
            "departure_time": departures,
            "reward_on_time": np.nan,
            "outcome_known_time": arrivals + 0.1,
            "is_reward": False,
        }
    )
    times = np.sort(
        np.concatenate(
            [rng.uniform(95.0, edges[-1] + 5.0, size=4000), arrivals, departures]
        )
    )
    state = behavioral_location_from_port_experiences(times, experiences)
    inside = in_well_visit(times, arrivals, departures)
    np.testing.assert_array_equal(
        state["occupancy_state"].astype(object).to_numpy() == "at_well", inside
    )
    index = well_visit_index(times, arrivals, departures)
    np.testing.assert_array_equal(
        state["occupancy_well_id"].to_numpy(dtype=float, na_value=np.nan)[inside],
        experiences["port"].to_numpy()[index[inside]],
    )
