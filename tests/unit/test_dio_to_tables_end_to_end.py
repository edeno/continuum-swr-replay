"""DIO events through the real builders to the stop and event tables.

Every other builder test hands the table builders a hand-written trials frame
and an adapted port-experience stream. That skips the two places the outcome
clock is actually set -- the pump-to-poke assignment in
:func:`make_well_visits_df` and the omission boundary in
:func:`make_port_experiences_df` -- so a fixture on the wrong omission clock
runs the whole suite on a clock the pipeline never uses.

This test starts from beam breaks and pump onsets, runs the real chain
(well visits -> movement trials -> port experiences -> stop / event table) and
pins the boundary that separates the two clocks: a bout that starts inside the
arrival -> outcome gap of an omission still reads the *previous* outcome, and a
bout that starts after the boundary reads the omission.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.replay.tables_event import build_event_table
from continuum_swr_replay.analysis.replay.tables_stop import build_stop_table
from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC
from continuum_swr_replay.data_loaders.trials import (
    make_port_experiences_df,
    make_trials_df_from_well_visits,
    make_well_visits_df,
)

# One task visit per row: (well, poke-in, poke-out, pump onset or None).
# Four movement trials over three wells; the third visit is the omission.
_VISITS = (
    (0, 1.0, 2.0, 1.05),
    (2, 5.0, 6.0, 5.08),
    (4, 9.0, 10.0, None),
    (0, 13.0, 14.0, 13.06),
    (2, 17.0, 18.0, 17.05),
)
_OMISSION_ARRIVAL = 9.0
_OMISSION_KNOWN = _OMISSION_ARRIVAL + OMISSION_KNOWN_LATENCY_SEC
# One bout inside the arrival -> outcome gap, one after it, both at the
# omission well; a third inside the following rewarded visit.
_EARLY_BOUT = (_OMISSION_ARRIVAL + 0.1, _OMISSION_ARRIVAL + 0.2)
_LATE_BOUT = (_OMISSION_ARRIVAL + 0.5, _OMISSION_ARRIVAL + 0.8)
_REWARD_BOUT = (13.2, 13.6)
_BOUTS = (_EARLY_BOUT, _LATE_BOUT, _REWARD_BOUT)
# Arm the animal occupies during each bout: leaf arm == well id.
_BOUT_ARM = (4, 4, 0)


def _bandit_graph() -> nx.Graph:
    """Canonical spider track: six leaf arms, three trunk arms."""
    graph = nx.Graph()
    for well, junction, edge_id in (
        (0, 6, 0),
        (1, 6, 1),
        (2, 7, 2),
        (3, 7, 3),
        (4, 8, 4),
        (5, 8, 5),
        (6, 9, 6),
        (7, 9, 7),
        (8, 9, 8),
    ):
        graph.add_edge(well, junction, edge_id=edge_id, distance=1.0)
    return graph


def _dio_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beam-break bouts and reward-pump onsets for the synthetic session."""
    beam_breaks = pd.DataFrame(
        {
            "start_time": [poke_in for _, poke_in, _, _ in _VISITS],
            "end_time": [poke_out for _, _, poke_out, _ in _VISITS],
            "event_name": [f"Poke{well + 1}" for well, _, _, _ in _VISITS],
        }
    )
    rewarded = [(well, pump) for well, _, _, pump in _VISITS if pump is not None]
    pump_events = pd.DataFrame(
        {
            "start_time": [pump for _, pump in rewarded],
            "event_name": [f"Pump{well + 1}" for well, _ in rewarded],
        }
    )
    return beam_breaks, pump_events


def _data_over_time() -> pd.DataFrame:
    """Decoder output: immobile only inside the three bouts, one event."""
    index = pd.Index(np.round(np.arange(0.0, 20.0, 0.02), 6), name="time")
    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 20200101,
            "epoch": "02_r1",
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "map_estimate": np.nan,
            "head_speed": 20.0,
            "track_segment_id": np.nan,
        },
        index=index,
    )
    for (start, end), arm in zip(_BOUTS, _BOUT_ARM, strict=True):
        immobile = (index >= start) & (index <= end)
        dot.loc[immobile, "head_speed"] = 0.0
        dot.loc[immobile, "track_segment_id"] = float(arm)
    # One non-local event inside the bout that starts before the omission
    # boundary, so the event table is exercised on the same clock.
    event = (index >= _EARLY_BOUT[0] + 0.02) & (index <= _EARLY_BOUT[0] + 0.06)
    dot.loc[event, "non_local_event"] = 1
    dot.loc[event, "Non-Local_Continuous_prob"] = 0.95
    dot.loc[event, "map_estimate"] = 2.0  # first bin of arm 1
    return dot


class _EnvStub:
    """Minimal ``Environment`` stand-in: two bins per arm plus junction bins."""

    class _Df:
        _d = {
            "linear_position": np.concatenate(
                [np.arange(18, dtype=float), np.array([100.0, 101.0, 102.0])]
            ),
            "edge_id": np.concatenate(
                [np.repeat(np.arange(9), 2).astype(int), np.array([-1, -1, -1])]
            ),
        }

        def __getitem__(self, key):
            return self._d[key]

    place_bin_centers_nodes_df_ = _Df()


def _conditional_posterior(dot: pd.DataFrame) -> xr.DataArray:
    """Uniform-over-real-bins posterior aligned to ``dot``."""
    n_bins = 21
    posterior = np.full((len(dot), n_bins), 1.0 / n_bins)
    return xr.DataArray(
        posterior,
        dims=("time", "position"),
        coords={"time": dot.index.to_numpy(), "position": np.arange(n_bins)},
    )


@pytest.fixture(scope="module")
def session():
    """Run the real DIO -> tables chain once for the whole module."""
    beam_breaks, pump_events = _dio_events()
    graph = _bandit_graph()
    well_visits = make_well_visits_df(beam_breaks, pump_events)
    trials = make_trials_df_from_well_visits(well_visits)
    port_experiences = make_port_experiences_df(well_visits, trials)
    dot = _data_over_time()
    stop_table = build_stop_table(
        dot, trials, port_experiences, graph, speed_threshold=4.0
    )
    event_table = build_event_table(
        dot,
        _EnvStub(),
        trials,
        port_experiences,
        graph,
        conditional_posterior=_conditional_posterior(dot),
    )
    return {
        "well_visits": well_visits,
        "trials": trials,
        "port_experiences": port_experiences,
        "stop_table": stop_table,
        "event_table": event_table,
    }


def _stop_starting_at(stop_table: pd.DataFrame, start: float) -> pd.Series:
    match = stop_table.loc[np.isclose(stop_table["stop_start"], start, atol=0.021)]
    assert len(match) == 1, (
        f"expected exactly one bout starting near {start}; "
        f"got {match['stop_start'].tolist()}"
    )
    return match.iloc[0]


def test_the_dio_chain_recovers_the_task_visits(session):
    """Five task visits, four movement trials, one omission at well 4."""
    experiences = session["port_experiences"]
    assert len(experiences) == len(_VISITS)
    assert len(session["trials"]) == len(_VISITS) - 1
    assert experiences["port"].tolist() == [well for well, _, _, _ in _VISITS]
    assert experiences["is_reward"].tolist() == [
        pump is not None for _, _, _, pump in _VISITS
    ]


def test_omission_outcome_is_known_at_the_pump_latency_boundary(session):
    """The omission's clock is arrival + the latency, not its departure."""
    omission = session["port_experiences"].iloc[2]
    assert omission["port"] == 4
    assert not bool(omission["is_reward"])
    assert omission["arrival_time"] == pytest.approx(_OMISSION_ARRIVAL)
    assert omission["outcome_known_time"] == pytest.approx(_OMISSION_KNOWN)
    # The departure clock the fixtures used to carry is 0.7 s later.
    assert omission["departure_time"] == pytest.approx(10.0)


def test_bout_inside_the_arrival_to_outcome_gap_reads_the_previous_outcome(session):
    """A bout that starts before the boundary cannot see the omission."""
    early = _stop_starting_at(session["stop_table"], _EARLY_BOUT[0])
    assert early["stop_start"] < _OMISSION_KNOWN
    # The most recent *known* outcome is the rewarded visit at well 2.
    assert early["known_reward_outcome_lag_1"] == pytest.approx(1.0)


def test_bout_after_the_boundary_reads_the_omission(session):
    """The same visit, past the boundary, reports the omission."""
    late = _stop_starting_at(session["stop_table"], _LATE_BOUT[0])
    assert late["stop_start"] > _OMISSION_KNOWN
    assert late["known_reward_outcome_lag_1"] == pytest.approx(0.0)


def test_stop_outcome_clock_is_the_boundary_for_an_omission(session):
    """``outcome_known_time_s`` names the boundary from arrival onward."""
    stop_table = session["stop_table"]
    for bout in (_EARLY_BOUT, _LATE_BOUT):
        stop = _stop_starting_at(stop_table, bout[0])
        assert stop["outcome_known_time_s"] == pytest.approx(_OMISSION_KNOWN)
        assert stop["last_well_arrival_time_s"] == pytest.approx(_OMISSION_ARRIVAL)


def test_stop_outcome_clock_is_the_pump_onset_for_a_reward(session):
    """A rewarded visit's boundary is the recorded pump onset."""
    stop = _stop_starting_at(session["stop_table"], _REWARD_BOUT[0])
    assert stop["outcome_known_time_s"] == pytest.approx(13.06)
    assert stop["last_well_arrival_time_s"] == pytest.approx(13.0)
    # Its own outcome is known by the time the bout starts.
    assert stop["known_reward_outcome_lag_1"] == pytest.approx(1.0)


def test_event_table_shares_the_stop_clock(session):
    """The one non-local event lands in the pre-boundary bout."""
    events = session["event_table"]
    assert len(events) == 1
    early = _stop_starting_at(session["stop_table"], _EARLY_BOUT[0])
    assert events.iloc[0]["stop_id"] == early["stop_id"]
