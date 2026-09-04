"""Regression test for ContFrag ``arm_fraction`` denominator.

The non-local table divides ``arm_count_hard`` by the count of
real-arm bins so per-event arm fractions sum to 1. The ContFrag
table used to divide by ``n_timebins`` (all event bins, including
junction / missing-arm bins) while the numerator only counted real
arms — so the per-event sum dropped below 1 whenever any event bin
landed on a junction. The fix uses ``n_timebins - n_junction_bins``
so the convention matches.

This exercises the production builder
(:func:`build_contfrag_event_arm_table`) end-to-end so a regression
that pivots back to dividing by ``n_timebins`` would actually fail.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_arm_table as _build_contfrag_event_arm_table,
)
from continuum_swr_replay.analysis.contfrag.tables import (
    build_contfrag_event_table as _build_contfrag_event_table,
)
from continuum_swr_replay.analysis.replay.covariates import (
    arm_never_visited_per_arm as _arm_never_visited_per_arm,
)
from continuum_swr_replay.analysis.replay.covariates import (
    recent_reward_rate_per_arm as _recent_reward_rate_per_arm,
)
from continuum_swr_replay.analysis.replay.covariates import (
    rolling_reward_count_per_arm as _rolling_reward_count_per_arm,
)
from continuum_swr_replay.analysis.replay.covariates import (
    visit_recency_per_arm as _visit_recency_per_arm,
)
from tests.unit._port_experience_fixtures import port_experiences_from_trials


def _experiences(trials):
    return port_experiences_from_trials(trials, prepend_initial=True)


def build_contfrag_event_table(dot, env, trials, graph, **kwargs):
    return _build_contfrag_event_table(
        dot, env, trials, _experiences(trials), graph, **kwargs
    )


def build_contfrag_event_arm_table(event_table, dot, env, trials, graph, **kwargs):
    return _build_contfrag_event_arm_table(
        event_table,
        dot,
        env,
        trials,
        _experiences(trials),
        graph,
        **kwargs,
    )


def visit_recency_per_arm(time, trials, graph):
    return _visit_recency_per_arm(time, _experiences(trials), graph)


def arm_never_visited_per_arm(time, trials, graph):
    return _arm_never_visited_per_arm(time, _experiences(trials), graph)


def rolling_reward_count_per_arm(time, trials, graph):
    return _rolling_reward_count_per_arm(time, _experiences(trials), graph)


def recent_reward_rate_per_arm(time, trials, graph, k=10):
    return _recent_reward_rate_per_arm(time, _experiences(trials), graph, k=k)


class _EnvStub:
    """Minimal ``Environment`` stand-in.

    ``place_bin_centers_nodes_df_`` holds 2 bins per real arm (0-8)
    plus 3 junction bins (edge_id = -1) so the ContFrag builder's
    junction-bin filter has something to exclude.
    """

    class _Df:
        _arm_lp = np.arange(18, dtype=float)
        _arm_eid = np.repeat(np.arange(9), 2).astype(int)
        _junc_lp = np.array([100.0, 101.0, 102.0])
        _junc_eid = np.array([-1, -1, -1])
        d = {
            "linear_position": np.concatenate([_arm_lp, _junc_lp]),
            "edge_id": np.concatenate([_arm_eid, _junc_eid]),
        }

        def __getitem__(self, k):
            return self.d[k]

    place_bin_centers_nodes_df_ = _Df()


def _bandit_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, eid in (
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
        g.add_edge(u, v, edge_id=eid, distance=1.0)
    return g


def _empty_trials() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "to_well",
            "from_well",
            "from_patch",
            "to_patch",
            "is_reward",
            "is_patch_change",
        ]
    )


def _build_dot(
    leaf_bins: dict[int, int],
    junction_bins: int,
    *,
    event_id: int = 1,
) -> pd.DataFrame:
    """Construct a ContFrag dot frame with one SWR event whose bin
    composition is specified by ``leaf_bins`` (per-arm counts) and
    ``junction_bins`` (count of junction-decoded bins).

    Each arm-bin sets ``map_estimate`` to that arm's first place-bin
    (``arm * 2``); junction bins set ``map_estimate`` to NaN —
    ``map_estimate_to_arm`` returns NaN there, the builder's
    ``decoded_arm`` series fails ``isin(ALL_ARMS)``, and those bins
    are counted in ``n_junction_bins``. Continuous/Fragmented
    probabilities are set to (0.5, 0.5) on every in-event bin so the
    dominant-state machinery doesn't short-circuit. Outside the
    event, all bins are mobile and SWR_event=0.
    """
    n_in_event = sum(leaf_bins.values()) + junction_bins
    n_padding = 5  # mobile padding so the slice has clear edges
    total = n_in_event + n_padding * 2
    idx = pd.Index(np.arange(total, dtype=float) * 0.01, name="time")

    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 1,
            "epoch": "02_r1",
            "SWR_event": 0,
            "MUA_event": 0,
            "Continuous_prob": 0.5,
            "Fragmented_prob": 0.5,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )

    # Build the in-event slice: arm bins first, then junction bins.
    bin_order: list[float] = []
    for arm, count in leaf_bins.items():
        bin_order.extend([float(arm * 2)] * count)  # leaf place bin
    bin_order.extend([float("nan")] * junction_bins)  # junction → NaN map_estimate

    event_slice = slice(n_padding, n_padding + n_in_event)
    rows = dot.index[event_slice]
    dot.loc[rows, "SWR_event"] = event_id
    dot.loc[rows, "head_speed"] = 1.0  # immobile
    dot.loc[rows, "map_estimate"] = bin_order
    # ``track_segment_id`` supplies the independent physical-arm diagnostic.
    arm_with_most_bins = max(leaf_bins, key=leaf_bins.get) if leaf_bins else 0
    dot.loc[rows, "track_segment_id"] = float(arm_with_most_bins)
    return dot


def _build_three_event_dot() -> pd.DataFrame:
    """Three separated immobile events with different decoded arms."""
    idx = pd.Index(np.arange(60, dtype=float) * 0.01, name="time")
    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 1,
            "epoch": "02_r1",
            "SWR_event": 0,
            "MUA_event": 0,
            "Continuous_prob": 0.5,
            "Fragmented_prob": 0.5,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )
    for event_id, (start, arm) in enumerate(((5, 0), (25, 2), (45, 4)), start=1):
        rows = dot.index[start : start + 3]
        dot.loc[rows, "SWR_event"] = event_id
        dot.loc[rows, "head_speed"] = 1.0
        dot.loc[rows, "map_estimate"] = float(arm * 2)
        dot.loc[rows, "track_segment_id"] = float(arm)
    return dot


def _three_trial_history() -> pd.DataFrame:
    """Visits whose arrival and outcome clocks bracket the three events."""
    return pd.DataFrame(
        {
            "start_time": [-0.20, 0.10, 0.32],
            "goal_arrival_time": [0.00, 0.20, 0.40],
            "reward_on_time": [0.01, 0.255, np.nan],
            "outcome_known_time": [0.01, 0.255, 0.42],
            "end_time": [0.02, 0.30, 0.42],
            "from_well": [1, 0, 2],
            "to_well": [0, 2, 4],
            "from_patch": [1, 1, 2],
            "to_patch": [1, 2, 3],
            "is_reward": [True, True, False],
            "is_patch_change": [False, True, True],
        }
    )


class TestContfragArmFractionDenominator:
    """Production-builder regression for the ContFrag arm-fraction
    denominator. The pre-fix builder divided by ``n_timebins`` (all
    event bins, including junctions); the post-fix builder divides
    by ``n_timebins - n_junction_bins`` so the per-arm fractions sum
    to 1.0 over an event's real bins.
    """

    @pytest.fixture
    def env(self):
        return _EnvStub()

    @pytest.fixture
    def graph(self):
        return _bandit_graph()

    def test_per_event_fractions_sum_to_one_with_junction_bins(self, env, graph):
        # 5 real bins (3 on arm 0, 2 on arm 1) + 5 junction bins.
        # Pre-fix denominator was n_timebins=10 → fractions sum to
        # (3+2)/10 = 0.5. Post-fix divides by 5 → fractions sum to 1.
        dot = _build_dot({0: 3, 1: 2}, junction_bins=5)
        ev = build_contfrag_event_table(
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
        )
        assert len(ev) == 1, f"expected 1 event in fixture; got {len(ev)}"

        eat = build_contfrag_event_arm_table(
            ev,
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
        )
        # 9 arms × 1 event.
        assert len(eat) == 9
        total = float(eat["arm_fraction"].sum())
        assert np.isclose(total, 1.0, atol=1e-12), (
            f"per-event arm_fraction sum = {total} (expected 1.0). "
            "If this drifted to 0.5, the builder is dividing by "
            "n_timebins again instead of n_timebins - n_junction_bins."
        )
        # And the count split should match the input: arm 0 = 3/5,
        # arm 1 = 2/5, others 0.
        by_arm = eat.set_index("candidate_arm")["arm_fraction"]
        assert np.isclose(by_arm.loc[0], 3 / 5)
        assert np.isclose(by_arm.loc[1], 2 / 5)

    def test_posterior_fraction_is_emitted_and_normalized(self, env, graph):
        dot = _build_dot({0: 3, 1: 2}, junction_bins=5)
        events = build_contfrag_event_table(
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
        )
        n_position_bins = len(env.place_bin_centers_nodes_df_["edge_id"])
        posterior = xr.DataArray(
            np.full((len(dot), n_position_bins), 1.0 / n_position_bins),
            dims=("time", "position"),
            coords={"time": dot.index.to_numpy()},
        )
        event_arms = build_contfrag_event_arm_table(
            events,
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
            conditional_posterior=posterior,
        )
        assert "arm_fraction_posterior" in event_arms
        assert event_arms["arm_fraction_posterior"].sum() == pytest.approx(1.0)

    def test_event_with_only_junction_bins_yields_nan_arm_fraction(self, env, graph):
        # All 5 in-event bins land on a junction → no real arms →
        # denominator is zero → every arm_fraction is NaN.
        dot = _build_dot({}, junction_bins=5)
        ev = build_contfrag_event_table(
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
        )
        assert len(ev) == 1
        eat = build_contfrag_event_arm_table(
            ev,
            dot,
            env,
            _empty_trials(),
            graph,
            event_source="SWR",
        )
        assert eat["arm_fraction"].isna().all(), (
            "every-junction event should propagate NaN arm_fraction; "
            f"got {eat['arm_fraction'].tolist()}."
        )

    def test_novelty_column_and_swr_interaction_follow_completed_visits(
        self, env, graph
    ):
        dot = _build_dot({0: 3, 1: 2}, junction_bins=0)
        trials = pd.DataFrame(
            {
                "start_time": [-1.0],
                "goal_arrival_time": [-0.5],
                "end_time": [-0.4],
                "from_well": [1],
                "to_well": [0],
                "from_patch": [1],
                "to_patch": [1],
                "is_reward": [True],
                "is_patch_change": [False],
            }
        )
        ev = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
        eat = build_contfrag_event_arm_table(
            ev, dot, env, trials, graph, event_source="SWR"
        ).set_index("candidate_arm")

        expected = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float)
        np.testing.assert_array_equal(
            eat.loc[range(9), "is_candidate_arm_never_visited"], expected
        )
        np.testing.assert_array_equal(
            eat.loc[range(9), "swr_x_is_candidate_arm_never_visited"], expected
        )

    def test_history_is_frozen_at_event_start_when_event_crosses_arrival(
        self, env, graph
    ):
        dot = _build_dot({0: 3, 1: 2}, junction_bins=0)
        trials = pd.DataFrame(
            {
                "start_time": [-1.0],
                # Event spans 0.05--0.09 and has midpoint 0.07. The arrival
                # would leak into history under the retired midpoint cutoff.
                "goal_arrival_time": [0.07],
                "end_time": [0.10],
                "from_well": [1],
                "to_well": [0],
                "from_patch": [1],
                "to_patch": [1],
                "is_reward": [True],
                "is_patch_change": [False],
            }
        )

        ev = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
        assert ev.iloc[0]["occupancy_state"] == "mixed"
        eat = build_contfrag_event_arm_table(
            ev, dot, env, trials, graph, event_source="SWR"
        )

        by_arm = eat.set_index("candidate_arm")
        # The event precedes the first movement destination, but the starting
        # port is already a valid past experience. The destination remains
        # novel, proving event-start rather than midpoint alignment.
        assert by_arm.loc[1, "visit_recency_trials"] == 1.0
        assert by_arm.loc[1, "is_candidate_arm_never_visited"] == 0.0
        assert by_arm.loc[0, "visit_recency_trials"] == 2.0
        assert by_arm.loc[0, "is_candidate_arm_never_visited"] == 1.0

    def test_batched_history_matches_scalar_reference_at_each_event_start(
        self, env, graph
    ):
        """Batching preserves row order, NaNs, and the event-start cutoff.

        Event 2 begins before its current reward is delivered but ends after
        delivery.  The scalar reference is evaluated at each event start, so
        this also detects an accidental switch back to midpoint history.
        """
        dot = _build_three_event_dot()
        trials = _three_trial_history()
        events = build_contfrag_event_table(dot, env, trials, graph, event_source="SWR")
        event_arms = build_contfrag_event_arm_table(
            events, dot, env, trials, graph, event_source="SWR"
        )

        assert events["event_start_s"].tolist() == pytest.approx([0.05, 0.25, 0.45])
        np.testing.assert_array_equal(
            event_arms["candidate_arm"].to_numpy().reshape(len(events), 9),
            np.tile(np.arange(9), (len(events), 1)),
        )

        scalar_functions = {
            "visit_recency_trials": visit_recency_per_arm,
            "is_candidate_arm_never_visited": arm_never_visited_per_arm,
            "reward_count_candidate_arm_last10_known_trials": rolling_reward_count_per_arm,
            "reward_fraction_candidate_arm_last10_visits": lambda t, trials, graph: (
                recent_reward_rate_per_arm(t, trials, graph, k=10)
            ),
        }
        for column, function in scalar_functions.items():
            expected = np.asarray(
                [
                    [function(t, trials, graph)[arm] for arm in range(9)]
                    for t in events["event_start_s"].to_numpy(dtype=float)
                ]
            )
            actual = event_arms[column].to_numpy(dtype=float).reshape(len(events), 9)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        start_reward = rolling_reward_count_per_arm(0.25, trials, graph)[2]
        midpoint_reward = rolling_reward_count_per_arm(0.26, trials, graph)[2]
        assert start_reward == 0.0
        assert midpoint_reward == 1.0
        assert (
            event_arms.loc[
                (event_arms["SWR_event"] == 2) & (event_arms["candidate_arm"] == 2),
                "reward_count_candidate_arm_last10_known_trials",
            ].item()
            == start_reward
        )
