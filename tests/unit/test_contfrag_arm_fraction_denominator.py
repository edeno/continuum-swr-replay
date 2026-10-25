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
from continuum_swr_replay.analysis.replay._helpers import build_well_to_arm_map
from continuum_swr_replay.analysis.replay.covariates import (
    arm_never_visited_all_events,
    recent_reward_rate_all_events,
    rolling_reward_count_all_events,
    visit_recency_all_events,
)
from continuum_swr_replay.analysis.replay.tables_arm_patch import (
    build_event_patch_table,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
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


def arm_history(function, times, trials, graph):
    """``(n_times, 9)`` history over the nine arms from a vectorized helper."""
    return function(
        np.asarray(times, dtype=float),
        _experiences(trials),
        build_well_to_arm_map(graph),
    )


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
            "movement_start_time": [-0.20, 0.10, 0.32],
            "destination_first_poke_in_time": [0.00, 0.20, 0.40],
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
                "movement_start_time": [-1.0],
                "destination_first_poke_in_time": [-0.5],
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
                "movement_start_time": [-1.0],
                # Event spans 0.05--0.09 and has midpoint 0.07. The arrival
                # would leak into history under the retired midpoint cutoff.
                "destination_first_poke_in_time": [0.07],
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

        history_functions = {
            "visit_recency_trials": visit_recency_all_events,
            "is_candidate_arm_never_visited": arm_never_visited_all_events,
            "reward_count_candidate_arm_last10_known_trials": (
                rolling_reward_count_all_events
            ),
            "reward_fraction_candidate_arm_last10_visits": recent_reward_rate_all_events,
        }
        event_start_times = events["event_start_s"].to_numpy(dtype=float)
        for column, function in history_functions.items():
            expected = arm_history(function, event_start_times, trials, graph)
            actual = event_arms[column].to_numpy(dtype=float).reshape(len(events), 9)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        reward_counts = arm_history(
            rolling_reward_count_all_events, [0.25, 0.26], trials, graph
        )
        start_reward, midpoint_reward = reward_counts[0, 2], reward_counts[1, 2]
        assert start_reward == 0.0
        assert midpoint_reward == 1.0
        assert (
            event_arms.loc[
                (event_arms["SWR_event"] == 2) & (event_arms["candidate_arm"] == 2),
                "reward_count_candidate_arm_last10_known_trials",
            ].item()
            == start_reward
        )


def _build_two_event_dot() -> pd.DataFrame:
    """Two SWR events: one with undecoded (junction) bins, one without.

    Event 1 has 5 real bins (3 on arm 0, 2 on arm 1) and 5 junction bins, so its
    bin total (``n_timebins`` = 10) differs from its decoded-bin count (5).
    Event 2 has 5 real bins (2 on arm 2, 2 on arm 4, 1 on arm 5) and no
    junction bins. The two frames are separated in time by one second.
    """
    first = _build_dot({0: 3, 1: 2}, junction_bins=5, event_id=1)
    second = _build_dot({2: 2, 4: 2, 5: 1}, junction_bins=0, event_id=2)
    second.index = pd.Index(second.index.to_numpy() + 1.0, name="time")
    return pd.concat([first, second])


class TestContfragEventPatchTable:
    """``build_event_patch_table`` accepts the real ContFrag builders' tables.

    The ContFrag event table carries ``n_timebins``, not ``n_nonlocal_bins``.
    The patch builder used to require ``n_nonlocal_bins`` and so raised on every
    ContFrag table. Its ``patch_fraction`` must be the sum of the ContFrag
    ``arm_fraction`` of the patch's three arms.
    """

    @pytest.fixture
    def tables(self):
        env = _EnvStub()
        graph = _bandit_graph()
        dot = _build_two_event_dot()
        events = build_contfrag_event_table(
            dot, env, _empty_trials(), graph, event_source="SWR"
        )
        event_arms = build_contfrag_event_arm_table(
            events, dot, env, _empty_trials(), graph, event_source="SWR"
        )
        return events, event_arms

    def test_fixture_has_an_event_with_undecoded_bins(self, tables):
        events, _ = tables
        assert "n_nonlocal_bins" not in events
        by_event = events.sort_values("event_start_s")
        assert by_event["n_timebins"].tolist() == [10, 5]
        assert by_event["n_undecoded_bins"].tolist() == [5, 0]

    def test_patch_table_builds_from_contfrag_tables(self, tables):
        events, event_arms = tables
        patches = build_event_patch_table(events, event_arms)
        assert len(patches) == 3 * len(events)

    def test_patch_fraction_is_sum_of_arm_fractions(self, tables):
        events, event_arms = tables
        patches = build_event_patch_table(events, event_arms)
        arm_patch = event_arms.assign(
            patch=event_arms["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)
        )
        expected = (
            arm_patch.groupby(["event_id", "patch"])["arm_fraction"]
            .sum(min_count=1)
            .rename("expected")
            .reset_index()
        )
        merged = patches.merge(
            expected, on=["event_id", "patch"], validate="one_to_one"
        )
        assert len(merged) == len(patches)
        np.testing.assert_allclose(
            merged["patch_fraction"].to_numpy(dtype=float),
            merged["expected"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )

    def test_patch_fractions_sum_to_one_within_each_event(self, tables):
        events, event_arms = tables
        patches = build_event_patch_table(events, event_arms)
        totals = patches.groupby("event_id")["patch_fraction"].sum()
        assert len(totals) == len(events)
        np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=0.0, atol=1e-12)

    def test_undecoded_event_is_divided_by_decoded_bins_not_n_timebins(self, tables):
        events, event_arms = tables
        patches = build_event_patch_table(events, event_arms)
        first = events.loc[events["n_undecoded_bins"] > 0, "event_id"].item()
        by_patch = patches[patches["event_id"] == first].set_index("patch")
        # 5 real bins in patch 1 (arms 0 and 1) out of 5 decoded bins.
        assert by_patch.loc[1, "patch_fraction"] == pytest.approx(1.0)
        assert by_patch.loc[2, "patch_fraction"] == pytest.approx(0.0)
        assert by_patch.loc[3, "patch_fraction"] == pytest.approx(0.0)


def _synthetic_patch_inputs(**event_columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One event, nine arms: 4 hard counts on arm 0 and 4 on arm 2."""
    arms = list(range(9))
    event_arms = pd.DataFrame(
        {
            "event_id": [1] * 9,
            "candidate_arm": arms,
            "arm_count_hard": [4.0, 0, 4.0, 0, 0, 0, 0, 0, 0],
            "is_same_patch_as_current": np.zeros(9, dtype=int),
            "reward_count_candidate_arm_last10_known_trials": np.zeros(9),
            "visit_recency_trials": np.full(9, np.nan),
            "is_candidate_arm_never_visited": np.full(9, np.nan),
        }
    )
    events = pd.DataFrame(
        {
            "animal": ["test"],
            "session_date": [20200101],
            "epoch": ["01_r1"],
            "event_id": [1],
            "last_rewarded_arm": [np.nan],
            "prev_patch": [np.nan],
            "did_last_completed_trip_switch_patch": [0],
            "is_swr": [False],
            **{name: [value] for name, value in event_columns.items()},
        }
    )
    return events, event_arms


class TestPatchFractionBinTotalColumn:
    def test_nonlocal_bins_wins_when_both_columns_exist(self):
        events, event_arms = _synthetic_patch_inputs(
            n_nonlocal_bins=10, n_timebins=99, n_undecoded_bins=2
        )
        by_patch = build_event_patch_table(events, event_arms).set_index("patch")
        # (10 - 2) = 8 decoded bins; 99 would give 4 / 97.
        assert by_patch["patch_fraction"].tolist() == pytest.approx([0.5, 0.5, 0.0])

    def test_n_timebins_is_used_when_n_nonlocal_bins_is_absent(self):
        events, event_arms = _synthetic_patch_inputs(n_timebins=10, n_undecoded_bins=2)
        by_patch = build_event_patch_table(events, event_arms).set_index("patch")
        assert by_patch["patch_fraction"].tolist() == pytest.approx([0.5, 0.5, 0.0])

    def test_neither_bin_total_column_raises_and_names_both(self):
        events, event_arms = _synthetic_patch_inputs(n_undecoded_bins=2)
        with pytest.raises(ValueError, match="n_nonlocal_bins") as excinfo:
            build_event_patch_table(events, event_arms)
        assert "n_timebins" in str(excinfo.value)

    def test_missing_n_undecoded_bins_raises(self):
        events, event_arms = _synthetic_patch_inputs(n_nonlocal_bins=10)
        with pytest.raises(ValueError, match="n_undecoded_bins"):
            build_event_patch_table(events, event_arms)
