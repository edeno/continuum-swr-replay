"""Tests for :mod:`continuum_swr_replay.analysis.replay.tables`.

Covers the table invariant list on a small synthetic session —
schema, within-event sum invariants, range invariants, cross-table ID
joins. Also contains one regression test for a known bug
(:func:`test_event_arm_table_respects_non_default_event_table_index`).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
)
from continuum_swr_replay.analysis.replay.tables import (
    EVENT_ARM_STANDARDIZED_COLS,
    build_event_arm_table,
    build_event_patch_table,
    build_event_sequence_table,
    build_event_table,
    build_stop_table,
)

# ── Fixtures ──


class _EnvStub:
    """Minimal ``Environment`` stand-in.

    ``place_bin_centers_nodes_df_`` holds 2 bins per real arm (0-8)
    plus 3 junction bins (edge_id = -1). That gives
    ``posterior_to_arm_marginal`` real arm-mass to marginalize and
    real junction-bin mass to exclude, exercising the junction-bin
    handling in the same path production code uses.
    """

    class _Df:
        # Per-arm bins spread along [0, 90); junction bins at 100.
        _arm_lp = np.arange(18, dtype=float)  # 2 bins per arm
        _arm_eid = np.repeat(np.arange(9), 2).astype(int)  # 0,0,1,1,...,8,8
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


def _tiny_dot_with_two_events() -> pd.DataFrame:
    """100 timebins, two non-local events on leaf arms 1 and 3.

    Event 1 (indices 20-29): animal on arm 1, SWR present, no MUA.
    Event 2 (indices 60-69): animal on arm 3, MUA present, no SWR.
    ``map_estimate`` at the first bin of each arm drives
    ``map_estimate_to_arm`` to the right arm.
    """
    idx = pd.Index(np.linspace(0, 10, 100), name="time")
    dot = pd.DataFrame(
        {
            "animal": "chimi",
            "day": 20200212,
            "epoch": "02_r1",
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
        },
        index=idx,
    )
    for ev_id, arm, sl, swr, mua in [
        (1, 1, slice(20, 30), 1, 0),
        (2, 3, slice(60, 70), 0, 1),
    ]:
        rows = dot.index[sl]
        dot.loc[rows, "non_local_event"] = ev_id
        dot.loc[rows, "Non-Local_Continuous_prob"] = 0.9
        dot.loc[rows, "head_speed"] = 1.0
        dot.loc[rows, "map_estimate"] = float(arm * 2)  # first bin of this arm
        dot.loc[rows, "track_segment_id"] = arm
        dot.loc[rows, "SWR_event"] = swr
        dot.loc[rows, "MUA_event"] = mua
    return dot


def _tiny_posterior(dot: pd.DataFrame) -> xr.DataArray:
    """Synthetic conditional non-local posterior aligned to ``dot``.

    Event 1 mass on arm 1 (bins 2-3), event 2 mass on arm 3 (bins
    6-7). Everywhere else uniform over real bins. Rows always sum to
    1 over place bins.
    """
    n_time = len(dot)
    n_bins = 21  # 18 arm bins + 3 junction
    post = np.full((n_time, n_bins), 1.0 / n_bins)  # baseline uniform
    ev_ids = dot["non_local_event"].to_numpy()
    for ev_id, arm_bins in [(1, (2, 3)), (2, (6, 7))]:
        mask = ev_ids == ev_id
        row = np.zeros(n_bins)
        # Put 90% on the target arm's two bins, 10% spread over rest.
        row[list(arm_bins)] = 0.45
        other_bins = np.setdiff1d(np.arange(n_bins), list(arm_bins))
        row[other_bins] = 0.10 / other_bins.size
        post[mask] = row
    return xr.DataArray(
        post,
        dims=("time", "position"),
        coords={"time": dot.index.to_numpy(), "position": np.arange(n_bins)},
    )


def _location_contract_session():
    """Three retained events: transit, at one well, and mixed."""
    times = np.arange(0.0, 9.01, 0.1)
    dot = pd.DataFrame(
        {
            "animal": "test",
            "day": 1,
            "epoch": "02_r1",
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "map_estimate": np.nan,
            "head_speed": 10.0,
            "track_segment_id": np.nan,
            "projected_x_position": np.nan,
            "projected_y_position": np.nan,
        },
        index=pd.Index(times, name="time"),
    )
    for event_id, start, end, physical_arm in (
        (1, 2.0, 2.4, 7),
        (2, 4.0, 4.4, 1),
        # Crosses the second goal arrival at 8.0. History must remain frozen
        # at the 7.8 event start, even though the midpoint is at arrival.
        (3, 7.8, 8.2, 7),
    ):
        mask = (dot.index >= start) & (dot.index <= end)
        dot.loc[mask, "non_local_event"] = event_id
        dot.loc[mask, "Non-Local_Continuous_prob"] = 0.9
        dot.loc[mask, "head_speed"] = 1.0
        dot.loc[mask, "map_estimate"] = 2.0
        dot.loc[mask, "track_segment_id"] = physical_arm

    trials = pd.DataFrame(
        {
            "start_time": [1.0, 6.5],
            "goal_arrival_time": [3.0, 8.0],
            "end_time": [3.0, 8.0],
            "from_well": [0, 1],
            "to_well": [1, 2],
            "is_reward": [False, True],
            "is_patch_change": [False, True],
        }
    )
    graph = _bandit_graph()
    for node in graph:
        graph.nodes[node]["pos"] = (float(node * 10), 0.0)
    dot.loc[:, "projected_x_position"] = 10.0
    dot.loc[:, "projected_y_position"] = 0.0
    return dot, _EnvStub(), trials, graph


def test_event_and_stop_tables_label_location_without_filtering_events():
    dot, env, trials, graph = _location_contract_session()

    events = build_event_table(dot, env, trials, graph).sort_values("non_local_event")
    stops = build_stop_table(dot, trials, graph).sort_values("stop_start")
    event_arms = build_event_arm_table(events, dot, env, trials, graph)
    event_patches = build_event_patch_table(events, event_arms)

    assert events["non_local_event"].tolist() == [1, 2, 3]
    assert events["location_state"].tolist() == [
        "in_transit",
        "at_well",
        "mixed",
    ]
    assert events["at_well"].tolist() == [False, True, False]
    np.testing.assert_allclose(
        events["location_well"], [np.nan, 1.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        events["location_arm"], [np.nan, 1.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        events["current_arm"], events["location_arm"], equal_nan=True
    )
    assert events["physical_arm"].tolist() == [7.0, 1.0, 7.0]
    assert (events["nearest_well_distance_cm_max"] == 0.0).all()
    assert events["time_since_last_port_entry"].dropna().ge(0).all()
    np.testing.assert_allclose(
        events["time_since_port_entry"],
        events["time_since_last_port_entry"],
        equal_nan=True,
    )
    mixed_history = event_arms.loc[
        event_arms["non_local_event"].eq(3) & event_arms["arm"].eq(2)
    ].iloc[0]
    mixed_event = events.loc[events["non_local_event"].eq(3)].iloc[0]
    assert mixed_event["time_since_last_port_entry"] == pytest.approx(4.8)
    assert pd.isna(mixed_event["time_since_reward"])
    assert mixed_history["arm_never_visited"] == 1.0
    assert mixed_history["visit_recency"] == 2.0

    # Each event is its own immobility bout in this fixture, so the stop and
    # event labels should agree exactly and no cohort row is lost.
    assert stops["location_state"].tolist() == events["location_state"].tolist()
    assert stops["at_well"].tolist() == events["at_well"].tolist()
    assert stops["completed_trial_id"].tolist() == [pd.NA, 0, 0]
    np.testing.assert_allclose(
        stops["completed_trial_arm"], [np.nan, 1.0, 1.0], equal_nan=True
    )
    np.testing.assert_allclose(
        stops["recent_switch_rate"].to_numpy(dtype=float),
        [np.nan, 0.0, 0.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        stops["leaky_q_value_diff"].to_numpy(dtype=float),
        [np.nan, 0.0, 0.0],
        equal_nan=True,
    )
    assert int(stops["n_events"].sum()) == len(events) == 3
    for table, rows_per_event in ((event_arms, 9), (event_patches, 3)):
        assert len(table) == len(events) * rows_per_event
        states = table.groupby("event_id", sort=False)["location_state"].first()
        expected = events.set_index("event_id")["location_state"].reindex(states.index)
        pd.testing.assert_series_equal(states, expected, check_names=False)


@pytest.fixture
def tiny_session():
    dot = _tiny_dot_with_two_events()
    env = _EnvStub()
    trials = pd.DataFrame(
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
    graph = _bandit_graph()
    return dot, env, trials, graph


@pytest.fixture
def tiny_session_with_posterior(tiny_session):
    dot, env, trials, graph = tiny_session
    return dot, env, trials, graph, _tiny_posterior(dot)


# ── Regression tests ──


def test_physical_trunk_is_kept_separate_from_at_well_location_arm():
    """A modal trunk segment must not overwrite behavioral well location."""
    dot = _tiny_dot_with_two_events()
    # Force both events onto trunk arm 7 (patch junction 7 ↔ center 9).
    for sl in (slice(20, 30), slice(60, 70)):
        rows = dot.index[sl]
        dot.loc[rows, "track_segment_id"] = 7

    # Stub trials with a single completed trip to well 3. Both events occur
    # during the subsequent dwell even though this synthetic physical-segment
    # field says trunk 7; the two facts must be exposed independently.
    trials = pd.DataFrame(
        {
            "start_time": [0.0],
            "end_time": [1.0],
            "from_well": [0],
            "to_well": [3],
            "from_patch": [1],
            "to_patch": [2],
            "is_reward": [True],
            "is_patch_change": [True],
        }
    )
    env = _EnvStub()
    graph = _bandit_graph()

    et = build_event_table(dot, env, trials, graph)
    assert len(et) == 2
    assert (et["location_state"] == "at_well").all()
    assert (et["location_arm"] == 3).all()
    assert (et["current_arm"] == 3).all(), (
        f"location-arm compatibility alias failed — got {et['current_arm'].tolist()}"
    )
    assert (et["physical_arm"] == 7).all()


def test_event_arm_table_respects_non_default_event_table_index(tiny_session):
    """``build_event_arm_table`` must tolerate any index on ``event_table``.

    Regression for a bug where the builder used the DataFrame's index
    label as a positional offset into the per-event predictor arrays.
    Worked on a ``RangeIndex(start=0)`` but silently corrupted the
    output for any frame produced by ``pd.concat`` across sessions
    (whose index is no longer zero-based).
    """
    dot, env, trials, graph = tiny_session
    et = build_event_table(dot, env, trials, graph)
    assert len(et) == 2

    # Shift the index to simulate concat of a prior session.
    shifted = et.copy()
    shifted.index = shifted.index + 500

    eat_shifted = build_event_arm_table(shifted, dot, env, trials, graph)
    eat_baseline = build_event_arm_table(et, dot, env, trials, graph)

    # Output must have 9 rows per event regardless of index.
    assert len(eat_shifted) == 2 * 9

    # Per-event per-arm predictor columns populated for every event.
    # Before the fix these were NaN (or IndexError) for everything
    # past the first session's rows.
    for col in ("dist_to_arm", "arm_dist_rank"):
        # ``dist_to_arm`` is NaN when current_arm is NaN — accept that,
        # but require the number of non-NaN rows to match the baseline.
        assert eat_shifted[col].notna().sum() == eat_baseline[col].notna().sum(), (
            f"column {col} lost values when event_table had a non-zero-based index"
        )


def test_event_arm_table_emits_time_local_novelty_and_swr_interaction(tiny_session):
    """Novelty follows completed visits and its SWR interaction is exact."""
    dot, env, empty_trials, graph = tiny_session
    et = build_event_table(dot, env, empty_trials, graph)
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 4.0],
            "goal_arrival_time": [1.0, 5.0],
            "end_time": [1.5, 5.5],
            "from_well": [1, 0],
            "to_well": [0, 1],
            "from_patch": [1, 1],
            "to_patch": [1, 1],
            "is_reward": [True, False],
            "is_patch_change": [False, False],
        }
    )

    eat = build_event_arm_table(et, dot, env, trials, graph)
    event_ids = et["event_id"].tolist()
    first = eat[eat["event_id"] == event_ids[0]].set_index("arm")
    second = eat[eat["event_id"] == event_ids[1]].set_index("arm")

    np.testing.assert_array_equal(
        first.loc[range(9), "arm_never_visited"].to_numpy(),
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        second.loc[range(9), "arm_never_visited"].to_numpy(),
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    for frame in (first, second):
        # Centering uses the *visited* leaf arms only, so it is their mean
        # that is zero. Including the never-visited arms -- whose value is
        # the right-censoring fill rather than a measured recency -- would
        # drag the visited arms by an arbitrary amount.
        novelty = frame.loc[range(6), "arm_never_visited"].to_numpy()
        demeaned = frame.loc[range(6), "visit_recency_demeaned"].to_numpy()
        assert demeaned[novelty == 0.0].mean() == pytest.approx(0.0)
        # The censored arms stay above every visited arm rather than
        # pulling them negative.
        assert demeaned[novelty == 1.0].min() > demeaned[novelty == 0.0].max()
        np.testing.assert_array_equal(
            frame.loc[range(6, 9), "visit_recency_demeaned"].to_numpy(),
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            frame["swr_x_arm_never_visited"].to_numpy(),
            frame["is_swr"].astype(int).to_numpy()
            * frame["arm_never_visited"].to_numpy(),
        )


def test_value_block_predictors_resolve_and_populate(tiny_session):
    """Mv/Pv predictors resolve to builder columns, and the contingency
    join populates them end-to-end.

    Guards the central rename-desync risk of the harmonization PR: a model
    spec naming a column the builders no longer emit, and the NonLocal
    contingency join (build_event_arm_table(contingency=...)) producing values
    rather than silently staying NaN.
    """
    from continuum_swr_replay.analysis.replay.model_specs import (
        ARM_FE_PREDICTORS,
        MODEL_HIERARCHY,
        MODEL_PATCH,
    )

    dot, env, trials, graph = tiny_session
    et = build_event_table(dot, env, trials, graph)
    eat = build_event_arm_table(et, dot, env, trials, graph)

    # value-block columns are emitted (present even without a contingency)
    for col in ("reward_rate_arm_k10", "p_reward_arm", "q_arm", "arm_uncertainty"):
        assert col in eat.columns

    # every Mv predictor (bar the fit-time arm fixed effects) resolves
    arm_fe = set(ARM_FE_PREDICTORS)
    mv = {p for p in MODEL_HIERARCHY["Mv"].predictors if p not in arm_fe}
    assert mv <= set(eat.columns), f"Mv predictors absent: {mv - set(eat.columns)}"

    # patch table emits the patch value aggregates; every Pv predictor resolves
    ept = build_event_patch_table(et, eat)
    pv = {p for p in MODEL_PATCH["Pv"].predictors if p not in arm_fe}
    assert pv <= set(ept.columns), f"Pv predictors absent: {pv - set(ept.columns)}"

    # contingency join routes each leaf arm to ITS OWN port value (distinct per
    # port -> catches a broadcast/transpose bug that `.notna().any()` would miss)
    stop_ids = eat["stop_id"].dropna().unique()
    if len(stop_ids):
        cont = pd.DataFrame({"stop_id": stop_ids})
        for i in range(6):
            cont[f"p_reward_port{i}"] = 0.1 * (i + 1)
            cont[f"q_port{i}"] = 0.05 * (i + 1)
            cont[f"betadist_var{i + 1}"] = 0.01 * (i + 1)  # beta variance (1-indexed)
        eat_c = build_event_arm_table(et, dot, env, trials, graph, contingency=cont)
        present = eat_c["stop_id"].isin(stop_ids)
        leaf = present & (eat_c["arm"] < 6)
        arm_j = eat_c.loc[leaf, "arm"].to_numpy()
        np.testing.assert_allclose(
            eat_c.loc[leaf, "p_reward_arm"].to_numpy(), 0.1 * (arm_j + 1)
        )
        np.testing.assert_allclose(
            eat_c.loc[leaf, "q_arm"].to_numpy(), 0.05 * (arm_j + 1)
        )
        # arm_uncertainty routes leaf arm j to betadist_var{j+1} (variance counterpart)
        np.testing.assert_allclose(
            eat_c.loc[leaf, "arm_uncertainty"].to_numpy(), 0.01 * (arm_j + 1)
        )
        # trunk arms (6-8) are not reward sites -> structural 0.0 for a present stop
        trunk = present & (eat_c["arm"] >= 6)
        assert (eat_c.loc[trunk, "p_reward_arm"] == 0.0).all()
        assert (eat_c.loc[trunk, "q_arm"] == 0.0).all()
        assert (eat_c.loc[trunk, "arm_uncertainty"] == 0.0).all()


def test_value_block_absent_stop_drops_all_arms(tiny_session):
    """A stop not in the contingency -> ALL nine arms NaN (uniform drop at fit
    time), not a spurious 0.0; stops that ARE present still populate. Guards the
    builder's stop_id keying through the real prepare_data call path."""
    dot, env, trials, graph = tiny_session
    et = build_event_table(dot, env, trials, graph)
    eat0 = build_event_arm_table(et, dot, env, trials, graph)
    stop_ids = eat0["stop_id"].dropna().unique()
    if len(stop_ids) < 2:
        pytest.skip("need >=2 stops to split present/absent")
    covered = stop_ids[: max(1, len(stop_ids) // 2)]  # contingency covers only some
    cont = pd.DataFrame({"stop_id": covered})
    for i in range(6):
        cont[f"p_reward_port{i}"] = 0.1 * (i + 1)
        cont[f"q_port{i}"] = 0.05 * (i + 1)
        cont[f"betadist_var{i + 1}"] = 0.01 * (i + 1)
    eat = build_event_arm_table(et, dot, env, trials, graph, contingency=cont)

    absent = eat[eat["stop_id"].notna() & ~eat["stop_id"].isin(covered)]
    assert len(absent), "fixture produced no absent-stop rows"
    assert absent["p_reward_arm"].isna().all()  # uniform NaN, not 0.0
    assert absent["q_arm"].isna().all()
    assert absent["arm_uncertainty"].isna().all()
    present_leaf = eat[eat["stop_id"].isin(covered) & (eat["arm"] < 6)]
    assert present_leaf["p_reward_arm"].notna().all()


# ── Schema / dtype tests ──


class TestEventTableSchema:
    def test_canonical_columns_present(self, tiny_session_with_posterior):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        for col in (
            "animal",
            "day",
            "epoch",
            "non_local_event",
            "event_id",
            "stop_idx",
            "stop_id",
            "event_start",
            "event_end",
            "event_duration",
            "time_since_stop",
            "is_swr",
            "is_mua",
            "whole_event_speed_eligible",
            "current_arm",
            "prev_arm",
            "next_arm",
            "last_rewarded_arm",
            "n_nonlocal_bins",
            "nonlocal_weight_sum",
            "posterior_mass_total",
            "posterior_peak_mean",
            "posterior_entropy_mean",
            "posterior_hpd_width_50",
            "posterior_hpd_width_80",
            "posterior_hpd_width_95",
            "n_junction_bins",
            "n_unique_arms",
        ):
            assert col in et.columns, f"missing canonical column: {col}"

    def test_bool_columns_are_bool(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        assert et["is_swr"].dtype == bool
        assert et["is_mua"].dtype == bool
        assert et["whole_event_speed_eligible"].dtype == bool
        assert et["whole_event_speed_eligible"].all()
        assert str(et["at_well"].dtype) == "boolean"

    def test_empty_session_returns_empty_frame_with_schema(self, tiny_session):
        dot, env, trials, graph = tiny_session
        dot = dot.copy()
        dot["non_local_event"] = 0  # no events
        et = build_event_table(dot, env, trials, graph)
        assert len(et) == 0
        # Column list is still the canonical one.
        assert "event_id" in et.columns
        assert "stop_id" in et.columns


# ── Within-event sum invariants ──


class TestEventTableInvariants:
    def test_custom_nonlocal_threshold_reaches_classified_table(
        self, tiny_session_with_posterior
    ):
        """Every stage validates labeled probabilities against one threshold."""
        dot, env, trials, graph, posterior = tiny_session_with_posterior
        custom = dot.copy()
        event_one = custom["non_local_event"].to_numpy() == 1
        custom.loc[event_one, "Non-Local_Continuous_prob"] = 0.75
        custom.loc[event_one, "Non-Local_Fragmented_prob"] = 0.0

        table = build_event_table(
            custom,
            env,
            trials,
            graph,
            conditional_posterior=posterior,
            nonlocal_threshold=0.7,
        )

        assert 1 in table["non_local_event"].to_numpy()

    def test_probability_contract_is_reported_before_speed_contract(
        self, tiny_session_with_posterior
    ):
        """A doubly invalid label reports the upstream detector violation."""
        dot, env, trials, graph, posterior = tiny_session_with_posterior
        invalid = dot.copy()
        event_rows = np.flatnonzero(invalid["non_local_event"].to_numpy() == 1)
        middle = event_rows[len(event_rows) // 2]
        invalid.iloc[middle, invalid.columns.get_loc("head_speed")] = 10.0
        invalid.iloc[
            middle,
            invalid.columns.get_loc("Non-Local_Continuous_prob"),
        ] = 0.5
        invalid.iloc[
            middle,
            invalid.columns.get_loc("Non-Local_Fragmented_prob"),
        ] = 0.0

        with pytest.raises(
            EventDefinitionError, match="probability-threshold contract"
        ):
            build_event_table(
                invalid,
                env,
                trials,
                graph,
                conditional_posterior=posterior,
            )

    @pytest.mark.parametrize("bad_speed", [4.0, np.nan, np.inf, -np.inf, 10.0])
    def test_nonlocal_event_speed_contract_fails_atomically(
        self, tiny_session_with_posterior, bad_speed
    ):
        """Legacy labels with one bad bin fail instead of being truncated."""
        dot, env, trials, graph, posterior = tiny_session_with_posterior
        clean_event_table = build_event_table(
            dot, env, trials, graph, conditional_posterior=posterior
        )
        invalid = dot.copy()
        event_rows = np.flatnonzero(invalid["non_local_event"].to_numpy() == 1)
        invalid.iloc[
            event_rows[len(event_rows) // 2],
            invalid.columns.get_loc("head_speed"),
        ] = bad_speed

        match = "whole-event immobility contract"
        with pytest.raises(EventDefinitionError, match=match):
            build_event_table(
                invalid,
                env,
                trials,
                graph,
                conditional_posterior=posterior,
            )
        with pytest.raises(EventDefinitionError, match=match):
            build_event_sequence_table(invalid, env, posterior)
        with pytest.raises(EventDefinitionError, match=match):
            build_event_arm_table(
                clean_event_table,
                invalid,
                env,
                trials,
                graph,
                conditional_posterior=posterior,
            )
        with pytest.raises(EventDefinitionError, match=match):
            build_stop_table(invalid, trials, graph)

    @pytest.mark.parametrize("bad_probability", [0.8, 0.5, np.nan, np.inf, -np.inf])
    def test_nonlocal_probability_contract_fails_atomically(
        self, tiny_session_with_posterior, bad_probability
    ):
        """A bad interior probability rejects stale labels without clipping."""
        dot, env, trials, graph, posterior = tiny_session_with_posterior
        clean_event_table = build_event_table(
            dot, env, trials, graph, conditional_posterior=posterior
        )
        invalid = dot.copy()
        event_rows = np.flatnonzero(invalid["non_local_event"].to_numpy() == 1)
        middle = event_rows[len(event_rows) // 2]
        invalid.iloc[
            middle,
            invalid.columns.get_loc("Non-Local_Continuous_prob"),
        ] = bad_probability
        invalid.iloc[
            middle,
            invalid.columns.get_loc("Non-Local_Fragmented_prob"),
        ] = 0.0

        match = "probability-threshold contract"
        with pytest.raises(EventDefinitionError, match=match):
            build_event_table(
                invalid,
                env,
                trials,
                graph,
                conditional_posterior=posterior,
            )
        with pytest.raises(EventDefinitionError, match=match):
            build_event_sequence_table(invalid, env, posterior)
        with pytest.raises(EventDefinitionError, match=match):
            build_event_arm_table(
                clean_event_table,
                invalid,
                env,
                trials,
                graph,
                conditional_posterior=posterior,
            )
        with pytest.raises(EventDefinitionError, match=match):
            build_stop_table(invalid, trials, graph)

    @pytest.mark.parametrize(
        ("mua_start_offset", "expected_is_mua"),
        [(0, False), (-1, True)],
    )
    def test_mua_uses_strict_event_level_overlap(
        self, tiny_session, mua_start_offset, expected_is_mua
    ):
        """Boundary contact is not overlap; positive duration marks all bins."""
        dot, env, trials, graph = tiny_session
        dot = dot.copy()
        dot["MUA_event"] = 0
        event_rows = np.flatnonzero(dot["non_local_event"].to_numpy() == 1)
        start = event_rows[-1] + mua_start_offset
        stop = event_rows[-1] + 4
        dot.iloc[start:stop, dot.columns.get_loc("MUA_event")] = 7
        dot.iloc[start:stop, dot.columns.get_loc("head_speed")] = 1.0

        event = build_event_table(dot, env, trials, graph)
        event_1 = event.loc[event["non_local_event"] == 1].iloc[0]

        assert bool(event_1["is_mua"]) is expected_is_mua

    def test_n_nonlocal_bins_matches_groupby_size(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        # Event 1 covers slice(20,30) = 10 bins. Event 2 covers 10 bins.
        np.testing.assert_array_equal(et["n_nonlocal_bins"].to_numpy(), [10, 10])

    def test_nonlocal_weight_sum_equals_sum_of_nonlocal_prob(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        # Non-Local_Continuous_prob = 0.9 × 10 bins = 9.0 per event.
        np.testing.assert_allclose(et["nonlocal_weight_sum"].to_numpy(), [9.0, 9.0])

    def test_legacy_and_label_timing_share_one_atomic_window(self, tiny_session):
        dot, env, trials, graph = tiny_session
        event = build_event_table(dot, env, trials, graph)

        for canonical, alias in (
            ("event_start", "label_event_start"),
            ("event_end", "label_event_end"),
            ("event_duration", "label_event_duration"),
            ("event_midpoint_time", "label_event_midpoint_time"),
        ):
            np.testing.assert_array_equal(event[canonical], event[alias])

    @pytest.mark.parametrize("external_col", ["SWR_event", "MUA_event"])
    def test_movement_spanning_external_event_is_excluded(
        self, tiny_session, external_col
    ):
        """A mobile SWR/MUA cannot mark an immobile non-local event.

        The external event overhangs the non-local window into mobile
        bins, so the whole-event rule rejects it while the non-local
        event itself stays eligible. Without the shared speed filter in
        ``_filter_external_event_labels`` / ``event_table`` the overhang
        is invisible and the flag flips to ``True``.
        """
        dot, env, trials, graph = tiny_session
        dot = dot.copy()
        dot[external_col] = 0
        # Event 1 spans bins 20-29 (head_speed 1.0). Bins 30-32 sit
        # outside it at head_speed 10.0, so the external event is
        # movement-spanning while event 1 remains immobile.
        dot.iloc[20:33, dot.columns.get_loc(external_col)] = 1
        assert dot["head_speed"].to_numpy()[30:33].min() > 4.0

        event = build_event_table(dot, env, trials, graph)
        event_1 = event.loc[event["non_local_event"] == 1].iloc[0]

        flag = "is_swr" if external_col == "SWR_event" else "is_mua"
        assert bool(event_1[flag]) is False

    @pytest.mark.parametrize("external_col", ["SWR_event", "MUA_event"])
    def test_stop_table_excludes_movement_spanning_external_event(
        self, tiny_session, external_col
    ):
        """Stop-level counts use the same speed-eligible cohort."""
        dot, env, trials, graph = tiny_session
        dot = dot.copy()
        dot[external_col] = 0
        dot.iloc[20:33, dot.columns.get_loc(external_col)] = 1

        stop = build_stop_table(dot, trials, graph)

        count_col = "n_SWR" if external_col == "SWR_event" else "n_MUA"
        assert stop[count_col].sum() == 0

    def test_nonlocal_prob_within_unit_interval(self, tiny_session):
        # ``nonlocal_prob`` values in [0, 1]. Exposed here via
        # ``nonlocal_weight_sum`` / ``n_nonlocal_bins``, which must be
        # between 0 and 1 when averaged.
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        avg = et["nonlocal_weight_sum"] / et["n_nonlocal_bins"]
        assert ((avg >= 0) & (avg <= 1)).all()

    def test_hpd_width_monotonic_50_80_95(self, tiny_session_with_posterior):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        assert (
            et["posterior_hpd_width_50"] <= et["posterior_hpd_width_80"] + 1e-9
        ).all()
        assert (
            et["posterior_hpd_width_80"] <= et["posterior_hpd_width_95"] + 1e-9
        ).all()

    def test_posterior_peak_mean_within_unit_interval(
        self, tiny_session_with_posterior
    ):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        assert (
            (et["posterior_peak_mean"] >= 0) & (et["posterior_peak_mean"] <= 1)
        ).all()

    def test_posterior_entropy_mean_nonnegative(self, tiny_session_with_posterior):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        assert (et["posterior_entropy_mean"] >= 0).all()

    def test_posterior_mass_total_equals_nonlocal_weight_sum(
        self, tiny_session_with_posterior
    ):
        """Under the renormalize convention in
        :func:`summaries.posterior_to_arm_marginal`, junction mass is
        redistributed over real arms so ``posterior_mass_total`` and
        ``nonlocal_weight_sum`` must be identical.
        """
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        np.testing.assert_allclose(
            et["posterior_mass_total"].to_numpy(),
            et["nonlocal_weight_sum"].to_numpy(),
            atol=1e-12,
        )


class TestEventArmTableInvariants:
    def test_nine_rows_per_event(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        assert len(eat) == 9 * len(et)

    def test_standardized_z_are_per_session_nan_placeholders(self, tiny_session):
        """Phase-6 build contract: the per-session build carries the ``<col>_z``
        columns (so the canonical order is stable) but leaves them all-NaN — the real
        pooled-GLOBAL z is filled post-concat in ``prepare_data.main``. This proves the
        per-session build adds only NaN placeholders (raw columns are built normally),
        i.e. the standardization did not leak a wrong per-session z into the table."""
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        for c in EVENT_ARM_STANDARDIZED_COLS:
            zc = f"{c}_z"
            assert zc in eat.columns, f"missing standardized placeholder {zc}"
            assert eat[zc].isna().all(), (
                f"{zc} must be an all-NaN placeholder in the per-session build "
                "(global z is filled post-concat in prepare_data)"
            )

    def test_sum_arm_count_hard_equals_n_bins_minus_junction(
        self, tiny_session_with_posterior
    ):
        """``sum(arm_count_hard) == n_nonlocal_bins``.

        Correct only when ``n_junction_bins == 0``; on the general
        definition the sum is ``n_nonlocal_bins − n_junction_bins``.
        Both quantities are in ``event_table``.
        """
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        eat = build_event_arm_table(
            et, dot, env, trials, graph, conditional_posterior=post
        )
        sum_hard = eat.groupby("event_id")["arm_count_hard"].sum()
        n_bins = et.set_index("event_id")["n_nonlocal_bins"].loc[sum_hard.index]
        n_junc = et.set_index("event_id")["n_junction_bins"].loc[sum_hard.index]
        np.testing.assert_array_equal(sum_hard.to_numpy(), (n_bins - n_junc).to_numpy())

    def test_arm_fraction_sums_to_1_when_no_junction_bins(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        # Our synthetic ``map_estimate`` always lands on a real arm,
        # so n_junction_bins is 0 and arm_fraction sums to exactly 1.
        assert (et["n_junction_bins"] == 0).all()
        frac_sum = eat.groupby("event_id")["arm_fraction"].sum()
        np.testing.assert_allclose(frac_sum.to_numpy(), 1.0, atol=1e-12)

    def test_arm_fraction_sums_to_1_when_junction_bins_present(self, tiny_session):
        """arm_fraction must sum to 1 per event even when some timebins
        have no arm attribution (NaN ``decoded_arm`` — counted as
        ``n_junction_bins``). ``arm_count_hard`` excludes those timebins
        from the numerator, so the denominator must exclude them too
        — otherwise the fraction deflates below 1 whenever any MAP
        failed to attribute to a real arm.
        """
        dot, env, trials, graph = tiny_session
        # Force half of event 1's timebins to have NaN map_estimate —
        # decoded_arm becomes NaN there, so n_junction_bins = 5.
        event_1_rows = dot.index[dot["non_local_event"] == 1]
        dot.loc[event_1_rows[:5], "map_estimate"] = np.nan

        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)

        # Event 1 now has junction bins; event 2 does not.
        ev1 = et[et["non_local_event"] == 1].iloc[0]
        assert ev1["n_junction_bins"] == 5
        assert ev1["n_nonlocal_bins"] == 10

        frac_sum = eat.groupby("non_local_event")["arm_fraction"].sum()
        np.testing.assert_allclose(frac_sum.to_numpy(), 1.0, atol=1e-12)

    def test_arm_fraction_nan_for_all_junction_event(self, tiny_session):
        """Events where every timebin has NaN decoded_arm have a
        denominator of zero — arm_fraction must be NaN for all 9 arms
        (not +inf, not 0, not silently 0/0)."""
        dot, env, trials, graph = tiny_session
        event_1_rows = dot.index[dot["non_local_event"] == 1]
        dot.loc[event_1_rows, "map_estimate"] = np.nan

        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)

        event_1_eat = eat[eat["non_local_event"] == 1]
        assert event_1_eat["arm_fraction"].isna().all()
        # Event 2 is untouched and still sums to 1.
        event_2_sum = eat[eat["non_local_event"] == 2]["arm_fraction"].sum()
        assert np.isclose(event_2_sum, 1.0)

    def test_sum_arm_posterior_mass_equals_posterior_mass_total(
        self, tiny_session_with_posterior
    ):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        eat = build_event_arm_table(
            et, dot, env, trials, graph, conditional_posterior=post
        )
        sum_mass = eat.groupby("event_id")["arm_posterior_mass"].sum()
        pmt = et.set_index("event_id")["posterior_mass_total"].loc[sum_mass.index]
        np.testing.assert_allclose(sum_mass.to_numpy(), pmt.to_numpy(), atol=1e-10)

    def test_is_current_indicator_is_valid(self, tiny_session):
        """With no trials every event is location-unknown, so
        ``is_current`` is missing rather than a fabricated zero.
        """
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        assert eat["is_current"].isna().all()

    def test_is_current_is_one_hot_at_well_and_missing_off_well(self):
        """``is_current`` is defined only where the animal's location is.

        An at-well event marks exactly one arm; transit and mixed events
        carry NaN on every arm, so complete-case fitting drops them instead
        of reading them as "not the current arm" nine times over.
        """
        dot, env, trials, graph = _location_contract_session()
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)

        state = et.set_index("event_id")["location_state"].to_dict()
        by_event = dict(eat.groupby("event_id").__iter__())
        assert len(by_event) == 3

        seen = set()
        for event_id, g in by_event.items():
            ic = g["is_current"].to_numpy(dtype=float)
            seen.add(state[event_id])
            if state[event_id] == "at_well":
                # Well 1 is degree-1, so it touches exactly arm 1.
                assert np.isfinite(ic).all()
                assert ic.sum() == 1.0
                assert g.loc[g["is_current"] == 1.0, "arm"].tolist() == [1]
                # The elsewhere-residual stays defined here.
                assert g["is_last_rewarded_elsewhere"].notna().all()
            else:
                assert np.isnan(ic).all(), (
                    f"{state[event_id]} leaked a numeric is_current"
                )
                assert g["is_same_patch"].isna().all()
                assert g["is_last_rewarded_elsewhere"].isna().all()
                assert g["swr_x_current"].isna().all()
        assert seen == {"in_transit", "at_well", "mixed"}

    def test_dist_to_arm_nonnegative_and_rank_1_is_min(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        for _, g in eat.groupby("event_id"):
            finite = g.dropna(subset=["dist_to_arm"])
            if finite.empty:
                continue
            assert (finite["dist_to_arm"] >= 0).all()
            rank_1 = finite[finite["arm_dist_rank"] == 1]
            assert rank_1["dist_to_arm"].min() == finite["dist_to_arm"].min()

    def test_interactions_zero_where_indicator_zero(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        eat = build_event_arm_table(et, dot, env, trials, graph)
        # For each interaction term the right-hand factor is a 0/1
        # indicator — every nonzero product row must have the
        # indicator == 1. We don't check the left-hand (numeric)
        # factor; the asymmetric check is enough to catch broadcast
        # drift.
        for prod, indicator_col in (
            ("time_x_current", "is_current"),
            ("swr_x_current", "is_current"),
            ("mua_x_prev", "is_prev"),
        ):
            p = eat[prod].to_numpy(dtype=float)
            ind = eat[indicator_col].to_numpy(dtype=float)
            # ``is_current`` is NaN where the animal's location is
            # unobserved; the product must carry that NaN through
            # rather than collapsing it to a number.
            assert np.array_equal(np.isnan(p), np.isnan(ind)), (
                f"{prod} does not propagate {indicator_col} missingness"
            )
            finite = ~np.isnan(p)
            nz_mask = finite & (p != 0)
            assert (ind[nz_mask] == 1).all(), (
                f"{prod} nonzero with {indicator_col} == 0"
            )


# ── Cross-table invariants ──


class TestCrossTable:
    def test_every_event_stop_id_exists_in_stop_table(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        st = build_stop_table(dot, trials, graph)
        assert set(et["stop_id"]).issubset(set(st["stop_id"]))

    def test_stop_table_n_events_matches_event_table_groupby(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        st = build_stop_table(dot, trials, graph)
        # Every stop's ``n_events`` equals the number of event_table
        # rows for that stop_id.
        from_events = et.groupby("stop_id").size()
        from_stops = st.set_index("stop_id")["n_events"]
        joined = from_events.align(from_stops, join="left")[0].fillna(0).astype(int)
        for stop_id, n in joined.items():
            if stop_id in from_stops.index:
                assert from_stops[stop_id] == n

    def test_sum_n_events_equals_event_table_length(self, tiny_session):
        dot, env, trials, graph = tiny_session
        et = build_event_table(dot, env, trials, graph)
        st = build_stop_table(dot, trials, graph)
        assert int(st["n_events"].sum()) == len(et)

    def test_n_non_local_is_identical_to_n_events(self, tiny_session):
        """The two stop-level count columns must never diverge.

        ``n_events`` is the response column the per-stop rate models fit and
        the only count name the ContFrag stop table carries; ``n_non_local``
        is the newer repo-wide name. Both are written from the same array,
        and consumers pick whichever they know
        (``tables.py`` reads ``out.get("n_non_local", out.get("n_events"))``),
        so a divergence would silently change which count a model saw. Only
        ``n_events`` was pinned, so setting ``n_non_local`` to something else
        passed the whole suite.
        """
        dot, env, trials, graph = tiny_session
        st = build_stop_table(dot, trials, graph)
        pd.testing.assert_series_equal(
            st["n_non_local"], st["n_events"], check_names=False
        )

    def test_event_sequence_row_count_equals_sum_n_nonlocal_bins(
        self, tiny_session_with_posterior
    ):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        est = build_event_sequence_table(dot, env, post)
        assert len(est) == int(et["n_nonlocal_bins"].sum())

    def test_event_sequence_arm_posterior_rows_sum_to_one(
        self, tiny_session_with_posterior
    ):
        dot, env, trials, graph, post = tiny_session_with_posterior
        est = build_event_sequence_table(dot, env, post)
        arm_cols = [f"arm_posterior_{a}" for a in range(9)]
        np.testing.assert_allclose(
            est[arm_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-10
        )

    def test_event_sequence_event_ids_match_event_table(
        self, tiny_session_with_posterior
    ):
        dot, env, trials, graph, post = tiny_session_with_posterior
        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)
        est = build_event_sequence_table(dot, env, post)
        assert set(et["event_id"]) == set(est["event_id"].unique())


# ── Trajectory rollup wiring ──

from continuum_swr_replay.analysis.replay.trajectory import (  # noqa: E402
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
)


def _traj_env():
    """Env stub with node_id + a bin-centre chain graph for graph distances."""
    env = _EnvStub()
    lp = np.concatenate([np.arange(18, dtype=float), [100.0, 101.0, 102.0]])
    eid = np.concatenate([np.repeat(np.arange(9), 2), [-1, -1, -1]]).astype(int)
    node_id = np.arange(len(lp))
    env.place_bin_centers_nodes_df_ = pd.DataFrame(
        {"node_id": node_id, "linear_position": lp, "edge_id": eid}
    )
    g = nx.Graph()
    for i in range(len(node_id) - 1):
        g.add_edge(
            int(node_id[i]), int(node_id[i + 1]), distance=abs(lp[i + 1] - lp[i])
        )
    env.track_graph_with_bin_centers_edges_ = g
    # Mirrors how ``Environment`` populates this at fit time
    # (``get_track_grid``): all-pairs Dijkstra over the bin-centre graph.
    env.distance_between_nodes_ = dict(
        nx.all_pairs_dijkstra_path_length(g, weight="distance")
    )
    return env


def _augmented_dot():
    """tiny dot + the augmented per-bin columns set to known constants."""
    dot = _tiny_dot_with_two_events()
    constants = {
        "map_speed": 5.0,
        "step_distance": 1.0,
        "ahead_behind_distance": 3.0,
        "hpd": 12.0,
        "Local_prob": 0.1,
        "No-Spike_prob": 0.0,
        "run_speed_baseline": 50.0,
        "ripple_power": 20.0,
        "SWR_zscore": 1.5,
        "mua_rate": 100.0,
        "MUA_zscore": 2.0,
        "theta_power": 0.3,
        "theta_power_zscore": 0.5,
    }
    for col, val in constants.items():
        dot[col] = val
    return dot


class TestTrajectoryRollupWiring:
    def test_all_trajectory_columns_present_and_computed(self):
        dot = _augmented_dot()
        env = _traj_env()
        trials = pd.DataFrame(
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
        graph = _bandit_graph()
        post = _tiny_posterior(dot)

        et = build_event_table(dot, env, trials, graph, conditional_posterior=post)

        for col in (*SHARED_TRAJECTORY_COLUMNS, *NONLOCAL_ONLY_COLUMNS):
            assert col in et.columns, f"missing trajectory column: {col}"

        assert len(et) == 2
        # constant per-bin z-scored signals -> per-event mean equals the constant
        np.testing.assert_allclose(et["ripple_power_z"], 1.5)  # from SWR_zscore
        np.testing.assert_allclose(et["mua_rate_z"], 2.0)  # from MUA_zscore
        np.testing.assert_allclose(et["theta_z"], 0.5)  # from theta_power_zscore
        np.testing.assert_allclose(et["hpd_coverage_cm"], 12.0)
        # compression_factor = median(map_speed) / run_speed_baseline = 5/50
        np.testing.assert_allclose(et["replay_speed_median"], 5.0)
        np.testing.assert_allclose(et["compression_factor"], 0.1)
        # each event sits on a single leaf arm -> one patch, terminates at well
        np.testing.assert_array_equal(et["n_patches_visited"], [1, 1])
        np.testing.assert_array_equal(et["crosses_trunk"], [False, False])
        np.testing.assert_array_equal(et["terminates_at_well"], [True, True])
        # path_length is the sum of the (constant 1.0) step_distance over bins
        assert (et["path_length"] > 0).all()

    def test_unaugmented_dot_emits_nan_trajectory_columns(self):
        """A pre-augmentation dot still yields the columns, as NaN."""
        dot = _tiny_dot_with_two_events()  # no step_distance etc.
        env = _traj_env()
        trials = pd.DataFrame(
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
        et = build_event_table(dot, env, trials, _bandit_graph())
        for col in (*SHARED_TRAJECTORY_COLUMNS, *NONLOCAL_ONLY_COLUMNS):
            assert col in et.columns
        assert et["path_length"].isna().all()
        assert et["ripple_power_z"].isna().all()
