"""Tests for the covariate module extraction.

Phase-0 of the covariate harmonization moved the behavioral covariate
definitions out of ``_helpers`` into ``covariates`` with a re-export shim. This
checks the move is transparent: every moved name resolves to the *same object*
whether imported from ``covariates``, from the ``_helpers`` shim, or (for the
public ones) off the package namespace.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import continuum_swr_replay.analysis.replay as replay_pkg
from continuum_swr_replay.analysis.replay import _helpers, covariates

# Names that moved to covariates.py and are re-exported from _helpers.
SHIMMED_NAMES = [
    "UNCERTAINTY_COVARIATES",
    "visit_recency_all_events",
    "cumulative_reward_all_events",
    "recent_reward_rate_all_events",
    "recent_trial_reward_rate",
    "lagged_trial_rewards",
    "_leaky_q_value_diff",
    "_omission_streak",
    "add_uncertainty_covariates",
    "visit_recency_per_arm",
    "cumulative_reward_per_arm",
    "recent_reward_rate_per_arm",
]

# Public covariate names also exposed on the package namespace.
PUBLIC_NAMES = ["UNCERTAINTY_COVARIATES", "add_uncertainty_covariates"]


def test_shim_reexports_same_objects():
    """Each moved name is the identical object via covariates and via _helpers."""
    for name in SHIMMED_NAMES:
        via_cov = getattr(covariates, name)
        via_helpers = getattr(_helpers, name)
        assert via_cov is via_helpers, f"{name} differs between covariates and _helpers"


def test_public_names_resolve_via_package():
    """The public covariate names resolve off the package and match covariates."""
    for name in PUBLIC_NAMES:
        assert getattr(replay_pkg, name) is getattr(covariates, name)
        assert name in replay_pkg.__all__


def test_q_learner_lives_in_covariates():
    """The promoted fitted-alpha leaky-Q learner is importable from covariates."""
    assert callable(covariates._q_chosen)
    assert callable(covariates._fit_alpha)


def test_covariates_imports_standalone_no_deadlock():
    """Importing ``covariates`` first in a *fresh* interpreter must not deadlock.

    ``_helpers`` and ``covariates`` form a re-export cycle (covariates imports
    ``_trial_cutoff_times`` / ``build_well_to_arm_map`` back from ``_helpers``).
    A subprocess is required to make ``covariates`` the entry import: within this
    test process both modules are already cached, so an inline import would only
    re-assert identity, not the load order. The shim resolves because importing
    the submodule first runs the package ``__init__``, which loads ``_helpers``
    before ``.covariates``.
    """
    code = (
        "from continuum_swr_replay.analysis.replay.covariates import "
        "recent_reward_rate_all_events as a;"
        "from continuum_swr_replay.analysis.replay._helpers import "
        "recent_reward_rate_all_events as b;"
        "assert a is b"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ── q_all_arm (per-arm leaky-Q rollout) ──


def test_q_all_arm_chosen_column_matches_q_chosen():
    """q_all_arm[i, arm[i]] reproduces _q_chosen exactly (same delta-rule)."""
    rng = np.random.default_rng(0)
    n = 200
    arm = rng.integers(0, 6, n)
    reward = rng.integers(0, 2, n).astype(float)
    grp = np.repeat([0, 1, 2, 3], n // 4)  # 4 sessions, reset at each
    alpha = 0.2
    full = covariates.q_all_arm(arm, reward, grp, alpha)
    chosen = covariates._q_chosen(arm, reward, grp, alpha)
    assert full.shape == (n, 6)
    np.testing.assert_allclose(full[np.arange(n), arm], chosen)


def test_q_all_arm_pre_update_and_group_reset():
    """Pre-update snapshot and per-group reset to init are exact on a hand case."""
    # one group, arm 0 rewarded then arm 0 omitted; init 0.5, alpha 0.5.
    arm = np.array([0, 0, 1])
    reward = np.array([1.0, 0.0, 1.0])
    grp = np.array([0, 0, 0])
    q = covariates.q_all_arm(arm, reward, grp, 0.5, init=0.5)
    # trial 0: all arms at init before update
    np.testing.assert_allclose(q[0], np.full(6, 0.5))
    # trial 1: arm 0 updated to 0.5 + 0.5*(1-0.5)=0.75; others still 0.5
    assert q[1, 0] == 0.75
    assert q[1, 1] == 0.5
    # new group resets every arm to init
    q2 = covariates.q_all_arm(
        np.array([0, 0]), np.array([1.0, 1.0]), np.array([0, 1]), 0.5
    )
    np.testing.assert_allclose(q2[1], np.full(6, 0.5))  # second row is a fresh group


# ── attach_contingency_value_block (shared per-arm value join) ──


def _toy_contingency(stop_id):
    """One contingency row: ports 0..5 = 0.0,0.2,0.4,0.6,0.8,1.0; q = p/2."""
    row = {"stop_id": stop_id}
    for i in range(6):
        row[f"p_reward_port{i}"] = i * 0.2
        row[f"q_port{i}"] = i * 0.1
    return pd.DataFrame([row])


def test_value_block_leaf_trunk_and_present():
    """Leaf arm j <- port{j}; trunk arms (6-8) -> 0.0 for a mapped stop."""
    arm = np.array([0, 1, 5, 6, 7, 8])
    stop_id = np.full(6, 111, dtype=np.int64)
    cont = _toy_contingency(111)
    p, q = covariates.attach_contingency_value_block(arm, stop_id, cont)
    np.testing.assert_allclose(p[:3], [0.0, 0.2, 1.0])  # leaves 0,1,5
    np.testing.assert_allclose(q[:3], [0.0, 0.1, 0.5])
    np.testing.assert_allclose(p[3:], [0.0, 0.0, 0.0])  # trunk arms, stop present
    np.testing.assert_allclose(q[3:], [0.0, 0.0, 0.0])


def test_value_block_absent_stop_all_nan():
    """A stop_id with no contingency row -> NaN on every arm (leaf and trunk)."""
    arm = np.array([0, 5, 6, 8])
    stop_id = np.full(4, 999, dtype=np.int64)  # not in the contingency
    cont = _toy_contingency(111)
    p, q = covariates.attach_contingency_value_block(arm, stop_id, cont)
    assert np.isnan(p).all()
    assert np.isnan(q).all()


def test_value_block_none_all_nan():
    """No artifact supplied (None) -> legitimately all-NaN, no error."""
    arm = np.array([0, 6])
    sid = np.array([1, 2])
    p, q = covariates.attach_contingency_value_block(arm, sid, None)
    assert np.isnan(p).all() and np.isnan(q).all()


def test_value_block_missing_cols_raises():
    """A supplied-but-stale artifact (no per-port cols) must fail loudly, not
    silently blank the whole value block: qvals_table is the sole source."""
    arm = np.array([0, 6])
    sid = np.array([1, 2])
    bad = pd.DataFrame({"stop_id": [1, 2]})  # no port/q cols
    with pytest.raises(ValueError, match="missing per-arm value columns"):
        covariates.attach_contingency_value_block(arm, sid, bad)


# ── load_contingency (shared value-artifact load seam) ──
# Concept: qvals_table is the SOLE value source, so its absence must degrade
# visibly (warning + None), not silently; presence returns the frame.


def test_load_contingency_present_returns_frame(tmp_path, caplog):
    df = pd.DataFrame({"stop_id": [1, 2, 3]})
    path = tmp_path / "qvals_table.pkl"
    df.to_pickle(path)
    with caplog.at_level(logging.INFO):
        out = covariates.load_contingency(path)
    pd.testing.assert_frame_equal(out, df)
    assert any("3 stop rows" in r.message for r in caplog.records)


def test_load_contingency_absent_returns_none_and_warns(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.pkl"
    with caplog.at_level(logging.WARNING):
        out = covariates.load_contingency(missing)
    assert out is None
    assert any(
        r.levelno == logging.WARNING and "no qvals_table" in r.message
        for r in caplog.records
    )


def test_load_contingency_absent_yields_all_nan_value_block(tmp_path):
    """End-to-end fallback contract: a missing artifact -> None -> the value
    block is all-NaN (drops uniformly at fit time), never a spurious 0.0."""
    cont = covariates.load_contingency(tmp_path / "nope.pkl")
    p, q = covariates.attach_contingency_value_block(
        np.array([0, 6]), np.array([1, 2]), cont
    )
    assert np.isnan(p).all() and np.isnan(q).all()


def test_value_block_warns_on_duplicate_stop_id():
    """Duplicate stop_id with differing values warns (silent-pick guard)."""
    a = _toy_contingency(111)
    b = _toy_contingency(111)
    b.loc[0, "p_reward_port0"] = 0.9  # conflicting value for the same stop
    cont = pd.concat([a, b], ignore_index=True)
    with pytest.warns(UserWarning, match="duplicate stop_id"):
        covariates.attach_contingency_value_block(np.array([0]), np.array([111]), cont)


def test_value_block_plain_key_populates_suffixed_key_misses():
    """Regression for the contfrag join-key bug: the contingency is keyed by the
    plain hash_stop_id(animal,day,epoch,stop_idx); joining on ContFrag's own
    brain_area/event_source-suffixed stop_id matches nothing (all-NaN). This locks
    the builder's choice to recompute the plain key."""
    animal, day, epoch = "x", 1, "02_r1"
    stop_idx = np.array([0, 0])
    arm = np.array([0, 6])  # one leaf, one trunk
    plain = _helpers.hash_stop_id(
        np.array([animal, animal]),
        np.array([day, day]),
        np.array([epoch, epoch]),
        stop_idx,
    )
    suffixed = _helpers.hash_stop_id(
        np.array([animal, animal]),
        np.array([day, day]),
        np.array([epoch, epoch]),
        stop_idx,
        brain_area="HPC",
        event_source="SWR",
    )
    assert plain[0] != suffixed[0]  # the two keyings genuinely differ
    cont = _toy_contingency(int(plain[0]))
    # plain key -> leaf populated, trunk 0.0
    p_plain, _ = covariates.attach_contingency_value_block(arm, plain, cont)
    assert not np.isnan(p_plain[0]) and p_plain[1] == 0.0
    # suffixed key (the bug) -> no match -> all NaN
    p_suf, _ = covariates.attach_contingency_value_block(arm, suffixed, cont)
    assert np.isnan(p_suf).all()


def _single_arm_trials(n_trials: int, rewarded_indices: set[int]) -> pd.DataFrame:
    """``n_trials`` consecutive trips that all end at well 0 (leaf arm 0)."""
    return pd.DataFrame(
        {
            "start_time": np.arange(n_trials, dtype=float) * 10.0,
            "end_time": np.arange(n_trials, dtype=float) * 10.0 + 9.0,
            "goal_arrival_time": np.arange(n_trials, dtype=float) * 10.0 + 8.0,
            "from_well": 1,
            "to_well": 0,
            "is_reward": [i in rewarded_indices for i in range(n_trials)],
            "is_patch_change": False,
        }
    )


def test_visit_recency_and_never_visited_are_time_local_and_arm_typed():
    """Recency uses only completed history; trunks are structural, not novel."""
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 10.0, 20.0],
            "end_time": [9.0, 19.0, 29.0],
            "goal_arrival_time": [8.0, 18.0, 28.0],
            "from_well": [1, 0, 1],
            "to_well": [0, 1, 0],
            "is_reward": [True, False, True],
            "is_patch_change": [False, False, False],
        }
    )
    event_times = np.array([7.0, 9.0, 19.0, 30.0])
    well_to_arm = {well: well for well in range(6)}

    recency = covariates.visit_recency_all_events(event_times, trials, well_to_arm)
    never_visited = covariates.arm_never_visited_all_events(
        event_times, trials, well_to_arm
    )
    combined_recency, combined_never_visited = (
        covariates.visit_recency_and_never_visited_all_events(
            event_times, trials, well_to_arm
        )
    )

    expected_recency = np.array(
        [
            [np.nan] * 9,
            [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_never_visited = np.array(
        [
            [np.nan] * 9,
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(recency, expected_recency)
    np.testing.assert_array_equal(never_visited, expected_never_visited)
    np.testing.assert_array_equal(combined_recency, expected_recency)
    np.testing.assert_array_equal(combined_never_visited, expected_never_visited)
    assert recency.shape == never_visited.shape == (len(event_times), 9)


def test_recency_and_never_visited_ignore_appended_future_trials():
    """Appending trials after an event cannot retroactively change covariates."""
    trials = _single_arm_trials(5, rewarded_indices=set())
    event_times = np.array([15.0])  # only trial 0 (arrival 8 s) has completed
    well_to_arm = {well: well for well in range(6)}

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        short = function(event_times, trials.iloc[:2], well_to_arm)
        long = function(event_times, trials, well_to_arm)
        np.testing.assert_array_equal(short, long)


def test_visit_recency_centering_is_within_event_and_future_invariant():
    recency = np.array(
        [
            [1, 2, 3, 4, 5, 6, 0, 0, 0],
            [6, 5, 4, 3, 2, 1, 0, 0, 0],
            [np.nan] * 9,
        ],
        dtype=float,
    )
    visited = np.zeros_like(recency)
    centered = covariates.center_visit_recency_within_event(recency, visited)
    np.testing.assert_allclose(centered[:2, :6].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_array_equal(centered[:2, 6:], 0.0)
    assert np.isnan(centered[2]).all()

    appended = np.vstack([recency, [[100, 1, 1, 1, 1, 1, 0, 0, 0]]])
    appended_visited = np.vstack([visited, [np.zeros(9)]])
    np.testing.assert_array_equal(
        covariates.center_visit_recency_within_event(appended, appended_visited)[:3],
        centered,
    )


def test_centering_excludes_the_right_censoring_fill_from_the_mean():
    """A never-visited arm must not displace the arms that were visited.

    Arm 3 is censored at 201 in a 200-trial session while the visited
    arms sit at 1-5. Including the fill would drag the mean to ~36 and
    report every visited arm as ~-33; only the censored arm's own value
    may depend on the censoring rule.
    """
    recency = np.array([[1, 2, 3, 201, 4, 5, 0, 0, 0]], dtype=float)
    never_visited = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=float)

    centered = covariates.center_visit_recency_within_event(recency, never_visited)

    visited_mean = np.mean([1, 2, 3, 4, 5])
    np.testing.assert_allclose(
        centered[0, [0, 1, 2, 4, 5]],
        np.array([1, 2, 3, 4, 5]) - visited_mean,
    )
    # The censored arm keeps its offset from the visited mean, and stays
    # far above every visited arm rather than dragging them down.
    assert centered[0, 3] == pytest.approx(201 - visited_mean)
    assert (centered[0, [0, 1, 2, 4, 5]] > -10).all()


def test_centering_leaves_an_all_censored_event_uncentered():
    """With no visited leaf arm there is no mean to subtract."""
    recency = np.array([[7, 7, 7, 7, 7, 7, 0, 0, 0]], dtype=float)
    never_visited = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=float)
    centered = covariates.center_visit_recency_within_event(recency, never_visited)
    assert np.isnan(centered[0, :6]).all()


def test_centering_rejects_mismatched_never_visited_shape():
    recency = np.zeros((2, 9))
    with pytest.raises(ValueError, match="never_visited must match"):
        covariates.center_visit_recency_within_event(recency, np.zeros((2, 6)))


def test_recency_rejects_incomplete_leaf_topology():
    """A missing leaf mapping is invalid, not a structural trunk zero."""
    trials = _single_arm_trials(1, rewarded_indices=set())
    incomplete = {well: well for well in range(5)}

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        with pytest.raises(ValueError, match="cover every canonical"):
            function(np.array([20.0]), trials, incomplete)


def test_recency_rejects_wrong_well_keys_despite_complete_arm_values():
    """A fake well cannot substitute for a missing canonical reward site."""
    trials = _single_arm_trials(1, rewarded_indices=set())
    wrong_well = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 99: 5}

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        with pytest.raises(ValueError, match="reward-site well"):
            function(np.array([20.0]), trials, wrong_well)


def test_cumulative_reward_counts_only_the_last_ten_trials():
    """``cumulative_reward_all_events`` is a rolling k=10 window, not a total.

    The name says "cumulative" but the helper subtracts the cumsum 10 trials
    back, so a reward older than that must not be counted. Nothing pinned the
    window: changing it from 10 to 5 left the whole suite green, because the
    builder fixture's ``trials`` frame is empty and every trial-derived column
    comes out NaN.

    Twelve completed trips, all to well 0, rewarded on trips 0 and 2. At an
    event after all twelve, the window covers trips 2-11, so trip 0's reward
    is outside it and the count is 1 -- not 2 (a true cumulative total) and
    not 0 (a 5-trial window).
    """
    trials = _single_arm_trials(12, rewarded_indices={0, 2})
    event_times = np.array([200.0])  # after every trip

    out = covariates.cumulative_reward_all_events(
        event_times, trials, well_to_arm={0: 0, 1: 1}
    )

    assert out[0, 0] == 1.0, (
        f"arm 0 should count only the reward inside the 10-trial window, got "
        f"{out[0, 0]}"
    )


def test_cumulative_reward_window_is_ten_not_a_running_total():
    """A reward far outside the window contributes nothing, however old."""
    trials = _single_arm_trials(30, rewarded_indices={0, 1, 2})
    out = covariates.cumulative_reward_all_events(
        np.array([400.0]), trials, well_to_arm={0: 0, 1: 1}
    )
    assert out[0, 0] == 0.0, (
        f"all three rewards are >10 trials back, so the window is empty; got "
        f"{out[0, 0]}"
    )


class TestAlignCountMatrix:
    """Direct coverage for the reindex + fill behind the arm-count responses.

    The fill fires only for an event whose every in-event timebin failed to
    attribute to a real arm -- 0 of 65,913 events on the shipped corpus -- so
    it was reachable only through two builder-level tests and the fill value
    itself was never asserted. It is dtype-dependent, which is easy to break
    silently.
    """

    def test_reindexes_rows_to_target_order(self):
        counts = np.array([[1, 2], [3, 4]], dtype=np.int64)
        out = _helpers.align_count_matrix(
            counts, np.array([10, 20]), np.array([20, 10]), n_arms=2
        )
        np.testing.assert_array_equal(out, np.array([[3, 4], [1, 2]]))

    def test_missing_event_fills_zero_for_integer_counts(self):
        """A hard count genuinely observed zero real-arm bins."""
        counts = np.array([[1, 2]], dtype=np.int64)
        out = _helpers.align_count_matrix(
            counts, np.array([10]), np.array([10, 99]), n_arms=2
        )
        np.testing.assert_array_equal(out[1], np.array([0, 0]))
        assert out.dtype == np.int64

    def test_missing_event_fills_nan_for_float_counts(self):
        """A fraction's 0/0 must stay undefined, not become 0 or +inf."""
        counts = np.array([[1.0, 2.0]], dtype=float)
        out = _helpers.align_count_matrix(
            counts, np.array([10]), np.array([10, 99]), n_arms=2
        )
        assert np.isnan(out[1]).all()


# ── Guards for invariants that survived mutation testing ──


def test_omission_streak_is_undefined_without_a_resolved_outcome():
    """A streak of leading omissions needs a known outcome to start from.

    Returning 0 would put every pre-first-outcome stop in the same bucket as a
    stop whose most recent trial was rewarded -- a fabricated reference
    category, systematically the earliest stops of each session.
    """
    no_history = np.full((1, 10), np.nan)
    assert np.isnan(covariates._omission_streak(no_history)[0])
    # A genuine "last trial rewarded" is the value it must not be confused with.
    rewarded = np.array([[1.0] + [0.0] * 9])
    assert covariates._omission_streak(rewarded)[0] == 0.0
    # An unresolved tail still truncates the count rather than voiding it.
    partial = np.array([[0.0, np.nan] + [np.nan] * 8])
    assert covariates._omission_streak(partial)[0] == 1.0


def test_reward_rate_arm_window_is_ten_visits_not_five():
    """The column is named k10; pin the window so a 5->10 migration can't half-land."""
    trials = pd.DataFrame(
        {
            "start_time": np.arange(12, dtype=float),
            "goal_arrival_time": np.arange(12, dtype=float) + 0.5,
            "end_time": np.arange(12, dtype=float) + 0.5,
            "from_well": [0] * 12,
            "to_well": [1] * 12,
            # First two visits rewarded, next ten omitted.
            "is_reward": [True, True] + [False] * 10,
            "is_patch_change": [False] * 12,
        }
    )
    w2a = {i: i for i in range(6)}
    rate = covariates.recent_reward_rate_all_events(
        np.array([11.9]), trials, w2a, k=10
    )
    # Last 10 visits = trials 2..11, all omitted -> 0.0. A k=5 window also
    # gives 0.0, so include the earlier query that separates them.
    early = covariates.recent_reward_rate_all_events(
        np.array([6.9]), trials, w2a, k=10
    )
    # By t=6.9 seven visits have completed: 2 rewarded of 7 -> 2/7 at k=10,
    # but 0/5 at k=5.
    assert early[0, 1] == pytest.approx(2 / 7)
    assert rate[0, 1] == pytest.approx(0.0)


def test_reward_rate_arm_counts_only_resolved_visits():
    """A visit whose outcome has not resolved must not enter the rate.

    Two stages compose here: how many trials have resolved by the query time
    (equality eligible), then how many of THIS arm's visits fall among them.
    The second search must exclude the visit whose index equals the resolved
    count -- that trial is the first unresolved one. `visit_recency` has a
    guard for its equivalent boundary; this twin did not, and the mutation
    leaked one trial's outcome into 6.8% of event-arm rows.
    """
    trials = pd.DataFrame(
        {
            "start_time": [0.0, 10.0],
            "goal_arrival_time": [5.0, 15.0],
            "end_time": [6.0, 16.0],
            "from_well": [0, 1],
            "to_well": [1, 1],
            # Visit 0 omitted, visit 1 rewarded. Leaking visit 1 turns 0.0 into 0.5.
            "is_reward": [False, True],
            "is_patch_change": [False, False],
        }
    )
    w2a = {i: i for i in range(6)}
    # Query after visit 0's outcome (6.0) but before visit 1's (16.0).
    between = covariates.recent_reward_rate_all_events(
        np.array([12.0]), trials, w2a, k=10
    )
    assert between[0, 1] == pytest.approx(0.0), "unresolved visit's reward leaked"
    # Once visit 1 resolves, it counts: 1 of 2 rewarded.
    after = covariates.recent_reward_rate_all_events(
        np.array([16.0]), trials, w2a, k=10
    )
    assert after[0, 1] == pytest.approx(0.5)
