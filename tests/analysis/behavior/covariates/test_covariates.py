"""Behavioral-covariate package ownership and public-import contracts."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import (
    covariates,
    history_basis,
    spatial_mapping,
)
from continuum_swr_replay.analysis.behavior.covariates.history import (
    diagnostics,
    value_uncertainty,
)
from continuum_swr_replay.analysis.event_tables import _common as table_common
from continuum_swr_replay.analysis.replay_models import per_bin_design
from continuum_swr_replay.tables import event_ids
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.io import write_table
from tests.support._imports import SRC, imported_modules
from tests.support._port_experience_fixtures import port_experiences_from_trials

# Scientific definitions must have one owning module.  The generic
# table-building modules (``event_tables._common``, ``behavior.spatial_mapping``)
# must not re-export covariate functions.
HISTORY_NAMES = [
    "UNCERTAINTY_COVARIATES",
    "visit_recency_all_events",
    "rolling_reward_count_all_events",
    "recent_reward_rate_all_events",
    "recent_trial_reward_rate",
    "lagged_trial_rewards",
    "add_uncertainty_covariates",
]

CONTEXT_NAMES = [
    "add_trial_context",
    "behavioral_location_from_port_experiences",
    "summarize_behavioral_location",
    "trial_context_at_times",
]

CLOCK_NAMES = ["trial_arrival_times", "trial_outcome_times"]


def test_covariate_definitions_have_one_owner():
    """Package exports are identical to their coherent owning submodules."""
    from continuum_swr_replay.analysis.behavior.covariates import (
        clocks,
        context,
        history,
    )

    for name in HISTORY_NAMES:
        assert getattr(covariates, name) is getattr(history, name)
    for name in CONTEXT_NAMES:
        assert getattr(covariates, name) is getattr(context, name)
    for name in CLOCK_NAMES:
        assert getattr(covariates, name) is getattr(clocks, name)


@pytest.mark.parametrize("module", [table_common, spatial_mapping])
def test_generic_helper_modules_have_no_covariate_reexport_shim(module):
    """The generic helper modules must not regain scientific covariate ownership."""
    for name in (*HISTORY_NAMES, *CONTEXT_NAMES, *CLOCK_NAMES):
        assert not hasattr(module, name), f"{module.__name__} re-exports {name}"


def test_q_learner_lives_in_its_diagnostics_module():
    """The fitted-alpha leaky-Q learner is owned by ``history.diagnostics``."""
    assert callable(diagnostics._q_chosen)
    assert callable(diagnostics._fit_alpha)


def test_unobserved_initial_occupancy_is_rejected_as_noncanonical():
    trials = pd.DataFrame(
        {
            "movement_start_time": [1.0],
            "destination_first_poke_in_time": [2.0],
            "end_time": [3.0],
            "from_well": [0],
            "to_well": [1],
            "is_reward": [True],
            "reward_on_time": [2.2],
            "outcome_known_time": [2.2],
            "is_patch_change": [False],
        }
    )
    experiences = port_experiences_from_trials(
        trials, prepend_initial=True, initial_reward=None
    )

    with pytest.raises(ValueError, match="physical poke"):
        covariates.canonical_port_value_history(experiences)


def test_covariates_imports_standalone_without_cycle():
    """Every covariate submodule imports in a fresh interpreter in either order."""
    code = (
        "from continuum_swr_replay.analysis.behavior.covariates.context import "
        "add_trial_context as context_fn;"
        "from continuum_swr_replay.analysis.behavior.covariates.history import "
        "recent_reward_rate_all_events as history_fn;"
        "from continuum_swr_replay.analysis.behavior.covariates.clocks import "
        "trial_outcome_times as clock_fn;"
        "from continuum_swr_replay.analysis.behavior import covariates;"
        "assert covariates.add_trial_context is context_fn;"
        "assert covariates.recent_reward_rate_all_events is history_fn;"
        "assert covariates.trial_outcome_times is clock_fn"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", [history_basis, per_bin_design])
def test_shared_primitive_module_does_not_import_covariates_package(module):
    """Static guard against recreating the former two-way import dependency.

    ``covariates.history.reward_history`` imports the raised-cosine basis and
    its projection from ``history_basis``, so ``history_basis`` importing the
    covariates back would be a cycle; ``per_bin_design``, which owned those
    primitives before, carries the same constraint.
    """
    imported = imported_modules(Path(module.__file__), SRC)
    assert not any(".behavior.covariates" in m for m in imported), sorted(imported)


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
    chosen = diagnostics._q_chosen(arm, reward, grp, alpha)
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
        row[f"task_reward_probability_port_{i}"] = i * 0.2
        row[f"reward_belief_mean_port_{i}"] = i * 0.1
        row[f"reward_belief_variance_port_{i}"] = 0.01 * (i + 1)
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
    with pytest.raises(
        ValueError, match=r"qvals_table\) lacks .*task_reward_probability"
    ):
        covariates.attach_contingency_value_block(arm, sid, bad)


@pytest.mark.parametrize("bad_arm", [-1, 1.5, 9, np.inf])
def test_value_block_rejects_noncanonical_arm_ids(bad_arm):
    with pytest.raises(ValueError, match="arm IDs"):
        covariates.attach_contingency_value_block(
            np.array([bad_arm]), np.array([111]), _toy_contingency(111)
        )


# ── load_contingency (shared value-artifact load seam) ──
# Concept: qvals_table is the SOLE value source, so its absence must degrade
# visibly (warning + None), not silently; presence returns the frame.


def test_load_contingency_present_returns_frame(tmp_path, caplog):
    df = stamp_covariate_schema(pd.DataFrame({"stop_id": [1, 2, 3]}), table="qvals")
    path = write_table(df, tmp_path / "qvals_table.parquet")
    with caplog.at_level(logging.INFO):
        out = covariates.load_contingency(path)
    pd.testing.assert_frame_equal(out, df)
    assert any("3 stop rows" in r.message for r in caplog.records)


def test_load_contingency_absent_returns_none_and_warns(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.parquet"
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
    cont = covariates.load_contingency(tmp_path / "nope.parquet")
    p, q = covariates.attach_contingency_value_block(
        np.array([0, 6]), np.array([1, 2]), cont
    )
    assert np.isnan(p).all() and np.isnan(q).all()


def test_value_block_rejects_duplicate_stop_id():
    """A duplicate scientific key is an alignment failure, not a warning."""
    a = _toy_contingency(111)
    b = _toy_contingency(111)
    b.loc[0, "task_reward_probability_port_0"] = (
        0.9  # conflicting value for the same stop
    )
    cont = pd.concat([a, b], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate stop_id"):
        covariates.attach_contingency_value_block(np.array([0]), np.array([111]), cont)


def test_value_block_rejects_partially_missing_matched_row():
    cont = _toy_contingency(111)
    cont.loc[0, "reward_belief_mean_port_4"] = np.nan

    with pytest.raises(ValueError, match="finite"):
        covariates.attach_contingency_value_block(
            np.array([0, 4]), np.array([111, 111]), cont
        )


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("task_reward_probability_port_2", 1.1),
        ("reward_belief_mean_port_3", -0.1),
    ],
)
def test_value_block_rejects_out_of_range_probability_or_q(column, bad):
    cont = _toy_contingency(111)
    cont.loc[0, column] = bad

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        covariates.attach_contingency_value_block(np.array([0]), np.array([111]), cont)


def test_value_block_reports_finiteness_before_range_on_mixed_violation():
    """A row both out-of-range and non-finite reports the finiteness problem
    first (a NaN is a worse corruption than an out-of-range value). Pins the
    check ordering so a future DRY pass cannot silently flip it.
    """
    cont = _toy_contingency(111)
    cont.loc[0, "task_reward_probability_port_2"] = 1.5  # out of [0, 1]
    cont.loc[0, "reward_belief_mean_port_3"] = np.nan  # non-finite
    with pytest.raises(ValueError, match="finite reward-belief means"):
        covariates.attach_contingency_value_block(np.array([0]), np.array([111]), cont)


def test_uncertainty_block_rejects_duplicate_stop_id():
    cont = _toy_contingency(111)
    cont = pd.concat([cont, cont], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate stop_id"):
        covariates.attach_arm_uncertainty(np.array([0]), np.array([111]), cont)


def test_uncertainty_block_rejects_invalid_beta_variance():
    cont = _toy_contingency(111)
    cont.loc[0, "reward_belief_variance_port_3"] = 0.3

    with pytest.raises(ValueError, match=r"\[0, 0.25\]"):
        covariates.attach_arm_uncertainty(np.array([0]), np.array([111]), cont)


def test_value_block_plain_key_populates_suffixed_key_misses():
    """Regression for the contfrag join-key bug: the contingency is keyed by the
    plain hash_stop_id(animal,day,epoch,stop_idx); joining on ContFrag's own
    brain_area/event_source-suffixed stop_id matches nothing (all-NaN). This locks
    the builder's choice to recompute the plain key."""
    animal, day, epoch = "x", 1, "02_r1"
    stop_idx = np.array([0, 0])
    arm = np.array([0, 6])  # one leaf, one trunk
    plain = event_ids.hash_stop_id(
        np.array([animal, animal]),
        np.array([day, day]),
        np.array([epoch, epoch]),
        stop_idx,
    )
    suffixed = event_ids.hash_stop_id(
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
            "movement_start_time": np.arange(n_trials, dtype=float) * 10.0,
            "end_time": np.arange(n_trials, dtype=float) * 10.0 + 9.0,
            "destination_first_poke_in_time": np.arange(n_trials, dtype=float) * 10.0
            + 8.0,
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
            "movement_start_time": [0.0, 10.0, 20.0],
            "end_time": [9.0, 19.0, 29.0],
            "destination_first_poke_in_time": [8.0, 18.0, 28.0],
            "from_well": [1, 0, 1],
            "to_well": [0, 1, 0],
            "is_reward": [True, False, True],
            "is_patch_change": [False, False, False],
        }
    )
    event_times = np.array([7.0, 9.0, 19.0, 30.0])
    well_to_arm = {well: well for well in range(6)}

    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    recency = covariates.visit_recency_all_events(event_times, experiences, well_to_arm)
    never_visited = covariates.arm_never_visited_all_events(
        event_times, experiences, well_to_arm
    )
    combined_recency, combined_never_visited = (
        covariates.visit_recency_and_never_visited_all_events(
            event_times, experiences, well_to_arm
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


def test_initial_port_experience_seeds_visit_and_reward_history():
    """History starts with the origin experience, not the first destination."""
    experiences = pd.DataFrame(
        {
            "experience_id": [0, 1],
            "movement_trial_number": pd.array([pd.NA, 1], dtype="Int64"),
            "is_initial_experience": [True, False],
            "is_boundary_derived": [False, False],
            "outcome_observed": [True, True],
            "port": [3, 2],
            "patch": [2, 2],
            "arrival_time": [0.0, 10.0],
            "departure_time": [2.0, 12.0],
            "reward_on_time": [1.0, np.nan],
            "outcome_known_time": [1.0, 10.3],
            "is_reward": [True, False],
        }
    )
    well_to_arm = {well: well for well in range(6)}

    recency, novel = covariates.visit_recency_and_never_visited_all_events(
        np.array([1.5, 10.1]), experiences, well_to_arm
    )
    reward_fraction = covariates.recent_reward_rate_all_events(
        np.array([1.5, 10.1]), experiences, well_to_arm, k=10
    )

    assert recency[0, 3] == 1.0
    assert novel[0, 3] == 0.0
    assert reward_fraction[0, 3] == 1.0
    assert recency[1, 2] == 1.0
    assert recency[1, 3] == 2.0
    assert novel[1, 3] == 0.0
    # The destination omission is unresolved at 10.1; only the initial reward
    # contributes to reward history.
    assert reward_fraction[1, 3] == 1.0


def _seed_experiences() -> pd.DataFrame:
    """Two experiences: an initial rewarded port 3, then an omission at port 2.

    The initial outcome is known at t=1.0; the destination outcome only at
    t=10.3. A query at t=1.5 therefore has exactly one known outcome — the
    initial one — which the destination-only (pre-migration) history would have
    missed entirely.
    """
    return pd.DataFrame(
        {
            "experience_id": [0, 1],
            "movement_trial_number": pd.array([pd.NA, 1], dtype="Int64"),
            "is_initial_experience": [True, False],
            "is_boundary_derived": [False, False],
            "outcome_observed": [True, True],
            "port": [3, 2],
            "patch": [2, 2],
            "arrival_time": [0.0, 10.0],
            "departure_time": [2.0, 12.0],
            "reward_on_time": [1.0, np.nan],
            "outcome_known_time": [1.0, 10.3],
            "is_reward": [True, False],
        }
    )


def test_initial_experience_counts_in_rolling_reward_count():
    """``reward_count_candidate_arm_last10_known_trials`` counts the initial
    reward: without it, a query before the first movement would see zero."""
    experiences = _seed_experiences()
    well_to_arm = {well: well for well in range(6)}
    counts = covariates.rolling_reward_count_all_events(
        np.array([1.5, 10.5]), experiences, well_to_arm, window=10
    )
    # At t=1.5 only the initial reward (arm 3) is known.
    assert counts[0, 3] == 1.0
    assert counts[0, 2] == 0.0
    # At t=10.5 the destination omission (arm 2) is now known but adds no reward.
    assert counts[1, 3] == 1.0
    assert counts[1, 2] == 0.0


def test_initial_experience_counts_in_stop_reward_fraction():
    """``reward_fraction_last10_known_trials`` (arm-agnostic) includes the
    initial experience as the earliest known outcome."""
    experiences = _seed_experiences()
    frac = covariates.recent_trial_reward_rate(np.array([1.5, 10.5]), experiences, k=10)
    # Only the initial reward is known at 1.5 -> 1/1.
    assert frac[0] == 1.0
    # Initial reward + destination omission known at 10.5 -> 1/2.
    assert frac[1] == 0.5


def test_initial_experience_is_the_first_reward_lag():
    """``known_reward_outcome_lag_1`` is the initial outcome until a later
    experience resolves, then it advances to the newer known outcome."""
    experiences = _seed_experiences()
    lags = covariates.lagged_trial_rewards(np.array([1.5, 10.5]), experiences, k=10)
    # At 1.5 the only known outcome is the initial reward.
    assert lags[0, 0] == 1.0
    assert np.isnan(lags[0, 1])
    # At 10.5 the most recent known outcome is the destination omission (0.0),
    # and the initial reward is one lag back.
    assert lags[1, 0] == 0.0
    assert lags[1, 1] == 1.0


def test_port_experience_reward_clock_disagreement_fails_loudly():
    trials = _single_arm_trials(2, rewarded_indices={0})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    rewarded = experiences["is_reward"].to_numpy(dtype=bool)
    experiences.loc[rewarded, "reward_on_time"] += 0.01

    with pytest.raises(ValueError, match="reward_on_time equal"):
        covariates.recent_trial_reward_rate(np.array([100.0]), experiences)


def test_recency_and_never_visited_ignore_appended_future_trials():
    """Appending trials after an event cannot retroactively change covariates."""
    trials = _single_arm_trials(5, rewarded_indices=set())
    event_times = np.array([15.0])  # only trial 0 (arrival 8 s) has completed
    well_to_arm = {well: well for well in range(6)}
    experiences = port_experiences_from_trials(trials, prepend_initial=False)

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        short = function(
            event_times,
            port_experiences_from_trials(trials.iloc[:2], prepend_initial=False),
            well_to_arm,
        )
        long = function(event_times, experiences, well_to_arm)
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
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    incomplete = {well: well for well in range(5)}

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        with pytest.raises(ValueError, match="cover every canonical"):
            function(np.array([20.0]), experiences, incomplete)


def test_recency_rejects_wrong_well_keys_despite_complete_arm_values():
    """A fake well cannot substitute for a missing canonical reward site."""
    trials = _single_arm_trials(1, rewarded_indices=set())
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    wrong_well = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 99: 5}

    for function in (
        covariates.visit_recency_all_events,
        covariates.arm_never_visited_all_events,
    ):
        with pytest.raises(ValueError, match="reward-site well"):
            function(np.array([20.0]), experiences, wrong_well)


def test_rolling_reward_count_counts_only_the_last_ten_trials():
    """``rolling_reward_count_all_events`` is a rolling k=10 window, not a total.

    The helper subtracts the cumsum 10 trials back, so a reward older than that
    must not be counted. Nothing pinned the window: changing it from 10 to 5
    left the whole suite green, because the
    builder fixture's ``trials`` frame is empty and every trial-derived column
    comes out NaN.

    Twelve completed trips, all to well 0, rewarded on trips 0 and 2. At an
    event after all twelve, the window covers trips 2-11, so trip 0's reward
    is outside it and the count is 1 -- not 2 (a true cumulative total) and
    not 0 (a 5-trial window).
    """
    trials = _single_arm_trials(12, rewarded_indices={0, 2})
    event_times = np.array([200.0])  # after every trip

    out = covariates.rolling_reward_count_all_events(
        event_times,
        port_experiences_from_trials(trials, prepend_initial=False),
        well_to_arm={0: 0, 1: 1},
    )

    assert out[0, 0] == 1.0, (
        f"arm 0 should count only the reward inside the 10-trial window, got "
        f"{out[0, 0]}"
    )


def test_rolling_reward_count_window_is_ten_not_a_running_total():
    """A reward far outside the window contributes nothing, however old."""
    trials = _single_arm_trials(30, rewarded_indices={0, 1, 2})
    out = covariates.rolling_reward_count_all_events(
        np.array([400.0]),
        port_experiences_from_trials(trials, prepend_initial=False),
        well_to_arm={0: 0, 1: 1},
    )
    assert out[0, 0] == 0.0, (
        f"all three rewards are >10 trials back, so the window is empty; got "
        f"{out[0, 0]}"
    )


def test_rolling_reward_count_fills_unchanged_column():
    """The renamed helper produces unchanged values under its frozen column.

    Two guarantees the honesty rename must preserve:

    1. **Output values unchanged.** On a deterministic fixture the helper
       returns the expected rolling reward counts -- the rename touched the
       name, not the arithmetic.
    2. **Binding unchanged.** The registry still names this function as the
       producer of the frozen persisted column
       ``reward_count_candidate_arm_last10_known_trials`` (shipped in
       ``event_arm_table.pkl``), so the rename can't silently detach the
       function from the column it fills.
    """
    # Three completed trips to well 0 (arm 0), rewarded on trips 0 and 2, with
    # an event after all three. The 10-trial window covers all three trips, so
    # arm 0's count is 2 and every other arm is a structural 0 (history exists,
    # no reward there) -- not NaN.
    trials = _single_arm_trials(3, rewarded_indices={0, 2})
    out = covariates.rolling_reward_count_all_events(
        np.array([200.0]),
        port_experiences_from_trials(trials, prepend_initial=False),
        well_to_arm={0: 0, 1: 1},
    )
    assert out.shape[0] == 1
    assert out[0, 0] == 2.0, out[0, 0]
    assert np.all(out[0, 1:] == 0.0), out[0, 1:]

    definition = covariates.definition_for(
        "reward_count_candidate_arm_last10_known_trials"
    )
    assert definition.producer == (
        "covariates.history.rolling_reward_count_all_events"
    ), definition.producer


def test_no_caller_uses_old_reward_count_name():
    """The pre-rename names must appear nowhere in the source tree.

    Guards against a caller (or a re-added deprecation alias) reintroducing the
    misleading ``cumulative_reward_*`` names. The needles are assembled from
    fragments so this test's own text does not match them.
    """
    needles = ("cumulative_reward" + "_all_events", "cumulative_reward" + "_per_arm")
    repo_root = Path(__file__).resolve().parents[4]
    offenders: list[str] = []
    for root in ("src", "scripts", "tests"):
        for path in (repo_root / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(needle in text for needle in needles):
                offenders.append(str(path.relative_to(repo_root)))
    assert not offenders, f"old reward-count name still present in: {offenders}"


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
        out = table_common.align_count_matrix(
            counts, np.array([10, 20]), np.array([20, 10]), n_arms=2
        )
        np.testing.assert_array_equal(out, np.array([[3, 4], [1, 2]]))

    def test_missing_event_fills_zero_for_integer_counts(self):
        """A hard count genuinely observed zero real-arm bins."""
        counts = np.array([[1, 2]], dtype=np.int64)
        out = table_common.align_count_matrix(
            counts, np.array([10]), np.array([10, 99]), n_arms=2
        )
        np.testing.assert_array_equal(out[1], np.array([0, 0]))
        assert out.dtype == np.int64

    def test_missing_event_fills_nan_for_float_counts(self):
        """A fraction's 0/0 must stay undefined, not become 0 or +inf."""
        counts = np.array([[1.0, 2.0]], dtype=float)
        out = table_common.align_count_matrix(
            counts, np.array([10]), np.array([10, 99]), n_arms=2
        )
        assert np.isnan(out[1]).all()


class TestEventLabelAlignment:
    def test_accepts_same_event_set_in_different_order(self):
        table_common.assert_event_label_alignment(
            np.array([2, 1, 2, 1]), np.array([1, 2]), source_name="dot"
        )

    @pytest.mark.parametrize(
        "source,target",
        [
            ([1], [1, 2]),
            ([1, 2, 3], [1, 2]),
            ([1, np.nan], [1]),
            ([1, 1.5], [1]),
        ],
    )
    def test_rejects_missing_extra_or_nonfinite_event_ids(self, source, target):
        with pytest.raises(ValueError, match="event-label alignment"):
            table_common.assert_event_label_alignment(
                np.asarray(source), np.asarray(target), source_name="dot"
            )

    def test_requires_same_single_natural_session(self):
        source = pd.DataFrame(
            {"animal": ["a", "a"], "day": [1, 1], "epoch": ["e", "e"]}
        )
        target = pd.DataFrame({"animal": ["b"], "day": [1], "epoch": ["e"]})

        with pytest.raises(ValueError, match="session alignment failed"):
            table_common.assert_single_session_alignment(
                source, target, source_name="dot"
            )

        table_common.assert_single_session_alignment(
            source,
            target.assign(animal="a"),
            source_name="dot",
        )


# ── Guards for invariants that survived mutation testing ──


def test_omission_streak_is_undefined_without_a_resolved_outcome():
    """A streak of leading omissions needs a known outcome to start from.

    Returning 0 would put every pre-first-outcome stop in the same bucket as a
    stop whose most recent trial was rewarded -- a fabricated reference
    category, systematically the earliest stops of each session.
    """
    no_history = np.full((1, 10), np.nan)
    assert np.isnan(value_uncertainty._omission_streak(no_history)[0])
    # A genuine "last trial rewarded" is the value it must not be confused with.
    rewarded = np.array([[1.0] + [0.0] * 9])
    assert value_uncertainty._omission_streak(rewarded)[0] == 0.0
    # An unresolved tail still truncates the count rather than voiding it.
    partial = np.array([[0.0, np.nan] + [np.nan] * 8])
    assert value_uncertainty._omission_streak(partial)[0] == 1.0


def test_reward_rate_arm_window_is_ten_visits_not_five():
    """The column is named k10; pin the window so a 5->10 migration can't half-land."""
    trials = pd.DataFrame(
        {
            "movement_start_time": np.arange(12, dtype=float),
            "destination_first_poke_in_time": np.arange(12, dtype=float) + 0.5,
            "end_time": np.arange(12, dtype=float) + 0.5,
            "from_well": [0] * 12,
            "to_well": [1] * 12,
            # First two visits rewarded, next ten omitted.
            "is_reward": [True, True] + [False] * 10,
            "is_patch_change": [False] * 12,
        }
    )
    w2a = {i: i for i in range(6)}
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    rate = covariates.recent_reward_rate_all_events(
        np.array([11.9]), experiences, w2a, k=10
    )
    # Last 10 visits = trials 2..11, all omitted -> 0.0. A k=5 window also
    # gives 0.0, so include the earlier query that separates them.
    early = covariates.recent_reward_rate_all_events(
        np.array([6.9]), experiences, w2a, k=10
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
            "movement_start_time": [0.0, 10.0],
            "destination_first_poke_in_time": [5.0, 15.0],
            "end_time": [6.0, 16.0],
            "from_well": [0, 1],
            "to_well": [1, 1],
            # Visit 0 omitted, visit 1 rewarded. Leaking visit 1 turns 0.0 into 0.5.
            "is_reward": [False, True],
            "is_patch_change": [False, False],
        }
    )
    w2a = {i: i for i in range(6)}
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    # Query after visit 0's outcome (6.0) but before visit 1's (16.0).
    between = covariates.recent_reward_rate_all_events(
        np.array([12.0]), experiences, w2a, k=10
    )
    assert between[0, 1] == pytest.approx(0.0), "unresolved visit's reward leaked"
    # Once visit 1 resolves, it counts: 1 of 2 rewarded.
    after = covariates.recent_reward_rate_all_events(
        np.array([16.0]), experiences, w2a, k=10
    )
    assert after[0, 1] == pytest.approx(0.5)


# ── Reward-history raised-cosine features ──


def _reward_basis() -> np.ndarray:
    """The (3, 10) log-spaced raised-cosine kernel the features project onto."""
    return per_bin_design.raised_cosine_history_basis(10, 3)


def test_reward_history_features_are_bump_weighted_means_of_the_lag_vector():
    """With ten known outcomes every bump has full support, so each feature is
    the plain weighted mean of the lag vector -- the definition the coefficient
    is read against. A transposed or unnormalized projection changes the value.
    """
    # Rewarded on trials 0, 3, 4, 7 -> a non-monotone lag vector, so a bump
    # mix-up cannot be masked by a flat history.
    trials = _single_arm_trials(10, rewarded_indices={0, 3, 4, 7})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    query = np.array([100.0])  # after every outcome (last is known at 99.0)

    features, incomplete = covariates.reward_history_basis_features(query, experiences)
    lags = covariates.lagged_trial_rewards(query, experiences, k=10)
    assert np.isfinite(lags).all()

    basis = _reward_basis()
    expected = (basis @ lags[0]) / basis.sum(axis=1)
    np.testing.assert_allclose(features[0], expected)
    # The most recent outcome (trial 9, omitted) dominates bump 0 and the
    # oldest (trial 0, rewarded) dominates bump 2.
    assert features[0, 0] < features[0, 2]
    assert incomplete[0] == 0.0


def test_one_known_outcome_defines_only_the_two_recent_bumps():
    """Bump 2 has zero weight at lag 1, so a single known outcome leaves it
    undefined: it must be zero-filled and flagged, never silently 0 as a
    reward-free history.
    """
    trials = _single_arm_trials(1, rewarded_indices={0})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)

    features, incomplete = covariates.reward_history_basis_features(
        np.array([9.5]), experiences
    )
    assert features[0, 0] == pytest.approx(1.0)
    assert features[0, 1] == pytest.approx(1.0)
    assert features[0, 2] == 0.0
    assert incomplete[0] == 1.0


def test_second_known_outcome_completes_the_slow_bump():
    """Bump 2's earliest support is lag 2, so with exactly two known outcomes it
    equals the *second* most recent outcome -- not the most recent one.
    """
    # Trial 0 rewarded, trial 1 omitted: lag 1 = 0, lag 2 = 1.
    trials = _single_arm_trials(2, rewarded_indices={0})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)

    features, incomplete = covariates.reward_history_basis_features(
        np.array([19.5]), experiences
    )
    lags = covariates.lagged_trial_rewards(np.array([19.5]), experiences, k=10)
    assert lags[0, 0] == 0.0 and lags[0, 1] == 1.0
    assert features[0, 2] == pytest.approx(1.0)
    assert incomplete[0] == 0.0


def test_query_before_any_known_outcome_is_all_zero_and_flagged():
    """No history at all is zero-filled, and only the indicator distinguishes it
    from a genuinely reward-free history.
    """
    trials = _single_arm_trials(3, rewarded_indices={0, 1, 2})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)

    features, incomplete = covariates.reward_history_basis_features(
        np.array([0.5]), experiences
    )
    np.testing.assert_array_equal(features[0], np.zeros(3))
    assert incomplete[0] == 1.0


def test_reward_history_features_are_weighted_rates_in_the_unit_interval():
    """Features are reward *fractions*, so coefficients are per unit rate. Any
    value outside [0, 1] means the normalization was dropped.
    """
    trials = _single_arm_trials(12, rewarded_indices={1, 2, 5, 8, 11})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)
    query = np.arange(9.5, 120.0, 10.0)

    features, _ = covariates.reward_history_basis_features(query, experiences)
    assert features.min() >= 0.0
    assert features.max() <= 1.0
    # The fixture must exercise both ends, or the bound test is vacuous.
    assert features.max() > 0.5


def test_reward_history_features_advance_on_the_outcome_clock():
    """Inside the arrival-to-feedback gap the pending outcome is not yet known;
    reading it would leak the current trip's reward into its own stop.
    """
    # Trials 0 and 1 omitted, trial 2 rewarded. Arrival at 28.0, outcome at 29.0.
    trials = _single_arm_trials(3, rewarded_indices={2})
    experiences = port_experiences_from_trials(trials, prepend_initial=False)

    in_gap, _ = covariates.reward_history_basis_features(np.array([28.5]), experiences)
    after, _ = covariates.reward_history_basis_features(np.array([29.5]), experiences)
    # Before feedback the history is two omissions; after, the reward enters
    # as the newest lag and carries bump 0's full weight.
    basis = _reward_basis()
    np.testing.assert_allclose(in_gap[0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(after[0, 0], basis[0, 0] / basis[0, :3].sum())
    assert after[0, 0] > 0.4


def test_reward_basis_from_lags_is_nan_only_where_a_bump_has_no_support():
    """Partial history must not poison the bumps that *do* have finite support;
    a row-wise NaN rule would drop every feature whenever any lag is missing.
    """
    basis = _reward_basis()
    lags = np.full((2, 10), np.nan)
    lags[0, 0] = 1.0  # only lag 1 known -> bump 2 (zero weight at lag 1) undefined
    lags[1, 9] = 1.0  # only lag 10 known -> bump 0 (zero weight at lag 10) undefined

    out = covariates.reward_basis_from_lags(lags, basis)
    np.testing.assert_allclose(out[0, :2], 1.0)
    assert np.isnan(out[0, 2])
    assert np.isnan(out[1, 0])
    np.testing.assert_allclose(out[1, 1:], 1.0)
