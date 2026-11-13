"""Unit tests for the pure (DB-free) helpers in the qvals-table loader.

Covers the authoritative timing/model trial join, stop-start broadcast, nominal
per-port values, model-column adapter, outcome-availability clock, and persisted
scientific identities.  DataJoint reads remain outside this unit boundary; the
tests use real pandas frames through the same join and broadcast operations as
the production builder.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.covariates.qvals_table import (
    KEYS,
    REQUIRED_FITTED_MODEL_COVARIATES,
    _add_nominal_contingency,
    _broadcast_trial_covariates_to_stops,
    _join_model_to_trial_timebase,
    _model_covariate_columns,
    _outcome_known_time,
    _validate_qvals_output,
    add_beta_belief_entropy,
    add_experienced_value_columns,
    assign_stops_to_trials,
    canonicalize_behavior_model_columns,
    peri_switch_index,
)
from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC


def _valid_qvals_output() -> pd.DataFrame:
    chosen_belief = np.array([0.3, 0.7])
    trial_reward = np.array([0.0, 1.0])
    rpe = trial_reward - chosen_belief
    frame = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "task_reward_probability_chosen_arm": [0.2, 0.8],
            "chosen_arm": [0, 1],
            "reward_belief_mean_chosen_arm_preoutcome": chosen_belief,
            "rpe_postoutcome": rpe,
            "rpe_postoutcome_abs": np.abs(rpe),
            "relative_value_switch_minus_stay_preoutcome": [0.1, -0.1],
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.05, 0.06],
            "reward_belief_mean_update_l1_alternative_patches_postoutcome": [
                0.01,
                0.02,
            ],
            "reward_belief_differential_entropy_leaf_mean_preoutcome": [0.0, -0.1],
            "outcome_known_time": [1.0, 2.0],
            "qvals_context_time": [0.5, 1.5],
            "trial_reward": trial_reward,
        }
    )
    for arm in range(6):
        frame[f"task_reward_probability_port_{arm}"] = 0.1 * (arm + 1)
        frame[f"reward_belief_mean_port_{arm}"] = 0.1 * (arm + 1)
        frame[f"reward_belief_variance_port_{arm}"] = 0.01 * (arm + 1)
    return frame


def _trial_times(trial_numbers=(0, 1, 2)) -> pd.DataFrame:
    """Complete behavior timing rows for one epoch."""
    trial_numbers = list(trial_numbers)
    n = len(trial_numbers)
    leaf = np.arange(1, n + 1)
    reward = np.asarray(trial_numbers) % 2
    poke_out = 10.0 * (np.asarray(trial_numbers, dtype=float) + 1.0)
    return pd.DataFrame(
        {
            "nwb_file_name": ["rat20200101_.nwb"] * n,
            "epoch": [2] * n,
            "trial_number_by_epoch": trial_numbers,
            "poke_in_ts": poke_out - 2.0,
            "poke_out_ts": poke_out,
            "reward_on_ts": np.where(reward == 1, poke_out - 1.0, np.nan),
            "reward_off_ts": np.where(reward == 1, poke_out - 0.5, np.nan),
            "leaf": leaf,
            "reward": reward,
        }
    )


def _model_trials(trial_numbers=(0, 1, 2)) -> pd.DataFrame:
    """Canonical model rows with every paper-fitted qvals covariate."""
    trial_numbers = list(trial_numbers)
    n = len(trial_numbers)
    leaf = np.asarray(trial_numbers) + 1
    reward = np.asarray(trial_numbers) % 2
    chosen_mean = 0.2 + 0.1 * np.asarray(trial_numbers, dtype=float)
    frame = pd.DataFrame(
        {
            "nwb_file_name": ["rat20200101_.nwb"] * n,
            "epoch": [2] * n,
            "trial_number_by_epoch": trial_numbers,
            "leaf": leaf,
            "reward": reward,
            "chosen_arm": leaf - 1,
            "reward_belief_mean_chosen_arm_preoutcome": chosen_mean,
            "rpe_postoutcome": reward - chosen_mean,
            "rpe_postoutcome_abs": np.abs(reward - chosen_mean),
            "relative_value_switch_minus_stay_preoutcome": np.linspace(-0.1, 0.1, n),
            "reward_belief_mean_update_l1_all_ports_postoutcome": np.full(n, 0.05),
            "reward_belief_mean_update_l1_alternative_patches_postoutcome": np.full(
                n, 0.01
            ),
            "reward_belief_differential_entropy_leaf_mean_preoutcome": np.full(n, -0.2),
        }
    )
    for arm in range(6):
        frame[f"reward_belief_mean_port_{arm}"] = 0.1 * (arm + 1)
        frame[f"reward_belief_variance_port_{arm}"] = 0.01 * (arm + 1)
    assert set(REQUIRED_FITTED_MODEL_COVARIATES).issubset(frame.columns)
    return frame


def test_required_fitted_model_covariates_are_not_silently_optional():
    model = _model_trials().drop(columns=["rpe_postoutcome"])

    with pytest.raises(ValueError, match="trial table lacks required column"):
        _model_covariate_columns(model)


def test_trial_join_rejects_duplicate_keys_in_either_source():
    times = _trial_times()
    model = _model_trials()
    model_cols = _model_covariate_columns(model)

    duplicate_times = pd.concat([times, times.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="timing trial keys are duplicated"):
        _join_model_to_trial_timebase(duplicate_times, model, model_cols)

    duplicate_model = pd.concat([model, model.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="model trial keys are duplicated"):
        _join_model_to_trial_timebase(times, duplicate_model, model_cols)


def test_trial_join_rejects_nonconsecutive_complete_timing_keys():
    times = _trial_times((0, 2))
    model = _model_trials((0, 2))

    with pytest.raises(ValueError, match="timing trial numbers must be consecutive"):
        _join_model_to_trial_timebase(times, model, _model_covariate_columns(model))


def test_trial_join_rejects_model_only_trial_keys():
    times = _trial_times((0, 1))
    model = _model_trials((0, 1, 2))

    with pytest.raises(ValueError, match="model-only trial keys"):
        _join_model_to_trial_timebase(times, model, _model_covariate_columns(model))


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        ("leaf", 6, "leaf labels disagree"),
        ("reward", 0, "reward labels disagree"),
        ("rpe_postoutcome", 999.0, "rpe_postoutcome != reward -"),
        ("rpe_postoutcome_abs", 999.0, "rpe_postoutcome_abs != abs"),
    ],
)
def test_trial_join_rejects_cross_source_and_rpe_identity_failures(
    column, bad, message
):
    times = _trial_times()
    model = _model_trials()
    model.loc[1, column] = bad

    with pytest.raises(ValueError, match=message):
        _join_model_to_trial_timebase(times, model, _model_covariate_columns(model))


def test_internal_missing_model_trial_preserves_later_stop_alignment():
    """A model gap keeps its timing interval and cannot shift trial 2 to trial 1."""
    times = _trial_times()
    model = _model_trials((0, 2))
    model_cols = _model_covariate_columns(model)

    joined = _join_model_to_trial_timebase(times, model, model_cols)

    assert joined["trial_number_by_epoch"].tolist() == [0, 1, 2]
    assert joined["_model_row_available"].tolist() == [True, False, True]
    assert np.isnan(joined.loc[1, "reward_belief_mean_chosen_arm_preoutcome"])
    assert joined.loc[2, "reward_belief_mean_chosen_arm_preoutcome"] == pytest.approx(
        0.4
    )
    # The vendored code uses adjacent rows for these two fitted quantities.
    # They must be blanked rather than silently bridging trial 0 -> trial 2.
    assert np.isnan(joined.loc[0, "reward_belief_mean_update_l1_all_ports_postoutcome"])
    assert np.isnan(joined.loc[2, "relative_value_switch_minus_stay_preoutcome"])
    # Direct current-trial quantities remain valid on trial 2.
    assert joined.loc[2, "rpe_postoutcome"] == pytest.approx(-0.4)
    np.testing.assert_allclose(joined["poke_out_ts"], [10.0, 20.0, 30.0])
    assert joined.attrs["timing_trials_without_model_by_session"] == [
        {
            "nwb_file_name": "rat20200101_.nwb",
            "epoch": 2,
            "n_timing_trials_without_model": 1,
        }
    ]

    stops = pd.DataFrame(
        {
            "animal": ["rat", "rat"],
            "session_date": ["20200101", "20200101"],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": ["missing-model-trial", "later-model-trial"],
            "stop_start": [15.0, 25.0],
            "stop_end": [16.0, 26.0],
        }
    )
    out = _broadcast_trial_covariates_to_stops("rat", joined, model_cols, stops)

    assert out["stop_id"].tolist() == ["later-model-trial"]
    assert out["trial_number_by_epoch"].tolist() == [2]
    assert out["reward_belief_mean_chosen_arm_preoutcome"].tolist() == [
        pytest.approx(0.4)
    ]
    assert out.attrs["n_stops_missing_model_trial"] == 1


def test_complete_trial_join_is_identical_to_legacy_inner_join():
    """The hardening must not numerically alter complete, already-aligned data."""
    times = _trial_times()
    model = _model_trials()
    model_cols = _model_covariate_columns(model)
    expected = times.merge(
        model[[*KEYS, "leaf", "reward", *model_cols]],
        on=KEYS,
        suffixes=("_time", "_val"),
        validate="one_to_one",
    )

    actual = _join_model_to_trial_timebase(times, model, model_cols).drop(
        columns="_model_row_available"
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_qvals_output_requires_unique_stop_ids():
    frame = _valid_qvals_output()
    frame.loc[1, "stop_id"] = frame.loc[0, "stop_id"]
    with pytest.raises(ValueError, match="one nonmissing row per stop_id"):
        _validate_qvals_output(frame)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_qvals_output_requires_finite_probabilities_in_unit_interval(bad):
    frame = _valid_qvals_output()
    frame.loc[0, "reward_belief_mean_port_3"] = bad
    with pytest.raises(ValueError, match="probability/value columns"):
        _validate_qvals_output(frame)


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        ("reward_belief_variance_port_1", 0.3, "posterior variances"),
        ("reward_belief_differential_entropy_leaf_mean_preoutcome", 0.1, "entropy"),
        ("outcome_known_time", np.inf, "times"),
        ("trial_reward", 0.5, "trial_reward"),
    ],
)
def test_qvals_output_validates_uncertainty_timing_and_reward(column, bad, message):
    frame = _valid_qvals_output()
    frame.loc[0, column] = bad

    with pytest.raises(ValueError, match=message):
        _validate_qvals_output(frame)


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        ("rpe_postoutcome", 0.0, "rpe_postoutcome != trial_reward -"),
        ("rpe_postoutcome_abs", 0.0, "rpe_postoutcome_abs != abs"),
    ],
)
def test_qvals_output_validates_exact_rpe_identities(column, bad, message):
    frame = _valid_qvals_output()
    frame.loc[0, column] = bad

    with pytest.raises(ValueError, match=message):
        _validate_qvals_output(frame)


@pytest.mark.parametrize(
    ("rewarded", "reward_on", "expected"),
    (
        (True, 10.2, 10.2),
        # Omission: poke_in + the latency by which a pump would have fired,
        # NOT poke-out. Poke-out is a behavioural boundary and lands a median
        # 880 ms after the information exists.
        (False, np.nan, 10.0 + OMISSION_KNOWN_LATENCY_SEC),
    ),
)
def test_outcome_known_time_uses_reward_on_or_omission_latency(
    rewarded, reward_on, expected
):
    assert _outcome_known_time(
        rewarded=rewarded,
        poke_in_time=10.0,
        poke_out_time=14.0,
        reward_on_time=reward_on,
    ) == pytest.approx(expected)


def test_omission_boundary_is_capped_at_poke_out():
    """A poke shorter than the latency is settled by the departure itself."""
    short = _outcome_known_time(
        rewarded=False,
        poke_in_time=10.0,
        poke_out_time=10.1,  # shorter than OMISSION_KNOWN_LATENCY_SEC
        reward_on_time=np.nan,
    )
    assert short == pytest.approx(10.1)


def test_omission_latency_exceeds_every_observed_reward_latency():
    """The boundary must not declare an omission while a pump could fire.

    Max observed reward latency across 18,753 rewarded task visits in the
    refreshed 218-session behavioral cache is 0.2361 s.
    """
    assert OMISSION_KNOWN_LATENCY_SEC > 0.2361


def test_outcome_known_time_rejects_missing_physical_boundary():
    with pytest.raises(ValueError, match="poke_in"):
        _outcome_known_time(
            rewarded=False,
            poke_in_time=np.nan,
            poke_out_time=14.0,
            reward_on_time=np.nan,
        )
    with pytest.raises(ValueError, match="poke_out"):
        _outcome_known_time(
            rewarded=False,
            poke_in_time=10.0,
            poke_out_time=np.nan,
            reward_on_time=np.nan,
        )
    with pytest.raises(ValueError, match="reward_on"):
        _outcome_known_time(
            rewarded=True,
            poke_in_time=10.0,
            poke_out_time=14.0,
            reward_on_time=np.nan,
        )


@pytest.mark.parametrize(
    ("rewarded", "poke_in", "poke_out", "reward_on", "match"),
    [
        (True, 10.0, 14.0, 9.9, "within"),
        (True, 10.0, 14.0, 14.1, "within"),
        (False, 10.0, 14.0, 12.5, "omission"),
        (False, 14.0, 10.0, np.nan, "before"),
    ],
)
def test_outcome_known_time_rejects_inconsistent_physical_clock(
    rewarded, poke_in, poke_out, reward_on, match
):
    with pytest.raises(ValueError, match=match):
        _outcome_known_time(
            rewarded=rewarded,
            poke_in_time=poke_in,
            poke_out_time=poke_out,
            reward_on_time=reward_on,
        )


def _toy_times(leaves, ports_pct, *, epoch="02_r1", block=0):
    """Build a minimal ``times`` frame for one epoch/contingency block.

    ports_pct: (n_trials, 6) nominal p(reward) per leaf, in PERCENT (as stored
    in ``p_rew_leaf1..6``).
    """
    ports_pct = np.asarray(ports_pct, dtype=float)
    n = len(leaves)
    df = {
        "nwb_file_name": ["x20200101_.nwb"] * n,
        "epoch": [epoch] * n,
        "trial_number_by_epoch": list(range(n)),
        "leaf": list(leaves),
        "contingency_count": [block] * n,
    }
    for i in range(1, 7):
        df[f"p_rew_leaf{i}"] = ports_pct[:, i - 1]
    return pd.DataFrame(df)


# ── peri_switch_index (signed trials-to-nearest-switch) ──
# Concept: +k trials AFTER the nearest switch, -k BEFORE, 0 at a switch; the
# switch is a between-trial patch change. Oracle values read off the definition.


def test_peri_switch_index_hand():
    # patches switch at index 3 (0->1) and 5 (1->0)
    patch = np.array([0, 0, 0, 1, 1, 0, 0])
    tss, isp = peri_switch_index(patch)
    np.testing.assert_array_equal(tss, [-3, -2, -1, 0, 1, 0, 1])
    np.testing.assert_array_equal(isp, [0, 0, 0, 1, 1, 1, 1])


def test_peri_switch_index_no_switch_is_nan():
    tss, isp = peri_switch_index(np.array([2, 2, 2, 2]))
    assert np.isnan(tss).all() and np.isnan(isp).all()


def test_peri_switch_index_clips():
    patch = np.array([0] * 30 + [1])  # switch at idx 30; idx 0 is 30 trials before
    tss, _ = peri_switch_index(patch, clip=20)
    assert tss[0] == -20 and tss[30] == 0


def test_peri_switch_invariant_property():
    """CONTRACT for arbitrary patch sequences: at a switch tss==0; magnitude is the
    trial-distance to the nearest switch; sign is + iff that switch is at/behind."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        patch = rng.integers(0, 3, size=int(rng.integers(2, 40)))
        tss, isp = peri_switch_index(patch, clip=1000)
        sw = np.where(np.r_[False, patch[1:] != patch[:-1]])[0]
        if sw.size == 0:
            assert np.isnan(tss).all()
            continue
        for i in range(len(patch)):
            dist = np.abs(sw - i).min()
            assert abs(tss[i]) == dist
            behind = (sw <= i).any() and (i - sw[sw <= i].max()) <= dist
            assert bool(isp[i]) == behind
            if i in sw:
                assert tss[i] == 0


def test_add_nominal_contingency_adds_peri_switch():
    # leaves 1,1,1,3,3,1,1 -> patches 0,0,0,1,1,0,0 (switch at trial 3 and 5)
    t = _toy_times([1, 1, 1, 3, 3, 1, 1], [[50] * 6] * 7)
    out = _add_nominal_contingency(t)
    assert out["periswitch_index_trials"].tolist() == [-3, -2, -1, 0, 1, 0, 1]
    assert out["is_periswitch_post"].tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_per_port_identities_and_patch_value():
    # leaf 3; ports (pct) = [20,50,80,80,20,20] -> frac [.2,.5,.8,.8,.2,.2]
    t = _toy_times([3], [[20, 50, 80, 80, 20, 20]])
    out = _add_nominal_contingency(t)
    # task_reward_probability_port_{i} == p_rew_leaf{i+1} / 100
    for i in range(6):
        assert out[f"task_reward_probability_port_{i}"].iloc[0] == pytest.approx(
            [0.2, 0.5, 0.8, 0.8, 0.2, 0.2][i]
        )
    # task_reward_probability_chosen_arm == p_reward_port[leaf-1]
    assert out["task_reward_probability_chosen_arm"].iloc[0] == pytest.approx(
        out["task_reward_probability_port_2"].iloc[0]
    )
    assert out["task_reward_probability_best_arm"].iloc[0] == pytest.approx(0.8)
    # patch means (WELL_PATCH=[0,0,1,1,2,2]): {.35, .8, .2}; diff = .8 - .35
    assert out["task_reward_probability_patch_mean_gap_best_vs_second"].iloc[
        0
    ] == pytest.approx(0.45)


def test_task_reward_probability_chosen_arm_matches_across_rows():
    leaves = [1, 4, 6]
    ports = [[80, 20, 20, 50, 20, 20]] * 3
    out = _add_nominal_contingency(_toy_times(leaves, ports))
    port_mat = out[[f"task_reward_probability_port_{i}" for i in range(6)]].to_numpy()
    arm = out["leaf"].to_numpy() - 1
    expected = port_mat[np.arange(len(out)), arm]
    np.testing.assert_allclose(
        out["task_reward_probability_chosen_arm"].to_numpy(), expected
    )


def test_trials_since_block_change_is_within_block_cumcount():
    # two blocks of the same epoch -> counter resets at the block boundary
    a = _toy_times([1, 2, 3], [[50] * 6] * 3, block=0)
    b = _toy_times([4, 5], [[50] * 6] * 2, block=1)
    b["trial_number_by_epoch"] = [3, 4]
    out = _add_nominal_contingency(pd.concat([a, b], ignore_index=True))
    assert out["trials_since_block_change"].tolist() == [0, 1, 2, 0, 1]


@pytest.mark.parametrize("bad_leaf", [0, 7, -1])
def test_leaf_range_guard_raises(bad_leaf):
    t = _toy_times([bad_leaf], [[50] * 6])
    with pytest.raises(ValueError, match="leaf outside 1..6"):
        _add_nominal_contingency(t)


# ── assign_stops_to_trials (stop -> trial by poke-out interval) ──
# Concept: trial t spans [poke_out[t-1], poke_out[t]) (poke_out[-1] = -inf);
# stops at/after the last poke_out have no enclosing trial (-> -1, dropped).


@pytest.mark.parametrize(
    "mid, expected",
    [
        (15, 1),  # interior of [10, 20)
        (10, 1),  # left boundary == poke_out[0]: belongs to the STARTING trial 1
        (20, 2),  # == poke_out[1]: rolls to trial 2, not trial 1
        (5, 0),  # before the first poke_out -> trial 0
        (30, -1),  # exactly the last poke_out -> no enclosing trial (dropped)
        (99, -1),  # after the last poke_out -> dropped
    ],
)
def test_assign_stop_boundaries(mid, expected):
    po = np.array([10.0, 20.0, 30.0])
    # oracle values are read off the interval definition by hand, not searchsorted
    assert assign_stops_to_trials(po, np.array([mid]))[0] == expected


def test_assign_stops_vectorized_matches_scalar():
    po = np.array([10.0, 20.0, 30.0])
    mids = np.array([5, 10, 15, 20, 25, 30, 99], dtype=float)
    np.testing.assert_array_equal(
        assign_stops_to_trials(po, mids), [0, 1, 1, 2, 2, -1, -1]
    )


def test_assign_stops_interval_invariant_property():
    """For arbitrary (sorted) poke-outs and stops, the CONTRACT must hold:
    a stop assigned to trial t satisfies poke_out[t-1] <= mid < poke_out[t]
    (poke_out[-1] = -inf); a dropped stop (-1) sits at/after the last poke_out.
    Validates the rule independent of the searchsorted mechanism."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        n = int(rng.integers(1, 8))
        po = np.cumsum(rng.uniform(0.5, 3.0, size=n))  # strictly increasing
        mids = rng.uniform(po[0] - 2, po[-1] + 2, size=20)
        ti = assign_stops_to_trials(po, mids)
        for m, t in zip(mids, ti, strict=True):
            if t == -1:
                assert m >= po[-1]
            else:
                lo = -np.inf if t == 0 else po[t - 1]
                assert lo <= m < po[t]


def test_stop_start_context_does_not_jump_to_midpoint_trial():
    """A long speed bout keeps the trial active at its canonical start.

    The same bout's midpoint lies after a poke-out boundary; assigning by the
    midpoint would attach trial 1 even though all stop-level context is frozen
    at the trial-0 bout start.
    """
    poke_out = np.array([10.0, 20.0])
    stop_start = np.array([9.0])
    stop_midpoint = np.array([11.0])

    assert assign_stops_to_trials(poke_out, stop_start).tolist() == [0]
    assert assign_stops_to_trials(poke_out, stop_midpoint).tolist() == [1]


def test_stop_broadcast_assigns_qvals_from_stop_start_not_midpoint():
    """Pin the production broadcast caller using real pandas frames."""
    per_trial = pd.DataFrame(
        {
            "nwb_file_name": ["rat20200101_.nwb"] * 2,
            "epoch": [2, 2],
            "trial_number_by_epoch": [0, 1],
            "poke_in_ts": [8.0, 18.0],
            "poke_out_ts": [10.0, 20.0],
            "reward_on_ts": [np.nan, 19.0],
            "reward_off_ts": [np.nan, 19.5],
            "reward_time": [0, 1],
            "_model_row_available": [True, True],
            "reward_belief_mean_chosen_arm_preoutcome": [0.1, 0.9],
        }
    )
    per_trial.attrs["leaf_agree"] = 1.0
    per_trial.attrs["reward_agree"] = 1.0
    per_trial.attrs["entropy_cov"] = 1.0
    stops = pd.DataFrame(
        {
            "animal": ["rat"],
            "session_date": ["20200101"],
            "epoch": ["02_r1"],
            "stop_id": ["stop-0"],
            "stop_start": [9.0],
            # The midpoint is 11.0, after trial 0's poke-out boundary.
            "stop_end": [13.0],
        }
    )

    result = _broadcast_trial_covariates_to_stops(
        "rat",
        per_trial,
        ["reward_belief_mean_chosen_arm_preoutcome"],
        stops,
    )

    assert result.loc[0, "trial_number_by_epoch"] == 0
    assert result.loc[0, "qvals_context_time"] == 9.0
    assert result.loc[0, "reward_belief_mean_chosen_arm_preoutcome"] == pytest.approx(
        0.1
    )
    # Omission trial (poke_in 8.0, poke_out 10.0): the boundary is
    # poke_in + the omission latency, not poke-out.
    assert result.loc[0, "outcome_known_time"] == pytest.approx(
        8.0 + OMISSION_KNOWN_LATENCY_SEC
    )


# ── add_experienced_value_columns (model belief of the chosen arm) ──
# Concept: q_chosen is the model value for the arm the animal chose (Q{leaf}),
# with correct 1-based-leaf -> 0-based-arm indexing; rpe = reward - q_chosen.


def _toy_vals(leaves, rewards, q_rows):
    df = {"leaf": list(leaves), "reward": list(rewards)}
    q = np.asarray(q_rows, dtype=float)
    for i in range(1, 7):
        df[f"Q{i}"] = q[:, i - 1]
    return pd.DataFrame(df)


@pytest.mark.parametrize("reward, exp_rpe", [(1, 0.67), (0, -0.33)])
def test_experienced_value_picks_chosen_arm(reward, exp_rpe):
    # distinct Q per port so picking the wrong arm gives a different literal
    q = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    out = add_experienced_value_columns(_toy_vals([3], [reward], [q]))
    assert out["arm"].iloc[0] == 2  # native realized arm: leaf 3 -> arm 2
    assert out["q_chosen"].iloc[0] == pytest.approx(0.33)  # Q3, not Q2/Q4
    assert not any(c.startswith("q_port") for c in out.columns)
    assert out["rpe"].iloc[0] == pytest.approx(exp_rpe)
    assert out["rpe_abs"].iloc[0] == pytest.approx(abs(exp_rpe))


def test_behavior_model_adapter_translates_native_realized_arm_and_beliefs():
    """External model vocabulary changes only at the qvals ownership boundary."""
    native = add_experienced_value_columns(
        _toy_vals([3], [1], [[0.11, 0.22, 0.33, 0.44, 0.55, 0.66]])
    )
    for port in range(6):
        native[f"betadist_mu{port + 1}"] = native[f"Q{port + 1}"]

    canonical = canonicalize_behavior_model_columns(native)

    assert canonical.loc[0, "chosen_arm"] == 2
    assert canonical.loc[
        0, "reward_belief_mean_chosen_arm_preoutcome"
    ] == pytest.approx(0.33)
    assert canonical.loc[0, "reward_belief_mean_port_2"] == pytest.approx(0.33)
    assert "candidate_arm" not in canonical
    assert not {"arm", "q_chosen", "Q3", "q_port2", "betadist_mu3"} & set(canonical)


def test_native_qvals_vocabulary_canonicalizes_end_to_end():
    """Native spellings become the canonical schema; values are untouched."""
    native = pd.DataFrame(
        {
            "stop_id": [1, 2],
            # numerical-path family: two spellings of one quantity
            "Q1": [0.5, 0.6],
            "betadist_mu1": [0.5 + 1e-16, 0.6 - 1e-16],
            "prev_global_variance": [0.1, 0.2],
            "global_leaf_var_sum_preoutcome_sess": [0.1, 0.2],
            # plain renames
            "betadist_var1": [0.01, 0.02],
            "Qstem1": [0.4, 0.7],
            "rpe": [-0.1, 0.3],
            "arm": [2, 5],
            "p_reward_current": [0.2, 0.8],
            "chosen_leaf_mu_update_abs_postoutcome_sess": [np.nan, 0.125],
            # dead outputs of the withleaf fit
            "depletion_leaf_1": [1.0, 1.0],
            "depletion_mean_stem_3": [1.0, 1.0],
        }
    )

    out = canonicalize_behavior_model_columns(native)

    assert list(out.columns) == [
        "stop_id",
        "reward_belief_mean_port_0",
        "reward_belief_variance_leaf_sum_preoutcome",
        "reward_belief_variance_port_0",
        "decision_value_patch_1",
        "rpe_postoutcome",
        "chosen_arm",
        "task_reward_probability_chosen_arm",
        "reward_belief_mean_update_chosen_arm_postoutcome_abs",
    ]
    for canonical, source in [
        ("reward_belief_mean_port_0", "Q1"),
        ("reward_belief_variance_leaf_sum_preoutcome", "prev_global_variance"),
        ("rpe_postoutcome", "rpe"),
        ("chosen_arm", "arm"),
        (
            "reward_belief_mean_update_chosen_arm_postoutcome_abs",
            "chosen_leaf_mu_update_abs_postoutcome_sess",
        ),
    ]:
        np.testing.assert_array_equal(
            out[canonical].to_numpy(), native[source].to_numpy()
        )


def test_qvals_ingestion_leaves_canonical_columns_alone():
    frame = pd.DataFrame({"stop_id": [1], "rpe_postoutcome_z": [0.3]})
    pd.testing.assert_frame_equal(canonicalize_behavior_model_columns(frame), frame)


@pytest.mark.parametrize(
    "q1, mu1, match",
    [
        ([0.5, 0.6], [0.5, 0.61], "exceed tolerance"),
        ([0.5, np.nan], [0.5, 0.6], "missingness"),
        ([np.nan, np.nan], [np.nan, np.nan], "unverifiable"),
        ([0.5, np.inf], [0.5, np.inf], "infinity"),
    ],
)
def test_qvals_ingestion_rejects_disagreeing_numerical_paths(q1, mu1, match):
    native = pd.DataFrame({"Q1": q1, "betadist_mu1": mu1})
    with pytest.raises(ValueError, match=match):
        canonicalize_behavior_model_columns(native)


def test_qvals_ingestion_rejects_an_incomplete_numerical_path_family():
    with pytest.raises(ValueError, match="incomplete"):
        canonicalize_behavior_model_columns(pd.DataFrame({"Q1": [0.5]}))


def test_qvals_ingestion_rejects_a_name_collision():
    native = pd.DataFrame({"arm": [2], "chosen_arm": [2]})
    with pytest.raises(ValueError, match="collides"):
        canonicalize_behavior_model_columns(native)


def test_rpe_is_surprise_relative_to_model_belief():
    """Directionality that a nominal implementation would NOT satisfy: for equal
    reward, RPE is larger on a low-belief arm than on a high-belief arm."""
    q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    out = add_experienced_value_columns(
        _toy_vals(leaves=[1, 6], rewards=[1, 1], q_rows=[q, q])
    )
    rpe_low, rpe_high = out["rpe"].iloc[0], out["rpe"].iloc[1]
    assert rpe_low > rpe_high
    assert rpe_low == pytest.approx(0.9) and rpe_high == pytest.approx(0.1)


@pytest.mark.parametrize("bad_leaf", [0, 7, -1])
def test_experienced_value_leaf_guard_raises(bad_leaf):
    v = _toy_vals([bad_leaf], [1], [[0.5] * 6])
    with pytest.raises(ValueError, match="leaf outside 1..6"):
        add_experienced_value_columns(v)


# ── add_beta_belief_entropy (beta belief entropy) ──
# beta_reward_entropy = mean over the 6 leaves of the Beta(α,β) DIFFERENTIAL entropy
# H = betaln(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b). Validated against scipy.


def _beta_frame(ab_by_leaf):
    """One-row ``vals`` frame with ``betadist_α{i}``/``betadist_β{i}`` for i in 1..6.

    ``ab_by_leaf`` is a list of six ``(alpha, beta)`` tuples (Greek-named columns
    exactly as the fit emits them).
    """
    row = {}
    for i, (alpha, beta) in enumerate(ab_by_leaf, start=1):
        row[f"betadist_α{i}"] = [float(alpha)]
        row[f"betadist_β{i}"] = [float(beta)]
    return pd.DataFrame(row)


def test_beta_belief_entropy_matches_scipy_oracle():
    from scipy.stats import beta as beta_dist

    ab = [(1.0, 1.0), (2.0, 5.0), (10.0, 3.0), (3.0, 3.0), (1.5, 4.0), (8.0, 2.0)]
    out = add_beta_belief_entropy(_beta_frame(ab))
    oracle = float(np.mean([beta_dist(a, b).entropy() for a, b in ab]))
    np.testing.assert_allclose(
        out["beta_reward_entropy"].to_numpy(), [oracle], atol=1e-10
    )


def test_beta_belief_entropy_uniform_is_zero():
    # Beta(1,1) is uniform -> differential entropy exactly 0; mean over 6 leaves = 0.
    out = add_beta_belief_entropy(_beta_frame([(1.0, 1.0)] * 6))
    np.testing.assert_allclose(out["beta_reward_entropy"].to_numpy(), [0.0], atol=1e-12)


def test_beta_belief_entropy_decreases_with_concentration():
    # A concentrated belief (high α+β) is more certain -> lower differential entropy.
    diffuse = add_beta_belief_entropy(_beta_frame([(1.0, 1.0)] * 6))[
        "beta_reward_entropy"
    ].iloc[0]
    concentrated = add_beta_belief_entropy(_beta_frame([(20.0, 20.0)] * 6))[
        "beta_reward_entropy"
    ].iloc[0]
    assert concentrated < diffuse


def test_beta_belief_entropy_missing_param_raises():
    frame = _beta_frame([(2.0, 2.0)] * 6).drop(columns=["betadist_α3"])
    with pytest.raises(
        ValueError, match=r"lacks required column\(s\) \['betadist_α3'\]"
    ):
        add_beta_belief_entropy(frame)
