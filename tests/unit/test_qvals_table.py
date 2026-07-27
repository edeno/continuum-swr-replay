"""Unit tests for the pure (DB-free) helpers in the qvals_table loader.

Covers ``_add_nominal_contingency`` — the per-port nominal value block and its
leaf-range guard. These lock the identities every downstream consumer relies on
(``attach_contingency_value_block`` maps ``p_reward_port{leaf-1}`` -> the chosen
arm), which the range-only runtime verifier does not assert. The DataJoint-backed
paths (``_fetch_model_df`` / ``build_animal``) are exercised by the slow runner
smoke test, not here.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.qvals_table import (
    _add_nominal_contingency,
    add_beta_belief_entropy,
    add_experienced_value_columns,
    assign_stops_to_trials,
    peri_switch_index,
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
    assert out["trials_since_switch"].tolist() == [-3, -2, -1, 0, 1, 0, 1]
    assert out["is_post_switch"].tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_per_port_identities_and_patch_value():
    # leaf 3; ports (pct) = [20,50,80,80,20,20] -> frac [.2,.5,.8,.8,.2,.2]
    t = _toy_times([3], [[20, 50, 80, 80, 20, 20]])
    out = _add_nominal_contingency(t)
    # p_reward_port{i} == p_rew_leaf{i+1} / 100
    for i in range(6):
        assert out[f"p_reward_port{i}"].iloc[0] == pytest.approx([0.2, 0.5, 0.8, 0.8, 0.2, 0.2][i])
    # p_reward_current == p_reward_port[leaf-1]
    assert out["p_reward_current"].iloc[0] == pytest.approx(out["p_reward_port2"].iloc[0])
    assert out["p_reward_best"].iloc[0] == pytest.approx(0.8)
    # patch means (WELL_PATCH=[0,0,1,1,2,2]): {.35, .8, .2}; diff = .8 - .35
    assert out["value_diff_patch"].iloc[0] == pytest.approx(0.45)


def test_p_reward_current_matches_chosen_arm_across_rows():
    leaves = [1, 4, 6]
    ports = [[80, 20, 20, 50, 20, 20]] * 3
    out = _add_nominal_contingency(_toy_times(leaves, ports))
    port_mat = out[[f"p_reward_port{i}" for i in range(6)]].to_numpy()
    arm = out["leaf"].to_numpy() - 1
    expected = port_mat[np.arange(len(out)), arm]
    np.testing.assert_allclose(out["p_reward_current"].to_numpy(), expected)


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
        (15, 1),   # interior of [10, 20)
        (10, 1),   # left boundary == poke_out[0]: belongs to the STARTING trial 1
        (20, 2),   # == poke_out[1]: rolls to trial 2, not trial 1
        (5, 0),    # before the first poke_out -> trial 0
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
    assert out["arm"].iloc[0] == 2                       # leaf 3 -> arm 2
    assert out["q_chosen"].iloc[0] == pytest.approx(0.33)  # Q3, not Q2/Q4
    assert out["q_port0"].iloc[0] == pytest.approx(0.11)
    assert out["q_port5"].iloc[0] == pytest.approx(0.66)
    assert out["rpe"].iloc[0] == pytest.approx(exp_rpe)
    assert out["rpe_abs"].iloc[0] == pytest.approx(abs(exp_rpe))


def test_rpe_is_surprise_relative_to_model_belief():
    """Directionality that a nominal implementation would NOT satisfy: for equal
    reward, RPE is larger on a low-belief arm than on a high-belief arm."""
    q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    out = add_experienced_value_columns(
        _toy_vals(leaves=[1, 6], rewards=[1, 1], q_rows=[q, q])
    )
    rpe_low, rpe_high = out["rpe"].iloc[0], out["rpe"].iloc[1]  # arm0 q=.1, arm5 q=.9
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
    np.testing.assert_allclose(out["beta_reward_entropy"].to_numpy(), [oracle], atol=1e-10)


def test_beta_belief_entropy_uniform_is_zero():
    # Beta(1,1) is uniform -> differential entropy exactly 0; mean over 6 leaves = 0.
    out = add_beta_belief_entropy(_beta_frame([(1.0, 1.0)] * 6))
    np.testing.assert_allclose(out["beta_reward_entropy"].to_numpy(), [0.0], atol=1e-12)


def test_beta_belief_entropy_decreases_with_concentration():
    # A concentrated belief (high α+β) is more certain -> lower differential entropy.
    diffuse = add_beta_belief_entropy(_beta_frame([(1.0, 1.0)] * 6))["beta_reward_entropy"].iloc[0]
    concentrated = add_beta_belief_entropy(
        _beta_frame([(20.0, 20.0)] * 6)
    )["beta_reward_entropy"].iloc[0]
    assert concentrated < diffuse


def test_beta_belief_entropy_missing_param_raises():
    frame = _beta_frame([(2.0, 2.0)] * 6).drop(columns=["betadist_α3"])
    with pytest.raises(ValueError, match="beta posterior params missing"):
        add_beta_belief_entropy(frame)
