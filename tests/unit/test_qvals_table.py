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
