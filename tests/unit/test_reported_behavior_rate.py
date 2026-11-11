"""The behavior summary's per-bin rate values: the reward gate, the lag-1 kernel and the
corrected history peak, on synthetic fits."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.model_specs import (
    PER_BIN_DESIGN,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import reward_kernel_weights
from continuum_swr_replay.reported import behavior

ANIMALS = ("a", "b", "c", "d", "e")


def _fits(outcome: str, coefs, separated=None, converged=None) -> pd.DataFrame:
    """One full-rung fit per row of ``coefs``, two sessions per animal in turn."""
    coefs = np.asarray(coefs, dtype=float)
    n = len(coefs)
    frame = pd.DataFrame(
        {
            "animal": [ANIMALS[i % 5] for i in range(n)],
            "session_date": [f"d{i}" for i in range(n)],
            "epoch": ["02_r1"] * n,
            "model": behavior.FULL_RATE_RUNG,
            "outcome": outcome,
            "separated_terms": separated if separated is not None else [""] * n,
            "converged": converged if converged is not None else [True] * n,
        }
    )
    for k, term in enumerate(REWARD_HISTORY_COLUMNS):
        frame[f"coef_{term}"] = coefs[:, k]
        # The fits carry each predictor's SD; the runaway backstop reads
        # |coef| * sd, so large coefficients on these small-SD features are kept.
        frame[f"sd_{term}"] = 0.1
    return frame


def _lag_one(coefs) -> np.ndarray:
    weights = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
    return np.asarray(coefs, dtype=float) @ weights[:, 0]


def test_the_reward_gate_leaves_out_a_fit_with_any_separated_reward_term():
    coefs = np.tile([1.0, -0.5, 0.2], (10, 1))
    separated = [""] * 10
    separated[3] = f"{REWARD_HISTORY_COLUMNS[1]}|hist_rcos_0"
    # A separated self-history term alone does not drop the reward kernel.
    separated[4] = "hist_rcos_0"
    fits = _fits("non_local_swr", coefs, separated)

    kept = behavior._reward_gated(fits)

    assert len(kept) == 9
    assert fits.index[3] not in kept.index
    summary = behavior._lag_one_summary(fits)
    assert summary["n"] == 9 and summary["n_dropped"] == 1


def test_the_lag_one_summary_reports_the_kernel_not_the_first_coefficient():
    # A large first coefficient cancelled by the second: exp(c0) is 20x, but
    # the kernel at lag 1 is below 1.
    c0 = np.log(20.0)
    c1 = -c0 * 0.5 / 0.066  # about -22.7: makes 0.32*c0 + 0.066*c1 < 0
    coefs = np.tile([c0, c1, 0.0], (10, 1))
    fits = _fits("non_local_swr", coefs, converged=[True] * 9 + [False])

    summary = behavior._lag_one_summary(fits)

    assert summary["rr_animal"] == pytest.approx(np.exp(_lag_one([c0, c1, 0.0])))
    assert summary["rr_animal"] < 1
    assert summary["animals_positive"] == 0
    assert summary["n_nonconverged"] == 1


def test_the_paired_test_compares_lag_one_kernels_within_session():
    rng = np.random.default_rng(0)
    n = 20
    # SWR-coupled: larger lag-1 kernel; non-SWR: a larger first coefficient
    # offset by the second, so the coefficient comparison would run the other way.
    swr = np.column_stack([np.full(n, 1.0), np.zeros(n), np.zeros(n)])
    swr[:, 0] += rng.normal(0, 0.05, n)
    no_swr = np.column_stack([np.full(n, 2.0), np.full(n, -12.0), np.zeros(n)])
    no_swr[:, 0] += rng.normal(0, 0.05, n)
    assert (_lag_one(swr) > _lag_one(no_swr)).all()
    assert (swr[:, 0] < no_swr[:, 0]).all()
    fits = pd.concat(
        [_fits("non_local_swr", swr), _fits("non_local_no_swr", no_swr)],
        ignore_index=True,
    )

    paired = behavior._paired_lag_one(fits)

    assert paired["n"] == n
    assert paired["positive"] == 1.0
    assert paired["animals_positive"] == 5


def test_the_paired_test_drops_a_session_gated_in_either_stream():
    coefs = np.tile([1.0, 0.0, 0.0], (10, 1))
    separated = [""] * 10
    separated[2] = REWARD_HISTORY_COLUMNS[0]
    fits = pd.concat(
        [
            _fits("non_local_swr", coefs + 0.1),
            _fits("non_local_no_swr", coefs, separated),
        ],
        ignore_index=True,
    )

    assert behavior._paired_lag_one(fits)["n"] == 9


def _kernel(log_rr, se, ok):
    return {
        "log_rr": np.asarray(log_rr, dtype=float),
        "se": np.asarray(se, dtype=float),
        "ok": np.asarray(ok, dtype=bool),
    }


def test_the_history_peak_interval_is_widened_for_the_lags_searched():
    # z = 2.2 at the largest of 8 measurable lags: above 1.96, below the
    # Bonferroni threshold for 8 lags (about 2.73).
    log_rr = np.r_[0.22, np.zeros(9)]
    se = np.full(10, 0.1)
    many = _kernel(log_rr, se, [True] * 8 + [False] * 2)
    one = _kernel(log_rr, se, [True] + [False] * 9)

    assert behavior._peak(many)[2] is False
    assert behavior._peak(one)[2] is True
    # A large peak still clears the widened interval.
    assert behavior._peak(
        _kernel(np.r_[0.5, np.zeros(9)], se, [True] * 8 + [False] * 2)
    )[2]


def test_fits_from_the_old_all_bins_design_are_refused(tmp_path):
    """A per-bin fit table without the at-risk design label is not summarized."""
    path = tmp_path / "all_models.pkl"
    _fits("non_local_swr", [[0.3, 0.1, 0.0]]).to_pickle(path)
    with pytest.raises(ValueError, match="refit"):
        behavior._read_per_bin_fits(path)

    _fits("non_local_swr", [[0.3, 0.1, 0.0]]).assign(
        per_bin_design=PER_BIN_DESIGN
    ).to_pickle(path)
    assert len(behavior._read_per_bin_fits(path)) == 1
