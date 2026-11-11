"""The behavior summary's per-bin rate values: the corrected history peak and the
refusal of fits from the old design, on synthetic fits."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.model_specs import PER_BIN_DESIGN
from continuum_swr_replay.reported import behavior


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
    fits = pd.DataFrame({"model": ["M0_const"], "outcome": ["non_local_swr"]})
    fits.to_pickle(path)
    with pytest.raises(ValueError, match="refit"):
        behavior._read_per_bin_fits(path)

    fits.assign(per_bin_design=PER_BIN_DESIGN).to_pickle(path)
    assert len(behavior._read_per_bin_fits(path)) == 1
