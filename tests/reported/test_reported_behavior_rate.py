"""The behavior summary refuses per-bin fits from the old design."""

import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.model_specs import PER_BIN_DESIGN
from continuum_swr_replay.reported import behavior


def test_fits_from_the_old_all_bins_design_are_refused(tmp_path):
    """A per-bin fit table without the at-risk design label is not summarized."""
    path = tmp_path / "all_models.pkl"
    fits = pd.DataFrame({"model": ["M0_const"], "outcome": ["non_local_swr"]})
    fits.to_pickle(path)
    with pytest.raises(ValueError, match="refit"):
        behavior._read_per_bin_fits(path)

    fits.assign(per_bin_design=PER_BIN_DESIGN).to_pickle(path)
    assert len(behavior._read_per_bin_fits(path)) == 1


def test_a_table_with_any_fit_from_another_design_is_refused(tmp_path):
    """One row on the at-risk design does not vouch for the others."""
    path = tmp_path / "all_models.pkl"
    pd.DataFrame(
        {
            "model": ["M0_const", "M0_const"],
            "outcome": ["non_local_swr", "swr"],
            "per_bin_design": [PER_BIN_DESIGN, "all_bins_onset_history"],
        }
    ).to_pickle(path)
    with pytest.raises(ValueError, match="refit"):
        behavior._read_per_bin_fits(path)
