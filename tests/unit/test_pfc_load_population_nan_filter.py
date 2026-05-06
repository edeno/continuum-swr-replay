"""Regression test: ``load_population_data`` must drop neurons with
NaN modulation in *any* of the four returned arrays.

The pre-fix filter only checked ``swr_all`` and ``no_swr_all``, so
neurons with NaN entries in ``nl_swr_all`` / ``nl_no_swr_all``
slipped through and downstream consumers saw NaN-laced non-local
modulation traces.

This test writes a tiny synthetic batch to a temp dir and asserts
the returned modulation arrays are NaN-free across all four channels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.load_population import (
    load_population_data,
)


def _required_pval_columns() -> list[str]:
    """Match the FDR-correction loop in load_population_data."""
    fdr_tests = [
        "SWR vs. constant",
        "non_local vs. constant",
        "non_local_no_SWR vs. constant",
        "SWR + non_local_no_SWR vs. SWR",
        "SWR + non_local_no_SWR vs. non_local_no_SWR",
        "non_local_SWR + non_local_no_SWR vs. non_local",
    ]
    return [f"{t}: pval" for t in fdr_tests]


class TestLoadPopulationNanFilter:
    def test_neurons_with_nan_in_non_local_swr_dropped(self, tmp_path):
        n_lag_times = 201
        # Two neurons. Neuron 0: clean across all four arrays.
        # Neuron 1: NaN only in ``nl_swr`` — used to slip past the
        # filter under the pre-fix code.
        swr = np.zeros((n_lag_times, 2))
        no_swr = np.zeros((n_lag_times, 2))
        nl_swr = np.zeros((n_lag_times, 2))
        nl_swr[10, 1] = np.nan
        nl_no_swr = np.zeros((n_lag_times, 2))

        df_rows = pd.DataFrame(
            {
                "nwb_file_name": ["chimi20200212_.nwb"] * 2,
                "immobility_avg_firing_rate": [10.0, 10.0],
            }
        )
        for col in _required_pval_columns():
            df_rows[col] = 0.5  # non-NaN ⇒ ``has_fit`` is True

        # Write the batch in the canonical filename pattern.
        base = tmp_path / "session_mPFC_modulation"
        df_rows.to_pickle(base.with_suffix(".pkl"))
        np.save(base.with_suffix(".SWR_modulation.npy"), swr)
        np.save(base.with_suffix(".no_SWR_modulation.npy"), no_swr)
        np.save(base.with_suffix(".non_local_SWR_modulation.npy"), nl_swr)
        np.save(base.with_suffix(".non_local_no_SWR_modulation.npy"), nl_no_swr)

        df, modulations, _ = load_population_data(tmp_path, min_firing_rate=0.0)
        assert df.shape[0] == 1, (
            f"expected 1 neuron retained, got {df.shape[0]} — neuron with "
            "NaN in non_local_swr modulation slipped past the filter"
        )
        for key in ("swr", "no_swr", "non_local_swr", "non_local_no_swr"):
            assert not np.any(np.isnan(modulations[key])), (
                f"modulations[{key!r}] still has NaN entries; "
                "the NaN filter is not enforced for that channel"
            )
