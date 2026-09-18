"""``load_population_data`` must drop neurons with NaN modulation in *any* of
the four shipped arrays, and must keep each session's rows paired with its own
modulation columns.

The pre-fix filter only checked ``swr_all`` and ``no_swr_all``, so neurons with
NaN entries in ``nl_swr_all`` / ``nl_no_swr_all`` slipped through and
downstream consumers saw NaN-laced non-local modulation traces.

The fifth array, the matched non-event control, is deliberately *not* part of
that drop rule: it is the baseline the shipped curves are read against, and
gating on it would change which units the population holds. It is loaded and
sliced with them, and may carry NaN.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.glm import MATCHED_CONTROL_PLACEMENT_METHOD
from continuum_swr_replay.analysis.pfc.heldout import (
    CONTROL_NULL_EVENT_POLICY,
    DEFAULT_CONTROL_DRAWS,
)
from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    HELDOUT_USABLE_DRAW_COLUMNS,
    LEGACY_FDR_TESTS,
    MATCHED_NULL_OBSERVED_COLUMNS,
    load_population_data,
    significant_fraction,
)

N_LAG_TIMES = 201

#: The five arrays a bundle writes, in the order the loader reads them.
MODULATION_ARRAYS = (
    "SWR_modulation",
    "no_SWR_modulation",
    "non_local_SWR_modulation",
    "non_local_no_SWR_modulation",
    "matched_control_modulation",
)


def heldout_statistics(n_neurons: int) -> dict[str, list[float]]:
    """The out-of-fold columns a current bundle carries, all well posed."""
    columns: dict[str, list[float]] = {
        "SWR: heldout_deviance": [100.0] * n_neurons,
        "min_heldout_model": ["SWR"] * n_neurons,
    }
    for column in HELDOUT_P_COLUMNS.values():
        columns[column] = [0.025] * n_neurons
    for column in HELDOUT_USABLE_DRAW_COLUMNS.values():
        columns[column] = [DEFAULT_CONTROL_DRAWS] * n_neurons
    for column in MATCHED_NULL_OBSERVED_COLUMNS.values():
        columns[column] = [1.0] * n_neurons
    return columns


def write_bundle(
    base,
    nwb_files: list[str],
    n_mod_cols: int,
    *,
    swr_marker: float = 0.0,
    statistics: dict[str, list[float]] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
    provenance_overrides: dict[str, object] | None = None,
) -> None:
    """Write one session's pickle and its five modulation arrays."""
    frame = pd.DataFrame(
        {
            "nwb_file_name": nwb_files,
            "immobility_avg_firing_rate": [10.0] * len(nwb_files),
        }
    )
    for column, values in (
        statistics if statistics is not None else heldout_statistics(len(nwb_files))
    ).items():
        frame[column] = values
    frame.to_pickle(base.with_suffix(".pkl"))
    provenance = {
        "estimator": "ridge_poisson_glm",
        "inference": "blocked_heldout_deviance",
        "empirical_pvalue_method": "plus_one",
        "control_null_event_policy": CONTROL_NULL_EVENT_POLICY,
        "penalty_sigma_log_rate": 3.0,
        "n_folds": 5,
        "n_control_draws": DEFAULT_CONTROL_DRAWS,
        "min_usable_null_draws": DEFAULT_CONTROL_DRAWS,
        "matched_control_placement_method": MATCHED_CONTROL_PLACEMENT_METHOD,
        "null_usable_draws": {
            name: [DEFAULT_CONTROL_DRAWS] * len(nwb_files)
            for name in HELDOUT_USABLE_DRAW_COLUMNS
        },
        **(provenance_overrides or {}),
    }
    base.with_suffix(".provenance.json").write_text(json.dumps(provenance))

    default = {
        "SWR_modulation": np.full((N_LAG_TIMES, n_mod_cols), swr_marker, dtype=float),
    }
    for name in MODULATION_ARRAYS:
        values = (arrays or {}).get(
            name, default.get(name, np.zeros((N_LAG_TIMES, n_mod_cols)))
        )
        np.save(base.with_suffix(f".{name}.npy"), values)


class TestLoadPopulationNanFilter:
    def test_nan_in_a_secondary_pvalue_stays_untested(self, tmp_path):
        """A unit tested for one class and not another stays untested there.

        A unit whose session had no events of some class carries NaN for that
        class's empirical p-value while its primary column is finite, so it
        passes the population gate. The raw-p preview must leave that class out
        of its denominator rather than count the unit as a negative.
        """
        statistics = heldout_statistics(1)
        statistics[HELDOUT_P_COLUMNS["non_local_no_SWR"]] = [np.nan]
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            statistics=statistics,
        )

        df, _, _ = load_population_data(tmp_path, min_firing_rate=0.0)

        assert len(df) == 1
        assert pd.isna(df[HELDOUT_P_COLUMNS["non_local_no_SWR"]].iloc[0])
        assert significant_fraction(df)["non_local_no_SWR"] == (0, 0)
        assert significant_fraction(df)["SWR"] == (1, 1)
        # No BH column is created: a 1/1001 Monte Carlo grid cannot support
        # unit-wise correction over thousands of dependent units, so the raw
        # p-value is all the loader ships.
        assert f"{HELDOUT_P_COLUMNS['SWR']}: heldout_q" not in df

    def test_a_session_whose_every_fit_failed_does_not_keyerror(self, tmp_path):
        """A fallback bundle writes only firing rates and all-NaN curves."""
        nan_mod = np.full((N_LAG_TIMES, 1), np.nan)
        frame = pd.DataFrame(
            {
                "nwb_file_name": ["chimi20200212_.nwb"],
                "immobility_avg_firing_rate": [10.0],
                "run_avg_firing_rate": [5.0],
            }
        )
        base = tmp_path / "session_mPFC_modulation"
        frame.to_pickle(base.with_suffix(".pkl"))
        for name in MODULATION_ARRAYS:
            np.save(base.with_suffix(f".{name}.npy"), nan_mod)

        # The bundle carries none of the held-out columns, so it is refused
        # as a bundle of this estimator rather than silently returning a
        # population with no significance at all.
        with pytest.raises(ValueError, match="missing held-out column"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_missing_primary_fit_statistic_does_not_gate_a_finite_curve(self, tmp_path):
        """Inferential availability must not define a descriptive population."""
        statistics = heldout_statistics(2)
        statistics["SWR: heldout_deviance"] = [100.0, np.nan]
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 2,
            2,
            statistics=statistics,
        )

        df, modulations, _ = load_population_data(tmp_path)

        assert len(df) == 2
        assert df["SWR: heldout_deviance"].isna().sum() == 1
        assert modulations["swr"].shape[1] == 2

    def test_low_firing_rate_does_not_gate_the_default_population(self, tmp_path):
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 2,
            2,
        )
        frame_path = tmp_path / "session_mPFC_modulation.pkl"
        frame = pd.read_pickle(frame_path)
        frame["immobility_avg_firing_rate"] = [0.0, 10.0]
        frame.to_pickle(frame_path)

        descriptive, modulations, _ = load_population_data(tmp_path)
        sensitivity, _, _ = load_population_data(tmp_path, min_firing_rate=1.0)

        assert len(descriptive) == 2
        assert modulations["swr"].shape[1] == 2
        assert len(sensitivity) == 1

    def test_the_rate_screen_counts_what_it_dropped_and_what_had_no_rate(
        self, tmp_path, caplog
    ):
        """A NaN rate fails the comparison silently; the count must be logged."""
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 3,
            3,
        )
        frame_path = tmp_path / "session_mPFC_modulation.pkl"
        frame = pd.read_pickle(frame_path)
        frame["immobility_avg_firing_rate"] = [0.0, np.nan, 10.0]
        frame.to_pickle(frame_path)

        with caplog.at_level(
            logging.INFO, logger="continuum_swr_replay.analysis.pfc.load_population"
        ):
            sensitivity, _, _ = load_population_data(tmp_path, min_firing_rate=1.0)

        assert len(sensitivity) == 1
        assert "n_dropped_by_rate=2" in caplog.text
        assert "n_nan_rate=1" in caplog.text

    def test_infinite_primary_curve_value_is_not_descriptive_data(self, tmp_path):
        swr = np.zeros((N_LAG_TIMES, 2))
        swr[10, 1] = np.inf
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 2,
            2,
            arrays={"SWR_modulation": swr},
        )

        df, modulations, _ = load_population_data(tmp_path)

        assert len(df) == 1
        assert np.isfinite(modulations["swr"]).all()

    def test_a_unit_with_nan_in_the_replay_split_curve_is_dropped(self, tmp_path):
        nl_swr = np.zeros((N_LAG_TIMES, 2))
        nl_swr[10, 1] = np.nan
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 2,
            2,
            arrays={"non_local_SWR_modulation": nl_swr},
        )

        df, modulations, _ = load_population_data(tmp_path, min_firing_rate=0.0)

        assert df.shape[0] == 1, (
            "the unit with NaN in its non_local_SWR curve slipped past the filter"
        )
        for key in ("swr", "no_swr", "non_local_swr", "non_local_no_swr"):
            assert not np.any(np.isnan(modulations[key]))

    def test_the_matched_control_curve_is_loaded_and_does_not_gate(self, tmp_path):
        """The baseline curve travels with the population, NaN and all.

        A unit whose engagement model was not estimable has a NaN control
        curve; dropping it would change the population the shipped curves
        describe, so the curve is returned as it is and reduced with
        ``nanmean``.
        """
        matched = np.zeros((N_LAG_TIMES, 2))
        matched[:, 1] = np.nan
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"] * 2,
            2,
            arrays={"matched_control_modulation": matched},
        )

        df, modulations, _ = load_population_data(tmp_path, min_firing_rate=0.0)

        assert len(df) == 2
        assert modulations["matched_control"].shape == (N_LAG_TIMES, 2)
        assert np.isnan(modulations["matched_control"][:, 1]).all()


class TestBundleGeneration:
    """The loader must not read one estimator's bundles as the other's."""

    def test_a_likelihood_ratio_bundle_is_refused(self, tmp_path):
        statistics = heldout_statistics(1)
        statistics["SWR vs. constant: pval"] = [0.5]
        statistics["min_aic_model"] = ["SWR"]
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            statistics=statistics,
        )

        with pytest.raises(ValueError, match="earlier estimator"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_the_superseded_bundles_can_still_be_read_on_purpose(self, tmp_path):
        statistics = {f"{test}: pval": [0.5] for test in LEGACY_FDR_TESTS}
        statistics["min_aic_model"] = ["SWR"]
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            statistics=statistics,
        )

        df, _, _ = load_population_data(
            tmp_path, min_firing_rate=0.0, inference="legacy"
        )

        assert pd.notna(df["SWR vs. constant: fdr_q"].iloc[0])

    def test_an_out_of_fold_bundle_is_not_read_as_a_legacy_one(self, tmp_path):
        write_bundle(tmp_path / "session_mPFC_modulation", ["chimi20200212_.nwb"], 1)

        with pytest.raises(ValueError, match="scored out of fold"):
            load_population_data(tmp_path, min_firing_rate=0.0, inference="legacy")

    def test_a_pre_review_empirical_pvalue_bundle_is_refused(self, tmp_path):
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            provenance_overrides={"empirical_pvalue_method": "raw_exceedance"},
        )

        with pytest.raises(ValueError, match="incompatible held-out provenance"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_an_all_event_observed_null_bundle_is_refused(self, tmp_path):
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            provenance_overrides={
                "control_null_event_policy": "all_events_even_if_null_is_thinner"
            },
        )

        with pytest.raises(ValueError, match="incompatible held-out provenance"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_a_disjoint_window_null_bundle_is_refused(self, tmp_path):
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            provenance_overrides={
                "matched_control_placement_method": "disjoint_control_windows"
            },
        )

        with pytest.raises(ValueError, match="incompatible held-out provenance"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_a_heldout_bundle_without_provenance_is_refused(self, tmp_path):
        base = tmp_path / "session_mPFC_modulation"
        write_bundle(base, ["chimi20200212_.nwb"], 1)
        base.with_suffix(".provenance.json").unlink()

        with pytest.raises(ValueError, match="no readable provenance sidecar"):
            load_population_data(tmp_path, min_firing_rate=0.0)


class TestLoadPopulationCrossSessionAlignment:
    """Per-session shape validation must prevent cross-session misalignment.

    Pre-fix the loader only checked array row count (n_lag_times) and appended
    each session's pickle and arrays independently, then ``pd.concat`` +
    ``np.hstack`` assembled them. If a single session had
    ``len(pickle) != n_modulation_columns``, every later session's neurons
    silently shifted along the modulation axis.
    """

    def test_per_session_row_column_mismatch_raises(self, tmp_path):
        write_bundle(
            tmp_path / "sessA_mPFC_modulation",
            ["chimi20200212_.nwb", "chimi20200212_.nwb"],
            n_mod_cols=1,
            swr_marker=10.0,
        )
        write_bundle(
            tmp_path / "sessB_mPFC_modulation",
            ["j1620210701_.nwb"],
            n_mod_cols=2,
            swr_marker=21.0,
        )

        with pytest.raises(ValueError, match="Per-session shape mismatch"):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_consistent_per_session_shapes_pair_rows_with_columns(self, tmp_path):
        write_bundle(
            tmp_path / "sessA_mPFC_modulation",
            ["chimi20200212_.nwb", "chimi20200212_.nwb"],
            n_mod_cols=2,
            swr_marker=10.0,
        )
        write_bundle(
            tmp_path / "sessB_mPFC_modulation",
            ["j1620210701_.nwb"],
            n_mod_cols=1,
            swr_marker=21.0,
        )

        df, mods, _ = load_population_data(tmp_path, min_firing_rate=0.0)

        assert len(df) == 3
        assert mods["swr"].shape[1] == 3
        assert df["animal"].tolist() == ["chimi", "chimi", "j16"]
        np.testing.assert_array_equal(mods["swr"][0, :], [10.0, 10.0, 21.0])


class TestTheControlCurveFallbackIsLegacyOnly:
    """A missing control array is a vintage on the legacy path and a fault here.

    The matched non-event control postdates the first bundles, so a tree
    written before it legitimately has no such array and an audit reads one on
    purpose. A *current* bundle always writes it, so a missing one there is a
    truncated write, and filling it with NaN would hide that.
    """

    def test_a_legacy_tree_without_the_array_reads_it_as_absent(self, tmp_path):
        statistics = {f"{test}: pval": [0.5] for test in LEGACY_FDR_TESTS}
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            statistics=statistics,
        )
        (tmp_path / "session_mPFC_modulation.matched_control_modulation.npy").unlink()

        _, modulations, _ = load_population_data(
            tmp_path, min_firing_rate=0.0, inference="legacy"
        )

        assert np.isnan(modulations["matched_control"]).all()

    def test_a_current_bundle_without_the_array_is_a_fault(self, tmp_path):
        write_bundle(tmp_path / "session_mPFC_modulation", ["chimi20200212_.nwb"], 1)
        (tmp_path / "session_mPFC_modulation.matched_control_modulation.npy").unlink()

        with pytest.raises(FileNotFoundError):
            load_population_data(tmp_path, min_firing_rate=0.0)

    def test_a_tree_of_neither_generation_is_refused_on_the_legacy_path(self, tmp_path):
        # Neither the held-out columns nor the likelihood-ratio ones.
        write_bundle(
            tmp_path / "session_mPFC_modulation",
            ["chimi20200212_.nwb"],
            1,
            statistics={"run_avg_firing_rate": [5.0]},
        )

        with pytest.raises(ValueError, match="neither generation"):
            load_population_data(tmp_path, min_firing_rate=0.0, inference="legacy")
