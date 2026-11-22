"""Tests for the per-time-bin counts by signal level and non-local probability,
on tables small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties import nonlocal_by_signal as nbs
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)


def test_counts_split_each_signal_bin_by_probability_with_0_8_as_an_edge():
    """Signal bins [0, 1), [1, 2), [2, 3); a probability of exactly 0.8 is not
    non-local (the last bin holds those above 0.8), 0 is in the first bin."""
    signal = np.array([0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 1.0])
    probability = np.array([0.9, 0.8, 0.0, 0.95, 0.81, 0.3, 1.0])
    counts = nbs.count_by_signal_and_probability(
        signal, probability, np.array([0.0, 1.0, 2.0, 3.0])
    )
    assert len(counts) == 3 * 5
    by_cell = counts.set_index(["signal_low", "probability_low"])["n_bins"]
    assert by_cell[(0.0, 0.8)] == 1  # 0.9
    assert by_cell[(0.0, 0.6)] == 1  # exactly 0.8
    assert by_cell[(0.0, 0.0)] == 1  # 0
    assert by_cell[(2.0, 0.8)] == 2
    assert by_cell[(2.0, 0.2)] == 1
    assert by_cell[(1.0, 0.8)] == 1  # a signal on an edge goes up
    assert counts["n_bins"].sum() == 7


def test_counts_leave_out_bins_outside_the_edges_and_not_finite():
    counts = nbs.count_by_signal_and_probability(
        np.array([-5.0, 0.5, np.nan, 0.5]),
        np.array([0.9, 0.9, 0.9, np.nan]),
        np.array([0.0, 1.0]),
    )
    assert counts["n_bins"].tolist() == [0, 0, 0, 0, 1]


def test_signal_edges_count_every_level_and_split_at_their_thresholds():
    for edges, threshold in (
        (nbs.RIPPLE_POWER_EDGES, KAY_ZSCORE_THRESHOLD),
        (nbs.MUA_EDGES, nbs.MUA_LEVEL_Z),
    ):
        assert edges[0] == -np.inf and edges[-1] == np.inf
        assert (np.diff(edges) > 0).all()
        assert threshold in edges
        counts = nbs.count_by_signal_and_probability(
            np.array([-7.0, threshold, 40.0]), np.full(3, 0.9), edges
        )
        occupied = counts[counts["n_bins"] > 0]
        assert occupied["signal_low"].tolist() == [-np.inf, threshold, 10.0]
    assert nbs.MUA_LEVEL_Z == 2.0


def _dot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # immobile, moving, immobile in excluded time, immobile, immobile
            "head_speed": [1.0, 20.0, 1.0, 1.0, 1.0],
            "in_excluded_time": [False, False, True, False, False],
            "SWR_zscore": [0.1, 0.1, 0.1, 0.1, 3.1],
            "MUA_zscore": [-0.6, 0.0, 0.0, 2.2, np.nan],
            "Non-Local_Continuous_prob": [0.5, 0.9, 0.9, 0.1, 0.9],
            "Non-Local_Fragmented_prob": [0.4, 0.0, 0.0, 0.1, 0.0],
        }
    )


def test_rest_signal_counts_count_the_immobile_analyzed_bins_of_each_signal():
    counts = nbs.rest_signal_counts(_dot(), np.ones(5, dtype=bool))
    assert list(counts.columns) == [
        "signal", "signal_low", "signal_high", "probability_low",
        "probability_high", "n_bins",
    ]  # fmt: skip
    ripple = nbs.nonlocal_counts(counts, "ripple_power").set_index("signal_low")
    assert len(ripple) == len(nbs.RIPPLE_POWER_EDGES) - 1
    assert ripple["n_bins"].sum() == 3
    assert ripple.loc[0.0, ["n_bins", "n_nonlocal"]].tolist() == [2, 1]
    assert ripple.loc[3.0, ["n_bins", "n_nonlocal"]].tolist() == [1, 1]
    # MUA: the last bin has no MUA and is in no MUA row
    mua = nbs.nonlocal_counts(counts, "mua").set_index("signal_low")
    assert mua["n_bins"].sum() == 2
    assert mua.loc[-0.75, ["n_bins", "n_nonlocal"]].tolist() == [1, 1]
    assert mua.loc[2.0, ["n_bins", "n_nonlocal"]].tolist() == [1, 0]
    # outside every well visit, a bin is not counted
    inside = np.array([True, True, True, True, False])
    counts = nbs.rest_signal_counts(_dot(), inside)
    ripple = nbs.nonlocal_counts(counts, "ripple_power").set_index("signal_low")
    assert ripple["n_bins"].sum() == 2
    assert ripple.loc[3.0, ["n_bins", "n_nonlocal"]].tolist() == [0, 0]
    with pytest.raises(ValueError, match="one bool per bin"):
        nbs.rest_signal_counts(_dot(), inside[:4])


def test_nonlocal_counts_keep_each_epochs_columns_and_refuse_an_unknown_signal():
    epoch = nbs.rest_signal_counts(_dot(), np.ones(5, dtype=bool))
    two = pd.concat(
        [
            epoch.assign(epoch_id="e1", animal="a"),
            epoch.assign(epoch_id="e2", animal="a"),
        ]
    )[["epoch_id", "animal", *epoch.columns]]
    counts = nbs.nonlocal_counts(two, "ripple_power")
    assert list(counts.columns) == [
        "epoch_id", "animal", "signal_low", "signal_high", "n_bins", "n_nonlocal",
    ]  # fmt: skip
    assert counts.groupby("epoch_id")["n_nonlocal"].sum().tolist() == [2, 2]
    with pytest.raises(ValueError, match="no 'theta' rows"):
        nbs.nonlocal_counts(two, "theta")
