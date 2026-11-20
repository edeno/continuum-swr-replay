"""Tests for the non-local restriction scores over panel h's windows, on tables
small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    SCORE_COLUMNS,
)

N = 10
STEP = 0.002


def _bins() -> pd.DataFrame:
    """Ten bins: ``none`` = 0, ``swr`` = bin index, ``full`` = 2 x bin index;
    the two-sided scores are the filter scores plus 1. Bins 2-3 inside an SWR,
    bin 6 moving, bin 9 excluded."""
    index = np.arange(N, dtype=np.float32)
    category = ["immobile_outside_swr"] * N
    category[2] = category[3] = "immobile_in_swr"
    category[6] = "moving"
    category[9] = "excluded"
    return pd.DataFrame(
        {
            "time": 100.0 + STEP * np.arange(N),
            "none": np.zeros(N, dtype=np.float32),
            "swr": index,
            "full": 2 * index,
            "none_two_sided": np.ones(N, dtype=np.float32),
            "swr_two_sided": index + 1,
            "full_two_sided": 2 * index + 1,
            "in_swr": np.isin(np.arange(N), [2, 3]),
            "category": pd.Categorical(category),
        }
    )


def _windows() -> pd.DataFrame:
    """Windows 0-1 (an event), 1-3 (overlapping it, inside the SWR), 5-6 (a
    moving bin) and 8-9 (an excluded bin)."""
    return pd.DataFrame(
        {
            "window_id": [0, 1, 2, 3],
            "event_id": ["a", "b", "a", "b"],
            "kind": ["event", "event", "control", "control"],
            "group": ["no SWR", "SWR", "no SWR", "SWR"],
            "s": [0, 1, 5, 8],
            "e": [1, 3, 6, 9],
        }
    )


def test_window_sums_count_and_sum_each_windows_bins():
    sums = rw.restriction_window_sums(_bins(), _windows())
    np.testing.assert_array_equal(sums["n_bins"], [2, 3, 2, 2])
    np.testing.assert_array_equal(sums["n_bins_in_swr"], [0, 2, 0, 0])
    np.testing.assert_array_equal(sums["n_bins_immobile"], [2, 3, 1, 1])
    np.testing.assert_array_equal(sums["n_bins_excluded"], [0, 0, 0, 1])
    np.testing.assert_array_equal(sums["n_bins_not_finite"], [0, 0, 0, 0])
    np.testing.assert_allclose(sums["duration_s"], np.array([2, 3, 2, 2]) * STEP)
    # swr sums the bin indices: 0+1, 1+2+3, 5+6, 8+9
    np.testing.assert_allclose(sums["swr"], [1, 6, 11, 17])
    np.testing.assert_allclose(sums["full"], [2, 12, 22, 34])
    np.testing.assert_allclose(sums["none"], 0)
    np.testing.assert_allclose(sums["full_two_sided"], [4, 15, 24, 36])
    assert all(sums[c].dtype == np.float64 for c in SCORE_COLUMNS)
    assert sums[list(rw.WINDOW_KEYS)].equals(_windows()[list(rw.WINDOW_KEYS)])


def test_a_score_that_is_not_finite_makes_its_windows_sum_nan_and_is_counted():
    bins = _bins()
    bins.loc[1, "full"] = np.nan
    sums = rw.restriction_window_sums(bins, _windows())
    np.testing.assert_array_equal(sums["n_bins_not_finite"], [1, 1, 0, 0])
    assert sums["full"].isna().tolist() == [True, True, False, False]
    # the other columns of those windows are kept
    np.testing.assert_allclose(sums["swr"], [1, 6, 11, 17])
    np.testing.assert_allclose(sums.loc[2:, "full"], [22, 34])


def test_a_window_outside_the_bins_raises():
    windows = _windows()
    windows.loc[3, "e"] = N
    with pytest.raises(ValueError, match="outside"):
        rw.restriction_window_sums(_bins(), windows)


def _holdout(windows: pd.DataFrame) -> pd.DataFrame:
    """Two tetrodes per window: 1 and 2 spikes in window 0, 0 and 0 in window 1,
    3 and 0 in 2, 1 and 1 in 3."""
    rows = windows.loc[np.repeat(windows.index, 2), list(rw.WINDOW_KEYS)]
    rows = rows.assign(
        tetrode=np.tile([0, 1], len(windows)),
        n_bins=np.repeat((windows["e"] - windows["s"] + 1).to_numpy(), 2),
        n_spikes=[1, 2, 0, 0, 3, 0, 1, 1],
    )
    return rows.iloc[::-1].reset_index(drop=True)


def test_spike_counts_sum_the_tetrodes_of_each_window_in_window_order():
    spikes = rw.window_spike_counts(_holdout(_windows()), _windows())
    assert spikes.tolist() == [3, 0, 3, 2]
    assert spikes.index.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "change",
    [
        lambda h: h[h["window_id"] != 2],  # a window missing
        lambda h: h.assign(group=np.where(h["window_id"] == 1, "no SWR", h["group"])),
        lambda h: h.assign(n_bins=h["n_bins"] + (h["window_id"] == 0)),  # length
    ],
)
def test_windows_that_are_not_the_holdout_runs_raise(change):
    with pytest.raises(ValueError, match="not the held-out"):
        rw.window_spike_counts(change(_holdout(_windows())), _windows())


def test_a_window_without_one_of_the_tetrodes_rows_raises():
    """Its spike count would leave that tetrode's spikes out."""
    holdout = _holdout(_windows())
    missing = holdout[~((holdout["window_id"] == 2) & (holdout["tetrode"] == 0))]
    with pytest.raises(ValueError, match="one row per tetrode"):
        rw.window_spike_counts(missing, _windows())
    twice = pd.concat([holdout, holdout.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per tetrode"):
        rw.window_spike_counts(twice, _windows())


def _scored() -> pd.DataFrame:
    sums = rw.restriction_window_sums(_bins(), _windows())
    return sums.assign(n_spikes=[3, 0, 3, 2])


def test_contrasts_per_spike_and_per_second():
    out = rw.window_contrasts(_scored())
    # full - swr filter: 1, 6, 11, (17, but excluded)
    np.testing.assert_allclose(
        out["full_minus_swr_per_spike"], [1 / 3, np.nan, 11 / 3, np.nan]
    )
    np.testing.assert_allclose(
        out["full_minus_swr_per_s"],
        [1 / (2 * STEP), 6 / (3 * STEP), 11 / (2 * STEP), np.nan],
    )
    # swr - none two-sided: (1+2) - 2 = 1, (2+3+4) - 3 = 6, (6+7) - 2 = 11
    np.testing.assert_allclose(
        out["swr_minus_none_two_sided_per_spike"], [1 / 3, np.nan, 11 / 3, np.nan]
    )
    np.testing.assert_allclose(
        out["full_minus_none_per_s"],
        [2 / (2 * STEP), 12 / (3 * STEP), 22 / (2 * STEP), np.nan],
    )


def _cohort() -> pd.DataFrame:
    """Two animals; animal a has two epochs. Event windows of the "no SWR"
    group with ``full - swr`` (filter) per spike of: a/e1 1 and 3 (median 2),
    a/e2 4 (median 4), b/e3 10 and 20 and 30 (median 20); one "SWR" event and
    one control per epoch with ``full - swr`` per spike 1."""
    rows = []
    for animal, epoch, gains in (
        ("a", "e1", [1, 3]),
        ("a", "e2", [4]),
        ("b", "e3", [10, 20, 30]),
    ):
        for gain in gains:
            rows.append((animal, epoch, "event", "no SWR", gain))
        rows.append((animal, epoch, "event", "SWR", 1))
        rows.append((animal, epoch, "control", "no SWR", 1))
    table = pd.DataFrame(rows, columns=["animal", "epoch_id", "kind", "group", "gain"])
    n = len(table)
    # one spike per window and one bin of 1 s; full - swr = gain, swr - none = 0
    zeros = np.zeros(n)
    return table.assign(
        n_spikes=1,
        n_bins_excluded=0,
        duration_s=1.0,
        none=zeros,
        swr=zeros,
        full=table["gain"].to_numpy(float),
        none_two_sided=zeros,
        swr_two_sided=zeros,
        full_two_sided=2 * table["gain"].to_numpy(float),
    )


def test_statistics_are_the_per_animal_median_of_epoch_medians_and_their_mean():
    stats = rw.restriction_window_statistics(_cohort(), n_boot=200, seed=1)
    # 2 families x 3 contrasts x 2 units, events (2 groups) and controls (1)
    assert len(stats) == 2 * 3 * 2 * 3
    assert list(stats.columns[:5]) == ["family", "contrast", "unit", "kind", "group"]
    key = stats.set_index(["family", "contrast", "unit", "kind", "group"])
    row = key.loc[("filter", "full_minus_swr", "per_spike", "event", "no SWR")]
    # a: median of epoch medians 2 and 4 = 3; b: 20
    np.testing.assert_allclose(row["animals"], [3.0, 20.0])
    assert list(row["animal_names"]) == ["a", "b"]
    assert row["estimate"] == pytest.approx(11.5)
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
    assert row["n_events"] == 6 and row["n_scored"] == 6 and row["n_animals"] == 2
    two_sided = key.loc[("two_sided", "full_minus_swr", "per_s", "event", "no SWR")]
    assert two_sided["estimate"] == pytest.approx(23.0)
    control = key.loc[("filter", "full_minus_swr", "per_spike", "control", "no SWR")]
    assert control["estimate"] == pytest.approx(1.0) and control["n_events"] == 3
    null = key.loc[("filter", "swr_minus_none", "per_s", "event", "SWR")]
    assert null["estimate"] == pytest.approx(0.0)


def test_statistics_leave_out_and_count_windows_with_excluded_bins():
    cohort = _cohort()
    cohort.loc[cohort["gain"] == 30, "n_bins_excluded"] = 1
    stats = rw.restriction_window_statistics(cohort, n_boot=50)
    key = stats.set_index(["family", "contrast", "unit", "kind", "group"])
    row = key.loc[("filter", "full_minus_swr", "per_spike", "event", "no SWR")]
    # b: median of 10 and 20 = 15
    np.testing.assert_allclose(row["animals"], [3.0, 15.0])
    assert row["n_events"] == 6 and row["n_scored"] == 5
