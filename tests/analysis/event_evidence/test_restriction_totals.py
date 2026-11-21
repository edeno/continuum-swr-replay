"""Tests for the non-local restriction contrast over all immobile time outside
SWRs, on tables small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_totals as rt

BIN_S = 0.5


def _sums() -> pd.DataFrame:
    """Animal a's epochs a1 (10 bins, 5 s) and a2 (20 bins, 10 s), animal b's b1
    (4 bins, 2 s). ``full - swr`` outside SWRs: filter 10, 40, -2 (2, 4 and -1
    per second); two-sided 20, 60, 2 (4, 6 and 1 per second). Each epoch's
    ``immobile_in_swr`` row has ``full - swr`` of 100, which must not count."""
    rows = []
    for epoch, animal, n_bins, gain, gain_two_sided in (
        ("a1", "a", 10, 10.0, 20.0),
        ("a2", "a", 20, 40.0, 60.0),
        ("b1", "b", 4, -2.0, 2.0),
    ):
        for category, full, full_two_sided in (
            ("immobile_outside_swr", gain, gain_two_sided),
            ("immobile_in_swr", 100.0, 100.0),
        ):
            rows.append(
                {
                    "category": category,
                    "n_bins": n_bins,
                    "none": -1.0,
                    "swr": -1.0,
                    "full": full - 1.0,
                    "none_two_sided": -2.0,
                    "swr_two_sided": -2.0,
                    "full_two_sided": full_two_sided - 2.0,
                    "epoch_id": epoch,
                    "animal": animal,
                }
            )
    return pd.DataFrame(rows)


def _windows() -> pd.DataFrame:
    """No-SWR event windows with ``full - swr`` (filter; two-sided twice it): a1
    5 and 2, a2 10, b1 none. Left out: an SWR event, a no-SWR control and a
    no-SWR event with a bin in excluded time, each with 100."""
    rows = [
        ("a1", "event", "no SWR", 0, 5.0),
        ("a1", "event", "no SWR", 0, 2.0),
        ("a1", "event", "SWR", 0, 100.0),
        ("a1", "control", "no SWR", 0, 100.0),
        ("a1", "event", "no SWR", 1, 100.0),
        ("a2", "event", "no SWR", 0, 10.0),
        ("b1", "event", "SWR", 0, 100.0),
    ]
    table = pd.DataFrame(
        rows, columns=["epoch_id", "kind", "group", "n_bins_excluded", "gain"]
    )
    return table.assign(
        swr=-3.0,
        full=table["gain"] - 3.0,
        swr_two_sided=-4.0,
        full_two_sided=2 * table["gain"] - 4.0,
    )


def _bin_s() -> pd.Series:
    return pd.Series(BIN_S, index=["a1", "a2", "b1"])


def test_epoch_gains_are_the_categorys_contrast_per_second_and_its_share():
    epochs = rt.epoch_outside_swr_gains(_sums(), _bin_s(), _windows())
    assert list(epochs["epoch_id"]) == ["a1", "a2", "b1"]
    assert list(epochs["animal"]) == ["a", "a", "b"]
    np.testing.assert_allclose(epochs["seconds"], [5.0, 10.0, 2.0])
    np.testing.assert_allclose(epochs["full_minus_swr"], [10.0, 40.0, -2.0])
    np.testing.assert_allclose(epochs["gain_per_s"], [2.0, 4.0, -1.0])
    np.testing.assert_allclose(epochs["gain_per_s_two_sided"], [4.0, 6.0, 1.0])
    np.testing.assert_allclose(epochs["no_swr_events_full_minus_swr"], [7.0, 10.0, 0])
    # b1's total is not positive: no share
    np.testing.assert_allclose(epochs["share_in_no_swr_events"], [0.7, 0.25, np.nan])
    # b1 has no no-SWR event: a share of 0
    np.testing.assert_allclose(
        epochs["share_in_no_swr_events_two_sided"], [0.7, 20 / 60, 0.0]
    )


def test_statistics_are_the_mean_over_animals_of_their_epoch_medians():
    epochs = rt.epoch_outside_swr_gains(_sums(), _bin_s(), _windows())
    stats = rt.outside_swr_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * 2
    key = stats.set_index(["family", "measure"])
    gain = key.loc[("filter", "gain_per_s")]
    # a: median of 2 and 4 = 3; b: -1
    np.testing.assert_allclose(gain["animals"], [3.0, -1.0])
    assert gain["estimate"] == pytest.approx(1.0)
    assert gain["ci_low"] <= gain["estimate"] <= gain["ci_high"]
    assert gain["n_animals"] == 2 and gain["n_animals_positive"] == 1
    assert gain["n_epochs"] == 3 and gain["n_epochs_positive"] == 2
    assert key.loc[("two_sided", "gain_per_s"), "estimate"] == pytest.approx(3.0)
    share = key.loc[("filter", "share_in_no_swr_events")]
    # a: median of 0.7 and 0.25; b has no share
    assert share["estimate"] == pytest.approx(0.475)
    assert share["n_animals"] == 1 and share["n_epochs"] == 2
    share = key.loc[("two_sided", "share_in_no_swr_events")]
    assert share["estimate"] == pytest.approx(((0.7 + 1 / 3) / 2 + 0.0) / 2)
    assert share["n_epochs"] == 3 and share["n_epochs_positive"] == 2


def test_an_epoch_without_a_time_step_or_with_two_outside_rows_raises():
    with pytest.raises(ValueError, match="no time step"):
        rt.epoch_outside_swr_gains(_sums(), _bin_s().drop("a2"), _windows())
    sums = _sums()
    twice = pd.concat([sums, sums.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one immobile_outside_swr"):
        rt.epoch_outside_swr_gains(twice, _bin_s(), _windows())
