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
    5 and 2, a2 10, b1 none. Left out of that share: an SWR event, a no-SWR
    control and a no-SWR event with a bin in excluded time, each with 100.

    Outside SWRs (``*_outside_swr``, two-sided twice the filter score), with
    ``n_out`` bins there: each no-SWR event its whole gain over 2 bins (a2's
    over 4); the SWR events 1 (a1) and 3 (b1) over 2 bins, their time outside
    the SWR; the excluded no-SWR event 0.5 over 1 bin; the control 100 over 4,
    which never counts."""
    rows = [
        ("a1", "event", "no SWR", 0, 5.0, 5.0, 2, 0),
        ("a1", "event", "no SWR", 0, 2.0, 2.0, 2, 10),
        ("a1", "event", "SWR", 0, 100.0, 1.0, 2, 20),
        ("a1", "control", "no SWR", 0, 100.0, 100.0, 4, 25),
        ("a1", "event", "no SWR", 1, 100.0, 0.5, 1, 30),
        ("a2", "event", "no SWR", 0, 10.0, 10.0, 4, 0),
        ("b1", "event", "SWR", 0, 100.0, 3.0, 2, 0),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "epoch_id", "kind", "group", "n_bins_excluded", "gain", "outside",
            "n_bins_outside_swr", "s",
        ],
    )  # fmt: skip
    return table.assign(
        e=table["s"] + 5,
        swr=-3.0,
        full=table["gain"] - 3.0,
        swr_two_sided=-4.0,
        full_two_sided=2 * table["gain"] - 4.0,
        duration_outside_swr_s=table["n_bins_outside_swr"] * BIN_S,
        none_outside_swr=0.0,
        swr_outside_swr=0.0,
        full_outside_swr=table["outside"],
        none_two_sided_outside_swr=0.0,
        swr_two_sided_outside_swr=0.0,
        full_two_sided_outside_swr=2 * table["outside"],
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


def test_epoch_gains_split_the_time_outside_swrs_into_events_and_the_rest():
    epochs = rt.epoch_outside_swr_gains(_sums(), _bin_s(), _windows())
    # the SWR events' time outside the SWR: 1 of a1's 10, none of a2's 40, b1's
    # 3 of a total that is not positive
    np.testing.assert_allclose(
        epochs["swr_events_outside_swr_full_minus_swr"], [1.0, 0.0, 3.0]
    )
    np.testing.assert_allclose(
        epochs["share_in_swr_events_outside_swr"], [0.1, 0.0, np.nan]
    )
    np.testing.assert_allclose(
        epochs["share_in_swr_events_outside_swr_two_sided"], [0.1, 0.0, 3.0]
    )
    # the rest: a1 5 s less 7 event bins (3.5 s), 10 - 8.5 = 1.5 nats (the
    # excluded event's bins are event time too); a2 10 - 2 s, 40 - 10; b1
    # 2 - 1 s, -2 - 3
    np.testing.assert_allclose(epochs["outside_events_seconds"], [1.5, 8.0, 1.0])
    np.testing.assert_allclose(
        epochs["outside_events_full_minus_swr"], [1.5, 30.0, -5.0]
    )
    np.testing.assert_allclose(epochs["outside_events_gain_per_s"], [1.0, 3.75, -5.0])
    # two-sided: 20 - 17, 60 - 20, 2 - 6
    np.testing.assert_allclose(
        epochs["outside_events_gain_per_s_two_sided"], [2.0, 5.0, -4.0]
    )


def test_overlapping_event_windows_or_a_window_table_without_outside_sums_raise():
    windows = _windows()
    windows.loc[1, "s"] = 3  # into a1's first event, 0-5
    with pytest.raises(ValueError, match="overlap"):
        rt.epoch_outside_swr_gains(_sums(), _bin_s(), windows)
    old = _windows().drop(columns=["full_outside_swr"])
    with pytest.raises(ValueError, match="Rerun scripts/event_evidence"):
        rt.epoch_outside_swr_gains(_sums(), _bin_s(), old)


def test_statistics_are_the_mean_over_animals_of_their_epoch_medians():
    epochs = rt.epoch_outside_swr_gains(_sums(), _bin_s(), _windows())
    stats = rt.outside_swr_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * 4
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
    # a: median of 0.1 and 0; b has no share
    flanks = key.loc[("filter", "share_in_swr_events_outside_swr")]
    assert flanks["estimate"] == pytest.approx(0.05) and flanks["n_epochs"] == 2
    # a: median of 1 and 3.75; b: -5
    rest = key.loc[("filter", "outside_events_gain_per_s")]
    np.testing.assert_allclose(rest["animals"], [2.375, -5.0])
    assert rest["n_epochs_positive"] == 2


def _immobile_sums() -> pd.DataFrame:
    """Per epoch and category: ``n_bins``, ``swr - none`` and ``full - none``
    (``none`` 0; two-sided twice the filter scores). Over the immobile bins, at
    0.5 s a bin: a1 10 bins, 6 and 9 (1.2 and 1.8 per s); a2 10 bins, 4 and 10
    (0.8, 2.0); b1 4 bins, 3 and 2 (1.5, 1.0). Moving bins never count."""
    rows = [
        ("a1", "immobile_in_swr", 2, 6.0, 5.0),
        ("a1", "immobile_outside_swr", 8, 0.0, 4.0),
        ("a1", "moving", 10, 100.0, 300.0),
        ("a2", "immobile_in_swr", 4, 4.0, 4.0),
        ("a2", "immobile_outside_swr", 6, 0.0, 6.0),
        ("b1", "immobile_in_swr", 2, 3.0, 2.0),
        ("b1", "immobile_outside_swr", 2, 0.0, 0.0),
    ]
    table = pd.DataFrame(
        rows, columns=["epoch_id", "category", "n_bins", "swr", "full"]
    )
    return table.assign(
        animal=table["epoch_id"].str[0],
        none=0.0,
        none_two_sided=0.0,
        swr_two_sided=2 * table["swr"],
        full_two_sided=2 * table["full"],
    )


def test_immobile_gains_are_each_decoders_gain_over_local_only_per_second():
    epochs = rt.epoch_immobile_gains(_immobile_sums(), _bin_s())
    assert list(epochs["epoch_id"]) == ["a1", "a2", "b1"]
    np.testing.assert_allclose(epochs["seconds"], [5.0, 5.0, 2.0])
    np.testing.assert_allclose(epochs["swr_minus_none_per_s"], [1.2, 0.8, 1.5])
    np.testing.assert_allclose(epochs["full_minus_none_per_s"], [1.8, 2.0, 1.0])
    np.testing.assert_allclose(epochs["full_minus_swr_per_s"], [0.6, 1.2, -0.5])
    np.testing.assert_allclose(
        epochs["full_minus_none_per_s_two_sided"], [3.6, 4.0, 2.0]
    )
    np.testing.assert_allclose(epochs["share_captured_by_swr_only"], [2 / 3, 0.4, 1.5])
    assert epochs["full_above_swr_only"].tolist() == [True, True, False]


def test_immobile_statistics_count_the_epochs_with_full_above_swr_only():
    epochs = rt.epoch_immobile_gains(_immobile_sums(), _bin_s())
    stats = rt.immobile_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * 4
    key = stats.set_index(["family", "measure"])
    swr = key.loc[("filter", "swr_minus_none_per_s")]
    np.testing.assert_allclose(swr["animals"], [1.0, 1.5])  # a: median 1.2, 0.8
    assert swr["estimate"] == pytest.approx(1.25)
    assert swr["ci_low"] <= swr["estimate"] <= swr["ci_high"]
    full = key.loc[("filter", "full_minus_none_per_s")]
    assert full["estimate"] == pytest.approx((1.9 + 1.0) / 2)
    difference = key.loc[("filter", "full_minus_swr_per_s")]
    assert difference["estimate"] == pytest.approx((0.9 - 0.5) / 2)
    assert difference["n_animals_positive"] == 1
    assert difference["n_epochs_positive"] == 2
    share = key.loc[("two_sided", "share_captured_by_swr_only")]
    assert share["estimate"] == pytest.approx(((2 / 3 + 0.4) / 2 + 1.5) / 2)
    assert (stats["n_epochs_full_above_swr_only"] == 2).all()


def test_an_epoch_without_a_time_step_or_with_two_outside_rows_raises():
    with pytest.raises(ValueError, match="no time step"):
        rt.epoch_outside_swr_gains(_sums(), _bin_s().drop("a2"), _windows())
    sums = _sums()
    twice = pd.concat([sums, sums.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one immobile_outside_swr"):
        rt.epoch_outside_swr_gains(twice, _bin_s(), _windows())
