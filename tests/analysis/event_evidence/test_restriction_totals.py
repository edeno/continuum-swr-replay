"""Tests for the non-local restriction contrasts over all immobile time and
outside SWRs, inside a well visit or not, on tables small enough to check by
hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_totals as rt
from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    SCORE_COLUMNS,
)

BIN_S = 0.5


def _time_sums() -> pd.DataFrame:
    """Rows of (epoch, category, in visit, in an event window): bins, spikes,
    ``swr - none`` and ``full - none`` (``none`` 0; two-sided twice the filter
    scores).

    Outside SWRs (``swr`` 0, so ``full - swr`` is ``full``), inside a visit:
    a1 10 bins, 6 spikes, 10 nats, of which the rest (no event window) 6 bins,
    2 spikes, 4 nats; a2 10 bins, 10 spikes, 10 nats, rest 8, 8, 8; b1 4 bins,
    2 spikes, -2 nats, all rest. a1 also has 10 bins, 10 spikes and 100 nats
    outside a visit. In SWRs, inside a visit: a1 2 bins, 2 spikes, swr 6, full
    7; b1 2, 2, 3, 2; a1 outside a visit 2, 2, 50, 50. Moving bins never
    count."""
    rows = [
        ("a1", "immobile_outside_swr", True, True, 4, 4, 0.0, 6.0),
        ("a1", "immobile_outside_swr", True, False, 6, 2, 0.0, 4.0),
        ("a1", "immobile_outside_swr", False, False, 10, 10, 0.0, 100.0),
        ("a1", "immobile_in_swr", True, True, 2, 2, 6.0, 7.0),
        ("a1", "immobile_in_swr", False, False, 2, 2, 50.0, 50.0),
        ("a1", "moving", True, False, 10, 10, 100.0, 300.0),
        ("a2", "immobile_outside_swr", True, True, 2, 2, 0.0, 2.0),
        ("a2", "immobile_outside_swr", True, False, 8, 8, 0.0, 8.0),
        ("b1", "immobile_outside_swr", True, False, 4, 2, 0.0, -2.0),
        ("b1", "immobile_in_swr", True, True, 2, 2, 3.0, 2.0),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "epoch_id", "category", "in_well_visit", "in_event_window", "n_bins",
            "n_spikes", "swr", "full",
        ],
    )  # fmt: skip
    return table.assign(
        animal=table["epoch_id"].str[0],
        none=0.0,
        none_two_sided=0.0,
        swr_two_sided=2 * table["swr"],
        full_two_sided=2 * table["full"],
    )


def _windows() -> pd.DataFrame:
    """Event windows with ``full - swr`` over their bins outside SWRs inside a
    visit (``in_visit``) and over all of them (``outside``; ``swr`` 0,
    two-sided twice the filter score): a1's no-SWR event 3 and 3.5, its SWR
    event 2 and 2; a2's no-SWR event 1 and 1; b1 none. Never counted: a1's
    no-SWR control and its no-SWR event with a bin in excluded time, 100 each."""
    rows = [
        ("a1", "event", "no SWR", 0, 3.0, 3.5),
        ("a1", "event", "SWR", 0, 2.0, 2.0),
        ("a1", "control", "no SWR", 0, 100.0, 100.0),
        ("a1", "event", "no SWR", 1, 100.0, 100.0),
        ("a2", "event", "no SWR", 0, 1.0, 1.0),
    ]
    table = pd.DataFrame(
        rows,
        columns=["epoch_id", "kind", "group", "n_bins_excluded", "in_visit", "outside"],
    )
    columns = dict.fromkeys((*rw.OUTSIDE_SWR_COLUMNS, *rw.BIN_FACT_COLUMNS), 0.0)
    table = table.assign(**columns)
    for suffix, scale in (("", 1.0), ("_two_sided", 2.0)):
        table[f"full{suffix}_outside_swr"] = scale * table["outside"]
        table[f"full{suffix}{rw.IN_VISIT_SUFFIX}"] = scale * table["in_visit"]
    return table


def _bin_s() -> pd.Series:
    return pd.Series(BIN_S, index=["a1", "a2", "b1"])


def test_time_sums_partition_the_bins_by_category_visit_and_event_window():
    """Six bins: an event window over bins 1-2, a control over 4-5 (it marks
    nothing); bins 0-2 inside a visit."""
    category = ["immobile_outside_swr"] * 3 + ["immobile_in_swr", "moving", "excluded"]
    bins = pd.DataFrame(
        {
            "time": np.arange(6.0),
            "category": pd.Categorical(category),
            **{c: np.arange(6, dtype=np.float32) for c in SCORE_COLUMNS},
        }
    )
    facts = pd.DataFrame(
        {
            "time": np.arange(6.0),
            "n_spikes": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
            "in_well_visit": np.arange(6) < 3,
        }
    )
    windows = pd.DataFrame({"kind": ["event", "control"], "s": [1, 4], "e": [2, 5]})
    sums = rt.restriction_time_sums(bins, facts, windows)
    assert list(sums.columns[:5]) == [*rt.TIME_SUM_KEYS, "n_bins", "n_spikes"]
    assert len(sums) == 16 and sums["n_bins"].sum() == 6
    key = sums.set_index(list(rt.TIME_SUM_KEYS))
    inside = key.loc[("immobile_outside_swr", True, True)]
    assert inside["n_bins"] == 2 and inside["n_spikes"] == 5
    assert inside["full"] == pytest.approx(3.0)
    rest = key.loc[("immobile_outside_swr", True, False)]
    assert rest["n_bins"] == 1 and rest["n_spikes"] == 1 and rest["swr"] == 0.0
    assert key.loc[("moving", False, False), "n_spikes"] == 5
    assert key.loc[("excluded", False, False), "none_two_sided"] == 5.0
    assert key.loc[("immobile_in_swr", True, True), "n_bins"] == 0
    assert sums["full"].dtype == np.float64 and sums["n_bins"].dtype == np.int64
    with pytest.raises(ValueError, match="time"):
        rt.restriction_time_sums(bins, facts.assign(time=facts["time"] + 1), windows)


def test_outside_swr_gains_inside_a_visit_per_spike_and_per_second():
    epochs = rt.epoch_outside_swr_gains(_time_sums(), _bin_s(), _windows())
    assert list(epochs["epoch_id"]) == ["a1", "a2", "b1"]
    assert list(epochs["animal"]) == ["a", "a", "b"]
    np.testing.assert_array_equal(epochs["n_spikes"], [6, 10, 2])
    np.testing.assert_allclose(epochs["seconds"], [5.0, 5.0, 2.0])
    np.testing.assert_allclose(epochs["full_minus_swr"], [10.0, 10.0, -2.0])
    np.testing.assert_allclose(epochs["gain_per_spike"], [10 / 6, 1.0, -1.0])
    np.testing.assert_allclose(epochs["gain_per_s"], [2.0, 2.0, -1.0])
    np.testing.assert_allclose(epochs["gain_per_spike_two_sided"], [20 / 6, 2.0, -2.0])
    # the rest, outside every event window
    np.testing.assert_allclose(
        epochs["outside_events_gain_per_spike"], [2.0, 1.0, -1.0]
    )
    np.testing.assert_allclose(epochs["outside_events_gain_per_s"], [4 / 3, 2.0, -1.0])
    np.testing.assert_allclose(epochs["outside_events_seconds"], [3.0, 4.0, 2.0])


def test_outside_swr_shares_are_on_the_bins_inside_a_visit():
    epochs = rt.epoch_outside_swr_gains(_time_sums(), _bin_s(), _windows())
    np.testing.assert_allclose(epochs["no_swr_events_full_minus_swr"], [3.0, 1.0, 0.0])
    # b1's total is not positive: no share
    np.testing.assert_allclose(epochs["share_in_no_swr_events"], [0.3, 0.1, np.nan])
    np.testing.assert_allclose(
        epochs["share_in_swr_events_outside_swr"], [0.2, 0.0, np.nan]
    )
    np.testing.assert_allclose(
        epochs["share_in_no_swr_events_two_sided"], [0.3, 0.1, np.nan]
    )


def test_outside_swr_gains_over_all_time():
    epochs = rt.epoch_outside_swr_gains(
        _time_sums(), _bin_s(), _windows(), in_well_visit_only=False
    )
    np.testing.assert_array_equal(epochs["n_spikes"], [16, 10, 2])
    np.testing.assert_allclose(epochs["full_minus_swr"], [110.0, 10.0, -2.0])
    np.testing.assert_allclose(epochs["gain_per_s"], [11.0, 2.0, -1.0])
    # a1's rest: 16 bins, 12 spikes, 104 nats
    np.testing.assert_allclose(
        epochs["outside_events_gain_per_spike"], [104 / 12, 1.0, -1.0]
    )
    # the windows' sums over all their bins outside SWRs
    np.testing.assert_allclose(
        epochs["share_in_no_swr_events"], [3.5 / 110, 0.1, np.nan]
    )


def test_immobile_gains_inside_a_visit_per_spike_and_per_second():
    epochs = rt.epoch_immobile_gains(_time_sums(), _bin_s())
    assert list(epochs["epoch_id"]) == ["a1", "a2", "b1"]
    np.testing.assert_array_equal(epochs["n_spikes"], [8, 10, 4])
    np.testing.assert_allclose(epochs["seconds"], [6.0, 5.0, 3.0])
    np.testing.assert_allclose(epochs["swr_minus_none_per_spike"], [0.75, 0.0, 0.75])
    np.testing.assert_allclose(epochs["full_minus_none_per_spike"], [17 / 8, 1.0, 0.0])
    np.testing.assert_allclose(epochs["full_minus_swr_per_spike"], [11 / 8, 1.0, -0.75])
    np.testing.assert_allclose(epochs["full_minus_none_per_s"], [17 / 6, 2.0, 0.0])
    np.testing.assert_allclose(
        epochs["full_minus_none_per_spike_two_sided"], [17 / 4, 2.0, 0.0]
    )
    # b1's full decoder gains nothing: no share
    np.testing.assert_allclose(
        epochs["share_captured_by_swr_only"], [6 / 17, 0.0, np.nan]
    )
    assert epochs["full_above_swr_only"].tolist() == [True, True, False]


def test_immobile_gains_over_all_time():
    epochs = rt.epoch_immobile_gains(_time_sums(), _bin_s(), in_well_visit_only=False)
    # a1: 24 bins, 20 spikes, swr 56, full 167
    np.testing.assert_array_equal(epochs["n_spikes"], [20, 10, 4])
    np.testing.assert_allclose(epochs["full_minus_swr_per_s"], [111 / 12, 2.0, -1.0])
    np.testing.assert_allclose(
        epochs["share_captured_by_swr_only"], [56 / 167, 0.0, np.nan]
    )


def test_immobile_statistics_count_the_epochs_with_full_above_swr_only():
    epochs = rt.epoch_immobile_gains(_time_sums(), _bin_s())
    stats = rt.immobile_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * len(rt.IMMOBILE_MEASURES)
    key = stats.set_index(["family", "measure"])
    difference = key.loc[("filter", "full_minus_swr_per_spike")]
    # a: median of 11/8 and 1; b: -0.75
    np.testing.assert_allclose(difference["animals"], [(11 / 8 + 1) / 2, -0.75])
    assert difference["estimate"] == pytest.approx(((11 / 8 + 1) / 2 - 0.75) / 2)
    assert difference["ci_low"] <= difference["estimate"] <= difference["ci_high"]
    assert difference["n_animals_positive"] == 1
    assert difference["n_epochs_positive"] == 2
    share = key.loc[("filter", "share_captured_by_swr_only")]
    # a: median of 6/17 and 0; b has none
    assert share["estimate"] == pytest.approx(3 / 17) and share["n_epochs"] == 2
    assert (stats["n_epochs_full_above_swr_only"] == 2).all()


def test_outside_swr_statistics_are_the_mean_over_animals_of_their_epoch_medians():
    epochs = rt.epoch_outside_swr_gains(_time_sums(), _bin_s(), _windows())
    stats = rt.outside_swr_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * len(rt.MEASURES)
    key = stats.set_index(["family", "measure"])
    gain = key.loc[("filter", "gain_per_spike")]
    np.testing.assert_allclose(gain["animals"], [(10 / 6 + 1) / 2, -1.0])
    assert gain["n_epochs"] == 3 and gain["n_epochs_positive"] == 2
    share = key.loc[("filter", "share_in_no_swr_events")]
    assert share["estimate"] == pytest.approx(0.2) and share["n_animals"] == 1


def test_time_sums_with_a_row_twice_or_an_epoch_without_a_time_step_raise():
    with pytest.raises(ValueError, match="no time step"):
        rt.epoch_immobile_gains(_time_sums(), _bin_s().drop("a2"))
    sums = _time_sums()
    twice = pd.concat([sums, sums.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="twice"):
        rt.epoch_outside_swr_gains(twice, _bin_s(), _windows())
