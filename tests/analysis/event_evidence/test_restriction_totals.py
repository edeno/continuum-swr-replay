"""Tests for the non-local restriction contrasts over all immobile time and
outside SWRs, inside a well visit, on tables small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_totals as rt
from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    SCORE_COLUMNS,
)
from tests.support._restriction_tables import (
    bin_facts_table,
    restriction_bins,
    time_sums,
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
    return time_sums(rows)


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
    bins = restriction_bins(
        np.arange(6.0),
        dict.fromkeys(SCORE_COLUMNS, np.arange(6)),
        np.arange(6) == 3,
        category,
        place_sampling_s=np.zeros(6, dtype=np.float32),
    )
    facts = bin_facts_table(np.arange(6.0), [1, 2, 3, 4, 5, 6], np.arange(6) < 3)
    windows = pd.DataFrame({"kind": ["event", "control"], "s": [1, 4], "e": [2, 5]})
    sums = rt.restriction_time_sums(bins, facts, windows)
    assert list(sums.columns[:8]) == [*rt.TIME_SUM_KEYS, "n_bins", "n_spikes"]
    assert len(sums) == 384 and sums["n_bins"].sum() == 6
    # summed over the newer keys, the rows of the first three
    key = sums.groupby(["category", "in_well_visit", "in_event_window"]).sum(
        numeric_only=True
    )
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
    # the rest, elsewhere: 1 - 0.3 - 0.2 and 1 - 0.1 - 0.0; none without a total
    np.testing.assert_allclose(epochs["share_outside_events"], [0.5, 0.9, np.nan])
    np.testing.assert_allclose(
        epochs["share_outside_events_two_sided"], [0.5, 0.9, np.nan]
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
    # the share elsewhere: a's median of 0.5 and 0.9; b has none
    rest = key.loc[("filter", "share_outside_events")]
    assert rest["estimate"] == pytest.approx(0.7) and rest["n_epochs"] == 2


def test_inside_swr_gains_are_on_the_immobile_bins_in_swrs_inside_a_visit():
    """In SWRs inside a visit: a1 2 bins, 2 spikes, full - swr = 7 - 6 = 1; a2
    none; b1 2, 2, 2 - 3 = -1. a1's bins in SWRs outside a visit and its moving
    bins never count."""
    epochs = rt.epoch_inside_swr_gains(_time_sums(), _bin_s())
    assert list(epochs["epoch_id"]) == ["a1", "a2", "b1"]
    assert list(epochs["animal"]) == ["a", "a", "b"]
    np.testing.assert_array_equal(epochs["n_spikes"], [2, 0, 2])
    np.testing.assert_allclose(epochs["seconds"], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(epochs["full_minus_swr"], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(epochs["gain_per_spike"], [0.5, np.nan, -0.5])
    np.testing.assert_allclose(epochs["gain_per_s"], [1.0, np.nan, -1.0])
    np.testing.assert_allclose(epochs["gain_per_spike_two_sided"], [1.0, np.nan, -1.0])
    stats = rt.inside_swr_gain_statistics(epochs, n_boot=200, seed=1)
    assert len(stats) == 2 * len(rt.SUBSET_MEASURES)
    gain = stats.set_index(["family", "measure"]).loc[("filter", "gain_per_spike")]
    # a: its one epoch with SWR bins; b: -0.5
    np.testing.assert_allclose(gain["animals"], [0.5, -0.5])
    assert gain["estimate"] == pytest.approx(0.0)
    assert gain["n_epochs"] == 2 and gain["n_epochs_positive"] == 1
    assert gain["n_animals_positive"] == 1


def test_time_sums_with_a_row_twice_or_an_epoch_without_a_time_step_raise():
    with pytest.raises(ValueError, match="no time step"):
        rt.epoch_immobile_gains(_time_sums(), _bin_s().drop("a2"))
    sums = _time_sums()
    twice = pd.concat([sums, sums.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="twice"):
        rt.epoch_outside_swr_gains(twice, _bin_s(), _windows())


def _keyed_bins():
    """Eight bins 0.1 s apart, all immobile outside SWRs and inside a visit but
    bin 0, inside an SWR; ``full`` = bin index, ``swr`` 0, one spike a bin.
    Bin 7 is where SWR detection was not observed. Place sampling: NaN, 5,
    10, 59, 60, 61, 100, 100 s."""
    n = 8
    category = ["immobile_in_swr"] + ["immobile_outside_swr"] * (n - 1)
    scores = dict.fromkeys(SCORE_COLUMNS, np.zeros(n))
    scores = {**scores, "full": np.arange(n), "full_two_sided": 2 * np.arange(n)}
    bins = restriction_bins(
        0.1 * np.arange(n),
        scores,
        np.arange(n) == 0,
        category,
        place_sampling_s=np.array(
            [np.nan, 5, 10, 59, 60, 61, 100, 100], dtype=np.float32
        ),
    )
    facts = bin_facts_table(
        0.1 * np.arange(n), np.ones(n), np.ones(n, bool), np.arange(n) != 7
    )
    windows = pd.DataFrame({"kind": [], "s": [], "e": []})
    return bins, facts, windows


def test_margin_bands_count_bins_to_the_nearest_bin_of_the_mask():
    """At 0.1 s a bin, the margins 0.1, 0.25 and 0.5 s are 1, 2 (0.25/0.1
    rounds to 2) and 5 bins: a bin is beyond a margin when the nearest mask bin
    is more bins away than that."""
    in_swr = np.zeros(12, dtype=bool)
    in_swr[[0, 11]] = True
    np.testing.assert_allclose(
        rt.swr_margin_bands(in_swr, 0.1),
        [0, 0, 0.1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.1, 0, 0],
    )
    np.testing.assert_allclose(rt.swr_margin_bands(np.zeros(3, bool), 0.1), 0.5)


def test_place_sampling_bands_take_the_largest_threshold_reached():
    np.testing.assert_allclose(
        rt.place_sampling_bands(np.array([np.nan, 0, 9.9, 10, 59.9, 60, 1e3])),
        [0, 0, 0, 10, 10, 60, 60],
    )


def test_time_sums_key_the_observed_flag_margin_and_place_sampling():
    sums = rt.restriction_time_sums(*_keyed_bins())
    kept = sums[sums["n_bins"] > 0].set_index(list(rt.TIME_SUM_KEYS))
    # bin 7: not observed, 7 bins (0.7 s) from bin 0, sampled 100 s
    row = kept.loc[("immobile_outside_swr", True, False, False, 0.5, 60.0)]
    assert row["n_bins"] == 1 and row["full"] == 7.0
    # bin 1 is 1 bin from the mask (within 0.1 s); bin 2 is 2 bins (beyond
    # 0.1 s, not beyond 0.25 s = 2 bins)
    assert kept.loc[("immobile_outside_swr", True, False, True, 0.0, 0.0), "full"] == 1
    assert kept.loc[("immobile_outside_swr", True, False, True, 0.1, 10.0), "full"] == 2
    with pytest.raises(ValueError, match="swr_observed"):
        bins, facts, windows = _keyed_bins()
        rt.restriction_time_sums(bins, facts.drop(columns="swr_observed"), windows)


def _keyed_sums() -> pd.DataFrame:
    bins, facts, windows = _keyed_bins()
    sums = rt.restriction_time_sums(bins, facts, windows)
    return sums.assign(epoch_id="a1", animal="a")


def test_the_gain_outside_swrs_leaves_out_bins_where_detection_was_not_observed():
    """Bins 1-6 (bin 7 not observed): full - swr 21 over 6 spikes, 0.6 s."""
    sums = _keyed_sums()
    epochs = rt.epoch_outside_swr_gains(
        sums, pd.Series({"a1": 0.1}), _windows().iloc[:0]
    )
    assert epochs.loc[0, "n_bins"] == 6
    assert epochs.loc[0, "gain_per_spike"] == pytest.approx(21 / 6)
    assert epochs.loc[0, "gain_per_s"] == pytest.approx(21 / 0.6)
    counts = rt.outside_swr_not_observed_counts(sums)
    assert counts == {"n_bins": 1, "n_epochs": 1}
    # all immobile time keeps it: bins 0-7, 28 nats
    immobile = rt.epoch_immobile_gains(sums, pd.Series({"a1": 0.1}))
    assert immobile.loc[0, "n_bins"] == 8


@pytest.mark.parametrize(
    ("keyword", "threshold", "kept_bins"),
    [
        # margins: bins 2-6 (more than 1 bin), 3-6 (2), 6 (5)
        ("beyond_swr_margin_s", 0.1, [2, 3, 4, 5, 6]),
        ("beyond_swr_margin_s", 0.25, [3, 4, 5, 6]),
        ("beyond_swr_margin_s", 0.5, [6]),
        # place sampling: bins 2-6 (at least 10 s), 4-6 (at least 60 s)
        ("place_sampling_at_least_s", 10.0, [2, 3, 4, 5, 6]),
        ("place_sampling_at_least_s", 60.0, [4, 5, 6]),
    ],
)
def test_subset_gains_keep_the_bins_of_the_part(keyword, threshold, kept_bins):
    epochs = rt.epoch_outside_swr_subset_gains(
        _keyed_sums(), pd.Series({"a1": 0.1}), **{keyword: threshold}
    )
    row = epochs.iloc[0]
    assert row["n_bins"] == len(kept_bins) and row["n_spikes"] == len(kept_bins)
    assert row["gain_per_spike"] == pytest.approx(np.mean(kept_bins))
    assert row["gain_per_s"] == pytest.approx(sum(kept_bins) / (0.1 * len(kept_bins)))
    assert row["gain_per_spike_two_sided"] == pytest.approx(2 * np.mean(kept_bins))
    assert row["share_of_bins"] == pytest.approx(len(kept_bins) / 6)


def test_subset_gains_refuse_an_unnamed_threshold():
    with pytest.raises(ValueError, match="one of"):
        rt.epoch_outside_swr_subset_gains(
            _keyed_sums(), pd.Series({"a1": 0.1}), beyond_swr_margin_s=0.2
        )


def test_subset_statistics_have_a_row_per_part_family_and_measure():
    sums = pd.concat(
        [
            _keyed_sums(),
            _keyed_sums().assign(epoch_id="b1", animal="b", full=lambda t: -t["full"]),
        ],
        ignore_index=True,
    )
    stats = rt.outside_swr_subset_statistics(
        sums, pd.Series({"a1": 0.1, "b1": 0.1}), n_boot=50, seed=1
    )
    n_parts = len(rt.OUTSIDE_SWR_MARGINS_S) + len(rt.PLACE_SAMPLING_THRESHOLDS_S)
    assert len(stats) == n_parts * 2 * len(rt.SUBSET_MEASURES)
    assert "share_of_bins" not in set(stats["measure"])
    key = stats.set_index(["subset", "threshold_s", "family", "measure"])
    row = key.loc[("beyond_swr_margin", 0.25, "filter", "gain_per_spike")]
    # a: mean of bins 3-6 = 4.5; b: -4.5
    np.testing.assert_allclose(row["animals"], [4.5, -4.5])
    assert row["estimate"] == pytest.approx(0.0)
    assert row["n_epochs"] == 2 and row["n_epochs_positive"] == 1
    shares = rt.outside_swr_subset_shares(sums, pd.Series({"a1": 0.1, "b1": 0.1}))
    assert len(shares) == n_parts
    share = shares.loc[("place_sampling", 60.0)]
    # bins 4-6 of 6 in both epochs: each animal 0.5
    assert share["estimate"] == pytest.approx(0.5)
    assert share["animal_min"] == pytest.approx(0.5)
    assert share["n_animals"] == 2 and share["n_epochs"] == 2


def test_time_sums_of_the_earlier_design_are_refused():
    earlier = _time_sums().drop(columns=["swr_observed"])
    with pytest.raises(ValueError, match="made before"):
        rt.epoch_outside_swr_gains(earlier, _bin_s(), _windows())


def test_subset_shares_are_the_mean_over_animals_of_their_epoch_medians():
    """a1 keeps bins 3-6 of 1-6 beyond 0.25 s (2 bins); b1 has the scoring
    mask at bins 0 and 7, so only bins 3 and 4 are more than 2 bins from it:
    animals 4/6 and 2/6, no bootstrap."""
    bins, facts, windows = _keyed_bins()
    bins["in_swr"] = np.isin(np.arange(8), [0, 7])
    other = rt.restriction_time_sums(bins, facts, windows)
    sums = pd.concat(
        [_keyed_sums(), other.assign(epoch_id="b1", animal="b")], ignore_index=True
    )
    shares = rt.outside_swr_subset_shares(sums, pd.Series({"a1": 0.1, "b1": 0.1}))
    share = shares.loc[("beyond_swr_margin", 0.25)]
    assert share["estimate"] == pytest.approx((4 / 6 + 2 / 6) / 2)
    assert share["animal_min"] == pytest.approx(2 / 6)


def _four_model_time_sums() -> pd.DataFrame:
    """:func:`_time_sums` with the fourth model: ``swr_allowed`` half of
    ``full`` in each row (two-sided twice the filter score, as the others)."""
    sums = _time_sums()
    for suffix in ("", "_two_sided"):
        sums[f"swr_allowed{suffix}"] = 0.5 * sums[f"full{suffix}"]
    return sums


def test_time_sums_sum_a_fourth_model_when_the_bins_hold_it():
    category = ["immobile_outside_swr"] * 3 + ["immobile_in_swr", "moving", "excluded"]
    scores = dict.fromkeys(SCORE_COLUMNS, np.arange(6))
    bins = restriction_bins(
        np.arange(6.0),
        scores,
        np.arange(6) == 3,
        category,
        place_sampling_s=np.zeros(6, dtype=np.float32),
        swr_allowed=np.full(6, 2.0, dtype=np.float32),
        swr_allowed_two_sided=np.full(6, 3.0, dtype=np.float32),
    )
    facts = bin_facts_table(np.arange(6.0), [1, 2, 3, 4, 5, 6], np.arange(6) < 3)
    windows = pd.DataFrame({"kind": ["event"], "s": [1], "e": [2]})
    sums = rt.restriction_time_sums(bins, facts, windows)
    assert sums["swr_allowed"].sum() == 12.0
    assert sums["swr_allowed_two_sided"].sum() == 18.0
    plain = rt.restriction_time_sums(
        bins.drop(columns=["swr_allowed", "swr_allowed_two_sided"]), facts, windows
    )
    assert "swr_allowed" not in plain.columns
    pd.testing.assert_frame_equal(plain, sums[list(plain.columns)])


def test_contrast_gains_inside_outside_and_over_all_immobile_time():
    """Inside SWRs inside a visit: a1 full 7, swr_allowed 3.5, swr 6 over 2
    bins and 2 spikes; b1 2, 1, 3. Outside SWRs inside a visit: a1 full 10,
    swr_allowed 5, swr 0 over 10 bins and 6 spikes. All immobile time inside a
    visit: a1 full 17, swr_allowed 8.5, none 0 over 12 bins and 8 spikes."""
    sums = _four_model_time_sums()
    inside = rt.epoch_contrast_gains(sums, _bin_s(), time="inside_swr")
    assert list(inside["epoch_id"]) == ["a1", "a2", "b1"]
    np.testing.assert_allclose(inside["full_minus_swr_allowed"], [3.5, 0.0, 1.0])
    np.testing.assert_allclose(
        inside["full_minus_swr_allowed_per_spike"], [1.75, np.nan, 0.5]
    )
    np.testing.assert_allclose(
        inside["swr_allowed_minus_swr_per_s"], [-2.5, np.nan, -2]
    )
    np.testing.assert_allclose(
        inside["swr_allowed_minus_swr_per_s_two_sided"], [-5.0, np.nan, -4.0]
    )
    outside = rt.epoch_contrast_gains(sums, _bin_s(), time="outside_swr")
    np.testing.assert_allclose(outside["swr_allowed_minus_swr"], [5.0, 5.0, -1.0])
    np.testing.assert_allclose(
        outside["full_minus_swr_allowed_per_spike"], [5 / 6, 0.5, -0.5]
    )
    immobile = rt.epoch_contrast_gains(sums, _bin_s(), time="immobile")
    np.testing.assert_allclose(
        immobile["swr_allowed_minus_none_per_spike"], [8.5 / 8, 0.5, 0.0]
    )
    # the canonical contrasts are the existing functions' values
    np.testing.assert_allclose(
        immobile["full_minus_swr_per_spike"],
        rt.epoch_immobile_gains(sums, _bin_s())["full_minus_swr_per_spike"],
    )
    np.testing.assert_allclose(
        outside["full_minus_swr_per_s_two_sided"],
        rt.epoch_outside_swr_gains(sums, _bin_s(), _windows())["gain_per_s_two_sided"],
    )
    stats = rt.contrast_gain_statistics(
        inside, ["full_minus_swr_allowed"], n_boot=50, seed=1
    ).set_index(["family", "measure"])
    row = stats.loc[("filter", "full_minus_swr_allowed_per_spike")]
    # a: its one epoch with SWR bins (1.75); b: 0.5
    np.testing.assert_allclose(row["animals"], [1.75, 0.5])
    assert len(stats) == 2 * 2


def test_contrast_gains_refuse_a_contrast_the_time_sums_do_not_hold():
    with pytest.raises(ValueError, match="do not hold the models"):
        rt.epoch_contrast_gains(
            _time_sums(), _bin_s(), time="immobile", contrasts=["swr_allowed_minus_swr"]
        )
    with pytest.raises(ValueError, match="unknown contrasts"):
        rt.epoch_contrast_gains(
            _time_sums(), _bin_s(), time="immobile", contrasts=["x"]
        )
    with pytest.raises(ValueError, match="not one of"):
        rt.epoch_contrast_gains(_time_sums(), _bin_s(), time="moving")
    # by default, the contrasts the time sums hold
    canonical = rt.epoch_contrast_gains(_time_sums(), _bin_s(), time="immobile")
    assert "full_minus_swr_allowed" not in canonical.columns
    assert "full_minus_none_per_s" in canonical.columns
