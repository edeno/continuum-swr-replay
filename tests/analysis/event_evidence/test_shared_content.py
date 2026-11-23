"""Shared content: the agreement per spike by the width of the others' estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.shared_content import (
    WIDTH_BIN_LABELS,
    agreement_by_width_statistics,
    shared_content_statistics,
    width_bins,
    width_matched,
)


def test_the_bins_are_fixed_and_closed_on_the_left():
    width = pd.Series([0.0, 24.9, 25.0, 49.9, 50.0, 99.9, 100.0, 199.9, 200.0, 400.0])
    assert list(width_bins(width)) == [
        "<25", "<25", "25–50", "25–50", "50–100", "50–100",
        "100–200", "100–200", "≥200", "≥200",
    ]  # fmt: skip
    assert WIDTH_BIN_LABELS == ("<25", "25–50", "50–100", "100–200", "≥200")


def _windows() -> pd.DataFrame:
    """No-SWR events of animals a (epochs a1, a2) and b (b1), and one SWR event.

    Narrow (<25 cm): a1 has 1, 2, -1; a2 has 3, 5; b1 has 0.5, -0.5, -2.
    Wide (>=200 cm): a1 has -1; b1 has -3, 1. One SWR event in a1, narrow, 4.
    An event without spikes (NaN) in a1, narrow, counts nowhere. The
    reference is the agreement minus 1 except in the first event (no donor,
    NaN), so the difference is 1 wherever both are known."""
    rows = [
        ("a", "a1", "no SWR", 10.0, 1.0),
        ("a", "a1", "no SWR", 12.0, 2.0),
        ("a", "a1", "no SWR", 20.0, -1.0),
        ("a", "a2", "no SWR", 5.0, 3.0),
        ("a", "a2", "no SWR", 8.0, 5.0),
        ("b", "b1", "no SWR", 15.0, 0.5),
        ("b", "b1", "no SWR", 16.0, -0.5),
        ("b", "b1", "no SWR", 17.0, -2.0),
        ("a", "a1", "no SWR", 250.0, -1.0),
        ("b", "b1", "no SWR", 300.0, -3.0),
        ("b", "b1", "no SWR", 220.0, 1.0),
        ("a", "a1", "SWR", 10.0, 4.0),
        ("a", "a1", "no SWR", 10.0, np.nan),
    ]
    windows = pd.DataFrame(
        rows, columns=["animal", "epoch_id", "group", "hpd_cm", "agreement_per_spike"]
    )
    windows["agreement_null_per_spike"] = windows["agreement_per_spike"] - 1.0
    windows.loc[0, "agreement_null_per_spike"] = np.nan
    # every donor in its window's width bin (no donor: NaN)
    windows["hpd_null_cm"] = windows["hpd_cm"]
    windows.loc[0, "hpd_null_cm"] = np.nan
    windows["agreement_minus_null_per_spike"] = (
        windows["agreement_per_spike"] - windows["agreement_null_per_spike"]
    )
    return windows


def test_the_agreement_by_width_is_the_gain_statistic_per_bin():
    stats = agreement_by_width_statistics(_windows(), n_boot=50, seed=1)
    narrow = stats.loc[("<25", "no SWR")]
    # a: epoch medians 1 (a1) and 4 (a2), their median 2.5; b: median -0.5
    np.testing.assert_allclose(narrow["animals"], [2.5, -0.5])
    assert narrow["estimate"] == pytest.approx(1.0)
    assert narrow["n_animals_positive"] == 1
    assert narrow["n_epochs"] == 3 and narrow["n_scored"] == 8
    # shares above zero: a1 2/3, a2 1, b1 1/3; a's median 5/6, b 1/3
    assert narrow["share_positive"] == pytest.approx((5 / 6 + 1 / 3) / 2)
    assert narrow["share_positive_animal_min"] == pytest.approx(1 / 3)
    assert narrow["share_positive_animal_max"] == pytest.approx(5 / 6)
    # 8 of the 11 scored no-SWR events are narrow
    assert narrow["share_of_events"] == pytest.approx(8 / 11)
    # per epoch: a1 3 of 4, a2 2 of 2, b1 3 of 5; a's mean 7/8, b's 3/5
    np.testing.assert_allclose(narrow["share_of_events_animals"], [7 / 8, 3 / 5])
    assert narrow["share_of_events_animal_mean"] == pytest.approx((7 / 8 + 3 / 5) / 2)
    assert narrow["share_of_events_animal_min"] == pytest.approx(3 / 5)
    assert narrow["share_of_events_animal_max"] == pytest.approx(7 / 8)
    wide = stats.loc[("≥200", "no SWR")]
    # a: -1; b: median of -3 and 1 is -1
    assert wide["estimate"] == pytest.approx(-1.0)
    assert wide["share_of_events"] == pytest.approx(3 / 11)
    # a1 1 of 4 (a2 none: 0), so a's mean 1/8; b1 2 of 5
    assert wide["share_of_events_animal_mean"] == pytest.approx((1 / 8 + 2 / 5) / 2)
    assert wide["n_epochs"] == 2
    swr = stats.loc[("<25", "SWR")]
    assert swr["estimate"] == pytest.approx(4.0) and swr["share_of_events"] == 1.0
    # no row for a bin a group has no event in
    assert ("25–50", "no SWR") not in stats.index


def test_the_reference_and_the_difference_are_over_the_events_with_a_donor():
    """The reference leaves out the event without a donor; the difference is 1
    in every event that has both."""
    stats = shared_content_statistics(_windows(), n_boot=50, seed=1)
    narrow = stats.reference.loc[("<25", "no SWR")]
    assert narrow["n_scored"] == 7
    # a1 without its first event: 2 - 1 and -1 - 1, median -0.5; a2 median 3;
    # a: median of -0.5 and 3 = 1.25; b: -1.5
    np.testing.assert_allclose(narrow["animals"], [1.25, -1.5])
    difference = stats.difference.loc[("<25", "no SWR")]
    assert difference["estimate"] == pytest.approx(1.0)
    assert difference["n_animals_positive"] == 2
    pd.testing.assert_frame_equal(
        stats.agreement, agreement_by_width_statistics(_windows(), n_boot=50, seed=1)
    )


def test_an_animal_without_events_in_a_bin_counts_zero_in_its_share():
    """Animal a has one narrow and one wide event, b two narrow ones: the wide
    bin's mean share over animals is (1/2 + 0) / 2, its smallest 0."""
    windows = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "epoch_id": ["a1", "a1", "b1", "b1"],
            "group": "no SWR",
            "hpd_cm": [10.0, 300.0, 10.0, 12.0],
            "agreement_per_spike": [1.0, -1.0, 0.5, 0.2],
        }
    )
    wide = agreement_by_width_statistics(windows, n_boot=20).loc[("≥200", "no SWR")]
    assert wide["share_of_events_animal_mean"] == pytest.approx(0.25)
    assert wide["share_of_events_animal_min"] == 0.0
    assert wide["share_of_events_animal_max"] == pytest.approx(0.5)
    assert wide["share_of_events_animal_names"].tolist() == ["a", "b"]
    np.testing.assert_allclose(wide["share_of_events_animals"], [0.5, 0.0])


def _donor_width_windows() -> pd.DataFrame:
    """Narrow no-SWR events (10 cm) of animals a, b and c, each with agreement
    0.5, half with a narrow donor (reference -0.5) and half with a wide donor
    (300 cm, reference -0.1): the reference depends on the donor's width only,
    as in the data."""
    rows = []
    for animal in "abc":
        for donor, reference in ((10.0, -0.5), (300.0, -0.1)) * 2:
            rows.append((animal, f"{animal}1", 10.0, donor, 0.5, reference))
    windows = pd.DataFrame(
        rows,
        columns=[
            "animal", "epoch_id", "hpd_cm", "hpd_null_cm", "agreement_per_spike",
            "agreement_null_per_spike",
        ],
    ).assign(group="no SWR")  # fmt: skip
    windows["agreement_minus_null_per_spike"] = (
        windows["agreement_per_spike"] - windows["agreement_null_per_spike"]
    )
    return windows


def test_the_reference_is_read_over_the_donors_of_the_same_width():
    """Unmatched, the narrow bin's reference would mix narrow and wide donors
    (each animal's median -0.3, the difference 0.8); matched it is the narrow
    donors' -0.5 and the difference 1.0. The agreement is over every event."""
    windows = _donor_width_windows()
    assert width_matched(windows).tolist() == [True, False] * 6
    stats = shared_content_statistics(windows, n_boot=20)
    narrow = ("<25", "no SWR")
    assert stats.reference.loc[narrow, "estimate"] == pytest.approx(-0.5)
    assert stats.difference.loc[narrow, "estimate"] == pytest.approx(1.0)
    assert stats.reference.loc[narrow, "n_scored"] == 6
    assert stats.agreement.loc[narrow, "n_scored"] == 12
    unmatched = agreement_by_width_statistics(
        windows, n_boot=20, column="agreement_null_per_spike"
    )
    assert unmatched.loc[narrow, "estimate"] == pytest.approx(-0.3)
