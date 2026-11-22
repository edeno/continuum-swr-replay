"""Shared content: the agreement per spike by the spread of the others' estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.shared_content import (
    SPREAD_BIN_LABELS,
    agreement_by_spread_statistics,
    spread_bins,
)


def test_the_bins_are_fixed_and_closed_on_the_left():
    spread = pd.Series([0.0, 24.9, 25.0, 49.9, 50.0, 99.9, 100.0, 400.0])
    assert list(spread_bins(spread)) == [
        "<25", "<25", "25–50", "25–50", "50–100", "50–100",
        "≥100", "≥100",
    ]  # fmt: skip
    assert SPREAD_BIN_LABELS == ("<25", "25–50", "50–100", "≥100")


def _windows() -> pd.DataFrame:
    """No-SWR events of animals a (epochs a1, a2) and b (b1), and one SWR event.

    Narrow (<25 cm): a1 has 1, 2, -1; a2 has 3, 5; b1 has 0.5, -0.5, -2.
    Wide (>=100 cm): a1 has -1; b1 has -3, 1. One SWR event in a1, narrow, 4.
    An event without spikes (NaN) in a1, narrow, counts nowhere."""
    rows = [
        ("a", "a1", "no SWR", 10.0, 1.0),
        ("a", "a1", "no SWR", 12.0, 2.0),
        ("a", "a1", "no SWR", 20.0, -1.0),
        ("a", "a2", "no SWR", 5.0, 3.0),
        ("a", "a2", "no SWR", 8.0, 5.0),
        ("b", "b1", "no SWR", 15.0, 0.5),
        ("b", "b1", "no SWR", 16.0, -0.5),
        ("b", "b1", "no SWR", 17.0, -2.0),
        ("a", "a1", "no SWR", 150.0, -1.0),
        ("b", "b1", "no SWR", 200.0, -3.0),
        ("b", "b1", "no SWR", 120.0, 1.0),
        ("a", "a1", "SWR", 10.0, 4.0),
        ("a", "a1", "no SWR", 10.0, np.nan),
    ]
    return pd.DataFrame(
        rows,
        columns=["animal", "epoch_id", "group", "spread_cm", "agreement_per_spike"],
    )


def test_the_agreement_by_spread_is_the_gain_statistic_per_bin():
    stats = agreement_by_spread_statistics(_windows(), n_boot=50, seed=1)
    narrow = stats.loc[("<25", "no SWR")]
    # a: epoch medians 1 (a1) and 4 (a2), their median 2.5; b: median -0.5
    np.testing.assert_allclose(narrow["animals"], [2.5, -0.5])
    assert narrow["estimate"] == pytest.approx(1.0)
    assert narrow["n_epochs"] == 3 and narrow["n_scored"] == 8
    # shares above zero: a1 2/3, a2 1, b1 1/3; a's median 5/6, b 1/3
    assert narrow["share_positive"] == pytest.approx((5 / 6 + 1 / 3) / 2)
    assert narrow["share_positive_animal_min"] == pytest.approx(1 / 3)
    assert narrow["share_positive_animal_max"] == pytest.approx(5 / 6)
    # 8 of the 11 scored no-SWR events are narrow
    assert narrow["share_of_events"] == pytest.approx(8 / 11)
    wide = stats.loc[("≥100", "no SWR")]
    # a: -1; b: median of -3 and 1 is -1
    assert wide["estimate"] == pytest.approx(-1.0)
    assert wide["share_of_events"] == pytest.approx(3 / 11)
    assert wide["n_epochs"] == 2
    swr = stats.loc[("<25", "SWR")]
    assert swr["estimate"] == pytest.approx(4.0) and swr["share_of_events"] == 1.0
    # no row for a bin a group has no event in
    assert ("25–50", "no SWR") not in stats.index
