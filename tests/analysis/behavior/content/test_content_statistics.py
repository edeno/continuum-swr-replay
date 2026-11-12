"""The content statistics (trunk shares, event length, the concentration
contrast), on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.content import content_statistics as stats

CURRENT_ARM = {"a": 2, "b": 2, "c": 5}


def _arm_table() -> pd.DataFrame:
    """Per animal, one non-SWR and one SWR event. Non-SWR: 0.1 on arm 7, 0.1 on
    arm 6. SWR: 0.2 on arm 7, 0.3 on arm 8. The current arm of animals a and b
    is 2 (patch 2, whose trunk is arm 7); of animal c, 5 (patch 3, trunk 8)."""
    rows = []
    for animal, current in CURRENT_ARM.items():
        for is_swr, trunk in ((False, {6: 0.1, 7: 0.1}), (True, {7: 0.2, 8: 0.3})):
            event = f"{animal}-{int(is_swr)}"
            leaf = (1.0 - sum(trunk.values())) / 6
            for arm in range(9):
                rows.append(
                    {
                        "animal": animal,
                        "session_date": "d1",
                        "epoch": 2,
                        "event_id": event,
                        "is_swr": is_swr,
                        "candidate_arm": arm,
                        "is_current_arm": int(arm == current),
                        "arm_fraction": trunk.get(arm, leaf if arm < 6 else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_each_patch_has_its_own_trunk():
    assert stats.PATCH_TRUNK == {1: 6, 2: 7, 3: 8}


def test_trunk_shares_split_by_the_current_patch():
    events = stats.trunk_shares(_arm_table()).set_index("event_id")
    assert len(events) == 6
    assert events["is_swr"].dtype == bool
    assert events.loc["a-0", "animal"] == "a"
    assert events.loc["a-0", "session_date"] == "d1"
    assert events.loc["a-0", "epoch"] == 2
    assert not events.loc["a-0", "is_swr"]
    assert events.loc["a-0", "trunk"] == pytest.approx(0.2)
    assert events.loc["a-0", "current_patch_trunk"] == pytest.approx(0.1)
    assert events.loc["a-1", "is_swr"]
    assert events.loc["a-1", "trunk"] == pytest.approx(0.5)
    assert events.loc["a-1", "current_patch_trunk"] == pytest.approx(0.2)
    assert events.loc["a-1", "other_trunks"] == pytest.approx(0.3)
    # animal c's current patch is patch 3: its trunk is arm 8
    assert events.loc["c-0", "current_patch_trunk"] == 0.0
    assert events.loc["c-0", "other_trunks"] == pytest.approx(0.2)
    assert events.loc["c-1", "current_patch_trunk"] == pytest.approx(0.3)
    assert events.loc["c-1", "other_trunks"] == pytest.approx(0.2)


def test_trunk_shares_refuse_an_event_without_exactly_one_current_arm():
    two = _arm_table()
    two.loc[
        (two["event_id"] == "a-1") & (two["candidate_arm"] == 3), "is_current_arm"
    ] = 1
    with pytest.raises(ValueError, match="1 events do not have exactly one current"):
        stats.trunk_shares(two)
    none = _arm_table()
    none.loc[none["event_id"].isin(["a-1", "b-0"]), "is_current_arm"] = 0
    with pytest.raises(ValueError, match="2 events do not have exactly one current"):
        stats.trunk_shares(none)


def test_trunk_shares_refuse_an_event_of_unknown_swr_status():
    table = _arm_table().astype({"is_swr": object})
    table.loc[table["event_id"] == "b-0", "is_swr"] = None
    with pytest.raises(ValueError, match="1 events have a missing is_swr"):
        stats.trunk_shares(table)


def test_trunk_shares_refuse_a_missing_share():
    # a missing share is not a share of zero
    table = _arm_table()
    table.loc[
        (table["event_id"] == "c-1") & (table["candidate_arm"] == 7), "arm_fraction"
    ] = np.nan
    with pytest.raises(ValueError, match="1 events have a missing arm_fraction"):
        stats.trunk_shares(table)


def _events() -> pd.DataFrame:
    """Per-event trunk shares of three animals in five sessions (trunk, current
    patch's trunk). Animal a's SWR events and animal b's non-SWR events lie in
    two sessions of unequal sizes, so the session level moves their means."""
    rows = [
        ("a", "d1", 1, True, 0.3, 0.1),
        ("a", "d1", 1, True, 0.3, 0.1),
        ("a", "d1", 2, True, 0.9, 0.4),
        ("a", "d1", 1, False, 0.2, 0.1),
        ("b", "d2", 1, True, 0.3, 0.3),
        ("b", "d2", 1, False, 0.5, 0.2),
        ("b", "d2", 1, False, 0.7, 0.2),
        ("b", "d2", 2, False, 0.3, 0.0),
        ("c", "d3", 1, True, 0.75, 0.6),
        ("c", "d3", 1, False, 0.1, 0.05),
    ]
    events = pd.DataFrame(
        rows,
        columns=[
            "animal",
            "session_date",
            "epoch",
            "is_swr",
            "trunk",
            "current_patch_trunk",
        ],
    )
    events["other_trunks"] = events["trunk"] - events["current_patch_trunk"]
    return events


def test_trunk_share_statistics_are_means_over_animals_of_session_means():
    table = stats.trunk_share_statistics(_events())
    assert list(table.index) == ["trunk", "current_patch_trunk", "other_trunks"]
    # SWR: a's sessions 0.3 (of 0.3 and 0.3) and 0.9 give 0.6 (its events' mean
    # is 0.5), b 0.3, c 0.75: mean 0.55 (median 0.6). Non-SWR: a 0.2, b's
    # sessions 0.6 and 0.3 give 0.45 (its events' mean is 0.5), c 0.1: mean 0.25.
    assert table.loc["trunk"].to_dict() == {
        "swr_mean": pytest.approx(0.55),
        "non_swr_mean": pytest.approx(0.25),
        "n_animals": 3,
        "n_animals_swr_higher": 2,
    }
    # SWR: a 0.25 (sessions 0.1 and 0.4), b 0.3, c 0.6; non-SWR: a 0.1, b 0.1
    # (sessions 0.2 and 0.0), c 0.05
    assert table.loc["current_patch_trunk"].to_dict() == {
        "swr_mean": pytest.approx(1.15 / 3),
        "non_swr_mean": pytest.approx(0.25 / 3),
        "n_animals": 3,
        "n_animals_swr_higher": 3,
    }
    # SWR: a 0.35 (sessions 0.2 and 0.5), b 0.0, c 0.15; non-SWR: a 0.1, b 0.35
    # (sessions 0.4 and 0.3), c 0.05
    assert table.loc["other_trunks"].to_dict() == {
        "swr_mean": pytest.approx(0.5 / 3),
        "non_swr_mean": pytest.approx(0.5 / 3),
        "n_animals": 3,
        "n_animals_swr_higher": 2,
    }


def test_the_current_patch_fraction_is_a_ratio_of_the_animal_s_means():
    fractions = stats.current_patch_trunk_fraction(_events())
    # SWR: 0.25 / 0.6, 0.3 / 0.3, 0.6 / 0.75 (the mean of animal a's sessions'
    # own ratios would be 0.39, not 0.42)
    assert fractions["swr"] == pytest.approx((0.25 / 0.6 + 1.0 + 0.8) / 3)
    # non-SWR: 0.1 / 0.2, 0.1 / 0.45, 0.05 / 0.1
    assert fractions["non_swr"] == pytest.approx((0.5 + 0.1 / 0.45 + 0.5) / 3)
    only_a = stats.current_patch_trunk_fraction(_events().query("animal == 'a'"))
    assert only_a["swr"] == pytest.approx(0.25 / 0.6)


def test_the_trunk_statistics_refuse_an_animal_without_both_kinds_of_event():
    # animal c keeps only its non-SWR event
    events = _events().iloc[[0, 1, 2, 3, 4, 5, 6, 7, 9]]
    with pytest.raises(ValueError, match=r"without both SWR and non-SWR.*\['c'\]"):
        stats.trunk_share_statistics(events)
    with pytest.raises(ValueError, match=r"without both SWR and non-SWR.*\['c'\]"):
        stats.current_patch_trunk_fraction(events)


def test_event_length_statistics_compare_the_class_medians():
    events = pd.DataFrame(
        {
            "animal": ["a", "b", "a", "a", "b", "a", "b", "b"],
            "is_swr": [True, True, True, False, False, False, False, True],
            "n_nonlocal_bins": [10.0, 20.0, 30.0, 4.0, 6.0, 8.0, 100.0, np.nan],
        }
    )
    assert stats.event_length_statistics(events) == {
        "median_bins": 10.0,  # of 4, 6, 8, 10, 20, 30, 100
        "median_bins_swr": 20.0,
        "median_bins_non_swr": 7.0,
        "swr_to_non_swr_ratio": pytest.approx(20 / 7),
    }


def test_the_concentration_contrast_as_ratios_and_the_length_s_share():
    result = stats.concentration_contrast_statistics(2.0, 0.5)
    assert result == {
        "ratio": pytest.approx(7.38905609893065),  # e ** 2
        "length_adjusted_ratio": pytest.approx(1.6487212707001282),  # e ** 0.5
        "length_share": 0.75,
    }
