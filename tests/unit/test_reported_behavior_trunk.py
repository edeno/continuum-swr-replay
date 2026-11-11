"""Part 3's trunk-share and paired current-arm values (claim 19a), on synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import (
    current_arm_paired_values,
    trunk_shares,
    trunk_values,
)

ANIMALS = ("a", "b", "c")


def _arm_table() -> pd.DataFrame:
    """Per animal, one non-SWR and one SWR event, both with the current arm 2
    (patch 2, whose trunk is arm 7). Non-SWR: 0.1 on arm 7, 0.1 on arm 6.
    SWR: 0.2 on arm 7, 0.3 on arm 8."""
    rows = []
    for animal in ANIMALS:
        for is_swr, trunk in ((False, {6: 0.1, 7: 0.1}), (True, {7: 0.2, 8: 0.3})):
            event = f"{animal}-{int(is_swr)}"
            leaf = (1.0 - sum(trunk.values())) / 6
            for arm in range(9):
                rows.append(
                    {
                        "animal": animal,
                        "event_id": event,
                        "is_swr": is_swr,
                        "candidate_arm": arm,
                        "is_current_arm": int(arm == 2),
                        "arm_fraction": trunk.get(arm, leaf if arm < 6 else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_trunk_shares_split_by_the_current_patch():
    events = trunk_shares(_arm_table()).set_index("event_id")
    assert events.loc["a-0", "trunk"] == pytest.approx(0.2)
    assert events.loc["a-0", "current_patch_trunk"] == pytest.approx(0.1)
    assert events.loc["a-1", "current_patch_trunk"] == pytest.approx(0.2)
    assert events.loc["a-1", "other_trunks"] == pytest.approx(0.3)


def test_trunk_values_summarize_over_animals():
    values = {v.key: v.value for v in trunk_values(_arm_table(), "x")}
    assert values["trunk_share_swr"] == pytest.approx(0.5)
    assert values["trunk_share_non_swr"] == pytest.approx(0.2)
    assert values["trunk_share_animals_swr_higher"] == 3
    assert values["trunk_share_current_patch_swr"] == pytest.approx(0.2)
    assert values["trunk_share_other_patches_non_swr"] == pytest.approx(0.1)
    assert values["trunk_current_patch_fraction_swr"] == pytest.approx(0.4)
    assert values["trunk_current_patch_fraction_non_swr"] == pytest.approx(0.5)


def test_an_event_without_one_current_arm_is_left_out():
    table = _arm_table()
    table.loc[
        (table["event_id"] == "a-1") & (table["candidate_arm"] == 3), "is_current_arm"
    ] = 1
    events = trunk_shares(table)
    assert "a-1" not in set(events["event_id"])


def _fits(differences: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for animal, diffs in differences.items():
        for s, d in enumerate(diffs):
            for population, coef in (("SWR", 1.0 + d), ("nonSWR", 1.0)):
                rows.append(
                    {
                        "model": "C2_base",
                        "population": population,
                        "animal": animal,
                        "session_date": s,
                        "epoch": "02_r1",
                        "coef_is_current_arm": coef,
                        "bse_is_current_arm": 0.1,
                        "sd_is_current_arm": 0.3,
                        "separated_terms": "",
                        "converged": True,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_difference_is_within_session_then_over_animals():
    fits = _fits(
        {
            "a": [0.5] * 6,
            "b": [-0.2] * 6,
            "c": [0.1] * 6,
            "d": [0.3] * 6,
            "e": [-0.4] * 6,
        }
    )
    values = {v.key: v.value for v in current_arm_paired_values(fits, "x")}
    assert values["current_arm_paired_difference"] == pytest.approx(0.1)
    assert values["current_arm_paired_difference_animals_positive"] == 3


def test_paired_difference_drops_a_session_missing_one_split():
    fits = _fits({a: [0.2] * 6 for a in "abcde"})
    fits = fits.drop(
        fits.index[(fits["population"] == "SWR") & (fits["session_date"] == 0)]
    )
    values = current_arm_paired_values(fits, "x")
    assert "25 sessions" in values[0].description
    assert np.isclose(values[0].value, 0.2)
