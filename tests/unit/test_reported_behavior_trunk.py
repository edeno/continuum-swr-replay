"""The behavior summary's trunk-share and paired current-arm values (question 3.1), on synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import (
    current_arm_paired_values,
    trunk_values,
)

# Per animal, one event of each kind, with the current arm 2 (patch 2, whose
# trunk is arm 7): the event's share on arm 7 and on another trunk (arm 8).
TRUNK_CONTENT = {
    ("a", False): (0.1, 0.1),
    ("a", True): (0.2, 0.3),
    ("b", False): (0.3, 0.3),
    ("b", True): (0.1, 0.2),
    ("c", False): (0.05, 0.25),
    ("c", True): (0.6, 0.2),
}


def test_trunk_values_summarize_over_animals():
    arm = pd.DataFrame(
        [
            {
                "animal": animal,
                "event_id": f"{animal}-{int(is_swr)}",
                "is_swr": is_swr,
                "candidate_arm": candidate,
                "is_current_arm": int(candidate == 2),
                "arm_fraction": {7: current, 8: other}.get(
                    candidate, (1 - current - other) / 6 if candidate < 6 else 0.0
                ),
            }
            for (animal, is_swr), (current, other) in TRUNK_CONTENT.items()
            for candidate in range(9)
        ]
    )
    values = {v.key: v for v in trunk_values(arm, "x")}
    # all trunks, SWR: 0.5, 0.3, 0.8; non-SWR: 0.2, 0.6, 0.3 (b's is higher)
    assert values["trunk_share_swr"].value == pytest.approx(0.5)
    assert "over the 3 animals" in values["trunk_share_swr"].description
    assert values["trunk_share_non_swr"].value == pytest.approx(0.3)
    assert values["trunk_share_animals_swr_higher"].value == 2
    # the current patch's trunk, SWR: 0.2, 0.1, 0.6; non-SWR: 0.1, 0.3, 0.05
    assert values["trunk_share_current_patch_swr"].value == pytest.approx(0.2)
    assert values["trunk_share_current_patch_non_swr"].value == pytest.approx(0.1)
    assert values["trunk_share_current_patch_animals_swr_higher"].value == 2
    # the other trunks, SWR: 0.3, 0.2, 0.2; non-SWR: 0.1, 0.3, 0.25 (only a's rises)
    assert values["trunk_share_other_patches_swr"].value == pytest.approx(0.2)
    assert values["trunk_share_other_patches_non_swr"].value == pytest.approx(0.25)
    assert values["trunk_share_other_patches_animals_swr_higher"].value == 1
    # SWR: 0.2 / 0.5, 0.1 / 0.3, 0.6 / 0.8; non-SWR: 0.1 / 0.2, 0.3 / 0.6, 0.05 / 0.3
    assert values["trunk_current_patch_fraction_swr"].value == pytest.approx(0.4)
    assert values["trunk_current_patch_fraction_non_swr"].value == pytest.approx(0.5)


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
    assert "over the 5 animals" in values[0].description
    assert np.isclose(values[0].value, 0.2)
