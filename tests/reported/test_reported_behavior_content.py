"""The behavior summary's content-contrast, concentration and estimator values,
on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import (
    CONTENT_SPECS,
    ESTIMATOR_TERMS,
    ESTIMATORS,
    breadth_derived_values,
    content_contrast_values,
    estimator_values,
)

SOURCES = {
    "": "fraction",
    "_rewarded": "rewarded",
    "_omitted": "omitted",
    "_dm": "dm",
    "_dm_len": "dm with length",
}

# The rows the test reads: (file suffix, rung, population, term) -> what the
# audited aggregate holds for it.
CURRENT, BREADTH = "is_current_arm", "log_phi_is_swr"
BELIEF = "swr_x_reward_belief_mean_candidate_arm_preoutcome"
KNOWN = {
    ("", "C2_base", "SWR", CURRENT): {
        "coef_animal_median": 1.5,
        "n_sessions": 200,
        "n_animals_pos": 5,
        "session_signrank_fdr_q": 0.001,
    },
    ("", "C2_base", "nonSWR", CURRENT): {
        "coef_animal_median": 0.7,
        "n_sessions": 190,
        "n_animals_pos": 4,
        "session_signrank_fdr_q": np.nan,
    },
    ("_dm", "C2_base", "all", BREADTH): {
        "coef_animal_median": 2.0,
        "pct_positive_session": 0.8,
    },
    ("_dm_len", "C2_base", "all", BREADTH): {
        "coef_animal_median": 0.5,
        "pct_positive_session": 0.6,
    },
    ("_omitted", "CS1_current", "all", "swr_x_is_current_arm"): {"n_animals": 4},
    ("", "CS5b_valuevsprevgoal", "all", BELIEF): {"n_sessions_flagged_separated": 7},
    ("_dm", "CS5b_valuevsprevgoal", "all", BELIEF): {"n_sessions_flagged_separated": 3},
}


def _audited() -> dict[str, pd.DataFrame]:
    """One audited row per reported contrast: the ``KNOWN`` rows as given, every
    other with five animals and numbers no assertion reads."""
    rows: dict[str, list[dict]] = {suffix: [] for suffix in SOURCES}
    for suffix, model, population, term in sorted(set(CONTENT_SPECS.values())):
        row = {
            "model": model,
            "population": population,
            "term": term,
            "coef_animal_median": -9.0,
            "n_sessions": 1,
            "n_animals": 5,
            "n_animals_pos": 0,
            "pct_positive_session": 0.0,
            "n_sessions_flagged_separated": 0,
            "session_signrank_fdr_q": 0.9,
        }
        rows[suffix].append(row | KNOWN.get((suffix, model, population, term), {}))
    return {suffix: pd.DataFrame(frame) for suffix, frame in rows.items()}


def test_content_contrast_values_read_each_row_of_its_own_bundle():
    values = {v.key: v for v in content_contrast_values(_audited(), SOURCES)}
    read = {key: value.value for key, value in values.items()}
    assert read["base_current_swr"] == 1.5
    assert read["base_current_swr_sessions"] == 200
    assert read["base_current_swr_animals_positive"] == 5
    assert read["base_current_swr_q"] == 0.001
    assert read["base_current_non_swr"] == 0.7
    assert read["base_current_non_swr_sessions"] == 190
    assert read["base_current_non_swr_animals_positive"] == 4
    assert "base_current_non_swr_q" not in read  # no q without a test
    # the two concentration contrasts come from different bundles
    assert read["dm_breadth"] == 2.0
    assert read["dm_breadth_positive_fraction"] == 0.8
    assert values["dm_breadth"].source == "dm"
    assert read["dm_breadth_length_adjusted"] == 0.5
    assert read["dm_breadth_length_adjusted_positive_fraction"] == 0.6
    assert values["dm_breadth_length_adjusted"].source == "dm with length"
    assert read["swr_prevgoal_separated"] == 7
    assert read["dm_swr_prevgoal_separated"] == 3
    # the omitted stratum may have fewer than five animals, and says how many
    assert read["swr_current_direct_omitted_animals"] == 4
    assert values["swr_current_direct_omitted"].source == "omitted"
    assert values["swr_current_direct_rewarded"].source == "rewarded"
    # a key's questions follow its leading words
    assert values["base_current_swr"].questions == ("3.1",)
    assert values["armreward_swr"].questions == ("3.2",)
    assert values["swr_armreward_direct"].questions == ("3.2", "3.6")
    assert values["dm_swr_armvalue_direct"].questions == ("3.6",)
    assert values["armuncert_all"].questions == ("3.7",)


def test_content_contrast_values_refuse_a_missing_row_and_a_missing_animal():
    audited = _audited()
    fraction = audited[""]
    is_base = (fraction["model"] == "C2_base") & (fraction["population"] == "SWR")
    audited[""] = fraction[~is_base]
    with pytest.raises(ValueError, match="base_current_swr: 0 audited rows"):
        content_contrast_values(audited, SOURCES)
    audited = _audited()
    audited["_dm"].loc[audited["_dm"]["term"] == BREADTH, "n_animals"] = 4
    with pytest.raises(ValueError, match="dm_breadth: 4 animals, not 5"):
        content_contrast_values(audited, SOURCES)


def test_breadth_values_turn_the_contrasts_into_ratios_and_compare_lengths():
    content = {v.key: v for v in content_contrast_values(_audited(), SOURCES)}
    events = pd.DataFrame(
        {
            "animal": ["a", "b", "a", "a", "b", "a", "b"],
            "is_swr": [True, True, True, False, False, False, False],
            "n_nonlocal_bins": [10, 20, 30, 4, 6, 8, 100],
        }
    )
    values = {v.key: v.value for v in breadth_derived_values(content, events, "x")}
    assert values == {
        "dm_breadth_ratio": pytest.approx(7.38905609893065),  # e ** 2
        "dm_breadth_length_adjusted_ratio": pytest.approx(1.6487212707001282),
        "dm_breadth_length_share": 0.75,  # 1 - 0.5 / 2
        "event_bins_median_swr": 20.0,
        "event_bins_median_non_swr": 7.0,
        "event_bins_median": 10.0,  # of 4, 6, 8, 10, 20, 30 and 100
        "event_bins_ratio": pytest.approx(20 / 7),
    }


def _estimator_tables(n_animals: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison's summary and bootstrap. Term ``i`` (in the order of
    ``ESTIMATOR_TERMS``) under estimator ``j`` (count, fraction,
    Dirichlet-multinomial) has the median ``i + j / 10``, ``j + 2`` positive
    animals, ``100 + 10 i + j`` sessions, the AIC difference ``-(10 i + j)``
    and a bootstrap interval of the median -+ 0.05."""
    summary, bootstrap = [], []
    for i, (rung, term) in enumerate(ESTIMATOR_TERMS.values()):
        row = {"model": rung, "term": term}
        for j, (word, estimator) in enumerate(ESTIMATORS.items()):
            row |= {
                f"n_animals_{word}": n_animals,
                f"median_coef_{word}": i + j / 10,
                f"n_animals_pos_{word}": j + 2,
                f"n_sessions_{word}": 100 + 10 * i + j,
                f"median_delta_aic_{word}": -(10 * i + j),
            }
            bootstrap.append(
                {
                    "model": rung,
                    "term": term,
                    "estimator": estimator,
                    "p2_5": i + j / 10 - 0.05,
                    "p97_5": i + j / 10 + 0.05,
                }
            )
        summary.append(row)
    bootstrap.append(
        {
            "model": "C2_base",
            "term": BREADTH,
            "estimator": "dirichlet_multinomial",
            "p2_5": 0.4,
            "p97_5": 1.6,
        }
    )
    return pd.DataFrame(summary), pd.DataFrame(bootstrap)


def test_estimator_values_read_each_term_under_each_estimator():
    values = {v.key: v for v in estimator_values(*_estimator_tables(), "x")}
    read = {key: value.value for key, value in values.items()}
    assert read["est_current_count"] == 0.0
    assert read["est_current_dm"] == pytest.approx(0.2)
    assert read["est_swr_current_fraction"] == pytest.approx(1.1)
    assert read["est_swr_current_fraction_low"] == pytest.approx(1.05)
    assert read["est_swr_current_fraction_high"] == pytest.approx(1.15)
    assert read["est_swr_armreward_count_sessions"] == 120
    assert read["est_swr_armreward_dm_sessions"] == 122
    # two terms of one rung
    assert read["est_swr_armvalue_dm"] == pytest.approx(3.2)
    assert read["est_armvalue_dm"] == pytest.approx(4.2)
    assert read["est_swr_armvalue_count_animals_positive"] == 2
    assert read["est_swr_armvalue_dm_animals_positive"] == 4
    assert read["est_armvalue_rung_delta_aic_count"] == -30.0
    assert read["est_armvalue_rung_delta_aic_fraction"] == -31.0
    assert read["est_armvalue_rung_delta_aic_dm"] == -32.0
    assert read["est_breadth_dm_low"] == 0.4
    assert read["est_breadth_dm_high"] == 1.6
    assert values["est_breadth_dm_low"].questions == ("3.1",)
    assert values["est_current_count"].questions == ("3.1", "3.6", "3.7")


def test_estimator_values_refuse_a_missing_animal():
    with pytest.raises(ValueError, match="not 5 animals"):
        estimator_values(*_estimator_tables(n_animals=4), "x")
