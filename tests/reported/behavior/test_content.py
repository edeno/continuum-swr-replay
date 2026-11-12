"""The behavior summary's content values: the content-contrast, concentration
and estimator values, the leaf ladder, the horse race, the trunk shares and the
paired current-arm difference, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import content
from continuum_swr_replay.reported.behavior.content import (
    CONTENT_SPECS,
    ESTIMATOR_TERMS,
    ESTIMATORS,
    breadth_derived_values,
    content_contrast_values,
    current_arm_paired_values,
    estimator_values,
    horserace_values,
    trunk_values,
)
from tests.support._behavior_fits import coefficient_fits, values_by_key

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
    with pytest.raises(ValueError, match="dm_breadth: 4 animals, not the 5 analysed"):
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
                    "n_animals": n_animals,
                    "low": i + j / 10 - 0.05,
                    "high": i + j / 10 + 0.05,
                }
            )
        summary.append(row)
    bootstrap.append(
        {
            "model": "C2_base",
            "term": BREADTH,
            "estimator": "dirichlet_multinomial",
            "n_animals": n_animals,
            "low": 0.4,
            "high": 1.6,
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
    assert "small-sample t correction over the 5 animals" in (
        values["est_current_dm_low"].description
    )


def test_estimator_values_refuse_a_percentile_bootstrap():
    summary, bootstrap = _estimator_tables()
    old = bootstrap.rename(columns={"low": "p2_5", "high": "p97_5"})
    with pytest.raises(ValueError, match="holds percentiles"):
        estimator_values(summary, old, "x")


def test_estimator_values_refuse_a_missing_animal():
    with pytest.raises(ValueError, match="4 animals, not the 5 analysed"):
        estimator_values(*_estimator_tables(n_animals=4), "x")


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
                "session_date": "d1",
                "epoch": "02_r1",
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
    assert values["trunk_share_swr"].value == pytest.approx(1.6 / 3)
    assert "over the 3 animals" in values["trunk_share_swr"].description
    assert values["trunk_share_non_swr"].value == pytest.approx(1.1 / 3)
    assert values["trunk_share_animals_swr_higher"].value == 2
    # the current patch's trunk, SWR: 0.2, 0.1, 0.6; non-SWR: 0.1, 0.3, 0.05
    assert values["trunk_share_current_patch_swr"].value == pytest.approx(0.3)
    assert values["trunk_share_current_patch_non_swr"].value == pytest.approx(0.15)
    assert values["trunk_share_current_patch_animals_swr_higher"].value == 2
    # the other trunks, SWR: 0.3, 0.2, 0.2; non-SWR: 0.1, 0.3, 0.25 (only a's rises)
    assert values["trunk_share_other_patches_swr"].value == pytest.approx(0.7 / 3)
    assert values["trunk_share_other_patches_non_swr"].value == pytest.approx(0.65 / 3)
    assert values["trunk_share_other_patches_animals_swr_higher"].value == 1
    # SWR: 0.2 / 0.5, 0.1 / 0.3, 0.6 / 0.8; non-SWR: 0.1 / 0.2, 0.3 / 0.6, 0.05 / 0.3
    assert values["trunk_current_patch_fraction_swr"].value == pytest.approx(
        (0.4 + 1 / 3 + 0.75) / 3
    )
    assert values["trunk_current_patch_fraction_non_swr"].value == pytest.approx(
        (0.5 + 0.5 + 1 / 6) / 3
    )


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


def _rung_fit(aic, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """One rung's fit statistics in the six sessions, beside ``frame``'s columns."""
    fit = frame if frame is not None else coefficient_fits("unused", [0.0] * 6, 0.1)
    return fit.assign(llf=-40.0, aic=aic, bic=aic, n_params=3, n_events=50)


def test_leaf_values_read_each_rung_s_own_term_and_step(tmp_path):
    rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # median 0.35
    # Each step's per-session AIC(child) - AIC(parent), sessions in ANIMALS order
    # (a, a, b, c, d, e):
    # M4 -> Mu  [-10, -6, -1, -3, 4, -5]: median -4, four below -2; animal
    #           medians -8, -1, -3, 4, -5, four negative. The child is better.
    # M4 -> Mv  [3, 1, 5, -4, 2, 6]: median 2.5, one below -2; animal medians
    #           2, 5, -4, 2, 6, one negative. The parent is better.
    # Mu -> Mvu [-1, 0.5, -2.5, 1.5, -0.5, 3]: median 0, one below -2; animal
    #           medians -0.25, -2.5, 1.5, -0.5, 3, three negative.
    m4 = np.full(6, 100.0)
    mu = m4 + [-10, -6, -1, -3, 4, -5]
    mv = m4 + [3, 1, 5, -4, 2, 6]
    mvu = mu + [-1, 0.5, -2.5, 1.5, -0.5, 3]
    _rung_fit(m4).to_pickle(tmp_path / "M4.pkl")
    _rung_fit(
        mu,
        coefficient_fits(
            "reward_belief_variance_candidate_arm_preoutcome_z", rising, 0.05
        ),
    ).to_pickle(tmp_path / "Mu.pkl")
    _rung_fit(
        mv,
        coefficient_fits(
            "reward_belief_mean_candidate_arm_preoutcome",
            [-2 * c for c in rising],
            0.07,
        ),
    ).to_pickle(tmp_path / "Mv.pkl")
    _rung_fit(mvu).to_pickle(tmp_path / "Mvu.pkl")
    values = values_by_key(content.leaf_values(tmp_path, "source"))
    assert values["leaf_uncertainty"] == pytest.approx(0.35)
    assert values["leaf_value"] == pytest.approx(-0.7)
    assert values["leaf_uncertainty_sessions"] == 6
    assert values["leaf_uncertainty_animals_positive"] == 5
    assert values["leaf_value_animals_positive"] == 0
    assert values["leaf_uncertainty_se"] == pytest.approx(0.05)
    assert values["leaf_value_se"] == pytest.approx(0.07)
    assert values["leaf_step_uncertainty"] == -4.0
    assert values["leaf_step_uncertainty_clear"] == 4
    assert values["leaf_step_uncertainty_animals"] == 4
    assert values["leaf_step_value"] == 2.5
    assert values["leaf_step_value_clear"] == 1
    assert values["leaf_step_value_sessions"] == 6
    assert values["leaf_step_value_animals"] == 1
    assert values["leaf_step_both"] == 0.0
    assert values["leaf_step_both_clear"] == 1
    assert values["leaf_step_both_animals"] == 3


def test_leaf_steps_ignore_a_stored_comparison(tmp_path):
    """A bundle's model_comparison.pkl written in the old sign (parent minus
    child) is not read: the step comes from the rungs' own AICs."""
    m4 = np.full(6, 100.0)
    mu = m4 - 10.0
    for name, aic in (("M4", m4), ("Mvu", mu)):
        _rung_fit(aic).to_pickle(tmp_path / f"{name}.pkl")
    for name, aic, term in (
        ("Mu", mu, "reward_belief_variance_candidate_arm_preoutcome_z"),
        ("Mv", m4, "reward_belief_mean_candidate_arm_preoutcome"),
    ):
        _rung_fit(aic, coefficient_fits(term, [0.1] * 6, 0.05)).to_pickle(
            tmp_path / f"{name}.pkl"
        )
    pd.DataFrame(
        {"parent": ["M4"], "child": ["Mu"], "n_animals": 5, "delta_aic_median": 10.0}
    ).to_pickle(tmp_path / "model_comparison.pkl")
    values = values_by_key(content.leaf_values(tmp_path, "source"))
    assert values["leaf_step_uncertainty"] == -10.0


def test_horserace_values_name_each_population_and_predictor():
    predictors = {
        "reward_belief_variance_candidate_arm_preoutcome_z": 0.03,
        "reward_belief_mean_candidate_arm_preoutcome_z": -0.001,
        "distance_to_candidate_arm_cm_z": -0.01,
        "visit_recency_trials_z": 0.002,
    }
    summary = pd.DataFrame(
        [
            {
                "population": pop,
                "predictor": p,
                "median": m,
                "n_animals_consistent": 4,
                "median_cond_number": 3.0 + i,
            }
            for i, pop in enumerate(("all", "SWR", "non_SWR"))
            for p, m in predictors.items()
        ]
    )
    values = {v.key: v for v in horserace_values(summary, "x")}
    assert values["horserace_uncertainty_swr"].value == pytest.approx(0.03)
    assert values["horserace_value_non_swr_animals"].value == 4
    assert values["horserace_condition_number_non_swr"].value == pytest.approx(5.0)
    assert len(values) == 3 * (4 * 2 + 1)
