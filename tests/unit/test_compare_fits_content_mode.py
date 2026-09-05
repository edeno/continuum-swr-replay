"""The content audit must state the response scales it is joining, not blend them.

The content ladders changed response, not specification, so a reader of the audit
needs three things kept straight: coefficients ARE comparable term for term; the
"before" standard error is a model SE and the "after" one the aggregation reads
is an event-clustered sandwich; and the AIC-scale columns are per timebin before
and per event after, so they must appear as separate columns and never be
differenced. These tests pin each of those, plus the per-animal sign count that
carries the population claim, on tiny synthetic bundles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.qa.compare_fits import (
    build_content_report,
    collect_content_inputs,
    content_delta_aic_table,
    content_rung_session_counts,
    content_scale_table,
    content_separation_table,
    content_term_table,
    contrast_stratum_table,
)

TERM = "is_current_arm"
SC = ["animal", "session_date", "epoch"]


def _fits(
    coefs: dict[str, list[float]],
    *,
    response: str | None,
    bse: float,
    bse_cluster: float | None,
    model: str = "M1",
    sd: float | None = 1.0,
    separated_terms: str | None = "",
) -> pd.DataFrame:
    """Per-session fits for one rung; ``response=None`` is a pre-integrity frame."""
    rows = []
    for animal, values in coefs.items():
        for i, value in enumerate(values):
            row = {
                "animal": animal,
                "session_date": 20200101 + i,
                "epoch": "02_r1",
                "model": model,
                f"coef_{TERM}": value,
                f"bse_{TERM}": bse,
            }
            if bse_cluster is not None:
                row[f"bse_cluster_{TERM}"] = bse_cluster
            if sd is not None:
                row[f"sd_{TERM}"] = sd
            if separated_terms is not None:
                row["separated_terms"] = separated_terms
            if response is not None:
                row["response_col"] = response
                row["estimator"] = (
                    "conditional_poisson_fraction"
                    if response.endswith("_fraction")
                    else "conditional_poisson_count"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _old() -> pd.DataFrame:
    return _fits(
        {"a": [0.8, 1.0], "b": [1.2], "c": [-0.4]},
        response=None,
        bse=0.05,
        bse_cluster=None,
    )


def _new() -> pd.DataFrame:
    return _fits(
        {"a": [0.8, 1.0], "b": [1.2], "c": [0.4]},
        response="arm_fraction",
        bse=0.05,
        bse_cluster=0.30,
    )


def test_se_columns_keep_the_model_and_clustered_medians_apart():
    """The old SE and the SE the aggregation now reads are different quantities."""
    table = content_term_table(_old(), _new()).set_index("term").loc[TERM]

    assert table["bse_model_median_before"] == pytest.approx(0.05)
    assert table["bse_cluster_median_after"] == pytest.approx(0.30)
    # The new fits' own model SE is reported too, so the 6x gap between a model
    # SE and its clustered counterpart cannot be misread as an SE that grew
    # because the coefficients moved.
    assert table["bse_model_median_after"] == pytest.approx(0.05)


def test_a_count_fit_with_no_clustered_se_reports_no_clustered_median():
    """A pre-fix frame has no ``bse_cluster_`` column; the cell must stay empty."""
    table = content_term_table(_old(), _old()).set_index("term").loc[TERM]

    assert np.isnan(table["bse_cluster_median_after"])


def test_per_animal_sign_counts_are_medians_within_animal():
    """One animal's many sessions must not outvote the others' sign."""
    table = content_term_table(_old(), _new()).set_index("term").loc[TERM]

    # animal a (median +0.9) and b (+1.2) are positive in both fits; c flips
    # from -0.4 to +0.4, so the count moves 2/3 -> 3/3 rather than 4/5 -> 5/5.
    assert table["n_animals_pos_before"] == "2/3"
    assert table["n_animals_pos_after"] == "3/3"


def test_the_response_scale_of_each_side_is_reported():
    scale = content_scale_table(_old(), _new()).set_index("fit")

    assert scale.loc["before", "response_col"] == "(unstated)"
    assert scale.loc["before", "estimator"] == "(unstated)"
    assert scale.loc["after", "response_col"] == "arm_fraction"
    assert scale.loc["after", "estimator"] == "conditional_poisson_fraction"
    assert scale.loc["after", "n_sessions"] == 4


def test_delta_aic_before_and_after_stay_in_separate_columns():
    """The two AIC scales are joined on the step, never combined into one number."""
    old_cmp = pd.DataFrame(
        [{"parent": "M0", "child": "M1", "n_sessions": 218, "delta_aic_median": 989.7}]
    )
    new_cmp = pd.DataFrame(
        [{"parent": "M0", "child": "M1", "n_sessions": 215, "delta_aic_median": 41.1}]
    )

    table = content_delta_aic_table(old_cmp, new_cmp)

    assert list(table.columns) == [
        "parent",
        "child",
        "n_sessions_before",
        "delta_aic_median_before",
        "n_sessions_after",
        "delta_aic_median_after",
    ]
    assert table.loc[0, "delta_aic_median_before"] == pytest.approx(989.7)
    assert table.loc[0, "delta_aic_median_after"] == pytest.approx(41.1)


def test_rung_counts_show_a_rung_missing_from_one_side_as_zero():
    old = pd.concat([_old(), _old().assign(model="M2")], ignore_index=True)

    counts = content_rung_session_counts(old, _new()).set_index("model")

    assert counts.loc["M1", "n_sessions_before"] == 4
    assert counts.loc["M1", "n_sessions_after"] == 4
    assert counts.loc["M2", "n_sessions_after"] == 0


def test_separation_counts_split_the_flag_from_the_effect_scale_backstop():
    """A runaway |coef|*sd is not a separation diagnosis and is counted apart."""
    old = _fits({"a": [0.5, 20.0]}, response=None, bse=0.05, bse_cluster=None)
    new = _fits(
        {"a": [0.5, 0.6]},
        response="arm_fraction",
        bse=0.05,
        bse_cluster=0.3,
        separated_terms=TERM,
    )

    table = content_separation_table(old, new).set_index("term").loc[TERM]

    assert table["n_flagged_before"] == 0
    assert table["n_runaway_before"] == 1
    assert table["n_flagged_after"] == 2
    assert table["n_runaway_after"] == 0


def test_separation_table_omits_terms_neither_fit_excludes():
    assert content_separation_table(_old(), _new()).empty


def test_strata_are_joined_side_by_side_on_the_spec():
    audited = pd.DataFrame(
        [
            {
                "level": "candidate_arm",
                "model": "CS1_current",
                "population": "all",
                "term": "swr_x_is_current_arm",
                "n_sessions": 200,
                "coef_animal_median": 0.4,
                "n_animals_pos": 5,
            }
        ]
    )
    strata = {
        "all": audited,
        "rewarded": audited.assign(coef_animal_median=0.6, n_sessions=120),
    }

    table = contrast_stratum_table(strata, {"swr_x_is_current_arm"})

    assert len(table) == 1
    assert table.loc[0, "coef_animal_median_all"] == pytest.approx(0.4)
    assert table.loc[0, "coef_animal_median_rewarded"] == pytest.approx(0.6)
    assert table.loc[0, "n_sessions_rewarded"] == 120


def test_a_bundle_present_on_only_one_side_is_named_not_dropped(tmp_path):
    """A refit that has not run yet must be visible in the audit, not omitted."""
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    (old_root / "models").mkdir(parents=True)
    (new_root / "models").mkdir(parents=True)
    _old().to_pickle(old_root / "models" / "all_models.pkl")

    bundles, contrast, inputs, missing = collect_content_inputs(
        old_root, new_root, ("models", "models_patch")
    )

    assert set(bundles) == {"models"}
    assert bundles["models"][1].empty  # no "after" side yet
    assert any("no refit bundle" in line for line in missing)
    assert any("models_patch" in line for line in missing)
    assert contrast["audited_before"].empty


def test_the_report_states_that_the_aic_scales_differ(tmp_path):
    """The scale warning is not optional prose -- a reader must not difference them."""
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    (old_root / "models").mkdir(parents=True)
    (new_root / "models").mkdir(parents=True)
    _old().to_pickle(old_root / "models" / "all_models.pkl")
    _new().to_pickle(new_root / "models" / "all_models.pkl")

    bundles, contrast, inputs, missing = collect_content_inputs(
        old_root, new_root, ("models",)
    )
    report = build_content_report(bundles, contrast, inputs, "; ".join(missing))

    assert "different scales" in report
    assert "event-clustered" in report
    assert "conditional_poisson_fraction" in report
    # The one-sided artifacts are named rather than silently absent.
    assert "swr_contrast_content_audited.pkl" in report
