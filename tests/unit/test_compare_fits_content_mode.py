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
    _disjoint_rungs_note,
    _md,
    _stratum_power_note,
    audited_term_drop_table,
    build_content_report,
    collect_content_inputs,
    content_delta_aic_table,
    content_rung_session_counts,
    content_scale_table,
    content_separation_table,
    content_term_table,
    contrast_audited_table,
    contrast_design_delta_table,
    contrast_stratum_table,
    rung_sort_key,
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


def test_term_table_counts_unidentified_sessions_and_skips_their_ses():
    """A session that cannot inform the term has a NaN coefficient and infinite
    SEs: it is counted, and it stays out of the SE medians as it does out of the
    coefficient median."""
    new = _new()
    new[f"bse_cluster_{TERM}"] = [0.1, 0.2, 0.3, np.inf]
    new[f"bse_{TERM}"] = [0.01, 0.02, 0.03, np.inf]
    new.loc[3, f"coef_{TERM}"] = np.nan
    new["unidentified_terms"] = ["", "", "", f"{TERM}:separated_to_minus_inf"]

    row = content_term_table(_old(), new).set_index("term").loc[TERM]

    assert row["n_unidentified_after"] == 1
    assert row["n_unidentified_before"] == 0
    assert row["n_sessions_after"] == 3
    assert row["bse_cluster_median_after"] == pytest.approx(0.2)
    assert row["bse_model_median_after"] == pytest.approx(0.02)


def test_the_drop_table_counts_unidentified_sessions_as_the_gate_does():
    """``n_fitted`` and the share include the sessions that cannot inform the
    term, as the aggregation's refusal gate does; a frame from before the
    column reports none."""
    audited = pd.DataFrame(
        {
            "model": ["CS1", "CS4"],
            "population": ["all", "all"],
            "cohort": ["at_well", "at_well"],
            "term": ["a", "b"],
            "n_sessions": [16, 18],
            "n_dropped": [2, 1],
            "n_unidentified": [2, 1],
        }
    )

    table = audited_term_drop_table(audited).set_index("term")

    assert table.loc["a", "n_fitted"] == 20
    assert table.loc["a", "pct_dropped"] == pytest.approx(20.0)
    assert table.loc["b", "pct_dropped"] == pytest.approx(10.0)
    legacy = audited_term_drop_table(audited.drop(columns="n_unidentified"))
    assert (legacy["n_unidentified"] == 0).all()


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


_CONTRAST_BOTH_SIDES = (
    "swr_contrast_content_audited.pkl",
    "swr_contrast_content_arm.pkl",
)
_CONTRAST_STRATA = (
    "swr_contrast_content_audited_rewarded.pkl",
    "swr_contrast_content_audited_omitted.pkl",
)


def _contrast_artifacts(old_root, new_root, skip=()):
    """Empty is_swr-contrast artifacts in each root's ``models_swr_contrast/``."""
    for root, names in (
        (old_root, _CONTRAST_BOTH_SIDES),
        (new_root, _CONTRAST_BOTH_SIDES + _CONTRAST_STRATA),
    ):
        (root / "models_swr_contrast").mkdir(parents=True, exist_ok=True)
        for name in names:
            if name not in skip:
                pd.DataFrame().to_pickle(root / "models_swr_contrast" / name)


def test_a_missing_contrast_artifact_fails_naming_its_directory(tmp_path):
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    _contrast_artifacts(old_root, new_root)
    (new_root / "models_swr_contrast" / _CONTRAST_BOTH_SIDES[0]).unlink()

    with pytest.raises(SystemExit) as exc:
        collect_content_inputs(old_root, new_root, ())

    message = str(exc.value)
    assert _CONTRAST_BOTH_SIDES[0] in message
    assert str(new_root / "models_swr_contrast") in message


def test_an_outcome_stratum_not_refit_yet_is_named_not_required(tmp_path):
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    _contrast_artifacts(old_root, new_root, skip=_CONTRAST_STRATA[1:])

    _, contrast, _, missing = collect_content_inputs(old_root, new_root, ())

    assert contrast["stratum_omitted"].empty
    assert any("omitted" in line and _CONTRAST_STRATA[1] in line for line in missing)


def test_contrast_artifacts_left_in_models_are_not_read(tmp_path):
    """No fallback to ``models/``, where the fits lived before 2026-10-24."""
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    _contrast_artifacts(old_root, new_root)
    legacy = old_root / "models"
    legacy.mkdir()
    for name in _CONTRAST_BOTH_SIDES:
        (old_root / "models_swr_contrast" / name).rename(legacy / name)

    with pytest.raises(SystemExit, match="models_swr_contrast"):
        collect_content_inputs(old_root, new_root, ())


def test_a_bundle_present_on_only_one_side_is_named_not_dropped(tmp_path):
    """A refit that has not run yet must be visible in the audit, not omitted."""
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    (old_root / "models").mkdir(parents=True)
    (new_root / "models").mkdir(parents=True)
    _old().to_pickle(old_root / "models" / "all_models.pkl")
    _contrast_artifacts(old_root, new_root)

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
    _contrast_artifacts(old_root, new_root)

    bundles, contrast, inputs, missing = collect_content_inputs(
        old_root, new_root, ("models",)
    )
    report = build_content_report(bundles, contrast, inputs, "; ".join(missing))

    assert "different scales" in report
    assert "event-clustered" in report
    assert "conditional_poisson_fraction" in report
    # The contrast artifacts read are named in the input table.
    assert "swr_contrast_content_audited.pkl" in report


def test_a_rung_whose_design_changed_is_named_before_its_coefficients_are_read():
    """A rung that gained a control term is not a pure response comparison."""
    old = _fits({"a": [0.5, 0.6]}, response=None, bse=0.05, bse_cluster=None).assign(
        model="CS1_current"
    )
    new = old.assign(**{"coef_log_n_nonlocal_bins_z_x_is_current_arm": 0.2})
    new = new.assign(response_col="arm_fraction")

    table = content_term_table(old, new)  # coefficients still join fine
    delta = contrast_design_delta_table(old, new)

    assert TERM in set(table["term"])
    assert len(delta) == 1
    assert delta.loc[0, "model"] == "CS1_current"
    assert "log_n_nonlocal_bins_z_x_is_current_arm" in delta.loc[0, "only_after"]
    assert delta.loc[0, "only_before"] == "—"


def test_identical_designs_produce_no_design_delta_rows():
    assert contrast_design_delta_table(_old(), _new()).empty


def test_a_rung_absent_before_is_marked_not_fit():
    old = _old().assign(model="CS1_current")
    new = pd.concat(
        [old.assign(response_col="arm_fraction"), _new().assign(model="CS5b_new")],
        ignore_index=True,
    )

    delta = contrast_design_delta_table(old, new).set_index("model")

    assert delta.loc["CS5b_new", "only_before"] == "(rung not fit)"


# --- Fix round: units, pairing, empty rows, integer rendering -------------------


def _audited(std_factor: float, coef: float, **overrides) -> pd.DataFrame:
    """One audited contrast row, at a stated per-SD unit."""
    row = {
        "level": "candidate_arm",
        "model": "CS5_armvalue",
        "population": "all",
        "term": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
        "n_sessions": 212,
        "n_animals": 5,
        "n_dropped": 6,
        "coef_animal_median": coef,
        "n_animals_pos": 0,
        "coef_session_median": coef,
        "typical_session_se": 0.3,
        "se_source": "model",
        "std_unit": "per SD",
        "std_factor": std_factor,
        "std_coef_animal_median": coef * std_factor,
        "session_signrank_fdr_q": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_the_per_sd_unit_of_each_side_is_printed_beside_its_std_coef():
    """A std_coef stated per a redefined SD must carry the SD it is stated per.

    Ruling 5 moved the arm belief SD onto leaf-arm support (0.2493 -> 0.1190), so
    a before/after std_coef pair mixes a unit redefinition with an effect change
    unless both factors are on the page.
    """
    before = _audited(std_factor=1 / 0.249346, coef=-0.996)
    after = _audited(std_factor=1 / 0.118986, coef=0.2366, se_source="cluster")

    table = contrast_audited_table(before, after)

    assert "std_factor_before" in table.columns
    assert "std_factor_after" in table.columns
    assert "std_unit_before" in table.columns
    assert "std_unit_after" in table.columns
    assert table.loc[0, "std_factor_before"] != pytest.approx(
        table.loc[0, "std_factor_after"]
    )


def test_the_report_warns_that_the_per_sd_unit_changed(tmp_path):
    old_root, new_root = tmp_path / "backup", tmp_path / "live"
    (old_root / "models").mkdir(parents=True)
    (new_root / "models").mkdir(parents=True)
    _old().to_pickle(old_root / "models" / "all_models.pkl")
    _new().to_pickle(new_root / "models" / "all_models.pkl")
    _contrast_artifacts(old_root, new_root)

    bundles, contrast, inputs, missing = collect_content_inputs(
        old_root, new_root, ("models",)
    )
    report = build_content_report(bundles, contrast, inputs, "; ".join(missing))

    assert "per-SD unit also changed" in report
    assert "must not be differenced across" in report


def test_strata_carry_their_per_sd_factor():
    audited = _audited(std_factor=8.4, coef=0.24)
    strata = {"all": audited, "rewarded": audited.assign(coef_animal_median=0.16)}

    table = contrast_stratum_table(
        strata, {"swr_x_reward_belief_mean_candidate_arm_preoutcome"}
    )

    assert table.loc[0, "std_factor_all"] == pytest.approx(8.4)
    assert table.loc[0, "std_factor_rewarded"] == pytest.approx(8.4)


def test_a_rung_is_not_given_the_terms_of_other_rungs_in_the_same_bundle():
    """A bundle frame is the union of every rung's design; a rung owns its own."""
    m1 = _fits({"a": [0.5, 0.6]}, response="arm_fraction", bse=0.05, bse_cluster=0.3)
    m2 = m1.assign(model="M2")
    m2["coef_is_prev_arm"] = 0.2
    bundle = pd.concat([m1, m2], ignore_index=True)

    table = content_term_table(bundle, bundle)

    m1_terms = set(table.loc[table.model == "M1", "term"])
    m2_terms = set(table.loc[table.model == "M2", "term"])
    assert m1_terms == {TERM}  # not is_prev_arm, which only M2 was fit with
    assert m2_terms == {TERM, "is_prev_arm"}
    # And no all-zero placeholder row survives.
    assert (table["n_sessions_before"] > 0).all()


def test_paired_medians_use_only_the_sessions_both_fits_reached():
    """An unpaired median can move because the cohort moved, not the effect."""
    old = _fits({"a": [1.0, 1.0, 1.0]}, response=None, bse=0.05, bse_cluster=None)
    # Same first two sessions with the same coefficient, plus a third session the
    # old fit never reached carrying a wildly different value.
    new = _fits(
        {"a": [1.0, 1.0, 1.0, 9.0]},
        response="arm_fraction",
        bse=0.05,
        bse_cluster=0.3,
    )

    row = content_term_table(old, new).set_index("term").loc[TERM]

    assert row["n_sessions_before"] == 3
    assert row["n_sessions_after"] == 4
    assert row["n_paired"] == 3
    assert row["coef_median_after"] == pytest.approx(1.0)  # unpaired median
    assert row["coef_median_paired_before"] == pytest.approx(1.0)
    assert row["coef_median_paired_after"] == pytest.approx(1.0)


def test_paired_columns_are_empty_when_no_session_is_shared():
    old = _old().assign(model="Mv")
    new = _new().assign(model="M1")

    table = content_term_table(old, new).set_index("model")

    assert table.loc["Mv", "n_paired"] == 0
    assert np.isnan(table.loc["Mv", "coef_median_paired_before"])


def test_count_columns_render_without_decimals():
    frame = pd.DataFrame([{"n_sessions_before": 218.0, "coef_median_before": 0.8732}])

    rendered = _md(frame, floatfmt="{:.4f}")

    assert "| 218 |" in rendered
    assert "0.8732" in rendered
    assert "218.0000" not in rendered


def test_a_bundle_whose_rungs_do_not_overlap_says_so():
    """`models/` held a partial belief run before the guard existed."""
    old = pd.concat(
        [_old().assign(model=name) for name in ("Mv", "Mu", "Mvu")], ignore_index=True
    )
    new = pd.concat(
        [_new().assign(model=name) for name in ("M0a", "M1")], ignore_index=True
    )

    note = _disjoint_rungs_note(old, new)

    assert "No rung is present on both sides" in note
    assert "Mu, Mv, Mvu" in note
    assert "M0a, M1" in note
    # A bundle that does share rungs gets no note.
    assert _disjoint_rungs_note(new, new) == ""


def test_the_disjointness_note_states_no_cause_the_frames_cannot_show():
    """WHY two sides are disjoint is bundle history, supplied by the caller.

    The rung sets establish only THAT they are disjoint; asserting a cause for
    every bundle would attribute one bundle's history to all of them.
    """
    old = _old().assign(model="Mu")
    new = _new().assign(model="M1")

    derived = _disjoint_rungs_note(old, new)
    explained = _disjoint_rungs_note(old, new, note="A partial run published here.")

    assert "fitted disjoint rung sets" in derived
    assert "partial" not in derived
    assert explained.startswith(derived.rstrip("\n"))
    assert "A partial run published here." in explained


def test_a_thin_stratum_reports_how_many_sessions_it_rests_on():
    table = pd.DataFrame(
        [
            {"n_sessions_all": 218.0, "n_sessions_omitted": 11.0},
            {"n_sessions_all": 216.0, "n_sessions_omitted": 13.0},
        ]
    )

    note = _stratum_power_note(table, ["all", "omitted"])

    assert "`all` 216-218" in note
    assert "`omitted` 11-13" in note
    assert "not population evidence" in note


def test_an_empty_stratum_stays_in_the_table_and_reports_zero_sessions():
    """A stratum that fitted nothing is a finding, not a column to drop."""
    audited = _audited(std_factor=8.4, coef=0.24)
    terms = {"swr_x_reward_belief_mean_candidate_arm_preoutcome"}

    table = contrast_stratum_table({"all": audited, "omitted": pd.DataFrame()}, terms)

    assert "n_sessions_omitted" in table.columns
    assert "coef_animal_median_omitted" in table.columns
    assert table["n_sessions_omitted"].isna().all()
    assert table.loc[0, "coef_animal_median_all"] == pytest.approx(0.24)
    assert "`omitted` 0 sessions" in _stratum_power_note(table, ["all", "omitted"])


def test_a_missing_label_renders_as_an_em_dash_not_the_string_nan():
    """An outer join leaves NaN in label columns; "nan" reads as a value."""
    frame = pd.DataFrame([{"se_source_after": np.nan, "n_sessions_after": np.nan}])

    rendered = _md(frame)

    assert "nan" not in rendered
    assert rendered.count("—") == 2


def test_each_rung_prints_next_to_its_uncontrolled_twin():
    """The direct and total SWR effects of one term are read as a pair.

    Alphabetically ``CS5b_valuevsprevgoal`` falls between ``CS5_armvalue`` and
    its twin, which would put the pair on either side of a third rung.
    """
    term = "swr_x_reward_belief_mean_candidate_arm_preoutcome"
    rungs = ["CS5nt_armvalue", "CS5b_valuevsprevgoal", "CS5_armvalue", "CS1_current"]
    audited = pd.concat(
        [_audited(std_factor=8.4, coef=0.2).assign(model=name) for name in rungs],
        ignore_index=True,
    ).assign(term=term)

    table = contrast_audited_table(audited, audited, {term})

    assert list(table["model"]) == [
        "CS1_current",
        "CS5_armvalue",
        "CS5nt_armvalue",
        "CS5b_valuevsprevgoal",
    ]
    assert rung_sort_key("CS5nt_armvalue") == ("CS5_armvalue", 1)
    assert rung_sort_key("CPS1nt_current") == ("CPS1_current", 1)
    assert rung_sort_key("C13_armuncert") == ("C13_armuncert", 0)
