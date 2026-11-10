"""Part 0's values are what the literature survey says, under the stated rules.

The introduction quotes modes, ranges and shares from the survey of
Supplementary Table S1. These tests check the rules on small surveys -- only
bare numbers count, a cell of an unlisted form is an error, a tied mode is an
error, the trigger shares match their stated definitions -- pin a few values
checked by hand against the survey, and check that the committed
``paper/reported/part0.json`` holds what the survey in the repository gives, so
a corrected survey cannot leave a stale number in the text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.reported import part0, values

SUMMARY = Path(__file__).resolve().parents[2] / "paper" / "reported" / "part0.json"


@pytest.fixture(scope="module")
def fresh() -> values.Summary:
    return part0.summarize(part0.default_inputs(), "by", "commit")


def test_only_bare_numbers_count():
    survey = pd.DataFrame(
        {
            "x": [
                "3",
                " 2.5 ",
                ">4",
                "1-3",
                "1.8-1.9",
                "15% of ensemble",
                "1/3 of template cells",
                "Adaptive",
                "Adaptive (25.6 ms base bins)",
                "#N/A",
                "Not reported",
                "1 (inferred)",
                "20 (track); 80 (arena)",
            ]
        }
    )
    numbers = part0.numeric(survey)["x"]
    assert numbers.iloc[:2].tolist() == [3.0, 2.5]
    assert numbers.iloc[2:].isna().all()


@pytest.mark.parametrize("cell", ["4 cm/s", "", "inf", "1e2", "2,5", "~3", "three"])
def test_a_cell_of_an_unlisted_form_is_an_error(cell):
    survey = pd.DataFrame({"Animal Speed (cm/s)": ["5", cell]})
    with pytest.raises(ValueError, match="neither a bare number"):
        part0.numeric(survey)


def test_tied_mode_is_an_error():
    with pytest.raises(ValueError, match="tied"):
        part0._mode(pd.Series([2.0, 2.0, 3.0, 3.0]), "threshold")
    assert part0._mode(pd.Series([2.0, 3.0, 3.0]), "threshold") == 3.0


def test_a_parameter_with_no_numeric_cell_is_an_error():
    numbers = pd.DataFrame({c: [1.0, 1.0, 2.0] for c in part0.PARAMETER_COLUMNS})
    numbers["Animal Speed (cm/s)"] = float("nan")
    with pytest.raises(ValueError, match="speed: no numeric cells"):
        part0.parameter_values(numbers, "s")


def test_whole_values_print_whole_and_others_keep_figures():
    assert part0._quantity("speed_min", 1.0, "cm/s", "d", "s").style == "int"
    assert part0._quantity("speed_min", 2.5, "cm/s", "d", "s").style == "sig2"


def test_trigger_and_decoder_shares():
    survey = pd.DataFrame(
        {
            "Detection": ["MUA", "MUA", "SWR", "SWR, MUA", "MUA,SWR", "Decoding"],
            "Decoding": [
                "Bayes",
                "UMAP\nBayes",
                "Rank order correlation",
                "bayes",
                "Bayesian decoder",
                "non-Bayesian state space",
            ],
        }
    )
    shares = {v.key: v.value for v in part0.method_values(survey, "s")}
    assert shares["papers"] == 6
    assert shares["mua_only_fraction"] == pytest.approx(2 / 6)
    assert shares["swr_only_fraction"] == pytest.approx(1 / 6)
    assert shares["swr_and_mua_fraction"] == pytest.approx(2 / 6)
    assert shares["bayes_fraction"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("cell", ["SWR + MUA", "LFP", ""])
def test_an_unknown_trigger_is_an_error(cell):
    with pytest.raises(ValueError, match="Detection names"):
        part0.triggers(pd.Series(["SWR", cell]))


def test_threshold_pools_the_ripple_and_multiunit_columns():
    numbers = pd.DataFrame(
        {
            "SWR Z-score Thresh. (STD)": [3.0, 2.0, float("nan")],
            "MUA Z-score Thresh. (STD)": [3.0, float("nan"), 4.0],
            **{
                column: [1.0, 1.0, 2.0]
                for column in part0.PARAMETER_COLUMNS
                if "Thresh." not in column
            },
        }
    )
    pooled = {v.key: v.value for v in part0.parameter_values(numbers, "s")}
    assert pooled["threshold_mode"] == 3.0
    assert pooled["threshold_min"] == 2.0
    assert pooled["threshold_max"] == 4.0
    assert pooled["speed_mode"] == 1.0


def test_survey_values_checked_by_hand(fresh):
    # Counted from paper/tables/table_s1_literature_parameters.csv by hand, so a
    # wrong column or a swapped bound fails here even after a regeneration.
    expected = {
        "papers": 57,
        "threshold_mode": 3,
        "threshold_min": 1,
        "threshold_max": 8,
        "band_low_mode": 150,
        "band_high_mode": 250,
        "speed_mode": 5,
        "cells_mode": 5,
        "shortest_mode": 50,
        "shortest_min": 15,
        "longest_mode": 500,
        "longest_min": 300,
        "longest_max": 2000,
        "time_bin_mode": 20,
        "position_bin_mode": 2,
        "mua_only_fraction": 24 / 57,
        "swr_only_fraction": 16 / 57,
        "swr_and_mua_fraction": 13 / 57,
        "bayes_fraction": 46 / 57,
    }
    got = {k: fresh.values[k].value for k in expected}
    assert got == pytest.approx(expected)


def test_committed_summary_is_current(fresh):
    committed = values.read_summary(SUMMARY)
    rebuild = (
        "run `uv run --frozen python scripts/reported/summarize_part.py --part 0` "
        "and `make reported-values`"
    )
    assert committed.inputs == fresh.inputs, f"the survey changed: {rebuild}"
    assert committed.values == fresh.values, f"part0.json is stale: {rebuild}"
