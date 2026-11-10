"""Part 0's values are what the literature survey says, under the stated rules.

The introduction quotes modes, ranges and shares from the 57-paper survey
(Supplementary Table S1). These tests check the rules on small surveys -- only
bare numbers count, a tied mode is an error, the trigger shares match their
stated definitions -- and that the committed ``paper/reported/part0.json`` holds
the values the installed survey gives, so a corrected survey cannot leave a
stale number in the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.reported import part0

SUMMARY = Path(__file__).resolve().parents[2] / "paper" / "reported" / "part0.json"


def test_only_bare_numbers_count():
    survey = pd.DataFrame(
        {
            "x": [
                "3",
                " 2.5 ",
                ">4",
                "1-3",
                "Adaptive",
                "#N/A",
                "Not reported",
                "1 (inferred)",
            ]
        }
    )
    numbers = part0.numeric(survey)["x"]
    assert numbers.iloc[:2].tolist() == [3.0, 2.5]
    assert numbers.iloc[2:].isna().all()


def test_tied_mode_is_an_error():
    with pytest.raises(ValueError, match="tied"):
        part0._mode(pd.Series([2.0, 2.0, 3.0, 3.0]), "threshold")
    assert part0._mode(pd.Series([2.0, 3.0, 3.0]), "threshold") == 3.0


def test_whole_values_print_whole_and_others_keep_figures():
    assert part0._quantity("speed_min", 1.0, "cm/s", "d", "s").style == "int"
    assert part0._quantity("speed_min", 2.5, "cm/s", "d", "s").style == "sig2"


def test_trigger_and_decoder_shares():
    survey = pd.DataFrame(
        {
            "Detection": ["MUA", "MUA", "SWR", "SWR, MUA", "Decoding"],
            "Decoding": [
                "Bayes",
                "UMAP\nBayes",
                "Rank order correlation",
                "bayes",
                "State Space",
            ],
        }
    )
    values = {v.key: v.value for v in part0.method_values(survey, "s")}
    assert values["papers"] == 5
    assert values["mua_only_fraction"] == pytest.approx(2 / 5)
    assert values["swr_only_fraction"] == pytest.approx(1 / 5)
    assert values["swr_and_mua_fraction"] == pytest.approx(1 / 5)
    assert values["bayes_fraction"] == pytest.approx(3 / 5)


def test_threshold_pools_the_ripple_and_multiunit_columns():
    numbers = pd.DataFrame(
        {
            "SWR Z-score Thresh. (STD)": [3.0, 2.0, float("nan")],
            "MUA Z-score Thresh. (STD)": [3.0, float("nan"), 4.0],
            **{
                column: [1.0, 1.0, 2.0]
                for column in (
                    "SWR Low Band (Hz)",
                    "SWR High Band (Hz)",
                    "Animal Speed (cm/s)",
                    "Min. Cells (#)",
                    "Min. Duration (ms)",
                    "Max Duration (ms)",
                    "Time bin (ms)",
                    "Position bin (cm)",
                )
            },
        }
    )
    values = {v.key: v.value for v in part0.parameter_values(numbers, "s")}
    assert values["threshold_mode"] == 3.0
    assert values["threshold_min"] == 2.0
    assert values["threshold_max"] == 4.0
    assert values["speed_mode"] == 1.0


def test_committed_summary_is_current():
    fresh = part0.summarize(part0.default_inputs(), "by", "commit")
    committed = json.loads(SUMMARY.read_text())
    assert committed["inputs"] == fresh.inputs, (
        "the survey changed: run `make summaries` (part 0) and `make reported-values`"
    )
    assert {k: v["value"] for k, v in committed["values"].items()} == {
        k: v.value for k, v in fresh.values.items()
    }
