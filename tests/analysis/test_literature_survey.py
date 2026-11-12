"""The literature-survey statistics, on surveys small enough to count by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis import literature_survey

NAN = np.nan


def test_the_mode_is_the_single_most_common_value():
    assert literature_survey.mode(pd.Series([2.0, 3.0, 3.0]), "x") == 3.0
    # The mode, not the mean (8/3) or the median (2): a wrong rule fails.
    assert literature_survey.mode(pd.Series([1.0, 1.0, 2.0, 4.0, 5.0]), "x") == 1.0


def test_a_tied_mode_is_an_error():
    with pytest.raises(ValueError, match="threshold: the mode is tied"):
        literature_survey.mode(pd.Series([2.0, 2.0, 3.0, 3.0]), "threshold")


def test_a_parameter_pools_its_columns_and_drops_the_cells_that_are_not_numbers():
    numbers = pd.DataFrame(
        {
            "swr": [1.0, 2.0, NAN, 8.0],
            "mua": [3.0, NAN, 3.0, NAN],
            "other": [100.0, 100.0, 100.0, 100.0],
        }
    )
    stats = literature_survey.parameter_statistics(numbers, ("swr", "mua"), "x")
    # A paper with a number in both columns counts twice: 1, 2, 8, 3, 3.
    assert stats == {"n_cells": 5, "mode": 3.0, "min": 1.0, "max": 8.0}
    one = literature_survey.parameter_statistics(numbers, ("mua",), "x")
    assert one == {"n_cells": 2, "mode": 3.0, "min": 3.0, "max": 3.0}


def test_a_parameter_with_no_number_is_an_error():
    numbers = pd.DataFrame({"speed": [NAN, NAN]})
    with pytest.raises(ValueError, match="speed: no numeric cells"):
        literature_survey.parameter_statistics(numbers, ("speed",), "speed")


def test_trigger_and_decoder_shares():
    triggers = pd.Series(
        [
            frozenset({"MUA"}),
            frozenset({"MUA"}),
            frozenset({"SWR"}),
            frozenset({"SWR", "MUA"}),
            frozenset({"MUA", "SWR"}),
            frozenset({"Decoding"}),
            frozenset({"SWR", "MUA", "Decoding"}),
        ]
    )
    decoding = pd.Series(
        [
            "Bayes",
            "UMAP\nBayes",
            "Rank order correlation",
            "bayes",
            "Bayesian decoder",
            "non-Bayesian state space",
            "naive-Bayes",
        ]
    )
    shares = literature_survey.method_shares(triggers, decoding)
    assert shares["n_papers"] == 7
    assert shares["mua_only_fraction"] == pytest.approx(2 / 7)
    assert shares["swr_only_fraction"] == pytest.approx(1 / 7)
    # A paper that also names Decoding is in none of the trigger shares.
    assert shares["swr_and_mua_fraction"] == pytest.approx(2 / 7)
    assert shares["bayes_fraction"] == pytest.approx(5 / 7)
