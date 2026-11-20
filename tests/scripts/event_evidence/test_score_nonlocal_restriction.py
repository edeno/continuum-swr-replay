"""The non-local restriction runner's bin categories and model checks."""

import numpy as np
import pandas as pd
import pytest

from scripts.event_evidence import score_nonlocal_restriction as runner


def _frame(speed, excluded):
    return pd.DataFrame(
        {
            "head_speed": np.asarray(speed, dtype=float),
            "in_excluded_time": np.asarray(excluded, dtype=bool),
        },
        index=pd.Index(np.arange(len(speed)) * 0.002, name="time"),
    )


def test_each_bin_gets_its_category():
    # bins: excluded (slow, in SWR), excluded (fast), immobile in SWR,
    # immobile outside SWR, at the threshold (moving), fast (moving),
    # no speed (moving), immobile outside SWR
    frame = _frame(
        [1.0, 10.0, 0.5, 3.9, 4.0, 20.0, np.nan, 0.0],
        [True, True, False, False, False, False, False, False],
    )
    in_swr = np.array([True, False, True, False, True, True, True, False])
    labels = runner.bin_categories(frame, in_swr, speed_threshold=4.0)
    assert list(labels) == [
        "excluded",
        "excluded",
        "immobile_in_swr",
        "immobile_outside_swr",
        "moving",
        "moving",
        "moving",
        "immobile_outside_swr",
    ]
    assert list(labels.categories) == list(runner.CATEGORIES)


def test_categories_reject_a_mismatched_swr_mask():
    with pytest.raises(ValueError, match="shape"):
        runner.bin_categories(_frame([1.0, 2.0], [False, False]), np.array([True]))


def _record(**changes):
    record = {"all_finite": True}
    for model in ("none", "swr", "full"):
        record |= {
            f"marginal_agrees_{model}": True,
            f"marginal_log_likelihood_{model}": -10.0,
            f"marginal_difference_{model}": 0.0,
            f"causal_mass_deviation_{model}": 1e-6,
        }
    return record | changes


def test_a_clean_record_passes_the_checks():
    assert runner.check_failures(_record()) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"marginal_agrees_full": False}, "full: library marginal"),
        ({"causal_mass_deviation_none": 2e-3}, "none: filtered posterior mass"),
        ({"causal_mass_deviation_swr": float("nan")}, "swr: filtered"),
        ({"all_finite": False}, "not finite"),
    ],
)
def test_each_failed_check_is_named(changes, expected):
    failures = runner.check_failures(_record(**changes))
    assert len(failures) == 1 and expected in failures[0]
