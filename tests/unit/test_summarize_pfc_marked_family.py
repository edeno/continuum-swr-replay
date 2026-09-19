"""Aggregation rules of the marked-family population summary."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "summarize_pfc_marked_family",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "summarize_pfc_marked_family.py",
)
summarize = importlib.util.module_from_spec(_SPEC)
sys.modules["summarize_pfc_marked_family"] = summarize
_SPEC.loader.exec_module(summarize)


def _units() -> pd.DataFrame:
    """Two animals; animal a has two sessions, animal b one, with a blow-up."""
    rows = []
    for animal, session, deltas in (
        ("a", "a1", [1.0, -1.0, 2.0, np.nan]),  # 2/3 positive over finite
        ("a", "a2", [-1.0, -2.0, 3.0, -4.0]),  # 1/4 positive
        ("b", "b1", [5.0, 5000.0, -1.0, 1.0]),  # blow-up excluded -> 2/3
    ):
        for delta in deltas:
            rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "model": "m",
                    "event_class": "non_local_no_SWR",
                    "target": "t",
                    "delta": delta,
                    "curve_finite": True,
                }
            )
    return pd.DataFrame(rows)


def test_fraction_positive_is_animal_balanced_and_excludes_nan_and_blowups():
    out = summarize.animal_balanced_fraction_positive(_units(), blowup=1000.0)
    row = out.iloc[0]
    assert row.n_units == 12
    assert row.n_usable == 10
    assert row.n_excluded_blowup == 1
    assert row.n_sessions == 3 and row.n_animals == 2
    # animal a: median(2/3, 1/4) = 11/24; animal b: 2/3; animal-balanced median
    expected_a = np.median([2 / 3, 1 / 4])
    assert row.frac_pos == np.median([expected_a, 2 / 3])
    assert row.frac_pos_min == expected_a and row.frac_pos_max == 2 / 3


def test_population_curve_peak_and_sign_consistency():
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    curves = {
        ("m", "non_local_no_SWR", "t"): {
            "a1": 2 * bump,
            "a2": 0.5 * bump,
            "b1": -0.5 * bump,
        }
    }
    out = summarize.population_curve_summary(
        curves, lag, {"a1": "a", "a2": "a", "b1": "b"}
    )
    row = out.iloc[0]
    assert row.peak_lag_ms == 0.0
    assert row.peak_value == np.mean([2, 0.5, -0.5])
    assert row.sessions_same_sign == 2 / 3
    assert row.animals_same_sign == 1 and row.curve_n_animals == 2
