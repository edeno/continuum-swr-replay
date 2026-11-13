"""The firing-rate gate table's own counts: why units leave, and the per-band rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS

_SPEC = importlib.util.spec_from_file_location(
    "part4_rate_threshold_sensitivity",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "pfc"
    / "part4_rate_threshold_sensitivity.py",
)
sensitivity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sensitivity)

NAN = np.nan


def test_named_joins_a_results_labels_into_its_names():
    assert sensitivity._named({"odds_ratio": 2.0, "n_tested": 7}, "overlap") == {
        "overlap.odds_ratio": 2.0,
        "overlap.n_tested": 7,
    }
    table = pd.DataFrame(
        {"n_tested": [3, 4], "fraction": [0.5, 0.25]}, index=["SWR", "non_local"]
    )
    assert sensitivity._named(table, "called_fractions") == {
        "called_fractions.SWR.n_tested": 3,
        "called_fractions.SWR.fraction": 0.5,
        "called_fractions.non_local.n_tested": 4,
        "called_fractions.non_local.fraction": 0.25,
    }
    by_group = pd.DataFrame(
        {"n_units": [1, 2]},
        index=pd.MultiIndex.from_tuples(
            [("all", "enhanced"), ("min_events", "suppressed")],
            names=["subset", "direction"],
        ),
    )
    assert sensitivity._named(by_group, "timing") == {
        "timing.all.enhanced.n_units": 1,
        "timing.min_events.suppressed.n_units": 2,
    }


def test_funnel_counts_why_units_leave_at_a_gate():
    raw = pd.DataFrame(
        {
            "immobility_avg_firing_rate": [0.05, 0.5, 1.0, 2.0, 3.0],
            "has_fit": [True, True, True, False, True],
        }
    )
    # At 1 Hz three units pass (the one exactly at the gate too). One of them
    # has no fit; of the two with a fit the loaded population holds one, so the
    # other was dropped for a non-finite curve.
    assert sensitivity.funnel(raw, 1.0, n_kept=1) == {
        "funnel_units_in_bundles": 5,
        "funnel_units_rate_pass": 3,
        "funnel_units_no_fit": 1,
        "funnel_units_nonfinite_curve": 1,
        "funnel_units_kept": 1,
    }


def test_band_rows_give_each_bands_own_called_fractions():
    # Six units: one below 0.1 Hz, two in [0.1, 0.5), one in [0.5, 1), two at
    # 1 Hz or above (the one exactly at 1 Hz belongs there).
    rate = [0.05, 0.3, 0.3, 0.7, 1.0, 2.0]
    population = pd.DataFrame({"immobility_avg_firing_rate": rate})
    for column in HELDOUT_P_COLUMNS.values():
        population[column] = [0.01, 0.01, 0.5, 0.5, 0.01, 0.5]
    population[HELDOUT_P_COLUMNS["engagement"]] = [0.5, 0.01, 0.01, 0.5, 0.5, 0.5]
    extended = pd.DataFrame(
        {
            "immobility_avg_firing_rate": rate,
            "SWR_no_non_local: p_heldout": [NAN, 0.01, NAN, 0.5, 0.01, 0.01],
        }
    )
    spikes = pd.Series([1.0, 10.0, 30.0, 50.0, 200.0, 400.0])

    rows = pd.DataFrame(sensitivity.band_rows(population, extended, spikes))
    values = rows.set_index(["selection", "key"])["value"]

    assert list(rows["selection"].unique()) == [
        "band [0, 0.1) Hz",
        "band [0.1, 0.5) Hz",
        "band [0.5, 1) Hz",
        "band [1, inf) Hz",
    ]
    low = values["band [0.1, 0.5) Hz"]
    assert low["units"] == 2
    assert (low["spikes_fit_median"], low["spikes_fit_min"]) == (20.0, 10.0)
    assert (low["SWR_units_tested"], low["SWR_called_fraction"]) == (2, 0.5)
    assert low["engagement_called_fraction"] == 1.0
    # one of the two is tested for SWRs without replay, and called
    assert low["SWR_no_non_local_units_tested"] == 1
    assert low["SWR_no_non_local_called_fraction"] == 1.0
    high = values["band [1, inf) Hz"]
    assert high["units"] == 2
    assert high["SWR_called_fraction"] == 0.5
    assert high["SWR_no_non_local_units_tested"] == 2
    assert high["SWR_no_non_local_called_fraction"] == 1.0
    assert values["band [0.5, 1) Hz"]["SWR_no_non_local_called_fraction"] == 0.0
    # the lowest band's one unit is not tested for SWRs without replay: no share
    lowest = values["band [0, 0.1) Hz"]
    assert lowest["SWR_called_fraction"] == 1.0
    assert lowest["SWR_no_non_local_units_tested"] == 0
    assert np.isnan(lowest["SWR_no_non_local_called_fraction"])
    assert lowest["spikes_fit_min"] == pytest.approx(1.0)
