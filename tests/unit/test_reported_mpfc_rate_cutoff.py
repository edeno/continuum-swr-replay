"""The mPFC summary's 1 Hz check: the main values count every unit, the check only units
at or above 1 Hz."""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS
from continuum_swr_replay.reported.mpfc import (
    CHECK_FIRING_RATE_HZ,
    CHECK_KEYS,
    TAGS,
    _value_factory,
    called_fraction_values,
    rate_cutoff_check_values,
)

#: Rates on both sides of the cutoff; exactly 1 Hz is kept.
RATES = np.array([0.05, 0.3, 0.99, 1.0, 1.01, 2.0, 4.0, 9.0, 0.5, 1.5] * 4)


def _population() -> pd.DataFrame:
    n = len(RATES)
    rng = np.random.default_rng(0)
    population = pd.DataFrame(
        {
            "immobility_avg_firing_rate": RATES,
            "animal": np.repeat(["chimi", "j16", "peanut", "senor"], n // 4),
            "nwb_file_name": np.repeat(
                [
                    "chimi20200212_.nwb",
                    "j1620210706_.nwb",
                    "peanut20201128_.nwb",
                    "senor20201030_.nwb",
                ],
                n // 4,
            ),
            "non_local_SWR + non_local_no_SWR vs. non_local: heldout_delta_dev": (
                rng.normal(size=n)
            ),
            "min_heldout_model": np.where(np.arange(n) % 2 == 0, "constant", "SWR"),
            "min_heldout_model_replay_split": np.where(
                np.arange(n) % 3 == 0, "non_local_SWR + non_local_no_SWR", "constant"
            ),
        }
    )
    # Low-rate units are called for everything; among the kept ones (positions
    # 3, 4, 5, 6, 7, 9 of each block of ten) every animal has units called for
    # both replay classes, for one only and for neither, so the called shares
    # differ between the population and its 1 Hz subset.
    low = RATES < CHECK_FIRING_RATE_HZ
    position = np.arange(n) % 10
    swr = low | np.isin(position, [3, 4, 9])
    non = low | np.isin(position, [3, 5, 9])
    for name, column in HELDOUT_P_COLUMNS.items():
        called = non if name == "non_local_no_SWR" else swr
        population[column] = np.where(called, 0.01, 0.4)
    for column in ("non_local_SWR", "non_local_no_SWR", "SWR", "matched_control"):
        population[f"{column}: peak_log_modulation"] = rng.choice([-1, 1], n) * (
            0.1 + rng.random(n)
        )
    return population


def test_the_check_keeps_units_at_or_above_one_hz():
    population = _population()
    make = _value_factory("synthetic", TAGS["paper"])
    values = {v.key: v for v in rate_cutoff_check_values(population, None, make)}
    kept = int((RATES >= CHECK_FIRING_RATE_HZ).sum())
    assert 0 < kept < len(RATES)
    assert values["check_one_hz_units"].value == kept
    assert "at or above 1 Hz" in values["check_one_hz_units"].description
    # the check's called share is the share among the kept units only
    gated = population[population["immobility_avg_firing_rate"] >= 1.0]
    called = (gated[HELDOUT_P_COLUMNS["non_local_SWR"]] < 0.05).mean()
    assert values["check_one_hz_swr_replay_called_fraction"].value == called
    main = {v.key: v for v in called_fraction_values(population, make)}
    assert main["swr_replay_called_fraction"].value > called
    assert main["swr_replay_units_tested"].value == len(RATES)
    # only the check keys, renamed, are returned
    assert {k.removeprefix("check_one_hz_") for k in values} <= set(CHECK_KEYS)
    assert all(k.startswith("check_one_hz_") for k in values)
