"""One epoch's counts per detector, from native to eligible, and the bridging rows."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.adapters import (
    NATIVE_IS_ELIGIBLE,
    InventoryTables,
)
from continuum_swr_replay.analysis.inventory_comparison.epoch_comparison import (
    BRIDGING,
    bridging_set_aside,
    detector_counts,
)

KEY = ("chimi", 20200225, "02_r1")


def tables(n_inventory: int, n_set_aside: int, run_id: str) -> InventoryTables:
    return InventoryTables(
        inventory=pd.DataFrame({"event_key": range(n_inventory)}),
        observations=pd.DataFrame({"run_id": [run_id, run_id]}),
        set_aside=pd.DataFrame({"event_key": range(n_set_aside)}),
    )


KEPT = pd.DataFrame(
    {
        "detector": ["kay", "kay", "kay"],
        "event_start_s": [1.0, 9.0, 21.0],
        "event_end_s": [2.0, 11.0, 22.0],
    }
)
"""Kay keeps three events; the one at 9-11 s crosses its exposure's edge."""
MOVING = pd.DataFrame({"detector": ["kay", "mua"]})
BRIDGING_EVENTS = pd.DataFrame({"detector": ["kay"]})
COVERAGE = {
    (*KEY, "kay"): SimpleNamespace(exposure=np.array([[0.0, 10.0], [20.0, 25.0]])),
    (*KEY, "mua"): SimpleNamespace(exposure=np.empty((0, 2))),
}


def counts(inventories):
    return detector_counts(
        inventories, KEPT, MOVING, BRIDGING_EVENTS, COVERAGE, KEY
    ).set_index("detector")


def test_counts_from_native_to_eligible_and_the_rate_in_exposure():
    """Kay: 5 in the inventory (3 kept, 1 moving, 1 bridging) and 2 set aside
    by its adapter; 2 of its 3 kept events lie wholly in its 15 s of exposure.
    MUA: one moving event and no exposure, so no rate."""
    out = counts({"kay": tables(5, 2, "swr"), "mua": tables(1, 0, "decode")})

    kay, mua = out.loc["kay"], out.loc["mua"]
    assert (kay.n_native, kay.n_set_aside, kay.n_moving, kay.n_eligible) == (
        7,
        3,
        1,
        3,
    )
    assert kay.exposure_s == 15.0
    assert kay.n_in_exposure == 2
    assert kay.rate_per_s == pytest.approx(2 / 15)
    assert kay.run_id == "swr" and kay.native_note == ""
    assert (mua.n_native, mua.n_eligible, mua.n_moving, mua.n_in_exposure) == (
        1,
        0,
        1,
        0,
    )
    assert mua.exposure_s == 0.0 and np.isnan(mua.rate_per_s)
    assert mua.native_note == NATIVE_IS_ELIGIBLE["mua"]
    assert list(out.loc[:, ["animal", "day", "epoch"]].iloc[0]) == list(KEY)


def test_refuses_counts_that_do_not_add_up():
    with pytest.raises(RuntimeError, match="not eligible"):
        counts({"kay": tables(6, 2, "swr"), "mua": tables(1, 0, "decode")})


def test_bridging_events_are_set_aside_over_their_own_spans():
    bridging = pd.DataFrame(
        {
            "animal": ["chimi"],
            "day": [20200225],
            "epoch": ["02_r1"],
            "detector": ["kay"],
            "event_key": ["kay-1"],
            "event_start_s": [3.0],
            "event_end_s": [3.5],
        }
    )

    out = bridging_set_aside(bridging)

    assert out.loc[0, "reason"] == BRIDGING
    assert (out.loc[0, "interval_start_s"], out.loc[0, "interval_end_s"]) == (3.0, 3.5)
    assert out.loc[0, "event_key"] == "kay-1"
