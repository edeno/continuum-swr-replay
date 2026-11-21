"""The content table contract (``table_contracts.validate_canonical_table_contract``)
on the small canonical arm and patch tables, with an event that crosses its
visit's final poke-out (``tests/support/_content_tables.py``).

Event ``e1`` starts inside a visit at well 1 (arm 1, patch 1) and crosses the
final poke-out: its whole-window location is ``mixed`` with no well, and its
current arm and patch are the visit's.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.event_tables.table_contracts import (
    validate_canonical_table_contract,
)
from tests.support._content_tables import canonical_arm_table, canonical_patch_table


def _tables():
    return {
        "arm": canonical_arm_table(crossing_event="e1"),
        "patch": canonical_patch_table(crossing_event="e1"),
    }


@pytest.mark.parametrize("kind", ["arm", "patch"])
def test_a_crossing_event_is_accepted(kind):
    validate_canonical_table_contract(
        _tables()[kind], hierarchy_kind=kind, table_label=kind
    )


def test_the_crossing_event_follows_the_visit_well():
    arms, patches = _tables()["arm"], _tables()["patch"]
    crossing = arms[arms["event_id"] == "e1"]
    assert crossing["visit_context"].eq("spans_visit_end").all()
    assert crossing["occupancy_well_id"].isna().all()
    assert crossing["occupancy_well_arm"].eq(1).all()
    np.testing.assert_array_equal(
        crossing["is_current_arm"], [float(arm == 1) for arm in range(9)]
    )
    crossing = patches[patches["event_id"] == "e1"].sort_values("patch")
    np.testing.assert_array_equal(crossing["is_current_patch"], [1.0, 0.0, 0.0])


@pytest.mark.parametrize("kind", ["arm", "patch"])
def test_in_visit_is_exactly_at_well(kind):
    table = _tables()[kind]
    table.loc[table["event_id"] == "e0", "visit_context"] = "spans_visit_end"
    with pytest.raises(ValueError, match="in_visit"):
        validate_canonical_table_contract(table, hierarchy_kind=kind, table_label=kind)


@pytest.mark.parametrize("kind", ["arm", "patch"])
def test_spans_visit_end_only_on_mixed_rows(kind):
    table = _tables()[kind]
    rows = table["event_id"] == "e1"
    table.loc[rows, "occupancy_state"] = "in_transit"
    with pytest.raises(ValueError, match="spans_visit_end"):
        validate_canonical_table_contract(table, hierarchy_kind=kind, table_label=kind)


@pytest.mark.parametrize("kind", ["arm", "patch"])
def test_the_crossing_event_needs_the_visit_well(kind):
    table = _tables()[kind]
    rows = table["event_id"] == "e1"
    table.loc[rows, "occupancy_well_arm"] = np.nan
    with pytest.raises(ValueError, match="occupancy_well_arm|is_current"):
        validate_canonical_table_contract(table, hierarchy_kind=kind, table_label=kind)
