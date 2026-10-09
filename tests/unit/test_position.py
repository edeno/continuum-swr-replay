"""Unit tests for position data processing.

Tests position interpolation and linearization functions used in the
data loading pipeline, and the epoch position helper that ``load_data`` and the
SWR runner share. The helper's Spyglass calls are stand-ins, so these tests run
offline.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest
from track_linearization import make_track_graph

from continuum_swr_replay.data_loaders import position
from continuum_swr_replay.data_loaders.position import get_interpolated_position_info

# 3-node linear track shared by every test in this module; the production
# call signature requires both ``track_graph`` and the parallel
# ``edge_order`` + ``edge_spacing`` lists, so bundle them once.
_EDGE_ORDER = [(0, 1), (1, 2)]
_EDGE_SPACING = [10.0]


@pytest.fixture
def linear_3node_track():
    return make_track_graph(
        node_positions=[(0, 0), (10, 0), (20, 0)],
        edges=_EDGE_ORDER,
    )


def _interpolate(position_info, time, track_graph, **kwargs):
    """Thin wrapper that fills in the per-module edge_order / edge_spacing."""
    return get_interpolated_position_info(
        position_info,
        time=time,
        track_graph=track_graph,
        edge_order=_EDGE_ORDER,
        edge_spacing=_EDGE_SPACING,
        **kwargs,
    )


class TestGetInterpolatedPositionInfo:
    """Test position interpolation and linearization."""

    def test_interpolation_preserves_original_times(self, linear_3node_track):
        """When the interpolation grid equals the input grid, original
        column values must be returned exactly — no rounding, no
        propagation through the interpolator."""
        original_times = np.array([0.0, 1.0, 2.0, 3.0])
        original_x = np.array([0.0, 5.0, 10.0, 15.0])
        original_speed = np.array([5.0, 7.0, 9.0, 11.0])
        position_info = pd.DataFrame(
            {
                "head_position_x": original_x,
                "head_position_y": np.zeros(4),
                "head_speed": original_speed,
            },
            index=original_times,
        )

        result = _interpolate(position_info, original_times, linear_3node_track)

        assert {"linear_position", "projected_x_position", "track_segment_id"} <= set(
            result.columns
        )
        np.testing.assert_array_equal(result["head_position_x"].to_numpy(), original_x)
        np.testing.assert_array_equal(result["head_speed"].to_numpy(), original_speed)

    def test_interpolation_to_new_times(self, linear_3node_track):
        """New time points should have linearly interpolated values."""
        original_times = np.array([0.0, 2.0])
        position_info = pd.DataFrame(
            {
                "head_position_x": [0.0, 10.0],
                "head_position_y": [0.0, 0.0],
            },
            index=original_times,
        )

        new_times = np.array([0.0, 1.0, 2.0])
        result = _interpolate(position_info, new_times, linear_3node_track)

        assert len(result) == 3
        assert result.index.tolist() == list(new_times)
        # Midpoint at t=1.0: 5.0 cm under linear interpolation.
        np.testing.assert_allclose(
            result.loc[1.0, "head_position_x"],
            5.0,
            rtol=0.01,
            err_msg="Expected linear interpolation at midpoint",
        )

    def test_interpolation_multiple_columns(self, linear_3node_track):
        """All columns should be interpolated independently."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame(
            {
                "head_position_x": [0.0, 10.0],
                "head_position_y": [0.0, 5.0],
                "head_speed": [0.0, 10.0],
                "head_orientation": [0.0, np.pi],
            },
            index=original_times,
        )

        result = _interpolate(position_info, np.array([0.5]), linear_3node_track)

        np.testing.assert_allclose(result.loc[0.5, "head_position_x"], 5.0, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, "head_position_y"], 2.5, rtol=0.01)
        np.testing.assert_allclose(result.loc[0.5, "head_speed"], 5.0, rtol=0.01)
        np.testing.assert_allclose(
            result.loc[0.5, "head_orientation"], np.pi / 2, rtol=0.01
        )

    def test_custom_position_columns(self, linear_3node_track):
        """Should handle custom position column names."""
        original_times = np.array([0.0, 1.0])
        position_info = pd.DataFrame(
            {
                "custom_x": [0.0, 10.0],
                "custom_y": [0.0, 0.0],
            },
            index=original_times,
        )

        result = _interpolate(
            position_info,
            original_times,
            linear_3node_track,
            position_columns=["custom_x", "custom_y"],
        )

        assert "linear_position" in result.columns
        assert "custom_x" in result.columns

    def test_preserves_time_ordering(self, linear_3node_track):
        """Output should maintain time ordering even with unsorted input times."""
        position_info = pd.DataFrame(
            {
                "head_position_x": [0.0, 5.0, 10.0],
                "head_position_y": [0.0, 0.0, 0.0],
            },
            index=[0.0, 1.0, 2.0],
        )

        unsorted_times = np.array([2.0, 0.5, 1.5, 0.0])
        result = _interpolate(position_info, unsorted_times, linear_3node_track)

        assert set(result.index) == set(unsorted_times)

    def test_adds_linearization_columns(self, linear_3node_track):
        """Should add all required linearization columns."""
        position_info = pd.DataFrame(
            {
                "head_position_x": [0.0, 10.0],
                "head_position_y": [0.0, 0.0],
            },
            index=[0.0, 1.0],
        )

        result = _interpolate(position_info, np.array([0.0, 1.0]), linear_3node_track)

        required_columns = {
            "linear_position",
            "projected_x_position",
            "projected_y_position",
            "track_segment_id",
        }
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"

    def test_linear_position_monotonic_for_forward_movement(self, linear_3node_track):
        """Linear position should increase monotonically for forward movement."""
        times = np.arange(0.0, 3.0, 0.1)
        positions = np.linspace(0.0, 15.0, len(times))

        position_info = pd.DataFrame(
            {
                "head_position_x": positions,
                "head_position_y": np.zeros(len(times)),
            },
            index=times,
        )

        result = _interpolate(position_info, times, linear_3node_track)

        linear_pos = result["linear_position"].values
        diffs = np.diff(linear_pos)
        assert np.all(diffs >= -0.01), (
            "Linear position should be monotonically increasing for forward movement"
        )

    def test_respects_edge_spacing(self, linear_3node_track):
        """Linearization output should fall within the total track length."""
        position_info = pd.DataFrame(
            {
                "head_position_x": [0.0, 10.0],
                "head_position_y": [0.0, 0.0],
            },
            index=[0.0, 1.0],
        )

        result = _interpolate(position_info, np.array([0.0, 1.0]), linear_3node_track)

        assert result["linear_position"].min() >= 0.0
        assert result["linear_position"].max() <= 20.0  # Total track length


# --- The epoch position that load_data and the SWR runner share -------------

_NWB_FILE_NAME = "testrat20990101_.nwb"
"""A synthetic session: no real recording has this name."""
_EPOCH_NAME = "02_r1"


class _FakePositionIntervalMap:
    """``PositionIntervalMap``: records the restriction, serves one name."""

    def __init__(self):
        self.restrictions: list[dict] = []

    def __and__(self, restriction):
        self.restrictions.append(restriction)
        return self

    def fetch1(self, attribute):
        assert attribute == "position_interval_name"
        return "pos 1 valid times"


def test_epoch_position_drops_nan_rows_like_load_data(monkeypatch, caplog):
    """The ``PositionIntervalMap`` lookup, ``get_position_info``, then
    ``load_data``'s drop of rows with NaN head x or y, with its log line.

    A NaN in any other column keeps its row: the drop looks at position only.
    """
    interval_map = _FakePositionIntervalMap()
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = interval_map
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    time = pd.Index(100.0 + np.arange(8) / 500, name="time")
    position_info = pd.DataFrame(
        {
            "head_position_x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "head_position_y": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0],
            "head_speed": [0.5, np.nan, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "track_segment_id": [0, 0, 0, 6, 6, 6, 1, 1],
        },
        index=time,
    )
    track_graph = object()
    position_calls = []

    def fake_get_position_info(nwb_file_name, epoch_name, pos_name):
        position_calls.append((nwb_file_name, epoch_name, pos_name))
        return {
            "position_info": position_info.copy(),
            "linear_edge_order": [(0, 6)],
            "linear_edge_spacing": [15.0],
            "track_graph": track_graph,
        }

    monkeypatch.setattr(position, "get_position_info", fake_get_position_info)

    with caplog.at_level(logging.INFO, logger=position.__name__):
        position_data = position.get_epoch_position_info(_NWB_FILE_NAME, _EPOCH_NAME)

    assert interval_map.restrictions == [
        {"nwb_file_name": _NWB_FILE_NAME, "interval_list_name": _EPOCH_NAME}
    ]
    assert position_calls == [(_NWB_FILE_NAME, _EPOCH_NAME, "pos 1 valid times")]
    pd.testing.assert_frame_equal(
        position_data["position_info"], position_info.drop(index=time[[2, 5]])
    )
    assert position_data["track_graph"] is track_graph
    assert position_data["linear_edge_order"] == [(0, 6)]
    assert position_data["linear_edge_spacing"] == [15.0]
    assert "Dropped 2 time points with NaN position (25.0%)" in caplog.messages


@pytest.fixture
def bandit_task(monkeypatch):
    """``bandit_task``, imported offline with ``spyglass.common`` stubbed.

    It imports ``spyglass.common`` at import time, which opens a DataJoint
    connection. A module imported here binds the stub, so it is forgotten
    afterwards and a later import gets the real one.
    """
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    name = "continuum_swr_replay.data_loaders.bandit_task"
    fresh = name not in sys.modules
    yield importlib.import_module(name)
    if fresh:
        sys.modules.pop(name, None)
        vars(sys.modules["continuum_swr_replay.data_loaders"]).pop("bandit_task", None)


def test_load_data_takes_position_from_get_epoch_position_info(
    bandit_task, monkeypatch
):
    """``load_data``'s position is ``get_epoch_position_info``'s, so the SWR
    runner, which calls the same helper, shares its speed and span."""
    position_data = {
        "position_info": pd.DataFrame(
            {"head_position_x": [1.0], "head_position_y": [2.0], "head_speed": [0.5]},
            index=pd.Index([100.0], name="time"),
        ),
        "linear_edge_order": [(0, 6)],
        "linear_edge_spacing": [15.0],
        "track_graph": object(),
    }
    calls = []

    def fake_get_epoch_position_info(nwb_file_name, epoch_name):
        calls.append((nwb_file_name, epoch_name))
        return position_data

    monkeypatch.setattr(
        bandit_task, "get_epoch_position_info", fake_get_epoch_position_info
    )

    result = bandit_task.load_data(_NWB_FILE_NAME, _EPOCH_NAME, include=frozenset())

    assert calls == [(_NWB_FILE_NAME, _EPOCH_NAME)]
    assert set(result) == set(position_data)
    for key, value in position_data.items():
        assert result[key] is value


# ---------------------------------------------------------------------------
# A PositionOutput entry held in two parts (j1620210710 02_r1 and 04_r2)
# ---------------------------------------------------------------------------

_MERGE_ID = "9ba1f7d5-c9a3-b19a-19a6-e92b8674866e"
_TWO_PARENTS = (
    "Found  2 potential parents: [FreeTable(`common_position`.`__interval_position_"
    "info`), FreeTable(`position_v1_trodes_position`.`__trodes_pos_v1`)]"
)


class _Restricted:
    """``table & restriction``, answering only what the merge table answers."""

    def __init__(self, table, restriction):
        self.table, self.restriction = table, restriction

    def fetch1(self, *attributes):
        row = self.table.rows[self.restriction["merge_id"]]
        return row[attributes[0]] if attributes else dict(row)

    def fetch1_dataframe(self):
        return self.table.fetch1_dataframe(self.restriction)


class _Table:
    def __init__(self, rows=None, frame=None, refuse=None):
        self.rows, self.frame, self.refuse, self.restrictions = rows, frame, refuse, []

    def __and__(self, restriction):
        self.restrictions.append(dict(restriction))
        return _Restricted(self, restriction)

    def fetch1_dataframe(self, restriction):
        if self.refuse is not None:
            raise ValueError(self.refuse)
        return self.frame


def _position_output_stub(monkeypatch) -> types.SimpleNamespace:
    """``spyglass.position.PositionOutput`` with one entry whose merge id is in
    both its CommonPos and its TrodesPosV1 part, as the lab database holds
    j1620210710's: the master names CommonPos, and the merge table's own
    ``fetch1_dataframe`` refuses the entry."""
    common_parent = _Table(frame=pd.DataFrame({"head_speed": [1.0]}))
    trodes_parent = _Table(frame=pd.DataFrame({"head_speed": [99.0]}))
    master = _Table(
        rows={_MERGE_ID: {"merge_id": _MERGE_ID, "source": "CommonPos"}},
        refuse=_TWO_PARENTS,
    )
    part_key = {
        "nwb_file_name": "j1620210710_.nwb",
        "interval_list_name": "pos 1 valid times",
    }
    common_part = _Table(
        rows={
            _MERGE_ID: {
                "merge_id": _MERGE_ID,
                **part_key,
                "position_info_param_name": "default_decoding",
            }
        }
    )
    trodes_part = _Table(
        rows={
            _MERGE_ID: {
                "merge_id": _MERGE_ID,
                **part_key,
                "trodes_pos_params_name": "default_decoding",
            }
        }
    )
    parents = {"CommonPos": common_parent, "TrodesPosV1": trodes_parent}

    class PositionOutput:
        CommonPos = common_part
        TrodesPosV1 = trodes_part

        def __call__(self):  # the instance stands in for the class too
            return self

        def __and__(self, restriction):
            return master & restriction

        def merge_get_parent_class(self, source):
            return parents[source]

    module = types.ModuleType("spyglass.position")
    module.PositionOutput = PositionOutput()
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.position", module)
    return types.SimpleNamespace(
        master=master,
        common_parent=common_parent,
        trodes_parent=trodes_parent,
        part_key=part_key,
    )


def test_a_merge_entry_held_in_two_parts_is_read_from_its_masters_source(
    monkeypatch,
):
    """The merge table's ``fetch1_dataframe`` finds two parents and refuses;
    the entry is read as ``load_data`` reads it, from the source table itself:
    the part the master names gives the parent's key."""
    stub = _position_output_stub(monkeypatch)
    PositionOutput = sys.modules["spyglass.position"].PositionOutput
    with pytest.raises(ValueError, match="2 potential parents"):
        (PositionOutput & {"merge_id": _MERGE_ID}).fetch1_dataframe()

    frame = position.get_merge_position_info(_MERGE_ID)

    pd.testing.assert_frame_equal(frame, stub.common_parent.frame)
    (restriction,) = stub.common_parent.restrictions
    assert restriction == {
        "merge_id": _MERGE_ID,
        **stub.part_key,
        "position_info_param_name": "default_decoding",
    }
    assert stub.trodes_parent.restrictions == []
