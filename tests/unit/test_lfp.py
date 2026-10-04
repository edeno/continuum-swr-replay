"""Unit tests for LFP data processing.

Tests the LFP filtering and ripple data processing functions, and the read-only
fetch of the CA1 ripple-band inputs. That fetch runs offline here: every
Spyglass module it imports is a stand-in in ``sys.modules``, and no stand-in has
an insert, populate or delete, so a write would fail the test.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import continuum_swr_replay.types as project_types
from continuum_swr_replay.data_loaders import lfp
from continuum_swr_replay.data_loaders.lfp import filter_ripple_lfp_time


class TestRippleLFPFiltering:
    """Test temporal filtering of ripple LFP data to position bounds."""

    def test_filter_to_position_bounds(self):
        """Ripple LFP data outside position time range should be removed."""
        # Create sample ripple data with time index from 0 to 10 seconds
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0, 3.0], "end_time": [1.5, 3.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 4),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3", "electrode4"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 4),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3", "electrode4"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 2.0 to 8.0 seconds
        position_time = np.arange(2.0, 8.0, 0.01)

        # Apply filter (returns new dict)
        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Check that filtered data is within bounds
        assert ripple_data["ripple_filtered_lfps"].index.min() >= 2.0
        assert ripple_data["ripple_filtered_lfps"].index.max() <= 8.0

        assert ripple_data["ripple_lfps"].index.min() >= 2.0
        assert ripple_data["ripple_lfps"].index.max() <= 8.0

        assert ripple_data["ripple_consensus_trace"].index.min() >= 2.0
        assert ripple_data["ripple_consensus_trace"].index.max() <= 8.0

        assert ripple_data["zscored_ripple_consensus_trace"].index.min() >= 2.0
        assert ripple_data["zscored_ripple_consensus_trace"].index.max() <= 8.0

        # Check that ripple_times IS filtered: [1.0, 1.5] is before position
        # start (2.0), so only the [3.0, 3.5] event should remain
        assert len(ripple_data["ripple_times"]) == 1
        assert ripple_data["ripple_times"].iloc[0]["start_time"] == 3.0
        assert ripple_data["ripple_times"].iloc[0]["end_time"] == 3.5

    def test_ripple_straddling_start_boundary_is_dropped(self):
        """A ripple starting before position time should be dropped entirely."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({
                "start_time": [1.5, 3.0],
                "end_time": [2.5, 3.5],
            }),
            "ripple_filtered_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="c",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="zc",
            ),
        }

        # Position starts at 2.0 — the [1.5, 2.5] ripple straddles the boundary
        position_time = np.arange(2.0, 8.0, 0.01)
        result = filter_ripple_lfp_time(ripple_data, position_time)

        # Only the fully-contained [3.0, 3.5] ripple should survive
        assert len(result["ripple_times"]) == 1
        assert result["ripple_times"].iloc[0]["start_time"] == 3.0

    def test_ripple_straddling_end_boundary_is_dropped(self):
        """A ripple ending after position time should be dropped entirely."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.001), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({
                "start_time": [3.0, 7.5],
                "end_time": [3.5, 8.5],
            }),
            "ripple_filtered_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.zeros((len(time_index), 1)), index=time_index, columns=["e1"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="c",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.zeros(len(time_index)), index=time_index, name="zc",
            ),
        }

        # Position ends at 8.0 — the [7.5, 8.5] ripple straddles the boundary
        position_time = np.arange(2.0, 8.0, 0.01)
        result = filter_ripple_lfp_time(ripple_data, position_time)

        # Only the fully-contained [3.0, 3.5] ripple should survive
        assert len(result["ripple_times"]) == 1
        assert result["ripple_times"].iloc[0]["start_time"] == 3.0

    def test_empty_dataframes(self):
        """Empty ripple data should not crash."""
        # Create empty DataFrames and Series with proper structure
        empty_time_index = pd.Index([], dtype=np.float64, name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [], "end_time": []}),
            "ripple_filtered_lfps": pd.DataFrame(
                index=empty_time_index,
                columns=["electrode1"],
            ),
            "ripple_lfps": pd.DataFrame(
                index=empty_time_index,
                columns=["electrode1"],
            ),
            "ripple_consensus_trace": pd.Series(
                index=empty_time_index,
                dtype=np.float64,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                index=empty_time_index,
                dtype=np.float64,
                name="zscored_consensus_trace",
            ),
        }

        position_time = np.arange(1.0, 5.0, 0.01)

        # Should not raise
        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should still be empty
        assert len(ripple_data["ripple_filtered_lfps"]) == 0
        assert len(ripple_data["ripple_lfps"]) == 0
        assert len(ripple_data["ripple_consensus_trace"]) == 0
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == 0

    def test_boundary_conditions_exact_start_end(self):
        """Data at exact boundaries should be included (slice is inclusive)."""
        # Create data from 1.0 to 5.0
        time_index = pd.Index(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [2.0], "end_time": [3.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 2.0 to 4.0 (exact boundaries)
        position_time = np.array([2.0, 3.0, 4.0])

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should include boundary times (pandas slice is inclusive on both ends)
        assert len(ripple_data["ripple_filtered_lfps"]) == 3
        np.testing.assert_array_equal(
            ripple_data["ripple_filtered_lfps"].index.to_numpy(),
            np.array([2.0, 3.0, 4.0])
        )

        assert len(ripple_data["ripple_consensus_trace"]) == 3
        np.testing.assert_array_equal(
            ripple_data["ripple_consensus_trace"].index.to_numpy(),
            np.array([2.0, 3.0, 4.0])
        )

    def test_all_data_outside_bounds(self):
        """If all ripple data outside position bounds, should return empty data."""
        # Ripple data from 0 to 1 second
        time_index = pd.Index(np.arange(0.0, 1.0, 0.01), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [0.2], "end_time": [0.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["electrode1", "electrode2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time from 5 to 10 seconds (completely outside ripple data)
        position_time = np.arange(5.0, 10.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Should have empty filtered data
        assert len(ripple_data["ripple_filtered_lfps"]) == 0
        assert len(ripple_data["ripple_lfps"]) == 0
        assert len(ripple_data["ripple_consensus_trace"]) == 0
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == 0

    def test_no_data_filtered_all_within_bounds(self):
        """If all ripple data within position bounds, all should be preserved."""
        # Ripple data from 2.0 to 4.0
        time_index = pd.Index(np.arange(2.0, 4.0, 0.01), name="time")
        original_length = len(time_index)

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [2.5], "end_time": [3.5]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(original_length, 3),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(original_length, 3),
                index=time_index,
                columns=["electrode1", "electrode2", "electrode3"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(original_length),
                index=time_index,
                name="consensus_trace",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(original_length),
                index=time_index,
                name="zscored_consensus_trace",
            ),
        }

        # Position time encompasses all ripple data (1.0 to 5.0)
        position_time = np.arange(1.0, 5.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # All data should be preserved
        assert len(ripple_data["ripple_filtered_lfps"]) == original_length
        assert len(ripple_data["ripple_lfps"]) == original_length
        assert len(ripple_data["ripple_consensus_trace"]) == original_length
        assert len(ripple_data["zscored_ripple_consensus_trace"]) == original_length

    def test_preserves_data_structure(self):
        """Filtering should preserve DataFrame/Series structure and column names."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.01), name="time")

        ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0], "end_time": [2.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 3),
                index=time_index,
                columns=["CA1_01", "CA1_02", "CA1_03"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 3),
                index=time_index,
                columns=["CA1_01", "CA1_02", "CA1_03"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="consensus",
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
                name="zscored",
            ),
        }

        position_time = np.arange(2.0, 5.0, 0.01)

        ripple_data = filter_ripple_lfp_time(ripple_data, position_time)

        # Check that column names are preserved
        assert list(ripple_data["ripple_filtered_lfps"].columns) == ["CA1_01", "CA1_02", "CA1_03"]
        assert list(ripple_data["ripple_lfps"].columns) == ["CA1_01", "CA1_02", "CA1_03"]

        # Check that Series names are preserved
        assert ripple_data["ripple_consensus_trace"].name == "consensus"
        assert ripple_data["zscored_ripple_consensus_trace"].name == "zscored"

        # Check that data types are preserved
        assert isinstance(ripple_data["ripple_filtered_lfps"], pd.DataFrame)
        assert isinstance(ripple_data["ripple_lfps"], pd.DataFrame)
        assert isinstance(ripple_data["ripple_consensus_trace"], pd.Series)
        assert isinstance(ripple_data["zscored_ripple_consensus_trace"], pd.Series)

    def test_returns_new_dict(self):
        """Function should return new dict (functional style)."""
        time_index = pd.Index(np.arange(0.0, 10.0, 0.1), name="time")

        original_ripple_data = {
            "ripple_times": pd.DataFrame({"start_time": [1.0], "end_time": [2.0]}),
            "ripple_filtered_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["e1", "e2"],
            ),
            "ripple_lfps": pd.DataFrame(
                np.random.RandomState(42).randn(len(time_index), 2),
                index=time_index,
                columns=["e1", "e2"],
            ),
            "ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
            ),
            "zscored_ripple_consensus_trace": pd.Series(
                np.random.RandomState(42).randn(len(time_index)),
                index=time_index,
            ),
        }

        # Store original length
        original_length = len(original_ripple_data["ripple_filtered_lfps"])

        position_time = np.arange(2.0, 5.0, 0.01)

        # Function returns new dict (functional style)
        result = filter_ripple_lfp_time(original_ripple_data, position_time)

        assert result is not None, "Function should return new dict"
        assert result is not original_ripple_data, "Should be a new dict object"
        assert len(result["ripple_filtered_lfps"]) < original_length, "Returned data should be filtered"
        assert len(original_ripple_data["ripple_filtered_lfps"]) == original_length, "Original data should be unchanged"


# --- The ripple keys and the band-electrode check ---------------------------


def test_ripple_keys_are_what_get_ripple_used():
    """Both restrictions, pinned to the literals ``get_ripple`` built before the
    refactor moved them into ``_ripple_keys``."""
    ripple_key, lfp_band_restriction = lfp._ripple_keys("chimi20200225_.nwb", "02_r1")

    assert ripple_key == {
        "nwb_file_name": "chimi20200225_.nwb",
        "target_interval_list_name": "02_r1 noPrePostTrialTimes",
        "group_name": "CA1",
        "ripple_param_name": "default",
        "filter_name": "ripple_150_250",
        "filter_sampling_rate": 1000,
    }
    assert lfp_band_restriction == {
        "nwb_file_name": "chimi20200225_.nwb",
        "filter_name": "ripple_150_250",
        "target_interval_list_name": "02_r1 noPrePostTrialTimes",
        "lfp_band_sampling_rate": 1000,
    }


def test_band_electrode_sets_must_match():
    """Equal sets pass in any order; a missing or an extra electrode is named."""
    lfp.check_band_electrodes(np.array([0, 4, 8]), np.array([8, 0, 4]))

    with pytest.raises(ValueError, match=r"missing from the band: \[12\]"):
        lfp.check_band_electrodes(np.array([0, 4, 8]), np.array([0, 4, 8, 12]))
    with pytest.raises(ValueError, match=r"extra in the band: \[16\]"):
        lfp.check_band_electrodes(np.array([0, 4, 8, 16]), np.array([0, 4, 8]))


# --- get_ripple_band_inputs, against stand-ins for Spyglass ------------------

_NWB_FILE_NAME = "testrat20990101_.nwb"
"""A synthetic session: no real recording has this name."""
_BAND_KEY = {
    "nwb_file_name": _NWB_FILE_NAME,
    "lfp_merge_id": "00000000-0000-0000-0000-000000000001",
    "filter_name": "ripple_150_250",
    "filter_sampling_rate": 1000,
    "target_interval_list_name": "02_r1 noPrePostTrialTimes",
    "lfp_band_sampling_rate": 1000,
}
_LFP_KEY = {"merge_id": _BAND_KEY["lfp_merge_id"]}
_INVALID_ELECTRODE_INDEX = 99999999
"""What Spyglass's ``get_electrode_indices`` returns for an electrode it lacks."""


class _FakeTable:
    """A Spyglass table: records each restriction and serves canned reads.

    It has no insert, populate or delete, so a write fails the test.
    """

    def __init__(self, fetch=None, fetch1=None, nwb=None):
        self.restrictions: list[dict] = []
        self.fetch_results = fetch or {}
        self.fetch1_results = fetch1 or {}
        self.nwb = nwb

    def __call__(self):
        return self

    def __and__(self, restriction):
        self.restrictions.append(restriction)
        return self

    def fetch(self, *attributes):
        return self.fetch_results[attributes]

    def fetch1(self, *attributes):
        return self.fetch1_results[attributes]

    def fetch_nwb(self):
        return [self.nwb]


def _fake_get_electrode_indices(series, electrode_ids):
    """Spyglass's ``get_electrode_indices``, with each column's electrode given
    directly instead of through the file's electrode table."""
    return [
        series.electrode_ids.index(electrode_id)
        if electrode_id in series.electrode_ids
        else _INVALID_ELECTRODE_INDEX
        for electrode_id in electrode_ids
    ]


class _FakeLFPBandV1(_FakeTable):
    """``LFPBandV1`` with the two steps of ``make`` the fetch reuses."""

    def __init__(self, interval, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval
        self.interval_calls: list[tuple] = []
        self.load_calls: list[tuple] = []

    def _compute_interval_times(self, key, lfp_key):
        self.interval_calls.append((key, lfp_key))
        return 1000.0, self.interval, key["target_interval_list_name"], 1000, 1

    def _load_lfp_data_with_referencing(
        self, lfp_object, elect_ids, ref_ids, valid_times
    ):
        """Spyglass's referencing: in the stored dtype, from an unreferenced copy."""
        self.load_calls.append(
            (lfp_object, list(elect_ids), list(ref_ids), valid_times)
        )
        data = lfp_object.data.copy()
        original = data.copy()
        elect_index = _fake_get_electrode_indices(lfp_object, elect_ids)
        ref_index = _fake_get_electrode_indices(lfp_object, ref_ids)
        for index, column in enumerate(elect_index):
            if ref_ids[index] != -1:
                data[:, column] = original[:, column] - original[:, ref_index[index]]
        return lfp_object.timestamps.copy(), data, elect_index


def _raw_lfp() -> SimpleNamespace:
    """Ten int16 samples on four raw columns, holding electrodes 8, 0, 5 and 4.

    Electrodes 0 and 4 are referenced to 5, and 8 has no reference. Three
    subtractions leave the int16 range: 0 - 5 at samples 0 and 3, and 4 - 5 at
    sample 6. Sample 0 lies outside the band. Electrode 8's large value at
    sample 5 is no subtraction, so it does not count.
    """
    data = np.random.default_rng(0).integers(-1000, 1000, size=(10, 4)).astype(np.int16)
    e8, e0, e5, e4 = range(4)
    data[0, [e0, e5, e4]] = [30000, -10000, 0]
    data[3, [e0, e5, e4]] = [20000, -20000, 0]
    data[6, [e0, e5, e4]] = [0, 10000, -30000]
    data[5, e8] = 32000
    return SimpleNamespace(
        data=data,
        timestamps=100.0 + np.arange(10) / 1000,
        unit="volts",
        electrode_ids=[8, 0, 5, 4],
    )


@pytest.fixture
def fake_spyglass(monkeypatch):
    """Every Spyglass module the fetch imports, as stand-ins in ``sys.modules``.

    The stored band covers raw samples 1-8 on electrodes 0, 4 and 8. The band
    selection lists its electrodes out of order, as a database fetch may.
    """
    raw = _raw_lfp()
    band = SimpleNamespace(
        data=np.arange(24, dtype=np.int16).reshape(8, 3),
        timestamps=raw.timestamps[1:9].copy(),
        electrode_ids=[0, 4, 8],
    )
    interval = SimpleNamespace(times=np.array([[raw.timestamps[1], raw.timestamps[8]]]))
    fakes = SimpleNamespace(
        raw=raw,
        band=band,
        interval=interval,
        band_v1=_FakeLFPBandV1(
            interval, fetch1={("KEY",): dict(_BAND_KEY)}, nwb={"lfp_band": band}
        ),
        band_electrode=_FakeTable(
            fetch={
                ("electrode_id", "reference_elect_id"): (
                    np.array([8, 0, 4]),
                    np.array([-1, 5, 5]),
                )
            }
        ),
        ripple_electrode=_FakeTable(fetch={("electrode_id",): np.array([0, 4, 8])}),
        lfp_output=_FakeTable(nwb={"lfp": raw}),
        lfp_v1=_FakeTable(fetch1={("filter_name",): "LFP 0-400 Hz"}),
        fir_filter=_FakeTable(fetch1={("filter_coeff",): np.array([0.25, 0.5, 0.25])}),
    )
    fakes.lfp_output.merge_get_parent = lambda restriction: fakes.lfp_v1 & restriction

    modules = {
        "spyglass": {},
        "spyglass.common": {"FirFilterParameters": fakes.fir_filter},
        "spyglass.lfp.analysis.v1.lfp_band": {
            "LFPBandSelection": SimpleNamespace(LFPBandElectrode=fakes.band_electrode),
            "LFPBandV1": fakes.band_v1,
        },
        "spyglass.lfp.lfp_merge": {"LFPOutput": fakes.lfp_output},
        "spyglass.ripple.v1": {
            "RippleLFPSelection": SimpleNamespace(
                RippleLFPElectrode=fakes.ripple_electrode
            )
        },
        "spyglass.utils.nwb_helper_fn": {
            "get_electrode_indices": _fake_get_electrode_indices
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return fakes


def test_ripple_band_inputs_reproduce_make_and_count_overflow(fake_spyglass):
    """The fetch reuses ``LFPBandV1.make``'s own steps, read-only.

    It takes the band key from ``_ripple_keys``, sorts the band's electrodes as
    make does and checks them against the CA1 selection for that same band key.
    The referencing comes from make's own step, in the stored dtype. Wrapped
    subtractions are counted against a float64 subtraction, over every raw
    sample. The float64 copy holds the band's electrodes in order, at the
    band's timestamps.
    """
    fakes = fake_spyglass
    inputs = lfp.get_ripple_band_inputs(_NWB_FILE_NAME, "02_r1")

    _, lfp_band_restriction = lfp._ripple_keys(_NWB_FILE_NAME, "02_r1")
    assert fakes.band_v1.restrictions[0] == lfp_band_restriction
    assert fakes.band_electrode.restrictions == [_BAND_KEY]
    assert fakes.ripple_electrode.restrictions == [{**_BAND_KEY, "group_name": "CA1"}]
    assert fakes.lfp_output.restrictions == [_LFP_KEY]
    assert fakes.lfp_v1.restrictions == [_LFP_KEY]
    assert fakes.fir_filter.restrictions == [
        {"filter_name": "ripple_150_250", "filter_sampling_rate": 1000}
    ]
    assert fakes.band_v1.interval_calls == [(_BAND_KEY, _LFP_KEY)]
    assert (
        fakes.raw,
        [0, 4, 8],
        [5, 5, -1],
        fakes.interval,
    ) in fakes.band_v1.load_calls

    np.testing.assert_array_equal(inputs["electrode_ids"], [0, 4, 8])
    np.testing.assert_array_equal(inputs["reference_ids"], [5, 5, -1])
    np.testing.assert_array_equal(inputs["lfp_band_elect_index"], [1, 3, 0])

    raw = fakes.raw.data
    referenced = raw.copy()
    referenced[:, 1] = raw[:, 1] - raw[:, 2]  # electrode 0 - 5, wrapping in int16
    referenced[:, 3] = raw[:, 3] - raw[:, 2]  # electrode 4 - 5
    assert inputs["raw_referenced_native"].dtype == np.int16
    np.testing.assert_array_equal(inputs["raw_referenced_native"], referenced)
    np.testing.assert_array_equal(inputs["raw_timestamps"], fakes.raw.timestamps)
    assert inputs["reference_overflow_count"] == 3

    np.testing.assert_array_equal(inputs["time"], fakes.band.timestamps)
    assert inputs["raw_referenced"].dtype == np.float64
    np.testing.assert_array_equal(
        inputs["raw_referenced"], referenced[1:9][:, [1, 3, 0]].astype(np.float64)
    )
    assert inputs["ripple_band"].dtype == np.float64
    np.testing.assert_array_equal(
        inputs["ripple_band"], fakes.band.data.astype(np.float64)
    )
    assert inputs["ripple_band_dtype"] == "int16"

    assert inputs["band_valid_interval"] is fakes.interval
    assert inputs["decimation"] == 1
    assert inputs["sampling_frequency"] == 1000.0
    np.testing.assert_array_equal(inputs["band_filter_coeff"], [0.25, 0.5, 0.25])
    assert inputs["band_filter_name"] == "ripple_150_250"
    assert inputs["lfp_filter_name"] == "LFP 0-400 Hz"
    assert inputs["lfp_units"] == "volts"
    assert set(inputs) == set(project_types.RippleBandInputs.__annotations__)


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        ("ca1_selection", r"missing from the band: \[12\]"),
        ("band_columns", r"columns do not hold electrodes \[0, 4, 8\] in order"),
        ("band_timestamps", r"8 of the ripple band's 8 timestamps"),
    ],
)
def test_ripple_band_inputs_refuse_misaligned_inputs(fake_spyglass, breakage, message):
    """A CA1 selection that differs from the band, a stored band whose columns
    are not its electrodes in order, and band timestamps the raw LFP lacks each
    stop the fetch rather than mislabel a channel or a sample."""
    fakes = fake_spyglass
    if breakage == "ca1_selection":
        fakes.ripple_electrode.fetch_results[("electrode_id",)] = np.array(
            [0, 4, 8, 12]
        )
    elif breakage == "band_columns":
        fakes.band.electrode_ids = [4, 0, 8]
    else:
        fakes.band.timestamps = fakes.band.timestamps + 0.0005

    with pytest.raises(ValueError, match=message):
        lfp.get_ripple_band_inputs(_NWB_FILE_NAME, "02_r1")
