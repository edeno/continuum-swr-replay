"""Tests for the PFC path's entry point to the canonical events and SWRs.

The point of this module is that Figure 4's events are the *same objects* the
content and rate families describe. So the tests check the two things that
could silently break that: the session restriction and the SWR interval
population, the latter against the builder these loaders promise to agree with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events import analysed_time, swr_inventory
from continuum_swr_replay.analysis.events.immobility import (
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    IS_SWR_DEFINITION,
    canonical_event_input_hashes,
    load_session_canonical_events,
    load_session_swr_intervals,
    session_data_over_time_path,
)
from continuum_swr_replay.tables.io import read_table, sha256_file, write_table
from tests.unit._swr_inventory_fixtures import RUN_ID, write_swr_inventory


@pytest.fixture
def event_table_path(tmp_path):
    """A two-session event table, deliberately out of start-time order."""
    frame = pd.DataFrame(
        {
            "animal": ["j16"] * 4 + ["wilbur"],
            "session_date": [20210706] * 4 + [20210329],
            "epoch": ["02_r1", "02_r1", "02_r1", "04_r2", "02_r1"],
            "event_id": np.array([11, 12, 13, 14, 15], dtype=np.uint64),
            "event_start_s": [30.0, 10.0, 20.0, 40.0, 50.0],
            "event_end_s": [30.5, 10.5, 20.5, 40.5, 50.5],
            "is_swr": [True, False, True, False, True],
            "occupancy_state": ["at_well", "at_well", "in_transit", "mixed", "at_well"],
            "peak_nonlocal_prob": [0.9, 0.95, 0.85, 0.99, 0.91],
        }
    )
    path = tmp_path / "event_table.parquet"
    write_table(analysed_time.stamp_swr_source(frame), path)
    return path


def _synthetic_dot(labels, speed, *, animal="j16", day=20210706, epoch="02_r1"):
    """A minimal data_over_time frame: one SWR label column and a speed."""
    time = np.arange(len(labels), dtype=float) / 500.0
    return pd.DataFrame(
        {
            "SWR_event": np.asarray(labels, dtype=int),
            "head_speed": np.asarray(speed, dtype=float),
            "animal": animal,
            "day": day,
            "epoch": epoch,
        },
        index=time,
    )


class TestLoadSessionCanonicalEvents:
    def test_only_that_sessions_events_are_returned_and_they_are_sorted(
        self, event_table_path
    ):
        events, _ = load_session_canonical_events(
            "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
        )

        # The other epoch (14) and the other animal (15) are excluded; the
        # remaining three arrive in start-time order, not table order.
        assert list(events.event_id) == [12, 13, 11]
        np.testing.assert_allclose(events.start_time.to_numpy(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(events.end_time.to_numpy(), [10.5, 20.5, 30.5])
        assert list(events.is_swr) == [False, True, True]
        assert list(events.occupancy_state) == ["at_well", "in_transit", "at_well"]
        assert list(events.index) == [0, 1, 2]

    def test_the_shipped_columns_are_exactly_what_the_glm_is_keyed_on(
        self, event_table_path
    ):
        events, _ = load_session_canonical_events(
            "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
        )

        # A stray extra column would reach the design-matrix builder untested;
        # the renamed timing columns are the ones it reads.
        assert list(events.columns) == [
            "event_id",
            "start_time",
            "end_time",
            "is_swr",
            "occupancy_state",
        ]

    def test_the_hash_is_of_the_table_that_was_read(self, event_table_path):
        _, sha = load_session_canonical_events(
            "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
        )

        assert sha == sha256_file(event_table_path)

    def test_a_session_absent_from_the_event_table_is_refused(self, event_table_path):
        # Returning an empty frame would fit a session with no event regressors
        # at all and report it as a successful run.
        with pytest.raises(ValueError, match="no events for"):
            load_session_canonical_events(
                "chimi20200212_.nwb", "02_r1", event_table_path=event_table_path
            )

    def test_a_missing_event_table_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session_canonical_events(
                "j1620210706_.nwb",
                "02_r1",
                event_table_path=tmp_path / "absent.parquet",
            )

    @pytest.mark.parametrize(
        "stamp", [None, "kay@an_earlier_run+artifacts@none"], ids=["unstamped", "other"]
    )
    def test_a_table_from_another_swr_source_is_refused(self, event_table_path, stamp):
        # Its is_swr would be split against this source's live SWR intervals.
        frame = read_table(event_table_path)
        frame.attrs.pop("swr_source")
        if stamp is not None:
            frame.attrs["swr_source"] = stamp
        write_table(frame, event_table_path)

        with pytest.raises(swr_inventory.SwrSourceError, match="event table"):
            load_session_canonical_events(
                "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
            )

    def test_a_table_from_the_configured_swr_source_is_read(self, event_table_path):
        assert read_table(event_table_path).attrs["swr_source"] == (
            swr_inventory.swr_source_name()
        )

        events, _ = load_session_canonical_events(
            "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
        )

        assert len(events) == 3

    def test_a_table_without_the_event_columns_is_refused(self, tmp_path):
        frame = pd.DataFrame(
            {
                "animal": ["j16"],
                "session_date": [20210706],
                "epoch": ["02_r1"],
                "event_id": np.array([1], dtype=np.uint64),
            }
        )
        path = tmp_path / "event_table.parquet"
        write_table(analysed_time.stamp_swr_source(frame), path)

        with pytest.raises(KeyError, match="event_start_s"):
            load_session_canonical_events(
                "j1620210706_.nwb", "02_r1", event_table_path=path
            )


class TestLoadSessionSwrIntervals:
    """The intervals are the SWR source's, gated and bounded on the frame's grid.

    The frame's own ``SWR_event`` (the decode's stored ripples) marks nothing
    the source has, so an interval read from it would show.
    """

    @pytest.fixture
    def source(self, tmp_path, monkeypatch):
        def write(intervals):
            root = tmp_path / "swr"
            write_swr_inventory(root, {"j1620210706_02_r1_HPC": intervals})
            monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_ROOT", root)
            monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_RUN_ID", RUN_ID)

        return write

    def test_the_intervals_equal_the_canonical_builders_on_the_sources_labels(
        self, tmp_path, source
    ):
        # Three source SWRs: the middle one spans a moving sample and so is
        # removed atomically by the whole-event rule, exactly as the classified
        # event table's SWR population is.
        source([(1 / 500, 2 / 500), (4 / 500, 6 / 500), (8 / 500, 9 / 500)])
        stored = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        speed = [1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 2.0, 2.0, 1.0]
        dot = _synthetic_dot(stored, speed)
        path = tmp_path / "j1620210706_02_r1_HPC_80_data_over_time.pkl"
        dot.to_pickle(path)

        intervals, _ = load_session_swr_intervals(
            "j1620210706_.nwb", "02_r1", dot_path=path
        )

        expected = intervals_from_labels(
            filter_event_labels_by_speed(
                analysed_time.with_swr_source(dot), ["SWR_event"], speed_threshold=4.0
            ),
            "SWR_event",
        )
        np.testing.assert_allclose(
            intervals.start_time.to_numpy(), expected.start_time.to_numpy()
        )
        np.testing.assert_allclose(
            intervals.end_time.to_numpy(), expected.end_time.to_numpy()
        )
        assert list(intervals.columns) == ["start_time", "end_time"]
        # The moving-sample event is the one that must be gone, and the stored
        # ripple at the last sample was never read.
        assert len(intervals) == 2
        np.testing.assert_allclose(
            intervals.start_time.to_numpy(), np.array([1, 8]) / 500.0
        )
        np.testing.assert_allclose(
            intervals.end_time.to_numpy(), np.array([2, 9]) / 500.0
        )

    def test_a_session_with_no_eligible_swr_returns_an_empty_frame(
        self, tmp_path, source
    ):
        source([(1 / 500, 2 / 500)])
        dot = _synthetic_dot([0, 1, 1, 0], [1.0, 9.0, 1.0, 1.0])
        path = tmp_path / "dot.pkl"
        dot.to_pickle(path)

        intervals, _ = load_session_swr_intervals(
            "j1620210706_.nwb", "02_r1", dot_path=path
        )

        assert list(intervals.columns) == ["start_time", "end_time"]
        assert len(intervals) == 0

    def test_the_hash_is_of_the_frame_that_was_read(self, tmp_path, source):
        source([])
        dot = _synthetic_dot([0, 1, 1, 0], [1.0, 1.0, 1.0, 1.0])
        path = tmp_path / "dot.pkl"
        dot.to_pickle(path)

        _, sha = load_session_swr_intervals("j1620210706_.nwb", "02_r1", dot_path=path)

        assert sha == sha256_file(path)

    def test_a_missing_cached_frame_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no cached decode frame"):
            load_session_swr_intervals(
                "j1620210706_.nwb", "02_r1", dot_path=tmp_path / "absent.pkl"
            )


class TestSessionDataOverTimePath:
    def test_the_threshold_tag_names_the_cached_frame(self, tmp_path):
        path = session_data_over_time_path(
            "j1620210706_.nwb", "02_r1", data_dir=tmp_path
        )

        assert path.name == "j1620210706_02_r1_HPC_80_data_over_time.pkl"


class TestCanonicalEventInputHashes:
    def test_a_missing_input_hashes_to_none_so_the_session_is_refit(
        self, tmp_path, monkeypatch
    ):
        import continuum_swr_replay.analysis.pfc.canonical_events as module

        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path)
        monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_ROOT", tmp_path / "swr")

        provenance = canonical_event_input_hashes(
            "j1620210706_.nwb", "02_r1", event_table_path=tmp_path / "absent.parquet"
        )

        assert provenance["is_swr_definition"] == IS_SWR_DEFINITION
        assert provenance["immobility_inequality"] == "<"
        assert provenance["canonical_event_table_sha256"] is None
        assert provenance["data_over_time_sha256"] is None
        assert provenance["swr_events_sha256"] is None

    def test_present_inputs_hash_to_the_files_on_disk(
        self, tmp_path, event_table_path, monkeypatch
    ):
        import continuum_swr_replay.analysis.pfc.canonical_events as module

        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path)
        write_swr_inventory(tmp_path / "swr", {"j1620210706_02_r1_HPC": []})
        monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_ROOT", tmp_path / "swr")
        monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_RUN_ID", RUN_ID)
        dot = _synthetic_dot([0, 1, 1, 0], [1.0, 1.0, 1.0, 1.0])
        dot_path = tmp_path / "j1620210706_02_r1_HPC_80_data_over_time.pkl"
        dot.to_pickle(dot_path)

        provenance = canonical_event_input_hashes(
            "j1620210706_.nwb", "02_r1", event_table_path=event_table_path
        )

        assert provenance["canonical_event_table_sha256"] == sha256_file(
            event_table_path
        )
        assert provenance["data_over_time_sha256"] == sha256_file(dot_path)
        assert provenance["data_over_time_path"] == str(dot_path)
        events = tmp_path / "swr" / RUN_ID / "j1620210706_02_r1_HPC_kay_events.parquet"
        assert provenance["swr_interval_source"] == f"kay@{RUN_ID}+artifacts@none"
        assert provenance["swr_events_path"] == str(events)
        assert provenance["swr_events_sha256"] == sha256_file(events)
