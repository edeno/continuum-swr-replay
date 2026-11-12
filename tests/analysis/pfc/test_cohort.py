"""Tests for the PFC cohorts: the epochs with sorted units, and the shared stops."""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.cohort import (
    load_session_stop_cohort,
    split_sorted_epochs,
)
from continuum_swr_replay.tables.io import write_table


@pytest.fixture
def stop_table_path(tmp_path):
    """A two-session stop table with every occupancy state represented."""
    frame = pd.DataFrame(
        {
            "animal": ["j16"] * 4 + ["wilbur"],
            "session_date": [20210706] * 4 + [20210329],
            "epoch": ["02_r1", "02_r1", "02_r1", "04_r2", "02_r1"],
            "stop_id": ["a", "b", "c", "d", "e"],
            "stop_start": [10.0, 20.0, 30.0, 40.0, 50.0],
            "stop_end": [11.0, 21.0, 31.0, 41.0, 51.0],
            "occupancy_state": [
                "at_well",
                "mixed",
                "in_transit",
                "at_well",
                "at_well",
            ],
        }
    )
    path = tmp_path / "stop_table.parquet"
    write_table(frame, path)
    return path


def test_at_well_keeps_the_well_anchored_stops_of_that_session(stop_table_path):
    stops, report = load_session_stop_cohort(
        "j1620210706_.nwb", "02_r1", cohort="at_well", stop_table_path=stop_table_path
    )

    # The session's in-transit stop, its other epoch, and the other animal are
    # all excluded; the well-anchored `mixed` stop is kept.
    assert list(stops["stop_id"]) == ["a", "b"]
    assert report["animal"] == "j16"
    assert report["session_date"] == 20210706
    assert report["epoch"] == "02_r1"
    assert report["n_stops_before"] == 3
    assert report["n_stops_after"] == 2
    assert report["cohort"] == "at_well"


def test_all_stops_keeps_every_stop_of_that_session(stop_table_path):
    stops, report = load_session_stop_cohort(
        "j1620210706_.nwb", "02_r1", cohort="all_stops", stop_table_path=stop_table_path
    )

    assert list(stops["stop_id"]) == ["a", "b", "c"]
    assert report["n_stops_after"] == 3


def test_the_report_hashes_the_table_that_was_read(stop_table_path):
    _, report = load_session_stop_cohort(
        "j1620210706_.nwb", "02_r1", stop_table_path=stop_table_path
    )

    from continuum_swr_replay.runs.io import sha256_file

    assert report["stop_table"] == str(stop_table_path)
    assert report["stop_table_sha256"] == sha256_file(stop_table_path)


def test_a_session_absent_from_the_stop_table_is_refused(stop_table_path):
    # Silently returning an empty cohort would fit the whole epoch instead.
    with pytest.raises(ValueError, match="no stops for"):
        load_session_stop_cohort(
            "chimi20200212_.nwb", "02_r1", stop_table_path=stop_table_path
        )


def test_an_epoch_absent_from_the_stop_table_is_refused(stop_table_path):
    with pytest.raises(ValueError, match="no stops for"):
        load_session_stop_cohort(
            "j1620210706_.nwb", "99_r9", stop_table_path=stop_table_path
        )


def test_the_pfc_epoch_cohort_is_the_analysed_epochs_of_sorted_sessions():
    """Two sorted files of three; the unsorted one's epochs are left out, and
    both parts keep the register's order. A sorted file the register does not
    analyse adds nothing."""
    analysed = pd.DataFrame(
        {"animal": ["chimi", "chimi", "chimi", "wilbur"]},
        index=pd.MultiIndex.from_tuples(
            [
                ("chimi20200224_.nwb", "02_r1"),
                ("chimi20200223_.nwb", "02_r1"),
                ("chimi20200224_.nwb", "04_r2"),
                ("wilbur20210329_.nwb", "02_r1"),
            ],
            names=["nwb_file_name", "interval_list_name"],
        ),
    )

    cohort, left_out = split_sorted_epochs(
        analysed, ["wilbur20210329_.nwb", "chimi20200224_.nwb", "j1620210706_.nwb"]
    )

    assert list(cohort.index) == [
        ("chimi20200224_.nwb", "02_r1"),
        ("chimi20200224_.nwb", "04_r2"),
        ("wilbur20210329_.nwb", "02_r1"),
    ]
    assert list(cohort["animal"]) == ["chimi", "chimi", "wilbur"]
    assert list(left_out) == [("chimi20200223_.nwb", "02_r1")]


def test_no_sorted_session_leaves_every_epoch_out():
    analysed = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [("chimi20200223_.nwb", "02_r1"), ("chimi20200223_.nwb", "04_r2")],
            names=["nwb_file_name", "interval_list_name"],
        )
    )

    cohort, left_out = split_sorted_epochs(analysed, [])

    assert cohort.empty
    assert list(left_out) == list(analysed.index)
