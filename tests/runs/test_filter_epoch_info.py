"""The runners' epoch filters (``runs.cli``): ``filter_epoch_info`` over an epoch
table, ``read_session_list`` for ``--sessions``, ``epoch_ids_matching`` over
epoch ids.

``--epoch`` is matched against the full epoch name. It used to keep only the
number before ``_`` (``"02_r1"`` -> ``2``), so ``--epoch 02_r1`` also selected
``02_r3`` of another day, and a mistyped tag such as ``02_r9`` still selected
every epoch 2.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.runs.cli import (
    epoch_ids_matching,
    filter_epoch_info,
    read_session_list,
)


def _epoch_info() -> pd.DataFrame:
    """``get_epoch_info``'s shape: two animals, epoch 2 on two days."""
    index = pd.MultiIndex.from_tuples(
        [
            ("a20200101_.nwb", "02_r1"),
            ("a20200101_.nwb", "04_r2"),
            ("a20200102_.nwb", "02_r3"),
            ("b20200101_.nwb", "02_r1"),
        ],
        names=["nwb_file_name", "interval_list_name"],
    )
    return pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b"],
            "date": [20200101, 20200101, 20200102, 20200101],
            "epoch": [2, 4, 2, 2],
            "epoch_tag": ["r1", "r2", "r3", "r1"],
        },
        index=index,
    )


def test_epoch_matches_the_full_epoch_name():
    kept = filter_epoch_info(_epoch_info(), epoch="02_r1")
    assert list(kept.index) == [
        ("a20200101_.nwb", "02_r1"),
        ("b20200101_.nwb", "02_r1"),
    ]


def test_epoch_with_a_wrong_tag_matches_nothing():
    kept = filter_epoch_info(_epoch_info(), epoch="02_r9")
    assert kept.empty
    assert list(kept.columns) == list(_epoch_info().columns)


def test_filters_combine_and_one_row_stays_a_frame():
    kept = filter_epoch_info(_epoch_info(), animal="a", date=20200102, epoch="02_r3")
    assert isinstance(kept, pd.DataFrame)
    assert list(kept.index) == [("a20200102_.nwb", "02_r3")]


def test_no_filter_keeps_every_epoch():
    pd.testing.assert_frame_equal(filter_epoch_info(_epoch_info()), _epoch_info())


def _sessions(keys: list[tuple[str, str]], **columns) -> pd.DataFrame:
    """Session table indexed the way ``get_epoch_info`` returns it."""
    index = pd.MultiIndex.from_tuples(
        keys, names=["nwb_file_name", "interval_list_name"]
    )
    frame = pd.DataFrame(
        {
            "animal": [key[0].split("2")[0] for key in keys],
            "date": list(range(len(keys))),
        },
        index=index,
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestReadSessionList:
    def test_it_keeps_file_order_and_drops_repeats_and_comments(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text(
            "# worker 0\nc_.nwb 03_r1\n\na_.nwb,01_r1  # inline comment\nc_.nwb 03_r1\n"
        )

        assert read_session_list(path) == [("c_.nwb", "03_r1"), ("a_.nwb", "01_r1")]

    def test_it_refuses_a_malformed_line(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text("a_.nwb 01_r1\na_.nwb 01_r1 extra\n")

        with pytest.raises(ValueError, match="sessions.txt:2: expected"):
            read_session_list(path)

    def test_it_refuses_a_file_naming_no_sessions(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text("# nothing here\n\n")

        with pytest.raises(ValueError, match="names no sessions"):
            read_session_list(path)


class TestFilterEpochInfo:
    def test_a_session_file_sets_the_run_order(self, tmp_path):
        epochs = _sessions(
            [("a_.nwb", "01_r1"), ("b_.nwb", "02_r1"), ("c_.nwb", "03_r1")]
        )
        session_file = tmp_path / "sessions.txt"
        session_file.write_text("c_.nwb 03_r1\na_.nwb 01_r1\n")

        selected = filter_epoch_info(epochs, sessions=session_file)

        assert list(selected.index) == [("c_.nwb", "03_r1"), ("a_.nwb", "01_r1")]

    def test_without_a_session_file_the_source_order_is_kept(self):
        epochs = _sessions(
            [("a_.nwb", "01_r1"), ("b_.nwb", "02_r1"), ("c_.nwb", "03_r1")]
        )

        selected = filter_epoch_info(epochs)

        assert list(selected.index) == list(epochs.index)

    def test_animal_date_and_epoch_filters_compose(self):
        epochs = _sessions(
            [("a20200101_.nwb", "01_r1"), ("a20200101_.nwb", "02_r1")],
            animal=["a", "a"],
            date=[20200101, 20200101],
        )

        selected = filter_epoch_info(epochs, animal="a", date=20200101, epoch="02_r1")

        assert list(selected.index) == [("a20200101_.nwb", "02_r1")]
        assert filter_epoch_info(epochs, animal="b").empty

    def test_an_unavailable_requested_session_is_refused(self, tmp_path):
        epochs = _sessions([("a_.nwb", "01_r1")])
        session_file = tmp_path / "sessions.txt"
        session_file.write_text("z_.nwb 09_r9\n")

        with pytest.raises(ValueError, match="unavailable session"):
            filter_epoch_info(epochs, sessions=session_file)


def test_epoch_ids_match_animal_date_and_epoch_in_their_given_order():
    """The date is the NWB stem's last 8 characters, so j16's own digits are
    not read as part of it; with no filter nothing is parsed."""
    epoch_ids = [
        "chimi20990225_04_r2_HPC",
        "chimi20990225_02_r1_HPC",
        "chimi20990226_02_r1_HPC",
        "j1620990707_06_r3_HPC",
    ]

    assert epoch_ids_matching(epoch_ids, date=20990225, epoch="02_r1") == [
        "chimi20990225_02_r1_HPC"
    ]
    assert epoch_ids_matching(epoch_ids, date=20990707) == ["j1620990707_06_r3_HPC"]
    assert epoch_ids_matching(epoch_ids, animal="chimi", epoch="02_r1") == [
        "chimi20990225_02_r1_HPC",
        "chimi20990226_02_r1_HPC",
    ]
    assert epoch_ids_matching(epoch_ids, animal="j16") == ["j1620990707_06_r3_HPC"]
    assert epoch_ids_matching([*epoch_ids, "not-an-id"]) == [*epoch_ids, "not-an-id"]
    with pytest.raises(ValueError):
        epoch_ids_matching(["not-an-id"], animal="chimi")
