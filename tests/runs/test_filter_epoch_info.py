"""``filter_epoch_info`` matches ``--epoch`` against the full epoch name.

It used to keep only the number before ``_`` (``"02_r1"`` -> ``2``), so
``--epoch 02_r1`` also selected ``02_r3`` of another day, and a mistyped tag
such as ``02_r9`` still selected every epoch 2.
"""

from __future__ import annotations

from argparse import Namespace

import pandas as pd

from continuum_swr_replay.runs.cli import filter_epoch_info


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


def _args(animal=None, date=None, epoch=None) -> Namespace:
    return Namespace(animal=animal, date=date, epoch=epoch)


def test_epoch_matches_the_full_epoch_name():
    kept = filter_epoch_info(_epoch_info(), _args(epoch="02_r1"))
    assert list(kept.index) == [
        ("a20200101_.nwb", "02_r1"),
        ("b20200101_.nwb", "02_r1"),
    ]


def test_epoch_with_a_wrong_tag_matches_nothing():
    kept = filter_epoch_info(_epoch_info(), _args(epoch="02_r9"))
    assert kept.empty
    assert list(kept.columns) == list(_epoch_info().columns)


def test_filters_combine_and_one_row_stays_a_frame():
    kept = filter_epoch_info(
        _epoch_info(), _args(animal="a", date=20200102, epoch="02_r3")
    )
    assert isinstance(kept, pd.DataFrame)
    assert list(kept.index) == [("a20200102_.nwb", "02_r3")]


def test_no_filter_keeps_every_epoch():
    pd.testing.assert_frame_equal(
        filter_epoch_info(_epoch_info(), _args()), _epoch_info()
    )
