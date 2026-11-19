"""The epochs of an event table, as the per-event runs hold them."""

import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_epoch_ids,
    event_table_epochs,
)

EVENTS = pd.DataFrame(
    {
        "animal": ["chimi", "chimi", "wilbur"],
        "session_date": [20200212, 20200212, 20210329],
        "epoch": ["02_r1", "02_r1", "04_r2"],
    }
)


def test_event_epoch_ids_name_each_rows_epoch():
    assert event_epoch_ids(EVENTS).tolist() == [
        "chimi20200212_02_r1_HPC",
        "chimi20200212_02_r1_HPC",
        "wilbur20210329_04_r2_HPC",
    ]


def test_an_analyzed_epoch_without_events_is_not_expected_and_an_unanalyzed_one_refused():
    analyzed = [
        "chimi20200212_02_r1_HPC",
        "wilbur20210329_04_r2_HPC",
        "wilbur20210329_06_r3_HPC",  # no event
    ]
    assert event_table_epochs(EVENTS, analyzed) == [
        "chimi20200212_02_r1_HPC",
        "wilbur20210329_04_r2_HPC",
    ]
    with pytest.raises(ValueError, match="not analyzed"):
        event_table_epochs(EVENTS, analyzed[:1])
