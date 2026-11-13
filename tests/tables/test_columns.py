"""The required-column check."""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.tables.columns import require_columns


def test_a_table_with_every_column_passes():
    frame = pd.DataFrame(columns=["stop_id", "n_events", "exposure"])

    require_columns(frame, ["exposure", "stop_id"], "stop table")
    require_columns(frame, [], "stop table")


def test_the_message_names_the_table_and_the_missing_columns_sorted():
    frame = pd.DataFrame(columns=["stop_id"])

    # a column asked twice is named once, and a set gives the same message
    message = r"^stop table lacks required column\(s\) \['exposure', 'n_events'\]$"
    with pytest.raises(ValueError, match=message):
        require_columns(
            frame, ["n_events", "stop_id", "exposure", "n_events"], "stop table"
        )
    with pytest.raises(ValueError, match=message):
        require_columns(frame, {"n_events", "exposure"}, "stop table")


def test_an_empty_table_is_checked_too():
    with pytest.raises(ValueError, match="'stop_id'"):
        require_columns(pd.DataFrame(), ["stop_id"], "stop table")
