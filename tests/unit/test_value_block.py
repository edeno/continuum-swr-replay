"""Tests for the qvals value-block clock guard.

``qvals_table`` joins onto the stop table by ``stop_id``, a hash that carries
no timestamp. These tests pin the guard that compares the two clocks so a
renumbered decode fails loudly instead of misjoining.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.covariates import assert_qvals_aligned


def _stop_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": [11, 12, 13],
            "epoch": ["02_r1", "02_r1", "04_r2"],
            "stop_start": [100.0, 140.0, 900.0],
        }
    )


def _qvals(stop_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": stop_table["stop_id"],
            "epoch": stop_table["epoch"],
            "qvals_context_time": stop_table["stop_start"],
        }
    )


def test_matching_clocks_pass():
    stop = _stop_table()
    assert assert_qvals_aligned(stop, _qvals(stop)) is None


def test_shifted_context_time_raises():
    stop = _stop_table()
    qvals = _qvals(stop)
    qvals.loc[1, "qvals_context_time"] = 140.5
    with pytest.raises(ValueError, match="off the stop table's clock"):
        assert_qvals_aligned(stop, qvals)


def test_drift_within_tolerance_passes():
    stop = _stop_table()
    qvals = _qvals(stop)
    qvals.loc[1, "qvals_context_time"] = 140.0 + 5e-7
    assert assert_qvals_aligned(stop, qvals) is None


def test_qvals_stop_id_absent_from_stop_table_raises():
    stop = _stop_table()
    qvals = _qvals(stop)
    # A renumbered decode: same bout start, different hash.
    qvals.loc[0, "stop_id"] = 999
    with pytest.raises(ValueError, match="not in the stop table"):
        assert_qvals_aligned(stop, qvals)


def test_error_names_the_first_offending_stop_and_both_times():
    stop = _stop_table()
    qvals = _qvals(stop)
    qvals.loc[1, "qvals_context_time"] = 141.25
    with pytest.raises(ValueError) as excinfo:
        assert_qvals_aligned(stop, qvals)
    message = str(excinfo.value)
    assert "stop_id=12" in message
    assert "141.25" in message
    assert "140.0" in message


def test_session_scoped_stop_table_ignores_other_sessions():
    """One session's stop table must not be judged against the whole corpus.

    ``qvals_table`` has no animal/date column, so a per-session caller passes
    the corpus-wide artifact. Rows whose bout start lies outside the session's
    own span belong to another session and are not this table's to account for.
    """
    stop = _stop_table().iloc[:2]  # the 02_r1 session only
    qvals = _qvals(_stop_table())
    assert assert_qvals_aligned(stop, qvals) is None


def test_no_qvals_rows_for_the_session_is_not_an_error():
    stop = _stop_table().iloc[[2]]
    qvals = _qvals(_stop_table()).iloc[:2]
    assert assert_qvals_aligned(stop, qvals) is None


def test_absent_contingency_is_not_an_error():
    assert assert_qvals_aligned(_stop_table(), None) is None


def test_missing_columns_raise():
    stop = _stop_table()
    with pytest.raises(ValueError, match="missing columns"):
        assert_qvals_aligned(stop, _qvals(stop).drop(columns=["qvals_context_time"]))
    with pytest.raises(ValueError, match="missing columns"):
        assert_qvals_aligned(stop.drop(columns=["stop_start"]), _qvals(stop))
