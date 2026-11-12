"""aggregate_contrast must return a schema-carrying frame when nothing survives.

When every term is unidentified in more than ``MAX_UNIDENTIFIED_SHARE`` of the
fitted sessions, no aggregate row is emitted. The result must still be a frame
with the full output schema, so a caller that selects a column
(``agg["population"].map(...)`` in the rate/content contrast runners) gets an
empty frame rather than a KeyError. A mostly-separated corpus is a valid,
reportable "nothing identified" -- not a crash after a multi-hour fit.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    _AGG_COLUMNS,
    aggregate_contrast,
)


def _all_separated_models() -> pd.DataFrame:
    """Six session-fits for one (model, population), every one flagged separated."""
    return pd.DataFrame(
        {
            "model": ["G0_base"] * 6,
            "population": ["non_local"] * 6,
            "animal": list("aabbcc"),
            "session_date": [1, 2, 1, 2, 1, 2],
            "epoch": ["e"] * 6,
            "coef_x": [1.0] * 6,
            "bse_x": [0.1] * 6,
            "pval_x": [0.5] * 6,
            "converged": [True] * 6,
            "separated": [True] * 6,
        }
    )


def test_all_gated_out_returns_empty_frame_with_schema():
    agg = aggregate_contrast(
        _all_separated_models(), [("G0_base", "non_local", "x")], "rate", {"x": 1.0}
    )

    assert len(agg) == 0
    # the exact columns a caller selects must be present on the empty frame
    for column in (*_AGG_COLUMNS, "session_signrank_fdr_q"):
        assert column in agg.columns, f"empty agg missing {column}"
    # the operation that used to crash must now be a no-op
    agg["population"] = agg["population"].map(lambda p: p)


def test_identified_term_still_emits_a_row():
    """Guard the fix did not suppress a genuinely identifiable term."""
    models = _all_separated_models()
    models["separated"] = False  # now identified

    agg = aggregate_contrast(
        models, [("G0_base", "non_local", "x")], "rate", {"x": 1.0}
    )

    assert len(agg) == 1
    assert agg.iloc[0]["model"] == "G0_base"
    assert agg.iloc[0]["n_sessions"] == 6


def test_empty_and_populated_schemas_match():
    """The hand-maintained _AGG_COLUMNS empty schema must equal the columns a
    populated aggregation produces, or a caller's column op KeyErrors only on
    all-separated corpora (the rare, hard-to-reproduce case)."""
    identified = _all_separated_models()
    identified["separated"] = False
    populated = aggregate_contrast(
        identified, [("G0_base", "non_local", "x")], "rate", {"x": 1.0}
    )
    empty = aggregate_contrast(
        _all_separated_models(), [("G0_base", "non_local", "x")], "rate", {"x": 1.0}
    )
    assert list(empty.columns) == list(populated.columns)
