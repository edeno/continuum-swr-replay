"""Regression test for ``compute_session_quality_summary`` empty-calib path.

``compute_calibration_curve`` returns an empty DataFrame (no
``calibration_slope`` / ``is_monotonic_decreasing`` columns) when
there's nothing to calibrate. The summary used to do
``calib_df.groupby(...).first()[["calibration_slope", ...]]`` which
raised ``KeyError: ['calibration_slope', 'is_monotonic_decreasing']``.

The fix substitutes a zero-row, schema-complete placeholder when
the calib output is empty so the downstream merge still has the
expected columns.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    compute_session_quality_summary,
)


def _empty_schema_frame() -> pd.DataFrame:
    """Schema-complete frame with zero rows that the summary should
    be able to digest without crashing."""
    columns = [
        "animal",
        "day",
        "epoch",
        "head_speed",
        "linear_position",
        "Local_prob",
        "No-Spike_prob",
        "Non-Local_Continuous_prob",
        "Non-Local_Fragmented_prob",
        "ahead_behind_distance",
        "hpd",
        "non_local_event",
        "SWR_event",
        "MUA_event",
        "theta_power_zscore",
        "max_distance",
        "max_coverage",
        "max_linear_position",
    ]
    return pd.DataFrame(
        {
            **{c: pd.Series(dtype="float64") for c in columns},
            "in_excluded_time": pd.Series(dtype=bool),
        }
    )


class TestEmptyCalibSummary:
    def test_empty_schema_input_does_not_crash_on_calib_join(self):
        # Should not raise KeyError on calibration_slope /
        # is_monotonic_decreasing.
        out = compute_session_quality_summary(_empty_schema_frame())
        # Empty input ⇒ empty output is the natural result; the
        # crucial property is that we *don't* raise.
        assert isinstance(out, pd.DataFrame)
