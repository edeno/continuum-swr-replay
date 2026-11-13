"""Regression test for the drift check in HPC + ContFrag get_data_over_time.

The aggregations build ``data_over_time`` on ``position_info.index``
but read column values from ``results.time``-indexed arrays. A
length-only check passed when the two axes had the same row count
but were time-shifted; that silently attached event labels /
probabilities / MAP stats to the wrong position rows.

Both aggregators now route through the shared helper
:func:`continuum_swr_replay.analysis.detection.nonlocal_events.validate_results_position_time_alignment`.
This test exercises that helper directly — a regression that
keeps the same error strings but computes drift against the wrong
axis, swaps the reducer, or weakens the tolerance fails the
behavioral assertions below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.nonlocal_events import (
    validate_results_position_time_alignment,
)


class TestValidateResultsPositionTimeAlignment:
    def test_aligned_axes_pass(self):
        t = np.linspace(100.0, 110.0, 1000)
        validate_results_position_time_alignment(t, pd.Index(t, name="time"))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            validate_results_position_time_alignment(
                np.arange(100), pd.Index(np.arange(99))
            )

    def test_same_length_time_shifted_axes_raise(self):
        t = np.linspace(100.0, 110.0, 1000)
        with pytest.raises(ValueError, match="drifts from position_info.index"):
            validate_results_position_time_alignment(t, pd.Index(t + 0.5, name="time"))

    def test_within_tolerance_passes(self):
        """1e-9 drift sits well below the 1e-6 tolerance — confirms the
        boundary, so a regression that tightens the tolerance to e.g.
        1e-12 (rejecting genuine float-roundtrip drift) trips here."""
        t = np.linspace(100.0, 110.0, 1000)
        validate_results_position_time_alignment(t, pd.Index(t + 1e-9, name="time"))

    def test_just_above_tolerance_raises(self):
        """A regression that loosens the tolerance to e.g. 1e-3 must
        not silently accept drift just above 1e-6."""
        t = np.linspace(100.0, 110.0, 1000)
        with pytest.raises(ValueError, match="drifts from position_info.index"):
            validate_results_position_time_alignment(t, pd.Index(t + 2e-6, name="time"))

    def test_max_reducer_not_mean(self):
        """Pin the reducer: a *single* drifting bin must trip the
        guard, not just be averaged into a passing global mean.
        Regression: switching to ``mean`` would let one large drift
        slip through whenever the rest of the axis is tightly
        aligned."""
        t = np.linspace(100.0, 110.0, 1000)
        idx_vals = t.copy()
        idx_vals[500] = idx_vals[500] + 1.0  # one bin off by 1s
        with pytest.raises(ValueError, match="drifts from position_info.index"):
            validate_results_position_time_alignment(t, pd.Index(idx_vals, name="time"))

    def test_drift_computed_against_position_index_not_zero(self):
        """A regression that diffed ``results_time`` against itself,
        zero, or any constant other than ``position_index`` would let
        a constant-offset position axis slip through (both axes
        re-encoded as the same offset)."""
        t = np.linspace(100.0, 110.0, 1000)
        with pytest.raises(ValueError, match="drifts from position_info.index"):
            validate_results_position_time_alignment(t, pd.Index(t + 5.0, name="time"))
