"""Tests for PFC GLM functions.

Tests the reindex_by function tolerance parameter and related utilities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.glm import reindex_by


class TestReindexByTolerance:
    """Test tolerance parameter in reindex_by function."""

    def test_reindex_without_tolerance_matches_distant_points(self):
        """Without tolerance, nearest neighbor matches even distant points."""
        data = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.5, 1.5, 5.0])  # 5.0 is far from any point

        # Suppress expected warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = reindex_by(data, new_index, tolerance=None)

        assert len(result) == 3
        # All values should be present (5.0 matches to 2.0)
        assert not np.isnan(result.iloc[2]), "Without tolerance, distant point should match"

    def test_reindex_with_tolerance_rejects_distant_points(self):
        """With tolerance, distant points should become NaN."""
        data = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        # Use unambiguous distances: 0.1 from 0.0, 1.1 from 1.0, 5.0 is far from all
        new_index = pd.Index([0.1, 1.1, 5.0])

        result = reindex_by(data, new_index, tolerance=0.5)

        assert len(result) == 3
        # Close points should match (0.1 -> 0.0, 1.1 -> 1.0)
        assert result.iloc[0] == 1.0, "0.1 should match to 0.0 (distance 0.1)"
        assert result.iloc[1] == 2.0, "1.1 should match to 1.0 (distance 0.1)"
        # Distant point should be NaN
        assert np.isnan(result.iloc[2]), "5.0 should be NaN (distance 3.0 > tolerance 0.5)"

    def test_reindex_tolerance_exact_match(self):
        """Exact matches should always work regardless of tolerance."""
        data = pd.Series([10.0, 20.0, 30.0], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.0, 1.0, 2.0])  # Exact matches

        result = reindex_by(data, new_index, tolerance=0.001)

        np.testing.assert_array_equal(result.values, [10.0, 20.0, 30.0])

    def test_reindex_tolerance_with_dataframe(self):
        """Tolerance should work with DataFrames too."""
        data = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [10.0, 20.0, 30.0],
        }, index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.1, 0.9, 5.0])

        result = reindex_by(data, new_index, tolerance=0.2)

        assert result.shape == (3, 2)
        # Close points match
        assert result.iloc[0]['a'] == 1.0
        assert result.iloc[1]['a'] == 2.0
        # Distant point is NaN
        assert np.isnan(result.iloc[2]['a'])
        assert np.isnan(result.iloc[2]['b'])

    def test_reindex_tolerance_zero_rejects_all_non_exact(self):
        """Zero tolerance should only allow exact matches."""
        data = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.0001, 1.0, 2.0001])

        result = reindex_by(data, new_index, tolerance=0.0)

        assert np.isnan(result.iloc[0]), "0.0001 should be NaN (not exact match)"
        assert result.iloc[1] == 2.0, "1.0 should match exactly"
        assert np.isnan(result.iloc[2]), "2.0001 should be NaN (not exact match)"

    def test_reindex_preserves_column_order(self):
        """DataFrame columns should be preserved after reindexing."""
        data = pd.DataFrame({
            'col_z': [1.0, 2.0],
            'col_a': [3.0, 4.0],
            'col_m': [5.0, 6.0],
        }, index=[0.0, 1.0])
        new_index = pd.Index([0.5])

        result = reindex_by(data, new_index, tolerance=0.6)

        assert list(result.columns) == ['col_z', 'col_a', 'col_m']

    def test_reindex_empty_input(self):
        """Empty input should return empty output."""
        data = pd.Series([], dtype=float)
        new_index = pd.Index([0.0, 1.0])

        result = reindex_by(data, new_index, tolerance=0.1)

        assert len(result) == 2
        assert np.all(np.isnan(result))

    def test_reindex_typical_neuroscience_use_case(self):
        """Test typical use case: resampling spike counts to position time.

        Spike counts live at well-separated times; the position grid
        samples some bins close to a spike, others far from any
        spike. Tolerance ``0.5`` s. The fixture is chosen so all
        nearest-neighbor distances are unambiguously above or below
        tolerance — the close→match, far→NaN contract holds without
        depending on float roundoff at the tolerance boundary.

            position_time | nearest spike (dist) | value
            ------------- | --------------------- | -----
            0.0           | 0.0 (exact)           | 1.0
            1.0           | 1.2 (0.2 ≤ tol)       | 4.0
            2.0           | 2.3 (0.3 ≤ tol)       | 9.0
            3.0           | 2.3 (0.7 > tol)       | NaN
            4.0           | 4.4 (0.4 ≤ tol)       | 16.0
            5.0           | 4.4 (0.6 > tol)       | NaN
            6.0           | 6.1 (0.1 ≤ tol)       | 36.0

        Pinning the full expected vector catches:
        - A regression that drops the ``<=`` boundary (tolerance
          becomes strict ``<``): row at distance exactly ``tol``
          would flip to NaN, but no row here sits at the boundary,
          so the test stays robust to that detail.
        - A regression that loosens the comparator (``<=`` ->
          ``<= 1.5*tol``): distant rows (3.0, 5.0) would stop being
          NaN.
        - A regression that swaps tie-break direction in
          ``method='nearest'`` for asymmetric-distance bins: would
          shift values across rows.
        """
        spike_counts = pd.Series(
            [1.0, 4.0, 9.0, 16.0, 36.0],
            index=[0.0, 1.2, 2.3, 4.4, 6.1],
        )
        position_time = pd.Index(np.arange(0.0, 7.0, 1.0))

        result = reindex_by(spike_counts, position_time, tolerance=0.5)

        assert len(result) == len(position_time)
        expected = np.array(
            [1.0, 4.0, 9.0, np.nan, 16.0, np.nan, 36.0]
        )
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_reindex_without_tolerance_raises_warning(self):
        """Calling with tolerance=None should raise a UserWarning."""
        data = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.5, 1.5, 2.5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reindex_by(data, new_index, tolerance=None)

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "tolerance=None" in str(w[0].message)
            assert "arbitrarily distant" in str(w[0].message)

    def test_reindex_with_tolerance_no_warning(self):
        """Calling with explicit tolerance should not raise a warning."""
        data = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        new_index = pd.Index([0.5, 1.5, 2.5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reindex_by(data, new_index, tolerance=0.5)

            # Filter out any unrelated warnings
            tolerance_warnings = [
                warning for warning in w
                if "tolerance=None" in str(warning.message)
            ]
            assert len(tolerance_warnings) == 0
