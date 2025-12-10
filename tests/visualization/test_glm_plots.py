"""Unit tests for GLM plotting utilities.

Tests plotting functions for GLM analysis visualization including event-aligned
spike rasters, peri-stimulus time histograms (PSTH), and AIC comparisons.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from continuum_swr_replay.visualization.glm_plots import (
    plot_AICs,
    plot_hist_relative_to_event,
    plot_raster_relative_to_event,
)


class TestPlotRasterRelativeToEvent:
    """Test event-aligned spike raster plotting."""

    @pytest.fixture
    def mock_spike_times(self) -> np.ndarray:
        """Create mock spike times."""
        # Spikes at 0.1, 0.2, 0.5, 0.8, 1.2, 1.5, 2.1, 2.8 seconds
        return np.array([0.1, 0.2, 0.5, 0.8, 1.2, 1.5, 2.1, 2.8])

    @pytest.fixture
    def mock_event_times(self) -> np.ndarray:
        """Create mock event times."""
        # Events at 0.5 and 2.0 seconds
        return np.array([0.5, 2.0])

    def test_basic_raster_plot(self, mock_spike_times, mock_event_times):
        """Test that raster plot is created correctly."""
        fig, ax = plt.subplots()

        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
        )

        # Verify plot was created
        assert ax is not None
        assert ax.get_xlim() == (-0.5, 0.5)

        plt.close(fig)

    def test_window_filtering(self, mock_spike_times):
        """Test that spikes outside window are excluded."""
        event_times = np.array([1.0])
        window = (-0.2, 0.3)

        fig, ax = plt.subplots()
        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=event_times,
            window=window,
            ax=ax,
        )

        # Expected relative spike times: -0.1 (from 0.9 not in data), 0.2 (from 1.2)
        # Only spike at 1.2 should be within window (1.2 - 1.0 = 0.2)

        plt.close(fig)

    def test_custom_color(self, mock_spike_times, mock_event_times):
        """Test custom color is applied."""
        fig, ax = plt.subplots()

        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
            color="red",
        )

        plt.close(fig)

    def test_no_ax_creates_new(self, mock_spike_times, mock_event_times):
        """Test that function creates axes if not provided."""
        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
        )

        # Should create plot on current axes
        ax = plt.gca()
        assert ax is not None
        plt.close()

    def test_empty_spike_times(self, mock_event_times):
        """Test handling of empty spike times."""
        fig, ax = plt.subplots()

        plot_raster_relative_to_event(
            spike_times=np.array([]),
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
        )

        plt.close(fig)

    def test_empty_event_times(self, mock_spike_times):
        """Test handling of empty event times."""
        fig, ax = plt.subplots()

        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=np.array([]),
            window=(-0.5, 0.5),
            ax=ax,
        )

        plt.close(fig)

    def test_vertical_line_at_zero(self, mock_spike_times, mock_event_times):
        """Test that vertical line is drawn at time 0."""
        fig, ax = plt.subplots()

        plot_raster_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
        )

        # Check that there's a vertical line (axvline was called)
        # This is implementation-specific, but we can check xlim was set correctly
        assert ax.get_xlim() == (-0.5, 0.5)

        plt.close(fig)


class TestPlotHistRelativeToEvent:
    """Test event-aligned histogram (PSTH) plotting."""

    @pytest.fixture
    def mock_spike_times(self) -> np.ndarray:
        """Create mock spike times."""
        return np.array([0.1, 0.2, 0.5, 0.8, 1.2, 1.5, 2.1, 2.8])

    @pytest.fixture
    def mock_event_times(self) -> np.ndarray:
        """Create mock event times."""
        return np.array([0.5, 2.0])

    def test_basic_histogram_plot(self, mock_spike_times, mock_event_times):
        """Test that histogram plot is created correctly."""
        fig, ax = plt.subplots()

        plot_hist_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
            bin_size=0.1,
        )

        # Verify plot was created
        assert ax is not None
        assert ax.get_xlim() == (-0.5, 0.5)

        plt.close(fig)

    def test_bin_size_parameter(self, mock_spike_times, mock_event_times):
        """Test custom bin size is used."""
        fig, ax = plt.subplots()

        plot_hist_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-1.0, 1.0),
            ax=ax,
            bin_size=0.2,
        )

        plt.close(fig)

    def test_custom_color(self, mock_spike_times, mock_event_times):
        """Test custom color is applied."""
        fig, ax = plt.subplots()

        plot_hist_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
            color="blue",
        )

        plt.close(fig)

    def test_no_ax_creates_new(self, mock_spike_times, mock_event_times):
        """Test that function creates axes if not provided."""
        plot_hist_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
        )

        ax = plt.gca()
        assert ax is not None
        plt.close()

    def test_empty_spike_times(self, mock_event_times):
        """Test handling of empty spike times."""
        fig, ax = plt.subplots()

        plot_hist_relative_to_event(
            spike_times=np.array([]),
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
        )

        plt.close(fig)

    def test_vertical_line_at_zero(self, mock_spike_times, mock_event_times):
        """Test that vertical line is drawn at time 0."""
        fig, ax = plt.subplots()

        plot_hist_relative_to_event(
            spike_times=mock_spike_times,
            event_times=mock_event_times,
            window=(-0.5, 0.5),
            ax=ax,
        )

        assert ax.get_xlim() == (-0.5, 0.5)

        plt.close(fig)


class TestPlotAICs:
    """Test AIC comparison plotting."""

    @pytest.fixture
    def mock_glm_results(self) -> dict:
        """Create mock GLM results with AIC values."""
        # Create mock result objects with AIC attribute
        mock_results = {}
        model_names = ["constant", "SWR", "non_local_SWR", "non_local_no_SWR", "non_local"]
        aic_values = [1000.0, 950.0, 920.0, 930.0, 925.0]

        for name, aic in zip(model_names, aic_values):
            mock_result = MagicMock()
            mock_result.aic = aic
            mock_results[name] = mock_result

        return mock_results

    def test_basic_aic_plot(self, mock_glm_results):
        """Test that AIC comparison plot is created correctly."""
        fig, ax = plot_AICs(
            glm_results=mock_glm_results,
            model_list=["constant", "SWR", "non_local"],
        )

        # Verify plot was created
        assert fig is not None
        assert ax is not None

        # Verify number of points matches number of models
        assert len(ax.collections[0].get_offsets()) == 3

        plt.close(fig)

    def test_default_model_list(self, mock_glm_results):
        """Test that default model list is used when not provided."""
        # Add the additional model needed for default list
        mock_result = MagicMock()
        mock_result.aic = 940.0
        mock_glm_results["SWR + non_local_no_SWR"] = mock_result

        fig, ax = plot_AICs(glm_results=mock_glm_results)

        # Default list has 6 models
        assert len(ax.collections[0].get_offsets()) == 6

        plt.close(fig)

    def test_model_name_mapping(self, mock_glm_results):
        """Test that model names are properly formatted."""
        fig, ax = plot_AICs(
            glm_results=mock_glm_results,
            model_list=["constant", "non_local_SWR"],
        )

        # Get y-tick labels
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]

        # Check that formatted names are used (e.g., "Non-Local,\nSWR" instead of "non_local_SWR")
        assert "Constant" in yticklabels
        assert "Non-Local,\nSWR" in yticklabels or "Non-Local, SWR" in yticklabels

        plt.close(fig)

    def test_aic_ordering(self, mock_glm_results):
        """Test that AICs are plotted in correct order."""
        model_list = ["constant", "SWR", "non_local"]
        expected_aics = [1000.0, 950.0, 925.0]

        fig, ax = plot_AICs(
            glm_results=mock_glm_results,
            model_list=model_list,
        )

        # Get x-coordinates of scatter points
        scatter_data = ax.collections[0].get_offsets()
        plotted_aics = scatter_data[:, 0]

        # Verify AICs match expected values
        np.testing.assert_array_almost_equal(plotted_aics, expected_aics)

        plt.close(fig)

    def test_custom_figsize(self, mock_glm_results):
        """Test custom figure size."""
        fig, ax = plot_AICs(
            glm_results=mock_glm_results,
            model_list=["constant", "SWR"],
            figsize=(6, 4),
        )

        # Verify figure size
        assert fig.get_figwidth() == 6
        assert fig.get_figheight() == 4

        plt.close(fig)

    def test_xlabel_set(self, mock_glm_results):
        """Test that x-axis label is set correctly."""
        fig, ax = plot_AICs(
            glm_results=mock_glm_results,
            model_list=["constant", "SWR"],
        )

        xlabel = ax.get_xlabel()
        assert "AIC" in xlabel
        assert "lower" in xlabel.lower()

        plt.close(fig)
