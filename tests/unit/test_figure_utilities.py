"""Tests for figure_utilities module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from continuum_swr_replay.figure_utilities import (
    GOLDEN_RATIO,
    MM_TO_INCHES,
    OKABE_ITO,
    ONE_COLUMN,
    TWO_COLUMN,
    WONG,
    cm_to_inch,
    figure_size_from_panels,
    label_panels,
    save_figure,
    set_figure_defaults,
)


class TestConstants:
    """Test module constants."""

    def test_mm_to_inches(self):
        """Test millimeter to inches conversion factor."""
        assert MM_TO_INCHES == pytest.approx(1.0 / 25.4)

    def test_golden_ratio(self):
        """Test golden ratio constant."""
        expected = (np.sqrt(5) - 1.0) / 2.0
        assert GOLDEN_RATIO == pytest.approx(expected)

    def test_one_column_width(self):
        """Test single column width matches standard."""
        assert ONE_COLUMN == pytest.approx(89.0 * MM_TO_INCHES)
        assert ONE_COLUMN == pytest.approx(3.5, abs=0.01)

    def test_two_column_width(self):
        """Test two column width matches standard."""
        assert TWO_COLUMN == pytest.approx(178.0 * MM_TO_INCHES)
        assert TWO_COLUMN == pytest.approx(7.0, abs=0.01)


class TestColorPalettes:
    """Test color palette definitions."""

    def test_wong_palette_length(self):
        """Test Wong palette has 8 colors."""
        assert len(WONG) == 8

    def test_okabe_ito_palette_length(self):
        """Test Okabe-Ito palette has 8 colors."""
        assert len(OKABE_ITO) == 8

    def test_wong_colors_are_hex(self):
        """Test Wong palette colors are valid hex strings."""
        for color in WONG:
            assert color.startswith("#")
            assert len(color) == 7
            int(color[1:], 16)  # Should not raise

    def test_okabe_ito_colors_are_hex(self):
        """Test Okabe-Ito palette colors are valid hex strings."""
        for color in OKABE_ITO:
            assert color.startswith("#")
            assert len(color) == 7
            int(color[1:], 16)  # Should not raise


class TestCmToInch:
    """Test centimeter to inch conversion."""

    def test_conversion_accuracy(self):
        """Test cm to inch conversion is accurate."""
        assert cm_to_inch(2.54) == pytest.approx(1.0)
        assert cm_to_inch(10.0) == pytest.approx(3.937, abs=0.001)
        assert cm_to_inch(0.0) == 0.0

    def test_negative_values(self):
        """Test conversion with negative values."""
        assert cm_to_inch(-2.54) == pytest.approx(-1.0)


class TestFigureSizeFromPanels:
    """Test figure size calculation from panel dimensions."""

    def test_single_panel_default(self):
        """Test single panel with default settings."""
        width, height = figure_size_from_panels(nrows=1, ncols=1)
        assert width > 0
        assert height > 0
        # Height should be based on golden ratio (with margins added)
        # The panel dimensions use golden ratio, but margins affect final ratio
        assert height < width  # Should still be landscape-ish

    def test_multiple_panels(self):
        """Test multiple panels increase figure size."""
        w1, h1 = figure_size_from_panels(nrows=1, ncols=1)
        w2, h2 = figure_size_from_panels(nrows=1, ncols=2)
        w3, h3 = figure_size_from_panels(nrows=2, ncols=1)

        assert w2 > w1  # More columns = wider
        assert h3 > h1  # More rows = taller

    def test_custom_panel_size(self):
        """Test custom panel dimensions."""
        width, height = figure_size_from_panels(
            nrows=1, ncols=1, panel_width_cm=10.0, panel_height_cm=5.0
        )
        # Check approximate sizes (margins add to total)
        assert width == pytest.approx(cm_to_inch(11.0), abs=0.1)  # 10 + margins
        assert height == pytest.approx(cm_to_inch(6.0), abs=0.1)  # 5 + margins


class TestSetFigureDefaults:
    """Test figure default configuration."""

    def test_paper_context(self):
        """Test paper context sets appropriate defaults."""
        set_figure_defaults(context="paper")
        assert mpl.rcParams["figure.dpi"] == 450
        assert mpl.rcParams["savefig.dpi"] == 450

    def test_presentation_context(self):
        """Test presentation context sets appropriate defaults."""
        set_figure_defaults(context="presentation")
        assert mpl.rcParams["figure.dpi"] == 150
        assert mpl.rcParams["savefig.dpi"] == 150
        # Fonts should be larger
        assert mpl.rcParams["font.size"] >= 16

    def test_presentation_dark_context(self):
        """Test dark presentation context."""
        set_figure_defaults(context="presentation_dark")
        assert mpl.rcParams["figure.facecolor"] == "#1a1a1a"
        assert mpl.rcParams["axes.labelcolor"] == "white"

    def test_invalid_context_raises(self):
        """Test invalid context raises ValueError."""
        with pytest.raises(ValueError, match="Unknown context"):
            set_figure_defaults(context="invalid")


class TestSaveFigure:
    """Test figure saving functionality."""

    def test_save_with_default_formats(self):
        """Test saving figure with default formats (PDF and PNG)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = Path(tmpdir) / "test_figure"
            save_figure(str(output_path))

            # Check both formats were created
            assert (Path(tmpdir) / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "test_figure.png").exists()

            plt.close(fig)

    def test_save_with_custom_formats(self):
        """Test saving figure with custom formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = Path(tmpdir) / "test_figure"
            save_figure(str(output_path), formats=["pdf", "svg"])

            # Check only requested formats were created
            assert (Path(tmpdir) / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "test_figure.svg").exists()
            assert not (Path(tmpdir) / "test_figure.png").exists()

            plt.close(fig)

    def test_facecolor_none_uses_figure_facecolor(self):
        """Test that facecolor=None uses the figure's current facecolor.

        This is the key test for FU-1.1: fig.get_facecolor() returns a tuple,
        so the type hint must accept tuple[float, ...], not just str.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Set a specific facecolor
            fig.set_facecolor("#1a1a1a")

            # This should work without type errors
            # fig.get_facecolor() returns a tuple of floats (RGBA)
            output_path = Path(tmpdir) / "test_figure"
            save_figure(str(output_path), facecolor=None)

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_facecolor_string_accepted(self):
        """Test that string facecolor is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = Path(tmpdir) / "test_figure"
            save_figure(str(output_path), facecolor="#ffffff")

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_facecolor_tuple_accepted(self):
        """Test that tuple facecolor (RGBA) is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = Path(tmpdir) / "test_figure"
            # RGBA tuple should be accepted
            save_figure(str(output_path), facecolor=(1.0, 1.0, 1.0, 1.0))

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_custom_dpi(self):
        """Test saving with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = Path(tmpdir) / "test_figure"
            save_figure(str(output_path), dpi=300)

            assert (Path(tmpdir) / "test_figure.png").exists()
            plt.close(fig)


class TestLabelPanels:
    """Test panel labeling functionality."""

    def test_label_single_panel(self):
        """Test labeling a single panel."""
        fig, ax = plt.subplots()
        label_panels([ax])
        # Check that text was added (can't easily verify content in test)
        plt.close(fig)

    def test_label_multiple_panels(self):
        """Test labeling multiple panels."""
        fig, axes = plt.subplots(2, 2)
        label_panels(axes.flatten())
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom labels."""
        fig, axes = plt.subplots(1, 2)
        label_panels(axes.flatten(), labels=["A", "B"])
        plt.close(fig)

    def test_ndarray_axes_flattened(self):
        """Test that ndarray axes are properly flattened."""
        fig, axes = plt.subplots(2, 2)
        # Should work with 2D array
        label_panels(axes)
        plt.close(fig)

    def test_label_count_mismatch_raises(self):
        """Test that mismatched label count raises error."""
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(ValueError):
            # strict=True in zip should catch length mismatch
            label_panels(axes.flatten(), labels=["A", "B"])
        plt.close(fig)
