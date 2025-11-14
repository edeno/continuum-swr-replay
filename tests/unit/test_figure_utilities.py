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

            save_figure("test_figure", base_dir=tmpdir)

            # Check both formats were created
            assert (Path(tmpdir) / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "test_figure.png").exists()

            plt.close(fig)

    def test_save_with_custom_formats(self):
        """Test saving figure with custom formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            save_figure("test_figure", base_dir=tmpdir, formats=["pdf", "svg"])

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
            save_figure("test_figure", base_dir=tmpdir, facecolor=None)

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_facecolor_string_accepted(self):
        """Test that string facecolor is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            save_figure("test_figure", base_dir=tmpdir, facecolor="#ffffff")

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_facecolor_tuple_accepted(self):
        """Test that tuple facecolor (RGBA) is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # RGBA tuple should be accepted
            save_figure("test_figure", base_dir=tmpdir, facecolor=(1.0, 1.0, 1.0, 1.0))

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

    def test_custom_dpi(self):
        """Test saving with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            save_figure("test_figure", base_dir=tmpdir, dpi=300)

            assert (Path(tmpdir) / "test_figure.png").exists()
            plt.close(fig)

    def test_dpi_is_always_int(self):
        """Test that DPI is always cast to int, even if rcParams returns float.

        This is the key test for FU-1.4: mpl.rcParams can return float values
        (e.g., 450.0), but the dpi parameter should be int for type safety.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set rcParams to a float value (common scenario)
            original_dpi = mpl.rcParams["savefig.dpi"]
            mpl.rcParams["savefig.dpi"] = 450.0  # Float value

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # This should work without type errors, even though rcParams has float
            save_figure("test_figure", base_dir=tmpdir)

            assert (Path(tmpdir) / "test_figure.pdf").exists()
            plt.close(fig)

            # Restore original
            mpl.rcParams["savefig.dpi"] = original_dpi

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError.

        This is the key test for FU-1.5: save_figure should validate
        format strings before attempting to save, preventing cryptic
        matplotlib errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Invalid format should raise ValueError
            with pytest.raises(ValueError, match="Invalid format"):
                save_figure("test_figure", base_dir=tmpdir, formats=["invalid"])

            with pytest.raises(ValueError, match="Invalid format"):
                save_figure("test_figure", base_dir=tmpdir, formats=["pdf", "bad_format"])

            plt.close(fig)

    def test_valid_formats_accepted(self):
        """Test that all valid matplotlib formats are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Common valid formats should work
            for fmt in ["pdf", "png", "svg", "eps", "jpg"]:
                save_figure("test_figure", base_dir=tmpdir, formats=[fmt])
                assert (Path(tmpdir) / f"test_figure.{fmt}").exists()
                (Path(tmpdir) / f"test_figure.{fmt}").unlink()

            plt.close(fig)

    def test_format_error_message_is_helpful(self):
        """Test that format validation error message lists valid formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Error message should mention valid formats
            with pytest.raises(ValueError) as exc_info:
                save_figure("test_figure", base_dir=tmpdir, formats=["xyz"])

            error_msg = str(exc_info.value)
            # Should mention the invalid format
            assert "xyz" in error_msg
            # Should list valid formats (at least some of them)
            assert "pdf" in error_msg.lower() or "png" in error_msg.lower()

            plt.close(fig)

    def test_base_dir_none_uses_figure_dir(self):
        """Test that base_dir=None saves to FIGURE_DIR by default.

        This is the key test for FU-2.1: save_figure should default to
        FIGURE_DIR when base_dir is not specified, making it easier to
        save figures without specifying the full path every time.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock FIGURE_DIR to point to our temp directory
            from continuum_swr_replay import figure_utilities
            original_figure_dir = getattr(figure_utilities, "FIGURE_DIR", None)

            # Set FIGURE_DIR to temp directory
            figure_utilities.FIGURE_DIR = Path(tmpdir)

            try:
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])

                # Save with base_dir=None (should use FIGURE_DIR)
                save_figure("test_figure", base_dir=None)

                # Check files were created in FIGURE_DIR
                assert (Path(tmpdir) / "test_figure.pdf").exists()
                assert (Path(tmpdir) / "test_figure.png").exists()

                plt.close(fig)
            finally:
                # Restore original FIGURE_DIR
                if original_figure_dir is not None:
                    figure_utilities.FIGURE_DIR = original_figure_dir

    def test_base_dir_string_accepted(self):
        """Test that base_dir accepts string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Save with string base_dir
            save_figure("test_figure", base_dir=tmpdir)

            # Check files were created in specified directory
            assert (Path(tmpdir) / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "test_figure.png").exists()

            plt.close(fig)

    def test_base_dir_path_accepted(self):
        """Test that base_dir accepts Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Save with Path base_dir
            save_figure("test_figure", base_dir=Path(tmpdir))

            # Check files were created in specified directory
            assert (Path(tmpdir) / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "test_figure.png").exists()

            plt.close(fig)

    def test_figure_name_with_subdirectory_and_base_dir(self):
        """Test that subdirectories in figure_name work with base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Save with subdirectory in figure_name
            save_figure("subfolder/test_figure", base_dir=tmpdir)

            # Check files were created in base_dir/subfolder
            assert (Path(tmpdir) / "subfolder" / "test_figure.pdf").exists()
            assert (Path(tmpdir) / "subfolder" / "test_figure.png").exists()

            plt.close(fig)

    def test_figure_name_with_subdirectory_no_base_dir(self):
        """Test that subdirectories in figure_name work when base_dir=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock FIGURE_DIR to point to our temp directory
            from continuum_swr_replay import figure_utilities
            original_figure_dir = getattr(figure_utilities, "FIGURE_DIR", None)

            # Set FIGURE_DIR to temp directory
            figure_utilities.FIGURE_DIR = Path(tmpdir)

            try:
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])

                # Save with subdirectory and base_dir=None
                save_figure("subfolder/test_figure", base_dir=None)

                # Check files were created in FIGURE_DIR/subfolder
                assert (Path(tmpdir) / "subfolder" / "test_figure.pdf").exists()
                assert (Path(tmpdir) / "subfolder" / "test_figure.png").exists()

                plt.close(fig)
            finally:
                # Restore original FIGURE_DIR
                if original_figure_dir is not None:
                    figure_utilities.FIGURE_DIR = original_figure_dir

    def test_empty_figure_raises_error(self):
        """Test that saving a figure with no axes raises ValueError.

        This is the key test for FU-2.2: save_figure should validate that
        the figure actually has content (axes) before attempting to save,
        preventing confusing empty figure files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a figure but don't add any axes
            fig = plt.figure()

            # Should raise ValueError for empty figure
            with pytest.raises(ValueError, match="no axes"):
                save_figure("empty_figure", base_dir=tmpdir)

            plt.close(fig)

    def test_figure_with_axes_saves_successfully(self):
        """Test that figures with axes save without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create figure with axes - should work fine
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            # Should not raise
            save_figure("valid_figure", base_dir=tmpdir)

            assert (Path(tmpdir) / "valid_figure.pdf").exists()
            plt.close(fig)

    def test_empty_figure_error_message_is_helpful(self):
        """Test that empty figure error message provides helpful guidance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plt.figure()

            with pytest.raises(ValueError) as exc_info:
                save_figure("empty_figure", base_dir=tmpdir)

            error_msg = str(exc_info.value).lower()
            # Should mention the problem and suggest a solution
            assert "no axes" in error_msg or "empty" in error_msg
            # Should suggest adding axes or subplots
            assert "axes" in error_msg or "subplot" in error_msg

            plt.close(fig)


class TestValidateFigureQuality:
    """Test figure quality validation functionality."""

    def test_validation_does_not_modify_figure_layout(self):
        """Test that validate_figure_quality() doesn't modify figure state.

        This is the key test for FU-2.3: validation functions should be
        read-only and not have side effects like calling tight_layout().
        """
        from continuum_swr_replay.figure_utilities import validate_figure_quality

        fig, axes = plt.subplots(2, 2, constrained_layout=True, dpi=450)
        for ax in axes.flatten():
            ax.plot([0, 1], [0, 1])

        # Get layout state before validation
        # We'll check that the figure's layout engine state doesn't change
        layout_engine_before = fig.get_layout_engine()

        # Call validation
        validate_figure_quality()

        # Layout engine should be unchanged
        layout_engine_after = fig.get_layout_engine()
        assert type(layout_engine_before) == type(layout_engine_after)

        plt.close(fig)

    def test_validation_returns_true_for_good_figure(self):
        """Test that validation returns True for properly configured figures."""
        from continuum_swr_replay.figure_utilities import validate_figure_quality

        fig, ax = plt.subplots(constrained_layout=True, dpi=450)
        ax.plot([0, 1], [0, 1])

        # Should pass validation
        result = validate_figure_quality()
        assert result is True

        plt.close(fig)

    def test_validation_warns_low_dpi(self):
        """Test that validation warns about low DPI."""
        from continuum_swr_replay.figure_utilities import validate_figure_quality

        fig, ax = plt.subplots(constrained_layout=True, dpi=72)
        ax.plot([0, 1], [0, 1])

        # Should warn about low DPI
        with pytest.warns(UserWarning, match="Low DPI"):
            result = validate_figure_quality()
            assert result is False

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
        """Test that mismatched label count raises error with helpful message.

        This is the key test for FU-2.4: label_panels should provide clear
        error messages showing both the number of labels provided and the
        number of axes, helping users quickly identify the mismatch.
        """
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(ValueError) as exc_info:
            label_panels(axes.flatten(), labels=["A", "B"])

        error_msg = str(exc_info.value)
        # Should mention both counts
        assert "3" in error_msg  # 3 axes
        assert "2" in error_msg  # 2 labels
        # Should use clear language
        assert "label" in error_msg.lower() or "axes" in error_msg.lower()

        plt.close(fig)
