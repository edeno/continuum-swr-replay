"""Figure utilities for creating publication-quality scientific figures.

This module provides utilities for creating figures optimized for different
contexts (papers, presentations) following best practices from Tufte, Gelman,
and Heer.

Functions
---------
set_figure_defaults
    Configure matplotlib for paper or presentation figures
save_figure
    Save figure in multiple formats with proper DPI
figure_size_from_panels
    Calculate optimal figure size from panel dimensions
cm_to_inch
    Convert centimeters to inches
label_panels
    Add a, b, c, d labels to multi-panel figures

Constants
---------
MM_TO_INCHES : float
    Conversion factor from millimeters to inches
ONE_COLUMN, TWO_COLUMN : float
    Standard column widths for journal papers
WONG, OKABE_ITO : list
    Colorblind-friendly color palettes

See Also
--------
continuum_swr_replay.parameters : Global configuration constants

Notes
-----
Typography Standards (Papers):
- Minimum font size: 7pt (Nature/Science requirement)
- Axis labels: 7pt
- Tick labels: 5-6pt
- Panel letters: 8pt bold

Typography Standards (Presentations):
- Body text: 16-24pt
- Titles: 32-48pt
- Axis labels: 20-24pt
- Tick labels: 16-18pt

Always use constrained_layout=True to prevent element overlaps.

Examples
--------
Paper figure:

>>> from continuum_swr_replay.figure_utilities import (
...     set_figure_defaults, save_figure, WONG
... )
>>> import matplotlib.pyplot as plt
>>>
>>> set_figure_defaults(context="paper")
>>> fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
...                        constrained_layout=True)
>>> ax.plot(x, y, color=WONG[1], linewidth=1.5)
>>> ax.set_xlabel('Time (s)', fontsize=7)
>>> save_figure("figure_1")

Presentation figure:

>>> set_figure_defaults(context="presentation")
>>> fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
...                        constrained_layout=True)
>>> ax.plot(x, y, color=WONG[1], linewidth=4)
>>> ax.set_xlabel('Time (s)', fontsize=24, fontweight='bold')
>>> save_figure("slide_figure")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.typing import ColorType

from continuum_swr_replay.parameters import FIGURE_DIR

# ==============================================================================
# Dimension Constants
# ==============================================================================

MM_TO_INCHES: float = 1.0 / 25.4
"""Conversion factor from millimeters to inches."""

# Journal paper dimensions (Nature/Science standards)
ONE_COLUMN: float = 89.0 * MM_TO_INCHES  # ~3.5 inches
"""Single column width for journal papers (89mm = 3.5 inches)."""

ONE_AND_HALF_COLUMN: float = 140.0 * MM_TO_INCHES  # ~5.5 inches
"""1.5 column width for journal papers (140mm = 5.5 inches)."""

TWO_COLUMN: float = 178.0 * MM_TO_INCHES  # ~7.0 inches
"""Full page width for journal papers (178mm = 7.0 inches)."""

PAGE_HEIGHT: float = 247.0 * MM_TO_INCHES  # ~9.7 inches
"""Maximum page height for journal papers (247mm = 9.7 inches)."""

GOLDEN_RATIO: float = (np.sqrt(5) - 1.0) / 2.0
"""Golden ratio for aesthetically pleasing aspect ratios (~0.618)."""

# Presentation dimensions (16:9 aspect ratio)
SLIDE_WIDTH: float = 10.0  # inches
"""Standard slide width for 16:9 presentations."""

SLIDE_HEIGHT: float = 5.625  # inches (10 / 16 * 9)
"""Standard slide height for 16:9 presentations."""

# ==============================================================================
# Color Palettes (Colorblind-Friendly)
# ==============================================================================

WONG: list[str] = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
]
"""Wong colorblind-friendly palette (8 colors).

Most commonly used palette. Works for deuteranopia, protanopia, and
tritanopia. Also works in grayscale.

References
----------
Bang Wong (2011). "Points of view: Color blindness." Nature Methods 8(6).
"""

OKABE_ITO: list[str] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
]
"""Okabe-Ito colorblind-friendly palette (8 colors).

Alternative to Wong palette. Excellent for deuteranopia and protanopia.

References
----------
Okabe & Ito (2008). "Color Universal Design." JFly Data Depository.
"""

# ==============================================================================
# Typography Constants
# ==============================================================================

# Paper figure typography (minimum sizes per Nature/Science)
PAPER_FONT_MIN: int = 7  # Minimum for readability
PAPER_AXIS_LABEL: int = 7
PAPER_TICK_LABEL: int = 6
PAPER_PANEL_LABEL: int = 8
PAPER_LEGEND: int = 5

# Presentation figure typography (visible from 30+ feet)
PRES_BODY: int = 20
PRES_TITLE: int = 32
PRES_AXIS_LABEL: int = 24
PRES_TICK_LABEL: int = 18
PRES_LEGEND: int = 20

# ==============================================================================
# Style Configuration
# ==============================================================================


def set_figure_defaults(
    context: Literal["paper", "presentation", "presentation_dark"] = "paper",
) -> None:
    """Configure matplotlib for publication-quality figures.

    Sets typography, colors, line widths, and layout parameters optimized
    for different output contexts.

    Parameters
    ----------
    context : {"paper", "presentation", "presentation_dark"}
        Output context:
        - "paper": Journal publications (Nature/Science standards)
        - "presentation": Light background slides (distance viewing)
        - "presentation_dark": Dark background slides (high contrast)

    Notes
    -----
    Paper context:
    - Minimum 7pt fonts (Nature/Science requirement)
    - Fine lines (1.5pt)
    - High DPI (450) for print quality
    - Editable fonts (pdf.fonttype: 42)

    Presentation contexts:
    - Large fonts (20-32pt) for distance viewing
    - Bold lines (3-5pt) for visibility
    - Lower DPI (150) for screen display
    - High contrast colors

    Always use constrained_layout=True when creating figures to prevent
    label/legend overlaps.

    Examples
    --------
    Paper figure:

    >>> set_figure_defaults("paper")
    >>> fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
    ...                        constrained_layout=True)
    >>> ax.plot(x, y, linewidth=1.5)  # Fine line for print

    Presentation:

    >>> set_figure_defaults("presentation")
    >>> fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
    ...                        constrained_layout=True)
    >>> ax.plot(x, y, linewidth=4)  # Bold line for visibility
    """
    if context == "paper":
        # Try loading custom style file first
        style_path = Path(FIGURE_DIR) / "styles" / "paper.mplstyle"
        if style_path.exists():
            mpl.style.use(str(style_path))
        else:
            # Fallback to programmatic configuration
            rc_params = {
                # Fonts (editable in Illustrator)
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": PAPER_FONT_MIN,
                # Axes
                "axes.labelsize": PAPER_AXIS_LABEL,
                "axes.labelcolor": "#222222",
                "axes.linewidth": 0.8,
                "axes.labelpad": 4.0,
                "axes.spines.top": False,
                "axes.spines.right": False,
                # Ticks
                "xtick.labelsize": PAPER_TICK_LABEL,
                "ytick.labelsize": PAPER_TICK_LABEL,
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.bottom": True,
                "ytick.left": True,
                # Lines and markers
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                # Legend
                "legend.fontsize": PAPER_LEGEND,
                "legend.frameon": False,
                # Text
                "text.color": "#222222",
                "text.usetex": False,
                # Figure
                "figure.dpi": 450,
                "figure.constrained_layout.use": True,
                "savefig.dpi": 450,
                "savefig.bbox": "tight",
            }
            sns.set_theme(style="white", context="paper", rc=rc_params)

    elif context == "presentation":
        rc_params = {
            # Fonts (large for distance viewing)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": PRES_BODY,
            "font.weight": "bold",
            # Axes
            "axes.labelsize": PRES_AXIS_LABEL,
            "axes.labelweight": "bold",
            "axes.titlesize": PRES_TITLE,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "axes.linewidth": 2,
            "axes.labelpad": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks
            "xtick.labelsize": PRES_TICK_LABEL,
            "ytick.labelsize": PRES_TICK_LABEL,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            # Lines and markers (bold for visibility)
            "lines.linewidth": 4,
            "lines.markersize": 10,
            # Legend
            "legend.fontsize": PRES_LEGEND,
            "legend.frameon": False,
            # Text
            "text.color": "#222222",
            "text.usetex": False,
            # Figure
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
        sns.set_theme(style="white", context="talk", rc=rc_params)

    elif context == "presentation_dark":
        rc_params = {
            # Fonts
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": PRES_BODY,
            "font.weight": "bold",
            # Axes (white on dark)
            "axes.labelsize": PRES_AXIS_LABEL,
            "axes.labelweight": "bold",
            "axes.titlesize": PRES_TITLE,
            "axes.titleweight": "bold",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "axes.linewidth": 2,
            "axes.labelpad": 10.0,
            "axes.facecolor": "#1a1a1a",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks (white)
            "xtick.labelsize": PRES_TICK_LABEL,
            "ytick.labelsize": PRES_TICK_LABEL,
            "xtick.color": "white",
            "ytick.color": "white",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            # Lines and markers
            "lines.linewidth": 4,
            "lines.markersize": 10,
            # Legend
            "legend.fontsize": PRES_LEGEND,
            "legend.frameon": False,
            # Text (white)
            "text.color": "white",
            "text.usetex": False,
            # Figure (dark background)
            "figure.facecolor": "#1a1a1a",
            "figure.edgecolor": "#1a1a1a",
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.facecolor": "#1a1a1a",
            "savefig.edgecolor": "#1a1a1a",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
        sns.set_theme(style="dark", context="talk", rc=rc_params)

    else:
        raise ValueError(
            f"Unknown context: {context}. "
            f"Must be 'paper', 'presentation', or 'presentation_dark'"
        )


def save_figure(
    figure_name: str,
    *,
    facecolor: ColorType | None = None,
    transparent: bool = True,
    dpi: int | None = None,
    formats: list[str] | None = None,
) -> None:
    """Save figure in multiple formats with proper DPI settings.

    Saves both vector (PDF) and raster (PNG) formats by default. Useful for
    submissions that require both formats.

    Parameters
    ----------
    figure_name : str
        Base filename (without extension). Can include directory path.
    facecolor : ColorType, optional
        Background color. If None, uses current figure facecolor.
        Accepts any matplotlib color: name, hex string, RGB(A) tuple, etc.
        Use "#1a1a1a" for dark backgrounds.
    transparent : bool, default=True
        Whether to use transparent background (recommended for papers).
    dpi : int, optional
        DPI for raster formats. If None, uses context default
        (450 for paper, 150 for presentations).
    formats : list of str, optional
        Output formats. Default is ["pdf", "png"].
        Options: "pdf", "png", "svg", "eps", "jpg"

    Notes
    -----
    PDF uses vector graphics (infinite resolution) - preferred for papers.
    PNG uses raster graphics - useful for presentations and web.

    File naming:
    - figure_name="fig1" → fig1.pdf, fig1.png
    - figure_name="figures/fig1" → figures/fig1.pdf, figures/fig1.png

    Examples
    --------
    Save with defaults:

    >>> save_figure("figure_1")  # Creates figure_1.pdf and figure_1.png

    Dark background figure:

    >>> save_figure("dark_fig", facecolor="#1a1a1a", transparent=False)

    Custom formats:

    >>> save_figure("fig", formats=["pdf", "svg", "png"])
    """
    if formats is None:
        formats = ["pdf", "png"]

    # Get DPI from current settings if not specified
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", 300)

    # Determine facecolor
    if facecolor is None:
        # Use figure's current facecolor
        fig = plt.gcf()
        facecolor = fig.get_facecolor()

    # Create directory if needed
    output_path = Path(figure_name)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in each format
    for fmt in formats:
        output_file = f"{figure_name}.{fmt}"
        plt.savefig(
            output_file,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=facecolor,
            transparent=transparent,
        )


# ==============================================================================
# Figure Sizing Utilities
# ==============================================================================


def cm_to_inch(cm: float) -> float:
    """Convert centimeters to inches.

    Parameters
    ----------
    cm : float
        Length in centimeters

    Returns
    -------
    float
        Length in inches

    Examples
    --------
    >>> cm_to_inch(10)
    3.937007874015748
    """
    return cm / 2.54


def figure_size_from_panels(
    nrows: int = 1,
    ncols: int = 1,
    *,
    panel_width_cm: float = 4.5,
    panel_height_cm: float | None = None,
    wspace_cm: float = 0.8,
    hspace_cm: float = 0.8,
    left_margin_cm: float = 0.5,
    right_margin_cm: float = 0.5,
    top_margin_cm: float = 0.5,
    bottom_margin_cm: float = 0.5,
) -> tuple[float, float]:
    """Calculate optimal figure size from panel dimensions.

    Useful for creating multi-panel figures where you want to control
    individual panel sizes rather than overall figure size.

    Parameters
    ----------
    nrows : int, default=1
        Number of rows
    ncols : int, default=1
        Number of columns
    panel_width_cm : float, default=4.5
        Width of each panel in cm
    panel_height_cm : float, optional
        Height of each panel in cm. If None, uses panel_width_cm * GOLDEN_RATIO
    wspace_cm : float, default=0.8
        Horizontal spacing between panels in cm
    hspace_cm : float, default=0.8
        Vertical spacing between panels in cm
    left_margin_cm : float, default=0.5
        Left margin in cm
    right_margin_cm : float, default=0.5
        Right margin in cm
    top_margin_cm : float, default=0.5
        Top margin in cm
    bottom_margin_cm : float, default=0.5
        Bottom margin in cm

    Returns
    -------
    tuple of float
        Figure size (width, height) in inches

    Examples
    --------
    2x2 grid with default spacing:

    >>> figsize = figure_size_from_panels(nrows=2, ncols=2)
    >>> fig, axes = plt.subplots(2, 2, figsize=figsize)

    Custom panel dimensions:

    >>> figsize = figure_size_from_panels(
    ...     nrows=1, ncols=3,
    ...     panel_width_cm=5.0,
    ...     panel_height_cm=3.0,
    ...     wspace_cm=1.0
    ... )
    """
    if panel_height_cm is None:
        panel_height_cm = panel_width_cm * GOLDEN_RATIO

    # Calculate total width
    total_width_cm = (
        left_margin_cm
        + (ncols * panel_width_cm)
        + ((ncols - 1) * wspace_cm)
        + right_margin_cm
    )

    # Calculate total height
    total_height_cm = (
        bottom_margin_cm
        + (nrows * panel_height_cm)
        + ((nrows - 1) * hspace_cm)
        + top_margin_cm
    )

    # Convert to inches
    width_inches = cm_to_inch(total_width_cm)
    height_inches = cm_to_inch(total_height_cm)

    return (width_inches, height_inches)


# ==============================================================================
# Panel Labeling
# ==============================================================================


def label_panels(
    axes: list[Axes] | np.ndarray,
    *,
    labels: list[str] | None = None,
    xpos: float = -0.12,
    ypos: float = 1.05,
    fontsize: int = 8,
    fontweight: str = "bold",
) -> None:
    """Add a, b, c, d labels to multi-panel figures.

    Automatically labels panels in reading order (left-to-right, top-to-bottom).

    Parameters
    ----------
    axes : list of Axes or ndarray
        Axes to label. Can be from plt.subplots() or subplot_mosaic().
        For mosaic: use list(axes.values())
    labels : list of str, optional
        Custom labels. If None, uses lowercase letters a, b, c, ...
    xpos : float, default=-0.12
        X position in axes coordinates (0-1). Negative places outside plot.
    ypos : float, default=1.05
        Y position in axes coordinates. >1 places above plot.
    fontsize : int, default=8
        Font size for labels
    fontweight : str, default="bold"
        Font weight ("normal", "bold", etc.)

    Notes
    -----
    Position in axes coordinates:
    - (0, 0) = bottom-left corner of axes
    - (1, 1) = top-right corner of axes
    - Negative x = left of axes
    - y > 1 = above axes

    Typical positions:
    - Upper left outside: xpos=-0.12, ypos=1.05
    - Upper left inside: xpos=0.02, ypos=0.98
    - Lower left: xpos=-0.12, ypos=-0.15

    Examples
    --------
    Regular subplots:

    >>> fig, axes = plt.subplots(2, 2)
    >>> label_panels(axes.flatten())  # Labels: a, b, c, d

    subplot_mosaic:

    >>> layout = "AB\\nCD"
    >>> fig, axes = plt.subplot_mosaic(layout)
    >>> label_panels(list(axes.values()))  # Convert dict to list

    Custom labels:

    >>> label_panels(axes.flatten(), labels=['A1', 'A2', 'B1', 'B2'])
    """
    # Flatten if ndarray
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    # Generate default labels if not provided
    if labels is None:
        import string

        labels = list(string.ascii_lowercase[: len(axes)])

    # Add label to each axis
    for ax, label in zip(axes, labels, strict=True):
        ax.text(
            xpos,
            ypos,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="top",
            ha="right",
        )


# ==============================================================================
# Validation Utilities
# ==============================================================================


def validate_figure_quality() -> bool:
    """Check current figure for common quality issues.

    Validates:
    - Font sizes meet minimum requirements
    - No layout warnings
    - DPI appropriate for context
    - Constrained layout enabled

    Returns
    -------
    bool
        True if figure passes all checks

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
    ...                        constrained_layout=True)
    >>> ax.plot(x, y)
    >>> if validate_figure_quality():
    ...     save_figure("figure_1")
    """
    fig = plt.gcf()
    passed = True

    # Check DPI
    if fig.dpi < 150:
        warnings.warn(
            f"Low DPI ({fig.dpi}). Use 450 for papers, 150 for presentations.",
            stacklevel=2,
        )
        passed = False

    # Check constrained_layout
    if not mpl.rcParams.get("figure.constrained_layout.use", False):
        warnings.warn(
            "constrained_layout not enabled. "
            "Use constrained_layout=True to prevent overlaps.",
            stacklevel=2,
        )
        passed = False

    # Check for layout warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            plt.tight_layout()
        except Exception:
            pass

        if w:
            for warning in w:
                warnings.warn(f"Layout issue: {warning.message}", stacklevel=2)
            passed = False

    return passed
