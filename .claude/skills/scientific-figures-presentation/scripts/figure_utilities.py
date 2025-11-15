"""
Core utility functions for creating presentation figures.

Provides set_figure_defaults() and save_figure() functions that configure
matplotlib for presentation-quality output.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_figure_defaults(context="presentation"):
    """
    Set matplotlib defaults for presentation figures.
    
    Parameters
    ----------
    context : str, default="presentation"
        Context for figure style. Options:
        - "presentation": Light background, large fonts for talks
        - "presentation_dark": Dark background for dark rooms
    
    Examples
    --------
    >>> set_figure_defaults(context="presentation")
    >>> fig, ax = plt.subplots()
    >>> # Figure will use presentation defaults
    """
    if context == "presentation":
        style_path = os.path.join(SCRIPT_DIR, "presentation.mplstyle")
        if os.path.exists(style_path):
            mpl.style.use(style_path)
        else:
            # Fallback: set defaults programmatically
            _set_presentation_defaults()
    elif context == "presentation_dark":
        style_path = os.path.join(SCRIPT_DIR, "presentation_dark.mplstyle")
        if os.path.exists(style_path):
            mpl.style.use(style_path)
        else:
            # Fallback: set defaults programmatically
            _set_presentation_dark_defaults()
    else:
        raise ValueError(f"Unknown context '{context}'. "
                        "Use 'presentation' or 'presentation_dark'.")


def _set_presentation_defaults():
    """Set presentation defaults programmatically (fallback)."""
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 24,
        'axes.titlesize': 32,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'axes.linewidth': 2,
        'figure.dpi': 150,
    })


def _set_presentation_dark_defaults():
    """Set dark presentation defaults programmatically (fallback)."""
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'axes.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.labelsize': 24,
        'axes.titlesize': 32,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'axes.linewidth': 2,
        'figure.dpi': 150,
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
    })


def save_figure(figure_name, facecolor=None, transparent=False, dpi=300):
    """
    Save figure in multiple formats optimized for presentations.
    
    Parameters
    ----------
    figure_name : str
        Base name for figure file (without extension)
    facecolor : str or None, default=None
        Background color for saved figure. Use '#1a1a1a' for dark backgrounds.
    transparent : bool, default=False
        Whether to save with transparent background
    dpi : int, default=300
        Resolution for rasterized outputs
    
    Examples
    --------
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure('my_plot')  # Creates my_plot.png and my_plot.pdf
    
    >>> # For dark background
    >>> save_figure('dark_plot', facecolor='#1a1a1a')
    """
    # Save as PDF (vector)
    if facecolor is None:
        plt.savefig(
            f"{figure_name}.pdf",
            transparent=transparent,
            dpi=dpi,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{figure_name}.pdf",
            transparent=transparent,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=facecolor,
        )
    
    # Save as PNG (raster, best for slides)
    plt.savefig(
        f"{figure_name}.png",
        transparent=transparent,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=facecolor if facecolor else 'white',
    )
    
    print(f"Saved: {figure_name}.pdf and {figure_name}.png")
