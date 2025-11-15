"""
Utility for adding alphabetical panel labels to Matplotlib axes.
"""

import string
from typing import List, Optional, Sequence

import matplotlib.axes


def label_panels(
    axes: Sequence[matplotlib.axes.Axes],
    labels: Optional[List[str]] = None,
    *,
    xpos: float = -0.1,
    ypos: float = 1.05,
    fontsize: int = 8,
    fontweight: str = "bold",
) -> None:
    """Adds alphabetical panel labels (a, b, c...) to a list of subplots.

    The function places labels relative to each subplot's axes, ensuring they
    remain correctly positioned.

    Parameters
    ----------
    axes : Sequence[matplotlib.axes.Axes]
        A flat sequence of Matplotlib Axes objects to be labeled. If you have
        a 2D array of axes from `plt.subplots`, you must flatten it first
        (e.g., `axes.flatten()`).
    labels : Optional[List[str]], optional
        A list of strings to use as labels. If None, lowercase letters
        (a, b, c...) are used. By default None.
    xpos : float, optional
        The x-position of the label in axes-fractional coordinates,
        by default -0.1 (i.e., to the left of the y-axis).
    ypos : float, optional
        The y-position of the label in axes-fractional coordinates,
        by default 1.05 (i.e., above the plot area).
    fontsize : int, optional
        The font size of the panel labels, by default 8.
    fontweight : str, optional
        The font weight of the panel labels, by default "bold".
    """
    if labels is None:
        labels = list(string.ascii_lowercase)

    # Ensure there are enough labels for the axes provided
    if len(labels) < len(axes):
        raise ValueError("Not enough labels provided for the number of axes.")

    for ax, label in zip(axes, labels):
        ax.text(
            xpos,
            ypos,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            ha="right",  # Align to the right of the given xpos
            va="top",
        )
