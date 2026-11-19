"""The project's color palettes.

- :data:`COLORS` — one color per event type and decoder state (SWR, MUA,
  Non-Local, Local, No-Spike, ...), used by every figure.
- :data:`CONTFRAG_COLORS` — the Continuous/Fragmented classifier's two states.

Two arm palettes live here too:

- :func:`build_arm_color_map` — the legacy palette of
  ``scripts/plot_replay_content_time_slice.py`` (removed in 82b66d0). Samples
  ``matplotlib.colormaps['tab10']`` in the order the track graph was
  linearized. Categorical, not patch-aware.
- :func:`patch_grouped_arm_palette` — the palette ``plot_session_tables``
  uses. Arms are grouped by patch (Blues / Greens / Reds) with
  intra-patch lightness variation so patch membership is visually
  obvious. Trunk arms (the darkest shade of their patch) sit past the
  two leaf arms.

Both return ``dict[int, tuple[float, float, float, float]]`` mapping
``arm_id`` (edge_id, 0–8) to an RGBA tuple.
"""

from __future__ import annotations

from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from numpy.typing import NDArray

from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    LEAF_ARMS,
    TRACK_SEGMENT_TO_PATCH,
)

_LEAF_ARMS = set(LEAF_ARMS)  # arms touching a well: 0-5

COLORS: dict[str, str] = {
    "SWR": "#d95f02",
    # A non-local event partly in an SWR: a sand between SWR and Non-Local.
    "SWR partial": "#b5825a",
    "MUA": "#7570b3",
    "Non-Local": "#1b9e77",
    "Local": "#add8e6",  # Soft Blue for Local
    "No-Spike": "#b0b0b0",  # Light Gray for No-Spike
    "All Events": "#d9d9d9",  # Light Gray for background/all-events reference
    # Joint SWR + non-SWR-non-local model bar in Figure 4's AIC
    # comparison. Same Dark2 family as SWR/MUA/Non-Local.
    "Combined": "#e6ab02",
}
"""Color scheme for plotting the event types and decoder states."""

LIGHT_COLOR_LIGHTNESS = 0.6
"""A color lighter than this (HLS lightness) is too pale for text on white, and
too pale under white text."""

TEXT_LIGHTNESS = 0.4
"""The lightness a pale color is darkened to when text is written in it."""


def readable_text_color(color: str) -> str:
    """The color to write text in ``color`` on white.

    A pale color (``COLORS["Local"]``'s light blue, ``COLORS["No-Spike"]``'s
    gray) is darkened to :data:`TEXT_LIGHTNESS`, keeping its hue, so the text can
    be read and still names its state by color; any other color is returned as
    given.
    """
    hue, lightness, saturation = rgb_to_hls(*to_rgb(color))
    if lightness <= LIGHT_COLOR_LIGHTNESS:
        return color
    return to_hex(hls_to_rgb(hue, TEXT_LIGHTNESS, saturation))


def text_color_on(fill: str) -> str:
    """Black on a pale fill, white on any other: the color of text drawn over ``fill``."""
    return "black" if rgb_to_hls(*to_rgb(fill))[1] > LIGHT_COLOR_LIGHTNESS else "white"


CONTFRAG_COLORS: dict[str, str] = {
    "Continuous": "#1b9e77",  # Teal green (smooth trajectory)
    "Fragmented": "#d95f02",  # Orange (disjointed representation)
}
"""Colors of the ContFrag classifier's Continuous and Fragmented states."""


def build_arm_color_map(
    track_graph: nx.Graph,
    edge_order: NDArray[np.int_],
) -> dict[int, tuple[float, float, float, float]]:
    """Legacy tab10 palette keyed by arm ID, ordered by linearized edges.

    Direct port of the helper previously inlined in
    ``scripts/plot_replay_content_time_slice.py`` (removed in 82b66d0).
    Samples ``tab10`` in the order ``edge_order`` iterates the track graph's edges;
    keeps the visual consistency between arm markers and the 1D track
    overlay in that script.

    Parameters
    ----------
    track_graph : networkx.Graph
        Graph with ``edge_id`` attributes on edges.
    edge_order : array-like of (u, v) node pairs
        Canonical edge order used by ``plot_graph_as_1D``.

    Returns
    -------
    dict[int, tuple]
        ``{arm_id: rgba}`` for every arm in :data:`ALL_ARMS`. Arms that
        ``edge_order`` doesn't visit (can happen on minimal fixtures)
        fall back to ``tab10[arm % 10]``.
    """
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = getattr(cmap, "colors", None)
    if colors is not None:
        edge_colors = np.array(colors)
    else:
        edge_colors = np.array([cmap(i / 10) for i in range(10)])

    arm_color_map: dict[int, tuple[float, float, float, float]] = {}
    for edge_ind, edge in enumerate(edge_order):
        eid = track_graph.edges[tuple(edge)].get("edge_id")
        if eid is not None:
            arm_color_map[int(eid)] = tuple(edge_colors[edge_ind % len(edge_colors)])

    for arm in ALL_ARMS:
        if arm not in arm_color_map:
            arm_color_map[arm] = tuple(edge_colors[arm % len(edge_colors)])
    return arm_color_map


def patch_grouped_arm_palette(
    *,
    patch_cmaps: tuple[str, str, str] = ("Blues", "Greens", "Reds"),
    leaf_levels: tuple[float, float] = (0.45, 0.65),
    trunk_level: float = 0.85,
) -> dict[int, tuple[float, float, float, float]]:
    """Patch-grouped palette where arm color encodes patch + role.

    Patch 1 (arms 0, 1, 6) → ``Blues``;
    Patch 2 (arms 2, 3, 7) → ``Greens``;
    Patch 3 (arms 4, 5, 8) → ``Reds``.

    Within a patch, the two leaf arms use lighter shades
    (``leaf_levels``) and the trunk arm uses the darkest shade
    (``trunk_level``). A reader can tell (a) which patch an arm
    belongs to from the hue and (b) whether it is a leaf or trunk
    from the saturation, at a glance.

    Parameters
    ----------
    patch_cmaps : tuple[str, str, str]
        Three matplotlib colormap names, one per patch (1, 2, 3).
    leaf_levels : tuple[float, float]
        Two sampling positions in [0, 1] for the two leaf arms of
        each patch.
    trunk_level : float
        Sampling position in [0, 1] for the trunk arm of each patch.

    Returns
    -------
    dict[int, tuple]
        ``{arm_id: rgba}`` for every arm in :data:`ALL_ARMS`.

    Notes
    -----
    Leaf arm order within a patch follows ``sorted(arms_in_patch)``:
    arm 0 before arm 1 in patch 1, arm 2 before arm 3 in patch 2,
    arm 4 before arm 5 in patch 3. The trunk arms (6, 7, 8) are fixed
    by :data:`TRACK_SEGMENT_TO_PATCH`.
    """
    palette: dict[int, tuple[float, float, float, float]] = {}
    # Group arms by patch.
    arms_by_patch: dict[int, list[int]] = {}
    for arm_id, patch_id in TRACK_SEGMENT_TO_PATCH.items():
        arms_by_patch.setdefault(patch_id, []).append(arm_id)

    for patch_id, cmap_name in enumerate(patch_cmaps, start=1):
        cmap = plt.get_cmap(cmap_name)
        arms = sorted(arms_by_patch.get(patch_id, []))
        leaf_arms = sorted(a for a in arms if a in _LEAF_ARMS)
        trunk_arms = sorted(a for a in arms if a not in _LEAF_ARMS)

        for arm, level in zip(leaf_arms[: len(leaf_levels)], leaf_levels, strict=False):
            palette[arm] = tuple(cmap(level))
        for arm in trunk_arms:
            palette[arm] = tuple(cmap(trunk_level))

    # Final safety net: any arm not covered (degenerate fixture
    # without full patch coverage) falls back to a gray so it's
    # visibly different from patch colors.
    for arm in ALL_ARMS:
        palette.setdefault(arm, (0.5, 0.5, 0.5, 1.0))
    return palette
