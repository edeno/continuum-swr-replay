#!/usr/bin/env python
"""Generate Figure 3: what SWR and no-SWR non-local events are like.

The immobile non-local events of the canonical event table that start inside
a well visit (from the first poke-in at a well to the final poke-out; selected
once, where the table is read, by ``cohort_selection.select_visit_context``;
the content's time bins are read for those events), split by whether they
overlap an SWR (Kay's detector, the SWR source). The starting hypothesis
is that the two are different kinds of event, so every panel draws them
separately and none pools them; the legend gives each kind's number of events.
Each row asks one thing:

    Size and movement
        a — Duration.
        b — The mean share of an event in Non-Local Continuous and in Non-Local
            Fragmented (each event's mean state probability, each event weighted
            alike); bars the animals' mean, dots the animals
            (``event_statistics.state_composition_statistics``).
        c — Spatial extent of the decoded content.
        d — Replay speed (each event's median over its time bins), twice on
            one axis: the distribution, and under it the share of events at
            or below each speed.
    Where on the track the content is
        e — Where each kind's content is: its content time per cm along every
            arm, drawn on the track once per kind, against what content spread
            evenly over the track would give
            (``content_along_arms.log2_density_against_even``). The track
            names its center, a junction and a well, the ends of f's axis.
        f — Content time per cm from the center out to the junctions (trunk
            arms) and from the junctions out to the wells (leaf arms), against
            what content spread evenly over the track would give (the dashed
            line; ``content_along_arms.even_content``).

The figure describes the content from the decode and the track alone. What needs
the animal's behavior to say is left to the behavior figure: content by distance
from the animal and by which well it is at
(``content_along_arms.distance_from_animal_shares``, once a panel here), by the
animal's choices and rewards, and when the events happen. The state the events
occur in is not drawn here either: ripple power, MUA, theta and head speed
around the events are Figure 2's, and slow and fast gamma during them are a
supplementary figure (``generate_supp_figure03.py``).

b comes before c and d because the far mode of c and the fastest events of d
(its end bin) are the events with Fragmented time. d's axis is linear: a decoded
position that moves one place bin in one time bin has a speed of 1,000 cm/s,
which the smoothing spreads over the bins around it, so the values an event's
median takes below about 100 cm/s are the smoother's, and a log axis would give
them two fifths of its length. The number of arms the decoded position touches is
not drawn: it restates c in arms, and an arm counts when one time bin's position
is on it. Its values stay in the events summary.

Panels a, c and d mark each class's median (triangle;
``event_statistics.class_medians``), the events summary's
``<property>_median_*`` values (``paper/reported/events.json``), and under it
each animal's median (ticks; ``event_statistics.animal_class_medians``), the
medians the summary's ``<property>_animals_in_direction`` counts compare. Panels
e and f place every event time bin's MAP position on its arm
(``measure_content_along_arms.py``; ``Processed-Data/content_along_arms``) and
draw ``content_along_arms.content_density_along_arms``, the animals' mean; f
quotes the share of content time within ``AT_WELL_CM`` of a well, the summary's
``*_content_at_well_fraction`` (``content_along_arms.content_place_shares``,
``place_share_statistics``). Each weighs each epoch of an animal the same and
each animal the same.

Usage
-----
    python scripts/figures/generate_figure03.py
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    content_density_along_arms,
    content_place_shares,
    even_content,
    leaf_and_trunk_profiles,
    log2_density_against_even,
    place_share_statistics,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    AT_WELL_CM,
    animal_class_medians,
    class_medians,
    state_composition_statistics,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    processed_data_dir,
)
from continuum_swr_replay.reported.events import content_along_arms_bins
from continuum_swr_replay.reported.inputs import default_inputs
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.palettes import EVENT_KINDS
from continuum_swr_replay.visualization.panel_layout import place_letters
from continuum_swr_replay.visualization.publication_style import (
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    set_publication_style,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

# Inches: saved with 0.1 in margins, about 17.3 cm (Cell Press full width).
WIDTH = 6.5
HEIGHT = 4.3


# The track drawn in panel e: one epoch's track graph (every epoch's arms run the
# same way, ``content_along_arms.check_arm_orientation``).
TRACK_EPOCH = "wilbur20210329_02_r1_HPC"
# Half the width of an arm drawn in panel e, track units (cm).
ARM_HALF_WIDTH_CM = 9.0
# Panels e and f: segments per arm, about 4 cm (two 2-cm place bins) each.
N_SEGMENTS = 13
# Panel e: the color scale's limit, log2 of a kind's density over even content's,
# and its tick labels: a segment beyond an end is drawn in the end's color.
DENSITY_LIMIT_LOG2 = 2
DENSITY_TICK_LABELS = ("≤¼×", "even", "≥4×")
# Panel e: the color of content sparser than even content, for either kind.
C_BELOW_EVEN = "#6e6e6e"

# Panels a, c, d: (column, x label, scale, bin edges, x ticks). A tick at an
# end of the axis that starts with ≤ or ≥ says its bin also holds the events
# beyond the axis; an end without one holds none (``draw_histogram`` checks).
HISTOGRAMS = {
    "duration": (
        "event_duration_s", "Duration (ms)", 1e3,
        np.arange(0, 401, 8), {0: "0", 200: "200", 400: "≥400"},
    ),
    "extent": (
        "spatial_extent_cm", "Spatial extent (cm)", 1.0,
        np.linspace(0, 240, 41), {0: "0", 100: "100", 200: "200"},
    ),
    "speed": (
        "replay_speed_median_cm_s", "Replay speed (cm/s)", 1.0,
        np.arange(0, 1501, 100),
        {0: "0", 500: "500", 1000: "1000", 1500: "≥1500"},
    ),
}  # fmt: skip


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs() -> dict:
    """The events, their content's time bins, and the other inputs."""
    inputs = default_inputs()
    events = select_visit_context(read_swr_table(inputs.event_table))
    n_epochs = len(analyzed_epoch_ids())
    along, _ = content_along_arms_bins(
        events,
        inputs.content_along_arms,
        n_epochs,
        [
            "arm",
            "fraction",
            "arm_length_cm",
            "place_bin",
            "n_place_bins",
            "distance_to_well_cm",
            "distance_to_animal_cm",
        ],
    )
    density = content_density_along_arms(along, N_SEGMENTS)
    even = even_content(along, AT_WELL_CM)
    env = pd.read_pickle(decode_paths(TRACK_EPOCH, processed_data_dir()).environment)
    return {
        "events": events,
        "place_shares": place_share_statistics(content_place_shares(along))["means"],
        "profiles": leaf_and_trunk_profiles(density),
        "even": even,
        "against_even": log2_density_against_even(density, even["density_per_cm"]),
        "track_graph": env.track_graph,
        "edge_order": [tuple(edge) for edge in env.edge_order],
    }


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def kind_labels(events: pd.DataFrame) -> list[str]:
    """Each kind's name with its number of events, for the figure's legend."""
    return [
        f"{name} (n = {int((events['is_swr'] == is_swr).sum()):,} events)"
        for is_swr, (name, _) in EVENT_KINDS.items()
    ]


def draw_histogram(ax, events: pd.DataFrame, key: str) -> None:
    """A property's distribution for each kind, % of the kind's events per bin,
    with each kind's median (triangle) and, under it, each animal's (ticks).

    The events beyond an end of the axis are counted in its end bin, whose tick
    then says so (≤ or ≥). An event beyond an end that is not marked raises, and
    so does an event without a value: the legend's n is every event.
    """
    column, label, scale, edges, ticks = HISTOGRAMS[key]
    if events[column].isna().any():
        raise ValueError(f"{label}: an event has no value")
    medians = class_medians(events, column)
    animals = animal_class_medians(events, column)
    marked_below = ticks.get(edges[0], "").startswith("≤")
    marked_above = ticks.get(edges[-1], "").startswith("≥")
    top = 0.0
    for is_swr, (_, color) in EVENT_KINDS.items():
        values = scale * events.loc[events["is_swr"] == is_swr, column].to_numpy()
        below, above = (values < edges[0]).any(), (values > edges[-1]).any()
        if (below and not marked_below) or (above and not marked_above):
            raise ValueError(f"{label}: an event lies beyond the axis, unmarked")
        values = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        share = 100 * counts / counts.sum()
        ax.stairs(share, edges, color=color, fill=True, alpha=0.25, lw=0)
        ax.stairs(share, edges, color=color, lw=0.8)
        top = max(top, share.max())
    for (_, color), word in zip(EVENT_KINDS.values(), ("non_swr", "swr"), strict=True):
        ax.plot(
            scale * medians[f"median_{word}"], top * 1.19, marker="v", ms=3.5,
            color=color, lw=0,
        )  # fmt: skip
        ax.vlines(scale * animals[word], top * 1.04, top * 1.12, color=color, lw=0.6)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, top * 1.26)
    ax.set_xlabel(label)
    ax.set_xticks(list(ticks), list(ticks.values()))


def draw_cumulative(ax, events: pd.DataFrame, key: str) -> None:
    """The share of each kind's events at or below each value, on the
    histogram's axis. A curve crosses the 50 % line at the kind's median, and
    what it lacks of 100 % at the end of the axis is the share of events beyond
    it (the histogram's end bin)."""
    column, label, scale, edges, ticks = HISTOGRAMS[key]
    if events[column].isna().any():
        raise ValueError(f"{label}: an event has no value")
    grid = np.linspace(edges[0], edges[-1], 301)
    ax.axhline(50, color="0.8", lw=0.5)
    for is_swr, (_, color) in EVENT_KINDS.items():
        values = scale * events.loc[events["is_swr"] == is_swr, column].to_numpy()
        at_or_below = np.searchsorted(np.sort(values), grid, side="right")
        ax.plot(grid, 100 * at_or_below / len(values), color=color, lw=0.8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(label)
    ax.set_xticks(list(ticks), list(ticks.values()))


def _keep_side(
    polygon: np.ndarray, point: np.ndarray, toward: np.ndarray
) -> np.ndarray:
    """The part of a convex polygon on the side of the line through ``point``
    that ``toward`` (the line's normal) points to."""
    side = (polygon - point) @ toward
    kept = []
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if side[i] >= 0:
            kept.append(polygon[i])
        if (side[i] >= 0) != (side[j] >= 0):
            crossing = side[i] / (side[i] - side[j])
            kept.append(polygon[i] + crossing * (polygon[j] - polygon[i]))
    return np.asarray(kept).reshape(-1, 2)


def track_bands(
    track_graph, edge_order, values: pd.Series
) -> tuple[list[np.ndarray], np.ndarray]:
    """Each arm segment as a band ``ARM_HALF_WIDTH_CM`` to each side of its arm,
    and its value.

    Where arms meet, a band ends on the lines halfway between its arm and each
    neighbor, so the bands of a junction fit together: none lies over another
    and no segment is hidden.

    Parameters
    ----------
    track_graph : networkx.Graph
        Nodes with ``pos`` (cm), edges with ``edge_id`` (the arm).
    edge_order : sequence of (node, node)
        The arms, each from its first node (segment 0) to its second.
    values : pd.Series
        One value per segment, indexed by ``arm`` and ``segment``.

    Returns
    -------
    polygons : list of NDArray, each shape (n_corners, 2)
        One per arm and segment, in the order of ``edge_order``.
    shown : NDArray, shape (n_arms * N_SEGMENTS,)
        Each polygon's value.
    """
    position = {
        node: np.asarray(track_graph.nodes[node]["pos"], dtype=float)
        for node in track_graph
    }
    polygons, shown = [], []
    for first, second in edge_order:
        arm = track_graph.edges[first, second]["edge_id"]
        along = position[second] - position[first]
        unit = along / np.hypot(*along)
        normal = np.array([-unit[1], unit[0]]) * ARM_HALF_WIDTH_CM
        # At each end: nearer this arm than each other arm that leaves the node.
        cuts = []
        for node, outward in ((first, unit), (second, -unit)):
            for other in track_graph.neighbors(node):
                if other not in (first, second):
                    toward = position[other] - position[node]
                    cuts.append((position[node], outward - toward / np.hypot(*toward)))
        for segment in range(N_SEGMENTS):
            a = position[first] + along * segment / N_SEGMENTS
            b = position[first] + along * (segment + 1) / N_SEGMENTS
            polygon = np.array([a + normal, b + normal, b - normal, a - normal])
            for point, toward in cuts:
                polygon = _keep_side(polygon, point, toward)
            if len(polygon) < 3:
                raise ValueError(
                    f"arm {arm}'s segment {segment} has no room to be drawn"
                )
            polygons.append(polygon)
            shown.append(values[(arm, segment)])
    return polygons, np.asarray(shown, dtype=float)


def draw_density_tracks(axes, track_graph, edge_order, against_even: pd.Series) -> None:
    """The track once per kind (``axes``: no SWR, then SWR), each segment colored
    by log2 of the kind's content density over even content's
    (``content_along_arms.log2_density_against_even``): gray where sparser than
    even, the kind's color where denser. The scale saturates at
    ``DENSITY_LIMIT_LOG2``, and its end labels say so."""
    limit = DENSITY_LIMIT_LOG2
    for ax, (is_swr, (name, color)) in zip(axes, EVENT_KINDS.items(), strict=True):
        polygons, values = track_bands(
            track_graph, edge_order, against_even.loc[is_swr]
        )
        # Filled bands, not thick lines: the journal caps line width.
        bands = PolyCollection(
            polygons,
            array=np.clip(values, -limit, limit),
            cmap=LinearSegmentedColormap.from_list(
                name, [C_BELOW_EVEN, "#f4f4f4", color]
            ),
            norm=TwoSlopeNorm(0.0, -limit, limit),
            linewidths=0,
        )
        ax.add_collection(bands)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=SMALL_SIZE, color=color, pad=2)
        bar = plt.colorbar(
            bands, ax=ax, orientation="horizontal", shrink=0.6, aspect=14, pad=0.03,
            ticks=[-limit, 0, limit],
        )  # fmt: skip
        bar.ax.set_xticklabels(DENSITY_TICK_LABELS)
        bar.ax.tick_params(length=2)
        bar.outline.set_linewidth(0.5)


def label_track(ax, track_graph) -> None:
    """Name the track's places on a drawn track: its center, a junction and a
    well, the ends of panel f's axis.

    No name lies over a band and none needs a line. The well's name is beside
    the end of its arm. The junction's and the center's sit in the notch where
    two arms part: the junction's with a corner at the notch, the center's
    under it, as far down as its width needs to clear the two arms. The
    junction named is the topmost and the well its right one. The figure is
    drawn once here, to measure the center's name.
    """
    position = {
        node: np.asarray(track_graph.nodes[node]["pos"], dtype=float)
        for node in track_graph
    }
    is_well = {node: track_graph.degree(node) == 1 for node in track_graph}
    junctions = [
        node
        for node in track_graph
        if not is_well[node] and any(is_well[n] for n in track_graph.neighbors(node))
    ]
    (center,) = [n for n in track_graph if not is_well[n] and n not in junctions]
    junction = max(junctions, key=lambda node: position[node][1])
    well = max(
        (n for n in track_graph.neighbors(junction) if is_well[n]),
        key=lambda node: position[node][0],
    )
    half = ARM_HALF_WIDTH_CM
    gap = 2.0  # cm between a name and the nearest band

    def notch(node, first, second):
        """Where the bands of two arms that leave ``node`` part, the unit vector
        from the node into the opening between them, and half its angle."""
        toward = [
            (position[other] - position[node])
            / np.hypot(*(position[other] - position[node]))
            for other in (first, second)
        ]
        opening = toward[0] + toward[1]
        sine = abs(toward[0][0] * toward[1][1] - toward[0][1] * toward[1][0])
        half_angle = 0.5 * np.arccos(np.clip(toward[0] @ toward[1], -1, 1))
        return (
            position[node] + opening * half / sine,
            opening / np.hypot(*opening),
            half_angle,
        )

    style = {"color": "0.3", "fontsize": SMALL_SIZE}
    ax.annotate(
        "well", position[well], position[well] + [half + gap, 0], ha="left",
        va="center", **style,
    )  # fmt: skip
    # The junction: a corner of its name at the notch between its trunk arm and
    # the named well's arm, the name running into the opening.
    tip, into, _ = notch(junction, center, well)
    ax.annotate(
        "junction", position[junction], tip + gap * into,
        ha="left" if into[0] > 0 else "right",
        va="top" if into[1] < 0 else "bottom", **style,
    )  # fmt: skip
    # The center: under the notch between the two arms that leave it downward.
    lower = sorted(track_graph.neighbors(center), key=lambda n: position[n][1])[:2]
    tip, into, half_angle = notch(center, *lower)
    name = ax.annotate(
        "center", position[center], tip, ha="center",
        va="top" if into[1] < 0 else "bottom", **style,
    )  # fmt: skip
    ax.figure.draw_without_rendering()
    extent = name.get_window_extent().transformed(ax.transData.inverted())
    name.set_position(
        tip
        + (0.5 * extent.width / np.tan(half_angle) + gap / np.sin(half_angle)) * into
    )


def draw_profiles(ax, profiles: pd.DataFrame, shares: pd.DataFrame, even: dict) -> None:
    """Content time per cm along the trunk arms (center to junction) and the
    leaf arms (junction to well), each kind, with its share at a well; and what
    content spread evenly over the track would give (``even``:
    ``content_along_arms.even_content``), a flat line and its share at a well."""
    x = (np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS
    even_percent = 100 * even["density_per_cm"]
    ax.axhline(even_percent, color="0.5", lw=0.6, ls=(0, (4, 2)), zorder=0, gid="even")
    ax.annotate(
        "even", (0.03, even_percent), xycoords=ax.get_yaxis_transform(),
        xytext=(0, -1.5), textcoords="offset points", ha="left", va="top",
        color="0.4", fontsize=SMALL_SIZE,
    )  # fmt: skip
    ax.text(
        0.03, 0.79, f"even: {100 * even['at_well_fraction']:.0f}% at a well",
        transform=ax.transAxes, color="0.4", ha="left", va="top",
        fontsize=SMALL_SIZE,
    )  # fmt: skip
    for is_swr, (name, color) in EVENT_KINDS.items():
        trunk = profiles.loc[(is_swr, "trunk"), "density_per_cm"].to_numpy()
        leaf = profiles.loc[(is_swr, "leaf"), "density_per_cm"].to_numpy()
        ax.plot(x, 100 * trunk, color=color)
        ax.plot(1 + x, 100 * leaf, color=color)
        ax.text(
            0.03, 0.97 - 0.09 * is_swr,
            f"{name}: {100 * shares.loc[is_swr, 'at_well']:.0f}% at a well",
            transform=ax.transAxes, color=color, ha="left", va="top",
            fontsize=SMALL_SIZE,
        )  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1], ["0.1", "0.2", "0.5", "1"])
    ax.axvline(1, color="0.8", lw=0.5, zorder=0)
    ax.set_xticks([0, 1, 2], ["center", "junction", "well"])
    ax.set_xlim(0, 2)
    for x_mid, name in ((0.5, "trunk arms"), (1.5, "leaf arms")):
        ax.text(
            x_mid, -0.17, name, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=SMALL_SIZE, color="0.35",
        )  # fmt: skip
    ax.set_ylabel("% content time per cm")
    ax.set_title(
        f"at a well: <{AT_WELL_CM:.0f} cm from one", fontsize=SMALL_SIZE, pad=8
    )


def draw_state_composition(ax, statistics: pd.DataFrame) -> None:
    """Panel b: the mean share of an event in Continuous and in Fragmented.

    Bars are the animals' mean, dots the animals
    (``event_statistics.state_composition_statistics``).
    """
    for g, (is_swr, (_, color)) in enumerate(EVENT_KINDS.items()):
        row = statistics.loc["SWR" if is_swr else "no SWR"]  # the statistic's names
        for x, state in enumerate(("continuous", "fragmented")):
            position = x + (g - 0.5) * 0.36
            ax.bar(position, 100 * row[f"{state}_animal_mean"], width=0.34, color=color)
            animals = 100 * row[f"{state}_animals"]
            ax.scatter(
                position + np.linspace(-0.08, 0.08, len(animals)), animals, s=3,
                color="k", lw=0, zorder=3,
            )  # fmt: skip
    ax.set_xticks([0, 1], ["Continuous", "Fragmented"], rotation=30, ha="right")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("% of event\n(mean over events)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# The panel letters as they sit in the figure: the letters of a row share a
# height, the letters of a column a left edge.
LETTER_ROWS = ("abcd", "ef")
LETTER_COLUMNS = ("ae", "b", "cf", "d")


def generate_figure(inputs: dict):
    """Build Figure 3 from :func:`read_inputs`."""
    events = inputs["events"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # One grid for both rows, so the panels of a column share a left edge. The
    # widths are the upper row's needs: four bars in b, two stacked axes in d.
    grid = GridSpec(
        2, 4, figure=fig, height_ratios=[1.0, 1.1], hspace=0.7,
        width_ratios=[1.1, 0.72, 1.0, 1.3], wspace=0.62,
        left=0.08, right=0.99, top=0.91, bottom=0.13,
    )  # fmt: skip

    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, i]) for i in range(3))
    d_grid = grid[0, 3].subgridspec(2, 1, hspace=0.25)
    ax_d = fig.add_subplot(d_grid[0])
    ax_d_cumulative = fig.add_subplot(d_grid[1], sharex=ax_d)
    draw_state_composition(ax_b, state_composition_statistics(events))
    for ax, key in ((ax_a, "duration"), (ax_c, "extent"), (ax_d, "speed")):
        draw_histogram(ax, events, key)
    draw_cumulative(ax_d_cumulative, events, "speed")
    ax_d.set_xlabel("")
    ax_d.tick_params(labelbottom=False)
    for ax in (ax_a, ax_c, ax_d):
        ax.set_ylabel("% events")
    ax_d_cumulative.set_ylabel("% events\nat or below")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in EVENT_KINDS.values()]
    fig.legend(
        handles, kind_labels(events), frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8, columnspacing=1.5,
    )  # fmt: skip

    # The lower row's two panels are each two columns wide. Panel e is the track
    # once per kind under one title, at the height of f's.
    ax_e_frame = fig.add_subplot(grid[1, :2])
    ax_e_frame.axis("off")
    ax_e_frame.set_title(
        "Content time per cm, against even", fontsize=SMALL_SIZE, pad=8
    )
    e_grid = grid[1, :2].subgridspec(1, 2, wspace=0.1)
    ax_e = [fig.add_subplot(e_grid[0]), fig.add_subplot(e_grid[1])]
    draw_density_tracks(
        ax_e, inputs["track_graph"], inputs["edge_order"], inputs["against_even"]
    )
    label_track(ax_e[0], inputs["track_graph"])
    ax_f = fig.add_subplot(grid[1, 2:])
    draw_profiles(ax_f, inputs["profiles"], inputs["place_shares"], inputs["even"])

    fig.draw_without_rendering()
    panels_by_letter = {
        "a": [ax_a], "b": [ax_b], "c": [ax_c], "d": [ax_d, ax_d_cumulative],
        "e": [ax_e_frame, *ax_e], "f": [ax_f],
    }  # fmt: skip
    place_letters(fig, panels_by_letter, LETTER_ROWS, LETTER_COLUMNS)
    check_placement(fig, panels_by_letter)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure03")
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure03", base_dir=args.output_dir)
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
