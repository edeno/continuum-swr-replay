#!/usr/bin/env python
"""Generate Figure 3: what SWR and no-SWR non-local events are like.

The immobile non-local events of the canonical event table, split by whether
they overlap an SWR (Kay's detector, the SWR source). The starting hypothesis
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
    Where the content is, and the brain state
        e — Content time by track distance from the animal, at a well (within
            ``AT_WELL_CM`` of one) and elsewhere: content at a well far from
            the animal is at another well, near it at the animal's own (the
            two labels; the summary's ``*_content_remote_well_fraction`` and
            ``*_content_own_well_fraction`` divide at ``FAR_CM``).
        f — Where on the track each kind's content is denser: the ratio of no-SWR
            to SWR content time per cm along every arm.
        g — Content time per cm from the center out to the junctions (trunk
            arms) and from the junctions out to the wells (leaf arms).
        h — Slow and fast gamma during each kind of event, between the levels
            of the same epochs' quiet-rest and running windows as long as the
            kind's events. (Ripple power, MUA, theta and head speed around the
            events are Figure 2's.)

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
e–g place every event time bin's MAP position on its arm
(``measure_content_along_arms.py``; ``Processed-Data/content_along_arms``): e
draws ``content_along_arms.distance_from_animal_shares``, f and g
``content_along_arms.content_density_along_arms``, each the animals' mean; g
quotes the share of content time within ``AT_WELL_CM`` of a well, the summary's
``*_content_at_well_fraction`` (``content_along_arms.content_place_shares``,
``place_share_statistics``). Every one of them weighs each epoch of an animal the
same and each animal the same. Panel h's animal values are
``event_statistics.animal_signal_levels`` with ``matched``: the values the
summary's ``<signal>_animals_*`` counts compare. An event's gamma is a mean
over its bins, and a mean of skewed power over a window sits above the median
of single bins, so the reference is windows of the events' own length.

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
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    content_density_along_arms,
    content_place_shares,
    distance_from_animal_shares,
    leaf_and_trunk_profiles,
    log2_density_ratio,
    place_share_statistics,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    AT_WELL_CM,
    FAR_CM,
    animal_class_medians,
    animal_signal_levels,
    class_medians,
    state_composition_statistics,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    processed_data_dir,
)
from continuum_swr_replay.reported.events import (
    content_along_arms_bins,
    default_inputs,
    with_brain_state,
)
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.simulated_run_panels import (
    SMALL_SIZE,
    check_journal_limits,
    drawn_extent,
    letter_panel,
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
HEIGHT = 4.6

C_NO_SWR = COLORS["No SWR"]
C_SWR = COLORS["SWR"]
# Panel h's rest and running levels: neutral, not decoder-state colors.
C_REST = "white"
C_RUN = "black"
# The two kinds of event, named as in Figure 2.
KINDS = ((False, "No SWR", C_NO_SWR), (True, "SWR", C_SWR))

# The track drawn in panel f: one epoch's track graph (every epoch's arms run the
# same way, ``content_along_arms.check_arm_orientation``).
TRACK_EPOCH = "wilbur20210329_02_r1_HPC"
# Half the width of an arm drawn in panel f, track units (cm).
ARM_HALF_WIDTH_CM = 9.0
# Panels f and g: segments per arm, about 4 cm (two 2-cm place bins) each.
N_SEGMENTS = 13
# Panel f: the color scale's limit, log2 of the no-SWR to SWR density ratio.
RATIO_LIMIT_LOG2 = 2
# Panel e: distance from the animal, cm (the track's longest is under 240).
DISTANCE_EDGES_CM = np.arange(0, 241, 8)

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

# Panel h: (signal, event column, label). Theta is Figure 2's.
SIGNALS = (
    ("slow_gamma", "slow_gamma_z", "Slow γ"),
    ("fast_gamma", "fast_gamma_z", "Fast γ"),
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs() -> dict:
    """The events with their gamma, their content's time bins, and the other inputs."""
    inputs = default_inputs()
    events = read_swr_table(inputs.event_table)
    n_epochs = len(analyzed_epoch_ids())
    events, reference, _ = with_brain_state(events, inputs, n_epochs)
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
    env = pd.read_pickle(decode_paths(TRACK_EPOCH, processed_data_dir()).environment)
    return {
        "events": events,
        "reference": reference,
        "place_shares": place_share_statistics(content_place_shares(along))["means"],
        "ratio": log2_density_ratio(density),
        "distance_shares": distance_from_animal_shares(
            along, DISTANCE_EDGES_CM, AT_WELL_CM
        ),
        "profiles": leaf_and_trunk_profiles(density),
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
        for is_swr, name, _ in KINDS
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
    for is_swr, _, color in KINDS:
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
    for (_, _, color), word in zip(KINDS, ("non_swr", "swr"), strict=True):
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
    for is_swr, _, color in KINDS:
        values = scale * events.loc[events["is_swr"] == is_swr, column].to_numpy()
        at_or_below = np.searchsorted(np.sort(values), grid, side="right")
        ax.plot(grid, 100 * at_or_below / len(values), color=color, lw=0.8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(label)
    ax.set_xticks(list(ticks), list(ticks.values()))


def draw_distance_from_animal(axes, shares: pd.DataFrame) -> None:
    """Each kind's content time by distance from the animal (``axes``: no SWR,
    then SWR), stacked: at a well (dark) and elsewhere (light). The upper axis
    says which wells the dark bars are: the animal's own within ``FAR_CM`` of it,
    other wells beyond."""
    width = np.diff(DISTANCE_EDGES_CM)
    top = 100 * shares.sum(axis=1).max()
    for ax, (is_swr, name, color) in zip(axes, KINDS, strict=True):
        kind = shares.loc[is_swr]
        left = kind.index.to_numpy()
        well = 100 * kind["at_well"].to_numpy()
        ax.bar(left, well, width, align="edge", color=color, lw=0)
        ax.bar(
            left, 100 * kind["elsewhere"].to_numpy(), width, bottom=well,
            align="edge", color=color, alpha=0.3, lw=0,
        )  # fmt: skip
        ax.set_xlim(DISTANCE_EDGES_CM[0], DISTANCE_EDGES_CM[-1])
        ax.set_ylim(0, top * 1.05)
        ax.text(
            0.55, 0.95, name, transform=ax.transAxes, ha="center", va="top",
            color=color, fontsize=SMALL_SIZE,
        )  # fmt: skip
    # One row, over the bars they name: side by side they fit only from the
    # axis's left edge to the far wells' bars.
    for x, align, name in (
        (DISTANCE_EDGES_CM[0] + 2, "left", "own well"),
        (DISTANCE_EDGES_CM[-1] - 2 * FAR_CM - 4, "right", "other wells"),
    ):
        axes[0].text(
            x, 0.70, name, transform=axes[0].get_xaxis_transform(), ha=align,
            va="bottom", color="0.3", fontsize=SMALL_SIZE,
        )  # fmt: skip
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Content distance\nfrom animal (cm)")
    axes[0].set_ylabel("% content time", y=0.0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=alpha, lw=0)
        for alpha in (1.0, 0.3)
    ]
    # Above the panel: inside it, the key covers the bars. One entry per line:
    # side by side they reach under the next panel's letter.
    axes[0].legend(
        handles, [f"at a well (<{AT_WELL_CM:.0f} cm)", "elsewhere"],
        frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=1,
        fontsize=SMALL_SIZE, handlelength=0.8, labelspacing=0.2,
        handletextpad=0.4, borderaxespad=0,
    )  # fmt: skip


def draw_ratio_track(ax, track_graph, edge_order, ratio: pd.Series) -> None:
    """Each segment of the track colored by log2 of its no-SWR to SWR content
    density (``content_along_arms.log2_density_ratio``), on a scale between the
    two kinds' colors; the scale saturates at ``RATIO_LIMIT_LOG2``."""
    polygons, values = [], []
    for first, second in edge_order:
        arm = track_graph.edges[first, second]["edge_id"]
        start = np.asarray(track_graph.nodes[first]["pos"], dtype=float)
        step = np.asarray(track_graph.nodes[second]["pos"], dtype=float) - start
        normal = np.array([-step[1], step[0]]) / np.hypot(*step) * ARM_HALF_WIDTH_CM
        step = step / N_SEGMENTS
        for segment in range(N_SEGMENTS):
            a, b = start + step * segment, start + step * (segment + 1)
            polygons.append([a + normal, b + normal, b - normal, a - normal])
            values.append(ratio[(arm, segment)])
    # Filled bands, not thick lines: the journal caps line width.
    bands = PolyCollection(
        polygons,
        array=np.asarray(values),
        cmap=LinearSegmentedColormap.from_list("kinds", [C_SWR, "#f7f7f7", C_NO_SWR]),
        norm=TwoSlopeNorm(0.0, -RATIO_LIMIT_LOG2, RATIO_LIMIT_LOG2),
        linewidths=0,
    )
    ax.add_collection(bands)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    steps = np.arange(-RATIO_LIMIT_LOG2, RATIO_LIMIT_LOG2 + 1)
    # Under the track: beside it, the labels run into the next panel.
    bar = plt.colorbar(
        bands, ax=ax, orientation="horizontal", shrink=0.75, aspect=18, pad=0.03,
        ticks=steps,
    )  # fmt: skip
    # The ends and the middle are labeled; the ticks between are 2×.
    limit = f"{2**RATIO_LIMIT_LOG2:g}×"
    labels = {steps[0]: f"{limit} SWR", 0: "equal", steps[-1]: f"{limit} No SWR"}
    bar.ax.set_xticklabels([labels.get(k, "") for k in steps])
    bar.ax.tick_params(length=2)
    bar.outline.set_linewidth(0.5)
    ax.set_title("Content time per cm,\nNo SWR : SWR", fontsize=SMALL_SIZE, pad=2)


def draw_profiles(ax, profiles: pd.DataFrame, shares: pd.DataFrame) -> None:
    """Content time per cm along the trunk arms (center to junction) and the
    leaf arms (junction to well), each kind, with its share at a well."""
    x = (np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS
    for is_swr, name, color in KINDS:
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
    for g, (is_swr, _, color) in enumerate(KINDS):
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


def draw_brain_state(axes, events: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Each animal's level of each signal in each kind of event, between the
    levels of quiet-rest (white) and running (black) windows as long as the
    kind's events (``axes``: one per signal, stacked, sharing the x axis)."""
    for ax, (signal, column, label) in zip(axes, SIGNALS, strict=True):
        levels = animal_signal_levels(events, reference, signal, column, matched=True)
        centers = []
        for k, (is_swr, _, color) in enumerate(KINDS):
            x = 3.7 * k + np.arange(3)
            values = levels.loc[is_swr, ["rest", "event", "running"]]
            for _, row in values.iterrows():
                ax.plot(x, row.to_numpy(), color="#999999", lw=0.5, zorder=1)
            for i, dot_color in enumerate((C_REST, color, C_RUN)):
                ax.scatter(
                    np.full(len(values), x[i]), values.iloc[:, i], s=7,
                    color=dot_color, edgecolors="black", linewidths=0.5, zorder=2,
                )  # fmt: skip
            centers.append(x[1])
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.set_title(label, fontsize=SMALL_SIZE, pad=2)
        # One label per kind; the key says which dots are rest and running.
        ax.set_xticks(centers, [name for _, name, _ in KINDS])
        ax.set_xlim(-0.7, centers[-1] + 1.7)
    axes[0].tick_params(labelbottom=False)
    axes[0].set_ylabel("Power (z)", y=-0.2)
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=2.6, mfc=color, mec="black", mew=0.5)
        for color in (C_REST, C_RUN)
    ]
    axes[0].legend(
        handles, ["rest", "running"], frameon=False, loc="lower left",
        bbox_to_anchor=(-0.05, 1.22), ncol=2, fontsize=SMALL_SIZE,
        handletextpad=0.0, columnspacing=0.6, borderaxespad=0,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# The panel letters as they sit in the figure: the letters of a row share a
# height, the letters of a column a left edge.
LETTER_ROWS = ("abcd", "efgh")
LETTER_COLUMNS = ("ae", "bf", "cg", "dh")


def letter_positions(fig, panels: dict[str, list]) -> dict[str, tuple[float, float]]:
    """Each panel letter's top left corner, in figure fractions.

    The letters of a row sit at one height, just above the highest of the row's
    drawn tops; the letters of a column sit at one left edge, the leftmost of
    the column's drawn extents (the rule of Figure 2's ``place_letters``). The
    figure must already be drawn.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    panels : dict
        Each letter's axes.
    """
    extents = {letter: drawn_extent(fig, axes) for letter, axes in panels.items()}
    tops = {
        letter: min(max(extents[other].y1 for other in row) + 0.02, 0.985)
        for row in LETTER_ROWS
        for letter in row
    }
    lefts = {
        letter: max(min(extents[other].x0 for other in column) - 0.004, 0.0)
        for column in LETTER_COLUMNS
        for letter in column
    }
    return {letter: (lefts[letter], tops[letter]) for letter in panels}


def generate_figure(inputs: dict):
    """Build Figure 3 from :func:`read_inputs`."""
    events = inputs["events"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # One grid for both rows, so the panels of a column share a left edge.
    grid = GridSpec(
        2, 4, figure=fig, height_ratios=[1.0, 1.25], hspace=0.7,
        width_ratios=[1.0, 1.15, 1.05, 0.95], wspace=0.58,
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
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in KINDS]
    fig.legend(
        handles, kind_labels(events), frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8, columnspacing=1.5,
    )  # fmt: skip

    e_grid = grid[1, 0].subgridspec(2, 1, hspace=0.15)
    ax_e = [fig.add_subplot(e_grid[0])]
    ax_e.append(fig.add_subplot(e_grid[1], sharex=ax_e[0]))
    draw_distance_from_animal(ax_e, inputs["distance_shares"])
    ax_f = fig.add_subplot(grid[1, 1])
    draw_ratio_track(ax_f, inputs["track_graph"], inputs["edge_order"], inputs["ratio"])
    ax_g = fig.add_subplot(grid[1, 2])
    draw_profiles(ax_g, inputs["profiles"], inputs["place_shares"])
    h_grid = grid[1, 3].subgridspec(2, 1, hspace=0.4)
    ax_h = [fig.add_subplot(h_grid[0])]
    ax_h.append(fig.add_subplot(h_grid[1], sharex=ax_h[0]))
    draw_brain_state(ax_h, events, inputs["reference"])

    fig.draw_without_rendering()
    panels_by_letter = {
        "a": [ax_a], "b": [ax_b], "c": [ax_c], "d": [ax_d, ax_d_cumulative],
        "e": ax_e, "f": [ax_f], "g": [ax_g], "h": ax_h,
    }  # fmt: skip
    for letter, (left, top) in letter_positions(fig, panels_by_letter).items():
        letter_panel(fig, letter, left, top)
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
