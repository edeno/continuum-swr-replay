#!/usr/bin/env python
"""Generate Figure 3: what SWR and no-SWR non-local events are like.

The immobile non-local events of the canonical event table, split by whether
they overlap an SWR (Kay's detector, the SWR source). Each row asks one thing:

    Size and movement
        a — Duration.
        b — Spatial extent of the decoded content.
        c — Replay speed (each event's median).
        d — Arms the decoded position visits (of the track's nine).
        e — The mean share of an event in Non-Local Continuous and in Non-Local
            Fragmented (each event's mean state probability, each event weighted
            alike); bars the animals' mean, dots the animals
            (``event_statistics.state_composition_statistics``).
    Where the content is
        f — Content time by track distance from the animal, at a well (within
            ``AT_WELL_CM`` of one) and elsewhere: content at a well far from
            the animal is at another well, near it at the animal's own.
        g — Where on the track each kind's content is denser: the ratio of no-SWR
            to SWR content time per cm along every arm.
        h — Content time per cm from the center out to the junctions (trunk
            arms) and from the junctions out to the wells (leaf arms).
    Brain state
        i — Slow and fast gamma during the events, against the same epochs'
            quiet rest and running. (Ripple power, MUA, theta and head speed
            around the events are Figure 2's.)

Panels a–c mark each class's median (``event_statistics.class_medians``),
the events summary's ``<property>_median_*`` values
(``paper/reported/events.json``). Panels
f–h place every event time bin's MAP position on its arm
(``measure_content_along_arms.py``; ``Processed-Data/content_along_arms``): f
draws ``content_along_arms.distance_from_animal_shares``, g and h
``content_along_arms.content_density_along_arms``, each the animals' mean; h
quotes the share of content time within ``AT_WELL_CM`` of a well, the summary's
``*_content_at_well_fraction`` (``content_along_arms.content_place_shares``,
``place_share_statistics``). Every one of them weighs each epoch of an animal the
same and each animal the same. Panel i's animal values are
``event_statistics.animal_signal_levels``: the values the summary's
``<signal>_animals_*`` counts compare.

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
    animal_signal_levels,
    class_medians,
    state_composition_statistics,
)
from continuum_swr_replay.data_loaders.constants import ALL_ARMS
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
    LABEL_SIZE,
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
HEIGHT = 6.3

C_NO_SWR = COLORS["No SWR"]
C_SWR = COLORS["SWR"]
# Panel i's rest and running levels: neutral, not decoder-state colors.
C_REST = "white"
C_RUN = "black"
KINDS = ((False, "no SWR", C_NO_SWR), (True, "SWR", C_SWR))

# The track drawn in panel g: one epoch's track graph (every epoch's arms run the
# same way, ``content_along_arms.check_arm_orientation``).
TRACK_EPOCH = "wilbur20210329_02_r1_HPC"
# Half the width of an arm drawn in panel g, track units (cm).
ARM_HALF_WIDTH_CM = 9.0
# Panels g and h: segments per arm, about 4 cm (two 2-cm place bins) each.
N_SEGMENTS = 13
# Panel g: the color scale's limit, log2 of the no-SWR to SWR density ratio.
RATIO_LIMIT_LOG2 = 2
# Panel f: distance from the animal, cm (the track's longest is under 240).
DISTANCE_EDGES_CM = np.arange(0, 241, 8)
# Panel d: the track's arms.
N_ARMS = len(ALL_ARMS)

# Panels a-c: (column, x label, scale, bin edges, log x).
HISTOGRAMS = {
    "duration": (
        "event_duration_s", "Duration (ms)", 1e3,
        np.arange(0, 401, 8), False,
    ),
    "extent": (
        "spatial_extent_cm", "Spatial extent (cm)", 1.0,
        np.linspace(0, 240, 41), False,
    ),
    "speed": (
        "replay_speed_median_cm_s", "Replay speed (cm/s)", 1.0,
        np.geomspace(1, 1e5, 41), True,
    ),
}  # fmt: skip

# Panel i: (signal, event column, label). Theta is Figure 2's.
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


def animal_signal_values(
    events: pd.DataFrame, reference: pd.DataFrame, signal: str, column: str
) -> pd.DataFrame:
    """Each animal's rest, no-SWR, SWR and running value of one signal, z.

    ``event_statistics.animal_signal_levels``: the event values the summary's
    animal counts compare, and the levels they are compared with. Each class's
    levels come from the animal's epochs with an event of that class; every
    analyzed epoch holds both kinds, so the two classes' levels must be equal.
    """
    levels = animal_signal_levels(events, reference, signal, column)
    non_swr, swr = levels.loc[False], levels.loc[True]
    for level in ("rest", "running"):
        if not np.allclose(non_swr[level], swr.loc[non_swr.index, level]):
            raise ValueError(f"{signal}: the classes' {level} levels differ")
    return pd.DataFrame(
        {
            "rest": non_swr["rest"],
            "no SWR": non_swr["event"],
            "SWR": swr["event"],
            "running": non_swr["running"],
        }
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def draw_histogram(ax, events: pd.DataFrame, key: str) -> None:
    """A property's distribution for each kind, % of the kind's events per bin,
    with each kind's median (triangle)."""
    column, label, scale, edges, log_x = HISTOGRAMS[key]
    medians = class_medians(events, column)
    top = 0.0
    for is_swr, _, color in KINDS:
        values = scale * events.loc[events["is_swr"] == is_swr, column].to_numpy()
        values = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        share = 100 * counts / counts.sum()
        ax.stairs(share, edges, color=color, fill=True, alpha=0.25, lw=0)
        ax.stairs(share, edges, color=color, lw=0.8)
        top = max(top, share.max())
    for (_, _, color), m in zip(
        KINDS, (medians["median_non_swr"], medians["median_swr"]), strict=True
    ):
        ax.plot(scale * m, top * 1.08, marker="v", ms=3.5, color=color, lw=0)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel(label)
    if log_x:  # the end bins hold the clipped values
        ax.set_xticks([1, 1e2, 1e4, 1e5], ["≤1", "$10^2$", "$10^4$", "≥$10^5$"])


def draw_arm_counts(ax, events: pd.DataFrame) -> None:
    """The share of each kind's events visiting each number of the nine arms."""
    values = np.arange(1, N_ARMS + 1)
    for k, (is_swr, _, color) in enumerate(KINDS):
        counts = events.loc[events["is_swr"] == is_swr, "n_unique_arms"]
        if not counts.between(1, N_ARMS).all():
            raise ValueError(f"an event visits no arm or more than {N_ARMS}")
        share = counts.value_counts(normalize=True).reindex(values, fill_value=0)
        ax.bar(values + (k - 0.5) * 0.4, 100 * share, width=0.4, color=color, lw=0)
    ax.set_xticks(values)
    ax.set_xlim(0.4, N_ARMS + 0.6)


def draw_distance_from_animal(axes, shares: pd.DataFrame) -> None:
    """Each kind's content time by distance from the animal (``axes``: no SWR,
    then SWR), stacked: at a well (dark) and elsewhere (light)."""
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
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Content distance\nfrom animal (cm)")
    axes[0].set_ylabel("% content time", y=0.0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=alpha, lw=0)
        for alpha in (1.0, 0.3)
    ]
    axes[1].legend(
        handles, [f"at a well (<{AT_WELL_CM:.0f} cm)", "elsewhere"],
        frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.8),
        fontsize=SMALL_SIZE, handlelength=0.8, borderaxespad=0,
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
    bar = plt.colorbar(
        bands, ax=ax, shrink=0.7, aspect=12, pad=0.02, ticks=steps
    )  # fmt: skip
    labels = [f"{2 ** abs(k):g}×" if k else "equal" for k in steps]
    labels[0] += " SWR"
    labels[-1] += " no SWR"
    bar.ax.set_yticklabels(labels)
    bar.ax.tick_params(length=2)
    bar.outline.set_linewidth(0.5)
    ax.set_title("Content time per cm,\nno SWR : SWR", fontsize=SMALL_SIZE, pad=2)


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
    """Panel e: the mean share of an event in Continuous and in Fragmented.

    Bars are the animals' mean, dots the animals
    (``event_statistics.state_composition_statistics``).
    """
    for g, (_, group, color) in enumerate(KINDS):
        row = statistics.loc[group]
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
    """Each animal's rest, no-SWR, SWR and running level of each signal."""
    colors = (C_REST, C_NO_SWR, C_SWR, C_RUN)
    for ax, (signal, column, label) in zip(axes, SIGNALS, strict=True):
        values = animal_signal_values(events, reference, signal, column)
        x = np.arange(4)
        for _, row in values.iterrows():
            ax.plot(x, row.to_numpy(), color="#999999", lw=0.5, zorder=1)
        for i, color in enumerate(colors):
            ax.scatter(
                np.full(len(values), i), values.iloc[:, i], s=7, color=color,
                edgecolors="black", linewidths=0.5, zorder=2,
            )  # fmt: skip
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.set_title(label, fontsize=LABEL_SIZE, pad=2)
        ax.set_xticks(x, values.columns, rotation=45, ha="right")
        ax.set_xlim(-0.5, 3.5)
    axes[0].set_ylabel("Power (z)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def generate_figure(inputs: dict):
    """Build Figure 3 from :func:`read_inputs`."""
    events = inputs["events"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = GridSpec(
        3, 1, figure=fig, height_ratios=[1.0, 1.05, 1.0], hspace=0.75,
        left=0.08, right=0.99, top=0.94, bottom=0.08,
    )  # fmt: skip

    row1 = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1.3, 0.9], wspace=0.55)
    ax_a, ax_b, ax_c, ax_d, ax_e = (fig.add_subplot(row1[i]) for i in range(5))
    draw_state_composition(ax_e, state_composition_statistics(events))
    for ax, key in ((ax_a, "duration"), (ax_b, "extent"), (ax_c, "speed")):
        draw_histogram(ax, events, key)
    ax_a.set_ylabel("% events")
    draw_arm_counts(ax_d, events)
    ax_d.set_xlabel("Arms visited")
    ax_d.set_ylabel("% events")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in KINDS]
    fig.legend(
        handles, [n for _, n, _ in KINDS], frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8, columnspacing=1.5,
    )  # fmt: skip

    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.6, 1.3], wspace=0.6)
    f_grid = row2[0].subgridspec(2, 1, hspace=0.15)
    ax_f = [fig.add_subplot(f_grid[0])]
    ax_f.append(fig.add_subplot(f_grid[1], sharex=ax_f[0]))
    draw_distance_from_animal(ax_f, inputs["distance_shares"])
    ax_g = fig.add_subplot(row2[1])
    draw_ratio_track(ax_g, inputs["track_graph"], inputs["edge_order"], inputs["ratio"])
    ax_h = fig.add_subplot(row2[2])
    draw_profiles(ax_h, inputs["profiles"], inputs["place_shares"])

    row3 = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.5)
    ax_i = [fig.add_subplot(row3[i]) for i in range(len(SIGNALS))]
    draw_brain_state(ax_i, events, inputs["reference"])

    fig.draw_without_rendering()
    rows = (
        ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")),
        ((ax_f[0], "f"), (ax_g, "g"), (ax_h, "h")),
        ((ax_i[0], "i"),),
    )
    for row in rows:
        top = min(drawn_extent(fig, [ax for ax, _ in row]).y1 + 0.02, 0.985)
        for i, (ax, text) in enumerate(row):
            left = 0.005 if i == 0 else drawn_extent(fig, [ax]).x0 - 0.004
            letter_panel(fig, text, left, top)
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
