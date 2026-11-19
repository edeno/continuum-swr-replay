#!/usr/bin/env python
"""Generate Figure 3: what SWR and no-SWR non-local events are like.

The immobile non-local events of the canonical event table, split by whether
they overlap an SWR (Kay's detector, the SWR source). Each row asks one thing:

    Size and movement
        a — Duration.
        b — Spatial extent of the decoded content.
        c — Replay speed (each event's median).
        d — Arms the decoded position visits (of the track's nine).
    Where the content is
        e — Start distance: track distance from the animal to the event's first
            decoded position.
        f — Where on the track each kind's content is denser: the ratio of no-SWR
            to SWR content time per cm along every arm.
        g — Content time per cm from the center out to the junctions (trunk
            arms) and from the junctions out to the wells (leaf arms).
    Brain state
        h — Theta and slow and fast gamma during the events, against the same
            epochs' quiet rest and running. (Ripple power and MUA around the
            events are Figure 2's.)

Panels a–c and e draw each class's median and the SWR minus no-SWR difference
(duration-matched, except for duration itself) with its interval from the
property bootstrap (``bootstrap_event_properties.py``'s ``intervals.csv``): the
events summary's ``<property>_*`` values (``paper/reported/events.json``). Panels
f and g place every event time bin's MAP position on its arm
(``measure_content_along_arms.py``; ``Processed-Data/content_along_arms``) and
draw ``content_along_arms.content_density_along_arms``, the animals' mean; g
quotes the share of content time within ``AT_WELL_CM`` of a well, the summary's
``*_content_at_well_fraction`` (``event_statistics.content_occupancy_shares``). Panel h's animal values are
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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    content_density_along_arms,
    leaf_and_trunk_profiles,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    AT_WELL_CM,
    animal_signal_levels,
    class_medians,
    content_occupancy_shares,
)
from continuum_swr_replay.data_loaders.constants import ALL_ARMS
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    CONTENT_ALONG_ARMS_DIR,
    content_occupancy_dir,
    processed_data_dir,
)
from continuum_swr_replay.reported.events import default_inputs
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.simulated_run_panels import (
    LABEL_SIZE,
    PANEL_LETTER_SIZE,
    SMALL_SIZE,
    check_journal_limits,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
plt.rcParams.update(
    {
        "font.size": SMALL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": False,
    }
)
warnings.filterwarnings("ignore")

# Inches: saved with 0.1 in margins, about 17.3 cm (Cell Press full width).
WIDTH = 6.5
HEIGHT = 6.3

C_NL = COLORS["Non-Local"]
C_SWR = COLORS["SWR"]
C_REST = COLORS["Local"]
C_RUN = COLORS["No-Spike"]
KINDS = ((False, "no SWR", C_NL), (True, "SWR", C_SWR))

# The track drawn in panel f: one epoch's track graph (every epoch's arms run the
# same way, ``content_along_arms.check_arm_orientation``).
TRACK_EPOCH = "wilbur20210329_02_r1_HPC"
# Half the width of an arm drawn in panel f, track units (cm).
ARM_HALF_WIDTH_CM = 9.0
# Panels f and g: segments per arm, about 4 cm (two 2-cm place bins) each.
N_SEGMENTS = 13
# Panel f: the color scale's limit, log2 of the no-SWR to SWR density ratio.
RATIO_LIMIT_LOG2 = 2.0
# Panel d: the track's arms.
N_ARMS = len(ALL_ARMS)

# Panels a-c and f: (column, bootstrap property, x label, scale, bin edges, log x).
HISTOGRAMS = {
    "duration": (
        "event_duration_s", "duration", "Duration (ms)", 1e3,
        np.arange(0, 401, 8), False,
    ),
    "extent": (
        "spatial_extent_cm", "extent", "Spatial extent (cm)", 1.0,
        np.linspace(0, 240, 41), False,
    ),
    "speed": (
        "replay_speed_median_cm_s", "speed", "Replay speed (cm/s)", 1.0,
        np.geomspace(1, 1e5, 41), True,
    ),
    "start_distance": (
        "start_distance_cm", "start_distance", "Start distance\nfrom animal (cm)",
        1.0, np.linspace(0, 240, 41), False,
    ),
}  # fmt: skip
UNITS = {"duration": "ms", "extent": "cm", "speed": "cm/s", "start_distance": "cm"}

# Panel h: (signal, event column, label).
SIGNALS = (
    ("theta", "theta_z", "Theta"),
    ("slow_gamma", "slow_gamma_z", "Slow γ"),
    ("fast_gamma", "fast_gamma_z", "Fast γ"),
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_inputs() -> dict:
    """The events with their start distances and gamma, and the other inputs."""
    inputs = default_inputs()
    events = read_swr_table(inputs.event_table)
    n_events = len(events)
    extras = pd.concat(
        map(pd.read_parquet, sorted(inputs.event_start_rest_theta.glob("*.parquet")))
    )[["event_id", "start_distance_cm"]]
    gamma = pd.concat(
        pd.read_parquet(path, columns=["event_id", "slow_gamma_z", "fast_gamma_z"])
        for path in sorted(inputs.event_gamma.glob("*_events.parquet"))
    )
    events = events.merge(extras, on="event_id", how="left", validate="1:1").merge(
        gamma, on="event_id", how="left", validate="1:1"
    )
    assert len(events) == n_events
    missing = events[["start_distance_cm", "slow_gamma_z", "fast_gamma_z"]].isna()
    if missing.any().any():
        raise ValueError(f"events missing a joined value:\n{missing.sum()}")

    reference = pd.read_parquet(inputs.brain_state_reference)
    gamma_reference = pd.read_parquet(inputs.event_gamma / "reference.parquet")
    gamma_columns = [c for c in gamma_reference.columns if "gamma" in c]
    reference = reference.merge(
        gamma_reference[[*SESSION_COLUMNS, *gamma_columns]],
        on=list(SESSION_COLUMNS),
        how="inner",
        validate="1:1",
    )
    occupancy = pd.read_csv(
        content_occupancy_dir() / "content_occupancy_per_session.csv"
    )
    along = pd.concat(
        pd.read_parquet(path, columns=["event_id", "arm", "fraction", "arm_length_cm"])
        for path in sorted(
            (processed_data_dir() / CONTENT_ALONG_ARMS_DIR).glob("*.parquet")
        )
    )
    n_bins = len(along)
    along = along.merge(
        events[["event_id", "animal", "is_swr"]], on="event_id", validate="m:1"
    )
    if len(along) != n_bins or along["event_id"].nunique() != n_events:
        raise ValueError("the content-along-arms bins do not cover the event table")
    density = content_density_along_arms(along, N_SEGMENTS)
    env = joblib.load(decode_paths(TRACK_EPOCH, processed_data_dir()).environment)
    return {
        "events": events,
        "intervals": pd.read_csv(
            inputs.property_bootstrap / "intervals.csv", index_col=0
        ),
        "reference": reference,
        "occupancy_shares": content_occupancy_shares(occupancy),
        "density": density,
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
    analysed epoch holds both kinds, so the two classes' levels must be equal.
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


def draw_histogram(ax, events: pd.DataFrame, intervals: pd.DataFrame, key: str):
    """A property's distribution for each kind, its medians and the difference."""
    column, prop, label, scale, edges, log_x = HISTOGRAMS[key]
    medians = class_medians(events, column)
    top = 0.0
    for is_swr, _, color in KINDS:
        values = scale * events.loc[events["is_swr"] == is_swr, column].to_numpy()
        values = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        share = counts / counts.sum()
        ax.stairs(share, edges, color=color, fill=True, alpha=0.25, lw=0)
        ax.stairs(share, edges, color=color, lw=0.8)
        top = max(top, share.max())
    for (_, _, color), m in zip(
        KINDS, (medians["median_non_swr"], medians["median_swr"]), strict=True
    ):
        ax.plot(scale * m, top * 1.08, marker="v", ms=3.5, color=color, lw=0)
    row = intervals.loc[prop]
    unit = UNITS[key]
    word = "difference" if key == "duration" else "matched_difference"
    heading = "SWR − no SWR" if key == "duration" else "Δ matched"
    ax.set_title(
        f"{heading}: {row[word]:+.0f} {unit}\n"
        f"[{row[f'{word}_low']:.0f}, {row[f'{word}_high']:.0f}]",
        fontsize=SMALL_SIZE,
        pad=2,
        linespacing=1.1,
    )
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


def draw_ratio_track(ax, track_graph, edge_order, density: pd.DataFrame) -> None:
    """Each segment of the track colored by log2 of its no-SWR to SWR content
    density, on a scale between the two kinds' colors."""
    ratio = np.log2(
        density.loc[False, "density_per_cm"] / density.loc[True, "density_per_cm"]
    )
    cmap = LinearSegmentedColormap.from_list("kinds", [C_SWR, "#f7f7f7", C_NL])
    norm = TwoSlopeNorm(0.0, -RATIO_LIMIT_LOG2, RATIO_LIMIT_LOG2)
    for first, second in edge_order:
        arm = track_graph.edges[first, second]["edge_id"]
        start = np.asarray(track_graph.nodes[first]["pos"], dtype=float)
        step = np.asarray(track_graph.nodes[second]["pos"], dtype=float) - start
        normal = np.array([-step[1], step[0]]) / np.hypot(*step) * ARM_HALF_WIDTH_CM
        step = step / N_SEGMENTS
        for segment in range(N_SEGMENTS):
            a, b = start + step * segment, start + step * (segment + 1)
            corners = np.array([a + normal, b + normal, b - normal, a - normal])
            value = np.clip(ratio[(arm, segment)], -RATIO_LIMIT_LOG2, RATIO_LIMIT_LOG2)
            # A filled band, not a thick line: the journal caps line width.
            ax.fill(corners[:, 0], corners[:, 1], color=cmap(norm(value)), lw=0)
    ax.set_aspect("equal")
    ax.axis("off")
    bar = plt.colorbar(
        plt.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.7, aspect=12, pad=0.02,
        ticks=[-2, -1, 0, 1, 2],
    )  # fmt: skip
    bar.ax.set_yticklabels(["4× SWR", "2×", "equal", "2×", "4× no SWR"])
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
        group = "SWR" if is_swr else "nonSWR"
        ax.text(
            0.03, 0.97 - 0.09 * is_swr,
            f"{name}: {100 * shares.loc[group, 'at_well']:.0f}% at a well",
            transform=ax.transAxes, color=color, ha="left", va="top",
            fontsize=SMALL_SIZE,
        )  # fmt: skip
    ax.set_yscale("log")
    ax.axvline(1, color="0.8", lw=0.5, zorder=0)
    ax.set_xticks([0, 1, 2], ["center", "junction", "well"])
    ax.set_xlim(0, 2)
    ax.set_ylabel("% content time per cm")
    ax.set_title(
        f"at a well: <{AT_WELL_CM:.0f} cm from one", fontsize=SMALL_SIZE, pad=8
    )


def draw_brain_state(axes, events: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Each animal's rest, no-SWR, SWR and running level of each signal."""
    colors = (C_REST, C_NL, C_SWR, C_RUN)
    for ax, (signal, column, label) in zip(axes, SIGNALS, strict=True):
        values = animal_signal_values(events, reference, signal, column)
        x = np.arange(4)
        for _, row in values.iterrows():
            ax.plot(x, row.to_numpy(), color="#999999", lw=0.5, zorder=1)
        for i, color in enumerate(colors):
            ax.scatter(
                np.full(len(values), i), values.iloc[:, i], s=6, color=color,
                lw=0, zorder=2,
            )  # fmt: skip
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.set_title(label, fontsize=LABEL_SIZE, pad=2)
        ax.set_xticks(x, values.columns, rotation=45, ha="right")
        ax.set_xlim(-0.5, 3.5)
    axes[0].set_ylabel("Power (z)")


def letter(fig, renderer, ax, text: str, x: float | None, y: float) -> None:
    """A panel letter at figure height y, at the axes' drawn left or at x."""
    box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    fig.text(
        max(box.x0 - 0.004, 0.0) if x is None else x,
        y,
        text,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def generate_figure(inputs: dict):
    """Build Figure 3 from :func:`read_inputs`."""
    events = inputs["events"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = GridSpec(
        3, 1, figure=fig, height_ratios=[1.0, 1.05, 1.0], hspace=0.75,
        left=0.08, right=0.99, top=0.96, bottom=0.08,
    )  # fmt: skip

    row1 = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.5)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(row1[i]) for i in range(4))
    for ax, key in ((ax_a, "duration"), (ax_b, "extent"), (ax_c, "speed")):
        draw_histogram(ax, events, inputs["intervals"], key)
    ax_a.set_ylabel("Share of events")
    draw_arm_counts(ax_d, events)
    ax_d.set_xlabel("Arms visited")
    ax_d.set_ylabel("% events")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in KINDS]
    ax_d.legend(
        handles, [n for _, n, _ in KINDS], frameon=False, loc="upper right",
        handlelength=0.8, borderaxespad=0,
    )  # fmt: skip

    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.6, 1.3], wspace=0.6)
    ax_e = fig.add_subplot(row2[0])
    draw_histogram(ax_e, events, inputs["intervals"], "start_distance")
    ax_e.set_ylabel("Share of events")
    ax_f = fig.add_subplot(row2[1])
    draw_ratio_track(
        ax_f, inputs["track_graph"], inputs["edge_order"], inputs["density"]
    )
    ax_g = fig.add_subplot(row2[2])
    draw_profiles(ax_g, inputs["profiles"], inputs["occupancy_shares"])

    row3 = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.5)
    ax_h = [fig.add_subplot(row3[i]) for i in range(len(SIGNALS))]
    draw_brain_state(ax_h, events, inputs["reference"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = 0.005
    rows = (
        ((ax_a, "a", x_left), (ax_b, "b", None), (ax_c, "c", None),
         (ax_d, "d", None)),
        ((ax_e, "e", x_left), (ax_f, "f", None), (ax_g, "g", None)),
        ((ax_h[0], "h", x_left),),
    )  # fmt: skip
    for row in rows:
        top = max(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y1
            for ax, _, _ in row
        )
        for ax, text, x in row:
            letter(fig, renderer, ax, text, x, min(top - 0.012, 0.985))
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
