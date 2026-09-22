#!/usr/bin/env python
"""Supplemental figure: what each detector does to a replay of known shape.

Summarizes the sweep in `scripts/qa/sweep_memoryless_detection.py`, which
injected 4536 replays of controlled path length, duration, firing rate and
distance from the animal's own place into a simulated recording and recorded
what the memoryless sliding-window detector and this project's switching
state-space model each reported. Both read one encoding model, fitted once.

Reads the aggregates the sweep writes, so the figure and the text of
`docs/design/replay_detection_comparison/02-memoryless-sequence.md` cannot
disagree. Re-run the sweep to change them.

Usage:
    python scripts/figures/generate_memoryless_sweep.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from continuum_swr_replay.figure_utilities import (
    TWO_COLUMN,
    WONG,
    label_panels,
    save_figure,
    set_figure_defaults,
)

SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_sweep_summary.csv"
)
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = WONG[:8]

SLIDING = "sliding-window detector"
STATE_SPACE = "state-space detector"
DISTANCE_ORDER = ("(40, 80]", "(80, 120]", "(120, 200]")


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No sweep summary at {path}. Run "
            "`python scripts/qa/sweep_memoryless_detection.py` first."
        )
    return pd.read_csv(path)


def detection_grid(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Detection rate as path length by duration."""
    rows = summary[
        (summary["summary"] == "detection_rate_by_path_and_duration")
        & (summary.measure == measure)
    ]
    return rows.pivot(index="path_cm", columns="duration_s", values="value")


def plot_detection(ax, grid: pd.DataFrame, title: str, show_y: bool):
    image = ax.imshow(
        100 * grid.to_numpy(),
        vmin=0,
        vmax=100,
        cmap="magma",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"{1000 * c:.0f}" for c in grid.columns], fontsize=5)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"{r:.0f}" for r in grid.index], fontsize=5)
    ax.set_xlabel("how long the replay really lasted (ms)")
    if show_y:
        ax.set_ylabel("path the replay travels (cm)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            percent = 100 * grid.iat[row, column]
            ax.text(
                column,
                row,
                f"{percent:.0f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if percent < 55 else "black",
            )
    return image


def plot_duration_bias(ax, summary: pd.DataFrame):
    """Reported duration over true duration, for events both procedures found."""
    rows = summary[summary["summary"] == "median_duration_ratio_by_duration"]
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    for measure, color, label in (
        ("sliding_duration_ratio", VERMILION, SLIDING),
        ("state_space_duration_ratio", BLUE, STATE_SPACE),
    ):
        series = rows[rows.measure == measure].sort_values("duration_s")
        ax.plot(
            1000 * series.duration_s,
            series.value,
            color=color,
            lw=1.0,
            marker="o",
            ms=2.5,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1x", "2x", "5x", "10x"])
    ax.set_xlabel("how long the replay really lasted (ms)")
    ax.set_ylabel("reported duration / actual duration")
    ax.set_title(
        "how long each detector says an event lasted,\nagainst how long it was",
        fontsize=6,
    )
    ax.legend(fontsize=5, frameon=False, loc="upper right", handlelength=1.3)
    ax.text(
        250,
        5.5,
        "a real 40 ms replay is\nreported as 300 ms",
        fontsize=5,
        color=VERMILION,
        ha="center",
        va="center",
        linespacing=1.3,
    )


def plot_absorption(ax, summary: pd.DataFrame):
    """The over-call against how far the replay sits from the animal's own place."""
    rows = summary[summary["summary"] == "median_by_distance_from_animal"]
    order = [
        bin_name for bin_name in DISTANCE_ORDER if bin_name in set(rows.distance_bin_cm)
    ]
    at = np.arange(len(order))

    def series(measure):
        indexed = rows[rows.measure == measure].set_index("distance_bin_cm").value
        return np.array([indexed[bin_name] for bin_name in order])

    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.plot(
        at, series("sliding_duration_ratio"), color=VERMILION, lw=1.0, marker="o", ms=3
    )
    ax.plot(
        at, series("state_space_duration_ratio"), color=BLUE, lw=1.0, marker="o", ms=3
    )
    for x, local in zip(at, series("sliding_local_fraction"), strict=True):
        ax.annotate(
            f"{100 * local:.0f}% of the event's 80 ms windows\n"
            "decode the animal's own place",
            xy=(x, 1.0),
            xytext=(x + (0.42 if x == 0 else 0.0), 1.62 if x == 0 else 1.32),
            fontsize=4.5,
            color="0.35",
            ha="center",
            va="center",
            linespacing=1.25,
        )
    ax.set_xticks(at)
    ax.set_xticklabels([bin_name.replace(", ", "-").strip("(]") for bin_name in order])
    ax.set_ylim(0.85, 2.6)
    ax.set_xlabel("replay's distance from the animal's place (cm)")
    ax.set_ylabel("reported duration / actual duration")
    ax.set_title(
        "events are reported too long only when the replay is near\n"
        "the animal's own place, where local coding is absorbed into them",
        fontsize=6,
    )


def make_figure(summary: pd.DataFrame):
    overall = summary[summary["summary"] == "overall"].set_index("measure").value

    set_figure_defaults(context="paper")
    fig = plt.figure(
        figsize=(TWO_COLUMN, TWO_COLUMN * 0.62), dpi=450, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.07)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.85])

    inner = grid[0, :].subgridspec(1, 2, wspace=0.06)
    ax_sliding = fig.add_subplot(inner[0])
    ax_state = fig.add_subplot(inner[1])
    image = plot_detection(
        ax_sliding,
        detection_grid(summary, "sliding_matches"),
        f"{SLIDING}\nreported {100 * overall['sliding_matches']:.0f}% of the "
        "replays that were put in",
        show_y=True,
    )
    plot_detection(
        ax_state,
        detection_grid(summary, "state_space_matches"),
        f"{STATE_SPACE}\nreported {100 * overall['state_space_matches']:.0f}% of them",
        show_y=False,
    )
    bar = fig.colorbar(image, ax=[ax_sliding, ax_state], fraction=0.03, pad=0.01)
    bar.set_label("replays reported (%)", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)

    ax_duration = fig.add_subplot(grid[1, 0])
    plot_duration_bias(ax_duration, summary)
    ax_distance = fig.add_subplot(grid[1, 1])
    plot_absorption(ax_distance, summary)

    label_panels(
        [ax_sliding, ax_state, ax_duration, ax_distance], xpos=-0.09, ypos=1.10
    )
    fig.suptitle(
        f"{int(overall['n_cells'])} made-up replays, each with a path, a duration and "
        "a firing rate we chose, written into the spike trains\nof a simulated "
        "recording one at a time, so that the right answer is known\n"
        "A detector counts as reporting a replay only when what it returns is that "
        "replay: on its path, and not covering far more ground.\n"
        f"{100 * overall['found_by_state_space_only']:.0f}% were reported by the "
        f"{STATE_SPACE} alone, "
        f"{100 * overall['found_by_sliding_window_only']:.0f}% by the {SLIDING} "
        f"alone. {100 * overall['sliding_accepted_but_not_the_injection']:.0f}% of "
        f"the {SLIDING}'s accepted events are not the replay that was put in.",
        fontsize=6,
        linespacing=1.6,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/memoryless_comparison")
    )
    args = parser.parse_args()

    summary = read_summary(args.summary)
    fig = make_figure(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure("memoryless_sweep", base_dir=args.output_dir)
    plt.close(fig)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
