#!/usr/bin/env python
"""Generate Figure 4e: Population average modulation time courses.

Mean ± SEM of SWR and non-SWR replay modulation across all mPFC neurons.
Shows whether a population-level replay signal exists independent of SWR.

Usage:
    python scripts/generate_figure04e_population_average.py
"""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.figure_utilities import PANEL_SINGLE, save_figure
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.visualization.palettes import COLORS

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_population_average(
    modulations: dict[str, np.ndarray],
    lag_time: np.ndarray,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot population mean ± SEM modulation time courses."""
    fig, ax = plt.subplots(figsize=PANEL_SINGLE)
    lag_ms = lag_time * 1000

    for name, color, label in [
        ("SWR", COLORS["SWR"], "SWR"),
        ("no_SWR", COLORS["Non-Local"], "Non-SWR replay"),
    ]:
        arr = modulations[name]
        mean = np.nanmean(arr, axis=1)
        sem = np.nanstd(arr, axis=1) / np.sqrt(arr.shape[1])

        ax.plot(lag_ms, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(lag_ms, mean - sem, mean + sem, color=color, alpha=0.2)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Time from event onset (ms)")
    ax.set_ylabel("Log firing rate change")
    ax.set_xlim(lag_ms[0], lag_ms[-1])

    # Add fold-change ticks on right
    ax_right = ax.twinx()
    yticks = ax.get_yticks()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels([f"{np.exp(y):.2f}x" for y in yticks])
    ax_right.set_ylabel("Fold change")
    sns.despine(ax=ax_right, right=False, left=True)

    ax.legend(frameon=False, loc="lower left")
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=3, direction="out")
    ax.set_title(f"Population average (n={modulations['SWR'].shape[1]:,})")

    save_figure(plt.gcf(), figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    print(f"Saved: {figure_name}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure04")
    )
    parser.add_argument(
        "--format", type=str, choices=["pdf", "png", "svg"], default=None
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-firing-rate", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=processed_data_dir())
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [args.format] if args.format else None

    df, modulations, lag_time = load_population_data(
        args.data_dir, min_firing_rate=args.min_firing_rate
    )

    plot_population_average(
        modulations,
        lag_time,
        args.output_dir,
        "fig04e_population_average",
        formats,
        args.dpi,
    )


if __name__ == "__main__":
    main()
