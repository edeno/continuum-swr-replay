#!/usr/bin/env python
"""Generate Figure 4d: Population modulation heatmaps.

Two side-by-side heatmaps showing peri-event modulation time courses for
all mPFC neurons — SWR modulation (left) and non-SWR replay modulation
(right). Neurons are sorted by peak SWR response time, with the same
ordering applied to the replay panel for direct comparison.

Usage:
    python scripts/generate_figure04d_population_heatmaps.py
    python scripts/generate_figure04d_population_heatmaps.py --min-firing-rate 2.0
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
from continuum_swr_replay.figure_utilities import TWO_COLUMN, save_figure
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_population_heatmaps(
    swr_mod: np.ndarray,
    no_swr_mod: np.ndarray,
    lag_time: np.ndarray,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot side-by-side heatmaps of SWR and non-SWR replay modulation."""
    # Sort neurons by peak SWR modulation time
    peak_times = lag_time[np.argmax(np.abs(swr_mod), axis=0)]
    sort_idx = np.argsort(peak_times)

    swr_sorted = swr_mod[:, sort_idx]
    no_swr_sorted = no_swr_mod[:, sort_idx]

    vmin, vmax = -np.log(2), np.log(2)
    lag_ms = lag_time * 1000

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(TWO_COLUMN, TWO_COLUMN * 0.4),
        sharey=True,
        constrained_layout=True,
    )

    # SWR heatmap
    im1 = ax1.imshow(
        swr_sorted.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax1.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Time from SWR (ms)")
    ax1.set_ylabel("Neuron (sorted by SWR peak time)")
    ax1.set_title("SWR modulation")

    # Non-SWR replay heatmap (same neuron order)
    im2 = ax2.imshow(
        no_swr_sorted.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], no_swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax2.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Time from non-SWR replay (ms)")
    ax2.set_title("Non-SWR replay modulation")

    for ax in (ax1, ax2):
        sns.despine(ax=ax)
        ax.tick_params(axis="both", which="both", length=3, direction="out")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label("Log firing rate change")
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(["0.5x", "1x", "2x"])

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    print(f"Saved: {figure_name}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure04"),
    )
    parser.add_argument("--format", type=str, choices=["pdf", "png", "svg"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-firing-rate", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [args.format] if args.format else None

    df, modulations, lag_time = load_population_data(
        args.data_dir, min_firing_rate=args.min_firing_rate
    )

    plot_population_heatmaps(
        modulations["SWR"],
        modulations["no_SWR"],
        lag_time,
        args.output_dir,
        "fig04d_population_heatmaps",
        formats,
        args.dpi,
    )


if __name__ == "__main__":
    main()
