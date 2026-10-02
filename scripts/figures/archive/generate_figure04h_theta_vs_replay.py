#!/usr/bin/env python
"""Generate Figure 4h: Theta modulation vs. replay modulation scatter.

Shows whether neurons that are theta-modulated during running are also
replay-modulated during immobility, or whether these are separate populations.

Usage:
    python scripts/generate_figure04h_theta_vs_replay.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.figure_utilities import PANEL_SINGLE_SQUARE, save_figure
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_theta_vs_replay(
    df,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Scatter: theta modulation vs peak replay modulation magnitude."""
    x = df["theta_mod"]
    y = np.abs(df["non_local_no_SWR_max_mod"])
    replay_sig = df["non_local_no_SWR vs. constant: fdr_q"] < 0.05

    valid = x.notna() & y.notna() & (y < 5)  # clip extreme modulations
    x, y, sig = x[valid], y[valid], replay_sig[valid]

    rho, p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=PANEL_SINGLE_SQUARE)

    # Plot non-significant in gray, significant in green
    ax.scatter(
        x[~sig], y[~sig], s=1, alpha=0.1, color="gray",
        rasterized=True, label=f"Not sig. (n={(~sig).sum():,})",
    )
    ax.scatter(
        x[sig], y[sig], s=2, alpha=0.2, color=COLORS["Non-Local"],
        rasterized=True, label=f"Replay sig. (n={sig.sum():,})",
    )

    ax.set_xlabel("Theta modulation depth (running)")
    ax.set_ylabel("|Peak replay modulation| (immobility)")
    ax.set_title(f"Theta vs. replay (rho={rho:.2f}, p={p:.1e})")

    ax.legend(frameon=False, fontsize=7, loc="upper right", markerscale=5)
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=3, direction="out")

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    print(f"Saved: {figure_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/figure04"))
    parser.add_argument("--format", type=str, choices=["pdf", "png", "svg"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-firing-rate", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [args.format] if args.format else None

    df, _, _ = load_population_data(args.data_dir, min_firing_rate=args.min_firing_rate)

    plot_theta_vs_replay(
        df, args.output_dir, "fig04h_theta_vs_replay", formats, args.dpi,
    )


if __name__ == "__main__":
    main()
