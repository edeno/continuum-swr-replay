#!/usr/bin/env python
"""Generate Figure 4g: Non-SWR replay robustness scatter.

Compares peak non-SWR replay modulation estimated two ways:
1. SWR-controlled model (controlling for all SWRs)
2. Replay-split model (controlling for SWR-coincident replay only)

Tight correlation along the identity line means the replay response is
robust to how ripple confounds are controlled.

Usage:
    python scripts/generate_figure04g_robustness.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.figure_utilities import PANEL_SINGLE_SQUARE, save_figure
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_robustness_scatter(
    df,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Scatter: SWR-controlled vs replay-split non-SWR replay modulation."""
    x = df["non_local_no_SWR_max_mod"]
    y = df["non_local_no_SWR_split_max_mod"]
    valid = x.notna() & y.notna() & (x.abs() < 5) & (y.abs() < 5)  # clip extremes
    x, y = x[valid], y[valid]

    r, p = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=PANEL_SINGLE_SQUARE)

    ax.scatter(x, y, s=1, alpha=0.15, color="k", rasterized=True)

    lim = (
        max(
            abs(x.quantile(0.01)),
            abs(x.quantile(0.99)),
            abs(y.quantile(0.01)),
            abs(y.quantile(0.99)),
        )
        * 1.1
    )
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.5, label="Identity")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("Peak mod. (SWR-controlled)")
    ax.set_ylabel("Peak mod. (replay-split)")
    ax.set_title(f"Robustness check (r={r:.2f})")
    ax.set_aspect("equal")

    ax.legend(frameon=False, fontsize=7, loc="upper left")
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=3, direction="out")

    save_figure(figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    print(f"Saved: {figure_name}")


def main() -> None:
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
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [args.format] if args.format else None

    df, _, _ = load_population_data(args.data_dir, min_firing_rate=args.min_firing_rate)

    plot_robustness_scatter(
        df,
        args.output_dir,
        "fig04g_robustness_scatter",
        formats,
        args.dpi,
    )


if __name__ == "__main__":
    main()
