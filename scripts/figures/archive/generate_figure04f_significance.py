#!/usr/bin/env python
"""Generate Figure 4f: Fraction of significantly modulated neurons and AIC model selection.

Bar charts showing:
1. Fraction of neurons significantly modulated by each event type (FDR q<0.05)
2. AIC model selection — which model best explains each neuron's immobility firing

Usage:
    python scripts/generate_figure04f_significance.py
"""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.figure_utilities import (
    FIGURE_TWO_PANEL_HORIZONTAL,
    save_figure,
)
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.visualization.palettes import COLORS

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_significance_bars(
    df,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot fraction of neurons significant for each LR test."""
    tests = {
        "SWR\nvs. constant": "SWR vs. constant: fdr_q",
        "Replay\nvs. constant": "non_local vs. constant: fdr_q",
        "non-SWR replay\nvs. constant": "non_local_no_SWR vs. constant: fdr_q",
        "Replay beyond\nSWR": "SWR + non_local_no_SWR vs. SWR: fdr_q",
    }

    n_total = len(df)
    fig, ax = plt.subplots(figsize=FIGURE_TWO_PANEL_HORIZONTAL)

    labels = list(tests.keys())
    fracs = []
    counts = []
    for col in tests.values():
        sig = (df[col] < 0.05).sum()
        fracs.append(100 * sig / n_total)
        counts.append(sig)

    colors_list = [
        COLORS["SWR"],
        COLORS["Non-Local"],
        COLORS["Non-Local"],
        COLORS["Non-Local"],
    ]

    bars = ax.bar(labels, fracs, color=colors_list, alpha=0.7, edgecolor="none")

    for bar, count, frac in zip(bars, counts, fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            frac + 1.5,
            f"{count}\n({frac:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.axhline(5, color="gray", linewidth=0.5, linestyle="--", label="5% chance level")
    ax.set_ylabel("% of neurons (FDR q < 0.05)")
    ax.set_ylim(0, max(fracs) * 1.3)
    ax.set_title(f"Significant modulation (n={n_total:,})")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=3, direction="out")

    save_figure(plt.gcf(), figure_name, base_dir=output_dir, dpi=dpi, formats=formats)
    plt.close(fig)
    print(f"Saved: {figure_name}")


def plot_aic_model_selection(
    df,
    output_dir: Path,
    figure_name: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Plot AIC model selection distribution."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=FIGURE_TWO_PANEL_HORIZONTAL, constrained_layout=True
    )

    n_total = len(df)

    # SWR-controlled decomposition
    model_order_swr = ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"]
    model_labels_swr = ["Constant", "SWR", "Replay", "SWR +\nnon-SWR replay"]
    model_colors_swr = ["gray", COLORS["SWR"], COLORS["Non-Local"], "#e6ab02"]

    counts_swr = df["min_aic_model"].value_counts()
    vals_swr = [100 * counts_swr.get(m, 0) / n_total for m in model_order_swr]

    bars1 = ax1.bar(
        model_labels_swr, vals_swr, color=model_colors_swr, alpha=0.7, edgecolor="none"
    )
    for bar, val in zip(bars1, vals_swr):
        if val > 2:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.0f}%",
                ha="center",
                fontsize=7,
            )
    ax1.set_ylabel("% of neurons")
    ax1.set_title("Best model (SWR-controlled)")
    sns.despine(ax=ax1)

    # Replay-split decomposition
    model_order_split = ["constant", "non_local", "non_local_SWR + non_local_no_SWR"]
    model_labels_split = [
        "Constant",
        "Replay\n(unified)",
        "SWR-replay +\nnon-SWR replay",
    ]
    model_colors_split = ["gray", COLORS["Non-Local"], "#e6ab02"]

    counts_split = df["min_aic_model_replay_split"].value_counts()
    vals_split = [100 * counts_split.get(m, 0) / n_total for m in model_order_split]

    bars2 = ax2.bar(
        model_labels_split,
        vals_split,
        color=model_colors_split,
        alpha=0.7,
        edgecolor="none",
    )
    for bar, val in zip(bars2, vals_split):
        if val > 2:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.0f}%",
                ha="center",
                fontsize=7,
            )
    ax2.set_ylabel("% of neurons")
    ax2.set_title("Best model (replay-split)")
    sns.despine(ax=ax2)

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", which="both", length=3, direction="out")

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

    df, _, _ = load_population_data(args.data_dir, min_firing_rate=args.min_firing_rate)

    plot_significance_bars(
        df,
        args.output_dir,
        "fig04f_significance_bars",
        formats,
        args.dpi,
    )
    plot_aic_model_selection(
        df,
        args.output_dir,
        "fig04f_aic_model_selection",
        formats,
        args.dpi,
    )


if __name__ == "__main__":
    main()
