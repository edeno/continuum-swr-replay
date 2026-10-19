"""Compute decoding quality metrics across all sessions.

Loads pre-computed data_over_time pickle files, computes quality metrics,
saves a summary CSV, and generates diagnostic plots.

Usage:
    uv run python scripts/qa/run_quality_metrics.py
    uv run python scripts/qa/run_quality_metrics.py --output quality_summary.csv
    uv run python scripts/qa/run_quality_metrics.py --animal chimi
    uv run python scripts/qa/run_quality_metrics.py --no_plots
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.hippocampus.hpc_raw_stats import (
    RAW_STAT_COLUMNS,
    TRANSITION_MATRIX_COLUMNS,
    apply_raw_stats_flags,
)
from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    compute_calibration_curve,
    compute_error_by_track_segment,
    compute_non_local_event_context,
    compute_session_quality_summary,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    read_dot_with_swr_source,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_decode_files
from continuum_swr_replay.decode_files import data_over_time_suffix
from continuum_swr_replay.parameters import (
    FIGURE_DIR,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

RAW_STATS_JOIN_COLS = list(SESSION_COLUMNS_DAY)
RAW_STATS_DATA_COLS = list(RAW_STAT_COLUMNS) + list(TRANSITION_MATRIX_COLUMNS)


def maybe_merge_raw_stats(
    summary: pd.DataFrame,
    raw_stats_csv: Path,
) -> pd.DataFrame:
    """Merge an optional hpc_raw_stats.csv into the quality summary.

    Drops any raw-stat columns already present on ``summary`` (so repeated
    runs don't accumulate ``_x``/``_y`` suffixes), then left-merges on
    animal/day/epoch. If the CSV does not exist or is empty, returns
    ``summary`` unchanged.
    """
    if not raw_stats_csv.exists():
        logger.info("No raw-stats CSV at %s; skipping merge", raw_stats_csv)
        return summary
    raw = pd.read_csv(raw_stats_csv)
    if raw.empty:
        return summary
    raw["day"] = raw["day"].astype(str)
    summary = summary.copy()
    summary["day"] = summary["day"].astype(str)
    drop_cols = [c for c in RAW_STATS_DATA_COLS if c in summary.columns]
    if drop_cols:
        summary = summary.drop(columns=drop_cols)
    keep = RAW_STATS_JOIN_COLS + [c for c in RAW_STATS_DATA_COLS if c in raw.columns]
    merged = summary.merge(
        raw[keep],
        on=RAW_STATS_JOIN_COLS,
        how="left",
        validate="many_to_one",
    )
    logger.info(
        "Merged raw stats for %d/%d sessions from %s",
        merged["rate_hz_per_tetrode"].notna().sum(),
        len(merged),
        raw_stats_csv.name,
    )
    return apply_raw_stats_flags(merged)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_all_sessions(
    animal: str | None = None,
    threshold: int = 80,
) -> pd.DataFrame:
    """Load and concatenate the analysed epochs' data_over_time pickle files.

    The epoch register's analysed epochs (``analysed_decode_files``), never a
    glob, so a pending epoch's decode in the folder is not read.
    """
    suffix = data_over_time_suffix(threshold)
    paths = analysed_decode_files(PROCESSED_DATA_DIR, suffix)

    if not paths:
        raise FileNotFoundError(
            f"No analysed epoch's *{suffix} found in {PROCESSED_DATA_DIR}"
        )

    if animal is not None:
        paths = [p for p in paths if p.stem.startswith(animal)]
        if not paths:
            raise ValueError(f"No files found for animal '{animal}'")

    logger.info("Loading %d session files...", len(paths))
    data = pd.concat(
        read_dot_with_swr_source(p, non_local_threshold=threshold / 100.0)
        for p in paths
    )

    n_sessions = data.groupby(["animal", "day", "epoch"]).ngroups
    logger.info("Loaded %d sessions (%d timebins)", n_sessions, len(data))
    return data


def print_summary(summary: pd.DataFrame) -> None:
    """Print quality summary to console."""
    n = len(summary)
    n_flagged = (summary["n_flags"] > 0).sum()

    print(f"\n{'=' * 60}")
    print(f"Decoding Quality Summary: {n} sessions")
    print(f"{'=' * 60}")
    print(
        f"  Median error (cm):    {summary['median_error_cm'].median():.1f} (median across sessions)"
    )
    print(
        f"  Running fraction:     {summary['running_fraction'].median():.2f} (median across sessions)"
    )
    print(
        f"  Local fraction:       {summary['local_fraction'].median():.2f} (median during movement)"
    )
    print(
        "  Immobile local/nl:    "
        f"{summary['immobile_local_fraction'].median():.2f} / "
        f"{summary['immobile_nonlocal_fraction'].median():.2f} (median during immobility)"
    )
    if "p90_running_local_error_cm" in summary.columns:
        print(
            "  Running err p90/p99:  "
            f"{summary['p90_running_local_error_cm'].median():.1f} / "
            f"{summary['p99_running_local_error_cm'].median():.1f} cm "
            "(median of per-session tail)"
        )
    if "frac_running_bouts_nonlocal_dominant" in summary.columns:
        print(
            "  Frac bouts NL-dom:    "
            f"{summary['frac_running_bouts_nonlocal_dominant'].median():.3f} "
            "(median across sessions)"
        )
    if "rate_hz_per_tetrode" in summary.columns:
        print(
            "  Hz/tetrode (HPC):     "
            f"{summary['rate_hz_per_tetrode'].median():.1f} "
            "(median across sessions)"
        )
    if "wf_per_tet_abs_median_max" in summary.columns:
        print(
            "  Worst-tet amp med:    "
            f"{summary['wf_per_tet_abs_median_max'].median():.0f} "
            "(median across sessions)"
        )
    if "T_local_local" in summary.columns:
        diag_cols = ["T_local_local", "T_nospike_nospike", "T_nlc_nlc", "T_nlf_nlf"]
        present = [c for c in diag_cols if c in summary.columns]
        meds = [f"{summary[c].median():.3f}" for c in present]
        print(
            "  Self-transitions:     "
            + " / ".join(meds)
            + "  (Local / No-Spike / NL-C / NL-F)"
        )
    print(f"  No-spike/MUA rho:     {summary['nospike_mua_rho'].median():.3f} (median)")
    print(
        f"  Calibration slope:    {summary['calibration_slope'].median():.1f} (median)"
    )
    print(f"  Sessions with flags:  {n_flagged}/{n}")

    flag_cols = [c for c in summary.columns if c.startswith("flag_")]
    for col in flag_cols:
        count = summary[col].sum()
        if count > 0:
            print(f"    {col}: {count} sessions")

    if n_flagged > 0:
        print("\nFlagged sessions:")
        flagged = summary.loc[
            summary["n_flags"] > 0, ["animal", "day", "epoch", "n_flags"] + flag_cols
        ]
        print(flagged.to_string(index=False))
    print()


def plot_diagnostics(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate diagnostic plots."""
    figure_utilities.set_figure_defaults(context="paper")
    output_path = FIGURE_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _plot_error_distribution(summary, output_path)
        _plot_state_fractions(summary, output_path)
        _plot_nonlocal_movement(summary, output_path)
        _plot_calibration_curves(data, output_path)
        _plot_event_context(data, output_path)
        _plot_nospike_correlation(summary, output_path)
        _plot_segment_error(data, output_path)


def _plot_error_distribution(summary: pd.DataFrame, output_path) -> None:
    """A: Distribution of median decoding error across sessions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(
        summary["median_error_cm"].dropna(), bins=30, edgecolor="black", alpha=0.7
    )
    axes[0].set_xlabel("Median decoding error (cm)")
    axes[0].set_ylabel("Number of sessions")
    axes[0].set_title("Error distribution")

    animals = sorted(summary["animal"].unique())
    box_data = [
        summary.loc[summary["animal"] == a, "median_error_cm"].dropna() for a in animals
    ]
    axes[1].boxplot(box_data, labels=animals)
    axes[1].set_xlabel("Animal")
    axes[1].set_ylabel("Median decoding error (cm)")
    axes[1].set_title("Error by animal")

    fig.tight_layout()
    fig.savefig(output_path / "error_distribution.pdf")
    plt.close(fig)


def _plot_state_fractions(summary: pd.DataFrame, output_path) -> None:
    """B: State fractions during movement."""
    fraction_cols = [
        "local_fraction",
        "nonlocal_continuous_fraction",
        "nonlocal_fragmented_fraction",
        "nospike_fraction",
    ]
    labels = ["Local", "Non-Local Cont.", "Non-Local Frag.", "No-Spike"]

    valid = summary.dropna(subset=fraction_cols)
    if valid.empty:
        return

    sorted_idx = valid["local_fraction"].sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))

    bottom = np.zeros(len(valid))
    for col, lbl in zip(fraction_cols, labels, strict=True):
        values = valid.loc[sorted_idx, col].to_numpy()
        ax.bar(range(len(valid)), values, bottom=bottom, label=lbl, width=1.0)
        bottom += values

    ax.set_xlabel("Session (sorted by local fraction)")
    ax.set_ylabel("Fraction of movement time")
    ax.set_title("State fractions during movement")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, len(valid) - 0.5)

    fig.tight_layout()
    fig.savefig(output_path / "state_fractions.pdf")
    plt.close(fig)


def _plot_nonlocal_movement(summary: pd.DataFrame, output_path) -> None:
    """C: Non-local during movement rate and duration."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rate = summary["nonlocal_movement_rate_per_min"].dropna()
    axes[0].hist(rate, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Non-local events per minute (during movement)")
    axes[0].set_ylabel("Number of sessions")
    axes[0].set_title("Non-local rate during movement")

    duration = summary["nonlocal_movement_median_duration_ms"].dropna()
    duration = duration[duration > 0]
    if len(duration) > 0:
        axes[1].hist(duration, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Median duration (ms)")
    axes[1].set_ylabel("Number of sessions")
    axes[1].set_title("Non-local duration during movement")

    fig.tight_layout()
    fig.savefig(output_path / "nonlocal_during_movement.pdf")
    plt.close(fig)


def _plot_calibration_curves(data: pd.DataFrame, output_path) -> None:
    """D: Calibration curves by animal."""
    calib = compute_calibration_curve(data)
    animals = sorted(calib["animal"].unique()) if "animal" in calib.columns else []
    n_animals = len(animals)
    # ``compute_calibration_curve`` legitimately returns an empty
    # frame on degenerate inputs (all-NaN ``Local_prob``, no
    # decoder bins survive the filter, etc.). ``plt.subplots(1, 0)``
    # raises ``ValueError: Number of columns must be a positive
    # integer, not 0``, so log + skip the panel rather than crash
    # the whole quality-metrics pipeline.
    if n_animals == 0:
        print(
            "[run_quality_metrics] no calibration data available; "
            "skipping calibration_curves.pdf"
        )
        return

    fig, axes = plt.subplots(1, n_animals, figsize=(4 * n_animals, 4), squeeze=False)

    for ax, animal in zip(axes[0], animals, strict=True):
        animal_data = calib.loc[calib["animal"] == animal]
        for (_, _day, _epoch), group in animal_data.groupby(
            ["animal", "day", "epoch"], sort=False
        ):
            sorted_g = group.sort_values("prob_bin_center")
            ax.plot(
                sorted_g["prob_bin_center"],
                sorted_g["median_error_cm"],
                alpha=0.3,
                color="gray",
                linewidth=0.5,
            )

        # Animal mean
        mean_calib = animal_data.groupby("prob_bin_center")["median_error_cm"].mean()
        ax.plot(mean_calib.index, mean_calib.values, color="black", linewidth=2)
        ax.set_xlabel("Local probability")
        ax.set_ylabel("Median error (cm)")
        ax.set_title(animal)

    fig.suptitle("Calibration curves (error vs. local probability)")
    fig.tight_layout()
    fig.savefig(output_path / "calibration_curves.pdf")
    plt.close(fig)


def _plot_event_context(data: pd.DataFrame, output_path) -> None:
    """E: Non-local event context breakdown."""
    context = compute_non_local_event_context(data)
    if context.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sorted_ctx = context.sort_values("frac_swr", ascending=False).reset_index(drop=True)
    bottom = np.zeros(len(sorted_ctx))

    for col, lbl, color in [
        ("frac_swr", "SWR", "#1f77b4"),
        ("frac_theta", "Theta", "#2ca02c"),
        ("frac_neither", "Neither", "#d62728"),
    ]:
        values = sorted_ctx[col].to_numpy()
        ax.bar(
            range(len(sorted_ctx)),
            values,
            bottom=bottom,
            label=lbl,
            color=color,
            width=1.0,
        )
        bottom += values

    ax.set_xlabel("Session (sorted by SWR fraction)")
    ax.set_ylabel("Fraction of non-local events")
    ax.set_title("Non-local event context")
    ax.legend()
    ax.set_xlim(-0.5, len(sorted_ctx) - 0.5)

    fig.tight_layout()
    fig.savefig(output_path / "event_context.pdf")
    plt.close(fig)


def _plot_nospike_correlation(summary: pd.DataFrame, output_path) -> None:
    """F: No-spike/MUA correlation distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    rho = summary["nospike_mua_rho"].dropna()
    ax.hist(rho, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Spearman rho (No-Spike prob vs. MUA z-score)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("No-spike / firing rate correspondence")

    fig.tight_layout()
    fig.savefig(output_path / "nospike_mua_correlation.pdf")
    plt.close(fig)


def _plot_segment_error(data: pd.DataFrame, output_path) -> None:
    """G: Error by track segment."""
    seg = compute_error_by_track_segment(data)
    if seg.empty:
        return

    seg["session"] = seg["animal"] + "_" + seg["day"].astype(str) + "_" + seg["epoch"]

    pivot = seg.pivot_table(
        index="session",
        columns="track_segment_id",
        values="median_error_cm",
        aggfunc="first",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, max(4, len(pivot) * 0.15)),
        layout="constrained",
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=4)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Track segment ID")
    ax.set_ylabel("Session")
    ax.set_title("Median error by track segment (cm)")
    fig.colorbar(im, ax=ax, label="Error (cm)")

    fig.savefig(output_path / "segment_error_heatmap.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute decoding quality metrics")
    parser.add_argument(
        "--output",
        type=str,
        default="quality_summary.csv",
        help="Output CSV filename (saved in PROCESSED_DATA_DIR)",
    )
    parser.add_argument("--animal", type=str, default=None, help="Filter to one animal")
    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Non-local probability threshold (matches pickle filename, default 80)",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip diagnostic plots")
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Skip metric computation, load existing CSV and regenerate plots",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="quality_metrics",
        help="Subdirectory under FIGURE_DIR for plots",
    )
    parser.add_argument(
        "--raw_stats_csv",
        type=str,
        default="hpc_raw_stats.csv",
        help=(
            "Optional raw-data stats CSV (from scripts/qa/compute_hpc_raw_stats.py) "
            "to merge into the summary. If the file does not exist, the merge "
            "is silently skipped."
        ),
    )
    args = parser.parse_args()

    data = load_all_sessions(animal=args.animal, threshold=args.threshold)
    non_local_threshold = args.threshold / 100.0

    output_path = PROCESSED_DATA_DIR / args.output

    if args.plots_only:
        logger.info("Loading existing summary from %s", output_path)
        summary = pd.read_csv(output_path)
    else:
        logger.info("Computing quality metrics...")
        summary = compute_session_quality_summary(
            data,
            non_local_threshold=non_local_threshold,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        summary = maybe_merge_raw_stats(
            summary,
            PROCESSED_DATA_DIR / args.raw_stats_csv,
        )
        summary.to_csv(output_path, index=False)
        logger.info("Saved summary to %s", output_path)

    print_summary(summary)

    if not args.no_plots:
        logger.info("Generating diagnostic plots...")
        plot_diagnostics(data, summary, output_dir=args.figure_dir)
        logger.info("Saved plots to %s", FIGURE_DIR / args.figure_dir)


if __name__ == "__main__":
    main()
