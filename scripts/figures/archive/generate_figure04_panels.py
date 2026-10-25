#!/usr/bin/env python3
"""Generate Figure 4: mPFC neuron modulation during non-local events.

This script analyzes modulation of medial prefrontal cortex (mPFC) neurons around
different event types using Generalized Linear Models (GLM). It generates:

1. GLM modulation plots showing firing rate changes relative to events
2. Spike rasters aligned to event times
3. Peri-stimulus time histograms (PSTH) of neural activity
4. AIC comparisons across GLM models

Event types analyzed:
- SWR: Sharp-wave ripples
- Non-local events (with/without SWR overlap)
- Non-local events excluding SWR overlap

The script fits GLM models with spline basis functions to capture temporal dynamics
of neural responses during immobility periods.

Usage
-----
# Analyze default session and neuron
$ python generate_figure04.py

# Analyze specific session and neuron
$ python generate_figure04.py --nwb-file-name wilbur20210329_.nwb \\
    --epoch-name 02_r1 --neuron-ind 27

# Save to custom directory
$ python generate_figure04.py --output-dir paper/figures/figure04
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.analysis.hippocampus import load_results
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    get_confident_times,
    get_overlap_times,
)
from continuum_swr_replay.analysis.pfc import (
    fit_immobility_models,
    make_design_matrices,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.figure_utilities import (
    ONE_COLUMN,
    get_figure_size,
    save_figure,
)
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.visualization.glm_plots import (
    plot_AICs,
    plot_hist_relative_to_event,
    plot_raster_relative_to_event,
)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


def plot_glm_results(
    lag_time: np.ndarray,
    spline_basis: np.ndarray,
    neuron_spike_times: np.ndarray,
    glm_results: dict,
    event_times: dict,
    model_list: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot GLM modulation results for multiple models.

    Creates a multi-panel figure showing GLM modulation coefficients, spike rasters,
    and firing rate histograms for different event types.

    Parameters
    ----------
    lag_time : np.ndarray
        Time lags for GLM in seconds
    spline_basis : np.ndarray
        Spline basis functions for GLM coefficients
    neuron_spike_times : np.ndarray
        Spike times for the neuron being analyzed
    glm_results : dict
        Dictionary of fitted GLM results keyed by model name
    event_times : dict
        Dictionary of event DataFrames with start_time column
    model_list : list[str] | None
        List of model names to plot. Default: ["SWR", "non_local_SWR",
        "non_local_no_SWR", "non_local"]

    Returns
    -------
    fig : Figure
        Matplotlib Figure object
    axes : np.ndarray
        Array of axes (3 rows × len(model_list) columns)
    """
    if model_list is None:
        model_list = ["SWR", "non_local_SWR", "non_local_no_SWR", "non_local"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(model_list),
        figsize=get_figure_size(3, len(model_list), base_width=ONE_COLUMN * len(model_list) * 0.7),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    if len(model_list) == 1:
        axes = np.expand_dims(axes, 1)

    modulation_bounds = np.log(0.5), np.log(2.0)
    lag_bounds = lag_time[0], lag_time[-1]

    models = [glm_results[event_name] for event_name in model_list]

    # Model name formatting
    model_list_to_mode_names = {
        "SWR": "SWR",
        "non_local_SWR": "Non-Local,\nSWR",
        "non_local_no_SWR": "Non-Local,\nNo SWR",
        "non_local": "Non-Local",
    }
    model_names = [model_list_to_mode_names[model_name] for model_name in model_list]

    event_start_times = [
        event_times[model_name].start_time
        for model_name in model_list
    ]

    # Row 1: GLM modulation coefficients
    for ax, model, model_name in zip(axes[0], models, model_names, strict=True):
        modulation = model.params[1:] @ spline_basis.T
        ax.plot(lag_time, modulation)
        ax.fill_between(
            lag_time,
            (model.params[1:] + model.bse[1:]) @ spline_basis.T,
            (model.params[1:] - model.bse[1:]) @ spline_basis.T,
            alpha=0.25,
        )
        ax.axhline(0, color="black", linestyle="--")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(model_name)
        ax.set_ylim(modulation_bounds)
        ax.set_xticks((lag_bounds[0], 0, lag_bounds[1]))
        ax.set_yticks((modulation_bounds[0], 0, modulation_bounds[1]))
        ax.set_ylabel("Mod.")
        ax.set_yticklabels(
            (np.exp(modulation_bounds[0]), 1, np.exp(modulation_bounds[1]))
        )

    # Row 2: Spike rasters
    for ax, event_time in zip(axes[1], event_start_times, strict=True):
        plot_raster_relative_to_event(neuron_spike_times, event_time, lag_bounds, ax)
        ax.set_ylabel("Event")

    # Row 3: PSTHs
    for ax, event_time in zip(axes[2], event_start_times, strict=True):
        plot_hist_relative_to_event(neuron_spike_times, event_time, lag_bounds, ax)
        ax.set_ylabel("Rate\n[sp/s]")

    # Apply consistent styling
    for ax in axes.flatten():
        sns.despine(ax=ax)

    fig.supxlabel("Time from\nevent start (s)")

    return fig, axes


def main() -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nwb-file-name",
        type=str,
        default="wilbur20210329_.nwb",
        help="NWB file name to analyze (default: wilbur20210329_.nwb)",
    )
    parser.add_argument(
        "--epoch-name",
        type=str,
        default="02_r1",
        help="Epoch name to analyze (default: 02_r1)",
    )
    parser.add_argument(
        "--neuron-ind",
        type=int,
        default=27,
        help="mPFC neuron index to analyze (default: 27)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure04"),
        help="Directory to save figures (default: paper/figures/figure04)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default=None,
        help="Figure format (default: both pdf and png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster formats (default: 300)"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (default: save only)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Load data and results
    print(f"Loading data for {args.nwb_file_name}, {args.epoch_name}...")
    results, _, _ = load_results(
        args.nwb_file_name, args.epoch_name, decoding_brain_area="HPC"
    )
    data = load_data(args.nwb_file_name, args.epoch_name)

    # Compute non-local probability
    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")

    # Detect confident non-local events
    print("Detecting non-local events...")
    non_local_times = get_confident_times(
        non_local_prob,
        data["position_info"].head_speed.to_numpy(),
        data["position_info"].index.to_numpy(),
        DEFAULT_NON_LOCAL_THRESHOLD,
    )
    non_local_times = non_local_times[
        non_local_times.max_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ]

    # Separate non-local events by SWR overlap
    non_local_SWR_times, non_local_no_SWR_times = get_overlap_times(
        non_local_times, data["ripple_times"]
    )

    # Organize event times
    event_times = {
        "non_local_no_SWR": non_local_no_SWR_times,
        "non_local_SWR": non_local_SWR_times,
        "non_local": non_local_times,
        "SWR": data["ripple_times"],
    }

    # Create design matrices for GLM
    print("Creating GLM design matrices...")
    time = data["position_info"].index.to_numpy()
    is_immobile = (
        data["position_info"].head_speed <= DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )

    immobility_design_matrices, spline_basis, lag_time = make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=DEFAULT_SAMPLING_FREQUENCY_HZ,
        lags=np.arange(-100, 101, 1),
    )

    # Get neuron spike data
    print(f"Analyzing mPFC neuron #{args.neuron_ind}...")
    neuron_spike_times = data["spike_times"]["mPFC"][args.neuron_ind]
    neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)

    # Fit GLM models
    print("Fitting GLM models...")
    glm_results, glm_models = fit_immobility_models(
        is_immobile, immobility_design_matrices, neuron_spike_count
    )

    # Generate plots
    print("Generating figures...")
    formats = [args.format] if args.format else None

    # Figure 1: Single model (non_local_no_SWR)
    fig1, _ = plot_glm_results(
        lag_time,
        spline_basis,
        neuron_spike_times,
        glm_results,
        event_times,
        model_list=["non_local_no_SWR"],
    )
    save_figure(plt.gcf(), "figure04_panel_a", base_dir=args.output_dir, dpi=args.dpi, formats=formats)
    print("Saved: figure04_panel_a")
    if not args.show_plots:
        plt.close(fig1)

    # Figure 2: Two models comparison
    fig2, _ = plot_glm_results(
        lag_time,
        spline_basis,
        neuron_spike_times,
        glm_results,
        event_times,
        model_list=["non_local_no_SWR", "SWR"],
    )
    save_figure(plt.gcf(), "figure04_panel_b", base_dir=args.output_dir, dpi=args.dpi, formats=formats)
    print("Saved: figure04_panel_b")
    if not args.show_plots:
        plt.close(fig2)

    # Figure 3: AIC comparison
    model_list_for_aic = [
        "constant",
        "SWR",
        "SWR + non_local_no_SWR",
        "non_local",
    ]
    fig3, _ = plot_AICs(glm_results, model_list=model_list_for_aic)
    save_figure(plt.gcf(), "figure04_panel_c", base_dir=args.output_dir, dpi=args.dpi, formats=formats)
    print("Saved: figure04_panel_c")
    if not args.show_plots:
        plt.close(fig3)

    # Optional: Load and display processed modulation data
    if (processed_data_dir() / "*_mPFC_modulation.pkl").exists():
        print("Loading processed mPFC modulation data...")
        data_path = processed_data_dir().glob("*_mPFC_modulation.pkl")
        df = pd.concat([pd.read_pickle(p) for p in data_path])
        df["animal"] = df["nwb_file_name"].str[:-13]
        print(f"Loaded modulation data: {len(df)} entries")
        print(df.head())

    if args.show_plots:
        plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
