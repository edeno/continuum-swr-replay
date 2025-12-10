#!/usr/bin/env python
"""Generate Figure 1: Simulated non-local detection example.

This script generates a simple simulation demonstrating the non-local detection
algorithm on synthetic data. It creates diagnostic plots showing:
- Simulated position, speed, and spike raster
- Detector initial conditions and state transitions
- Full non-local model visualization

Usage:
    python scripts/generate_figure01.py
    python scripts/generate_figure01.py --output-dir paper/figures/figure01
    python scripts/generate_figure01.py --format pdf  # PDF only
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization import plot_non_local_model

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.figure_utilities import (
    PANEL_SINGLE_TALL,
    PANEL_WIDE,
    TWO_COLUMN,
    get_figure_size,
    save_figure,
)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


@dataclass
class SimulationOutput:
    """Container for simulation results and visualizations.

    Attributes
    ----------
    diagnostic_fig : plt.Figure
        Figure with position, speed, and spike raster panels
    detector : NonLocalSortedSpikesDetector
        Fitted detector model
    results : dict
        Decoding results from detector.predict()
    time : np.ndarray
        Time points (seconds)
    position : np.ndarray
        Position values (cm)
    spike_times : list
        List of spike time arrays per neuron
    speed : np.ndarray
        Speed values (cm/s)
    is_event : np.ndarray
        Boolean mask for event periods
    """

    diagnostic_fig: plt.Figure
    detector: NonLocalSortedSpikesDetector
    results: dict
    time: np.ndarray
    position: np.ndarray
    spike_times: list
    speed: np.ndarray
    is_event: np.ndarray


def add_event_shading(
    ax: Axes, time: np.ndarray, is_event: np.ndarray, max_height: float
) -> None:
    """Add shaded regions for event periods to an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to add shading to
    time : np.ndarray
        Time points
    is_event : np.ndarray
        Boolean mask for event periods
    max_height : float
        Maximum height of shading region
    """
    ax.fill_between(time, is_event * max_height, alpha=0.2, label="Event Periods")


def generate_simulation_panels(n_neurons: int = 30) -> SimulationOutput:
    """Generate simulated data and create diagnostic panels.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate (default: 30)

    Returns
    -------
    SimulationOutput
        Container with all simulation data and results
    """
    # Generate simulated data
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=n_neurons)

    # Create diagnostic panels (3 stacked panels, single column width)
    fig, axes = plt.subplots(
        3, 1, figsize=get_figure_size(3, 1), sharex=True, constrained_layout=True
    )

    # Position panel
    axes[0].plot(time, position, label="Position")
    axes[0].set_ylabel("Position (cm)")
    add_event_shading(axes[0], time, is_event, np.max(position))
    axes[0].legend()

    # Speed panel
    axes[1].plot(time, speed, label="Speed", color="orange")
    axes[1].set_ylabel("Speed (cm/s)")
    add_event_shading(axes[1], time, is_event, np.max(speed))

    # Spike raster panel
    axes[2].eventplot(spike_times, lineoffsets=1, linelengths=0.5, color="black")
    axes[2].set_ylabel("Spike Times")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([])
    axes[2].set_title("Spike Raster Plot")
    add_event_shading(axes[2], time, is_event, len(spike_times))
    axes[2].set_xlim(0, time[-1])

    # Apply consistent styling
    for ax in axes:
        sns.despine(ax=ax)

    # Fit detector
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    # Run decoding
    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=True,
    )

    return SimulationOutput(
        diagnostic_fig=fig,
        detector=detector,
        results=results,
        time=time,
        position=position,
        spike_times=spike_times,
        speed=speed,
        is_event=is_event,
    )


def main() -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure01"),
        help="Output directory for figures (default: paper/figures/figure01)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default=None,
        help="Output format (default: both pdf and png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (default: 300)",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=30,
        help="Number of neurons to simulate (default: 30)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (default: save only)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.n_neurons < 1:
        parser.error("--n-neurons must be positive")

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate simulation
    print(f"Generating simulation with {args.n_neurons} neurons...")
    sim = generate_simulation_panels(n_neurons=args.n_neurons)

    # Save diagnostic panels
    plt.figure(sim.diagnostic_fig.number)
    save_figure(
        "figure01_simulation",
        base_dir=args.output_dir,
        dpi=args.dpi,
        formats=[args.format] if args.format else None,
    )

    # Generate and save detector diagnostic plots
    print("Generating detector diagnostics...")

    sim.detector.plot_initial_conditions()
    save_figure(
        "figure01_initial_conditions",
        base_dir=args.output_dir,
        dpi=args.dpi,
        formats=[args.format] if args.format else None,
    )

    sim.detector.plot_discrete_state_transition()
    save_figure(
        "figure01_discrete_transition",
        base_dir=args.output_dir,
        dpi=args.dpi,
        formats=[args.format] if args.format else None,
    )

    sim.detector.plot_continuous_state_transition()
    save_figure(
        "figure01_continuous_transition",
        base_dir=args.output_dir,
        dpi=args.dpi,
        formats=[args.format] if args.format else None,
    )

    # Generate full model visualization
    print("Generating full model visualization...")
    plot_non_local_model(
        sim.time,
        sim.position,
        sim.spike_times,
        sim.speed,
        sim.detector,
        sim.results,
        figsize=(20, 10),
    )
    save_figure(
        "figure01_full_model",
        base_dir=args.output_dir,
        dpi=args.dpi,
        formats=[args.format] if args.format else None,
    )

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")

    print("Figure 1 generation complete!")


if __name__ == "__main__":
    main()
