#!/usr/bin/env python
"""Generate Figure 1: Simulated non-local detection example.

This script generates a simple simulation demonstrating the non-local detection
algorithm on synthetic data. It creates diagnostic plots showing:
- Simulated position, speed, and spike raster
- Detector initial conditions and state transitions
- Full non-local model visualization

Usage:
    python scripts/generate_figure01.py
    python scripts/generate_figure01.py --output-dir paper/figures --format pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization import plot_non_local_model


def generate_simulation_panels(n_neurons: int = 30) -> tuple:
    """Generate simulated data and create diagnostic panels.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate (default: 30)

    Returns
    -------
    tuple
        (fig, detector, results) - matplotlib figure and model objects
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

    # Create diagnostic panels
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True, constrained_layout=True)
    axes[0].plot(time, position, label="Position")
    axes[0].set_ylabel("Position (cm)")
    axes[0].fill_between(
        time, is_event * np.max(position), alpha=0.2, label="Event Periods"
    )
    axes[0].legend()

    axes[1].plot(time, speed, label="Speed", color="orange")
    axes[1].set_ylabel("Speed (cm/s)")
    axes[1].fill_between(
        time, is_event * np.max(speed), alpha=0.2, label="Event Periods"
    )

    axes[2].eventplot(spike_times, lineoffsets=1, linelengths=0.5, color="black")
    axes[2].set_ylabel("Spike Times")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([])
    axes[2].set_title("Spike Raster Plot")
    axes[2].fill_between(
        time, is_event * len(spike_times), alpha=0.2, label="Event Periods"
    )
    axes[2].set_xlim(0, time[-1])

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

    return fig, detector, results, time, position, spike_times, speed


def main() -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures (default: paper/figures)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format (default: pdf)",
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

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate simulation
    print(f"Generating simulation with {args.n_neurons} neurons...")
    fig, detector, results, time, position, spike_times, speed = (
        generate_simulation_panels(n_neurons=args.n_neurons)
    )

    # Save diagnostic panels
    output_file = args.output_dir / f"figure01_simulation.{args.format}"
    fig.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # Generate detector diagnostic plots
    print("Generating detector diagnostics...")
    fig2 = detector.plot_initial_conditions()
    output_file2 = args.output_dir / f"figure01_initial_conditions.{args.format}"
    fig2.savefig(output_file2, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file2}")

    fig3 = detector.plot_discrete_state_transition()
    output_file3 = args.output_dir / f"figure01_discrete_transition.{args.format}"
    fig3.savefig(output_file3, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file3}")

    fig4 = detector.plot_continuous_state_transition()
    output_file4 = args.output_dir / f"figure01_continuous_transition.{args.format}"
    fig4.savefig(output_file4, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file4}")

    # Generate full model visualization
    print("Generating full model visualization...")
    fig5 = plot_non_local_model(
        time,
        position,
        spike_times,
        speed,
        detector,
        results,
        figsize=(20, 10),
    )
    output_file5 = args.output_dir / f"figure01_full_model.{args.format}"
    fig5.savefig(output_file5, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file5}")

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")

    print("Figure 1 generation complete!")


if __name__ == "__main__":
    main()
