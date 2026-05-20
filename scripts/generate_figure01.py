#!/usr/bin/env python
"""Generate Figure 1: Switching State Space Model for Non-Local Detection.

This figure explains the non-local detector model to the reader in three rows:

    Row 1 (Model Structure):
        A: State transition diagram — 4 discrete states with transition arrows
        B: Observation model schematic — two scenarios side-by-side
           ("Local wins" and "Non-Local wins"), each showing the four
           states' likelihoods under that scenario

    Row 2 (Continuous Dynamics):
        C: Self-transition kernels for all four states — Local is uniform
           (re-anchored by the observation kernel each step under multi-bin
           local), No-Spike is identity (single-bin), Continuous is a
           Gaussian random walk, Fragmented is uniform.

    Row 3 (Simulated Example — single panel D):
        Spike raster, non-local posterior, state probabilities, speed

The narrative flow is: what are the states → how do they interpret spikes →
how do their spatial dynamics differ → what does the output look like.

Usage:
    python scripts/generate_figure01.py
    python scripts/generate_figure01.py --n-neurons 40 --seed 42
    python scripts/generate_figure01.py --output-dir paper/figures/figure01
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    WONG,
    label_panels,
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.parameters import LOCAL_POSITION_STD

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colors — colorblind-friendly, consistent across figure
# ---------------------------------------------------------------------------
STATE_COLORS = {
    "Local": WONG[5],  # Blue
    "No-Spike": "#999999",  # Gray
    "Non-Local\nContinuous": WONG[1],  # Orange
    "Non-Local\nFragmented": WONG[6],  # Vermillion
}
STATE_NAMES_SHORT = ["Local", "No-Spike", "Continuous", "Fragmented"]
STATE_COLORS_SHORT = [WONG[5], "#999999", WONG[1], WONG[6]]
ANIMAL_POS_COLOR = "magenta"


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Discrete State Diagram
# ═══════════════════════════════════════════════════════════════════════════


def plot_state_diagram(ax):
    """Draw 4-state diagram with transition arrows.

    States arranged in a 2x2 grid. Self-loop probabilities annotated.
    Key cross-state arrows shown with faint lines.

    The data limits and gridspec width_ratios are tuned so Panel A
    renders at roughly the same width and height as one of Panel B's
    scenario 2x2 blocks (~730 x 820 px at the configured DPI). Aspect
    is locked equal, and the axes is anchored to the south edge of
    its slot so the bottom of Panel A's circles aligns with the
    bottom of Panel B's state cells.
    """
    # x_range / y_range chosen to match scenario-block aspect 730/820 ≈ 0.89.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.set_anchor("S")
    ax.axis("off")

    positions = {
        "Local": (0.25, 0.80),
        "No-Spike": (0.75, 0.80),
        "Non-Local\nContinuous": (0.25, 0.20),
        "Non-Local\nFragmented": (0.75, 0.20),
    }

    self_probs = {
        "Local": "0.99",
        "No-Spike": "0.98",
        "Non-Local\nContinuous": "0.98",
        "Non-Local\nFragmented": "0.98",
    }

    radius = 0.16

    for state_name, (cx, cy) in positions.items():
        circle = mpatches.Circle(
            (cx, cy),
            radius,
            facecolor=STATE_COLORS[state_name],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            zorder=10,
        )
        ax.add_patch(circle)
        ax.text(
            cx, cy, state_name,
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="white", zorder=11, linespacing=0.9,
        )

    # Self-loop arrows
    loop_offsets = {
        "Local": (-0.08, 0.12),
        "No-Spike": (0.08, 0.12),
        "Non-Local\nContinuous": (-0.08, -0.12),
        "Non-Local\nFragmented": (0.08, -0.12),
    }
    for state_name, (cx, cy) in positions.items():
        ox, oy = loop_offsets[state_name]
        arrow = mpatches.FancyArrowPatch(
            (cx + ox, cy + oy * 0.5), (cx + ox * 0.5, cy + oy),
            arrowstyle="->,head_length=3,head_width=2",
            connectionstyle="arc3,rad=0.4", mutation_scale=1,
            linewidth=0.8, color="black", alpha=0.5, zorder=5,
        )
        ax.add_patch(arrow)
        ax.text(
            cx + ox * 1.5, cy + oy * 1.5, self_probs[state_name],
            ha="center", va="center", fontsize=4.5, color="black", alpha=0.7,
        )

    # Cross-state arrows
    cross_transitions = [
        ("Local", "No-Spike"),
        ("Local", "Non-Local\nContinuous"),
        ("Local", "Non-Local\nFragmented"),
        ("No-Spike", "Non-Local\nContinuous"),
        ("No-Spike", "Non-Local\nFragmented"),
        ("Non-Local\nContinuous", "Local"),
        ("Non-Local\nContinuous", "Non-Local\nFragmented"),
        ("Non-Local\nFragmented", "Local"),
    ]
    for from_state, to_state in cross_transitions:
        x1, y1 = positions[from_state]
        x2, y2 = positions[to_state]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        shrink = radius * 1.15
        arrow = mpatches.FancyArrowPatch(
            (x1 + dx / dist * shrink, y1 + dy / dist * shrink),
            (x2 - dx / dist * shrink, y2 - dy / dist * shrink),
            arrowstyle="->,head_length=3,head_width=1.5", mutation_scale=1,
            linewidth=0.5, color="black", alpha=0.25, zorder=4,
        )
        ax.add_patch(arrow)


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Observation Model Schematic
# ═══════════════════════════════════════════════════════════════════════════


def _draw_state_cells(fig, gs_inner, x, likelihood, animal_pos):
    """Draw the 4-state likelihood cells for one scenario.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    gs_inner : SubplotSpec
        A 2x2 sub-grid slot. Cells are laid out row-major:
        (Local, No-Spike) on top; (Continuous, Fragmented) on bottom.
    x : np.ndarray, shape (n_pts,)
        Position grid on [0, 1] for the schematic.
    likelihood : np.ndarray, shape (n_pts,)
        The place-field curve λ(x).
    animal_pos : float
        The animal's position in [0, 1]. Controls where the Local
        kernel sits and where the magenta dashed line is drawn.

    Returns
    -------
    list of matplotlib.axes.Axes
        The four sub-axes (Local, No-Spike, Continuous, Fragmented).
    """
    configs = [
        ("Local", STATE_COLORS_SHORT[0]),
        ("No-Spike", STATE_COLORS_SHORT[1]),
        ("Continuous\n Non-Local", STATE_COLORS_SHORT[2]),
        ("Fragmented\nNon-Local", STATE_COLORS_SHORT[3]),
    ]
    sub_axes = []

    # Pick annotation placement so the arrow doesn't cross the formula text.
    # The text is anchored on its edge nearest the bump (ha=left/right) so it
    # only extends away from the bump; the arrow tail then sits at that edge
    # and the shaft sweeps across empty space toward xy.
    if animal_pos >= 0.5:
        annot_xtext = animal_pos - 0.25
        annot_ha = "right"
    else:
        annot_xtext = animal_pos + 0.25
        annot_ha = "left"

    for idx, (name, color) in enumerate(configs):
        row, col = idx // 2, idx % 2
        sub_ax = fig.add_subplot(gs_inner[row, col])
        sub_axes.append(sub_ax)

        if name == "Local":
            sigma_local = 0.05  # schematic kernel width on the [0,1] x-axis
            kernel = np.exp(-0.5 * ((x - animal_pos) / sigma_local) ** 2)
            local_like = likelihood * kernel
            sub_ax.plot(x, likelihood, color="#aaaaaa", linewidth=0.8, alpha=0.4)
            sub_ax.fill_between(x, likelihood, alpha=0.2, color="#aaaaaa")
            sub_ax.plot(x, local_like, color=color, linewidth=1.0)
            sub_ax.fill_between(x, local_like, alpha=0.45, color=color)
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--")
            # Arrow tip stays visible even when the bump is tiny
            peak_y = max(float(local_like.max()), 0.10)
            sub_ax.annotate(
                r"$\lambda(x)\,\mathcal{N}(x_{\mathrm{animal}}, \sigma_L)$",
                xy=(animal_pos, peak_y),
                xytext=(annot_xtext, 0.95),
                fontsize=5, color=color, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
                ha=annot_ha,
            )

        elif name == "No-Spike":
            sub_ax.axhline(0.05, color=color, linewidth=1.2, linestyle="-")
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--")
            sub_ax.text(
                0.5, 0.12, r"$\lambda \approx 0$",
                ha="center", va="bottom", fontsize=6, color=color,
                fontweight="bold", transform=sub_ax.transAxes,
            )

        else:  # Continuous or Fragmented Non-Local
            sub_ax.plot(x, likelihood, color=color, linewidth=1.0)
            sub_ax.fill_between(x, likelihood, alpha=0.3, color=color)
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--", alpha=0.4)

        sub_ax.set_xlim(0, 1)
        sub_ax.set_ylim(-0.05, 1.2)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            sub_ax.spines[spine].set_visible(False)
        sub_ax.spines["bottom"].set_linewidth(0.5)

        sub_ax.set_title(
            name, fontsize=5.5, color=color, fontweight="bold", pad=2,
        )
        # Label the position axis on the bottom row of each scenario only;
        # all four cells share the same [0, 1] position axis.
        if row == 1:
            sub_ax.set_xlabel("position", fontsize=5, labelpad=3)

    return sub_axes


def plot_observation_models(fig, gs_slot):
    """Show how each state's spike likelihood reacts under two scenarios.

    Two scenarios are placed side-by-side, each containing the 4-state
    2x2 grid (Local / No-Spike on top; Continuous / Fragmented Non-Local
    on bottom):

    - "Local wins": x_animal coincides with the place-field peak. The
      cell fires where its tuning says it should, so the Local
      likelihood (λ(x)·N(x_animal, σ_L)) is large at the animal's
      position. The Non-Local likelihood λ(x) is also peaked there, but
      spread across all bins, so per-bin Non-Local mass is lower.
    - "Non-Local wins": x_animal is far from the place-field peak. The
      cell fires where it "shouldn't" — Local likelihood is tiny
      because λ(x_animal) is near zero. Non-Local likelihood is still
      peaked at the place-field center, far from the animal,
      consistent with replay of a different location.

    Returns all sub-axes (8 total), Local-wins block first then
    Non-Local-wins. The first is used by ``label_panels`` for the
    panel "b" label.
    """
    n_pts = 200
    x = np.linspace(0, 1, n_pts)
    likelihood_center = 0.4
    likelihood = np.exp(-0.5 * ((x - likelihood_center) / 0.10) ** 2)

    # Outer 2x2: thin title row above content row, two scenarios as columns.
    gs_outer = gs_slot.subgridspec(
        2, 2, height_ratios=[0.12, 1.0], wspace=0.20, hspace=0.10,
    )

    scenarios = [
        # (header, animal_pos)
        ("Local wins", likelihood_center),   # animal at place-field peak
        ("Non-Local wins", 0.55),            # animal ~1.5σ_λ from peak;
        #                                       Local bump shrinks to ~30% of
        #                                       Non-Local peak but stays visible
    ]

    all_axes = []
    for col, (header, animal_pos) in enumerate(scenarios):
        title_ax = fig.add_subplot(gs_outer[0, col])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.0, header,
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=title_ax.transAxes,
        )

        gs_scen = gs_outer[1, col].subgridspec(2, 2, wspace=0.10, hspace=0.55)
        all_axes.extend(
            _draw_state_cells(fig, gs_scen, x, likelihood, animal_pos)
        )

    return all_axes


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — Continuous Transition Kernels
# ═══════════════════════════════════════════════════════════════════════════


def plot_transition_kernels(fig, gs_slot):
    """Plot the self-transition kernel for each of the four states.

    With ``local_position_std`` set (multi-bin Local), the detector auto-
    upgrades any ``Discrete()`` transition involving the Local state to
    ``Uniform()`` (see ``base.initialize_continuous_state_transition``).
    Effective self-transition diagonals are:

    - Local → Local: Uniform. The Local state spans all bins; its next-step
      prior is uniform and the observation kernel re-anchors the posterior
      around the new animal position.
    - No-Spike → No-Spike: Discrete (identity). No-Spike stays single-bin;
      the auto-upgrade does not apply (neither source nor destination is
      Local).
    - Non-Local Continuous → Non-Local Continuous: Gaussian random walk
      (smooth replay trajectories).
    - Non-Local Fragmented → Non-Local Fragmented: Uniform (jump to any
      position).
    """
    gs_inner = gs_slot.subgridspec(1, 4, wspace=0.25)

    n_bins = 50
    configs = [
        {
            "name": "Local\n→ Local",
            "subtitle": "Uniform (re-anchored)",
            "color": STATE_COLORS_SHORT[0],
            "kind": "uniform",
        },
        {
            "name": "No-Spike\n→ No-Spike",
            "subtitle": "Identity",
            "color": STATE_COLORS_SHORT[1],
            "kind": "identity",
        },
        {
            "name": "Non-Local Continuous\n→ Non-Local Continuous",
            "subtitle": "Gaussian random walk",
            "color": STATE_COLORS_SHORT[2],
            "kind": "gaussian",
            "sigma": 5.0,
        },
        {
            "name": "Non-Local Fragmented\n→ Non-Local Fragmented",
            "subtitle": "Uniform (jumps)",
            "color": STATE_COLORS_SHORT[3],
            "kind": "uniform",
        },
    ]

    axes = []
    for idx, config in enumerate(configs):
        sub_ax = fig.add_subplot(gs_inner[idx])
        axes.append(sub_ax)

        if config["kind"] == "identity":
            trans = np.eye(n_bins)
        elif config["kind"] == "uniform":
            trans = np.ones((n_bins, n_bins)) / n_bins
        else:  # gaussian
            trans = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                trans[i, :] = np.exp(
                    -0.5 * ((np.arange(n_bins) - i) / config["sigma"]) ** 2
                )
                trans[i, :] /= trans[i, :].sum()

        # Shared vmax so the four kernels are visually comparable; identity's
        # diagonal saturates, Gaussian band sits mid-range, uniform reads as a
        # dim background — a clean gradient of spatial spread.
        sub_ax.imshow(
            trans, aspect="equal", cmap="magma",
            vmin=0, vmax=0.15,
            interpolation="nearest", origin="lower",
        )

        for spine in sub_ax.spines.values():
            spine.set_edgecolor(config["color"])
            spine.set_linewidth(1.5)

        sub_ax.set_xticks([])
        sub_ax.set_yticks([])

        sub_ax.set_title(
            config["name"], fontsize=5, color=config["color"],
            fontweight="bold", pad=3, linespacing=0.9,
        )
        sub_ax.text(
            0.5, -0.04, config["subtitle"],
            ha="center", va="top", fontsize=4.5,
            transform=sub_ax.transAxes, color="#555555",
            style="italic",
        )

        sub_ax.set_xlabel("position$_t$", fontsize=5, labelpad=12)
        if idx == 0:
            sub_ax.set_ylabel("position$_{t+1}$", fontsize=5, labelpad=2)

    return axes


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — Simulated Data + Decoding (single composite panel)
# ═══════════════════════════════════════════════════════════════════════════

def sort_neurons_by_place_field(spike_times, detector):
    """Sort neurons by place field peak for ordered raster display."""
    key = list(detector.encoding_model_.keys())[0]
    place_fields = np.asarray(detector.encoding_model_[key]["place_fields"])
    peak_positions = np.argmax(place_fields, axis=1)
    sort_order = np.argsort(peak_positions)
    return [spike_times[i] for i in sort_order], sort_order


def plot_spike_raster(ax, spike_times, detector):
    """Spike raster sorted by place field peak."""
    sorted_spikes, _ = sort_neurons_by_place_field(spike_times, detector)
    ax.eventplot(
        sorted_spikes, lineoffsets=1, linelengths=0.8,
        colors="black", linewidths=0.3,
    )
    ax.set_ylabel("Neuron", fontsize=7, labelpad=4)
    ax.set_yticks([1, len(spike_times)])
    ax.set_ylim(0, len(spike_times))
    ax.tick_params(labelsize=5, labelbottom=False)


def plot_posterior(ax, time, position, results):
    """Non-local posterior heatmap with true position overlay."""
    acausal = results["acausal_posterior"]
    unstacked = acausal.unstack("state_bins")
    state_vals = unstacked.coords["state"].values
    nl_states = [s for s in state_vals if "Non-Local" in str(s)]
    nl_pos = unstacked.sel(state=nl_states).sum("state").values
    pos_coord = unstacked.coords["position"].values

    vmax = max(np.percentile(nl_pos[nl_pos > 0], 99), 1e-6)
    ax.pcolormesh(
        time, pos_coord, nl_pos.T,
        cmap="bone_r", vmin=0, vmax=vmax, rasterized=True,
    )
    ax.plot(time, position, color=WONG[1], linewidth=0.8, alpha=0.9)
    ax.set_ylabel("Non-local\nposterior\n(cm)", fontsize=7, labelpad=4)
    ax.tick_params(labelsize=5, labelbottom=False)


def plot_state_probabilities(ax, time, results, detector):
    """State probabilities as line plots (raw, no smoothing)."""
    state_probs = results["acausal_state_probabilities"].values

    # Local/No-Spike in background, non-local on top
    for i in [0, 1]:
        ax.plot(
            time, state_probs[:, i], color=STATE_COLORS_SHORT[i],
            label=STATE_NAMES_SHORT[i], linewidth=0.6, alpha=0.4,
        )
    for i in [2, 3]:
        ax.plot(
            time, state_probs[:, i], color=STATE_COLORS_SHORT[i],
            label=STATE_NAMES_SHORT[i], linewidth=1.0, alpha=0.9,
        )
    ax.set_ylabel("P(state)", fontsize=7, labelpad=4)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=5, labelbottom=False)
    ax.legend(
        fontsize=4, loc="upper right", ncol=4, frameon=False,
        handlelength=1, columnspacing=0.5,
    )


def plot_speed(ax, time, speed):
    """Speed trace with immobility threshold."""
    ax.fill_between(time, speed, alpha=0.5, color="#888888", linewidth=0)
    ax.axhline(4.0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.text(
        time[-1], 4.5, "4 cm/s", ha="right", va="bottom",
        fontsize=4, alpha=0.5,
    )
    ax.set_ylabel("Speed\n(cm/s)", fontsize=7, labelpad=4)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=5)


# ═══════════════════════════════════════════════════════════════════════════
# Main Figure Assembly
# ═══════════════════════════════════════════════════════════════════════════


def generate_figure(n_neurons=30, seed=42):
    """Generate complete Figure 1.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    np.random.seed(seed)

    # --- Simulate and decode ---
    print(f"Simulating {n_neurons} neurons...")
    (
        speed, position, spike_times, time, event_times,
        sampling_frequency, is_event, place_fields,
    ) = make_simulated_data(n_neurons=n_neurons)

    print("Fitting detector and estimating parameters (EM)...")
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**8),
        },
        local_position_std=LOCAL_POSITION_STD,
    )
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time=time,
        is_training=~is_event,
    )

    # --- Create figure ---
    set_figure_defaults(context="paper")

    fig_height = min(TWO_COLUMN * 1.35, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)

    # 3 rows: model structure | transition kernels | simulated example.
    # Row 1 (panel C) is thinner than row 0 because two of its four
    # heatmaps are uniform (solid color, no internal structure), so
    # large cells waste space.
    gs_main = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.4, 0.55, 2.0],
        hspace=0.20,
    )

    # ── Row 0: State Diagram (a) + Observation Models (b) ──
    # Panel B contains two scenarios (Local-wins, Non-Local-wins), each a
    # 2x2 of state cells. width_ratios are tuned so Panel A renders at
    # roughly the same width as one scenario block; aspect+anchor inside
    # plot_state_diagram matches the height too.
    gs_row0 = gs_main[0].subgridspec(1, 2, wspace=0.15, width_ratios=[1, 2.2])

    ax_a = fig.add_subplot(gs_row0[0])
    plot_state_diagram(ax_a)

    b_axes = plot_observation_models(fig, gs_row0[1])

    # ── Row 1: Transition Kernels (c) ──
    kernel_axes = plot_transition_kernels(fig, gs_main[1])

    # ── Row 2: Simulated Example (d) — single panel, 4 sub-rows ──
    gs_row2 = gs_main[2].subgridspec(
        4, 1, height_ratios=[0.8, 1.2, 0.8, 0.5], hspace=0.12,
    )

    ax_raster = fig.add_subplot(gs_row2[0])
    plot_spike_raster(ax_raster, spike_times, detector)

    ax_posterior = fig.add_subplot(gs_row2[1], sharex=ax_raster)
    plot_posterior(ax_posterior, time, position, results)

    ax_states = fig.add_subplot(gs_row2[2], sharex=ax_raster)
    plot_state_probabilities(ax_states, time, results, detector)

    ax_speed = fig.add_subplot(gs_row2[3], sharex=ax_raster)
    plot_speed(ax_speed, time, speed)

    for a in [ax_raster, ax_posterior, ax_states, ax_speed]:
        a.set_xlim(time[0], time[-1])

    # ── Panel Labels ──
    label_panels(
        [ax_a, b_axes[0], kernel_axes[0], ax_raster],
        labels=["a", "b", "c", "d"],
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure01"),
    )
    parser.add_argument("--n-neurons", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _ = generate_figure(n_neurons=args.n_neurons, seed=args.seed)

    print(f"Saving to {args.output_dir}...")
    save_figure("figure01", base_dir=args.output_dir, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
