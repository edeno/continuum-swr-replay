#!/usr/bin/env python
"""Generate Figure 1: Switching State Space Model for Non-Local Detection.

This figure explains the non-local detector model to the reader in three rows:

    Row 1 (Model Structure):
        A: State transition diagram — 4 discrete states with transition arrows
        B: Observation model schematic — three scenarios side-by-side
           ("Local wins", "Non-Local wins", and "No-Spike wins"), each showing
           Local, No-Spike, and merged Non-Local observation likelihoods

    Row 2 (Continuous Dynamics):
        C: Self-transition kernels for all four states — Local is uniform
           (re-anchored by the observation kernel each step under multi-bin
           local), No-Spike is identity (single-bin), Continuous is a
           Gaussian random walk, Fragmented is uniform.

    Row 3 (Simulated Example — single panel D):
        Spike raster, non-local posterior, four state probabilities,
        collapsed non-local probability with decision threshold, speed

The narrative flow is: what are the states → how do they interpret spikes →
how do their spatial dynamics differ → what does the output look like.

Usage:
    python scripts/figures/generate_figure01.py
    python scripts/figures/generate_figure01.py --n-neurons 40 --seed 42
    python scripts/figures/generate_figure01.py --output-dir paper/figures/figure01
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
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    WONG,
    label_panels,
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.parameters import (
    LOCAL_POSITION_STD,
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.visualization.palettes import COLORS

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
NON_LOCAL_PROB_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD

# Stickiness from the real fitted detectors. The installed non_local_detector
# default [1e6, 1e6, 300, 300] makes EM prune Local->No-Spike to ~0, hiding the
# No-Spike state; this value keeps it alive and matches the real pipeline.
REAL_DISCRETE_TRANSITION_STICKINESS = np.array([30.0, 100000.0, 30.0, 200.0])

# Decode every Nth sample so the demo bin width (~2 ms) matches the real
# pipeline; the simulator emits 1500 Hz, which starves the No-Spike evidence.
DECODE_BIN_STRIDE = 3

# Lengthen the simulator's 0.5 s no-spike event so the No-Spike state clears the
# transition-driven edge ramps and reads at full strength in panel D.
NO_SPIKE_EVENT_LEN = 0.75


def add_panel_header(ax, title):
    """Draw a centered panel-level header on a dedicated header axes."""
    ax.axis("off")
    ax.text(
        0.5,
        0.65,
        title,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#222222",
        transform=ax.transAxes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Discrete State Diagram
# ═══════════════════════════════════════════════════════════════════════════


def plot_state_diagram(ax):
    """Draw 4-state diagram with transition arrows.

    States arranged in a 2x2 grid. Self-loop probabilities are pulled
    from the detector defaults, and all cross-state transitions are shown
    as bidirectional connections.

    The data limits and gridspec width ratios are tuned so Panel A
    carries the transition topology at roughly the same visual scale as
    a single Panel B likelihood scenario. Aspect is locked equal so the
    state circles remain circular.
    """
    # Aspect locked equal so state circles render as circles. Anchor "N"
    # aligns Panel A's top with the top of the panel-B scenario plots (no
    # title row sits above either now), so the two panels share a
    # horizontal baseline.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")

    positions = {
        "Local": (0.25, 0.80),
        "No-Spike": (0.75, 0.80),
        "Non-Local\nContinuous": (0.25, 0.20),
        "Non-Local\nFragmented": (0.75, 0.20),
    }

    default_detector = NonLocalSortedSpikesDetector()
    transition_matrix = np.asarray(default_detector.discrete_transition_type.values)
    self_probs = {
        state_name.replace("Non-Local ", "Non-Local\n"): f"{prob:.2f}"
        for state_name, prob in zip(
            default_detector.state_names, np.diag(transition_matrix), strict=True
        )
    }

    radius = 0.18

    # Group the two non-local states without erasing their distinct
    # continuous/fragmented colors. This also links panel A to the collapsed
    # non-local color used in panels B and D.
    nonlocal_group_x = 0.04
    nonlocal_group_y = -0.005
    nonlocal_group_width = 0.92
    nonlocal_group_height = 0.43
    ax.add_patch(
        mpatches.Rectangle(
            (nonlocal_group_x, nonlocal_group_y),
            nonlocal_group_width,
            nonlocal_group_height,
            facecolor=COLORS["Non-Local"],
            edgecolor="none",
            alpha=0.14,
            zorder=1,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (nonlocal_group_x, nonlocal_group_y),
            nonlocal_group_width,
            nonlocal_group_height,
            facecolor="none",
            edgecolor=COLORS["Non-Local"],
            linewidth=0.8,
            alpha=0.62,
            zorder=2,
        )
    )

    # Cross-state arrows: one bidirectional connection per unordered state pair.
    cross_transitions = [
        ("Local", "No-Spike"),
        ("Local", "Non-Local\nContinuous"),
        ("Local", "Non-Local\nFragmented"),
        ("No-Spike", "Non-Local\nContinuous"),
        ("No-Spike", "Non-Local\nFragmented"),
        ("Non-Local\nContinuous", "Non-Local\nFragmented"),
    ]
    for from_state, to_state in cross_transitions:
        x1, y1 = positions[from_state]
        x2, y2 = positions[to_state]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        shrink = radius * 1.16
        arrow = mpatches.FancyArrowPatch(
            (x1 + dx / dist * shrink, y1 + dy / dist * shrink),
            (x2 - dx / dist * shrink, y2 - dy / dist * shrink),
            arrowstyle="<->,head_length=3,head_width=1.7",
            mutation_scale=1,
            linewidth=0.8,
            color="#555555",
            alpha=0.55,
            zorder=4,
        )
        ax.add_patch(arrow)

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
            cx,
            cy,
            state_name,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="white",
            zorder=11,
            linespacing=0.9,
        )

    # Self-loop arrows routed to the outside corners so they remain legible
    # and do not collide with the bidirectional cross-state arrows.
    loop_specs = {
        "Local": {
            "start": (-0.55, 0.75),
            "end": (0.12, 1.06),
            "rad": -0.75,
            "label": (-0.92, 1.28),
        },
        "No-Spike": {
            "start": (-0.12, 1.06),
            "end": (0.55, 0.75),
            "rad": -0.75,
            "label": (0.92, 1.28),
        },
        "Non-Local\nContinuous": {
            "start": (0.12, -1.06),
            "end": (-0.55, -0.75),
            "rad": -0.75,
            "label": (-0.92, -1.28),
        },
        "Non-Local\nFragmented": {
            "start": (0.55, -0.75),
            "end": (-0.12, -1.06),
            "rad": -0.75,
            "label": (0.92, -1.28),
        },
    }
    for state_name, (cx, cy) in positions.items():
        spec = loop_specs[state_name]
        start = (cx + radius * spec["start"][0], cy + radius * spec["start"][1])
        end = (cx + radius * spec["end"][0], cy + radius * spec["end"][1])
        arrow = mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="->,head_length=3.5,head_width=2.2",
            connectionstyle=f"arc3,rad={spec['rad']}",
            mutation_scale=1,
            linewidth=0.9,
            color="black",
            alpha=0.58,
            zorder=12,
        )
        ax.add_patch(arrow)
        lx = cx + radius * spec["label"][0]
        ly = cy + radius * spec["label"][1]
        ax.text(
            lx,
            ly,
            self_probs[state_name],
            ha="center",
            va="center",
            fontsize=5.5,
            color="black",
            alpha=0.8,
            zorder=13,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Observation Model Schematic
# ═══════════════════════════════════════════════════════════════════════════


def _draw_scenario_overlay(ax, x, likelihood, animal_pos, scenario_kind):
    """Draw all three state likelihoods overlaid on one axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Single axes to draw into.
    x : np.ndarray, shape (n_pts,)
        Position grid on [0, 1] for the schematic.
    likelihood : np.ndarray, shape (n_pts,)
        The shared place-field curve λ(x), peaking at 1.0.
    animal_pos : float
        Animal's position in [0, 1]. Controls where the Local kernel
        sits and where the magenta position line is drawn.
    scenario_kind : {"local", "nonlocal", "nospike"}
        Determines the relative heights of the three curves so the
        "winning" state's curve is visibly tallest. Also drives the
        per-scenario placement of in-plot curve labels.

    Notes
    -----
    Three curves are drawn (collapsing the four states down to three
    visually distinct likelihoods, since Non-Local Continuous and
    Non-Local Fragmented have identical observation likelihoods; they
    only differ in the transition kernel, shown in panel c):
    - Local: blue, bump = λ(x)·N(x_animal, σ_L).
    - Non-Local: teal, bump = λ(x), broader than Local.
    - No-Spike: gray horizontal line at the No-Spike likelihood value.

    Each curve is labeled in-plot with colored text (no legend, no
    leader arrows), placed at scenario-specific positions that sit
    inside whitespace and clearly identify the curve.
    """
    local_color = STATE_COLORS_SHORT[0]  # blue
    nospike_color = STATE_COLORS_SHORT[1]  # gray
    # Non-Local in panel (b) merges the Continuous and Fragmented states;
    # use the collapsed non-local color shared with panels A and D.
    nonlocal_color = COLORS["Non-Local"]

    sigma_local = 0.05
    kernel = np.exp(-0.5 * ((x - animal_pos) / sigma_local) ** 2)

    # Faithful per-position observation likelihoods (unnormalized kernel,
    # peak 1):
    #   Local: λ(x) · N(x; x_a, σ_L)       — concentrated near x_a
    #   Non-Local: λ(x)                     — spread over the track
    # At x = x_a the kernel equals 1, so Local(x_a) = Non-Local(x_a) =
    # λ(x_a). The "wins" story for Local vs Non-Local is therefore
    # CONCENTRATION (narrow vs broad), not peak height; for No-Spike it
    # is the bin-less likelihood value relative to the curves' peaks.
    if scenario_kind == "local":
        local_curve = likelihood * kernel
        nonlocal_curve = likelihood
        nospike_y = 0.10
    elif scenario_kind == "nonlocal":
        local_curve = likelihood * kernel
        nonlocal_curve = likelihood
        nospike_y = 0.10
    elif scenario_kind == "nospike":
        # Empty bin schematic: make the position-dependent alternatives
        # broad and low so the reader sees weak spatial evidence, with
        # the bin-less No-Spike likelihood clearly winning.
        nospike_scale = 0.30
        broad_nonlocal = np.exp(-0.5 * ((x - animal_pos) / 0.50) ** 2)
        broad_local_kernel = 0.70 + 0.30 * np.exp(-0.5 * ((x - animal_pos) / 0.40) ** 2)
        nonlocal_curve = nospike_scale * (0.82 + 0.18 * broad_nonlocal)
        local_curve = nonlocal_curve * broad_local_kernel
        nospike_y = 0.70
    else:
        raise ValueError(f"Unknown scenario_kind: {scenario_kind!r}")

    # Non-Local drawn first so Local's fill sits on top where they overlap.
    ax.fill_between(
        x,
        nonlocal_curve,
        facecolor=nonlocal_color,
        linewidth=0.0,
        alpha=0.25,
    )
    ax.plot(x, nonlocal_curve, color=nonlocal_color, linewidth=1.0)

    ax.plot(x, local_curve, color=local_color, linewidth=1.0)
    ax.fill_between(x, local_curve, alpha=0.40, color=local_color)

    # Dashed to signal that No-Spike is a scalar likelihood (no position
    # dependence), unlike the Local/Non-Local curves which are functions of x.
    ax.axhline(nospike_y, color=nospike_color, linewidth=1.2, linestyle="--")

    ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8, linestyle="-")
    animal_label_y = {
        "local": 1.10,
        "nonlocal": 0.98,
        "nospike": 1.10,
    }
    ax.text(
        animal_pos + 0.035,
        animal_label_y[scenario_kind],
        "Animal Position",
        color=ANIMAL_POS_COLOR,
        fontsize=5.5,
        ha="left",
        va="center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xlabel("position", fontsize=6, labelpad=3)

    # In-plot labels, one per curve, color-coded. Positions per scenario
    # are chosen to land in whitespace next to the relevant curve, avoiding
    # overlap with other curves or the animal-position line.
    label_specs = {
        "local": [
            ("Local", 0.72, 0.82, "left", local_color),
            ("Non-Local", 0.72, 0.55, "left", nonlocal_color),
            ("No-Spike", 0.97, 0.18, "right", nospike_color),
        ],
        "nonlocal": [
            ("Non-Local", 0.40, 1.12, "center", nonlocal_color),
            ("Local", 0.72, 0.42, "left", local_color),
            ("No-Spike", 0.97, 0.18, "right", nospike_color),
        ],
        "nospike": [
            ("No-Spike", 0.97, 0.78, "right", nospike_color),
            ("Non-Local", 0.72, 0.30, "left", nonlocal_color),
            ("Local", 0.40, 0.50, "center", local_color),
        ],
    }
    for name, lx, ly, ha, color in label_specs[scenario_kind]:
        ax.text(
            lx,
            ly,
            name,
            color=color,
            fontsize=6.5,
            fontweight="bold",
            ha=ha,
            va="center",
        )


def plot_observation_models(fig, gs_slot):
    """Show how the state likelihoods react under three scenarios.

    Each scenario is one plot overlaying three state likelihoods on a
    shared position axis:

    - "Local wins": x_animal coincides with the place-field peak. The
      cell fires where its tuning says it should — Local likelihood
      (λ(x)·N(x_animal, σ_L)) peaks tall at the animal's position;
      Non-Local likelihood λ(x) is also peaked there but per-bin mass
      is lower; No-Spike is small.
    - "Non-Local wins": x_animal is far from the place-field peak.
      Local is tiny because λ(x_animal) is near zero; Non-Local is
      tall at the place-field center, consistent with replay of a
      different location.
    - "No-Spike wins": no spike in the bin. No-Spike's bin-less
      likelihood (drawn as a horizontal line) is larger than the peak
      of either position-dependent likelihood.

    Non-Local Continuous and Non-Local Fragmented are collapsed to a
    single "Non-Local" curve here because they share an observation
    likelihood; their distinction lives in the transition kernels
    (panel c) and dynamics (panel d).

    Returns 3 axes (one per scenario). The first is used by
    ``label_panels`` for the panel "b" label.
    """
    n_pts = 200
    x = np.linspace(0, 1, n_pts)
    likelihood_center = 0.4
    likelihood = np.exp(-0.5 * ((x - likelihood_center) / 0.10) ** 2)

    # Keep the three observation scenarios visually grouped: the internal
    # spacing is tighter than the A/B panel separation in the outer grid.
    gs_outer = gs_slot.subgridspec(1, 3, wspace=0.04)

    scenarios = [
        # (header, animal_pos, scenario_kind)
        ("Local wins", likelihood_center, "local"),
        ("Non-Local wins", 0.55, "nonlocal"),
        ("No-Spike wins", likelihood_center, "nospike"),
    ]

    all_axes = []
    for col, (header, animal_pos, scenario_kind) in enumerate(scenarios):
        scen_ax = fig.add_subplot(gs_outer[0, col])
        _draw_scenario_overlay(
            scen_ax,
            x,
            likelihood,
            animal_pos,
            scenario_kind,
        )
        scen_ax.set_title(header, fontsize=7, fontweight="bold", pad=2)
        all_axes.append(scen_ax)

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
    - No-Spike → No-Spike: Scalar self-transition with no spatial position.
    - Non-Local Continuous → Non-Local Continuous: Gaussian random walk
      (smooth replay trajectories).
    - Non-Local Fragmented → Non-Local Fragmented: Uniform (jump to any
      position).
    """
    side_margin = 0.45
    group_left = side_margin / (4 + 2 * side_margin)
    group_right = 1 - group_left
    gs_inner = gs_slot.subgridspec(
        2,
        6,
        height_ratios=[0.13, 1.0],
        width_ratios=[side_margin, 1, 1, 1, 1, side_margin],
        hspace=0.03,
        wspace=0.06,
    )

    header_ax = fig.add_subplot(gs_inner[0, :])
    add_panel_header(header_ax, "Continuous position transitions")
    header_ax.text(
        group_right,
        0.65,
        "Color = transition probability",
        ha="right",
        va="center",
        fontsize=5.5,
        color="#555555",
        transform=header_ax.transAxes,
    )

    n_bins = 50
    configs = [
        {
            "name": "Local\n→ Local",
            "subtitle": "Uniform; local kernel re-anchors",
            "color": STATE_COLORS_SHORT[0],
            "kind": "uniform",
            "show_local_anchor": True,
        },
        {
            "name": "No-Spike\n→ No-Spike",
            "subtitle": "No spatial position",
            "color": STATE_COLORS_SHORT[1],
            "kind": "scalar",
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

    axes = [header_ax]
    for idx, config in enumerate(configs):
        sub_ax = fig.add_subplot(gs_inner[1, idx + 1])
        axes.append(sub_ax)

        if config["kind"] == "scalar":
            sub_ax.set_xlim(0, 1)
            sub_ax.set_ylim(0, 1)
            sub_ax.set_aspect("equal")
            scalar_box = mpatches.Rectangle(
                (0.24, 0.24),
                0.52,
                0.52,
                facecolor=config["color"],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.75,
            )
            sub_ax.add_patch(scalar_box)
            sub_ax.text(
                0.5,
                0.5,
                "single\nstate",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="white",
                linespacing=0.9,
            )
        elif config["kind"] == "uniform":
            trans = np.ones((n_bins, n_bins)) / n_bins
            sub_ax.imshow(
                trans,
                aspect="equal",
                cmap="magma",
                vmin=0,
                vmax=0.15,
                interpolation="nearest",
                origin="lower",
            )
            if config.get("show_local_anchor", False):
                anchor_y = n_bins * 0.67
                sub_ax.axhline(
                    anchor_y,
                    color=ANIMAL_POS_COLOR,
                    linewidth=0.9,
                    linestyle="-",
                    zorder=5,
                )
                sub_ax.text(
                    2,
                    anchor_y + 2,
                    "Animal\nPosition",
                    color=ANIMAL_POS_COLOR,
                    fontsize=5.3,
                    ha="left",
                    va="bottom",
                    linespacing=0.85,
                    zorder=6,
                )
        else:  # gaussian
            trans = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                trans[i, :] = np.exp(
                    -0.5 * ((np.arange(n_bins) - i) / config["sigma"]) ** 2
                )
                trans[i, :] /= trans[i, :].sum()
            # Shared vmax so the uniform panels and Gaussian band are visually
            # comparable while No-Spike remains visibly non-spatial.
            sub_ax.imshow(
                trans,
                aspect="equal",
                cmap="magma",
                vmin=0,
                vmax=0.15,
                interpolation="nearest",
                origin="lower",
            )

        if config["kind"] == "scalar":
            for spine in sub_ax.spines.values():
                spine.set_visible(False)
        else:
            for spine in sub_ax.spines.values():
                spine.set_edgecolor(config["color"])
                spine.set_linewidth(1.5)

        sub_ax.set_xticks([])
        sub_ax.set_yticks([])

        sub_ax.set_title(
            config["name"],
            fontsize=5.5,
            color=config["color"],
            fontweight="bold",
            pad=3,
            linespacing=0.9,
        )
        sub_ax.text(
            0.5,
            -0.04,
            config["subtitle"],
            ha="center",
            va="top",
            fontsize=5,
            transform=sub_ax.transAxes,
            color="#555555",
            style="italic",
        )

        if config["kind"] != "scalar" and not config.get("show_local_anchor", False):
            sub_ax.set_xlabel("position$_t$", fontsize=5.5, labelpad=12)
        if idx == 0 and not config.get("show_local_anchor", False):
            sub_ax.set_ylabel("position$_{t+1}$", fontsize=5.5, labelpad=2)

    return axes


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — Simulated Data + Decoding (single composite panel)
# ═══════════════════════════════════════════════════════════════════════════


def get_non_local_probability(results):
    """Return P(Non-Local Continuous) + P(Non-Local Fragmented)."""
    non_local_prob = results.acausal_state_probabilities.sel(
        states=["Non-Local Continuous", "Non-Local Fragmented"]
    ).sum("states")
    return np.asarray(non_local_prob.values).squeeze()


def get_threshold_crossing_intervals(time, mask):
    """Convert a boolean time-bin mask to contiguous plotting intervals."""
    time = np.asarray(time)
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.size == 0 or not mask.any():
        return []

    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    dt = float(np.nanmedian(np.diff(time))) if time.size > 1 else 0.0

    intervals = []
    for start, stop in zip(starts, stops, strict=True):
        left = max(time[0], time[start] - dt / 2)
        right = min(time[-1], time[stop - 1] + dt / 2)
        intervals.append((left, right))
    return intervals


def shade_non_local_intervals(axes, intervals):
    """Mark threshold-crossing non-local bins consistently across panel D."""
    for ax in axes:
        for start, end in intervals:
            ax.axvspan(
                start,
                end,
                color=COLORS["Non-Local"],
                alpha=0.14,
                linewidth=0,
                zorder=2.5,
            )
            ax.axvline(
                start,
                color=COLORS["Non-Local"],
                alpha=0.25,
                linewidth=0.45,
                zorder=2.6,
            )
            ax.axvline(
                end,
                color=COLORS["Non-Local"],
                alpha=0.25,
                linewidth=0.45,
                zorder=2.6,
            )


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
        sorted_spikes,
        lineoffsets=1,
        linelengths=0.8,
        colors="black",
        linewidths=0.3,
        zorder=4,
    )
    ax.set_ylabel("Neuron", fontsize=7, labelpad=4)
    ax.set_yticks([1, len(spike_times)])
    ax.set_ylim(0, len(spike_times))
    ax.tick_params(labelsize=5.5, labelbottom=False)


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
        time,
        pos_coord,
        nl_pos.T,
        cmap="bone_r",
        vmin=0,
        vmax=vmax,
        rasterized=True,
        zorder=1,
    )
    # Magenta for the animal's true position, consistent with the position line
    # in panel B and avoiding collision with WONG[1] used for Continuous state.
    ax.plot(
        time,
        position,
        color=ANIMAL_POS_COLOR,
        linewidth=0.8,
        alpha=0.9,
        zorder=4,
    )
    ax.set_ylabel("Decoded\nposition\n(cm)", fontsize=7, labelpad=4)
    ax.tick_params(labelsize=5.5, labelbottom=False)


def plot_state_probabilities(ax, time, results):
    """Plot the four raw HMM state probabilities."""
    state_probs = results["acausal_state_probabilities"].values

    for i, (name, color) in enumerate(
        zip(
            STATE_NAMES_SHORT,
            STATE_COLORS_SHORT,
            strict=True,
        )
    ):
        ax.plot(
            time,
            state_probs[:, i],
            color=color,
            label=name,
            linewidth=0.75 if i < 2 else 0.95,
            alpha=0.55 if i < 2 else 0.9,
            zorder=4 if i >= 2 else 3,
        )
    ax.set_ylabel("State\nprob.", fontsize=7, labelpad=4)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=5.5, labelbottom=False)
    ax.legend(
        fontsize=5.5,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=4,
        frameon=False,
        handlelength=1,
        columnspacing=0.5,
    )


def plot_non_local_decision(
    ax,
    time,
    non_local_prob,
    threshold=NON_LOCAL_PROB_THRESHOLD,
):
    """Plot collapsed non-local probability and event threshold."""
    ax.plot(
        time,
        non_local_prob,
        color=COLORS["Non-Local"],
        linewidth=1.15,
        zorder=4,
    )
    ax.fill_between(
        time,
        0,
        non_local_prob,
        where=non_local_prob > threshold,
        color=COLORS["Non-Local"],
        alpha=0.22,
        step="pre",
        linewidth=0,
        zorder=3,
    )
    ax.axhline(
        threshold,
        color="lightgrey",
        linestyle="--",
        linewidth=0.75,
        alpha=0.8,
        zorder=3,
    )
    ax.text(
        time[-1],
        threshold + 0.035,
        f"P(non-local) > {threshold:.1f}",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#555555",
    )
    equation_text = [
        ("P(non-local) = ", COLORS["Non-Local"]),
        ("P(continuous)", STATE_COLORS_SHORT[2]),
        (" + ", COLORS["Non-Local"]),
        ("P(fragmented)", STATE_COLORS_SHORT[3]),
    ]
    equation = HPacker(
        children=[
            TextArea(
                text,
                textprops={
                    "fontsize": 5.5,
                    "color": color,
                },
            )
            for text, color in equation_text
        ],
        align="baseline",
        pad=0,
        sep=1.0,
    )
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower right",
            child=equation,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(0.985, 0.16),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
    )
    ax.set_ylabel("Non-local\nprob.", fontsize=7, labelpad=4)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=5.5, labelbottom=False)


def plot_speed(ax, time, speed):
    """Speed trace with immobility threshold."""
    ax.fill_between(time, speed, alpha=0.5, color="#888888", linewidth=0)
    ax.axhline(
        MAX_SPEED_THRESHOLD_CM_PER_SEC,
        color="black",
        linewidth=0.5,
        linestyle=":",
        alpha=0.4,
    )
    ax.text(
        time[-1],
        MAX_SPEED_THRESHOLD_CM_PER_SEC + 0.5,
        f"{MAX_SPEED_THRESHOLD_CM_PER_SEC:g} cm/s",
        ha="right",
        va="bottom",
        fontsize=5,
        alpha=0.7,
    )
    ax.set_ylabel("Speed\n(cm/s)", fontsize=7, labelpad=4)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=5.5)


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
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=n_neurons)

    # Extend the no-spike event (event index 3) to NO_SPIKE_EVENT_LEN by removing
    # spikes over the longer window and holding it out of training.
    ns_start = event_times[3, 0]
    ns_end = ns_start + NO_SPIKE_EVENT_LEN
    spike_times = [st[(st < ns_start) | (st > ns_end)] for st in spike_times]
    event_times[3, 1] = ns_end
    is_event = is_event | ((time >= ns_start) & (time <= ns_end))

    # Stride the decode/behavior grids to ~2 ms; spike_times stay full-resolution.
    time = time[::DECODE_BIN_STRIDE]
    position = position[::DECODE_BIN_STRIDE]
    speed = speed[::DECODE_BIN_STRIDE]
    is_event = is_event[::DECODE_BIN_STRIDE]

    print("Fitting detector and estimating parameters (EM)...")
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**8),
        },
        local_position_std=LOCAL_POSITION_STD,
        discrete_transition_stickiness=REAL_DISCRETE_TRANSITION_STICKINESS,
    )
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time=time,
        is_training=~is_event,
    )
    non_local_prob = get_non_local_probability(results)
    non_local_intervals = get_threshold_crossing_intervals(
        time,
        non_local_prob > NON_LOCAL_PROB_THRESHOLD,
    )

    # --- Create figure ---
    set_figure_defaults(context="paper")

    # Keep the full-width manuscript format, but avoid over-allocating the
    # simulated example. The model rows get enough height for readable labels,
    # while panel D remains a compact worked example.
    fig_height = min(TWO_COLUMN * 1.18, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.02,
        h_pad=0.02,
        wspace=0.03,
        hspace=0.03,
    )

    # 3 rows: model structure | transition kernels | simulated example.
    # Row 1 (panel C) is thinner than row 0 because two of its four
    # heatmaps are uniform (solid color, no internal structure), so
    # large cells waste space.
    gs_main = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[0.70, 0.55, 1.50],
        hspace=0.16,
    )

    # ── Row 0: State Diagram (a) + Observation Models (b) ──
    # Panel B contains three scenarios (Local-wins, Non-Local-wins,
    # No-Spike-wins). width_ratios are tuned so Panel A can carry the
    # transition topology without crowding Panel B's likelihood scenarios.
    gs_row0 = gs_main[0].subgridspec(
        2,
        2,
        height_ratios=[0.12, 1.0],
        hspace=0.03,
        wspace=0.14,
        width_ratios=[1, 2.8],
    )

    ax_a_header = fig.add_subplot(gs_row0[0, 0])
    add_panel_header(ax_a_header, "Discrete states and transitions")

    ax_a = fig.add_subplot(gs_row0[1, 0])
    plot_state_diagram(ax_a)

    ax_b_header = fig.add_subplot(gs_row0[0, 1])
    add_panel_header(ax_b_header, "Observation likelihoods")

    plot_observation_models(fig, gs_row0[1, 1])

    # ── Row 1: Transition Kernels (c) ──
    kernel_axes = plot_transition_kernels(fig, gs_main[1])

    # ── Row 2: Simulated Example (d) — single panel, 5 sub-rows ──
    gs_row2 = gs_main[2].subgridspec(
        5,
        1,
        height_ratios=[0.65, 0.95, 0.58, 0.58, 0.38],
        hspace=0.10,
    )

    ax_raster = fig.add_subplot(gs_row2[0])
    plot_spike_raster(ax_raster, spike_times, detector)

    ax_posterior = fig.add_subplot(gs_row2[1], sharex=ax_raster)
    plot_posterior(ax_posterior, time, position, results)

    ax_states = fig.add_subplot(gs_row2[2], sharex=ax_raster)
    plot_state_probabilities(ax_states, time, results)

    ax_nonlocal = fig.add_subplot(gs_row2[3], sharex=ax_raster)
    plot_non_local_decision(ax_nonlocal, time, non_local_prob)

    ax_speed = fig.add_subplot(gs_row2[4], sharex=ax_raster)
    plot_speed(ax_speed, time, speed)

    panel_d_axes = [ax_raster, ax_posterior, ax_states, ax_nonlocal, ax_speed]
    for a in panel_d_axes:
        a.set_xlim(time[0], time[-1])

    # Mark non-local events only on the neural/model rows. The speed row
    # carries its own immobility threshold cue.
    shade_non_local_intervals(
        [ax_raster, ax_posterior, ax_states, ax_nonlocal],
        non_local_intervals,
    )

    # ── Panel Labels ──
    label_panels([ax_a_header, ax_b_header, ax_raster], labels=["a", "b", "d"])
    label_panels([kernel_axes[0]], labels=["c"], xpos=-0.02)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure01"),
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
