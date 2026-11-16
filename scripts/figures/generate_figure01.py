#!/usr/bin/env python
"""Generate Figure 1: the switching state-space model for non-local detection.

The figure explains the model and then shows it at work on simulated data:

    a: the four discrete states (Local, No-Spike, and the two non-local states,
       Continuous and Fragmented) and the transitions between them
    b: how each state scores the same spikes — the likelihood of the spikes over
       position, the positions each state allows, and the evidence that results,
       in three scenarios
    c: how the represented position moves within each state
    d: a simulated run — raster, decoded position, non-local probability, speed
    e: one example of each kind of simulated spiking in d, at a millisecond
       scale — raster, decoded position, the four state probabilities

The narrative is: what the states are -> how they score spikes -> how their
positions move -> what the model reports over a run -> what each state looks
like inside an event.

Panels a-c are schematics drawn from illustrative curves, not from a fit. Panels
d and e are one simulation (``make_simulated_data``) decoded by the detector of
``make_example_detector``.

Usage:
    python scripts/figures/generate_figure01.py
    python scripts/figures/generate_figure01.py --n-neurons 40 --seed 42
    python scripts/figures/generate_figure01.py --output-dir paper/figures/figure01
"""

import argparse
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.transforms import Bbox
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    DECODE_BIN_STRIDE,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    build_speed_eligible_event_cohort,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    get_confident_times,
    nonlocal_state_probability,
)
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    LOCAL_POSITION_STD,
    NON_LOCAL_MINIMUM_DURATION_S,
)
from continuum_swr_replay.visualization.figure_utilities import (
    TWO_COLUMN,
    WONG,
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.visualization.palettes import COLORS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colors — colorblind-friendly, consistent across the figure
# ---------------------------------------------------------------------------
STATE_NAMES_SHORT = ["Local", "No-Spike", "Continuous", "Fragmented"]
STATE_COLORS_SHORT = [WONG[5], "#999999", WONG[1], WONG[6]]
LOCAL_COLOR, NO_SPIKE_COLOR, CONTINUOUS_COLOR, FRAGMENTED_COLOR = STATE_COLORS_SHORT
NON_LOCAL_COLOR = COLORS["Non-Local"]
ANIMAL_POS_COLOR = "magenta"

LABEL_SIZE = 7
SMALL_SIZE = 6
TICK_SIZE = 5.5
PANEL_LETTER_SIZE = 9

# A demo value for the simulation, chosen so EM keeps the No-Spike state alive:
# with the installed non_local_detector default stickiness
# [1e5, 1e5, 300, 300], EM prunes Local->No-Spike to ~0 on the simulated data
# and the No-Spike state disappears from the simulated panels. It is NOT the
# real decode's prior: the real decodes use setting P30-B
# (``continuum_swr_replay.decoding.transitions.nonlocal_transition_settings``).
# Panel a therefore draws the transitions without their probabilities.
DEMO_DISCRETE_TRANSITION_STICKINESS = np.array([30.0, 100000.0, 30.0, 200.0])

# Lengthen the simulator's 0.5 s no-spike event so the No-Spike state clears the
# transition-driven edge ramps and reads at full strength.
NO_SPIKE_EVENT_LEN = 0.75

# Panel e's Local example is taken from the pause that holds the first sweep: it
# starts and ends this long before the sweep does.
LOCAL_EXAMPLE_LEAD_S = (1.25, 0.85)
# Panel d's flags for the Local example and the first sweep are about 1 s apart,
# so each is moved this far away from the other.
EXAMPLE_FLAG_SPREAD_S = 0.4

FIGURE_HEIGHT = 8.3  # inches
# The data axes of panels d and e, in figure fractions.
DATA_LEFT, DATA_RIGHT = 0.095, 0.985
# A state diagram spans 1 unit across and this many up (see ``plot_state_diagram``).
STATE_DIAGRAM_HEIGHT = 1.14


def make_example_detector() -> NonLocalSortedSpikesDetector:
    """The unfitted detector the simulated panels fit.

    Returns
    -------
    NonLocalSortedSpikesDetector
        Sorted-spikes KDE likelihood, ``LOCAL_POSITION_STD`` Local kernel, the
        library's default initial discrete transition matrix and
        ``DEMO_DISCRETE_TRANSITION_STICKINESS``.
    """
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**8),
        },
        local_position_std=LOCAL_POSITION_STD,
        discrete_transition_stickiness=DEMO_DISCRETE_TRANSITION_STICKINESS,
    )


def _hide_top_right(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Panel a — the discrete states
# ═══════════════════════════════════════════════════════════════════════════


def plot_state_diagram(ax: Axes) -> None:
    """Draw the four states, the non-local group and the transitions.

    Every pair of states is joined by a two-way arrow and every state has a
    self-loop. No probability is written: the transitions are learned from the
    data, and the simulation's starting values are a demo setting
    (``DEMO_DISCRETE_TRANSITION_STICKINESS``).

    The data limits span 1 across and ``STATE_DIAGRAM_HEIGHT`` up with an equal
    aspect, so an axes of that shape is filled exactly.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, -0.12 + STATE_DIAGRAM_HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")

    centers = {
        "Local": (0.25, 0.80),
        "No-Spike": (0.75, 0.80),
        "Continuous": (0.25, 0.24),
        "Fragmented": (0.75, 0.24),
    }
    radius = 0.17

    for facecolor, edgecolor, alpha in (
        (NON_LOCAL_COLOR, "none", 0.13),
        ("none", NON_LOCAL_COLOR, 1.0),
    ):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.01, 0.0),
                0.98,
                0.48,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.8,
                alpha=alpha,
            )
        )
    ax.text(
        0.5,
        -0.07,
        "Non-local",
        color=NON_LOCAL_COLOR,
        fontsize=LABEL_SIZE,
        fontweight="bold",
        ha="center",
        va="center",
    )

    names = list(centers)
    for i, from_state in enumerate(names):
        for to_state in names[i + 1 :]:
            start = np.array(centers[from_state])
            end = np.array(centers[to_state])
            unit = (end - start) / np.hypot(*(end - start))
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    start + unit * radius * 1.12,
                    end - unit * radius * 1.12,
                    arrowstyle="<->,head_length=2.6,head_width=1.5",
                    mutation_scale=1,
                    linewidth=0.7,
                    color="#777777",
                    zorder=3,
                )
            )

    for (name, (cx, cy)), color in zip(
        centers.items(), STATE_COLORS_SHORT, strict=True
    ):
        ax.add_patch(
            mpatches.Circle(
                (cx, cy),
                radius,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
        )
        ax.text(
            cx,
            cy,
            name,
            color="white",
            fontsize=5.0,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        # Self-loop on the outer corner, away from the cross-state arrows.
        side = -1 if cx < 0.5 else 1
        vert = 1 if cy > 0.5 else -1
        ax.add_patch(
            mpatches.FancyArrowPatch(
                (cx + side * radius * 0.55, cy + vert * radius * 0.75),
                (cx - side * radius * 0.12, cy + vert * radius * 1.06),
                connectionstyle=f"arc3,rad={0.9 * side * vert}",
                arrowstyle="->,head_length=2.6,head_width=1.5",
                mutation_scale=1,
                linewidth=0.7,
                color="#444444",
                zorder=7,
                clip_on=False,
            )
        )


# ═══════════════════════════════════════════════════════════════════════════
# Panel b — how each state scores the spikes
# ═══════════════════════════════════════════════════════════════════════════


def plot_spike_scoring(fig: Figure, slot: SubplotSpec) -> None:
    """Three scenarios: the same kind of evidence, scored by each state.

    Each scenario stacks the likelihood of the observed spikes over position,
    the positions each state allows (Local: a narrow band at the animal;
    Non-local: the whole track), and the relative evidence for Local, Non-local
    and No-Spike. The first two scenarios share one likelihood and differ only
    in where the animal is.

    The curves are illustrative. The evidence is computed from them the way the
    model does: a state's evidence is the likelihood averaged over the positions
    the state allows (the Local kernel is normalized, so it acts as a prior over
    position centered on the animal), and No-Spike's is a single number. The
    three are scaled to sum to 1. The No-Spike values are chosen by hand: small
    when spikes are observed, large when none are.
    """
    x = np.linspace(0, 1, 400)
    place_field = np.exp(-0.5 * ((x - 0.35) / 0.09) ** 2)
    local_kernel_std = 0.045
    # (title, animal position, likelihood of the spikes over position,
    #  likelihood of the spikes under No-Spike)
    scenarios = [
        ("Spikes match the\nanimal's position", 0.35, 0.02 + place_field, 1e-3),
        ("Spikes match\nanother position", 0.78, 0.02 + place_field, 1e-3),
        ("No spikes", 0.35, 0.10 + 0.06 * np.cos(2 * np.pi * x), 1.0),
    ]
    evidence_colors = [LOCAL_COLOR, NON_LOCAL_COLOR, NO_SPIKE_COLOR]
    gs = slot.subgridspec(
        3, 3, height_ratios=[1.0, 0.36, 0.62], hspace=0.5, wspace=0.14
    )

    for col, (title, animal, likelihood, no_spike_likelihood) in enumerate(scenarios):
        kernel = np.exp(-0.5 * ((x - animal) / local_kernel_std) ** 2)
        kernel /= kernel.sum()
        evidence = np.array(
            [(kernel * likelihood).sum(), likelihood.mean(), no_spike_likelihood]
        )
        evidence /= evidence.sum()

        ax = fig.add_subplot(gs[0, col])
        ax.fill_between(x, likelihood, color="#bbbbbb", linewidth=0)
        ax.plot(x, likelihood, color="#333333", linewidth=0.9)
        ax.axvline(animal, color=ANIMAL_POS_COLOR, linewidth=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            title, fontsize=SMALL_SIZE, fontweight="bold", pad=3, linespacing=0.95
        )
        if col == 0:
            ax.set_ylabel("Likelihood\nof spikes", fontsize=SMALL_SIZE, labelpad=3)
            ax.text(
                animal + 0.04,
                1.06,
                "animal",
                color=ANIMAL_POS_COLOR,
                fontsize=SMALL_SIZE,
                ha="left",
                va="center",
            )

        ax_allowed = fig.add_subplot(gs[1, col])
        strips = [kernel / kernel.max(), np.full_like(x, 0.55)]
        for row, (strip, color) in enumerate(
            zip(strips, (LOCAL_COLOR, NON_LOCAL_COLOR), strict=True)
        ):
            bottom = 1 - row - 0.82
            ax_allowed.imshow(
                strip[None, :],
                cmap=LinearSegmentedColormap.from_list("strip", ["white", color]),
                vmin=0,
                vmax=1,
                aspect="auto",
                extent=(0, 1, bottom, 1 - row),
            )
            ax_allowed.add_patch(
                mpatches.Rectangle(
                    (0, bottom), 1, 0.82, fill=False, edgecolor=color, linewidth=0.5
                )
            )
        ax_allowed.axvline(animal, color=ANIMAL_POS_COLOR, linewidth=0.9)
        ax_allowed.set_xlim(0, 1)
        ax_allowed.set_ylim(-0.9, 1.05)
        ax_allowed.set_xticks([])
        ax_allowed.set_yticks([])
        for spine in ax_allowed.spines.values():
            spine.set_visible(False)
        ax_allowed.set_xlabel("Position", fontsize=SMALL_SIZE, labelpad=1)
        if col == 0:
            ax_allowed.set_yticks([0.59, -0.41])
            ax_allowed.set_yticklabels(["Local", "Non-local"], fontsize=SMALL_SIZE)
            ax_allowed.tick_params(axis="y", length=0, pad=2)
            for label, color in zip(
                ax_allowed.get_yticklabels(),
                (LOCAL_COLOR, NON_LOCAL_COLOR),
                strict=True,
            ):
                label.set_color(color)
            ax_allowed.text(
                0.0,
                1.12,
                "Positions each state allows",
                fontsize=TICK_SIZE,
                color="#555555",
                ha="left",
                va="bottom",
                clip_on=False,
            )

        ax_evidence = fig.add_subplot(gs[2, col])
        ax_evidence.barh([2, 1, 0], evidence, color=evidence_colors, height=0.72)
        favored = int(np.argmax(evidence))
        ax_evidence.text(
            evidence[favored] + 0.03,
            2 - favored,
            "favored",
            fontsize=TICK_SIZE,
            va="center",
            ha="left",
            color=evidence_colors[favored],
            fontweight="bold",
        )
        ax_evidence.set_xlim(0, 1.45)
        ax_evidence.set_ylim(-0.6, 2.6)
        ax_evidence.set_xticks([0, 1])
        ax_evidence.tick_params(axis="x", labelsize=TICK_SIZE, length=2, pad=1)
        ax_evidence.set_yticks([])
        _hide_top_right(ax_evidence)
        ax_evidence.spines["bottom"].set_bounds(0, 1)
        if col == 1:
            ax_evidence.set_xlabel(
                "Relative evidence for each state", fontsize=SMALL_SIZE, labelpad=1
            )
        if col == 0:
            ax_evidence.set_yticks([2, 1, 0])
            ax_evidence.set_yticklabels(
                ["Local", "Non-local", "No-Spike"], fontsize=SMALL_SIZE
            )
            ax_evidence.tick_params(axis="y", length=0, pad=2)
            for label, color in zip(
                ax_evidence.get_yticklabels(), evidence_colors, strict=True
            ):
                label.set_color(color)


# ═══════════════════════════════════════════════════════════════════════════
# Panel c — how the represented position moves within each state
# ═══════════════════════════════════════════════════════════════════════════


def plot_position_transitions(fig: Figure, slot: SubplotSpec) -> None:
    """Position-to-position transition of each state, in panel a's 2 x 2 layout.

    - Local: the next position is at the animal, whatever the current one. In the
      detector the Local transition is uniform and the likelihood kernel pins it
      to the animal (``base.initialize_continuous_state_transition``); the panel
      draws the combined effect, a band at the animal's position.
    - No-Spike: the state has no position.
    - Continuous: a Gaussian random walk, so the next position is nearby.
    - Fragmented: uniform, so the next position is anywhere.
    """
    gs = slot.subgridspec(2, 3, width_ratios=[1, 1, 0.08], hspace=0.4, wspace=0.16)
    n_bins = 50
    bins = np.arange(n_bins)
    animal_bin = 0.68 * n_bins

    # (n_bins, n_bins): row = position now, column = position next
    # Each row is scaled by the whole Gaussian's mass, not by the part that falls
    # on the track, so the band is the same at the track's ends as in its middle.
    random_walk_std = 5.0
    random_walk = np.exp(
        -0.5 * ((bins[None, :] - bins[:, None]) / random_walk_std) ** 2
    ) / (random_walk_std * np.sqrt(2 * np.pi))
    uniform = np.full((n_bins, n_bins), 1 / n_bins)
    at_animal = np.exp(-0.5 * ((bins[None, :] - animal_bin) / 3.5) ** 2) * np.ones(
        (n_bins, 1)
    )
    at_animal /= at_animal.sum(axis=1, keepdims=True)

    cells = [
        ("Local", LOCAL_COLOR, at_animal, "at the animal"),
        ("No-Spike", NO_SPIKE_COLOR, None, "no position"),
        ("Continuous", CONTINUOUS_COLOR, random_walk, "nearby"),
        ("Fragmented", FRAGMENTED_COLOR, uniform, "anywhere"),
    ]
    vmax = max(t.max() for _, _, t, _ in cells if t is not None)
    image = None
    for k, (name, color, transition, rule) in enumerate(cells):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_anchor("N" if k < 2 else "S")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)
        if transition is None:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_linestyle((0, (2, 2)))
        else:
            image = ax.imshow(
                transition.T,
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
        if name == "Local":
            ax.plot(
                [1.0, 1.09],
                [animal_bin / n_bins] * 2,
                color=ANIMAL_POS_COLOR,
                linewidth=1.2,
                transform=ax.transAxes,
                clip_on=False,
                solid_capstyle="butt",
            )
        ax.set_title(
            f"{name}\n{rule}",
            fontsize=TICK_SIZE,
            fontweight="bold",
            color=color,
            pad=2,
            linespacing=0.95,
        )
        if name == "Continuous":
            ax.set_xlabel("Position now", fontsize=SMALL_SIZE, labelpad=2)
            ax.set_ylabel("Position next", fontsize=SMALL_SIZE, labelpad=2)

    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[:, 2]))
    colorbar.set_ticks([0, vmax])
    colorbar.set_ticklabels(["0", "max"])
    colorbar.ax.tick_params(labelsize=TICK_SIZE, length=2, pad=1)
    colorbar.set_label("Transition probability", fontsize=SMALL_SIZE, labelpad=-4)
    colorbar.outline.set_linewidth(0.4)


# ═══════════════════════════════════════════════════════════════════════════
# Panels d and e — the simulation and its decode
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Example:
    """One column of panel e and its numbered flag in panel d."""

    state: str
    spiking: str  # what the simulated spikes do
    color: str
    pattern: tuple[float, float] | None  # s; when the simulated pattern occurs
    window: tuple[float, float]  # s; the time shown
    flag_shift: float = 0.0  # s; how far panel d's flag sits from where it points

    @property
    def origin(self) -> float:
        """Time zero of the column, in s: the pattern's start, else the window's."""
        return self.pattern[0] if self.pattern is not None else self.window[0]

    @property
    def flag_time(self) -> float:
        """Where panel d's flag points, in s."""
        return float(np.mean(self.pattern if self.pattern is not None else self.window))


@dataclass(frozen=True)
class DecodedSimulation:
    """The simulation and what the detector decoded from it."""

    time: np.ndarray  # (n_time,), s
    position: np.ndarray  # (n_time,), cm
    speed: np.ndarray  # (n_time,), cm/s
    sorted_spikes: list[np.ndarray]  # one array of spike times (s) per neuron
    state_probabilities: np.ndarray  # (n_time, 4), in STATE_NAMES_SHORT order
    posterior: np.ndarray  # (n_time, n_position), summed over the states
    posterior_vmax: float  # top of the posterior's colour scale
    position_bins: np.ndarray  # (n_position,), cm
    non_local_probability: np.ndarray  # (n_time,)
    event_times: np.ndarray  # (n_events, 2), s; the simulator's inserted patterns
    non_local_events: list[tuple[float, float]]  # s; by the project's event rule

    def next_time_at_track_start(self, after: float) -> float:
        """Middle of the animal's first visit to the track's start after ``after``, in s."""
        at_start = (self.time > after) & (self.position <= self.position.min() + 1.0)
        first = int(np.flatnonzero(at_start)[0])
        last = first + int(np.argmin(at_start[first:]))  # the visit's end
        return float(self.time[(first + last) // 2])

    def examples(self) -> list[Example]:
        """Panel e's columns, in time order.

        One Local window and the simulator's four inserted patterns: a sweep, a
        scatter of positions, a second sweep and a silence. The simulator's
        fifth, "local" event has zero duration and is not shown.
        """

        def around(pattern: tuple[float, float]) -> tuple[float, float]:
            pad = 0.6 * (pattern[1] - pattern[0]) + 0.03
            return (pattern[0] - pad, pattern[1] + pad)

        sweep_1, scatter, sweep_2, silence = (
            (float(start), float(end)) for start, end in self.event_times[:4]
        )
        local_window = (
            sweep_1[0] - LOCAL_EXAMPLE_LEAD_S[0],
            sweep_1[0] - LOCAL_EXAMPLE_LEAD_S[1],
        )
        in_window = (self.time >= local_window[0]) & (self.time <= local_window[1])
        if not (self.speed[in_window] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC).all():
            raise ValueError(
                f"the Local example window {local_window} s is not within a pause: "
                "the simulated behavior has changed"
            )
        sweeping = "spikes sweep\nalong the track"
        return [
            Example(
                "Local",
                "spikes at the\nanimal's position",
                LOCAL_COLOR,
                None,
                local_window,
                flag_shift=-EXAMPLE_FLAG_SPREAD_S,
            ),
            Example(
                "Continuous",
                sweeping,
                CONTINUOUS_COLOR,
                sweep_1,
                around(sweep_1),
                flag_shift=EXAMPLE_FLAG_SPREAD_S,
            ),
            Example(
                "Fragmented",
                "spikes at\nscattered positions",
                FRAGMENTED_COLOR,
                scatter,
                around(scatter),
            ),
            Example("Continuous", sweeping, CONTINUOUS_COLOR, sweep_2, around(sweep_2)),
            Example(
                "No-Spike", "no spikes\n", NO_SPIKE_COLOR, silence, around(silence)
            ),
        ]


def detect_non_local_events(
    non_local_probability: xr.DataArray, speed: np.ndarray, time: np.ndarray
) -> list[tuple[float, float]]:
    """The non-local events of a decode, by the rule the analyses use.

    The two steps of ``analysis/detection/nonlocal_events.py``: runs of
    non-local probability above ``DEFAULT_NON_LOCAL_THRESHOLD`` that last
    ``NON_LOCAL_MINIMUM_DURATION_S`` (``get_confident_times``), kept only when
    the animal is immobile for the whole event
    (``build_speed_eligible_event_cohort``).

    Parameters
    ----------
    non_local_probability : xr.DataArray, shape (n_time,)
        P(Non-Local Continuous) + P(Non-Local Fragmented).
    speed : np.ndarray, shape (n_time,)
        The animal's speed, in cm/s.
    time : np.ndarray, shape (n_time,)
        The decode's time bins, in s.

    Returns
    -------
    list of (start, end)
        Each event's first and last time bin, in s.
    """
    detected = get_confident_times(
        non_local_probability.assign_coords(time=time), speed, time
    )
    events, _ = build_speed_eligible_event_cohort(
        detected,
        speed=speed,
        time=time,
        event_col="non_local_event",
        preserve_source_ids=False,
    )
    return list(
        zip(events["start_time"].tolist(), events["end_time"].tolist(), strict=True)
    )


def sort_neurons_by_place_field(
    spike_times: list[np.ndarray], detector: NonLocalSortedSpikesDetector
) -> list[np.ndarray]:
    """Order the neurons' spike times by place-field peak, for the rasters."""
    key = list(detector.encoding_model_.keys())[0]
    place_fields = np.asarray(detector.encoding_model_[key]["place_fields"])
    return [spike_times[i] for i in np.argsort(np.argmax(place_fields, axis=1))]


def simulate_and_decode(n_neurons: int = 30, seed: int = 0) -> DecodedSimulation:
    """Simulate a run with inserted spiking patterns and decode it.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate.
    seed : int
        Seed of ``make_simulated_data``'s random generator.

    Returns
    -------
    DecodedSimulation
    """
    print(f"Simulating {n_neurons} neurons...")
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=n_neurons, seed=seed)
    event_times = np.array(event_times, dtype=float)

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
    detector = make_example_detector()
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time=time,
        is_training=~is_event,
    )
    time = np.asarray(time)
    speed = np.asarray(speed).squeeze()
    non_local_probability = nonlocal_state_probability(
        results.acausal_state_probabilities
    ).squeeze()
    non_local_events = detect_non_local_events(non_local_probability, speed, time)

    # Every state that has a position (No-Spike has none, so its bins are NaN).
    by_state = results["acausal_posterior"].unstack("state_bins")
    posterior = np.asarray(by_state.sum("state").transpose("time", "position").values)

    # The colour scale is set by the non-local posterior inside the non-local
    # events. Local's posterior is more concentrated and would set a scale on
    # which a non-local event is faint; on this scale Local saturates.
    non_local_posterior = np.asarray(
        by_state.sel(state=["Non-Local Continuous", "Non-Local Fragmented"])
        .sum("state")
        .transpose("time", "position")
        .values
    )
    in_event = np.zeros(time.size, dtype=bool)
    for start, end in non_local_events:
        in_event |= (time >= start) & (time <= end)
    scale_from = non_local_posterior[in_event] if in_event.any() else posterior

    return DecodedSimulation(
        time=time,
        position=np.asarray(position).squeeze(),
        speed=speed,
        sorted_spikes=sort_neurons_by_place_field(spike_times, detector),
        state_probabilities=np.asarray(results["acausal_state_probabilities"].values),
        posterior=posterior,
        posterior_vmax=float(np.percentile(scale_from, 99)),
        position_bins=np.asarray(by_state.coords["position"].values),
        non_local_probability=np.asarray(non_local_probability.values),
        event_times=event_times,
        non_local_events=non_local_events,
    )


def _example_badge(color: str) -> dict:
    return {"boxstyle": "circle,pad=0.18", "facecolor": color, "edgecolor": "none"}


def _plot_posterior(
    ax: Axes,
    sim: DecodedSimulation,
    rows: slice,
    time: np.ndarray,
    *,
    linewidth: float,
    dashes: tuple[float, float] | None = None,
) -> None:
    """Posterior over position with the animal's position on top.

    ``time`` is the x coordinate of ``rows`` (s in panel d, ms in panel e), on
    an even grid.
    """
    time_step = time[1] - time[0]
    bin_size = sim.position_bins[1] - sim.position_bins[0]
    # Every decoded bin is given to the renderer, which averages the bins that
    # share a pixel ("antialiased"); none is dropped.
    ax.imshow(
        sim.posterior[rows].T,
        origin="lower",
        aspect="auto",
        extent=(
            time[0] - time_step / 2,
            time[-1] + time_step / 2,
            sim.position_bins[0] - bin_size / 2,
            sim.position_bins[-1] + bin_size / 2,
        ),
        cmap="bone_r",
        vmin=0,
        vmax=sim.posterior_vmax,
        interpolation="antialiased",
        rasterized=True,
    )
    (line,) = ax.plot(
        time, sim.position[rows], color=ANIMAL_POS_COLOR, linewidth=linewidth
    )
    if dashes:
        line.set_dashes(dashes)


def plot_simulated_run(
    fig: Figure, slot: SubplotSpec, sim: DecodedSimulation, examples: list[Example]
) -> None:
    """Panel d: raster, decoded position, non-local probability and speed.

    The raster is sorted by place-field position and sits directly above the
    decoded position, so the spikes can be read against the decode. Numbered
    flags mark the examples of panel e.
    """
    gs = slot.subgridspec(4, 1, height_ratios=[0.9, 0.9, 0.5, 0.42], hspace=0.22)
    ax_raster = fig.add_subplot(gs[0])
    ax_posterior = fig.add_subplot(gs[1], sharex=ax_raster)
    ax_non_local = fig.add_subplot(gs[2], sharex=ax_raster)
    ax_speed = fig.add_subplot(gs[3], sharex=ax_raster)
    n_neurons = len(sim.sorted_spikes)

    ax_raster.eventplot(
        sim.sorted_spikes,
        lineoffsets=np.arange(1, n_neurons + 1),
        linelengths=0.85,
        colors="black",
        linewidths=0.3,
        rasterized=True,
    )
    ax_raster.set_ylim(0.3, n_neurons + 0.7)
    ax_raster.set_yticks([1, n_neurons])
    ax_raster.set_ylabel("Neuron, by\nfield position", fontsize=LABEL_SIZE, labelpad=3)
    for k, example in enumerate(examples):
        ax_raster.annotate(
            str(k + 1),
            xy=(example.flag_time, 1.0),
            xytext=(example.flag_time + example.flag_shift, 1.2),
            xycoords=ax_raster.get_xaxis_transform(),
            textcoords=ax_raster.get_xaxis_transform(),
            fontsize=SMALL_SIZE,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            annotation_clip=False,
            bbox=_example_badge(example.color),
            arrowprops={
                "arrowstyle": "-",
                "color": example.color,
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    _plot_posterior(
        ax_posterior, sim, slice(None), sim.time, linewidth=0.6, dashes=(2.5, 3.5)
    )
    ax_posterior.set_ylabel("Decoded\nposition (cm)", fontsize=LABEL_SIZE, labelpad=3)
    ax_posterior.set_ylim(sim.position_bins.min(), sim.position_bins.max())
    # Labeled where the animal first turns at the track's start: the space above
    # that turn is empty.
    ax_posterior.text(
        sim.next_time_at_track_start(examples[1].window[1]),
        0.62 * sim.position_bins.max(),
        "animal's\nposition",
        fontsize=TICK_SIZE,
        color=ANIMAL_POS_COLOR,
        ha="center",
        va="center",
        linespacing=0.95,
    )

    for start, end in sim.non_local_events:
        ax_non_local.axvspan(start, end, color=NON_LOCAL_COLOR, alpha=0.13, linewidth=0)
    ax_non_local.plot(
        sim.time, sim.non_local_probability, color=NON_LOCAL_COLOR, linewidth=1.0
    )
    ax_non_local.axhline(
        DEFAULT_NON_LOCAL_THRESHOLD, color="#999999", linestyle="--", linewidth=0.6
    )
    ax_non_local.text(
        sim.non_local_events[0][1] + 1.2,  # just after the first event
        DEFAULT_NON_LOCAL_THRESHOLD - 0.06,
        f"event threshold, {DEFAULT_NON_LOCAL_THRESHOLD:g}",
        fontsize=TICK_SIZE,
        color="#555555",
        ha="left",
        va="top",
    )
    ax_non_local.set_ylim(0, 1.05)
    ax_non_local.set_yticks([0, 1])
    ax_non_local.set_ylabel("Non-local\nprob.", fontsize=LABEL_SIZE, labelpad=3)
    definition = HPacker(
        children=[
            TextArea(text, textprops={"fontsize": SMALL_SIZE, "color": color})
            for text, color in (
                ("P(non-local) = ", NON_LOCAL_COLOR),
                ("P(Continuous)", CONTINUOUS_COLOR),
                (" + ", NON_LOCAL_COLOR),
                ("P(Fragmented)", FRAGMENTED_COLOR),
            )
        ],
        align="baseline",
        pad=0,
        sep=1.0,
    )
    ax_non_local.add_artist(
        AnchoredOffsetbox(
            loc="center right",
            child=definition,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(1.0, 0.42),
            bbox_transform=ax_non_local.transAxes,
            borderpad=0.0,
        )
    )

    ax_speed.fill_between(sim.time, sim.speed, color="#999999", linewidth=0)
    ax_speed.axhline(
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        color="black",
        linewidth=0.5,
        linestyle=":",
    )
    # Labeled in the pause that holds the last example.
    ax_speed.text(
        examples[-1].flag_time,
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC + 3,
        f"{DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC:g} cm/s",
        fontsize=TICK_SIZE,
        color="#333333",
        ha="center",
        va="bottom",
    )
    ax_speed.set_ylim(0, None)
    ax_speed.set_ylabel("Speed\n(cm/s)", fontsize=LABEL_SIZE, labelpad=3)
    ax_speed.set_xlabel("Time (s)", fontsize=LABEL_SIZE, labelpad=2)

    # The run is shown until the animal is back at the track's start after the
    # last example; the pauses after that hold no example.
    overview_end = sim.next_time_at_track_start(examples[-1].window[1])
    axes = (ax_raster, ax_posterior, ax_non_local, ax_speed)
    for ax in axes:
        ax.set_xlim(sim.time[0], overview_end)
        ax.tick_params(labelsize=TICK_SIZE, length=2)
        _hide_top_right(ax)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def plot_examples(
    fig: Figure, slot: SubplotSpec, sim: DecodedSimulation, examples: list[Example]
) -> None:
    """Panel e: raster, decoded position and state probabilities per example.

    Each column is one example on its own millisecond axis, with zero at the
    start of the simulated pattern (the start of the window for Local). A bar
    above the raster marks when the simulated pattern occurs; shading marks the
    non-local events (``detect_non_local_events``).
    """
    gs = slot.subgridspec(
        3, len(examples), height_ratios=[1.0, 1.0, 0.75], hspace=0.14, wspace=0.12
    )
    n_neurons = len(sim.sorted_spikes)
    for k, example in enumerate(examples):
        start, end = example.window
        origin = example.origin
        rows = slice(*np.searchsorted(sim.time, [start, end]))
        time_ms = (sim.time[rows] - origin) * 1e3
        ax_raster = fig.add_subplot(gs[0, k])
        ax_posterior = fig.add_subplot(gs[1, k], sharex=ax_raster)
        ax_states = fig.add_subplot(gs[2, k], sharex=ax_raster)
        axes = (ax_raster, ax_posterior, ax_states)

        ax_raster.eventplot(
            [
                (spikes[(spikes >= start) & (spikes <= end)] - origin) * 1e3
                for spikes in sim.sorted_spikes
            ],
            lineoffsets=np.arange(1, n_neurons + 1),
            linelengths=0.85,
            colors="black",
            linewidths=0.6,
        )
        ax_raster.set_ylim(0.3, n_neurons + 0.7)
        ax_raster.set_yticks([1, n_neurons])
        if example.pattern is not None:
            ax_raster.plot(
                [0, (example.pattern[1] - origin) * 1e3],
                [1.05, 1.05],
                color=example.color,
                linewidth=2.2,
                solid_capstyle="butt",
                transform=ax_raster.get_xaxis_transform(),
                clip_on=False,
            )
        ax_raster.text(
            0.5,
            1.47,
            example.state,
            transform=ax_raster.transAxes,
            fontsize=LABEL_SIZE,
            fontweight="bold",
            color=example.color,
            ha="center",
            va="center",
        )
        ax_raster.text(
            0.5,
            1.39,
            example.spiking,
            transform=ax_raster.transAxes,
            fontsize=TICK_SIZE,
            color="#444444",
            ha="center",
            va="top",
            linespacing=1.0,
        )
        ax_raster.text(
            0.03,
            1.47,
            str(k + 1),
            transform=ax_raster.transAxes,
            fontsize=SMALL_SIZE,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=_example_badge(example.color),
        )

        _plot_posterior(ax_posterior, sim, rows, time_ms, linewidth=0.8)
        ax_posterior.set_ylim(sim.position_bins.min(), sim.position_bins.max() + 4)

        for i, (name, color) in enumerate(
            zip(STATE_NAMES_SHORT, STATE_COLORS_SHORT, strict=True)
        ):
            ax_states.plot(
                time_ms,
                sim.state_probabilities[rows, i],
                color=color,
                linewidth=1.0,
                label=name,
            )
        ax_states.set_ylim(0, 1.05)
        ax_states.set_yticks([0, 1])
        ax_states.set_xlabel("Time (ms)", fontsize=LABEL_SIZE, labelpad=2)

        for nl_start, nl_end in sim.non_local_events:
            if nl_end > start and nl_start < end:
                for ax in axes:
                    ax.axvspan(
                        (nl_start - origin) * 1e3,
                        (nl_end - origin) * 1e3,
                        color=NON_LOCAL_COLOR,
                        alpha=0.13,
                        linewidth=0,
                        zorder=0,
                    )
        for ax in axes:
            ax.set_xlim((start - origin) * 1e3, (end - origin) * 1e3)
            ax.tick_params(labelsize=TICK_SIZE, length=2)
            _hide_top_right(ax)
            if k > 0:
                ax.tick_params(labelleft=False)
        ax_raster.tick_params(labelbottom=False)
        ax_posterior.tick_params(labelbottom=False)

        if k == 0:
            ax_raster.set_ylabel("Neuron", fontsize=LABEL_SIZE, labelpad=3)
            ax_posterior.set_ylabel(
                "Decoded\nposition (cm)", fontsize=LABEL_SIZE, labelpad=3
            )
            ax_states.set_ylabel("State\nprobability", fontsize=LABEL_SIZE, labelpad=3)
            ax_states.text(
                0.0,
                -0.8,
                "Bar: when the simulated spike pattern occurs.\n"
                "Shading: a non-local event (non-local probability above "
                f"{DEFAULT_NON_LOCAL_THRESHOLD:g} for at least "
                f"{NON_LOCAL_MINIMUM_DURATION_S * 1e3:g} ms while the animal is "
                "still).",
                fontsize=TICK_SIZE,
                color="#555555",
                ha="left",
                va="top",
                transform=ax_states.transAxes,
            )
        if k == len(examples) - 1:
            ax_states.legend(
                fontsize=TICK_SIZE,
                frameon=False,
                ncol=4,
                handlelength=1.0,
                columnspacing=0.8,
                loc="upper right",
                bbox_to_anchor=(1.0, -0.72),
                borderaxespad=0,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Main Figure Assembly
# ═══════════════════════════════════════════════════════════════════════════


def _axes_added(fig: Figure, draw: Callable[[], None]) -> list[Axes]:
    """Run ``draw`` and return the axes it added to ``fig``, in order."""
    before = set(fig.axes)
    draw()
    return [ax for ax in fig.axes if ax not in before]


def _drawn_extent(fig: Figure, axes: Sequence[Axes]) -> Bbox:
    """Bounding box of everything ``axes`` draw (labels included), in figure fractions."""
    return Bbox.union([ax.get_tightbbox() for ax in axes]).transformed(
        fig.transFigure.inverted()
    )


def _shift_vertically(axes: Sequence[Axes], dy: float) -> None:
    for ax in axes:
        box = ax.get_position(original=True)
        ax.set_position([box.x0, box.y0 + dy, box.width, box.height])


def generate_figure(n_neurons: int = 30, seed: int = 0) -> Figure:
    """Generate complete Figure 1.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate.
    seed : int
        Seed of ``make_simulated_data``'s random generator. The default, 0, is
        the simulator's own default, which the figure has always been drawn
        with.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sim = simulate_and_decode(n_neurons=n_neurons, seed=seed)
    examples = sim.examples()

    set_figure_defaults(context="paper")
    # The rows are placed by hand below, from what each one draws, so the
    # automatic layout the paper defaults switch on must be off.
    fig = plt.figure(figsize=(TWO_COLUMN, FIGURE_HEIGHT), dpi=450, layout="none")

    # Provisional vertical positions; the rows are re-spaced below.
    gs_bc = fig.add_gridspec(
        1,
        2,
        left=0.31,
        right=0.955,
        top=0.93,
        bottom=0.755,
        width_ratios=[2.5, 1.45],
        wspace=0.25,
    )
    gs_d = fig.add_gridspec(
        1, 1, left=DATA_LEFT, right=DATA_RIGHT, top=0.66, bottom=0.375
    )
    gs_e = fig.add_gridspec(
        1, 1, left=DATA_LEFT, right=DATA_RIGHT, top=0.30, bottom=0.055
    )
    axes_b = _axes_added(fig, lambda: plot_spike_scoring(fig, gs_bc[0]))
    axes_c = _axes_added(fig, lambda: plot_position_transitions(fig, gs_bc[1]))
    axes_d = _axes_added(fig, lambda: plot_simulated_run(fig, gs_d[0], sim, examples))
    axes_e = _axes_added(fig, lambda: plot_examples(fig, gs_e[0], sim, examples))
    fig.align_ylabels([*axes_d, *axes_e[:3]])
    fig.draw_without_rendering()

    # Panel a is as tall as panel b's drawing (the top of its plots to the bottom
    # of its axis label) and starts at the left edge of panel d's labels.
    top = axes_b[0].get_position().y1
    bottom = _drawn_extent(fig, axes_b).y0
    width = (top - bottom) * FIGURE_HEIGHT / STATE_DIAGRAM_HEIGHT / TWO_COLUMN
    ax_a = fig.add_axes((_drawn_extent(fig, axes_d).x0, bottom, width, top - bottom))
    plot_state_diagram(ax_a)
    fig.draw_without_rendering()

    # Equal gaps between the rows, measured from one row's drawing to the next's.
    rows = [[ax_a, *axes_b, *axes_c], axes_d, axes_e]
    extents = [_drawn_extent(fig, row) for row in rows]
    margin = 0.012
    gap = (1 - 2 * margin - sum(extent.height for extent in extents)) / (len(rows) - 1)
    row_top = 1 - margin
    for row, extent in zip(rows, extents, strict=True):
        _shift_vertically(row, row_top - extent.y1)
        row_top -= extent.height + gap
    fig.draw_without_rendering()

    # Panel letters: a, d and e share a left edge; a, b and c share a top edge.
    extents = [_drawn_extent(fig, row) for row in rows]
    left = min(extent.x0 for extent in extents)
    for letter, x, y in (
        ("a", left, extents[0].y1),
        ("b", _drawn_extent(fig, axes_b).x0, extents[0].y1),
        ("c", _drawn_extent(fig, axes_c).x0, extents[0].y1),
        ("d", left, extents[1].y1),
        ("e", left, extents[2].y1),
    ):
        fig.text(
            x,
            y,
            letter,
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
            ha="left",
            va="top",
        )

    return fig


def main(argv: Sequence[str] | None = None) -> None:
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the simulated data's random generator (default 0).",
    )
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = generate_figure(n_neurons=args.n_neurons, seed=args.seed)

    print(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure01", base_dir=args.output_dir, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
