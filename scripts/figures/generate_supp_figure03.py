#!/usr/bin/env python
"""Generate Supplementary Figure S3, the supplement to Figure 3: example events
of each kind, how fast their position distribution moves, which of their
content is at a well far from the animal and in which patch, how many wells an
event represents and how many patches it spans, where its content starts and
ends, and slow and fast gamma during them.

The immobile non-local events of the canonical event table that start inside
a well visit (``cohort_selection.select_visit_context``), split by whether
they overlap an SWR. The panels follow the text of Figure 3.

a — Six example events of the example session (:data:`EXAMPLES`,
    ``parameters.EXAMPLE_EPOCH``), three with no SWR whose content stays at one
    remote place and three with an SWR whose content extends, each a column as
    in Figure 2b (``visualization.example_event_panels.draw_examples``): the
    ripple trace with the SWR threshold and the SWRs, the tetrodes' spikes with
    the event shaded, the decoded position with the animal's (magenta) and the
    linearized track, and the state probabilities.

b — How fast the position distribution moves, no-SWR against SWR events: each
    event's transport-slope speed (``event_speed.transport_slope_speed``; the
    event-speed run, ``scripts/event_properties/measure_event_speed.py``), the
    slope, at least 0, of the earth mover's distance on the track between the
    Continuous-state position distributions of two of its time bins against
    their lag, over lags of 10 to 20 ms. Figure 3d draws the speed of the MAP
    position; this one reads the whole distribution and sets aside the
    bin-to-bin difference that does not grow with the lag. The percent of a
    class's events at or below each speed (the bins of
    ``event_statistics.TRANSPORT_SPEED_BIN_EDGES_CM_S`` summed), a thin line
    per animal and their mean in bold
    (``event_statistics.transport_speed_distribution``), over the events that
    have a speed: one too short for three lags, or with too few time bins that
    are mostly Continuous, has none.

c — The share of each kind's content time at a well, apart for the wells far
    from the animal (remote) and the animal's own: every one of these events
    happens with the animal at a well, so content at "a well" (Figure 3f) could
    be content at the animal's. Each animal's share for no-SWR and for SWR
    events, joined by a line, and the animals' mean (a bar)
    (``content_along_arms.content_place_shares`` of the content-mass run, the
    events summary's ``*_content_remote_well_fraction`` and
    ``*_content_own_well_fraction``; a well is within 20 cm, far is more than
    20 cm from the animal along the track), with the mean's 95 % interval as a
    line beside the dots (``content_along_arms.place_share_intervals``).

d — Whether an event represents one location: the share of each kind's events
    that represent no well, one well, and two or more, where an event
    represents a well when at least a tenth of its content time
    (``content_along_arms.WELL_CONTENT_SHARE``) is within 20 cm of it; bars the
    animals' mean, dots the animals (``content_along_arms.well_count_shares``).

e — Which patch the content is in: the share of each kind's content time in
    the animal's own patch (the patch of the well it is at) and in the other
    two, which lie beyond the center junction, at the wells and off them; bars
    the animals' mean, dots the animals
    (``content_along_arms.patch_content_shares``). Content spread evenly over
    the track would put a third in the animal's patch.

f — How much of the track an event spans: the share of each kind's events
    whose content is in one patch, in two and in all three, a patch counting
    when it holds at least a tenth of the event's content time
    (``content_along_arms.event_patch_counts``, ``patch_count_shares``).

g — Where content starts and where it ends: the share of each kind's events
    whose content at the first time bin, and at the last, is near the animal
    (within 20 cm) and at a remote well (``content_along_arms.event_start_end``,
    ``start_end_shares``; the decode's most probable place at that time bin).

h — Whether the events whose content stays in one place are the ones at a
    remote well: the mean share of an event's content time at a remote well, by
    the event's spatial extent (the classes of
    ``content_along_arms.EXTENT_BIN_EDGES_CM``; Figure 3c draws the extent), a
    thin line per animal and their mean in bold
    (``content_along_arms.remote_well_share_by_extent``: event, then epoch,
    then animal).

i, j — Whether the two kinds differ in the local field potential: for each
    band of ``gamma_power.GAMMA_BANDS`` (i, slow gamma; j, fast gamma) and each
    kind of event, every animal's level during the events, between the levels
    of the same epochs' quiet-rest and running windows as long as the kind's
    events; quiet rest is the rest inside a well visit. An event's gamma is a
    mean over its bins, and a mean of skewed power over a window sits above the
    median of single bins, so the reference is windows of the events' own
    length. The animal values are ``event_statistics.animal_signal_levels``
    with ``matched``: the values the events summary's ``<band>_animals_*``
    counts compare (``paper/reported/events.json``; the gamma run is
    ``scripts/event_properties/measure_event_gamma.py``). Ripple power, MUA,
    theta and head speed around the events are Figure 2's.

Usage
-----
    python scripts/figures/generate_supp_figure03.py
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    EXTENT_BIN_EDGES_CM,
    PATCH_CONTENT_CLASSES,
    PATCH_COUNT_CLASSES,
    START_END_CLASSES,
    WELL_COUNT_CLASSES,
    content_place_shares,
    patch_content_shares,
    patch_count_shares,
    place_share_intervals,
    remote_well_share_by_extent,
    start_end_shares,
    well_count_shares,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    TRANSPORT_SPEED_BIN_EDGES_CM_S,
    animal_signal_levels,
    transport_speed_distribution,
)
from continuum_swr_replay.analysis.event_properties.gamma_power import GAMMA_BANDS
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    time_mask_staleness,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.parameters import EXAMPLE_EPOCH, EXAMPLE_NWB_FILE
from continuum_swr_replay.reported.events import (
    content_mass_along_arms,
    content_mass_by_patch,
    with_brain_state,
    with_event_place_shares,
    with_event_speed,
    with_event_start_end,
)
from continuum_swr_replay.reported.inputs import default_inputs, read_example_decode
from continuum_swr_replay.runs.staleness import read_output_manifest
from continuum_swr_replay.visualization.example_event_panels import draw_examples
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.heldout_gain_panels import (
    GROUP_COLORS,
    name_groups,
)
from continuum_swr_replay.visualization.palettes import ANIMAL_LINE_COLOR, EVENT_KINDS
from continuum_swr_replay.visualization.panel_layout import place_letters
from continuum_swr_replay.visualization.publication_style import (
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    set_publication_style,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

# Inches: the full page width.
WIDTH = 6.5
# The journal's page is 20 cm tall with the saving margin.
HEIGHT = 7.6

# The rest and running levels: neutral, not decoder-state colors.
C_REST = "white"
C_RUN = "black"
# Each panel: (band, event column, title).
SIGNALS = tuple(
    (band, f"{band}_z", f"{name} gamma ({low:.0f}–{high:.0f} Hz)")
    for name, (band, (low, high)) in zip(
        ("Slow", "Fast"), GAMMA_BANDS.items(), strict=True
    )
)


def read_inputs() -> dict:
    """The events with their gamma, each epoch's rest and running levels, the
    selected events' ``transport_speed_distribution`` (``speed``), each animal's
    shares of content time by place (``places``:
    ``content_along_arms.content_place_shares``) and its
    ``remote_well_share_by_extent`` (``by_extent``).

    Raises
    ------
    ValueError
        If the brain-state or the gamma reference was not made over the
        analyses' subset or from the current session cache's well visits
        (``prepare_data.time_mask_staleness``).
    """
    inputs = default_inputs()
    events = select_visit_context(read_swr_table(inputs.event_table))
    selected = events
    events, reference, gamma_reference = with_brain_state(
        events, inputs, len(analyzed_epoch_ids())
    )
    for folder, levels in (
        (inputs.brain_state_reference.parent, reference),
        (inputs.event_gamma, gamma_reference),
    ):
        note = time_mask_staleness(
            read_output_manifest(folder), levels["epoch_id"], str(folder)
        )
        if note is not None:
            raise ValueError(note)
    # panel c: the selected events, as the events summary reads them
    n_epochs = len(analyzed_epoch_ids())
    with_speed, _ = with_event_speed(selected, inputs.event_speed, n_epochs)
    # panels d and e: the content-mass run, as Figure 3 e and f read it
    along, _ = content_mass_along_arms(
        selected, inputs.content_mass_along_arms, n_epochs
    )
    with_places, _ = with_event_place_shares(
        selected, inputs.content_mass_along_arms, n_epochs
    )
    by_patch, _ = content_mass_by_patch(
        selected, inputs.content_mass_along_arms, n_epochs
    )
    # where content starts and ends: the decode's place at the first and last bin
    with_ends, _ = with_event_start_end(
        with_places, inputs.content_along_arms, n_epochs
    )
    return {
        "events": events,
        "reference": reference,
        "speed": transport_speed_distribution(with_speed),
        "places": content_place_shares(along),
        "place_intervals": place_share_intervals(along),
        "well_counts": well_count_shares(with_places, column="content_n_wells"),
        "patch_counts": patch_count_shares(with_places, column="content_n_patches"),
        "patch_content": patch_content_shares(by_patch),
        "start_end": start_end_shares(with_ends),
        "session": read_example_decode(EXAMPLE_NWB_FILE, EXAMPLE_EPOCH),
        "examples": example_windows(selected),
        "by_extent": remote_well_share_by_extent(
            with_places, column="content_remote_well"
        ),
    }


def draw_gamma(axes, events: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Each animal's level of each band in each kind of event, between the
    levels of quiet-rest (white) and running (black) windows as long as the
    kind's events (``axes``: one per band)."""
    for ax, (band, column, title) in zip(axes, SIGNALS, strict=True):
        levels = animal_signal_levels(events, reference, band, column, matched=True)
        centers = []
        for k, (is_swr, (_, color)) in enumerate(EVENT_KINDS.items()):
            x = 3.7 * k + np.arange(3)
            values = levels.loc[is_swr, ["rest", "event", "running"]]
            for _, row in values.iterrows():
                ax.plot(x, row.to_numpy(), color=ANIMAL_LINE_COLOR, lw=0.5, zorder=1)
            for i, dot_color in enumerate((C_REST, color, C_RUN)):
                ax.scatter(
                    np.full(len(values), x[i]), values.iloc[:, i], s=9,
                    color=dot_color, edgecolors="black", linewidths=0.5, zorder=2,
                )  # fmt: skip
            centers.append(x[1])
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=SMALL_SIZE)
        # One label per kind; the key says which dots are rest and running.
        ax.set_xticks(centers, [name for name, _ in EVENT_KINDS.values()])
        ax.set_xlim(-0.7, centers[-1] + 1.7)
    axes[0].set_ylabel("Power (z)")


SPEED_TICKS_CM_S = (0, 500, 1000)


def draw_transport_speed(ax: Axes, distribution: pd.DataFrame) -> None:
    """Panel b, the speed panel: for no-SWR and SWR events, the percent of the
    class's events (with a speed) whose transport-slope speed is at most each
    speed, a thin line per animal and their mean in bold in the class's color
    (``distribution``: :func:`read_inputs`' ``speed``)."""
    for group, color in GROUP_COLORS.items():
        mine = distribution[distribution["group"] == group]
        by_animal = mine.pivot(index="bin_high", columns="animal", values="percent")
        within = by_animal.cumsum()
        edges = within.index.to_numpy()
        ax.plot(edges, within.to_numpy(), color=color, lw=0.5, alpha=0.5)
        ax.plot(edges, within.mean(axis=1).to_numpy(), color=color, lw=1.2)
    name_groups(ax, 0.98, 0.3, 0.1, "right")
    ax.set_xlim(TRANSPORT_SPEED_BIN_EDGES_CM_S[0], TRANSPORT_SPEED_BIN_EDGES_CM_S[-1])
    ax.set_xticks(SPEED_TICKS_CM_S)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Speed (cm/s)")
    ax.set_ylabel("Cumulative %\nof events")
    ax.set_title("Distribution's speed", fontsize=SMALL_SIZE)


WELL_PLACES = (("remote_well", "remote well"), ("own_well", "own well"))
"""Panel d's two places and their names under the axis."""


def draw_well_shares(
    ax: Axes, shares: pd.DataFrame, intervals: pd.DataFrame | None = None
) -> None:
    """Panel c: each animal's share of no-SWR and of SWR content time at a
    remote well and at its own well, the two kinds joined by a line per animal,
    and the animals' mean as a bar (``shares``: :func:`read_inputs`'
    ``places``, indexed by ``is_swr`` and ``animal``); with ``intervals``
    (``content_along_arms.place_share_intervals``), the mean's 95 % interval
    as a line beside the dots."""
    for k, (column, _) in enumerate(WELL_PLACES):
        # one row per animal; columns no-SWR, SWR (the order of EVENT_KINDS)
        by_kind = 100 * shares[column].unstack("is_swr").to_numpy()
        x = 2.6 * k + np.arange(2)
        ax.plot(x, by_kind.T, color=ANIMAL_LINE_COLOR, lw=0.5, zorder=1)
        for i, (_, color) in enumerate(EVENT_KINDS.values()):
            ax.scatter(
                np.full(len(by_kind), x[i]), by_kind[:, i], s=9, color=color,
                edgecolors="black", linewidths=0.5, zorder=2,
            )  # fmt: skip
            if intervals is not None:
                row = intervals.loc[(("non_swr", "swr")[i], column)]
                side = x[i] + (-0.42 if i == 0 else 0.42)
                ax.plot(
                    [side, side], [100 * row["ci_low"], 100 * row["ci_high"]],
                    color=color, lw=0.8, zorder=3, gid="interval",
                )  # fmt: skip
            mean = by_kind[:, i].mean()
            ax.plot(
                [x[i] - 0.3, x[i] + 0.3], [mean, mean], color=color, lw=1.5, zorder=3
            )
    ax.set_xticks([0.5, 3.1], [name for _, name in WELL_PLACES])
    ax.set_xlim(-0.7, 4.3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Content time (%)")
    ax.set_title("At a well", fontsize=SMALL_SIZE)
    name_groups(ax, 0.98, 1.0, 0.1, "right")


def draw_remote_well_by_extent(ax: Axes, by_extent: pd.DataFrame) -> None:
    """Panel h: the mean share of an event's content time at a remote well in
    each spatial-extent class, a thin line per animal and their mean in bold in
    the kind's color (``by_extent``: :func:`read_inputs`' ``by_extent``)."""
    n_classes = len(EXTENT_BIN_EDGES_CM) - 1
    for is_swr, (_, color) in EVENT_KINDS.items():
        by_animal = (
            100 * by_extent.loc[is_swr, "remote_well"].unstack("animal")
        ).reindex(range(n_classes))
        ax.plot(by_animal.index, by_animal.to_numpy(), color=color, lw=0.5, alpha=0.5)
        ax.plot(by_animal.index, by_animal.mean(axis=1), color=color, lw=1.2)
    edges = EXTENT_BIN_EDGES_CM
    labels = [f"<{edges[1]:g}"]
    labels += [
        f"{low:g}–{high:g}" for low, high in zip(edges[1:-2], edges[2:-1], strict=True)
    ]
    labels += [f"≥{edges[-2]:g}"]
    ax.set_xticks(range(n_classes), labels)
    ax.set_xlim(-0.3, n_classes - 0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Event's spatial extent (cm)")
    ax.set_ylabel("Of the event, at a\nremote well (%)")
    ax.set_title("Remote well, by extent", fontsize=SMALL_SIZE)


def draw_class_bars(
    ax: Axes,
    shares: pd.DataFrame,
    classes: Sequence[str],
    labels: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    """A share per class and kind: bars the animals' mean, dots the animals
    (``shares``: one column per class, indexed by ``is_swr`` and ``animal``)."""
    for g, (is_swr, (_, color)) in enumerate(EVENT_KINDS.items()):
        mine = 100 * shares.loc[is_swr]
        for x, name in enumerate(classes):
            position = x + (g - 0.5) * 0.36
            ax.bar(position, mine[name].mean(), width=0.34, color=color)
            ax.scatter(
                position + np.linspace(-0.08, 0.08, len(mine)), mine[name], s=3,
                color="k", lw=0, zorder=3,
            )  # fmt: skip
    ax.set_xticks(range(len(classes)), labels)
    ax.set_xlim(-0.55, len(classes) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=SMALL_SIZE)


def draw_well_counts(ax: Axes, shares: pd.DataFrame) -> None:
    """Panel d: the share of each kind's events that represent no well, one
    well, and two or more (``shares``: :func:`read_inputs`' ``well_counts``)."""
    draw_class_bars(
        ax, shares, WELL_COUNT_CLASSES, ["none", "one", "two\nor more"],
        "Wells an event represents", "Events (%)",
    )  # fmt: skip


def draw_patch_content(ax: Axes, shares: pd.DataFrame) -> None:
    """Panel e: the share of each kind's content time in the animal's own patch
    and in the other two, at the wells and elsewhere (``shares``:
    :func:`read_inputs`' ``patch_content``)."""
    draw_class_bars(
        ax, shares, PATCH_CONTENT_CLASSES,
        ["wells", "other", "wells", "other"],
        "Content by patch", "Content time (%)",
    )  # fmt: skip
    for x, name in ((0.5, "own patch"), (2.5, "other patches")):
        ax.text(
            x, -0.2, name, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=SMALL_SIZE, color="0.35",
        )  # fmt: skip
    ax.axvline(1.5, color="0.8", lw=0.5, zorder=0)


def draw_patch_counts(ax: Axes, shares: pd.DataFrame) -> None:
    """Panel f: the share of each kind's events whose content spans one patch,
    two and three (``shares``: :func:`read_inputs`' ``patch_counts``)."""
    draw_class_bars(
        ax, shares, PATCH_COUNT_CLASSES, ["one", "two", "three"],
        "Patches an event spans", "Events (%)",
    )  # fmt: skip


def draw_start_end(ax: Axes, shares: pd.DataFrame) -> None:
    """Panel g: the share of each kind's events whose content starts, and
    ends, near the animal and at a remote well (``shares``:
    :func:`read_inputs`' ``start_end``)."""
    draw_class_bars(
        ax, shares, START_END_CLASSES, ["starts", "ends", "starts", "ends"],
        "Where content starts and ends", "Events (%)",
    )  # fmt: skip
    for x, name in ((0.5, "near the animal"), (2.5, "at a remote well")):
        ax.text(
            x, -0.2, name, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=SMALL_SIZE, color="0.35",
        )  # fmt: skip
    ax.axvline(1.5, color="0.8", lw=0.5, zorder=0)


EXAMPLES = {
    False: (7156386530554463803, 5354495646056365027, 611840950996746651),
    True: (3352114973072553739, 225761306712118988, 7036534662482105035),
}
"""Panel a's example events of the example session (``parameters.EXAMPLE_EPOCH``)
by kind, chosen by eye from a sheet of its candidates: events with no SWR whose
content stays at one remote place (extent at most 10 cm, at least nine tenths
of the content time at a remote well, HPD coverage under 30 cm, 40 ms or
longer), and events with an SWR whose content extends (extent at least 60 cm,
Fragmented under a tenth, 80 ms or longer). :func:`example_windows` checks each
one's kind."""


def example_windows(events: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Panel a's examples as (start, end) seconds by title, no-SWR ones first;
    a title is the kind's name and the event's duration.

    Raises
    ------
    AssertionError
        If an example is not one row of ``events``, or is not of its kind.
    """
    windows = {}
    for is_swr, event_ids in EXAMPLES.items():
        for event_id in event_ids:
            rows = events[events["event_id"] == event_id]
            if len(rows) != 1:
                raise AssertionError(f"event_id {event_id} matches {len(rows)} rows")
            row = rows.iloc[0]
            if bool(row["is_swr"]) != is_swr:
                raise AssertionError(f"event {event_id} is not of its kind")
            title = f"{EVENT_KINDS[is_swr][0]}, {1e3 * row['event_duration_s']:.0f} ms"
            if title in windows:
                raise AssertionError(f"two examples are titled {title!r}")
            windows[title] = (float(row["event_start_s"]), float(row["event_end_s"]))
    return windows


def generate_figure(inputs: dict):
    """Build the figure from :func:`read_inputs`."""
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[1.5, 1.0, 1.0, 1.0], hspace=0.62,
        left=0.09, right=0.99, top=0.975, bottom=0.055,
    )  # fmt: skip
    session = inputs["session"]
    max_pos = float(np.ceil(session["data"]["position_info"].linear_position.max()))
    ax_a = draw_examples(fig, outer[0], session, inputs["examples"], max_pos)
    rows = [outer[k].subgridspec(1, 3, wspace=0.5) for k in (1, 2, 3)]
    ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = (
        fig.add_subplot(row[k]) for row in rows for k in range(3)
    )
    draw_transport_speed(ax_b, inputs["speed"])
    draw_well_shares(ax_c, inputs["places"], inputs["place_intervals"])
    draw_well_counts(ax_d, inputs["well_counts"])
    draw_patch_content(ax_e, inputs["patch_content"])
    draw_patch_counts(ax_f, inputs["patch_counts"])
    draw_start_end(ax_g, inputs["start_end"])
    draw_remote_well_by_extent(ax_h, inputs["by_extent"])
    draw_gamma((ax_i, ax_j), inputs["events"], inputs["reference"])
    # the gamma panels' key, inside the first of them
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=3, mfc=color, mec="black", mew=0.5)
        for color in (C_REST, C_RUN)
    ]
    ax_i.legend(
        handles, ["quiet rest", "running"], frameon=False, loc="upper left",
        handletextpad=0.1, borderaxespad=0.0, fontsize=SMALL_SIZE,
    )  # fmt: skip
    fig.draw_without_rendering()
    panels = {
        "a": [ax_a], "b": [ax_b], "c": [ax_c], "d": [ax_d], "e": [ax_e],
        "f": [ax_f], "g": [ax_g], "h": [ax_h], "i": [ax_i], "j": [ax_j],
    }  # fmt: skip
    place_letters(
        fig, panels, rows=["a", "bcd", "efg", "hij"], columns=["abeh", "cfi", "dgj"]
    )
    check_placement(fig, panels)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/supp_figure03")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_figure(read_inputs())
    check_journal_limits(fig)
    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "supp_figure03", base_dir=args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
