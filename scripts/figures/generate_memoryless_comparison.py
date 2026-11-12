#!/usr/bin/env python
"""Supplemental figure: two replay detectors on replays whose shape is known.

Upper panels run the ported Widloski & Foster sliding-window detector and this
project's switching state-space model over the Figure 1 simulation (25 neurons
here where Figure 1 uses 30, and a plain fit rather than EM), so the two
procedures can be read against a handful of labeled events.

Lower panels summarize the sweep in `scripts/detection/sweep_memoryless_detection.py`,
which wrote replays of chosen path length, duration, firing rate and distance
from the animal one at a time into a simulated recording and recorded what each
detector reported. Those panels read the aggregates the sweep saves, so the
figure and the numbers quoted from the sweep cannot disagree; re-run the sweep
to change them.

The figure's caption is prose this script does not write, and no draft of it is
kept in the repository. Its numbers are reported values
(`continuum_swr_replay.reported.detection`): the sweep's from the sweeps' CSVs, the
labeled recording's (a-f) from the JSON this script writes
(:data:`RECORDING_NUMBERS_PATH`). Where the prose commits to a verdict
-- an event missed, declined, found, passing or failing the significance test --
the script checks it and stops rather than draw a figure the caption contradicts.

In both halves the two detectors read one encoding model, fitted once.
The upper panels' simulation and its decode are
``memoryless_sequence.comparison_simulation``'s, which the reference check
(``scripts/qa/compare_memoryless_to_reference.py``) shares.

Usage:
    python scripts/figures/generate_memoryless_comparison.py
"""

import argparse
import json
import string
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    sweep_statistics,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.comparison_simulation import (
    CONTINUOUS_INBOUND_EVENT,
    CONTINUOUS_OUTBOUND_EVENT,
    REMOTE_STATIONARY_EVENT,
    SHORT_PATH,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    STRAY_WINDOW_EVENT,
    STRAY_WINDOW_N_LOCAL,
    STRAY_WINDOW_STILL_S,
    simulate_and_decode,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.window_statistics import (
    NO_CENTER,
    dispersion_cm,
)
from continuum_swr_replay.figure_utilities import (
    TWO_COLUMN,
    WONG,
    save_figure,
    set_figure_defaults,
)
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_sweep_summary.csv"
)
"""Aggregates of the parameter sweep, drawn in the lower panels."""

DISTANCE_ORDER = ("(0, 40]", "(40, 80]", "(80, 120]", "(120, 200]")
"""Every bin the sweep's `pd.cut` can produce, in order. Panel (k) draws the
ones the summary holds and refuses a summary holding any other."""

CONTROLS_SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_controls_summary.csv"
)
"""Aggregates of the controls sweep: no replay, bursts, jumps, chains."""

RECORDING_NUMBERS_PATH = Path(
    "docs/design/replay_detection_comparison/memoryless_comparison_recording.json"
)
"""The numbers the caption states about the one labeled recording (a-f),
which this script computes; Part 1's reported values read them."""

HELDOUT_SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_heldout_summary.csv"
)
"""Aggregates of the held-out sweep: support by fold, estimator, class and truth."""

ROW_HEIGHTS = [0.80, 0.95, 1.05, 0.75, 0.75, 0.70]
"""Relative heights of the figure's rows: overview, events, sweep, timing,
controls, held-out evidence."""

HELDOUT_GROUPS = (
    ("replay,\nours only", ("ours_only",), "continuous_replay"),
    ("replay,\nshared", ("shared",), "continuous_replay"),
    ("jumping\nburst", ("ours_only", "shared"), "non_local"),
    ("place-less\nburst", ("ours_only", "shared"), "no_content"),
)
"""The kinds of event the held-out panels compare: by which detectors found
them and what they truly are."""

FOLD_HATCH = {"A": None, "B": None, "C": "////"}
"""Fold C's evaluation cells cover only half the track."""

OUTPUTS = (
    ("sliding_accepted", "sliding, before test", 0.4),
    ("sliding_published", "sliding, published", 1.0),
    ("state_space_canonical", "ours, non-local", 1.0),
    ("state_space_continuous", "ours, continuous", 0.4),
)
"""The four scored outputs of the controls, with their bar opacity."""

MULTIPART_CONDITIONS = (
    "jump_120ms",
    "jump_200ms",
    "chain_continuing",
    "chain_elsewhere",
)
"""The controls conditions whose events have two parts."""

SEGMENTATION_STYLE = (
    ("spanning", "0.2", "one event over all parts"),
    ("one_per_part", "0.55", "one event per part"),
    ("other", "0.8", "other"),
    ("parts_missing", "1.0", "a part missed"),
)
"""How a multi-part event was carved, as the stacked bars shade it."""

LETTER_RISE_PT = 9.0
"""Panel letters ride this far above the axes: the height of a panel title,
so letter and title share a line."""

LETTER_GAP_PT = 4.0
"""And this far left of whatever the panel draws furthest left."""

EVENT_LABELS = [
    "continuous\noutbound",
    "fragmented",
    "continuous\ninbound",
    "no spike",
    "local",
    "remote\nstationary*",
    "short path\n30 cm*",
    "short replay\n80 ms*",
    "one window\nfar away*",
]
"""Labels for the simulation's events. A star marks one added here; the rest
come with the package's simulation."""

NON_LOCAL_PROB_THRESHOLD = DEFAULT_NON_LOCAL_THRESHOLD
NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The project's non-local event threshold and nominal minimum, applied with a
timestamp-span test (:func:`state_space_intervals`): non-local probability
above 0.8 over a first-to-last span of at least 15 ms, i.e. 9 samples on this
script's 2 ms grid. ``get_confident_times`` counts samples instead (7 on the
real decode grid; ``docs/known_issues/nonlocal_minimum_duration_rounding.md``).
The span test is kept so the published comparison does not change.

The state-space model is otherwise left on the library's defaults. Figure 1
sets a discrete transition stickiness, but stickiness is a Dirichlet
concentration read only by the EM M-step, and this comparison calls `fit`
rather than `estimate_parameters`, so passing one would change nothing here
and would read as tuning.
"""

BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = WONG[:8]

SLIDING = "sliding-window detector"
STATE_SPACE = "state-space detector"
"""One name each, used in both figures. Neither says "our": a figure should
name a method, not a side."""


def state_space_intervals(decoded: dict) -> list[tuple[float, float]]:
    """When the switching state-space model calls the representation non-local.

    The comparison needs this. Every event the sliding-window detector misses
    has to be an event some detector can find, or the figure shows the limits
    of the data rather than the limits of the method.

    Reads the detector fitted in :func:`simulate_and_decode`, which is also the
    one whose place fields the sliding-window likelihood was given, so the two
    procedures differ in how they use the encoding model and not in what it is.
    """
    time = decoded["time"]
    results = decoded["detector"].predict(
        position_time=time,
        position=decoded["position"],
        spike_times=decoded["spike_times"],
        time=time,
    )
    probability = np.asarray(
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .values
    ).squeeze()
    if not np.isfinite(probability).all() or probability.max() <= 0.0:
        raise RuntimeError(
            "The state-space model returned no probability mass; its state "
            "probabilities must sum to one at every time."
        )

    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    return [
        (time[start], time[stop])
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True
        )
        if time[stop] - time[start] >= NON_LOCAL_MINIMUM_DURATION_S
    ]


def candidates_overlapping(decoded: dict, event: int) -> list[int]:
    """Every candidate sequence that overlaps a labeled event, accepted or not."""
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    return [
        i
        for i, (start, stop) in enumerate(candidates.runs)
        if decoded["midpoint_s"][start] < event_end
        and event_start < decoded["midpoint_s"][stop]
    ]


def accepted_overlapping(decoded: dict, event: int) -> list[int]:
    """The subset that met the duration and dispersion criteria."""
    is_replay = decoded["candidates"].is_replay
    return [i for i in candidates_overlapping(decoded, event) if is_replay[i]]


def is_reported(decoded: dict) -> np.ndarray:
    """Per candidate: replay under the whole published definition."""
    reported = np.zeros(len(decoded["candidates"].runs), dtype=bool)
    for i, result in decoded["shuffles"].items():
        reported[i] = result.passes
    return reported


def reported_overlapping(decoded: dict, event: int) -> list[int]:
    """The subset that also beat the place-cell identity shuffle: replay under
    the whole published definition."""
    return [
        i for i in accepted_overlapping(decoded, event) if decoded["shuffles"][i].passes
    ]


def verdict(decoded: dict, event: int) -> str:
    """What the sliding-window detector did with one labeled event.

    ``found`` when a sequence overlapping it meets the whole published
    definition; ``failed the shuffle`` when one met the duration and
    dispersion criteria but not the significance test; ``declined`` when a
    candidate was assembled but none was accepted; ``missed`` when no
    candidate overlaps it at all. Read from the data, so a panel title cannot
    drift away from the panel.
    """
    if reported_overlapping(decoded, event):
        return "found"
    if accepted_overlapping(decoded, event):
        return "failed the shuffle"
    if candidates_overlapping(decoded, event):
        return "declined"
    return "missed"


def duration_ratio(decoded: dict, event: int) -> float:
    """How much longer the detector calls an event than the event really is.

    Over the longest accepted sequence overlapping it, whether or not it then
    passes the shuffle, which never moves a sequence's bounds. Raises when
    there is none, so a title cannot be formatted from a detection that did
    not happen.
    """
    accepted = accepted_overlapping(decoded, event)
    if not accepted:
        raise RuntimeError(
            f"No accepted sequence overlaps event {event} ({EVENT_LABELS[event]!r}); "
            "the figure's titles and caption assume one. Re-read the data before "
            "regenerating."
        )
    candidates = decoded["candidates"]
    event_start, event_end = decoded["event_times"][event]
    return float(
        max(candidates.duration_s[i] for i in accepted) / (event_end - event_start)
    )


def outcome_title(decoded: dict, event: int, label: str) -> str:
    """One-line panel title stating what the detector did, from the data."""
    what = verdict(decoded, event)
    if what == "found":
        return f"{label}: {duration_ratio(decoded, event):.1f}x long"
    if what == "failed the shuffle":
        return f"{label}: {duration_ratio(decoded, event):.1f}x long, fails shuffle"
    return f"{label}: {what}"


def shuffle_of(decoded: dict, event: int):
    """The shuffle result of the longest accepted sequence overlapping ``event``."""
    accepted = accepted_overlapping(decoded, event)
    longest = max(accepted, key=lambda i: decoded["candidates"].duration_s[i])
    return decoded["shuffles"][longest]


def draws_from_passing(result) -> int:
    """How many more draws below the observed value than a pass allows.

    As the reference tests it, a half passes when ``(1 + k) / (n + 1) <= 0.05``
    with ``k`` the draws strictly below the observed value: at most 4 of 100.
    Zero means the candidate passes; one means a single draw decided it.
    """
    allowed = int(np.floor(SIGNIFICANCE_LEVEL * (result.n_shuffles + 1) + 1e-9)) - 1
    worst = max(
        int(np.count_nonzero(result.null_spread_cm < result.observed_spread_cm)),
        int(np.count_nonzero(result.null_jump_cm < result.observed_jump_cm)),
    )
    return max(0, worst - allowed)


def decoded_center_cm(decoded: dict) -> np.ndarray:
    """Center of each window, NaN where the window has none."""
    center = decoded["candidates"].decisions.center_index
    out = np.full(center.size, np.nan)
    present = center >= 0
    out[present] = decoded["place_bin_centers_cm"][center[present]]
    return out


def local_fraction_of_event(decoded: dict, event: int) -> float:
    """Share of the accepted event overlapping ``event`` that is local coding.

    Over the longest accepted sequence, the one :func:`duration_ratio` reads.
    """
    accepted = accepted_overlapping(decoded, event)
    if not accepted:
        raise RuntimeError(f"No accepted sequence overlaps event {event}.")
    longest = max(accepted, key=lambda i: decoded["candidates"].duration_s[i])
    return local_fraction(decoded, longest)


def best_dispersion_cm(decoded: dict, event: int) -> float:
    """The largest dispersion any candidate overlapping ``event`` reached."""
    dispersion = decoded["candidates"].dispersion_cm
    return max(
        (float(dispersion[i]) for i in candidates_overlapping(decoded, event)),
        default=float("nan"),
    )


def outside_pieces(decoded: dict, sequence: int) -> np.ndarray:
    """Windows of a sequence that were kept but belong to none of its pieces.

    A single kept window between two runs is dropped before the merge, as the
    reference drops it, yet still counts toward the merged sequence's
    dispersion. Returns window indices.
    """
    candidates = decoded["candidates"]
    start, stop = candidates.runs[sequence]
    in_piece = np.zeros(stop - start + 1, dtype=bool)
    for piece in candidates.pieces[sequence]:
        first, last = candidates.pre_merge_runs[piece]
        in_piece[first - start : last - start + 1] = True
    kept = candidates.decisions.gates.kept[start : stop + 1]
    return start + np.flatnonzero(kept & ~in_piece)


def stray_window_dispersions(decoded: dict) -> tuple[float, float, int]:
    """The stray-window event's dispersion with and without the windows
    outside its pieces, and the sequence they come from.

    Over the longest accepted sequence overlapping it, the one the panel shows.
    """
    accepted = accepted_overlapping(decoded, STRAY_WINDOW_EVENT)
    if not accepted:
        raise RuntimeError(
            "The caption says the stray-window event is called replay; no "
            "accepted sequence overlaps it. Rewrite the caption."
        )
    candidates = decoded["candidates"]
    sequence = max(accepted, key=lambda i: candidates.duration_s[i])
    start, stop = candidates.runs[sequence]
    center = candidates.decisions.center_index[start : stop + 1].copy()
    center[np.isin(np.arange(start, stop + 1), outside_pieces(decoded, sequence))] = (
        NO_CENTER
    )
    center[~candidates.decisions.gates.kept[start : stop + 1]] = NO_CENTER
    return (
        float(candidates.dispersion_cm[sequence]),
        dispersion_cm(center, decoded["bin_distance_cm"]),
        sequence,
    )


CAPTION_SETTINGS = {
    "n_shuffles": 100,
    "spread_threshold_cm": 10.0,
    "stride_ms": 4.0,
    "speed_threshold_cm_per_s": 4.0,
    "dispersion_threshold_cm": 12.0,
    "stray_window_still_ms": 120.0,
    "stray_window_n_local": 2,
}
"""Settings the caption's prose states. They are method, not results, so the
caption types them; :func:`check_caption_settings` stops if the code moves."""


def check_caption_settings(config: MemorylessSequenceConfig) -> None:
    """Stop when a setting the caption states is no longer the code's."""
    actual = {
        "n_shuffles": DEFAULT_N_SHUFFLES,
        "spread_threshold_cm": config.spread_threshold_cm,
        "stride_ms": 1000 * config.stride_s,
        "speed_threshold_cm_per_s": config.speed_threshold_cm_per_s,
        "dispersion_threshold_cm": config.dispersion_threshold_cm,
        "stray_window_still_ms": 1000 * STRAY_WINDOW_STILL_S,
        "stray_window_n_local": STRAY_WINDOW_N_LOCAL,
    }
    moved = {
        k: (v, actual[k])
        for k, v in CAPTION_SETTINGS.items()
        if not np.isclose(v, actual[k])
    }
    if moved:
        raise RuntimeError(
            f"The caption states settings the code no longer uses (caption, code): "
            f"{moved}. Rewrite the caption and CAPTION_SETTINGS."
        )


def check_caption_claims(decoded: dict, config: MemorylessSequenceConfig) -> None:
    """Stop rather than write prose the panels contradict.

    The caption commits to a verdict for each of the four zoomed events and to
    the state-space detector reporting all of them. Each is read from the
    data here; a threshold, seed or library change that moves one turns into
    an error naming it instead of a caption that says the opposite of its
    figure.
    """
    expected = {
        CONTINUOUS_OUTBOUND_EVENT: "failed the shuffle",
        CONTINUOUS_INBOUND_EVENT: "found",
        SHORT_PATH_EVENT: "declined",
        REMOTE_STATIONARY_EVENT: "declined",
    }
    if not accepted_overlapping(decoded, SHORT_REPLAY_EVENT):
        raise RuntimeError(
            "The caption says the 80 ms replay meets the duration and dispersion "
            "criteria; no accepted sequence overlaps it. Rewrite the caption."
        )
    for event, want in expected.items():
        got = verdict(decoded, event)
        if got != want:
            raise RuntimeError(
                f"The caption says the sliding-window detector {want} "
                f"{EVENT_LABELS[event]!r}; the data say {got}. Rewrite the caption."
            )
    for event in (SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT):
        if not best_dispersion_cm(decoded, event) < config.dispersion_threshold_cm:
            raise RuntimeError(
                f"The caption says {EVENT_LABELS[event]!r} was declined for too "
                "little dispersion, but a candidate overlapping it clears the "
                "threshold. Rewrite the caption."
            )
    zoomed = (
        CONTINUOUS_OUTBOUND_EVENT,
        SHORT_REPLAY_EVENT,
        SHORT_PATH_EVENT,
        REMOTE_STATIONARY_EVENT,
    )
    full, pieces, sequence = stray_window_dispersions(decoded)
    stray_start = decoded["stray_window"].stray_window_start_s
    stray_index = int(
        np.argmin(
            np.abs(decoded["midpoint_s"] - stray_start - config.window_duration_s / 2)
        )
    )
    if not (
        full > config.dispersion_threshold_cm > pieces
        and outside_pieces(decoded, sequence).tolist() == [stray_index]
    ):
        raise RuntimeError(
            "The caption says the stray-window event is called replay only "
            "because of its one window outside every piece; the data say "
            f"dispersion {full:.1f} cm with it, {pieces:.1f} cm without, and "
            f"windows {outside_pieces(decoded, sequence).tolist()} outside the "
            f"pieces against the placed one, {stray_index}. Rewrite the caption."
        )
    if not np.isfinite(bound_error_ms(decoded, zoomed)):
        raise RuntimeError(
            "The caption says the state-space detector reports all four zoomed "
            "events; it does not. Rewrite the caption."
        )
    for event in (CONTINUOUS_OUTBOUND_EVENT, CONTINUOUS_INBOUND_EVENT):
        if not local_fraction_of_event(decoded, event) > 0.5:
            raise RuntimeError(
                "The caption says most of each reported continuous replay is the "
                f"animal's own place; {EVENT_LABELS[event]!r} is not. Rewrite it."
            )
    if not shuffle_of(decoded, SHORT_REPLAY_EVENT).passes:
        raise RuntimeError(
            "The caption says the 80 ms replay passes the significance test; it "
            "fails it. Rewrite the caption."
        )
    if (
        shuffle_of(decoded, CONTINUOUS_OUTBOUND_EVENT).passes
        or not shuffle_of(decoded, CONTINUOUS_INBOUND_EVENT).passes
    ):
        raise RuntimeError(
            "The caption says the outbound continuous replay fails the significance "
            "test and the inbound one passes it; the data disagree. Rewrite the "
            "caption."
        )
    stray = decoded["shuffles"][sequence]
    if stray.p_spread <= SIGNIFICANCE_LEVEL or stray.p_jump <= SIGNIFICANCE_LEVEL:
        raise RuntimeError(
            "The caption says the stray-window event fails both halves of the "
            "significance test; it passes one. Rewrite the caption."
        )
    start_s, end_s = decoded["event_times"][STRAY_WINDOW_EVENT]
    if any(a < end_s and start_s < b for a, b in decoded["state_space_intervals"]):
        raise RuntimeError(
            "The caption says the state-space detector reports no non-local event "
            "over the stray-window event; it reports one. Rewrite the caption."
        )


def local_fraction(decoded: dict, sequence: int, tolerance_cm: float = 15.0) -> float:
    """Share of a sequence's kept windows whose center is the animal's own place.

    Kept windows only, as the sweep scores them: a window inside a merged gap
    was rejected and is not one of the event's decoded positions.
    """
    candidates = decoded["candidates"]
    start, stop = candidates.runs[sequence]
    center_cm = decoded_center_cm(decoded)[start : stop + 1]
    animal_cm = np.interp(
        decoded["midpoint_s"][start : stop + 1], decoded["time"], decoded["position"]
    )
    kept = candidates.assembly.kept[start : stop + 1]
    return float(np.mean(np.abs(center_cm[kept] - animal_cm[kept]) < tolerance_cm))


def bound_error_ms(decoded: dict, events: tuple[int, ...]) -> float:
    """Largest error in either bound of the state-space model's reports.

    The caption used to state a round number for this. It is read off the
    data now, over exactly the events the figure shows.
    """
    worst = 0.0
    for event in events:
        start_s, end_s = decoded["event_times"][event]
        overlapping = [
            (a, b)
            for a, b in decoded["state_space_intervals"]
            if a < end_s and start_s < b
        ]
        if not overlapping:
            return float("nan")
        worst = max(
            worst,
            abs(overlapping[0][0] - start_s),
            abs(overlapping[-1][1] - end_s),
        )
    return 1000 * worst


def plot_overview(ax, decoded: dict) -> None:
    """The whole recording: what was decoded, and what the detector kept."""
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    center_cm = decoded_center_cm(decoded)
    kept = candidates.decisions.gates.kept

    # the package's "local" entry is a zero-width marker for an unmanipulated
    # pause; the remote stationary event sits there now
    labelled = [
        (start_s, end_s, label)
        for (start_s, end_s), label in zip(
            decoded["event_times"], EVENT_LABELS, strict=True
        )
        if end_s > start_s
    ]
    # two rows, alternating in time order, so neighbours never share a row
    for row, (start_s, end_s, label) in enumerate(sorted(labelled)):
        ax.axvspan(start_s, end_s, color=YELLOW, alpha=0.6, lw=0)
        ax.text(
            0.5 * (start_s + end_s),
            190 if row % 2 else 234,
            label,
            fontsize=5,
            ha="center",
            va="bottom",
            color="0.25",
            linespacing=0.9,
        )

    ax.plot(decoded["time"], decoded["position"], color=GREEN, lw=0.6, zorder=1)
    ax.scatter(
        midpoint_s[~kept],
        center_cm[~kept],
        s=0.4,
        color="0.55",
        lw=0,
        zorder=2,
    )
    ax.scatter(midpoint_s[kept], center_cm[kept], s=0.6, color=SKY, lw=0, zorder=3)

    for start_s, end_s in decoded["state_space_intervals"]:
        ax.plot(
            [start_s, end_s],
            [-22, -22],
            color=BLUE,
            lw=2.4,
            solid_capstyle="butt",
            zorder=4,
        )

    reported = is_reported(decoded)
    for sequence, (start, stop) in enumerate(candidates.runs):
        accepted = reported[sequence]
        ax.plot(
            midpoint_s[[start, stop]],
            [-14, -14],
            color=VERMILION if accepted else "0.6",
            lw=2.4 if accepted else 1.2,
            solid_capstyle="butt",
            zorder=4,
        )

    handles = [
        Line2D([], [], color=GREEN, lw=1.0, label="where the animal is"),
        Line2D(
            [],
            [],
            color=SKY,
            marker="o",
            ms=2.0,
            lw=0,
            label="decoded place, window kept",
        ),
        Line2D(
            [],
            [],
            color="0.55",
            marker="o",
            ms=2.0,
            lw=0,
            label="decoded place, window rejected",
        ),
        Patch(color=YELLOW, alpha=0.6, label="labeled event"),
        Patch(color="0.75", label="posterior, panels b-f"),
        Line2D(
            [],
            [],
            color=VERMILION,
            marker="*",
            ms=4.0,
            lw=0,
            label="kept window outside every piece",
        ),
        Line2D([], [], color=VERMILION, lw=2.4, label=f"event, {SLIDING}"),
        Line2D([], [], color="0.6", lw=1.2, label="candidate, rejected"),
        Line2D([], [], color=BLUE, lw=2.4, label=f"event, {STATE_SPACE}"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
        fontsize=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.0,
    )

    ax.set_ylim(-30, 268)
    ax.set_xlim(decoded["time"][0], decoded["time"][-1])
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("position (cm)")
    ax.set_xlabel("time (s)")


def zoom_window(decoded: dict, event: int) -> tuple[float, float, list[int]]:
    """The stretch to show around one labeled event.

    Wide enough to hold the event itself and every candidate sequence that
    touches it, which is the point: the sequences reach well beyond the event.
    """
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    true_start, true_end = decoded["event_times"][event]

    overlapping = [
        i
        for i, (start, stop) in enumerate(candidates.runs)
        if midpoint_s[start] < true_end + 0.1 and true_start - 0.1 < midpoint_s[stop]
    ]
    spans = [midpoint_s[candidates.runs[i]] for i in overlapping]
    return (
        min([span[0] for span in spans] + [true_start]) - 0.25,
        max([span[1] for span in spans] + [true_end]) + 0.25,
        overlapping,
    )


def plot_event(ax, decoded: dict, event: int) -> None:
    """One labeled event, with the sequence the detector built around it."""
    candidates = decoded["candidates"]
    midpoint_s = decoded["midpoint_s"]
    center_cm = decoded_center_cm(decoded)
    true_start, true_end = decoded["event_times"][event]

    window_lo, window_hi, overlapping = zoom_window(decoded, event)
    shown = (midpoint_s >= window_lo) & (midpoint_s <= window_hi)

    extent = (
        midpoint_s[shown][0],
        midpoint_s[shown][-1],
        decoded["place_bin_centers_cm"][0],
        decoded["place_bin_centers_cm"][-1],
    )
    # the grey wash is the posterior itself; panel b's label says so once
    ax.imshow(
        decoded["posterior"][shown].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="bone_r",
        vmin=0.0,
        vmax=np.percentile(decoded["posterior"][shown], 99.9),
        interpolation="nearest",
    )
    ax.axvspan(
        true_start, max(true_end, true_start + 0.02), color=YELLOW, alpha=0.45, lw=0
    )

    kept = candidates.decisions.gates.kept & shown
    ax.scatter(midpoint_s[kept], center_cm[kept], s=1.6, color=SKY, lw=0, zorder=3)
    # a kept window outside every piece still counts toward dispersion
    stray = np.concatenate(
        [outside_pieces(decoded, i) for i in overlapping] or [np.array([], int)]
    )
    ax.scatter(
        midpoint_s[stray],
        center_cm[stray],
        s=22,
        marker="*",
        color=VERMILION,
        lw=0,
        zorder=5,
    )
    ax.plot(
        decoded["time"],
        decoded["position"],
        color=GREEN,
        lw=1.0,
        ls="--",
        zorder=2,
        label="where the animal is",
    )

    for start_s, end_s in decoded["state_space_intervals"]:
        if start_s > window_hi or end_s < window_lo:
            continue
        ax.plot(
            [start_s, end_s],
            [-16, -16],
            color=BLUE,
            lw=2.6,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )

    reported = is_reported(decoded)
    for i in overlapping:
        start, stop = candidates.runs[i]
        accepted = reported[i]
        ax.plot(
            midpoint_s[[start, stop]],
            [-9, -9],
            color=VERMILION if accepted else "0.6",
            lw=2.6 if accepted else 1.4,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )

    ax.set_xlim(window_lo, window_hi)
    ax.set_ylim(-20, 178)
    ax.set_xlabel("time (s)")
    ax.set_title(EVENT_LABELS[event].replace("\n", " "), fontsize=6)


def detection_grid(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Detection rate as path length by duration."""
    rows = summary[
        (summary["summary"] == "detection_rate_by_path_and_duration")
        & (summary.measure == measure)
    ]
    return rows.pivot(index="path_cm", columns="duration_s", values="value")


def plot_detection(ax, grid: pd.DataFrame, title: str, show_y: bool):
    """Percent of the swept replays each detector reported, path by duration."""
    image = ax.imshow(
        100 * grid.to_numpy(),
        vmin=0,
        vmax=100,
        cmap="magma",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"{1000 * c:.0f}" for c in grid.columns], fontsize=5)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"{r:.0f}" for r in grid.index], fontsize=5)
    ax.set_xlabel("duration (ms)")
    if show_y:
        ax.set_ylabel("path length (cm)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            percent = 100 * grid.iat[row, column]
            ax.text(
                column,
                row,
                f"{percent:.0f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if percent < 55 else "black",
            )
    return image


TIMING_MEASURES = tuple(
    f"{detector}_{edge}_error_ms_{statistic}"
    for detector in ("sliding", "state_space")
    for edge in ("start", "end")
    for statistic in ("median", "q25", "q75")
)
"""The timing errors the onset and offset panels draw, in ms, by true duration."""


def plot_edge_error(ax, summary: pd.DataFrame, edge: str) -> None:
    """Reported minus true start or end, for replays both procedures found."""
    timing = sweep_statistics.sweep_by_duration(
        summary, "timing_by_duration", TIMING_MEASURES
    )
    duration_ms = 1000 * timing.index.to_numpy()
    ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
    for detector, color, label in (
        ("sliding", VERMILION, SLIDING),
        ("state_space", BLUE, STATE_SPACE),
    ):
        median, low, high = (
            timing[f"{detector}_{edge}_error_ms_{statistic}"].to_numpy()
            for statistic in ("median", "q25", "q75")
        )
        ax.fill_between(duration_ms, low, high, color=color, alpha=0.2, lw=0)
        ax.plot(
            duration_ms, median, color=color, lw=1.0, marker="o", ms=2.5, label=label
        )
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel(f"{edge} error (ms)")
    ax.set_title("onset" if edge == "start" else "offset", fontsize=6)


def plot_absorption(ax, summary: pd.DataFrame):
    """The over-call against how far the replay sits from the animal's own place."""
    rows = summary[summary["summary"] == "median_by_distance_from_animal"]
    present = set(rows.distance_bin_cm)
    if not present <= set(DISTANCE_ORDER):
        raise ValueError(
            f"The sweep summary holds distance bins {sorted(present - set(DISTANCE_ORDER))} "
            "this panel does not know how to place; update DISTANCE_ORDER."
        )
    order = list(DISTANCE_ORDER)
    at = np.arange(len(order))

    def series(measure):
        indexed = rows[rows.measure == measure].set_index("distance_bin_cm").value
        return np.array([indexed.get(bin_name, np.nan) for bin_name in order])

    # an expected bin with no jointly reported replay is a result, not a gap
    for x, bin_name in zip(at, order, strict=True):
        if bin_name not in present:
            ax.text(
                x,
                1.0,
                "none\nreported\nby both",
                fontsize=4.5,
                color="0.45",
                ha="center",
                va="bottom",
                linespacing=0.9,
            )

    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.plot(
        at, series("sliding_duration_ratio"), color=VERMILION, lw=1.0, marker="o", ms=3
    )
    ax.plot(
        at, series("state_space_duration_ratio"), color=BLUE, lw=1.0, marker="o", ms=3
    )
    for x, local in zip(at, series("sliding_local_fraction"), strict=True):
        if not np.isfinite(local):
            continue
        ax.annotate(
            f"{100 * local:.0f}%",
            xy=(x, 1.0),
            xytext=(x, 1.28),
            fontsize=5,
            color="0.35",
            ha="center",
            va="center",
        )
    ax.set_xticks(at)
    ax.set_xticklabels([bin_name.replace(", ", "-").strip("(]") for bin_name in order])
    # every bin gets a full slot, so an empty one is not drawn on the spine
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.85, 2.6)
    ax.set_xlabel("distance from the animal (cm)")
    ax.set_ylabel("reported / actual duration")
    ax.set_title("reported duration by distance", fontsize=6)


def output_color(detector: str) -> str:
    return VERMILION if detector.startswith("sliding") else BLUE


def no_replay_minutes(controls: pd.DataFrame) -> float:
    rows = controls[
        (controls["summary"] == "no_replay") & (controls.measure == "immobile_minutes")
    ]
    return float(rows.value.iloc[0])


def plot_bursts(ax, controls: pd.DataFrame) -> None:
    """Share of bursts with an event reported on them, for each output."""
    width = 0.2
    for group, condition in enumerate(("placeless_burst", "jumping_burst")):
        for offset, (detector, label, alpha) in enumerate(OUTPUTS):
            share = sweep_statistics.controls_value(
                controls, "bursts", "called_rate", condition, None, detector
            )
            ax.bar(
                group + (offset - 1.5) * width,
                100 * share,
                width=width,
                color=output_color(detector),
                alpha=alpha,
                label=label if group == 0 else None,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["place-less\nburst", "jumping\nburst"])
    # headroom: the no-replay note above the bars, the legend above it
    ax.set_ylim(0, 165)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("bursts called (%)")
    ax.set_title("bursts", fontsize=6)
    minutes = no_replay_minutes(controls)
    ax.text(
        0.5,
        118,
        f"no replay: no event from any\noutput in {minutes:.1f} min of rest",
        ha="center",
        va="center",
        fontsize=4.5,
        color="0.3",
    )
    ax.legend(fontsize=4.5, frameon=False, loc="upper left", handlelength=1.0, ncol=2)


def plot_carving(
    ax,
    controls: pd.DataFrame,
    conditions: tuple[str, ...],
    xlabel: str,
    title: str,
    tick,
    legend: bool,
    show_y: bool = True,
) -> None:
    """How each detector carved multi-part events, pooled over ``conditions``."""
    detectors = ("sliding_published", "state_space_canonical")
    by_level = {
        (detector, measure): sweep_statistics.parts_by_level(
            controls, conditions, measure, detector
        )
        for detector in detectors
        for measure in (
            *(f"segmentation_{label}" for label, _, _ in SEGMENTATION_STYLE),
            "part_recovery",
        )
    }
    levels = by_level[detectors[0], "part_recovery"].index
    width = 0.36
    for group in range(len(levels)):
        for offset, detector in enumerate(detectors):
            x = group + (offset - 0.5) * width
            bottom = 0.0
            for label, shade, name in SEGMENTATION_STYLE:
                share = by_level[detector, f"segmentation_{label}"].iloc[group]
                ax.bar(
                    x,
                    100 * share,
                    bottom=100 * bottom,
                    width=width,
                    color=shade,
                    edgecolor=output_color(detector),
                    linewidth=0.8,
                    label=name if legend and group == 0 and offset == 0 else None,
                )
                bottom += share
            recovery = by_level[detector, "part_recovery"].iloc[group]
            # the two detectors' numbers at two heights, so neighbours do not run
            # together
            ax.text(
                x,
                102 + 9 * offset,
                f"{100 * recovery:.0f}",
                ha="center",
                va="bottom",
                fontsize=4,
                color=output_color(detector),
            )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([tick(level) for level in levels])
    # headroom for the recovery numbers and, above them, the legend
    ax.set_ylim(0, 162)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    if show_y:
        ax.set_ylabel("trials (%)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)
    if legend:
        ax.legend(
            fontsize=4.5, frameon=False, loc="upper left", handlelength=1.0, ncol=2
        )


def plot_heldout_support(ax, heldout: pd.DataFrame) -> None:
    """Share of informative events with held-out support, by kind and fold."""
    width = 0.26
    for group, (_, classes, truth) in enumerate(HELDOUT_GROUPS):
        for offset, (fold, hatch) in enumerate(FOLD_HATCH.items()):
            x = group + (offset - 1) * width
            ours = sweep_statistics.pooled_heldout_rate(heldout, fold, classes, truth)
            rate, n = ours["rate"], ours["n_events"]
            other = sweep_statistics.pooled_heldout_rate(
                heldout, fold, classes, truth, estimator="sliding_window"
            )["rate"]
            if not n:
                ax.text(x, 3, "-", ha="center", va="bottom", fontsize=4.5, color="0.5")
                continue
            ax.bar(
                x,
                100 * rate,
                width=width,
                color=BLUE,
                alpha=0.35 + 0.25 * offset,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                linewidth=0,
                label=f"fold {fold}" if group == 0 else None,
            )
            ax.plot(
                x,
                100 * other,
                "o",
                color=VERMILION,
                ms=2.2,
                label="sliding window's content"
                if group == 0 and offset == 0
                else None,
            )
            ax.text(x, 100 * rate + 3, f"{n}", ha="center", va="bottom", fontsize=3.8)
    ax.set_xticks(range(len(HELDOUT_GROUPS)))
    ax.set_xticklabels([label for label, _, _ in HELDOUT_GROUPS])
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("held-out support (%)")
    ax.set_title("held-out support, informative events", fontsize=6)
    ax.legend(fontsize=4.5, frameon=False, loc="upper right", handlelength=1.0, ncol=2)


def plot_heldout_coverage(ax, heldout: pd.DataFrame) -> None:
    """Share of each kind of replay the evaluation cells can judge at all."""
    width = 0.26
    for group, (_, classes, truth) in enumerate(HELDOUT_GROUPS[:2]):
        for offset, (fold, hatch) in enumerate(FOLD_HATCH.items()):
            share = sweep_statistics.pooled_heldout_rate(
                heldout, fold, classes, truth, measure="informative_share"
            )["rate"]
            ax.bar(
                group + (offset - 1) * width,
                100 * share,
                width=width,
                color=BLUE,
                alpha=0.35 + 0.25 * offset,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                linewidth=0,
            )
    ax.set_xticks(range(2))
    ax.set_xticklabels([label for label, _, _ in HELDOUT_GROUPS[:2]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("informative (%)")
    ax.set_title("events the evaluation cells can judge", fontsize=6)


def place_panel_letters(fig, rows: list[list]) -> None:
    """Draw each panel's letter on its title line, flush with a common margin.

    The project's `label_panels` positions a letter in axes fractions, which
    is right for a grid of equal panels. Here the four rows differ in height
    and in how wide a y label each carries, so one fraction puts the letters
    at four different heights and four different depths. This measures the
    drawn figure instead: every letter sits on its row's title line, and the
    first letter of each row sits on one left margin.

    Constrained layout settles only when the figure is drawn, so this must
    draw first and must run after every panel is filled in.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    rise = LETTER_RISE_PT / (72 * fig.get_figheight())
    gap = LETTER_GAP_PT / (72 * fig.get_figwidth())

    def drawn_left(ax) -> float:
        """Left edge of everything the panel draws, y label and ticks included."""
        return to_figure.transform_bbox(ax.get_tightbbox(renderer)).x0

    margin = min(drawn_left(row[0]) for row in rows) - gap
    letters = iter(string.ascii_lowercase)
    for row in rows:
        for column, ax in enumerate(row):
            fig.text(
                margin if column == 0 else drawn_left(ax) - gap,
                to_figure.transform_bbox(ax.bbox).y1 + rise,
                next(letters),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="center",
            )


def make_figure(
    n_neurons: int,
    config: MemorylessSequenceConfig,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    heldout: pd.DataFrame,
):
    decoded = simulate_and_decode(n_neurons, config)
    decoded["state_space_intervals"] = state_space_intervals(decoded)
    check_caption_claims(decoded, config)

    set_figure_defaults(context="paper")
    # this project's defaults turn constrained layout on, so the grid must not
    # also set its own margins: the two fight and the panels collapse
    fig = plt.figure(
        figsize=(TWO_COLUMN, TWO_COLUMN * 1.72), dpi=450, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.09)
    # one column of rows, each row with its own columns, so the layout solves
    # each row's widths on its own; one shared grid couples every row's
    # decorations and collapses
    grid = GridSpec(len(ROW_HEIGHTS), 1, figure=fig, height_ratios=ROW_HEIGHTS)

    ax_overview = fig.add_subplot(grid[0])
    plot_overview(ax_overview, decoded)

    # one line each, so these titles sit at the same height as every other
    # row's and the panel letters can share one offset; the verdict in each
    # is read from the data
    outcomes = (
        (CONTINUOUS_OUTBOUND_EVENT, "continuous"),
        (SHORT_REPLAY_EVENT, "80 ms replay"),
        (SHORT_PATH_EVENT, "30 cm path"),
        (REMOTE_STATIONARY_EVENT, "remote, held still"),
        (STRAY_WINDOW_EVENT, "one far window"),
    )
    zooms = []
    events = grid[1].subgridspec(1, len(outcomes))
    for column, (event, label) in enumerate(outcomes):
        ax = fig.add_subplot(events[column])
        plot_event(ax, decoded, event)
        title = (
            f"{label}: "
            + {"found": "found", "failed the shuffle": "fails shuffle"}.get(
                verdict(decoded, event), verdict(decoded, event)
            )
            if event == STRAY_WINDOW_EVENT
            else outcome_title(decoded, event, label)
        )
        ax.set_title(title, fontsize=5)
        if column:
            ax.set_yticklabels([])
        zooms.append(ax)
    zooms[0].set_ylabel("decoded place (cm)")

    heatmaps = grid[2].subgridspec(1, 2, wspace=0.06)
    ax_sliding = fig.add_subplot(heatmaps[0])
    ax_state = fig.add_subplot(heatmaps[1])
    image = plot_detection(
        ax_sliding, detection_grid(summary, "sliding_matches"), SLIDING, show_y=True
    )
    plot_detection(
        ax_state,
        detection_grid(summary, "state_space_matches"),
        STATE_SPACE,
        show_y=False,
    )
    bar = fig.colorbar(image, ax=[ax_sliding, ax_state], fraction=0.03, pad=0.01)
    bar.set_label("replays reported (%)", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)

    timing = grid[3].subgridspec(1, 3)
    ax_onset = fig.add_subplot(timing[0])
    plot_edge_error(ax_onset, summary, "start")
    ax_onset.legend(fontsize=5, frameon=False, loc="center right", handlelength=1.3)
    ax_offset = fig.add_subplot(timing[1])
    plot_edge_error(ax_offset, summary, "end")
    ax_distance = fig.add_subplot(timing[2])
    plot_absorption(ax_distance, summary)

    carving = grid[4].subgridspec(1, 4)
    ax_bursts = fig.add_subplot(carving[0])
    plot_bursts(ax_bursts, controls)
    ax_jumps = fig.add_subplot(carving[1])
    plot_carving(
        ax_jumps,
        controls,
        ("jump_120ms", "jump_200ms"),
        "jump (cm)",
        "mid-event jump",
        lambda level: f"{level:.0f}",
        legend=True,
    )
    ax_continuing = fig.add_subplot(carving[2])
    plot_carving(
        ax_continuing,
        controls,
        ("chain_continuing",),
        "gap (ms)",
        "chain, continuing",
        lambda level: f"{1000 * level:.0f}",
        legend=False,
        show_y=False,
    )
    evidence = grid[5].subgridspec(1, 2, width_ratios=[2.2, 1])
    ax_support = fig.add_subplot(evidence[0])
    plot_heldout_support(ax_support, heldout)
    ax_coverage = fig.add_subplot(evidence[1])
    plot_heldout_coverage(ax_coverage, heldout)
    ax_elsewhere = fig.add_subplot(carving[3])
    plot_carving(
        ax_elsewhere,
        controls,
        ("chain_elsewhere",),
        "gap (ms)",
        "chain, elsewhere",
        lambda level: f"{1000 * level:.0f}",
        legend=False,
        show_y=False,
    )

    place_panel_letters(
        fig,
        [
            [ax_overview],
            zooms,
            [ax_sliding, ax_state],
            [ax_onset, ax_offset, ax_distance],
            [ax_bursts, ax_jumps, ax_continuing, ax_elsewhere],
            [ax_support, ax_coverage],
        ],
    )

    return fig, decoded


def check_summary_claims(summary: pd.DataFrame) -> None:
    """The caption's claims about the sweep, checked against the summary.

    Companion to :func:`check_caption_claims`, which covers the decoded
    recording. Anything the caption states about the sweep as a fact rather
    than interpolates as a number is asserted here.
    """
    by_path = summary[summary["summary"] == "by_path"]
    below = by_path[
        (by_path.measure == "state_space_replay_matches")
        & (by_path.path_cm < SHORT_PATH.path_cm + 15.0)
    ]
    if below.empty or (below.value > 0).any():
        raise RuntimeError(
            "The caption says the state-space detector, held to the published "
            "replay definition, reports nothing below 45 cm; the summary disagrees."
        )
    jump_pass = sweep_statistics.sweep_jump_pass_statistics(summary)
    if not jump_pass["shortest_fraction"] < jump_pass["longest_fraction"]:
        raise RuntimeError(
            "The caption says the jump half passes fewer of the shortest replays "
            "than of the longest; the summary's pass rates by duration disagree."
        )
    timing = sweep_statistics.sweep_timing_statistics(summary)
    if not (
        timing["sliding_start_early_min_ms"] > 0
        and timing["sliding_end_late_min_ms"] > 0
    ):
        raise RuntimeError(
            "The caption says the sliding window starts early and ends late at "
            "every duration; the summary's median timing errors disagree."
        )
    errors = sweep_statistics.sweep_by_duration(
        summary, "timing_by_duration", TIMING_MEASURES
    )
    sliding_onset, state_space_onset, sliding_offset, state_space_offset = (
        errors[f"{detector}_{edge}_error_ms_median"].to_numpy()
        for edge in ("start", "end")
        for detector in ("sliding", "state_space")
    )
    sliding_offset_q25 = errors["sliding_end_error_ms_q25"].to_numpy()
    sliding_offset_q75 = errors["sliding_end_error_ms_q75"].to_numpy()
    if not (
        sliding_offset_q75 - sliding_offset > sliding_offset - sliding_offset_q25
    ).all():
        raise RuntimeError(
            "The caption says the sliding window's late ends carry a long tail; "
            "its offset errors are not skewed late at every duration."
        )
    if not (
        (sliding_onset < state_space_onset).all()
        and (sliding_offset > state_space_offset).all()
    ):
        raise RuntimeError(
            "The caption contrasts the sliding window's early starts and late ends "
            "with the state-space detector's; at some duration they do not differ "
            "that way."
        )
    by_distance = summary[summary["summary"] == "median_by_distance_from_animal"]
    local = (
        by_distance[by_distance.measure == "sliding_local_fraction"]
        .set_index("distance_bin_cm")
        .value
    )
    ratio = (
        by_distance[by_distance.measure == "sliding_duration_ratio"]
        .set_index("distance_bin_cm")
        .value
    )
    near, far = "(40, 80]", ["(80, 120]", "(120, 200]"]
    if not (local.get(near, 0.0) > 0.0 and all(local.get(b, 0.0) == 0.0 for b in far)):
        raise RuntimeError(
            "The caption says absorbed local coding reaches about 80 cm and no "
            "further; the summary's local fractions disagree."
        )
    if not all(ratio.get(b, np.nan) > 1.0 for b in far):
        raise RuntimeError(
            "The caption says the window floor still leaves events too long "
            "beyond 80 cm; the summary's duration ratios disagree."
        )


def check_controls_claims(controls: pd.DataFrame) -> None:
    """The caption's claims about the controls, checked against their summary."""
    no_replay = controls[
        (controls["summary"] == "no_replay") & (controls.measure == "n_events")
    ]
    if len(no_replay) != len(OUTPUTS) or (no_replay.value != 0).any():
        raise RuntimeError(
            "The caption says no output reported an event where there was no "
            "replay; the controls summary disagrees."
        )
    for condition in ("jump_120ms", "jump_200ms"):
        if sweep_statistics.controls_value(
            controls,
            "parts",
            "segmentation_spanning",
            condition,
            80.0,
            "sliding_published",
        ):
            raise RuntimeError(
                "The caption says the sliding window never reports one event across "
                "an 80 cm jump; the controls summary disagrees."
            )
        spanning = sweep_statistics.parts_by_level(
            controls, (condition,), "segmentation_spanning", "state_space_canonical"
        )
        if (spanning < 0.5).any():
            raise RuntimeError(
                "The caption says the state-space detector reports one event over "
                "both parts of most jumps at every size; the summary disagrees."
            )
    continuing = {
        detector: sweep_statistics.carving_statistics(
            controls, ("chain_continuing",), "spanning", detector
        )
        for detector in ("sliding_published", "state_space_canonical")
    }
    for detector in ("sliding_published", "state_space_canonical"):
        if continuing[detector]["at_lowest_level"] < 0.5:
            raise RuntimeError(
                "The caption says both detectors report one event over a chain at "
                "the shortest gap; the controls summary disagrees."
            )
    for detector in ("sliding_published", "state_space_canonical"):
        recovery = sweep_statistics.part_recovery_statistics(
            controls, MULTIPART_CONDITIONS, detector
        )
        if recovery["recovery_in_event"] <= recovery["recovery_alone"]:
            raise RuntimeError(
                "The caption says pooled recovery is higher inside the event than "
                "alone; the controls summary disagrees."
            )
        if recovery["n_trials_more_in_event"] <= recovery["n_trials_fewer_in_event"]:
            raise RuntimeError(
                "The caption says more trials gain parts inside the event than "
                "lose them; the controls summary disagrees."
            )
    for detector in ("sliding_published", "state_space_canonical"):
        if continuing[detector]["at_highest_level"]:
            raise RuntimeError(
                "The caption says neither detector reports one event over a chain "
                "at the longest gap; the controls summary disagrees."
            )
    elsewhere = sweep_statistics.parts_by_level(
        controls, ("chain_elsewhere",), "segmentation_spanning", "sliding_published"
    )
    # a missing share is not zero
    if (elsewhere != 0).any():
        raise RuntimeError(
            "The caption says the sliding window never reports one event over a "
            "chain whose parts are far apart; the controls summary disagrees."
        )


def check_heldout_claims(heldout: pd.DataFrame) -> None:
    """The caption's claims about held-out evidence, checked against its summary."""
    if "comparator_only" in sweep_statistics.heldout_event_classes(heldout):
        raise RuntimeError(
            "The caption says the sliding window reported no event our detector "
            "missed; the held-out summary holds comparator-only events."
        )
    placeless = {
        estimator: sweep_statistics.heldout_fold_statistics(
            heldout,
            tuple(FOLD_HATCH),
            ("ours_only", "shared"),
            "no_content",
            estimator=estimator,
        )
        for estimator in ("state_space", "sliding_window")
    }
    if any(statistics["n_folds_with_support"] for statistics in placeless.values()):
        raise RuntimeError(
            "The caption says no informative place-less burst event has "
            "held-out support; the held-out summary disagrees."
        )
    by_class = heldout[heldout["summary"] == "by_class"]
    local = by_class[(by_class.truth == "local") & (by_class.measure == "n_events_all")]
    if local.value.sum() > 0:
        raise RuntimeError(
            "The caption says no event of pure local coding reached the held-out "
            "test; the held-out summary holds some."
        )
    if not placeless["state_space"]["n_events"]:
        raise RuntimeError(
            "The caption states a place-less burst result; no place-less burst "
            "event is informative, so there is none to state."
        )
    for fold in FOLD_HATCH:
        for estimator in ("state_space", "sliding_window"):
            ours_only = sweep_statistics.pooled_heldout_rate(
                heldout,
                fold,
                ("ours_only",),
                "continuous_replay",
                estimator=estimator,
            )
            if ours_only["n_events"] and ours_only["rate"] < 0.8:
                raise RuntimeError(
                    "The caption says most informative replays only our detector "
                    "reported have held-out support in every fold; the summary "
                    "disagrees."
                )


def recording_numbers(decoded: dict) -> dict[str, float | int]:
    """The numbers the caption states about the one labeled recording (a-f).

    The caption no longer interpolates them: they go to
    :data:`RECORDING_NUMBERS_PATH`, ``continuum_swr_replay.reported.detection``
    reads them into Part 1's summary, and the emitter writes them into the
    caption's markers. Durations in ms, distances in cm, shares in [0, 1].
    """
    outbound = shuffle_of(decoded, CONTINUOUS_OUTBOUND_EVENT)
    inbound = shuffle_of(decoded, CONTINUOUS_INBOUND_EVENT)
    short = shuffle_of(decoded, SHORT_REPLAY_EVENT)
    stray_full, stray_pieces, stray_sequence = stray_window_dispersions(decoded)
    stray = decoded["shuffles"][stray_sequence]
    zoomed = (
        CONTINUOUS_OUTBOUND_EVENT,
        SHORT_REPLAY_EVENT,
        SHORT_PATH_EVENT,
        REMOTE_STATIONARY_EVENT,
    )
    return {
        "outbound_duration_ratio": duration_ratio(decoded, CONTINUOUS_OUTBOUND_EVENT),
        "outbound_local_fraction": local_fraction_of_event(
            decoded, CONTINUOUS_OUTBOUND_EVENT
        ),
        "outbound_p_spread": float(outbound.p_spread),
        "outbound_p_jump": float(outbound.p_jump),
        "outbound_draws_from_passing": draws_from_passing(outbound),
        "inbound_p_spread": float(inbound.p_spread),
        "inbound_p_jump": float(inbound.p_jump),
        "short_replay_duration_ratio": duration_ratio(decoded, SHORT_REPLAY_EVENT),
        "short_replay_local_fraction": local_fraction_of_event(
            decoded, SHORT_REPLAY_EVENT
        ),
        "short_replay_p_spread": float(short.p_spread),
        "short_replay_p_jump": float(short.p_jump),
        "short_path_dispersion_cm": best_dispersion_cm(decoded, SHORT_PATH_EVENT),
        "remote_stationary_dispersion_cm": best_dispersion_cm(
            decoded, REMOTE_STATIONARY_EVENT
        ),
        "state_space_bound_error_ms": bound_error_ms(decoded, zoomed),
        "stray_window_distance_cm": abs(
            decoded["stray_window_remote_cm"] - decoded["stray_window_local_cm"]
        ),
        "stray_window_local_fraction": local_fraction_of_event(
            decoded, STRAY_WINDOW_EVENT
        ),
        "stray_window_duration_ms": 1000
        * float(decoded["candidates"].duration_s[stray_sequence]),
        "stray_window_dispersion_cm": stray_full,
        "stray_window_pieces_dispersion_cm": stray_pieces,
        "stray_window_p_spread": float(stray.p_spread),
        "stray_window_p_jump": float(stray.p_jump),
    }


def write_recording_numbers(numbers: dict, n_neurons: int, path: Path) -> None:
    """Save :func:`recording_numbers` with the provenance the sweeps' CSVs carry."""
    path.write_text(
        json.dumps(
            {
                "produced_by": produced_by(__file__),
                "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": described_commit(),
                "n_neurons": n_neurons,
                "numbers": numbers,
            },
            indent=1,
            allow_nan=False,
        )
        + "\n"
    )


def read_summary(path: Path, producer: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No summary at {path}. Run `python {producer}` first.")
    return pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> None:
    # the CPU unless JAX_PLATFORMS names another
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neurons", type=int, default=25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--controls-summary", type=Path, default=CONTROLS_SUMMARY_PATH)
    parser.add_argument("--heldout-summary", type=Path, default=HELDOUT_SUMMARY_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/memoryless_comparison")
    )
    parser.add_argument(
        "--recording-numbers", type=Path, default=RECORDING_NUMBERS_PATH
    )
    args = parser.parse_args(argv)

    config = MemorylessSequenceConfig(sampling_frequency=500.0)
    summary = read_summary(
        args.summary, "scripts/detection/sweep_memoryless_detection.py"
    )
    controls = read_summary(
        args.controls_summary, "scripts/detection/sweep_memoryless_controls.py"
    )
    heldout = read_summary(
        args.heldout_summary, "scripts/detection/sweep_memoryless_heldout.py"
    )
    check_caption_settings(config)
    check_summary_claims(summary)
    check_controls_claims(controls)
    check_heldout_claims(heldout)
    fig, decoded = make_figure(args.n_neurons, config, summary, controls, heldout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "memoryless_comparison", base_dir=args.output_dir)
    plt.close(fig)
    write_recording_numbers(
        recording_numbers(decoded), args.n_neurons, args.recording_numbers
    )

    candidates = decoded["candidates"]
    print(
        f"{len(candidates.runs)} candidate sequences, "
        f"{int(candidates.is_replay.sum())} accepted"
    )
    for sequence in np.flatnonzero(candidates.is_replay):
        start, stop = candidates.runs[sequence]
        print(
            f"  [{decoded['midpoint_s'][start]:7.3f}, {decoded['midpoint_s'][stop]:7.3f}] "
            f"dur {candidates.duration_s[sequence]:.3f} s  "
            f"disp {candidates.dispersion_cm[sequence]:5.1f} cm  "
            f"local {100 * local_fraction(decoded, sequence):.0f}%"
        )
    print(f"saved to {args.output_dir}; caption numbers to {args.recording_numbers}")


if __name__ == "__main__":
    main()
