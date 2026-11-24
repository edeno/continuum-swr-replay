"""Figure 2: what it draws, from which values, and what it refuses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from continuum_swr_replay.analysis.detector_comparison import (  # noqa: E402
    nonlocal_comparison,
)
from continuum_swr_replay.analysis.event_definition import (  # noqa: E402
    analyzed_time,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (  # noqa: E402
    animal_signal_counts,
    boundary_offset_distribution,
    boundary_offsets,
    overlap_class_animal_shares,
    overlap_class_statistics,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (  # noqa: E402
    RIPPLE_POWER_EDGES,
    nonlocal_counts,
)
from continuum_swr_replay.analysis.event_properties.triggered_analysis import (  # noqa: E402
    TRIGGERED_SIGNALS,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analyzed_animals,
)
from continuum_swr_replay.reported import inputs as stored_inputs  # noqa: E402
from continuum_swr_replay.reported.questions import FIGURES  # noqa: E402
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from continuum_swr_replay.visualization import heldout_gain_panels  # noqa: E402
from tests.support._figure02_tables import (  # noqa: E402
    GAIN_SPLIT,
    IN_EVENTS,
    INSIDE,
    OUTSIDE,
    OUTSIDE_EVENTS,
    assert_spread_about,
    dots_at,
    figure02_events,
    gain_statistics,
    has_zero_line,
    means_at,
    restriction_statistics,
    texts,
)
from tests.support._tetrode_holdout_tables import run_table  # noqa: E402

FRAME_COUNTS = {"swr": 3, "non_swr": 5, "total": 8}
"""The decode frames' event counts by SWR overlap that agree with
``figure02_events``'s overlap classes."""


@pytest.fixture
def figure02(offline):
    """The figure's module, imported with ``spyglass`` stubbed."""
    return offline("scripts.figures.generate_figure02")


# ---------------------------------------------------------------------------
# The whole figure, from small inputs
# ---------------------------------------------------------------------------


def _track() -> tuple[nx.Graph, list[tuple[int, int]]]:
    """A track of three arms from a center (3) to three wells (0, 1, 2), each
    arm's ``distance`` its length (cm), and its arms in drawing order."""
    positions = {0: (0.0, 100.0), 1: (90.0, -50.0), 2: (-90.0, -50.0), 3: (0.0, 0.0)}
    graph = nx.Graph()
    for node, position in positions.items():
        graph.add_node(node, pos=position)
    order = [(3, 0), (3, 1), (3, 2)]
    for arm, (first, second) in enumerate(order):
        distance = float(np.hypot(*np.subtract(positions[first], positions[second])))
        graph.add_edge(first, second, distance=distance, edge_id=arm)
    return graph, order


def _session(figure02) -> dict:
    """An example session over panel a's run and rest: the animal runs and
    rests on a three-arm track, five tetrodes spike, the decode is Local but
    for one non-local stretch, and two SWRs."""
    start = figure02.OVERVIEW_START_S - 1.0
    time = start + np.arange(0, figure02.OVERVIEW_DURATION_S + 2.0, 0.002)
    graph, order = _track()
    centers = np.arange(5.0, 330.0, 10.0)
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    probabilities = np.tile([0.9, 0.04, 0.03, 0.03], (time.size, 1))
    nonlocal_time = (time > start + 9.0) & (time < start + 9.2)
    probabilities[nonlocal_time] = [0.02, 0.02, 0.9, 0.06]
    position = 160.0 + 150.0 * np.sin(2 * np.pi * (time - start) / 8.0)
    position[time > start + 7.0] = 300.0
    speed = np.where(time > start + 7.0, 1.0, 30.0)
    bins = [
        (state, center) for state in states if state != "No-Spike" for center in centers
    ]
    posterior = np.full((time.size, len(bins)), 1.0 / len(centers))
    results = xr.Dataset(
        {
            "acausal_posterior": (("time", "state_bins"), posterior),
            "acausal_state_probabilities": (("time", "states"), probabilities),
        },
        coords={
            "time": time,
            "states": states,
            "state_bins": pd.MultiIndex.from_tuples(bins, names=["state", "position"]),
        },
    )
    angle = np.deg2rad(position)
    position_info = pd.DataFrame(
        {
            "linear_position": position,
            "head_speed": speed,
            "head_position_x": 50 * np.cos(angle),
            "head_position_y": 50 * np.sin(angle),
        },
        index=pd.Index(time, name="time"),
    )
    ripple = pd.Series(
        np.where((time > start + 8.0) & (time < start + 8.1), 6.0, 0.0), index=time
    )
    rng = np.random.default_rng(0)
    data = {
        "track_graph": graph,
        "linear_edge_order": order,
        "linear_edge_spacing": 15.0,
        "position_info": position_info,
        "zscored_ripple_consensus_trace": ripple,
        "ripple_times": pd.DataFrame(
            {
                "start_time": [start + 8.0, start + 10.0],
                "end_time": [start + 8.1, start + 10.05],
            }
        ),
        "spike_times": {
            "HPC": [np.sort(rng.uniform(time[0], time[-1], 500)) for _ in range(5)]
        },
    }
    env = SimpleNamespace(
        place_bin_centers_=centers, is_track_interior_=np.ones(centers.size, bool)
    )
    return {
        "data": data,
        "results": results,
        "detector": SimpleNamespace(encoding_model_={}),
        "env": env,
    }


def _examples(figure02) -> dict[str, tuple[float, float]]:
    """Panel b's windows: one event a second in the rest."""
    start = figure02.OVERVIEW_START_S - 1.0
    titles = [*figure02.EXAMPLES, "SWR, mostly Local"]
    return {
        title: (start + 8.0 + 0.6 * k, start + 8.1 + 0.6 * k)
        for k, title in enumerate(titles)
    }


def _detector_shares(scale: float) -> pd.DataFrame:
    """``nonlocal_events_overlapped``' rows: each figure detector's share."""
    detectors = nonlocal_comparison.FIGURE_DETECTORS
    shares = scale * np.linspace(0.3, 0.9, len(detectors))
    return pd.DataFrame(
        {
            "animal_mean": shares,
            "animals": [share + np.linspace(-0.05, 0.05, 5) for share in shares],
        },
        index=pd.Index(detectors, name="detector"),
    )


def _ripple_power_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored counts by ripple power of every analyzed animal, and the
    states by ripple power: 8 non-local bins below the lowest finite edge, 4
    above the highest, and 4 just above the threshold for every animal after
    the first; Local time at -1 z, non-local time at 3 z."""
    low, high = RIPPLE_POWER_EDGES[:-1], RIPPLE_POWER_EDGES[1:]
    animals = list(analyzed_animals())
    epochs = []
    for i, animal in enumerate(animals):
        n_nonlocal = np.zeros(len(low), dtype=int)
        n_nonlocal[0], n_nonlocal[-1] = 8, 4
        n_nonlocal[np.flatnonzero(low == 2.0)[0]] = 4 * i
        # each signal bin's non-local bins in the last probability bin, the rest
        # in the first
        for probability_low, n in ((0.0, 20 - n_nonlocal), (0.8, n_nonlocal)):
            epochs.append(
                pd.DataFrame(
                    {
                        "epoch_id": f"{animal}_e",
                        "animal": animal,
                        "signal": "ripple_power",
                        "signal_low": low,
                        "signal_high": high,
                        "probability_low": probability_low,
                        "probability_high": probability_low + 0.2,
                        "n_bins": n,
                    }
                )
            )
    counts = pd.concat(epochs, ignore_index=True)
    states = pd.DataFrame(
        [
            {"state": state, "animal": animal, "signal_low": lo, "signal_high": hi,
             "percent": 100.0 * (lo == peak)}
            for state, peak in (("local", -1.0), ("non_local", 3.0))
            for animal in animals
            for lo, hi in zip(low, high, strict=True)
        ]
    )  # fmt: skip
    return counts, states


def _triggered() -> tuple[dict, pd.DataFrame]:
    """Panel g's curves (each signal's quartiles by SWR overlap, a rise at the
    start) and the brain-state reference's levels."""
    time = np.linspace(-0.15, 0.15, 151)
    rise = np.where(time > 0, 1.0, 0.0)
    curves = {
        signal: {
            "time": time,
            **{
                f"{kind}_{name}": scale * rise + offset
                for kind, scale in (("overlap", 3.0), ("no_overlap", 1.5))
                for name, offset in (("q25", -0.5), ("median", 0.0), ("q75", 0.5))
            },
        }
        for signal in TRIGGERED_SIGNALS
    }
    reference = pd.DataFrame(
        {"rest": [0.0, 0.0, -0.2], "running": [np.nan, np.nan, 1.5]},
        index=["ripple", "mua", "theta"],
    )
    return curves, reference


PARTICIPATION = {
    ("no SWR", "control"): [0.3, 0.35, 0.4, 0.27, 0.33],
    ("no SWR", "event"): [0.5, 0.55, 0.6, 0.45, 0.48],
    ("SWR", "control"): [0.45, 0.5, 0.36, 0.48, 0.42],
    ("SWR", "event"): [0.7, 0.8, 0.66, 0.72, 0.68],
}
"""The participation panel's animals: each group's controls and events."""


def _participation() -> pd.DataFrame:
    """``participation_statistics``' rows indexed by group and kind."""
    rows = {
        key: {
            "participation_animals": np.asarray(values),
            "animal_names": np.array(list("abcde")),
        }
        for key, values in PARTICIPATION.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["group", "kind"]
    return table


GAIN_OVER_ANIMAL = GAIN_SPLIT["gain_per_spike"]
GAIN_OVER_OTHER = ([0.2, 0.1, 0.35, 0.15, 0.18], [0.22, 0.3, 0.14, 0.25, 0.2])
"""The left-out panel: each comparison's animals, no SWR then SWR."""


def _inputs(figure02) -> dict:
    """What :func:`read_inputs` gives, each panel's from a small table."""
    events = figure02_events()
    counts, states = _ripple_power_counts()
    kay = pd.Series(
        {
            "no_nonlocal_fraction_animal_mean": 0.6,
            "mostly_local_fraction_animal_mean": 0.4,
            "n_events": 1234,
        }
    )
    content = pd.DataFrame(
        {
            "with_nonlocal": np.linspace(0.35, 0.45, 5),
            "mostly_local": np.linspace(0.35, 0.45, 5),
            "other": np.linspace(0.15, 0.25, 5),
        }
    )
    session = _session(figure02)
    return {
        "session": session,
        "overview": pd.DataFrame(
            {
                "event_start_s": [figure02.OVERVIEW_START_S + 8.0],
                "event_end_s": [figure02.OVERVIEW_START_S + 8.2],
            }
        ),
        "examples": _examples(figure02),
        "swr_content": (kay, content),
        "event_overlap": (
            overlap_class_statistics(events),
            overlap_class_animal_shares(events),
        ),
        "detector_overlap": (_detector_shares(1.0), _detector_shares(0.8)),
        "boundary_offsets": boundary_offset_distribution(
            boundary_offsets(events), events
        ),
        "event_traces": _triggered(),
        "participation": _participation(),
        "nonlocal_ripple_power": (
            animal_signal_counts(nonlocal_counts(counts, "ripple_power")),
            states,
        ),
        "left_out": {
            "animal": gain_statistics(*GAIN_OVER_ANIMAL),
            "other": gain_statistics(*GAIN_OVER_OTHER),
        },
        "decoder_comparison": restriction_statistics(),
    }


@pytest.fixture
def drawn(figure02):
    """The figure drawn from :func:`_inputs`."""
    fig = figure02.generate_figure(_inputs(figure02))
    yield fig
    plt.close(fig)


def test_the_figure_draws_its_panels_in_the_cited_order(figure02, drawn):
    """The panels a to k in the layout's order, each row's letters at one
    height and each column's at one left edge, inside the journal's limits;
    the layout is the question list's Figure 2."""
    assert figure02.PANELS == (
        "overview", "examples", "swr_content", "event_overlap", "detector_overlap",
        "boundary_offsets", "event_traces", "participation", "nonlocal_ripple_power",
        "left_out", "decoder_comparison",
    )  # fmt: skip
    assert list(figure02.read_inputs.__annotations__) == [
        "nwb_file_name",
        "epoch_name",
        "return",
    ]
    letters = {t.get_text(): t.get_position() for t in drawn.texts}
    assert list(letters) == list(figure02.LETTERS.values())
    assert "".join(letters) == FIGURES["2"].panels
    assert FIGURES["2"].script == "scripts/figures/generate_figure02.py"
    rows = [letters[row[0]][1] for row in figure02.LETTER_ROWS]
    assert rows == sorted(rows, reverse=True)
    for row in figure02.LETTER_ROWS:
        assert len({letters[k][1] for k in row}) == 1, row
    for column in figure02.LETTER_COLUMNS:
        assert len({letters[k][0] for k in column}) == 1, column
    assert figure02.LETTER_ROWS == ("a", "b", "cdef", "ghi", "jk")
    assert figure02.LETTER_COLUMNS == ("abcgj", "d", "ehk", "fi")
    figure02.check_journal_limits(drawn)


def test_the_figure_is_placed_in_the_font_of_a_machine_without_arial(figure02):
    """The figure settles and passes ``check_placement`` (in
    :func:`generate_figure`) in DejaVu Sans too, the font matplotlib falls back
    to (the continuous-integration machine's), whose glyphs are wider."""
    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        fig = figure02.generate_figure(_inputs(figure02))
    plt.close(fig)


def test_each_panels_values_carry_its_letter(figure02):
    """The events summary tags each panel's values with the panel's letter, as
    the figure places it; the restriction panel draws the summary's own
    definitions."""
    from continuum_swr_replay.analysis.event_evidence import restriction_totals
    from continuum_swr_replay.reported import events

    letter = {panel: f"2{figure02.LETTERS[panel]}" for panel in figure02.PANELS}
    for tag, panel in {
        "swr_content": "swr_content",
        "cohort": "event_overlap",
        "nonlocal_found": "detector_overlap",
        "boundaries": "boundary_offsets",
        "ripple": "event_traces",
        "mua": "event_traces",
        "theta": "event_traces",
        "participation": "participation",
        "nonlocal_power": "nonlocal_ripple_power",
        "tetrode_holdout_extra_figure": "left_out",
        "nonlocal_restriction_figure": "decoder_comparison",
    }.items():
        assert events.TAGS[tag].figures == (letter[panel],), tag
    # the gain over local coding is the supplement's too
    assert events.TAGS["tetrode_holdout"].figures[0] == letter["left_out"]
    assert events.TAGS["shared_content"].figures == ()
    # one definition of the drawn statistic and its choices, in the analysis
    # layer, used by both
    assert (
        figure02.restriction_figure_statistics
        is restriction_totals.restriction_figure_statistics
    )
    for name in (
        "RESTRICTION_FIGURE_EVENT_GROUP",
        "RESTRICTION_FIGURE_FAMILY",
        "RESTRICTION_FIGURE_INSIDE_SWR",
        "RESTRICTION_FIGURE_OUTSIDE_EVENTS",
        "RESTRICTION_FIGURE_OUTSIDE_SWR",
    ):
        assert getattr(figure02, name) is getattr(restriction_totals, name), name
    # panel e's detectors are the analysis layer's
    assert list(figure02.DETECTOR_LABELS) == list(nonlocal_comparison.FIGURE_DETECTORS)


# ---------------------------------------------------------------------------
# The panels
# ---------------------------------------------------------------------------


def test_the_examples_are_checked_rows_of_the_event_table(figure02):
    """Panel b: each example is the one row of its event_id, of the subclass it
    is drawn as, a partial one running past its SWR on its title's side alone;
    the Local SWR is the source's, empty of events, and mostly Local."""
    ids = [event_id for event_id, _ in figure02.EXAMPLES.values()]
    subclasses = [subclass for _, subclass in figure02.EXAMPLES.values()]
    nan = np.nan
    events = pd.DataFrame(
        {
            "event_id": [*ids, 1],
            "overlap_subclass": [*subclasses, "no_overlap"],
            "event_start_s": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "event_end_s": [10.1, 20.1, 30.1, 40.1, 50.1, 60.1],
            "matched_swr_start": [9.9, 20.05, 29.9, nan, nan, nan],
            "matched_swr_end": [10.2, 20.2, 30.05, nan, nan, nan],
        }
    )
    start = figure02.LOCAL_SWR_START_S
    ripples = pd.DataFrame({"start_time": [start], "end_time": [start + 0.1]})

    def results(local: float) -> xr.Dataset:
        time = start + np.arange(-0.05, 0.15, 0.002)
        states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
        probabilities = np.tile([local, 1 - local, 0.0, 0.0], (time.size, 1))
        return xr.Dataset(
            {"acausal_state_probabilities": (("time", "states"), probabilities)},
            coords={"time": time, "states": states},
        )

    windows = figure02.example_windows(events, ripples, results(0.95))
    assert list(windows) == [*figure02.EXAMPLES, "SWR, mostly Local"]
    assert windows["Inside SWR"] == (10.0, 10.1)
    assert windows["SWR, mostly Local"] == (start, start + 0.1)

    wrong = events.assign(
        overlap_subclass=["no_overlap", *events["overlap_subclass"][1:]]
    )
    with pytest.raises(AssertionError, match="Inside SWR"):
        figure02.example_windows(wrong, ripples, results(0.95))
    both = events.assign(matched_swr_end=[10.2, 20.08, 30.05, *[np.nan] * 3])
    with pytest.raises(AssertionError, match="Starts before SWR.*start alone"):
        figure02.example_windows(both, ripples, results(0.95))
    with pytest.raises(AssertionError, match="not one of the source's"):
        figure02.example_windows(
            events, ripples.assign(start_time=start + 1.0), results(0.95)
        )
    overlapping = pd.concat(
        [
            events,
            pd.DataFrame(
                {
                    "event_id": [2],
                    "overlap_subclass": ["swr_contains_event"],
                    "event_start_s": [start + 0.01],
                    "event_end_s": [start + 0.05],
                }
            ),
        ]
    )
    with pytest.raises(AssertionError, match="overlaps the Local SWR"):
        figure02.example_windows(overlapping, ripples, results(0.95))
    with pytest.raises(AssertionError, match="not mostly Local"):
        figure02.example_windows(events, ripples, results(0.5))


def test_the_event_overlap_panel_prints_the_digits_of_the_events_summary(figure02):
    """Panel d: the bars are the mean over animals of the animals' dots, their
    percents rounded by the summary's rule (a half rounds away from zero: 12.5
    % is 13), the title gives the count of every event, and each animal is one
    dot per bar."""
    from continuum_swr_replay.reported.events import event_cohort_values
    from continuum_swr_replay.reported.formatting import formatted

    events = figure02_events()
    animals = overlap_class_animal_shares(events)
    fig, ax = plt.subplots()
    figure02.draw_events_bars(ax, overlap_class_statistics(events), animals)
    assert [t.get_text() for t in ax.texts] == ["13%", "25%", "63%"]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["No SWR", "Partly in SWR", "Inside SWR"]  # bottom to top
    assert ax.get_title(loc="right") == "Non-local events (n = 8)"
    assert all(len(dots.get_offsets()) == len(animals) for dots in ax.collections)
    widths = [bar.get_width() for bar in ax.patches]  # top to bottom
    np.testing.assert_allclose(
        widths, 100 * animals[list(figure02.OVERLAP_CLASSES)].mean().to_numpy()
    )
    cohort = {v.key: v for v in event_cohort_values(events, "s")}
    assert formatted(cohort["non_swr_fraction_animal_mean"]) == "63 %"
    plt.close(fig)


def test_the_participation_panel_joins_each_animals_rest_to_its_events(figure02):
    """Panel h: for no-SWR then SWR events, one line per animal from its
    Local-rest controls to its events, the controls in the Local state's color
    and the events in their class's, in percent."""
    fig, ax = plt.subplots()
    figure02.draw_participation(ax, _participation())
    for group, color in heldout_gain_panels.GROUP_COLORS.items():
        rest_x, event_x = figure02.REST_EVENT_X[group]
        for x, kind, dot_color in (
            (rest_x, "control", figure02.COLORS["Local"]),
            (event_x, "event", color),
        ):
            (dots,) = [
                c for c in ax.collections
                if isinstance(c, matplotlib.collections.PathCollection)
                and np.allclose(c.get_offsets()[:, 0], x)
            ]  # fmt: skip
            np.testing.assert_allclose(
                dots.get_offsets()[:, 1], 100 * np.asarray(PARTICIPATION[(group, kind)])
            )
            np.testing.assert_allclose(dots.get_facecolor()[0], to_rgba(dot_color))
        pairs = [
            line for line in ax.lines
            if np.allclose(line.get_xdata(), (rest_x, event_x))
        ]  # fmt: skip
        assert len(pairs) == 5
    assert ax.get_ylabel() == "Tetrodes active (%)"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Matched\nlocal\nrest", "No\nSWR", "Matched\nlocal\nrest", "SWR",
    ]  # fmt: skip
    plt.close(fig)


def test_the_ripple_power_panel_writes_the_share_the_events_summary_gives(figure02):
    """Panel i: the share of non-local time below the SWR threshold is the
    events summary's value (the animals' mean, the bins beyond the axis
    counted), rounded by the summary's rule; each animal is one line on the
    bottom axis, with the animals' mean."""
    from matplotlib.offsetbox import AnchoredOffsetbox

    from continuum_swr_replay.reported.events import nonlocal_power_values
    from continuum_swr_replay.reported.formatting import formatted
    from continuum_swr_replay.visualization import signal_share_panels

    counts, states = _ripple_power_counts()
    n_animals = len(analyzed_animals())
    fig, (ax_spread, ax_share) = plt.subplots(2, 1, sharex=True)
    figure02.draw_share_by_power(
        ax_spread,
        ax_share,
        animal_signal_counts(nonlocal_counts(counts, "ripple_power")),
        states,
    )
    value = {v.key: v for v in nonlocal_power_values(counts, "s")}[
        "nonlocal_time_below_swr_threshold_fraction"
    ]
    # 8 / (12 + 4 i) over i = 0..4: mean 0.4371, although no bin on the axis
    # below the threshold holds a non-local bin
    assert formatted(value) == "44 %"
    assert "44% below\nthreshold" in texts(ax_spread)
    # the top strip: the two classes' means and the threshold; the bottom: the
    # animals, their mean and the threshold
    assert len(ax_spread.lines) == 3
    assert len(ax_share.lines) == n_animals + 2
    local, nonlocal_ = ax_spread.lines[:2]
    assert local.get_xdata()[np.argmax(local.get_ydata())] == pytest.approx(-0.875)
    assert nonlocal_.get_xdata()[np.argmax(nonlocal_.get_ydata())] == pytest.approx(
        3.125
    )
    (title,) = [a for a in ax_spread.artists if isinstance(a, AnchoredOffsetbox)]
    words = title.get_child().get_children()
    assert [w.get_text() for w in words] == ["Local", "and", "non-local", "time"]
    for word, (_, state) in zip(words, signal_share_panels.STRIP_TITLE, strict=True):
        if state is not None:  # a class's name takes its curve's color
            assert word._text.get_color() == signal_share_panels.readable_text_color(
                signal_share_panels.STATE_STYLES[state][1]
            )
    for ax in (ax_spread, ax_share):
        assert ax.get_xlim() == (-2.0, 10.0)
    plt.close(fig)


def test_the_left_out_panel_draws_both_comparisons_on_one_axis(figure02):
    """Panel j: the drawing of the test under the heading; then one axis with
    the fit to the decoded place over 1, the animal's place and 2, another
    event's place, each for no-SWR then SWR events (a dot per animal in the
    group's color, the mean with its interval beside them), the comparisons
    named under the axis, the groups named in their colors, a dashed zero line
    named "equal fit" and the y label."""
    from continuum_swr_replay.visualization import evidence_drawings

    fig, (ax_drawing, ax) = plt.subplots(1, 2)
    statistics = {
        "animal": gain_statistics(*GAIN_OVER_ANIMAL),
        "other": gain_statistics(*GAIN_OVER_OTHER),
    }
    figure02.draw_left_out_fit(ax_drawing, ax, statistics)
    fig.canvas.draw()
    animals = {"animal": GAIN_OVER_ANIMAL, "other": GAIN_OVER_OTHER}
    colors = heldout_gain_panels.GROUP_COLORS
    for (comparison, group), x in figure02.FIT_X.items():
        values = animals[comparison][list(colors).index(group)]
        dots = dots_at(ax, x)
        np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
        np.testing.assert_allclose(dots.get_facecolors()[0], to_rgba(colors[group]))
        assert np.allclose(means_at(ax, x), np.mean(values))
    assert has_zero_line(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(
        figure02.FIT_GROUPS.values()
    )
    assert {"equal fit", "No SWR", "SWR"} <= set(texts(ax))
    assert ax.get_ylabel() == figure02.FIT_YLABEL
    drawn = texts(ax_drawing)
    assert figure02.LEFT_OUT_HEADING in drawn
    assert set(evidence_drawings.LEFT_OUT_TEXTS.values()) <= set(drawn)
    plt.close(fig)


def test_the_decoder_comparison_draws_its_test_and_two_axes(figure02):
    """Panel k: the drawing of the two decoders under the heading; then the
    full minus the SWR-only decoder on two axes, "All immobility time" (inside
    SWRs, outside SWRs) and, on its own scale, "Outside SWRs" (the no-SWR
    held-out events, the other time): dark gray dots spread about each column,
    the mean with its interval beside them, a dashed zero line named "equal
    fit", one y label."""
    from continuum_swr_replay.visualization import evidence_drawings

    fig, (ax_drawing, ax_time, ax_outside) = plt.subplots(1, 3)
    figure02.draw_decoder_need(
        ax_drawing, ax_time, ax_outside, restriction_statistics()
    )
    fig.canvas.draw()
    for ax, columns in (
        (ax_time, ((0.0, INSIDE), (1.0, OUTSIDE))),
        (ax_outside, ((0.0, IN_EVENTS), (1.0, OUTSIDE_EVENTS))),
    ):
        for x, values in columns:
            dots = dots_at(ax, x)
            np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
            assert_spread_about(dots, x)
            np.testing.assert_allclose(
                dots.get_facecolors()[0], to_rgba(heldout_gain_panels.NEUTRAL_DOTS)
            )
            assert np.allclose(means_at(ax, x), np.mean(values))
        assert has_zero_line(ax)
        assert texts(ax) == ["equal fit"]
    assert ax_time.get_ylim()[0] < min(INSIDE)
    assert [t.get_text() for t in ax_time.get_xticklabels()] == [
        "Inside\nSWRs",
        "Outside\nSWRs",
    ]
    assert [t.get_text() for t in ax_outside.get_xticklabels()] == [
        "No-SWR\nevents",
        "Other\ntime",
    ]
    assert (ax_time.get_title(), ax_outside.get_title()) == (
        "All immobility time",
        "Outside SWRs",
    )
    assert ax_time.get_ylabel() == figure02.DECODER_YLABEL
    assert ax_outside.get_ylabel() == ""
    drawn = texts(ax_drawing)
    assert figure02.DECODER_HEADING in drawn
    assert set(evidence_drawings.RESTRICTION_TEXTS.values()) <= set(drawn)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The inputs: the frames' counts
# ---------------------------------------------------------------------------


def test_an_event_table_that_is_not_the_frames_is_refused(figure02):
    """Panel d's counts come from the event table the summary cites, and the
    figure is refused when that table's event counts by SWR overlap are not the
    decode frames', or the table holds an event that is not immobile."""
    events = figure02_events()
    classes = overlap_class_statistics(events)
    figure02.check_table_against_frames(classes, FRAME_COUNTS)
    with pytest.raises(ValueError, match="counts by SWR overlap are not"):
        figure02.check_table_against_frames(
            classes, {**FRAME_COUNTS, "non_swr": 6, "total": 9}
        )
    moving = events.assign(whole_event_speed_eligible=[True] * 7 + [False])
    with pytest.raises(ValueError, match="not whole-event immobile"):
        overlap_class_statistics(moving)


def _write_event_table(events: pd.DataFrame) -> None:
    """``events`` as the canonical event table of the data root."""
    path = stored_inputs.default_inputs().event_table
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(analyzed_time.stamp_swr_source(events), path)


def _stored_inputs(figure02, monkeypatch, frames: dict) -> None:
    """The stored windows with ``frames`` as the frames' event counts (one
    epoch), the example session and the detector comparison stubbed."""
    epochs = pd.DataFrame(
        {
            "epoch_id": ["e"],
            "animal": ["a"],
            "no_spike_level": [0.0],
            "n_swr": [frames["swr"]],
            "n_non_swr": [frames["non_swr"]],
            "n_events": [frames["total"]],
        }
    )
    monkeypatch.setattr(figure02, "read_windows", lambda folder: (None, epochs))
    data = {"ripple_times": pd.DataFrame(columns=["start_time", "end_time"])}
    monkeypatch.setattr(
        figure02, "read_session", lambda *args: {"data": data, "results": None}
    )
    monkeypatch.setattr(
        figure02,
        "detector_comparison",
        lambda events: dict.fromkeys(
            ("kay", "kay_content_animals", "overlap", "with_nonlocal")
        ),
    )
    monkeypatch.setattr(figure02, "stored_windows_of", lambda windows, events: None)
    monkeypatch.setattr(figure02, "stored_triggered_curves", lambda windows: None)


def test_the_event_table_is_checked_against_the_frames_counts(
    figure02, processed_data_root, monkeypatch
):
    """The event table is checked against the frames' event counts the stored
    windows record."""
    _write_event_table(figure02_events())
    # the frames give one more non-SWR event than the table holds
    _stored_inputs(figure02, monkeypatch, {**FRAME_COUNTS, "non_swr": 6, "total": 9})
    with pytest.raises(ValueError, match="counts by SWR overlap are not"):
        figure02.read_inputs(epoch_name="e1")


def test_the_whole_table_is_checked_and_the_selected_events_drawn(
    figure02, processed_data_root, monkeypatch
):
    """Two non-SWR events on the track, between two visits: the frames hold
    them, so the check against the frames counts them; every panel is read
    from the other eight, the events that start inside a well visit."""
    outside = figure02_events().iloc[[3, 4]].assign(visit_context="track_leaf_arm")
    _write_event_table(pd.concat([figure02_events(), outside], ignore_index=True))
    _stored_inputs(figure02, monkeypatch, {**FRAME_COUNTS, "non_swr": 7, "total": 10})
    seen = []
    counted = figure02.overlap_class_statistics

    def count(table):
        seen.append(len(table))
        return counted(table)

    class Read(Exception):
        pass

    def stop(table, analyzed):
        seen.append(table)
        raise Read

    monkeypatch.setattr(figure02, "overlap_class_statistics", count)
    monkeypatch.setattr(figure02, "event_table_epochs", stop)
    monkeypatch.setattr(figure02, "parse_nwb_filename", lambda name: ("a", "d1"))
    with pytest.raises(Read):
        figure02.read_inputs(epoch_name="e1")
    # the check saw all ten
    assert seen[0] == 10
    assert len(seen[1]) == 10


# ---------------------------------------------------------------------------
# The inputs: what is refused
# ---------------------------------------------------------------------------


class _Read(Exception):
    """Stops the inputs at the read under test."""


def _one_event_outside(figure02, root: Path, monkeypatch):
    """The figure with its eight events in a visit and a ninth on the track,
    every reader before the held-out tetrode run's stubbed, a held-out tetrode
    run held out in folds of whole trials, and a per-bin facts run made from
    the current session cache, all under the data root ``root``."""
    from continuum_swr_replay.analysis.event_evidence import heldout_runs
    from continuum_swr_replay.analysis.event_tables import prepare_data
    from continuum_swr_replay.paths import (
        EVENT_BIN_FACTS_DIR,
        EVENT_TETRODE_HOLDOUT_DIR,
    )

    inside = figure02_events().assign(event_id=range(8))
    outside = inside.iloc[[3]].assign(event_id=99, visit_context="track_leaf_arm")
    _write_event_table(pd.concat([inside, outside]))
    _stored_inputs(figure02, monkeypatch, {**FRAME_COUNTS, "non_swr": 6, "total": 9})
    monkeypatch.setattr(prepare_data, "SESSION_CACHE_DIR", root / "cache")
    entry = prepare_data.behavioral_context_cache_path({"session_id": "e"})
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"the visits")
    (root / EVENT_TETRODE_HOLDOUT_DIR).mkdir(parents=True)
    (root / EVENT_TETRODE_HOLDOUT_DIR / "manifest.json").write_text(
        json.dumps({"settings": {"n_folds": 5, "fold_design": "trials"}})
    )
    (root / EVENT_BIN_FACTS_DIR).mkdir()
    (root / EVENT_BIN_FACTS_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {"session_cache": prepare_data.session_cache_record(["e"])},
                "epochs": {"e": {"status": "ok"}},
            }
        )
    )
    # each stored per-event table holds a row of every event, the ninth too;
    # the held-out event run is read by the shared reader, the held-out
    # tetrode run (with every column the runner writes) by its own
    rows = pd.DataFrame({"event_id": [*range(8), 99]})
    monkeypatch.setattr(
        stored_inputs,
        "read_complete_run",
        lambda *args: rows.assign(starts_in_well_visit=1),
    )
    monkeypatch.setattr(
        heldout_runs,
        "read_complete_run",
        lambda *args: run_table(event_id=[*range(8), 99]),
    )
    for name, value in {
        "parse_nwb_filename": lambda name: ("a", "d1"),
        "example_windows": lambda *args: {},
        "read_checked": lambda folder, name, file="counts.parquet": None,
        "reference_medians": lambda *args: None,
        "animal_nonlocal_counts": lambda *args: None,
        "state_distribution": lambda *args: None,
        "event_table_epochs": lambda *args: ["e"],
        "window_scores": lambda table: table.assign(kind="event"),
        "holdout_statistics": lambda windows, kind, column: None,
        "read_participation": lambda inputs, epochs, event_ids: None,
        "well_visit_times": lambda identifier: None,
        "check_visit_context_starts": lambda table, visits: None,
    }.items():
        monkeypatch.setattr(figure02, name, value)
    monkeypatch.setattr(stored_inputs, "restriction_inputs_note", lambda *args: None)


def _heldout_events_run(
    root: Path, facts_sha256=None, restriction_sha256=None, restriction_settings=None
) -> None:
    """A restriction run's manifest (the fold fit on whole trials, or
    ``restriction_settings``) and a held-out event run's manifest that records
    the current per-bin facts and restriction runs' digests (or
    ``facts_sha256``, ``restriction_sha256``)."""
    from continuum_swr_replay.paths import (
        EVENT_BIN_FACTS_DIR,
        EVENT_NONLOCAL_RESTRICTION_DIR,
        EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    )
    from continuum_swr_replay.runs.io import sha256_file

    restriction = root / EVENT_NONLOCAL_RESTRICTION_DIR
    restriction.mkdir()
    settings = restriction_settings or {"fit": "folds", "fold_design": "trials"}
    (restriction / "manifest.json").write_text(json.dumps({"settings": settings}))
    folder = root / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR
    folder.mkdir()
    facts = facts_sha256 or sha256_file(root / EVENT_BIN_FACTS_DIR / "manifest.json")
    restriction_digest = restriction_sha256 or sha256_file(
        restriction / "manifest.json"
    )
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "bin_facts": {"sha256": facts},
                    "nonlocal_restriction": {"sha256": restriction_digest},
                },
                "epochs": {"e": {"status": "ok", "bin_s": 0.002}},
            }
        )
    )


def test_the_held_out_tetrode_run_is_read_over_the_selected_events(
    figure02, processed_data_root, monkeypatch
):
    """Panel j is over the events that start inside a visit: the held-out
    tetrode run's rows of the ninth event (on the track) are not read."""
    _one_event_outside(figure02, processed_data_root, monkeypatch)
    seen = []

    def stop(windows):
        seen.append(windows)
        raise _Read

    monkeypatch.setattr(figure02, "window_scores", stop)
    with pytest.raises(_Read):
        figure02.read_inputs(epoch_name="e1")
    assert sorted(seen[0]["event_id"]) == list(range(8))


def test_a_held_out_tetrode_run_not_on_trial_folds_is_refused(
    figure02, processed_data_root, monkeypatch
):
    """Panel j, as the events summary: a held-out tetrode run whose folds are
    not whole trials is refused before it is read."""
    from continuum_swr_replay.paths import EVENT_TETRODE_HOLDOUT_DIR

    _one_event_outside(figure02, processed_data_root, monkeypatch)
    (processed_data_root / EVENT_TETRODE_HOLDOUT_DIR / "manifest.json").write_text(
        json.dumps({"settings": {"n_folds": 5, "fold_design": "blocks"}})
    )

    def refuse(*args):
        raise AssertionError("the held-out tetrode run was read without its check")

    monkeypatch.setattr(figure02, "window_scores", refuse)
    with pytest.raises(ValueError, match="whole trials"):
        figure02.read_inputs(epoch_name="e1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"restriction_sha256": "0" * 64},
            "made from another nonlocal_restriction run than the current one",
        ),
        ({"restriction_settings": {"n_folds": 5}}, "is not the fold fit"),
        (
            {"restriction_settings": {"fit": "folds", "fold_design": "contiguous"}},
            "not 'trials'",
        ),
    ],
    ids=["another restriction run", "not the fold fit", "contiguous blocks"],
)
def test_a_heldout_event_run_not_from_the_current_fold_fit_is_refused(
    figure02, processed_data_root, monkeypatch, change, message
):
    """Panel k draws only the fold fit on whole trials, from a held-out event
    run made from the restriction run as it is now."""
    _one_event_outside(figure02, processed_data_root, monkeypatch)
    _heldout_events_run(processed_data_root, **change)

    def refuse(*args):
        raise AssertionError("panel k was drawn from another restriction run")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError, match=message) as refused:
        figure02.read_inputs(epoch_name="e1")
    assert "detect_heldout_events.py" in str(refused.value)


def test_a_restriction_run_off_the_current_decodes_or_swr_source_is_refused(
    figure02, processed_data_root, monkeypatch
):
    """Panel k is not drawn from a restriction run made from other decode
    frames or another SWR source than the current ones: the summary's reason,
    and the runner to make again."""
    _one_event_outside(figure02, processed_data_root, monkeypatch)
    _heldout_events_run(processed_data_root)
    seen = []

    def note(folder, epochs):
        seen.append((folder.name, list(epochs)))
        return "the restriction run records another SWR source; rerun x before citing"

    def refuse(*args):
        raise AssertionError("panel k was drawn from other decodes")

    monkeypatch.setattr(stored_inputs, "restriction_inputs_note", note)
    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError) as refused:
        figure02.read_inputs(epoch_name="e1")
    assert str(refused.value) == (
        f"Figure 2: panel {figure02.LETTERS['decoder_comparison']} is not drawn: "
        "the restriction run records another SWR source. "
        "Rerun scripts/event_evidence/score_nonlocal_restriction.py and the runs "
        "made from it."
    )
    assert seen == [("event_nonlocal_restriction", ["e"])]


def test_the_heldout_events_that_start_inside_a_visit_are_read(
    figure02, processed_data_root, monkeypatch
):
    """Panel k reads the held-out event run: its events that start inside a
    well visit (not the two planted outside one) and its time sums whole."""
    _one_event_outside(figure02, processed_data_root, monkeypatch)
    _heldout_events_run(processed_data_root)
    heldout = pd.DataFrame(
        {"heldout_event_id": range(5), "starts_in_well_visit": [1, 0, 1, 1, 0]}
    ).astype({"starts_in_well_visit": bool})
    read = {"heldout_events": heldout, "heldout_time_sums": pd.DataFrame(range(9))}
    monkeypatch.setattr(
        stored_inputs, "read_complete_run", lambda folder, name, epochs: read[name]
    )
    seen = []

    def stop(windows, time_sums, bin_s):
        seen.append((windows, time_sums, bin_s))
        raise _Read

    monkeypatch.setattr(figure02, "restriction_figure_statistics", stop)
    with pytest.raises(_Read):
        figure02.read_inputs(epoch_name="e1")
    windows, time_sums, bin_s = seen[0]
    assert windows["heldout_event_id"].tolist() == [0, 2, 3]
    assert len(time_sums) == 9  # the time sums are read whole
    assert bin_s.to_dict() == {"e": 0.002}


@pytest.mark.parametrize("problem", ["no session cache", "another session cache"])
def test_bin_facts_made_from_other_visits_are_refused(
    figure02, processed_data_root, monkeypatch, problem
):
    """Panel k's time inside a well visit is the per-bin facts' flag: a facts
    run that records no session cache (made before it did) or another than the
    current one is refused, naming the runs to make again."""
    from continuum_swr_replay.paths import EVENT_BIN_FACTS_DIR

    _one_event_outside(figure02, processed_data_root, monkeypatch)
    path = processed_data_root / EVENT_BIN_FACTS_DIR / "manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "no session cache":
        manifest["inputs"].pop("session_cache")
    else:
        manifest["inputs"]["session_cache"] = {"sha256": "0" * 64}
    path.write_text(json.dumps(manifest))
    _heldout_events_run(processed_data_root)
    note = {
        "no session cache": "records no session-cache SHA-256",
        "another session cache": "other well visits than the current",
    }[problem]

    def refuse(*args):
        raise AssertionError("panel k was drawn from the facts without their check")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError, match=f"{EVENT_BIN_FACTS_DIR}.*{note}") as refused:
        figure02.read_inputs(epoch_name="e1")
    message = str(refused.value)
    # what is not current, then the runners to make again, in that order
    assert "before citing" not in message
    assert message.endswith(
        ". Rerun scripts/event_evidence/measure_bin_facts.py, then "
        "scripts/event_evidence/detect_heldout_events.py."
    )


def test_a_heldout_event_run_not_made_from_the_current_bin_facts_is_refused(
    figure02, processed_data_root, monkeypatch
):
    """Panel k draws the held-out event run, whose events' start and time sums
    take their well visits from the per-bin facts: a run that recorded another
    facts run than the current one is refused, naming the run to make again."""
    _one_event_outside(figure02, processed_data_root, monkeypatch)
    _heldout_events_run(processed_data_root, facts_sha256="0" * 64)

    def refuse(*args):
        raise AssertionError("panel k was drawn from another facts run's events")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(
        ValueError, match="made from another bin_facts run than the current one"
    ) as refused:
        figure02.read_inputs(epoch_name="e1")
    assert "detect_heldout_events.py" in str(refused.value)


def test_every_events_label_is_checked_against_the_visits(
    figure02, processed_data_root, monkeypatch
):
    """The ninth event (on the track, starting at 3 s) is labeled outside every
    visit; visits that hold it make the figure refuse the table."""
    from continuum_swr_replay.analysis.event_definition import cohort_selection

    _one_event_outside(figure02, processed_data_root, monkeypatch)
    monkeypatch.setattr(
        figure02,
        "check_visit_context_starts",
        cohort_selection.check_visit_context_starts,
    )
    monkeypatch.setattr(
        figure02, "event_table_epochs", lambda table, analyzed: ["ad1_e1_HPC"]
    )
    visits = (np.array([-1.0]), np.array([10.0]))
    monkeypatch.setattr(figure02, "well_visit_times", lambda identifier: visits)
    monkeypatch.setattr(
        cohort_selection, "event_epoch_ids", lambda t: pd.Series("ad1_e1_HPC", t.index)
    )
    with pytest.raises(ValueError, match="1 of 9 events' visit_context disagrees"):
        figure02.read_inputs(epoch_name="e1")


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"event_table": {"sha256": "0" * 64}}, "another event table"),
        ({"settings": {}}, "not made over the events that start inside"),
    ],
    ids=["another event table", "no subset"],
)
def test_a_participation_run_the_summary_marks_pending_is_refused(
    figure02, processed_data_root, monkeypatch, manifest, message
):
    """Panel h is refused when the events summary marks its values pending:
    the participation run was made from another event table, or not over the
    analyses' subset."""
    from continuum_swr_replay.paths import EVENT_PARTICIPATION_DIR
    from continuum_swr_replay.runs.io import sha256_file

    _write_event_table(figure02_events())
    inputs = stored_inputs.default_inputs()
    current = {"event_table": {"sha256": sha256_file(inputs.event_table)}}
    folder = processed_data_root / EVENT_PARTICIPATION_DIR
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({**current, **manifest}))

    def refuse(*args):
        raise AssertionError("panel h was read from a run the summary pends")

    monkeypatch.setattr(figure02, "read_complete_run", refuse)
    with pytest.raises(ValueError, match=message) as refused:
        figure02.read_participation(inputs, ["e"], pd.Series([0]))
    assert str(refused.value).startswith(
        f"Figure 2: panel {figure02.LETTERS['participation']} is not drawn: "
    )
    assert str(refused.value).endswith(
        "Rerun scripts/event_evidence/measure_event_participation.py."
    )


def test_a_stored_session_that_is_not_current_refuses_panel_a(figure02, monkeypatch):
    """Panels a and b read the stored example session; one that is not current
    is refused as panel a, naming the runner."""
    epoch = "wilbur20210329_02_r1_HPC"
    monkeypatch.setattr(
        figure02,
        "read_manifest",
        lambda run: {"epochs": {epoch: {"decode": {"sha256": "d" * 64}}}},
    )

    def stale(folder, epoch_id, decode_sha256):
        assert (folder.name, epoch_id, decode_sha256) == (epoch, epoch, "d" * 64)
        raise ValueError("it was made from another decode frame")

    monkeypatch.setattr(figure02, "read_example_session", stale)
    with pytest.raises(ValueError) as refused:
        figure02.read_session("wilbur20210329_.nwb", "02_r1")
    assert str(refused.value) == (
        f"Figure 2: panel {figure02.LETTERS['overview']} is not drawn: it was made "
        "from another decode frame. Rerun scripts/decoding/store_example_session.py."
    )
