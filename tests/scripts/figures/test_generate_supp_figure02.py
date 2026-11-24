"""Figure 2's supplement: what it draws and from which values."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.reported.questions import FIGURES  # noqa: E402
from scripts.figures import generate_supp_figure02 as figure  # noqa: E402
from tests.scripts.figures.test_figure_script_fixes import (  # noqa: E402
    GAIN_SPLIT,
    OUTSIDE,
    _figure02_events,
    _gain_statistics,
    _means_at,
    _measure_rows,
    _outside_rows,
    assert_spread_about,
)

TRIAL_FOLD_SETTINGS = {"n_folds": 5, "fold_design": "trials"}
MARGINS = {0.1: [0.022, 0.019, 0.026, 0.030, 0.021]}
MARGINS[0.25] = list(np.asarray(MARGINS[0.1]) - 0.001)
MARGINS[0.5] = list(np.asarray(MARGINS[0.1]) - 0.002)


def _parts() -> pd.DataFrame:
    """``outside_swr_subset_statistics``' rows of the margins indexed by subset,
    threshold, family and measure: each margin's gain per spike."""
    tables = []
    for threshold, values in MARGINS.items():
        rows = _measure_rows({"gain_per_spike": values}).reset_index()
        tables.append(rows.assign(subset="beyond_swr_margin", threshold_s=threshold))
    return pd.concat(tables).set_index(["subset", "threshold_s", "family", "measure"])


NO_SPIKE_TIME = np.linspace(-0.15, 0.15, 151)


def _no_spike_trace() -> dict:
    """MUA at 1 z before the start, falling to -1 z after it."""
    median = np.where(NO_SPIKE_TIME < 0, 1.0, -1.0)
    return {
        "time": NO_SPIKE_TIME,
        "q25": median - 0.5,
        "median": median,
        "q75": median + 0.5,
        "level": 0.2,
        "periods": pd.Series({"a": 3, "b": 2}),
        "median_duration": 0.018,
    }


def _timing() -> pd.DataFrame:
    """``boundary_timing_statistics``' nine cells: the animals' mean shares
    0.05 to 0.7 (the middle cell the largest), each animal range 0.01 either
    side, the pooled shares (not drawn) a little off."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        TIMING_CLASSES,
    )

    shares = np.array([0.05, 0.05, 0.05, 0.05, 0.7, 0.15, 0.05, 0.1, 0.05])
    cells = pd.MultiIndex.from_product(
        [TIMING_CLASSES, TIMING_CLASSES], names=["start", "end"]
    )
    return pd.DataFrame(
        {
            "share_animal_mean": shares, "animal_min": shares - 0.01,
            "animal_max": shares + 0.01, "share": shares[::-1],
            "n_events": (1000 * shares).astype(int),
            "animals": [np.array([s - 0.01, s, s + 0.01]) for s in shares],
            "animal_names": [np.array(list("abc"))] * len(shares),
        },
        index=cells,
    )  # fmt: skip


def _mua_states() -> pd.DataFrame:
    """``state_distribution`` of MUA: Local time all at -1 z, non-local time
    all at 3 z, for two animals."""
    from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
        MUA_EDGES,
    )

    return pd.DataFrame(
        [
            {"state": state, "animal": animal, "signal_low": lo, "signal_high": hi,
             "percent": 100.0 * (lo == peak)}
            for state, peak in (("local", -1.0), ("non_local", 3.0))
            for animal in ("a", "b")
            for lo, hi in zip(MUA_EDGES[:-1], MUA_EDGES[1:], strict=True)
        ]
    )  # fmt: skip


def _hpd() -> pd.DataFrame:
    """``hpd_distribution`` of the HPD test's events."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        hpd_distribution,
    )
    from tests.analysis.event_properties.test_event_statistics import _hpd_events

    return hpd_distribution(_hpd_events())


def _probability() -> pd.DataFrame:
    """``animal_probability_distribution`` of two animals' epochs: most of the
    time in the first bin, some in the last, a little between."""
    from continuum_swr_replay.analysis.decode_quality import nonlocal_probability
    from tests.analysis.decode_quality.test_nonlocal_probability import _stored

    counts = _stored(
        {
            ("a", "1"): {0.0: 900, 0.5: 10, 0.98: 90},
            ("b", "1"): {0.0: 800, 0.3: 20, 0.9: 30, 0.98: 150},
        }
    )
    return nonlocal_probability.animal_probability_distribution(counts)


def _decoded_distance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``animal_distance_by_local_probability`` of two animals' sessions: the
    distance falls from 30 cm to 3 cm over the ten Local-probability bins."""
    from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
        animal_distance_by_local_probability,
    )

    centers = np.arange(0.05, 1.0, 0.1)
    rows = [
        {"animal": animal, "day": 1, "epoch": epoch, "prob_bin_center": c,
         "median_error_cm": scale * (30 - 27 * c)}
        for animal, epoch, scale in (("a", 1, 1.0), ("a", 2, 1.2), ("b", 1, 0.8))
        for c in centers
    ]  # fmt: skip
    return animal_distance_by_local_probability(pd.DataFrame(rows))


def _mua_counts() -> pd.DataFrame:
    """``animal_signal_counts`` of two animals over the MUA bins: 20 bins in each,
    non-local in 2 below 2 z and 10 at or above."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        animal_signal_counts,
    )
    from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
        MUA_EDGES,
    )

    low, high = MUA_EDGES[:-1], MUA_EDGES[1:]
    return animal_signal_counts(
        pd.concat(
            pd.DataFrame(
                {
                    "animal": animal, "epoch_id": f"{animal}_e", "signal_low": low,
                    "signal_high": high, "n_bins": 20,
                    "n_nonlocal": np.where(low >= 2.0, 10, 2),
                }
            )
            for animal in ("a", "b")
        )
    )  # fmt: skip


def _sweep() -> pd.DataFrame:
    """``sweep_statistics`` of the rule sweep test's counts."""
    from continuum_swr_replay.analysis.event_properties.rule_sweep import (
        sweep_statistics,
    )
    from tests.analysis.event_properties.test_rule_sweep import _counts

    return sweep_statistics(_counts())


def _calibration() -> pd.DataFrame:
    """The in-visit calibration's bins (the builder test's)."""
    from tests.reported.test_reported_events_values import _state_calibration

    return _state_calibration()[0]


TRACE_TIME = np.linspace(-0.15, 0.15, 151)


def _event_traces() -> dict:
    """Two animals' median traces per class and signal: SWR events at the
    animal's level plus 1 after the start, no-SWR events plus 0.5."""
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        TRIGGERED_SIGNALS,
    )

    def medians(height: float) -> pd.DataFrame:
        return pd.DataFrame(
            {a: level + height * (TRACE_TIME > 0) for a, level in (("a", 0), ("b", 1))},
            index=TRACE_TIME,
        )

    return {
        "time": TRACE_TIME,
        "overlap": {signal: medians(1.0) for signal in TRIGGERED_SIGNALS},
        "no_overlap": {signal: medians(0.5) for signal in TRIGGERED_SIGNALS},
    }


def _four_decoders():
    """``swr_allowed_statistics`` of the builder test's inputs."""
    from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
        swr_allowed_statistics,
    )
    from tests.reported.test_reported_events_values import _swr_allowed_inputs

    sums, bin_s, events = _swr_allowed_inputs()
    return swr_allowed_statistics(events, sums, bin_s)


def _statistics() -> dict:
    return {
        "calibration": _calibration(),
        "sweep": _sweep(),
        "mua": _mua_counts(),
        "mua_states": _mua_states(),
        "hpd": _hpd(),
        "timing": _timing(),
        "probability": _probability(),
        "distance": _decoded_distance(),
        "split": {
            column: _gain_statistics(*groups) for column, groups in GAIN_SPLIT.items()
        },
        "outside": _outside_rows(),
        "parts": _parts(),
        "no_spike": _no_spike_trace(),
        "event_traces": _event_traces(),
        "four_decoders": _four_decoders(),
    }


@pytest.fixture(scope="module")
def drawn():
    """The figure drawn once from :func:`_statistics`, for the tests that only
    read it."""
    fig = figure.generate_figure(_statistics())
    yield fig
    plt.close(fig)


def test_the_figure_draws_its_panels_in_the_cited_order(drawn):
    """In five rows, each a group the text cites in turn: the decoder (a to d),
    the count is robust (e, f), the held-out tests (g to i), the animals agree
    and the posterior's spread (j, k), the fourth decoder (l); each row's
    letters at one height; the letters in four columns, each at one left edge
    (the rows' first letters; b, h; c, f, i; d, k), where the column's panels'
    drawings start; no heading; within the common 20 cm height."""
    assert figure.PANELS == (
        "state_calibration", "no_spike_mua", "nonlocal_probability",
        "decoded_distance", "rule_sweep", "boundary_timing", "nonlocal_mua",
        "left_out_split", "outside_distance",
        "event_traces", "hpd", "four_decoders",
    )  # fmt: skip
    fig = drawn
    assert figure.panel_axes(fig, "no_spike_mua")[0].get_title() == "No-Spike state"
    assert (
        figure.panel_axes(fig, "boundary_timing")[0].get_title(loc="left")
        == "Event against its SWR"
    )
    letters = {
        t.get_text(): t.get_position() for t in fig.texts if len(t.get_text()) == 1
    }
    assert sorted(letters) == list("abcdefghijkl")
    rows = ("abcd", "ef", "ghi", "jk", "l")
    for row in rows:
        assert len({letters[k][1] for k in row}) == 1, row
    heights = [letters[row[0]][1] for row in rows]
    assert heights == sorted(heights, reverse=True)
    columns = ("aegjl", "bh", "cfi", "dk")
    for column in columns:
        assert len({letters[k][0] for k in column}) == 1, column
    edges = [letters[column[0]][0] for column in columns]
    assert edges == sorted(edges)
    by_letter = {figure.LETTERS[panel]: panel for panel in figure.PANELS}
    for column in columns:  # each panel's drawing starts at its column's edge
        starts = [
            figure.drawn_extent(fig, figure.panel_axes(fig, by_letter[k])).x0
            for k in column
        ]
        np.testing.assert_allclose(starts, starts[0], atol=1e-3)
    assert "Immobile, at wells" not in [
        t.get_text() for ax in fig.axes for t in ax.texts
    ]
    figure.check_journal_limits(fig)


def test_the_split_panel_draws_the_fit_over_the_animals_place_by_bins(drawn):
    """Panel h: Figure 2j's first comparison for no-SWR then SWR events, three
    positions each (all bins, the bins where the scored tetrode spiked, the
    silent bins): a dot per animal in the group's color, the mean with its
    interval beside them; the parts named, the groups named over their
    positions, a dashed zero line named "equal fit", the title and y label."""
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    (ax,) = figure.panel_axes(drawn, "left_out_split")
    dots = [
        c for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]  # fmt: skip
    assert len(dots) == 6
    for k, (_group, color) in enumerate(panels.GROUP_COLORS.items()):
        for j, groups in enumerate(GAIN_SPLIT.values()):
            x = panels.split_x(k, j)
            (mine,) = [d for d in dots if np.allclose(d.get_offsets()[:, 0], x)]
            np.testing.assert_allclose(mine.get_offsets()[:, 1], groups[k])
            np.testing.assert_allclose(mine.get_facecolor()[0], to_rgba(color))
            assert np.allclose(_means_at(ax, x), np.mean(groups[k]))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == 2 * list(panels.SPLIT_LABELS.values())
    assert {"No SWR", "SWR", "equal fit"} <= {t.get_text() for t in ax.texts}
    assert ax.get_title() == "Decoded place over\nanimal's place"
    assert ax.get_ylabel() == "Left-out tetrode:\nbetter fit (nats/spike)"


def test_each_panels_values_carry_its_letter():
    """The events summary tags each panel's values with the panel's letter, as
    the figure places it."""
    from continuum_swr_replay.reported import events

    panels = {
        "state_calibration": "state_calibration",
        "rule_sweep": "rule_sweep",
        "boundary_timing": "boundary_timing",
        "nonlocal_probability": "nonlocal_probability",
        "calibration": "decoded_distance",
        "nonlocal_mua": "nonlocal_mua",
        "tetrode_holdout_split": "left_out_split",
        "nonlocal_restriction_distance": "outside_distance",
        "hpd": "hpd",
        "nonlocal_restriction_swr_allowed": "four_decoders",
    }
    for tag, panel in panels.items():
        assert events.TAGS[tag].figures == (f"S2{figure.LETTERS[panel]}",), tag
    # the gain over local coding is Figure 2j's too
    assert events.TAGS["tetrode_holdout"].figures == (
        "2j",
        f"S2{figure.LETTERS['left_out_split']}",
    )
    assert set(panels.values()) == set(figure.PANELS) - {"no_spike_mua", "event_traces"}


def test_the_hpd_panel_draws_the_distribution_of_each_class(drawn):
    """For each class, a thin line per animal and their mean in bold, in the
    class's color, on a log x axis with the named ticks; a dotted line at the
    tight cut; the classes named in their colors; a y axis from 0."""
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    distribution = _hpd()
    (ax,) = figure.panel_axes(drawn, "hpd")
    n_bins = len(figure.HPD_BIN_EDGES_CM) - 1
    curves = [line for line in ax.lines if len(line.get_xdata()) == n_bins]
    assert len(curves) == 2 * (2 + 1)  # two animals and their mean, per class
    for group, color in panels.GROUP_COLORS.items():
        mine = [c for c in curves if to_rgba(c.get_color()) == to_rgba(color)]
        mean = max(mine, key=lambda c: c.get_linewidth())
        expected = (
            distribution[distribution["group"] == group]
            .pivot(index="bin_low", columns="animal", values="percent")
            .mean(axis=1)
        )
        np.testing.assert_allclose(mean.get_ydata(), expected.to_numpy())
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "10", "25", "50", "100", "400",
    ]  # fmt: skip
    (cut,) = [line for line in ax.lines if line.get_linestyle() == ":"]
    np.testing.assert_allclose(cut.get_xdata(), figure.HPD_TIGHT_CM)
    assert {"No SWR", "SWR", "50 cm"} <= {t.get_text() for t in ax.texts}
    assert ax.get_ylim()[0] == 0
    assert ax.get_title() == "Posterior spread"


def test_the_probability_panel_draws_each_animals_time_by_nonlocal_probability(
    drawn,
):
    """A thin line per animal and their mean in bold, the percent of the time
    per probability bin at the bins' centers on a log y axis, the event rule's
    cut dotted and named "used", the title in plain words."""
    distribution = _probability()
    (ax,) = figure.panel_axes(drawn, "nonlocal_probability")
    by_animal = distribution.pivot(index="bin_low", columns="animal", values="percent")
    curves = [line for line in ax.lines if len(line.get_xdata()) == 50]
    assert len(curves) == 3
    mean = max(curves, key=lambda c: c.get_linewidth())
    np.testing.assert_allclose(mean.get_xdata(), by_animal.index + 0.01)
    np.testing.assert_allclose(mean.get_ydata(), by_animal.mean(axis=1))
    assert ax.get_yscale() == "log"
    (cut,) = [line for line in ax.lines if line.get_linestyle() == ":"]
    np.testing.assert_allclose(cut.get_xdata(), 0.8)
    assert "used " in [t.get_text() for t in ax.texts]
    assert ax.get_xlabel() == "Non-local probability"
    assert ax.get_title() == "Non-local probability,\nimmobile time"


def test_the_decoded_place_panel_draws_each_animals_distance_and_their_mean(drawn):
    """A thin line per animal (the median over its sessions) and the mean over
    animals in bold against the Local probability; no word of calibration."""
    by_animal, statistics = _decoded_distance()
    (ax,) = figure.panel_axes(drawn, "decoded_distance")
    curves = [line for line in ax.lines if len(line.get_xdata()) == 10]
    assert len(curves) == 3
    mean = max(curves, key=lambda c: c.get_linewidth())
    np.testing.assert_allclose(mean.get_ydata(), statistics["animal_mean"])
    # animal a: the mean of its two sessions, 1.1 times the shape
    np.testing.assert_allclose(by_animal["a"], 1.1 * (30 - 27 * by_animal.index))
    assert ax.get_xlabel() == "Local probability"
    assert ax.get_ylabel() == "Distance from\nthe animal (cm)"
    assert ax.get_title() == "Decoded place against\nthe animal's place"
    assert "calibrat" not in (ax.get_title() + ax.get_ylabel()).lower()


def test_the_timing_panel_draws_a_row_per_case_with_its_drawing(drawn):
    """Panel f: nine rows, each named beside a drawing of the case (the five
    with a bound the same, then the four with both bounds apart); a bar of the
    mean over animals with a dot per animal and the percent at its end; the
    two bars of a drawing named over the drawings; the note of what "same"
    means."""
    ax_cases, ax = figure.panel_axes(drawn, "boundary_timing")
    names = [t.get_text() for t in ax_cases.texts]
    # each row's name, then the two bars' names over the drawings
    assert names == [*(name for name, _, _ in figure.TIMING_CASES), "SWR", "event"]
    assert names[0] == "Same start and end"
    # one bound apart, each case beside its opposite; then both apart
    assert names[1:5] == [
        "Starts later",
        "Starts earlier",
        "Ends earlier",
        "Ends later",
    ]
    assert names[5:9] == [
        "Shorter at both ends",
        "Longer at both ends",
        "Starts and ends later",
        "Starts and ends earlier",
    ]
    widths = [bar.get_width() for bar in ax.patches]
    # together-together 70, starts later 10, ends later 15, the other six 5
    np.testing.assert_allclose(widths, [70, 10, 5, 5, 15, 5, 5, 5, 5])
    texts = [t.get_text() for t in ax.texts]
    assert texts[:9] == ["70%", "10%", "5%", "5%", "15%", "5%", "5%", "5%", "5%"]
    assert ax.get_xlim() == (0.0, 80.0)  # the largest animal's 71 and room
    assert figure.TIMING_NOTE in texts
    dots = [
        c for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]  # fmt: skip
    assert len(dots) == 9
    np.testing.assert_allclose(sorted(dots[0].get_offsets()[:, 0]), [69, 70, 71])
    assert ax.get_xlabel() == "% of overlapping pairs"
    assert ax_cases.get_title(loc="left") == "Event against its SWR"
    assert ax.get_ylim() == ax_cases.get_ylim()


def test_the_distance_panel_draws_the_outside_swr_gain_then_away_from_swrs(drawn):
    """Outside SWRs, then only farther than each margin from every SWR: dots
    spread about each column, the mean with its interval, a thin line through
    the means, the margins' label, the difference's y label, the zero line and
    "full better". The gain at well-sampled places is not drawn."""
    fig = drawn
    (ax,) = figure.panel_axes(fig, "outside_distance")
    xs = [0.0, *(figure.SERIES_START + k for k in range(len(MARGINS)))]
    for x, values in zip(xs, [OUTSIDE, *MARGINS.values()], strict=True):
        (dots,) = [
            c
            for c in ax.collections
            if isinstance(c, matplotlib.collections.PathCollection)
            and np.allclose(c.get_offsets()[:, 0], x, atol=0.2)
        ]
        np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
        assert_spread_about(dots, x)
        assert np.allclose(_means_at(ax, x), np.mean(values))
    (series,) = [line for line in ax.lines if len(line.get_xdata()) == len(xs)]
    np.testing.assert_allclose(
        series.get_ydata(), [np.mean(v) for v in [OUTSIDE, *MARGINS.values()]]
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Outside\nSWRs", "100", "250", "500",
    ]  # fmt: skip
    texts = [t.get_text() for t in ax.texts]
    assert "at least this far from\nany SWR (ms)" in texts
    assert "↑ full better" in texts
    assert ax.get_ylabel() == "Held-out log score,\nfull − SWR-only\n(nats/spike)"
    assert ax.get_ylim()[0] < 0


def test_the_mua_panel_draws_where_the_nonlocal_time_sits_in_mua(drawn):
    """As Figure 2i draws ripple power: top, Local and non-local time spread
    over MUA (the mean over animals of each), named, with the share of
    non-local time below the MUA level written; bottom, the share of bins
    decoded non-local at each level; a dashed line at the level on both."""
    fig = drawn
    ax_spread, ax_share = figure.panel_axes(fig, "nonlocal_mua")
    assert len(ax_spread.lines) == 2 + 1  # the two classes, the level
    assert len(ax_share.lines) == 2 + 2  # two animals, their mean, the level
    for ax in (ax_spread, ax_share):
        (level,) = [line for line in ax.lines if line.get_linestyle() == "--"]
        np.testing.assert_allclose(level.get_xdata(), figure.MUA_LEVEL_Z)
    local, nonlocal_ = ax_spread.lines[:2]
    assert local.get_xdata()[np.argmax(local.get_ydata())] == pytest.approx(-0.875)
    assert nonlocal_.get_xdata()[np.argmax(nonlocal_.get_ydata())] == pytest.approx(
        3.125
    )
    # non-local bins below 2 z: 2 in each of the 17 bins below (the open one
    # too) against 10 in each of the 33 at or above
    below = 17 * 2 / (17 * 2 + 33 * 10)
    texts = [t.get_text() for t in ax_spread.texts]
    assert f"{round(100 * below)}% below\n2 z" in texts
    assert ax_share.get_xlabel() == "MUA (z)"
    from matplotlib.offsetbox import AnchoredOffsetbox

    from continuum_swr_replay.visualization import signal_share_panels

    (title,) = [a for a in ax_spread.artists if isinstance(a, AnchoredOffsetbox)]
    words = title.get_child().get_children()
    assert [w.get_text() for w in words] == ["Local", "and", "non-local", "time"]
    for word, (_, state) in zip(words, signal_share_panels.STRIP_TITLE, strict=True):
        if state is not None:  # a class's name takes its curve's color
            assert word._text.get_color() == signal_share_panels.readable_text_color(
                signal_share_panels.STATE_STYLES[state][1]
            )


def test_the_sweep_panel_draws_the_share_with_no_swr_at_each_setting(drawn):
    """Three small axes sharing the y axis, one per swept setting: a thin line
    per animal and their mean over animals in bold, in percent, the rule's value
    marked by a dotted line; under each, the events at each value in
    thousands; the minimum run length drawn in ms (2 ms a sample)."""
    from continuum_swr_replay.analysis.event_properties.rule_sweep import (
        RULE,
        SETTINGS,
    )

    sweep = _sweep()
    fig = drawn
    axes = figure.panel_axes(fig, "rule_sweep")
    assert len(axes) == 6
    shares, counts = axes[:3], axes[3:]
    for ax, ax_count, (setting, values) in zip(
        shares, counts, SETTINGS.items(), strict=True
    ):
        mine = sweep[sweep["setting"] == setting]
        scale = figure.SWEEP_SCALES[setting]
        lines = [line for line in ax.lines if len(line.get_xdata()) == len(values)]
        assert len(lines) == 2 + 1  # two animals and their mean
        np.testing.assert_allclose(lines[-1].get_xdata(), scale * np.asarray(values))
        np.testing.assert_allclose(
            lines[-1].get_ydata(),
            100 * mine["no_swr_fraction_animal_mean"].to_numpy(),
        )
        (count,) = [
            line for line in ax_count.lines if len(line.get_xdata()) == len(values)
        ]
        np.testing.assert_allclose(
            count.get_ydata(), mine["n_events"].to_numpy() / 1000
        )
        (rule,) = [line for line in ax.lines if line.get_linestyle() == ":"]
        np.testing.assert_allclose(rule.get_xdata(), scale * RULE[setting])
        assert ax.get_shared_y_axes().joined(ax, shares[0])
        assert ax_count.get_shared_x_axes().joined(ax_count, ax)
    assert figure.SWEEP_SCALES["minimum_samples"] == pytest.approx(2.0, rel=1e-3)
    assert shares[0].get_ylabel() == "No SWR (%)"
    assert counts[0].get_ylabel() == "Events\n(1000s)"
    # one heading over the event rule's two axes, one over the detector's; the
    # value used marked on all three
    assert "Event rule" in [t.get_text() for t in shares[0].texts]
    assert "SWR detector" in [t.get_text() for t in shares[2].texts]
    for ax in shares:
        assert " used" in [t.get_text() for t in ax.texts]
    assert [ax.get_xlabel() for ax in counts] == list(figure.SWEEP_XLABELS.values())


def test_the_calibration_panel_draws_each_states_evidence_by_its_probability(drawn):
    """Local, Continuous and Fragmented on one axis, each the animals' mean
    with its interval near the probability bins' centers in the state's color,
    named at its line's end; a dashed zero line; a y label that says what the
    evidence is against; No-Spike is not drawn."""
    from matplotlib.colors import to_rgba

    fig = drawn
    (ax,) = figure.panel_axes(fig, "state_calibration")
    centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    curves = [line for line in ax.lines if len(line.get_xdata()) == 5]
    assert len(curves) == 3
    for k, (curve, state) in enumerate(
        zip(curves, figure.CALIBRATION_COLORS, strict=True)
    ):
        np.testing.assert_allclose(
            curve.get_xdata(), centers + (k - 1) * figure.CALIBRATION_DODGE
        )
        np.testing.assert_allclose(curve.get_ydata(), k + np.arange(5))
        np.testing.assert_allclose(
            to_rgba(curve.get_color()), to_rgba(figure.CALIBRATION_COLORS[state])
        )
    assert [t.get_text() for t in ax.texts] == ["Local", "Continuous", "Fragmented"]
    assert any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )
    assert ax.get_ylabel() == figure.CALIBRATION_YLABEL
    assert "mismatched place" in ax.get_ylabel() and "nats/spike" in ax.get_ylabel()
    assert ax.get_xlabel() == "State probability"


def test_the_no_spike_panel_draws_the_mua_around_no_spike_starts(drawn):
    """The median with its interquartile band, a dotted line at the start, a
    dashed line at the level outside the periods, and its labels."""
    trace = _no_spike_trace()
    fig = drawn
    (ax,) = figure.panel_axes(fig, "no_spike_mua")
    (median,) = [line for line in ax.lines if len(line.get_xdata()) == 151]
    np.testing.assert_allclose(median.get_xdata(), 1000 * NO_SPIKE_TIME)
    np.testing.assert_allclose(median.get_ydata(), trace["median"])
    assert any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0.2)
        for line in ax.lines
    )
    assert any(
        line.get_linestyle() == ":" and np.allclose(line.get_xdata(), 0)
        for line in ax.lines
    )
    assert len(ax.collections) == 1  # the band
    assert ax.get_xlabel() == "Time from No-Spike\nstart (ms)"
    assert ax.get_ylabel() == "MUA (z)"
    assert ax.get_title() == "No-Spike state"


def test_the_no_spike_trace_is_over_every_epochs_periods(monkeypatch):
    """Each analyzed epoch's frame read once: its periods (the rule, with its
    well visits) give the windows, and its level outside them; the periods
    are counted by animal."""
    import xarray as xr

    from tests.analysis.event_definition.test_no_spike_periods import _frame

    frame, arrival, departure = _frame()
    frame = frame.assign(animal="a", day=1, epoch=2)
    suffix = figure.data_over_time_suffix()
    paths = [Path(f"a1_02_r1_HPC{suffix}"), Path(f"a2_02_r1_HPC{suffix}")]
    read, visits = [], []
    monkeypatch.setattr(figure, "processed_data_dir", lambda: Path("."))
    monkeypatch.setattr(
        figure, "analyzed_decode_files", lambda folder, suffix, require_all: paths
    )
    monkeypatch.setattr(
        figure, "read_dot_with_swr_source", lambda path: read.append(path) or frame
    )
    monkeypatch.setattr(
        figure,
        "well_visit_times",
        lambda epoch_id: visits.append(epoch_id) or (arrival, departure),
    )

    def windows(data, event_times, signals, window_sec):
        assert window_sec == figure.TRIGGER_WINDOW_S
        values = np.tile(np.array([2.0, 0.0]), (len(event_times), 1))[..., None]
        return xr.DataArray(
            values,
            dims=("event", "time", "signal"),
            coords={"time": [-0.002, 0.0], "signal": signals},
        )

    monkeypatch.setattr(figure, "get_signal_windows", windows)
    trace = figure.read_frame_traces(workers=1)["no_spike"]
    assert read == paths
    assert visits == ["a1_02_r1_HPC", "a2_02_r1_HPC"]
    np.testing.assert_allclose(trace["median"], [2.0, 0.0])
    assert trace["periods"].to_dict() == {"a": 2}  # one period per epoch
    assert trace["level"] == pytest.approx(1.0)
    assert trace["median_duration"] == pytest.approx(9 * 0.002)


def test_the_traces_panel_draws_each_animals_median_per_class(drawn):
    """One axis per Figure 2g signal, a thin line per animal and class in the
    class's color, the start dotted, the classes named."""
    from matplotlib.colors import to_rgba

    axes = figure.panel_axes(drawn, "event_traces")
    traces = _event_traces()
    assert [ax.get_title() for ax in axes] == list(figure.TRACE_LABELS.values())
    for ax, signal in zip(axes, figure.TRIGGERED_SIGNALS, strict=True):
        for kind, (_, color) in figure.TRACE_CLASSES.items():
            mine = [
                line for line in ax.lines
                if len(line.get_xdata()) == TRACE_TIME.size
                and to_rgba(line.get_color()) == to_rgba(color)
            ]  # fmt: skip
            assert len(mine) == 2
            np.testing.assert_allclose(
                np.column_stack([line.get_ydata() for line in mine]),
                traces[kind][signal].to_numpy(),
            )
            np.testing.assert_allclose(mine[0].get_xdata(), 1000 * TRACE_TIME)
    assert {"SWR", "No SWR"} <= {t.get_text() for t in axes[-1].texts}


def test_the_figure_is_the_question_lists_s2_figure():
    """Values tagged ``S2`` (by panel) name this script."""
    assert FIGURES["S2"].panels == "abcdefghijkl"
    assert FIGURES["S2"].script == ("scripts/figures/generate_supp_figure02.py")


class _HeldoutRestriction:
    """What ``read_heldout_restriction`` gives, with every note None but those
    given."""

    def __init__(self, **notes):
        self.events = pd.DataFrame({"heldout_event_id": [0, 1]})
        self.time_sums = pd.DataFrame({"n": range(3)})
        self.bin_s = pd.Series({"e": 0.002})
        for name in (
            "restriction_run",
            "bin_facts_run",
            "session_cache",
            "event_table",
        ):
            setattr(self, name, notes.get(name))


def _patch_reading(monkeypatch, stored, seen, heldout=None, note=None):
    for name, value in {
        "read_output_manifest": lambda folder: {"settings": TRIAL_FOLD_SETTINGS},
        "read_swr_table": lambda path: stored["table"],
        "event_table_epochs": lambda table, analyzed: ["e"],
        "analyzed_epoch_ids": lambda: ["e"],
        "read_complete_run": lambda folder, name, epochs: stored["scores"],
        "window_scores": lambda table: seen.setdefault("tetrode", []).append(table)
        or table,
        "holdout_statistics": lambda windows, kind, column: seen.setdefault(
            "split", windows
        ),
        "read_nonlocal_probability": lambda folder: "probability",
        "read_decoded_distance": lambda folder: "distance",
        "sha256_file": lambda path: "0" * 64,
        "read_heldout_restriction": lambda *args: heldout or _HeldoutRestriction(),
        "restriction_inputs_note": lambda folder, epochs: note,
        "outside_swr_statistics": lambda sums, bin_s, events: seen.setdefault(
            "restriction", (sums, bin_s, events)
        ),
        "read_frame_traces": lambda events, workers: {
            "no_spike": "trace",
            "event_traces": "traces",
        },
        "hpd_distribution": lambda events: seen.setdefault("hpd", events),
        "read_mua_counts": lambda folder: "mua",
        "read_mua_states": lambda folder: "mua_states",
        "read_rule_sweep": lambda folder, table: "sweep",
        "read_state_calibration": lambda folder: "calibration",
        "swr_allowed_statistics": lambda events, sums, bin_s: seen.setdefault(
            "four_decoders", (events, sums, bin_s)
        ),
        "outside_swr_subset_statistics": lambda sums, bin_s, subsets: pd.DataFrame(
            {
                "subset": ["beyond_swr_margin"],
                "threshold_s": [0.1],
                "family": ["filter"],
                "measure": ["gain_per_spike"],
            }
        ),
    }.items():
        monkeypatch.setattr(figure, name, value)


def test_the_figure_reads_the_held_out_runs_over_the_events_that_start_in_a_visit(
    monkeypatch,
):
    """The stored held-out tetrode run (held out in
    folds of whole trials) scores every event; the figure, like the events
    summary, reads the rows (every tetrode's) of the events that start inside a
    well visit (panel h, the events alone), and the held-out event run's
    events and time sums by the summary's reader."""
    inside = _figure02_events().assign(event_id=range(8))
    outside = inside.iloc[[3]].assign(event_id=99, visit_context="track_leaf_arm")
    stored = {
        "table": pd.concat([inside, outside]),
        "scores": pd.DataFrame(
            {"event_id": [0, 0, *range(1, 8), 99, 99], "kind": "event"}
        ),
    }
    seen = {}
    _patch_reading(monkeypatch, stored, seen)
    inputs = figure.read_inputs()
    # the held-out tetrode run (panel h)
    assert [list(t["event_id"]) for t in seen["tetrode"]] == [[0, 0, *range(1, 8)]]
    assert list(seen["split"]["event_id"]) == [0, 0, *range(1, 8)]
    sums, bin_s, events = seen["restriction"]
    assert len(sums) == 3 and bin_s.to_dict() == {"e": 0.002} and len(events) == 2
    assert seen["hpd"]["event_id"].tolist() == list(range(8))
    assert list(inputs) == [
        "calibration",
        "probability",
        "distance",
        "sweep",
        "mua",
        "mua_states",
        "hpd",
        "timing",
        "split",
        "outside",
        "parts",
        "four_decoders",
        "no_spike",
        "event_traces",
    ]
    events, sums, bin_s = seen["four_decoders"]
    assert len(sums) == 3 and bin_s.to_dict() == {"e": 0.002} and len(events) == 2


@pytest.mark.parametrize(
    ("notes", "inputs_note", "rerun"),
    [
        ({}, "the restriction run records another SWR source", "score_nonlocal"),
        ({"restriction_run": "made from another restriction run"}, None, "detect"),
        ({"bin_facts_run": "made from another bin_facts run"}, None, "detect"),
        ({"session_cache": "other well visits than the current"}, None, "measure_bin"),
    ],
    ids=["decodes or SWR source", "restriction run", "bin facts", "session cache"],
)
def test_the_figure_refuses_a_held_out_event_run_that_is_not_current(
    monkeypatch, notes, inputs_note, rerun
):
    """Panel i is refused as Figure 2k is: from a restriction run off
    the current decodes or SWR source, or a held-out event run not made from
    the current restriction run, per-bin facts run or session cache."""
    stored = {
        "table": _figure02_events().assign(event_id=range(8)),
        "scores": pd.DataFrame({"event_id": [0]}),
    }

    def refuse(*args):
        raise AssertionError("drawn from inputs that are not current")

    _patch_reading(
        monkeypatch, stored, {}, heldout=_HeldoutRestriction(**notes), note=inputs_note
    )
    monkeypatch.setattr(figure, "outside_swr_statistics", refuse)
    with pytest.raises(ValueError, match="panel i is not drawn") as refused:
        figure.read_inputs()
    assert rerun in str(refused.value)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_folds": 5, "margin_bins": 2}, "fold_design None, not 'trials'"),
        (
            {"n_folds": 5, "fold_design": "contiguous"},
            "fold_design 'contiguous', not 'trials'",
        ),
    ],
    ids=["no fold design", "another design"],
)
def test_the_figure_refuses_a_held_out_run_not_on_trial_folds(
    monkeypatch, settings, message
):
    """A held-out tetrode run of an older fold design is refused before any
    table is read, as the events summary refuses it."""
    monkeypatch.setattr(
        figure, "read_output_manifest", lambda folder: {"settings": settings}
    )

    def read_swr_table(path):
        raise AssertionError("read before the refusal")

    monkeypatch.setattr(figure, "read_swr_table", read_swr_table)
    with pytest.raises(ValueError, match=message):
        figure.read_inputs()


def test_the_figure_refuses_a_fourth_decoder_run_that_is_not_current(monkeypatch):
    """Panel l is refused on the same grounds as panel i, from its own
    held-out event run (read by the same reader from its own folders)."""
    stored = {
        "table": _figure02_events().assign(event_id=range(8)),
        "scores": pd.DataFrame({"event_id": [0]}),
    }
    _patch_reading(monkeypatch, stored, {})
    current, stale = (
        _HeldoutRestriction(),
        _HeldoutRestriction(restriction_run="made from another restriction run"),
    )
    four_dir = figure.default_inputs().nonlocal_restriction_swr_allowed_heldout_events
    monkeypatch.setattr(
        figure,
        "read_heldout_restriction",
        lambda heldout_dir, *args: stale if heldout_dir == four_dir else current,
    )
    with pytest.raises(ValueError, match="panel l is not drawn") as refused:
        figure.read_inputs()
    assert "detect_heldout_events" in str(refused.value)


def test_the_four_decoder_panel_draws_the_decoders_with_the_fourth(drawn):
    """Panel l: the SWR-only, SWR-allowed (open dots) and full decoders over
    the local-only decoder; at each time, the full minus the SWR-allowed
    decoder (filled) and the SWR-allowed minus the SWR-only decoder (open),
    named in a legend; and the full minus the SWR-allowed decoder in the
    held-out no-SWR events, each from ``swr_allowed_statistics``' rows."""
    from matplotlib.colors import to_rgba

    statistics = _four_decoders()
    ax_decoders, ax_difference, ax_events = figure.panel_axes(drawn, "four_decoders")

    def dots_at(ax, x):
        (dots,) = [
            c for c in ax.collections
            if isinstance(c, matplotlib.collections.PathCollection)
            and np.isclose(np.mean(c.get_offsets()[:, 0]), x, atol=0.1)
        ]  # fmt: skip
        return dots

    def animals(time, measure):
        return statistics.times.loc[(time, "filter", measure), "animals"]

    assert [t.get_text() for t in ax_decoders.get_xticklabels()] == [
        "SWR-\nonly", "SWR-\nallowed", "Full",
    ]  # fmt: skip
    for x, measure in enumerate(figure.SWR_ALLOWED_FIGURE_DECODERS):
        dots = dots_at(ax_decoders, float(x))
        np.testing.assert_allclose(
            sorted(dots.get_offsets()[:, 1]), sorted(animals("immobile", measure))
        )
    # the SWR-allowed decoder in open dots of the SWR-only decoder's gray
    assert len(dots_at(ax_decoders, 1.0).get_facecolors()) == 0
    np.testing.assert_allclose(
        dots_at(ax_decoders, 1.0).get_edgecolors()[0],
        to_rgba(figure.NEUTRAL_DOTS),
    )
    np.testing.assert_allclose(
        dots_at(ax_decoders, 0.0).get_facecolors()[0], to_rgba(figure.NEUTRAL_DOTS)
    )
    for k, time in enumerate(figure.FOUR_DECODER_TIMES):
        for j, measure in enumerate(figure.SWR_ALLOWED_FIGURE_DIFFERENCES):
            x = k * figure.FOUR_DECODER_GROUP + j * figure.FOUR_DECODER_STEP
            dots = dots_at(ax_difference, x)
            np.testing.assert_allclose(
                sorted(dots.get_offsets()[:, 1]), sorted(animals(time, measure))
            )
            assert (len(dots.get_facecolors()) == 0) == (j == 1)
            assert np.allclose(
                _means_at(ax_difference, x),
                statistics.times.loc[(time, "filter", measure), "estimate"],
            )
    assert [t.get_text() for t in ax_difference.get_xticklabels()] == [
        "All\nimmobility", "Inside\nSWRs", "Outside\nSWRs",
    ]  # fmt: skip
    assert [t.get_text() for t in ax_difference.get_legend().get_texts()] == [
        "Full \u2212 SWR-allowed", "SWR-allowed \u2212 SWR-only",
    ]  # fmt: skip
    row = statistics.events.loc["no SWR"]
    dots = dots_at(ax_events, 0.0)
    np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(row["animals"]))
    assert np.allclose(_means_at(ax_events, 0.0), row["estimate"])
    assert ax_events.get_title() == "Outside SWRs,\nheld-out events"
    assert ax_events.get_ylabel() == (
        "Held-out log score,\nfull \u2212 SWR-allowed\n(nats/spike)"
    )
