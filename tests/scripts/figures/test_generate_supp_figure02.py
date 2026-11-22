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
SAMPLED = {10.0: [0.029, 0.021, 0.024, 0.037, 0.023]}
SAMPLED[60.0] = list(np.asarray(SAMPLED[10.0]) + 0.001)


def _parts() -> pd.DataFrame:
    """``outside_swr_subset_statistics``' rows indexed by subset, threshold,
    family and measure: each part's gain per spike."""
    tables = []
    for subset, parts in (("beyond_swr_margin", MARGINS), ("place_sampling", SAMPLED)):
        for threshold, values in parts.items():
            rows = _measure_rows({"gain_per_spike": values}).reset_index()
            tables.append(rows.assign(subset=subset, threshold_s=threshold))
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


SHARED = {
    ("no SWR", 0): [0.3, 0.25, 0.35, 0.2, 0.4],
    ("SWR", 0): [0.2, 0.25, 0.3, 0.15, 0.22],
}
"""Panel c's animals in the narrowest spread bin; each wider bin is 0.1 lower."""


SHARE_OF_EVENTS = {"no SWR": [0.12, 0.2, 0.43, 0.25], "SWR": [0.1, 0.3, 0.4, 0.2]}
"""Each group's share of its events in each spread bin."""


def _shared_statistics() -> pd.DataFrame:
    """``agreement_by_spread_statistics``' rows: each bin and group's animals,
    their mean and an interval of 0.05 either side."""
    rows = []
    for k, label in enumerate(figure.SPREAD_BIN_LABELS):
        for group in ("no SWR", "SWR"):
            animals = np.asarray(SHARED[(group, 0)]) - 0.1 * k
            mean = float(animals.mean())
            rows.append(
                {
                    "spread_bin": label, "group": group, "animals": animals,
                    "animal_names": np.array(list("abcde")), "estimate": mean,
                    "ci_low": mean - 0.05, "ci_high": mean + 0.05,
                    "share_of_events": SHARE_OF_EVENTS[group][k],
                }
            )  # fmt: skip
    return pd.DataFrame(rows).set_index(["spread_bin", "group"])


def _timing() -> pd.DataFrame:
    """``boundary_timing_statistics``' nine cells: shares 0.05 to 0.7 (the
    middle cell the largest), each animal range 0.01 either side."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        TIMING_CLASSES,
    )

    shares = np.array([0.05, 0.05, 0.05, 0.05, 0.7, 0.15, 0.05, 0.1, 0.05])
    cells = pd.MultiIndex.from_product(
        [TIMING_CLASSES, TIMING_CLASSES], names=["start", "end"]
    )
    return pd.DataFrame(
        {
            "share": shares, "animal_min": shares - 0.01,
            "animal_max": shares + 0.01, "n_events": (1000 * shares).astype(int),
        },
        index=cells,
    )  # fmt: skip


def _hpd() -> pd.DataFrame:
    """``hpd_distribution`` of the HPD test's events."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        hpd_distribution,
    )
    from tests.analysis.event_properties.test_event_statistics import _hpd_events

    return hpd_distribution(_hpd_events())


SPLIT = {
    "gain_per_spike": ([0.52, 0.47, 0.25, 0.43, 0.46], [0.86, 0.8, 0.7, 0.75, 0.9]),
    "spiking_gain_per_spike": ([0.3, 0.2, 0.1, 0.2, 0.25], [0.6, 0.5, 0.4, 0.5, 0.6]),
    "silent_gain_per_spike": ([0.2, 0.25, 0.15, 0.2, 0.2], [0.25, 0.3, 0.3, 0.2, 0.3]),
}
"""Each part's animals, no SWR then SWR."""


def _split_statistics() -> dict:
    """``holdout_statistics`` of each part of the gain, by column."""
    return {column: _gain_statistics(*groups) for column, groups in SPLIT.items()}


RIPPLE_GRID = np.linspace(-2.0, 12.0, 141)


def _ripple_modes() -> dict:
    """Two animals' peak-power densities (normal, modes 1 and 2 z) and the one
    mode statistic: one mode in 5 of 5 animals."""
    densities = pd.DataFrame(
        {
            animal: np.exp(-0.5 * (RIPPLE_GRID - mode) ** 2) / np.sqrt(2 * np.pi)
            for animal, mode in (("a", 1.0), ("b", 2.0))
        },
        index=RIPPLE_GRID,
    )
    return {
        "densities": densities,
        "modes": {"n_animals_unimodal": 5, "mode_animal_min": 1.0,
                  "mode_animal_max": 2.0},
        "n_animals": 5,
    }  # fmt: skip


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


def _statistics() -> dict:
    return {
        "calibration": _calibration(),
        "sweep": _sweep(),
        "mua": _mua_counts(),
        "ripple_modes": _ripple_modes(),
        "hpd": _hpd(),
        "timing": _timing(),
        "tetrode": _split_statistics(),
        "shared": _shared_statistics(),
        "outside": _outside_rows(),
        "parts": _parts(),
        "no_spike": _no_spike_trace(),
    }


def test_the_figure_draws_its_panels_in_the_cited_order():
    """In four rows, each a group the text cites in turn: the decoder (a, b),
    the count is robust (c, d), the events are coordinated and specific (e to
    h), ripple power and the held-out test (i, j); each row's letters at one
    height, the rows' first letters at one left edge, no heading."""
    assert figure.PANELS == (
        "state_calibration", "no_spike_mua", "rule_sweep", "boundary_timing",
        "nonlocal_mua", "heldout_gain", "hpd", "shared_content", "ripple_modes",
        "outside_distance",
    )  # fmt: skip
    statistics = _statistics()
    fig = figure.generate_figure(statistics)
    ax_gain = figure.panel_axes(fig, "heldout_gain")[0]
    assert ax_gain.get_title() == "Held-out tetrode"
    assert ax_gain.get_ylabel() == "Held-out gain\nover local (nats/spike)"
    assert figure.panel_axes(fig, "no_spike_mua")[0].get_title() == "No-Spike state"
    assert (
        figure.panel_axes(fig, "boundary_timing")[0].get_title()
        == "Timing against the SWR"
    )
    assert figure.panel_axes(fig, "shared_content")[0].get_title() == "Shared content"
    letters = {
        t.get_text(): t.get_position() for t in fig.texts if len(t.get_text()) == 1
    }
    assert sorted(letters) == list("abcdefghij")
    rows = ("ab", "cd", "efgh", "ij")
    for row in rows:
        assert len({letters[k][1] for k in row}) == 1, row
    heights = [letters[row[0]][1] for row in rows]
    assert heights == sorted(heights, reverse=True)
    assert len({letters[row[0]][0] for row in rows}) == 1
    assert "Immobile, at wells" not in [
        t.get_text() for ax in fig.axes for t in ax.texts
    ]
    figure.check_journal_limits(fig)
    plt.close(fig)


def test_the_held_out_gain_panel_splits_the_gain_by_bins_with_and_without_spikes():
    """For no-SWR then SWR events, three positions in the group's color (all
    bins, the bins where the scored tetrode spiked, the silent bins): a dot per
    animal, the mean with its interval beside them; the parts named so that a
    reader needs no caption, the groups named over their positions, a dashed
    zero line with its note."""
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    fig = figure.generate_figure(_statistics())
    (ax,) = figure.panel_axes(fig, "heldout_gain")
    dots = [
        c for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]  # fmt: skip
    assert len(dots) == 6
    for k, (_group, color) in enumerate(panels.GROUP_COLORS.items()):
        for j, groups in enumerate(SPLIT.values()):
            x = panels.split_x(k, j)
            (mine,) = [d for d in dots if np.allclose(d.get_offsets()[:, 0], x)]
            np.testing.assert_allclose(mine.get_offsets()[:, 1], groups[k])
            np.testing.assert_allclose(mine.get_facecolor()[0], to_rgba(color))
            assert np.allclose(_means_at(ax, x), np.mean(groups[k]))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == 2 * list(panels.SPLIT_LABELS.values())
    assert list(panels.SPLIT_LABELS.values()) == [
        "All bins", "Bins with\na spike", "Silent\nbins",
    ]  # fmt: skip
    texts = [t.get_text() for t in ax.texts]
    assert {"No SWR", "SWR", "no gain"} <= set(texts)
    assert any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )
    plt.close(fig)


def test_each_panels_values_carry_its_letter():
    """The events summary tags each panel's values with the panel's letter, as
    the figure places it."""
    from continuum_swr_replay.reported import events

    panels = {
        "state_calibration": "state_calibration",
        "rule_sweep": "rule_sweep",
        "rule_sweep_swr": "ripple_modes",
        "boundary_timing": "boundary_timing",
        "nonlocal_mua": "nonlocal_mua",
        "hpd": "hpd",
        "tetrode_holdout": "heldout_gain",
        "spike_rate": "heldout_gain",
        "shared_content": "shared_content",
        "class_distribution": "ripple_modes",
        "nonlocal_restriction_distance": "outside_distance",
    }
    for tag, panel in panels.items():
        assert events.TAGS[tag].figures == (f"S2{figure.LETTERS[panel]}",), tag
    assert set(panels.values()) == set(figure.PANELS) - {"no_spike_mua"}


def test_panel_b_draws_the_timing_table():
    """The nine shares as a 3 by 3 heat map (rows the start, columns the end),
    each cell's percent and its animals' range, white on the dark cell and
    black on the pale ones, and the labels."""
    fig = figure.generate_figure(_statistics())
    (ax,) = figure.panel_axes(fig, "boundary_timing")
    (image,) = ax.images
    np.testing.assert_allclose(
        image.get_array(), 100 * _timing()["share"].to_numpy().reshape(3, 3)
    )
    texts = {t.get_text(): t for t in ax.texts}
    assert texts["70%"].get_position() == pytest.approx((1, 1 + figure.SHARE_DROP))
    assert texts["15%"].get_position()[0] == 2  # start together, end later
    assert "69\u201371" in texts
    assert texts["70%"].get_color() == "white"
    assert texts["5%"].get_color() == "black"
    assert [t.get_text() for t in ax.get_yticklabels()] == list(figure.TIMING_LABELS)
    assert ax.get_ylabel() == "Event starts,\nvs SWR start"
    assert ax.get_xlabel() == "Event ends, vs SWR end"
    plt.close(fig)


def test_panel_b_names_the_four_corner_cells():
    """The corner cells, where the event and its SWR overlap but neither
    bound is together, are named in small text at the top of each cell: start
    and end earlier, the event leads; start earlier and end later, it
    contains the SWR; start later and end earlier, it is inside the SWR;
    start and end later, it lags. The middle row and column have no name."""
    fig = figure.generate_figure(_statistics())
    (ax,) = figure.panel_axes(fig, "boundary_timing")
    named = {
        t.get_text(): t.get_position()
        for t in ax.texts
        if t.get_text() in figure.TIMING_CORNERS.values()
    }
    assert named == {
        "event leads": pytest.approx((0, figure.CORNER_DROP)),
        "contains SWR": pytest.approx((2, figure.CORNER_DROP)),
        "inside SWR": pytest.approx((0, 2 + figure.CORNER_DROP)),
        "event lags": pytest.approx((2, 2 + figure.CORNER_DROP)),
    }
    assert figure.CORNER_DROP < figure.SHARE_DROP < figure.RANGE_DROP
    for t in ax.texts:
        if t.get_text() in named:
            assert t.get_fontsize() >= 6
    plt.close(fig)


def test_panel_d_draws_the_agreement_by_spread():
    """Each group's dots per spread bin on its side of the bin, its means with
    their intervals farther out joined by a thin line in the group's color, the
    bins' labels, a dashed zero line and which way means shared content."""
    fig = figure.generate_figure(_statistics())
    ax_share, ax = figure.panel_axes(fig, "shared_content")
    assert [t.get_text() for t in ax.get_xticklabels()] == list(
        figure.SPREAD_BIN_LABELS
    )
    bins = np.arange(len(figure.SPREAD_BIN_LABELS))
    for side, group in ((-1, "no SWR"), (1, "SWR")):
        expected = [np.mean(SHARED[(group, 0)]) - 0.1 * k for k in bins]
        means = bins + side * (figure.SHARED_OFFSET + figure.SHARED_MEAN_OFFSET)
        (series,) = [
            line for line in ax.lines
            if len(line.get_xdata()) == bins.size
            and np.allclose(line.get_xdata(), means)
            and line.get_linestyle() != "None"  # not the means' markers
        ]  # fmt: skip
        np.testing.assert_allclose(series.get_ydata(), expected, atol=1e-12)
        for k in bins:
            (dots,) = [
                c for c in ax.collections
                if isinstance(c, matplotlib.collections.PathCollection)
                and np.allclose(
                    c.get_offsets()[:, 0], k + side * figure.SHARED_OFFSET, atol=0.1
                )
            ]  # fmt: skip
            np.testing.assert_allclose(
                sorted(dots.get_offsets()[:, 1]),
                sorted(np.asarray(SHARED[(group, 0)]) - 0.1 * k),
            )
    assert any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )
    # a strip above, sharing the x axis: each group's percent of its events per
    # bin as two narrow bars side by side, in the groups' colors
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    bars = list(ax_share.patches)
    assert len(bars) == 2 * len(bins)
    for side, (group, color) in zip((-1, 1), panels.GROUP_COLORS.items(), strict=True):
        mine = [b for b in bars if to_rgba(b.get_facecolor()) == to_rgba(color)]
        np.testing.assert_allclose(
            [b.get_x() + b.get_width() / 2 for b in mine],
            bins + side * figure.SHARE_BAR_WIDTH / 2,
        )
        np.testing.assert_allclose(
            [b.get_height() for b in mine], 100 * np.asarray(SHARE_OF_EVENTS[group])
        )
    assert ax_share.get_shared_x_axes().joined(ax_share, ax)
    assert ax_share.get_ylabel() == "% of\nevents"
    assert ax_share.get_title() == "Shared content"
    assert not any(t.get_visible() and t.get_text() for t in ax_share.get_xticklabels())
    texts = [t.get_text() for t in ax.texts]
    assert "\u2191 shared\ncontent" in texts
    assert {"No SWR", "SWR"} <= set(texts)
    assert ax.get_ylabel() == "Held-out agreement\n(nats/spike)"
    assert ax.get_xlabel() == "Width of the other tetrodes'\nposition estimate (cm)"
    plt.close(fig)


def test_panel_e_draws_the_outside_swr_gain_then_away_from_swrs():
    """Outside SWRs, then only farther than each margin from every SWR: dots
    spread about each column, the mean with its interval, a thin line through
    the means, the margins' label, the difference's y label, the zero line and
    "full better". The gain at well-sampled places is not drawn."""
    fig = figure.generate_figure(_statistics())
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
    plt.close(fig)


def test_the_hpd_panel_draws_the_distribution_of_each_class():
    """For each class, a thin line per animal and their mean in bold, in the
    class's color, on a log x axis with the named ticks; a dotted line at the
    tight cut; the classes named in their colors; a y axis from 0."""
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    distribution = _hpd()
    fig = figure.generate_figure(_statistics())
    (ax,) = figure.panel_axes(fig, "hpd")
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
        "10", "25", "50", "100", "200", "400",
    ]  # fmt: skip
    (cut,) = [line for line in ax.lines if line.get_linestyle() == ":"]
    np.testing.assert_allclose(cut.get_xdata(), figure.HPD_TIGHT_CM)
    assert {"No SWR", "SWR", "50 cm"} <= {t.get_text() for t in ax.texts}
    assert ax.get_ylim()[0] == 0
    assert ax.get_title() == "Posterior spread"
    plt.close(fig)


def test_the_ripple_panel_draws_the_peak_power_and_the_share_at_each_threshold():
    """Two axes sharing the ripple-power x axis. Top: each animal's density of
    the events' peak ripple power as a thin line, their mean in bold, the
    one-mode result in its title, no split by SWR label. Bottom: the percent of events with
    no SWR at each SWR threshold, a thin line per animal and the cohort's
    line. One dashed line at the threshold used on both, labeled once; each
    axis names what its x is."""
    from continuum_swr_replay.analysis.event_properties.rule_sweep import (
        SWR_THRESHOLDS_Z,
    )

    fig = figure.generate_figure(_statistics())
    ax_peak, ax_share = figure.panel_axes(fig, "ripple_modes")
    densities = _ripple_modes()["densities"]
    curves = [
        line for line in ax_peak.lines if len(line.get_xdata()) == RIPPLE_GRID.size
    ]
    assert len(curves) == 3  # two animals and their mean
    np.testing.assert_allclose(
        curves[-1].get_ydata(), densities.mean(axis=1).to_numpy()
    )
    assert curves[-1].get_linewidth() > curves[0].get_linewidth()
    assert ax_peak.get_title() == "One mode in 5 of 5 animals (dip test)"
    assert not ax_peak.collections  # no band, no stacked label
    sweep = _sweep()
    mine = sweep[sweep["setting"] == "swr_threshold_z"]
    shares = [
        line
        for line in ax_share.lines
        if len(line.get_xdata()) == len(SWR_THRESHOLDS_Z)
    ]
    assert len(shares) == 2 + 1
    np.testing.assert_allclose(shares[-1].get_xdata(), SWR_THRESHOLDS_Z)
    np.testing.assert_allclose(
        shares[-1].get_ydata(), 100 * mine["no_swr_fraction"].to_numpy()
    )
    for ax in (ax_peak, ax_share):
        (threshold,) = [line for line in ax.lines if line.get_linestyle() == "--"]
        np.testing.assert_allclose(threshold.get_xdata(), figure.KAY_ZSCORE_THRESHOLD)
    labels = [t.get_text() for ax in (ax_peak, ax_share) for t in ax.texts]
    assert labels.count("SWR threshold used") == 1
    assert "Event peak" in [t.get_text() for t in ax_peak.texts]
    assert "No SWR at this threshold" in [t.get_text() for t in ax_share.texts]
    assert ax_share.get_shared_x_axes().joined(ax_share, ax_peak)
    assert ax_share.get_xlabel() == "Ripple power (z)"
    assert ax_peak.get_xlim() == figure.RIPPLE_X_LIMITS_Z
    plt.close(fig)


def test_the_mua_panel_draws_where_the_nonlocal_time_sits_in_mua():
    """As Figure 2i draws ripple power: top, each animal's non-local time
    spread over MUA with the share below the MUA level written; bottom, the
    share of bins decoded non-local at each level; a dashed line at the level
    on both."""
    fig = figure.generate_figure(_statistics())
    ax_spread, ax_share = figure.panel_axes(fig, "nonlocal_mua")
    for ax in (ax_spread, ax_share):
        assert len(ax.lines) == 2 + 2  # two animals, their mean, the level
        (level,) = [line for line in ax.lines if line.get_linestyle() == "--"]
        np.testing.assert_allclose(level.get_xdata(), figure.MUA_LEVEL_Z)
    # non-local bins below 2 z: 2 in each of the 17 bins below (the open one
    # too) against 10 in each of the 33 at or above
    below = 17 * 2 / (17 * 2 + 33 * 10)
    assert ax_spread.texts[0].get_text() == f"{round(100 * below)}% below\n2 z"
    assert ax_share.get_xlabel() == "MUA (z)"
    assert ax_spread.get_title() == "Non-local time"
    plt.close(fig)


def test_the_sweep_panel_draws_the_share_with_no_swr_at_each_setting():
    """Two small axes sharing the y axis, one per setting of the event rule: a
    thin line per animal and the pooled share in bold, in percent, the value
    used dotted and marked, and the title "Event rule" over both (the SWR
    detector's threshold is in the ripple panel)."""
    from continuum_swr_replay.analysis.event_properties.rule_sweep import (
        RULE,
        SETTINGS,
    )

    sweep = _sweep()
    fig = figure.generate_figure(_statistics())
    axes = figure.panel_axes(fig, "rule_sweep")
    assert len(axes) == 2
    for ax, setting in zip(axes, figure.SWEEP_XLABELS, strict=True):
        values = SETTINGS[setting]
        mine = sweep[sweep["setting"] == setting]
        lines = [line for line in ax.lines if len(line.get_xdata()) == len(values)]
        assert len(lines) == 2 + 1  # two animals and the pooled share
        np.testing.assert_allclose(lines[-1].get_xdata(), values)
        np.testing.assert_allclose(
            lines[-1].get_ydata(), 100 * mine["no_swr_fraction"].to_numpy()
        )
        (rule,) = [line for line in ax.lines if line.get_linestyle() == ":"]
        np.testing.assert_allclose(rule.get_xdata(), RULE[setting])
        assert ax.get_shared_y_axes().joined(ax, axes[0])
        assert " used" in [t.get_text() for t in ax.texts]
    assert axes[0].get_ylabel() == "Events with\nno SWR (%)"
    assert "Event rule" in [t.get_text() for t in axes[0].texts]
    assert list(figure.SWEEP_XLABELS) == ["probability_threshold", "minimum_samples"]
    assert [ax.get_xlabel() for ax in axes] == list(figure.SWEEP_XLABELS.values())
    plt.close(fig)


def test_the_calibration_panel_draws_each_states_evidence_by_its_probability():
    """Local, Continuous and Fragmented on one axis, each the animals' mean
    with its interval near the probability bins' centers in the state's color,
    named at its line's end; a dashed zero line; a y label that says what the
    evidence is against; No-Spike is not drawn."""
    from matplotlib.colors import to_rgba

    fig = figure.generate_figure(_statistics())
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
    plt.close(fig)


def test_panel_a_draws_the_mua_around_no_spike_starts():
    """The median with its interquartile band, a dotted line at the start, a
    dashed line at the level outside the periods, and its labels."""
    trace = _no_spike_trace()
    fig = figure.generate_figure(_statistics())
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
    plt.close(fig)


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
    trace = figure.read_no_spike_trace()
    assert read == paths
    assert visits == ["a1_02_r1_HPC", "a2_02_r1_HPC"]
    np.testing.assert_allclose(trace["median"], [2.0, 0.0])
    assert trace["periods"].to_dict() == {"a": 2}  # one period per epoch
    assert trace["level"] == pytest.approx(1.0)
    assert trace["median_duration"] == pytest.approx(9 * 0.002)


def test_the_figure_is_the_question_lists_held_out_tetrode_figure():
    """Values tagged ``S2`` (by panel) name this script."""
    assert FIGURES["S2"].panels == "abcdefghij"
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
        "window_scores": lambda table: seen.setdefault("tetrode", table),
        "holdout_statistics": lambda windows, kind, column: windows,
        "agreement_by_spread_statistics": lambda windows: windows,
        "sha256_file": lambda path: "0" * 64,
        "read_heldout_restriction": lambda *args: heldout or _HeldoutRestriction(),
        "restriction_inputs_note": lambda folder, epochs: note,
        "epoch_outside_swr_gains": lambda sums, bin_s, events: seen.setdefault(
            "restriction", (sums, bin_s, events)
        ),
        "outside_swr_gain_statistics": lambda epochs: pd.DataFrame(
            {"family": ["filter"], "measure": ["gain_per_spike"]}
        ),
        "read_no_spike_trace": lambda: "trace",
        "read_hpd": lambda events: seen.setdefault("hpd", events),
        "read_mua_counts": lambda folder: "mua",
        "read_rule_sweep": lambda folder, table: "sweep",
        "read_state_calibration": lambda folder: "calibration",
        "read_ripple_modes": lambda events, folder, table: seen.setdefault(
            "ripple", events
        ),
        "outside_swr_subset_statistics": lambda sums, bin_s: pd.DataFrame(
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
    """The stored held-out tetrode run (held out in folds of whole trials)
    scores every event; the figure, like the events summary, reads the rows
    (every tetrode's) of the events that start inside a well visit, and the
    held-out event run's events and time sums by the summary's reader."""
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
    assert list(seen["tetrode"]["event_id"]) == [0, 0, *range(1, 8)]
    sums, bin_s, events = seen["restriction"]
    assert len(sums) == 3 and bin_s.to_dict() == {"e": 0.002} and len(events) == 2
    assert seen["hpd"]["event_id"].tolist() == list(range(8))
    assert seen["ripple"]["event_id"].tolist() == list(range(8))
    assert list(inputs) == [
        "calibration",
        "sweep",
        "mua",
        "ripple_modes",
        "hpd",
        "timing",
        "tetrode",
        "shared",
        "outside",
        "parts",
        "no_spike",
    ]


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
    """Panel e is refused as Figure 2j is: from a restriction run off
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
    monkeypatch.setattr(figure, "epoch_outside_swr_gains", refuse)
    with pytest.raises(ValueError, match="panel j is not drawn") as refused:
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
