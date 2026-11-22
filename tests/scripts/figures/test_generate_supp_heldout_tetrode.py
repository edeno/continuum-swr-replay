"""The held-out tetrode supplementary figure: what it draws and from which values."""

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
from scripts.figures import generate_supp_heldout_tetrode as figure  # noqa: E402
from tests.scripts.figures.test_figure_script_fixes import (  # noqa: E402
    OUTSIDE,
    SHARES,
    _figure02_events,
    _gain_statistics,
    _means_at,
    _measure_rows,
    _outside_rows,
    assert_draws_group_gain,
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


def _statistics() -> dict:
    return {
        "tetrode": _gain_statistics([0.52, 0.47, 0.25, 0.43, 0.46], [0.86, 0.8, 0.7]),
        "outside": _outside_rows(),
        "parts": _parts(),
        "no_spike": _no_spike_trace(),
    }


def test_the_figure_draws_five_panels_in_order():
    """a, the held-out tetrode gain as before; b and c, the outside-SWR gain
    then its parts away from SWRs and at well-sampled places; d, the shares,
    in one row under the heading over b to d; e, the No-Spike MUA, under a. A
    row's letters at one height; a's and e's at one left edge."""
    statistics = _statistics()
    fig = figure.generate_figure(statistics)
    ax_a, ax_b, ax_c, ax_d, ax_e = fig.axes
    assert_draws_group_gain(ax_a, statistics["tetrode"])
    assert ax_a.get_title() == "Held-out tetrode"
    assert ax_a.get_ylabel() == "Held-out gain\nover local (nats/spike)"
    letters = {
        t.get_text(): t.get_position() for t in fig.texts if len(t.get_text()) == 1
    }
    assert sorted(letters) == list("abcde")
    assert len({letters[k][1] for k in "abcd"}) == 1
    assert letters["e"][1] < letters["a"][1]
    assert letters["e"][0] == letters["a"][0]
    assert "Immobile, at wells" in [t.get_text() for t in ax_b.texts]
    figure.check_journal_limits(fig)
    plt.close(fig)


@pytest.mark.parametrize(
    ("panel", "parts", "names", "label"),
    [
        (1, MARGINS, ["100", "250", "500"], "at least this far from\nany SWR (ms)"),
        (
            2,
            SAMPLED,
            ["10", "60"],
            "training data at the\nanimal's place (s, at least)",
        ),
    ],
    ids=["away from SWRs", "well-sampled places"],
)
def test_panels_b_and_c_draw_the_outside_swr_gain_then_its_parts(
    panel, parts, names, label
):
    """Outside SWRs, then each part: dots spread about each column, the mean
    with its interval, a thin line through the means, the parts' label, the
    difference's y label, the zero line and "full better"."""
    fig = figure.generate_figure(_statistics())
    ax = fig.axes[panel]
    xs = [0.0, *(figure.SERIES_START + k for k in range(len(parts)))]
    for x, values in zip(xs, [OUTSIDE, *parts.values()], strict=True):
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
        series.get_ydata(), [np.mean(v) for v in [OUTSIDE, *parts.values()]]
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Outside\nSWRs", *names]
    texts = [t.get_text() for t in ax.texts]
    assert label in texts and "↑ full better" in texts
    assert ax.get_ylabel() == "Held-out log score,\nfull − SWR-only\n(nats/spike)"
    # b and c share one y range
    assert fig.axes[1].get_ylim() == fig.axes[2].get_ylim()
    plt.close(fig)


def test_panel_d_names_each_share_beside_its_dots():
    """The three shares in percent in one column, in panel d's colors and a
    gray for the rest, each a dot per animal (spread), the mean with its
    interval and its name beside them, no lighter than the reference gray; no x
    tick labels; a dashed zero line; a y label that says what is described."""
    from matplotlib.colors import to_hex, to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    fig = figure.generate_figure(_statistics())
    ax = fig.axes[3]
    dots = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]
    assert len(dots) == 3
    means = _means_at(ax, 0.0)
    for drawn, mean, (measure, values) in zip(dots, means, SHARES.items(), strict=True):
        np.testing.assert_allclose(
            sorted(drawn.get_offsets()[:, 1]), sorted(100 * np.asarray(values))
        )
        assert_spread_about(drawn, 0.0)
        np.testing.assert_allclose(
            drawn.get_facecolors()[0], to_rgba(panels.SHARE_COLORS[measure])
        )
        assert np.allclose(mean, 100 * np.mean(values)), measure
    names = list(ax.texts)
    assert [n.get_text() for n in names] == [
        "No SWR",
        "Partly in SWR",
        "Outside events",
    ]
    for name, values in zip(names, SHARES.values(), strict=True):
        assert name.xy[1] == pytest.approx(100 * np.mean(values))
        assert name.get_ha() == "left"
    # the gray dots' name is in the reference gray, no lighter
    assert to_hex(names[2].get_color()) == panels.REFERENCE_COLOR
    assert list(ax.get_xticks()) == []
    assert ax.get_ylim()[0] < 100 * min(SHARES["share_outside_events"])
    assert ax.get_ylabel() == "Share of the\noutside-SWR gain (%)"
    plt.close(fig)


def test_panel_e_draws_the_mua_around_no_spike_starts():
    """The median with its interquartile band, a dotted line at the start, a
    dashed line at the level outside the periods, and its labels."""
    trace = _no_spike_trace()
    fig = figure.generate_figure(_statistics())
    ax = fig.axes[4]
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
    assert ax.get_xlabel() == "Time from No-Spike start (ms)"
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
    """Values tagged ``heldout_tetrode`` name this script."""
    assert FIGURES["heldout_tetrode"].script == (
        "scripts/figures/generate_supp_heldout_tetrode.py"
    )


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
        "holdout_statistics": lambda windows, kind: windows,
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
        "scores": pd.DataFrame({"event_id": [0, 0, *range(1, 8), 99, 99]}),
    }
    seen = {}
    _patch_reading(monkeypatch, stored, seen)
    inputs = figure.read_inputs()
    assert list(seen["tetrode"]["event_id"]) == [0, 0, *range(1, 8)]
    sums, bin_s, events = seen["restriction"]
    assert len(sums) == 3 and bin_s.to_dict() == {"e": 0.002} and len(events) == 2
    assert list(inputs) == ["tetrode", "outside", "parts", "no_spike"]


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
    """Panels b to d are refused as Figure 2g is: from a restriction run off
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
    with pytest.raises(ValueError, match="panels b to d are not drawn") as refused:
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
