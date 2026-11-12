"""Publication-facing inference labels for Figure 4."""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from continuum_swr_replay.analysis.pfc.heldout import DEFAULT_CONTROL_DRAWS
from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS
from continuum_swr_replay.visualization import figure_utilities

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "generate_figure04",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "figures"
    / "generate_figure04.py",
)
figure04 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure04)


def test_panel_d_uses_the_canonical_control_draw_count():
    assert figure04.DEFAULT_CONTROL_DRAWS == DEFAULT_CONTROL_DRAWS == 1_000

    frame = pd.DataFrame({column: [0.01, 0.5] for column in HELDOUT_P_COLUMNS.values()})
    fig, ax = plt.subplots()
    try:
        figure04.plot_significance(frame, ax)
        assert "R=1000, min p=0.001" in ax.get_title(loc="left")
    finally:
        plt.close(fig)


def test_panel_d_draws_the_threshold_and_the_draw_count_it_is_given():
    frame = pd.DataFrame({column: [0.01, 0.5] for column in HELDOUT_P_COLUMNS.values()})
    fig, ax = plt.subplots()
    try:
        figure04.plot_significance(frame, ax, alpha=0.6, n_control_draws=99)
        # at 0.6 both units are called, at the default 0.05 one of the two
        assert [bar.get_height() for bar in ax.patches] == [100.0] * 4
        assert ax.get_title(loc="left") == (
            "Raw empirical p < 0.6 (descriptive)\nR=99, min p=0.010"
        )
        # the dashed line is the threshold, as a percent
        assert list(ax.lines[0].get_ydata()) == pytest.approx([60.0, 60.0])
    finally:
        plt.close(fig)


def test_panel_d_bars_are_the_called_and_best_model_shares_as_percents():
    """Four units. SWR: one untested, two of the three tested called.
    Engagement: three of four called. The other tests: one of four. Best
    model: one constant, two SWR, and one unit with none, which is left out."""
    pvalues = {name: [0.01, 0.5, 0.5, 0.5] for name in HELDOUT_P_COLUMNS}
    pvalues["SWR"] = [0.01, 0.01, 0.5, np.nan]
    pvalues["engagement"] = [0.01, 0.01, 0.01, 0.5]
    frame = pd.DataFrame(
        {HELDOUT_P_COLUMNS[name]: values for name, values in pvalues.items()}
    )
    frame["min_heldout_model"] = ["constant", "SWR", "SWR", None]

    fig, (called_ax, best_ax) = plt.subplots(1, 2)
    try:
        figure04.plot_significance(frame, called_ax)
        figure04.plot_best_model_selection(frame, best_ax)

        assert [t.get_text() for t in called_ax.get_xticklabels()] == [
            "SWR",
            "Replay",
            "no-SWR",
            "Beyond\ncontrol",
        ]
        assert [bar.get_height() for bar in called_ax.patches] == pytest.approx(
            [200 / 3, 25.0, 25.0, 75.0]
        )
        assert [t.get_text() for t in called_ax.texts] == ["67%", "25%", "25%", "75%"]
        assert [t.get_text() for t in best_ax.get_xticklabels()] == [
            "Const.",
            "SWR",
            "Replay",
            "SWR+\nno-SWR",
        ]
        assert [bar.get_height() for bar in best_ax.patches] == pytest.approx(
            [100 / 3, 200 / 3, 0.0, 0.0]
        )
        assert [t.get_text() for t in best_ax.texts] == ["33%", "67%"]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Layout and clipping: drawn at the figure's real size, in the production style
# ---------------------------------------------------------------------------

#: Text this close (points) counts as touching: two tick labels one point apart
#: still read as one word.
MIN_TEXT_GAP_PT = 1.0

_BASIS_CENTERS = np.linspace(-0.2, 0.2, 6)


def _synthetic_inputs(non_swr_peak=1.6, n_units=150):
    """Inputs of ``assemble_figure`` with the shapes the real ones have.

    The non-SWR replay curve is built to rise above 2x (log 2) late in the
    window, as the example neuron's does. Seeded, so a failure repeats.
    """
    rng = np.random.default_rng(0)
    lag_time = np.arange(-100, 101) / 500
    basis = np.exp(-((lag_time[:, None] - _BASIS_CENTERS[None]) ** 2) / (2 * 0.05**2))
    swr_params = np.array([0.1, 0.2, 0.0, -0.1, 0.2, 0.3])
    non_swr_params = np.array([-0.1, -0.1, 0.1, 0.3, non_swr_peak, 0.6])
    # The joint ``SWR + non_local_no_SWR`` fit panel a reads both curves off:
    # ``[intercept, SWR block, non_local_no_SWR block]``, the design's order.
    params = np.concatenate([[0.0], swr_params, non_swr_params])
    covariance = np.diag(np.r_[0.0, np.full(2 * _BASIS_CENTERS.size, 0.2**2)])
    glm_results = {
        "SWR + non_local_no_SWR": SimpleNamespace(params=params, covariance=covariance),
    }
    starts = np.arange(20) * 5.0
    events = pd.DataFrame({"start_time": starts, "end_time": starts + 0.1})
    pop_df = pd.DataFrame(
        {
            column: np.where(rng.uniform(size=n_units) < fraction, 0.01, 0.5)
            for column, fraction in zip(
                HELDOUT_P_COLUMNS.values(), [0.31, 0.40, 0.23, 0.0, 0.23], strict=True
            )
        }
    )
    pop_df["min_heldout_model"] = rng.choice(
        ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"],
        size=n_units,
        p=[0.48, 0.14, 0.33, 0.05],
    )
    return {
        "lag_time": lag_time,
        "spline_basis": basis,
        "neuron_spike_times": np.sort(rng.uniform(0, 100, 800)),
        "glm_results": glm_results,
        "event_times": {"SWR": events, "non_local_no_SWR": events},
        "pop_df": pop_df,
        "modulations": {
            name: rng.normal(0, 0.1, (lag_time.size, n_units))
            for name in ("swr", "no_swr")
        },
        "pop_lag_time": lag_time,
    }


@pytest.fixture
def paper_style():
    """The style the figure is saved in, without leaking it into other tests.

    ``set_figure_defaults("paper")`` resets rcParams and sets font size 7, dpi 450
    and constrained layout. Inside ``rc_context`` it is undone on exit, and a
    layout test cannot pass only because another test's import left a style on.
    """
    with matplotlib.rc_context():
        figure_utilities.set_figure_defaults(context="paper")
        # The style asks for Arial first and falls back to DejaVu Sans, so the
        # drawn text depends on the host's fonts. DejaVu Sans ships with
        # matplotlib and is the wider of the two: a layout that clears it clears
        # Arial, and the test no longer depends on what is installed.
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        yield


@pytest.fixture
def drawn_figure(paper_style):
    """The assembled Figure 4 on synthetic inputs, drawn once at real size."""
    fig = figure04.assemble_figure(**_synthetic_inputs())
    fig.canvas.draw()
    try:
        yield fig
    finally:
        plt.close(fig)


def _axis_with_title(fig, prefix):
    (ax,) = [
        ax
        for ax in fig.axes
        if any(
            ax.get_title(loc=loc).startswith(prefix)
            for loc in ("left", "center", "right")
        )
    ]
    return ax


def _axis_with_ylabel(fig, text):
    (ax,) = [ax for ax in fig.axes if ax.get_ylabel() == text]
    return ax


def _drawn_tick_labels(axis):
    """Tick labels of ticks inside the view interval: the ones that are drawn."""
    low, high = sorted(axis.get_view_interval())
    axis.get_ticklabels()  # refreshes each tick's text and position
    ticks = axis.get_major_ticks(len(axis.get_ticklocs()))
    return [
        tick.label1
        for tick in ticks
        if low <= tick.get_loc() <= high
        and tick.label1.get_visible()
        and tick.label1.get_text()
    ]


def _drawn_texts(ax):
    """Every text the axes draws: titles, annotations, axis labels, ticks, legend."""
    texts = [child for child in ax.get_children() if isinstance(child, Text)]
    texts += [ax.xaxis.label, ax.yaxis.label]
    texts += _drawn_tick_labels(ax.xaxis) + _drawn_tick_labels(ax.yaxis)
    if ax.get_legend() is not None:
        texts += ax.get_legend().get_texts()
    return [text for text in texts if text.get_visible() and text.get_text().strip()]


def _extent(fig, text):
    return text.get_window_extent(fig.canvas.get_renderer())


def _touching(fig, first, second):
    """Whether two artists' extents come within ``MIN_TEXT_GAP_PT`` of each other."""
    pad = 0.5 * MIN_TEXT_GAP_PT * fig.dpi / 72
    return _extent(fig, first).padded(pad).overlaps(_extent(fig, second).padded(pad))


def _panel_letter(fig, letter):
    (text,) = [
        text for ax in fig.axes for text in ax.texts if text.get_text() == letter
    ]
    return text


@pytest.mark.parametrize("title_prefix", ["Raw empirical", "Best model"])
def test_panel_d_x_tick_labels_do_not_overlap(drawn_figure, title_prefix):
    """The four bar labels ran together ("SWRReplayno-SWRBeyond control") at the
    figure's real size, in both panel d axes."""
    ax = _axis_with_title(drawn_figure, title_prefix)

    labels = _drawn_tick_labels(ax.xaxis)

    assert len(labels) == 4
    touching = [
        (first.get_text(), second.get_text())
        for first, second in combinations(labels, 2)
        if _touching(drawn_figure, first, second)
    ]
    assert touching == []


@pytest.mark.parametrize("letter", ["a", "b", "c", "d"])
def test_a_panel_letter_overlaps_no_text(drawn_figure, letter):
    """The "d" sat on the left end of the panel d title ("Raw empirical p < ...")."""
    label = _panel_letter(drawn_figure, letter)

    others = [
        text
        for ax in drawn_figure.axes
        for text in _drawn_texts(ax)
        if text is not label
    ]
    touching = [
        text.get_text() for text in others if _touching(drawn_figure, label, text)
    ]
    assert touching == []


def test_panel_d_titles_do_not_overlap_each_other(
    drawn_figure,
):
    first = _axis_with_title(drawn_figure, "Raw empirical")
    second = _axis_with_title(drawn_figure, "Best model")
    titles = [
        [text for text in _drawn_texts(ax) if text.get_text().startswith(prefix)]
        for ax, prefix in ((first, "Raw empirical"), (second, "Best model"))
    ]
    (title_one,), (title_two,) = titles

    assert not _touching(drawn_figure, title_one, title_two)


def _draw_overlay(inputs):
    """Panel a drawn on its own axes, for the limits and legend of the curves."""
    fig, (ax_mod, ax_psth) = plt.subplots(2, 1, sharex=True)
    configs = [
        {"curve": "SWR", "label": "SWR", "color": "C1", "event_key": "SWR"},
        {
            "curve": "no_SWR",
            "label": "non-SWR replay",
            "color": "C2",
            "event_key": "non_local_no_SWR",
        },
    ]
    figure04.plot_single_neuron_overlay(
        axes=[ax_mod, ax_psth],
        lag_time=inputs["lag_time"],
        spline_basis=inputs["spline_basis"],
        neuron_spike_times=inputs["neuron_spike_times"],
        glm_results=inputs["glm_results"],
        event_times=inputs["event_times"],
        model_configs=configs,
    )
    fig.canvas.draw()
    return fig, ax_mod


def _curve_lines(ax, n_lags):
    """The drawn modulation curves: not the two dashed reference lines."""
    lines = [line for line in ax.lines if len(line.get_xdata()) == n_lags]
    assert len(lines) == 2
    return lines


def test_the_modulation_axis_limits_contain_a_curve_above_2x(paper_style):
    """The non-SWR replay curve rose above the fixed 2.0 top limit between
    about 0.08 and 0.17 s, so part of it was not drawn."""
    inputs = _synthetic_inputs(non_swr_peak=1.6)
    fig, ax_mod = _draw_overlay(inputs)
    try:
        lines = _curve_lines(ax_mod, inputs["lag_time"].size)
        assert max(line.get_ydata().max() for line in lines) > np.log(2.0)

        low, high = ax_mod.get_ylim()

        for line in lines:
            assert low <= line.get_ydata().min()
            assert line.get_ydata().max() <= high
    finally:
        plt.close(fig)


def test_the_modulation_axis_limits_contain_the_standard_error_bands(paper_style):
    inputs = _synthetic_inputs(non_swr_peak=1.6)
    fig, ax_mod = _draw_overlay(inputs)
    try:
        low, high = ax_mod.get_ylim()
        bands = [
            collection.get_paths()[0].vertices[:, 1]
            for collection in ax_mod.collections
        ]

        assert len(bands) == 2
        # The non-SWR replay band reaches above 2x too, not only its curve.
        assert max(band.max() for band in bands) > np.log(2.0)
        for band in bands:
            assert low <= band.min()
            assert band.max() <= high
    finally:
        plt.close(fig)


def test_the_modulation_axis_keeps_its_half_to_double_range_for_a_flat_neuron(
    paper_style,
):
    inputs = _synthetic_inputs()
    for result in inputs["glm_results"].values():
        result.params = 0.1 * result.params
        result.covariance = 0.01 * result.covariance
    fig, ax_mod = _draw_overlay(inputs)
    try:
        assert ax_mod.get_ylim() == (np.log(0.5), np.log(2.0))
    finally:
        plt.close(fig)


def test_the_modulation_axis_ticks_are_fold_changes_inside_the_limits(paper_style):
    """Each tick sits at the log of the fold change it is labelled with, the
    1x reference line is kept, and the labels are plain numbers."""
    fig, ax_mod = _draw_overlay(_synthetic_inputs(non_swr_peak=1.6))
    try:
        low, high = ax_mod.get_ylim()
        locations = ax_mod.get_yticks()
        labels = [label.get_text() for label in ax_mod.get_yticklabels()]

        assert all(low <= location <= high for location in locations)
        assert len(labels) >= 3
        assert "1" in labels
        for location, label in zip(locations, labels, strict=True):
            assert np.isclose(np.exp(location), float(label))
        reference = [
            line
            for line in ax_mod.lines
            if len(line.get_ydata()) == 2 and np.all(np.asarray(line.get_ydata()) == 0)
        ]
        assert len(reference) == 1
    finally:
        plt.close(fig)


def test_the_panel_a_legend_is_outside_the_modulation_axes(paper_style):
    """The legend, top right, covered the green curve and its band."""
    fig, ax_mod = _draw_overlay(_synthetic_inputs())
    try:
        renderer = fig.canvas.get_renderer()
        legend = ax_mod.get_legend()

        assert not legend.get_window_extent(renderer).overlaps(
            ax_mod.get_window_extent(renderer)
        )
    finally:
        plt.close(fig)


def test_the_panel_a_legend_is_clear_of_every_other_text(drawn_figure):
    ax_mod = _axis_with_ylabel(drawn_figure, "GLM\nmodulation")
    legend = ax_mod.get_legend()

    others = [
        text
        for ax in drawn_figure.axes
        for text in _drawn_texts(ax)
        if text not in legend.get_texts()
    ]
    assert [
        text.get_text() for text in others if _touching(drawn_figure, legend, text)
    ] == []


def test_panel_a_draws_the_joint_models_two_blocks_in_design_order(paper_style):
    """Panel a is the population's quantity: the SWR and non-SWR replay kernels
    of the joint ``SWR + non_local_no_SWR`` fit, block 0 and block 1 in the
    design's order, each with its pointwise standard error from the joint
    fit's covariance (off-diagonal terms included)."""
    inputs = _synthetic_inputs(non_swr_peak=1.6)
    rng = np.random.default_rng(1)
    n_params = inputs["glm_results"]["SWR + non_local_no_SWR"].params.size
    root = 0.1 * rng.normal(size=(n_params, n_params))
    covariance = root @ root.T
    inputs["glm_results"]["SWR + non_local_no_SWR"].covariance = covariance
    basis = inputs["spline_basis"]
    n_basis = basis.shape[1]
    params = inputs["glm_results"]["SWR + non_local_no_SWR"].params

    fig, ax_mod = _draw_overlay(inputs)
    try:
        lines = _curve_lines(ax_mod, inputs["lag_time"].size)
        bands = [
            collection.get_paths()[0].vertices for collection in ax_mod.collections
        ]
        for line, band, block in zip(lines, bands, (0, 1), strict=True):
            block_slice = slice(1 + block * n_basis, 1 + (block + 1) * n_basis)
            expected = basis @ params[block_slice]
            np.testing.assert_allclose(line.get_ydata(), expected)
            block_covariance = covariance[block_slice, block_slice]
            standard_error = np.sqrt(
                np.einsum("lp,pq,lq->l", basis, block_covariance, basis)
            )
            np.testing.assert_allclose(
                band[:, 1].max(), (expected + standard_error).max()
            )
            np.testing.assert_allclose(
                band[:, 1].min(), (expected - standard_error).min()
            )
    finally:
        plt.close(fig)


def test_a_non_canonical_prior_is_refused_before_anything_is_loaded(monkeypatch):
    """The runner writes a sensitivity bundle without the null panel d draws,
    so the figure refuses it up front instead of reading the sigma=3 bundle
    under a sigma=1 panel a."""

    def _must_not_load(*args, **kwargs):
        raise AssertionError("loaded data for a refused prior")

    monkeypatch.setattr(figure04, "load_results", _must_not_load)
    with pytest.raises(ValueError, match="sensitivity bundle"):
        figure04.generate_figure(penalty_sigma=1.0)


def test_panel_c_curves_are_relative_to_each_units_value_at_the_window_start():
    # Two units with the same event-locked bump on different whole-window
    # levels: relative to -200 ms, both start at zero and the mean is the bump.
    lag_time = np.arange(-100, 101) / 500
    bump = np.exp(-0.5 * (lag_time / 0.05) ** 2)
    curves = np.stack([bump - 0.5, bump + 0.2], axis=1)
    modulations = {"swr": curves, "no_swr": curves}
    fig, ax = plt.subplots()

    figure04.plot_population_average(modulations, lag_time, ax)

    mean_line = ax.get_lines()[0]
    np.testing.assert_allclose(mean_line.get_ydata(), bump - bump[0], atol=1e-12)
    assert "-200 ms" in ax.get_ylabel()
    plt.close(fig)
