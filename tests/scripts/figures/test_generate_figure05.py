"""Publication-facing inference labels for Figure 5."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.heldout import DEFAULT_CONTROL_DRAWS
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA,
    HELDOUT_P_COLUMNS,
)
from continuum_swr_replay.visualization import figure_utilities
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.publication_style import (
    check_journal_limits,
    set_publication_style,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "generate_figure05",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "figures"
    / "generate_figure05.py",
)
figure05 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure05)


def _one_session(frame):
    """``frame``'s units, all in one session of one animal."""
    return frame.assign(animal="a", nwb_file_name="a20200101_.nwb", epoch_name="02_r1")


def test_panel_d_uses_the_canonical_control_draw_count():
    assert figure05.DEFAULT_CONTROL_DRAWS == DEFAULT_CONTROL_DRAWS == 1_000

    frame = _one_session(
        pd.DataFrame({column: [0.01, 0.5] for column in HELDOUT_P_COLUMNS.values()})
    )
    fig, ax = plt.subplots()
    try:
        figure05.plot_significance(frame, ax)
        assert "R=1000, min p=0.001" in ax.get_title(loc="left")
    finally:
        plt.close(fig)


def test_panel_d_defaults_to_the_summarys_called_threshold():
    # The same object, not an equal literal: the panel and the summary's called
    # fractions read one threshold.
    default = inspect.signature(figure05.plot_significance).parameters["alpha"].default
    assert default is CALLED_ALPHA


def test_panel_d_draws_the_threshold_and_the_draw_count_it_is_given():
    frame = _one_session(
        pd.DataFrame({column: [0.01, 0.5] for column in HELDOUT_P_COLUMNS.values()})
    )
    fig, ax = plt.subplots()
    try:
        figure05.plot_significance(frame, ax, alpha=0.6, n_control_draws=99)
        # at 0.6 both units are called, at the default 0.05 one of the two
        assert [bar.get_height() for bar in ax.patches] == [100.0] * 4
        assert ax.get_title(loc="left") == (
            "Raw empirical p < 0.6\n(descriptive)\nR=99, min p=0.010"
        )
        # the dashed line is the threshold, as a percent, and it is named
        assert list(ax.lines[0].get_ydata()) == pytest.approx([60.0, 60.0])
        assert ax.texts[0].get_text() == "chance"
    finally:
        plt.close(fig)


def test_panel_d_bars_are_the_called_and_best_model_shares_as_percents():
    """Four units. SWR: one untested, two of the three tested called.
    Engagement: three of four called. The other tests: one of four. Best
    model: one constant, two SWR, and one unit with none, which is left out."""
    pvalues = {name: [0.01, 0.5, 0.5, 0.5] for name in HELDOUT_P_COLUMNS}
    pvalues["SWR"] = [0.01, 0.01, 0.5, np.nan]
    pvalues["engagement"] = [0.01, 0.01, 0.01, 0.5]
    frame = _one_session(
        pd.DataFrame(
            {HELDOUT_P_COLUMNS[name]: values for name, values in pvalues.items()}
        )
    )
    frame["min_heldout_model"] = ["constant", "SWR", "SWR", None]

    fig, (called_ax, best_ax) = plt.subplots(1, 2)
    try:
        figure05.plot_significance(frame, called_ax)
        figure05.plot_best_model_selection(frame, best_ax)

        assert [t.get_text() for t in called_ax.get_xticklabels()] == [
            "SWR",
            "Non-local",
            "No SWR",
            "Beyond\ncontrol",
        ]
        assert [bar.get_height() for bar in called_ax.patches] == pytest.approx(
            [200 / 3, 25.0, 25.0, 75.0]
        )
        assert [t.get_text() for t in called_ax.texts] == [
            "chance",
            "67%",
            "25%",
            "25%",
            "75%",
        ]
        # one animal with one session: its dot is on the bar's top
        dots = [c.get_offsets()[:, 1].tolist() for c in called_ax.collections]
        assert dots == [pytest.approx([h]) for h in (200 / 3, 25.0, 25.0, 75.0)]
        assert [t.get_text() for t in best_ax.get_xticklabels()] == [
            "Constant",
            "SWR",
            "Non-local",
            "SWR +\nNo SWR",
        ]
        assert [bar.get_height() for bar in best_ax.patches] == pytest.approx(
            [100 / 3, 200 / 3, 0.0, 0.0]
        )
        assert [t.get_text() for t in best_ax.texts] == ["33%", "67%"]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Layout: drawn at the figure's real size, in the production style
# ---------------------------------------------------------------------------

_BASIS_CENTERS = np.linspace(-0.2, 0.2, 6)


def _synthetic_inputs(non_swr_peak=1.6, n_units=150):
    """Inputs of ``assemble_figure`` with the shapes the real ones have.

    The No SWR curve is built to rise above 2x (log 2) late in the window, as
    the example unit's does. Seeded, so a failure repeats.
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
    # Panels c and d group the units by session and animal: five animals of
    # two sessions each.
    session = np.arange(n_units) % 10
    pop_df["animal"] = [f"animal{s // 2}" for s in session]
    pop_df["nwb_file_name"] = [f"animal{s // 2}2020010{s % 2}_.nwb" for s in session]
    pop_df["epoch_name"] = "02_r1"
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

    ``set_figure_defaults("paper")`` resets rcParams and ``set_publication_style``
    sets the paper figures' sizes and turns the layout engine off. Inside
    ``rc_context`` both are undone on exit, and a layout test cannot pass only
    because another test's import left a style on.
    """
    with matplotlib.rc_context():
        figure_utilities.set_figure_defaults(context="paper")
        set_publication_style()
        # The style asks for Arial first and falls back to DejaVu Sans, so the
        # drawn text depends on the host's fonts. DejaVu Sans ships with
        # matplotlib and is the wider of the two: a layout that clears it clears
        # Arial, and the test no longer depends on what is installed.
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        yield


@pytest.fixture
def drawn_figure(paper_style):
    """The assembled Figure 5 on synthetic inputs, drawn once at real size.

    ``assemble_figure`` refuses a figure in which a text runs into another or a
    panel into another (``check_placement``), so the figure existing is that
    check passing.
    """
    fig = figure05.assemble_figure(**_synthetic_inputs())
    fig.canvas.draw()
    try:
        yield fig
    finally:
        plt.close(fig)


def _axis_with_ylabel(fig, text):
    (ax,) = [ax for ax in fig.axes if ax.get_ylabel() == text]
    return ax


def _panel_letter(fig, letter):
    (text,) = [text for text in fig.texts if text.get_text() == letter]
    return text


def test_the_assembled_figure_is_inside_the_journals_limits(drawn_figure):
    """Text of 6 to 8 pt, lines of 0.5 to 1.5 pt, and a saved size within a
    page's width: the limits Figures 1 to 3 are held to."""
    check_journal_limits(drawn_figure)


def test_the_panel_letters_sit_by_row_and_by_column(drawn_figure):
    """The letters of a row share a height and the letters of a column a left
    edge, as in Figures 2 and 3."""
    position = {
        letter: _panel_letter(drawn_figure, letter).get_position() for letter in "abcd"
    }
    assert position["a"][1] == position["b"][1] > position["c"][1] == position["d"][1]
    assert position["a"][0] == position["c"][0] < position["b"][0] == position["d"][0]


def test_the_figure_legend_names_the_two_kinds_in_their_colors(drawn_figure):
    """One legend for the figure, the kinds named as in Figures 2 and 3."""
    (legend,) = drawn_figure.legends
    names = [text.get_text() for text in legend.get_texts()]
    colors = [handle.get_facecolor() for handle in legend.legend_handles]
    assert names == ["No SWR", "SWR"]
    assert [matplotlib.colors.to_hex(c) for c in colors] == [
        COLORS["No SWR"],
        COLORS["SWR"],
    ]
    assert all(ax.get_legend() is None for ax in drawn_figure.axes)


def test_panel_b_draws_the_swr_panel_first_and_marks_the_scales_ends(drawn_figure):
    """The SWR panel sets the units' order, so it is first; a change beyond the
    color scale is drawn in its end color, which the end labels say."""
    titles = [ax.get_title() for ax in drawn_figure.axes if ax.images]
    assert titles == ["SWR", "No SWR"]
    scale = _axis_with_ylabel(drawn_figure, "Firing rate change")
    assert [t.get_text() for t in scale.get_yticklabels()] == ["≤0.5×", "1×", "≥2×"]


def _draw_overlay(inputs):
    """Panel a drawn on its own axes, for the limits and legend of the curves."""
    fig, (ax_mod, ax_psth) = plt.subplots(2, 1, sharex=True)
    configs = [
        {"curve": "SWR", "label": "SWR", "color": "C1", "event_key": "SWR"},
        {
            "curve": "no_SWR",
            "label": "No SWR",
            "color": "C2",
            "event_key": "non_local_no_SWR",
        },
    ]
    figure05.plot_single_neuron_overlay(
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
    """Each tick sits at the log of the fold change it is labeled with, the
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


def test_panel_a_draws_no_legend_of_its_own(paper_style):
    """The figure's legend names the kinds; one inside panel a covered a curve."""
    fig, ax_mod = _draw_overlay(_synthetic_inputs())
    try:
        assert ax_mod.get_legend() is None
    finally:
        plt.close(fig)


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

    monkeypatch.setattr(figure05, "prepare_session", _must_not_load)
    with pytest.raises(ValueError, match="sensitivity bundle"):
        figure05.generate_figure(penalty_sigma=1.0)


def test_panel_c_curves_are_relative_to_each_units_value_at_the_window_start():
    # Two units with the same event-locked bump on different whole-window
    # levels: relative to -200 ms, both start at zero and the mean is the bump.
    lag_time = np.arange(-100, 101) / 500
    bump = np.exp(-0.5 * (lag_time / 0.05) ** 2)
    curves = np.stack([bump - 0.5, bump + 0.2], axis=1)
    modulations = {"swr": curves, "no_swr": curves}
    pop_df = pd.DataFrame(
        {
            "nwb_file_name": ["a20200101_.nwb", "b20200101_.nwb"],
            "epoch_name": ["02_r1", "02_r1"],
            "animal": ["a", "b"],
        }
    )
    fig, ax = plt.subplots()

    figure05.plot_population_average(modulations, pop_df, lag_time, ax)

    mean_line = ax.get_lines()[0]
    np.testing.assert_allclose(mean_line.get_ydata(), bump - bump[0], atol=1e-12)
    assert "-200 ms" in ax.get_ylabel()
    plt.close(fig)


def test_panel_c_draws_the_mean_over_animals_with_the_sem_over_animals():
    # Animal a holds three units in two sessions, animal b one unit. Each
    # unit's curve is flat after -200 ms at its own level.
    lag_time = np.array([-0.2, 0.0, 0.2])
    levels = np.array([1.0, 3.0, 5.0, 8.0])  # a1, a1, a2, b1
    curves = np.outer([0.0, 1.0, 1.0], levels)
    modulations = {"swr": curves, "no_swr": curves}
    pop_df = pd.DataFrame(
        {
            "nwb_file_name": ["a20200101_.nwb"] * 2
            + ["a20200102_.nwb", "b20200101_.nwb"],
            "epoch_name": ["02_r1"] * 4,
            "animal": ["a", "a", "a", "b"],
        }
    )
    fig, ax = plt.subplots()

    figure05.plot_population_average(modulations, pop_df, lag_time, ax)

    # Session a1 is 2, a2 is 5, so animal a is 3.5 and b is 8: the mean over
    # animals is 5.75 (the unit mean would be 4.25), the SEM |8 - 3.5| / 2.
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [0.0, 5.75, 5.75])
    (band, _) = ax.collections
    upper = band.get_paths()[0].vertices[:, 1].max()
    assert upper == pytest.approx(5.75 + 2.25)
    assert "mean over animals" in ax.get_ylabel()
    assert ax.get_legend() is None
    assert ax.get_title() == "Mean over 2 animals ± SEM (4 units)"
    plt.close(fig)
