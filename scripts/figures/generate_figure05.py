#!/usr/bin/env python
"""Generate Figure 5: mPFC units respond to SWRs and to no-SWR non-local events.

The two kinds of event are drawn separately and named as in Figures 2 and 3
("SWR", "No SWR"); the figure's legend gives their colors. Here "SWR" is every
detected SWR (the SWR source's events) and "No SWR" every non-local event that
overlaps none: the two kernel blocks of the joint ``SWR + non_local_no_SWR``
model. Each row asks one thing:

    One unit, then every unit
        a — An example mPFC unit: its fitted modulation by each kind of event
            (band: the pointwise penalized posterior-mode standard error, from
            the joint fit's coefficient covariance) over its firing rate around
            the starts of each kind's events.
        b — Every unit's modulation by each kind, the units in one order (the
            time of the peak of their SWR modulation). The color scale's ends
            say that they also hold the changes beyond them.
    The population
        c — Population modulation by each kind, each unit's curve relative to
            its own value at -200 ms (the reference of the Part 4 timing
            values), reduced unit -> session -> animal: the mean over animals,
            ± SEM over animals.
        d — Left: the share of units whose event term improves held-out
            deviance beyond the matched non-event control's null, bars pooled
            over units (the mpfc summary's ``*_called_fraction`` values) and
            dots one per animal
            (``population_statistics.called_fraction_by_animal``). Right: each
            unit's best model by held-out deviance.

Every fitted quantity in the figure comes from the estimator the bundles were
fit with: a Poisson GLM with a Gaussian prior on the event-basis coefficients,
and a peri-event kernel that returns to baseline at both ends of the ±250 ms
fit window. The population significance is an empirical p-value from repeated
matched-control draws (R = 1,000, with the finite-sample plus-one correction,
so its minimum is 1/(R+1) ≈ 0.001). Panel d's unit share is a descriptive
raw-p preview; population inference requires aggregation through session and
animal.

The figure follows the standards of Figures 1 to 3: the publication style, one
column grid for both rows, letters placed by row and column, and the checks
that no text or panel runs into another (``check_placement``) and that the
saved figure is inside the journal's limits (``check_journal_limits``).

Usage:
    python scripts/figures/generate_figure05.py
    python scripts/figures/generate_figure05.py --output-dir paper/figures/figure05
"""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.pfc import (
    fit_immobility_models,
    make_design_matrices,
    population_statistics,
    prepare_session,
)
from continuum_swr_replay.analysis.pfc.batched_glm import DEFAULT_PENALTY_SIGMA
from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
from continuum_swr_replay.analysis.pfc.fit import model_penalties
from continuum_swr_replay.analysis.pfc.heldout import (
    DEFAULT_CONTROL_DRAWS,
    empirical_pvalue_floor,
)
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA,
    MIN_FIRING_RATE_HZ,
    load_population_data,
)
from continuum_swr_replay.analysis.pfc.modulation import (
    curve_coefficients,
    curve_standard_error,
    in_report_window,
    peri_event_rate,
)
from continuum_swr_replay.intervals import get_spikecount_per_time_bin
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.runs.device import configure_jax_gpu
from continuum_swr_replay.visualization.figure_utilities import save_figure
from continuum_swr_replay.visualization.palettes import COLORS
from continuum_swr_replay.visualization.panel_layout import place_letters
from continuum_swr_replay.visualization.publication_style import (
    SMALL_SIZE,
    check_journal_limits,
    check_placement,
    set_publication_style,
)

figure_utilities.set_figure_defaults(context="paper")
set_publication_style()
warnings.filterwarnings("ignore")

# Inches: saved with 0.1 in margins, about 17 cm (Cell Press full width).
WIDTH = 6.5
HEIGHT = 5.0

# The two kinds of event, named and colored as in Figures 2 and 3: the bundle's
# curve, the example session's event set, the population's curve, the name and
# the color.
KINDS = (
    ("no_SWR", "non_local_no_SWR", "no_swr", "No SWR", COLORS["No SWR"]),
    ("SWR", "SWR", "swr", "SWR", COLORS["SWR"]),
)

# ---------------------------------------------------------------------------
# Panel a — Single-neuron example (overlaid SWR + non-SWR modulation)
# ---------------------------------------------------------------------------


#: Panel a's PSTH bin width, s.
PSTH_BIN_S = 0.010

#: The range panel a always shows: 0.5x to 2x, on the log-modulation scale.
MODULATION_REFERENCE_BOUNDS = (np.log(0.5), np.log(2.0))


def modulation_axis_bounds(drawn):
    """Limits of the modulation axis, on the log scale the curves are drawn on.

    The 0.5x to 2x reference range, widened where a curve or its band leaves it,
    with 5 % of the drawn span as margin so a line on the limit is not half
    clipped. A fixed range would cut off any curve or band that leaves it.

    Parameters
    ----------
    drawn : list of np.ndarray
        Every curve and band edge drawn on the axis (log modulation).

    Returns
    -------
    tuple of float
        ``(low, high)`` in log modulation.
    """
    data_low = min(float(np.min(values)) for values in drawn)
    data_high = max(float(np.max(values)) for values in drawn)
    margin = 0.05 * (data_high - data_low)
    return (
        min(MODULATION_REFERENCE_BOUNDS[0], data_low - margin),
        max(MODULATION_REFERENCE_BOUNDS[1], data_high + margin),
    )


def modulation_axis_ticks(bounds):
    """Fold-change ticks on the modulation axis: each power of two inside it.

    Always 0.5, 1 and 2 (the reference range), plus 0.25, 4, 8 ... when the
    limits were widened.
    """
    tolerance = 1e-9
    low = int(np.ceil(bounds[0] / np.log(2) - tolerance))
    high = int(np.floor(bounds[1] / np.log(2) + tolerance))
    return [2.0**power for power in range(low, high + 1)]


def plot_single_neuron_overlay(
    axes,
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_results,
    event_times,
    model_configs,
):
    """Plot one unit's modulation by each kind of event over its firing rate
    around the starts of each kind's events.

    Sub-row 1 : one modulation-curve axis with a curve per config (the two
                kinds of event on the same axes).
    Sub-row 2 : the firing rate around event starts, one line per kind.

    Each curve is read off its model's basis block exactly as the bundle's
    shipped curves are (:data:`~continuum_swr_replay.analysis.pfc.modulation.CURVE_SOURCES`),
    so the example unit shows the quantity the population panels draw. The
    kinds are named by the figure's legend, so no legend is drawn here.

    Parameters
    ----------
    axes : list of 2 matplotlib.axes.Axes
        [ax_modulation, ax_psth]
    lag_time : np.ndarray
        Time lags in seconds (covers both positive and negative lags).
    spline_basis : np.ndarray, shape (n_lags, n_basis)
        Spline basis one event block of the parameters multiplies.
    neuron_spike_times : np.ndarray
        Spike times for the example unit (seconds).
    glm_results : dict[str, fitted GLM result]
        Full dict of fitted GLMs; only the models the configs' curves are read
        off are plotted. Each needs ``params`` and ``covariance``, the
        penalized posterior covariance of the parameters. The band is
        ``curve +/- sqrt(b(lag)' cov b(lag))`` over the curve's own block, off
        diagonal terms included, so it is the pointwise standard error the
        shipped estimator gives -- not a maximum-likelihood interval drawn
        around a penalized point estimate.
    event_times : dict[str, pd.DataFrame]
        Event interval tables keyed by name (columns: start_time, end_time).
    model_configs : list of dict
        Each entry: ``{"curve": str, "label": str, "color": str,
        "event_key": str}``. ``curve`` is a key into ``CURVE_SOURCES`` (e.g.
        ``"SWR"`` or ``"no_SWR"``, the two blocks of the joint
        ``SWR + non_local_no_SWR`` model), ``event_key`` is a key into
        ``event_times`` (used for the firing rate).

    """
    ax_mod, ax_psth = axes
    lag_bounds = lag_time[0], lag_time[-1]
    lag_ms = 1e3 * lag_time

    # ── Sub-row 1: overlaid modulation curves ──
    # The block is the design's own order (``CURVE_SOURCES``), not the order
    # the model's name lists its classes in.
    n_basis = spline_basis.shape[1]
    drawn = []
    for cfg in model_configs:
        model, coefficients = curve_coefficients(cfg["curve"], n_basis)
        result = glm_results[model]
        modulation, standard_error = curve_standard_error(
            spline_basis,
            np.asarray(result.params)[coefficients],
            np.asarray(result.covariance)[coefficients, coefficients],
        )
        upper = modulation + standard_error
        lower = modulation - standard_error
        drawn += [modulation, upper, lower]

        ax_mod.plot(lag_ms, modulation, color=cfg["color"], label=cfg["label"])
        ax_mod.fill_between(lag_ms, lower, upper, color=cfg["color"], alpha=0.2, lw=0)

    ax_mod.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.axvline(0, color="black", linestyle="--", linewidth=0.5)
    modulation_bounds = modulation_axis_bounds(drawn)
    ax_mod.set_ylim(modulation_bounds)
    modulation_ticks = modulation_axis_ticks(modulation_bounds)
    ax_mod.set_yticks(np.log(modulation_ticks))
    ax_mod.set_yticklabels([f"{tick:g}" for tick in modulation_ticks])
    ax_mod.set_ylabel("GLM\nmodulation")
    ax_mod.tick_params(labelbottom=False)
    ax_mod.set_title(
        "Example mPFC unit (penalized fit, ±1 posterior SE)",
        fontsize=SMALL_SIZE,
        pad=4,
    )

    # ── Sub-row 2: the firing rate around event starts ──
    # The rate in 10-ms lag bins around each event start, per kind of event.
    for cfg in model_configs:
        events = event_times[cfg["event_key"]]
        if len(events) == 0:
            continue
        psth_centers, rate = peri_event_rate(
            neuron_spike_times,
            events["start_time"].to_numpy(),
            lag_bounds,
            bin_s=PSTH_BIN_S,
        )
        ax_psth.plot(1e3 * psth_centers, rate, color=cfg["color"], label=cfg["label"])

    ax_psth.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_psth.set_xlim(lag_ms[0], lag_ms[-1])
    ax_psth.set_xticks((lag_ms[0], 0, lag_ms[-1]))
    ax_psth.set_xlabel("Time from event start (ms)")
    ax_psth.set_ylabel("Firing\nrate (Hz)")


# ---------------------------------------------------------------------------
# Panel b — Population heatmaps (accepts axes)
# ---------------------------------------------------------------------------


#: Panel b's color scale ends at this rate change and its inverse; a larger
#: change is drawn in the end's color, which the scale's end labels say.
MODULATION_COLOR_LIMIT = 2.0


def plot_population_heatmaps(modulations, lag_time, axes):
    """Every unit's modulation by each kind of event, on ``axes`` (SWR, then
    No SWR), the units in one order: the time of the peak of their SWR
    modulation. The SWR panel is first because it sets the order.

    Parameters
    ----------
    modulations : dict[str, np.ndarray], each (n_lag, n_neuron)
        Population modulation curves (log rate), keys ``swr`` and ``no_swr``.
    lag_time : np.ndarray, shape (n_lag,)
        Lags of ``modulations``, in seconds.
    axes : list of 2 matplotlib.axes.Axes
        The SWR panel and the No SWR panel.

    Returns
    -------
    matplotlib.image.AxesImage
        The last image, for the shared color scale.
    """
    peak_times = lag_time[np.argmax(np.abs(modulations["swr"]), axis=0)]
    order = np.argsort(peak_times)
    limit = np.log(MODULATION_COLOR_LIMIT)
    lag_ms = 1e3 * lag_time
    for ax, (_, _, key, name, color) in zip(axes, reversed(KINDS), strict=True):
        image = ax.imshow(
            modulations[key][:, order].T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=[lag_ms[0], lag_ms[-1], len(order), 0],
            interpolation="none",
        )
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        # Half way to each end: the two panels' end ticks would run together.
        ax.set_xticks((0.5 * lag_ms[0], 0, 0.5 * lag_ms[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.set_title(name, fontsize=SMALL_SIZE, color=color, pad=2)
    return image


# ---------------------------------------------------------------------------
# Panel c — Population average (accepts ax)
# ---------------------------------------------------------------------------


def plot_population_average(modulations, pop_df, lag_time, ax):
    """Plot the animal-balanced population modulation ± SEM over animals.

    Each unit's curve is taken relative to its own value at the window's first
    lag (−200 ms), the reference the Part 4 timing values use, so the panel shows
    the event-locked change rather than the whole-window level that low-rate
    units carry. The curves are then reduced unit -> session -> animal
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.animal_balanced_unit_curve`):
    every animal weighs the same however many of the units are its, and the
    band is the standard error over animals. The kinds are named by the
    figure's legend.

    Parameters
    ----------
    modulations : dict[str, np.ndarray], each (n_lag, n_neuron)
        Population modulation curves, keys ``swr`` and ``no_swr``.
    pop_df : pd.DataFrame
        One row per unit, in the order of the columns of ``modulations``, with
        ``nwb_file_name``, ``epoch_name`` and ``animal``.
    lag_time : np.ndarray, shape (n_lag,)
        Lags of ``modulations``, in seconds.
    ax : matplotlib.axes.Axes
        The panel's axes.
    """
    lag_ms = lag_time * 1000
    sessions = (pop_df["nwb_file_name"] + "_" + pop_df["epoch_name"]).to_numpy()
    animals = pop_df["animal"].to_numpy()

    for _, _, name, label, color in KINDS:
        arr = modulations[name] - modulations[name][:1]
        mean, sem = population_statistics.animal_balanced_unit_curve(
            arr, sessions, animals
        )

        ax.plot(lag_ms, mean, color=color, label=label)
        ax.fill_between(lag_ms, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time from event start (ms)")
    ax.set_ylabel(f"Log firing rate change re {lag_ms[0]:.0f} ms\n(mean over animals)")
    ax.set_xlim(lag_ms[0], lag_ms[-1])
    ax.set_xticks((lag_ms[0], 0, lag_ms[-1]))
    ax.set_title(
        f"Mean over {pop_df['animal'].nunique()} animals ± SEM "
        f"({modulations['swr'].shape[1]:,} units)",
        fontsize=SMALL_SIZE,
        pad=4,
    )


# ---------------------------------------------------------------------------
# Panel d — Held-out significance + best model (accepts axes)
# ---------------------------------------------------------------------------

#: Panel d's bars: how each is labeled, which held-out test it reads, and its
#: color. "Non-local" is every non-local event, with or without an SWR. The
#: "beyond control" bar is the engagement comparison -- does a no-SWR event
#: predict held-out spikes beyond a duration-matched non-event baseline drawn
#: from the same bouts?
SIGNIFICANCE_BARS = (
    ("SWR", "SWR", "SWR"),
    ("Non-local", "non_local", "Non-Local"),
    ("No SWR", "non_local_no_SWR", "No SWR"),
    ("Beyond\ncontrol", "engagement", "Combined"),
)


def _label_bars(ax, heights):
    """Write each bar's percent above it, in one row along the top of the axes:
    clear of the bars and of the animals' dots, whose spread would otherwise
    set each label's height."""
    for x, height in enumerate(heights):
        if height > 2:
            ax.annotate(
                f"{height:.0f}%",
                (x, 1.0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, -1),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=SMALL_SIZE,
            )


def plot_significance(
    df, ax, alpha=CALLED_ALPHA, n_control_draws=DEFAULT_CONTROL_DRAWS
):
    """Descriptive share of units whose regressor beats the control null: bars
    pooled over units, dots one per animal.

    A unit counts when its plus-one Monte Carlo p-value is below ``alpha``.
    The denominator is the units actually tested for that class: a unit whose
    session had no events of a class is untested for it, not a negative
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_fractions`).
    An animal's share is the mean of its sessions' shares
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_fraction_by_animal`).
    This is a descriptive implementation preview, not population inference:
    units within a session and animal are not independent replicates.

    The p-value is ``(r + 1) / (R + 1)`` over ``R`` usable draws, so it cannot
    be zero. Every share here inherits that resolution, which is why
    ``n_control_draws`` is drawn on the panel. The dashed line is ``alpha``,
    the share called by chance.

    Parameters
    ----------
    df : pd.DataFrame
        One row per unit: the held-out p-value columns, and the ``animal``,
        ``nwb_file_name`` and ``epoch_name`` the animals' shares group by.
    ax : matplotlib.axes.Axes
    alpha : float
        The raw empirical-p threshold.
    n_control_draws : int
        Matched-control draws in each unit's null.
    """
    fractions = population_statistics.called_fractions(df, alpha)["fraction"]
    ax.axhline(100 * alpha, color="0.5", linewidth=0.6, linestyle="--")
    ax.annotate(
        "chance",
        (1.0, 100 * alpha),
        xycoords=ax.get_yaxis_transform(),
        xytext=(2, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color="0.4",
        fontsize=SMALL_SIZE,
    )
    labels, heights, tops = [], [], []
    for x, (label, name, color) in enumerate(SIGNIFICANCE_BARS):
        height = 100 * fractions[name]
        animals = (
            100
            * population_statistics.called_fraction_by_animal(df, name, alpha)[
                "fraction"
            ].to_numpy()
        )
        ax.bar(x, height, width=0.7, color=COLORS[color], edgecolor="none")
        ax.scatter(
            x + np.linspace(-0.16, 0.16, len(animals)) * (len(animals) > 1),
            animals,
            s=3,
            color="k",
            lw=0,
            zorder=3,
        )
        labels.append(label)
        heights.append(height)
        tops.append(max(height, animals.max()))
    _label_bars(ax, heights)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("% units")
    ax.set_ylim(0, max(100 * alpha * 1.2, max(tops) * 1.25))
    # Say both the raw-p threshold and R on the panel: this is a descriptive
    # preview, and its Monte Carlo resolution is part of what the bars mean.
    # Left-aligned and in short lines: the panel is narrow.
    min_pvalue = empirical_pvalue_floor(n_control_draws)["min_pvalue"]
    ax.set_title(
        f"Raw empirical p < {alpha:g}\n(descriptive)\n"
        f"R={n_control_draws}, min p={min_pvalue:.3f}",
        fontsize=SMALL_SIZE,
        loc="left",
    )


def plot_best_model_selection(df, ax):
    """Distribution of the best model per unit, by held-out deviance.

    Each bar is a share of the units with a best model
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.best_model_fractions`).
    """
    model_order = ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"]
    model_labels = ["Constant", "SWR", "Non-local", "SWR +\nNo SWR"]
    model_colors = ["0.6", COLORS["SWR"], COLORS["Non-Local"], COLORS["Combined"]]

    fractions = population_statistics.best_model_fractions(df, model_order)
    heights = [100 * fractions[m] for m in model_order]

    ax.bar(
        range(len(heights)), heights, width=0.7, color=model_colors, edgecolor="none"
    )
    _label_bars(ax, heights)
    ax.set_xticks(range(len(model_labels)), model_labels, rotation=30, ha="right")
    ax.set_ylabel("% units")
    ax.set_ylim(0, max(heights) * 1.25)
    ax.set_title("Best model\n(held-out deviance)", fontsize=SMALL_SIZE, loc="left")


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------


def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    neuron_ind=27,
    data_dir=None,
    stop_cohort="at_well",
    fitter="jax",
    penalty_sigma=DEFAULT_PENALTY_SIGMA,
):
    """Generate composite Figure 5.

    Parameters
    ----------
    nwb_file_name : str
        NWB file for the single-neuron example.
    epoch_name : str
        Epoch name.
    neuron_ind : int
        mPFC neuron index.
    data_dir : Path
        Directory containing processed data.
    stop_cohort : {"at_well", "all_stops"}
        Population the panels describe. The example neuron is refit here, so it
        must be cut the same way the population bundle in ``data_dir`` was; the
        suffixed ``_allstops`` bundle is read for ``all_stops``.
    fitter : {"jax"}
        Backend the example neuron's GLMs are fit with -- the batched one the
        population bundle was fit with, and the only one with a penalized path.
    penalty_sigma : float
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale. Only the canonical prior is accepted: the runner writes a
        sensitivity bundle with ``n_control_draws = 0``, so it has no
        matched-control null for panel d.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``penalty_sigma`` is not the canonical prior, before anything is
        loaded or fit: that bundle has no held-out p-values for panel d.
    """
    if data_dir is None:
        data_dir = processed_data_dir()
    if float(penalty_sigma) != DEFAULT_PENALTY_SIGMA:
        raise ValueError(
            f"penalty_sigma={penalty_sigma} names a sensitivity bundle, which "
            "run_pfc_analysis_bandit.py writes without "
            "the matched-control null (n_control_draws=0), so panel d has no "
            f"held-out p-values to draw. Figure 5 is drawn at the canonical "
            f"prior ({DEFAULT_PENALTY_SIGMA})."
        )
    # --- Load single-neuron data ---
    # The example neuron is refit here, so its session is prepared exactly as
    # the runner prepares the population's (``analysis.pfc.session``): the
    # aligned decode and data, the fit mask and excluded time, and the canonical
    # event sets with the matched non-event control drawn on the session's seed.
    print(f"Loading data for {nwb_file_name}, {epoch_name}...")
    logger = logging.getLogger(f"figure05.{nwb_file_name}.{epoch_name}")
    cohort_stops = None
    if stop_cohort != "all_stops":
        cohort_stops, cohort_report = load_session_stop_cohort(
            nwb_file_name, epoch_name, cohort=stop_cohort
        )
        print(
            f"Stop cohort {stop_cohort}: "
            f"{cohort_report['n_stops_after']}/{cohort_report['n_stops_before']} stops"
        )
    prepared = prepare_session(
        nwb_file_name,
        epoch_name,
        stop_cohort=stop_cohort,
        cohort_stops=cohort_stops,
        sampling_frequency=SAMPLING_FREQUENCY,
        non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
        logger=logger,
    )
    data = prepared.data
    time = prepared.time
    lags = prepared.fit_lags
    fit_mask = prepared.fit_mask
    bin_counts = prepared.bin_counts
    event_times = prepared.event_times
    event_counts = prepared.event_counts
    print(
        f"Fit bins: {bin_counts['n_bins_fit']}/{bin_counts['n_bins_immobile']} "
        "immobile bins in cohort"
    )
    print(
        f"Events in cohort: {event_counts['n_nonlocal_after']}"
        f"/{event_counts['n_nonlocal_before']} non-local, "
        f"{event_counts['n_ripple_after']}/{event_counts['n_ripple_before']} SWRs; "
        f"matched non-event controls: {len(event_times['matched_control'])}"
        f"/{len(event_times['non_local'])} placed"
    )

    immobility_design_matrices, spline_basis, lag_time = make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=SAMPLING_FREQUENCY,
        lags=lags,
    )

    neuron_spike_times = data["spike_times"]["mPFC"][neuron_ind]
    neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)

    print(f"Fitting GLM models (fitter: {fitter}, prior sigma: {penalty_sigma})...")
    glm_results, _ = fit_immobility_models(
        fit_mask,
        immobility_design_matrices,
        neuron_spike_count,
        fitter=fitter,
        penalties=model_penalties(immobility_design_matrices, penalty_sigma),
    )

    # The models are fit over +/-250 ms and reported over +/-200 ms, as the
    # bundle's curves are: the spline's boundary behavior stays outside the
    # window the panel draws.
    report_mask = in_report_window(lag_time)
    spline_basis = spline_basis[report_mask]
    lag_time = lag_time[report_mask]

    # --- Load population data ---
    print(
        f"Loading population data (prior sigma {penalty_sigma}, "
        f"R={DEFAULT_CONTROL_DRAWS} control draws)..."
    )
    pop_df, modulations, pop_lag_time = load_population_data(
        data_dir,
        # No firing-rate cutoff, as Part 4's values (the lead, 2026-10-29):
        # every unit with a usable fit.
        min_firing_rate=MIN_FIRING_RATE_HZ,
        artifact_suffix=stop_cohort_suffix(stop_cohort),
    )

    return assemble_figure(
        lag_time=lag_time,
        spline_basis=spline_basis,
        neuron_spike_times=neuron_spike_times,
        glm_results=glm_results,
        event_times=event_times,
        pop_df=pop_df,
        modulations=modulations,
        pop_lag_time=pop_lag_time,
    )


# The panel letters as they sit in the figure: the letters of a row share a
# height, the letters of a column a left edge.
LETTER_ROWS = ("ab", "cd")
LETTER_COLUMNS = ("ac", "bd")


def assemble_figure(
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_results,
    event_times,
    pop_df,
    modulations,
    pop_lag_time,
):
    """Lay out and draw the four panels from already-loaded inputs.

    Split from ``generate_figure`` so the layout can be drawn, and checked for
    overlapping text, without loading a session or fitting a unit.

    Parameters
    ----------
    lag_time : np.ndarray, shape (n_report_lags,)
        Reported lags of the example unit's curves, in seconds.
    spline_basis : np.ndarray, shape (n_report_lags, n_basis)
        Spline basis one event block of the parameters multiplies.
    neuron_spike_times : np.ndarray
        Spike times of the example unit (seconds).
    glm_results : dict[str, fitted GLM result]
        The example unit's fits; the two kernel blocks of
        ``SWR + non_local_no_SWR`` are drawn, with its ``covariance``.
    event_times : dict[str, pd.DataFrame]
        Event interval tables for the example unit's firing rate.
    pop_df : pd.DataFrame
        One row per population unit (held-out p-values, best model, and the
        ``nwb_file_name``, ``epoch_name`` and ``animal`` panels c and d group
        by), in the order of the columns of ``modulations``.
    modulations : dict[str, np.ndarray], each (n_lag, n_neuron)
        Population modulation curves, keys ``swr`` and ``no_swr``.
    pop_lag_time : np.ndarray, shape (n_lag,)
        Lags of ``modulations``, in seconds.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If a text runs into another or a panel's drawing into another panel's
        (``check_placement``).
    """
    print("Assembling composite figure...")
    fig = plt.figure(figsize=(WIDTH, HEIGHT), layout="none")
    # One grid for both rows, so the panels of a column share a left edge:
    # a over c, b over d.
    grid = GridSpec(
        2, 4, figure=fig, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.75,
        left=0.085, right=0.935, top=0.9, bottom=0.12,
    )  # fmt: skip

    # ── Panel a: one unit, its modulation over its firing rate ──
    a_grid = grid[0, :2].subgridspec(2, 1, hspace=0.12, height_ratios=[1.0, 0.8])
    ax_a_mod = fig.add_subplot(a_grid[0])
    ax_a_psth = fig.add_subplot(a_grid[1], sharex=ax_a_mod)
    plot_single_neuron_overlay(
        axes=[ax_a_mod, ax_a_psth],
        lag_time=lag_time,
        spline_basis=spline_basis,
        neuron_spike_times=neuron_spike_times,
        glm_results=glm_results,
        event_times=event_times,
        model_configs=[
            {"curve": curve, "label": name, "color": color, "event_key": event_key}
            for curve, event_key, _, name, color in KINDS
        ],
    )

    # ── Panel b: every unit, each kind, one order, one color scale ──
    b_grid = grid[0, 2:].subgridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.14)
    ax_b = [fig.add_subplot(b_grid[0])]
    ax_b.append(fig.add_subplot(b_grid[1], sharey=ax_b[0]))
    image = plot_population_heatmaps(modulations, pop_lag_time, ax_b)
    ax_b[0].set_ylabel("mPFC unit, by time of SWR peak")
    ax_b[1].tick_params(labelleft=False)
    # One x label for both: centered under the pair, as a label of the first.
    first, second = (ax.get_position() for ax in ax_b)
    ax_b[0].set_xlabel("Time from event start (ms)")
    ax_b[0].xaxis.label.set_x((0.5 * (first.x0 + second.x1) - first.x0) / first.width)
    limit = np.log(MODULATION_COLOR_LIMIT)
    ax_b_scale = fig.add_subplot(b_grid[2])
    scale = fig.colorbar(image, cax=ax_b_scale, ticks=[-limit, 0, limit])
    scale.ax.set_yticklabels(
        [f"≤{1 / MODULATION_COLOR_LIMIT:g}×", "1×", f"≥{MODULATION_COLOR_LIMIT:g}×"]
    )
    scale.ax.tick_params(length=2)
    scale.outline.set_linewidth(0.5)
    scale.set_label("Firing rate change", labelpad=1)

    # ── Panel c: the population's modulation, the mean over animals ──
    ax_c = fig.add_subplot(grid[1, :2])
    plot_population_average(modulations, pop_df, pop_lag_time, ax_c)

    # ── Panel d: the share of units called, and each unit's best model ──
    ax_d = [fig.add_subplot(grid[1, 2]), fig.add_subplot(grid[1, 3])]
    plot_significance(pop_df, ax_d[0])
    plot_best_model_selection(pop_df, ax_d[1])

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for *_, color in KINDS]
    fig.legend(
        handles, [name for *_, name, _ in KINDS], frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8, columnspacing=1.5,
    )  # fmt: skip

    fig.draw_without_rendering()
    panels_by_letter = {
        "a": [ax_a_mod, ax_a_psth], "b": [*ax_b, ax_b_scale], "c": [ax_c], "d": ax_d,
    }  # fmt: skip
    place_letters(
        fig, panels_by_letter, LETTER_ROWS, LETTER_COLUMNS, above=0.02,
        left_of=0.004, ceiling=0.985,
    )  # fmt: skip
    check_placement(fig, panels_by_letter)
    return fig


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure05"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument("--neuron-ind", type=int, default=27)
    parser.add_argument("--data-dir", type=Path, default=processed_data_dir())
    parser.add_argument(
        "--stop-cohort",
        type=str,
        default="at_well",
        choices=["at_well", "all_stops"],
        help=(
            "Population the figure describes (default: at_well, the canonical "
            "bundle). 'all_stops' refits the example neuron without the stop "
            "restriction and reads the '_allstops' population bundle."
        ),
    )
    parser.add_argument(
        "--fitter",
        type=str,
        default="jax",
        choices=["jax"],
        help=(
            "Backend the example neuron's GLMs are fit with (default and only "
            "choice: jax, the one the population bundle was fit with and the "
            "only one with a penalized path)."
        ),
    )
    parser.add_argument(
        "--penalty-sigma",
        type=float,
        default=DEFAULT_PENALTY_SIGMA,
        help=(
            "Prior standard deviation of an event-basis coefficient, on the "
            f"log-rate scale (default: {DEFAULT_PENALTY_SIGMA}). The example "
            "neuron is refit under it and the population bundle fit under it "
            "is read. Only the canonical prior's bundle carries the null panel "
            "d needs, so another value is refused."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Fit on this GPU instead of the emptiest free one.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit the batched fitter to run on CPU when no GPU is free.",
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Before JAX initializes a backend (which is later than the import, and
    # checked afterward): the panel's fit runs on the same backend the
    # population bundle was fit with, on a card nobody else is using.
    if args.fitter == "jax":
        record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
        print(f"JAX device: {record}")

    fig = generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        neuron_ind=args.neuron_ind,
        data_dir=args.data_dir,
        stop_cohort=args.stop_cohort,
        fitter=args.fitter,
        penalty_sigma=args.penalty_sigma,
    )

    check_journal_limits(fig)
    print(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure05", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
