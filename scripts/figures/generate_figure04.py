#!/usr/bin/env python
"""Generate Figure 4: PFC neurons respond to replay regardless of SWR detection.

Composite figure with 4 panels in 3 rows:

    Row 1 (panel a):
        Single-neuron example — overlaid GLM modulation curves and PSTHs
        for SWR and non-SWR replay events on the same mPFC neuron. Shows
        that the neuron responds to both event types with similar shape.
        The curves are the two kernel blocks of the joint
        ``SWR + non_local_no_SWR`` model, the same quantity panels b and c
        draw for the population.

    Row 2 (panel b):
        Population heatmaps — SWR and non-SWR replay modulation with the
        same neuron sort order, showing parallel response structure
        across the whole population.

    Row 3 (panels c, d):
        c: Population average modulation curves (mean ± SEM) for SWR vs
           non-SWR replay, each unit's curve relative to its own value at
           −200 ms (the reference of the Part 4 timing values).
        d: Consolidated statistics — fraction of neurons whose event
           regressor improves held-out deviance beyond the matched
           non-event control's null, and the distribution of the best
           model by held-out deviance across the population.

Every fitted quantity in the figure comes from the same estimator the
bundles were fit with: a Poisson GLM with a Gaussian prior on the
event-basis coefficients, and a peri-event kernel that returns to
baseline at both ends of the ±250 ms fit window. The example neuron's
band is the pointwise penalised posterior-mode standard error of each
curve, from the joint fit's coefficient covariance, and the population
significance is an empirical p-value from repeated matched-control
draws (R = 1,000, with the finite-sample plus-one correction, so its minimum is
1/(R+1) ≈ 0.001). Panel d's unit fraction is a descriptive raw-p preview;
population inference requires aggregation through session and animal.

April 2026: the figure was tightened from 6 panels to 4. The previous
standalone single-neuron model selection was dropped (it duplicated the
population one), and the two side-by-side single-neuron columns were
collapsed into one overlay panel.

Usage:
    python scripts/figures/generate_figure04.py
    python scripts/figures/generate_figure04.py --output-dir paper/figures/figure04
"""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    excluded_intervals,
    read_artifact_event_drops,
)
from continuum_swr_replay.analysis.events.cohort_selection import stop_cohort_suffix
from continuum_swr_replay.analysis.events.swr_source import (
    ripple_power_inputs,
    swr_epoch_id,
)
from continuum_swr_replay.analysis.hippocampus import load_results
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    align_results_to_position,
)
from continuum_swr_replay.analysis.pfc import (
    build_session_event_sets,
    build_session_fit_mask,
    fit_immobility_models,
    make_design_matrices,
    make_matched_control_times,
    matched_control_seed,
    population_statistics,
)
from continuum_swr_replay.analysis.pfc.batched_glm import DEFAULT_PENALTY_SIGMA
from continuum_swr_replay.analysis.pfc.canonical_events import (
    load_session_canonical_events,
    load_session_swr_intervals,
    session_data_over_time_path,
)
from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
from continuum_swr_replay.analysis.pfc.design import symmetric_lags
from continuum_swr_replay.analysis.pfc.event_sets import DEFAULT_BASIS_HALF_WINDOW_S
from continuum_swr_replay.analysis.pfc.fit import model_penalties
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.heldout import DEFAULT_CONTROL_DRAWS
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.analysis.pfc.matched_control import artifact_dropped_events
from continuum_swr_replay.analysis.pfc.modulation import (
    REPORT_HALF_WINDOW_S,
    curve_coefficients,
)
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.intervals import get_spikecount_per_time_bin
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.reported.mpfc import MIN_FIRING_RATE_HZ
from continuum_swr_replay.visualization.palettes import COLORS

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Panel a — Single-neuron example (overlaid SWR + non-SWR modulation)
# ---------------------------------------------------------------------------


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
    """Plot overlaid GLM modulation curves for one neuron + per-event PSTHs.

    Sub-row 1 : one modulation-curve axis with a curve per config
                (typically SWR and non-SWR replay on the same axes).
    Sub-row 2 : overlaid PSTHs — one line per event type.

    Each curve is read off its model's basis block exactly as the bundle's
    shipped curves are (:data:`~continuum_swr_replay.analysis.pfc.modulation.CURVE_SOURCES`),
    so the example neuron shows the quantity the population panels draw.

    Parameters
    ----------
    axes : list of 2 matplotlib.axes.Axes
        [ax_modulation, ax_psth]
    lag_time : np.ndarray
        Time lags in seconds (covers both positive and negative lags).
    spline_basis : np.ndarray, shape (n_lags, n_basis)
        Spline basis one event block of the parameters multiplies.
    neuron_spike_times : np.ndarray
        Spike times for the example neuron (seconds).
    glm_results : dict[str, fitted GLM result]
        Full dict of fitted GLMs; only the models the configs' curves are read
        off are plotted. Each needs ``params`` and ``covariance``, the
        penalised posterior covariance of the parameters. The band is
        ``curve +/- sqrt(b(lag)' cov b(lag))`` over the curve's own block, off
        diagonal terms included, so it is the pointwise standard error the
        shipped estimator gives -- not a maximum-likelihood interval drawn
        around a penalised point estimate.
    event_times : dict[str, pd.DataFrame]
        Event interval tables keyed by name (columns: start_time, end_time).
    model_configs : list of dict
        Each entry: ``{"curve": str, "label": str, "color": str,
        "event_key": str}``. ``curve`` is a key into ``CURVE_SOURCES`` (e.g.
        ``"SWR"`` or ``"no_SWR"``, the two blocks of the joint
        ``SWR + non_local_no_SWR`` model), ``event_key`` is a key into
        ``event_times`` (used for the PSTH).

    """
    ax_mod, ax_psth = axes
    lag_bounds = lag_time[0], lag_time[-1]

    # ── Sub-row 1: overlaid modulation curves ──
    # The block is the design's own order (``CURVE_SOURCES``), not the order
    # the model's name lists its classes in.
    n_basis = spline_basis.shape[1]
    drawn = []
    for cfg in model_configs:
        model, coefficients = curve_coefficients(cfg["curve"], n_basis)
        result = glm_results[model]
        params = np.asarray(result.params)[coefficients]
        covariance = np.asarray(result.covariance)[coefficients, coefficients]
        modulation = spline_basis @ params
        variance = np.einsum("lp,pq,lq->l", spline_basis, covariance, spline_basis)
        standard_error = np.sqrt(np.clip(variance, 0.0, None))
        upper = modulation + standard_error
        lower = modulation - standard_error
        drawn += [modulation, upper, lower]

        ax_mod.plot(
            lag_time,
            modulation,
            color=cfg["color"],
            linewidth=1.2,
            label=cfg["label"],
        )
        ax_mod.fill_between(
            lag_time,
            lower,
            upper,
            color=cfg["color"],
            alpha=0.2,
        )

    ax_mod.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.axvline(0, color="black", linestyle="--", linewidth=0.5)
    modulation_bounds = modulation_axis_bounds(drawn)
    ax_mod.set_ylim(modulation_bounds)
    ax_mod.set_xticks((lag_bounds[0], 0, lag_bounds[1]))
    modulation_ticks = modulation_axis_ticks(modulation_bounds)
    ax_mod.set_yticks(np.log(modulation_ticks))
    ax_mod.set_yticklabels([f"{tick:g}" for tick in modulation_ticks])
    ax_mod.set_ylabel("GLM\nmodulation")
    # In the empty margin to the right of the axes: inside them it covered the
    # non-SWR replay curve and its band. Left out of the layout so the axes keep
    # the width they had.
    legend = ax_mod.legend(
        frameon=False,
        fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    legend.set_in_layout(False)
    ax_mod.tick_params(labelbottom=False)
    ax_mod.set_title(
        "Example mPFC neuron (penalised fit, ±1 posterior SE)", fontsize=8, pad=4
    )

    # ── Sub-row 2: overlaid PSTHs ──
    # Cheap manual PSTH: compute spike-count density in lag bins around
    # each event start for each event type.
    psth_bin_ms = 10.0
    psth_edges = np.arange(
        lag_bounds[0], lag_bounds[1] + psth_bin_ms / 1000.0, psth_bin_ms / 1000.0
    )
    psth_centers = 0.5 * (psth_edges[:-1] + psth_edges[1:])

    for cfg in model_configs:
        events = event_times[cfg["event_key"]]
        if len(events) == 0:
            continue
        starts = events["start_time"].to_numpy()
        rel = []
        for t0 in starts:
            in_window = (neuron_spike_times >= t0 + lag_bounds[0]) & (
                neuron_spike_times <= t0 + lag_bounds[1]
            )
            rel.append(neuron_spike_times[in_window] - t0)
        if rel:
            all_rel = np.concatenate(rel)
        else:
            all_rel = np.array([])
        counts, _ = np.histogram(all_rel, bins=psth_edges)
        rate = counts / (len(starts) * (psth_bin_ms / 1000.0))
        ax_psth.plot(
            psth_centers,
            rate,
            color=cfg["color"],
            linewidth=1.2,
            label=cfg["label"],
        )

    ax_psth.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_psth.set_xlim(lag_bounds)
    ax_psth.set_xlabel("Time from event (s)")
    ax_psth.set_ylabel("Firing\nrate (Hz)")


# ---------------------------------------------------------------------------
# Panel b — Population heatmaps (accepts axes)
# ---------------------------------------------------------------------------


def plot_population_heatmaps(swr_mod, no_swr_mod, lag_time, axes):
    """Plot side-by-side heatmaps on provided (ax1, ax2) axes."""
    ax1, ax2 = axes
    peak_times = lag_time[np.argmax(np.abs(swr_mod), axis=0)]
    sort_idx = np.argsort(peak_times)

    swr_sorted = swr_mod[:, sort_idx]
    no_swr_sorted = no_swr_mod[:, sort_idx]

    vmin, vmax = -np.log(2), np.log(2)
    lag_ms = lag_time * 1000

    im1 = ax1.imshow(
        swr_sorted.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax1.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Time from SWR (ms)")
    ax1.set_ylabel("Neuron")
    ax1.set_title("SWR modulation")

    ax2.imshow(
        no_swr_sorted.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], no_swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax2.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Time from replay (ms)")
    ax2.set_title("Non-SWR replay modulation")

    return im1


# ---------------------------------------------------------------------------
# Panel c — Population average (accepts ax)
# ---------------------------------------------------------------------------


def plot_population_average(modulations, lag_time, ax):
    """Plot population mean ± SEM modulation on provided ax.

    Each unit's curve is taken relative to its own value at the window's first
    lag (−200 ms), the reference the Part 4 timing values use, so the panel shows
    the event-locked change rather than the whole-window level that low-rate
    units carry.
    """
    lag_ms = lag_time * 1000

    for name, color, label in [
        ("swr", COLORS["SWR"], "SWR"),
        ("no_swr", COLORS["Non-Local"], "Non-SWR replay"),
    ]:
        arr = modulations[name] - modulations[name][:1]
        mean = np.nanmean(arr, axis=1)
        sem = np.nanstd(arr, axis=1) / np.sqrt(arr.shape[1])

        ax.plot(lag_ms, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(lag_ms, mean - sem, mean + sem, color=color, alpha=0.2)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time from event (ms)")
    ax.set_ylabel(f"Log firing rate change\nre {lag_ms[0]:.0f} ms")
    ax.set_xlim(lag_ms[0], lag_ms[-1])
    ax.legend(frameon=False)
    ax.set_title(f"Population (n={modulations['swr'].shape[1]:,})")


# ---------------------------------------------------------------------------
# Panel d — Held-out significance + best model (accepts axes)
# ---------------------------------------------------------------------------

#: Panel d's bars: which held-out test each one reads, and how it is labelled.
#: The "beyond control" bar is the engagement comparison -- does non-SWR replay
#: predict held-out spikes beyond a duration-matched non-event baseline drawn
#: from the same bouts?
SIGNIFICANCE_BARS = (
    ("SWR", "SWR", "SWR"),
    ("Replay", "non_local", "Non-Local"),
    ("no-SWR", "non_local_no_SWR", "Non-Local"),
    ("Beyond\ncontrol", "engagement", "Combined"),
)


def plot_significance(df, ax, alpha=0.05, n_control_draws=DEFAULT_CONTROL_DRAWS):
    """Descriptive fraction of units whose regressor beats the control null.

    A unit counts when its plus-one Monte Carlo p-value is below ``alpha``.
    The denominator is the units actually tested for that class: a unit whose
    session had no events of a class is untested for it, not a negative
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_fractions`).
    This is a descriptive implementation preview, not population inference:
    units within a session and animal are not independent replicates.

    The p-value is ``(r + 1) / (R + 1)`` over ``R`` usable draws, so it cannot
    be zero. Every fraction here inherits that resolution, which is why
    ``n_control_draws`` is drawn on the panel.
    """
    fractions = population_statistics.called_fractions(df, alpha)["fraction"]
    labels, fracs, colors_list = [], [], []
    for label, name, color in SIGNIFICANCE_BARS:
        labels.append(label)
        fracs.append(100 * fractions[name])
        colors_list.append(COLORS[color])

    bars = ax.bar(labels, fracs, color=colors_list, alpha=0.7, edgecolor="none")
    for bar, frac in zip(bars, fracs, strict=True):
        if frac > 2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                frac + 1,
                f"{frac:.0f}%",
                ha="center",
                fontsize=5,
            )

    ax.axhline(100 * alpha, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("% units")
    ax.set_ylim(0, max(100 * alpha * 1.2, max(fracs) * 1.3))
    ax.tick_params(axis="x", labelsize=5)
    # Say both the raw-p threshold and R on the panel: this is a descriptive
    # preview, and its Monte Carlo resolution is part of what the bars mean.
    # Left-aligned: the title is wider than the axes, and centred it ran under
    # the panel letter.
    ax.set_title(
        "Raw empirical p < "
        f"{alpha:g} (descriptive)\n"
        f"R={n_control_draws}, min p={1 / (n_control_draws + 1):.3f}",
        fontsize=7,
        loc="left",
    )


def plot_best_model_selection(df, ax):
    """Distribution of the best model per unit, by held-out deviance.

    Each bar is a share of every unit
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.best_model_fractions`).
    """
    model_order = ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"]
    model_labels = ["Const.", "SWR", "Replay", "SWR+\nno-SWR"]
    model_colors = ["gray", COLORS["SWR"], COLORS["Non-Local"], COLORS["Combined"]]

    fractions = population_statistics.best_model_fractions(df, model_order)
    vals = [100 * fractions[m] for m in model_order]

    bars = ax.bar(model_labels, vals, color=model_colors, alpha=0.7, edgecolor="none")
    for bar, val in zip(bars, vals, strict=True):
        if val > 2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.0f}%",
                ha="center",
                fontsize=5,
            )
    ax.set_ylabel("% units")
    ax.tick_params(axis="x", labelsize=5)
    ax.set_title("Best model\n(held-out deviance)", fontsize=7, loc="left")


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------

#: The lags the models are fit on (+/-250 ms at 500 Hz) and the ones the
#: curves are reported on (+/-200 ms). The bundle is fit on the wider window so
#: the spline's boundary behaviour falls outside the reported one; the example
#: neuron is fit the same way and its curve trimmed to match.
FIT_LAGS = symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, DEFAULT_SAMPLING_FREQUENCY_HZ)
REPORT_LAGS = symmetric_lags(REPORT_HALF_WINDOW_S, DEFAULT_SAMPLING_FREQUENCY_HZ)


def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    neuron_ind=27,
    data_dir=None,
    stop_cohort="at_well",
    fitter="jax",
    penalty_sigma=DEFAULT_PENALTY_SIGMA,
):
    """Generate composite Figure 4.

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
        population bundle was fit with, and the only one with a penalised path.
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
            f"held-out p-values to draw. Figure 4 is drawn at the canonical "
            f"prior ({DEFAULT_PENALTY_SIGMA})."
        )
    # --- Load single-neuron data ---
    print(f"Loading data for {nwb_file_name}, {epoch_name}...")
    results_hpc, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )
    from continuum_swr_replay.data_loaders import load_data

    # the ripple trace is the source's (the Kay consensus over R-T's tetrodes)
    data = load_data(
        nwb_file_name,
        epoch_name,
        ripple_power=ripple_power_inputs(swr_epoch_id(nwb_file_name, epoch_name)),
    )

    # Decodes saved before get_position_info was trimmed carry a few more time
    # bins than load_data now returns; refit the example neuron on the run the
    # two axes share, exactly as the runner does.
    results_hpc, position_info, alignment_counts = align_results_to_position(
        results_hpc, data["position_info"]
    )
    data = {**data, "position_info": position_info}
    if alignment_counts["dropped_at"] != "none":
        print(
            "Trimmed decode/position axes to their shared timestamps: dropped "
            f"{alignment_counts['n_results_bins_dropped']} results bins and "
            f"{alignment_counts['n_position_rows_dropped']} position rows at "
            f"the {alignment_counts['dropped_at']}"
        )

    position_time = position_info.index

    # The example neuron is refit here, so it must be cut exactly as the
    # population bundle was: the same event sets and the same fit mask the
    # runner builds, from the same two functions.
    cohort_stops = None
    if stop_cohort != "all_stops":
        cohort_stops, cohort_report = load_session_stop_cohort(
            nwb_file_name, epoch_name, cohort=stop_cohort
        )
        print(
            f"Stop cohort {stop_cohort}: "
            f"{cohort_report['n_stops_after']}/{cohort_report['n_stops_before']} stops"
        )
    # Design matrices
    time = position_time.to_numpy(dtype=float)
    lags = FIT_LAGS
    _, fit_mask, bin_counts = build_session_fit_mask(
        data["position_info"],
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        excluded_intervals=excluded_intervals(swr_epoch_id(nwb_file_name, epoch_name)),
        basis_half_window_s=DEFAULT_BASIS_HALF_WINDOW_S,
    )
    print(
        f"Fit bins: {bin_counts['n_bins_fit']}/{bin_counts['n_bins_immobile']} "
        "immobile bins in cohort"
    )

    # The example neuron's events are the canonical ones -- the shipped event
    # table's rows for this session and its own ``is_swr`` split -- so the panel
    # and the population bundle describe the same events.
    canonical_events, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    swr_intervals, _ = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    event_times, event_counts = build_session_event_sets(
        data,
        results_hpc,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(lags).max()) / DEFAULT_SAMPLING_FREQUENCY_HZ,
    )
    print(
        f"Events in cohort: {event_counts['n_nonlocal_after']}"
        f"/{event_counts['n_nonlocal_before']} non-local, "
        f"{event_counts['n_ripple_after']}/{event_counts['n_ripple_before']} SWRs"
    )

    # The example neuron is fit on the same ladder the population bundle was,
    # matched non-event control included, drawn with the same session-keyed
    # seed -- so the panel's engagement model is the population's.
    event_times["matched_control"] = make_matched_control_times(
        event_times,
        fit_mask,
        np.random.default_rng(matched_control_seed(nwb_file_name, epoch_name)),
        avoid=artifact_dropped_events(
            read_artifact_event_drops(
                session_data_over_time_path(
                    nwb_file_name,
                    epoch_name,
                    non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
                )
            )
        ),
    ).times
    print(
        f"Matched non-event controls: {len(event_times['matched_control'])}"
        f"/{len(event_times['non_local'])} placed"
    )

    immobility_design_matrices, spline_basis, lag_time = make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=DEFAULT_SAMPLING_FREQUENCY_HZ,
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
    # bundle's curves are: the spline's boundary behaviour stays outside the
    # window the panel draws.
    report_mask = np.isin(
        np.round(lag_time, 6),
        np.round(REPORT_LAGS / DEFAULT_SAMPLING_FREQUENCY_HZ, 6),
    )
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
    overlapping text, without loading a session or fitting a neuron.

    Parameters
    ----------
    lag_time : np.ndarray, shape (n_report_lags,)
        Reported lags of the example neuron's curves, in seconds.
    spline_basis : np.ndarray, shape (n_report_lags, n_basis)
        Spline basis one event block of the parameters multiplies.
    neuron_spike_times : np.ndarray
        Spike times of the example neuron (seconds).
    glm_results : dict[str, fitted GLM result]
        The example neuron's fits; the two kernel blocks of
        ``SWR + non_local_no_SWR`` are drawn, with its ``covariance``.
    event_times : dict[str, pd.DataFrame]
        Event interval tables for the example neuron's PSTHs.
    pop_df : pd.DataFrame
        One row per population unit (held-out p-values, best model).
    modulations : dict[str, np.ndarray], each (n_lag, n_neuron)
        Population modulation curves, keys ``swr`` and ``no_swr``.
    pop_lag_time : np.ndarray, shape (n_lag,)
        Lags of ``modulations``, in seconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- Create figure ---
    # 4-panel layout (was 6): the standalone single-neuron AIC was dropped
    # because it duplicated the population model selection in the stats
    # panel, and the two side-by-side single-neuron columns were collapsed
    # into one overlay panel.
    #
    #     Row 1: panel a — single neuron overlay (modulation + PSTH,
    #                      overlaid SWR and non-SWR replay)
    #     Row 2: panel b — population heatmaps (2 side by side)
    #     Row 3: panel c — population average
    #            panel d — consolidated stats (% over the null + best model)
    print("Assembling composite figure...")
    fig_height = min(8.5, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450, constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.4, 1.4, 1.2], hspace=0.4)

    # ── Row 1: panel a — overlaid single-neuron example ──
    # Two sub-rows inside the row: modulation curve on top, PSTH below.
    # Center it within the row with a 1:2:1 left/mid/right width split so
    # the curves don't stretch to full 2-column width.
    gs_a_outer = gs[0].subgridspec(1, 3, width_ratios=[0.6, 2.0, 0.6])
    gs_a = gs_a_outer[0, 1].subgridspec(2, 1, hspace=0.1, height_ratios=[1.0, 0.8])
    ax_a_mod = fig.add_subplot(gs_a[0])
    ax_a_psth = fig.add_subplot(gs_a[1], sharex=ax_a_mod)

    # The joint model's two kernels, as panels b and c draw them for the
    # population: the bundle's ``SWR`` and ``no_SWR`` curves.
    model_configs = [
        {
            "curve": "SWR",
            "label": "SWR",
            "color": COLORS["SWR"],
            "event_key": "SWR",
        },
        {
            "curve": "no_SWR",
            "label": "non-SWR replay",
            "color": COLORS["Non-Local"],
            "event_key": "non_local_no_SWR",
        },
    ]
    plot_single_neuron_overlay(
        axes=[ax_a_mod, ax_a_psth],
        lag_time=lag_time,
        spline_basis=spline_basis,
        neuron_spike_times=neuron_spike_times,
        glm_results=glm_results,
        event_times=event_times,
        model_configs=model_configs,
    )

    # ── Row 2: panel b — population heatmaps (side by side) ──
    gs_b = gs[1].subgridspec(1, 2, wspace=0.15)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])
    im = plot_population_heatmaps(
        modulations["swr"],
        modulations["no_swr"],
        pop_lag_time,
        [ax_b1, ax_b2],
    )
    vmin, vmax = -np.log(2), np.log(2)
    cbar = fig.colorbar(im, ax=[ax_b1, ax_b2], shrink=0.8, pad=0.02)
    cbar.set_label("Log FR change")
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(["0.5x", "1x", "2x"])

    # ── Row 3: panel c (population average) + panel d (stats) ──
    # Panel d needs the width: each of its bar plots has four category labels.
    # The narrow gap gives panel d the room without shrinking panel c.
    gs_cd = gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.1)

    # Panel c: population average modulation
    ax_c = fig.add_subplot(gs_cd[0])
    plot_population_average(modulations, pop_lag_time, ax_c)

    # Panel d: consolidated stats — % beating the control null + best model.
    # A small wspace, so each bar plot is wide enough for its four category
    # labels.
    gs_d = gs_cd[1].subgridspec(1, 2, wspace=0.05)
    ax_d1 = fig.add_subplot(gs_d[0])
    ax_d2 = fig.add_subplot(gs_d[1])
    plot_significance(pop_df, ax_d1)
    plot_best_model_selection(pop_df, ax_d2)

    # ── Panel Labels ──
    label_panels(
        [ax_a_mod, ax_b1, ax_c, ax_d1],
        labels=["a", "b", "c", "d"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
    )

    return fig


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure04"),
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
            "only one with a penalised path)."
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

    # Before JAX initialises a backend (which is later than the import, and
    # checked afterwards): the panel's fit runs on the same backend the
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

    print(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure04", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
