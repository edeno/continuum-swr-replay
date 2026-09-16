#!/usr/bin/env python
"""Generate Figure 4: PFC neurons respond to replay regardless of SWR detection.

Composite figure with 4 panels in 3 rows:

    Row 1 (panel a):
        Single-neuron example — overlaid GLM modulation curves and PSTHs
        for SWR and non-SWR replay events on the same mPFC neuron. Shows
        that the neuron responds to both event types with similar shape.

    Row 2 (panel b):
        Population heatmaps — SWR and non-SWR replay modulation with the
        same neuron sort order, showing parallel response structure
        across the whole population.

    Row 3 (panels c, d):
        c: Population average modulation curves (mean ± SEM) for SWR vs
           non-SWR replay.
        d: Consolidated statistics — fraction of neurons whose event
           regressor improves held-out deviance beyond the matched
           non-event control's null, and the distribution of the best
           model by held-out deviance across the population.

Every fitted quantity in the figure comes from the same estimator the
bundles were fit with: a Poisson GLM with a Gaussian prior on the
event-basis coefficients, and a peri-event kernel that returns to
baseline at both ends of the ±250 ms fit window. The example neuron's
band is the penalised posterior-mode standard error, and the population
significance is an empirical p-value from repeated matched-control
draws, Benjamini-Hochberg corrected across units.

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
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
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    load_session_canonical_events,
    load_session_swr_intervals,
)
from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
from continuum_swr_replay.analysis.pfc.glm import model_penalties
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.load_population import (
    load_population_data,
    significant_fraction,
)
from continuum_swr_replay.analysis.replay.cohort_selection import stop_cohort_suffix
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

#: Prior standard deviation of an event-basis coefficient, on the log-rate
#: scale, that the canonical bundles are fit at. The example neuron is refit in
#: this script, so it carries the same prior the population it sits beside was
#: fit under.
CANONICAL_PENALTY_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Panel a — Single-neuron example (overlaid SWR + non-SWR modulation)
# ---------------------------------------------------------------------------

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

    Sub-row 1 : one modulation-curve axis with a curve per model config
                (typically SWR and non-SWR replay on the same axes).
    Sub-row 2 : overlaid PSTHs — one line per event type.

    Parameters
    ----------
    axes : list of 2 matplotlib.axes.Axes
        [ax_modulation, ax_psth]
    lag_time : np.ndarray
        Time lags in seconds (covers both positive and negative lags).
    spline_basis : np.ndarray
        Spline basis matching ``glm_result.params[1:]``.
    neuron_spike_times : np.ndarray
        Spike times for the example neuron (seconds).
    glm_results : dict[str, fitted GLM result]
        Full dict of fitted GLMs; only the models in ``model_configs`` are
        plotted. ``bse`` is read as the penalised posterior-mode standard
        error the fit reports, so the band is the one the shipped estimator
        gives -- not a maximum-likelihood interval drawn around a penalised
        point estimate.
    event_times : dict[str, pd.DataFrame]
        Event interval tables keyed by name (columns: start_time, end_time).
    model_configs : list of dict
        Each entry: ``{"model": str, "label": str, "color": str,
        "event_key": str}``. ``model`` is a key into ``glm_results``,
        ``event_key`` is a key into ``event_times`` (used for the PSTH).

    """
    ax_mod, ax_psth = axes
    lag_bounds = lag_time[0], lag_time[-1]
    modulation_bounds = np.log(0.5), np.log(2.0)

    # ── Sub-row 1: overlaid modulation curves ──
    for cfg in model_configs:
        result = glm_results[cfg["model"]]
        modulation = result.params[1:] @ spline_basis.T
        upper = (result.params[1:] + result.bse[1:]) @ spline_basis.T
        lower = (result.params[1:] - result.bse[1:]) @ spline_basis.T

        ax_mod.plot(
            lag_time, modulation, color=cfg["color"],
            linewidth=1.2, label=cfg["label"],
        )
        ax_mod.fill_between(
            lag_time, lower, upper, color=cfg["color"], alpha=0.2,
        )

    ax_mod.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.set_ylim(modulation_bounds)
    ax_mod.set_xticks((lag_bounds[0], 0, lag_bounds[1]))
    ax_mod.set_yticks((modulation_bounds[0], 0, modulation_bounds[1]))
    ax_mod.set_yticklabels(
        (f"{np.exp(modulation_bounds[0]):.1f}", "1", f"{np.exp(modulation_bounds[1]):.1f}")
    )
    ax_mod.set_ylabel("GLM\nmodulation")
    ax_mod.legend(frameon=False, fontsize=6, loc="upper right")
    ax_mod.tick_params(labelbottom=False)
    ax_mod.set_title(
        "Example mPFC neuron (penalised fit, ±1 posterior SE)", fontsize=8, pad=4
    )

    # ── Sub-row 2: overlaid PSTHs ──
    # Cheap manual PSTH: compute spike-count density in lag bins around
    # each event start for each event type.
    psth_bin_ms = 10.0
    psth_edges = np.arange(lag_bounds[0], lag_bounds[1] + psth_bin_ms / 1000.0, psth_bin_ms / 1000.0)
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
            psth_centers, rate, color=cfg["color"], linewidth=1.2,
            label=cfg["label"],
        )

    ax_psth.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_psth.set_xlim(lag_bounds)
    ax_psth.set_xlabel("Time from event (s)")
    ax_psth.set_ylabel("Firing\nrate (Hz)")


# ---------------------------------------------------------------------------
# Panel d — Population heatmaps (accepts axes)
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
        swr_sorted.T, aspect="auto", cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax1.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Time from SWR (ms)")
    ax1.set_ylabel("Neuron")
    ax1.set_title("SWR modulation")

    ax2.imshow(
        no_swr_sorted.T, aspect="auto", cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        extent=[lag_ms[0], lag_ms[-1], no_swr_sorted.shape[1], 0],
        interpolation="none",
    )
    ax2.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Time from replay (ms)")
    ax2.set_title("Non-SWR replay modulation")

    return im1


# ---------------------------------------------------------------------------
# Panel e — Population average (accepts ax)
# ---------------------------------------------------------------------------

def plot_population_average(modulations, lag_time, ax):
    """Plot population mean ± SEM modulation on provided ax."""
    lag_ms = lag_time * 1000

    for name, color, label in [
        ("swr", COLORS["SWR"], "SWR"),
        ("no_swr", COLORS["Non-Local"], "Non-SWR replay"),
    ]:
        arr = modulations[name]
        mean = np.nanmean(arr, axis=1)
        sem = np.nanstd(arr, axis=1) / np.sqrt(arr.shape[1])

        ax.plot(lag_ms, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(lag_ms, mean - sem, mean + sem, color=color, alpha=0.2)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time from event (ms)")
    ax.set_ylabel("Log firing rate change")
    ax.set_xlim(lag_ms[0], lag_ms[-1])
    ax.legend(frameon=False)
    ax.set_title(f"Population (n={modulations['swr'].shape[1]:,})")


# ---------------------------------------------------------------------------
# Panel f — Significance + AIC (accepts axes)
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


def plot_significance(df, ax, alpha=0.05):
    """Fraction of units whose regressor beats the control-draw null.

    A unit counts when its empirical p-value -- the fraction of matched-control
    draws whose held-out improvement matched or beat the observed one --
    survives Benjamini-Hochberg across units at ``alpha``. The denominator is
    the units actually tested for that class: a unit whose session had no
    events of a class is untested for it, not a negative.
    """
    fractions = significant_fraction(df, alpha=alpha)
    labels, fracs, colors_list = [], [], []
    for label, name, color in SIGNIFICANCE_BARS:
        n_significant, n_tested = fractions[name]
        labels.append(label)
        fracs.append(100 * n_significant / n_tested if n_tested else np.nan)
        colors_list.append(COLORS[color])

    bars = ax.bar(labels, fracs, color=colors_list, alpha=0.7, edgecolor="none")
    for bar, frac in zip(bars, fracs, strict=True):
        if np.isfinite(frac) and frac > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 1,
                    f"{frac:.0f}%", ha="center", fontsize=5)

    ax.axhline(100 * alpha, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("% units")
    ax.set_ylim(0, np.nanmax(fracs) * 1.3)
    ax.tick_params(axis="x", labelsize=5)
    ax.set_title("Beats control null", fontsize=7)


def plot_best_model_selection(df, ax):
    """Distribution of the best model per unit, by held-out deviance."""
    n_total = len(df)
    model_order = ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"]
    model_labels = ["Const.", "SWR", "Replay", "SWR+\nno-SWR"]
    model_colors = ["gray", COLORS["SWR"], COLORS["Non-Local"], COLORS["Combined"]]

    counts = df["min_heldout_model"].value_counts()
    vals = [100 * counts.get(m, 0) / n_total for m in model_order]

    bars = ax.bar(model_labels, vals, color=model_colors, alpha=0.7, edgecolor="none")
    for bar, val in zip(bars, vals, strict=True):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}%", ha="center", fontsize=5)
    ax.set_ylabel("% units")
    ax.tick_params(axis="x", labelsize=5)
    ax.set_title("Best model\n(held-out deviance)", fontsize=7)


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------

#: The lags the models are fit on (+/-250 ms at 500 Hz) and the ones the
#: curves are reported on (+/-200 ms). The bundle is fit on the wider window so
#: the spline's boundary behaviour falls outside the reported one; the example
#: neuron is fit the same way and its curve trimmed to match.
FIT_LAGS = np.arange(-125, 126, 1)
REPORT_LAGS = np.arange(-100, 101, 1)


def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    neuron_ind=27,
    data_dir=PROCESSED_DATA_DIR,
    stop_cohort="at_well",
    fitter="jax",
    penalty_sigma=CANONICAL_PENALTY_SIGMA,
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
        scale. The panel is refit here, so it must carry the prior the
        population bundle in ``data_dir`` was fit under; a sensitivity bundle's
        panel is generated by naming its sigma.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- Load single-neuron data ---
    print(f"Loading data for {nwb_file_name}, {epoch_name}...")
    results_hpc, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )
    data = load_data(nwb_file_name, epoch_name)

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
    ).times
    print(
        f"Matched non-event controls: {len(event_times['matched_control'])}"
        f"/{len(event_times['non_local'])} placed"
    )

    immobility_design_matrices, spline_basis, lag_time = make_design_matrices(
        event_times, data["position_info"],
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
    print("Loading population data...")
    pop_df, modulations, pop_lag_time = load_population_data(
        data_dir,
        min_firing_rate=1.0,
        artifact_suffix=stop_cohort_suffix(stop_cohort),
    )

    # --- Create figure ---
    # 4-panel layout (was 6): the standalone single-neuron AIC was dropped
    # because it duplicated the population AIC in the stats panel, and
    # the two side-by-side single-neuron columns were collapsed into one
    # overlay panel.
    #
    #     Row 1: panel a — single neuron overlay (modulation + PSTH,
    #                      overlaid SWR and non-SWR replay)
    #     Row 2: panel b — population heatmaps (2 side by side)
    #     Row 3: panel c — population average
    #            panel d — consolidated stats (% over the null + best model)
    print("Assembling composite figure...")
    fig_height = min(8.5, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.4, 1.4, 1.2], hspace=0.4)

    # ── Row 1: panel a — overlaid single-neuron example ──
    # Two sub-rows inside the row: modulation curve on top, PSTH below.
    # Center it within the row with a 1:2:1 left/mid/right width split so
    # the curves don't stretch to full 2-column width.
    gs_a_outer = gs[0].subgridspec(1, 3, width_ratios=[0.6, 2.0, 0.6])
    gs_a = gs_a_outer[0, 1].subgridspec(2, 1, hspace=0.1, height_ratios=[1.0, 0.8])
    ax_a_mod = fig.add_subplot(gs_a[0])
    ax_a_psth = fig.add_subplot(gs_a[1], sharex=ax_a_mod)

    model_configs = [
        {
            "model": "SWR",
            "label": "SWR",
            "color": COLORS["SWR"],
            "event_key": "SWR",
        },
        {
            "model": "non_local_no_SWR",
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
        modulations["swr"], modulations["no_swr"],
        pop_lag_time, [ax_b1, ax_b2],
    )
    vmin, vmax = -np.log(2), np.log(2)
    cbar = fig.colorbar(im, ax=[ax_b1, ax_b2], shrink=0.8, pad=0.02)
    cbar.set_label("Log FR change")
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(["0.5x", "1x", "2x"])

    # ── Row 3: panel c (population average) + panel d (stats) ──
    gs_cd = gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.25)

    # Panel c: population average modulation
    ax_c = fig.add_subplot(gs_cd[0])
    plot_population_average(modulations, pop_lag_time, ax_c)

    # Panel d: consolidated stats — % beating the control null + best model
    gs_d = gs_cd[1].subgridspec(1, 2, wspace=0.3)
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/figure04"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument("--neuron-ind", type=int, default=27)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
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
        default=CANONICAL_PENALTY_SIGMA,
        help=(
            "Prior standard deviation of an event-basis coefficient, on the "
            f"log-rate scale (default: {CANONICAL_PENALTY_SIGMA}). The example "
            "neuron is refit here, so this must be the prior the population "
            "bundle in --data-dir was fit under."
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

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Before JAX initialises a backend (which is later than the import, and
    # checked afterwards): the panel's fit runs on the same backend the
    # population bundle was fit with, on a card nobody else is using.
    if args.fitter == "jax":
        record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
        print(f"JAX device: {record}")

    generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        neuron_ind=args.neuron_ind,
        data_dir=args.data_dir,
        stop_cohort=args.stop_cohort,
        fitter=args.fitter,
        penalty_sigma=args.penalty_sigma,
    )

    print(f"Saving to {args.output_dir}...")
    save_figure("figure04", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
