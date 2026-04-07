#!/usr/bin/env python
"""Generate Figure 4: PFC neurons respond to replay regardless of SWR detection.

Composite figure with 6 panels in 3 rows:

    Row 1 (panels a, b, c):
        Single-neuron example showing GLM modulation, spike raster, and PSTH
        for non-SWR replay (a) and SWR (b) events, plus AIC model comparison (c).

    Row 2 (panel d):
        Population heatmaps — SWR and non-SWR replay modulation (same neuron
        sort order) showing parallel response structure.

    Row 3 (panels e, f):
        e: Population average modulation curves (mean ± SEM)
        f: Significance bars (fraction of neurons modulated) + AIC model selection

Usage:
    python scripts/generate_figure04_composite.py
    python scripts/generate_figure04_composite.py --output-dir paper/figures/figure04
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
    DEFAULT_SAMPLING_FREQUENCY_HZ,
)
from continuum_swr_replay.analysis.hippocampus import (
    get_confident_times,
    get_overlap_times,
    load_results,
)
from continuum_swr_replay.analysis.pfc import (
    _fit_immobility_models,
    _make_design_matrices,
)
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.glm_plots import (
    plot_AICs,
    plot_hist_relative_to_event,
    plot_raster_relative_to_event,
)

figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Panels a-b — Single-neuron GLM results (one model per call)
# ---------------------------------------------------------------------------

def plot_single_model_glm(
    axes,
    model_name,
    lag_time,
    spline_basis,
    neuron_spike_times,
    glm_result,
    event_start_times,
    title=None,
):
    """Plot GLM modulation + raster + PSTH for a single model on 3 axes.

    Parameters
    ----------
    axes : list of 3 Axes
        [ax_modulation, ax_raster, ax_psth]
    model_name : str
        Display name for the model
    lag_time : np.ndarray
        Time lags in seconds
    spline_basis : np.ndarray
        Spline basis functions
    neuron_spike_times : np.ndarray
        Spike times for the neuron
    glm_result : statsmodels result
        Fitted GLM result
    event_start_times : pd.Series
        Event start times
    title : str, optional
        Panel title (defaults to model_name)
    """
    ax_mod, ax_raster, ax_psth = axes
    lag_bounds = lag_time[0], lag_time[-1]
    modulation_bounds = np.log(0.5), np.log(2.0)

    # Row 1: GLM modulation coefficients
    modulation = glm_result.params[1:] @ spline_basis.T
    ax_mod.plot(lag_time, modulation)
    ax_mod.fill_between(
        lag_time,
        (glm_result.params[1:] + glm_result.bse[1:]) @ spline_basis.T,
        (glm_result.params[1:] - glm_result.bse[1:]) @ spline_basis.T,
        alpha=0.25,
    )
    ax_mod.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_mod.set_title(title or model_name)
    ax_mod.set_ylim(modulation_bounds)
    ax_mod.set_xticks((lag_bounds[0], 0, lag_bounds[1]))
    ax_mod.set_yticks((modulation_bounds[0], 0, modulation_bounds[1]))
    ax_mod.set_ylabel("Mod.")
    ax_mod.set_yticklabels(
        (np.exp(modulation_bounds[0]), 1, np.exp(modulation_bounds[1]))
    )
    ax_mod.tick_params(labelbottom=False)

    # Row 2: Spike raster
    plot_raster_relative_to_event(neuron_spike_times, event_start_times, lag_bounds, ax_raster)
    ax_raster.set_ylabel("Event")
    ax_raster.tick_params(labelbottom=False)

    # Row 3: PSTH
    plot_hist_relative_to_event(neuron_spike_times, event_start_times, lag_bounds, ax_psth)
    ax_psth.set_ylabel("Rate\n[sp/s]")
    ax_psth.set_xlabel("Time from\nevent (s)")


# ---------------------------------------------------------------------------
# Panel c — Single-neuron AIC
# ---------------------------------------------------------------------------

def plot_single_neuron_aic(glm_results, ax, model_list=None):
    """Plot AIC comparison for a single neuron on provided axes."""
    if model_list is None:
        model_list = ["constant", "SWR", "SWR + non_local_no_SWR", "non_local"]

    model_labels = {
        "constant": "Const.",
        "SWR": "SWR",
        "SWR + non_local_no_SWR": "SWR +\nnon-SWR",
        "non_local": "Replay",
    }

    aics = [glm_results[m].aic for m in model_list]
    min_aic = min(aics)
    delta_aics = [a - min_aic for a in aics]
    labels = [model_labels.get(m, m) for m in model_list]

    colors = ["gray", COLORS["SWR"], "#e6ab02", COLORS["Non-Local"]]
    ax.bar(labels, delta_aics, color=colors[:len(model_list)], alpha=0.7, edgecolor="none")
    ax.set_ylabel("ΔAIC")
    ax.set_title("Model comparison")
    ax.tick_params(axis="both", which="both", length=3, direction="out")


# ---------------------------------------------------------------------------
# Panel d — Population heatmaps (accepts axes)
# ---------------------------------------------------------------------------

def plot_population_heatmaps_on_axes(swr_mod, no_swr_mod, lag_time, axes):
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

def plot_population_average_on_ax(modulations, lag_time, ax):
    """Plot population mean ± SEM modulation on provided ax."""
    lag_ms = lag_time * 1000

    for name, color, label in [
        ("SWR", COLORS["SWR"], "SWR"),
        ("no_SWR", COLORS["Non-Local"], "Non-SWR replay"),
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
    ax.set_title(f"Population (n={modulations['SWR'].shape[1]:,})")


# ---------------------------------------------------------------------------
# Panel f — Significance + AIC (accepts axes)
# ---------------------------------------------------------------------------

def plot_significance_on_ax(df, ax):
    """Plot fraction of significant neurons on provided ax."""
    tests = {
        "SWR": "SWR vs. constant: fdr_q",
        "Replay": "non_local vs. constant: fdr_q",
        "non-SWR\nreplay": "non_local_no_SWR vs. constant: fdr_q",
        "Replay\nbeyond SWR": "SWR + non_local_no_SWR vs. SWR: fdr_q",
    }

    n_total = len(df)
    labels = list(tests.keys())
    fracs = [100 * (df[col] < 0.05).sum() / n_total for col in tests.values()]
    colors_list = [COLORS["SWR"], COLORS["Non-Local"], COLORS["Non-Local"], COLORS["Non-Local"]]

    bars = ax.bar(labels, fracs, color=colors_list, alpha=0.7, edgecolor="none")
    for bar, frac in zip(bars, fracs):
        if frac > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 1,
                    f"{frac:.0f}%", ha="center", fontsize=5)

    ax.axhline(5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("% neurons\n(FDR q<0.05)")
    ax.set_ylim(0, max(fracs) * 1.3)


def plot_aic_selection_on_ax(df, ax):
    """Plot AIC model selection distribution on provided ax."""
    n_total = len(df)
    model_order = ["constant", "SWR", "non_local", "SWR + non_local_no_SWR"]
    model_labels = ["Const.", "SWR", "Replay", "SWR +\nnon-SWR"]
    model_colors = ["gray", COLORS["SWR"], COLORS["Non-Local"], "#e6ab02"]

    counts = df["min_aic_model"].value_counts()
    vals = [100 * counts.get(m, 0) / n_total for m in model_order]

    bars = ax.bar(model_labels, vals, color=model_colors, alpha=0.7, edgecolor="none")
    for bar, val in zip(bars, vals):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}%", ha="center", fontsize=5)
    ax.set_ylabel("% neurons")
    ax.set_title("Best model (AIC)")


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------

def generate_figure(
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    neuron_ind=27,
    data_dir=PROCESSED_DATA_DIR,
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

    non_local_prob = (
        results_hpc.acausal_state_probabilities.sel(states="Non-Local Continuous")
        + results_hpc.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    )

    non_local_times = get_confident_times(
        non_local_prob,
        data["position_info"].head_speed.to_numpy(),
        data["position_info"].index.to_numpy(),
        DEFAULT_NON_LOCAL_THRESHOLD,
    )
    non_local_times = non_local_times[
        non_local_times.max_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ]
    non_local_SWR_times, non_local_no_SWR_times = get_overlap_times(
        non_local_times, data["ripple_times"], results_hpc
    )

    event_times = {
        "non_local_no_SWR": non_local_no_SWR_times,
        "non_local_SWR": non_local_SWR_times,
        "non_local": non_local_times,
        "SWR": data["ripple_times"],
    }

    # Design matrices
    time = data["position_info"].index.to_numpy()
    is_immobile = data["position_info"].head_speed <= DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    immobility_design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times, data["position_info"],
        sampling_frequency=DEFAULT_SAMPLING_FREQUENCY_HZ,
        lags=np.arange(-100, 101, 1),
    )

    neuron_spike_times = data["spike_times"]["mPFC"][neuron_ind]
    neuron_spike_count = get_spikecount_per_time_bin(neuron_spike_times, time)

    print("Fitting GLM models...")
    glm_results, _ = _fit_immobility_models(
        is_immobile, immobility_design_matrices, neuron_spike_count
    )

    # --- Load population data ---
    print("Loading population data...")
    pop_df, modulations, pop_lag_time = load_population_data(
        data_dir, min_firing_rate=1.0
    )

    # --- Create figure ---
    print("Assembling composite figure...")
    fig_height = min(8.5, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450,
                     constrained_layout=True)

    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.0, 1.2, 1.5], hspace=0.4)

    # ── Row 1: Single neuron example (a, b, c) ──
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.5)

    # Panel a: non-SWR replay (3 sub-rows)
    gs_a = gs_top[0].subgridspec(3, 1, hspace=0.1)
    ax_a = [fig.add_subplot(gs_a[i]) for i in range(3)]

    plot_single_model_glm(
        ax_a, "Non-SWR Replay", lag_time, spline_basis,
        neuron_spike_times, glm_results["non_local_no_SWR"],
        event_times["non_local_no_SWR"].start_time,
        title="Non-SWR\nReplay",
    )

    # Panel b: SWR (3 sub-rows)
    gs_b = gs_top[1].subgridspec(3, 1, hspace=0.1)
    ax_b = [fig.add_subplot(gs_b[i]) for i in range(3)]

    plot_single_model_glm(
        ax_b, "SWR", lag_time, spline_basis,
        neuron_spike_times, glm_results["SWR"],
        event_times["SWR"].start_time,
        title="SWR",
    )

    # Panel c: AIC
    ax_c = fig.add_subplot(gs_top[2])
    plot_single_neuron_aic(glm_results, ax_c)

    # ── Row 2: Population heatmaps (d) ──
    gs_mid = gs[1].subgridspec(1, 2, wspace=0.15)
    ax_d1 = fig.add_subplot(gs_mid[0])
    ax_d2 = fig.add_subplot(gs_mid[1])

    im = plot_population_heatmaps_on_axes(
        modulations["SWR"], modulations["no_SWR"],
        pop_lag_time, [ax_d1, ax_d2],
    )

    vmin, vmax = -np.log(2), np.log(2)
    cbar = fig.colorbar(im, ax=[ax_d1, ax_d2], shrink=0.8, pad=0.02)
    cbar.set_label("Log FR change")
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(["0.5x", "1x", "2x"])

    # ── Row 3: Population average (e) + significance/AIC (f) ──
    gs_bot = gs[2].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.45)

    # Panel e: population average
    ax_e = fig.add_subplot(gs_bot[0])
    plot_population_average_on_ax(modulations, pop_lag_time, ax_e)

    # Panel f: significance bars + AIC selection
    gs_f = gs_bot[1].subgridspec(1, 2, wspace=0.4)
    ax_f1 = fig.add_subplot(gs_f[0])
    ax_f2 = fig.add_subplot(gs_f[1])

    plot_significance_on_ax(pop_df, ax_f1)
    plot_aic_selection_on_ax(pop_df, ax_f2)

    # ── Panel Labels ──
    label_panels(
        [ax_a[0], ax_b[0], ax_c, ax_d1, ax_e, ax_f1],
        labels=["a", "b", "c", "d", "e", "f"],
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
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        neuron_ind=args.neuron_ind,
        data_dir=args.data_dir,
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
