#!/usr/bin/env python
"""Generate Figure 3: Non-local events exist outside detected SWRs.

Composite figure with 4 panels:

    Row 1 (panel a — full width):
        Real data example showing ~2s around a non-local event WITHOUT
        SWR. Stacked sub-rows: SWR z-score (flat), spike raster,
        non-local posterior (shows clear replay), state probabilities.

    Row 2 (panels b, c, d):
        b: SWR z-score density for Local vs Non-Local timepoints —
           shows non-local activity lives at low SWR power, not just
           barely below the SWR threshold.
        c: Replay speed by SWR status — low-SWR replay moves more
           slowly.
        d: Replay position by SWR status — non-SWR replay represents
           reward zones.

Usage:
    python scripts/figures/generate_figure03.py
    python scripts/figures/generate_figure03.py --output-dir paper/figures/figure03
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    dropped_for_excluded_time,
    read_artifact_event_drops,
    swr_intervals,
)
from continuum_swr_replay.analysis.events.detection import (
    get_confident_times,
    get_overlap_times,
)
from continuum_swr_replay.analysis.events.eligibility import artifact_drop_reasons
from continuum_swr_replay.analysis.events.immobility import (
    build_speed_eligible_event_cohort,
)
from continuum_swr_replay.analysis.events.swr_source import (
    artifact_intervals,
    ripple_power_inputs,
    swr_epoch_id,
)
from continuum_swr_replay.analysis.hippocampus import (
    load_data_over_time,
    load_results,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    validate_results_position_time_alignment,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_panels import (
    plot_replay_position_by_swr,
    plot_replay_speed_by_swr,
    plot_swr_zscore_by_state,
)
from continuum_swr_replay.visualization.plots import (
    plot_acausal_probabilities,
    plot_conditional_non_local_posterior,
    plot_ripple_consensus_trace_zscore,
    plot_spike_times,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Panel a — Real data example (non-SWR non-local event)
# ---------------------------------------------------------------------------

# Round steps (s) for panel a's relative-time ticks, finest first.
_TICK_STEPS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
_MAX_TICKS = 6


def _relative_time_ticks(
    t_lo: float, t_hi: float, t0: float
) -> tuple[np.ndarray, list[str]]:
    """Ticks for an axis showing ``[t_lo, t_hi]`` s, labelled in seconds from ``t0``.

    The ticks sit at relative times that are multiples of a round step (0 at
    ``t0``), the finest step that gives at most ``_MAX_TICKS`` ticks. Each label
    is its tick's relative time, not that time rounded from an automatic tick.

    Parameters
    ----------
    t_lo, t_hi : float
        Axis limits, in absolute seconds.
    t0 : float
        The absolute time shown as 0, in seconds.

    Returns
    -------
    ticks : np.ndarray, shape (n_ticks,)
        Tick positions, in absolute seconds.
    labels : list of str
        The relative time of each tick, e.g. ``"0"``, ``"0.5"``.
    """
    rel_lo, rel_hi = t_lo - t0, t_hi - t0
    for step in _TICK_STEPS_S:
        first = int(np.ceil(rel_lo / step - 1e-9))
        last = int(np.floor(rel_hi / step + 1e-9))
        if last - first + 1 <= _MAX_TICKS:
            break
    relative = np.round(np.arange(first, last + 1) * step, 6)
    return t0 + relative, [f"{r:g}" for r in relative]


def _set_relative_time_ticks(axes, t0: float) -> None:
    """Tick each axis of ``axes`` in seconds from ``t0``, leaving its limits."""
    for ax in axes:
        ticks, labels = _relative_time_ticks(*ax.get_xlim(), t0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def find_non_swr_example_event(
    data, results, window_sec=1.0, *, artifacts=None, dropped=None
):
    """Find a representative non-local event NOT co-occurring with SWR.

    Selects the event with highest peak non-local probability that does
    not overlap with any detected SWR. The events are re-detected from the
    posterior, so the one analysed-time rule is applied here: an event in
    excluded time (`artifacts`, the mask, widened by 0.3 s), or overlapping an
    SWR or MUA event dropped for it (`dropped`), is never chosen.

    Returns
    -------
    time_slice : slice
        Time window centered on the event.
    """
    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    position_info = data["position_info"]
    speed = position_info.head_speed.to_numpy()
    position_time = position_info.index
    validate_results_position_time_alignment(results.time.values, position_time)
    non_local_prob = non_local_prob.assign_coords(
        time=position_time.to_numpy(dtype=float)
    )

    detected_non_local_times = get_confident_times(
        non_local_prob,
        speed,
        position_time.to_numpy(dtype=float),
        DEFAULT_NON_LOCAL_THRESHOLD,
    )
    non_local_times, _ = build_speed_eligible_event_cohort(
        detected_non_local_times,
        speed=speed,
        time=position_time,
        event_col="non_local_event",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    if artifacts is not None:
        reasons = artifact_drop_reasons(
            non_local_times["start_time"].to_numpy(),
            non_local_times["end_time"].to_numpy(),
            artifacts,
            dropped,
        )
        non_local_times = non_local_times.loc[pd.isna(reasons)]
    # The session's SWR source intervals, already speed-gated (set by the caller).
    ripple_times = data["ripple_times"]

    # Find events NOT overlapping with SWR
    _, non_swr_events = get_overlap_times(non_local_times, ripple_times)

    if len(non_swr_events) == 0:
        raise ValueError("No non-local events without SWR overlap found.")

    # Pick longest event (likely the clearest example)
    best_idx = non_swr_events["duration"].idxmax()
    event = non_swr_events.loc[best_idx]

    center = (event.start_time + event.end_time) / 2
    return slice(center - window_sec, center + window_sec)


# ---------------------------------------------------------------------------
# Main Figure Assembly
# ---------------------------------------------------------------------------


def generate_figure(
    # TODO: swap to a non-wilbur session so Fig 3a comes from a different
    # animal than Fig 2a (cross-cutting narrative requirement). Candidates
    # from other animals will be evaluated separately.
    nwb_file_name="wilbur20210329_.nwb",
    epoch_name="02_r1",
    data_dir=PROCESSED_DATA_DIR,
):
    """Generate composite Figure 3.

    Parameters
    ----------
    nwb_file_name : str
        NWB file for the real data example panel.
    epoch_name : str
        Epoch name for the real data example panel.
    data_dir : Path
        Directory containing processed data files.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- Load data ---
    logging.info("Loading session data for example panel...")
    epoch_id = swr_epoch_id(nwb_file_name, epoch_name)
    # the ripple trace is the source's (the Kay consensus over R-T's tetrodes)
    data = load_data(
        nwb_file_name, epoch_name, ripple_power=ripple_power_inputs(epoch_id)
    )
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    logging.info("Loading aggregated data for statistics panels...")
    data_over_time = load_data_over_time(data_dir)
    # Panel a's SWR selection and shading: the session's SWR source intervals,
    # not load_data's stored ripples.
    data["ripple_times"] = swr_intervals(
        data_over_time, nwb_file_name=nwb_file_name, epoch_name=epoch_name
    )

    # Find example non-SWR event
    logging.info("Selecting non-SWR example event...")
    time_slice = find_non_swr_example_event(
        data,
        results,
        artifacts=artifact_intervals(epoch_id),
        dropped=dropped_for_excluded_time(
            read_artifact_event_drops(decode_paths(epoch_id, data_dir).data_over_time)
        ),
    )

    # --- Create figure ---
    fig_height = min(7.0, PAGE_HEIGHT)
    fig = plt.figure(figsize=(TWO_COLUMN, fig_height), dpi=450, constrained_layout=True)

    # 2 rows: (a) example stack, (b/c/d) stats
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.8, 1.1], hspace=0.35)

    # ── Row 1: Real data example (panel a) — 4 stacked sub-rows ──
    gs_a = gs[0].subgridspec(4, 1, hspace=0.05, height_ratios=[0.6, 0.8, 1.2, 0.8])
    ax_swr = fig.add_subplot(gs_a[0])
    ax_raster = fig.add_subplot(gs_a[1], sharex=ax_swr)
    ax_posterior = fig.add_subplot(gs_a[2], sharex=ax_swr)
    ax_probs = fig.add_subplot(gs_a[3], sharex=ax_swr)

    plot_ripple_consensus_trace_zscore(data, time_slice, ax_swr)
    plot_spike_times(data, time_slice, detector, env, results, ax_raster)
    plot_conditional_non_local_posterior(
        data, time_slice, results, env, detector, ax_posterior
    )
    plot_acausal_probabilities(results, time_slice, ax_probs, plot_local=True)

    # Convert absolute timestamps to relative time (seconds from the window
    # start, ``time_slice.start`` = the event centre minus 1 s)
    _set_relative_time_ticks(
        [ax_swr, ax_raster, ax_posterior, ax_probs], time_slice.start
    )

    # Remove x-labels from non-bottom sub-rows
    for ax in [ax_swr, ax_raster, ax_posterior]:
        ax.tick_params(labelbottom=False)
    ax_probs.set_xlabel("Time (s)")

    # ── Row 2: panels b, c, d (stats panels) ──
    gs_bcd = gs[1].subgridspec(1, 3, wspace=0.4)

    # Panel b: SWR z-score by state
    ax_b = fig.add_subplot(gs_bcd[0])
    plot_swr_zscore_by_state(data_over_time, ax=ax_b)

    # Panel c: Replay speed by SWR
    ax_c = fig.add_subplot(gs_bcd[1])
    plot_replay_speed_by_swr(data_over_time, ax=ax_c)
    ax_c.set_xlim(0, 1500)  # Focus on meaningful range

    # Panel d: Replay position by SWR
    ax_d = fig.add_subplot(gs_bcd[2])
    plot_replay_position_by_swr(data_over_time, ax=ax_d)

    # ── Panel Labels ──
    label_panels(
        [ax_swr, ax_b, ax_c, ax_d],
        labels=["a", "b", "c", "d"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure03"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
    )

    logging.info(f"Saving to {args.output_dir}...")
    save_figure("figure03", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
