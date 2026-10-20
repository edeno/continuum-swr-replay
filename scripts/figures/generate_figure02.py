#!/usr/bin/env python
"""Generate Figure 2: The model finds SWR-coincident replay (and more).

Composite 5-panel figure with the following narrative:

    a — Three example non-local events, one per SWR-overlap class
        (SWR-contained / SWR-partial / non-SWR), picked deterministically
        from a single session. Anchors the reader in real data and
        previews panel b.

    b — Stacked bar of immobile non-local events by SWR-overlap class
        (SWR-contained / SWR-partial / non-SWR). Shows the three-way
        population breakdown that panel a previews.

    c — Event-triggered averages of non-local probability and SWR
        z-score around non-local event starts, split by SWR-overlap
        status. Two signals only; theta and MUA live in the supplement.

    d — NL/SWR timing offset histograms on the immobile overlap subset
        (NL-start minus SWR-start, NL-end minus SWR-end), one row per
        overlapping event (the matched SWR is the containing SWR when one
        exists, else the earliest overlapping SWR). Shows that even when
        events overlap, non-local events typically sit inside the SWR
        window but with a long tail that extends beyond it.

    e — Median decoding error during Local state across sessions.
        Positive control that Local-state decoding is accurate.

Panels a, b, and d are driven by a single classification source:
``build_classified_nonlocal_event_table`` is computed once over the
aggregated ``data_over_time`` (one row per non-local event, full
label-window timing, strict positive-duration SWR overlap, matched SWR).
Panel a filters it for the example session, panel b tallies its
``overlap_class``, and panel d subtracts ``matched_swr_*`` from the event
boundaries. Panel c keeps ``get_event_types`` (parity-tested against the
same table) for its event-triggered windows.

Remaining validation panels (non-local probability threshold,
calibration curve, full four-signal triggered averages, moving-vs-
immobile state fractions, state probability vs theta phase) live in the
supplementary "Decoder validation" figure.

Usage
-----
    python scripts/figures/generate_figure02.py
    python scripts/figures/generate_figure02.py --nwb-file-name chimi20200212_.nwb --epoch-name 02_r1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.events.analysed_time import swr_intervals
from continuum_swr_replay.analysis.events.classified_table import (
    assert_drawn_class_matches_ripples,
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.detection import select_typical_event_by_class
from continuum_swr_replay.analysis.events.swr_inventory import (
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.analysis.hippocampus import (
    compute_local_decoding_error,
    compute_swr_overlap_class_counts,
    get_event_types,
    load_data_over_time,
    load_results,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_panels import (
    compute_triggered_average_curves,
    plot_decoding_error_distribution,
    plot_event_example_stack,
    plot_event_timing_offsets,
    plot_swr_overlap_class_bar,
    plot_triggered_averages,
)
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_track_graph_2d,
)

# Signals shown in panel c (triggered averages). Theta/MUA live in the
# supplementary decoder-validation figure.
TRIGGERED_SIGNALS = ["Non-Local_prob", "SWR_zscore"]

# The render cache's version, part of its file name. The name also carries the
# SWR source's name and a fingerprint of the epoch register's analysed epochs
# (the cohort panels b-e summarise), so a new SWR inventory, artifact run or
# register change makes a new cache by itself. Bump this whenever the cached
# values could change for any other reason: a change to
# ``_compute_figure_inputs`` or to a library function it calls (the read through
# the SWR source, the event classification, the triggered averages,
# ``compute_local_decoding_error``), or to the decodes themselves. Without a
# bump the figure is drawn from the old cache, unless ``--force-recompute`` is
# passed.
# 1: the cache named ``..._figure02_whole_event_speed_v1_...``.
# 2: panel e's decoding error leaves excluded time out (task 26b).
FIGURE02_CACHE_VERSION = 2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
figure_utilities.set_figure_defaults(context="paper")
warnings.filterwarnings("ignore")

CLASS_ORDER = ("swr_contained", "swr_partial", "non_swr")
CLASS_TITLES = {
    "swr_contained": "SWR-contained",
    "swr_partial": "SWR-partial",
    "non_swr": "non-SWR",
}


# ---------------------------------------------------------------------------
# Event selection for panel a
# ---------------------------------------------------------------------------


def pick_example_events(
    classified_events_all: pd.DataFrame,
    nwb_file_name: str,
    epoch_name: str,
    peak_threshold: float = 0.9,
) -> dict:
    """Deterministic three-example selection for panel a.

    Filters the pre-built classified table to the example session,
    restricts to canonical whole-event-immobile events (matching panel b) with
    ``peak_nonlocal_prob >= peak_threshold``, then runs
    :func:`select_typical_event_by_class` per overlap class (median
    duration within class). Returns a dict ``{class -> pd.Series}`` whose
    series carry ``start_time`` / ``end_time`` / ``duration`` / ``peak_prob``
    (renamed from the table's columns) plus ``event_id`` / ``overlap_class``.
    """
    animal, day = parse_nwb_filename(nwb_file_name)
    session = classified_events_all[
        (classified_events_all["animal"] == animal)
        & (classified_events_all["day"] == day)
        & (classified_events_all["epoch"] == epoch_name)
        & classified_events_all["whole_event_speed_eligible"]
    ]
    # select_typical_event_by_class expects ``overlap_class``, ``duration``,
    # and (via the precomputed path) a ``peak_prob`` column.
    session = session.rename(
        columns={
            "peak_nonlocal_prob": "peak_prob",
            "event_duration": "duration",
            "event_start": "start_time",
            "event_end": "end_time",
        }
    )
    return select_typical_event_by_class(session, peak_threshold=peak_threshold)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def _figure02_cache_path(
    nwb_file_name: str, epoch_name: str, data_dir: Path
) -> Path:
    """Path for the cached Figure-2 render inputs (joblib bundle).

    A single joblib bundle is used rather than netCDF because the bundle
    mixes a session ``data`` dict, an xarray ``results`` Dataset whose
    ``state_bins`` MultiIndex netCDF cannot serialize, the fitted detector
    and environment objects, and several small DataFrames; joblib (pickle)
    preserves them all exactly.
    """
    animal, day = parse_nwb_filename(nwb_file_name)
    # ``load_data_over_time`` reads every analysed epoch, so the cohort is the
    # register's analysed epochs.
    cohort = hashlib.sha256("\n".join(analysed_epoch_ids()).encode()).hexdigest()
    return (
        data_dir
        / "intermediates"
        / f"{animal}{day}_{epoch_name}_figure02_v{FIGURE02_CACHE_VERSION}_"
        f"{swr_source_name()}_register{cohort[:12]}_cache.joblib"
    )


def _compute_figure_inputs(
    nwb_file_name: str,
    epoch_name: str,
    data_dir: Path,
) -> dict:
    """Run the expensive loads/derivations and return all render inputs.

    Everything the figure needs is reduced here to a single dict that is
    cheap to cache: the example session's ``data``/``results``/``detector``
    /``env`` plus the chosen example events (panel a), and the small
    aggregate products for panels b–e (overlap-class counts, NL/SWR timing
    offsets, triggered-average curves, per-session decoding-error table).
    The multi-GB ``data_over_time`` is consumed here and intentionally not
    returned, so the cached bundle stays small.
    """
    # ── Load data ──
    logging.info(f"Loading session {nwb_file_name} {epoch_name}...")
    # the ripple trace is the source's (the Kay consensus over R-T's tetrodes)
    data = dict(
        load_data(
            nwb_file_name,
            epoch_name,
            ripple_power=ripple_power_inputs(swr_epoch_id(nwb_file_name, epoch_name)),
        )
    )
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    logging.info("Loading aggregated data for stats panels...")
    data_over_time = load_data_over_time(data_dir)
    # Panel a's SWR shading: the session's SWR source intervals, the ones the
    # classes below are computed against (not load_data's stored ripples).
    data["ripple_times"] = swr_intervals(
        data_over_time, nwb_file_name=nwb_file_name, epoch_name=epoch_name
    )
    event_types = get_event_types(data_over_time)

    # Single classification source for panels a/b/d: the unfiltered
    # classified table with full label-window timing, computed once over
    # the multi-session aggregate. Panel c stays on ``get_event_types``
    # (parity-tested against this table).
    classified_events_all = build_classified_nonlocal_event_table(data_over_time)
    n_total = len(classified_events_all)
    immobile_mask = classified_events_all["whole_event_speed_eligible"].astype(bool)
    n_immobile = int(immobile_mask.sum())
    class_counts_log = classified_events_all.loc[immobile_mask][
        "overlap_class"
    ].value_counts()
    logging.info(
        "Classified %d non-local events; %d pass whole-event immobility; "
        "by class: contained=%d, partial=%d, non_swr=%d",
        n_total,
        n_immobile,
        int(class_counts_log.get("swr_contained", 0)),
        int(class_counts_log.get("swr_partial", 0)),
        int(class_counts_log.get("non_swr", 0)),
    )

    logging.info("Selecting three example events...")
    chosen = pick_example_events(classified_events_all, nwb_file_name, epoch_name)
    missing = set(CLASS_ORDER) - set(chosen)
    if missing:
        raise RuntimeError(
            f"Example session lacks events in classes: {sorted(missing)}"
        )
    for cls in CLASS_ORDER:
        ev = chosen[cls]
        logging.info(
            f"  {cls}: {1000 * ev.duration:.0f} ms, peak prob {ev.peak_prob:.2f}"
        )
        if ev.peak_prob < 0.9:
            logging.warning(
                "  %s example falls back below peak_threshold 0.9 "
                "(actual peak prob %.2f)",
                cls,
                ev.peak_prob,
            )

    # Guard: panel a shades data["ripple_times"]; fail if a chosen example would
    # be drawn with an SWR span that contradicts its column label.
    assert_drawn_class_matches_ripples(chosen, data["ripple_times"])
    # Cheap hash-collision guard: each chosen event_id must resolve to
    # exactly one classified row.
    for cls, ev in chosen.items():
        n = int((classified_events_all["event_id"] == ev["event_id"]).sum())
        if n != 1:
            raise AssertionError(
                f"event_id {ev['event_id']!r} (column {cls!r}) matches {n} rows "
                f"in classified_events_all (expected 1) — hash collision."
            )

    logging.info("Computing overlap-class counts (panel b)...")
    class_counts = compute_swr_overlap_class_counts(classified_events_all)
    logging.info(
        f"  {class_counts['total']:,} immobile NL events; "
        f"SWR-contained {class_counts['swr_contained']:,} / "
        f"SWR-partial {class_counts['swr_partial']:,} / "
        f"non-SWR {class_counts['non_swr']:,}"
    )

    logging.info("Computing NL/SWR timing offsets (panel d)...")
    # One row per immobile, SWR-overlapping event: NL boundary minus the
    # matched SWR's boundary (matched SWR = containing SWR for
    # swr_contained, else earliest overlapping SWR).
    overlap_pairs = classified_events_all[
        immobile_mask & (classified_events_all["overlap_class"] != "non_swr")
    ].copy()
    overlap_pairs["start_diff"] = (
        overlap_pairs["event_start"] - overlap_pairs["matched_swr_start"]
    )
    overlap_pairs["end_diff"] = (
        overlap_pairs["event_end"] - overlap_pairs["matched_swr_end"]
    )
    offsets = overlap_pairs[["animal", "day", "epoch", "start_diff", "end_diff"]]
    logging.info(f"  {len(offsets):,} immobile SWR-overlapping NL events")

    logging.info("Computing triggered-average curves (panel c)...")
    triggered_curves = compute_triggered_average_curves(
        data_over_time, event_types, TRIGGERED_SIGNALS, time_col="start_time",
    )

    logging.info("Computing per-session decoding error (panel e)...")
    error_df = compute_local_decoding_error(data_over_time)

    return {
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "data": data,
        "results": results,
        "detector": detector,
        "env": env,
        "chosen": chosen,
        "class_counts": class_counts,
        "offsets": offsets,
        "triggered_curves": triggered_curves,
        "error_df": error_df,
    }


def generate_figure(
    nwb_file_name: str = "wilbur20210329_.nwb",
    epoch_name: str = "02_r1",
    data_dir: Path = PROCESSED_DATA_DIR,
    window_sec: float = 0.4,
    use_cache: bool = True,
):
    """Build Figure 2, loading cached render inputs when available.

    Parameters
    ----------
    use_cache : bool, default True
        When True and a cached bundle exists under ``data_dir /
        intermediates``, load it and skip the expensive session load
        (~9 min) and aggregate processing (~11 min). When False
        (``--force-recompute``), recompute and overwrite the cache.
        Figure-only edits (layout, styling) reuse the cache.
    """
    cache_path = _figure02_cache_path(nwb_file_name, epoch_name, data_dir)
    if use_cache and cache_path.exists():
        logging.info(
            f"Loading cached render inputs from {cache_path} "
            "(use --force-recompute to rebuild)..."
        )
        inputs = joblib.load(cache_path)
    else:
        inputs = _compute_figure_inputs(nwb_file_name, epoch_name, data_dir)
        logging.info(f"Caching render inputs to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(inputs, cache_path)

    return _render_figure(inputs, window_sec=window_sec)


def _render_figure(inputs: dict, window_sec: float = 0.4):
    data = inputs["data"]
    results = inputs["results"]
    detector = inputs["detector"]
    env = inputs["env"]
    chosen = inputs["chosen"]
    class_counts = inputs["class_counts"]
    offsets = inputs["offsets"]
    triggered_curves = inputs["triggered_curves"]
    error_df = inputs["error_df"]

    # ── Layout ──
    fig_height = min(9.0, PAGE_HEIGHT)
    fig = plt.figure(
        figsize=(TWO_COLUMN, fig_height),
        dpi=450,
        constrained_layout=True,
    )
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[2.2, 0.5, 1.1],
        hspace=0.3,
    )

    # ── Row 1: panel a ──
    # Left: a shared 2D track + occupancy spatial key. Right: three example
    # columns × 4 sub-rows. A 1D linearized track-graph strip is attached
    # flush to the leftmost posterior as its spatial y-axis key.
    gs_a_outer = outer[0].subgridspec(1, 2, width_ratios=[0.7, 5.0], wspace=0.02)

    # 2D spatial key, vertically centred in the left region so it reads as
    # roughly square next to the tall example stack.
    gs_key = gs_a_outer[0].subgridspec(3, 1, height_ratios=[0.5, 2.0, 0.9])
    ax_key2d = fig.add_subplot(gs_key[1])
    plot_track_graph_2d(
        data["track_graph"],
        data["position_info"],
        ax=ax_key2d,
        edge_order=data["linear_edge_order"],
    )

    gs_a = gs_a_outer[1].subgridspec(
        4,
        3,
        hspace=0.04,
        wspace=0.12,
        height_ratios=[0.55, 0.9, 1.2, 0.75],
    )
    max_position = float(np.ceil(data["position_info"].linear_position.max()))

    first_column_axes: list = []
    for col_i, cls in enumerate(CLASS_ORDER):
        axes_col = [fig.add_subplot(gs_a[row, col_i]) for row in range(4)]
        for ax in axes_col[1:]:
            ax.sharex(axes_col[0])
        if col_i == 0:
            first_column_axes = axes_col

        event = chosen[cls]
        center = (event.start_time + event.end_time) / 2
        time_slice = slice(center - window_sec, center + window_sec)
        column_title = f"{CLASS_TITLES[cls]}\n{1000 * event.duration:.0f} ms"
        plot_event_example_stack(
            data=data,
            results=results,
            detector=detector,
            env=env,
            time_slice=time_slice,
            axes=axes_col,
            column_title=column_title,
            show_ylabels=(col_i == 0),
            relative_time_center=center,
        )

    # 1D linearized track-graph strip attached flush to the leftmost
    # posterior as an inset, sharing its linear-position y-axis. It becomes
    # the spatial y-key, so blank the posterior's own numeric y-axis.
    ax_post_left = first_column_axes[2]
    # Right edge of the inset sits at x=0 (the posterior's left edge) so the
    # linearized track is flush against the posterior it keys.
    ax_strip = ax_post_left.inset_axes([-0.12, 0.0, 0.12, 1.0])
    plot_linear_track_graph_strip(
        data["track_graph"],
        data["linear_edge_order"],
        data["linear_edge_spacing"],
        ax=ax_strip,
        max_position=max_position,
    )
    ax_post_left.set_ylabel("")
    ax_post_left.tick_params(labelleft=False)

    # ── Row 2: panel b (wide overlap stacked bar) ──
    ax_b = fig.add_subplot(outer[1])
    plot_swr_overlap_class_bar(class_counts, ax=ax_b)

    # ── Row 3: panels c (triggered averages), d (timing), e (error) ──
    # One flat 2×4 grid so every panel shares the same two grid rows: c is
    # a 2-row signal stack (col 0), while d's two histograms (cols 1-2) and
    # e (col 3) each span both rows. Sharing the grid keeps their tops and
    # bottoms aligned (a nested subgridspec for c would not align).
    gs_cde = outer[2].subgridspec(
        2,
        4,
        width_ratios=[1.35, 0.8, 0.8, 1.05],
        hspace=0.05,
        wspace=0.38,
    )

    ax_c_top = fig.add_subplot(gs_cde[0, 0])
    ax_c_bot = fig.add_subplot(gs_cde[1, 0], sharex=ax_c_top)
    ax_c_top.tick_params(labelbottom=False)
    plot_triggered_averages(
        data_over_time=None,
        event_types=None,
        time_col="start_time",
        axes=[ax_c_top, ax_c_bot],
        signals=TRIGGERED_SIGNALS,
        precomputed=triggered_curves,
    )

    ax_d_start = fig.add_subplot(gs_cde[:, 1])
    ax_d_end = fig.add_subplot(gs_cde[:, 2], sharey=ax_d_start)
    plot_event_timing_offsets(offsets, axes=[ax_d_start, ax_d_end])
    ax_d_end.tick_params(labelleft=False)
    ax_d_end.set_ylabel("")

    ax_e = fig.add_subplot(gs_cde[:, 3])
    plot_decoding_error_distribution(ax=ax_e, error_df=error_df)

    # ── Panel labels ──
    # d and e are placed relative to their own axes (right column / centered
    # bottom). a, b, and c are placed at a shared figure-x so the label
    # sits left of the 2D key (panel a) and b/c line up vertically — their
    # axes boxes start at different x (panel b's long y-tick label pushes
    # its box right), so a fixed per-axis offset would not align them.
    label_panels(
        [ax_d_start, ax_e],
        labels=["d", "e"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
    )
    # Finalize constrained_layout so axis positions are final before reading
    # them, then label a/b/c at the common left edge of their content.
    fig.canvas.draw()
    x_label = (
        min(
            ax_key2d.get_position().x0,
            ax_b.get_position().x0,
            ax_c_top.get_position().x0,
        )
        - 0.006
    )
    for ax, letter in (
        (first_column_axes[0], "a"),
        (ax_b, "b"),
        (ax_c_top, "c"),
    ):
        fig.text(
            x_label,
            ax.get_position().y1,
            letter,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/figure02"),
    )
    parser.add_argument("--nwb-file-name", type=str, default="wilbur20210329_.nwb")
    parser.add_argument("--epoch-name", type=str, default="02_r1")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--window-sec", type=float, default=0.4)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Re-run the expensive loads/derivations instead of loading the "
            "cached render inputs under data_dir/intermediates (overwrites "
            "the cache). Use after changing any data/computation upstream of "
            "the figure; figure-only edits can reuse the cache."
        ),
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
        window_sec=args.window_sec,
        use_cache=not args.force_recompute,
    )

    logging.info(f"Saving to {args.output_dir}...")
    save_figure("figure02", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
