#!/usr/bin/env python
"""Generate Figure 2: The model finds SWR-coincident replay (and more).

Composite 6-panel figure with the following narrative:

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

    e — State probability vs theta phase during movement. Positive
        control that the decoder respects circuit-level structure.

    f — Median decoding error during Local state across sessions.
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
immobile state fractions) live in the supplementary
"Decoder validation" figure.

Usage
-----
    python scripts/generate_figure02.py
    python scripts/generate_figure02.py --nwb-file-name chimi20200212_.nwb --epoch-name 02_r1
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.hippocampus import (
    assert_drawn_class_matches_ripples,
    build_classified_nonlocal_event_table,
    compute_swr_overlap_class_counts,
    get_event_types,
    load_data_over_time,
    load_results,
    select_typical_event_by_class,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_panels import (
    plot_decoding_error_distribution,
    plot_event_example_stack,
    plot_event_timing_offsets,
    plot_state_probability_by_theta_phase,
    plot_swr_overlap_class_bar,
    plot_triggered_averages,
)

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
    restricts to immobile (endpoints rule, matching panel b) events with
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
        & classified_events_all["endpoints_speed_below_thresh"]
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


def generate_figure(
    nwb_file_name: str = "wilbur20210329_.nwb",
    epoch_name: str = "02_r1",
    data_dir: Path = PROCESSED_DATA_DIR,
    window_sec: float = 0.4,
):
    # ── Load data ──
    logging.info(f"Loading session {nwb_file_name} {epoch_name}...")
    data = load_data(nwb_file_name, epoch_name)
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC"
    )

    logging.info("Loading aggregated data for stats panels...")
    data_over_time = load_data_over_time(data_dir)
    event_types = get_event_types(data_over_time)

    # Single classification source for panels a/b/d: the unfiltered
    # classified table with full label-window timing, computed once over
    # the multi-session aggregate. Panel c stays on ``get_event_types``
    # (parity-tested against this table).
    classified_events_all = build_classified_nonlocal_event_table(data_over_time)
    n_total = len(classified_events_all)
    immobile_mask = classified_events_all["endpoints_speed_below_thresh"].astype(bool)
    n_immobile = int(immobile_mask.sum())
    class_counts_log = classified_events_all.loc[immobile_mask][
        "overlap_class"
    ].value_counts()
    logging.info(
        "Classified %d non-local events; %d pass endpoints-immobility filter; "
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

    # Guard: the column header is label-derived, but panel a shades the
    # threshold-crossing ripple_times. Fail if a chosen example would be
    # drawn with a SWR span that contradicts its column label.
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

    # ── Layout ──
    fig_height = min(9.0, PAGE_HEIGHT)
    fig = plt.figure(
        figsize=(TWO_COLUMN, fig_height),
        dpi=450,
        constrained_layout=True,
    )
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[2.2, 0.5, 1.4, 1.2],
        hspace=0.35,
    )

    # ── Row 1: panel a (3 example columns × 4 sub-rows) ──
    gs_a = outer[0].subgridspec(
        4,
        3,
        hspace=0.08,
        wspace=0.25,
        height_ratios=[0.55, 0.9, 1.2, 0.75],
    )
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

    # ── Row 2: panel b (wide overlap stacked bar) ──
    ax_b = fig.add_subplot(outer[1])
    plot_swr_overlap_class_bar(class_counts, ax=ax_b)

    # ── Row 3: panel c (triggered averages, 2 signals) + panel d (timing) ──
    gs_cd = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)

    gs_c = gs_cd[0].subgridspec(2, 1, hspace=0.1)
    ax_c_top = fig.add_subplot(gs_c[0])
    ax_c_bot = fig.add_subplot(gs_c[1], sharex=ax_c_top)
    ax_c_top.tick_params(labelbottom=False)
    plot_triggered_averages(
        data_over_time,
        event_types,
        time_col="start_time",
        axes=[ax_c_top, ax_c_bot],
        signals=["Non-Local_prob", "SWR_zscore"],
    )

    gs_d = gs_cd[1].subgridspec(1, 2, wspace=0.15)
    ax_d_start = fig.add_subplot(gs_d[0])
    ax_d_end = fig.add_subplot(gs_d[1], sharey=ax_d_start)
    plot_event_timing_offsets(offsets, axes=[ax_d_start, ax_d_end])
    ax_d_end.tick_params(labelleft=False)
    ax_d_end.set_ylabel("")

    # ── Row 4: panel e (theta phase) + panel f (decoding error) ──
    gs_ef = outer[3].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)
    ax_e = fig.add_subplot(gs_ef[0])
    plot_state_probability_by_theta_phase(data_over_time, ax=ax_e)

    ax_f = fig.add_subplot(gs_ef[1])
    plot_decoding_error_distribution(data_over_time, ax=ax_f)

    # ── Panel labels ──
    # Larger font for readability in a two-column figure; push slightly
    # further outside the axes so they don't collide with plot content.
    label_panels(
        [first_column_axes[0], ax_b, ax_c_top, ax_d_start, ax_e, ax_f],
        labels=["a", "b", "c", "d", "e", "f"],
        xpos=-0.08,
        ypos=1.15,
        fontsize=11,
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
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
        window_sec=args.window_sec,
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
