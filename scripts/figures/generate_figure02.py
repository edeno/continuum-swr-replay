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

Panels b and d are drawn from the canonical event table
(``replay_content/event_table.parquet``), the table the events summary cites,
by the functions the summary uses (``analysis.hippocampus.event_statistics``):
panel b counts its ``overlap_class`` and panel d subtracts ``matched_swr_*``
from the event bounds. The percents and medians printed on them are those
functions' numbers, rounded by the summary's rule (``reported.formatting``),
so a panel prints the digits the text cites.

Panels a, c and e are drawn from the decode frames. Panel a picks its
examples from ``build_classified_nonlocal_event_table``, computed once over the
aggregated ``data_over_time`` (one row per non-local event, full label-window
timing, strict positive-duration SWR overlap, matched SWR); panel c keeps
``get_event_types`` (parity-tested against the same table) for its
event-triggered windows. The class counts of the events classified from the
frames must equal the canonical table's, or the figure is not drawn: the table
was then built from other decode frames than the ones read here. Equal counts
do not prove that the two hold the same events.

Remaining validation panels (non-local probability threshold,
calibration curve, full four-signal triggered averages, moving-vs-
immobile state fractions, state probability vs theta phase) live in the
supplementary "Decoder validation" figure.

Usage
-----
    python scripts/figures/generate_figure02.py
    python scripts/figures/generate_figure02.py --nwb-file-name wilbur20210329_.nwb --epoch-name 02_r1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import continuum_swr_replay.visualization.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.events.analysed_time import swr_intervals
from continuum_swr_replay.analysis.events.classified_table import (
    assert_drawn_class_matches_ripples,
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.events.nonlocal_detection import (
    select_typical_event_by_class,
)
from continuum_swr_replay.analysis.events.swr_source import (
    ripple_power_inputs,
    swr_epoch_id,
    swr_source_name,
)
from continuum_swr_replay.analysis.hippocampus import (
    compute_local_decoding_error,
    get_event_types,
    load_data_over_time,
    load_results,
)
from continuum_swr_replay.analysis.hippocampus.event_statistics import (
    boundary_offset_statistics,
    boundary_offsets,
    compute_swr_overlap_class_counts,
    overlap_class_statistics,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.epoch_register import (
    analysed_decode_files,
    analysed_epoch_ids,
)
from continuum_swr_replay.decoding.files import data_over_time_suffix, decode_paths
from continuum_swr_replay.decoding.identifiers import (
    format_epoch_identifier,
    parse_nwb_filename,
)
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.visualization.figure_panels import (
    compute_triggered_average_curves,
    plot_decoding_error_distribution,
    plot_event_example_stack,
    plot_event_timing_offsets,
    plot_swr_overlap_class_bar,
    plot_triggered_averages,
)
from continuum_swr_replay.visualization.figure_utilities import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    label_panels,
    save_figure,
)
from continuum_swr_replay.visualization.plots import (
    plot_linear_track_graph_strip,
    plot_track_graph_2d,
)

# Signals shown in panel c (triggered averages). Theta/MUA live in the
# supplementary decoder-validation figure.
TRIGGERED_SIGNALS = ["Non-Local_prob", "SWR_zscore"]

# The render cache's version, part of its file name. The name also carries the
# SWR source's name, a fingerprint of the epoch register's analysed epochs
# (the cohort panels b-e summarise) and a fingerprint of the decode files read
# (``_decode_files_fingerprint``: path, size and mtime), so a new SWR inventory,
# artifact run, register change or rewritten decode makes a new cache by itself.
# Bump this whenever the cached values could change for any other reason: a
# change to ``_compute_figure_inputs`` or to a library function it calls (the
# read through the SWR source, the event classification, the triggered
# averages, ``compute_local_decoding_error``). Without a bump the figure is
# drawn from the old cache, unless ``--force-recompute`` is passed.
# 1: the cache named ``..._figure02_whole_event_speed_v1_...``.
# 2: panel e's decoding error leaves excluded time out.
# 3: panels b and d are read from the canonical event table when the figure is
#    drawn; the cache keeps the class counts of the events classified from the
#    decode frames, to check against that table.
FIGURE02_CACHE_VERSION = 3

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


def _decode_files_fingerprint(
    nwb_file_name: str, epoch_name: str, data_dir: Path
) -> str:
    """SHA-256 over the decode files ``_compute_figure_inputs`` reads.

    The files are the analysed epochs' ``data_over_time`` (as
    ``load_data_over_time`` finds them) and the example session's HPC
    ``results.nc`` and ``detector.pkl`` (as ``load_results`` names them), all
    in `data_dir`. Each contributes its path, size and ``st_mtime_ns``, or
    ``missing``, so the fingerprint is a stat per file, not a read.
    """
    example = decode_paths(
        format_epoch_identifier(nwb_file_name, epoch_name, "HPC"), data_dir
    )
    files = [
        *analysed_decode_files(data_dir, data_over_time_suffix()),
        example.results,
        example.detector,
    ]
    lines = []
    for path in files:
        if path.exists():
            stat = path.stat()
            lines.append(f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}")
        else:
            lines.append(f"{path}\tmissing")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _figure02_cache_path(nwb_file_name: str, epoch_name: str, data_dir: Path) -> Path:
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
    decodes = _decode_files_fingerprint(nwb_file_name, epoch_name, data_dir)
    return (
        data_dir
        / "intermediates"
        / f"{animal}{day}_{epoch_name}_figure02_v{FIGURE02_CACHE_VERSION}_"
        f"{swr_source_name()}_register{cohort[:12]}_decodes{decodes[:12]}"
        "_cache.joblib"
    )


def _compute_figure_inputs(
    nwb_file_name: str,
    epoch_name: str,
    data_dir: Path,
) -> dict:
    """Run the expensive loads/derivations and return all render inputs.

    Everything the figure needs is reduced here to a single dict that is
    cheap to cache: the example session's ``data``/``results``/``detector``
    /``env`` plus the chosen example events (panel a), the triggered-average
    curves (panel c), the per-session decoding-error table (panel e), and the
    overlap-class counts of the events classified from the decode frames, which
    :func:`event_table_panels` checks the canonical event table against.
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
        nwb_file_name, epoch_name, decoding_brain_area="HPC", root=data_dir
    )

    logging.info("Loading aggregated data for stats panels...")
    data_over_time = load_data_over_time(data_dir)
    # Panel a's SWR shading: the session's SWR source intervals, the ones the
    # classes below are computed against (not load_data's stored ripples).
    data["ripple_times"] = swr_intervals(
        data_over_time, nwb_file_name=nwb_file_name, epoch_name=epoch_name
    )
    event_types = get_event_types(data_over_time)

    # The events classified from the decode frames: the unfiltered classified
    # table with full label-window timing, computed once over the
    # multi-session aggregate. Panel a picks its examples from it, and its
    # class counts are what the canonical event table (panels b and d) is
    # checked against. Panel c stays on ``get_event_types`` (parity-tested
    # against this table).
    classified_events_all = build_classified_nonlocal_event_table(data_over_time)
    classified_class_counts = compute_swr_overlap_class_counts(classified_events_all)
    logging.info(
        "Classified %d non-local events; %d pass whole-event immobility; "
        "by class: contained=%d, partial=%d, non_swr=%d",
        len(classified_events_all),
        classified_class_counts["total"],
        classified_class_counts["swr_contained"],
        classified_class_counts["swr_partial"],
        classified_class_counts["non_swr"],
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

    logging.info("Computing triggered-average curves (panel c)...")
    triggered_curves = compute_triggered_average_curves(
        data_over_time,
        event_types,
        TRIGGERED_SIGNALS,
        time_col="start_time",
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
        "classified_class_counts": classified_class_counts,
        "triggered_curves": triggered_curves,
        "error_df": error_df,
    }


def event_table_panels(
    events: pd.DataFrame, classified_class_counts: dict[str, int]
) -> dict:
    """Panels b and d, from the canonical event table.

    The table is checked against the decode frames by its class counts alone.
    Equal counts do not prove that the two hold the same events: a table built
    from other frames that happen to give the same three counts is not refused.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table (``replay_content/event_table.parquet``): one
        row per immobile non-local event.
    classified_class_counts : dict
        The overlap-class counts of the events classified from the decode
        frames (:func:`compute_swr_overlap_class_counts`), which panels a, c
        and e are drawn from.

    Returns
    -------
    panels : dict
        - ``class_counts``: the events of each overlap class, and their
          ``total`` (panel b's bar).
        - ``class_percents``: each class's share of the events as a whole
          percent, printed as the events summary prints ``non_swr_fraction``,
          without the percent sign (panel b's labels).
        - ``offsets``: one row per SWR-overlapping event, with
          ``start_offset_s`` and ``end_offset_s``, its start and end minus its
          matched SWR's, in seconds (panel d's histograms).
        - ``offset_medians``: for each of those columns, its median in whole
          milliseconds, printed as the events summary prints
          ``start_offset_median_ms`` and ``end_offset_median_ms``, with a plus
          sign when positive (panel d's labels).

    Raises
    ------
    ValueError
        If an event of the table is not immobile as a whole, or the table's
        class counts are not those of the events classified from the decode
        frames.
    """
    classes = overlap_class_statistics(events)
    class_counts = {cls: classes[f"n_{cls}"] for cls in CLASS_ORDER}
    class_counts["total"] = classes["n_events"]
    if class_counts != classified_class_counts:
        raise ValueError(
            "the canonical event table's overlap-class counts are not those of "
            f"the events classified from the decode frames (table {class_counts}, "
            f"frames {classified_class_counts}); rebuild the canonical tables from "
            "these decode frames before drawing the figure"
        )
    offsets = boundary_offsets(events)
    medians = boundary_offset_statistics(offsets)
    return {
        "class_counts": class_counts,
        "class_percents": {
            cls: format_number(class_counts[cls] / class_counts["total"], "pctnum")
            for cls in CLASS_ORDER
        },
        "offsets": offsets,
        "offset_medians": {
            f"{bound}_offset_s": _with_plus(
                format_number(1000 * medians[f"{bound}_offset_median_s"], "int")
            )
            for bound in ("start", "end")
        },
    }


def _with_plus(text: str) -> str:
    """A printed offset with a plus sign when it is positive: the formatter
    writes a negative with a true minus, and a positive and a zero bare."""
    return text if text.startswith("−") or text == "0" else f"+{text}"


def generate_figure(
    nwb_file_name: str = "wilbur20210329_.nwb",
    epoch_name: str = "02_r1",
    data_dir: Path | None = None,
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
    if data_dir is None:
        data_dir = processed_data_dir()
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

    event_table = table_path(
        "event_table",
        data_dir / replay_content_dir().relative_to(processed_data_dir()),
    )
    logging.info(f"Reading panels b and d from {event_table}...")
    panels = event_table_panels(
        pd.read_parquet(event_table), inputs["classified_class_counts"]
    )
    class_counts = panels["class_counts"]
    logging.info(
        f"  {class_counts['total']:,} immobile NL events; "
        f"SWR-contained {class_counts['swr_contained']:,} / "
        f"SWR-partial {class_counts['swr_partial']:,} / "
        f"non-SWR {class_counts['non_swr']:,}; "
        f"{len(panels['offsets']):,} overlap an SWR"
    )
    return _render_figure({**inputs, **panels}, window_sec=window_sec)


def _render_figure(inputs: dict, window_sec: float = 0.4):
    data = inputs["data"]
    results = inputs["results"]
    detector = inputs["detector"]
    env = inputs["env"]
    chosen = inputs["chosen"]
    class_counts = inputs["class_counts"]
    class_percents = inputs["class_percents"]
    offsets = inputs["offsets"]
    offset_medians = inputs["offset_medians"]
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
    plot_swr_overlap_class_bar(class_counts, class_percents, ax=ax_b)

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
    plot_event_timing_offsets(offsets, offset_medians, axes=[ax_d_start, ax_d_end])
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


def main(argv: Sequence[str] | None = None) -> None:
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
    parser.add_argument("--data-dir", type=Path, default=processed_data_dir())
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

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = generate_figure(
        nwb_file_name=args.nwb_file_name,
        epoch_name=args.epoch_name,
        data_dir=args.data_dir,
        window_sec=args.window_sec,
        use_cache=not args.force_recompute,
    )

    logging.info(f"Saving to {args.output_dir}...")
    save_figure(fig, "figure02", base_dir=args.output_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logging.info("Done.")


if __name__ == "__main__":
    main()
