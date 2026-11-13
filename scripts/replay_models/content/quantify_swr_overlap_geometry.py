"""Quantify the non-local/SWR overlap sub-cases on real data.

Prints a table of the sub-case counts of one session's non-local events.
Useful for Methods-section numbers and for auditing what the coarse three-way
split aggregates over. The events and their classes are the paper's: the rows
of the canonical event table (``replay_content/event_table.parquet``), whose
``overlap_class`` and ``overlap_subclass`` the classified pass
(``event_definition.classified_table.build_classified_nonlocal_event_table``)
derived against the paper's SWR source, after the one analysed-time rule.

Usage:
    uv run python scripts/replay_models/content/quantify_swr_overlap_geometry.py
    uv run python scripts/replay_models/content/quantify_swr_overlap_geometry.py \\
        --nwb wilbur20210329_.nwb --epoch 02_r1
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import read_swr_table
from continuum_swr_replay.decoding.identifiers import nwb_stem, parse_nwb_filename
from continuum_swr_replay.paths import table_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SUBCLASS_ORDER = [
    "no_overlap",
    "event_contains_swr",
    "single_swr_partial",
    "multi_swr_no_gap",
    "multi_swr_gap",
    "swr_contains_event",
]
SUBCLASS_DESCRIPTIONS = {
    "no_overlap": "no SWR overlaps the event",
    "event_contains_swr": "event strictly contains >= 1 SWR",
    "single_swr_partial": "single-SWR partial overlap at a boundary",
    "multi_swr_no_gap": "tiled by >= 2 SWRs with no gap",
    "multi_swr_gap": "overlaps >= 2 SWRs, event has non-SWR timebins",
    "swr_contains_event": "single SWR fully contains the event",
}
COARSE_ORDER = [
    ("swr_contained", "SWR-contained (event inside one SWR)"),
    ("swr_partial", "SWR-partial (event extends past SWR boundaries)"),
    ("non_swr", "non-SWR (no overlap)"),
]


#: The event table's columns the report reads, by the names it prints them under.
GEOMETRY_COLUMNS = {
    "non_local_event": "non_local_event",
    "event_id": "event_id",
    "event_start_s": "start_time",
    "event_end_s": "end_time",
    "event_duration_s": "duration",
    "overlap_class": "overlap_class",
    "overlap_subclass": "overlap_subclass",
    "n_swrs_overlapping": "n_swrs_overlapping",
}


def quantify_session(
    nwb_file_name: str,
    epoch_name: str,
    event_table: Path | None = None,
) -> pd.DataFrame:
    """One session's non-local events and their SWR-overlap geometry.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The session.
    event_table : Path, optional
        The canonical event table; ``table_path("event_table")`` by default.

    Returns
    -------
    pd.DataFrame, shape (n_events, 10)
        The session's rows of the event table in time order, with the
        ``GEOMETRY_COLUMNS`` (renamed) and ``nwb_file_name``, ``epoch_name``.
    """
    if event_table is None:
        event_table = table_path("event_table")
    animal, day = parse_nwb_filename(nwb_file_name)
    geom = read_swr_table(
        event_table,
        columns=list(GEOMETRY_COLUMNS),
        filters=[
            ("animal", "==", animal),
            ("session_date", "==", day),
            ("epoch", "==", epoch_name),
        ],
    )
    geom = (
        geom.rename(columns=GEOMETRY_COLUMNS)
        .sort_values("start_time")
        .reset_index(drop=True)
    )
    logger.info(f"  {len(geom)} immobile non-local events in the event table")
    geom["nwb_file_name"] = nwb_file_name
    geom["epoch_name"] = epoch_name
    return geom


def print_report(geom: pd.DataFrame, session_label: str) -> None:
    """Print a single-session count table."""
    n_events = len(geom)
    counts = geom["overlap_subclass"].value_counts()
    print()
    print(f"=== {session_label}  ({n_events} events) ===")
    name_col = max(len(s) for s in SUBCLASS_ORDER)
    print(f"{'subclass':>{name_col}} {'n':>6} {'pct':>7}   description")
    for sub in SUBCLASS_ORDER:
        n = int(counts.get(sub, 0))
        pct = 100.0 * n / n_events if n_events else 0.0
        print(f"{sub:>{name_col}} {n:>6} {pct:>6.1f}%   {SUBCLASS_DESCRIPTIONS[sub]}")

    # Coarse rollup.
    print()
    coarse_counts = geom["overlap_class"].value_counts()
    coarse_col = max(len(k) for k, _ in COARSE_ORDER)
    for coarse, label in COARSE_ORDER:
        n = int(coarse_counts.get(coarse, 0))
        pct = 100.0 * n / n_events if n_events else 0.0
        print(f"{coarse:>{coarse_col}} {n:>6} {pct:>6.1f}%   {label}")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nwb",
        default="wilbur20210329_.nwb",
        help="NWB filename (default: wilbur20210329_.nwb, the Fig 2 session)",
    )
    p.add_argument("--epoch", default="02_r1", help="Epoch name (default: 02_r1)")
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional: save per-event geometry CSV to this path",
    )
    args = p.parse_args(argv)

    geom = quantify_session(args.nwb, args.epoch)
    session_label = f"{nwb_stem(args.nwb)} {args.epoch}"
    print_report(geom, session_label)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        geom.to_csv(args.save, index=False)
        logger.info(f"Wrote per-event geometry to {args.save}")


if __name__ == "__main__":
    main()
