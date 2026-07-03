"""Quantify the non-local/SWR overlap sub-cases on real data.

Runs ``classify_swr_overlap_geometry`` over one or more sessions and prints
a table of sub-case counts. Useful for Methods-section numbers and for
auditing what the coarse three-way split aggregates over.

Usage:
    uv run python scripts/content/quantify_swr_overlap_geometry.py
    uv run python scripts/content/quantify_swr_overlap_geometry.py \\
        --nwb wilbur20210329_.nwb --epoch 02_r1
    uv run python scripts/content/quantify_swr_overlap_geometry.py --animal chimi
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.hippocampus import (
    classify_swr_overlap_geometry,
    get_confident_times,
    load_results,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

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
    "no_overlap":          "no SWR overlaps the event",
    "event_contains_swr":  "event strictly contains >= 1 SWR",
    "single_swr_partial":  "single-SWR partial overlap at a boundary",
    "multi_swr_no_gap":    "tiled by >= 2 SWRs with no gap",
    "multi_swr_gap":       "overlaps >= 2 SWRs, event has non-SWR timebins",
    "swr_contains_event":  "single SWR fully contains the event",
}
COARSE_ORDER = [
    ("swr_contained", "SWR-contained (event inside one SWR)"),
    ("swr_partial",   "SWR-partial (event extends past SWR boundaries)"),
    ("non_swr",       "non-SWR (no overlap)"),
]


def quantify_session(
    nwb_file_name: str,
    epoch_name: str,
    data_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load one session and return per-event geometry classification."""
    logger.info(f"Loading {nwb_file_name} {epoch_name}...")
    data = load_data(nwb_file_name, epoch_name)
    results, _, _ = load_results(
        nwb_file_name,
        epoch_name,
        decoding_brain_area="HPC",
        load_detector=False,
    )

    non_local_prob = (
        results.acausal_state_probabilities.sel(states="Non-Local Continuous")
        + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    )

    speed = data["position_info"].head_speed.to_numpy()
    time = data["position_info"].index.to_numpy()

    nl_events = get_confident_times(
        non_local_prob, speed, time, DEFAULT_NON_LOCAL_THRESHOLD,
    )
    nl_events = nl_events[
        nl_events.max_speed < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    ].reset_index(drop=True)
    logger.info(f"  {len(nl_events)} immobile non-local events detected")
    logger.info(f"  {len(data['ripple_times'])} SWR events")

    geom = classify_swr_overlap_geometry(nl_events, data["ripple_times"])
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
        print(
            f"{sub:>{name_col}} {n:>6} {pct:>6.1f}%   {SUBCLASS_DESCRIPTIONS[sub]}"
        )

    # Coarse rollup.
    print()
    coarse_counts = geom["overlap_class"].value_counts()
    coarse_col = max(len(k) for k, _ in COARSE_ORDER)
    for coarse, label in COARSE_ORDER:
        n = int(coarse_counts.get(coarse, 0))
        pct = 100.0 * n / n_events if n_events else 0.0
        print(f"{coarse:>{coarse_col}} {n:>6} {pct:>6.1f}%   {label}")


def main() -> None:
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
    args = p.parse_args()

    geom = quantify_session(args.nwb, args.epoch)
    session_label = f"{args.nwb.replace('_.nwb', '')} {args.epoch}"
    print_report(geom, session_label)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        geom.to_csv(args.save, index=False)
        logger.info(f"Wrote per-event geometry to {args.save}")


if __name__ == "__main__":
    main()
