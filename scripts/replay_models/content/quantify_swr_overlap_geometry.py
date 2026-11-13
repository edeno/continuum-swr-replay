"""Quantify the non-local/SWR overlap sub-cases on real data.

Runs ``classify_swr_overlap_geometry`` over one or more sessions and prints
a table of sub-case counts. Useful for Methods-section numbers and for
auditing what the coarse three-way split aggregates over. The SWRs are the
paper's SWR source (``analysis/event_definition/swr_source.py``: the direct Kay
inventory), on the decode grid, held to the one analysed-time rule as the
decode frame's read holds them. The non-local events are re-detected from the
posterior, so the rule is applied here (``event_definition.eligibility``): an event in
excluded time (the mask widened by 0.3 s), or overlapping an SWR or MUA event
dropped for it, is not counted.

Usage:
    uv run python scripts/replay_models/content/quantify_swr_overlap_geometry.py
    uv run python scripts/replay_models/content/quantify_swr_overlap_geometry.py \\
        --nwb wilbur20210329_.nwb --epoch 02_r1
    uv run python scripts/replay_models/content/quantify_swr_overlap_geometry.py --animal chimi
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.nonlocal_events import (
    validate_results_position_time_alignment,
)
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    dropped_for_excluded_time,
    read_artifact_event_drops,
    swr_event_labels,
)
from continuum_swr_replay.analysis.event_definition.eligibility import (
    artifact_drop_reasons,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    build_speed_eligible_event_cohort,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    classify_swr_overlap_geometry,
    get_confident_times,
    nonlocal_state_probability,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    artifact_intervals,
    swr_epoch_id,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import nwb_stem
from continuum_swr_replay.decoding.results import load_results
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.paths import processed_data_dir

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


def quantify_session(
    nwb_file_name: str,
    epoch_name: str,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load one session and return per-event geometry classification."""
    if data_dir is None:
        data_dir = processed_data_dir()
    logger.info(f"Loading {nwb_file_name} {epoch_name}...")
    data = load_data(nwb_file_name, epoch_name)
    results, _, _ = load_results(
        nwb_file_name,
        epoch_name,
        decoding_brain_area="HPC",
        load_detector=False,
    )

    position_time = data["position_info"].index
    speed = data["position_info"].head_speed.to_numpy()
    validate_results_position_time_alignment(results.time.values, position_time)
    non_local_prob = nonlocal_state_probability(
        results.acausal_state_probabilities
    ).assign_coords(time=position_time.to_numpy(dtype=float))

    detected_nl_events = get_confident_times(
        non_local_prob,
        speed,
        position_time.to_numpy(dtype=float),
        DEFAULT_NON_LOCAL_THRESHOLD,
    )
    nl_events, _ = build_speed_eligible_event_cohort(
        detected_nl_events,
        speed=speed,
        time=position_time,
        event_col="non_local_event",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    epoch_id = swr_epoch_id(nwb_file_name, epoch_name)
    # every event of the decode frame, with its reason under the one rule
    drops = read_artifact_event_drops(decode_paths(epoch_id, data_dir).data_over_time)
    reasons = artifact_drop_reasons(
        nl_events["start_time"].to_numpy(),
        nl_events["end_time"].to_numpy(),
        artifact_intervals(epoch_id),
        dropped_for_excluded_time(drops),
    )
    if pd.notna(reasons).any():
        logger.info(
            f"  {int(pd.notna(reasons).sum())} non-local events dropped by the "
            "analysed-time rule"
        )
    nl_events = nl_events.loc[pd.isna(reasons)].reset_index(drop=True)
    # The paper's SWRs: the SWR source's labels on this grid, as sampled bounds,
    # the ones the rule keeps (an SWR overlapping a dropped MUA event is not).
    kept_swrs = drops.loc[
        (drops["event_col"] == "SWR_event") & drops["drop_reason"].isna(), "label"
    ].to_numpy()
    swr_labels = swr_event_labels(position_time, epoch_id, speed=speed)
    swr_labels = np.where(np.isin(swr_labels, kept_swrs), swr_labels, 0)
    swr_events = intervals_from_labels(
        pd.DataFrame({"SWR_event": swr_labels}, index=position_time), "SWR_event"
    )
    logger.info(f"  {len(nl_events)} immobile non-local events detected")
    logger.info(f"  {len(swr_events)} whole-event-immobile SWR events")

    geom = classify_swr_overlap_geometry(nl_events, swr_events)
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
