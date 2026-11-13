#!/usr/bin/env python
"""Compare the PFC path's old, run-time event definition against the canonical one.

The mPFC unit GLM used to build its own non-local events from the decode
posterior (``get_confident_times`` at the non-local threshold, then the
whole-event immobility rule) and to split them on their overlap with the
separately detected ``ripple_times``. It now reads the shipped event table and
that table's own ``is_swr`` column. The old construction survives only here, so
a session can be asked what the change did to it rather than being assumed
equivalent.

Reported per session:

- how many events each definition yields;
- how many events change class (SWR to non-SWR and back), after matching the
  two event lists by temporal overlap;
- the distribution of the start-time shift between matched events;
- how many of the reclassified events are partial-overlap cases -- events the
  old inclusive split and the canonical strict positive-duration rule can
  legitimately disagree on.

Usage::

    uv run python scripts/pfc/diagnose_pfc_event_definition.py \\
        --animal wilbur --date 20210329 --epoch 02_r1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.decoding import identifiers

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / ".claude" / "research" / "pfc_event_diagnostic_outputs"


def build_old_event_sets(
    data: dict,
    results,
    *,
    immobility_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """The retired run-time construction, kept intact for this comparison.

    This is the block ``build_session_event_sets`` used to run: confident
    non-local windows from the decode posterior, the whole-event immobility
    rule on both populations, and the SWR split taken as overlap with
    ``data["ripple_times"]``. It predates the analysed-time rule and applies none
    on purpose, so an event that rule drops counts as an old event without a new
    match.
    """
    from continuum_swr_replay.analysis.events.immobility import (
        build_speed_eligible_event_cohort,
    )
    from continuum_swr_replay.analysis.events.nonlocal_detection import (
        get_confident_times,
        get_overlap_times,
    )
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )

    results, position_info, _ = align_results_to_position(
        results, data["position_info"]
    )
    position_time = position_info.index
    speed = position_info.head_speed.to_numpy()
    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    non_local_prob = non_local_prob.assign_coords(
        time=position_time.to_numpy(dtype=float)
    )

    detected_non_local_times = get_confident_times(
        non_local_prob,
        speed,
        position_time.to_numpy(dtype=float),
        non_local_threshold,
    )
    non_local_times, _ = build_speed_eligible_event_cohort(
        detected_non_local_times,
        speed=speed,
        time=position_time,
        event_col="non_local_event",
        speed_threshold=immobility_threshold,
    )
    ripple_times, _ = build_speed_eligible_event_cohort(
        data["ripple_times"],
        speed=speed,
        time=position_time,
        event_col="SWR_event",
        speed_threshold=immobility_threshold,
    )
    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, ripple_times
    )
    return {
        "non_local": non_local_times.reset_index(drop=True),
        "SWR": ripple_times.reset_index(drop=True),
        "non_local_SWR": non_local_swr_times,
        "non_local_no_SWR": non_local_no_swr_times,
    }


def total_overlap_seconds(
    starts: np.ndarray,
    ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Seconds each ``[start, end]`` shares with the union of the other set.

    Shapes ``(n_events,)`` in, ``(n_events,)`` out. The other intervals are
    assumed non-overlapping, which both SWR populations here are: they are
    derived from disjoint label runs.
    """
    if len(starts) == 0:
        return np.empty(0, dtype=float)
    if len(other_starts) == 0:
        return np.zeros(len(starts), dtype=float)
    overlap = np.minimum(ends[:, None], other_ends[None, :]) - np.maximum(
        starts[:, None], other_starts[None, :]
    )
    return np.clip(overlap, 0.0, None).sum(axis=1)


def match_by_overlap(new_events: pd.DataFrame, old_events: pd.DataFrame) -> np.ndarray:
    """Index into ``old_events`` of each new event's best temporal match.

    ``-1`` where no old event overlaps at all. Greatest shared duration wins,
    so a new event spanning two old ones is attributed to the larger.
    """
    if len(new_events) == 0:
        return np.empty(0, dtype=int)
    if len(old_events) == 0:
        return np.full(len(new_events), -1, dtype=int)
    overlap = np.minimum(
        new_events.end_time.to_numpy(float)[:, None],
        old_events.end_time.to_numpy(float)[None, :],
    ) - np.maximum(
        new_events.start_time.to_numpy(float)[:, None],
        old_events.start_time.to_numpy(float)[None, :],
    )
    overlap = np.clip(overlap, 0.0, None)
    best = overlap.argmax(axis=1)
    best[overlap.max(axis=1) <= 0.0] = -1
    return best


def build_comparison(
    nwb_file_name: str,
    epoch_name: str,
    *,
    immobility_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Per-event old-vs-new table and its summary, for one session."""
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_canonical_events,
        load_session_swr_intervals,
    )
    from continuum_swr_replay.data_loaders import load_data

    results, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=False
    )
    data = load_data(nwb_file_name, epoch_name)

    old = build_old_event_sets(
        data,
        results,
        immobility_threshold=immobility_threshold,
        non_local_threshold=non_local_threshold,
    )
    old_events = old["non_local"]
    old_is_swr = np.isin(
        old_events.start_time.to_numpy(float),
        old["non_local_SWR"].start_time.to_numpy(float),
    )

    new_events, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    new_swr, _ = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=non_local_threshold,
        speed_threshold=immobility_threshold,
    )

    matched = match_by_overlap(new_events, old_events)
    has_match = matched >= 0
    new_start = new_events.start_time.to_numpy(float)
    new_end = new_events.end_time.to_numpy(float)
    new_is_swr = new_events.is_swr.to_numpy(bool)

    old_start = np.where(
        has_match, old_events.start_time.to_numpy(float)[matched], np.nan
    )
    old_end = np.where(has_match, old_events.end_time.to_numpy(float)[matched], np.nan)
    matched_old_is_swr = np.where(has_match, old_is_swr[matched], False)

    canonical_overlap = total_overlap_seconds(
        new_start,
        new_end,
        new_swr.start_time.to_numpy(float),
        new_swr.end_time.to_numpy(float),
    )
    old_ripple_overlap = total_overlap_seconds(
        new_start,
        new_end,
        old["SWR"].start_time.to_numpy(float),
        old["SWR"].end_time.to_numpy(float),
    )
    duration = new_end - new_start
    with np.errstate(divide="ignore", invalid="ignore"):
        canonical_frac = np.where(duration > 0, canonical_overlap / duration, np.nan)

    per_event = pd.DataFrame(
        {
            "event_id": new_events.event_id.to_numpy(),
            "new_start_time": new_start,
            "new_end_time": new_end,
            "new_is_swr": new_is_swr,
            "occupancy_state": new_events.occupancy_state.to_numpy(),
            "has_old_match": has_match,
            "old_start_time": old_start,
            "old_end_time": old_end,
            "old_is_swr": np.where(has_match, matched_old_is_swr, np.nan),
            "start_shift_s": np.where(has_match, new_start - old_start, np.nan),
            "canonical_swr_overlap_s": canonical_overlap,
            "canonical_swr_overlap_frac": canonical_frac,
            "old_ripple_overlap_s": old_ripple_overlap,
            "reclassified": has_match & (new_is_swr != matched_old_is_swr),
            "is_partial_swr_overlap": (canonical_overlap > 0) & (canonical_frac < 1.0),
        }
    )

    shifts = per_event.loc[per_event.has_old_match, "start_shift_s"].to_numpy(float)
    reclassified = per_event.reclassified.to_numpy(bool)
    summary = {
        "nwb_file_name": nwb_file_name,
        "epoch_name": epoch_name,
        "n_events_old": int(len(old_events)),
        "n_events_new": int(len(new_events)),
        "n_swr_intervals_old": int(len(old["SWR"])),
        "n_swr_intervals_new": int(len(new_swr)),
        "n_swr_old": int(old_is_swr.sum()),
        "n_swr_new": int(new_is_swr.sum()),
        "n_new_without_old_match": int((~has_match).sum()),
        "n_old_without_new_match": int(
            len(old_events) - len(np.unique(matched[has_match]))
        ),
        "n_reclassified": int(reclassified.sum()),
        "n_reclassified_swr_to_non_swr": int((reclassified & ~new_is_swr).sum()),
        "n_reclassified_non_swr_to_swr": int((reclassified & new_is_swr).sum()),
        "n_reclassified_partial_overlap": int(
            (reclassified & per_event.is_partial_swr_overlap.to_numpy(bool)).sum()
        ),
        "start_shift_n_exact": int(np.sum(shifts == 0.0)) if shifts.size else 0,
        "start_shift_min_s": float(np.min(shifts)) if shifts.size else np.nan,
        "start_shift_median_s": float(np.median(shifts)) if shifts.size else np.nan,
        "start_shift_max_s": float(np.max(shifts)) if shifts.size else np.nan,
        "start_shift_abs_p95_s": (
            float(np.percentile(np.abs(shifts), 95)) if shifts.size else np.nan
        ),
    }
    return per_event, summary


def _print_summary(summary: dict) -> None:
    width = max(len(key) for key in summary)
    print(f"\n{'-' * (width + 24)}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<{width}}  {value:>20.6g}")
        else:
            print(f"{key:<{width}}  {value:>20}")
    print("-" * (width + 24))


def get_command_line_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--animal", type=str, required=True, help="e.g. wilbur")
    parser.add_argument("--date", type=int, required=True, help="e.g. 20210329")
    parser.add_argument("--epoch", type=str, required=True, help="e.g. 02_r1")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the per-event CSV (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--immobility-threshold",
        type=float,
        default=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        help=(
            "Speed threshold in cm/s for the whole-event immobility rule: the "
            "canonical value or a stricter (lower) one"
        ),
    )
    parser.add_argument(
        "--non-local-threshold",
        type=float,
        default=DEFAULT_NON_LOCAL_THRESHOLD,
        help="Posterior threshold the old construction detected events at",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = get_command_line_arguments(argv)
    nwb_file_name = identifiers.nwb_file_name(args.animal, args.date)

    per_event, summary = build_comparison(
        nwb_file_name,
        args.epoch,
        immobility_threshold=args.immobility_threshold,
        non_local_threshold=args.non_local_threshold,
    )

    _print_summary(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (
        f"{identifiers.session_name(nwb_file_name, args.epoch)}_event_definition.csv"
    )
    per_event.to_csv(out_path, index=False)
    print(f"Wrote {len(per_event)} rows to {out_path}")


if __name__ == "__main__":
    main()
