"""The files the mpfc summary reads, and whether the unit bundles are current:
each bundle's provenance sidecar records the event and stop tables it was fit
on (:func:`staleness`, counted by ``runs.staleness.stale_counts``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.staleness import stale_counts

TIMING_CALIBRATION = "pfc_response_followups/timing_calibration.json"
"""The timing calibration ``scripts/pfc/calibrate_pfc_kernel_timing.py`` writes,
relative to the processed-data root."""

SWR_WITHOUT_REPLAY_TREE = "pfc_response_followups/swr_without_replay"
"""The extended-event-set bundles (``run_pfc_analysis_bandit.py --event-set
with_swr_without_replay``), relative to the processed-data root."""

STOP_CONTEXT_RESULTS = "pfc_response_followups/stop_context/results.csv"
"""The stop-level context fits (``scripts/pfc/run_pfc_stop_context.py fit``)."""

MARKED_FAMILY_SUMMARY = (
    "pfc_response_followups/marked_family_well_unknown/summary.parquet"
)
"""The marked-family summary (``scripts/pfc/summarize_pfc_marked_family.py``)
of the follow-up models fit with the per-stop baseline prior and the
``well_unknown`` baseline column (bundle contract 3)."""


@dataclass(frozen=True)
class MpfcInputs:
    """The files the mpfc summary reads."""

    bundle_dir: Path
    pfc_cohort: Path
    event_table: Path
    stop_table: Path
    timing_calibration: Path
    swr_without_replay_dir: Path
    marked_family_summary: Path
    stop_context_results: Path


def default_inputs() -> MpfcInputs:
    data = processed_data_dir()
    return MpfcInputs(
        bundle_dir=data,
        pfc_cohort=data / "pfc_cohort_mPFC.json",
        event_table=table_path("event_table", replay_content_dir()),
        stop_table=table_path("stop_table", replay_content_dir()),
        timing_calibration=data / TIMING_CALIBRATION,
        swr_without_replay_dir=data / SWR_WITHOUT_REPLAY_TREE,
        marked_family_summary=data / MARKED_FAMILY_SUMMARY,
        stop_context_results=data / STOP_CONTEXT_RESULTS,
    )


def staleness(sidecars: list[dict[str, Any]], current: dict[str, str]) -> str | None:
    """Why the bundles are stale against the current tables, or None.

    A bundle records the SHA-256 of the canonical event table and the stop
    table it was fit on; a bundle fit on other tables describes other events.
    ``current`` holds the current tables' (``runs.staleness.table_hashes``).
    """
    return _stale_note(
        "the mPFC bundles", stale_counts(sidecars, current), len(sidecars)
    )


def _stale_note(what: str, stale: dict[str, int], total: int) -> str | None:
    """A pending note naming each recorded input that differs, or None."""
    if not any(stale.values()):
        return None
    return (
        f"{what} were made from other inputs than the current ones "
        f"({', '.join(f'{name} differs in {k} of {total}' for name, k in stale.items() if k)}); "
        "refit before citing"
    )
