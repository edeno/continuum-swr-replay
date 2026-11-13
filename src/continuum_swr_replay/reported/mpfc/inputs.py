"""The files the mpfc summary reads, and whether the unit bundles are current:
each bundle's provenance sidecar records the event and stop tables it was fit
on (:func:`staleness`, counted by ``runs.staleness.stale_counts``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continuum_swr_replay.paths import (
    PFC_MARKED_FAMILY_DIR,
    PFC_MARKED_FAMILY_SUMMARY,
    PFC_STOP_CONTEXT_DIR,
    PFC_STOP_CONTEXT_RESULTS,
    PFC_SWR_WITHOUT_REPLAY_DIR,
    PFC_TIMING_CALIBRATION,
    pfc_cohort_name,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.staleness import stale_counts


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
        pfc_cohort=data / pfc_cohort_name("mPFC"),
        event_table=table_path("event_table", replay_content_dir()),
        stop_table=table_path("stop_table", replay_content_dir()),
        timing_calibration=data / PFC_TIMING_CALIBRATION,
        swr_without_replay_dir=data / PFC_SWR_WITHOUT_REPLAY_DIR,
        marked_family_summary=data / PFC_MARKED_FAMILY_DIR / PFC_MARKED_FAMILY_SUMMARY,
        stop_context_results=data / PFC_STOP_CONTEXT_DIR / PFC_STOP_CONTEXT_RESULTS,
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
