"""The files the mpfc summary reads, and whether the unit bundles are current:
each bundle's provenance sidecar records the event and stop tables it was fit
on (:func:`staleness`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import sha256_file

STEM = "_mPFC_modulation"
"""The canonical (at-well, sigma 3) bundle name; other cohorts and priors carry
a suffix after it."""


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


def read_sidecars(bundle_dir: Path) -> list[dict[str, Any]]:
    """Every canonical bundle's provenance sidecar, in session order.

    Raises
    ------
    ValueError
        A bundle has no sidecar or a sidecar no bundle.
    """
    bundles = {
        p.name.removesuffix(f"{STEM}.pkl") for p in bundle_dir.glob(f"*{STEM}.pkl")
    }
    paths = sorted(bundle_dir.glob(f"*{STEM}.provenance.json"))
    described = {p.name.removesuffix(f"{STEM}.provenance.json") for p in paths}
    if not bundles or bundles != described:
        raise ValueError(
            f"bundles and sidecars in {bundle_dir} differ: "
            f"{sorted(bundles ^ described)}"
        )
    return [json.loads(p.read_text()) for p in paths]


def staleness(sidecars: list[dict[str, Any]], current: dict[str, str]) -> str | None:
    """Why the bundles are stale against the current tables, or None.

    A bundle records the SHA-256 of the canonical event table and the stop
    table it was fit on; a bundle fit on other tables describes other events.
    ``current`` holds the current tables' (:func:`table_hashes`).
    """
    stale = {
        name: sum(s.get(name) != sha for s in sidecars) for name, sha in current.items()
    }
    return _stale_note("the mPFC bundles", stale, len(sidecars))


def table_hashes(event_table: Path, stop_table: Path) -> dict[str, str]:
    """The SHA-256 of the current canonical event and stop tables, under the
    names the bundles' provenance records them by."""
    return {
        "canonical_event_table_sha256": sha256_file(event_table),
        "stop_table_sha256": sha256_file(stop_table),
    }


def _stale_note(what: str, stale: dict[str, int], total: int) -> str | None:
    """A pending note naming each recorded input that differs, or None."""
    if not any(stale.values()):
        return None
    return (
        f"{what} were made from other inputs than the current ones "
        f"({', '.join(f'{name} differs in {k} of {total}' for name, k in stale.items() if k)}); "
        "refit before citing"
    )
