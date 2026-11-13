"""The files the behavior summary reads, and whether the fits among them are
current: :func:`fit_staleness` reads a fit's provenance sidecar, and
:func:`_read_per_bin_fits` refuses a per-bin fit table from the old design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.content.estimator_comparison import (
    BOOTSTRAP_BASENAME,
    SUMMARY_BASENAME,
)
from continuum_swr_replay.analysis.behavior.content.fit_outputs import (
    belief_chain_output_subdir,
)
from continuum_swr_replay.analysis.behavior.content.uncertainty_content import (
    HORSERACE_SUMMARY,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    PER_BIN_DESIGN,
)
from continuum_swr_replay.analysis.behavior.rate.ridge_history_cv import (
    RIDGE_CV_FILE,
)
from continuum_swr_replay.paths import (
    CONTENT_ESTIMATOR_COMPARISON_DIR,
    CROSS_HISTORY_RIDGE_CV_DIR,
    EVENT_SOURCE_COMPARISON_DIR,
    UNCERTAINTY_CONTENT_DIR,
    event_source_comparison_fits_name,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    rel,
)
from continuum_swr_replay.runs.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
    recorded_input_paths,
)


@dataclass(frozen=True)
class BehaviorInputs:
    """The files the behavior summary reads."""

    per_bin: Path
    per_bin_all_stops: Path
    per_bin_in_transit: Path
    rate_contrast: Path
    per_bin_audit: Path
    per_stop_audit: Path
    per_stop: Path
    content_contrast_dir: Path
    estimator_summary: Path
    estimator_bootstrap: Path
    leaf_dir: Path
    choice_fits: Path
    stay_leave_dir: Path
    swr_power: Path
    event_table: Path
    stop_table: Path
    onset_table: Path
    event_arm_table: Path
    duration_aware_pooled: Path
    cross_ridge_cv: Path
    horserace_summary: Path


def default_inputs() -> BehaviorInputs:
    rc = replay_content_dir()
    estimators = rc / CONTENT_ESTIMATOR_COMPARISON_DIR
    return BehaviorInputs(
        per_bin=model_dir("per_bin_rate") / "all_models.pkl",
        per_bin_all_stops=model_dir("per_bin_rate") / "all_models_allstops.pkl",
        per_bin_in_transit=model_dir("per_bin_rate") / "all_models_intransit.pkl",
        rate_contrast=model_dir("per_bin_rate") / "swr_contrast_rate.pkl",
        per_bin_audit=model_dir("per_bin_rate") / "per_bin_rate_cohort_audit.csv",
        per_stop_audit=model_dir("per_stop_rate") / "all_models_cohort_audit.csv",
        per_stop=model_dir("per_stop_rate") / "all_models.pkl",
        content_contrast_dir=rc / "models_swr_contrast",
        estimator_summary=estimators / f"{SUMMARY_BASENAME}.csv",
        estimator_bootstrap=estimators / f"{BOOTSTRAP_BASENAME}.csv",
        leaf_dir=rc / belief_chain_output_subdir(),
        choice_fits=rc
        / EVENT_SOURCE_COMPARISON_DIR
        / event_source_comparison_fits_name("non_local", "HPC"),
        stay_leave_dir=rc,
        swr_power=model_dir("per_event_swr_power") / "swr_power_model.pkl",
        event_table=table_path("event_table", rc),
        stop_table=table_path("stop_table", rc),
        onset_table=table_path("event_onset_table", rc),
        event_arm_table=table_path("event_arm_table", rc),
        duration_aware_pooled=model_dir("per_bin_rate")
        / "duration_aware_history_pooled.pkl",
        cross_ridge_cv=model_dir("per_bin_rate")
        / CROSS_HISTORY_RIDGE_CV_DIR
        / RIDGE_CV_FILE,
        horserace_summary=rc / UNCERTAINTY_CONTENT_DIR / HORSERACE_SUMMARY,
    )


def fit_staleness(*artifacts: Path) -> str | None:
    """Why the values read from fitted ``artifacts`` are pending, or None.

    A fit's provenance sidecar records the SHA-256 of each table and upstream
    fit it read. A fit with no sidecar, one that records no input, or one that
    records other inputs than the current ones, cannot be tied to the current
    data.
    """
    notes = []
    for artifact in artifacts:
        sidecar = fit_provenance_path(artifact)
        if not sidecar.exists():
            notes.append(
                f"{rel(artifact)} has no provenance record; refit before citing"
            )
            continue
        inputs = recorded_input_paths(artifact)
        if not inputs:
            notes.append(
                f"{rel(artifact)} records no input it was fit on; refit before citing"
            )
            continue
        stale = check_fit_provenance(artifact, inputs)
        if stale:
            notes.append(
                f"{rel(artifact)} was fit on other inputs than the current ones "
                f"({', '.join(stale)}); refit before citing"
            )
    return "; ".join(notes) or None


def _read_per_bin_fits(path: Path) -> pd.DataFrame:
    """A per-bin rate fit table, refused unless every row is on the at-risk design.

    Fits from the old all-bins, onset-history design carry no
    ``per_bin_design`` column; their ``hist_rcos`` terms are not the
    ``PER_BIN_HISTORY_COLUMNS`` this module reads.
    """
    fits = pd.read_pickle(path)
    if "per_bin_design" not in fits or (fits["per_bin_design"] != PER_BIN_DESIGN).any():
        raise ValueError(
            f"{path}: not fit on the {PER_BIN_DESIGN!r} per-bin design; refit it"
        )
    return fits
