#!/usr/bin/env python3
"""Fit the recommended descriptive mPFC marked-kernel model family.

The core family asks whether the peri-replay response varies with spatial
content, represented-arm uncertainty/value/reward history, trajectory
geometry, behavioral context, and graded hippocampal coupling.  Optional
interaction models ask whether current-arm or uncertainty modulation changes
with exploration, reward, or ripple power.  Every family is a joint temporal
model: a target is compared with the otherwise identical full model lacking
only that target's spline block.

This runner produces no p-values and applies no firing-rate, p-value, or
model-choice filter to neurons.  Held-out deviance changes are descriptive
predictive comparisons.  It uses the shipped hippocampal replay-content
tables, but does not decode mPFC population activity.

Examples
--------
Smoke the nine core families for one session::

    python scripts/pfc/run_pfc_marked_model_family.py --animal wilbur \
        --date 20210329 --epoch 02_r1 --max-sessions 1 --allow-cpu \
        --output-dir /tmp/pfc_marked_family_smoke

Run core plus content-interaction sensitivities on GPU 0::

    python scripts/pfc/run_pfc_marked_model_family.py --gpu 0 \
        --sessions worker0.sessions --include-interactions

Run one process per GPU with disjoint session files. Six compact core models
share one feature width; the three content horse-races and three interactions
share the wider tier, so only two scorer shapes compile within a session.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    DEFAULT_PENALTY_SIGMA,
    prepare_heldout_response,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    MARKED_EVENT_CLASSES,
    MIN_MARKED_EVENTS_PER_CLASS,
    prepare_marked_session_context,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CORE_MODEL_NAMES,
    DEFAULT_MARK_PENALTY_SIGMA,
    INTERACTION_MODEL_NAMES,
    MARK_DESCRIPTIONS,
    PROFILED_BATCH_FEATURE_BUDGET,
    RECOMMENDED_MODEL_SPECS,
    MarkedFamilySpec,
    ModelIneligibleError,
    build_marked_family_ladder,
    fit_marked_family_ladder,
    load_session_recommended_event_marks,
)
from continuum_swr_replay.analysis.pfc.runner_support import (
    MAX_NONCONVERGED_UNIT_FRACTION,
    REPORT_HALF_WINDOW_S,
    STOP_COHORT,
    filter_epoch_info,
    is_nonempty_file,
    json_safe,
    session_seed,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    fit_provenance_path,
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import processed_data_dir, table_path
from continuum_swr_replay.runs.batch import setup_logger

OUTPUT_SUBDIR = "pfc_marked_model_family"
ESTIMATOR = "joint_multimark_ridge_poisson_temporal_glm"
INFERENCE = "descriptive_blocked_heldout_leave_one_mark_block_out"
BUNDLE_CONTRACT_VERSION = 2
# ``prepare_marked_session_context`` fits over lags -125..125 samples with an
# 11-function clamped cubic B-spline basis per kernel. Both are pinned into the
# contract here because the contract is built before the session is prepared;
# ``_check_prepared_window`` refuses the fit if the prepared context disagrees.
FIT_HALF_WINDOW_S = 0.25
N_BASIS_PER_KERNEL = 11


def family_artifact_path(
    output_dir: Path, epoch_identifier: str, model_name: str
) -> Path:
    """Return the primary statistics table for one session x model family.

    Parameters
    ----------
    output_dir : Path
        Directory holding the marked-family bundles.
    epoch_identifier : str
        ``{animal}{date}_{epoch}_{brain_area}`` identifier of the session.
    model_name : str
        Name of the fitted model family.

    Returns
    -------
    Path
        Path of the bundle's ``.pkl`` statistics table.
    """
    return output_dir / f"{epoch_identifier}_{model_name}_marked_family.pkl"


def family_companion_paths(artifact: Path, spec: MarkedFamilySpec) -> tuple[Path, ...]:
    """Return every array a complete eligible bundle must carry.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    spec : MarkedFamilySpec
        The fitted model family; one slope curve per event class and
        *scientific* mark, so nuisance marks get no companion file.

    Returns
    -------
    tuple of Path
        The lag axis, and each class's typical and per-target slope curves.
    """
    curves: list[Path] = [artifact.with_suffix(".lag_time.npy")]
    for event_class in MARKED_EVENT_CLASSES:
        curves.append(artifact.with_suffix(f".{event_class}.typical.npy"))
        curves.extend(
            artifact.with_suffix(f".{event_class}.{mark}.slope.npy")
            for mark in spec.scientific_marks
        )
    return tuple(curves)


def build_fit_contract(
    spec: MarkedFamilySpec,
    *,
    penalty_sigma: float,
    mark_penalty_sigma: float | None,
    n_folds: int,
    fold_seed: int,
    design_batch_size: int,
    min_marked_events: int,
    brain_area: str,
    input_hashes: Mapping[str, object],
    baseline_terms: str = "well_and_reward_outcome",
) -> dict[str, object]:
    """Return every model, data, fold and execution fact that defines a bundle.

    The same dict is both the skip-check comparison and the model half of the
    provenance sidecar, so a bundle can never be skipped on a contract the
    sidecar does not literally record.

    Parameters
    ----------
    spec : MarkedFamilySpec
        The model family being fit.
    penalty_sigma : float
        Ridge prior SD on the log-rate coefficients.
    n_folds : int
        Number of blocked held-out folds.
    fold_seed : int
        Seed of the blocked fold assignment.
    design_batch_size : int
        Requested number of design columns scored per batch.
    min_marked_events : int
        Finite marked events required per mark and event class.
    brain_area : str
        Recorded area the units come from.
    input_hashes : Mapping of str to object
        SHA-256 digests of every input the fit reads, and the SWR source's
        name (``swr_interval_source``).
    baseline_terms : {"well_and_reward_outcome", "none"}
        The per-stop baseline columns every design carries.

    Returns
    -------
    dict of str to object
        JSON-safe contract keys. ``estimator`` is included for the skip check;
        :func:`write_fit_provenance` records that key itself.
    """
    return {
        "estimator": ESTIMATOR,
        "inference": INFERENCE,
        "bundle_contract_version": BUNDLE_CONTRACT_VERSION,
        "model_name": spec.name,
        "model_tier": spec.tier,
        "baseline_terms": baseline_terms,
        "model_question": spec.question,
        "scientific_marks": list(spec.scientific_marks),
        "nuisance_marks": list(spec.nuisance_marks),
        "required_varying_nuisance_marks": list(spec.required_varying_nuisance_marks),
        "mark_scaling": "within_session_within_replay_class_population_sd_ddof0",
        "undefined_mark_policy": (
            "retain event in ordinary kernel; zero only its centered marked "
            "regressor; record availability and scale"
        ),
        "neuron_inclusion": "all recorded units; no firing-rate or p-value screen",
        "penalty_sigma_log_rate": float(penalty_sigma),
        "mark_penalty_sigma_log_rate": (
            None if mark_penalty_sigma is None else float(mark_penalty_sigma)
        ),
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed),
        "requested_design_batch_size": int(design_batch_size),
        "profiled_batch_feature_budget": PROFILED_BATCH_FEATURE_BUDGET,
        "min_finite_events_per_mark_per_class": int(min_marked_events),
        "sampling_frequency_hz": SAMPLING_FREQUENCY,
        "fit_half_window_s": FIT_HALF_WINDOW_S,
        "n_basis_per_kernel": N_BASIS_PER_KERNEL,
        "report_half_window_s": REPORT_HALF_WINDOW_S,
        "stop_cohort": STOP_COHORT,
        "brain_area": brain_area,
        **dict(input_hashes),
    }


def _bundle_is_current(
    artifact: Path,
    spec: MarkedFamilySpec,
    *,
    overwrite: bool,
    fit_contract: Mapping[str, object],
) -> bool:
    """Return whether a bundle or marker on disk matches `fit_contract` exactly.

    The sidecar is the bundle's commit marker: it is written last and removed
    before a refit, so its presence plus a matching contract is what makes a
    bundle skippable. An ``ineligible`` or ``no_units`` sidecar marks a session
    x model that intentionally has no primary artifact.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    spec : MarkedFamilySpec
        The model family, which fixes the set of companion files.
    overwrite : bool
        When True, nothing on disk is current.
    fit_contract : Mapping of str to object
        The contract from :func:`build_fit_contract`; every key must be
        recorded with the same value in the sidecar.

    Returns
    -------
    bool
        True only when this session x model may be skipped.
    """
    if overwrite:
        return False
    sidecar = fit_provenance_path(artifact)
    if not is_nonempty_file(sidecar):
        return False
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(record, Mapping):
        return False
    if not all(record.get(key) == value for key, value in fit_contract.items()):
        return False
    status = record.get("bundle_status")
    if status in {"ineligible", "no_units"}:
        return not artifact.exists()
    if status != "complete":
        return False
    return is_nonempty_file(artifact) and all(
        is_nonempty_file(path) for path in family_companion_paths(artifact, spec)
    )


def _remove_stale_bundle(artifact: Path, spec: MarkedFamilySpec) -> None:
    """Remove every component of one session x model before refitting it.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    spec : MarkedFamilySpec
        The model family, which fixes the set of companion files.

    Returns
    -------
    None
        The sidecar and all companions are unlinked, so an interrupted refit
        can never leave a mixture of old and new components that looks
        complete on the next run.
    """
    fit_provenance_path(artifact).unlink(missing_ok=True)
    artifact.unlink(missing_ok=True)
    for path in family_companion_paths(artifact, spec):
        path.unlink(missing_ok=True)


def _write_marker(
    artifact: Path,
    input_records: Mapping[str, object],
    fit_contract: Mapping[str, object],
    *,
    status: str,
    extra: Mapping[str, object],
) -> None:
    """Write a current sidecar for an intentionally absent primary artifact.

    Parameters
    ----------
    artifact : Path
        The primary ``.pkl`` path that is deliberately not written.
    input_records : Mapping of str to object
        The records :func:`record_inputs` returned when the tables were read.
    fit_contract : Mapping of str to object
        The contract this session x model would have been fit under.
    status : str
        ``"ineligible"`` or ``"no_units"``.
    extra : Mapping of str to object
        Further session facts merged into the sidecar.

    Returns
    -------
    None
        The sidecar is written; no primary artifact or companion is created.
    """
    contract_extra = {
        key: value for key, value in fit_contract.items() if key != "estimator"
    }
    write_fit_provenance(
        artifact,
        dict(input_records),
        estimator=ESTIMATOR,
        cli=sys.argv,
        extra=json_safe({**contract_extra, "bundle_status": status, **dict(extra)}),
    )


def _check_prepared_window(spline_basis: NDArray, lags: NDArray) -> None:
    """Refuse a fit whose prepared window differs from the pinned contract.

    ``fit_half_window_s`` and ``n_basis_per_kernel`` are contract keys, but the
    contract is built before the session is prepared so that a current bundle
    can be skipped without loading anything. This is where the pinned values
    are confronted with the window that was actually prepared.

    Parameters
    ----------
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions of the prepared session context.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples of the prepared session context.

    Returns
    -------
    None
        Returns only when the prepared window matches the pinned contract.

    Raises
    ------
    RuntimeError
        If the prepared half-window or basis width differs from the contract,
        which would make every sidecar written in this run misdescribe its fit.
    """
    half_window_s = float(np.abs(lags).max()) / SAMPLING_FREQUENCY
    n_basis = int(spline_basis.shape[1])
    if half_window_s != FIT_HALF_WINDOW_S or n_basis != N_BASIS_PER_KERNEL:
        raise RuntimeError(
            f"prepared fit window ({half_window_s} s, {n_basis} basis functions) "
            f"does not match the pinned contract ({FIT_HALF_WINDOW_S} s, "
            f"{N_BASIS_PER_KERNEL} basis functions)"
        )


def _audit_diagnostics(
    logger: logging.Logger, diagnostics: Mapping[str, object], n_units: int
) -> bool:
    """Report incomplete channels and refuse a severely nonconverged model.

    Parameters
    ----------
    logger : logging.Logger
        Session logger.
    diagnostics : Mapping of str to object
        ``MarkedFamilyResults.diagnostics``; the per-unit non-convergence count
        and the per-comparison NaN counts are read by key, so a diagnostics
        dict missing one raises ``KeyError`` rather than hiding it.
    n_units : int
        Units in the session's fit.

    Returns
    -------
    bool
        False when more than ``MAX_NONCONVERGED_UNIT_FRACTION`` of units have a
        non-converged full-data fit, in which case nothing may be written.
    """
    n_failed = int(diagnostics["n_units_full_fit_not_converged"])
    if n_failed:
        logger.warning(
            "%d of %d full-data unit fits did not converge", n_failed, n_units
        )
    nan_counts = diagnostics["n_units_nan_heldout_delta_by_comparison"]
    for comparison in sorted(nan_counts):
        if nan_counts[comparison]:
            logger.warning(
                "%s: %d of %d units have a non-finite held-out delta",
                comparison,
                nan_counts[comparison],
                n_units,
            )
    if n_failed > MAX_NONCONVERGED_UNIT_FRACTION * n_units:
        logger.error(
            "%d of %d units exceed the %.0f%% full-fit convergence gate",
            n_failed,
            n_units,
            100 * MAX_NONCONVERGED_UNIT_FRACTION,
        )
        return False
    return True


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    *,
    model_names: tuple[str, ...],
    brain_area: str = "mPFC",
    output_dir: Path | None = None,
    log_directory: Path = Path("logs"),
    overwrite: bool = False,
    penalty_sigma: float = DEFAULT_PENALTY_SIGMA,
    mark_penalty_sigma: float | None = DEFAULT_MARK_PENALTY_SIGMA,
    n_folds: int = 5,
    fold_seed: int | None = None,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
    event_arm_table_path: Path | None = None,
    event_table_path: Path | None = None,
    baseline_terms: str = "well_and_reward_outcome",
    gpu_record: Mapping[str, object] | None = None,
    failure_sink: list[tuple[str, str]] | None = None,
) -> bool:
    """Fit all requested families for one session, sharing preparation/folds.

    Session preparation, the fold assignment and the held-out response are
    shared across every requested family, so the families differ only in their
    design. Each family is fit inside its own guard: one family that raises
    does not cost the session the families that would have succeeded.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file name.
    epoch_name : str
        Session epoch (interval list) name.
    model_names : tuple of str
        Families to fit, as keys of ``RECOMMENDED_MODEL_SPECS``.
    brain_area : str, optional
        Recorded area whose units are fit (``"mPFC"`` or ``"OFC"``).
    output_dir : Path, optional
        Directory the bundles are written to; ``processed_data_dir() /
        OUTPUT_SUBDIR`` when None, resolved at the call.
    log_directory : Path, optional
        Directory the per-session log is written to.
    overwrite : bool, optional
        Refit even when a bundle under the same contract already exists.
    penalty_sigma : float, optional
        Ridge prior SD on the log-rate coefficients.
    mark_penalty_sigma : float or None, optional
        Ridge prior SD for the marked blocks; ``None`` shares ``penalty_sigma``.
    n_folds : int, optional
        Number of blocked held-out folds.
    fold_seed : int or None, optional
        Fold seed; derived from the session when None.
    design_batch_size : int, optional
        Requested number of design columns scored per batch.
    min_marked_events : int, optional
        Finite marked events required per mark and event class.
    event_arm_table_path : Path, optional
        Event-arm table supplying the per-arm content marks; the canonical
        ``table_path("event_arm_table")`` when None.
    event_table_path : Path, optional
        Event table supplying the per-event marks; the canonical
        ``table_path("event_table")`` when None.
    baseline_terms : {"well_and_reward_outcome", "none"}, optional
        Per-stop baseline columns every design carries; ``"none"`` reproduces
        the single-intercept designs.
    gpu_record : Mapping of str to object, optional
        The accelerator record from ``configure_jax_gpu``.
    failure_sink : list of tuple of str, optional
        Appended with ``(failure kind, message)`` for every family that failed
        and for a session-level failure, so the batch can report every cause.

    Returns
    -------
    bool
        True when every requested family is current, was fit, was recorded as
        ineligible, or had no units. False when the session raised, or when any
        family raised or failed its convergence gate.
    """
    if output_dir is None:
        output_dir = processed_data_dir() / OUTPUT_SUBDIR
    if event_arm_table_path is None:
        event_arm_table_path = table_path("event_arm_table")
    if event_table_path is None:
        event_table_path = table_path("event_table")
    unknown = sorted(set(model_names) - set(RECOMMENDED_MODEL_SPECS))
    if unknown or not model_names:
        raise ValueError(f"unknown or empty model selection: {unknown}")
    if len(model_names) != len(set(model_names)):
        raise ValueError("model_names cannot contain duplicates")
    fold_seed = (
        session_seed("marked_family_folds", nwb_file_name, epoch_name)
        if fold_seed is None
        else fold_seed
    )
    if gpu_record is None:
        gpu_record = {
            "jax_backend": "unknown",
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_name": None,
        }
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    logger = setup_logger(
        name_logfile=f"{epoch_identifier}_marked_family",
        path_logfile=log_directory / f"{epoch_identifier}_marked_family.log",
    )
    try:
        input_records = record_inputs(
            {
                "stop_table": table_path("stop_table"),
                "event_arm_table": event_arm_table_path,
                "event_table": event_table_path,
            }
        )
        canonical_hashes = canonical_event_input_hashes(nwb_file_name, epoch_name)
        input_hashes = {
            "stop_table_sha256": input_records["stop_table"]["sha256"],
            "event_arm_table_sha256": input_records["event_arm_table"]["sha256"],
            "event_table_sha256": input_records["event_table"]["sha256"],
            "canonical_event_table_sha256": canonical_hashes[
                "canonical_event_table_sha256"
            ],
            "data_over_time_sha256": canonical_hashes["data_over_time_sha256"],
            "swr_interval_source": canonical_hashes["swr_interval_source"],
            "swr_events_sha256": canonical_hashes["swr_events_sha256"],
        }
        specs = {name: RECOMMENDED_MODEL_SPECS[name] for name in model_names}
        artifacts = {
            name: family_artifact_path(output_dir, epoch_identifier, name)
            for name in model_names
        }
        contracts = {
            name: build_fit_contract(
                spec,
                penalty_sigma=penalty_sigma,
                mark_penalty_sigma=mark_penalty_sigma,
                n_folds=n_folds,
                fold_seed=fold_seed,
                design_batch_size=design_batch_size,
                min_marked_events=min_marked_events,
                brain_area=brain_area,
                input_hashes=input_hashes,
                baseline_terms=baseline_terms,
            )
            for name, spec in specs.items()
        }
        pending = [
            name
            for name in model_names
            if not _bundle_is_current(
                artifacts[name],
                specs[name],
                overwrite=overwrite,
                fit_contract=contracts[name],
            )
        ]
        if not pending:
            logger.info("All requested marked-family bundles are current; skipping")
            return True
        output_dir.mkdir(parents=True, exist_ok=True)

        preparation_started = time.perf_counter()
        cohort_stops, cohort_report = load_session_stop_cohort(
            nwb_file_name, epoch_name, cohort=STOP_COHORT
        )
        context = prepare_marked_session_context(
            nwb_file_name,
            epoch_name,
            brain_area=brain_area,
            sampling_frequency=SAMPLING_FREQUENCY,
            cohort_stops=cohort_stops,
            event_arm_table_path=event_arm_table_path,
            baseline_terms=baseline_terms,
        )
        marked_events, mark_hashes = load_session_recommended_event_marks(
            nwb_file_name,
            epoch_name,
            context.events,
            event_arm_table_path=event_arm_table_path,
            event_table_path=event_table_path,
        )
        preparation_seconds = time.perf_counter() - preparation_started
        if cohort_report["stop_table_sha256"] != input_hashes["stop_table_sha256"]:
            raise RuntimeError("stop table changed between hashing and loading")
        for key, value in mark_hashes.items():
            if value != input_hashes[key]:
                raise RuntimeError(f"{key} changed between hashing and loading")
        if (
            context.input_provenance["canonical_event_table_sha256"]
            != input_hashes["canonical_event_table_sha256"]
        ):
            raise RuntimeError("canonical event table changed during preparation")
        if (
            context.input_provenance["data_over_time_sha256"]
            != input_hashes["data_over_time_sha256"]
        ):
            raise RuntimeError("data-over-time input changed during preparation")
        if (
            context.input_provenance["swr_events_sha256"]
            != input_hashes["swr_events_sha256"]
        ):
            raise RuntimeError("SWR events input changed during preparation")
        _check_prepared_window(context.spline_basis, context.lags)

        common_extra = {
            "nwb_file_name": nwb_file_name,
            "epoch_name": epoch_name,
            "cohort_report": cohort_report,
            "cohort_info": context.cohort_info,
            "n_units": int(context.counts.shape[1]),
            "n_fit_bins": int(context.counts.shape[0]),
            "n_events_by_class": {
                event_class: int(len(marked_events[event_class]))
                for event_class in MARKED_EVENT_CLASSES
            },
            "session_preparation_seconds": preparation_seconds,
            "baseline_column_names": list(context.baseline_names),
            **context.baseline_report,
            "jax_backend": gpu_record["jax_backend"],
            "gpu_index": gpu_record["gpu_index"],
            "gpu_uuid": gpu_record["gpu_uuid"],
            "gpu_name": gpu_record["gpu_name"],
            **context.input_provenance,
        }
        if context.counts.shape[1] == 0:
            logger.warning(
                "No %s units; writing no-units markers for %s",
                brain_area,
                epoch_identifier,
            )
            for name in pending:
                _remove_stale_bundle(artifacts[name], specs[name])
                _write_marker(
                    artifacts[name],
                    input_records,
                    contracts[name],
                    status="no_units",
                    extra=common_extra,
                )
            return True

        folds = assign_blocked_folds(
            context.fit_mask,
            n_folds=n_folds,
            seed=fold_seed,
            support_radius_bins=int(np.abs(context.lags).max()),
        )
        prepared_response = prepare_heldout_response(context.counts, folds)
        failures: list[tuple[str, str]] = []
        report = np.abs(context.lag_time) <= REPORT_HALF_WINDOW_S + 1e-12
        for name in pending:
            spec = specs[name]
            artifact = artifacts[name]
            logger.info("Fitting marked family %s (%s)", name, spec.question)
            # Removed here, not before the session loaded: a family that is
            # never reached keeps whatever it had rather than losing it to a
            # session that failed on an earlier family.
            _remove_stale_bundle(artifact, spec)
            design_started = time.perf_counter()
            try:
                ladder = build_marked_family_ladder(
                    marked_events,
                    context.fit_mask,
                    context.spline_basis,
                    context.lags,
                    spec,
                    min_marked_events=min_marked_events,
                    baseline=context.baseline,
                )
                design_seconds = time.perf_counter() - design_started
                fit_started = time.perf_counter()
                fitted = fit_marked_family_ladder(
                    ladder,
                    context.counts,
                    folds,
                    penalty_sigma=penalty_sigma,
                    mark_penalty_sigma=mark_penalty_sigma,
                    design_batch_size=design_batch_size,
                    prepared_response=prepared_response,
                )
                fit_seconds = time.perf_counter() - fit_started
            except ModelIneligibleError as error:
                logger.warning("Model %s is ineligible: %s", name, error)
                _write_marker(
                    artifact,
                    input_records,
                    contracts[name],
                    status="ineligible",
                    extra={**common_extra, "ineligibility_reason": str(error)},
                )
                continue
            except MemoryError:
                raise
            except Exception as error:
                logger.exception("Model %s failed for %s", name, epoch_identifier)
                _remove_stale_bundle(artifact, spec)
                failures.append((type(error).__name__, f"{name}: {str(error)[:160]}"))
                continue
            n_units = int(context.counts.shape[1])
            if not _audit_diagnostics(logger, fitted.diagnostics, n_units):
                n_failed = int(fitted.diagnostics["n_units_full_fit_not_converged"])
                failures.append(
                    (
                        "NonconvergedFullFit",
                        f"{n_failed} of {n_units} units did not converge in "
                        f"model {name}",
                    )
                )
                continue

            statistics = fitted.statistics.copy()
            statistics.insert(0, "neuron_index", np.arange(len(statistics)))
            statistics["nwb_file_name"] = nwb_file_name
            statistics["epoch_name"] = epoch_name
            statistics["brain_area"] = brain_area
            statistics["stop_cohort"] = STOP_COHORT
            statistics["fit_spike_count"] = context.counts.sum(axis=0).astype(int)
            statistics["fit_firing_rate_hz"] = (
                context.counts.mean(axis=0) * SAMPLING_FREQUENCY
            )
            statistics.to_pickle(artifact)
            np.save(artifact.with_suffix(".lag_time.npy"), context.lag_time[report])
            for event_class in MARKED_EVENT_CLASSES:
                np.save(
                    artifact.with_suffix(f".{event_class}.typical.npy"),
                    fitted.typical_curves[event_class][report],
                )
                for mark in spec.scientific_marks:
                    np.save(
                        artifact.with_suffix(f".{event_class}.{mark}.slope.npy"),
                        fitted.mark_curves[event_class][mark][report],
                    )
            contract_extra = {
                key: value
                for key, value in contracts[name].items()
                if key != "estimator"
            }
            # The sidecar is written last: it is the bundle's commit marker.
            write_fit_provenance(
                artifact,
                input_records,
                estimator=ESTIMATOR,
                cli=sys.argv,
                extra=json_safe(
                    {
                        **contract_extra,
                        "bundle_status": "complete",
                        "mark_definitions": {
                            mark: MARK_DESCRIPTIONS[mark] for mark in spec.all_marks
                        },
                        "fold_sizes": np.bincount(folds, minlength=n_folds).tolist(),
                        "fold_support_radius_bins": int(np.abs(context.lags).max()),
                        "runner_timing_seconds": {
                            "session_preparation_shared": preparation_seconds,
                            "design_construction": design_seconds,
                            "fit": fit_seconds,
                        },
                        "marked_family_diagnostics": fitted.diagnostics,
                        **common_extra,
                    }
                ),
            )
            logger.info("Saved %s", artifact)
        if failures:
            if failure_sink is not None:
                failure_sink.extend(failures)
            return False
        return True
    except MemoryError:
        logger.exception(
            "Marked model family ran out of memory for %s", epoch_identifier
        )
        raise
    except Exception as error:
        logger.exception("Marked model family failed for %s", epoch_identifier)
        if failure_sink is not None:
            failure_sink.append((type(error).__name__, str(error)[:160]))
        return False


def get_command_line_arguments() -> Namespace:
    """Parse and validate the marked-family production contract.

    Returns
    -------
    Namespace
        Validated CLI arguments; ``--include-interactions`` has already been
        folded into ``models`` without introducing duplicates.
    """
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("--animal")
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch")
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--brain-area", choices=["mPFC", "OFC"], default="mPFC")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=processed_data_dir() / OUTPUT_SUBDIR
    )
    parser.add_argument("--log-directory", type=Path, default=Path("logs"))
    parser.add_argument(
        "--event-arm-table", type=Path, default=table_path("event_arm_table")
    )
    parser.add_argument("--event-table", type=Path, default=table_path("event_table"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(RECOMMENDED_MODEL_SPECS),
        default=list(CORE_MODEL_NAMES),
        help="Models to fit (default: the nine core families).",
    )
    parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Add exploration, reward, and ripple-power content interactions.",
    )
    parser.add_argument("--penalty-sigma", type=float, default=DEFAULT_PENALTY_SIGMA)
    parser.add_argument(
        "--mark-penalty-sigma",
        type=float,
        default=DEFAULT_MARK_PENALTY_SIGMA,
        help=(
            "Ridge prior SD for the marked blocks (log-rate units); the base "
            "kernels keep --penalty-sigma. Default: the module's "
            "DEFAULT_MARK_PENALTY_SIGMA, chosen by held-out deviance on the "
            "2026-09-18 smoke sessions."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument(
        "--design-batch-size", type=int, default=DEFAULT_NULL_DESIGN_BATCH_SIZE
    )
    parser.add_argument(
        "--min-marked-events", type=int, default=MIN_MARKED_EVENTS_PER_CLASS
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help=(
            "Fit with one intercept per session and no per-stop well and "
            "reward-outcome baseline columns (the contract-1 designs)."
        ),
    )
    args = parser.parse_args()
    if args.include_interactions:
        args.models = list(dict.fromkeys([*args.models, *INTERACTION_MODEL_NAMES]))
    if args.penalty_sigma <= 0 or not np.isfinite(args.penalty_sigma):
        parser.error("--penalty-sigma must be finite and positive")
    if args.mark_penalty_sigma is not None and (
        args.mark_penalty_sigma <= 0 or not np.isfinite(args.mark_penalty_sigma)
    ):
        parser.error("--mark-penalty-sigma must be finite and positive")
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2")
    if args.design_batch_size < 1:
        parser.error("--design-batch-size must be positive")
    if args.min_marked_events < 2:
        parser.error("--min-marked-events must be at least 2")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    return args


def _written_bundle_status(artifact: Path) -> str:
    """Classify one session x model outcome from what is on disk.

    Parameters
    ----------
    artifact : Path
        The session x model's primary ``.pkl`` statistics table.

    Returns
    -------
    str
        ``"complete"`` when the primary artifact exists, otherwise the marker
        status its sidecar records (``"ineligible"`` or ``"no_units"``), or
        ``"missing"`` when there is no readable sidecar.
    """
    if artifact.is_file():
        return "complete"
    sidecar = fit_provenance_path(artifact)
    if not is_nonempty_file(sidecar):
        return "missing"
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return "missing"
    status = record.get("bundle_status") if isinstance(record, Mapping) else None
    return str(status) if status in {"ineligible", "no_units"} else "missing"


def main() -> None:
    """Select sessions, claim one accelerator, and fit requested families.

    Returns
    -------
    None
        Prints how many session x model bundles were written, and how many
        were recorded as having no units or as ineligible.

    Raises
    ------
    RuntimeError
        If any session failed; the message names every failure cause the
        session reported, not only the first.
    MemoryError
        Propagated immediately from the session that exhausted memory.
    """
    args = get_command_line_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)
    epochs = filter_epoch_info(
        analysed_epochs(),
        sessions=args.sessions,
        animal=args.animal,
        date=args.date,
        epoch=args.epoch,
    )
    if args.max_sessions is not None:
        epochs = epochs.iloc[: args.max_sessions]
    if epochs.empty:
        raise ValueError("the requested filters selected no sessions")
    gpu_record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
    print(
        f"Processing {len(epochs)} sessions × {len(args.models)} model families "
        f"on {gpu_record['jax_backend']}"
    )
    failures: list[tuple[str, str]] = []
    counts = dict.fromkeys(("complete", "no_units", "ineligible", "missing"), 0)
    for epoch in epochs.itertuples():
        nwb_file_name, epoch_name = epoch.Index[0], epoch.Index[1]
        sink: list[tuple[str, str]] = []
        succeeded = run_analysis(
            nwb_file_name,
            epoch_name,
            model_names=tuple(args.models),
            brain_area=args.brain_area,
            output_dir=args.output_dir,
            log_directory=args.log_directory,
            overwrite=args.overwrite,
            penalty_sigma=args.penalty_sigma,
            mark_penalty_sigma=args.mark_penalty_sigma,
            n_folds=args.n_folds,
            fold_seed=args.fold_seed,
            design_batch_size=args.design_batch_size,
            min_marked_events=args.min_marked_events,
            event_arm_table_path=args.event_arm_table,
            event_table_path=args.event_table,
            baseline_terms="none" if args.no_baseline else "well_and_reward_outcome",
            gpu_record=gpu_record,
            failure_sink=sink,
        )
        identifier = (
            f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{args.brain_area}"
        )
        for name in args.models:
            status = _written_bundle_status(
                family_artifact_path(args.output_dir, identifier, name)
            )
            counts[status] += 1
        if not succeeded:
            # Every cause the session reported, not only the first: one session
            # can fail several families for unrelated reasons.
            detail = "; ".join(f"{kind}: {message}" for kind, message in sink)
            failures.append(
                (f"{nwb_file_name} {epoch_name}", detail or "see session log")
            )
    print(
        f"{counts['complete']} bundle(s) written, {counts['no_units']} no-units "
        f"marker(s), {counts['ineligible']} ineligible marker(s); "
        f"{len(failures)} session(s) failed"
    )
    if failures:
        raise RuntimeError(
            "marked-family failures: "
            + "; ".join(f"{session}: {detail}" for session, detail in failures)
        )


if __name__ == "__main__":
    sys.exit(main())
