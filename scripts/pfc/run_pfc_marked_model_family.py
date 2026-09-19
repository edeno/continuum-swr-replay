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

import hashlib
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
import pandas as pd

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    DEFAULT_PENALTY_SIGMA,
    prepare_heldout_response,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (
    DEFAULT_STOP_TABLE_PATH,
    load_session_stop_cohort,
)
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.heldout import assign_blocked_folds
from continuum_swr_replay.analysis.pfc.marked_glm import (
    MARKED_EVENT_CLASSES,
    prepare_marked_session_context,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    CORE_MODEL_NAMES,
    DEFAULT_EVENT_ARM_TABLE_PATH,
    DEFAULT_EVENT_TABLE_PATH,
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
from continuum_swr_replay.analysis.replay.fit_provenance import (
    fit_provenance_path,
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.decoding_common import setup_logger
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, SAMPLING_FREQUENCY

DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR / "pfc_marked_model_family"
ESTIMATOR = "joint_multimark_ridge_poisson_temporal_glm"
INFERENCE = "descriptive_blocked_heldout_leave_one_mark_block_out"
BUNDLE_CONTRACT_VERSION = 1
STOP_COHORT = "at_well"
REPORT_HALF_WINDOW_S = 0.2
MIN_MARKED_EVENTS_PER_CLASS = 10
MAX_NONCONVERGED_UNIT_FRACTION = 0.1


def session_seed(stream: str, nwb_file_name: str, epoch_name: str) -> int:
    """Stable session-specific seed for a named random stream."""
    key = f"{stream}|{nwb_file_name}|{epoch_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def family_artifact_path(
    output_dir: Path, epoch_identifier: str, model_name: str
) -> Path:
    """Primary statistics table for one session × model family."""
    return output_dir / f"{epoch_identifier}_{model_name}_marked_family.pkl"


def family_companion_paths(artifact: Path, spec: MarkedFamilySpec) -> tuple[Path, ...]:
    """All arrays required for a complete eligible bundle."""
    curves: list[Path] = [artifact.with_suffix(".lag_time.npy")]
    for event_class in MARKED_EVENT_CLASSES:
        curves.append(artifact.with_suffix(f".{event_class}.typical.npy"))
        curves.extend(
            artifact.with_suffix(f".{event_class}.{mark}.slope.npy")
            for mark in spec.scientific_marks
        )
    return tuple(curves)


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def build_fit_contract(
    spec: MarkedFamilySpec,
    *,
    penalty_sigma: float,
    n_folds: int,
    fold_seed: int,
    design_batch_size: int,
    min_marked_events: int,
    brain_area: str,
    input_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Every model, data, fold and execution fact that defines a bundle."""
    return {
        "estimator": ESTIMATOR,
        "inference": INFERENCE,
        "bundle_contract_version": BUNDLE_CONTRACT_VERSION,
        "model_name": spec.name,
        "model_tier": spec.tier,
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
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed),
        "requested_design_batch_size": int(design_batch_size),
        "profiled_batch_feature_budget": PROFILED_BATCH_FEATURE_BUDGET,
        "min_finite_events_per_mark_per_class": int(min_marked_events),
        "sampling_frequency_hz": SAMPLING_FREQUENCY,
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
    """Whether a complete, ineligible, or no-unit marker matches the contract."""
    if overwrite:
        return False
    sidecar = fit_provenance_path(artifact)
    if not _is_nonempty_file(sidecar):
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
    return _is_nonempty_file(artifact) and all(
        _is_nonempty_file(path) for path in family_companion_paths(artifact, spec)
    )


def _remove_stale_bundle(artifact: Path, spec: MarkedFamilySpec) -> None:
    """Remove all pieces before refitting; the sidecar remains the commit marker."""
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
    """Write a current sidecar for an intentionally absent primary artifact."""
    contract_extra = {
        key: value for key, value in fit_contract.items() if key != "estimator"
    }
    write_fit_provenance(
        artifact,
        dict(input_records),
        estimator=ESTIMATOR,
        cli=sys.argv,
        extra={**contract_extra, "bundle_status": status, **dict(extra)},
    )


def _audit_diagnostics(
    logger: logging.Logger, diagnostics: Mapping[str, object], n_units: int
) -> bool:
    """Report incomplete channels and refuse a severely nonconverged model."""
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
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    log_directory: Path = Path("logs"),
    overwrite: bool = False,
    penalty_sigma: float = DEFAULT_PENALTY_SIGMA,
    n_folds: int = 5,
    fold_seed: int | None = None,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
    event_arm_table_path: Path = DEFAULT_EVENT_ARM_TABLE_PATH,
    gpu_record: Mapping[str, object] | None = None,
    failure_sink: list[tuple[str, str]] | None = None,
) -> bool:
    """Fit all requested families for one session, sharing preparation/folds."""
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
                "stop_table": DEFAULT_STOP_TABLE_PATH,
                "event_arm_table": event_arm_table_path,
                "event_table": DEFAULT_EVENT_TABLE_PATH,
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
                n_folds=n_folds,
                fold_seed=fold_seed,
                design_batch_size=design_batch_size,
                min_marked_events=min_marked_events,
                brain_area=brain_area,
                input_hashes=input_hashes,
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
        for name in pending:
            _remove_stale_bundle(artifacts[name], specs[name])
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
        )
        marked_events, mark_hashes = load_session_recommended_event_marks(
            nwb_file_name,
            epoch_name,
            context.events,
            event_arm_table_path=event_arm_table_path,
            event_table_path=DEFAULT_EVENT_TABLE_PATH,
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
            "jax_backend": gpu_record["jax_backend"],
            "gpu_index": gpu_record["gpu_index"],
            "gpu_uuid": gpu_record["gpu_uuid"],
            "gpu_name": gpu_record["gpu_name"],
            **context.input_provenance,
        }
        if context.counts.shape[1] == 0:
            for name in pending:
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
            design_started = time.perf_counter()
            try:
                ladder = build_marked_family_ladder(
                    marked_events,
                    context.fit_mask,
                    context.spline_basis,
                    context.lags,
                    spec,
                    min_marked_events=min_marked_events,
                )
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
            design_seconds = time.perf_counter() - design_started
            fit_started = time.perf_counter()
            fitted = fit_marked_family_ladder(
                ladder,
                context.counts,
                folds,
                penalty_sigma=penalty_sigma,
                design_batch_size=design_batch_size,
                prepared_response=prepared_response,
            )
            fit_seconds = time.perf_counter() - fit_started
            if not _audit_diagnostics(
                logger, fitted.diagnostics, int(context.counts.shape[1])
            ):
                failures.append((name, "full-data convergence gate failed"))
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
            write_fit_provenance(
                artifact,
                input_records,
                estimator=ESTIMATOR,
                cli=sys.argv,
                extra={
                    **contract_extra,
                    "bundle_status": "complete",
                    "mark_definitions": {
                        mark: MARK_DESCRIPTIONS[mark] for mark in spec.all_marks
                    },
                    "fold_sizes": np.bincount(folds, minlength=n_folds).tolist(),
                    "fold_support_radius_bins": int(np.abs(context.lags).max()),
                    "n_basis_per_kernel": int(context.spline_basis.shape[1]),
                    "runner_timing_seconds": {
                        "session_preparation_shared": preparation_seconds,
                        "design_construction": design_seconds,
                        "fit": fit_seconds,
                    },
                    "marked_family_diagnostics": fitted.diagnostics,
                    **common_extra,
                },
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


def read_session_list(path: Path) -> list[tuple[str, str]]:
    """Read unique ``nwb_file_name epoch_name`` pairs in file order."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{number}: expected '<nwb_file_name> <epoch_name>', got {raw!r}"
            )
        pair = (fields[0], fields[1])
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path} names no sessions")
    return pairs


def _filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Apply CLI filters while preserving source or explicit session order."""
    keep = np.ones(len(epoch_info), dtype=bool)
    keys = list(
        zip(
            epoch_info.index.get_level_values("nwb_file_name"),
            epoch_info.index.get_level_values("interval_list_name"),
            strict=True,
        )
    )
    requested_order = read_session_list(args.sessions) if args.sessions else None
    if requested_order is not None:
        missing = [key for key in requested_order if key not in set(keys)]
        if missing:
            raise ValueError(f"session file names unavailable sessions: {missing[:10]}")
        requested = set(requested_order)
        keep &= np.array([key in requested for key in keys])
    if args.animal is not None:
        keep &= epoch_info.animal.to_numpy() == args.animal
    if args.date is not None:
        keep &= epoch_info.date.to_numpy() == args.date
    if args.epoch is not None:
        keep &= (
            epoch_info.index.get_level_values("interval_list_name").to_numpy()
            == args.epoch
        )
    selected = epoch_info.loc[keep]
    if requested_order is not None:
        order = {key: index for index, key in enumerate(requested_order)}
        selected_keys = list(
            zip(
                selected.index.get_level_values("nwb_file_name"),
                selected.index.get_level_values("interval_list_name"),
                strict=True,
            )
        )
        selected = selected.iloc[np.argsort([order[key] for key in selected_keys])]
    return selected


def get_command_line_arguments() -> Namespace:
    """Parse and validate the production contract."""
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-directory", type=Path, default=Path("logs"))
    parser.add_argument(
        "--event-arm-table", type=Path, default=DEFAULT_EVENT_ARM_TABLE_PATH
    )
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
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument(
        "--design-batch-size", type=int, default=DEFAULT_NULL_DESIGN_BATCH_SIZE
    )
    parser.add_argument(
        "--min-marked-events", type=int, default=MIN_MARKED_EVENTS_PER_CLASS
    )
    args = parser.parse_args()
    if args.include_interactions:
        args.models = list(dict.fromkeys([*args.models, *INTERACTION_MODEL_NAMES]))
    if args.penalty_sigma <= 0 or not np.isfinite(args.penalty_sigma):
        parser.error("--penalty-sigma must be finite and positive")
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2")
    if args.design_batch_size < 1:
        parser.error("--design-batch-size must be positive")
    if args.min_marked_events < 2:
        parser.error("--min-marked-events must be at least 2")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    return args


def main() -> None:
    """Select sessions, claim one accelerator, and fit requested families."""
    args = get_command_line_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)
    epochs = _filter_epoch_info(get_epoch_info(), args)
    if args.max_sessions is not None:
        epochs = epochs.iloc[: args.max_sessions]
    if epochs.empty:
        raise ValueError("the requested filters selected no sessions")
    gpu_record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
    print(
        f"Processing {len(epochs)} sessions × {len(args.models)} model families "
        f"on {gpu_record['jax_backend']}"
    )
    failures: list[tuple[str, str, str]] = []
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
            n_folds=args.n_folds,
            fold_seed=args.fold_seed,
            design_batch_size=args.design_batch_size,
            min_marked_events=args.min_marked_events,
            event_arm_table_path=args.event_arm_table,
            gpu_record=gpu_record,
            failure_sink=sink,
        )
        if not succeeded:
            kind, message = sink[0] if sink else ("Unknown", "see session log")
            failures.append((f"{nwb_file_name} {epoch_name}", kind, message))
    print(f"{len(epochs) - len(failures)} session(s) succeeded; {len(failures)} failed")
    if failures:
        raise RuntimeError(
            "marked-family failures: "
            + "; ".join(
                f"{session}: {kind}: {message}" for session, kind, message in failures
            )
        )


if __name__ == "__main__":
    sys.exit(main())
