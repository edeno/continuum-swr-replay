"""Fit content-marked mPFC replay kernels on the canonical at-well cohort.

Each replay event contributes the ordinary peri-onset spline basis and a
second copy scaled by its within-session, within-class current-arm content.
Both SWR-coupled and non-SWR content slopes are fit jointly.  Event duration
is included by default as its own marked nuisance kernel.  A class-specific
held-out comparison drops only that class's content kernel. Within-class mark
permutations provide a descriptive reference without changing any event onset.

The plus-one tail fraction is not treated as a calibrated conditional p-value:
content is associated with duration and context in these data, so unrestricted
labels are not conditionally exchangeable. It never selects neurons, events,
curves, or population summaries.

The corpus setting is ``--n-shuffles 100``. ``--n-shuffles 1000`` is an opt-in
sensitivity run only: it costs about 5 GPU-hours on the largest session, so a
whole-corpus pass at that setting takes GPU-days.

Examples
--------
Smoke one session on CPU::

    python scripts/pfc/run_pfc_marked_kernels.py --animal wilbur \
        --date 20210329 --epoch 02_r1 --max-sessions 1 --n-shuffles 0 \
        --allow-cpu --output-dir /tmp/pfc_marked_smoke

Production worker on GPU 0::

    python scripts/pfc/run_pfc_marked_kernels.py --gpu 0 \
        --sessions worker0.sessions --n-shuffles 100

Run one process per GPU with disjoint ``--sessions`` files. A process keeps
its selected card and reuses compiled equal-width scorers within each session.
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

# A worker drives one GPU and should not also create a pool of BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    DEFAULT_PENALTY_SIGMA,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (
    DEFAULT_STOP_TABLE_PATH,
    load_session_stop_cohort,
)
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.heldout import (
    EMPIRICAL_PVALUE_METHOD,
    MIN_USABLE_NULL_DRAWS,
    assign_blocked_folds,
    required_usable_null_draws,
)
from continuum_swr_replay.analysis.pfc.marked_glm import (
    DEFAULT_EVENT_ARM_TABLE_PATH,
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    build_marked_model_ladder,
    fit_marked_kernel_ladder,
    prepare_marked_session_context,
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
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.decoding_common import setup_logger
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, SAMPLING_FREQUENCY

# This reference is descriptive, not a selection rule. A real 430-unit A100 profile
# projected 1,000 exact five-fold refits at about 5.2 GPU-hours per session;
# 100 gives 1/101 tail-fraction resolution without making a descriptive
# analysis several GPU-days long across the corpus.
DEFAULT_MARK_SHUFFLES = 100
DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR / "pfc_marked_kernels"
ESTIMATOR = "joint_marked_ridge_poisson_glm"
INFERENCE = "descriptive_blocked_heldout_association_with_label_shuffle_reference"
MARK_SCALING = "within_session_within_replay_class_population_sd_ddof0"
SHUFFLE_POLICY = "permute_scientific_mark_among_events_within_session_and_class"
LABEL_SHUFFLE_TAIL_METHOD = EMPIRICAL_PVALUE_METHOD
BUNDLE_CONTRACT_VERSION = 1
FIT_HALF_WINDOW_S = 0.25

MARK_DESCRIPTIONS = {
    "current_arm_fraction": (
        "fraction of hard-decoded non-local event bins assigned to the arm "
        "occupied by the animal during the stop"
    ),
    "current_arm_fraction_posterior": (
        "fraction of decoded posterior arm mass assigned to the arm occupied "
        "by the animal during the stop"
    ),
}
NUISANCE_MARK_DESCRIPTIONS = {
    EVENT_DURATION_MARK: "natural log of event duration in seconds",
}


def marked_artifact_path(
    output_dir: Path,
    epoch_identifier: str,
    scientific_mark: str,
) -> Path:
    """Return the primary per-neuron table path for one marked-kernel bundle.

    Parameters
    ----------
    output_dir : Path
        Directory holding the marked-kernel bundles.
    epoch_identifier : str
        ``{animal}{date}_{epoch}_{brain_area}`` identifier of the session.
    scientific_mark : str
        Event property carried by the scientific marked kernel.

    Returns
    -------
    Path
        Path of the bundle's ``.pkl`` statistics table.
    """
    return output_dir / f"{epoch_identifier}_{scientific_mark}_marked_kernels.pkl"


def marked_companion_paths(artifact: Path) -> tuple[Path, ...]:
    """Return the files that must coexist with a complete bundle.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.

    Returns
    -------
    tuple of Path
        Lag axis, per-class typical and mark-slope curves, and the label
        shuffle reference archive.
    """
    curves = tuple(
        artifact.with_suffix(f".{event_class}.{kind}.npy")
        for event_class in MARKED_EVENT_CLASSES
        for kind in ("typical", "mark_slope")
    )
    return (
        artifact.with_suffix(".lag_time.npy"),
        *curves,
        artifact.with_suffix(".mark_label_shuffle_reference.npz"),
    )


def build_fit_contract(
    *,
    scientific_mark: str,
    nuisance_marks: tuple[str, ...],
    penalty_sigma: float,
    n_folds: int,
    fold_seed: int,
    n_shuffles: int,
    shuffle_seed: int,
    design_batch_size: int,
    brain_area: str,
    input_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Return every fact that defines one marked-kernel fit.

    The same dict is both the skip-check comparison and the model half of the
    provenance sidecar, so a bundle can never be skipped on a contract the
    sidecar does not literally record.

    Parameters
    ----------
    scientific_mark : str
        Event property carried by the scientific marked kernel.
    nuisance_marks : tuple of str
        Marked nuisance kernels fit jointly with the scientific mark.
    penalty_sigma : float
        Ridge prior SD on the log-rate coefficients.
    n_folds : int
        Number of blocked held-out folds.
    fold_seed : int
        Seed of the blocked fold assignment.
    n_shuffles : int
        Requested within-class mark permutations.
    shuffle_seed : int
        Seed of the mark permutations.
    design_batch_size : int
        Number of permuted designs scored per batch.
    brain_area : str
        Recorded area the units come from.
    input_hashes : Mapping of str to object
        SHA-256 digests of the tables the fit reads, including
        ``stop_table_sha256`` and ``event_arm_table_sha256``.

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
        "scientific_mark": scientific_mark,
        "nuisance_marks": list(nuisance_marks),
        "mark_scaling": MARK_SCALING,
        "shuffle_policy": SHUFFLE_POLICY,
        "label_shuffle_tail_method": LABEL_SHUFFLE_TAIL_METHOD,
        "penalty_sigma_log_rate": float(penalty_sigma),
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed),
        "n_mark_shuffles": int(n_shuffles),
        "mark_shuffle_seed": int(shuffle_seed),
        "min_usable_mark_shuffles": int(required_usable_null_draws(n_shuffles)),
        "design_batch_size": int(design_batch_size),
        "sampling_frequency_hz": SAMPLING_FREQUENCY,
        "fit_half_window_s": FIT_HALF_WINDOW_S,
        "report_half_window_s": REPORT_HALF_WINDOW_S,
        "stop_cohort": STOP_COHORT,
        "brain_area": brain_area,
        **dict(input_hashes),
    }


def _bundle_is_current(
    artifact: Path,
    *,
    overwrite: bool,
    fit_contract: Mapping[str, object],
) -> bool:
    """Return whether an existing bundle was fit under exactly `fit_contract`.

    The sidecar is the bundle's commit marker: it is written last and removed
    before a refit, so its presence plus a matching contract is what makes a
    bundle skippable. A sidecar recording ``n_units == 0`` is a no-units
    marker and has no companion files by construction.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.
    overwrite : bool
        When True, nothing on disk is current.
    fit_contract : Mapping of str to object
        The contract from :func:`build_fit_contract`; every key must be
        recorded with the same value in the sidecar.

    Returns
    -------
    bool
        True only when the bundle may be skipped.
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
    if record.get("n_units") == 0:
        return True
    return is_nonempty_file(artifact) and all(
        is_nonempty_file(path) for path in marked_companion_paths(artifact)
    )


def _remove_stale_bundle(artifact: Path) -> None:
    """Remove every component of a previous bundle before refitting.

    Parameters
    ----------
    artifact : Path
        The bundle's primary ``.pkl`` statistics table.

    Returns
    -------
    None
        The sidecar and all companions are unlinked, so an interrupted refit
        can never leave a mixture of old and new components that looks
        complete on the next run.
    """
    fit_provenance_path(artifact).unlink(missing_ok=True)
    artifact.unlink(missing_ok=True)
    for path in marked_companion_paths(artifact):
        path.unlink(missing_ok=True)


def _audit_marked_diagnostics(
    logger: logging.Logger,
    diagnostics: Mapping[str, object],
    n_units: int,
) -> bool:
    """Log the fit's NaN channels and say whether the session is certifiable.

    Parameters
    ----------
    logger : logging.Logger
        Session logger.
    diagnostics : Mapping of str to object
        ``MarkedKernelResults.diagnostics``; the per-unit NaN channels and the
        per-class event and permutation counts are read by key, so a
        diagnostics dict missing one raises ``KeyError`` rather than hiding it.
    n_units : int
        Units in the session's fit.

    Returns
    -------
    bool
        False when more than ``MAX_NONCONVERGED_UNIT_FRACTION`` of units have
        a non-converged full-data fit, in which case the bundle must not be
        written.
    """
    n_not_converged = int(diagnostics["n_units_full_fit_not_converged"])
    nan_delta = diagnostics["n_units_nan_heldout_delta_by_class"]
    nan_tail = diagnostics["n_units_nan_tail_fraction_by_class"]
    n_marked_events = diagnostics["n_marked_events_by_class"]
    n_permutations = diagnostics["n_distinct_permutations_by_class"]

    if n_not_converged:
        logger.warning(
            "%d of %d units did not converge on the full-data fit",
            n_not_converged,
            n_units,
        )
    for event_class in sorted(nan_delta):
        if nan_delta[event_class]:
            logger.warning(
                "%s: %d of %d units have a NaN held-out delta",
                event_class,
                nan_delta[event_class],
                n_units,
            )
    for event_class in sorted(nan_tail):
        if nan_tail[event_class]:
            logger.warning(
                "%s: %d of %d units have a NaN label-shuffle tail fraction",
                event_class,
                nan_tail[event_class],
                n_units,
            )
    for event_class in sorted(n_marked_events):
        logger.info(
            "%s: %d marked events, %d distinct mark permutations",
            event_class,
            n_marked_events[event_class],
            n_permutations[event_class],
        )

    if n_not_converged > MAX_NONCONVERGED_UNIT_FRACTION * n_units:
        logger.error(
            "%d of %d units (> %.0f%%) did not converge on the full-data fit; "
            "refusing to write a bundle for this session",
            n_not_converged,
            n_units,
            100 * MAX_NONCONVERGED_UNIT_FRACTION,
        )
        return False
    return True


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    *,
    brain_area: str,
    output_dir: Path,
    log_directory: Path,
    overwrite: bool,
    scientific_mark: str,
    nuisance_marks: tuple[str, ...],
    penalty_sigma: float,
    n_folds: int,
    n_shuffles: int,
    fold_seed: int | None,
    shuffle_seed: int | None,
    design_batch_size: int,
    event_arm_table_path: Path,
    gpu_record: dict[str, object],
    failure_sink: list[tuple[str, str]] | None = None,
) -> bool:
    """Fit, audit, and save one session's complete marked-kernel bundle.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file name.
    epoch_name : str
        Session epoch (interval list) name.
    brain_area : str
        Recorded area whose units are fit (``"mPFC"`` or ``"OFC"``).
    output_dir : Path
        Directory the bundle is written to.
    log_directory : Path
        Directory the per-session log is written to.
    overwrite : bool
        Refit even when a bundle under the same contract already exists.
    scientific_mark : str
        Event property carried by the scientific marked kernel.
    nuisance_marks : tuple of str
        Marked nuisance kernels fit jointly with the scientific mark.
    penalty_sigma : float
        Ridge prior SD on the log-rate coefficients.
    n_folds : int
        Number of blocked held-out folds.
    n_shuffles : int
        Requested within-class mark permutations.
    fold_seed : int or None
        Fold seed; derived from the session when None.
    shuffle_seed : int or None
        Mark-permutation seed; derived from the session when None.
    design_batch_size : int
        Number of permuted designs scored per batch.
    event_arm_table_path : Path
        Event-arm table supplying the content marks.
    gpu_record : dict of str to object
        The accelerator record from ``configure_jax_gpu``.
    failure_sink : list of tuple of str, optional
        Appended with ``(exception type name, first 120 characters of the
        message)`` when the session fails, so the batch can report why every
        failed session failed.

    Returns
    -------
    bool
        True when the session is current, was fit, or has no units. False when
        the session raised, or when too many units failed to converge for the
        bundle to be certified.
    """
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    logger = setup_logger(
        name_logfile=f"{epoch_identifier}_marked",
        path_logfile=log_directory / f"{epoch_identifier}_marked.log",
    )
    if fold_seed is None:
        fold_seed = session_seed("marked_folds", nwb_file_name, epoch_name)
    if shuffle_seed is None:
        shuffle_seed = session_seed(
            "marked_content_shuffles", nwb_file_name, epoch_name
        )

    try:
        artifact = marked_artifact_path(output_dir, epoch_identifier, scientific_mark)
        input_records = record_inputs(
            {
                "stop_table": DEFAULT_STOP_TABLE_PATH,
                "event_arm_table": event_arm_table_path,
            }
        )
        input_provenance = {
            **canonical_event_input_hashes(nwb_file_name, epoch_name),
            "event_arm_table_sha256": input_records["event_arm_table"]["sha256"],
        }
        fit_contract = build_fit_contract(
            scientific_mark=scientific_mark,
            nuisance_marks=nuisance_marks,
            penalty_sigma=penalty_sigma,
            n_folds=n_folds,
            fold_seed=fold_seed,
            n_shuffles=n_shuffles,
            shuffle_seed=shuffle_seed,
            design_batch_size=design_batch_size,
            brain_area=brain_area,
            input_hashes={
                "stop_table_sha256": input_records["stop_table"]["sha256"],
                "event_arm_table_sha256": input_provenance["event_arm_table_sha256"],
                "canonical_event_table_sha256": input_provenance[
                    "canonical_event_table_sha256"
                ],
                "data_over_time_sha256": input_provenance["data_over_time_sha256"],
            },
        )
        # write_fit_provenance records the estimator from its own argument and
        # refuses an extra that would overwrite it.
        contract_extra = {
            key: value for key, value in fit_contract.items() if key != "estimator"
        }
        if _bundle_is_current(artifact, overwrite=overwrite, fit_contract=fit_contract):
            logger.info("Current marked-kernel bundle exists; skipping")
            return True
        _remove_stale_bundle(artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)

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
        preparation_seconds = time.perf_counter() - preparation_started
        if cohort_report["stop_table_sha256"] != input_records["stop_table"]["sha256"]:
            raise RuntimeError("stop table changed between hashing and session loading")
        for key, expected in input_provenance.items():
            if context.input_provenance.get(key) != expected:
                raise RuntimeError(
                    f"input {key!r} changed between hashing and session loading"
                )
        if context.counts.shape[1] == 0:
            logger.warning(
                "No %s units; writing a no-units marker for %s",
                brain_area,
                epoch_identifier,
            )
            write_fit_provenance(
                artifact,
                input_records,
                estimator=ESTIMATOR,
                cli=sys.argv,
                extra=json_safe(
                    {
                        **contract_extra,
                        "n_units": 0,
                        "cohort_report": cohort_report,
                        "cohort_info": context.cohort_info,
                        "nwb_file_name": nwb_file_name,
                        "epoch_name": epoch_name,
                        **context.input_provenance,
                    }
                ),
            )
            return True

        design_started = time.perf_counter()
        ladder = build_marked_model_ladder(
            context.events,
            context.fit_mask,
            context.spline_basis,
            context.lags,
            scientific_mark=scientific_mark,
            nuisance_marks=nuisance_marks,
        )
        design_seconds = time.perf_counter() - design_started
        folds = assign_blocked_folds(
            context.fit_mask,
            n_folds=n_folds,
            seed=fold_seed,
            support_radius_bins=int(np.abs(context.lags).max()),
        )
        required_shuffles = required_usable_null_draws(n_shuffles)
        logger.info(
            "Fitting %d units on %d bins, %d folds, %d mark shuffles",
            context.counts.shape[1],
            context.counts.shape[0],
            n_folds,
            n_shuffles,
        )
        fit_started = time.perf_counter()
        fitted = fit_marked_kernel_ladder(
            ladder,
            context.counts,
            folds,
            penalty_sigma=penalty_sigma,
            n_shuffles=n_shuffles,
            shuffle_seed=shuffle_seed,
            min_usable_shuffles=max(1, required_shuffles),
            design_batch_size=design_batch_size,
        )
        fit_seconds = time.perf_counter() - fit_started
        if not _audit_marked_diagnostics(
            logger, fitted.diagnostics, int(context.counts.shape[1])
        ):
            if failure_sink is not None:
                failure_sink.append(
                    (
                        "NonconvergedFullFit",
                        f"{fitted.diagnostics['n_units_full_fit_not_converged']} of "
                        f"{context.counts.shape[1]} units did not converge",
                    )
                )
            return False

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

        report = np.abs(context.lag_time) <= REPORT_HALF_WINDOW_S + 1e-12
        statistics.to_pickle(artifact)
        np.save(artifact.with_suffix(".lag_time.npy"), context.lag_time[report])
        for event_class in MARKED_EVENT_CLASSES:
            np.save(
                artifact.with_suffix(f".{event_class}.typical.npy"),
                fitted.typical_curves[event_class][report],
            )
            np.save(
                artifact.with_suffix(f".{event_class}.mark_slope.npy"),
                fitted.mark_curves[event_class][report],
            )
        np.savez_compressed(
            artifact.with_suffix(".mark_label_shuffle_reference.npz"),
            **fitted.randomization_reference,
        )

        # The sidecar is written last: it is the bundle's commit marker.
        sidecar = write_fit_provenance(
            artifact,
            input_records,
            estimator=ESTIMATOR,
            cli=sys.argv,
            extra=json_safe(
                {
                    **contract_extra,
                    "scientific_mark_definition": MARK_DESCRIPTIONS[scientific_mark],
                    "nuisance_mark_definitions": {
                        mark: NUISANCE_MARK_DESCRIPTIONS[mark]
                        for mark in nuisance_marks
                    },
                    "joint_scientific_mark_fit": True,
                    "shuffle_interpretation": (
                        "unrestricted label-randomization reference under a strong "
                        "exchangeability assumption; not a calibrated conditional "
                        "p-value and never used to filter units or events"
                    ),
                    "label_shuffle_exchangeability_warning": (
                        "content labels are not generally exchangeable over duration, "
                        "occupied arm, stop context, or session time"
                    ),
                    "fold_sizes": np.bincount(folds, minlength=n_folds).tolist(),
                    "fold_support_radius_bins": int(np.abs(context.lags).max()),
                    "n_basis_per_kernel": int(context.spline_basis.shape[1]),
                    "n_units": int(context.counts.shape[1]),
                    "n_fit_bins": int(context.counts.shape[0]),
                    "n_events_by_class": {
                        name: int(len(context.events[name]))
                        for name in MARKED_EVENT_CLASSES
                    },
                    "n_events_with_scientific_mark_by_class": {
                        name: int(
                            ladder.events[name][f"{scientific_mark}_available"].sum()
                        )
                        for name in MARKED_EVENT_CLASSES
                    },
                    "undefined_mark_event_policy": (
                        "retain in ordinary and nuisance kernels; assign zero only "
                        "to the centered scientific-mark regressor; never permute "
                        "into or out of unavailable events"
                    ),
                    "runner_timing_seconds": {
                        "session_preparation": preparation_seconds,
                        "design_construction": design_seconds,
                        "fit_and_shuffle": fit_seconds,
                    },
                    "marked_diagnostics": fitted.diagnostics,
                    "cohort_report": cohort_report,
                    "cohort_info": context.cohort_info,
                    "nwb_file_name": nwb_file_name,
                    "epoch_name": epoch_name,
                    "jax_backend": gpu_record["jax_backend"],
                    "gpu_index": gpu_record["gpu_index"],
                    "gpu_uuid": gpu_record["gpu_uuid"],
                    "gpu_name": gpu_record["gpu_name"],
                    **context.input_provenance,
                }
            ),
        )
        logger.info("Saved %s and provenance %s", artifact, sidecar)
        return True
    except MemoryError:
        # A worker that ran out of memory will do so again on the next session,
        # only later and with a less legible traceback.
        logger.exception(
            "Marked-kernel analysis ran out of memory for %s", epoch_identifier
        )
        raise
    except Exception as error:
        logger.exception("Marked-kernel analysis failed for %s", epoch_identifier)
        if failure_sink is not None:
            failure_sink.append((type(error).__name__, str(error)[:120]))
        return False


def get_command_line_arguments() -> Namespace:
    """Parse and validate the marked-kernel production contract.

    Returns
    -------
    Namespace
        Validated CLI arguments.
    """
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("--animal")
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch")
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--brain-area", default="mPFC", choices=["mPFC", "OFC"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-directory", type=Path, default=Path("logs"))
    parser.add_argument(
        "--event-arm-table", type=Path, default=DEFAULT_EVENT_ARM_TABLE_PATH
    )
    parser.add_argument(
        "--scientific-mark",
        choices=sorted(MARK_DESCRIPTIONS),
        default="current_arm_fraction",
    )
    parser.add_argument(
        "--nuisance-marks",
        nargs="*",
        choices=[EVENT_DURATION_MARK],
        default=[EVENT_DURATION_MARK],
        help=(
            "Marked nuisance kernels fit jointly (default: log event duration). "
            "Pass the flag with no values to omit nuisance marks."
        ),
    )
    parser.add_argument("--penalty-sigma", type=float, default=DEFAULT_PENALTY_SIGMA)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_MARK_SHUFFLES,
        help=(
            f"Within-class mark permutations (default {DEFAULT_MARK_SHUFFLES}, the "
            "corpus setting). 1000 is an opt-in sensitivity run only: about 5 "
            "GPU-hours on the largest session, so GPU-days over the corpus."
        ),
    )
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument(
        "--design-batch-size", type=int, default=DEFAULT_NULL_DESIGN_BATCH_SIZE
    )
    args = parser.parse_args()
    if args.penalty_sigma <= 0 or not np.isfinite(args.penalty_sigma):
        parser.error("--penalty-sigma must be finite and positive")
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2")
    if args.n_shuffles < 0:
        parser.error("--n-shuffles cannot be negative")
    if 0 < args.n_shuffles < MIN_USABLE_NULL_DRAWS:
        parser.error(f"--n-shuffles must be 0 or at least {MIN_USABLE_NULL_DRAWS}")
    if args.design_batch_size < 1:
        parser.error("--design-batch-size must be positive")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    if len(args.nuisance_marks) != len(set(args.nuisance_marks)):
        parser.error("--nuisance-marks cannot contain duplicates")
    return args


def main() -> None:
    """Select sessions, claim one accelerator, and run them serially.

    Returns
    -------
    None
        Prints how many sessions carry a bundle and how many had no units.

    Raises
    ------
    RuntimeError
        If any session failed; the message names every failure and its cause.
    MemoryError
        Propagated immediately from the session that exhausted memory.
    """
    args = get_command_line_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)

    epoch_info = filter_epoch_info(
        get_epoch_info(),
        sessions=args.sessions,
        animal=args.animal,
        date=args.date,
        epoch=args.epoch,
    )
    if args.max_sessions is not None:
        epoch_info = epoch_info.iloc[: args.max_sessions]
    if epoch_info.empty:
        raise ValueError("the requested filters selected no sessions")

    gpu_record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
    print(
        f"Processing {len(epoch_info)} sessions on {gpu_record['jax_backend']} "
        f"with {args.n_shuffles} within-class mark shuffles"
    )
    failures: list[tuple[str, str, str]] = []
    n_with_bundle = 0
    n_without_units = 0
    for epoch in epoch_info.itertuples():
        nwb_file_name, epoch_name = epoch.Index[0], epoch.Index[1]
        failure_sink: list[tuple[str, str]] = []
        succeeded = run_analysis(
            nwb_file_name=nwb_file_name,
            epoch_name=epoch_name,
            brain_area=args.brain_area,
            output_dir=args.output_dir,
            log_directory=args.log_directory,
            overwrite=args.overwrite,
            scientific_mark=args.scientific_mark,
            nuisance_marks=tuple(args.nuisance_marks),
            penalty_sigma=args.penalty_sigma,
            n_folds=args.n_folds,
            n_shuffles=args.n_shuffles,
            fold_seed=args.fold_seed,
            shuffle_seed=args.shuffle_seed,
            design_batch_size=args.design_batch_size,
            event_arm_table_path=args.event_arm_table,
            gpu_record=gpu_record,
            failure_sink=failure_sink,
        )
        if not succeeded:
            kind, message = (
                failure_sink[0] if failure_sink else ("Unknown", "see the session log")
            )
            failures.append((f"{nwb_file_name} {epoch_name}", kind, message))
            continue
        # A no-units session writes the sidecar marker only, never the table.
        identifier = (
            f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{args.brain_area}"
        )
        artifact = marked_artifact_path(
            args.output_dir, identifier, args.scientific_mark
        )
        if artifact.is_file():
            n_with_bundle += 1
        else:
            n_without_units += 1
    print(
        f"{n_with_bundle} session(s) carry a marked-kernel bundle, "
        f"{n_without_units} had no {args.brain_area} units, "
        f"{len(failures)} failed"
    )
    if failures:
        raise RuntimeError(
            f"marked-kernel analysis failed for {len(failures)} session(s): "
            + "; ".join(
                f"{session}: {kind}: {message}" for session, kind, message in failures
            )
        )


if __name__ == "__main__":
    sys.exit(main())
