"""Batch PFC GLM analysis of neuron modulation during non-local hippocampal events.

This script runs GLM-based analysis of prefrontal cortex (mPFC/OFC) neuron firing
rate modulation around sharp-wave ripples (SWR) and non-local hippocampal events.
For each session, it fits Poisson GLM models with convolutional spline basis functions
to quantify how each neuron's firing rate changes relative to event onsets.

The analysis pipeline per session:
1. Load the session's non-local events from the canonical event table
2. Split them on that table's ``is_swr`` column (strict positive-duration
   overlap with the SWR source's speed-gated ``SWR_event`` labels, the direct
   Kay inventory of ``analysis/event_definition/swr_source.py``), so Figure 4
   describes the same events, with the same classification, as the content
   and rate families
3. Fit GLMs for each neuron comparing models: constant, SWR, non-local, combined
4. Compute theta modulation during running
5. Save per-session neuron statistics and modulation time courses

Requires HPC decoding results to already exist (run run_decoding_bandit.py first).

Usage
-----
# Analyze all mPFC sessions
python run_pfc_analysis_bandit.py --brain_area mPFC

# Analyze specific animal
python run_pfc_analysis_bandit.py --animal wilbur --brain_area mPFC

# Analyze specific session
python run_pfc_analysis_bandit.py --animal wilbur --date 20210329 --epoch 02_r1

# Overwrite existing results
python run_pfc_analysis_bandit.py --brain_area mPFC --overwrite

# Use more Dask workers
python run_pfc_analysis_bandit.py --brain_area mPFC --n_workers 16

# Sensitivity bundle: no stop restriction, written as *_modulation_allstops.*
python run_pfc_analysis_bandit.py --brain_area mPFC --stop-cohort all_stops

# Smoke one session into a scratch directory
python run_pfc_analysis_bandit.py --max-sessions 1 --output-dir /tmp/pfc_smoke

# Sensitivity bundle at another prior, written as *_modulation_sigma1.*
python run_pfc_analysis_bandit.py --brain_area mPFC --penalty-sigma 1

# One worker of a partitioned batch: its own sessions, its own card
python run_pfc_analysis_bandit.py --brain_area mPFC --n_workers 1 \
    --gpu 2 --sessions worker0.sessions
"""

import json
import logging
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from continuum_swr_replay.runs.threads import pin_blas_threads

# Pin BLAS/OpenMP threads to 1 per process to prevent thread oversubscription
# when running multiple Dask workers. Each worker runs one session serially,
# and statsmodels GLM (IRLS) is single-threaded anyway.
pin_blas_threads("NUMEXPR_NUM_THREADS")

import dask  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dask.distributed import Client, LocalCluster  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from continuum_swr_replay.analysis.event_definition.cohort_selection import (  # noqa: E402
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.pfc import bundle_files, run_session  # noqa: E402
from continuum_swr_replay.analysis.pfc.batched_glm import (  # noqa: E402
    DEFAULT_PENALTY_SIGMA,
    implied_prior_sd,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (  # noqa: E402
    IS_SWR_DEFINITION,
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (  # noqa: E402
    load_session_stop_cohort,
    split_sorted_epochs,
)
from continuum_swr_replay.analysis.pfc.event_sets import (  # noqa: E402
    EVENT_CLASSES,
    EVENT_SETS,
    EXTENDED_EVENT_CLASSES,
)
from continuum_swr_replay.analysis.pfc.heldout import (  # noqa: E402
    CONTROL_NULL_EVENT_POLICY,
    DEFAULT_CONTROL_DRAWS,
    EMPIRICAL_PVALUE_METHOD,
    MIN_USABLE_NULL_DRAWS,
    required_usable_null_draws,
)
from continuum_swr_replay.analysis.pfc.load_population import (  # noqa: E402
    penalty_sigma_suffix,
)
from continuum_swr_replay.analysis.pfc.matched_control import (  # noqa: E402
    MATCHED_CONTROL_PLACEMENT_METHOD,
)
from continuum_swr_replay.analysis.pfc.modulation import (  # noqa: E402
    EXTENDED_CURVE_SOURCES,
    in_report_window,
)
from continuum_swr_replay.analysis.pfc.runner_support import (  # noqa: E402
    session_seed,
)
from continuum_swr_replay.analysis.pfc.scoring import (  # noqa: E402
    extended_curves,
    score_session,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analyzed_epochs,
)
from continuum_swr_replay.data_loaders.spikes import (  # noqa: E402
    pfc_sorted_nwb_file_names,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    format_epoch_identifier,
)
from continuum_swr_replay.parameters import (  # noqa: E402
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.paths import (  # noqa: E402
    pfc_cohort_name,
    processed_data_dir,
    table_path,
)
from continuum_swr_replay.runs.batch import (  # noqa: E402
    raise_open_file_limit,
    release_nwb_files,
    setup_logger,
)
from continuum_swr_replay.runs.cli import (  # noqa: E402
    add_epoch_filters,
    read_session_list,
)
from continuum_swr_replay.runs.device import configure_jax_gpu  # noqa: E402
from continuum_swr_replay.runs.fit_provenance import (  # noqa: E402
    git_hash as loaded_commit,
)
from continuum_swr_replay.runs.fit_provenance import (  # noqa: E402
    matching_sidecar,
    write_fit_provenance,
)
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.io import write_atomically  # noqa: E402

#: Populations the PFC GLMs may be fit on. ``at_well`` is the canonical shared
#: cohort and publishes the unsuffixed artifact names; ``all_stops`` applies no
#: stop restriction and publishes ``_allstops`` names. ``in_transit`` is not
#: offered here -- there is no power for it at the single-unit level.
STOP_COHORTS = ("at_well", "all_stops")

warnings.filterwarnings("ignore", message=".*Skipped checksum.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="datajoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hdmf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynwb")

#: Prior standard deviation of an event-basis coefficient, on the log-rate
#: scale, that the canonical bundle is fit at. What it says about a *curve* is
#: not this number -- the coefficients are B-spline weights -- so the sidecar
#: records the implied per-lag and sustained standard deviations beside it,
#: from ``implied_prior_sd``.
CANONICAL_PENALTY_SIGMA = DEFAULT_PENALTY_SIGMA

#: The estimator and the inference each bundle is produced by, recorded in the
#: provenance sidecar and compared by the skip check: a bundle fit by maximum
#: likelihood and tested by a likelihood ratio is not this runner's bundle.
ESTIMATOR = "ridge_poisson_glm"
INFERENCE = "blocked_heldout_deviance"

#: Columns the maximum-likelihood ladder wrote that this estimator cannot
#: produce. A penalized fit has no likelihood-ratio reference distribution and
#: no AIC at a stated number of parameters, so the statistics table must not
#: carry either: a reader who found a ``: pval`` column beside a ridge fit
#: would read an asymptotic test of a model that was never fit unpenalized.
RETIRED_STATISTIC_SUFFIXES = (": lr", ": pval", ": aic")
RETIRED_STATISTIC_COLUMNS = ("min_aic_model", "min_aic_model_replay_split")


def modulation_path(
    save_folder: Path,
    epoch_identifier: str,
    stop_cohort: str,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
) -> Path:
    """Artifact path for one session, named for how it was cut and fit.

    The canonical at-well bundle at the canonical prior keeps the unsuffixed
    name every reader already hard-codes, so a sensitivity bundle -- another
    population, or another prior -- can never be mistaken for it.
    """
    suffix = stop_cohort_suffix(stop_cohort) + penalty_sigma_suffix(penalty_sigma)
    return bundle_files.bundle_path(save_folder, epoch_identifier, suffix)


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
    stop_cohort: str = "at_well",
    stop_table_sha256: str | None = None,
    event_table_sha256: str | None = None,
    data_over_time_sha256: str | None = None,
    fitter: str | None = None,
    *,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
    n_folds: int | None = None,
    fold_seed: int | None = None,
    n_control_draws: int | None = None,
    control_draw_seed: int | None = None,
    swr_interval_source: str | None = None,
    swr_events_sha256: str | None = None,
    event_set: str = "canonical",
    null_classes: Sequence[str] | None = None,
) -> bool:
    """Check if this session's bundle already exists *for this population*.

    Distinct from ``decoding.files.results_exist`` because the PFC
    pipeline writes a different set of files (modulation .npy arrays
    plus a single .pkl) than the decoder pipeline (results .nc, detector
    .pkl, environment .pkl). The file names alone never settle it. A bundle
    written before the matched non-event control existed is missing that
    array, so it is refit rather than kept under a name a current run wants. An at-well
    bundle never satisfies an ``all_stops`` request; a bundle that does not
    *state* its population in a provenance sidecar is not accepted as one (the
    canonical name is unsuffixed, so a bundle fit before the cohort existed
    would otherwise be kept silently by a skip); the stop table the bundle was
    cut from must be the one on disk now, because regenerating the stop table
    changes which bins and events the cohort contains; and the bundle must
    state the canonical event definition and have been cut from the event
    table and cached decode frame now on disk, because a bundle fit on a
    run-time re-detection of events, or on an earlier generation of those two
    inputs, describes different events.

    Parameters
    ----------
    stop_table_sha256 : str or None
        Hash of the stop table this run would use, as
        ``load_session_stop_cohort`` reports it. ``None`` for ``all_stops``,
        which reads no stop table -- and which therefore only matches a sidecar
        that also recorded no stop table.
    event_table_sha256, data_over_time_sha256 : str or None
        Hashes of the canonical event table and the session's cached decode
        frame this run would read. ``None`` matches no existing bundle, so a
        session whose inputs could not be hashed is refit rather than skipped.
    swr_interval_source, swr_events_sha256 : str or None
        The SWR source (``swr_source_name()``) and the hash of its events file this
        run would read the SWR intervals from: a bundle fit on the stored
        ripples, or on another inventory run, describes other SWRs.
    fitter : str or None
        The backend this run would fit with. A bundle fit with the other one is
        not this run's bundle: the two agree to the equivalence audit's
        tolerances, not bit-for-bit, so a batch that means to be one estimator
        throughout must not silently keep sessions from the other. ``None``
        (the default) matches a sidecar that names no fitter, which is how a
        bundle written before the flag existed is read.
    penalty_sigma : float
        Prior this run would fit at. It is already in the artifact name for a
        sensitivity bundle, but the canonical name is unsuffixed, so the
        sidecar is what tells a ridge bundle from the maximum-likelihood one
        that occupied the same name before.
    n_folds, fold_seed, n_control_draws, control_draw_seed : int or None
        The inference this run would produce. A bundle scored on a different
        number of folds, a different split, or a null of a different size is
        not this run's bundle: its held-out differences and its empirical
    p-values are not the ones a current batch means to ship. ``None``
    matches a sidecar that records no value, which is how a bundle written
    before the held-out inference existed is read -- and such a bundle also
    fails the estimator comparison above, so it is refit.

    An extended-event-set bundle (``event_set``) must carry its two extra
    curves and record the same event set and the same null classes
    (``null_classes``; None is every class); a sidecar written before the
    event set existed reads as ``"canonical"``.

    The sidecar must also name the current matched-control placement method.
    The former disjoint-duration-window policy can leave every null draw
    thinner than the real class on dense bouts, so those bundles do not carry
    the same empirical test even when their p-value column names match.
    """
    if overwrite:
        return False
    artifact = modulation_path(
        save_folder, epoch_identifier, stop_cohort, penalty_sigma
    )
    save_base = artifact.with_suffix("")
    required_files = [
        save_base.with_suffix(".pkl"),
        save_base.with_suffix(".SWR_modulation.npy"),
        save_base.with_suffix(".no_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_no_SWR_modulation.npy"),
        save_base.with_suffix(".matched_control_modulation.npy"),
    ]
    if event_set != "canonical":
        required_files += [
            save_base.with_suffix(f".{name}_modulation.npy")
            for name in EXTENDED_CURVE_SOURCES
        ]
    if not all(f.exists() for f in required_files):
        return False
    record = matching_sidecar(
        artifact,
        {
            "null_classes": None if null_classes is None else sorted(null_classes),
            "stop_cohort": stop_cohort,
            "stop_table_sha256": stop_table_sha256,
            "is_swr_definition": IS_SWR_DEFINITION,
            "canonical_event_table_sha256": event_table_sha256,
            "data_over_time_sha256": data_over_time_sha256,
            "swr_interval_source": swr_interval_source,
            "swr_events_sha256": swr_events_sha256,
            "fitter": fitter,
            "estimator": ESTIMATOR,
            "inference": INFERENCE,
            "penalty_sigma_log_rate": penalty_sigma,
            "n_folds": n_folds,
            "fold_seed": fold_seed,
            "n_control_draws": n_control_draws,
            "control_draw_seed": control_draw_seed,
            "empirical_pvalue_method": EMPIRICAL_PVALUE_METHOD,
            "control_null_event_policy": CONTROL_NULL_EVENT_POLICY,
            "min_usable_null_draws": (
                None
                if n_control_draws is None
                else required_usable_null_draws(n_control_draws)
            ),
            "matched_control_placement_method": MATCHED_CONTROL_PLACEMENT_METHOD,
        },
    )
    return (
        record is not None
        and record.get("event_set", "canonical") == event_set
        and (
            n_control_draws in (None, 0)
            or isinstance(record.get("null_usable_draws"), dict)
        )
    )


def drop_retired_statistics(neuron_stats: pd.DataFrame) -> pd.DataFrame:
    """Remove the maximum-likelihood ladder's columns from a statistics table.

    The per-unit fits are penalized and the inference is out of fold, so the
    likelihood-ratio statistics, their asymptotic p-values and the AICs are not
    quantities this estimator produces. They are dropped before the bundle is
    written rather than shipped alongside the held-out columns, where a reader
    could pick either.
    """
    retired = [
        column
        for column in neuron_stats.columns
        if column.endswith(RETIRED_STATISTIC_SUFFIXES)
        or column in RETIRED_STATISTIC_COLUMNS
    ]
    return neuron_stats.drop(columns=retired)


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    log_directory: str | Path = "logs",
    overwrite: bool = False,
    git_hash: str | None = None,
    stop_cohort: str = "at_well",
    output_dir: Path | None = None,
    fitter: str = "jax",
    fit_diagnostics: bool = False,
    gpu: int | None = None,
    allow_cpu: bool = False,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
    n_folds: int = 5,
    n_control_draws: int = DEFAULT_CONTROL_DRAWS,
    fold_seed: int | None = None,
    control_draw_seed: int | None = None,
    event_set: str = "canonical",
    null_classes: Sequence[str] | None = None,
    *,
    cli: list[str],
) -> None:
    """Run PFC GLM analysis for a single session.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "wilbur20210329_.nwb")
    epoch_name : str
        Epoch name (e.g., "02_r1")
    brain_area : str
        Brain area to analyze ("mPFC" or "OFC")
    sampling_frequency : int
        Sampling frequency in Hz
    log_directory : str | Path
        Directory for log files
    overwrite : bool
        Whether to overwrite existing results
    git_hash : str, optional
        The commit ``main`` loaded, logged and recorded in the sidecar. A dask
        worker imports the code later than ``main`` does, after the checkout
        may have moved on, so ``main`` passes its own. Omitted, it is the
        commit this process loaded.
    stop_cohort : {"at_well", "all_stops"}
        Population to fit: the well-anchored stops the hippocampal ladders are
        read on, or no stop restriction at all.
    output_dir : Path, optional
        Directory the artifacts and their provenance sidecar are written to;
        ``paths.processed_data_dir()`` when None.
    fitter : {"jax"}
        How the GLMs are fit: the batched backend, which fits every neuron of
        the session against each design at once and is the only one with a
        penalized path. The sidecar records it.
    fit_diagnostics : bool
        Also write ``<artifact>.fit_diagnostics.npz`` -- per-neuron, per-model
        parameters and fit criteria, for the fitter equivalence audit. Forces
        the serial neuron path.
    gpu : int, optional
        Fit on this GPU instead of the emptiest free one. The card is still
        claimed against sibling workers, but its occupancy is not checked.
    allow_cpu : bool
        Permit the batched fitter to run on CPU when no GPU is free. Off by
        default: the CPU path is many times slower, and a batch that quietly
        fell back to it would not be noticed until it overran.
    penalty_sigma : float
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale. A value other than the canonical one writes a suffixed
        sensitivity bundle and skips the null, whose draws would then describe
        a prior the canonical bundle is not fit under.
    n_folds : int
        Folds the held-out deviance is summed over. Whole immobility runs whose
        event-kernel supports overlap are assigned together, so one event
        cannot enter both train and test.
    n_control_draws : int
        Draws of the matched non-event control the observed held-out
        improvements are read against. ``0`` skips the null.
    fold_seed, control_draw_seed : int, optional
        Seeds of the fold split and of the null draws. Derived from the session
        key when omitted, so a session refit on its own reproduces the batch's
        split and null exactly.
    event_set : {"canonical", "with_swr_without_replay"}
        ``"with_swr_without_replay"`` adds the SWRs that overlap no non-local
        event as their own class, with its models, its null, and two more
        curves (``<artifact>.SWR_no_non_local_modulation.npy`` and
        ``<artifact>.non_local_SWR_3class_modulation.npy``).
    null_classes : sequence of str, optional
        Draw the null for these classes only (every class when None).
    cli : list of str
        The command line recorded in the sidecar: ``main``'s program and the
        arguments it parsed.

    Notes
    -----
    With ``fitter="jax"`` the device is chosen before anything is loaded, and a
    device that cannot be chosen ends the session rather than degrading it: no
    free card (and no ``allow_cpu``) raises ``NoFreeGPUError``, and a process
    that had already initialized a JAX backend on every card raises
    ``WrongJAXDeviceError``. Both propagate out of this function -- they are
    raised before the ``try`` that turns a per-session failure into a log line,
    because they are a fact about the host rather than about the session, and a
    batch that hits them should stop rather than work through 70 more sessions
    failing the same way.
    """
    if output_dir is None:
        output_dir = processed_data_dir()
    epoch_identifier = format_epoch_identifier(nwb_file_name, epoch_name, brain_area)
    save_file = modulation_path(
        output_dir, epoch_identifier, stop_cohort, penalty_sigma
    )
    # A dask worker imports the library when it starts, which can be hours
    # after main did; if the checkout moved on in between, it runs newer code
    # than the commit main passed in.
    try:
        worker_commit = loaded_commit()
    except RuntimeError:
        # A worker that could not read git must not fail a session main has a
        # commit for; run on its own, the session has no commit to record.
        if git_hash is None:
            raise
        worker_commit = "unknown"
    if git_hash is None:
        git_hash = worker_commit
    if fold_seed is None:
        fold_seed = session_seed("folds", nwb_file_name, epoch_name)
    if control_draw_seed is None:
        control_draw_seed = session_seed("control_draws", nwb_file_name, epoch_name)
    # A bundle fit under another prior is a sensitivity bundle: it is written
    # for its modulation curves, and a null drawn under a prior the canonical
    # bundle is not fit under would not be a null for anything shipped.
    if float(penalty_sigma) != CANONICAL_PENALTY_SIGMA:
        n_control_draws = 0

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    # The cohort is resolved before the skip check, not after: the check needs
    # the stop table's hash to tell a bundle cut from the current table from one
    # cut from a previous generation of it. Reading and hashing the table is
    # seconds against a multi-hour fit.
    if stop_cohort == "all_stops":
        cohort_stops, cohort_report = None, {"cohort": stop_cohort}
    else:
        try:
            cohort_stops, cohort_report = load_session_stop_cohort(
                nwb_file_name, epoch_name, cohort=stop_cohort
            )
        except Exception as e:
            # One session missing from the stop table must not abort the batch;
            # the log is the per-session failure record, as it is for a failed
            # fit below.
            logger.exception(f"No stop cohort for {epoch_identifier}: {e}")
            return

    # The event definition is resolved before the skip check for the same
    # reason the cohort is: a bundle cut from a previous generation of the
    # canonical event table or of the cached decode frame describes different
    # events, so it is not this run's bundle.
    event_inputs = canonical_event_input_hashes(nwb_file_name, epoch_name)

    if results_exist(
        output_dir,
        epoch_identifier,
        overwrite,
        stop_cohort,
        cohort_report.get("stop_table_sha256"),
        event_inputs["canonical_event_table_sha256"],
        event_inputs["data_over_time_sha256"],
        fitter,
        penalty_sigma=penalty_sigma,
        n_folds=n_folds,
        fold_seed=fold_seed,
        n_control_draws=n_control_draws,
        control_draw_seed=control_draw_seed,
        swr_interval_source=event_inputs["swr_interval_source"],
        swr_events_sha256=event_inputs["swr_events_sha256"],
        event_set=event_set,
        null_classes=null_classes,
    ):
        logger.info(f"Results already exist for {epoch_identifier}. Skipping...")
        return

    logger.info(" START ".center(50, "#"))
    logger.info(f"Git Hash: {git_hash}")
    if worker_commit != git_hash:
        logger.warning(
            f"this process loaded {worker_commit}, not {git_hash}: the "
            "checkout changed after main started; the sidecar records both"
        )
    logger.info(f"brain_area: {brain_area}")
    logger.info(f"sampling_frequency: {sampling_frequency}")
    logger.info(f"immobility_threshold: {DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC}")
    logger.info(f"stop_cohort: {stop_cohort}")
    logger.info(f"event_set: {event_set} (null classes: {null_classes or 'all'})")
    logger.info(f"fitter: {fitter}")
    logger.info(f"penalty_sigma_log_rate: {penalty_sigma}")
    logger.info(f"n_folds: {n_folds} (fold_seed {fold_seed})")
    logger.info(
        f"n_control_draws: {n_control_draws} (control_draw_seed {control_draw_seed})"
    )

    diagnostics: dict | None = {} if fit_diagnostics else None
    # Filled by ``run_session`` with the bins, counts, designs and events it
    # fit on, so the out-of-fold score is taken on exactly that material
    # rather than on a second assembly of it.
    scoring_context: dict[str, object] = {}

    # Before JAX initializes a backend -- later than the import, and checked
    # afterward -- pick the card this process will fit on and bound what it
    # takes of it. Each Dask worker runs this for itself; the claim keeps two
    # of them off the same card, and a worker handed a second session gets the
    # card it is already on rather than a second one.
    gpu_record: dict[str, object] = {}
    if fitter == "jax":
        gpu_record = configure_jax_gpu(gpu=gpu, allow_cpu=allow_cpu)
        logger.info(
            "JAX device: %s%s",
            gpu_record["jax_backend"],
            (
                f" (GPU {gpu_record['gpu_index']}, {gpu_record['gpu_name']}, "
                f"{gpu_record['gpu_uuid']})"
                if gpu_record["gpu_index"] is not None
                else ""
            ),
        )

    try:
        logger.info(f"Processing {epoch_identifier}...")
        if cohort_stops is not None:
            logger.info(
                "Stop cohort %s: %d/%d stops (%s)",
                stop_cohort,
                cohort_report["n_stops_after"],
                cohort_report["n_stops_before"],
                cohort_report["rule"],
            )
        (
            neuron_stats,
            SWR_modulation,
            no_SWR_modulation,
            non_local_SWR_modulation,
            non_local_no_SWR_modulation,
            matched_control_modulation,
            cohort_info,
        ) = run_session(
            nwb_file_name,
            epoch_name,
            brain_area=brain_area,
            sampling_frequency=sampling_frequency,
            run_parallel=False,
            stop_cohort=stop_cohort,
            cohort_stops=cohort_stops,
            fitter=fitter,
            fit_diagnostics_out=diagnostics,
            penalty_sigma=penalty_sigma,
            scoring_out=scoring_context,
            event_set=event_set,
        )

        logger.info(f"Analyzed {len(neuron_stats)} neurons")

        # The statistics table's inference is the held-out score, so the
        # likelihood-ratio columns the maximum-likelihood ladder wrote are
        # dropped and the held-out ones take their place.
        neuron_stats = drop_retired_statistics(neuron_stats)
        n_basis_per_class: int | None = None
        implied_prior: dict[str, float] = {}
        scoring_diagnostics: dict[str, object] = {}
        extra_curves: dict[str, NDArray] = {}
        if len(neuron_stats) and event_set != "canonical":
            extra_curves = extended_curves(scoring_context, penalty_sigma)
        if len(neuron_stats):
            context = scoring_context
            n_basis_per_class = int(context["spline_basis"].shape[1])
            # What the prior says about a curve, not about a coefficient: the
            # basis is a partition of unity, so the prior on the kernel at one
            # lag is wider than sigma and the prior on a modulation held across
            # the window is narrower. Both are recorded, over the reported
            # window rather than the wider one the models are fit on.
            reported = in_report_window(context["lag_time"])
            implied_prior = implied_prior_sd(
                penalty_sigma, context["spline_basis"], reported
            )
            heldout_stats, scoring_diagnostics = score_session(
                context,
                modulation_curves={
                    "SWR": SWR_modulation,
                    "no_SWR": no_SWR_modulation,
                    "non_local_SWR": non_local_SWR_modulation,
                    "non_local_no_SWR": non_local_no_SWR_modulation,
                    "matched_control": matched_control_modulation,
                    **extra_curves,
                },
                penalty_sigma=penalty_sigma,
                n_folds=n_folds,
                fold_seed=fold_seed,
                n_control_draws=n_control_draws,
                control_draw_seed=control_draw_seed,
                logger=logger,
                null_classes=null_classes,
            )
            if len(heldout_stats) != len(neuron_stats):
                raise ValueError(
                    f"the held-out score describes {len(heldout_stats)} units "
                    f"and the fit {len(neuron_stats)}; they are the same "
                    "neurons in the same order or they are not comparable"
                )
            neuron_stats = pd.concat(
                [neuron_stats, heldout_stats.set_index(neuron_stats.index)], axis=1
            )
            logger.info(
                "Fold sizes: %s; events per fold: %s",
                scoring_diagnostics["fold_sizes"],
                scoring_diagnostics["events_per_fold"],
            )

        logger.info("Saving results...")

        save_file.parent.mkdir(parents=True, exist_ok=True)
        neuron_stats.to_pickle(save_file)
        np.save(
            save_file.with_suffix(".SWR_modulation.npy"),
            SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".no_SWR_modulation.npy"),
            no_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_SWR_modulation.npy"),
            non_local_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_no_SWR_modulation.npy"),
            non_local_no_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".matched_control_modulation.npy"),
            matched_control_modulation,
        )
        if event_set != "canonical":
            for name in EXTENDED_CURVE_SOURCES:
                np.save(
                    save_file.with_suffix(f".{name}_modulation.npy"),
                    extra_curves.get(name, np.empty((0, 0))),
                )

        if diagnostics:
            # Per-neuron, per-model parameters and fit criteria. Not part of
            # the bundle any reader loads; written only when a run asks for it,
            # so the fitter audit can compare the two backends at the fit level
            # rather than only through the statistics they feed.
            diagnostics_file = save_file.with_suffix(".fit_diagnostics.npz")
            np.savez_compressed(
                diagnostics_file,
                **{
                    f"{model_name}|{quantity}": values
                    for model_name, record in diagnostics.items()
                    for quantity, values in record.items()
                },
            )
            logger.info(f"Wrote fit diagnostics to {diagnostics_file}")

        # Every artifact states the population it was fit on: which cohort,
        # which stop table, and how many bins and events survived it.
        sidecar = write_fit_provenance(
            save_file,
            {"stop_table": table_path("stop_table")}
            if cohort_stops is not None
            else {},
            estimator=ESTIMATOR,
            cli=cli,
            commit=git_hash,
            extra={
                **(
                    {"worker_git_hash": worker_commit}
                    if worker_commit != git_hash
                    else {}
                ),
                "stop_cohort": stop_cohort,
                "event_set": event_set,
                "null_classes": (
                    None if null_classes is None else sorted(null_classes)
                ),
                **(
                    {
                        key: cohort_info.get(key)
                        for key in (
                            "n_swr_with_non_local",
                            "n_swr_no_non_local",
                            "n_swr_no_non_local_unknown",
                        )
                    }
                    if event_set != "canonical"
                    else {}
                ),
                # The estimator and the inference, in full: the prior the
                # coefficients were fit under, the blocked split the models
                # were scored out of, and the null the improvements are read
                # against. A bundle that does not state these cannot be told
                # from the maximum-likelihood one that held the same name.
                "inference": INFERENCE,
                "penalty_sigma_log_rate": float(penalty_sigma),
                "n_folds": int(n_folds),
                "fold_seed": int(fold_seed),
                "n_control_draws": int(n_control_draws),
                "control_draw_seed": int(control_draw_seed),
                "empirical_pvalue_method": EMPIRICAL_PVALUE_METHOD,
                "control_null_event_policy": CONTROL_NULL_EVENT_POLICY,
                "min_usable_null_draws": required_usable_null_draws(n_control_draws),
                "matched_control_placement_method": (MATCHED_CONTROL_PLACEMENT_METHOD),
                "n_basis_per_class": n_basis_per_class,
                "implied_prior_sd": implied_prior,
                "fold_sizes": scoring_diagnostics.get("fold_sizes"),
                "fold_support_radius_bins": scoring_diagnostics.get(
                    "fold_support_radius_bins"
                ),
                "events_per_fold": scoring_diagnostics.get("events_per_fold"),
                "non_estimable_models": scoring_diagnostics.get("non_estimable_models"),
                # Which real onsets enter the fixed matched test, and how many
                # windows each null draw actually placed against that tested
                # count. A thinner draw is not the reference the subset-
                # matched observed statistic needs.
                "null_windows": scoring_diagnostics.get("null_windows"),
                "null_usable_draws": scoring_diagnostics.get("null_usable_draws"),
                "engagement_null_overlap": scoring_diagnostics.get(
                    "engagement_null_overlap"
                ),
                "stop_table_sha256": cohort_report.get("stop_table_sha256"),
                "stop_cohort_report": cohort_report,
                "cohort_counts": cohort_info,
                "brain_area": brain_area,
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch_name,
                # How the models were fit: which backend, the convergence
                # tolerance both backends stop on, and -- for the batched
                # fitter -- the device and JAX version, since a GPU and a CPU
                # run are not bit-identical.
                "fitter": cohort_info.get("fitter"),
                "fitter_tol": cohort_info.get("fitter_tol"),
                "jax_backend": cohort_info.get("jax_backend"),
                "jax_version": cohort_info.get("jax_version"),
                # Which card the batched fits ran on, so a bundle names the
                # device as well as the estimator.
                "gpu_index": gpu_record.get("gpu_index"),
                "gpu_uuid": gpu_record.get("gpu_uuid"),
                "gpu_name": gpu_record.get("gpu_name"),
                # How many of the shipped fits never settled. A unit x model
                # that never settled is non-estimable for that neuron alone --
                # every statistic that reads it is NaN -- so these counts are
                # the only record of how much was dropped and from where.
                "n_model_fits": cohort_info.get("n_model_fits"),
                "n_nonconverged_model_fits": cohort_info.get(
                    "n_nonconverged_model_fits"
                ),
                "n_neurons_with_nonconverged_model": cohort_info.get(
                    "n_neurons_with_nonconverged_model"
                ),
                # Which events the fit describes, from the same block the skip
                # check above compares against.
                **event_inputs,
                # The cohort-restricted event counts, so the sidecar states the
                # split without a reader recomputing it.
                "n_events_total": cohort_info.get("n_nonlocal_after"),
                "n_events_swr": cohort_info.get("n_nonlocal_swr"),
                "n_events_non_swr": cohort_info.get("n_nonlocal_no_swr"),
                # The matched non-event control the engagement test is read
                # against: how it was drawn, and how many of the session's
                # events got one.
                "matched_control_seed": cohort_info.get("matched_control_seed"),
                "n_matched_control": cohort_info.get("n_matched_control"),
                "n_matched_control_unplaced": cohort_info.get(
                    "n_matched_control_unplaced"
                ),
                # ...and split by why: an onset outside every fitted bout is a
                # cohort fact, a saturated bout is an event-density one.
                "n_matched_control_onset_outside_fitted_bout": cohort_info.get(
                    "n_matched_control_onset_outside_fitted_bout"
                ),
                "n_matched_control_bout_saturated": cohort_info.get(
                    "n_matched_control_bout_saturated"
                ),
            },
        )

        logger.info(f"Saved results to {save_file}")
        logger.info(f"Wrote provenance to {sidecar}")
        logger.info("Done!")
    except Exception as e:
        logger.exception(f"Error processing {epoch_identifier}: {e}")


def run_analysis_then_release(
    nwb_file_name: str, epoch_name: str, brain_area: str = "mPFC", **kwargs: Any
) -> None:
    """``run_analysis``, then close the NWB files Spyglass opened for it.

    ``main``'s workers run one session at a time (``threads_per_worker=1``), and
    its serial path runs them one after another, so no other session in the
    process can still be reading those files (``release_nwb_files``). The release
    wraps ``run_analysis`` rather than living in it because that returns early
    and turns most failures into a log line: here it runs whether the session
    is fit, skipped, failed or raised. Whether the files were closed goes to
    the session's log.
    """
    try:
        run_analysis(nwb_file_name, epoch_name, brain_area=brain_area, **kwargs)
    finally:
        logging.getLogger(
            format_epoch_identifier(nwb_file_name, epoch_name, brain_area)
        ).info("NWB files Spyglass held open closed: %s", release_nwb_files())


def get_command_line_arguments(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local_scratch_dir",
        type=str,
        default="/tmp/dask_scratch",
        help="Local scratch directory for Dask workers (default: /tmp/dask_scratch)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of Dask workers (default: 4)",
    )
    add_epoch_filters(parser)
    parser.add_argument(
        "--brain_area",
        type=str,
        default="mPFC",
        choices=["mPFC", "OFC"],
        help="Brain area to analyze: mPFC or OFC (default: mPFC)",
    )
    parser.add_argument(
        "--log_directory",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--stop-cohort",
        type=str,
        default="at_well",
        choices=list(STOP_COHORTS),
        help=(
            "Population to fit (default: at_well). 'at_well' keeps only the "
            "immobility bins and events inside the well-anchored stops the "
            "hippocampal rate and content ladders are read on, and publishes "
            "the canonical artifact names; 'all_stops' applies no stop "
            "restriction and publishes '_allstops' names."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=processed_data_dir(),
        help=(
            "Directory for the modulation artifacts and their provenance "
            f"sidecars (default: {processed_data_dir()})"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Process at most this many sessions (smoke runs)",
    )
    parser.add_argument(
        "--fitter",
        type=str,
        default="jax",
        choices=["jax"],
        help=(
            "How the GLMs are fit (default and only choice: jax, the batched "
            "IRLS that fits every neuron of a session against each design at "
            "once). The per-neuron statsmodels IRLS has no penalized path, so "
            "a batch cannot be launched on it; it survives in the library as "
            "the control the fitter equivalence tests are read against."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "Fit on this GPU instead of the emptiest free one. The card is "
            "still claimed against sibling workers, but its occupancy is not "
            "checked."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Permit the batched fitter to run on CPU when no GPU is free. Off "
            "by default: the CPU path is many times slower, and a batch that "
            "quietly fell back to it would not be noticed until it overran."
        ),
    )
    parser.add_argument(
        "--fit-diagnostics",
        action="store_true",
        help=(
            "Also write <artifact>.fit_diagnostics.npz: per-neuron, per-model "
            "parameters and fit criteria, for the fitter equivalence audit. "
            "Forces the serial neuron path."
        ),
    )
    parser.add_argument(
        "--penalty-sigma",
        type=float,
        default=CANONICAL_PENALTY_SIGMA,
        help=(
            "Prior standard deviation of an event-basis coefficient, on the "
            f"log-rate scale (default: {CANONICAL_PENALTY_SIGMA}). A "
            "coefficient of this basis is a log-rate modulation near its knot, "
            "so the value reads as a rate multiplier. Another value writes a "
            "'_sigma<value>' sensitivity bundle and skips the null, which "
            "would otherwise describe a prior nothing is shipped under."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help=(
            "Folds the held-out deviance is summed over (default: 5). Whole "
            "immobility runs whose event-kernel supports overlap are assigned "
            "together, so one event cannot enter both train and test."
        ),
    )
    parser.add_argument(
        "--n-control-draws",
        type=int,
        default=DEFAULT_CONTROL_DRAWS,
        help=(
            "Draws of the matched non-event control that make the null each "
            f"held-out improvement is read against (default: "
            f"{DEFAULT_CONTROL_DRAWS}). 0 skips the null, and no empirical "
            "p-value is written."
        ),
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help=(
            "Seed of the run-to-fold assignment. Derived from the session key "
            "when omitted, so a session refit on its own reproduces the "
            "batch's split."
        ),
    )
    parser.add_argument(
        "--control-draw-seed",
        type=int,
        default=None,
        help=(
            "Seed the null's draw seeds are derived from. Derived from the "
            "session key when omitted."
        ),
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help=(
            "Fit only the sessions named in this file: one "
            "'<nwb_file_name> <epoch_name>' per line (comma or whitespace "
            "separated), '#' comments and blank lines ignored. For splitting "
            "a corpus across workers that cannot be named by animal or date. "
            "A named session that does not exist is an error, so a mistyped "
            "partition cannot quietly fit nothing."
        ),
    )

    parser.add_argument(
        "--event-set",
        type=str,
        default="canonical",
        choices=list(EVENT_SETS),
        help=(
            "Event classes to fit (default: canonical). "
            "'with_swr_without_replay' adds the SWRs that overlap no non-local "
            "event as a fifth class, with its own models, null and curves; it "
            "must write to its own --output-dir, never beside the canonical "
            "bundles."
        ),
    )
    parser.add_argument(
        "--null-classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Draw the matched-control null for these event classes only "
            "(default: every class of the event set). Recorded in the sidecar."
        ),
    )

    args = parser.parse_args(argv)
    if args.event_set != "canonical" and (
        args.output_dir.resolve() == processed_data_dir().resolve()
    ):
        parser.error(
            f"--event-set {args.event_set} writes bundles with the canonical "
            "names; give it its own --output-dir so they can never sit beside "
            "the canonical Figure 4 bundles."
        )
    if args.null_classes is not None:
        allowed = (
            EXTENDED_EVENT_CLASSES if args.event_set != "canonical" else EVENT_CLASSES
        )
        unknown = sorted(set(args.null_classes) - set(allowed))
        if unknown:
            parser.error(f"--null-classes {unknown} not in {list(allowed)}")
    if args.penalty_sigma <= 0:
        parser.error(
            f"--penalty-sigma {args.penalty_sigma} is not positive; it is a "
            "prior standard deviation on the log-rate scale."
        )
    if args.n_folds < 2:
        parser.error(
            f"--n-folds {args.n_folds} leaves nothing held out; the held-out "
            "deviance needs at least two folds."
        )
    if args.n_control_draws < 0:
        parser.error(
            f"--n-control-draws {args.n_control_draws} is negative; 0 skips the null."
        )
    if 0 < args.n_control_draws < MIN_USABLE_NULL_DRAWS:
        parser.error(
            f"--n-control-draws {args.n_control_draws} is below the "
            f"{MIN_USABLE_NULL_DRAWS}-draw reporting floor; use 0 to skip "
            "the null or request at least that many draws."
        )
    if args.gpu is not None and args.n_workers > 1:
        # --gpu names one card and every worker would be handed the same one:
        # they would all claim it, all preallocate a share of it, and every
        # sidecar would name a card most of them are not really alone on. One
        # worker per card is the layout that works -- separate processes, each
        # with its own --gpu and its own session list.
        parser.error(
            f"--gpu {args.gpu} names one card but --n_workers is "
            f"{args.n_workers}: every worker would be sent to that card. Run "
            "one process per card (--n_workers 1 --gpu N --sessions <list>), "
            "or drop --gpu and let each worker claim a free card of its own."
        )
    return args


def _session_keys(epoch_info: pd.DataFrame) -> list[tuple[str, str]]:
    """``(nwb_file_name, epoch_name)`` for every row, in row order."""
    return list(
        zip(
            epoch_info.index.get_level_values("nwb_file_name"),
            epoch_info.index.get_level_values("interval_list_name"),
            strict=True,
        )
    )


def _filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Filter epoch_info based on command line arguments.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        All available epochs
    args : Namespace
        Parsed command line arguments

    Returns
    -------
    pd.DataFrame
        Filtered epochs
    """
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    if getattr(args, "sessions", None) is not None:
        wanted = read_session_list(args.sessions)
        keys = _session_keys(epoch_info)
        missing = [key for key in wanted if key not in set(keys)]
        if missing:
            raise ValueError(
                f"{args.sessions} names {len(missing)} session(s) that are not "
                "analyzed epochs of the epoch register: "
                + ", ".join(f"{name} {epoch}" for name, epoch in missing[:10])
                + (", ..." if len(missing) > 10 else "")
            )
        requested = set(wanted)
        is_epoch = np.logical_and(
            np.array([key in requested for key in keys], dtype=bool), is_epoch
        )

    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    if args.epoch is not None:
        is_epoch = np.logical_and(
            epoch_info.index.get_level_values("interval_list_name") == args.epoch,
            is_epoch,
        )

    epoch_info = epoch_info.loc[is_epoch, :]

    if isinstance(epoch_info, pd.Series):
        epoch_info = epoch_info.to_frame().T

    return epoch_info


NO_SORTED_UNITS_REASON = (
    "the session has no curated cortical spike sorting (CuratedSpikeSorting, "
    "mountainsort4), a PFC-level condition, not an epoch exclusion"
)
"""Why an analyzed epoch is outside the PFC cohort."""


def write_cohort_record(
    path: Path,
    *,
    n_analyzed: int,
    pfc_cohort: pd.DataFrame,
    ran: pd.DataFrame,
    left_out: pd.Index,
    git_hash: str,
) -> None:
    """Record the PFC cohort beside the bundles: what ran, and why the rest of
    the register's analyzed epochs did not.

    Every partition of a run computes the same cohort, so every field but
    ``ran`` is the same whichever partition writes it last. ``ran`` is per
    writer: each invocation overwrites the file when it starts, so a batch
    split across parallel partitions (``--sessions``) leaves the last
    partition's share only. The union is the bundles on disk.
    """
    record = {
        "produced_by": produced_by(__file__),
        "git_hash": git_hash,
        # The British key the records already on disk carry; reported/mpfc reads it.
        "n_analysed": n_analyzed,
        "n_pfc_cohort": len(pfc_cohort),
        "ran": [f"{name} {epoch}" for name, epoch in ran.index],
        "left_out": {
            f"{name} {epoch}": NO_SORTED_UNITS_REASON for name, epoch in left_out
        },
    }
    text = json.dumps(record, indent=2) + "\n"
    write_atomically(path, lambda temporary: temporary.write_text(text))


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for batch PFC GLM analysis."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = get_command_line_arguments(argv)
    cli = [sys.argv[0], *argv]
    # before any cluster starts: its worker processes inherit this limit
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)
    Path(args.log_directory).mkdir(exist_ok=True, parents=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Read when this script imported its code, before any worker starts.
    git_hash = loaded_commit()

    # The PFC cohort: the register's analyzed epochs, whose session has sorted
    # units. An analyzed epoch without them is left out here, with that reason,
    # which pfc_cohort_<area>.json records beside the bundles.
    analyzed = analyzed_epochs()
    pfc_cohort, unsorted = split_sorted_epochs(analyzed, pfc_sorted_nwb_file_names())
    if len(unsorted):
        print(
            f"{len(unsorted)} analyzed epoch(s) left out at the PFC level, "
            f"{NO_SORTED_UNITS_REASON}: "
            + ", ".join(f"{name} {epoch}" for name, epoch in unsorted)
        )
    epoch_info = _filter_epoch_info(analyzed, args)
    epoch_info = epoch_info[epoch_info.index.isin(pfc_cohort.index)]
    if args.max_sessions is not None:
        epoch_info = epoch_info.iloc[: args.max_sessions]
    write_cohort_record(
        args.output_dir / pfc_cohort_name(args.brain_area),
        n_analyzed=len(analyzed),
        pfc_cohort=pfc_cohort,
        ran=epoch_info,
        left_out=unsorted,
        git_hash=git_hash,
    )

    print(
        f"Processing {len(epoch_info)} sessions for {args.brain_area} "
        f"(stop cohort: {args.stop_cohort}, fitter: {args.fitter}, "
        f"penalty sigma: {args.penalty_sigma}, folds: {args.n_folds}, "
        f"control draws: {args.n_control_draws})..."
    )

    if args.n_workers <= 1 or len(epoch_info) <= 1:
        for epoch in epoch_info.itertuples():
            run_analysis_then_release(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                brain_area=args.brain_area,
                sampling_frequency=SAMPLING_FREQUENCY,
                log_directory=args.log_directory,
                overwrite=args.overwrite,
                git_hash=git_hash,
                stop_cohort=args.stop_cohort,
                output_dir=args.output_dir,
                fitter=args.fitter,
                fit_diagnostics=args.fit_diagnostics,
                gpu=args.gpu,
                allow_cpu=args.allow_cpu,
                penalty_sigma=args.penalty_sigma,
                n_folds=args.n_folds,
                n_control_draws=args.n_control_draws,
                fold_seed=args.fold_seed,
                control_draw_seed=args.control_draw_seed,
                event_set=args.event_set,
                null_classes=args.null_classes,
                cli=cli,
            )
    else:
        cluster = LocalCluster(
            local_directory=args.local_scratch_dir,
            n_workers=args.n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="400GB",
        )

        with Client(cluster):
            results = [
                dask.delayed(run_analysis_then_release)(
                    nwb_file_name=epoch.Index[0],
                    epoch_name=epoch.Index[1],
                    brain_area=args.brain_area,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    log_directory=args.log_directory,
                    overwrite=args.overwrite,
                    git_hash=git_hash,
                    stop_cohort=args.stop_cohort,
                    output_dir=args.output_dir,
                    fitter=args.fitter,
                    fit_diagnostics=args.fit_diagnostics,
                    gpu=args.gpu,
                    allow_cpu=args.allow_cpu,
                    penalty_sigma=args.penalty_sigma,
                    n_folds=args.n_folds,
                    n_control_draws=args.n_control_draws,
                    fold_seed=args.fold_seed,
                    control_draw_seed=args.control_draw_seed,
                    event_set=args.event_set,
                    null_classes=args.null_classes,
                    cli=cli,
                )
                for epoch in epoch_info.itertuples()
            ]

            dask.compute(*results)


if __name__ == "__main__":
    sys.exit(main())
