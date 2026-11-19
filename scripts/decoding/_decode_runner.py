"""The flow both decode runners share, and the record of what differs.

Not a script. ``run_decoding_bandit.py`` (the NonLocal 4-state detector) and
``run_decoding_contfrag.py`` (the ContFrag 2-state classifier) import it as
``scripts.decoding._decode_runner``, after putting the repo root on
``sys.path``. Each runner holds a ``decoding.fit.DecodeRunner`` naming its
detector classes and arguments, its config, its aggregation and loader, its
file suffix and its log lines, and hands it to :func:`run_decode`. The fit
itself is the package's (``decoding.fit.perform_decoding``); what is here is
an epoch's decode around it.

An HPC decode, in either runner, reads the SWR source (``swr_source``): it
refuses to start unless the source names both its Kay inventory and its
artifact run (``hpc_artifact_intervals``, and ``parse_decode_args`` before a
batch), masks MUA detection and ``is_missing`` with the artifact run, leaves
the mask and every inventory event out of encoding (``hpc_decode_masks``),
fits on spike waveform features, refuses a resume decoded under other masks
(``require_saved_decode_masks``) and writes the source's ``SWR_event`` into
``data_over_time`` (``with_swr_source``).
"""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.event_definition.analyzed_time import with_swr_source
from continuum_swr_replay.analysis.event_definition.decode_masks import (
    hpc_artifact_intervals,
    hpc_decode_masks,
    hpc_ripple_power_inputs,
    require_hpc_decode_sources,
    require_saved_decode_masks,
)
from continuum_swr_replay.data_loaders import (
    get_training_timepoints,
    load_data,
)
from continuum_swr_replay.data_types import SessionData
from continuum_swr_replay.decoding.files import (
    decode_paths,
    non_local_threshold_percent,
    results_exist,
)
from continuum_swr_replay.decoding.fit import (
    DecodeRunner,
    perform_decoding,
)
from continuum_swr_replay.decoding.identifiers import (
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
    decode_identifier,
    format_epoch_identifier,
)
from continuum_swr_replay.decoding.transitions import NON_STATIONARY_REFUSAL
from continuum_swr_replay.parameters import (
    ROOT_DIR,
)
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.runs.batch import release_nwb_files, setup_logger
from continuum_swr_replay.runs.provenance import provenance_record
from continuum_swr_replay.tables.io import write_atomically


def save_results_and_model(
    logger: logging.Logger,
    results: xr.Dataset,
    detector: Any,
    env: Environment,
    data_over_time: pd.DataFrame,
    save_folder: Path,
    epoch_identifier: str,
    non_local_threshold: float | None = None,
    *,
    resumed: bool,
    provenance: str,
    decode_masks: str | None = None,
) -> None:
    """Save decoding results, detector model, environment, and derived data.

    Every file goes through ``write_atomically``, so a crash never leaves a
    truncated file that ``results_exist`` would trust on resume.

    A fresh save stamps ``provenance`` (the JSON of the epoch's PROVENANCE log
    line) into ``results.nc`` as ``attrs["provenance"]``, so the record outlives
    the log, which a resume rewrites, and an HPC decode's ``decode_masks``
    (the JSON of its ``DECODE MASKS`` log line) as ``attrs["decode_masks"]``.
    When ``resumed`` is True the results,
    detector and environment were read back from ``save_folder``, and
    ``results`` is still open lazily on its file there, so only
    ``data_over_time`` is written.

    ``data_over_time`` is ``{epoch_identifier}_{threshold}_data_over_time.pkl``,
    the threshold in percent, or ``{epoch_identifier}_data_over_time.pkl`` when
    ``non_local_threshold`` is None; every name is ``decode_paths``'.
    """
    paths = decode_paths(
        epoch_identifier,
        save_folder,
        threshold_percent=None
        if non_local_threshold is None
        else non_local_threshold_percent(non_local_threshold),
    )
    if resumed:
        logger.info("Resumed: saving data_over_time only...")
    else:
        logger.info("Saving results...")
        stamped = results.reset_index("state_bins")
        stamped.attrs["provenance"] = provenance
        if decode_masks is not None:
            stamped.attrs["decode_masks"] = decode_masks
        write_atomically(
            paths.results, lambda path: stamped.to_netcdf(path, engine="h5netcdf")
        )
        write_atomically(paths.detector, detector.save_model)
        write_atomically(paths.environment, env.save_environment)

    write_atomically(paths.data_over_time, data_over_time.to_pickle)


def organize_results(
    runner: DecodeRunner,
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str,
    non_local_threshold: float | None = None,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Organize decoding results and compute derived measures.

    ``runner.aggregate`` gets ``non_local_threshold`` unless it is None.

    For HPC, ``data_over_time`` is then read through the SWR source
    (``analyzed_time.with_swr_source``): ``SWR_event`` holds the configured Kay
    inventory's labels and ``swr_observed`` its coverage, and the stored
    ripples' labels and counts are gone.
    """
    time_diffs = np.diff(results.time.values)
    if len(time_diffs) == 0:
        raise ValueError(
            "Time array must have at least 2 elements to compute sampling frequency"
        )
    if np.any(time_diffs <= 0):
        n_bad = np.sum(time_diffs <= 0)
        bad_indices = np.where(time_diffs <= 0)[0][:5]
        bad_values = time_diffs[bad_indices]
        raise ValueError(
            f"Time array must be strictly increasing with positive differences. "
            f"Found {n_bad}/{len(time_diffs)} non-positive differences. "
            f"First problematic indices: {bad_indices.tolist()}, values: {bad_values.tolist()}"
        )

    median_dt = np.median(time_diffs)
    if median_dt <= 1e-10:
        raise ValueError(
            f"Median time difference too small ({median_dt}), cannot compute sampling frequency"
        )

    sampling_frequency = 1 / median_dt
    threshold = (
        {}
        if non_local_threshold is None
        else {"non_local_threshold": non_local_threshold}
    )
    data_over_time, posterior = runner.aggregate(
        data,
        results,
        env,
        nwb_file_name,
        epoch_name,
        brain_area,
        sampling_frequency=sampling_frequency,
        **threshold,
    )
    if brain_area == "HPC":
        # SWR_event from the SWR source, so the decode's provenance is the
        # analyses' source: the stored ripples' labels and counts are replaced.
        # The decode keeps every event it decoded; the analyses' read drops
        # the ones the artifact mask reaches, each with its reason. Its labels
        # are checked against the threshold they were made at.
        data_over_time = with_swr_source(
            data_over_time, drop_masked_events=False, **threshold
        )

    return data_over_time, posterior


def run_decode(
    runner: DecodeRunner,
    nwb_file_name: str,
    epoch_name: str,
    log_directory: str = "",
    overwrite: bool = False,
    training_type: str = DEFAULT_TRAINING_TYPE,
    non_stationary: bool = False,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    brain_area: str = "HPC",
    save_dir: Path | None = None,
    release_nwb: bool = False,
    non_local_threshold: float | None = None,
    create_figurl: bool = False,
) -> str | None:
    """Run `runner`'s decode for a single session.

    Results are written to ``save_dir`` (``paths.processed_data_dir()`` when
    None). If they are already there (and
    ``overwrite`` is False), the decode resumes: it logs ``RESUMED from existing
    results in <save_dir>``, reads them back from ``save_dir`` and rewrites only
    ``data_over_time``.

    An HPC decode reads the SWR source (``swr_source``) and refuses to start
    unless it names both its Kay inventory and its artifact run
    (``hpc_artifact_intervals``). Its artifact mask, the one every stage uses,
    masks MUA detection (``load_data``) and is its ``is_missing``; its encoding
    leaves the mask and every inventory event out (``hpc_decode_masks``, logged as
    ``DECODE MASKS`` and stamped into ``results.nc``); and its
    ``data_over_time`` carries the source's ``SWR_event``. An HPC resume
    refuses results whose stamped masks name other runs, or none
    (``require_saved_decode_masks``).

    With ``release_nwb``, the epoch ends, pass or fail, by closing the NWB files
    Spyglass holds open in this process (``release_nwb_files``) and logging
    whether that worked. Only for a process that runs one epoch at a time, as
    ``run_batch``'s GPU workers do: another task in the process could still be
    reading them.

    ``non_local_threshold`` and ``create_figurl`` are the NonLocal runner's
    alone (see ``decoding.fit.DecodeRunner``).

    Returns
    -------
    str | None
        ``None`` on success or graceful skip; the ``epoch_identifier``
        on caught exception so ``run_batch`` can aggregate failures.
    """
    if save_dir is None:
        save_dir = processed_data_dir()
    epoch_identifier = decode_identifier(
        format_epoch_identifier(
            nwb_file_name, epoch_name, brain_area, runner.file_suffix
        ),
        ripple_detector_name=ripple_detector_name,
        non_stationary=non_stationary,
        training_type=training_type,
    )

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    try:
        logger.info(runner.start_banner.center(50, "#"))
        provenance = provenance_record(ROOT_DIR)
        provenance_json = json.dumps(provenance, sort_keys=True)
        logger.info("PROVENANCE " + provenance_json)
        logger.info(f"Git Hash: {provenance['commit']}")
        if runner.model_line is not None:
            logger.info(runner.model_line)
        logger.info(f"training type: {training_type}")
        logger.info(f"non_stationary: {non_stationary}")
        if non_local_threshold is not None:
            logger.info(f"non_local_threshold: {non_local_threshold}")
        logger.info(f"ripple_detector_name: {ripple_detector_name}")
        logger.info(f"brain_area: {brain_area}")
        logger.info(f"save_dir: {save_dir.resolve()}")
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact_mask = None
        if brain_area == "HPC":
            # fail closed, before anything is loaded: an HPC decode's masks and
            # SWR labels need the SWR source's inventory and artifact runs
            artifact_mask = hpc_artifact_intervals(nwb_file_name, epoch_name)
        logger.info("Loading data...")
        data = load_data(
            nwb_file_name,
            epoch_name,
            ripple_detector_name=ripple_detector_name,
            mua_artifact_intervals=(
                None if artifact_mask is None else {"HPC": artifact_mask}
            ),
            # SWR_zscore: the inventory's Kay consensus, R-T's tetrodes, the
            # mask out of its normalization and NaN inside it
            ripple_power=(
                None
                if artifact_mask is None
                else hpc_ripple_power_inputs(nwb_file_name, epoch_name)
            ),
        )

        if not data["spike_times"]:
            logger.info("No spikes. Skipping...")
            logger.info("Done!")
            return

        if brain_area not in data["spike_times"]:
            logger.warning(f"No spike data available for {brain_area}. Skipping...")
            return

        decode_masks = None
        resumed = results_exist(save_dir, epoch_identifier, overwrite)
        if resumed:
            logger.info(f"RESUMED from existing results in {save_dir.resolve()}")
            results, detector, env = runner.load_saved(
                nwb_file_name,
                epoch_name,
                brain_area,
                non_stationary=non_stationary,
                ripple_detector_name=ripple_detector_name,
                training_type=training_type,
                root=save_dir,
            )
            if artifact_mask is not None:
                # the reused results must come from the configured masks
                require_saved_decode_masks(
                    results.attrs, str(decode_paths(epoch_identifier, save_dir).results)
                )
        else:
            is_missing = None
            if artifact_mask is None:
                is_training = get_training_timepoints(data, training_type=training_type)
            else:
                masks = hpc_decode_masks(
                    nwb_file_name,
                    epoch_name,
                    data["position_info"],
                    artifact_mask,
                    training_type,
                )
                decode_masks = json.dumps(masks.record, sort_keys=True)
                logger.info("DECODE MASKS " + decode_masks)
                is_training, is_missing = masks.is_training, masks.is_missing

            spike_waveform_features = data["spike_waveform_features"].get(
                brain_area, None
            )

            results, detector, env = perform_decoding(
                runner,
                logger,
                data["spike_times"][brain_area],
                spike_waveform_features,
                data["position_info"],
                data["linear_edge_order"],
                data["linear_edge_spacing"],
                data["track_graph"],
                is_training,
                brain_area=brain_area,
                non_stationary=non_stationary,
                is_missing=is_missing,
            )

        logger.info("Organizing results...")
        data_over_time, posterior = organize_results(
            runner,
            data,
            results,
            env,
            nwb_file_name,
            epoch_name,
            brain_area,
            non_local_threshold=non_local_threshold,
        )

        if create_figurl:
            runner.figurl(
                logger,
                data,
                results,
                data_over_time,
                posterior,
                epoch_identifier,
                brain_area=brain_area,
            )

        save_results_and_model(
            logger,
            results,
            detector,
            env,
            data_over_time,
            save_dir,
            epoch_identifier,
            non_local_threshold,
            resumed=resumed,
            provenance=provenance_json,
            decode_masks=decode_masks,
        )

        logger.info("Done!")
    except Exception as e:
        logger.exception(e)
        return epoch_identifier  # Sentinel: run_batch treats non-None as failure.
    finally:
        if release_nwb:
            logger.info(f"NWB files Spyglass held open closed: {release_nwb_files()}")


def parse_decode_args(
    parser: ArgumentParser, argv: Sequence[str] | None = None
) -> Namespace:
    """Parse a runner's command line, refusing what no decode may run.

    ``--non_stationary`` is refused (``NON_STATIONARY_REFUSAL``), and an HPC
    batch is refused before any cluster starts unless the SWR source names
    both of its runs (``require_hpc_decode_sources``).
    """
    args = parser.parse_args(argv)
    if args.non_stationary:
        parser.error(NON_STATIONARY_REFUSAL)
    if args.brain_area == "HPC":
        try:
            require_hpc_decode_sources()
        except ValueError as error:
            parser.error(str(error))
    return args
