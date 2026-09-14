"""Batch PFC GLM analysis of neuron modulation during non-local hippocampal events.

This script runs GLM-based analysis of prefrontal cortex (mPFC/OFC) neuron firing
rate modulation around sharp-wave ripples (SWR) and non-local hippocampal events.
For each session, it fits Poisson GLM models with convolutional spline basis functions
to quantify how each neuron's firing rate changes relative to event onsets.

The analysis pipeline per session:
1. Load the session's non-local events from the canonical event table
2. Split them on that table's ``is_swr`` column (strict positive-duration
   overlap with the speed-gated ``SWR_event`` labels), so Figure 4 describes
   the same events, with the same classification, as the content and rate
   families
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
"""

import json
import logging
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

# Pin BLAS/OpenMP threads to 1 per process to prevent thread oversubscription
# when running multiple Dask workers. Each worker runs one session serially,
# and statsmodels GLM (IRLS) is single-threaded anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from continuum_swr_replay.analysis.pfc import run_session
from continuum_swr_replay.analysis.pfc.canonical_events import (
    IS_SWR_DEFINITION,
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (
    DEFAULT_STOP_TABLE_PATH,
    load_session_stop_cohort,
)
from continuum_swr_replay.analysis.replay.cohort_selection import stop_cohort_suffix
from continuum_swr_replay.analysis.replay.fit_provenance import (
    fit_provenance_path,
    write_fit_provenance,
)
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.decoding_common import setup_logger
from continuum_swr_replay.parameters import (
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)

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


def modulation_path(save_folder: Path, epoch_identifier: str, stop_cohort: str) -> Path:
    """Artifact path for one session, named for the population it was fit on.

    The canonical at-well bundle keeps the unsuffixed name every reader already
    hard-codes, so a sensitivity bundle can never be mistaken for it.
    """
    suffix = stop_cohort_suffix(stop_cohort)
    return save_folder / f"{epoch_identifier}_modulation{suffix}.pkl"


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
    stop_cohort: str = "at_well",
    stop_table_sha256: str | None = None,
    event_table_sha256: str | None = None,
    data_over_time_sha256: str | None = None,
) -> bool:
    """Check if this session's bundle already exists *for this population*.

    Distinct from ``_decoding_common.results_exist`` because the PFC
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
    """
    if overwrite:
        return False
    artifact = modulation_path(save_folder, epoch_identifier, stop_cohort)
    save_base = artifact.with_suffix("")
    required_files = [
        save_base.with_suffix(".pkl"),
        save_base.with_suffix(".SWR_modulation.npy"),
        save_base.with_suffix(".no_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_no_SWR_modulation.npy"),
        save_base.with_suffix(".matched_control_modulation.npy"),
    ]
    if not all(f.exists() for f in required_files):
        return False
    sidecar = fit_provenance_path(artifact)
    if not sidecar.exists():
        return False
    record = json.loads(sidecar.read_text())
    return (
        record.get("stop_cohort") == stop_cohort
        and record.get("stop_table_sha256") == stop_table_sha256
        and record.get("is_swr_definition") == IS_SWR_DEFINITION
        and record.get("canonical_event_table_sha256") == event_table_sha256
        and record.get("data_over_time_sha256") == data_over_time_sha256
    )


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    immobility_threshold: float = MAX_SPEED_THRESHOLD_CM_PER_SEC,
    log_directory: str | Path = "logs",
    overwrite: bool = False,
    git_hash: str = "",
    stop_cohort: str = "at_well",
    output_dir: Path = PROCESSED_DATA_DIR,
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
    immobility_threshold : float
        Speed threshold for immobility classification (cm/s)
    log_directory : str | Path
        Directory for log files
    overwrite : bool
        Whether to overwrite existing results
    git_hash : str
        Pre-computed git hash for logging
    stop_cohort : {"at_well", "all_stops"}
        Population to fit: the well-anchored stops the hippocampal ladders are
        read on, or no stop restriction at all.
    output_dir : Path
        Directory the artifacts and their provenance sidecar are written to.
    """
    epoch_identifier = (
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    )
    save_file = modulation_path(output_dir, epoch_identifier, stop_cohort)

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
    ):
        logger.info(f"Results already exist for {epoch_identifier}. Skipping...")
        return

    logger.info(" START ".center(50, "#"))
    logger.info(f"Git Hash: {git_hash}")
    logger.info(f"brain_area: {brain_area}")
    logger.info(f"sampling_frequency: {sampling_frequency}")
    logger.info(f"immobility_threshold: {immobility_threshold}")
    logger.info(f"stop_cohort: {stop_cohort}")

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
            immobility_threshold=immobility_threshold,
            run_parallel=False,
            stop_cohort=stop_cohort,
            cohort_stops=cohort_stops,
        )

        logger.info(f"Analyzed {len(neuron_stats)} neurons")
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

        # Every artifact states the population it was fit on: which cohort,
        # which stop table, and how many bins and events survived it.
        sidecar = write_fit_provenance(
            save_file,
            {"stop_table": DEFAULT_STOP_TABLE_PATH} if cohort_stops is not None else {},
            estimator="poisson_glm",
            cli=sys.argv,
            extra={
                "stop_cohort": stop_cohort,
                "stop_table_sha256": cohort_report.get("stop_table_sha256"),
                "stop_cohort_report": cohort_report,
                "cohort_counts": cohort_info,
                "brain_area": brain_area,
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch_name,
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
            },
        )

        logger.info(f"Saved results to {save_file}")
        logger.info(f"Wrote provenance to {sidecar}")
        logger.info("Done!")
    except Exception as e:
        logger.exception(f"Error processing {epoch_identifier}: {e}")


def get_command_line_arguments() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local_scratch_dir", type=str, default="/tmp/dask_scratch",
        help="Local scratch directory for Dask workers (default: /tmp/dask_scratch)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Number of Dask workers (default: 4)",
    )
    parser.add_argument(
        "--animal", type=str, default=None,
        help="Filter to specific animal (e.g., wilbur)",
    )
    parser.add_argument(
        "--date", type=int, default=None,
        help="Filter to specific date (e.g., 20210329)",
    )
    parser.add_argument(
        "--epoch", type=str, default=None,
        help="Filter to specific epoch (e.g., 02_r1)",
    )
    parser.add_argument(
        "--brain_area", type=str, default="mPFC",
        choices=["mPFC", "OFC"],
        help="Brain area to analyze: mPFC or OFC (default: mPFC)",
    )
    parser.add_argument(
        "--log_directory", type=str, default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
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
        default=PROCESSED_DATA_DIR,
        help=(
            "Directory for the modulation artifacts and their provenance "
            f"sidecars (default: {PROCESSED_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Process at most this many sessions (smoke runs)",
    )

    return parser.parse_args()


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


def _filter_to_sorted_sessions(epoch_info: pd.DataFrame) -> pd.DataFrame:
    """Filter to sessions that have curated spike sorting results.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        Available epochs

    Returns
    -------
    pd.DataFrame
        Epochs with curated spike sorting
    """
    try:
        from spyglass.spikesorting.v0 import CuratedSpikeSorting

        restriction = {
            "preproc_params_name": "default",
            "sort_interval_name": (
                "sleeps_runs_noPrePostTrialTimes raw data valid times"
            ),
            "sorter_params_name": (
                "franklab_probe_ctx_30KHz_115rad_new_mountainsort2"
            ),
            "team_name": "ac_em_xs",
            "sorter": "mountainsort4",
        }

        sorted_nwb_file_names = np.unique(
            (CuratedSpikeSorting & restriction).fetch("nwb_file_name")
        )
        epoch_info = epoch_info[
            epoch_info.index.get_level_values("nwb_file_name").isin(
                sorted_nwb_file_names
            )
        ]
    except ImportError:
        logging.getLogger(__name__).warning(
            "Spyglass not available. Running on all sessions without filtering "
            "by curated spike sorting."
        )

    return epoch_info


def main() -> None:
    """Main entry point for batch PFC GLM analysis."""
    args = get_command_line_arguments()

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)
    Path(args.log_directory).mkdir(exist_ok=True, parents=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.rstrip()

    epoch_info = get_epoch_info()
    epoch_info = _filter_epoch_info(epoch_info, args)
    epoch_info = _filter_to_sorted_sessions(epoch_info)
    if args.max_sessions is not None:
        epoch_info = epoch_info.iloc[: args.max_sessions]

    print(
        f"Processing {len(epoch_info)} sessions for {args.brain_area} "
        f"(stop cohort: {args.stop_cohort})..."
    )

    if args.n_workers <= 1 or len(epoch_info) <= 1:
        for epoch in epoch_info.itertuples():
            run_analysis(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                brain_area=args.brain_area,
                sampling_frequency=SAMPLING_FREQUENCY,
                immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                log_directory=args.log_directory,
                overwrite=args.overwrite,
                git_hash=git_hash,
                stop_cohort=args.stop_cohort,
                output_dir=args.output_dir,
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
                dask.delayed(run_analysis)(
                    nwb_file_name=epoch.Index[0],
                    epoch_name=epoch.Index[1],
                    brain_area=args.brain_area,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                    log_directory=args.log_directory,
                    overwrite=args.overwrite,
                    git_hash=git_hash,
                    stop_cohort=args.stop_cohort,
                    output_dir=args.output_dir,
                )
                for epoch in epoch_info.itertuples()
            ]

            dask.compute(*results)


if __name__ == "__main__":
    sys.exit(main())
