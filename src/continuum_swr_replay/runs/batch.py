"""Running a batch of epochs: the cluster, one task per epoch, logs and the run lock.

Two drivers share the pieces here:

- :func:`run_batch`, the decoders' main loop (one ``dask.delayed`` call per
  epoch, on a GPU cluster or, with ``--allow_cpu``, a CPU one);
- :func:`run_epochs`, the detection and QA runners' loop (one submitted task
  per epoch, each record yielded as it finishes, so the caller alone writes
  the run's manifest).

A task logs to its own file (:func:`setup_logger`, :func:`run_epoch_logged`)
and closes the NWB files Spyglass opened for it (:func:`release_nwb_files`).
:func:`run_lock` keeps two runs from writing one run folder.

dask is imported inside the functions that start a cluster, so a runner that
needs only the lock, the logger or the NWB release does not load it.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import logging.config
import os
import resource
import sys
import traceback
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from continuum_swr_replay.data_loaders.epoch_register import epochs_to_run
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.runs.cli import filter_epoch_info

__all__ = [
    "cuda_cluster_class",
    "lost_epoch_record",
    "raise_open_file_limit",
    "release_nwb_files",
    "run_batch",
    "run_epoch_logged",
    "run_epoch_task",
    "run_epochs",
    "run_lock",
    "setup_logger",
]

_logger = logging.getLogger(__name__)
_warned_cache_name_missing = False
"""Whether ``release_nwb_files`` has said Spyglass's cache name is gone."""


class _SkipChecksumFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Skipped checksum" not in record.getMessage()


def setup_logger(
    name_logfile: str,
    path_logfile: str | Path,
    *,
    level: int = logging.INFO,
    include_warnings: bool = True,
) -> logging.Logger:
    """Set up a logger that logs to console and a file."""
    log_path = Path(path_logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = logging.getLevelName(level)

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "skip_checksum": {
                "()": _SkipChecksumFilter,
            },
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
                "datefmt": "%d-%b-%y %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_path),
                "mode": "w",
                "encoding": "utf-8",
                "formatter": "standard",
                "filters": ["skip_checksum"],
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["skip_checksum"],
            },
        },
        "loggers": {
            name_logfile: {
                "handlers": ["file", "console"],
                "level": level_name,
                "propagate": False,
            },
        },
    }

    if include_warnings:
        logging.captureWarnings(True)
        config["loggers"]["py.warnings"] = {
            "handlers": ["console"],
            "level": level_name,
            "propagate": False,
        }

    logging.config.dictConfig(config)
    return logging.getLogger(name_logfile)


def release_nwb_files() -> int:
    """Close every NWB file Spyglass holds open in this process; return how many.

    Spyglass caches each NWB file it opens, in ``spyglass.utils.nwb_helper_fn``'s
    module-level ``__open_nwb_files``, and never closes one. A long-lived worker
    therefore keeps one file descriptor per distinct file it has read (28 after
    one epoch's full ``load_data``), until it runs out (``[Errno 24] Too many
    open files``; Spyglass issue #1442). This closes them all with Spyglass's
    ``close_nwb_files``.

    Call it only where nothing else in this process can still be reading one of
    those files: data read lazily from a closed file cannot be read. The batch
    runners call it when an epoch ends, on workers that run one epoch at a time.

    It is cleanup, called from a ``finally``, so it never raises: an exception
    there would replace the epoch's own result, and in ``run_batch`` abort the
    whole batch. A failure is logged as a WARNING instead.

    Returns
    -------
    int
        How many files the cache held; all are now closed. -1 when the release
        failed (WARNING logged with the error), or when the count could not be
        read because Spyglass no longer names its cache ``__open_nwb_files``
        (WARNING logged once per process; the files are still closed).
    """
    global _warned_cache_name_missing
    try:
        # Imported here so that importing this module does not need Spyglass.
        from spyglass.utils import nwb_helper_fn

        open_files = vars(nwb_helper_fn).get("__open_nwb_files")
        if open_files is None:
            n_open = -1
            if not _warned_cache_name_missing:
                _warned_cache_name_missing = True
                _logger.warning(
                    "spyglass.utils.nwb_helper_fn has no __open_nwb_files: cannot "
                    "count the NWB files it holds open; closing them anyway"
                )
        else:
            n_open = len(open_files)
        nwb_helper_fn.close_nwb_files()
    except Exception as error:  # noqa: BLE001 - cleanup must not replace the result
        _logger.warning(
            "could not close the NWB files Spyglass held open: %r", error, exc_info=True
        )
        return -1
    return n_open


def raise_open_file_limit(target: int = 65536) -> tuple[int, int]:
    """Raise this process's soft limit on open files to `target`, at most the hard limit.

    A tmux session hands its processes a soft limit of 1024 open files, which a
    batch's long-lived workers can exhaust. Processes started afterwards,
    including dask's worker processes, inherit the raised limit, so call this
    before any cluster starts. A soft limit already at or above the new value is
    left as it is: this never lowers it.

    Parameters
    ----------
    target : int, optional
        The soft limit wanted. It is capped at the hard limit, which an
        unprivileged process cannot raise, unless that is unlimited.

    Returns
    -------
    old_soft : int
    new_soft : int
        The soft limit before and after; equal when it was not raised.
        ``resource.RLIM_INFINITY`` (-1 on Linux) stands for unlimited.
    """
    old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # RLIM_INFINITY is -1 here, so it must not take part in a min or a comparison.
    new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if old_soft == resource.RLIM_INFINITY or old_soft >= new_soft:
        return old_soft, old_soft
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    return old_soft, new_soft


def cuda_cluster_class() -> type | None:
    """``dask_cuda.LocalCUDACluster``, or None when dask_cuda does not import.

    Read when a batch starts, not when this module is imported, so that
    importing it loads no dask.
    """
    try:
        from dask_cuda import LocalCUDACluster
    except ImportError:
        return None
    return LocalCUDACluster


def run_batch(
    run_decode_fn: Callable,
    args: Namespace,
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Shared main loop: set up cluster, discover sessions, run in parallel.

    Parameters
    ----------
    run_decode_fn : Callable
        The per-session decoding function (e.g., ``run_decode`` from
        the NonLocal or ContFrag script).
    args : Namespace
        Parsed CLI arguments (must have common args from
        ``runs.cli.add_common_args``). ``args.save_dir`` reaches
        ``run_decode_fn`` resolved against this process's working directory,
        since a dask worker need not share it.
    extra_kwargs : dict, optional
        Additional keyword arguments to pass to ``run_decode_fn`` beyond
        the common ones.

    Notes
    -----
    The epochs are the epoch register's analysed ones, or with ``--pending-only``
    its pending ones alone (``epochs_to_run``), narrowed by ``--animal``,
    ``--date`` and ``--epoch``. Pending epochs are decoded into a staging folder
    and promoted with the register edit that makes them analysed
    (``docs/epoch_register.md``), so ``--pending-only`` refuses the canonical
    ``PROCESSED_DATA_DIR``: a reader of that folder must never meet a pending
    epoch's decode.

    Before the cluster starts, the soft open-file limit is raised
    (``raise_open_file_limit``), so that the worker processes inherit it.
    ``run_decode_fn`` is also given ``release_nwb``: True on the GPU cluster,
    whose workers run one epoch at a time, so each epoch closes the NWB files
    Spyglass opened for it (``release_nwb_files``); False on the CPU cluster,
    whose workers run several epochs at once.

    Raises
    ------
    RuntimeError
        If ``dask_cuda`` is not importable and ``--allow_cpu`` was not given.
        Without this check the batch would silently decode on a CPU cluster.
    ValueError
        If ``--pending-only`` would decode into ``PROCESSED_DATA_DIR``.
    """
    local_cuda_cluster = cuda_cluster_class()
    if local_cuda_cluster is None and not args.allow_cpu:
        raise RuntimeError(
            "dask_cuda is not importable, so this batch would decode on a CPU "
            "cluster. Install dask-cuda by hand (see docs/README.md), or pass "
            "--allow_cpu to decode on CPU."
        )

    if extra_kwargs is None:
        extra_kwargs = {}

    # Resolved once, here, before any worker sees it: a relative --save_dir
    # would otherwise be read against each worker's working directory.
    save_dir = Path(args.save_dir).resolve()
    if args.pending_only and save_dir == Path(PROCESSED_DATA_DIR).resolve():
        raise ValueError(
            "--pending-only decodes the register's pending epochs, which must "
            "not reach the canonical folder before they are analysed: give a "
            f"staging --save_dir, not {save_dir}."
        )

    # Before any cluster starts: a register that does not match Spyglass stops
    # the batch here, with no GPU worker started.
    epoch_info = filter_epoch_info(epochs_to_run(args.pending_only), args)

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

    # Before the cluster starts: its worker processes inherit this limit.
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)

    import dask
    from dask.distributed import Client, LocalCluster

    if local_cuda_cluster is not None:
        # One worker per GPU, one thread each: a worker process runs one epoch
        # at a time, so an epoch can close the NWB files Spyglass opened for it
        # without closing a file another epoch is reading.
        cluster = local_cuda_cluster(
            CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
            local_directory=args.local_scratch_dir,
            threads_per_worker=1,
        )
        release_nwb = True
    else:
        # LocalCluster's default workers run several epochs at once in threads
        # (for example 12 workers x 8 threads on 96 cores), and those threads share
        # Spyglass's per-process cache of open NWB files. Closing it after one
        # epoch could close a file another is still reading, so these epochs
        # leave their files open and rely on the raised open-file limit.
        cluster = LocalCluster(local_directory=args.local_scratch_dir)
        release_nwb = False

    with Client(cluster):
        log_directory = Path.cwd() / "logs"
        log_directory.mkdir(exist_ok=True, parents=True)

        delayed = [
            dask.delayed(run_decode_fn)(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                log_directory=log_directory,
                overwrite=args.overwrite,
                training_type=args.training_type,
                brain_area=args.brain_area,
                non_stationary=args.non_stationary,
                ripple_detector_name=args.ripple_detector_name,
                save_dir=save_dir,
                release_nwb=release_nwb,
                **extra_kwargs,
            )
            for epoch in epoch_info.itertuples()
        ]

        # Per-session run_decode_fn returns the epoch_identifier on
        # failure (after logging the exception) and None on success or
        # graceful skip. dask.compute returns a tuple of these.
        outcomes = dask.compute(*delayed)
        failures = [r for r in outcomes if r is not None]
        n_total = len(outcomes)

        if failures:
            sys.stderr.write(f"\n{len(failures)}/{n_total} session(s) FAILED:\n")
            for ident in failures:
                sys.stderr.write(f"  - {ident}\n")
            sys.stderr.write("See logs/<epoch_identifier>.log for tracebacks.\n")
            sys.exit(1)

        print(f"\nAll {n_total} session(s) completed.")


def lost_epoch_record(error: Exception) -> dict:
    """The record of an epoch whose task raised instead of returning one.

    ``status`` ``"failed"``, the error's ``repr`` and its traceback.
    """
    return {
        "status": "failed",
        "error": repr(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def run_epochs(
    epochs: Mapping[str, tuple],
    task: Callable[..., dict],
    cluster_factory: Callable[[], Any],
    *,
    key_prefix: str | None = None,
    lost: Callable[[Exception], dict] = lost_epoch_record,
) -> Iterator[tuple[str, dict]]:
    """Run each epoch as its own dask task; yield each record as it finishes.

    Nothing here writes the manifest: the caller does, after each record, so
    it has a single writer.

    Parameters
    ----------
    epochs : Mapping[str, tuple]
        Each epoch's id and the positional arguments its task is called with,
        such as ``{epoch_id: (nwb_file_name, epoch_name, kwargs, log_dir)}``.
    task : Callable
        One epoch's dask task, called with an entry's arguments; it returns
        the epoch's record (a dict), a failure included.
    cluster_factory : Callable
        Makes the cluster, closed after the last record; none is made for no
        epochs. Its workers should run one task at a time: ``setup_logger``
        reconfigures logging for the whole worker process, which closes the
        log of any other task running in it.
    key_prefix : str, optional
        The dask key of an epoch's task is ``{key_prefix}-{epoch_id}``, or the
        epoch id itself when None.
    lost : Callable, optional
        The record of an epoch whose task raised instead of returning, as when
        its worker died, from the error: :func:`lost_epoch_record` by default.

    Yields
    ------
    epoch_id : str
    record : dict
        The task's record, in the order the epochs finish.
    """
    if not epochs:
        return
    from dask.distributed import Client, as_completed

    with cluster_factory() as cluster, Client(cluster) as client:
        futures = {
            client.submit(
                task,
                *arguments,
                key=epoch_id if key_prefix is None else f"{key_prefix}-{epoch_id}",
            ): epoch_id
            for epoch_id, arguments in epochs.items()
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - the task could not return
                record = lost(error)
            yield futures[future], record


def run_epoch_logged(
    run: Callable[[logging.Logger], dict],
    *,
    log_name: str,
    log_path: Path,
    start: str,
    describe: Callable[[dict], str],
    log_provenance: bool,
) -> dict:
    """One epoch's task body: `run`, logged to its own file.

    When the epoch ends, pass or fail, the NWB files Spyglass opened for it are
    closed (:func:`release_nwb_files`): the workers run one task at a time, so
    no other epoch in the process reads them.

    Parameters
    ----------
    run : Callable
        Runs the epoch, given its logger; returns its record, whose
        ``status`` is ``"failed"`` for a failure it caught. An error it
        raises propagates, after the NWB files are closed.
    log_name, log_path : str, Path
        The logger's name and its file (:func:`setup_logger`).
    start : str
        Logged before `run` starts.
    describe : Callable
        The record's one-line summary, logged when `run` returns: as an ERROR
        followed by the record's ``traceback`` when it failed.
    log_provenance : bool
        Also log ``PROVENANCE`` and the record's ``provenance`` as sorted
        JSON, when the record holds one.

    Returns
    -------
    dict
        `run`'s record.
    """
    log = setup_logger(name_logfile=log_name, path_logfile=log_path)
    log.info("%s", start)
    try:
        record = run(log)
    finally:
        log.info("closed %d NWB files Spyglass held open", release_nwb_files())
    if log_provenance and "provenance" in record:
        log.info("PROVENANCE %s", json.dumps(record["provenance"], sort_keys=True))
    if record["status"] == "failed":
        log.error("%s\n%s", describe(record), record.get("traceback", ""))
    else:
        log.info("%s", describe(record))
    return record


def run_epoch_task(
    nwb_file_name: str,
    epoch_name: str,
    run_epoch_kwargs: dict,
    log_directory: Path,
    *,
    tag: str,
    run_epoch: Callable[..., dict],
    describe: Callable[[dict], str],
    epoch_id: Callable[[str, str], str],
    log_provenance: bool,
) -> dict:
    """One dask task: ``run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs)``,
    logged to ``log_directory / "{epoch id}_{tag}.log"`` (:func:`run_epoch_logged`).

    A runner binds everything after ``*`` with ``functools.partial`` (picklable
    for dask) and hands the result to :func:`run_epochs`, whose entries are
    ``(nwb_file_name, epoch_name, run_epoch_kwargs, log_directory)``.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
    run_epoch_kwargs : dict
        Passed to `run_epoch`; logged in the start line.
    log_directory : Path
    tag : str
        The runner's log suffix, such as ``"swr"``.
    run_epoch : Callable
        The runner's epoch body; returns the epoch's record.
    describe : Callable
        The record's one-line summary (:func:`run_epoch_logged`).
    epoch_id : Callable
        The epoch's id from its NWB file and epoch names (the runner passes
        ``swr_source.swr_epoch_id``; this package imports no analysis code).
    log_provenance : bool
        As :func:`run_epoch_logged`.
    """
    identifier = epoch_id(nwb_file_name, epoch_name)
    log_name = f"{identifier}_{tag}"
    return run_epoch_logged(
        lambda log: run_epoch(nwb_file_name, epoch_name, **run_epoch_kwargs),
        log_name=log_name,
        log_path=log_directory / f"{log_name}.log",
        start=f"{identifier}: run {run_epoch_kwargs}",
        describe=describe,
        log_provenance=log_provenance,
    )


@contextmanager
def run_lock(output_dir: Path) -> Iterator[None]:
    """Hold the run folder's lock for the block, or exit if another run has it.

    A non-blocking exclusive ``fcntl.lockf`` on ``output_dir / ".lock"``. Two
    runs sharing a run id would each rewrite the manifest from their own
    records, and the last to write would drop the other's. Only a lock held
    elsewhere stops the run with that message; any other failure to lock, such
    as a network file system without a lock manager, raises as itself.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".lock"
    with path.open("a") as handle:
        try:
            fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            sys.exit(
                f"Another run holds {path}: run {output_dir.name} is being "
                "written. Wait for it to finish, or pick another --run-id."
            )
        yield  # the lock goes when the file closes
