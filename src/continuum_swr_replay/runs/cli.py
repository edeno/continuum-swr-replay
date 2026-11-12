"""The runners' shared command line: run ids, the decoders' options, epoch filters,
and the fit figures' ``--allow-stale``."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.decoding.identifiers import (
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
    parse_epoch_identifier,
)
from continuum_swr_replay.paths import processed_data_dir

__all__ = [
    "RUN_ID_PATTERN",
    "add_common_args",
    "filter_epoch_info",
    "fit_figure_parser",
    "parse_run_id",
    "select_analysed_epochs",
]

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
"""A run id names one folder under its output root: no path separator."""


def parse_run_id(value: str) -> str:
    """Return `value` if it names one run folder: an argparse ``type``.

    Parameters
    ----------
    value : str
        The option's value, such as ``"a1b2c3d4e5f6"``.

    Returns
    -------
    str
        `value`, unchanged.

    Raises
    ------
    ArgumentTypeError
        Unless `value` is letters, digits, ``.``, ``_`` and ``-`` only, and
        not ``.`` or ``..``; argparse reports it as a usage error (exit 2).
    """
    if not RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ArgumentTypeError(
            f"{value!r} must name one folder: letters, digits, '.', '_' and '-', "
            "and not '.' or '..'."
        )
    return value


def filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Keep the epochs that ``--animal``, ``--date`` and ``--epoch`` name.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        ``get_epoch_info``'s frame (or a subset of its rows), indexed by
        ``(nwb_file_name, interval_list_name)``.
    args : Namespace
        Parsed arguments with ``animal``, ``date`` and ``epoch``; ``None``
        leaves that filter off. ``epoch`` is the full epoch name, such as
        ``"02_r1"``, matched exactly against ``interval_list_name``.

    Returns
    -------
    pd.DataFrame
        The matching rows, in their order.
    """
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    if args.epoch is not None:
        epoch_names = epoch_info.index.get_level_values("interval_list_name")
        is_epoch = np.logical_and(epoch_names == args.epoch, is_epoch)

    return epoch_info.loc[is_epoch, :]


def select_analysed_epochs(args: Namespace) -> list[str]:
    """The analysed epochs the arguments name, as epoch ids.

    ``args.epoch_id`` names one epoch of the epoch register's analysed ones (any
    other exits with a message); otherwise ``args.animal``, ``args.date`` and
    ``args.epoch`` narrow them, ``None`` leaving a filter off. The detection
    comparison's runners select their epochs by it.

    Parameters
    ----------
    args : Namespace
        Parsed arguments with ``epoch_id``, ``animal``, ``date`` and ``epoch``.

    Returns
    -------
    list of str
        Epoch ids such as ``"chimi20200225_02_r1_HPC"``, sorted.
    """
    analysed = analysed_epoch_ids()
    if args.epoch_id is not None:
        if args.epoch_id not in analysed:
            sys.exit(f"{args.epoch_id} is not an analysed epoch of the epoch register.")
        return [args.epoch_id]
    selected = []
    for identifier in analysed:
        key = parse_epoch_identifier(identifier)
        if (
            (args.animal is None or key.animal == args.animal)
            and (args.date is None or key.date == args.date)
            and (args.epoch is None or key.epoch_name == args.epoch)
        ):
            selected.append(identifier)
    return selected


def fit_figure_parser(description: str | None) -> ArgumentParser:
    """The command line of a script that plots stored fits: ``--allow-stale`` only.

    ``--allow-stale`` is the opt-out of the provenance gate
    (``analysis.replay.fit_provenance.require_fresh_inputs``), which refuses a
    fit whose input tables changed after it was written.

    Parameters
    ----------
    description : str or None
        The parser's description; a script passes its ``__doc__``.

    Returns
    -------
    ArgumentParser
        A parser whose namespace has ``allow_stale`` (False unless given).
    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Plot the fit even if its input tables have changed since it was "
            "fit (or carry no provenance sidecar at all)."
        ),
    )
    return parser


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    """Add CLI arguments shared by all decoding scripts."""
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7,8,9"
    )
    parser.add_argument("--local_scratch_dir", type=str, default=".local_scratch")
    parser.add_argument("--brain_area", type=str, default="HPC")
    parser.add_argument("--training_type", type=str, default=DEFAULT_TRAINING_TYPE)
    parser.add_argument(
        "--ripple_detector_name", type=str, default=DEFAULT_SWR_DEFINITION
    )
    parser.add_argument("--non_stationary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--animal", type=str)
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch", type=str)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=processed_data_dir(),
        help="Folder the decodes are written to, and resumed from "
        "(default: Processed-Data). Created if missing.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Decode on a CPU cluster when dask_cuda is not importable, "
        "instead of refusing to start.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Decode only the epochs the epoch register holds pending a Spyglass "
        "populate (the incremental update, run after the populates), into a "
        "staging --save_dir; the canonical folder is refused. By default the "
        "register's analysed epochs are decoded.",
    )
    return parser
