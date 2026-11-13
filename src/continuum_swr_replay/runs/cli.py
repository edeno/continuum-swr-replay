"""The runners' shared command line: run ids, the decoders' options, epoch filters
(:func:`add_epoch_filters`, read by :func:`filter_epoch_info`,
:func:`select_analysed_epochs` and :func:`epoch_ids_matching`), and the fit
figures' ``--allow-stale``."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Iterable
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
    "add_epoch_filters",
    "epoch_ids_matching",
    "filter_epoch_info",
    "fit_figure_parser",
    "parse_run_id",
    "read_session_list",
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


def read_session_list(path: Path) -> list[tuple[str, str]]:
    """Read unique ``nwb_file_name epoch_name`` pairs in file order.

    Parameters
    ----------
    path : Path
        Text file with one whitespace- or comma-separated session per line;
        ``#`` starts a comment.

    Returns
    -------
    list of tuple of str
        ``(nwb_file_name, epoch_name)`` pairs, de-duplicated, in file order.

    Raises
    ------
    ValueError
        If a line does not hold exactly two fields, or the file names no
        sessions.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{number}: expected '<nwb_file_name> <epoch_name>', "
                f"got {raw.strip()!r}"
            )
        key = (fields[0], fields[1])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    if not pairs:
        raise ValueError(f"{path} names no sessions")
    return pairs


def filter_epoch_info(
    epoch_info: pd.DataFrame,
    *,
    sessions: Path | None = None,
    animal: str | None = None,
    date: int | None = None,
    epoch: str | None = None,
) -> pd.DataFrame:
    """Keep the epochs that ``--sessions``, ``--animal``, ``--date`` and
    ``--epoch`` name (:func:`add_epoch_filters`); None leaves a filter off.

    The filters are taken as explicit fields rather than as a parsed
    ``Namespace`` so that runners with different remaining command lines
    still select epochs identically.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        Session table indexed by ``(nwb_file_name, interval_list_name)``.
    sessions : Path, optional
        ``--sessions`` file listing the sessions to keep, in the order they
        should run.
    animal : str, optional
        Keep only this animal.
    date : int, optional
        Keep only this session date.
    epoch : str, optional
        Keep only this interval list name: the full epoch name, such as
        ``"02_r1"``, matched exactly.

    Returns
    -------
    pd.DataFrame
        The selected rows, in `sessions` order when that file is given and in
        source order otherwise.

    Raises
    ------
    ValueError
        If `sessions` names a session that is not available.
    """
    keep = np.ones(len(epoch_info), dtype=bool)
    keys = list(
        zip(
            epoch_info.index.get_level_values("nwb_file_name"),
            epoch_info.index.get_level_values("interval_list_name"),
            strict=True,
        )
    )
    requested_order = read_session_list(sessions) if sessions is not None else None
    if requested_order is not None:
        available = set(keys)
        missing = [key for key in requested_order if key not in available]
        if missing:
            raise ValueError(f"{sessions} names unavailable session(s): {missing[:10]}")
        requested = set(requested_order)
        keep &= np.array([key in requested for key in keys], dtype=bool)
    if animal is not None:
        keep &= epoch_info.animal.to_numpy() == animal
    if date is not None:
        keep &= epoch_info.date.to_numpy() == date
    if epoch is not None:
        keep &= (
            epoch_info.index.get_level_values("interval_list_name").to_numpy() == epoch
        )
    filtered = epoch_info.loc[keep]
    if requested_order is not None:
        order = {key: index for index, key in enumerate(requested_order)}
        filtered_keys = list(
            zip(
                filtered.index.get_level_values("nwb_file_name"),
                filtered.index.get_level_values("interval_list_name"),
                strict=True,
            )
        )
        filtered = filtered.iloc[np.argsort([order[key] for key in filtered_keys])]
    return filtered


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
    return epoch_ids_matching(
        analysed, animal=args.animal, date=args.date, epoch=args.epoch
    )


def epoch_ids_matching(
    identifiers: Iterable[str],
    *,
    animal: str | None = None,
    date: int | None = None,
    epoch: str | None = None,
) -> list[str]:
    """The epoch ids of `animal`, recorded on `date` and named `epoch`, in their
    given order; None leaves a filter off.

    Parameters
    ----------
    identifiers : Iterable of str
        Epoch ids such as ``"chimi20200225_02_r1_HPC"``.
    animal : str, optional
    date : int, optional
        Such as ``20200225``: the last 8 digits of the NWB file's stem.
    epoch : str, optional
        The full epoch name, such as ``"02_r1"``.

    Returns
    -------
    list of str
        With no filter, `identifiers` as given, none parsed (so a name no
        decode writes still reaches a runner, which records it as failed).

    Raises
    ------
    ValueError
        With a filter, if an identifier cannot be parsed.
    """
    if animal is None and date is None and epoch is None:
        return list(identifiers)
    selected = []
    for identifier in identifiers:
        key = parse_epoch_identifier(identifier)
        if (
            (animal is None or key.animal == animal)
            and (date is None or key.date == date)
            and (epoch is None or key.epoch_name == epoch)
        ):
            selected.append(identifier)
    return selected


def add_epoch_filters(parser: ArgumentParser, *, epoch_id: bool = False) -> None:
    """Add a runner's epoch filters: ``--animal``, ``--date`` and ``--epoch``.

    They are what :func:`filter_epoch_info` and :func:`select_analysed_epochs`
    read; each left out keeps every epoch.

    Parameters
    ----------
    parser : ArgumentParser
    epoch_id : bool, optional
        Also add ``--epoch-id``, one epoch by its id, for the runners that select
        through :func:`select_analysed_epochs`.
    """
    parser.add_argument("--animal", help="Only this animal's epochs, such as chimi.")
    parser.add_argument(
        "--date", type=int, help="Only epochs recorded on this date, such as 20200225."
    )
    parser.add_argument("--epoch", help="Only this epoch, such as 02_r1.")
    if epoch_id:
        parser.add_argument(
            "--epoch-id",
            help="Only this epoch, such as chimi20200225_02_r1_HPC; not with "
            "--animal, --date or --epoch.",
        )


def fit_figure_parser(description: str | None) -> ArgumentParser:
    """The command line of a script that plots stored fits: ``--allow-stale`` only.

    ``--allow-stale`` is the opt-out of the provenance gate
    (``runs.fit_provenance.require_fresh_inputs``), which refuses a
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
    add_epoch_filters(parser)
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
