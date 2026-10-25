"""The epoch register: which run epochs are analysed, and why every other one is not.

The rule:

    An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) of
    a regular spatial-bandit session in Spyglass (or of a listed first-exposure
    session), with parsed trials (`noPrePostTrialTimes`), and every decode input
    resolves to exactly one row: position map, position info and linearization,
    ripple band, CA1 ripple times, HPC marks; and the artifact run finds no S
    block in the epoch. Every other run epoch of a project session is excluded
    with exactly one enumerated reason code and its evidence, in one register
    that every stage reads and the audit checks.

The register is ``epoch_register.csv`` beside this module, one row per
``(nwb_file_name, epoch)``: its ``status`` (:class:`Status`), its ``reason``
(:class:`Reason`, empty for an analysed epoch) and its ``evidence``. An
``epoch`` of ``*`` covers every epoch of a session none of whose run epochs is
known (no NWB file, or no run epoch in it). The register was generated once, on
2026-10-05, from the exclusion audit (``.claude/research/refresh_exclusions/
exclusions_table.csv``: 553 run epochs and 4 sessions with none, 557 rows) and
the populate plan (``.claude/research/refresh_populate/populate_matrix.csv``);
it is hand-maintained from then on. ``docs/epoch_register.md`` is generated from
it (``scripts/qa/write_epoch_register_doc.py``), and a unit test holds the two
together.

When several reasons hold for one epoch, the register records the first in the
rule's order: no data (``NO_NWB``, ``NO_RUN_EPOCH``), then the task
(``DEPLETION_TASK``), then Spyglass (``NOT_INGESTED``, ``NO_TRIAL_INTERVALS``),
then the decode inputs (``PENDING_POPULATE``, ``MISSING_INPUT``), then the
artifact run (``SUSTAINED_NOISE``). The evidence names the others. An epoch the
artifact run has read is populated, so ``PENDING_POPULATE`` no longer holds for
it.

The cohort is pinned here, not in Spyglass. A populate that makes a pending
epoch decodable lists it in ``get_epoch_info``, but it is analysed only once its
decodes exist and its entry is changed to ``analysed``. Every stage that
enumerates epochs from Spyglass reads :func:`analysed_epochs` (the decoders, the
SWR runner, the Widloski runner, the table builders, the PFC runner), and every
reader of a decode folder reads :func:`analysed_decode_files` instead of a glob.
``get_epoch_info`` listing an epoch the register does not, or not listing one
the register analyses, stops them: the cohort never changes silently.

The incremental update of the pending epochs: after their populates, decode them
alone with ``--pending-only`` (:func:`pending_epochs`) into a staging
``--save_dir``, and run the SWR and Widloski runners with ``--pending-only``
(the Widloski runner with ``--root`` at that staging folder). Then promote the
staged decodes into the canonical folder in the same change that flips their
entries to ``analysed`` and regenerates the doc. The canonical folder only ever
holds analysed epochs' decodes, which the audit checks (``decode_folder``).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ..decoding.identifiers import format_epoch_identifier, parse_nwb_filename
from .utils import get_epoch_info

REGISTER_PATH = Path(__file__).with_name("epoch_register.csv")
"""The committed register, one row per (nwb_file_name, epoch)."""

COLUMNS = ("nwb_file_name", "epoch", "status", "reason", "evidence")
SESSION_WIDE = "*"
"""The ``epoch`` of an entry that covers every epoch of its session."""

RULE = (
    "An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) "
    "of a regular spatial-bandit session in Spyglass (or of a listed "
    "first-exposure session), with parsed trials (`noPrePostTrialTimes`), and "
    "every decode input resolves to exactly one row: position map, position info "
    "and linearization, ripple band, CA1 ripple times, HPC marks; and the artifact "
    "run finds no S block in the epoch. Every other run epoch of a project session "
    "is excluded with exactly one enumerated reason code and its evidence, in one "
    "register that every stage reads and the audit checks."
)
"""The project lead's exclusion rule (2026-10-05), with the S-block clause added
on 2026-10-10."""

_RUN_EPOCH = re.compile(r"^[0-9]+_r[0-9]+$")


class Status(StrEnum):
    """Whether the stages analyse an epoch."""

    ANALYSED = "analysed"
    """In the cohort: every stage processes it."""
    PENDING = "pending"
    """Not yet analysed: its reason is ``PENDING_POPULATE``. Only a
    ``--pending-only`` run of the decoders, the SWR runner or the Widloski runner
    processes it."""
    EXCLUDED = "excluded"
    """Never analysed, for its reason."""


class Reason(StrEnum):
    """Why an epoch is not analysed. Exactly one per unanalysed entry."""

    NO_NWB = "NO_NWB"
    NO_RUN_EPOCH = "NO_RUN_EPOCH"
    DEPLETION_TASK = "DEPLETION_TASK"
    NOT_INGESTED = "NOT_INGESTED"
    NO_TRIAL_INTERVALS = "NO_TRIAL_INTERVALS"
    PENDING_POPULATE = "PENDING_POPULATE"
    MISSING_INPUT = "MISSING_INPUT"
    SUSTAINED_NOISE = "SUSTAINED_NOISE"


REASON_DEFINITIONS: dict[Reason, str] = {
    Reason.NO_NWB: (
        "No NWB file in either raw store and no Spyglass entry: there are no data."
    ),
    Reason.NO_RUN_EPOCH: "The session's NWB file holds no run epoch.",
    Reason.DEPLETION_TASK: (
        "A depletion session: a decaying reward schedule, a task variant, not the "
        "regular spatial bandit."
    ),
    Reason.NOT_INGESTED: "The session is not (fully) ingested in Spyglass.",
    Reason.NO_TRIAL_INTERVALS: (
        "Trial parsing failed: the epoch has no `noPrePostTrialTimes` interval."
    ),
    Reason.PENDING_POPULATE: (
        "A decode input is missing from Spyglass and a planned populate adds it; "
        "analysed once its decodes exist and its entry is updated."
    ),
    Reason.MISSING_INPUT: (
        "A decode input is missing from Spyglass and no fix is planned."
    ),
    Reason.SUSTAINED_NOISE: (
        "The artifact run finds an S block (sustained ripple-band noise) in the "
        "epoch. That noise's effect on the decode outside the mask could not be "
        "bounded, so the whole epoch is left out (project lead, 2026-10-10)."
    ),
}
"""One line per reason code."""


def reason_status(reason: Reason) -> Status:
    """The status an entry with this reason has: pending only for a populate."""
    return Status.PENDING if reason is Reason.PENDING_POPULATE else Status.EXCLUDED


class RegisterEntry(NamedTuple):
    """One register entry: an epoch's status, its reason and the evidence."""

    status: Status
    reason: Reason | None
    evidence: str


class EpochRegisterError(LookupError):
    """An epoch the register does not list, or a cohort Spyglass does not hold."""


def _problems(frame: pd.DataFrame) -> list[str]:
    """Every way `frame`'s rows break the register's contract."""
    problems = []
    keys = frame["nwb_file_name"] + " " + frame["epoch"]
    for key in sorted(set(keys[keys.duplicated()])):
        problems.append(f"{key} is registered more than once")
    session_wide = frame["epoch"] == SESSION_WIDE
    overlap = set(frame.loc[session_wide, "nwb_file_name"]) & set(
        frame.loc[~session_wide, "nwb_file_name"]
    )
    for name in sorted(overlap):
        problems.append(f"{name} has a session-wide '*' entry and epoch entries")
    for row in frame.itertuples(index=False):
        key = f"{row.nwb_file_name} {row.epoch}"
        if row.epoch != SESSION_WIDE and not _RUN_EPOCH.match(row.epoch):
            problems.append(f"{key}: epoch is neither '*' nor a run epoch name")
        if row.status not in set(Status):
            problems.append(
                f"{key}: status {row.status!r} is not one of "
                f"{', '.join(status.value for status in Status)}"
            )
            continue
        if row.status == Status.ANALYSED:
            if row.reason:
                problems.append(
                    f"{key}: an analysed entry has no reason, not {row.reason!r}"
                )
            if row.epoch == SESSION_WIDE:
                problems.append(f"{key}: a session-wide '*' entry must be excluded")
            continue
        if row.reason not in set(Reason):
            problems.append(f"{key}: reason {row.reason!r} is not a Reason code")
            continue
        expected = reason_status(Reason(row.reason))
        if row.status != expected:
            problems.append(
                f"{key}: PENDING_POPULATE is the reason of a pending entry, not of "
                "an excluded one"
                if row.reason == Reason.PENDING_POPULATE
                else f"{key}: a pending entry's reason is PENDING_POPULATE, not "
                f"{row.reason}"
            )
        if row.epoch == SESSION_WIDE and row.status != Status.EXCLUDED:
            problems.append(
                f"{key}: a session-wide '*' entry must be excluded, not {row.status}"
            )
        if not row.evidence.strip():
            problems.append(f"{key}: an unanalysed entry needs its evidence")
    return problems


def read_register(path: Path | None = None) -> pd.DataFrame:
    """Read and check an epoch register.

    Parameters
    ----------
    path : Path, optional
        A register CSV; ``REGISTER_PATH``, the committed one, by default.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(nwb_file_name, epoch)``, with columns ``status``,
        ``reason`` (``""`` for an analysed epoch) and ``evidence``.

    Raises
    ------
    ValueError
        If the file breaks the register's contract: an epoch registered twice,
        a session-wide entry beside epoch entries, an unknown status or reason,
        a reason that does not match the status, or an unanalysed entry with no
        evidence.
    """
    if path is None:
        path = REGISTER_PATH
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns {list(frame.columns)}, not {list(COLUMNS)}")
    problems = _problems(frame)
    if problems:
        raise ValueError(
            f"{path} is not a valid epoch register: " + "; ".join(problems[:10])
        )
    return frame.set_index(["nwb_file_name", "epoch"]).sort_index()


def register_entry(
    nwb_file_name: str, epoch: str, register: pd.DataFrame | None = None
) -> RegisterEntry:
    """The register's entry for an epoch: its own, or its session's ``*`` entry.

    Parameters
    ----------
    nwb_file_name, epoch : str
        Such as ``"chimi20200212_.nwb"`` and ``"02_r1"``.
    register : pd.DataFrame, optional
        ``read_register``'s frame; the committed register by default.

    Raises
    ------
    EpochRegisterError
        If the register has no entry for the epoch.
    """
    if register is None:
        register = read_register()
    for key in ((nwb_file_name, epoch), (nwb_file_name, SESSION_WIDE)):
        if key in register.index:
            row = register.loc[key]
            return RegisterEntry(
                Status(row["status"]),
                Reason(row["reason"]) if row["reason"] else None,
                row["evidence"],
            )
    raise EpochRegisterError(
        f"{nwb_file_name} {epoch} is not in the epoch register ({REGISTER_PATH.name})"
    )


def register_reason(nwb_file_name: str, epoch: str) -> Reason | None:
    """Why the register does not analyse an epoch; None if it does.

    Raises
    ------
    EpochRegisterError
        If the register has no entry for the epoch.
    """
    return register_entry(nwb_file_name, epoch).reason


def select_epochs(
    epoch_info: pd.DataFrame,
    statuses: set[Status],
    register: pd.DataFrame,
    reasons: Collection[Reason] = (),
) -> pd.DataFrame:
    """``epoch_info``'s rows whose register status is one of `statuses`, or
    whose reason is one of `reasons`.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        ``get_epoch_info``'s frame, indexed by ``(nwb_file_name,
        interval_list_name)``.
    statuses : set of Status
    register : pd.DataFrame
        ``read_register``'s frame.
    reasons : collection of Reason, optional
        Unanalysed entries also selected, such as ``SUSTAINED_NOISE`` for the
        artifact runner.

    Raises
    ------
    EpochRegisterError
        If `epoch_info` lists an epoch the register does not, or lacks an epoch
        the register gives one of `statuses` or `reasons`.
    """
    keys = list(epoch_info.index)
    unregistered, absent = registration_gaps(keys, register, statuses, reasons)
    if unregistered:
        raise EpochRegisterError(
            f"get_epoch_info lists {len(unregistered)} epoch(s) the epoch register "
            "does not; register each with its status, reason and evidence: "
            + ", ".join(f"{name} {epoch}" for name, epoch in unregistered)
        )
    if len(absent):
        by_status = ", ".join(
            f"{len(group)} {status} epoch(s)"
            for status, group in absent.groupby("status", sort=True)
        )
        raise EpochRegisterError(
            f"{by_status} of the register are not in get_epoch_info, so Spyglass "
            "cannot serve them (a pending one needs its populate first): "
            + ", ".join(f"{name} {epoch}" for name, epoch in absent.index)
        )
    entries = [register_entry(name, epoch, register) for name, epoch in keys]
    keep = [entry.status in statuses or entry.reason in reasons for entry in entries]
    return epoch_info[keep]


def registration_gaps(
    keys: list[tuple[str, str]],
    register: pd.DataFrame,
    statuses: set[Status],
    reasons: Collection[Reason] = (),
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """The listed epochs the register lacks, and its epochs of `statuses` (or
    `reasons`) not listed.

    Parameters
    ----------
    keys : list of (str, str)
        ``(nwb_file_name, epoch)`` of the epochs ``get_epoch_info`` lists.
    register : pd.DataFrame
        ``read_register``'s frame.
    statuses : set of Status
    reasons : collection of Reason, optional

    Returns
    -------
    unregistered : list of (str, str)
        Listed keys with no entry, theirs or their session's.
    absent : pd.DataFrame
        The register's epoch entries with one of `statuses` or `reasons` that
        `keys` lacks.
    """
    unregistered = [
        (name, epoch)
        for name, epoch in keys
        if (name, epoch) not in register.index
        and (name, SESSION_WIDE) not in register.index
    ]
    wanted = register[
        register["status"].isin(statuses) | register["reason"].isin(reasons)
    ]
    absent = wanted[~wanted.index.isin(keys)]
    return unregistered, absent


def analysed_epochs(
    include_pending: bool = False, include_sustained_noise: bool = False
) -> pd.DataFrame:
    """``get_epoch_info``'s epochs that the register analyses: the cohort.

    Parameters
    ----------
    include_pending : bool, optional
        Also the pending epochs. The artifact runner asks for both at once
        (``--include-pending``: runbook R9 runs the analysed and the pending
        epochs together). The incremental update of the decoders, the SWR runner
        and the Widloski runner takes :func:`pending_epochs` alone.
    include_sustained_noise : bool, optional
        Also the epochs excluded for ``SUSTAINED_NOISE``. Only the artifact
        runner asks for them, so that a new artifact run re-tests them; the
        audit's ``sustained_noise`` check needs their cores.

    Returns
    -------
    pd.DataFrame
        ``get_epoch_info``'s rows and columns for those epochs, in its order.

    Raises
    ------
    EpochRegisterError
        If ``get_epoch_info`` lists an epoch the register does not, or does not
        list an epoch the register analyses (or, with `include_pending`, one it
        holds pending, or with `include_sustained_noise`, one it excludes for
        ``SUSTAINED_NOISE``): the cohort would otherwise change without a
        register edit.
    """
    statuses = {Status.ANALYSED}
    if include_pending:
        statuses.add(Status.PENDING)
    reasons = {Reason.SUSTAINED_NOISE} if include_sustained_noise else set()
    return select_epochs(get_epoch_info(), statuses, read_register(), reasons)


def pending_epochs() -> pd.DataFrame:
    """``get_epoch_info``'s epochs that the register holds pending, and no other.

    For an incremental run over the pending epochs alone (``--pending-only``),
    after their populates.

    Returns
    -------
    pd.DataFrame
        ``get_epoch_info``'s rows and columns for those epochs, in its order.

    Raises
    ------
    EpochRegisterError
        If the register holds no pending epoch, so there is nothing to run; if
        ``get_epoch_info`` does not list one of them, so its populate has not
        happened; or if it lists an epoch the register does not.
    """
    register = read_register()
    if not (register["status"] == Status.PENDING).any():
        raise EpochRegisterError("the epoch register holds no pending epoch to run")
    return select_epochs(get_epoch_info(), {Status.PENDING}, register)


def epochs_to_run(pending_only: bool = False) -> pd.DataFrame:
    """The epochs a stage runs: the analysed ones, or with `pending_only` the
    pending ones alone (:func:`analysed_epochs`, :func:`pending_epochs`)."""
    return pending_epochs() if pending_only else analysed_epochs()


def analysed_epoch_ids(register: pd.DataFrame | None = None) -> list[str]:
    """``{stem}_{epoch}_HPC`` of every analysed epoch, sorted.

    The name every HPC decode file of the epoch starts with. Read from the
    register alone, not from Spyglass, so a reader of decode files needs no
    database.
    """
    if register is None:
        register = read_register()
    analysed = register.index[register["status"] == Status.ANALYSED]
    return sorted(
        format_epoch_identifier(name, epoch, "HPC") for name, epoch in analysed
    )


def analysed_animals(register: pd.DataFrame | None = None) -> tuple[str, ...]:
    """The animals with an analysed epoch, in register order.

    Read from the register alone, like :func:`analysed_epoch_ids`, so a script
    that lists the cohort's animals needs no database. The register is sorted
    by NWB file name, so the order is alphabetical:
    ``("chimi", "j16", "peanut", "senor", "wilbur")``.
    """
    if register is None:
        register = read_register()
    analysed = register.index[register["status"] == Status.ANALYSED]
    animals = (parse_nwb_filename(name)[0] for name, _ in analysed)
    return tuple(dict.fromkeys(animals))


def analysed_decode_files(
    directory: Path, suffix: str, *, require_all: bool = False
) -> list[Path]:
    """The analysed epochs' decode files ``{epoch_id}{suffix}`` in `directory`.

    The register's replacement for globbing a decode folder: a file of an epoch
    the register does not analyse (pending, excluded or unregistered) is never
    returned, whatever else the folder holds.

    Parameters
    ----------
    directory : Path
        A decode folder, such as ``paths.processed_data_dir()``.
    suffix : str
        What follows the epoch id, such as ``"_80_data_over_time.pkl"`` or
        ``"_contfrag_data_over_time.pkl"``.
    require_all : bool, optional
        Raise if an analysed epoch's file is missing, rather than leave it out.

    Returns
    -------
    list of Path
        Sorted by epoch id; only files that exist.

    Raises
    ------
    FileNotFoundError
        With `require_all`, if an analysed epoch's file is not in `directory`.
    """
    paths = [directory / f"{epoch_id}{suffix}" for epoch_id in analysed_epoch_ids()]
    present = [path for path in paths if path.exists()]
    if require_all and len(present) < len(paths):
        missing = [path.name for path in paths if not path.exists()]
        raise FileNotFoundError(
            f"{len(missing)} of the {len(paths)} analysed epochs have no "
            f"*{suffix} in {directory}: {', '.join(missing[:10])}"
            + (", ..." if len(missing) > 10 else "")
        )
    return present
