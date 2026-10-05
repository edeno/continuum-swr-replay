"""The epoch register: which run epochs are analysed, and why every other one is not.

The rule:

    An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) of
    a regular spatial-bandit session in Spyglass (or of a listed first-exposure
    session), with parsed trials (`noPrePostTrialTimes`), and every decode input
    resolves to exactly one row: position map, position info and linearization,
    ripple band, CA1 ripple times, HPC marks. Every other run epoch of a project
    session is excluded with exactly one enumerated reason code and its evidence,
    in one register that every stage reads and the audit checks.

The register is ``epoch_register.csv`` beside this module, one row per
``(nwb_file_name, epoch)``: its ``status`` (:class:`Status`), its ``reason``
(:class:`Reason`, empty for an analysed epoch) and its ``evidence``. An
``epoch`` of ``*`` covers every epoch of a session none of whose run epochs is
known (no NWB file, or no run epoch in it). The register was generated once, on
2026-10-05, from the exclusion audit (``.claude/research/refresh_exclusions/
exclusions_table.csv``, 557 run epochs) and the populate plan
(``.claude/research/refresh_populate/populate_matrix.csv``); it is
hand-maintained from then on. ``docs/epoch_register.md`` mirrors it, and a unit
test holds the two together.

When several reasons hold for one epoch, the register records the first in the
rule's order: no data (``NO_NWB``, ``NO_RUN_EPOCH``), then the task
(``DEPLETION_TASK``), then Spyglass (``NOT_INGESTED_BLOCKED``,
``NO_TRIAL_INTERVALS``), then the decode inputs (``PENDING_POPULATE``,
``MISSING_INPUT``). The evidence names the others.

The cohort is pinned here, not in Spyglass. A populate that makes a pending
epoch decodable lists it in ``get_epoch_info``, but it is analysed only once its
decodes exist and its entry is changed to ``analysed``. Every stage that
enumerates epochs reads :func:`analysed_epochs`: the decoders (with
``--include-pending`` for that incremental decode), the SWR runner, the Widloski
runner, the table builders and the PFC runner. ``get_epoch_info`` listing an
epoch the register does not, or not listing one the register analyses, stops
them: the cohort never changes silently.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd

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
    "and linearization, ripple band, CA1 ripple times, HPC marks. Every other run "
    "epoch of a project session is excluded with exactly one enumerated reason "
    "code and its evidence, in one register that every stage reads and the audit "
    "checks."
)
"""The project lead's exclusion rule, verbatim."""

_RUN_EPOCH = re.compile(r"^[0-9]+_r[0-9]+$")


class Status(StrEnum):
    """Whether the stages analyse an epoch."""

    ANALYSED = "analysed"
    """In the cohort: every stage processes it."""
    PENDING = "pending"
    """Not yet analysed: its reason is ``PENDING_POPULATE``. Only the decoders,
    given ``--include-pending``, process it."""
    EXCLUDED = "excluded"
    """Never analysed, for its reason."""


class Reason(StrEnum):
    """Why an epoch is not analysed. Exactly one per unanalysed entry."""

    NO_NWB = "NO_NWB"
    NO_RUN_EPOCH = "NO_RUN_EPOCH"
    DEPLETION_TASK = "DEPLETION_TASK"
    NOT_INGESTED_BLOCKED = "NOT_INGESTED_BLOCKED"
    NO_TRIAL_INTERVALS = "NO_TRIAL_INTERVALS"
    PENDING_POPULATE = "PENDING_POPULATE"
    MISSING_INPUT = "MISSING_INPUT"


REASON_DEFINITIONS: dict[Reason, str] = {
    Reason.NO_NWB: (
        "No NWB file in either raw store and no Spyglass entry: there are no data."
    ),
    Reason.NO_RUN_EPOCH: "The session's NWB file holds no run epoch.",
    Reason.DEPLETION_TASK: (
        "A depletion session (a decaying reward schedule) or the session run "
        "after the depletion block: a task variant, not the regular spatial bandit."
    ),
    Reason.NOT_INGESTED_BLOCKED: (
        "A regular session that Spyglass ingested only in part, whose "
        "re-ingestion is blocked."
    ),
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


def read_register(path: Path = REGISTER_PATH) -> pd.DataFrame:
    """Read and check an epoch register.

    Parameters
    ----------
    path : Path, optional
        A register CSV; the committed one by default.

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
    epoch_info: pd.DataFrame, statuses: set[Status], register: pd.DataFrame
) -> pd.DataFrame:
    """``epoch_info``'s rows whose register status is one of `statuses`.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        ``get_epoch_info``'s frame, indexed by ``(nwb_file_name,
        interval_list_name)``.
    statuses : set of Status
    register : pd.DataFrame
        ``read_register``'s frame.

    Raises
    ------
    EpochRegisterError
        If `epoch_info` lists an epoch the register does not, or lacks an epoch
        the register gives one of `statuses`.
    """
    keys = list(epoch_info.index)
    unregistered, absent = registration_gaps(keys, register, statuses)
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
    keep = [
        register_entry(name, epoch, register).status in statuses for name, epoch in keys
    ]
    return epoch_info[keep]


def registration_gaps(
    keys: list[tuple[str, str]], register: pd.DataFrame, statuses: set[Status]
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """The listed epochs the register lacks, and its epochs of `statuses` not listed.

    Parameters
    ----------
    keys : list of (str, str)
        ``(nwb_file_name, epoch)`` of the epochs ``get_epoch_info`` lists.
    register : pd.DataFrame
        ``read_register``'s frame.
    statuses : set of Status

    Returns
    -------
    unregistered : list of (str, str)
        Listed keys with no entry, theirs or their session's.
    absent : pd.DataFrame
        The register's epoch entries with one of `statuses` that `keys` lacks.
    """
    unregistered = [
        (name, epoch)
        for name, epoch in keys
        if (name, epoch) not in register.index
        and (name, SESSION_WIDE) not in register.index
    ]
    wanted = register[register["status"].isin(statuses)]
    absent = wanted[~wanted.index.isin(keys)]
    return unregistered, absent


def analysed_epochs(include_pending: bool = False) -> pd.DataFrame:
    """``get_epoch_info``'s epochs that the register analyses: the cohort.

    Parameters
    ----------
    include_pending : bool, optional
        Also the pending epochs, for the decoders' incremental decode of them.

    Returns
    -------
    pd.DataFrame
        ``get_epoch_info``'s rows and columns for those epochs, in its order.

    Raises
    ------
    EpochRegisterError
        If ``get_epoch_info`` lists an epoch the register does not, or does not
        list an epoch the register analyses (or, with `include_pending`, one it
        holds pending): the cohort would otherwise change without a register
        edit.
    """
    statuses = {Status.ANALYSED}
    if include_pending:
        statuses.add(Status.PENDING)
    return select_epochs(get_epoch_info(), statuses, read_register())
