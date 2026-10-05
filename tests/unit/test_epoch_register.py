"""The epoch register: its contents, its rule and the selection every stage uses.

Nothing here reaches the Spyglass database: ``get_epoch_info`` is replaced by a
frame of the epochs a test names, and a test of ``get_epoch_info`` itself
replaces ``spyglass.common`` in ``sys.modules``. The register checked is the
committed one; the malformed registers are written under ``tmp_path``.

The audit's keys and the populate matrix's keys are pinned under
``tests/unit/data/epoch_register/``: ``exclusions_table_keys.csv`` holds the
557 rows of the 2026-10-05 exclusion audit (``exclusions_table.csv``), with its
status, its class and whether the October re-decode made the epoch's decode;
``populate_matrix_keys.csv`` holds the 48 target epochs of the populate plan
(``populate_matrix.csv``), with their group.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.data_loaders import epoch_register, utils
from continuum_swr_replay.data_loaders.constants import (
    FIRST_EXPOSURE_DESCRIPTIONS,
    NWB_FILES,
    REGULAR_SESSION_DESCRIPTION,
)
from continuum_swr_replay.data_loaders.epoch_register import (
    EpochRegisterError,
    Reason,
    Status,
    analysed_epochs,
    read_register,
    register_entry,
    register_reason,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

DATA = Path(__file__).resolve().parent / "data" / "epoch_register"
AUDIT_KEYS = DATA / "exclusions_table_keys.csv"
POPULATE_KEYS = DATA / "populate_matrix_keys.csv"
COLUMNS = ["nwb_file_name", "epoch", "status", "reason", "evidence"]


def _audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_KEYS, dtype=str, keep_default_na=False)


def _populate() -> pd.DataFrame:
    return pd.read_csv(POPULATE_KEYS, dtype=str, keep_default_na=False)


def _epoch_info(keys: list[tuple[str, str]]) -> pd.DataFrame:
    """``get_epoch_info``'s frame for these (nwb_file_name, epoch) pairs."""
    animal_date = [parse_nwb_filename(nwb_file_name) for nwb_file_name, _ in keys]
    return pd.DataFrame(
        {
            "animal": [animal for animal, _ in animal_date],
            "date": [date for _, date in animal_date],
            "epoch": [int(epoch.split("_")[0]) for _, epoch in keys],
        },
        index=pd.MultiIndex.from_tuples(
            keys, names=["nwb_file_name", "interval_list_name"]
        ),
    )


def _registered(*statuses: Status) -> list[tuple[str, str]]:
    """The committed register's epoch-level keys with one of these statuses."""
    register = read_register()
    keep = register["status"].isin(statuses)
    return [key for key in register.index[keep] if key[1] != "*"]


def _decoded() -> list[tuple[str, str]]:
    """The 218 epochs the October re-decode decoded, pinned from the audit."""
    audit = _audit()
    decoded = audit[audit["decoded_in_redecode"] == "y"]
    return list(zip(decoded["nwb_file_name"], decoded["epoch"], strict=True))


def _write_register(path: Path, rows: list[list[str]]) -> Path:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# The committed register
# --------------------------------------------------------------------------


def test_the_register_covers_every_row_of_the_audit_and_nothing_else():
    audit = _audit()
    audited = set(
        zip(audit["nwb_file_name"], audit["epoch"].replace("", "*"), strict=True)
    )

    assert len(audit) == 557
    assert set(read_register().index) == audited


def test_no_epoch_has_two_statuses():
    frame = pd.read_csv(epoch_register.REGISTER_PATH, dtype=str, keep_default_na=False)
    assert list(frame.columns) == COLUMNS
    assert not frame.duplicated(["nwb_file_name", "epoch"]).any()
    session_wide = set(frame.loc[frame["epoch"] == "*", "nwb_file_name"])
    with_epochs = set(frame.loc[frame["epoch"] != "*", "nwb_file_name"])
    assert session_wide.isdisjoint(with_epochs)


def test_every_unanalysed_entry_has_one_reason_and_its_evidence():
    register = read_register()
    analysed = register["status"] == Status.ANALYSED

    assert (register.loc[analysed, "reason"] == "").all()
    others = register[~analysed]
    assert others["reason"].map(lambda reason: reason in set(Reason)).all()
    assert (others["evidence"].str.strip() != "").all()
    pending = others["status"] == Status.PENDING
    assert (others.loc[pending, "reason"] == Reason.PENDING_POPULATE).all()
    assert (others.loc[~pending, "reason"] != Reason.PENDING_POPULATE).all()


def test_the_analysed_epochs_are_the_ones_the_audit_found_decoded():
    audit = _audit()
    analysed = audit[audit["status"] == "analysed"]

    assert sorted(_registered(Status.ANALYSED)) == sorted(_decoded())
    assert sorted(_decoded()) == sorted(
        zip(analysed["nwb_file_name"], analysed["epoch"], strict=True)
    )
    assert len(_decoded()) == 218


def test_the_pending_epochs_are_the_populate_plans_fixable_groups():
    populate = _populate()
    fixable = populate[~populate["group"].str.startswith("G5")]
    blocked = populate[populate["group"].str.startswith("G5")]

    assert len(fixable) == 40
    assert sorted(_registered(Status.PENDING)) == sorted(
        zip(fixable["nwb_file_name"], fixable["epoch"], strict=True)
    )
    register = read_register()
    is_blocked = register["reason"] == Reason.NOT_INGESTED_BLOCKED
    assert sorted(register.index[is_blocked]) == sorted(
        zip(blocked["nwb_file_name"], blocked["epoch"], strict=True)
    )
    assert len(blocked) == 8


def test_every_reason_has_a_one_line_definition():
    assert set(epoch_register.REASON_DEFINITIONS) == set(Reason)
    for definition in epoch_register.REASON_DEFINITIONS.values():
        assert definition.strip() and "\n" not in definition


def test_the_rule_is_stated_in_the_module_docstring():
    docstring = " ".join((epoch_register.__doc__ or "").split())
    assert " ".join(epoch_register.RULE.split()) in docstring


# --------------------------------------------------------------------------
# Reading a register by hand edits
# --------------------------------------------------------------------------


GOOD_ROWS = [
    ["a20200101_.nwb", "02_r1", "analysed", "", ""],
    ["a20200101_.nwb", "04_r2", "pending", "PENDING_POPULATE", "no PIM row"],
    ["b20200101_.nwb", "*", "excluded", "NO_NWB", "no NWB file anywhere"],
]


def test_a_well_formed_register_is_read_by_epoch(tmp_path):
    register = read_register(_write_register(tmp_path / "r.csv", GOOD_ROWS))

    assert list(register.index.names) == ["nwb_file_name", "epoch"]
    assert register.loc[("a20200101_.nwb", "04_r2"), "status"] == "pending"


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        (["a20200101_.nwb", "02_r1", "excluded", "NO_NWB", "x"], "more than once"),
        (["b20200101_.nwb", "02_r1", "analysed", "", ""], "session-wide"),
        (["c20200101_.nwb", "02_r1", "kept", "", ""], "status"),
        (["c20200101_.nwb", "02_r1", "excluded", "NO_REASON", "x"], "reason"),
        (["c20200101_.nwb", "02_r1", "excluded", "NO_NWB", " "], "evidence"),
        (["c20200101_.nwb", "02_r1", "analysed", "NO_NWB", "x"], "reason"),
        (["c20200101_.nwb", "02_r1", "pending", "NO_NWB", "x"], "PENDING_POPULATE"),
        (["c20200101_.nwb", "02_r1", "excluded", "PENDING_POPULATE", "x"], "pending"),
        (["c20200101_.nwb", "2_s1", "excluded", "NO_NWB", "x"], "run epoch"),
        (["c20200101_.nwb", "*", "pending", "PENDING_POPULATE", "x"], "session-wide"),
    ],
    ids=[
        "duplicate",
        "star-and-epoch",
        "unknown-status",
        "unknown-reason",
        "blank-evidence",
        "analysed-with-reason",
        "pending-other-reason",
        "excluded-pending-reason",
        "not-a-run-epoch",
        "star-not-excluded",
    ],
)
def test_a_malformed_register_is_refused(tmp_path, bad_row, message):
    path = _write_register(tmp_path / "r.csv", [*GOOD_ROWS, bad_row])

    with pytest.raises(ValueError, match=message):
        read_register(path)


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------


def test_register_reason():
    assert register_reason("chimi20200213_.nwb", "02_r1") is None
    assert register_reason("senor20201104_.nwb", "06_r3") is Reason.PENDING_POPULATE
    assert register_reason("wilbur20210401_.nwb", "02_r1") is (
        Reason.NOT_INGESTED_BLOCKED
    )
    assert register_reason("chimi20200304_.nwb", "02_r1") is Reason.DEPLETION_TASK
    # a session-wide entry covers any epoch of its session
    assert register_reason("chimi20200302_.nwb", "02_r1") is Reason.NO_NWB
    assert register_reason("chimi20200301_.nwb", "02_r1") is Reason.NO_RUN_EPOCH


def test_an_unregistered_epoch_has_no_reason_but_an_error():
    with pytest.raises(EpochRegisterError, match="testrat20990101_.nwb 02_r1"):
        register_reason("testrat20990101_.nwb", "02_r1")
    with pytest.raises(EpochRegisterError):
        register_entry("chimi20200213_.nwb", "99_r9")


def test_register_entry_carries_status_reason_and_evidence():
    entry = register_entry("senor20201105_.nwb", "09_r4")

    assert entry.status is Status.PENDING
    assert entry.reason is Reason.PENDING_POPULATE
    assert "IntervalLinearizedPosition" in entry.evidence


# --------------------------------------------------------------------------
# analysed_epochs
# --------------------------------------------------------------------------


def test_analysed_epochs_are_the_218_decoded_ones_and_258_with_pending(monkeypatch):
    """Once the populates list all 258 epochs, the cohort is still the 218."""
    every = _registered(Status.ANALYSED, Status.PENDING)
    monkeypatch.setattr(epoch_register, "get_epoch_info", lambda: _epoch_info(every))

    assert sorted(analysed_epochs().index) == sorted(_decoded())
    assert len(analysed_epochs(include_pending=True)) == 258


def test_analysed_epochs_today_ignore_what_spyglass_lists_beyond_the_cohort(
    monkeypatch,
):
    """Today ``get_epoch_info`` lists 222 epochs: the 218, senor20201104 06_r3 and
    08_r4, senor20201105 09_r4, and wilbur20210326 03_r1 through the allowlist."""
    listed = [
        *_decoded(),
        ("senor20201104_.nwb", "06_r3"),
        ("senor20201104_.nwb", "08_r4"),
        ("senor20201105_.nwb", "09_r4"),
        ("wilbur20210326_.nwb", "03_r1"),
    ]
    monkeypatch.setattr(epoch_register, "get_epoch_info", lambda: _epoch_info(listed))

    selected = analysed_epochs()

    assert sorted(selected.index) == sorted(_decoded())
    assert list(selected.columns) == ["animal", "date", "epoch"]
    # the 36 pending epochs with no PositionIntervalMap row are not listed yet
    with pytest.raises(EpochRegisterError, match="36 pending epoch"):
        analysed_epochs(include_pending=True)


def test_analysed_epochs_raise_on_an_unregistered_epoch(monkeypatch):
    listed = [*_decoded(), ("testrat20990101_.nwb", "02_r1")]
    monkeypatch.setattr(epoch_register, "get_epoch_info", lambda: _epoch_info(listed))

    with pytest.raises(EpochRegisterError, match="testrat20990101_.nwb 02_r1"):
        analysed_epochs()


def test_analysed_epochs_raise_when_an_analysed_epoch_is_not_listed(monkeypatch):
    listed = _decoded()[1:]
    monkeypatch.setattr(epoch_register, "get_epoch_info", lambda: _epoch_info(listed))

    with pytest.raises(EpochRegisterError, match="1 analysed epoch"):
        analysed_epochs()


# --------------------------------------------------------------------------
# get_epoch_info's first-exposure allowlist
# --------------------------------------------------------------------------


def test_the_first_exposure_allowlist_admits_wilbur20210326_and_nothing_else():
    regular = [
        {"nwb_file_name": name, "session_description": REGULAR_SESSION_DESCRIPTION}
        for name in NWB_FILES
    ]
    restriction = utils.session_restriction()

    assert restriction[: len(regular)] == regular
    assert restriction[len(regular) :] == [
        {
            "nwb_file_name": "wilbur20210326_.nwb",
            "session_description": (
                "Spatial bandit task (regular) - exploration session"
            ),
        }
    ]
    assert set(FIRST_EXPOSURE_DESCRIPTIONS) <= set(NWB_FILES)


def test_get_epoch_info_restricts_session_to_the_admitted_descriptions(monkeypatch):
    """``get_epoch_info`` hands ``Session`` exactly ``session_restriction()``."""
    seen = []

    class _Stop(Exception):
        pass

    class _Session:
        def __and__(self, restriction):
            seen.append(restriction)
            raise _Stop

    common = types.ModuleType("spyglass.common")
    common.Session = _Session()
    common.IntervalList = object()
    common.PositionIntervalMap = object()
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    utils.get_epoch_info.cache_clear()
    try:
        with pytest.raises(_Stop):
            utils.get_epoch_info()
    finally:
        utils.get_epoch_info.cache_clear()

    assert seen == [utils.session_restriction()]
