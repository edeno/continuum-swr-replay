"""The epoch register: its contents, its rule and the selection every stage uses.

Nothing here reaches the Spyglass database: ``get_epoch_info`` is replaced by a
frame of the epochs a test names, and a test of ``get_epoch_info`` itself
replaces ``spyglass.common`` in ``sys.modules``.

The committed register is checked for its contract, and for covering every key
of the 2026-10-05 exclusion audit (``tests/unit/data/epoch_register/
exclusions_table_keys.csv``: the 557 keys of ``exclusions_table.csv``, a blank
epoch for a session with no run epoch). No test pins an epoch's status there
that a register edit could change: which epochs are analysed is the audit's to
check, live. The selections and lookups are tested on registers written under
``tmp_path``, which ``REGISTER_PATH`` is pointed at.
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
    analysed_decode_files,
    analysed_epoch_ids,
    analysed_epochs,
    epochs_to_run,
    pending_epochs,
    read_register,
    register_entry,
    register_reason,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

AUDIT_KEYS = (
    Path(__file__).resolve().parent
    / "data"
    / "epoch_register"
    / "exclusions_table_keys.csv"
)
COLUMNS = ["nwb_file_name", "epoch", "status", "reason", "evidence"]

ROWS = [
    ["a20200101_.nwb", "02_r1", "analysed", "", ""],
    ["a20200101_.nwb", "04_r2", "analysed", "", ""],
    ["a20200101_.nwb", "06_r3", "pending", "PENDING_POPULATE", "no PIM row"],
    ["b20200102_.nwb", "02_r1", "pending", "PENDING_POPULATE", "no LFPV1"],
    ["b20200102_.nwb", "04_r2", "excluded", "DEPLETION_TASK", "decay 80"],
    ["c20200103_.nwb", "*", "excluded", "NO_NWB", "no NWB file anywhere"],
]
"""A small register: two analysed, two pending, an excluded epoch, a
session-wide entry."""
ANALYSED = [("a20200101_.nwb", "02_r1"), ("a20200101_.nwb", "04_r2")]
PENDING = [("a20200101_.nwb", "06_r3"), ("b20200102_.nwb", "02_r1")]


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


def _write_register(path: Path, rows: list[list[str]]) -> Path:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def small_register(tmp_path, monkeypatch):
    """``REGISTER_PATH`` pointed at a register of `ROWS`."""
    path = _write_register(tmp_path / "epoch_register.csv", ROWS)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)
    return path


def _listing(monkeypatch, keys: list[tuple[str, str]]) -> None:
    """``get_epoch_info`` lists exactly `keys`."""
    monkeypatch.setattr(epoch_register, "get_epoch_info", lambda: _epoch_info(keys))


# --------------------------------------------------------------------------
# The committed register
# --------------------------------------------------------------------------


def test_the_register_covers_every_key_of_the_audit():
    audit = pd.read_csv(AUDIT_KEYS, dtype=str, keep_default_na=False)
    audited = set(
        zip(audit["nwb_file_name"], audit["epoch"].replace("", "*"), strict=True)
    )

    assert len(audit) == 557
    assert audited <= set(read_register().index)


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


def test_the_two_uningested_wilbur_sessions_are_not_ingested_not_depletion():
    """wilbur20210401 is partly ingested and blocked; wilbur20210416, "regular -
    after depletion", was never ingested. Neither is a depletion session."""
    register = read_register()
    for name, n_epochs in (("wilbur20210401_.nwb", 8), ("wilbur20210416_.nwb", 7)):
        rows = register.loc[name]
        assert len(rows) == n_epochs
        assert (rows["reason"] == Reason.NOT_INGESTED).all()
    assert (register.loc["wilbur20210416_.nwb", "evidence"].str.startswith(
        "Never ingested"
    )).all()  # fmt: skip
    assert (register.loc["wilbur20210404_.nwb", "reason"] == "DEPLETION_TASK").all()


def test_every_reason_has_a_one_line_definition():
    assert set(epoch_register.REASON_DEFINITIONS) == set(Reason)
    for definition in epoch_register.REASON_DEFINITIONS.values():
        assert definition.strip() and "\n" not in definition


def test_the_rule_is_stated_in_the_module_docstring():
    docstring = " ".join((epoch_register.__doc__ or "").split())
    assert " ".join(epoch_register.RULE.split()) in docstring


def test_the_committed_cohort_is_whatever_the_register_analyses(monkeypatch):
    """With Spyglass listing every analysed and pending epoch, as after the
    populates, the cohort is exactly the register's analysed entries."""
    register = read_register()
    keys = [key for key in register.index if key[1] != "*"]
    status = dict(zip(register.index, register["status"], strict=True))
    listed = [key for key in keys if status[key] != Status.EXCLUDED]
    _listing(monkeypatch, listed)

    assert set(analysed_epochs().index) == {
        key for key in listed if status[key] == Status.ANALYSED
    }
    assert set(pending_epochs().index) == {
        key for key in listed if status[key] == Status.PENDING
    }


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


def test_register_reason(small_register):
    assert register_reason("a20200101_.nwb", "02_r1") is None
    assert register_reason("a20200101_.nwb", "06_r3") is Reason.PENDING_POPULATE
    assert register_reason("b20200102_.nwb", "04_r2") is Reason.DEPLETION_TASK
    # a session-wide entry covers any epoch of its session
    assert register_reason("c20200103_.nwb", "08_r4") is Reason.NO_NWB


def test_an_unregistered_epoch_has_no_reason_but_an_error(small_register):
    with pytest.raises(EpochRegisterError, match="testrat20990101_.nwb 02_r1"):
        register_reason("testrat20990101_.nwb", "02_r1")
    with pytest.raises(EpochRegisterError):
        register_entry("a20200101_.nwb", "99_r9")


def test_register_entry_carries_status_reason_and_evidence(small_register):
    entry = register_entry("b20200102_.nwb", "02_r1")

    assert entry.status is Status.PENDING
    assert entry.reason is Reason.PENDING_POPULATE
    assert entry.evidence == "no LFPV1"


# --------------------------------------------------------------------------
# analysed_epochs, pending_epochs, epochs_to_run
# --------------------------------------------------------------------------


def test_analysed_epochs_keep_get_epoch_infos_rows_the_register_analyses(
    small_register, monkeypatch
):
    """Spyglass also lists pending epochs; the cohort is still the analysed."""
    listed = [PENDING[0], ANALYSED[1], ANALYSED[0], PENDING[1]]
    _listing(monkeypatch, listed)

    selected = analysed_epochs()

    assert list(selected.index) == [ANALYSED[1], ANALYSED[0]]  # its order
    assert list(selected.columns) == ["animal", "date", "epoch"]
    assert sorted(analysed_epochs(include_pending=True).index) == sorted(
        ANALYSED + PENDING
    )


def test_analysed_epochs_raise_on_an_unregistered_epoch(small_register, monkeypatch):
    _listing(monkeypatch, [*ANALYSED, ("testrat20990101_.nwb", "02_r1")])

    with pytest.raises(EpochRegisterError, match="testrat20990101_.nwb 02_r1"):
        analysed_epochs()


def test_analysed_epochs_raise_when_an_analysed_epoch_is_not_listed(
    small_register, monkeypatch
):
    _listing(monkeypatch, ANALYSED[1:])

    with pytest.raises(EpochRegisterError, match="1 analysed epoch"):
        analysed_epochs()


def test_pending_epochs_are_exactly_the_pending_ones(small_register, monkeypatch):
    _listing(monkeypatch, [*ANALYSED, *PENDING])

    assert list(pending_epochs().index) == PENDING
    assert list(epochs_to_run(pending_only=True).index) == PENDING
    assert list(epochs_to_run().index) == ANALYSED


def test_pending_epochs_must_be_listed_by_spyglass(small_register, monkeypatch):
    """A pending epoch whose populate has not happened is not listed."""
    _listing(monkeypatch, [*ANALYSED, PENDING[0]])

    with pytest.raises(EpochRegisterError, match="1 pending epoch"):
        pending_epochs()


def test_pending_epochs_refuse_a_register_with_none(tmp_path, monkeypatch):
    rows = [row for row in ROWS if row[2] != "pending"]
    path = _write_register(tmp_path / "epoch_register.csv", rows)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)
    _listing(monkeypatch, ANALYSED)

    with pytest.raises(EpochRegisterError, match="no pending epoch"):
        pending_epochs()


# --------------------------------------------------------------------------
# analysed_decode_files, the register's replacement for globbing a folder
# --------------------------------------------------------------------------


def test_analysed_epoch_ids_name_the_decodes(small_register):
    assert analysed_epoch_ids() == ["a20200101_02_r1_HPC", "a20200101_04_r2_HPC"]


def test_a_decode_folder_yields_only_analysed_epochs_files(small_register, tmp_path):
    folder = tmp_path / "decodes"
    folder.mkdir()
    for name in [
        "a20200101_02_r1_HPC_80_data_over_time.pkl",  # analysed
        "a20200101_06_r3_HPC_80_data_over_time.pkl",  # pending
        "b20200102_04_r2_HPC_80_data_over_time.pkl",  # excluded
        "z20990101_02_r1_HPC_80_data_over_time.pkl",  # unregistered
    ]:
        (folder / name).touch()

    found = analysed_decode_files(folder, "_80_data_over_time.pkl")

    # a20200101 04_r2 is analysed but has no file: left out, not raised
    assert found == [folder / "a20200101_02_r1_HPC_80_data_over_time.pkl"]
    with pytest.raises(FileNotFoundError, match="a20200101_04_r2_HPC_80"):
        analysed_decode_files(folder, "_80_data_over_time.pkl", require_all=True)


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
