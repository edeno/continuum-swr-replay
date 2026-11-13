"""The file names of a PFC modulation bundle, built and read back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum_swr_replay.analysis.pfc.bundle_files import (
    MPFC_STEM,
    bundle_identifiers,
    bundle_path,
    bundle_sessions,
    read_sidecars,
    session_bundle_path,
)

FOLDER = Path("/bundles")


def test_a_bundle_file_is_the_identifier_modulation_suffix_and_extension():
    assert bundle_path(FOLDER, "wilbur20210329_02_r1_mPFC") == (
        FOLDER / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    )
    assert bundle_path(
        FOLDER, "wilbur20210329_02_r1_OFC", "_allstops", ".SWR_modulation.npy"
    ) == (FOLDER / "wilbur20210329_02_r1_OFC_modulation_allstops.SWR_modulation.npy")
    assert session_bundle_path(
        FOLDER, "j1620210706_04_r2", "_sigma1", ".provenance.json"
    ) == (FOLDER / "j1620210706_04_r2_mPFC_modulation_sigma1.provenance.json")


def test_the_readers_list_one_suffix_and_area_and_read_the_names_back(tmp_path):
    names = [
        "j1620210706_10_r5_mPFC_modulation.pkl",
        "j1620210706_02_r1_mPFC_modulation.pkl",
        "j1620210706_02_r1_mPFC_modulation.provenance.json",
        "j1620210706_02_r1_mPFC_modulation_allstops.pkl",
        "j1620210706_02_r1_OFC_modulation.pkl",
        "j1620210706_02_r1_mPFC_modulation.SWR_modulation.npy",
    ]
    for name in names:
        (tmp_path / name).touch()

    sessions = bundle_sessions(tmp_path)
    assert list(sessions) == ["j1620210706_02_r1", "j1620210706_10_r5"]
    for session, path in sessions.items():
        assert session_bundle_path(tmp_path, session) == path

    assert list(bundle_sessions(tmp_path, "_allstops")) == ["j1620210706_02_r1"]
    assert list(bundle_sessions(tmp_path, extension=".provenance.json")) == [
        "j1620210706_02_r1"
    ]
    assert list(bundle_identifiers(tmp_path)) == [
        "j1620210706_02_r1_OFC",
        "j1620210706_02_r1_mPFC",
        "j1620210706_10_r5_mPFC",
    ]
    assert bundle_sessions(tmp_path / "absent") == {}


def test_read_sidecars_reads_the_canonical_mpfc_sidecars_in_session_order(tmp_path):
    assert MPFC_STEM == "_mPFC_modulation"
    for session in ("j1620210706_10_r5", "j1620210706_02_r1"):
        (tmp_path / f"{session}{MPFC_STEM}.pkl").touch()
        (tmp_path / f"{session}{MPFC_STEM}.provenance.json").write_text(
            json.dumps({"session": session})
        )
    # Neither a sensitivity bundle nor another area's is canonical mPFC.
    (tmp_path / f"j1620210706_02_r1{MPFC_STEM}_allstops.provenance.json").touch()
    (tmp_path / "j1620210706_02_r1_OFC_modulation.provenance.json").touch()

    assert read_sidecars(tmp_path) == [
        {"session": "j1620210706_02_r1"},
        {"session": "j1620210706_10_r5"},
    ]


def test_read_sidecars_refuses_no_bundle_or_an_unpaired_one(tmp_path):
    with pytest.raises(ValueError, match=r"differ: \[\]"):
        read_sidecars(tmp_path)
    (tmp_path / f"j1620210706_02_r1{MPFC_STEM}.pkl").touch()
    with pytest.raises(ValueError, match=r"differ: \['j1620210706_02_r1'\]"):
        read_sidecars(tmp_path)
