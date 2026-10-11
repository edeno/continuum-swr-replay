"""The table scripts' ``--expected-sessions`` default is the register's analysed count.

A literal default (218 until 2026-10-10) went stale whenever the register's
cohort changed. Each script now reads the number of analysed epochs from the
epoch register when it builds its parser. Here the register is a stand-in under
``tmp_path`` and each script's work is replaced by a recorder, so nothing is read
or written but the stand-in.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.data_loaders import epoch_register

REPO = Path(__file__).resolve().parents[2]

SCRIPTS: dict[str, tuple[str, list[str]]] = {
    "assemble_replay_content_shards": (
        "assemble_replay_content_shards",
        ["shard", "--output-dir", "out", "--contingency-dir", "contingency"],
    ),
    "assemble_contfrag_shards": (
        "assemble_contfrag_shards",
        [
            "shard",
            "--output-dir",
            "out",
            "--event-source",
            "SWR",
            "--brain-area",
            "HPC",
        ],
    ),
    "refresh_contfrag_behavioral_covariates": (
        "refresh_contfrag_behavioral_covariates",
        [
            "--source-dir",
            "source",
            "--output-dir",
            "out",
            "--replay-content-dir",
            "replay",
            "--event-source",
            "SWR",
            "--brain-area",
            "HPC",
        ],
    ),
    "write_regeneration_manifest": (
        "write_manifest",
        ["--replay-root", "replay", "--contfrag-root", "contfrag", "--output", "out"],
    ),
}
"""Script (under ``scripts/decoding``) -> the function its ``main`` hands
``expected_sessions`` to, and the arguments it requires."""


class _Stop(Exception):
    pass


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_the_default_expected_sessions_is_the_registers_analysed_count(
    script, tmp_path, monkeypatch
):
    rows = [
        ["a20200101_.nwb", epoch, "analysed", "", ""] for epoch in ("02_r1", "04_r2")
    ]
    rows += [
        ["a20200101_.nwb", "06_r3", "pending", "PENDING_POPULATE", "no PIM row"],
        ["a20200101_.nwb", "08_r4", "excluded", "SUSTAINED_NOISE", "an S block"],
    ]
    register = tmp_path / "epoch_register.csv"
    pd.DataFrame(rows, columns=list(epoch_register.COLUMNS)).to_csv(
        register, index=False
    )
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", register)
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "decoding"))
    module = importlib.import_module(f"scripts.decoding.{script}")
    function, argv = SCRIPTS[script]
    seen: dict[str, object] = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(module, function, record)
    monkeypatch.setattr(sys, "argv", [script, *argv])

    with pytest.raises(_Stop):
        module.main()

    assert seen["expected_sessions"] == 2


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_the_help_says_the_default_is_the_hpc_cohort(script, monkeypatch, capsys):
    """For ``--brain-area mPFC`` or ``OFC`` the default is not the area's count."""
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "decoding"))
    module = importlib.import_module(f"scripts.decoding.{script}")
    monkeypatch.setattr(sys, "argv", [script, "--help"])

    with pytest.raises(SystemExit):
        module.main()

    assert "analysed epochs (HPC cohort)" in " ".join(capsys.readouterr().out.split())
