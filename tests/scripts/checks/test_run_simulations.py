"""The simulation runner: its table, the record it writes with it, and its
choice of scenarios."""

from __future__ import annotations

import json

import pytest

from continuum_swr_replay.tables.io import read_table
from scripts.checks import run_simulations
from scripts.checks import simulate_declared_penalty as sim


def test_quick_run_writes_the_table_with_its_record(tmp_path, monkeypatch) -> None:
    """``--quick`` writes ``<model>.parquet`` with the seed, the data sets,
    the scenarios, the commit and each model's declared penalties in its
    ``attrs``; ``--reference`` writes the same table as JSON."""
    monkeypatch.setattr(sim, "TARGETS", (6.0,))
    reference = tmp_path / "reference.json"
    run_simulations.main(
        [
            "--model",
            "declared_penalty",
            "--quick",
            "--output-dir",
            str(tmp_path),
            "--reference",
            str(reference),
        ]
    )
    table = read_table(tmp_path / "declared_penalty.parquet")
    attrs = table.attrs
    assert attrs["model"] == "declared_penalty"
    assert attrs["seed"] == sim.DATA_SEED
    assert attrs["n_data_sets"] == 2 and attrs["n_draws"] == 200
    assert attrs["scenarios"] == ["rise", "flat"]
    assert isinstance(attrs["commit"], str) and attrs["commit"]
    log_penalty = attrs["calibration"]["target_edf_6"]["log_penalty"]
    assert attrs["log_penalties"] == {
        "target_edf_6": {sim.TERM: {"set": "fixed", "log_penalty": log_penalty}}
    }
    stored = json.loads(reference.read_text())
    assert stored["attrs"]["seed"] == sim.DATA_SEED
    assert [row["curve"] for row in stored["table"]] == ["rise", "flat"]
    assert stored["table"][0]["log_penalty"] == pytest.approx(log_penalty)


def test_one_scenario_keeps_its_data_sets(monkeypatch) -> None:
    """A run of the flat curve alone gives its rows of a run of both: the
    scenario keeps its place, and so its seeds."""
    monkeypatch.setattr(sim, "TARGETS", (6.0,))
    both = run_simulations.run("declared_penalty", quick=True)
    flat = run_simulations.run("declared_penalty", quick=True, scenarios=["flat"])
    assert flat.attrs["scenarios"] == ["flat"]
    assert flat.reset_index(drop=True).equals(
        both[both["curve"] == "flat"].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="not scenarios of the run"):
        run_simulations.run("declared_penalty", quick=True, scenarios=["none"])
