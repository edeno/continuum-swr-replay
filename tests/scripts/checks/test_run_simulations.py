"""The simulation runner: its table, the record it writes with it, and its
choice of scenarios."""

from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.simulation import CAUSES
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


def test_a_config_file_is_passed_to_the_list(tmp_path) -> None:
    """``--config`` passes its JSON to the module's ``simulation`` and names
    the table after the file: the visit rate list's quick configuration of
    the No SWR kind, one scenario."""
    config = tmp_path / "no_swr_quick.json"
    config.write_text(json.dumps({"kind": "No SWR"}))
    run_simulations.main(
        [
            "--model",
            "visit_rate",
            "--quick",
            "--scenarios",
            "no_effect",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path),
        ]
    )
    table = read_table(tmp_path / "visit_rate_no_swr_quick.parquet")
    assert table.attrs["config"]["kind"] == "No SWR"
    assert table.attrs["scenarios"] == ["no_effect"]
    assert set(table["kind"]) == {"No SWR"}
    assert set(table["model"]) == {"session_sd2", "animal"}


def test_a_config_for_a_list_without_configurations_is_refused(monkeypatch) -> None:
    """``--config`` for a module whose ``simulation`` takes no ``config`` is
    refused with a message that says so, before the list is built."""
    module = types.ModuleType("no_config_list")
    module.simulation = lambda quick, workers: pytest.fail("built the list")
    monkeypatch.setitem(sys.modules, "no_config_list", module)
    monkeypatch.setitem(run_simulations.MODELS, "no_config", "no_config_list")
    with pytest.raises(ValueError, match="takes no configuration"):
        run_simulations.run("no_config", quick=True, config={"n_data_sets": 2})


def test_the_declared_penalty_list_reads_a_config_mapping(monkeypatch) -> None:
    """The declared-penalty list takes ``--config``'s JSON object as its
    ``Config``."""
    seen = []

    def cohort(config):
        seen.append(config)
        raise RuntimeError("stop")

    monkeypatch.setattr(sim, "cohort", cohort)
    with pytest.raises(RuntimeError, match="stop"):
        sim.simulation(True, 1, {"n_data_sets": 3})
    assert seen == [sim.Config(n_data_sets=3)]
def _summary_row(**values) -> dict:
    """One summary row in which every fit converged and every unit and
    session has a value, with ``values`` set."""
    return {
        "scenario": "s",
        "model": "m",
        "estimand": "e",
        "share_without_value": 0.0,
        **{f"without_value_{cause}": 0.0 for cause in CAUSES},
        "share_not_converged": 0.0,
        "n_sessions": 15,
        "n_sessions_used_min": 15,
        "share_sessions_unused": 0.0,
        **values,
    }


def test_a_session_unused_by_a_day_fit_is_warned() -> None:
    """A run in which every unit of the fit has a value but an estimate left
    out one of the 15 sessions (a day fit whose day kept a value without it)
    is warned; a run that used every session is not."""
    lost = _summary_row(n_sessions_used_min=14, share_sessions_unused=1 / 15)
    lines = run_simulations.warnings(pd.DataFrame([_summary_row(), lost]))
    assert lines == [
        "warning: scenario s, model m, estimand e: an estimate used as few as 14 "
        "of the 15 sessions (6.7 % of the sessions unused, mean over the data sets)"
    ]
