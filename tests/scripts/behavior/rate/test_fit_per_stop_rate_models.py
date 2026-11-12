"""Command-line behaviour of the per-stop rate runner.

The runner writes ``models_per_stop_rate/all_models.pkl`` in place, so an
unrecognised flag must abort instead of silently refitting the canonical
artifact. These tests exercise the parser only: the fitting entry point is
replaced so that a run which reaches it is detectable as a failure.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from continuum_swr_replay.analysis.events.swr_source import SwrSourceError

_SPEC = importlib.util.spec_from_file_location(
    "fit_per_stop_rate_models",
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "behavior"
    / "rate"
    / "fit_per_stop_rate_models.py",
)
fpsr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fpsr)


@pytest.fixture()
def no_fitting(monkeypatch):
    """Replace the fitting entry point with a call recorder."""
    calls: list[tuple] = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("fit_per_stop_rate_hierarchy was called")

    monkeypatch.setattr(fpsr, "fit_per_stop_rate_hierarchy", _record)
    return calls


def test_help_exits_zero_without_fitting(capsys, no_fitting):
    with pytest.raises(SystemExit) as excinfo:
        fpsr.main(["--help"])

    assert excinfo.value.code == 0
    assert no_fitting == []
    help_text = capsys.readouterr().out
    assert "--max-per-animal" in help_text
    assert "--no-cv" in help_text
    assert "--stop-cohort" in help_text


def test_unknown_flag_exits_two_without_fitting(capsys, no_fitting):
    with pytest.raises(SystemExit) as excinfo:
        fpsr.main(["--not-a-flag"])

    assert excinfo.value.code == 2
    assert no_fitting == []
    assert "--not-a-flag" in capsys.readouterr().err


def test_unknown_stop_cohort_exits_two_without_fitting(capsys, no_fitting):
    """A cohort name the rule does not define must not reach the fitter."""

    with pytest.raises(SystemExit) as excinfo:
        fpsr.main(["--stop-cohort", "at_the_well"])

    assert excinfo.value.code == 2
    assert no_fitting == []
    assert "at_the_well" in capsys.readouterr().err


def _cohort_stop_table() -> pd.DataFrame:
    """Six stops of one session spanning the three occupancy states."""
    return pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "session_date": [1] * 6,
            "epoch": ["02_r1"] * 6,
            "stop_id": [f"s{i}" for i in range(6)],
            "occupancy_state": [
                "at_well",
                "at_well",
                "in_transit",
                "in_transit",
                "mixed",
                "at_well",
            ],
            "immobility_duration_s": [1.0] * 6,
            **{col: [0.0] * 6 for col in fpsr.RESPONSE_COLS.values()},
            **{col: [0.0] * 6 for col in fpsr.REWARD_HISTORY_COLUMNS},
            fpsr.REWARD_HISTORY_INCOMPLETE: [0.0] * 6,
        }
    )


@pytest.fixture()
def stubbed_run(monkeypatch, replay_content_root):
    """Run ``main`` without fitting: record what the fitter was handed."""
    seen: dict = {}
    stop = _cohort_stop_table()

    def _fit(stop_table, **kwargs):
        seen["stop_ids"] = list(stop_table["stop_id"])
        rate = pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "outcome": ["non_local"],
                "model": ["R0_offset"],
            }
        )
        return rate, []

    # stamped by the configured SWR source, as the shipped table is
    monkeypatch.setattr(fpsr, "read_table", lambda _path: stamp_swr_source(stop))
    monkeypatch.setattr(fpsr, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(fpsr, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(fpsr, "fit_per_stop_rate_hierarchy", _fit)
    return seen, replay_content_root


def test_default_cohort_is_at_well_and_publishes_the_canonical_names(stubbed_run):
    seen, tmp_path = stubbed_run

    fpsr.main([])

    # Only the well-anchored stops reached the fitter -- the 'mixed' stop s4 is
    # one of them (an at-well bout whose first samples read as in-transit).
    assert seen["stop_ids"] == ["s0", "s1", "s4", "s5"]
    out = tmp_path / "models_per_stop_rate" / "all_models.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["at_well"]
    provenance = json.loads((out.parent / "all_models.provenance.json").read_text())
    assert provenance["stop_cohort"] == "at_well"
    assert (out.parent / "all_models_cohort_audit.csv").exists()


def test_a_stop_table_from_another_swr_source_is_refused(monkeypatch, stubbed_run):
    """Its SWR counts would be another source's; nothing is fit or written."""
    seen, tmp_path = stubbed_run
    stop = _cohort_stop_table()
    stop.attrs["swr_source"] = "kay@another_run+artifacts@none"
    monkeypatch.setattr(fpsr, "read_table", lambda _path: stop)

    with pytest.raises(SwrSourceError, match="another_run"):
        fpsr.main([])

    assert seen == {}
    assert not (tmp_path / "models_per_stop_rate").exists()


def test_all_stops_cohort_publishes_the_suffixed_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    fpsr.main(["--stop-cohort", "all_stops", "--no-cv"])

    assert seen["stop_ids"] == [f"s{i}" for i in range(6)]
    out = tmp_path / "models_per_stop_rate" / "all_models_allstops.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["all_stops"]
    # The canonical name is never written by a sensitivity run.
    assert not (tmp_path / "models_per_stop_rate" / "all_models.pkl").exists()


def test_in_transit_cohort_publishes_its_own_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    fpsr.main(["--stop-cohort", "in_transit", "--no-cv"])

    assert seen["stop_ids"] == ["s2", "s3"]
    assert (tmp_path / "models_per_stop_rate" / "all_models_intransit.pkl").exists()


@pytest.mark.parametrize(
    ("converged", "n_flagged"), [(True, 0), (False, 1), (None, 1), ("absent", 1)]
)
def test_a_fit_without_a_true_convergence_flag_counts_as_not_converged(
    monkeypatch, converged, n_flagged
):
    """The cohort audit counts a fit as converged only when its flag is True:
    a flag that is missing or None has not shown convergence."""

    def _fit(cohort, predictors, *, response_col):
        fit = {"n_stops": len(cohort), "converged": converged, "separated": False}
        if converged == "absent":
            del fit["converged"]
        return fit

    monkeypatch.setattr(fpsr, "fit_per_stop_rate_model", _fit)

    _rows, records = fpsr.fit_per_stop_rate_hierarchy(
        _cohort_stop_table(), run_cv=False
    )

    assert len(records) == len(fpsr.MODEL_PER_STOP_RATE) * len(fpsr.RESPONSE_COLS)
    assert {r["n_sessions_flagged_nonconverged"] for r in records} == {n_flagged}
