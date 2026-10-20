"""The uncertainty-rate runner fits one stop cohort and says which.

The cohort restriction is the whole reason the rate and content families
describe the same population, and it is invisible in the fitted numbers, so it
is checked here directly: ``main`` is run with the fitter replaced by a
recorder, and the test reads back the ``stop_id``s that reached it, the
artifact suffix, the ``stop_cohort`` column on the rows, and the provenance
sidecar.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from scripts.rate import fit_uncertainty_rate as mod

_STATES = ("at_well", "at_well", "in_transit", "in_transit", "mixed", "at_well")


def _cohort_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two sessions of one animal, each spanning the three occupancy states.

    The runner drops each animal's first session before fitting, so the same
    six-state pattern appears in both: whichever cohort is selected, session 1
    is the exposure session that goes and session 2 is the one fit.
    """
    rows = [
        {
            "animal": "a",
            "session_date": session_date,
            "epoch": "02_r1",
            "stop_id": f"{prefix}{i}",
            "occupancy_state": state,
        }
        for session_date, prefix in ((1, "p"), (2, "s"))
        for i, state in enumerate(_STATES)
    ]
    stops = pd.DataFrame(rows)
    n = len(stops)
    qvals = pd.DataFrame(
        {
            "stop_id": stops["stop_id"],
            "outcome_known_time": [0.0] * n,
            "trial_reward": [0.0] * n,
            **{c: [0.0] * n for c in mod.VALUE_COVARIATES},
            "rpe_postoutcome": [0.0] * n,
            # distinct per stop, so every cohort's fitted rows have a finite
            # non-zero SD for the z-scoring the runner insists on
            **{
                c: [float(i + offset) for i in range(n)]
                for offset, c in enumerate(mod.RL_COVARIATES)
            },
        }
    )
    return stops, qvals


@pytest.fixture()
def stubbed_run(monkeypatch, tmp_path):
    """Run ``main`` without fitting: record what the fitter was handed."""
    seen: dict = {}
    stops, qvals = _cohort_tables()

    def _read_table(path):
        if "qvals" in str(path):
            return qvals
        # stamped by the configured SWR source, as the shipped tables are
        if "stop_table" in str(path):
            return stamp_swr_source(stops.copy())
        return stamp_swr_source(pd.DataFrame())

    def _fit(stop_table, onsets):
        seen["stop_ids"] = list(stop_table["stop_id"])
        models = pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [2],
                "epoch": ["02_r1"],
                "outcome": ["swr"],
                "model": [mod.MU_UNCERTAINTY.name],
            }
        )
        models.attrs["post_outcome_gate_audit"] = []
        return models

    monkeypatch.setattr(mod, "RC", tmp_path)
    monkeypatch.setattr(mod, "read_table", _read_table)
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(mod, "add_uncertainty_covariates", lambda frame, **k: frame)
    monkeypatch.setattr(mod, "assert_qvals_aligned", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "attach_is_swr_to_onsets", lambda onsets, events: pd.DataFrame()
    )
    monkeypatch.setattr(mod, "fit_uncertainty_rate", _fit)
    monkeypatch.setattr(mod, "_cohort_records", lambda *a, **k: [])
    return seen, tmp_path


def test_default_cohort_is_at_well_and_publishes_the_canonical_names(
    monkeypatch, stubbed_run
):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(mod.sys, "argv", ["fit_uncertainty_rate.py"])

    mod.main()

    # Only the well-anchored stops of the non-exposure session reached the
    # fitter -- the 'mixed' stop s4 is one of them (an at-well bout whose first
    # settling samples read as in-transit).
    assert seen["stop_ids"] == ["s0", "s1", "s4", "s5"]
    out = tmp_path / "models_per_bin_rate" / "uncertainty_rate.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["at_well"]
    provenance = json.loads(
        (out.parent / "uncertainty_rate.provenance.json").read_text()
    )
    assert provenance["stop_cohort"] == "at_well"
    assert (out.parent / "uncertainty_rate_cohort_audit.csv").exists()


def test_all_stops_cohort_publishes_the_suffixed_bundle(monkeypatch, stubbed_run):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(
        mod.sys, "argv", ["fit_uncertainty_rate.py", "--stop-cohort", "all_stops"]
    )

    mod.main()

    assert seen["stop_ids"] == [f"s{i}" for i in range(6)]
    out = tmp_path / "models_per_bin_rate" / "uncertainty_rate_allstops.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["all_stops"]
    # The canonical name is never written by a sensitivity run.
    assert not (tmp_path / "models_per_bin_rate" / "uncertainty_rate.pkl").exists()


def test_in_transit_cohort_publishes_its_own_bundle(monkeypatch, stubbed_run):
    seen, tmp_path = stubbed_run
    monkeypatch.setattr(
        mod.sys, "argv", ["fit_uncertainty_rate.py", "--stop-cohort", "in_transit"]
    )

    mod.main()

    assert seen["stop_ids"] == ["s2", "s3"]
    assert (
        tmp_path / "models_per_bin_rate" / "uncertainty_rate_intransit.pkl"
    ).exists()


def test_unknown_cohort_is_refused_by_the_parser(monkeypatch, capsys):
    monkeypatch.setattr(
        mod.sys, "argv", ["fit_uncertainty_rate.py", "--stop-cohort", "at_the_well"]
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 2
    assert "at_the_well" in capsys.readouterr().err
