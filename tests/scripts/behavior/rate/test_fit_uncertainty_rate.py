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
from continuum_swr_replay.data_loaders import epoch_register
from scripts.behavior.rate import fit_uncertainty_rate as mod

_STATES = ("at_well", "at_well", "in_transit", "in_transit", "mixed", "at_well")


def _register(path, rows: list[tuple[str, str, str, str, bool]]) -> None:
    """``REGISTER_PATH`` contents: (nwb_file_name, epoch, status, reason,
    first_exposure) per entry, with evidence wherever a reason needs it."""
    pd.DataFrame(
        [
            [name, epoch, status, reason, "x" if reason else "", str(flag)]
            for name, epoch, status, reason, flag in rows
        ],
        columns=list(epoch_register.COLUMNS),
    ).to_csv(path, index=False)


def test_the_register_s_first_exposure_sessions_are_dropped_and_no_other(
    tmp_path, monkeypatch
):
    """The register names the session, not its place in the stop table.

    ``a``'s exposure (20200101 02_r1) goes and its same-day 04_r2 stays; ``b``'s
    exposure is excluded from the register, so the stop table lacks it and
    ``b``'s first stop-table session (20200105) stays, as peanut's does.
    """
    path = tmp_path / "epoch_register.csv"
    _register(
        path,
        [
            ("a20200101_.nwb", "02_r1", "analysed", "", True),
            ("a20200101_.nwb", "04_r2", "analysed", "", False),
            ("a20200102_.nwb", "02_r1", "analysed", "", False),
            ("b20200104_.nwb", "02_r1", "excluded", "NO_TRIAL_INTERVALS", True),
            ("b20200105_.nwb", "02_r1", "analysed", "", False),
            ("b20200106_.nwb", "02_r1", "analysed", "", False),
        ],
    )
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)
    sessions = [
        ("a", 20200101, "02_r1"),
        ("a", 20200101, "04_r2"),
        ("a", 20200102, "02_r1"),
        ("b", 20200105, "02_r1"),
        ("b", 20200106, "02_r1"),
    ]
    stop = pd.DataFrame(
        [(*session, stop_id) for stop_id, session in enumerate(sessions * 2)],
        columns=["animal", "session_date", "epoch", "stop_id"],
    )

    out = mod._drop_exposure_sessions(stop)

    assert out["stop_id"].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert out.columns.equals(stop.columns)


def _cohort_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two sessions of one animal, each spanning the three occupancy states.

    The register marks session 1 as the animal's first exposure, and the runner
    drops it before fitting, so the same six-state pattern appears in both:
    whichever cohort is selected, session 1 goes and session 2 is the one fit.
    """
    rows = [
        {
            "animal": "a",
            "session_date": session_date,
            "epoch": "02_r1",
            "stop_id": f"{prefix}{i}",
            "occupancy_state": state,
        }
        for session_date, prefix in ((20200101, "p"), (20200102, "s"))
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
def stubbed_run(monkeypatch, replay_content_root, tmp_path):
    """Run ``main`` without fitting: record what the fitter was handed."""
    seen: dict = {}
    stops, qvals = _cohort_tables()
    register = tmp_path / "epoch_register.csv"
    _register(
        register,
        [
            ("a20200101_.nwb", "02_r1", "analysed", "", True),
            ("a20200102_.nwb", "02_r1", "analysed", "", False),
        ],
    )
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", register)

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
                "session_date": [20200102],
                "epoch": ["02_r1"],
                "outcome": ["swr"],
                "model": [mod.MU_UNCERTAINTY.name],
            }
        )
        models.attrs["post_outcome_gate_audit"] = []
        return models

    monkeypatch.setattr(mod, "read_table", _read_table)
    monkeypatch.setattr(mod, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(mod, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(mod, "add_uncertainty_covariates", lambda frame, **k: frame)
    monkeypatch.setattr(mod, "assert_qvals_aligned", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "attach_is_swr_to_onsets", lambda onsets, events: pd.DataFrame()
    )
    monkeypatch.setattr(mod, "attach_event_ends", lambda onsets, events: onsets)
    monkeypatch.setattr(mod, "fit_uncertainty_rate", _fit)

    def _records(models, stop_fitted, **counts):
        seen["n_sessions_postjoin"] = counts["n_sessions_postjoin"]
        return []

    monkeypatch.setattr(mod, "_cohort_records", _records)
    return seen, replay_content_root


def test_default_cohort_is_at_well_and_publishes_the_canonical_names(stubbed_run):
    seen, tmp_path = stubbed_run

    mod.main([])

    # Only the well-anchored stops of the non-exposure session reached the
    # fitter -- the 'mixed' stop s4 is one of them (an at-well bout whose first
    # settling samples read as in-transit).
    assert seen["stop_ids"] == ["s0", "s1", "s4", "s5"]
    # The audit's sessions_in counts the exposure session it then drops.
    assert seen["n_sessions_postjoin"] == 2
    out = tmp_path / "models_per_bin_rate" / "uncertainty_rate.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["at_well"]
    provenance = json.loads(
        (out.parent / "uncertainty_rate.provenance.json").read_text()
    )
    assert provenance["stop_cohort"] == "at_well"
    assert (out.parent / "uncertainty_rate_cohort_audit.csv").exists()


def test_all_stops_cohort_publishes_the_suffixed_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    mod.main(["--stop-cohort", "all_stops"])

    assert seen["stop_ids"] == [f"s{i}" for i in range(6)]
    out = tmp_path / "models_per_bin_rate" / "uncertainty_rate_allstops.pkl"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["all_stops"]
    # The canonical name is never written by a sensitivity run.
    assert not (tmp_path / "models_per_bin_rate" / "uncertainty_rate.pkl").exists()


def test_in_transit_cohort_publishes_its_own_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    mod.main(["--stop-cohort", "in_transit"])

    assert seen["stop_ids"] == ["s2", "s3"]
    assert (
        tmp_path / "models_per_bin_rate" / "uncertainty_rate_intransit.pkl"
    ).exists()


def test_unknown_cohort_is_refused_by_the_parser(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--stop-cohort", "at_the_well"])

    assert excinfo.value.code == 2
    assert "at_the_well" in capsys.readouterr().err
