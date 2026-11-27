"""Tests for the per-bin rate runner's command line: which stop cohort it fits
and which files it publishes. The fit loop is tested with the package
(``tests/analysis/replay_models/rate/test_per_bin_rate_ladder.py``)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    stamp_swr_source,
)

_SPEC = importlib.util.spec_from_file_location(
    "fit_per_bin_rate_models",
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "replay_models"
    / "rate"
    / "fit_per_bin_rate_models.py",
)
fprm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fprm)


def _cohort_stop_table() -> pd.DataFrame:
    """Six stops of one session spanning the occupancy states."""
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
        }
    )


@pytest.fixture()
def stubbed_run(monkeypatch, replay_content_root):
    """Run ``main`` without fitting: record what the fitter was handed."""
    seen: dict = {}

    stop = _cohort_stop_table()

    # stamped by the configured SWR source, as the shipped tables are
    stamped_stop = stamp_swr_source(stop.copy())
    stamped_other = stamp_swr_source(pd.DataFrame())

    def _read_table(path):
        return stamped_stop if "stop_table" in str(path) else stamped_other

    def _fit(stop_table, onsets, **kwargs):
        seen["stop_ids"] = list(stop_table["stop_id"])
        out = pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "outcome": ["non_local"],
                "model": ["M0_const"],
                "n_stops": [3],
            }
        )
        out.attrs["cohort_stop_drops"] = {}
        return out

    monkeypatch.setattr(fprm, "read_table", _read_table)
    monkeypatch.setattr(fprm, "require_covariate_schema", lambda *a, **k: None)
    monkeypatch.setattr(fprm, "record_inputs", lambda inputs: {})
    monkeypatch.setattr(fprm, "attach_is_swr_to_onsets", lambda o, e: pd.DataFrame())
    monkeypatch.setattr(fprm, "attach_event_ends", lambda o, e: o)
    monkeypatch.setattr(fprm, "fit_per_bin_rate_models", _fit)
    monkeypatch.setattr(fprm, "per_bin_cohort_records", lambda *a, **k: [])
    monkeypatch.setattr(
        fprm, "write_cohort_audit", lambda records, path: seen.setdefault("audit", path)
    )
    return seen, replay_content_root


def test_default_cohort_is_at_well_and_publishes_the_canonical_names(stubbed_run):
    seen, tmp_path = stubbed_run

    fprm.main([])

    # Only the well-anchored stops reached the fitter -- the 'mixed' stop s4 is
    # one of them (an at-well bout whose first samples read as in-transit).
    assert seen["stop_ids"] == ["s0", "s1", "s4", "s5"]
    out = tmp_path / "models_per_bin_rate" / "all_models.pkl"
    assert out.exists()
    assert seen["audit"].name == "per_bin_rate_cohort_audit.csv"
    models = pd.read_pickle(out)
    assert list(models["stop_cohort"]) == ["at_well"]
    provenance = json.loads((out.parent / "all_models.provenance.json").read_text())
    assert provenance["stop_cohort"] == "at_well"


def test_all_stops_cohort_publishes_the_suffixed_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    fprm.main(["--stop-cohort", "all_stops", "--max-per-animal", "1"])

    # No occupancy restriction: every stop is fit.
    assert seen["stop_ids"] == [f"s{i}" for i in range(6)]
    out = tmp_path / "models_per_bin_rate" / "all_models_allstops_smoke.pkl"
    assert out.exists()
    # The canonical name is never written by a sensitivity run.
    assert not (tmp_path / "models_per_bin_rate" / "all_models.pkl").exists()
    assert seen["audit"].name == "per_bin_rate_cohort_audit_allstops_smoke.csv"
    assert list(pd.read_pickle(out)["stop_cohort"]) == ["all_stops"]


def test_in_transit_cohort_publishes_its_own_bundle(stubbed_run):
    seen, tmp_path = stubbed_run

    fprm.main(["--stop-cohort", "in_transit"])

    assert seen["stop_ids"] == ["s2", "s3"]
    assert (tmp_path / "models_per_bin_rate" / "all_models_intransit.pkl").exists()


def test_unknown_cohort_is_refused_by_the_parser(capsys):
    with pytest.raises(SystemExit) as excinfo:
        fprm.main(["--stop-cohort", "at_the_well"])

    assert excinfo.value.code == 2
    assert "at_the_well" in capsys.readouterr().err


def test_tables_from_another_swr_source_are_refused(monkeypatch, stubbed_run):
    """The bins are dropped by the configured source's excluded time, so tables
    made under another source (another artifact run, say) would mix two masks."""
    from continuum_swr_replay import paths
    from continuum_swr_replay.analysis.event_definition import swr_source

    monkeypatch.setattr(paths, "ARTIFACT_INTERVALS_RUN", "another_run")

    with pytest.raises(swr_source.SwrSourceError, match="another_run"):
        fprm.main([])
