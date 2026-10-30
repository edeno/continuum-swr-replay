"""Part 4: values read from fits made on other inputs than the current ones are
pending, and a fit that does not cover the PFC cohort is refused."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from continuum_swr_replay.reported.part4 import (
    Part4Inputs,
    followup_values,
    marked_family_staleness,
    staleness,
    stop_context_staleness,
    table_hashes,
)

CLAIMS = ("20",)
SESSIONS = (("j1620210706_.nwb", "02_r1"), ("j1620210706_.nwb", "04_r2"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tables(tmp_path) -> tuple[Path, Path]:
    event, stop = tmp_path / "event_table.parquet", tmp_path / "stop_table.parquet"
    event.write_bytes(b"events")
    stop.write_bytes(b"stops")
    return event, stop


def test_bundle_staleness_names_the_count(tables):
    event, stop = tables
    fresh = {
        "canonical_event_table_sha256": _sha(event),
        "stop_table_sha256": _sha(stop),
    }
    sidecars = [dict(fresh), dict(fresh), dict(fresh)]
    assert staleness(sidecars, event, stop) is None
    sidecars[1]["stop_table_sha256"] = "old"
    note = staleness(sidecars, event, stop)
    assert "stop_table_sha256 differs in 1 bundles" in note
    assert "canonical_event_table_sha256" not in note


def _stop_context(root: Path, current: dict[str, str]) -> Path:
    """A stop-context output for two sessions, fit on the current inputs."""
    folder = root / "stop_context"
    (folder / "sessions").mkdir(parents=True)
    bundles, session_tables = {}, {}
    for nwb, epoch in SESSIONS:
        identifier = f"{nwb.removesuffix('_.nwb')}_{epoch}_mPFC"
        bundle = root / f"{identifier}_modulation.pkl"
        bundle.write_bytes(identifier.encode())
        table = folder / "sessions" / f"{identifier}.parquet"
        table.write_bytes(b"table " + identifier.encode())
        (folder / "sessions" / f"{identifier}.provenance.json").write_text(
            json.dumps({"input_provenance": dict(current)})
        )
        # recorded under another worktree's path: matched by file name
        bundles[f"/elsewhere/{bundle.name}"] = _sha(bundle)
        session_tables[f"/elsewhere/{table.name}"] = _sha(table)
    (folder / "results.provenance.json").write_text(
        json.dumps({"bundles": bundles, "session_tables": session_tables})
    )
    rows = [
        {
            "event_class": event_class,
            "sign": sign,
            "unit_set": unit_set,
            "estimator": "conditional",
            "term": term,
            "exp_coef": 1.08,
            "ci_low": 1.02,
            "ci_high": 1.14,
            "n_animals_pooled_sign": 1,
            "n_animals_estimated": 1,
            "loao_exp_coef_min": 1.02,
            "loao_exp_coef_max": 1.10,
            "n_sessions": len(SESSIONS),
            "n_animals": 1,
            "n_loao_estimated": 1,
        }
        for event_class, sign, unit_set, term in itertools.product(
            ("non_local_SWR", "non_local_no_SWR"),
            ("increase", "decrease"),
            ("called", "all"),
            ("rewarded", "next_trip_switches_patch"),
        )
    ]
    pd.DataFrame(rows).to_csv(folder / "results.csv", index=False)
    return folder / "results.csv"


def test_stop_context_staleness(tmp_path, tables):
    current = table_hashes(*tables)
    results = _stop_context(tmp_path, current)
    assert stop_context_staleness(results, tmp_path, current, 2) is None
    with pytest.raises(ValueError, match="not the 3 of the PFC cohort"):
        stop_context_staleness(results, tmp_path, current, 3)
    # one session assembled from another stop table
    provenance = results.parent / "sessions" / "j1620210706_04_r2_mPFC.provenance.json"
    provenance.write_text(
        json.dumps({"input_provenance": {**current, "stop_table_sha256": "old"}})
    )
    note = stop_context_staleness(results, tmp_path, current, 2)
    assert "stop_table_sha256 differs in 1 of 2" in note
    # a canonical bundle refit since
    (tmp_path / "j1620210706_02_r1_mPFC_modulation.pkl").write_bytes(b"refit")
    note = stop_context_staleness(results, tmp_path, current, 2)
    assert "bundle_sha256 differs in 1 of 2" in note


def _marked(root: Path, current: dict[str, str]) -> Path:
    folder = root / "marked_family_new_tree"
    folder.mkdir()
    for (nwb, epoch), model in itertools.product(SESSIONS, ("replay_content", "x")):
        stem = f"{nwb.removesuffix('_.nwb')}_{epoch}_mPFC_{model}_marked_family"
        (folder / f"{stem}.pkl").write_bytes(b"")
        (folder / f"{stem}.provenance.json").write_text(
            json.dumps({"nwb_file_name": nwb, "epoch_name": epoch, **current})
        )
    summary = folder / "summary.parquet"
    pd.DataFrame(
        [
            ("replay_content", "non_local_SWR", "current_arm_fraction", stat, value)
            for stat, value in (
                ("peak_value", -0.03),
                ("peak_lag_ms", 60.0),
                ("animals_same_sign", 5.0),
                ("n_animals", 5.0),
                ("frac_pos", 0.4),
                ("n_units", 10.0),
                ("n_usable", 10.0),
                ("n_excluded_blowup", 0.0),
                ("n_sessions", 2.0),
            )
        ],
        columns=["model", "event_class", "target", "statistic", "value"],
    ).to_parquet(summary)
    later = summary.stat().st_mtime + 10
    os.utime(summary, (later, later))
    return summary


def test_marked_family_staleness(tmp_path, tables):
    current = table_hashes(*tables)
    summary = _marked(tmp_path, current)
    assert marked_family_staleness(summary, current, 2) is None
    with pytest.raises(ValueError, match="2 sessions, not the 75"):
        marked_family_staleness(summary, current, 75)
    sidecar = next(summary.parent.glob("*_x_marked_family.provenance.json"))
    record = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**record, "canonical_event_table_sha256": "old"}))
    note = marked_family_staleness(summary, current, 2)
    assert "canonical_event_table_sha256 differs in 1 of 4" in note
    # a bundle rewritten after the summary
    bundle = next(summary.parent.glob("*_marked_family.pkl"))
    newer = summary.stat().st_mtime + 10
    os.utime(bundle, (newer, newer))
    note = marked_family_staleness(summary, current, 2)
    assert "bundle written after the summary differs in 1 of 4" in note


def test_the_pending_notes_reach_the_followup_values(tmp_path, tables):
    event, stop = tables
    current = table_hashes(event, stop)
    inputs = Part4Inputs(
        bundle_dir=tmp_path,
        pfc_cohort=tmp_path / "cohort.json",
        event_table=event,
        stop_table=stop,
        ml_tree=tmp_path,
        timing_calibration=tmp_path / "none.json",
        swr_without_replay_dir=tmp_path,
        marked_family_summary=_marked(tmp_path, current),
        stop_context_results=_stop_context(tmp_path, current),
    )
    values, records = followup_values(inputs, current, 2, CLAIMS)
    assert values and all(v.pending is None for v in values)
    assert len(records) == 2
    assert "--slope-permutations 0" in next(
        v.source for v in values if v.key.startswith("marked_")
    )
    stop.write_bytes(b"new stops")
    values, _ = followup_values(inputs, table_hashes(event, stop), 2, CLAIMS)
    assert all("stop_table_sha256 differs" in v.pending for v in values)
    marked = [v for v in values if v.key.startswith("marked_")]
    context = [v for v in values if v.key.startswith("stop_context_")]
    assert marked and context
    assert all("marked-family bundles" in v.pending for v in marked)
    assert all("stop-context fits" in v.pending for v in context)
    # a fit over fewer sessions than the cohort is refused
    with pytest.raises(ValueError, match="not the 3 of the PFC cohort"):
        followup_values(inputs, current, 3, CLAIMS)
