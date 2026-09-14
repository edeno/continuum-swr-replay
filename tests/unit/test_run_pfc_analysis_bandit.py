"""Command-line and artifact-naming behaviour of the PFC GLM runner.

The runner writes into the shared ``Processed-Data`` tree under names every
downstream reader hard-codes, so the population a run was fit on must be visible
in the artifact name and in its provenance sidecar -- never inferred.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_pfc_analysis_bandit",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "run_pfc_analysis_bandit.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_stop_cohort_defaults_to_the_canonical_at_well_population(monkeypatch):
    monkeypatch.setattr(runner.sys, "argv", ["run_pfc_analysis_bandit.py"])

    args = runner.get_command_line_arguments()

    assert args.stop_cohort == "at_well"
    assert args.output_dir == runner.PROCESSED_DATA_DIR
    assert args.max_sessions is None


def test_an_unoffered_cohort_is_rejected(monkeypatch):
    monkeypatch.setattr(
        runner.sys,
        "argv",
        ["run_pfc_analysis_bandit.py", "--stop-cohort", "in_transit"],
    )

    with pytest.raises(SystemExit) as excinfo:
        runner.get_command_line_arguments()

    assert excinfo.value.code == 2


def test_only_the_sensitivity_bundle_is_suffixed(tmp_path):
    canonical = runner.modulation_path(tmp_path, "wilbur20210329_02_r1_mPFC", "at_well")
    sensitivity = runner.modulation_path(
        tmp_path, "wilbur20210329_02_r1_mPFC", "all_stops"
    )

    assert canonical.name == "wilbur20210329_02_r1_mPFC_modulation.pkl"
    assert sensitivity.name == "wilbur20210329_02_r1_mPFC_modulation_allstops.pkl"


#: Hashes of the two canonical event inputs a test bundle claims to be cut
#: from. Named so a call site reads as "the current inputs".
EVENT_TABLE_SHA = "event111"
DOT_SHA = "dot222"


def _write_bundle_files(
    directory,
    epoch_identifier,
    stop_cohort,
    sidecar_cohort,
    sidecar_hash="abc123",
    is_swr_definition=runner.IS_SWR_DEFINITION,
    event_table_sha=EVENT_TABLE_SHA,
    data_over_time_sha=DOT_SHA,
):
    """Touch one bundle's six files, optionally with a provenance sidecar."""
    artifact = runner.modulation_path(directory, epoch_identifier, stop_cohort)
    base = artifact.with_suffix("")
    for suffix in (
        ".pkl",
        ".SWR_modulation.npy",
        ".no_SWR_modulation.npy",
        ".non_local_SWR_modulation.npy",
        ".non_local_no_SWR_modulation.npy",
        ".matched_control_modulation.npy",
    ):
        base.with_suffix(suffix).touch()
    if sidecar_cohort is not None:
        artifact.with_suffix(".provenance.json").write_text(
            json.dumps(
                {
                    "stop_cohort": sidecar_cohort,
                    "stop_table_sha256": sidecar_hash,
                    "is_swr_definition": is_swr_definition,
                    "canonical_event_table_sha256": event_table_sha,
                    "data_over_time_sha256": data_over_time_sha,
                }
            )
        )


def test_results_exist_does_not_confuse_the_two_cohorts(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "at_well")

    assert runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )
    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "all_stops", None, EVENT_TABLE_SHA, DOT_SHA
    )
    assert not runner.results_exist(
        tmp_path, epoch_identifier, True, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_cut_from_another_stop_table_is_refit(tmp_path):
    # Regenerating the stop table changes which bins and events the cohort
    # holds, so a bundle cut from the previous generation is not this run's
    # bundle even though it names the same cohort.
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", sidecar_hash="stale999"
    )

    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_that_does_not_state_its_population_is_refit(tmp_path):
    # The canonical name is unsuffixed, so a bundle written before the cohort
    # existed sits under exactly the name a canonical run wants. Skipping on the
    # name alone would keep an all-bins fit as if it were the at-well one.
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", sidecar_cohort=None)

    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_naming_another_population_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "all_stops")

    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_an_allstops_bundle_matches_when_no_stop_table_was_read(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "all_stops", "all_stops", sidecar_hash=None
    )

    assert runner.results_exist(
        tmp_path, epoch_identifier, False, "all_stops", None, EVENT_TABLE_SHA, DOT_SHA
    )


@pytest.fixture
def fake_session(monkeypatch):
    """Replace the per-session fit and the cohort load with recorded stubs."""
    calls: dict[str, object] = {}

    def _fake_cohort(nwb_file_name, epoch_name, *, cohort="at_well", **kwargs):
        calls["cohort"] = cohort
        stops = pd.DataFrame(
            {"stop_start": [10.0], "stop_end": [11.0], "occupancy_state": ["at_well"]}
        )
        report = {
            "cohort": cohort,
            "rule": "test rule",
            "n_stops_before": 3,
            "n_stops_after": 1,
            "stop_table_sha256": "deadbeef",
        }
        return stops, report

    def _fake_run_session(nwb_file_name, epoch_name, **kwargs):
        calls["run_session"] = kwargs
        neuron_stats = pd.DataFrame({"immobility_avg_firing_rate": [3.0, 4.0]})
        modulation = np.zeros((201, 2))
        return (
            neuron_stats,
            modulation,
            modulation,
            modulation,
            modulation,
            modulation,
            {
                "stop_cohort": kwargs["stop_cohort"],
                "n_bins_immobile": 1000,
                "n_bins_fit": 400,
                "n_nonlocal_before": 20,
                "n_nonlocal_after": 8,
                "n_ripple_before": 30,
                "n_ripple_after": 11,
                "n_nonlocal_dropped_overlapping_fit": 3,
                "n_ripple_dropped_overlapping_fit": 5,
                "n_nonlocal_swr": 5,
                "n_nonlocal_no_swr": 3,
                "n_events_outside_frame": 0,
                "matched_control_seed": 12345,
                "n_matched_control": 7,
                "n_matched_control_unplaced": 1,
            },
        )

    def _fake_event_inputs(nwb_file_name, epoch_name, **kwargs):
        calls["event_inputs"] = (nwb_file_name, epoch_name)
        return {
            "is_swr_definition": runner.IS_SWR_DEFINITION,
            "immobility_inequality": "<",
            "canonical_event_table_path": "/canonical/event_table.parquet",
            "canonical_event_table_sha256": EVENT_TABLE_SHA,
            "swr_interval_source": "data_over_time.SWR_event",
            "data_over_time_path": "/canonical/dot.pkl",
            "data_over_time_sha256": DOT_SHA,
        }

    monkeypatch.setattr(runner, "load_session_stop_cohort", _fake_cohort)
    monkeypatch.setattr(runner, "run_session", _fake_run_session)
    monkeypatch.setattr(runner, "canonical_event_input_hashes", _fake_event_inputs)
    return calls


def test_the_sidecar_states_the_population_the_fit_was_read_on(
    tmp_path, fake_session, monkeypatch
):
    monkeypatch.setattr(runner.sys, "argv", ["run_pfc_analysis_bandit.py"])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    assert artifact.exists(), "the fit wrote no artifact (run_analysis swallowed it)"
    sidecar = artifact.with_suffix(".provenance.json")
    record = json.loads(sidecar.read_text())
    assert record["stop_cohort"] == "at_well"
    assert record["stop_table_sha256"] == "deadbeef"
    assert record["cohort_counts"]["n_bins_fit"] == 400
    assert record["cohort_counts"]["n_nonlocal_after"] == 8
    # The leading-edge coverage diagnostic travels with the bundle.
    assert record["cohort_counts"]["n_nonlocal_dropped_overlapping_fit"] == 3
    assert record["cohort_counts"]["n_ripple_dropped_overlapping_fit"] == 5
    assert record["stop_cohort_report"]["n_stops_after"] == 1
    assert fake_session["cohort"] == "at_well"


def test_the_sidecar_states_the_event_definition_the_fit_was_read_on(
    tmp_path, fake_session, monkeypatch
):
    # A bundle that does not name its event definition cannot be told from one
    # fit on the retired run-time construction, which split events on a
    # separately detected ripple population.
    monkeypatch.setattr(runner.sys, "argv", ["run_pfc_analysis_bandit.py"])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["is_swr_definition"] == runner.IS_SWR_DEFINITION
    assert record["immobility_inequality"] == "<"
    assert record["canonical_event_table_path"] == "/canonical/event_table.parquet"
    assert record["canonical_event_table_sha256"] == EVENT_TABLE_SHA
    assert record["swr_interval_source"] == "data_over_time.SWR_event"
    assert record["data_over_time_path"] == "/canonical/dot.pkl"
    assert record["data_over_time_sha256"] == DOT_SHA
    # The cohort-restricted split, so a reader need not recompute it.
    assert record["n_events_total"] == 8
    assert record["n_events_swr"] == 5
    assert record["n_events_non_swr"] == 3
    # The matched non-event control the engagement test is read against: how
    # it was drawn, how many controls were placed, and how many events could
    # not be given one.
    assert record["matched_control_seed"] == 12345
    assert record["n_matched_control"] == 7
    assert record["n_matched_control_unplaced"] == 1


def test_a_bundle_fit_on_the_retired_event_definition_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path,
        epoch_identifier,
        "at_well",
        "at_well",
        is_swr_definition="ripple_times_overlap",
    )

    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_cut_from_another_event_table_or_decode_frame_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", event_table_sha="stale111"
    )
    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )

    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", data_over_time_sha="stale222"
    )
    assert not runner.results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_all_stops_fits_without_a_cohort_and_writes_the_suffixed_bundle(
    tmp_path, fake_session, monkeypatch
):
    monkeypatch.setattr(runner.sys, "argv", ["run_pfc_analysis_bandit.py"])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="all_stops",
        output_dir=output_dir,
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation_allstops.pkl"
    assert artifact.exists()
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["stop_cohort"] == "all_stops"
    # No stop table is read at all, so none is hashed as an input.
    assert record["inputs"] == {}
    assert record["stop_table_sha256"] is None
    # The fit is handed no stops -- "all_stops" is the absence of the
    # restriction, not a cohort that happens to keep everything.
    assert "cohort" not in fake_session
    assert fake_session["run_session"]["cohort_stops"] is None
