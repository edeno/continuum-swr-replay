"""Command-line and artifact-naming behaviour of the PFC GLM runner.

The runner writes into the shared ``Processed-Data`` tree under names every
downstream reader hard-codes, so the population a run was fit on must be visible
in the artifact name and in its provenance sidecar -- never inferred.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import fit_provenance
from continuum_swr_replay.tables.io import write_table

_SPEC = importlib.util.spec_from_file_location(
    "run_pfc_analysis_bandit",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "pfc"
    / "run_pfc_analysis_bandit.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_stop_cohort_defaults_to_the_canonical_at_well_population(
    processed_data_root,
):
    args = runner.get_command_line_arguments([])

    assert args.stop_cohort == "at_well"
    assert args.output_dir == processed_data_root
    assert args.max_sessions is None
    assert args.n_control_draws == runner.DEFAULT_CONTROL_DRAWS == 1_000


def test_an_unoffered_cohort_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        runner.get_command_line_arguments(["--stop-cohort", "in_transit"])

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

#: The SWR source a test bundle claims its SWR intervals came from, and the hash
#: of that source's events file.
SWR_SOURCE = "kay@testrun"
SWR_EVENTS_SHA = "swr333"


#: The estimator and inference a test bundle claims, and the values a
#: ``results_exist`` call in these tests asks for. The skip check compares them
#: all, so a bundle scored under a different prior, split or null is refit.
INFERENCE = {
    "penalty_sigma": runner.CANONICAL_PENALTY_SIGMA,
    "n_folds": 5,
    "fold_seed": 7,
    "n_control_draws": 40,
    "control_draw_seed": 9,
}


def _results_exist(*args, **overrides):
    """``runner.results_exist`` asking for the inference and the SWR source
    these tests write."""
    return runner.results_exist(
        *args,
        **{
            **INFERENCE,
            "swr_interval_source": SWR_SOURCE,
            "swr_events_sha256": SWR_EVENTS_SHA,
            **overrides,
        },
    )


def _write_bundle_files(
    directory,
    epoch_identifier,
    stop_cohort,
    sidecar_cohort,
    sidecar_hash="abc123",
    is_swr_definition=runner.IS_SWR_DEFINITION,
    event_table_sha=EVENT_TABLE_SHA,
    data_over_time_sha=DOT_SHA,
    swr_interval_source=SWR_SOURCE,
    swr_events_sha=SWR_EVENTS_SHA,
    fitter=None,
    estimator=runner.ESTIMATOR,
    inference=runner.INFERENCE,
    **sidecar_overrides,
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
                    "swr_interval_source": swr_interval_source,
                    "swr_events_sha256": swr_events_sha,
                    "fitter": fitter,
                    "estimator": estimator,
                    "inference": inference,
                    "penalty_sigma_log_rate": INFERENCE["penalty_sigma"],
                    "n_folds": INFERENCE["n_folds"],
                    "fold_seed": INFERENCE["fold_seed"],
                    "n_control_draws": INFERENCE["n_control_draws"],
                    "control_draw_seed": INFERENCE["control_draw_seed"],
                    "empirical_pvalue_method": runner.EMPIRICAL_PVALUE_METHOD,
                    "control_null_event_policy": runner.CONTROL_NULL_EVENT_POLICY,
                    "min_usable_null_draws": runner.required_usable_null_draws(
                        INFERENCE["n_control_draws"]
                    ),
                    "null_usable_draws": {},
                    "matched_control_placement_method": (
                        runner.MATCHED_CONTROL_PLACEMENT_METHOD
                    ),
                    **sidecar_overrides,
                }
            )
        )


def test_results_exist_does_not_confuse_the_two_cohorts(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "at_well")

    assert _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )
    assert not _results_exist(
        tmp_path, epoch_identifier, False, "all_stops", None, EVENT_TABLE_SHA, DOT_SHA
    )
    assert not _results_exist(
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

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_that_does_not_state_its_population_is_refit(tmp_path):
    # The canonical name is unsuffixed, so a bundle written before the cohort
    # existed sits under exactly the name a canonical run wants. Skipping on the
    # name alone would keep an all-bins fit as if it were the at-well one.
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", sidecar_cohort=None)

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_naming_another_population_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "all_stops")

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_an_allstops_bundle_matches_when_no_stop_table_was_read(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "all_stops", "all_stops", sidecar_hash=None
    )

    assert _results_exist(
        tmp_path, epoch_identifier, False, "all_stops", None, EVENT_TABLE_SHA, DOT_SHA
    )


@pytest.fixture
def fake_session(monkeypatch, tmp_path, replay_content_root):
    """Replace the per-session fit and the cohort load with recorded stubs."""
    calls: dict[str, object] = {}

    # The sidecar hashes the stop table the cohort was cut from; give it one
    # under the test's data root rather than read the shared Processed-Data tree.
    stop_table = replay_content_root / "stop_table.parquet"
    write_table(pd.DataFrame({"stop_start": [10.0], "stop_end": [11.0]}), stop_table)

    # These tests are about what the runner writes, not about which card it
    # fits on; without this they would claim a real GPU on a shared host.
    monkeypatch.setattr(
        runner,
        "configure_jax_gpu",
        lambda **kwargs: {
            "jax_backend": "cpu",
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_name": None,
        },
    )

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
        # The real one hands the scoring material back through this dict.
        if kwargs.get("scoring_out") is not None:
            kwargs["scoring_out"].update(
                {
                    "spline_basis": np.eye(11),
                    "lag_time": np.linspace(-0.25, 0.25, 11),
                }
            )
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
                "n_model_fits": 18,
                "n_nonconverged_model_fits": 2,
                "n_neurons_with_nonconverged_model": 1,
                "matched_control_seed": 12345,
                "n_matched_control": 7,
                "n_matched_control_unplaced": 1,
                "n_matched_control_onset_outside_fitted_bout": 1,
                "n_matched_control_bout_saturated": 0,
            },
        )

    def _fake_event_inputs(nwb_file_name, epoch_name, **kwargs):
        calls["event_inputs"] = (nwb_file_name, epoch_name)
        return {
            "is_swr_definition": runner.IS_SWR_DEFINITION,
            "immobility_inequality": "<",
            "canonical_event_table_path": "/canonical/event_table.parquet",
            "canonical_event_table_sha256": EVENT_TABLE_SHA,
            "swr_interval_source": SWR_SOURCE,
            "swr_events_path": "/canonical/swr/kay_events.parquet",
            "swr_events_sha256": SWR_EVENTS_SHA,
            "data_over_time_path": "/canonical/dot.pkl",
            "data_over_time_sha256": DOT_SHA,
        }

    def _fake_score(context, **kwargs):
        calls["score"] = kwargs
        n_draws = kwargs["n_control_draws"]
        statistics = pd.DataFrame(
            {
                "SWR: heldout_deviance": [100.0, 200.0],
                "SWR vs. constant: heldout_delta_dev": [1.5, -0.5],
                "SWR: p_heldout": [0.025, 0.9],
                "SWR: n_usable_null_draws": [n_draws, n_draws],
                "min_heldout_model": ["SWR", "constant"],
            }
        )
        return statistics, {
            "fold_sizes": [10, 10, 10, 10, 10],
            "events_per_fold": {"non_local_no_SWR": [2, 1, 3, 2, 2]},
            "non_estimable_models": [],
            "null_usable_draws": {
                name: [n_draws, n_draws]
                for name in (
                    "SWR",
                    "non_local",
                    "non_local_no_SWR",
                    "non_local_SWR",
                    "engagement",
                )
            },
        }

    monkeypatch.setattr(runner, "load_session_stop_cohort", _fake_cohort)
    monkeypatch.setattr(runner, "run_session", _fake_run_session)
    monkeypatch.setattr(runner, "canonical_event_input_hashes", _fake_event_inputs)
    monkeypatch.setattr(runner, "score_session", _fake_score)
    return calls


def test_the_sidecar_states_the_population_the_fit_was_read_on(
    tmp_path, fake_session, monkeypatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
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
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["is_swr_definition"] == runner.IS_SWR_DEFINITION
    assert record["immobility_inequality"] == "<"
    assert record["canonical_event_table_path"] == "/canonical/event_table.parquet"
    assert record["canonical_event_table_sha256"] == EVENT_TABLE_SHA
    assert record["swr_interval_source"] == SWR_SOURCE
    assert record["swr_events_sha256"] == SWR_EVENTS_SHA
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
    assert record["n_nonconverged_model_fits"] == 2
    assert record["n_neurons_with_nonconverged_model"] == 1
    assert record["n_model_fits"] == 18
    assert record["n_matched_control_onset_outside_fitted_bout"] == 1
    assert record["n_matched_control_bout_saturated"] == 0


def test_the_sidecar_records_the_commit_the_parent_passed_in(
    tmp_path, fake_session, monkeypatch
):
    # A dask worker imports the code after the parent did; the checkout may
    # have moved on by then, so it records the parent's commit, not its own.
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        git_hash="parent-commit-dirty",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["git_hash"] == "parent-commit-dirty"
    # This process loaded another commit than the parent's, so it may have run
    # newer library code: the sidecar records both, and the log says so.
    assert record["worker_git_hash"] == fit_provenance.git_hash()
    log = (tmp_path / "logs" / "wilbur20210329_02_r1_mPFC.log").read_text()
    assert "parent-commit-dirty" in log
    assert f"loaded {fit_provenance.git_hash()}" in log


def test_a_session_run_on_its_own_records_the_commit_this_process_loaded(
    tmp_path, fake_session, monkeypatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["git_hash"] == fit_provenance.git_hash()
    assert "worker_git_hash" not in record


def test_a_worker_that_cannot_read_git_still_fits_under_the_parents_commit(
    tmp_path, fake_session, monkeypatch
):
    # A worker that could not read git when it imported must not fail the
    # session: the parent's commit is the one recorded, and the worker's is
    # recorded as unknown.
    def no_git():
        raise RuntimeError("git could not be read at import")

    monkeypatch.setattr(runner, "loaded_commit", no_git)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        git_hash="parent-commit",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["git_hash"] == "parent-commit"
    assert record["worker_git_hash"] == "unknown"


def test_a_session_run_on_its_own_needs_a_readable_commit(
    tmp_path, fake_session, monkeypatch
):
    def no_git():
        raise RuntimeError("git could not be read at import")

    monkeypatch.setattr(runner, "loaded_commit", no_git)

    with pytest.raises(RuntimeError, match="git could not be read"):
        runner.run_analysis(
            "wilbur20210329_.nwb",
            "02_r1",
            log_directory=tmp_path / "logs",
            stop_cohort="at_well",
            output_dir=tmp_path,
            cli=["run_pfc_analysis_bandit.py"],
        )


def test_a_bundle_fit_on_the_retired_event_definition_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path,
        epoch_identifier,
        "at_well",
        "at_well",
        is_swr_definition="ripple_times_overlap",
    )

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


@pytest.mark.parametrize(
    ("source", "sha"),
    [
        ("data_over_time.SWR_event", None),
        ("kay@an_earlier_run", SWR_EVENTS_SHA),
        (SWR_SOURCE, "stale333"),
    ],
    ids=["stored-ripples", "another-inventory-run", "changed-events-file"],
)
def test_a_bundle_fit_on_other_swrs_is_refit(tmp_path, source, sha):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path,
        epoch_identifier,
        "at_well",
        "at_well",
        swr_interval_source=source,
        swr_events_sha=sha,
    )

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_cut_from_another_event_table_or_decode_frame_is_refit(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", event_table_sha="stale111"
    )
    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )

    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", data_over_time_sha="stale222"
    )
    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_a_bundle_fit_by_the_other_estimator_is_refit(tmp_path):
    # The two fitters agree to the equivalence audit's tolerances, not
    # bit-for-bit, so a batch that means to be one estimator throughout must
    # not keep sessions from the other under the name it is about to write.
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(
        tmp_path, epoch_identifier, "at_well", "at_well", fitter="statsmodels"
    )

    assert not _results_exist(
        tmp_path,
        epoch_identifier,
        False,
        "at_well",
        "abc123",
        EVENT_TABLE_SHA,
        DOT_SHA,
        "jax",
    )
    assert _results_exist(
        tmp_path,
        epoch_identifier,
        False,
        "at_well",
        "abc123",
        EVENT_TABLE_SHA,
        DOT_SHA,
        "statsmodels",
    )


def test_all_stops_fits_without_a_cohort_and_writes_the_suffixed_bundle(
    tmp_path, fake_session, monkeypatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="all_stops",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
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


# ---------------------------------------------------------------------------
# Partitioning a batch across workers
# ---------------------------------------------------------------------------


def _epoch_info(keys: list[tuple[str, str]]) -> pd.DataFrame:
    """An ``epoch_info`` frame holding exactly ``keys``, as the loader shapes it."""
    index = pd.MultiIndex.from_tuples(
        keys, names=["nwb_file_name", "interval_list_name"]
    )
    return pd.DataFrame(
        {
            "animal": [name.split("2")[0] for name, _ in keys],
            "date": [int(name[-13:-5]) for name, _ in keys],
        },
        index=index,
    )


class TestSessionPartitionFile:
    def test_only_the_named_sessions_are_fit(self, tmp_path):
        keys = [
            ("chimi20200223_.nwb", "02_r1"),
            ("chimi20200223_.nwb", "06_r3"),
            ("j1620210710_.nwb", "02_r1"),
        ]
        path = tmp_path / "worker1.sessions"
        path.write_text("j1620210710_.nwb 02_r1\nchimi20200223_.nwb 06_r3\n")
        args = runner.get_command_line_arguments(["--sessions", str(path)])

        filtered = runner._filter_epoch_info(_epoch_info(keys), args)

        assert set(runner._session_keys(filtered)) == {
            ("j1620210710_.nwb", "02_r1"),
            ("chimi20200223_.nwb", "06_r3"),
        }

    def test_a_session_that_does_not_exist_stops_the_worker(self, tmp_path):
        # A mistyped partition line must not quietly shrink the batch: the
        # missing session would simply never be fit, and the shortfall would
        # only show up when the audit counted 70 bundles instead of 71.
        keys = [("chimi20200223_.nwb", "02_r1")]
        path = tmp_path / "worker2.sessions"
        path.write_text("chimi20200223_.nwb 99_r9\n")
        args = runner.get_command_line_arguments(["--sessions", str(path)])

        with pytest.raises(ValueError, match="chimi20200223_.nwb 99_r9"):
            runner._filter_epoch_info(_epoch_info(keys), args)

    def test_a_partition_composes_with_the_animal_filter(self, tmp_path):
        keys = [
            ("chimi20200223_.nwb", "02_r1"),
            ("j1620210710_.nwb", "02_r1"),
        ]
        path = tmp_path / "worker3.sessions"
        path.write_text("chimi20200223_.nwb 02_r1\nj1620210710_.nwb 02_r1\n")
        args = runner.get_command_line_arguments(
            ["--sessions", str(path), "--animal", "chimi"]
        )

        filtered = runner._filter_epoch_info(_epoch_info(keys), args)

        assert runner._session_keys(filtered) == [("chimi20200223_.nwb", "02_r1")]


def test_one_card_cannot_be_handed_to_every_worker(capsys):
    # --gpu names one card; a multi-worker run would send every worker to it,
    # each claiming and preallocating a share of the same card while the
    # sidecars all named it. The layout that works is one process per card.

    with pytest.raises(SystemExit):
        runner.get_command_line_arguments(["--gpu", "3", "--n_workers", "6"])

    assert "every worker would be sent to that card" in capsys.readouterr().err


def test_one_card_and_one_worker_is_the_supported_layout():
    args = runner.get_command_line_arguments(["--gpu", "3", "--n_workers", "1"])

    assert (args.gpu, args.n_workers) == (3, 1)


# ---------------------------------------------------------------------------
# The estimator and the inference the bundle states
# ---------------------------------------------------------------------------


def test_only_a_bundle_at_another_prior_is_suffixed():
    assert runner.penalty_sigma_suffix(runner.CANONICAL_PENALTY_SIGMA) == ""
    assert runner.penalty_sigma_suffix(2.0) == "_sigma2"
    assert runner.penalty_sigma_suffix(4.5) == "_sigma4p5"
    # A decimal point would truncate the companion array names, which are
    # written by suffixing the stem.
    assert "." not in runner.penalty_sigma_suffix(4.5)


def test_the_sensitivity_bundle_cannot_take_the_canonical_name(tmp_path):
    canonical = runner.modulation_path(tmp_path, "wilbur20210329_02_r1_mPFC", "at_well")
    sensitivity = runner.modulation_path(
        tmp_path, "wilbur20210329_02_r1_mPFC", "at_well", 2.0
    )

    assert canonical.name == "wilbur20210329_02_r1_mPFC_modulation.pkl"
    assert sensitivity.name == "wilbur20210329_02_r1_mPFC_modulation_sigma2.pkl"


def test_the_seeds_are_a_function_of_the_session_and_the_draw():
    folds = runner.session_seed("folds", "wilbur20210329_.nwb", "02_r1")

    # Reproducible: a session refit on its own splits the same way.
    assert folds == runner.session_seed("folds", "wilbur20210329_.nwb", "02_r1")
    # Two draws of the same session are independent of each other, and of the
    # matched control's own draw.
    assert folds != runner.session_seed("control_draws", "wilbur20210329_.nwb", "02_r1")
    from continuum_swr_replay.analysis.pfc import matched_control_seed

    assert folds != matched_control_seed("wilbur20210329_.nwb", "02_r1")
    # Two sessions never share a draw.
    assert folds != runner.session_seed("folds", "wilbur20210329_.nwb", "04_r2")


@pytest.mark.parametrize(
    "stale",
    [
        {"estimator": "poisson_glm"},
        {"inference": "likelihood_ratio"},
        {"penalty_sigma_log_rate": 1.5},
        {"n_folds": 10},
        {"fold_seed": 11},
        {"n_control_draws": 20},
        {"control_draw_seed": 12},
        {"empirical_pvalue_method": "raw_exceedance"},
        {"control_null_event_policy": "all_events_even_if_null_is_thinner"},
        {"min_usable_null_draws": 0},
        {"matched_control_placement_method": "disjoint_control_windows"},
    ],
)
def test_a_bundle_produced_by_another_estimator_or_inference_is_refit(tmp_path, stale):
    # The canonical name is unsuffixed, so the maximum-likelihood bundle sits
    # under exactly the name this run wants; and a bundle scored on another
    # split or against a null of another size carries different held-out
    # differences and different p-values.
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "at_well", **stale)

    assert not _results_exist(
        tmp_path, epoch_identifier, False, "at_well", "abc123", EVENT_TABLE_SHA, DOT_SHA
    )


def test_the_sidecar_states_the_estimator_and_the_inference(
    tmp_path, fake_session, monkeypatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        n_folds=5,
        n_control_draws=40,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["estimator"] == "ridge_poisson_glm"
    assert record["inference"] == "blocked_heldout_deviance"
    assert record["penalty_sigma_log_rate"] == runner.CANONICAL_PENALTY_SIGMA
    # What that says about a curve, which is not the same number.
    assert record["implied_prior_sd"]["per_lag"] > 0
    assert record["implied_prior_sd"]["sustained"] > 0
    assert record["n_folds"] == 5
    assert record["n_control_draws"] == 40
    assert record["empirical_pvalue_method"] == "plus_one"
    assert record["control_null_event_policy"] == runner.CONTROL_NULL_EVENT_POLICY
    assert record["min_usable_null_draws"] == 40
    assert (
        record["matched_control_placement_method"]
        == runner.MATCHED_CONTROL_PLACEMENT_METHOD
    )
    assert record["n_basis_per_class"] == 11
    # Derived from the session key, so a lone refit reproduces the batch.
    assert record["fold_seed"] == runner.session_seed(
        "folds", "wilbur20210329_.nwb", "02_r1"
    )
    assert record["control_draw_seed"] == runner.session_seed(
        "control_draws", "wilbur20210329_.nwb", "02_r1"
    )
    # The hand-back the batched fitter used to do is gone; nothing counts it.
    assert "n_neurons_reference_fit" not in record
    # The prior the models were fit under reaches the fit itself.
    assert (
        fake_session["run_session"]["penalty_sigma"] == runner.CANONICAL_PENALTY_SIGMA
    )
    assert fake_session["score"]["penalty_sigma"] == runner.CANONICAL_PENALTY_SIGMA


def test_the_bundle_carries_the_heldout_columns_and_not_the_retired_ones(
    tmp_path, fake_session, monkeypatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        cli=["run_pfc_analysis_bandit.py"],
    )

    table = pd.read_pickle(output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl")
    assert "SWR vs. constant: heldout_delta_dev" in table.columns
    assert "SWR: p_heldout" in table.columns
    assert "min_heldout_model" in table.columns
    assert not [c for c in table.columns if c.endswith((": lr", ": pval", ": aic"))]
    assert "min_aic_model" not in table.columns


def test_a_sensitivity_prior_writes_its_own_bundle_and_draws_no_null(
    tmp_path, fake_session, monkeypatch
):
    # A null drawn under a prior nothing is shipped at is a null for nothing,
    # and it is the expensive half of the run.
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner.run_analysis(
        "wilbur20210329_.nwb",
        "02_r1",
        log_directory=tmp_path / "logs",
        stop_cohort="at_well",
        output_dir=output_dir,
        penalty_sigma=2.0,
        cli=["run_pfc_analysis_bandit.py"],
    )

    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation_sigma2.pkl"
    assert artifact.exists()
    assert not (output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl").exists()
    record = json.loads(artifact.with_suffix(".provenance.json").read_text())
    assert record["penalty_sigma_log_rate"] == 2.0
    assert record["n_control_draws"] == 0
    assert fake_session["score"]["n_control_draws"] == 0


def test_the_retired_statistics_are_dropped_before_the_bundle_is_written():
    table = pd.DataFrame(
        {
            "SWR vs. constant: lr": [1.0],
            "SWR vs. constant: pval": [0.2],
            "SWR vs. constant: frac_null_dev": [0.01],
            "theta vs. constant: pval": [0.3],
            "min_aic_model": ["SWR"],
            "min_aic_model_replay_split": ["constant"],
            "immobility_avg_firing_rate": [3.0],
        }
    )

    kept = runner.drop_retired_statistics(table)

    assert list(kept.columns) == [
        "SWR vs. constant: frac_null_dev",
        "immobility_avg_firing_rate",
    ]


def test_a_batch_cannot_be_launched_on_the_unpenalised_backend():
    # The per-unit statsmodels IRLS has no penalised path, so a batch fit with
    # it would be an unpenalised ladder written under a sidecar claiming a
    # prior. It is not offered at all.
    with pytest.raises(SystemExit) as excinfo:
        runner.get_command_line_arguments(["--fitter", "statsmodels"])

    assert excinfo.value.code == 2


def test_a_null_below_the_reporting_floor_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        runner.get_command_line_arguments(["--n-control-draws", "19"])

    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Open files: Spyglass's NWB cache and the open-file limit
# ---------------------------------------------------------------------------


def _recording_release(monkeypatch) -> list[bool]:
    """Replace ``release_nwb_files``; return the list each call appends to."""
    closed: list[bool] = []

    def release_nwb_files() -> bool:
        closed.append(True)
        return True

    monkeypatch.setattr(runner, "release_nwb_files", release_nwb_files)
    return closed


@pytest.mark.parametrize("outcome", ["fit", "fit_fails", "raises"])
def test_each_session_closes_the_nwb_files_spyglass_opened(
    tmp_path, fake_session, monkeypatch, outcome
):
    """A worker runs one session at a time, so when its session ends -- fit,
    failed inside ``run_analysis`` (which logs and returns) or raised out of it
    (no free card) -- it closes the NWB files Spyglass cached and logs that it did."""
    closed = _recording_release(monkeypatch)
    if outcome == "fit_fails":

        def run_session(*args, **kwargs):
            raise RuntimeError("simulated: the fit failed")

        monkeypatch.setattr(runner, "run_session", run_session)
    elif outcome == "raises":

        def configure_jax_gpu(**kwargs):
            raise RuntimeError("simulated: no free card")

        monkeypatch.setattr(runner, "configure_jax_gpu", configure_jax_gpu)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def session():
        runner.run_analysis_then_release(
            nwb_file_name="wilbur20210329_.nwb",
            epoch_name="02_r1",
            log_directory=tmp_path / "logs",
            output_dir=output_dir,
            cli=["run_pfc_analysis_bandit.py"],
        )

    if outcome == "raises":
        with pytest.raises(RuntimeError, match="no free card"):
            session()
    else:
        session()

    assert closed == [True]
    artifact = output_dir / "wilbur20210329_02_r1_mPFC_modulation.pkl"
    assert artifact.exists() == (outcome == "fit")
    log = (tmp_path / "logs" / "wilbur20210329_02_r1_mPFC.log").read_text()
    assert log.splitlines()[-1].endswith("NWB files Spyglass held open closed: True")


class _Client:
    """Stands in for ``dask.distributed.Client``: it connects to nothing."""

    def __init__(self, cluster: object) -> None:
        pass

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.mark.parametrize("n_workers", [1, 2], ids=["serial", "dask"])
def test_main_raises_the_limit_first_and_closes_each_sessions_files(
    tmp_path, monkeypatch, capsys, n_workers
):
    """The limit is raised before any cluster starts, so the workers inherit
    it, and every session, on either path, ends by closing its files.

    The cluster, the client and dask are stand-ins: each session's call runs
    here, as it is built.
    """
    keys = [("chimi20200223_.nwb", "02_r1"), ("chimi20200223_.nwb", "04_r2")]
    events: list[str] = []

    def raise_open_file_limit():
        events.append("limit")
        return 1024, 65536

    def local_cluster(**kwargs):
        events.append("cluster")
        assert kwargs["threads_per_worker"] == 1 and kwargs["processes"] is True
        return object()

    def run_analysis(nwb_file_name, epoch_name, **kwargs):
        events.append(f"run {epoch_name}")

    def release_nwb_files():
        events.append("release")
        return True

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    monkeypatch.setattr(runner, "analysed_epochs", lambda: _epoch_info(keys))
    monkeypatch.setattr(runner, "_filter_to_sorted_sessions", lambda info: info)
    monkeypatch.setattr(runner, "run_analysis", run_analysis)
    monkeypatch.setattr(runner, "release_nwb_files", release_nwb_files)
    monkeypatch.setattr(runner, "LocalCluster", local_cluster)
    monkeypatch.setattr(runner, "Client", _Client)
    monkeypatch.setattr(
        runner,
        "dask",
        types.SimpleNamespace(
            delayed=lambda function: function, compute=lambda *outcomes: outcomes
        ),
    )

    runner.main(
        [
            "--n_workers",
            str(n_workers),
            "--local_scratch_dir",
            str(tmp_path / "scratch"),
            "--log_directory",
            str(tmp_path / "logs"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    sessions = ["run 02_r1", "release", "run 04_r2", "release"]
    assert events == ["limit", *(["cluster"] if n_workers > 1 else []), *sessions]
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out


@pytest.mark.parametrize("n_workers", [1, 2], ids=["serial", "dask"])
def test_main_hands_every_session_the_commit_it_loaded(
    tmp_path, monkeypatch, n_workers
):
    """A dask worker imports the code later than ``main`` does, so the commit
    each session's sidecar records is the one ``main`` loaded, passed in."""
    keys = [("chimi20200223_.nwb", "02_r1"), ("chimi20200223_.nwb", "04_r2")]
    passed: list[str] = []
    clis: list[list[str]] = []

    def run_analysis(nwb_file_name, epoch_name, **kwargs):
        passed.append(kwargs["git_hash"])
        clis.append(kwargs["cli"])

    monkeypatch.setattr(runner, "loaded_commit", lambda: "parent-commit-dirty")
    monkeypatch.setattr(runner, "raise_open_file_limit", lambda: (1024, 65536))
    monkeypatch.setattr(runner, "analysed_epochs", lambda: _epoch_info(keys))
    monkeypatch.setattr(runner, "_filter_to_sorted_sessions", lambda info: info)
    monkeypatch.setattr(runner, "run_analysis", run_analysis)
    monkeypatch.setattr(runner, "release_nwb_files", lambda: True)
    monkeypatch.setattr(runner, "LocalCluster", lambda **kwargs: object())
    monkeypatch.setattr(runner, "Client", _Client)
    monkeypatch.setattr(
        runner,
        "dask",
        types.SimpleNamespace(
            delayed=lambda function: function, compute=lambda *outcomes: outcomes
        ),
    )

    argv = [
        "--n_workers",
        str(n_workers),
        "--local_scratch_dir",
        str(tmp_path / "scratch"),
        "--log_directory",
        str(tmp_path / "logs"),
        "--output-dir",
        str(tmp_path / "out"),
    ]

    runner.main(argv)

    assert passed == ["parent-commit-dirty", "parent-commit-dirty"]
    # Each sidecar records the arguments main was given, not pytest's.
    assert clis == [[sys.argv[0], *argv]] * 2
    cohort = json.loads((tmp_path / "out" / "pfc_cohort_mPFC.json").read_text())
    assert cohort["git_hash"] == "parent-commit-dirty"


def test_main_takes_the_registers_analysed_epochs_less_those_without_sorting(
    tmp_path, monkeypatch, capsys
):
    """The PFC cohort is the register's analysed epochs whose session has curated
    spike sorting. An analysed epoch without it is left out at the PFC level,
    with that reason, and is not an epoch exclusion."""
    keys = [
        ("chimi20200223_.nwb", "02_r1"),
        ("chimi20200223_.nwb", "04_r2"),
        ("chimi20200224_.nwb", "02_r1"),
    ]
    ran: list[tuple[str, str]] = []

    def run_analysis(nwb_file_name, epoch_name, **kwargs):
        ran.append((nwb_file_name, epoch_name))

    def sorted_sessions_only(epoch_info):
        names = epoch_info.index.get_level_values("nwb_file_name")
        return epoch_info[names == "chimi20200223_.nwb"]

    monkeypatch.setattr(runner, "raise_open_file_limit", lambda: (1024, 65536))
    monkeypatch.setattr(runner, "analysed_epochs", lambda: _epoch_info(keys))
    monkeypatch.setattr(runner, "_filter_to_sorted_sessions", sorted_sessions_only)
    monkeypatch.setattr(runner, "run_analysis", run_analysis)
    monkeypatch.setattr(runner, "release_nwb_files", lambda: True)

    runner.main(
        [
            "--n_workers",
            "1",
            "--local_scratch_dir",
            str(tmp_path / "scratch"),
            "--log_directory",
            str(tmp_path / "logs"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert ran == keys[:2]
    out = capsys.readouterr().out
    assert (
        "1 analysed epoch(s) left out at the PFC level, "
        f"{runner.NO_SORTED_UNITS_REASON}: chimi20200224_.nwb 02_r1"
    ) in out
    assert not hasattr(runner, "get_epoch_info")
    # and written down beside the bundles, not only printed
    cohort = json.loads((tmp_path / "out" / "pfc_cohort_mPFC.json").read_text())
    assert cohort["n_analysed"] == 3
    assert cohort["n_pfc_cohort"] == 2
    assert cohort["ran"] == ["chimi20200223_.nwb 02_r1", "chimi20200223_.nwb 04_r2"]
    assert cohort["left_out"] == {
        "chimi20200224_.nwb 02_r1": runner.NO_SORTED_UNITS_REASON
    }


def test_the_extended_event_set_needs_its_own_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CSR_PROCESSED_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        runner.get_command_line_arguments(
            ["--event-set", "with_swr_without_replay", "--output-dir", str(tmp_path)]
        )
    args = runner.get_command_line_arguments(
        [
            "--event-set",
            "with_swr_without_replay",
            "--output-dir",
            str(tmp_path / "extended"),
            "--null-classes",
            "SWR_no_non_local",
        ]
    )
    assert args.null_classes == ["SWR_no_non_local"]
    with pytest.raises(SystemExit):
        runner.get_command_line_arguments(["--null-classes", "SWR_no_non_local"])


def test_results_exist_keeps_the_event_sets_apart(tmp_path):
    epoch_identifier = "wilbur20210329_02_r1_mPFC"
    _write_bundle_files(tmp_path, epoch_identifier, "at_well", "at_well")
    common = (
        tmp_path,
        epoch_identifier,
        False,
        "at_well",
        "abc123",
        EVENT_TABLE_SHA,
        DOT_SHA,
    )

    # A canonical bundle (no event_set key) never satisfies an extended request.
    assert _results_exist(*common)
    assert not _results_exist(*common, event_set="with_swr_without_replay")

    _write_bundle_files(
        tmp_path,
        epoch_identifier,
        "at_well",
        "at_well",
        event_set="with_swr_without_replay",
    )
    # The extended request also needs the two extra curves on disk.
    assert not _results_exist(*common, event_set="with_swr_without_replay")
    base = runner.modulation_path(tmp_path, epoch_identifier, "at_well").with_suffix("")
    for name in runner.EXTENDED_CURVE_SOURCES:
        base.with_suffix(f".{name}_modulation.npy").touch()
    assert _results_exist(*common, event_set="with_swr_without_replay")
    assert not _results_exist(*common)
    assert not _results_exist(
        *common, event_set="with_swr_without_replay", null_classes=["SWR_no_non_local"]
    )
