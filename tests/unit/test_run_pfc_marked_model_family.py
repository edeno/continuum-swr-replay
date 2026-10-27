"""Runner-contract tests for the recommended marked-model family."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.marked_model_family import MarkedFamilyResults

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "run_pfc_marked_model_family.py"
)
_SPEC = importlib.util.spec_from_file_location("run_pfc_marked_model_family", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

MODEL = "belief_content"
OTHER_MODEL = "spatial_role"
SPEC = runner.RECOMMENDED_MODEL_SPECS[MODEL]
EPOCH_IDENTIFIER = "rat20200101_01_r1_mPFC"
INPUT_HASHES = {
    "stop_table_sha256": "stop",
    "event_arm_table_sha256": "arm",
    "event_table_sha256": "event",
    "canonical_event_table_sha256": "event",
    "data_over_time_sha256": "dot",
    "swr_interval_source": "kay",
    "swr_events_sha256": "swr",
}
GPU_RECORD = {
    "jax_backend": "cpu",
    "gpu_index": None,
    "gpu_uuid": None,
    "gpu_name": None,
}


def _refuse_non_json(token: str) -> None:
    """Fail on a bare ``NaN``/``Infinity`` token, which is not strict JSON."""
    raise AssertionError(f"sidecar carries the non-JSON token {token!r}")


def _contract(spec=SPEC, **overrides):
    """Build the runner's fit contract with the fixtures' values."""
    kwargs = {
        "penalty_sigma": 3.0,
        "mark_penalty_sigma": 1.0,
        "n_folds": 2,
        "fold_seed": 3,
        "design_batch_size": 4,
        "min_marked_events": 10,
        "brain_area": "mPFC",
        "input_hashes": INPUT_HASHES,
    }
    kwargs.update(overrides)
    return runner.build_fit_contract(spec, **kwargs)


def _context(n_units: int = 2) -> SimpleNamespace:
    """Mock one prepared session: 6 fit bins, the pinned 251-lag fit window."""
    fit_mask = pd.Series([True] * 6, index=np.arange(6) / 500.0)
    events = {
        event_class: pd.DataFrame({"event_id": [1, 2], "start_time": [0.002, 0.006]})
        for event_class in runner.MARKED_EVENT_CLASSES
    }
    return SimpleNamespace(
        fit_mask=fit_mask,
        counts=np.ones((6, n_units)),
        events=events,
        spline_basis=np.ones((251, runner.N_BASIS_PER_KERNEL)),
        lag_time=np.arange(-125, 126) / 500.0,
        lags=np.arange(-125, 126),
        cohort_info={"n_bins_fit": 6},
        input_provenance={
            "event_arm_table_sha256": "arm",
            "canonical_event_table_sha256": "event",
            "data_over_time_sha256": "dot",
            "swr_interval_source": "kay",
            "swr_events_sha256": "swr",
        },
        baseline=np.column_stack([np.ones(6), np.r_[np.ones(3), np.zeros(3)]]),
        baseline_names=("well_2", "rewarded_outcome"),
        baseline_report={
            "baseline_reference_well": 0,
            "baseline_dropped_constant": ["outcome_unknown"],
            "n_bins_without_well": 0,
        },
    )


def _result(
    context: SimpleNamespace, spec=SPEC, *, n_not_converged: int = 0
) -> MarkedFamilyResults:
    """Mock fitted results whose curves span the full 251-lag fit window."""
    n_units = context.counts.shape[1]
    n_lags = context.lag_time.size
    grid = np.arange(n_lags * n_units, dtype=float).reshape(n_lags, n_units)
    return MarkedFamilyResults(
        statistics=pd.DataFrame({"family_full_heldout_deviance": [1.0] * n_units}),
        typical_curves={
            event_class: grid + offset
            for offset, event_class in enumerate(runner.MARKED_EVENT_CLASSES)
        },
        mark_curves={
            event_class: {
                mark: -grid - offset
                for offset, mark in enumerate(spec.scientific_marks)
            }
            for event_class in runner.MARKED_EVENT_CLASSES
        },
        diagnostics={
            "n_units_full_fit_not_converged": n_not_converged,
            "n_units_nan_heldout_delta_by_comparison": {},
            # A real diagnostics channel can be NaN; the sidecar must stay JSON.
            "mean_heldout_delta": float("nan"),
        },
    )


def _install_mocks(monkeypatch, context: SimpleNamespace, *, build=None, fit=None):
    """Replace every Spyglass-backed input with the mocked session."""
    monkeypatch.setattr(
        runner,
        "setup_logger",
        lambda *a, **k: logging.getLogger("marked-family-runner-test"),
    )
    monkeypatch.setattr(
        runner,
        "record_inputs",
        lambda inputs: {
            "stop_table": {"sha256": "stop"},
            "event_arm_table": {"sha256": "arm"},
            "event_table": {"sha256": "event"},
        },
    )
    monkeypatch.setattr(
        runner,
        "canonical_event_input_hashes",
        lambda *a, **k: {
            "canonical_event_table_sha256": "event",
            "data_over_time_sha256": "dot",
            "swr_interval_source": "kay",
            "swr_events_sha256": "swr",
        },
    )
    monkeypatch.setattr(
        runner,
        "load_session_stop_cohort",
        lambda *a, **k: (
            pd.DataFrame({"start_time": [0.0], "end_time": [1.0]}),
            {"stop_table_sha256": "stop"},
        ),
    )
    monkeypatch.setattr(
        runner, "prepare_marked_session_context", lambda *a, **k: context
    )
    monkeypatch.setattr(
        runner,
        "load_session_recommended_event_marks",
        lambda *a, **k: (
            context.events,
            {"event_arm_table_sha256": "arm", "event_table_sha256": "event"},
        ),
    )
    monkeypatch.setattr(
        runner, "assign_blocked_folds", lambda *a, **k: np.array([0, 0, 0, 1, 1, 1])
    )
    monkeypatch.setattr(
        runner, "prepare_heldout_response", lambda *a, **k: SimpleNamespace()
    )
    # The ladder carries its spec, so a mocked fit can tell the families apart.
    monkeypatch.setattr(
        runner,
        "build_marked_family_ladder",
        build if build is not None else (lambda *a, **k: SimpleNamespace(spec=a[4])),
    )
    monkeypatch.setattr(
        runner,
        "fit_marked_family_ladder",
        fit
        if fit is not None
        else (lambda ladder, *a, **k: _result(context, ladder.spec)),
    )


def _run(tmp_path, **overrides):
    """Run the mocked session with the fixtures' contract values."""
    kwargs = {
        "model_names": (MODEL,),
        "output_dir": tmp_path,
        "log_directory": tmp_path / "logs",
        "mark_penalty_sigma": 1.0,
        "n_folds": 2,
        "fold_seed": 3,
        "gpu_record": GPU_RECORD,
    }
    kwargs.update(overrides)
    return runner.run_analysis("rat20200101_.nwb", "01_r1", **kwargs)


def _artifact(tmp_path, model: str = MODEL) -> Path:
    """Path of the bundle the mocked session writes for `model`."""
    return runner.family_artifact_path(tmp_path, EPOCH_IDENTIFIER, model)


def test_fit_contract_pins_question_marks_and_no_selection_rule():
    contract = _contract()

    assert contract["scientific_marks"] == list(SPEC.scientific_marks)
    assert contract["nuisance_marks"] == list(SPEC.nuisance_marks)
    assert contract["model_question"] == SPEC.question
    assert "no firing-rate or p-value" in contract["neuron_inclusion"]
    assert "shuffle" not in contract["inference"]
    assert contract["fit_half_window_s"] == runner.FIT_HALF_WINDOW_S
    assert contract["n_basis_per_kernel"] == runner.N_BASIS_PER_KERNEL


def test_current_marker_requires_valid_mapping_and_exact_contract(tmp_path):
    artifact = runner.family_artifact_path(tmp_path, "rat_epoch_mPFC", MODEL)
    sidecar = runner.fit_provenance_path(artifact)
    contract = _contract()
    sidecar.write_text("[]")
    assert not runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=contract
    )

    sidecar.write_text(json.dumps({**contract, "bundle_status": "ineligible"}))
    assert runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=contract
    )
    changed = {**contract, "penalty_sigma_log_rate": 4.0}
    assert not runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=changed
    )
    assert contract["mark_penalty_sigma_log_rate"] == 1.0
    changed = {**contract, "mark_penalty_sigma_log_rate": None}
    assert not runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=changed
    )


def test_bundle_is_stale_when_the_window_or_one_input_hash_changes(tmp_path):
    artifact = _artifact(tmp_path)
    artifact.write_bytes(b"complete")
    for path in runner.family_companion_paths(artifact, SPEC):
        path.write_bytes(b"complete")
    contract = _contract()
    runner.fit_provenance_path(artifact).write_text(
        json.dumps({**contract, "bundle_status": "complete"})
    )

    assert runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=contract
    )
    assert not runner._bundle_is_current(
        artifact,
        SPEC,
        overwrite=False,
        fit_contract={
            **contract,
            "fit_half_window_s": contract["fit_half_window_s"] * 2,
        },
    )
    assert not runner._bundle_is_current(
        artifact,
        SPEC,
        overwrite=False,
        fit_contract=_contract(
            input_hashes={**INPUT_HASHES, "event_table_sha256": "other"}
        ),
    )


def test_run_writes_only_target_curves_and_sidecar_last(tmp_path, monkeypatch):
    context = _context()
    _install_mocks(monkeypatch, context)

    succeeded = _run(tmp_path)

    assert succeeded
    artifact = _artifact(tmp_path)
    assert artifact.is_file()
    assert all(path.is_file() for path in runner.family_companion_paths(artifact, SPEC))
    record = json.loads(runner.fit_provenance_path(artifact).read_text())
    assert record["bundle_status"] == "complete"
    assert record["neuron_inclusion"].startswith("all recorded units")
    assert not list(tmp_path.glob("*.label_shuffle*"))
    table = pd.read_pickle(artifact)
    assert table.neuron_index.tolist() == [0, 1]
    assert table.fit_firing_rate_hz.tolist() == [500.0, 500.0]


def test_sidecar_is_strict_json_and_round_trips_into_the_skip_check(
    tmp_path, monkeypatch
):
    context = _context()
    _install_mocks(monkeypatch, context)

    assert _run(tmp_path)

    artifact = _artifact(tmp_path)
    text = runner.fit_provenance_path(artifact).read_text()
    record = json.loads(text, parse_constant=_refuse_non_json)
    assert record["estimator"] == runner.ESTIMATOR
    assert record["marked_family_diagnostics"]["mean_heldout_delta"] is None
    assert runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=_contract()
    )


def test_only_the_target_marks_get_a_slope_curve(tmp_path, monkeypatch):
    context = _context()
    _install_mocks(monkeypatch, context)

    assert _run(tmp_path)

    slopes = sorted(path.name for path in tmp_path.glob("*.slope.npy"))
    assert len(slopes) == 2 * len(runner.MARKED_EVENT_CLASSES) == 4
    assert (
        tmp_path
        / (
            "rat20200101_01_r1_mPFC_belief_content_marked_family"
            ".non_local_SWR.content_weighted_belief_value.slope.npy"
        )
    ).is_file()
    assert not any(mark in name for name in slopes for mark in SPEC.nuisance_marks)


def test_saved_curves_are_sliced_to_the_report_window(tmp_path, monkeypatch):
    context = _context()
    result = _result(context)
    _install_mocks(monkeypatch, context, fit=lambda *a, **k: result)

    assert _run(tmp_path)

    artifact = _artifact(tmp_path)
    lag_time = np.load(artifact.with_suffix(".lag_time.npy"))
    report = np.abs(context.lag_time) <= runner.REPORT_HALF_WINDOW_S + 1e-12
    assert context.lag_time.size == 251
    assert lag_time.size == 201
    np.testing.assert_allclose(lag_time, context.lag_time[report])
    for event_class in runner.MARKED_EVENT_CLASSES:
        np.testing.assert_allclose(
            np.load(artifact.with_suffix(f".{event_class}.typical.npy")),
            result.typical_curves[event_class][report],
        )
        for mark in SPEC.scientific_marks:
            np.testing.assert_allclose(
                np.load(artifact.with_suffix(f".{event_class}.{mark}.slope.npy")),
                result.mark_curves[event_class][mark][report],
            )


def test_ineligible_model_writes_a_marker_that_is_not_re_prepared(
    tmp_path, monkeypatch
):
    context = _context()

    def raise_ineligible(*args, **kwargs):
        raise runner.ModelIneligibleError("too few finite marked events")

    _install_mocks(monkeypatch, context, build=raise_ineligible)

    assert _run(tmp_path)

    artifact = _artifact(tmp_path)
    assert not artifact.exists()
    assert not any(
        path.exists() for path in runner.family_companion_paths(artifact, SPEC)
    )
    record = json.loads(runner.fit_provenance_path(artifact).read_text())
    assert record["bundle_status"] == "ineligible"
    assert record["ineligibility_reason"] == "too few finite marked events"
    assert runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=_contract()
    )

    def fail_if_prepared(*args, **kwargs):
        raise AssertionError("a current marker must not be re-prepared")

    monkeypatch.setattr(runner, "load_session_stop_cohort", fail_if_prepared)
    monkeypatch.setattr(runner, "prepare_marked_session_context", fail_if_prepared)
    assert _run(tmp_path)


def test_session_without_units_writes_a_marker_per_pending_model(
    tmp_path, monkeypatch, caplog
):
    context = _context(n_units=0)

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("a session with no units must not be fit")

    _install_mocks(monkeypatch, context, fit=fail_if_fit)

    with caplog.at_level(logging.WARNING, logger="marked-family-runner-test"):
        assert _run(tmp_path, model_names=(MODEL, OTHER_MODEL))

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "No mPFC units" in message and EPOCH_IDENTIFIER in message
        for message in messages
    )
    for model in (MODEL, OTHER_MODEL):
        artifact = _artifact(tmp_path, model)
        assert not artifact.exists()
        record = json.loads(runner.fit_provenance_path(artifact).read_text())
        assert record["bundle_status"] == "no_units"
        assert record["n_units"] == 0
        assert runner._bundle_is_current(
            artifact,
            runner.RECOMMENDED_MODEL_SPECS[model],
            overwrite=False,
            fit_contract=_contract(runner.RECOMMENDED_MODEL_SPECS[model]),
        )


def _write_stale_bundle(tmp_path, model: str) -> Path:
    """Leave a complete-looking bundle from a previous run on disk."""
    artifact = _artifact(tmp_path, model)
    spec = runner.RECOMMENDED_MODEL_SPECS[model]
    artifact.write_bytes(b"stale")
    for path in runner.family_companion_paths(artifact, spec):
        path.write_bytes(b"stale")
    runner.fit_provenance_path(artifact).write_text(json.dumps({"bundle_status": "x"}))
    return artifact


def test_a_session_that_never_loads_keeps_the_bundles_it_did_not_reach(
    tmp_path, monkeypatch
):
    context = _context()
    stale = [_write_stale_bundle(tmp_path, model) for model in (MODEL, OTHER_MODEL)]

    def raise_on_load(*args, **kwargs):
        raise RuntimeError("stop table unreadable")

    _install_mocks(monkeypatch, context)
    monkeypatch.setattr(runner, "load_session_stop_cohort", raise_on_load)
    sink: list[tuple[str, str]] = []

    assert _run(tmp_path, model_names=(MODEL, OTHER_MODEL), failure_sink=sink) is False
    assert sink == [("RuntimeError", "stop table unreadable")]
    for artifact in stale:
        assert artifact.read_bytes() == b"stale"
        assert runner.fit_provenance_path(artifact).is_file()


def test_one_failed_model_is_recorded_and_the_next_model_still_runs(
    tmp_path, monkeypatch
):
    context = _context()
    _write_stale_bundle(tmp_path, MODEL)

    def fit(ladder, *args, **kwargs):
        if ladder.spec.name == MODEL:
            raise RuntimeError("fit blew up")
        return _result(context, ladder.spec)

    _install_mocks(monkeypatch, context, fit=fit)
    sink: list[tuple[str, str]] = []

    assert _run(tmp_path, model_names=(MODEL, OTHER_MODEL), failure_sink=sink) is False

    failed = _artifact(tmp_path, MODEL)
    assert not failed.exists()
    assert not runner.fit_provenance_path(failed).exists()
    assert not any(
        path.exists() for path in runner.family_companion_paths(failed, SPEC)
    )
    assert len(sink) == 1
    assert sink[0][0] == "RuntimeError"
    assert MODEL in sink[0][1] and "fit blew up" in sink[0][1]

    survivor = _artifact(tmp_path, OTHER_MODEL)
    assert survivor.is_file()
    assert runner.fit_provenance_path(survivor).is_file()


def test_convergence_gate_records_the_counts_and_writes_nothing(tmp_path, monkeypatch):
    context = _context()
    _install_mocks(
        monkeypatch,
        context,
        fit=lambda ladder, *a, **k: _result(context, ladder.spec, n_not_converged=2),
    )
    sink: list[tuple[str, str]] = []

    assert _run(tmp_path, failure_sink=sink) is False

    artifact = _artifact(tmp_path)
    assert not artifact.exists()
    assert not runner.fit_provenance_path(artifact).exists()
    assert sink == [
        ("NonconvergedFullFit", f"2 of 2 units did not converge in model {MODEL}")
    ]


def test_interaction_models_are_appended_without_duplicates(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_pfc_marked_model_family.py"])
    assert runner.get_command_line_arguments().models == list(runner.CORE_MODEL_NAMES)

    monkeypatch.setattr(
        sys, "argv", ["run_pfc_marked_model_family.py", "--include-interactions"]
    )
    assert runner.get_command_line_arguments().models == [
        *runner.CORE_MODEL_NAMES,
        *runner.INTERACTION_MODEL_NAMES,
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pfc_marked_model_family.py",
            "--models",
            OTHER_MODEL,
            runner.INTERACTION_MODEL_NAMES[0],
            "--include-interactions",
        ],
    )
    assert runner.get_command_line_arguments().models == [
        OTHER_MODEL,
        *runner.INTERACTION_MODEL_NAMES,
    ]


@pytest.mark.parametrize(
    ("spline_basis", "lags"),
    [
        (np.ones((251, runner.N_BASIS_PER_KERNEL + 1)), np.arange(-125, 126)),
        (np.ones((51, runner.N_BASIS_PER_KERNEL)), np.arange(-25, 26)),
    ],
)
def test_a_prepared_window_off_the_pinned_contract_is_refused(spline_basis, lags):
    with pytest.raises(RuntimeError, match="pinned contract"):
        runner._check_prepared_window(spline_basis, lags)


def test_runner_contract_records_baseline_names_and_version_two(
    tmp_path, monkeypatch
):
    context = _context()
    _install_mocks(monkeypatch, context)

    assert _run(tmp_path)

    artifact = _artifact(tmp_path)
    record = json.loads(runner.fit_provenance_path(artifact).read_text())
    assert record["bundle_contract_version"] == 2
    assert record["baseline_terms"] == "well_and_reward_outcome"
    assert record["baseline_column_names"] == ["well_2", "rewarded_outcome"]
    assert record["baseline_reference_well"] == 0
    assert record["baseline_dropped_constant"] == ["outcome_unknown"]
    assert runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=_contract()
    )
    # A contract-1 sidecar never matches, so the old corpus is refit, not skipped.
    runner.fit_provenance_path(artifact).write_text(
        json.dumps({**record, "bundle_contract_version": 1})
    )
    assert not runner._bundle_is_current(
        artifact, SPEC, overwrite=False, fit_contract=_contract()
    )
    # Without the baseline the contract differs, so neither bundle stands in
    # for the other.
    assert _contract(baseline_terms="none") != _contract()
