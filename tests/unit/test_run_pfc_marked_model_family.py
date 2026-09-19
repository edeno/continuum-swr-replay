"""Runner-contract tests for the recommended marked-model family."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
SPEC = runner.RECOMMENDED_MODEL_SPECS[MODEL]
INPUT_HASHES = {
    "stop_table_sha256": "stop",
    "event_arm_table_sha256": "arm",
    "event_table_sha256": "event",
    "canonical_event_table_sha256": "event",
    "data_over_time_sha256": "dot",
}
GPU_RECORD = {
    "jax_backend": "cpu",
    "gpu_index": None,
    "gpu_uuid": None,
    "gpu_name": None,
}


def _contract(**overrides):
    kwargs = {
        "penalty_sigma": 3.0,
        "n_folds": 2,
        "fold_seed": 3,
        "design_batch_size": 4,
        "min_marked_events": 10,
        "brain_area": "mPFC",
        "input_hashes": INPUT_HASHES,
    }
    kwargs.update(overrides)
    return runner.build_fit_contract(SPEC, **kwargs)


def _context(n_units: int = 2) -> SimpleNamespace:
    fit_mask = pd.Series([True] * 6, index=np.arange(6) / 500.0)
    events = {
        event_class: pd.DataFrame({"event_id": [1, 2], "start_time": [0.002, 0.006]})
        for event_class in runner.MARKED_EVENT_CLASSES
    }
    return SimpleNamespace(
        fit_mask=fit_mask,
        counts=np.ones((6, n_units)),
        events=events,
        spline_basis=np.ones((251, 11)),
        lag_time=np.arange(-125, 126) / 500.0,
        lags=np.arange(-125, 126),
        cohort_info={"n_bins_fit": 6},
        input_provenance={
            "event_arm_table_sha256": "arm",
            "canonical_event_table_sha256": "event",
            "data_over_time_sha256": "dot",
        },
    )


def _result(context: SimpleNamespace) -> MarkedFamilyResults:
    n_units = context.counts.shape[1]
    grid = np.arange(251 * n_units, dtype=float).reshape(251, n_units)
    return MarkedFamilyResults(
        statistics=pd.DataFrame({"family_full_heldout_deviance": [1.0] * n_units}),
        typical_curves=dict.fromkeys(runner.MARKED_EVENT_CLASSES, grid),
        mark_curves={
            event_class: dict.fromkeys(SPEC.scientific_marks, grid)
            for event_class in runner.MARKED_EVENT_CLASSES
        },
        diagnostics={
            "n_units_full_fit_not_converged": 0,
            "n_units_nan_heldout_delta_by_comparison": {},
        },
    )


def _install_mocks(monkeypatch, context: SimpleNamespace) -> None:
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
    monkeypatch.setattr(
        runner,
        "build_marked_family_ladder",
        lambda *a, **k: SimpleNamespace(spec=SPEC),
    )
    monkeypatch.setattr(
        runner, "fit_marked_family_ladder", lambda *a, **k: _result(context)
    )


def test_fit_contract_pins_question_marks_and_no_selection_rule():
    contract = _contract()

    assert contract["scientific_marks"] == list(SPEC.scientific_marks)
    assert contract["nuisance_marks"] == list(SPEC.nuisance_marks)
    assert contract["model_question"] == SPEC.question
    assert "no firing-rate or p-value" in contract["neuron_inclusion"]
    assert "shuffle" not in contract["inference"]


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


def test_run_writes_only_target_curves_and_sidecar_last(tmp_path, monkeypatch):
    context = _context()
    _install_mocks(monkeypatch, context)

    succeeded = runner.run_analysis(
        "rat20200101_.nwb",
        "01_r1",
        model_names=(MODEL,),
        output_dir=tmp_path,
        log_directory=tmp_path / "logs",
        n_folds=2,
        fold_seed=3,
        gpu_record=GPU_RECORD,
    )

    assert succeeded
    artifact = runner.family_artifact_path(tmp_path, "rat20200101_01_r1_mPFC", MODEL)
    assert artifact.is_file()
    assert all(path.is_file() for path in runner.family_companion_paths(artifact, SPEC))
    record = json.loads(runner.fit_provenance_path(artifact).read_text())
    assert record["bundle_status"] == "complete"
    assert record["neuron_inclusion"].startswith("all recorded units")
    assert not list(tmp_path.glob("*.label_shuffle*"))
    table = pd.read_pickle(artifact)
    assert table.neuron_index.tolist() == [0, 1]
    assert table.fit_firing_rate_hz.tolist() == [500.0, 500.0]
