"""Runner contract tests for the descriptive marked-kernel analysis."""

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

from continuum_swr_replay.analysis.pfc.marked_glm import MarkedKernelResults

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "run_pfc_marked_kernels.py"
)
_SPEC = importlib.util.spec_from_file_location("run_pfc_marked_kernels", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

MARK = "current_arm_fraction"
EPOCH_IDENTIFIER = "rat20200101_01_r1_mPFC"
GPU_RECORD = {
    "jax_backend": "cpu",
    "gpu_index": None,
    "gpu_uuid": None,
    "gpu_name": None,
}
INPUT_HASHES = {
    "stop_table_sha256": "stop",
    "event_arm_table_sha256": "arm",
    "canonical_event_table_sha256": "event",
    "data_over_time_sha256": "dot",
    "swr_interval_source": "kay",
    "swr_events_sha256": "swr",
}


def _contract(**overrides):
    """Build the runner's fit contract with the fixtures' values."""
    kwargs = {
        "scientific_mark": MARK,
        "nuisance_marks": (runner.EVENT_DURATION_MARK,),
        "penalty_sigma": 3.0,
        "n_folds": 2,
        "fold_seed": 1,
        "n_shuffles": 100,
        "shuffle_seed": 2,
        "design_batch_size": 4,
        "brain_area": "mPFC",
        "input_hashes": INPUT_HASHES,
    }
    kwargs.update(overrides)
    return runner.build_fit_contract(**kwargs)


def _make_context(*, n_units: int = 2) -> SimpleNamespace:
    """Mock one prepared session: 4 fit bins, 251 fit lags, `n_units` units."""
    fit_mask = pd.Series([True, True, True, True], index=np.arange(4) / 500.0)
    counts = np.arange(4 * n_units, dtype=float).reshape(4, n_units) % 2
    events = {
        event_class: pd.DataFrame(
            {
                "event_id": [1, 2],
                "start_time": [0.002, 0.004],
                f"{MARK}_available": [True, True],
            }
        )
        for event_class in runner.MARKED_EVENT_CLASSES
    }
    return SimpleNamespace(
        fit_mask=fit_mask,
        counts=counts,
        events=events,
        spline_basis=np.ones((3, 1)),
        lag_time=np.arange(-125, 126) / 500.0,
        lags=np.arange(-125, 126),
        cohort_info={"n_bins_fit": 4},
        input_provenance={
            "event_arm_table_sha256": "arm",
            "canonical_event_table_sha256": "event",
            "data_over_time_sha256": "dot",
            "swr_interval_source": "kay",
            "swr_events_sha256": "swr",
        },
    )


def _make_diagnostics(*, n_not_converged: int = 0, n_nan: int = 0) -> dict:
    """Mock the diagnostics keys the runner audits."""
    return {
        "n_usable_shuffles": {},
        "n_units_full_fit_not_converged": n_not_converged,
        "n_units_nan_heldout_delta_by_class": dict.fromkeys(
            runner.MARKED_EVENT_CLASSES, n_nan
        ),
        "n_units_nan_tail_fraction_by_class": dict.fromkeys(
            runner.MARKED_EVENT_CLASSES, n_nan
        ),
        "n_marked_events_by_class": dict.fromkeys(runner.MARKED_EVENT_CLASSES, 2),
        "n_distinct_permutations_by_class": dict.fromkeys(
            runner.MARKED_EVENT_CLASSES, 2
        ),
    }


def _make_result(context, **diagnostics_kwargs) -> MarkedKernelResults:
    """Mock fitted results whose curves span the full 251-lag fit window."""
    n_units = context.counts.shape[1]
    n_lags = context.lag_time.size
    statistics = pd.DataFrame(
        {
            "marked_full_heldout_deviance": [10.0, 11.0][:n_units],
            "non_local_no_SWR: marked_label_shuffle_tail_fraction": [0.9, 0.01][
                :n_units
            ],
            "non_local_SWR: marked_label_shuffle_tail_fraction": [np.nan, 0.8][
                :n_units
            ],
        }
    )
    grid = np.arange(n_lags * n_units, dtype=float).reshape(n_lags, n_units)
    return MarkedKernelResults(
        statistics=statistics,
        typical_curves={
            name: grid + offset
            for offset, name in enumerate(runner.MARKED_EVENT_CLASSES)
        },
        mark_curves={
            name: -grid - offset
            for offset, name in enumerate(runner.MARKED_EVENT_CLASSES)
        },
        randomization_reference={
            name: np.zeros((1_000, n_units)) for name in runner.MARKED_EVENT_CLASSES
        },
        diagnostics=_make_diagnostics(**diagnostics_kwargs),
    )


def _install_session_mocks(monkeypatch, context, *, fit) -> None:
    """Replace every Spyglass-backed input with the mocked session."""
    monkeypatch.setattr(
        runner,
        "load_session_stop_cohort",
        lambda *a, **k: (
            pd.DataFrame({"start_time": [0.0], "end_time": [1.0]}),
            {"stop_table_sha256": "stop"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "record_inputs",
        lambda inputs: {
            "stop_table": {"sha256": "stop"},
            "event_arm_table": {"sha256": "arm"},
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
        runner, "prepare_marked_session_context", lambda *a, **k: context
    )
    monkeypatch.setattr(
        runner,
        "build_marked_model_ladder",
        lambda *a, **k: SimpleNamespace(events=context.events),
    )
    monkeypatch.setattr(
        runner, "assign_blocked_folds", lambda *a, **k: np.array([0, 0, 1, 1])
    )
    monkeypatch.setattr(runner, "fit_marked_kernel_ladder", fit)


def _run(tmp_path, **overrides):
    """Run the mocked session with the fixtures' contract values."""
    kwargs = {
        "brain_area": "mPFC",
        "output_dir": tmp_path / "out",
        "log_directory": tmp_path / "logs",
        "overwrite": False,
        "scientific_mark": MARK,
        "nuisance_marks": (runner.EVENT_DURATION_MARK,),
        "penalty_sigma": 3.0,
        "n_folds": 2,
        "n_shuffles": 100,
        "fold_seed": 1,
        "shuffle_seed": 2,
        "design_batch_size": 4,
        "event_arm_table_path": tmp_path / "arm.parquet",
        "gpu_record": GPU_RECORD,
    }
    kwargs.update(overrides)
    kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
    kwargs["log_directory"].mkdir(parents=True, exist_ok=True)
    return runner.run_analysis("rat20200101_.nwb", "01_r1", **kwargs)


def _artifact(tmp_path) -> Path:
    """Path of the bundle the mocked session writes."""
    return runner.marked_artifact_path(tmp_path / "out", EPOCH_IDENTIFIER, MARK)


def test_cli_defaults_to_100_shuffles_and_duration_adjustment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_pfc_marked_kernels.py"])

    args = runner.get_command_line_arguments()

    assert args.n_shuffles == 100
    assert args.nuisance_marks == [runner.EVENT_DURATION_MARK]
    assert args.scientific_mark == "current_arm_fraction"


@pytest.mark.parametrize("draws", [1, 19])
def test_cli_refuses_underresolved_nonzero_null(monkeypatch, draws):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pfc_marked_kernels.py", "--n-shuffles", str(draws)],
    )

    with pytest.raises(SystemExit):
        runner.get_command_line_arguments()


def test_bundle_is_current_requires_every_companion_file(tmp_path):
    artifact = tmp_path / f"session_{MARK}_marked_kernels.pkl"
    artifact.write_bytes(b"complete")
    for path in runner.marked_companion_paths(artifact):
        path.write_bytes(b"complete")
    contract = _contract()
    runner.fit_provenance_path(artifact).write_text(
        json.dumps({**contract, "n_units": 7})
    )

    assert runner._bundle_is_current(artifact, overwrite=False, fit_contract=contract)
    assert not runner._bundle_is_current(
        artifact, overwrite=True, fit_contract=contract
    )

    runner.marked_companion_paths(artifact)[-1].unlink()
    assert not runner._bundle_is_current(
        artifact, overwrite=False, fit_contract=contract
    )


def test_bundle_is_current_rejects_empty_or_malformed_components(tmp_path):
    artifact = tmp_path / f"session_{MARK}_marked_kernels.pkl"
    contract = _contract()
    artifact.touch()
    assert not runner._bundle_is_current(
        artifact, overwrite=False, fit_contract=contract
    )

    artifact.write_bytes(b"complete")
    for path in runner.marked_companion_paths(artifact):
        path.write_bytes(b"complete")
    for sidecar_text in ("not json", "[]", '{"inputs": []}'):
        runner.fit_provenance_path(artifact).write_text(sidecar_text)
        assert not runner._bundle_is_current(
            artifact, overwrite=False, fit_contract=contract
        )


def test_bundle_is_stale_when_one_contract_value_changes(tmp_path):
    artifact = tmp_path / f"session_{MARK}_marked_kernels.pkl"
    artifact.write_bytes(b"complete")
    for path in runner.marked_companion_paths(artifact):
        path.write_bytes(b"complete")
    runner.fit_provenance_path(artifact).write_text(
        json.dumps({**_contract(), "n_units": 7})
    )

    assert not runner._bundle_is_current(
        artifact, overwrite=False, fit_contract=_contract(penalty_sigma=1.0)
    )
    assert not runner._bundle_is_current(
        artifact,
        overwrite=False,
        fit_contract=_contract(input_hashes={**INPUT_HASHES, "stop_table_sha256": "x"}),
    )


def test_current_bundle_skips_before_scientific_preparation(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact = _artifact(tmp_path)
    artifact.write_bytes(b"complete")
    for path in runner.marked_companion_paths(artifact):
        path.write_bytes(b"complete")
    runner.fit_provenance_path(artifact).write_text(
        json.dumps({**_contract(), "n_units": 7})
    )
    context = _make_context()

    def fail_if_prepared(*args, **kwargs):
        raise AssertionError("scientific preparation should not run")

    _install_session_mocks(monkeypatch, context, fit=fail_if_prepared)
    monkeypatch.setattr(runner, "load_session_stop_cohort", fail_if_prepared)
    monkeypatch.setattr(runner, "prepare_marked_session_context", fail_if_prepared)

    assert _run(tmp_path)


def test_runner_saves_every_unit_and_uses_full_requested_shuffle_floor(
    monkeypatch, tmp_path
):
    context = _make_context()
    result = _make_result(context)
    captured = {}

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return result

    _install_session_mocks(monkeypatch, context, fit=fake_fit)
    monkeypatch.setattr(
        runner,
        "write_fit_provenance",
        lambda artifact, *args, **kwargs: artifact.with_suffix(".provenance.json"),
    )

    succeeded = _run(tmp_path, overwrite=True, n_shuffles=1_000)

    artifact = _artifact(tmp_path)
    saved = pd.read_pickle(artifact)
    assert succeeded
    assert len(saved) == 2
    assert saved.neuron_index.tolist() == [0, 1]
    assert saved["non_local_no_SWR: marked_label_shuffle_tail_fraction"].tolist() == [
        0.9,
        0.01,
    ]
    assert saved.fit_spike_count.dtype.kind == "i"
    assert captured["min_usable_shuffles"] == 1_000
    assert artifact.with_suffix(".mark_label_shuffle_reference.npz").exists()


def test_corpus_setting_keeps_its_own_shuffle_floor(monkeypatch, tmp_path):
    context = _make_context()
    result = _make_result(context)
    captured = {}

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return result

    _install_session_mocks(monkeypatch, context, fit=fake_fit)
    monkeypatch.setattr(
        runner,
        "write_fit_provenance",
        lambda artifact, *args, **kwargs: artifact.with_suffix(".provenance.json"),
    )

    assert _run(tmp_path, overwrite=True, n_shuffles=100)
    assert captured["min_usable_shuffles"] == 100


def test_saved_curves_are_sliced_to_the_report_window(monkeypatch, tmp_path):
    context = _make_context()
    result = _make_result(context)
    _install_session_mocks(monkeypatch, context, fit=lambda *a, **k: result)
    monkeypatch.setattr(
        runner,
        "write_fit_provenance",
        lambda artifact, *args, **kwargs: artifact.with_suffix(".provenance.json"),
    )

    assert _run(tmp_path, overwrite=True)

    artifact = _artifact(tmp_path)
    lag_time = np.load(artifact.with_suffix(".lag_time.npy"))
    report = np.abs(context.lag_time) <= runner.REPORT_HALF_WINDOW_S + 1e-12
    assert lag_time.size == 201
    assert np.all(np.abs(lag_time) <= runner.REPORT_HALF_WINDOW_S + 1e-12)
    np.testing.assert_allclose(lag_time, context.lag_time[report])
    for event_class in runner.MARKED_EVENT_CLASSES:
        np.testing.assert_allclose(
            np.load(artifact.with_suffix(f".{event_class}.typical.npy")),
            result.typical_curves[event_class][report],
        )
        np.testing.assert_allclose(
            np.load(artifact.with_suffix(f".{event_class}.mark_slope.npy")),
            result.mark_curves[event_class][report],
        )


def test_sidecar_round_trips_into_the_skip_check(monkeypatch, tmp_path):
    context = _make_context()
    result = _make_result(context)
    _install_session_mocks(monkeypatch, context, fit=lambda *a, **k: result)

    assert _run(tmp_path, overwrite=True)

    artifact = _artifact(tmp_path)
    sidecar = runner.fit_provenance_path(artifact)
    record = json.loads(sidecar.read_text())
    assert record["estimator"] == runner.ESTIMATOR
    assert record["n_units"] == 2
    assert record["n_events_with_scientific_mark_by_class"] == {
        name: int(context.events[name][f"{MARK}_available"].sum())
        for name in runner.MARKED_EVENT_CLASSES
    }
    assert record["nuisance_mark_definitions"] == {
        runner.EVENT_DURATION_MARK: "natural log of event duration in seconds"
    }
    assert runner._bundle_is_current(
        artifact, overwrite=False, fit_contract=_contract()
    )


def test_session_without_units_writes_a_marker_that_is_current(monkeypatch, tmp_path):
    context = _make_context(n_units=0)

    def fail_if_fit(*args, **kwargs):
        raise AssertionError("a session with no units must not be fit")

    _install_session_mocks(monkeypatch, context, fit=fail_if_fit)

    assert _run(tmp_path, overwrite=True)

    artifact = _artifact(tmp_path)
    assert not artifact.exists()
    assert not any(path.exists() for path in runner.marked_companion_paths(artifact))
    record = json.loads(runner.fit_provenance_path(artifact).read_text())
    assert record["n_units"] == 0
    assert runner._bundle_is_current(
        artifact, overwrite=False, fit_contract=_contract()
    )

    def fail_if_prepared(*args, **kwargs):
        raise AssertionError("a current marker must not be refit")

    monkeypatch.setattr(runner, "load_session_stop_cohort", fail_if_prepared)
    monkeypatch.setattr(runner, "prepare_marked_session_context", fail_if_prepared)
    assert _run(tmp_path)


def test_failed_fit_leaves_no_sidecar_or_stale_companion(monkeypatch, tmp_path):
    artifact = _artifact(tmp_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"stale")
    for path in runner.marked_companion_paths(artifact):
        path.write_bytes(b"stale")
    runner.fit_provenance_path(artifact).write_text(json.dumps({**_contract()}))
    context = _make_context()

    def raise_on_fit(*args, **kwargs):
        raise RuntimeError("fit blew up")

    _install_session_mocks(monkeypatch, context, fit=raise_on_fit)
    failure_sink: list[tuple[str, str]] = []

    assert _run(tmp_path, overwrite=True, failure_sink=failure_sink) is False
    assert not runner.fit_provenance_path(artifact).exists()
    assert not any(path.exists() for path in runner.marked_companion_paths(artifact))
    assert failure_sink == [("RuntimeError", "fit blew up")]


def test_nan_channels_are_logged(caplog):
    logger = logging.getLogger("marked_diagnostics_test")
    diagnostics = _make_diagnostics(n_not_converged=1, n_nan=3)

    with caplog.at_level(logging.INFO, logger=logger.name):
        certified = runner._audit_marked_diagnostics(logger, diagnostics, 100)

    assert certified
    messages = [record.getMessage() for record in caplog.records]
    assert any("did not converge" in message for message in messages)
    assert any("NaN held-out delta" in message for message in messages)
    assert any("NaN label-shuffle tail fraction" in message for message in messages)
    assert any("distinct mark permutations" in message for message in messages)


def test_too_many_nonconverged_units_refuses_the_bundle(monkeypatch, tmp_path, caplog):
    logger = logging.getLogger("marked_diagnostics_test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        assert not runner._audit_marked_diagnostics(
            logger, _make_diagnostics(n_not_converged=11), 100
        )

    context = _make_context()
    result = _make_result(context, n_not_converged=1)
    _install_session_mocks(monkeypatch, context, fit=lambda *a, **k: result)
    failure_sink: list[tuple[str, str]] = []

    assert _run(tmp_path, overwrite=True, failure_sink=failure_sink) is False

    artifact = _artifact(tmp_path)
    assert not artifact.exists()
    assert not runner.fit_provenance_path(artifact).exists()
    assert failure_sink[0][0] == "NonconvergedFullFit"
