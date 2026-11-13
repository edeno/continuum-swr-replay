"""Fail-closed schema checks for the replay-content model runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.swr_source import SwrSourceError
from continuum_swr_replay.analysis.replay_models.content import session_fits
from continuum_swr_replay.paths import PROCESSED_DATA_DIR_ENV
from continuum_swr_replay.runs.fit_provenance import check_fit_provenance
from continuum_swr_replay.tables.io import write_table
from continuum_swr_replay.tables.standardization import (
    attach_standardized,
    save_reference,
)
from scripts.replay_models.content import fit_replay_content_models as runner
from tests.support._content_tables import default_ladder_arm_table


def _use_data_root(monkeypatch, root):
    """Make `root` the data root, so every event source's input directory is
    under it."""
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(root))


def _partial_ladder_dir(input_dir, *rungs: str):
    """Where a run that fits only ``rungs`` publishes.

    A partial ladder is fit on its own complete-case cohort, so it is keyed by
    the rungs it requested and never replaces part of the canonical ``models/``
    bundle.
    """
    return input_dir / f"models_{'-'.join(rungs)}"


def test_complete_bundle_creates_explicit_output_parent(tmp_path):
    """A nested staging target must publish without caller-created parents."""
    output_dir = tmp_path / "candidate" / "models"
    result = pd.DataFrame({"animal": ["a"], "model": ["M0"]})

    runner._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"M0": result},
        combined=result,
        cohort_audits=[],
        logit_results={},
        fit_inputs={},
        estimator="conditional_poisson_fraction",
        cli=["fit_replay_content_models.py"],
    )

    assert (output_dir / "M0.pkl").exists()
    assert (output_dir / "all_models.pkl").exists()


def test_stale_canonical_table_aborts_before_fit_or_output(tmp_path, monkeypatch):
    """An old table cannot silently fit only the rungs it happens to satisfy."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    stale = default_ladder_arm_table().drop(
        columns=[
            "occupancy_state",
            "at_well",
            "occupancy_well_id",
            "occupancy_well_arm",
            "is_candidate_arm_never_visited",
        ]
    )
    write_table(stale, input_dir / "event_arm_table.parquet")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("schema validation must run before model fitting")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_must_not_run)

    with pytest.raises(
        ValueError, match="regenerate it with the current prepare script"
    ):
        runner.main(["--models", "M0"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_legacy_file_is_not_an_implicit_canonical_fallback(tmp_path, monkeypatch):
    """A legacy pickle cannot write into the canonical model directory."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    pd.DataFrame({"event_uid": [1], "arm_count": [1]}).to_pickle(
        input_dir / "long_count.pkl"
    )
    _use_data_root(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="Legacy fallback is disabled"):
        runner.main(["--models", "M0"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_a_table_from_another_swr_source_is_refused(tmp_path, monkeypatch):
    """Its events and is_swr would be another source's: nothing is fit."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = default_ladder_arm_table()
    frame.attrs["swr_source"] = "kay@another_run+artifacts@none"
    write_table(frame, input_dir / "event_arm_table.parquet")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("the source check must run before model fitting")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_must_not_run)

    with pytest.raises(SwrSourceError, match="another_run"):
        runner.main(["--models", "M0"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_complete_requested_run_writes_every_requested_result(tmp_path, monkeypatch):
    """A complete canonical request reaches the real fitter and canonical outputs."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")
    _use_data_root(monkeypatch, tmp_path)

    runner.main(["--models", "M0"])

    output_dir = _partial_ladder_dir(input_dir, "M0")
    assert (output_dir / "M0.pkl").exists()
    assert (output_dir / "all_models.pkl").exists()
    assert (output_dir / "all_models_cohort_audit.csv").exists()
    result = pd.read_pickle(output_dir / "all_models.pkl")
    assert result["model"].unique().tolist() == ["M0"]


def test_complete_default_ladder_is_dispatched_without_dropping_rungs(
    tmp_path, monkeypatch
):
    """The runner dispatches and persists the complete validated default ladder."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = default_ladder_arm_table()
    write_table(frame, input_dir / "event_arm_table.parquet")
    requested = runner.default_content_model_names(runner.MODEL_HIERARCHY)
    calls = []

    def _record_fit(long_df, model_name, predictors, **kwargs):
        calls.append(model_name)
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "model": [model_name],
                "n_events": [long_df["event_id"].nunique()],
                "estimator": ["conditional_poisson_fraction"],
                "response_col": ["arm_fraction"],
                "converged": [True],
                "separated": [False],
            }
        )

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _record_fit)
    monkeypatch.setattr(
        runner, "run_full_hierarchy_comparison", lambda *args, **kwargs: pd.DataFrame()
    )

    runner.main([])

    output_dir = input_dir / "models"
    assert tuple(calls) == requested
    assert all((output_dir / f"{model_name}.pkl").exists() for model_name in requested)
    combined = pd.read_pickle(output_dir / "all_models.pkl")
    assert tuple(combined["model"]) == requested


def test_missing_requested_predictor_rejects_the_whole_ladder(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(
        default_ladder_arm_table().drop(columns="swr_x_is_current_arm"),
        input_dir / "event_arm_table.parquet",
    )

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("the complete ladder must validate before fitting")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_must_not_run)

    with pytest.raises(ValueError, match=r"M6=.*swr_x_is_current_arm"):
        runner.main(["--models", "M5,M6"])

    assert not (_partial_ladder_dir(input_dir, "M5", "M6")).exists()


def test_uncertainty_model_rejects_legacy_all_arm_standardization(
    tmp_path, monkeypatch
):
    """A named ``arm_uncertainty_z`` cannot bypass its leaf-support contract."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = default_ladder_arm_table()
    frame["reward_belief_variance_candidate_arm_preoutcome"] = np.where(
        frame["candidate_arm"].lt(6), frame["candidate_arm"] + 1.0, 0.0
    )
    frame, legacy = attach_standardized(
        frame, ["reward_belief_variance_candidate_arm_preoutcome"]
    )
    write_table(frame, input_dir / "event_arm_table.parquet")
    # Simulate the shipped pre-fix sidecar: right mean/SD keys, no scientific
    # support metadata, and moments computed over all nine candidates.
    (input_dir / "standardization_reference.json").write_text(
        json.dumps(
            {
                "reward_belief_variance_candidate_arm_preoutcome": {
                    "mean": legacy["reward_belief_variance_candidate_arm_preoutcome"][
                        "mean"
                    ],
                    "std": legacy["reward_belief_variance_candidate_arm_preoutcome"][
                        "std"
                    ],
                }
            }
        )
    )

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("standardization validation must run before model fitting")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_must_not_run)

    with pytest.raises(ValueError, match="stale or invalid"):
        runner.main(["--models", "Mu"])
    assert not (_partial_ladder_dir(input_dir, "Mu")).exists()


def test_uncertainty_model_accepts_matching_leaf_support_reference(tmp_path):
    frame = default_ladder_arm_table()
    frame["reward_belief_variance_candidate_arm_preoutcome"] = np.where(
        frame["candidate_arm"].lt(6), frame["candidate_arm"] + 1.0, 0.0
    )
    frame, reference = attach_standardized(
        frame,
        ["reward_belief_variance_candidate_arm_preoutcome"],
        support_mask=frame["candidate_arm"].lt(6),
        support_label="leaf_arms",
    )
    save_reference(reference, tmp_path / "standardization_reference.json")

    runner.validate_requested_standardization(
        frame,
        union_predictors=("reward_belief_variance_candidate_arm_preoutcome_z",),
        input_dir=tmp_path,
        table_label="fresh event_arm_table",
    )


def test_numeric_current_indicator_off_well_fails_closed(tmp_path, monkeypatch):
    """Off-well zeros cannot silently admit undefined-current events to a fit."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = default_ladder_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "occupancy_state"] = "in_transit"
    frame.loc[event_mask, "at_well"] = False
    frame.loc[event_mask, ["occupancy_well_id", "occupancy_well_arm"]] = np.nan
    frame.loc[event_mask, "is_current_arm"] = 0.0
    write_table(frame, input_dir / "event_arm_table.parquet")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("location contract validation must run before fitting")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_must_not_run)

    with pytest.raises(ValueError, match="one-hot at wells and NaN off-well"):
        runner.main(["--models", "M0"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_late_empty_rung_aborts_without_publishing_any_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    def _fit_with_empty_late_rung(long_df, model_name, predictors, **kwargs):
        if model_name == "M1":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "model": [model_name],
                "n_events": [long_df["event_id"].nunique()],
                "estimator": ["conditional_poisson_fraction"],
                "response_col": ["arm_fraction"],
            }
        )

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_fits, "fit_model_per_session", _fit_with_empty_late_rung
    )

    with pytest.raises(RuntimeError, match="M1.*no session fits"):
        runner.main(["--models", "M0,M1"])

    assert not (_partial_ladder_dir(input_dir, "M0", "M1")).exists()


def test_late_fit_exception_leaves_existing_output_unchanged(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    output_dir = _partial_ladder_dir(input_dir, "M0", "M1")
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "previous_complete_run.txt"
    sentinel.write_text("unchanged")
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    def _fit_then_raise(long_df, model_name, predictors, **kwargs):
        if model_name == "M1":
            raise RuntimeError("late numerical failure")
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "model": [model_name],
                "n_events": [long_df["event_id"].nunique()],
                "estimator": ["conditional_poisson_fraction"],
                "response_col": ["arm_fraction"],
            }
        )

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _fit_then_raise)

    with pytest.raises(RuntimeError, match="late numerical failure"):
        runner.main(["--models", "M0,M1"])

    assert sentinel.read_text() == "unchanged"
    assert sorted(path.name for path in output_dir.iterdir()) == [sentinel.name]


def test_successful_publish_replaces_stale_output_directory(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    output_dir = _partial_ladder_dir(input_dir, "M0")
    output_dir.mkdir(parents=True)
    (output_dir / "M7.pkl").write_bytes(b"stale later rung")
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")
    _use_data_root(monkeypatch, tmp_path)

    runner.main(["--models", "M0"])

    assert not (output_dir / "M7.pkl").exists()
    assert (output_dir / "M0.pkl").exists()
    assert (output_dir / "all_models.pkl").exists()


def test_fit_result_must_report_exact_frozen_cohort_size(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    def _wrong_cohort_size(long_df, model_name, predictors, **kwargs):
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "model": [model_name],
                "n_events": [long_df["event_id"].nunique() - 1],
                "estimator": ["conditional_poisson_fraction"],
                "response_col": ["arm_fraction"],
            }
        )

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _wrong_cohort_size)

    with pytest.raises(RuntimeError, match="expected 2 from the frozen cohort"):
        runner.main(["--models", "M0"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_published_ladder_uses_intersection_of_successful_sessions(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = default_ladder_arm_table(n_events=4)
    frame.loc[frame["event_id"].ge(2), "animal"] = "b"
    write_table(frame, input_dir / "event_arm_table.parquet")

    def _fit_with_one_late_session_failure(long_df, model_name, predictors, **kwargs):
        rows = []
        for keys, group in long_df.groupby(list(runner.SESSION_COLUMNS)):
            if model_name == "M1" and keys[0] == "a":
                continue
            rows.append(
                {
                    **dict(zip(runner.SESSION_COLUMNS, keys, strict=True)),
                    "model": model_name,
                    "n_events": group["event_id"].nunique(),
                    "estimator": "conditional_poisson_fraction",
                    "response_col": "arm_fraction",
                }
            )
        return pd.DataFrame(rows)

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_fits, "fit_model_per_session", _fit_with_one_late_session_failure
    )
    monkeypatch.setattr(
        runner, "run_full_hierarchy_comparison", lambda *args, **kwargs: pd.DataFrame()
    )

    runner.main(["--models", "M0,M1"])

    output_dir = _partial_ladder_dir(input_dir, "M0", "M1")
    for model_name in ("M0", "M1"):
        result = pd.read_pickle(output_dir / f"{model_name}.pkl")
        assert result["animal"].tolist() == ["b"]
        assert result["n_events"].tolist() == [2]


def test_requested_cv_failure_aborts_without_publishing(tmp_path, monkeypatch):
    """CV that scores NO session is broken, and must not publish.

    The complementary case -- CV scoring *some* sessions -- must publish, with
    the primary coefficients keeping the full cohort. Letting a failed fold drop
    the session cost 5 of 213 sessions their coefficients purely because CV was
    requested, which made the reported effects depend on a diagnostic."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    # Two events are enough for the null primary fit but not 5-fold CV.
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")
    _use_data_root(monkeypatch, tmp_path)

    # The primary fit now survives a failed CV, so the abort comes from CV
    # scoring nothing at all -- which means CV is broken, not marginal.
    with pytest.raises(RuntimeError, match="no scored session"):
        runner.main(["--models", "M0", "--cv"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_requested_cv_metadata_is_required_at_publication(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    def _primary_without_cv(long_df, model_name, predictors, **kwargs):
        assert kwargs["run_cv"] is True
        return pd.DataFrame(
            {
                "animal": ["a"],
                "session_date": [1],
                "epoch": ["02_r1"],
                "model": [model_name],
                "n_events": [long_df["event_id"].nunique()],
                "estimator": ["conditional_poisson_fraction"],
                "response_col": ["arm_fraction"],
            }
        )

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(session_fits, "fit_model_per_session", _primary_without_cv)

    with pytest.raises(RuntimeError, match="CV result.*missing column"):
        runner.main(["--models", "M0", "--cv"])

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_complete_requested_cv_is_published_with_primary_fit(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    # Five-fold CV requires at least ten events; exercise the real null-model
    # fit and CV path rather than only a mocked result dictionary.
    write_table(
        default_ladder_arm_table(n_events=10), input_dir / "event_arm_table.parquet"
    )
    _use_data_root(monkeypatch, tmp_path)

    runner.main(["--models", "M0", "--cv"])

    result = pd.read_pickle(_partial_ladder_dir(input_dir, "M0") / "M0.pkl")
    assert result["n_events"].tolist() == [10]
    assert result["cv_n_folds"].tolist() == [5]
    assert np.isfinite(result["cv_ll_per_event_mean"]).all()


def _one_session_primary_result(long_df, model_name):
    return pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["02_r1"],
            "model": [model_name],
            "n_events": [long_df["event_id"].nunique()],
            "estimator": ["conditional_poisson_fraction"],
            "response_col": ["arm_fraction"],
        }
    )


def test_a_logit_rung_with_no_fitted_session_aborts_without_publishing(
    tmp_path, monkeypatch
):
    """A failed session is skipped, but a rung left with none still aborts."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_fits,
        "fit_model_per_session",
        lambda long_df, model_name, predictors, **kwargs: _one_session_primary_result(
            long_df, model_name
        ),
    )
    monkeypatch.setattr(
        session_fits,
        "conditional_logit_or_reason",
        lambda *args, **kwargs: (None, "refused"),
    )

    with pytest.raises(RuntimeError, match="produced no session fits"):
        runner.main(["--models", "M1", "--logit"])

    assert not (_partial_ladder_dir(input_dir, "M1")).exists()


def test_complete_requested_logit_is_staged_with_primary_fit(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")

    _use_data_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_fits,
        "fit_model_per_session",
        lambda long_df, model_name, predictors, **kwargs: _one_session_primary_result(
            long_df, model_name
        ),
    )
    monkeypatch.setattr(
        session_fits,
        "conditional_logit_or_reason",
        lambda group, **kwargs: (
            {
                "n_events": group["event_id"].nunique(),
                "estimator": "conditional_logit_dominant",
            },
            "",
        ),
    )

    runner.main(["--models", "M1", "--logit"])

    output_dir = _partial_ladder_dir(input_dir, "M1")
    assert (output_dir / "M1.pkl").exists()
    logit = pd.read_pickle(output_dir / "logit" / "M1_logit.pkl")
    assert logit["model"].tolist() == ["M1"]
    assert logit["n_events"].tolist() == [2]


def test_published_bundle_carries_the_hash_of_the_table_it_was_fit_on(
    tmp_path, monkeypatch
):
    """The sidecar is what lets a reader refuse a fit whose table has moved."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    table = input_dir / "event_arm_table.parquet"
    write_table(default_ladder_arm_table(), table)
    _use_data_root(monkeypatch, tmp_path)

    runner.main(["--models", "M0"])

    output_dir = _partial_ladder_dir(input_dir, "M0")
    inputs = {"event_arm_table": table}
    assert check_fit_provenance(output_dir / "all_models.pkl", inputs) == []
    assert check_fit_provenance(output_dir / "M0.pkl", inputs) == []
    record = json.loads((output_dir / "all_models.provenance.json").read_text())
    assert record["estimator"] == "conditional_poisson_fraction"

    table.write_bytes(table.read_bytes() + b"\n")
    assert check_fit_provenance(output_dir / "all_models.pkl", inputs) == [
        "event_arm_table"
    ]


def test_count_run_publishes_beside_the_fraction_bundle(tmp_path, monkeypatch):
    """A count fit is on a different scale, so it gets a different directory."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    write_table(default_ladder_arm_table(), input_dir / "event_arm_table.parquet")
    _use_data_root(monkeypatch, tmp_path)

    runner.main(["--models", "M0", "--response", "hard_count"])

    fraction_bundle = _partial_ladder_dir(input_dir, "M0")
    count_bundle = input_dir / "models_hard_count_M0"
    assert not fraction_bundle.exists()
    assert (count_bundle / "all_models.pkl").exists()
    result = pd.read_pickle(count_bundle / "all_models.pkl")
    assert result["estimator"].unique().tolist() == ["conditional_poisson_count"]
    assert result["response_col"].unique().tolist() == ["arm_count_hard"]
