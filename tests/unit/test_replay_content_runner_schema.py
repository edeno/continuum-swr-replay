"""Fail-closed schema checks for the replay-content model runner."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from continuum_swr_replay.analysis.events.swr_source import SwrSourceError
from continuum_swr_replay.analysis.replay.fit_provenance import check_fit_provenance
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.standardization import (
    attach_standardized,
    save_reference,
)
from scripts.content import fit_replay_content_models as runner


def _partial_ladder_dir(input_dir, *rungs: str):
    """Where a run that fits only ``rungs`` publishes.

    A partial ladder is fit on its own complete-case cohort, so it is keyed by
    the rungs it requested and never replaces part of the canonical ``models/``
    bundle.
    """
    return input_dir / f"models_{'-'.join(rungs)}"


def _canonical_arm_table(n_events: int = 2) -> pd.DataFrame:
    """Small canonical arm table with every default-ladder predictor."""
    arms = np.tile(np.arange(9), n_events)
    event_ids = np.repeat(np.arange(n_events), 9)
    frame = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "event_id": event_ids,
            "candidate_arm": arms,
            "arm_count_hard": (arms == (event_ids % 9)).astype(float),
            # One decoded bin per event, so the allocation fraction is the
            # count: each event puts all of its mass on one arm.
            "arm_fraction": (arms == (event_ids % 9)).astype(float),
            "occupancy_state": "at_well",
            "at_well": True,
            "occupancy_well_id": 0.0,
            "occupancy_well_arm": 0.0,
            "is_current_arm": (arms == 0).astype(float),
            "visit_recency_trials_event_leaf_centered": 0.0,
            "is_candidate_arm_never_visited": 0.0,
        }
    )
    requested = runner.default_content_model_names(runner.MODEL_HIERARCHY)
    for model_name in requested:
        for predictor in runner.MODEL_HIERARCHY[model_name].predictors:
            if predictor.startswith("arm_fe_") or predictor in frame:
                continue
            frame[predictor] = 0.0
    return stamp_swr_source(stamp_covariate_schema(frame, table="event_arm"))


def _canonical_patch_table(n_events: int = 2) -> pd.DataFrame:
    """Small canonical patch table with every default-ladder predictor."""
    patches = np.tile([1, 2, 3], n_events)
    event_ids = np.repeat(np.arange(n_events), 3)
    frame = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "event_id": event_ids,
            "patch": patches,
            "patch_count_hard": (patches == ((event_ids % 3) + 1)).astype(float),
            "patch_fraction": (patches == ((event_ids % 3) + 1)).astype(float),
            "occupancy_state": "at_well",
            "at_well": True,
            "occupancy_well_id": 0.0,
            "occupancy_well_arm": 0.0,
            "is_current_patch": (patches == 1).astype(float),
            "visit_recency_trials_patch": 0.0,
            "is_patch_never_visited": 0.0,
        }
    )
    requested = runner.default_content_model_names(runner.MODEL_PATCH)
    for model_name in requested:
        for predictor in runner.MODEL_PATCH[model_name].predictors:
            if predictor in frame:
                continue
            frame[predictor] = 0.0
    return stamp_swr_source(stamp_covariate_schema(frame, table="event_patch"))


def test_complete_bundle_creates_explicit_output_parent(tmp_path):
    """A nested staging target must publish without caller-created parents."""
    output_dir = tmp_path / "candidate" / "models"
    result = pd.DataFrame({"animal": ["a"], "model": ["M0"]})

    runner._write_complete_output_bundle(
        output_dir=output_dir,
        model_results={"M0": result},
        combined=result,
        cohort_audits=[],
        comparison=pd.DataFrame(),
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
    stale = _canonical_arm_table().drop(
        columns=[
            "occupancy_state",
            "at_well",
            "occupancy_well_id",
            "occupancy_well_arm",
            "is_candidate_arm_never_visited",
        ]
    )
    stale.to_pickle(input_dir / "event_arm_table.pkl")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("schema validation must run before model fitting")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_must_not_run)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    with pytest.raises(ValueError, match="current-location/novelty schema"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_legacy_file_is_not_an_implicit_canonical_fallback(tmp_path, monkeypatch):
    """A legacy pickle cannot write into the canonical model directory."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    pd.DataFrame({"event_uid": [1], "arm_count": [1]}).to_pickle(
        input_dir / "long_count.pkl"
    )
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    with pytest.raises(FileNotFoundError, match="Legacy fallback is disabled"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_complete_default_ladder_passes_preflight_without_dropping_rungs():
    """Every requested rung survives preflight when its real columns are present."""
    requested = runner.default_content_model_names(runner.MODEL_HIERARCHY)

    prepared, model_names, predictors, response_col, event_col = (
        runner.prepare_canonical_fit_request(
            _canonical_arm_table(),
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=requested,
            hierarchy_kind="arm",
            table_label="synthetic event_arm_table.pkl",
        )
    )

    assert model_names == requested
    assert "is_candidate_arm_never_visited" in predictors
    assert "is_candidate_arm_0" in prepared
    assert response_col == "arm_fraction"
    assert event_col == "event_id"


def test_a_table_from_another_swr_source_is_refused(tmp_path, monkeypatch):
    """Its events and is_swr would be another source's: nothing is fit."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = _canonical_arm_table()
    frame.attrs["swr_source"] = "kay@another_run+artifacts@none"
    frame.to_pickle(input_dir / "event_arm_table.pkl")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("the source check must run before model fitting")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_must_not_run)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    with pytest.raises(SwrSourceError, match="another_run"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_complete_requested_run_writes_every_requested_result(tmp_path, monkeypatch):
    """A complete canonical request reaches the real fitter and canonical outputs."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    runner.main()

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
    frame = _canonical_arm_table()
    frame.to_pickle(input_dir / "event_arm_table.pkl")
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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _record_fit)
    monkeypatch.setattr(
        runner, "run_full_hierarchy_comparison", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py"])

    runner.main()

    output_dir = input_dir / "models"
    assert tuple(calls) == requested
    assert all((output_dir / f"{model_name}.pkl").exists() for model_name in requested)
    combined = pd.read_pickle(output_dir / "all_models.pkl")
    assert tuple(combined["model"]) == requested


def test_missing_requested_predictor_rejects_the_whole_ladder(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().drop(columns="swr_x_is_current_arm").to_pickle(
        input_dir / "event_arm_table.pkl"
    )

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("the complete ladder must validate before fitting")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_must_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M5,M6"],
    )

    with pytest.raises(ValueError, match=r"M6=.*swr_x_is_current_arm"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M5", "M6")).exists()


def test_uncertainty_model_rejects_legacy_all_arm_standardization(
    tmp_path, monkeypatch
):
    """A named ``arm_uncertainty_z`` cannot bypass its leaf-support contract."""
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = _canonical_arm_table()
    frame["reward_belief_variance_candidate_arm_preoutcome"] = np.where(
        frame["candidate_arm"].lt(6), frame["candidate_arm"] + 1.0, 0.0
    )
    frame, legacy = attach_standardized(
        frame, ["reward_belief_variance_candidate_arm_preoutcome"]
    )
    frame.to_pickle(input_dir / "event_arm_table.pkl")
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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_must_not_run)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "Mu"])

    with pytest.raises(ValueError, match="stale or invalid"):
        runner.main()
    assert not (_partial_ladder_dir(input_dir, "Mu")).exists()


def test_uncertainty_model_accepts_matching_leaf_support_reference(tmp_path):
    frame = _canonical_arm_table()
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
    frame = _canonical_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "occupancy_state"] = "in_transit"
    frame.loc[event_mask, "at_well"] = False
    frame.loc[event_mask, ["occupancy_well_id", "occupancy_well_arm"]] = np.nan
    frame.loc[event_mask, "is_current_arm"] = 0.0
    frame.to_pickle(input_dir / "event_arm_table.pkl")

    def _fit_must_not_run(*args, **kwargs):
        pytest.fail("location contract validation must run before fitting")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_must_not_run)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    with pytest.raises(ValueError, match="one-hot at wells and NaN off-well"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_arm_current_indicator_must_match_occupancy_well_arm():
    frame = _canonical_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "is_current_arm"] = (
        frame.loc[event_mask, "candidate_arm"].eq(1).astype(float)
    )

    with pytest.raises(ValueError, match="must match occupancy_well_arm"):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt arm table",
        )


def test_unknown_location_events_are_accepted_with_null_at_well():
    """``unknown`` carries ``at_well = NA`` by construction, not by corruption.

    Events before the first completed trip have no observed location, so
    ``at_well`` is null while transit/mixed windows are explicitly False.
    A canonical table is entitled to contain them; rejecting the null would
    abort a whole session for a state this validator lists as valid.
    """
    frame = _canonical_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame["at_well"] = frame["at_well"].astype("boolean")
    frame.loc[event_mask, "occupancy_state"] = "unknown"
    frame.loc[event_mask, "at_well"] = pd.NA
    for column in ("occupancy_well_id", "occupancy_well_arm", "is_current_arm"):
        frame.loc[event_mask, column] = np.nan

    request = runner.prepare_canonical_fit_request(
        frame,
        hierarchy=runner.MODEL_HIERARCHY,
        model_names=("M0",),
        hierarchy_kind="arm",
        table_label="table with an unknown-location event",
    )

    assert request is not None


@pytest.mark.parametrize(
    ("state", "at_well_value"),
    [("at_well", pd.NA), ("in_transit", pd.NA), ("unknown", False)],
)
def test_at_well_null_pattern_must_track_location_state(state, at_well_value):
    """Null-ness is only permitted for ``unknown``, and required there."""
    frame = _canonical_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame["at_well"] = frame["at_well"].astype("boolean")
    frame.loc[event_mask, "occupancy_state"] = state
    frame.loc[event_mask, "at_well"] = at_well_value
    if state != "at_well":
        for column in (
            "occupancy_well_id",
            "occupancy_well_arm",
            "is_current_arm",
        ):
            frame.loc[event_mask, column] = np.nan

    with pytest.raises(ValueError, match="inconsistent with location_state"):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt arm table",
        )


def test_patch_current_indicator_must_match_mapped_occupancy_well_arm():
    frame = _canonical_patch_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "is_current_patch"] = (
        frame.loc[event_mask, "patch"].eq(2).astype(float)
    )

    with pytest.raises(ValueError, match="mapped occupancy_well_arm"):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_PATCH,
            model_names=("P0",),
            hierarchy_kind="patch",
            table_label="corrupt patch table",
        )


def test_location_reference_must_be_constant_across_candidate_rows():
    """Row-varying references cannot make a corrupt one-hot look valid."""
    frame = _canonical_arm_table()
    corrupt_row = frame["event_id"].eq(0) & frame["candidate_arm"].eq(1)
    # This row still has is_current=0 and arm=1, so the later row-wise
    # indicator check would pass (1 != 2) unless the event-level reference is
    # first required to be constant.
    frame.loc[corrupt_row, "occupancy_well_arm"] = 2.0

    with pytest.raises(ValueError, match="event context.*changes"):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="row-varying location table",
        )


@pytest.mark.parametrize("bad_count", [np.nan, np.inf, -1.0, 0.5])
def test_response_counts_must_be_finite_nonnegative_integers(bad_count):
    frame = _canonical_arm_table()
    frame.loc[0, "arm_count_hard"] = bad_count

    with pytest.raises(ValueError, match="finite nonnegative integer counts"):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt response table",
        )


def test_each_session_event_requires_the_exact_candidate_set():
    frame = _canonical_arm_table()
    frame.loc[frame.index[0], "candidate_arm"] = 1

    with pytest.raises(
        ValueError, match="exactly the expected candidate_arm candidates"
    ):
        runner.prepare_canonical_fit_request(
            frame,
            hierarchy=runner.MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt candidate table",
        )


def test_late_empty_rung_aborts_without_publishing_any_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_with_empty_late_rung)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0,M1"],
    )

    with pytest.raises(RuntimeError, match="M1.*no session fits"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0", "M1")).exists()


def test_late_fit_exception_leaves_existing_output_unchanged(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    output_dir = _partial_ladder_dir(input_dir, "M0", "M1")
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "previous_complete_run.txt"
    sentinel.write_text("unchanged")
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _fit_then_raise)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0,M1"],
    )

    with pytest.raises(RuntimeError, match="late numerical failure"):
        runner.main()

    assert sentinel.read_text() == "unchanged"
    assert sorted(path.name for path in output_dir.iterdir()) == [sentinel.name]


def test_successful_publish_replaces_stale_output_directory(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    output_dir = _partial_ladder_dir(input_dir, "M0")
    output_dir.mkdir(parents=True)
    (output_dir / "M7.pkl").write_bytes(b"stale later rung")
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    runner.main()

    assert not (output_dir / "M7.pkl").exists()
    assert (output_dir / "M0.pkl").exists()
    assert (output_dir / "all_models.pkl").exists()


def test_fit_result_must_report_exact_frozen_cohort_size(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _wrong_cohort_size)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    with pytest.raises(RuntimeError, match="expected 2 from the frozen cohort"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_published_ladder_uses_intersection_of_successful_sessions(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    frame = _canonical_arm_table(n_events=4)
    frame.loc[frame["event_id"].ge(2), "animal"] = "b"
    frame.to_pickle(input_dir / "event_arm_table.pkl")

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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        runner, "fit_model_per_session", _fit_with_one_late_session_failure
    )
    monkeypatch.setattr(
        runner, "run_full_hierarchy_comparison", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0,M1"],
    )

    runner.main()

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
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0", "--cv"],
    )

    # The primary fit now survives a failed CV, so the abort comes from CV
    # scoring nothing at all -- which means CV is broken, not marginal.
    with pytest.raises(RuntimeError, match="no scored session"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_requested_cv_metadata_is_required_at_publication(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

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

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "fit_model_per_session", _primary_without_cv)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0", "--cv"],
    )

    with pytest.raises(RuntimeError, match="CV result.*missing column"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M0")).exists()


def test_complete_requested_cv_is_published_with_primary_fit(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    # Five-fold CV requires at least ten events; exercise the real null-model
    # fit and CV path rather than only a mocked result dictionary.
    _canonical_arm_table(n_events=10).to_pickle(input_dir / "event_arm_table.pkl")
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0", "--cv"],
    )

    runner.main()

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


def test_requested_logit_failure_aborts_without_publishing(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "fit_model_per_session",
        lambda long_df, model_name, predictors, **kwargs: _one_session_primary_result(
            long_df, model_name
        ),
    )
    monkeypatch.setattr(runner, "fit_conditional_logit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M1", "--logit"],
    )

    with pytest.raises(RuntimeError, match="logit robustness fit failed"):
        runner.main()

    assert not (_partial_ladder_dir(input_dir, "M1")).exists()


def test_complete_requested_logit_is_staged_with_primary_fit(tmp_path, monkeypatch):
    input_dir = tmp_path / "replay_content"
    input_dir.mkdir()
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")

    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        runner,
        "fit_model_per_session",
        lambda long_df, model_name, predictors, **kwargs: _one_session_primary_result(
            long_df, model_name
        ),
    )
    monkeypatch.setattr(
        runner,
        "fit_conditional_logit",
        lambda group, **kwargs: {
            "n_events": group["event_id"].nunique(),
            "estimator": "conditional_logit_dominant",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M1", "--logit"],
    )

    runner.main()

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
    table = input_dir / "event_arm_table.pkl"
    _canonical_arm_table().to_pickle(table)
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_replay_content_models.py", "--models", "M0"])

    runner.main()

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
    _canonical_arm_table().to_pickle(input_dir / "event_arm_table.pkl")
    monkeypatch.setattr(runner, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_replay_content_models.py", "--models", "M0", "--response", "hard_count"],
    )

    runner.main()

    fraction_bundle = _partial_ladder_dir(input_dir, "M0")
    count_bundle = input_dir / "models_hard_count_M0"
    assert not fraction_bundle.exists()
    assert (count_bundle / "all_models.pkl").exists()
    result = pd.read_pickle(count_bundle / "all_models.pkl")
    assert result["estimator"].unique().tolist() == ["conditional_poisson_count"]
    assert result["response_col"].unique().tolist() == ["arm_count_hard"]
