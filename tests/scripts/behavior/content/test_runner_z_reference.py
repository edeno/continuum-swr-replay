"""F2: the contrast runner persists its fit-time _z reference and it is reproducible.

The is_swr-contrast content runner standardizes over its *fit cohort* (its own scope,
distinct from the builder's pooled-global table _z) through the shared
``attach_standardized`` entry point. The runner is NOT repointed at the builder _z;
instead it persists the run's own means/SDs -- into the provenance and into a
``<artifact-stem>.z_reference.json`` sidecar -- so the standardized coefficients are
reproducible. This locks that in, plus the separation of the arm and patch scopes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import contrast_aggregation
from continuum_swr_replay.analysis.events.analysed_time import stamp_swr_source
from continuum_swr_replay.analysis.events.swr_source import SwrSourceError
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.io import write_table
from continuum_swr_replay.tables.standardization import (
    attach_standardized,
    load_reference,
    save_reference,
)
from scripts.behavior.content import fit_swr_contrast_content as mod


def _args(**overrides) -> argparse.Namespace:
    params = {
        "max_per_animal": None,
        "response": "fraction",
        "estimator": "conditional_poisson",
        "min_events_session_phi": 40,
        "outcome_stratum": "all",
    }
    params.update(overrides)
    return argparse.Namespace(**params)


def _patch_frame(n_events: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_patches = 3
    patches = np.tile(np.arange(n_patches), n_events)
    return pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(n_events), n_patches),
            "n_nonlocal_bins": np.repeat(rng.integers(1, 90, size=n_events), n_patches),
            "time_since_stop_start_s": np.repeat(
                rng.uniform(0.1, 8.0, size=n_events), n_patches
            ),
            "is_current_patch": (patches == 0).astype(float),
            "is_prev_patch": (patches == 1).astype(float),
            "is_swr": np.repeat(rng.random(n_events) < 0.5, n_patches),
            "periswitch_index_trials": np.repeat(rng.normal(size=n_events), n_patches),
            "patch_switch_fraction_last5_completed_trips": np.repeat(
                rng.uniform(size=n_events), n_patches
            ),
        }
    )


def test_provenance_includes_fit_time_z_reference():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"periswitch_index_trials": rng.normal(size=200)})
    _, reference = attach_standardized(
        frame.copy(),
        ["periswitch_index_trials"],
        support_label="fit_cohort_all_rows",
    )
    prov = mod._provenance(_args(), 218, {"patch_cohort": reference})

    assert prov["fit_time_z_reference"] == {"patch_cohort": reference}
    moments = reference["periswitch_index_trials"]
    assert moments["mean"] == float(np.nanmean(frame.to_numpy()))
    assert moments["std"] == float(np.nanstd(frame.to_numpy()))
    assert moments["support"] == "fit_cohort_all_rows"


def test_declared_sources_cover_every_standardization_the_runner_makes():
    """A new fit-time ``_z`` must be declared, or its moments go unrecorded."""
    assert set(contrast_aggregation._ARM_Z_SOURCES) == set(
        contrast_aggregation._ZSCALARS
    ) | {
        "log1p_time_since_stop_start_s",
        "log_n_nonlocal_bins",
    }
    assert set(contrast_aggregation._PATCH_Z_SOURCES) == {
        "periswitch_index_trials",
        "patch_switch_fraction_last5_completed_trips",
        "log1p_time_since_stop_start_s",
        "log_n_nonlocal_bins",
    }


def test_patch_cohort_is_its_own_standardization_scope():
    """The patch reference is estimated on patch rows, not inherited from the arms."""
    _, reference = contrast_aggregation._attach_patch_design_columns(_patch_frame())
    frame = _patch_frame()
    for source in ("periswitch_index_trials", "log_n_nonlocal_bins"):
        values = (
            np.log(frame["n_nonlocal_bins"].to_numpy(dtype=float))
            if source == "log_n_nonlocal_bins"
            else frame[source].to_numpy(dtype=float)
        )
        assert reference[source]["mean"] == float(np.nanmean(values))
        assert reference[source]["std"] == float(np.nanstd(values))


def test_reference_round_trips_through_the_sidecar(tmp_path):
    """Reloading the persisted sidecar reproduces the fit's ``_z`` exactly."""
    designed, reference = contrast_aggregation._attach_patch_design_columns(
        _patch_frame()
    )
    path = mod.z_reference_path(tmp_path / "swr_contrast_content_patch.pkl")
    save_reference(reference, path)

    assert path.name == "swr_contrast_content_patch.z_reference.json"
    replayed, _ = attach_standardized(
        designed.loc[:, list(contrast_aggregation._PATCH_Z_SOURCES)].copy(),
        list(contrast_aggregation._PATCH_Z_SOURCES),
        reference=load_reference(path),
        support_label="fit_cohort_all_rows",
    )
    for source in contrast_aggregation._PATCH_Z_SOURCES:
        np.testing.assert_allclose(
            replayed[f"{source}_z"].to_numpy(),
            designed[f"{source}_z"].to_numpy(),
            atol=1e-12,
        )


def test_swr_content_assembly_rejects_legacy_uncertainty_reference(tmp_path):
    """The contrast runner cannot consume the shipped nine-arm uncertainty unit."""
    frame = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_date": [1, 1],
            "epoch": ["01_r1", "01_r1"],
            "event_id": [1, 1],
            "candidate_arm": [0, 1],
            "occupancy_state": ["at_well", "at_well"],
            "reward_belief_variance_candidate_arm_preoutcome": [1.0, 2.0],
            "reward_belief_variance_candidate_arm_preoutcome_z": [-1.0, 1.0],
        }
    )
    frame = stamp_swr_source(stamp_covariate_schema(frame, table="event_arm"))
    write_table(frame, tmp_path / "event_arm_table.parquet")
    (tmp_path / "standardization_reference.json").write_text(
        json.dumps(
            {
                "reward_belief_variance_candidate_arm_preoutcome": {
                    "mean": 1.5,
                    "std": 0.5,
                }
            }
        )
    )
    fit_inputs = {
        **contrast_aggregation.contrast_fit_inputs(),
        "event_arm_table": tmp_path / "event_arm_table.parquet",
    }

    with pytest.raises(ValueError, match="stale or invalid"):
        contrast_aggregation.assemble_arm(
            fit_inputs, tmp_path / "standardization_reference.json"
        )


def test_swr_content_assembly_refuses_a_table_from_another_swr_source(
    tmp_path, monkeypatch
):
    """Its is_swr would be another source's."""
    frame = pd.DataFrame({"event_id": [1], "candidate_arm": [0]})
    frame.attrs["swr_source"] = "kay@another_run+artifacts@none"
    write_table(frame, tmp_path / "event_arm_table.parquet")
    fit_inputs = {
        **contrast_aggregation.contrast_fit_inputs(),
        "event_arm_table": tmp_path / "event_arm_table.parquet",
    }
    monkeypatch.setattr(
        contrast_aggregation, "require_covariate_schema", lambda *a, **k: None
    )

    with pytest.raises(SwrSourceError, match="another_run"):
        contrast_aggregation.assemble_arm(
            fit_inputs, tmp_path / "standardization_reference.json"
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [([np.nan, np.nan], "no finite values"), ([1.0, np.inf], "infinity")],
)
def test_fit_time_standardization_rejects_invalid_support(values, message):
    with pytest.raises(ValueError, match=message):
        contrast_aggregation._standardize(
            pd.DataFrame({"periswitch_index_trials": values}),
            ["periswitch_index_trials"],
        )
