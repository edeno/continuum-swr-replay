"""Version literals are centralized and re-exported unchanged."""

from __future__ import annotations

import pytest

from continuum_swr_replay.tables import versions


def test_reexports_match_centralized_values():
    from continuum_swr_replay.analysis.behavior.tables.prepare_data import (
        BEHAVIORAL_CONTEXT_CACHE_VERSION,
    )
    from continuum_swr_replay.tables.covariate_registry import (
        COVARIATE_SCHEMA_VERSION,
    )

    assert COVARIATE_SCHEMA_VERSION == versions.COVARIATE_SCHEMA_VERSION
    assert BEHAVIORAL_CONTEXT_CACHE_VERSION == versions.BEHAVIORAL_CONTEXT_CACHE_VERSION


def test_version_triple_covers_all_three():
    assert set(versions.VERSION_TRIPLE) == {
        "covariate_schema_version",
        "behavioral_context_cache_version",
        "manifest_schema",
    }


def test_compatible_versions_pass():
    versions.assert_versions_compatible(
        covariate_schema_version=versions.COVARIATE_SCHEMA_VERSION,
        behavioral_context_cache_version=versions.BEHAVIORAL_CONTEXT_CACHE_VERSION,
    )


def test_mismatched_covariate_version_raises():
    with pytest.raises(ValueError, match="covariate schema version"):
        versions.assert_versions_compatible(covariate_schema_version="stale")


def test_mismatched_cache_version_raises():
    with pytest.raises(ValueError, match="behavioral cache version"):
        versions.assert_versions_compatible(behavioral_context_cache_version="stale")


def test_require_schema_rejects_old_version():
    """An old stamp raises through ``require_covariate_schema``."""
    pd = pytest.importorskip("pandas")
    from continuum_swr_replay.tables import covariate_registry as reg

    frame = reg.stamp_covariate_schema(
        pd.DataFrame({"stop_id": [1], "task_reward_probability_chosen_arm": [0.5]}),
        table="qvals",
    )
    frame.attrs["covariate_schema_version"] = "covariate_contract_v4_dio_poke_trials"

    with pytest.raises(ValueError, match="covariate schema version"):
        reg.require_covariate_schema(frame, table="qvals")


def test_stale_behavioral_cache_stamp_is_reported_not_raised():
    """A cache-version mismatch is a miss (a reason string), never an exception."""
    from continuum_swr_replay.analysis.behavior.tables import prepare_data

    assert prepare_data._behavioral_cache_version_error(None) is not None
    assert "comrie_photogate_trials_v3" in str(
        prepare_data._behavioral_cache_version_error("comrie_photogate_trials_v3")
    )
    assert (
        prepare_data._behavioral_cache_version_error(
            versions.BEHAVIORAL_CONTEXT_CACHE_VERSION
        )
        is None
    )
