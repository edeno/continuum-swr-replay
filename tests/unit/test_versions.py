"""Version literals are centralized and re-exported unchanged."""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.replay import versions


def test_reexports_match_centralized_values():
    from continuum_swr_replay.analysis.replay.covariate_registry import (
        COVARIATE_SCHEMA_VERSION,
    )
    from continuum_swr_replay.analysis.replay.prepare_data import (
        BEHAVIORAL_CONTEXT_CACHE_VERSION,
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
