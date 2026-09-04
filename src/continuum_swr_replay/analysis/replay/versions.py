"""Centralized schema/version literals for the covariate pipeline.

Three version strings, each versioned independently (today v5 / v4 / v1) but
semantically related, so all three should be reviewed together whenever the
behavioral derivation or the covariate contract changes. ``assert_versions_
compatible`` checks each stamped/cached value against its own constant; it does
not enforce that the three move in lockstep, and they legitimately do not.

- ``COVARIATE_SCHEMA_VERSION`` -- the canonical-names + row-key contract stamped
  onto every table (``covariate_registry.stamp_covariate_schema``).
- ``BEHAVIORAL_CONTEXT_CACHE_VERSION`` -- the versioned behavioral-context cache
  key; a change forces a cache rebuild rather than silently reinterpreting old
  pickles (``prepare_data.load_behavioral_context``).
- ``MANIFEST_SCHEMA`` -- the regeneration-manifest format
  (``write_regeneration_manifest``).

They lived in three separate modules and drifted (the arch review found the
cache at one lineage while the contract named another). Defining them here, with
``assert_versions_compatible`` to check a stamped/cached value against the
declared constant, keeps the bump a one-file edit and makes a drift loud.
"""

from __future__ import annotations

COVARIATE_SCHEMA_VERSION = "covariate_contract_v5_dio_poke_trials"
BEHAVIORAL_CONTEXT_CACHE_VERSION = "comrie_photogate_trials_v4"
MANIFEST_SCHEMA = "covariate_regeneration_bundle_v1"

#: The three versions as one record, for the manifest and for logging.
VERSION_TRIPLE: dict[str, str] = {
    "covariate_schema_version": COVARIATE_SCHEMA_VERSION,
    "behavioral_context_cache_version": BEHAVIORAL_CONTEXT_CACHE_VERSION,
    "manifest_schema": MANIFEST_SCHEMA,
}


def assert_versions_compatible(
    *,
    covariate_schema_version: str | None = None,
    behavioral_context_cache_version: str | None = None,
) -> None:
    """Raise if a stamped/cached version does not match the declared constant.

    Pass whichever versions a table or cache actually carries; each is checked
    against the single centralized constant. A mismatch means an artifact was
    produced under a different contract than the code now declares -- a stale
    bundle, not something to reinterpret.
    """
    if (
        covariate_schema_version is not None
        and covariate_schema_version != COVARIATE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"covariate schema version {covariate_schema_version!r} does not match "
            f"the declared {COVARIATE_SCHEMA_VERSION!r}; the artifact predates the "
            "current covariate contract and must be regenerated"
        )
    if (
        behavioral_context_cache_version is not None
        and behavioral_context_cache_version != BEHAVIORAL_CONTEXT_CACHE_VERSION
    ):
        raise ValueError(
            f"behavioral cache version {behavioral_context_cache_version!r} does not "
            f"match the declared {BEHAVIORAL_CONTEXT_CACHE_VERSION!r}; rebuild the "
            "behavioral-context cache"
        )
