# Phase 2 — Safe cleanups: one session_keys module, centralized versions, model_specs prose → docs

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Three independent, non-numerical cleanups bundled because none touches covariate/table math and each is verified by the existing suite alone. No regen byte-diff needed (no canonical table is rebuilt), but the full suite must stay green.

**Inputs to read first:**

- [session_keys contract](shared-contracts.md#session_keys) — the module to create and its invariants.
- `src/continuum_swr_replay/analysis/replay/covariate_registry.py:74,117` — `_ROW_KEYS` and `row_keys()` that back the new module.
- The ~20 session-key sites in overview.md "Session-key identity" — especially `analysis/hippocampus/event_table.py:77` (the `day` outlier).
- `src/continuum_swr_replay/analysis/replay/model_specs.py` — locate the prose blocks (module docstring + inter-spec narrative comments) vs the `ModelSpec` definitions; only prose moves.
- `covariate_registry.py:71-72`, `prepare_data.py:117`, `write_regeneration_manifest.py:527` — the version literals.

## Tasks

- **Create `src/continuum_swr_replay/session_keys.py`** per the contract. Then replace each of the ~20 local definitions with an import, migrating each call site to the matching typed helper (`session_key_set` / `n_sessions` / `session_key_frame`), preserving its current return type. `event_table.py:77` uses `SESSION_COLUMNS_DAY` — **do not rename its `day` column**. Delete the now-dead local `_SESSION_COL*` / `SESSION_COL*` definitions. Grep after: `grep -rn "SESSION_COL" src scripts` returns only `session_keys.py`, `event_table.py`'s import, and `covariate_registry._ROW_KEYS`.
- **Create `src/continuum_swr_replay/analysis/replay/versions.py`** holding the three literals (`COVARIATE_SCHEMA_VERSION`, `BEHAVIORAL_CONTEXT_CACHE_VERSION`, `manifest_schema` value) plus `assert_versions_compatible()` that checks the coupling the arch review flagged (the covariate contract's lineage substring matches the cache lineage). Re-export the two existing names from their current modules (`covariate_registry`, `prepare_data`) so importers don't break, but the *value* now lives in `versions.py`. Have `write_regeneration_manifest.py` call `assert_versions_compatible()` before writing a manifest.
- **Move the `model_specs.py` design-journal prose** to `docs/design/model_history.md` (create it; preserve the retraction/reversal narrative verbatim — it is the scientific audit trail). Leave `model_specs.py` as the terse `ModelSpec` catalog with a one-line docstring pointing at the history doc. Verify no code imported anything from the prose (it is comments/docstring, so nothing should).
- **Docs:** add a short "Session identity" note to `docs/README.md` pointing at `session_keys.py`, and a "Model catalog vs history" note pointing `model_specs.py` readers to `docs/design/model_history.md`.

## Deliberately not in this phase

- The publish chokepoint (phase 3) and any `write_table` call-site change — a version literal moving is not a write-path change.
- Splitting `tables.py` (phase 4). `tables.py` may import `session_keys`; it is not restructured here.
- Touching `model_specs.py`'s actual `ModelSpec` values — only prose moves.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_session_columns_sourced_from_registry` | `session_keys.SESSION_COLUMNS == row_keys(table="event")[:3]`; a registry change would propagate. |
| `test_session_key_helpers_types` | `session_key_set`→set, `n_sessions`→int, `session_key_frame`→DataFrame with columns in order. |
| `test_no_local_session_col_definitions` | Grep-style test: no `SESSION_COL* =` literal survives outside `session_keys.py` / registry. |
| `test_event_table_keeps_day_column` | The hippocampus event table still uses `day`; not renamed. |
| `test_assert_versions_compatible` | Compatible triple passes; a mismatched lineage raises. |
| `test_model_specs_catalog_intact` | Every `ModelSpec` name/predictor set is unchanged vs a snapshot of the current catalog. |
| existing suite | Green. |

## Fixtures

None beyond a small in-memory frame for the helper-type tests. The catalog-intact test snapshots the current `MODEL_*` structures at test-collection time.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Exactly one session-key definition remains; every migrated site kept its original return type.
- The `day` outlier is preserved, not normalized.
- Version *values* live only in `versions.py`; old names re-export.
- Only prose left `model_specs.py`; the catalog is byte-identical in its `ModelSpec` content.
- Doc notes added. Tests exercise behavior, not tautologies.
- No name references this plan.
