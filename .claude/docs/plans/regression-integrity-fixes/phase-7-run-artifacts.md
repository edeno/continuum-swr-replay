# Phase 7 — Versioned run artifacts and atomic promotion

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#atomic-runs)

Replace fixed-path, mixed-generation result loading with one manifest-validated run abstraction. This
phase changes storage/resolution only; it does not change fitted numbers. It ships before every
number-changing phase so later refits cannot recreate the stale-artifact problem.

**Inputs to read first:**

- [scripts/content/fit_replay_content_models.py:625](../../../../scripts/content/fit_replay_content_models.py#L625) — direct per-model, combined, comparison writes.
- [scripts/rate/fit_per_bin_rate_models.py:319](../../../../scripts/rate/fit_per_bin_rate_models.py#L319) — fixed `all_models.pkl` write.
- [scripts/rate/fit_per_stop_rate_models.py:168](../../../../scripts/rate/fit_per_stop_rate_models.py#L168) — fixed Family-A write.
- [scripts/content/fit_swr_contrast_content.py:529](../../../../scripts/content/fit_swr_contrast_content.py#L529) — multiple outputs plus the richest current provenance fragment.
- [scripts/rate/fit_swr_contrast_rate.py:410](../../../../scripts/rate/fit_swr_contrast_rate.py#L410) — fixed contrast outputs and limited metadata.
- [scripts/content/compare_event_source_models.py:536](../../../../scripts/content/compare_event_source_models.py#L536) — cohort provenance and fixed pickle write.
- [scripts/rate/summarize_per_bin_rate_models.py:155](../../../../scripts/rate/summarize_per_bin_rate_models.py#L155) and [scripts/rate/plot_per_bin_rate_summary.py:245](../../../../scripts/rate/plot_per_bin_rate_summary.py#L245) — representative direct consumers to migrate.

**Contracts referenced:**

- [Run artifact and promotion contract](shared-contracts.md#run-artifact-contract) — exact manifest and promotion rules; do not add a legacy fallback.

**Designs referenced:** [atomic run creation and promotion](designs.md#atomic-runs).

## Tasks

- Add `src/continuum_swr_replay/analysis/replay/run_artifacts.py` implementing the context-managed
  staging/promote API, streaming SHA-256 fingerprints, stable model-spec serialization, exact output
  schema/key validation, `CURRENT.json` resolution, and `RunValidationError` with actionable reasons.
- Add `analysis/replay/run_schema.py` containing the manifest schema version, required fields, and
  analysis-specific expected cell keys. Keep schema data separate from filesystem mechanics so tests
  can validate manifests without writing a full run.
- Migrate the canonical producers listed above, plus `fit_uncertainty_rate.py` and
  `fit_swr_power_model.py`, to write through `AnalysisRun`. Write every dataframe/table atomically;
  declare expected model/population/session coverage before fitting; record every skipped/failed cell.
- Migrate canonical result consumers and paper-facing plotters under `scripts/{content,rate,figures}/`
  from hard-coded pickle paths to `resolve_current_run(analysis)`. Use `rg` to enumerate every read of
  `all_models.pkl`, `swr_contrast_*.pkl`, choice comparison pickles, and power results. Delete the old
  direct-read code; absence of a promoted run is an error naming the command needed to create it.
- Generate `model_spec_sha256` from the actual serialized registry entries used by the runner—not the
  whole Python file—and record response, family/link, offset, standardized-column references, CLI,
  software versions, input hashes, and dirty-worktree hash.
- Add a migration/quarantine command `scripts/qa/audit_legacy_regression_artifacts.py` that inventories
  fixed-path artifacts and their dates/schemas but never promotes them. Phase 13 removes/quarantines
  them after the clean run; this phase only prevents their consumption.
- Update `docs/README.md` “Where results live” and runner module docstrings to document run directories,
  `CURRENT.json`, manifests, and the lack of legacy compatibility.

## Deliberately not in this phase

- Exact comparison cohorts/folds and their hashes (Phases 9a/9b); manifest fields may be empty only for
  an analysis that does not yet perform selection.
- Rebuilding tables or refitting models. Use tiny synthetic/smoke outputs solely to prove storage.
- PFC sidecar details; coordinate the common manifest fields with the dedicated PFC plan, but do not
  edit PFC GLMs here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_staged_run_exception_does_not_promote` | exception leaves `CURRENT.json` unchanged and incomplete staging unreadable |
| `test_manifest_rejects_missing_or_extra_schema` | unknown version, missing required fields/columns, incompatible extras fail closed |
| `test_output_hash_and_run_id_are_verified` | corrupt bytes or mixed run IDs cannot resolve |
| `test_expected_cells_require_result_or_failure` | every declared cell is present or has an explicit failure record |
| `test_atomic_promotion_updates_pointer_last` | completed run is renamed before `CURRENT.json`; interrupted update preserves old run |
| `test_dirty_run_records_diff_hash` | dirty state is explicit and reproducible, never silently reported clean |
| migrated producer smoke | one synthetic/single-session run resolves through the strict loader and direct legacy file does not |

## Fixtures

Use `tmp_path` run roots with two tiny DataFrames, a minimal model-spec registry, one intentionally
corrupt output, and an existing promoted run to test interrupted replacement. One real-session smoke
uses `--max-per-animal 1` where already supported; do not run a full fit.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- All manifest/promotion failure tests genuinely fail when validation is removed.
- No canonical consumer retains a hard-coded fallback to fixed legacy pickles.
- The staging directory cannot become current before every expected output/audit validates.
- No scientific formula, cohort, or coefficient changes slipped into this storage-only PR.
- User-facing result-location docs were updated in this PR.
- Every task is implemented, the deliberately excluded scientific changes remain out of scope, and old direct-write/read paths named above are removed.
- Validation tests pass, slow tests are marked, and tests exercise interruption/corruption rather than only their own mocks.
- Production names/docstrings do not mention this plan or phase numbers.
