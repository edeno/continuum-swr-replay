# Phase 7 — Source-valid inputs, versioned run artifacts, and atomic promotion

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design](designs.md#atomic-runs)

Make every session input prove its Spyglass source, valid time envelope, cache freshness, and decoder
alignment, then replace fixed-path, mixed-generation result loading with one manifest-validated run
abstraction. This phase builds validators and storage plumbing but does not promote regenerated
scientific results; Phase 13 performs the selective refresh/re-decode and number-changing cutover.

**Inputs to read first:**

- [src/continuum_swr_replay/data_loaders/position.py:106-115](../../../../src/continuum_swr_replay/data_loaders/position.py#L106) and [:221-293](../../../../src/continuum_swr_replay/data_loaders/position.py#L221) — interpolation, position/track-graph source fallbacks, and valid-interval handling to harden.
- [src/continuum_swr_replay/data_loaders/trials.py:191-258](../../../../src/continuum_swr_replay/data_loaders/trials.py#L191) and [:320-352](../../../../src/continuum_swr_replay/data_loaders/trials.py#L320) — endpoint clamping/nearest-well assignment and pump-to-visit mapping.
- [src/continuum_swr_replay/decoding_common.py:153-167](../../../../src/continuum_swr_replay/decoding_common.py#L153) and [scripts/decoding/run_decoding_bandit.py:346-407](../../../../scripts/decoding/run_decoding_bandit.py#L346) — presence-only decoder reuse and derived-data rewrite path.
- [scripts/decoding/run_decoding_contfrag.py:315-355](../../../../scripts/decoding/run_decoding_contfrag.py#L315) — the same presence-only reuse and rewrite path for ContFrag decodes.
- [scripts/decoding/prepare_contfrag_replay_content_data.py:95-148](../../../../scripts/decoding/prepare_contfrag_replay_content_data.py#L95) and [:391-404](../../../../scripts/decoding/prepare_contfrag_replay_content_data.py#L391) — glob-based ContFrag discovery and fixed table writes.
- [scripts/decoding/build_timebin_tables.py:83-105](../../../../scripts/decoding/build_timebin_tables.py#L83) — skip-on-error discovery and fixed onset/time-bin writes.
- [src/continuum_swr_replay/analysis/replay/prepare_data.py:150-195](../../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L150) — presence-only, unversioned trial cache.
- [src/continuum_swr_replay/analysis/replay/prepare_data.py:495-506](../../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L495) and [src/continuum_swr_replay/analysis/behavior/qvals_table.py:441-447](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L441) — fixed canonical-table/Q writes that must join the same generation.
- [scripts/content/fit_replay_content_models.py:625](../../../../scripts/content/fit_replay_content_models.py#L625) — direct per-model, combined, comparison writes.
- [scripts/rate/fit_per_bin_rate_models.py:319](../../../../scripts/rate/fit_per_bin_rate_models.py#L319) — fixed `all_models.pkl` write.
- [scripts/rate/fit_per_stop_rate_models.py:168](../../../../scripts/rate/fit_per_stop_rate_models.py#L168) — fixed Family-A write.
- [scripts/content/fit_swr_contrast_content.py:529](../../../../scripts/content/fit_swr_contrast_content.py#L529) — multiple outputs plus the richest current provenance fragment.
- [scripts/rate/fit_swr_contrast_rate.py:410](../../../../scripts/rate/fit_swr_contrast_rate.py#L410) — fixed contrast outputs and limited metadata.
- [scripts/content/compare_event_source_models.py:536](../../../../scripts/content/compare_event_source_models.py#L536) — cohort provenance and fixed pickle write.
- [scripts/rate/summarize_per_bin_rate_models.py:155](../../../../scripts/rate/summarize_per_bin_rate_models.py#L155) and [scripts/rate/plot_per_bin_rate_summary.py:245](../../../../scripts/rate/plot_per_bin_rate_summary.py#L245) — representative direct consumers to migrate.

**Contracts referenced:**

- [Session input-bundle contract](shared-contracts.md#session-input-contract) — exact source, timing,
  DIO/trial, cache, and decoder identities; do not weaken or add an allow-by-presence fallback.
- [Run artifact and promotion contract](shared-contracts.md#run-artifact-contract) — exact manifest and promotion rules; do not add a legacy fallback.

**Designs referenced:** [session input validation and selective refresh](designs.md#session-input-validation)
and [atomic run creation and promotion](designs.md#atomic-runs).

## Tasks

- Add `src/continuum_swr_replay/analysis/replay/source_integrity.py` and
  `scripts/qa/audit_replay_source_integrity.py` implementing the session input-bundle contract,
  complete expected-session accounting, typed failure reasons, and a machine-readable
  `source_input_audit.parquet`. The audit must reproduce the current out-of-interval/frozen-tail
  defect before any fix, but those current counts remain report values rather than unit-test literals.
- Snapshot the exact task/brain-area session query as `expected_sessions.parquet` and hash it as a run
  input. Add a checked-in `config/regression_session_exclusions.yaml` with scoped analysis family,
  typed reason code, reviewer, approval date, and source evidence. An expected session is complete only
  when it has an `ok` audit row or an already-approved, in-scope exclusion; a new query/QC/cache failure
  can never be converted to an exclusion during the production run.
- Refactor `get_position_info` to count the restricted `PositionOutput().merge_restrict()` and
  `IntervalLinearizedPosition` relations before fetch: exactly one preferred row records its
  merge/graph identity, zero rows deliberately enters a separately cardinality-checked legacy/animal
  fallback, and multiple rows fail. Replace both broad `except DataJointError` fallbacks, fingerprint
  graph topology/edge order/spacing, apply the union of all valid intervals, and measure raw frame
  gaps before linear interpolation separately within each declared valid interval. A gap over
  0.100 seconds inside an interval returns the contract's explicit QC failure; a declared gap between
  valid intervals does not.
- Change trial endpoint assignment to reject timestamps outside the position envelope, return the
  endpoint-to-well distance, and fail above 8 cm. Validate that each pump belongs to exactly one visit;
  treat an empty pump stream as an explicit session exception rather than silently declaring every
  trial an omission.
- Replace `_load_trials_track_graph`'s session-name pickle with a schema-versioned cache bundle keyed by
  source-query/index/DIO hashes plus the trial-builder schema. A mismatch rebuilds from source;
  `--from-cache` fails if source identity cannot be verified. Persist the resulting session-input
  manifest beside the cache and require its hash in downstream table manifests.
- Extend both bandit and ContFrag decoder cache checks beyond the three filenames: validate a decoding
  sidecar containing source/query identity, detector/ripple query, training type, parameters, software,
  results-time hash, and all output hashes. A legacy result may be attested and rewrapped only when
  those identities can be reconstructed from stored evidence and exactly match the locked run spec;
  time equality alone is insufficient. Otherwise it is audit-only and Phase 13 re-decodes it.
- Add an explicit `--output-root` to both decoder runners and every downstream table builder. Tier-1
  derived refreshes and tier-2 `--overwrite` decodes write only beneath a new candidate root; tests
  assert that no legacy input byte changes. Replace glob discovery with the expected-session manifest
  so a missing result becomes a structured failure instead of an omitted session.
- Add `src/continuum_swr_replay/analysis/replay/run_artifacts.py` implementing the context-managed
  staging/finalization API plus `freeze_release_candidate`, global `promote_analysis_release`,
  `publish_release_outputs`, streaming SHA-256 fingerprints, stable model-spec serialization, exact
  output schema/key validation, release-aware `CURRENT.json` resolution, generic optional-component
  resolution, and `RunValidationError` with actionable reasons. A family finalization or frozen
  candidate cannot advance canonical state.
- Add `analysis/replay/run_schema.py` containing the manifest schema version, required fields, and
  analysis-specific expected cell keys. Keep schema data separate from filesystem mechanics so tests
  can validate manifests without writing a full run.
- Migrate the canonical producers listed above, both decoder-preparation paths, the replay and ContFrag
  table writers, the onset/time-bin builder, `qvals_table.py`, `fit_uncertainty_rate.py`, and
  `fit_swr_power_model.py` to the staged output/run APIs. Write every dataframe/table atomically;
  declare expected model/population/session coverage before work starts; record every skipped/failed
  cell rather than `continue`-dropping it. The replay/ContFrag/Q/onset table generation hashes are
  required inputs to the final global release.
- Migrate canonical result consumers and paper-facing plotters under `scripts/{content,rate,figures}/`
  from hard-coded pickle paths to the run resolver. Use `rg` to enumerate every read of
  `all_models.pkl`, `swr_contrast_*.pkl`, choice comparison pickles, and power results. Production
  defaults use `resolve_current_run(analysis)` / `resolve_release_component(component)` and fail when
  no global release exists; development, candidate-report, and smoke commands may select a finalized
  noncanonical run or frozen candidate only with an explicit `--run-id` or
  `--release-candidate`. PFC Phase 4 owns adapting Figure 4 to the generic component resolver; this
  phase owns the resolver itself. Delete the old direct-read code and never make a legacy path the
  implicit fallback.
- Generate `model_spec_sha256` from the actual serialized registry entries used by the runner—not the
  whole Python file—and record response, family/link, offset, standardized-column references, CLI,
  software versions, input hashes, and dirty-worktree hash.
- Add a migration/quarantine command `scripts/qa/audit_legacy_regression_artifacts.py` that inventories
  fixed-path artifacts and their dates/schemas but never promotes them. Phase 13 quarantines
  them after the clean run; this phase only prevents their consumption. The Phase-13 action is a
  recoverable move into one hash-indexed, read-only legacy archive, not deletion.
- Update `docs/README.md` “Where results live,” data-loader/runner module docstrings, and
  `docs/known_issues/` with source-bundle fields, QC/exclusion reasons, run directories,
  `CURRENT.json`, manifests, and the lack of legacy compatibility.

## Deliberately not in this phase

- Exact comparison cohorts/folds and their hashes (Phases 9a/9b); manifest fields may be empty only for
  an analysis that does not yet perform selection.
- Decoder `training_type`/artifact-identifier semantics, ripple-query fixes, and trajectory-label
  repairs remain owned by [`review-findings-fixes`](../review-findings-fixes/PLAN.md). Consume those
  corrected identities in the manifest; do not reimplement their scientific fixes here.
- Refreshing/re-decoding production sessions, rebuilding tables, or refitting models. Use synthetic
  and read-only real-data audits solely to prove the contracts; explicit finalized test runs keep this
  PR independently usable, while Phase 13 owns all production refresh and the first global promotion.
- Changing decoder/detector algorithms or silently trimming saved decoder arrays to match position.
- PFC sidecar details; coordinate the common manifest fields with the dedicated PFC plan, but do not
  edit PFC GLMs here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_valid_interval_union_excludes_internal_and_outer_gaps` | only samples in an actual declared interval survive; outer-envelope slicing cannot pass |
| `test_position_and_graph_source_cardinality_and_fallback` | zero preferred rows records the exact checked fallback; one records merge/graph identity; duplicates and connection/query errors fail |
| `test_long_raw_position_gap_fails_before_interpolation` | a >0.100-s gap cannot be linearly bridged into valid decoder exposure |
| `test_declared_inter_interval_gap_is_not_raw_gap_failure` | only gaps internal to a valid interval are compared with 0.100 s |
| `test_trial_boundaries_and_pumps_fail_closed` | out-of-range/greater-than-8-cm endpoints, unmatched/ambiguous pumps, and unapproved empty pump streams fail |
| `test_session_cache_invalidates_on_source_or_builder_change` | changed interval/index/DIO/builder hash cannot hit the trial cache |
| `test_decoder_bundle_requires_equal_time_hashes` | results, position, and derived-data axes must agree; presence-only files are legacy |
| `test_decoder_reuse_requires_full_attestation` | equal time axes without exact source/detector/training/parameter/output provenance force tier 2 |
| `test_candidate_refresh_never_mutates_legacy_root` | tier 1 and tier 2 write beneath explicit output root, including `--overwrite` |
| `test_expected_sessions_require_ok_or_preapproved_exclusion` | missing/newly failed sessions cannot disappear or become ad hoc exclusions |
| `test_staged_run_exception_does_not_promote` | exception leaves global `CURRENT.json` unchanged and incomplete staging unreadable |
| `test_manifest_rejects_missing_or_extra_schema` | unknown version, missing required fields/columns, incompatible extras fail closed |
| `test_output_hash_and_run_id_are_verified` | corrupt bytes or mixed run IDs cannot resolve |
| `test_expected_cells_require_result_or_failure` | every declared cell is present or has an explicit failure record |
| `test_partial_family_release_cannot_promote` | missing/duplicate/staging family manifests cannot create a current release |
| `test_candidate_reporting_does_not_require_current_pointer` | explicit frozen candidate resolves for staged reports while default resolution still returns the old current release |
| `test_release_outputs_are_hashed_before_promotion` | missing/mutated staged report, figure, or doc bytes block release completion and leave `CURRENT.json` unchanged |
| `test_atomic_promotion_updates_pointer_last` | all family runs and the release manifest validate before the global pointer changes; interruption preserves the old release |
| `test_publish_release_outputs_copies_exact_bytes` | post-promotion publication resolves through global current and destination hashes equal the immutable release-output tree |
| `test_dirty_run_records_diff_hash` | dirty state is explicit and reproducible, never silently reported clean |
| current-session source audit (slow/integration) | all expected sessions produce `ok` or explicit failure rows; known invalid tails are detected with measured exposure |
| migrated producer smoke | replay, ContFrag, qvals, onset/time-bin, and one model output share the candidate generation and resolve by explicit run ID; direct legacy files do not |

## Fixtures

Use synthetic single/multiple valid intervals, 500-Hz analysis grids over 30-Hz raw frames, duplicate
position/graph source rows, out-of-range DIO events, far-from-well endpoints, unmatched pumps, two
cache schema versions, and mismatched decoder axes. Use `tmp_path` run roots with two tiny analyses,
minimal model-spec registries, one intentionally corrupt output, a missing required family, and an
existing promoted release to test interrupted replacement. The real-data audit is read-only and covers
all expected sessions; do not refit.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Position and graph fallbacks distinguish true absence from query/cardinality failure and record the
  exact graph geometry used by trial/arm construction.
- No path collapses multiple valid intervals, bridges a forbidden gap, clamps an invalid trial endpoint,
  or trusts a session-name cache without its source identity.
- The source audit reproduces and measures the stale-tail failure while leaving current promoted
  artifacts untouched.
- All manifest/promotion failure tests genuinely fail when validation is removed.
- No canonical consumer retains a hard-coded fallback to fixed legacy pickles.
- No individual family or staging directory can become current before the complete release validates.
- No scientific formula, cohort, coefficient, decoder, or detector changes slipped into this
  validation/storage PR.
- User-facing result-location docs were updated in this PR.
- Every task is implemented, the deliberately excluded scientific changes remain out of scope, and old direct-write/read paths named above are removed.
- Validation tests pass, slow tests are marked, and tests exercise interruption/corruption rather than only their own mocks.
- Production names/docstrings do not mention this plan or phase numbers.
