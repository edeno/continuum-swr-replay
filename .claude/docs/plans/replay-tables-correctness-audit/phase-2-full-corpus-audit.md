# Phase 2 — Read-Only Full-Corpus Audit and Manual QC

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#read-only-full-corpus-audit)

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/tables.py:215-1751` — production schemas and calculations; the independent audit may inspect but must not import private schema/calculation helpers.
- `src/continuum_swr_replay/analysis/replay/prepare_data.py:87-147,199-304,404-507` — producer inputs, session discovery, assembly-level retrospective/patch/standardization additions, and canonical output names. This phase does not call `main()`.
- `src/continuum_swr_replay/analysis/replay/retrospective.py:522-575` — assembly-level event columns that must be part of final-schema auditing.
- `src/continuum_swr_replay/visualization/session_tables.py:2723-2757,3096-3355` and `tests/visualization/test_session_tables.py:1218-1300` — existing real table panels and offline fixture pattern; use only as a comparison surface.
- `scripts/exploratory/_verify_covariate_invariants.py:41-210` — current print-only checker and its coverage gaps; do not treat its pass count as evidence.
- `scripts/exploratory/_audit_covariate_accuracy.py:201-221` — existing normalization check to reconcile with the new registry without duplicating conflicting rules.
- `tests/unit/test_replay_tables.py:597-647` — current cross-table/sequence assertions that are necessary but far from a corpus audit.
- `.claude/docs/reviews/01_definitions_event_tables_review.md` and `.claude/docs/reviews/06_data_provenance_regression_audit.md` — prior risks, artifact inventory, and provenance observations to turn into executable checks.

**Contracts referenced:**

- [Scientific grains and keys](shared-contracts.md#scientific-grains-and-keys) — all audit checks and examples are keyed at the correct grain.
- [Output contracts](shared-contracts.md#output-contracts) and [bundle contract](shared-contracts.md#cross-table-bundle-contract) — independent hard/identity registry.
- [Comparison policy](shared-contracts.md#comparison-policy) — fixed tolerances and exact NaN behavior.
- [Finding/severity contract](shared-contracts.md#finding-and-severity-contract) — automated and manual results block or warn consistently.
- [Artifact isolation and provenance](shared-contracts.md#artifact-isolation-and-provenance) — read-only sources and ignored output tree.

**Designs referenced:** [read-only full-corpus audit](designs.md#read-only-full-corpus-audit), [automated registry](designs.md#automated-registry), and [deterministic QC catalog](designs.md#deterministic-qc-catalog).

## Tasks

- Create `src/continuum_swr_replay/analysis/replay/table_audit.py` with immutable `AuditInputs`/`AuditResult` dataclasses, read-only loaders, a versioned independent schema/check registry, capped failure examples, session metrics, deterministic QC selection, and atomic report writers. It may call Phase 1 validators as additional defense, but expected schemas and identities are specified independently so producer and auditor cannot drift together.

- Implement path and provenance guards before loading table values:
  - resolve canonical five-table, standardization reference, q-value/contingency, event-onset, raw-dot, environment, result, detector, and session-cache paths;
  - require a new ignored output under `scratch/tables-review/<run_id>/audit`, outside the canonical directory and its ancestors/descendants;
  - record size/`mtime_ns` only for every raw dot/environment/result/cache source; SHA-256 before/after only the five canonical tables and tractable small auxiliary artifacts; never hash or open the approximately 271 GB posterior corpus;
  - record git commit, dirty diff hash, environment/lock hash, command, seed, expected/observed sessions, thresholds, elapsed time, peak RSS, and result status;
  - repeat the role-specific stat/hash policy at finalize and fail if anything changed.

- Freeze and review `config/replay_table_expected_sessions.csv` with the exact columns `animal,day,epoch,session_id,expect_dot,expect_environment,expect_result,expect_cache,posterior_status,posterior_evidence_token,posterior_evidence_sha256`. `posterior_status` is `available|unavailable|unattested`: the first two require a contemporaneous successful/absent-or-failed load attestation; a path or materialized output is never sufficient. Add `--expected-session-manifest`; count is only secondary. The streaming raw pass derives expected stop/event and retained-bin counts. Exact sequence participation is evaluated only for attested statuses; `unattested` emits a stable scientific-semantic finding and makes the Phase-2 source verdict `INCONCLUSIVE`, to be resolved by Phase 3's source-attested candidate.

- Build an independent final-assembled schema registry. Include builder columns, the retrospective columns merged at `prepare_data.py:281-297`, SWR split columns, and pooled z columns written at `:476-505`. Exact ordered columns, dtype families, nullability, and no silent extras are hard checks. Surface the current materialized `q_arm_hmm`, `q_arm_hmm_z`, and `swr_x_q_arm_hmm` columns as schema/provenance drift; do not hide them because a prior model used them.

- Implement the complete automated registry from [designs.md](designs.md#automated-registry), with stable check IDs and corpus/per-session rows:
  - input/session/schema/provenance;
  - key uniqueness, referential integrity, exact 9-arm/3-patch/sequence/stop cardinalities;
  - event/sequence, arm, patch, stop, retrospective, standardization, and cross-table identities;
  - exact broadcast equality and NaN masks;
  - independent interaction/indicator/value/history/leaf-aggregation recomputation.
  Every merge uses declared cardinality and every all-NaN aggregation uses a NaN-preserving operation. Label posterior-vector/MAP/peak/entropy/HPD/mass/fraction/multimodality checks `materialized_identity`: Phase 2 can prove their internal algebra but not source correctness without NetCDF; the stratified source parity gate is explicitly deferred to Phase 3.

- Add streaming raw-dot reconciliation. Read one `*_HPC_80_data_over_time.pkl` at a time, independently implement the strict three-clause retained mask and row-contiguous low-speed bouts, and compare:
  - retained event-label population, filtered/full boundaries, duration/bin/weight/MUA values;
  - strict positive-duration SWR classification and matched intervals;
  - stop boundaries/duration/distinct SWR/MUA/non-local counts and event assignment;
  - sequence time/non-local-probability rows where materialized.
  Do not import `_in_event_mask`, `_immobility_bout_intervals`, classified-event builders, or table schemas for these expected values. Call the Phase 1 `dot` contract separately under distinct check IDs.

- Add diagnostic—not direction-gated—session/animal summaries for every table response and downstream model predictor. Include robust quantiles, finite/missing fractions, MAD/IQR outliers, and session-tertile drift. Explicitly report:
  - full-label versus retained windows and labels spanning bout/filter gaps;
  - MUA full-window versus retained-window classification;
  - trunk current-arm fallback and content-support distributions;
  - probability overshoot and weight/count bounds;
  - junction/unmapped/all-junction frequency;
  - zero-event, zero-duration, long, and timestamp-gap stops plus alternative exposure totals;
  - contingency/value/uncertainty missingness and complete-case attrition;
  - session/animal distributions for SWR fraction, event rate, bins/duration/posterior confidence, events per stop, arms/patches, and every model response.
  A surprising biological distribution is a warning for inspection, not a hard failure by sign.

- Create `src/continuum_swr_replay/visualization/table_audit.py` with a headless source-to-table renderer. For every selected event/stop, render three deterministic images (`source`, `content`, `design`) with thresholds, raw labels, boundaries, table annotations, track/arm content, model rows, and explicit missing-data legends. Reuse `plot_session_tables` only as a thin comparison panel; independently render source traces and do not load posterior NetCDF.

- Create `scripts/qa/audit_replay_tables.py` with `audit`, `render-qc`, and `finalize` subcommands, argument parsing only, stable exit codes (`0` pass/complete, `1` scientific/manual failure, `INCONCLUSIVE` source status, or undispositioned concern, `2` IO/schema/configuration), and no project-path defaults in review mode. Automated outputs are immutable: `audit` refuses an existing run and creates a keyed `finding_dispositions.csv` review ledger, `render-qc` adds only images/index plus a hash-populated manual sheet, reviewers edit only the disposition ledger and manual-review fields, and `finalize` joins those ledgers into `final_findings.csv` plus new final-summary/report/hash files rather than overwriting automated results.

- Implement every exact schema/primary key/nullability/enum in [the versioned artifact design](designs.md#versioned-artifact-schemas): automated/final JSON reports, parent/active session identities, normalized artifact manifest, checks, immutable uncapped `automated_findings.csv`, keyed `finding_dispositions.csv`, finalize-created `final_findings.csv`, capped example parquet, exact session metrics/QC selections/manual review, image index, and three images per selection. Validate output order and schema against literal test expectations that do not import the writer or audit schema registry.

- Implement the exact tertile eligibility, scalar-quantile ranking, fallback order, deduplication, tie keys, clipping, and eight strata in [the deterministic QC design](designs.md#deterministic-qc-catalog), including `no_positive_junction_or_mapping_example` when the real zero-junction corpus forces fallback. Require exactly one completed row/selection and hash each image set. A `concern` requires disposition plus a second reviewer before finalize/Phase 3; `fail` is blocking.

- Add `tests/conftest.py` fixture `replay_audit_bundle`: two hand-authored sessions, three events, four stops (including zero-event and zero-duration), 27 arm rows, nine patch rows, contained/partial/non-SWR classes, one junction bin, one deliberately missing contingency group, exact lags/interactions/z reference, and literal expected audit rows. Construct expected tables by hand; never call production builders.

- Add tests:
  - `tests/unit/test_table_audit.py`: one clean pass and parameterized duplicate key, wrong session, missing arm/patch, orphan key, bad split, bad sequence ordinal/sum/argmax, wrong interaction, all-NaN-to-zero, wrong leaf mean, stale/extra schema, and wrong z corruption; assert stable check ID, `n_failed`, example key, severity, and exit classification;
  - `tests/visualization/test_table_audit.py`: three readable images/window, deterministic filenames/dimensions, threshold/boundary artists, missing-data legend, and no mocked table values;
  - `tests/integration/test_replay_table_audit_cli.py`: exact artifact names/literal schemas, atomic lifecycle, deterministic parent-to-pilot manifest projection/hashes, attested and unattested posterior statuses, immutable automated/finalized finding joins, exit codes, concern disposition, incomplete manual/disposition-ledger rejection, and role-specific source stat/hash equality;
  - `tests/integration/test_replay_table_audit_full_corpus.py`, marked `slow`: when explicitly opted in and configured, require exact identity equality with the 218-session manifest and run automated checks without QC; skip only for explicit absence, never for check failure;
  - fail-fast sentinels that raise if the Phase-2 path calls `xarray.open_dataset`, the posterior loader, Spyglass/DataJoint, or any cache writer.

- Add `docs/replay_table_audit.md` documenting input/output layout, command examples, severities, independent-versus-production checks, manual-QC procedure, and how to resume from a failed finding. Link it from `docs/README.md`. Mark the new CLI as the canonical invariant audit and update the two exploratory scripts only enough to point readers at it; do not maintain a second divergent rule set.

- Execute the audit in stages:
  1. synthetic/CLI tests;
  2. one session per animal (`--max-per-animal 1`) in a distinct pilot run directory with `/usr/bin/time -v`; deterministically project the first chronological row per animal from the frozen 218-row parent, write `active_session_manifest.csv`, record parent/subset hashes and selector, and gate against those five rows before reviewing elapsed/RSS/output size;
  3. all 218 sessions in a new full run directory, without QC rendering;
  4. render the 40 base windows plus failure exemplars;
  5. complete independent manual review and finalize.
  Record current baseline observations (218 sessions / five animals and the five row counts) in the report, but do not make row-count preservation a correctness criterion.

- Commit a sanitized snapshot under `docs/audits/replay-tables/<date>-<shortsha>/`: automated/final summaries, normalized and active manifests without machine-private prefixes, immutable automated findings, the finding-disposition ledger, finalized findings, QC selection, completed manual sheet, and final report. Keep row examples, logs, and images in ignored scratch storage and attach them as review artifacts. It is valid for the Phase 2 PR to report `FAIL` or `INCONCLUSIVE`; correctness of the audit and honest blocking are the acceptance criteria. Phase 3 cannot start until each hard/identity/manual concern has a resolution or author-approved expected-change ID; an explicitly unresolved posterior-attestation finding may proceed only because Phase 3 names the source-attested candidate and posterior spot-check as its resolution owner.

## Deliberately not in this phase

- Editing `tables.py`, its calculation helpers, thresholds, schemas, or model formulas.
- Calling `prepare_data.main`, rebuilding any table, refreshing a session cache, querying Spyglass, or opening the posterior NetCDF corpus.
- Whitelisting known schema drift, clipping probability overshoot, changing exposure/window/fallback/junction semantics, or choosing expected biological directions after seeing distributions.
- Fitting downstream models, changing a paper claim, regenerating figures, or promoting artifacts.
- Treating existing visualization panels, the old exploratory invariant count, or cold/warm same-code parity as an independent oracle.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_table_audit.py` | Independent registry passes the literal bundle and detects each targeted corruption with stable keys/counts/severity |
| `test_table_audit.py` diagnostics | Robust summaries retain NaNs, do not fail on biological sign, and deterministically identify constructed outliers |
| `test_table_audit.py` QC selection | Exact tertile/quantile/fallback/dedup/window algorithm yields eight rows/animal and records unavailable positive-junction examples honestly |
| `test_table_audit.py` visualization | Three offline images/window carry source thresholds/boundaries/content/design values and missing-data cues |
| `test_replay_table_audit_cli.py` | Atomic output, parent/active manifest identity, posterior-attestation states, immutable automated-to-final finding lifecycle, schemas, exit codes, incomplete review rejection, and zero source mutation |
| `test_replay_table_audit_full_corpus.py` (`slow`, opt-in) | Current identities equal the frozen 218-session manifest and every hard/identity result is emitted without catch-all skip |
| Manual review | All 40 base selections plus failure exemplars have one signed verdict and image hash |
| Canonical immutability check | Five tables/tractable auxiliaries have identical hashes; all raw/env/result/cache files have identical size/`mtime_ns`; no NetCDF/cache/upstream API was opened |

Run, in order:

```bash
uv run pytest -q tests/unit/test_table_audit.py tests/visualization/test_table_audit.py tests/integration/test_replay_table_audit_cli.py
/usr/bin/time -v uv run python scripts/qa/audit_replay_tables.py audit \
  --input-dir Processed-Data/replay_content \
  --source-dir Processed-Data \
  --output-dir scratch/tables-review/<pilot_run_id>/audit \
  --expected-session-manifest config/replay_table_expected_sessions.csv \
  --max-per-animal 1 --source-checks \
  --qc-per-animal 8 --seed 20260808
/usr/bin/time -v uv run python scripts/qa/audit_replay_tables.py audit \
  --input-dir Processed-Data/replay_content \
  --source-dir Processed-Data \
  --output-dir scratch/tables-review/<full_run_id>/audit \
  --expected-session-manifest config/replay_table_expected_sessions.csv \
  --source-checks \
  --qc-per-animal 8 --seed 20260808
uv run python scripts/qa/audit_replay_tables.py render-qc --audit-dir scratch/tables-review/<full_run_id>/audit
uv run python scripts/qa/audit_replay_tables.py finalize --audit-dir scratch/tables-review/<full_run_id>/audit
uv run pytest -q -m slow tests/integration/test_replay_table_audit_full_corpus.py
uv run pytest -q -m "not slow"
uv run ruff check src/continuum_swr_replay/analysis/replay/table_audit.py src/continuum_swr_replay/visualization/table_audit.py scripts/qa/audit_replay_tables.py tests/unit/test_table_audit.py tests/visualization/test_table_audit.py tests/integration/test_replay_table_audit_cli.py
```

## Fixtures

- The shared literal two-session `replay_audit_bundle` described above; expected frames, artifact schemas, and check results are hand-authored and small enough to inspect in tests. Tests may not import expected schemas from the audit registry/writer.
- Parameterized corruption functions return modified copies and never alter the clean fixture in place.
- The current 218-session source/table corpus is external read-only test data. The slow test requires explicit configured paths and emits one narrow missing-data skip reason.
- QC image tests use the literal raw/source arrays and a minimal real xarray time coordinate; they do not mock expected numerical table values or open decoder results.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- A reviewer independent of the table implementation checks at least one raw-dot reconciliation and one arm/patch/stop identity manually, confirms the audit registry does not import producer calculations, and signs the QC sheet.
- The PR includes canonical before/after hashes, resource measurements, the complete finding list/dispositions, and no changes to the five source artifacts.
