# Phase 1 — Contracts and Independent Reference Oracle

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#independent-slow-reference-oracle)

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/tables.py:176-209,215-669` — event annotation, strict retention mask, event population, two-window summaries, and classified merge.
- `src/continuum_swr_replay/analysis/replay/tables.py:672-1073` — sequence, stop, and SWR/non-SWR split behavior, including current fail-open paths.
- `src/continuum_swr_replay/analysis/replay/tables.py:1076-1538` — arm expansion/alignment and patch projection.
- `src/continuum_swr_replay/analysis/replay/tables.py:1541-1751` — current ordered schemas and empty-frame paths.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:29-266,332-425` — trial timing, bouts, posterior alignment, and stable IDs.
- `src/continuum_swr_replay/analysis/replay/summaries.py:59-317` — production posterior collapse and arm-response arithmetic that the oracle must not import.
- `src/continuum_swr_replay/analysis/hippocampus/event_table.py:206-365` — full-label classification and strict overlap semantics.
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:215-280` and `tests/analysis/hippocampus/test_data_aggregation.py:319-359` — partial validator to migrate and tests to relocate.
- `src/continuum_swr_replay/analysis/replay/prepare_data.py:199-304,408-506` — per-session assembly, concat, current one-line global invariant, standardization, and write boundary.
- `tests/unit/test_replay_tables.py:29-733` plus focused tests in `tests/unit/test_replay_summaries.py`, `test_replay_uncertainty.py`, `test_spatial_mapping.py`, `test_trial_sort_invariance.py`, and `test_trajectory_rollups.py` — preserve these focused checks and identify their table-level wiring gaps.

**Contracts referenced:**

- [Review boundary and ownership](shared-contracts.md#review-boundary-and-ownership) — formula changes and canonical regeneration are outside this PR.
- [Input contract](shared-contracts.md#input-contract) — reject malformed/misaligned inputs; never repair them.
- [Output contracts](shared-contracts.md#output-contracts) — exact grains, schemas, dtypes, and table identities, including empty outputs.
- [Cross-table bundle contract](shared-contracts.md#cross-table-bundle-contract) — replace the current single aggregate guard without weakening it.
- [Comparison policy](shared-contracts.md#comparison-policy) — predeclared exact/numerical tolerances.

**Designs referenced:** [contract module](designs.md#contract-module-and-failure-surface) and [independent slow reference oracle](designs.md#independent-slow-reference-oracle).

## Tasks

- Record the untouched baseline before edits. Run the current focused replay-table selection and save command, git SHA/diff hash, Python/package versions, count (`77 passed` is the current observation), wall time, and first failure if the baseline has drifted. Do not update expected outputs to accommodate a pre-existing failure.

- Create `src/continuum_swr_replay/analysis/replay/table_contracts.py` with:
  - `ColumnSpec`, exact schemas for all five builder tables plus exact assembled event/stop schemas after retrospective/SWR-split joins, stable `ReplayTableContractError.check_id`, and typed empty-frame construction;
  - session/time/label/state-probability `dot` validation, including absent-or-equal `time` column, exact one-bout-per-retained-label, required canonical MUA, and all-or-none trajectory-source columns;
  - posterior dimension/shape/time/position/finite/nonnegative/normalization validation with the shared tolerances;
  - retained event-population/session matching for downstream builders;
  - per-table key/schema/dtype/range/identity validators;
  - `validate_session_replay_tables(..., posterior_available=...)` before assembly joins and `validate_assembled_replay_tables(..., expected_session_manifest=..., standardization_reference=...)` after assembly, so additive columns are required rather than rejected and empty frames cannot hide missing sessions/posteriors.
  Error messages include table, session, failure count, and example natural keys. Validators must not mutate, reorder, fill, clip, coerce, or deduplicate caller data.

- Move `validate_dot_event_contract` out of `analysis/hippocampus/data_aggregation.py:215-280`, migrate `prepare_data.py` and all tests/importers to the new replay contract module, and remove the old symbol. Do not add a lazy wrapper or alias. Move/expand its tests into `tests/unit/test_replay_table_contracts.py`.

- Replace the private name-only schema ownership at `tables.py:1541-1751` with projections from the new typed schema constants. Make each public empty path return exact ordered columns and dtype families. Add explicit non-mutating-input tests for every builder and `_prepare_dot`.

- Wire validation at every public boundary in `tables.py`:
  - `build_event_table` (`:215-669`): validate before session extraction, validate optional posterior before alignment, prove the classified merge is one-to-one/total for the retained population, and validate the returned nonempty or empty table.
  - `build_event_sequence_table` (`:672-771`): validate required posterior and event retention, then validate exact time/ordinal/posterior rows before return.
  - `build_stop_table` (`:774-1006`): validate its narrower raw/trial/graph inputs, unique ordered bouts, counts/flags, context-at-start, and typed empty/error behavior.
  - `add_nonlocal_swr_split` (`:1009-1073`): preserve idempotence and zero-event behavior, but reject missing/null `stop_id`, `event_id`, or `is_swr`, duplicate events/stops, unknown stops, and nonadditive counts. Remove `fillna(False)` and the nonempty fail-open branch.
  - `build_event_arm_table` (`:1076-1370`): require event-table session and retained-label equality with the supplied `dot`/thresholds; validate posterior; replace label-index repetition at current line 1177 with positional repetition such as `event_table.iloc[np.repeat(np.arange(n_events), 9)]`; verify exact nine-arm sets, broadcasts, counts, mass, fractions, predictors, and every interaction.
  - `build_event_patch_table` (`:1373-1538`): require valid event and arm tables, total session/`is_swr` inputs, exact arm sets and key cardinality before groupby; explicitly broadcast `non_local_event` so the natural key is present; reject missing/duplicate/unknown alternatives; validate leaf-only means, NaN-preserving sums, three-patch sets, and interactions before return. Keep its intentional multi-session projection support, but validate every event/session group and migrate the ContFrag caller/test to the same exact schema. Remove the fixture-tolerant “all false” fallback for missing `is_swr` and optional session-key branch.

- Replace `prepare_data.py:454-472`'s single `sum(n_events)` gate with two explicit stages: call `validate_session_replay_tables` inside `_process_session` immediately after the four raw-derived builders and before the retrospective merge, then call `validate_assembled_replay_tables` after global retrospective columns, SWR split, patch projection, and pooled standardization but before any artifact write. Build and pass an exact session manifest with raw retained-event/stop counts and posterior availability plus the pooled z reference. An invalid session raises and terminates regeneration; it is not converted into the current broad skip-and-continue path. Preserve useful progress logging.

- Create `tests/oracles/__init__.py` and `tests/oracles/replay_table_reference.py` using the exact [reference design](designs.md#reference-algorithm). Include literal expected frames and assert the oracle against them before comparing production. Add a module-level import guard/test proving the oracle does not import any forbidden production replay module.

- Create `tests/unit/test_replay_table_contracts.py` covering:
  - clean, empty, mixed-session, missing-column, duplicate/nonfinite/unsorted time, conflicting `time` column, fractional/negative/discontiguous labels, and one retained label spanning two immobility bouts;
  - posterior reversed dimensions, time/position/shape drift, NaN/negative/nonunit rows, and the measured float32 normalization tolerance;
  - exact empty schema/order/dtype/nullability for every table;
  - mismatched session/event populations/thresholds across builders;
  - duplicate/missing/invalid arms and patches; duplicate sequence keys/bad ordinals;
  - split idempotence, zero-event behavior, missing/null classifications, duplicate IDs, unknown stops, dtype preservation, and additivity;
  - sorted-versus-shuffled trial invariance, non-mutating stable sorting, and rejection of ambiguous chronological ties;
  - bundle acceptance for both a zero-event session with stops and a no-immobility/no-stop session, exact posterior-session participation, and rejection for orphan/double-counted events;
  - missing required MUA rejection; no-trajectory legacy acceptance; complete-trajectory acceptance; and partial trajectory-source rejection;
  - arbitrary and duplicate pandas indices with identical keyed output;
  - input frames unchanged byte-for-byte/value-for-value after every public call.

- Create `tests/unit/test_replay_table_reference.py` covering all cases in the [adversarial matrix](designs.md#adversarial-case-matrix). Compare complete keyed core response columns, not merely row counts/non-null counts. Add an explicit shuffled/nonconsecutive-label test that compares every value to the baseline after key sorting.

- Extend existing table-level tests rather than relying only on helper tests:
  - event/classification: each filter clause, strict ties, NaNs, fragmented+continuous sum, filtered versus full-label window, SWR boundary touch, classified merge cardinality, upper-middle filtered/full midpoint for odd/even/gapped bins, and exact median `time_since_stop`;
  - stop: no-stop, zero/single-sample/multi-event bouts, distinct SWR/MUA labels, missing-MUA rejection, every lag/context/arm/patch field, and event labels near/across bout boundaries;
  - arm: exact soft/fraction/posterior responses, arm order, every indicator/interaction, graph distance/rank, visit/reward history, demeaning scope, placeholder/NaN behavior, and value permutation under shuffled keys;
  - patch: exact 3-row projection, conservation, leaf-only aggregation, session propagation, missing values, and all interactions;
  - sequence: exact schema/dtypes, time/order/reset, posterior vector/argmax/probability, shape/time mismatch, and all-junction behavior.

- Strengthen `tests/integration/test_from_cache_parity.py:29-57`: remove catch-all skip behavior, validate both bundles independently, compare by declared keys, and retain exact cold/warm parity. If Spyglass is unavailable, separate the cache-only real-session test from the explicitly marked external upstream test rather than skipping all parity assertions.

- Add `docs/replay_table_contracts.md` with the two-window definition, table grains/keys, input alignment rules, exact empty behavior, response identities, tolerances, and examples of fail-closed errors; link it from `docs/README.md`. Update any stale `data_aggregation` doc references in the same PR.

## Deliberately not in this phase

- Changing MUA to the full-label window, stop exposure to sample-count time, bout splitting on timestamp gaps, trunk fallback, probability clipping, or all-junction posterior semantics. Current behavior is pinned and alternatives are measured in Phase 2.
- Rebuilding or overwriting canonical tables, accepting materialized schema drift, or deleting retired columns from existing artifacts.
- Fitting any content/rate/choice/stay model or judging whether paper conclusions survive.
- Creating a broad “fix whatever the oracle finds” commit. A valid-input scientific mismatch receives a finding ID, a narrow correction PR, and a rerun of this phase.
- Refactoring direct helper modules for style/performance. Only boundary validation and independently demonstrated correctness defects are in scope.

## Validation slice

| Test | Asserts |
| --- | --- |
| Existing focused replay suite | Pre-change scientific outputs remain stable for valid current fixtures; no regression is hidden by new tests |
| `test_replay_table_contracts.py` | Every malformed/misaligned input fails at the public boundary; empty schemas/dtypes and non-mutation are exact |
| `test_replay_table_reference.py` | Core event/stop/arm/patch/sequence responses match literal expectations and an independent implementation on adversarial cases |
| Expanded table-level unit tests | Filter/merge/broadcast/projection/history/interaction wiring is correct, not merely the helper in isolation |
| Full unit suite | Direct helper and unrelated public behavior remains green |
| Strengthened cache parity (`integration`, external mark where needed) | A real cached build satisfies contracts and exact keyed parity; external unavailability cannot mask cache-only checks |

Run, in order:

```bash
uv run pytest -q \
  tests/unit/test_replay_tables.py \
  tests/unit/test_event_table_pre_trial_no_leak.py \
  tests/unit/test_stop_table_bout_start_no_leak.py \
  tests/unit/test_event_patch_table_nan.py \
  tests/unit/test_id_uniqueness_preconditions.py \
  tests/analysis/hippocampus/test_classified_event_table.py \
  tests/unit/test_swr_overlap_consistency.py
uv run pytest -q tests/unit/test_replay_table_contracts.py tests/unit/test_replay_table_reference.py
uv run pytest -q tests/unit
uv run pytest -q tests/integration/test_from_cache_parity.py
uv run ruff check src/continuum_swr_replay/analysis/replay/table_contracts.py src/continuum_swr_replay/analysis/replay/tables.py tests/oracles tests/unit/test_replay_table_contracts.py tests/unit/test_replay_table_reference.py
```

## Fixtures

- A hand-authored one-session environment with finite MAP positions mapping to nearest real arms, one NaN/unmapped MAP row, explicit posterior junction bins, literal posterior coordinates, literal SWR/MUA/event labels, and a small graph whose arm/well/patch distances can be checked on paper.
- A second session using overlapping local label numbers to prove that session keys prevent cross-session fusion.
- At least three retained events: non-SWR, SWR-contained, and SWR-partial/boundary-touch; four stops including zero-event and zero-duration; all nine arms and all three patches.
- Posterior variants built from literal arrays: one-hot, diffuse, permuted columns with matching coordinates, all junction mass, malformed shape/dims/time/normalization, and tie rows.
- A cache-only real-session fixture is loaded from configured project paths and never checked into git. External upstream coverage has a distinct marker and may skip only for a proven unavailable service, not for assertion/loader errors. Strict auditing of the known-drift materialized five-table bundle is owned by Phase 2 so this PR is not required to bless stale HMM columns.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- A second scientific-code reviewer verifies the oracle's independence, recomputes at least one fixture by hand, and confirms that no tolerance or expected value was chosen from the production output.
- The PR description includes the pre/post focused baseline, any intentionally changed adversarial/schema behavior, cache-only real-session status, and zero canonical artifact modifications.
