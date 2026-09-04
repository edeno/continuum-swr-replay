# Phase 3 — build_event_table into a stage pipeline

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [stage convention](shared-contracts.md#stage-convention)

`build_event_table` is ~535 lines across nine numbered `# --- N ---` banner blocks that a scientist can't hold in their head. Turn each banner into a named stage function so the builder is a readable spine. Output-preserving.

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/tables_event.py:100-635` — the whole `build_event_table`, and its nine banner blocks at `:252, :278, :290, :305, :434, :447, :457, :499, :524, :544`. Read what each block consumes and produces so the stage boundaries are the banner boundaries.
- `tables_event.py:544-633` — the classified-timing merge + the cross-pass timing verification that asserts builder-pass timing equals label-pass timing bit-for-bit, then drops the alias columns. This is load-bearing science; it becomes its own named stage, not hidden.
- [stage convention](shared-contracts.md#stage-convention).

## Tasks

- Extract one module-private stage function per banner (grouping the tightly-coupled summary blocks 3-5 is fine), named for the scientific step: e.g. `_annotate_dot_for_events` (block 1), `_select_in_event_timebins` (2), `_aggregate_event_features` (3-5), `_time_since_stop_per_event` (6), `_attach_posterior_rollups` (7), `_attach_trajectory_rollups` (7b), `_event_identifiers_and_hashes` (8), `_merge_classified_timing` + `_verify_builder_pass_timing` (9). Each moves its block verbatim; the spine calls them in order and returns the stamped table (see the stage-convention sketch). Delete the banner comments the stage names now replace.
- Keep `build_event_sequence_table` (same file) untouched unless it shares an extracted helper — if it does, the helper is module-private and shared, not duplicated.

## Deliberately not in this phase

- `build_stop_table` / `build_event_arm_table` (`tables_stop.py` / `tables_arm_patch.py`) — same god-function shape may apply, but out of scope; note as a follow-on.
- Any change to which bins define an event, the posterior marginalization, or the cross-pass verification logic — pure reorganization.

## Validation slice

| Test | Asserts |
| --- | --- |
| existing `test_builder_parity` / `test_replay_tables*` / `test_from_cache_parity` (integration) | Canonical `event_table` built post-extraction equals pre-extraction. |
| `test_event_table_stage_spine` (optional) | `build_event_table` body is a short sequence of the named stages (guards against a future re-inlining). |
| full unit suite | Green. |

## Fixtures

None new — the existing builder-parity integration tests are the verification; the phase-5 regen byte-diff is the corpus-level confirmation.

## Review

Before opening the PR, dispatch **both** reviewers (overview "Between-phase independent review"):
- `code-reviewer`: pure motion (each stage body unchanged), the cross-pass verification survives as its own stage, banners removed.
- **Independent readability agent**: given the new `build_event_table` spine + stages, confirm a scientist can now follow how an event table is built stage by stage without re-reading 500 lines — keep/adjust with evidence.
Confirm: builder-parity tests pass; stage names use scientific vocabulary, not "phase N" or plan references.
