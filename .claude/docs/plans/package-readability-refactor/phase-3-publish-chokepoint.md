# Phase 3 — Single publish_canonical() chokepoint

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make "one door onto the canonical tables" structural. Every writer that persists a canonical table stamps → validates → atomically writes through one function, so an unstamped or partial-set publish becomes impossible at the write, not caught downstream at fit time. Numerical-path phase: regen byte-diff required.

**Inputs to read first:**

- [publish_canonical contract](shared-contracts.md#publish_canonical) — the function to add and its invariants.
- `src/continuum_swr_replay/table_io.py` — `write_table` (lines ~66-114) and `umask_file_mode`; `publish_canonical` wraps `write_table`.
- The ~14 canonical-writer sites in overview.md "Canonical-table writers" — read each to confirm it currently does `stamp_covariate_schema(...)` then `write_table(...)`; those two lines collapse into one `publish_canonical(...)`.
- `scripts/decoding/regenerate_stop_table.py:200-204` and `analysis/replay/prepare_data.py:660-674` — the two guarded stop-table writers added earlier this session; their guards move behind the chokepoint or stay as callers of it.

## Tasks

- **Add `publish_canonical()` to `table_io.py`** per the contract (function-local import of `stamp_covariate_schema` to avoid a cycle; `basename` param for the source-tagged contfrag writers).
- **Route every canonical writer through it.** At each site in overview.md, replace the `stamp_covariate_schema(frame, table=T)` + `write_table(frame, path)` pair with `publish_canonical(frame, path, table=T)`. Sites: `prepare_data.py:674,1009`; `qvals_table.py:1073`; `prepare_contfrag_replay_content_data.py:496`; `assemble_replay_content_shards.py:170,202`; `build_timebin_tables.py:133,134`; `regenerate_stop_table.py:204`; `assemble_contfrag_shards.py:109`; `refresh_contfrag_behavioral_covariates.py:293`. For contfrag basenames pass `basename=basenames[name]`.
- **Leave non-canonical writes on `write_table`:** `regenerate_stop_table.py:202` (the `stop_table_previous` backup), `convert_tables_to_parquet.py:81` (migration tool), and the phase-0 baseline scripts. These persist frames that are not the live canonical set; document why in a one-line comment at each.
- **Keep the atomic-set guard visible.** `regenerate_stop_table` and the `--stop-only` bootstrap already refuse to write a partial/incomplete stop table (guards added this session). Confirm those guards still fire *before* `publish_canonical`; do not let the chokepoint silently absorb them — the session-parity and split-column checks stay.
- **Docs:** add a "Publishing a canonical table" section to `docs/README.md` (or the covariate-registry doc) stating `publish_canonical` is the only supported path and why direct `write_table` on a canonical table is a bug.
- **Regen byte-diff.** Rebuild the canonical tables via the warm `--from-cache` path and run `scripts/qa/compare_to_golden_baseline.py` — must be byte-identical (stamping already happened before; this is pure call-site consolidation).

## Deliberately not in this phase

- Splitting `tables.py` (phase 4) or changing builder signatures (phase 5). The chokepoint is a write-path change; builders are untouched.
- Adding new validation semantics. `publish_canonical` enforces exactly what `stamp_covariate_schema` already enforces — no stricter, no looser — so the byte-diff stays clean.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_publish_canonical_stamps_and_writes` | Output table carries the schema stamp and round-trips; equals a manual stamp+write. |
| `test_publish_canonical_rejects_bad_rows` | A frame with duplicate row keys raises (via `stamp_covariate_schema`), nothing written. |
| `test_publish_canonical_basename_override` | `basename=` writes to `dir/basename.parquet`. |
| `test_no_canonical_writer_calls_write_table_directly` | Grep-style: canonical-writer modules import `publish_canonical`, not bare `write_table` for canonical frames. |
| regen + `compare_to_golden_baseline` (slow, integration) | Byte-identical to the frozen v5 baseline. |
| existing suite | Green. |

## Fixtures

A small valid canonical `stop_table` and a duplicate-key variant (in `conftest.py`) for the stamp/reject tests. The regen step uses the warm behavioral cache — mark the test `slow`/`integration` and skip if the cache is absent.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every canonical writer routes through `publish_canonical`; only the documented non-canonical writes stay on `write_table`.
- The stop-table atomic-set guards still fire before the write (not swallowed).
- Regen byte-diff is clean; the grep-style test has no false negatives.
- Doc section added. Tests exercise behavior, not tautologies.
- No name references this plan.
