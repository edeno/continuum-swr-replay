# Phase 4 (D) — Results into `part4.json` and the overview

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: the project reports no number that a script did not compute into a summary file (`docs/design/2026-10-26-reported-values-design.md`). Phase 1 already wrote the timing values; this phase adds the results of phases 2 and 3b and the overview prose that cites them.

**Inputs to read first:**

- [src/continuum_swr_replay/reported/part4.py:809](../../../../src/continuum_swr_replay/reported/part4.py#L809) `overlap_and_effect_values` (the comparison of two classes this phase repeats for the new pair) and `:1054` `summarize`; `read_sidecars` / `staleness` (`:183`, `:204`).
- `docs/design/2026-10-26-reported-values-design.md` — keys, kinds, styles, `pending`, the changes list.
- `paper/reported/overview.md:140` — the Part 4 section.
- `Processed-Data/pfc_response_followups/swr_without_replay/` (phase 2) and `.../marked_family/summary.parquet` (phase 3b).

**Contracts referenced:** [Reported-value keys](shared-contracts.md#reported-value-keys), [Event classes and curves](shared-contracts.md#event-classes-and-curves), [Response window and unit sets](shared-contracts.md#response-window-and-unit-sets).

## Tasks

- **`Part4Inputs`** gains `swr_without_replay_dir` and `marked_family_summary`; `summarize` adds a source string and `inputs` records for each, and a `pending` note when the tree is stale against the current tables (same `staleness` rule) or absent (values skipped, never invented).
- **`swr_without_replay_values(population, curves, make)`**, keys `swr_no_replay_*`: sessions and events (`n_swr_no_non_local` summed, median per session, the unknown count), units tested, called fraction (raw p < 0.05, own null) pooled and per animal; overlap with SWR-coupled replay as `overlap_and_effect_values` does (both, either only, neither, expected, odds ratio pooled and by animal), the Spearman correlation of peak modulations, same-sign share, median absolute peak and rate multiplier for each of the two classes from the three-class model, share of units whose held-out deviance improves when `SWR_no_non_local` is added to the replay split, and timing (phase 1 functions) of the new class.
- **`marked_family_values(summary, make)`**, keys `marked_<model>_<target>_<class>_*`: per target and class, the joint and marginal population slope at its peak lag and that lag, whether it lies outside the null band (as 1/0 counts over animals: `..._animals_outside_band`), fraction of units with a held-out gain, and the number of sessions eligible. Mark words without digits.
- **Run** `summarize_part.py --part 4` and `emit_reported_values.py`; review `paper/reported/changes.md` (existing values must not change).
- **`paper/reported/overview.md` Part 4**: bullets **SWRs without replay?**, **What is replayed?**, **How strong?**, **Which stops?**, markers only; each bullet says the comparison in words and cites the keys.
- **Claims table**: no new claim rows without the lead; list candidate claims in the ledger.

## Deliberately not in this phase

- No refits. No figure (a figure is the lead's call after reading the values).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_swr_without_replay_values_keys` | synthetic population with the extended columns: documented keys, no digits, counts add up (both + only + neither = tested) |
| `test_marked_family_values_reads_summary` | synthetic `summary.parquet`: one value per documented key; absent file → no values and a logged reason |
| emit (manual) | `changes.md` lists only new keys; overview markers all resolve |

## Fixtures

Synthetic frames in the test module; the real trees for the summary run.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
