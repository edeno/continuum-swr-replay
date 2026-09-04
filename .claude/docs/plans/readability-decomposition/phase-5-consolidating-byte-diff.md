# Phase 5 — Consolidating regeneration byte-diff

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The one corpus-level gate: regenerate the canonical tables with all of phases 1-4 applied and confirm they are byte-identical to the frozen `_golden_baseline_v5_v4cache`. Per-phase verification was the suite (and the phase-4 equivalence test); this is the final proof that the whole decomposition preserved output. No code change.

**Baseline provenance (read this first):** the reference is `_golden_baseline_v5_v4cache`, a **tables-only** snapshot of the 5 canonical tables produced by committed master + the rebuilt **v4** behavioral cache (`comrie_photogate_trials_v4`), frozen at the start of this plan. It is NOT the older `_golden_baseline_v5` (Aug-29, v3 cache): the v3→v4 cache reclassifies 1361 stops / 12 events `is_rewarded_outcome` 0.0→NaN, so a v4 regen can never match the v3 baseline. That v3→v4 difference is a separate data-adoption question (models are still v3-fit); this gate isolates *only* whether the decomposition changed output, by diffing v4-regen against a v4 baseline. Models are not frozen in this baseline and not regenerated in this phase.

**Inputs to read first:**

- `scripts/qa/compare_to_golden_baseline.py` — the byte-diff tool (fixed two-reader form).
- The v4 baseline at `Processed-Data/replay_content/_golden_baseline_v5_v4cache/` (5 table parquets + `manifest.json` with per-table sha256/row counts).
- The warm cache exists at `Processed-Data/replay_content/_session_cache/comrie_photogate_trials_v4/` (218 entries), so `--from-cache` regenerates in ~80 min without Spyglass.

## Tasks

- **Smoke first** (per CLAUDE.md long-run discipline): confirm the from-cache regen hits the cache, not Spyglass, on the first few sessions (`grep 'from cache'` in the log), and estimate wall time before committing to the full run.
- **Full run** in tmux: `uv run python -m continuum_swr_replay.analysis.replay.prepare_data --from-cache`. This overwrites the production `event/event_arm/event_patch/stop/event_sequence` tables from the phases-1-4 code. Safe: the v4 baseline is frozen read-only; if the diff shows drift, restore production by copying the five parquet files back from `_golden_baseline_v5_v4cache`.
- **Byte-diff:** `uv run python scripts/qa/compare_to_golden_baseline.py --baseline Processed-Data/replay_content/_golden_baseline_v5_v4cache --tables event_table event_arm_table event_patch_table stop_table event_sequence_table`. Must report byte-identical (exit 0, no declared drift).
- If any table drifts: bisect by phase (the suite passed each, so drift implies a stage extraction or the vectorization changed a value the unit tests didn't cover); fix and re-run. Record the clean result.

## Deliberately not in this phase

- The onset/timebin tables (built by `build_timebin_tables.py`, not `prepare_data`) — unchanged by phases 1-4, so not part of this diff.
- Any new code.

## Validation slice

| Test | Asserts |
| --- | --- |
| `compare_to_golden_baseline` (slow, integration) | The five canonical tables regenerated from the phases-1-4 code are byte-identical to `_golden_baseline_v5_v4cache`. |
| full unit suite | Green (final confirmation). |

## Fixtures

The frozen `_golden_baseline_v5_v4cache` (tables-only) and the warm v4 behavioral cache (both already on disk).

## Review

Before closing the plan, dispatch **both** reviewers (overview "Between-phase independent review"):
- `code-reviewer`: confirm the diff is clean and, if any per-phase fix was needed, that it was output-preserving.
- **Independent readability agent** (whole-pipeline pass): re-run the scientist-readability lens over the full changed surface (the honest-name renames, both decomposed builders, the `history/` package) and give an overall verdict on whether a naive scientist can now trace the data processing — the closing check on the plan's goal.
Confirm: byte-diff clean; suite green.
