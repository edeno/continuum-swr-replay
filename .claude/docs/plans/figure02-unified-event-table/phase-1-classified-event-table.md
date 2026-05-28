# Phase 1 — Canonical per-event classification table

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/tables.py:120-499](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L120-L499) — current `_prepare_dot`, `_in_event_mask`, and `build_event_table`. The new module extracts the timing/overlap/peak-prob core; `build_event_table` is rewritten to compose on top of it.
- [src/continuum_swr_replay/analysis/replay/tables.py:1229-1275](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1229-L1275) — `_event_table_columns()`. Phase 1 adds new columns to this list and bumps the schema.
- [src/continuum_swr_replay/analysis/hippocampus/event_detection.py:255-509](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L255-L509) — `classify_swr_overlap` and `classify_swr_overlap_geometry`. Reused verbatim for `overlap_class` / `overlap_subclass`.
- [src/continuum_swr_replay/analysis/replay/_helpers.py:301-323](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L301-L323) — `hash_event_id` and `validate_event_ids`. Reused for `event_id` derivation in the new table.
- [scripts/prepare_replay_content_data.py:140-187](../../../../scripts/prepare_replay_content_data.py#L140-L187) — call site for `build_event_table`. Re-run after this phase to regenerate `event_table.pkl`.
- [tests/unit/test_event_table_pre_trial_no_leak.py](../../../../tests/unit/test_event_table_pre_trial_no_leak.py) — existing event-table test. Must continue to pass after the refactor.

**Contracts referenced:** None — this phase defines the new module's contract for phase 2 to consume.

## Tasks

### 1. Add `src/continuum_swr_replay/analysis/hippocampus/event_table.py`

New module with one public function `build_classified_nonlocal_event_table`. One row per non-local event in the input `dot` frame (`non_local_event > 0`), no speed/prob filter applied. The replay-content table filters this on top.

Signature:

```python
def build_classified_nonlocal_event_table(
    dot: pd.DataFrame,
    *,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.DataFrame:
    """One row per non-local event with timing, overlap class, and per-event summaries.

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed ``data_over_time`` slice for one or more sessions. Must
        carry ``animal``, ``day``, ``epoch``, ``non_local_event``,
        ``SWR_event``, ``head_speed``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``.
    speed_threshold : float
        Immobility threshold (cm/s) used to populate ``endpoints_speed_below_thresh``.

    Returns
    -------
    pd.DataFrame
        Columns: ``animal, day, epoch, non_local_event, event_id,
        event_start, event_end, event_duration, event_midpoint_time,
        max_speed, mean_speed, start_speed, end_speed,
        endpoints_speed_below_thresh, peak_nonlocal_prob, is_swr,
        overlap_class, overlap_subclass, n_swrs_overlapping,
        matched_swr_event, matched_swr_start, matched_swr_end``.
        Empty frame (correct columns, zero rows) if no events.
    """
```

**SWR-interval source — single canonical rule.** SWR intervals are *always* derived from `dot["SWR_event"]` labels via `intervals_from_labels(session_dot, "SWR_event")`. The function does not accept a `ripple_times` override. This matches the label source used by panels b/c/d today and removes the panel-a-vs-panel-b boundary-touch drift the reviewer flagged. The threshold-crossing `ripple_times` from `load_data` is *not* used for classification anywhere in Figure 2 after this phase — only for plotting (`plot_ripple_consensus_trace_zscore`, which is unchanged).

Derivation rules per column:

- `event_start = times[0]`, `event_end = times[-1]` where `times` are the *full* set of timebins for that event id (no speed/prob filter). `event_duration = event_end - event_start`. `event_midpoint_time = times[len(times) // 2]`.
- `event_id = hash_event_id(animal, day, epoch, non_local_event)` — reuse [_helpers.py:301](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L301). Call `validate_event_ids` on the output to enforce uniqueness within session.
- `max_speed = g["head_speed"].max()`, `mean_speed = g["head_speed"].mean()`, `start_speed = g["head_speed"].iloc[0]`, `end_speed = g["head_speed"].iloc[-1]`. For a single-timebin event, `iloc[0] == iloc[-1]` and the four scalars collapse to the same value — equivalent to the single-timepoint branch in [event_analysis.py:93-95](../../../../src/continuum_swr_replay/analysis/hippocampus/event_analysis.py#L93-L95). No special-case needed.
- `endpoints_speed_below_thresh = (start_speed < threshold) & (end_speed < threshold)` — matches `get_immobile_events(method="endpoints")` semantics at [event_analysis.py:97-101](../../../../src/continuum_swr_replay/analysis/hippocampus/event_analysis.py#L97-L101).
- `peak_nonlocal_prob = (g["Non-Local_Continuous_prob"] + g["Non-Local_Fragmented_prob"]).max()`.
- `overlap_class` / `overlap_subclass` / `n_swrs_overlapping` — call `classify_swr_overlap_geometry(classifier_events, swr_intervals_for_session)` per session. `swr_intervals_for_session` is computed via `intervals_from_labels(session_dot, "SWR_event")` (see "SWR-interval source" above). `classify_swr_overlap_geometry` expects input columns named `start_time` and `end_time`, while the canonical output columns are named `event_start` and `event_end`; build a temporary alias frame for the classifier, for example `classifier_events = events_df.rename(columns={"event_start": "start_time", "event_end": "end_time"})`, then copy the returned `overlap_class`, `overlap_subclass`, and `n_swrs_overlapping` columns back onto the canonical `events_df`. Do not pass a frame with only `event_start`/`event_end` directly to `classify_swr_overlap_geometry`.
- `is_swr = overlap_class != "non_swr"`.
- `matched_swr_event/start/end`: for `overlap_class == "swr_contained"`, pick the SWR that fully contains the event (if multiple, the earliest by `start_time`). For `"swr_partial"`, pick the earliest SWR with positive-duration overlap. For `"non_swr"`, NaN. The SWR id is the positive integer label from `dot["SWR_event"]`; expose as `matched_swr_event` (Int64, nullable). This requires `intervals_from_labels(..., "SWR_event")` to retain the `SWR_event` column, not drop it.
- **Note for panel d population (multi-SWR events).** The matched-SWR scheme picks one SWR per event. Events with `overlap_subclass in {"multi_swr_no_gap", "multi_swr_gap"}` lose their additional SWR pairings — these classes are rare (SWR detectors enforce minimum inter-event separation). Panel d's timing-offset histograms will therefore use *one row per overlapping event*, not *one row per (SWR, event) pair* as the old `compute_event_timing_offsets` did. Phase 2 documents this in the panel-d caption.

**Canonical home of `intervals_from_labels`.** Move the helper contract into `analysis/hippocampus/event_table.py` as a public helper and make `data_aggregation.py` import it (or keep `_intervals_from_labels = intervals_from_labels` as a temporary alias until Phase 2 deletes the old caller). The helper must support both production frames with time in the index and tests with a `"time"` column, and it must return the event label id:

```python
def intervals_from_labels(session_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return columns ``label_col, start_time, end_time`` for positive labels."""
```

Implementation requirements:

- If `"time"` is present, use that column; otherwise use `session_df.reset_index(names="time")`.
- Group only positive labels (`label_col > 0`) and return columns `[label_col, "start_time", "end_time"]`.
- Do not `reset_index(drop=True)`; callers need the SWR label id for `matched_swr_event`.
- Empty input returns a schema-complete frame with those three columns.

Implementation notes:

- Process one `(animal, day, epoch)` group at a time. SWR intervals are session-scoped, so cross-session leakage is impossible by construction.
- Empty-session short-circuit: if no rows in the input have `non_local_event > 0`, return an empty frame with the canonical column order — see `build_event_table`'s empty-frame pattern at [tables.py:287-291](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L287-L291).

**Register the new public API in the package's lazy-export map.** The `analysis/hippocampus/__init__.py` exposes its surface through a `_LAZY_EXPORTS` dict plus a mirrored `TYPE_CHECKING` import block ([__init__.py:75-122](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L75-L122) and [__init__.py:138-158](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L138-L158)). Add an entry for the new module so `from continuum_swr_replay.analysis.hippocampus import build_classified_nonlocal_event_table` works through the same path as the existing helpers:

- Add `"build_classified_nonlocal_event_table": "event_table"` (and `"intervals_from_labels": "event_table"` if it lives here per the "Canonical home" decision above) to `_LAZY_EXPORTS`.
- Add a matching `from .event_table import (...)  # noqa: F401` block under `TYPE_CHECKING`.
- `__all__` is derived from `_LAZY_EXPORTS.keys()` ([__init__.py:135](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L135)), so no separate edit is needed there.

### 2. Refactor `analysis/replay/tables.py:build_event_table` to compose on top

`build_event_table` retains the legacy per-event population, timing, and replay aggregates bit-identically while pulling SWR overlap classification and additive full-window summaries from the new classified table. The key invariant: **columns currently consumed by replay-content analyses are computed over the same filtered sub-window as today, so they cannot drift; only `is_swr` changes semantics.** Full label-window timing is added under new `label_event_*` column names.

The implementation uses **two groupbys** with a clear division of labor:

**Groupby A — classified table** (the new module).

- Operates on the full event-label window: every row with `non_local_event > 0` in `dot`.
- Produces, per event: `event_id`, full-window `event_start`, `event_end`, `event_duration`, `event_midpoint_time`, `max_speed`, `mean_speed`, `start_speed`, `end_speed`, `endpoints_speed_below_thresh`, `peak_nonlocal_prob`, `is_swr`, `overlap_class`, `overlap_subclass`, `n_swrs_overlapping`, `matched_swr_event`, `matched_swr_start`, `matched_swr_end`.

**Groupby B — legacy aggregates** (kept inside `build_event_table`).

- Operates on the filtered sub-window: rows where `(non_local_event > 0) & (head_speed < speed_threshold) & (nonlocal_prob > nonlocal_threshold)` — the same `_in_event_mask` as today at [tables.py:158-162](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L158-L162).
- Produces, per event: legacy filtered-window `event_start`, `event_end`, `event_duration`, `event_midpoint_time`, `is_mua`, `current_arm`, `prev_arm`, `next_arm`, `last_rewarded_arm`, `current_patch`, `prev_patch`, `next_patch`, `is_after_switch`, `is_after_omission`, `is_after_reward`, `is_after_stay`, `reward_outcome`, `is_stay`, `n_nonlocal_bins`, `nonlocal_weight_sum`, `n_junction_bins`, `n_unique_arms`, `time_since_stop`, `time_since_port_entry`, `time_since_reward`, `stop_idx`, `stop_id`, plus posterior columns when `conditional_posterior` is provided.
- This is the *same* groupby as today's [tables.py:311-465](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L311-L465), unchanged in scope.

**Join.** Merge the two on `event_id` (inner join). The replay-content event table keeps the legacy population semantics: an event appears in `build_event_table`'s output iff at least one of its timebins survived `_in_event_mask` (the existing condition). Events present in the classified table but filtered out by `_in_event_mask` are simply excluded from the inner join. This must preserve the `event_id` set, row count, and ordering of today's `event_table.pkl`.

**Boundary-column policy after the join.** Preserve existing replay timing columns from groupby B: `event_start`, `event_end`, `event_duration`, and `event_midpoint_time` remain the filtered survivor-window values. Rename groupby A timing before merge to `label_event_start`, `label_event_end`, `label_event_duration`, and `label_event_midpoint_time`. This keeps `stop_id` assignment, `time_since_stop`, event-arm reward/visit predictors, and downstream shuffles on their current timing contract while still carrying the full label-window boundaries needed for Figure 2 audits.

**Concrete changes in [tables.py](../../../../src/continuum_swr_replay/analysis/replay/tables.py):**

- Delete the inline `is_swr` derivation at [tables.py:368-377](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L368-L377). Keep the existing filtered-window boundary derivation in groupby B; do not replace replay `event_start` / `event_end` with full-window values.
- Replace the call-flow at [tables.py:283-312](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L283-L312) with: (a) build the classified table first (groupby A, on `dot` unfiltered), (b) compute the legacy aggregates as today (groupby B, on `dot` filtered by `_in_event_mask`), (c) inner-join on `event_id`.
- Keep the existing `_in_event_mask` definition unchanged at [tables.py:146-162](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L146-L162) — it defines which events appear in the replay table.
- Update `_event_table_columns()` at [tables.py:1229-1275](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1229-L1275) to add the new columns at the end. New columns: `label_event_start`, `label_event_end`, `label_event_duration`, `label_event_midpoint_time`, `max_speed`, `mean_speed`, `start_speed`, `end_speed`, `endpoints_speed_below_thresh`, `peak_nonlocal_prob`, `overlap_class`, `overlap_subclass`, `n_swrs_overlapping`, `matched_swr_event`, `matched_swr_start`, `matched_swr_end`.
- Update `build_event_table`'s docstring to (i) point at `build_classified_nonlocal_event_table`, (ii) state that existing replay timing columns stay filtered-window while `label_event_*` carries full-window timing, (iii) confirm legacy aggregates are computed over the filtered sub-window (unchanged from today).

### 3. Update the docstring at [tables.py:7](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L7) (module header)

The module header documents `is_swr` as "any `SWR_event > 0` within event." Replace with "strict positive-duration overlap with any SWR; see `analysis/hippocampus/event_table.py`." One line change.

### 4. Tests for `build_classified_nonlocal_event_table`

Create `tests/analysis/hippocampus/test_classified_event_table.py`. Fixtures: synthesize a `dot`-shaped DataFrame in `conftest.py` (or inline) with controlled event labels and SWR labels covering each geometry case from [classify_swr_overlap_geometry](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L339).

Tests to include (full code expected; do not stop at descriptions):

- `test_swr_contained_event`: event fully inside one SWR → `overlap_class == "swr_contained"`, `matched_swr_start <= event_start`, `matched_swr_end >= event_end`.
- `test_swr_partial_single`: event overlaps exactly one SWR with neither containing the other → `overlap_class == "swr_partial"`, `overlap_subclass == "single_swr_partial"`, `n_swrs_overlapping == 1`, `matched_swr_*` from that SWR.
- `test_non_swr`: event disjoint from all SWRs → `overlap_class == "non_swr"`, `matched_swr_*` is NaN, `is_swr == False`.
- `test_boundary_touch_is_non_swr`: SWR ends exactly at event start → `overlap_class == "non_swr"` (strict positive-duration). This is the case where the old inclusive `is_swr` would flip to True; new must stay False.
- `test_event_id_stable`: same input run twice yields identical `event_id` values.
- `test_event_id_unique_within_session`: `validate_event_ids` does not raise.
- `test_peak_prob`: synthesize a known non-local-prob trajectory → `peak_nonlocal_prob` equals the max of that trajectory within the event window.
- `test_max_speed_vs_endpoints`: event with high mid-window speed but immobile endpoints → `max_speed > threshold`, `endpoints_speed_below_thresh == True` (verifies the two flags are independent).
- `test_multiple_sessions_no_leak`: two sessions, each with their own SWR intervals; classifying one session's event must not see another session's SWRs.
- `test_empty_session`: no events → empty frame with the canonical column order.
- `test_no_overlapping_pairs_schema`: session has events but none overlap an SWR → frame is non-empty but `overlap_class` values are all `"non_swr"` and `matched_swr_*` columns are all NaN with the right dtypes. (This replaces the schema check in the deleted `tests/unit/test_event_timing_offsets_empty.py`.)
- `test_intervals_from_labels_accepts_index_time_and_retains_label`: helper works when time is the index and returns the positive `SWR_event` ids alongside `start_time`/`end_time`.
- `test_intervals_from_labels_accepts_time_column`: helper works when tests provide a `"time"` column instead of a time index.

Mark none as `@pytest.mark.slow` — these are fast synthetic tests.

### 5. Back up baseline, regen `event_table.pkl`, and diff old vs new

This task has **four sequential sub-steps**. Do not skip ahead — backing up the baseline AFTER the regen overwrites the only reference point.

**Output location — read before running anything.** `prepare_replay_content_data.py` writes all of its pickles (`event_table.pkl`, `event_arm_table.pkl`, `stop_table.pkl`, …) into `PROCESSED_DATA_DIR/replay_content/` (see `OUTPUT_DIR` at [prepare_replay_content_data.py:57](../../../../scripts/prepare_replay_content_data.py#L57)), NOT directly in `PROCESSED_DATA_DIR`. **The `--test` flag does NOT change the output filenames** — it only limits the run to 2 sessions per animal (`max_per_animal=2`, [prepare_replay_content_data.py:195-199](../../../../scripts/prepare_replay_content_data.py#L195-L199)) and then writes the *same* `event_table.pkl` name. So a `--test` run **overwrites the real, full `event_table.pkl` with a 2-sessions-per-animal partial table.** This is why the backup must happen first and the full regen must happen last.

**Step 5a — Back up the current `event_table.pkl` as the baseline.** Before running any regen (test or full), copy the existing file:

```bash
cp "$PROCESSED_DATA_DIR/replay_content/event_table.pkl" \
   "$PROCESSED_DATA_DIR/replay_content/event_table.pre_phase1.pkl"
ls -la "$PROCESSED_DATA_DIR/replay_content/event_table.pkl" \
       "$PROCESSED_DATA_DIR/replay_content/event_table.pre_phase1.pkl"
```

Confirm the copy exists and matches the original size. If `event_table.pkl` does not exist yet (greenfield, never regenerated), skip the backup and note in the PR description that there is no baseline to diff against — the full-regen output (step 5c) is itself the new baseline, and the diff step (5d) is skipped.

**Step 5b — Smoke-test regen on 2 sessions per animal.**

```bash
uv run python scripts/prepare_replay_content_data.py --test
```

⚠️ This **overwrites** `replay_content/event_table.pkl` (and the sibling pickles) with a partial 2-per-animal table. That is acceptable *only because step 5a already backed up the baseline.* Verify the partial output frame carries the new columns (`label_event_*`, `overlap_class`, `peak_nonlocal_prob`, `matched_swr_*`, etc.) and inspect the log for per-session event counts. Do **not** run the diff against this partial output.

**Step 5c — Full regen (mandatory after the smoke test).**

```bash
uv run python scripts/prepare_replay_content_data.py
```

This overwrites the partial `event_table.pkl` from step 5b with the full table. After this step, `replay_content/event_table.pkl` is the real post-phase-1 table and `replay_content/event_table.pre_phase1.pkl` is the only path back to the old table. **Skipping this step leaves the partial 2-per-animal test table in place — a silent data-loss footgun for every downstream consumer of `event_table.pkl`.**

**Step 5d — Diff script.** Write `scripts/diff_event_table_phase1.py` (deletable after this phase ships) that:

1. Loads `replay_content/event_table.pre_phase1.pkl` (baseline), `replay_content/event_table.pkl` (post-full-regen), and the post-regen `replay_content/stop_table.pkl` for stop-interval checks.
2. Joins on `event_id`.
3. Asserts **population identity**: same row count and same ordered `event_id` sequence as the baseline.
4. Asserts **legacy bit-identity** for: `event_start, event_end, event_duration, event_midpoint_time, current_arm, prev_arm, next_arm, last_rewarded_arm, current_patch, prev_patch, next_patch, is_mua, is_after_switch, is_after_omission, is_after_reward, is_after_stay, reward_outcome, is_stay, n_nonlocal_bins, nonlocal_weight_sum, posterior_*, n_junction_bins, n_unique_arms, time_since_stop, time_since_port_entry, time_since_reward, stop_idx, stop_id`.
5. Reports for **additive full-window timing columns** (`label_event_start`, `label_event_end`, `label_event_duration`, `label_event_midpoint_time`): count of events where label-window timing differs from legacy timing, median shift, and max shift. Confirm every legacy `event_start` still falls within its `stop_id` interval and every `label_event_*` value is finite for non-empty rows.
6. Reports `is_swr` drift: count and fraction of events where the new strict-positive-duration `is_swr` disagrees with the old inclusive-timebin value.
7. Reports classified-column completeness and counts: non-null fraction for `label_event_*`, `peak_nonlocal_prob`, `overlap_class`; class counts for `overlap_class`; and non-null matched-SWR fraction among `is_swr` rows. `event_table.pkl` itself must have zero new-vs-old event-id delta.

If any legacy-bit-identity column disagrees, **stop** — Groupby B (the filtered legacy aggregate path) is not preserving today's semantics. Investigate before continuing.

### 6. Run downstream replay-content tests and sanity-check

```bash
uv run pytest tests/unit/test_event_table_pre_trial_no_leak.py
uv run pytest tests/unit/test_event_patch_table_nan.py
uv run pytest tests/unit/test_id_uniqueness_preconditions.py
uv run pytest tests/unit/test_retrospective.py
uv run pytest tests/analysis/hippocampus/  # event detection / classification tests
```

If any of these fail, the refactor changed observable behavior beyond `is_swr`. Fix or surface as a real population change before merging.

The downstream regression-fitting scripts (`run_stage5b_retrospective.py`, `run_stage6_learning.py`) consume `event_table.pkl`. **They are not re-run in this phase** — phase-1 ships when the table regenerates, tests pass, and the diff report is acceptable. Re-running the regression models is its own follow-up (and is the user's call, since the model-comparison artifacts in the paper depend on those outputs).

### 7. Documentation updates within this phase

- Module docstring on `analysis/hippocampus/event_table.py` describing the new public function and what each new column means.
- Update [tables.py:7-30](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L7-L30) module header to point at the new classified-event-table module.
- Add a one-paragraph note to [CLAUDE.md](../../../../CLAUDE.md) under "Spike Data Processing" or a nearby section explaining that the canonical per-event table lives in `analysis/hippocampus/event_table.py` and that the replay-content tables build on it.

## Deliberately not in this phase

- **Figure 2 rewire.** Panel a/b/c/d still call their existing helpers. Phase 2 swaps panels a/b/d to consume the new table and parity-tests panel c against it.
- **Deleting `pick_example_events`, `compute_swr_overlap_class_counts`, `compute_event_timing_offsets`.** Phase 2's job. Phase 1 leaves them functional and unchanged.
- **Rerunning regression-fitting scripts (`run_stage5b_retrospective.py`, `run_stage6_learning.py`).** Out of scope; user decides when to refit the paper's replay-content models.
- **Caching `build_classified_nonlocal_event_table` to its own pickle.** It's a thin transformation of `dot`; the cached replay `event_table.pkl` carries the classified columns only for the replay `_in_event_mask` population and with full-window timing renamed to `label_event_*`. Figure 2 computes the unfiltered table on demand. Revisit only if Figure 2's panel-a path becomes a bottleneck.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_swr_contained_event` | Event fully inside one SWR → `overlap_class == "swr_contained"`; matched SWR boundaries span the event. |
| `…::test_swr_partial_single` | Event overlaps exactly one SWR partially → `overlap_class == "swr_partial"`, `overlap_subclass == "single_swr_partial"`, `n_swrs_overlapping == 1`. |
| `…::test_non_swr` | Disjoint event → `overlap_class == "non_swr"`, `matched_swr_*` is NaN, `is_swr == False`. |
| `…::test_boundary_touch_is_non_swr` | SWR ends at event start → not overlapping under strict positive-duration rule. |
| `…::test_event_id_stable` | Same input twice → same `event_id`. |
| `…::test_event_id_unique_within_session` | `validate_event_ids` passes on the output. |
| `…::test_peak_prob` | Peak Non-Local prob matches synthetic max within event window. |
| `…::test_max_speed_vs_endpoints` | `max_speed` and `endpoints_speed_below_thresh` are independent; high-mid-speed + immobile endpoints sets the flag True with max above threshold. |
| `…::test_multiple_sessions_no_leak` | SWR intervals from one session don't classify another session's events. |
| `…::test_empty_session` | No events → empty frame, canonical columns, no crash. |
| `…::test_no_overlapping_pairs_schema` | Events exist but none overlap an SWR → `overlap_class` all `"non_swr"`, `matched_swr_*` all NaN with correct dtypes. (Replaces deleted `tests/unit/test_event_timing_offsets_empty.py`.) |
| `…::test_intervals_from_labels_accepts_index_time_and_retains_label` | Interval helper accepts production-style time index and keeps the positive SWR label id. |
| `…::test_intervals_from_labels_accepts_time_column` | Interval helper accepts test-style `"time"` column and returns the same schema. |
| `tests/unit/test_event_table_pre_trial_no_leak.py` (existing) | Refactored `build_event_table` still passes; trial-derived NaN propagation unchanged. |
| `tests/unit/test_event_patch_table_nan.py` (existing) | Refactored `build_event_table` still passes; patch-NaN behavior unchanged. |
| `tests/unit/test_id_uniqueness_preconditions.py` (existing) | `event_id` uniqueness preconditions still hold. |
| `tests/unit/test_retrospective.py` (existing) | Downstream retrospective analysis still passes on the regenerated `event_table.pkl`. |
| Manual: `scripts/diff_event_table_phase1.py` output | Replay row count and ordered `event_id` sequence unchanged; legacy timing/aggregate columns bit-identical; `is_swr` drift <1%; label-vs-legacy timing deltas documented. |

## Fixtures

- Synthetic `dot`-shaped DataFrame for the new tests. Build it inline (`pd.DataFrame({...})`) with ~50 rows covering the geometry cases. No real-data fixture required for this module.

  **Required columns** (must match what `build_classified_nonlocal_event_table` reads, plus what the phase-2 parity test of `get_event_types` needs):
  - `animal` (str), `day` (int), `epoch` (str) — session keys.
  - `non_local_event` (int) — 0 = no event; positive ints label events.
  - `SWR_event` (int) — 0 = no SWR; positive ints label SWRs.
  - `MUA_event` (int, optional) — only needed if a test exercises `get_event_types`'s MUA path; default 0 otherwise.
  - `head_speed` (float, cm/s).
  - `Non-Local_Continuous_prob` (float, [0,1]), `Non-Local_Fragmented_prob` (float, [0,1]).
  - Index named `time` (float, seconds), monotonically increasing at a fixed dt (e.g., 1/500 s).

  **Geometry cases the fixture must cover** (one event per case, in distinct `non_local_event` ids within one session, sampled at 500 Hz):
  - one event fully inside one SWR (`swr_contained`),
  - one event overlapping exactly one SWR at one boundary (`single_swr_partial`),
  - one event strictly containing one SWR (`event_contains_swr`),
  - one event disjoint from all SWRs (`non_swr`),
  - one event whose start coincides with an SWR end and zero positive-duration overlap (boundary-touch; must classify as `non_swr` under strict positive-duration),
  - one event with high mid-window speed but immobile endpoints (for `test_max_speed_vs_endpoints`),
  - one single-timebin event (to verify the `iloc[0] == iloc[-1]` collapse),
  - one event in a *second* session with overlapping SWR ids in the first session's namespace (for `test_multiple_sessions_no_leak`).

  Each event's `Non-Local_*_prob` columns should peak somewhere inside the event so `peak_nonlocal_prob` is testable to known values.

- For the regen step, the real `data_over_time` pickles in `PROCESSED_DATA_DIR` are the input; the existing prepare script handles loading.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases (specifically: `pick_example_events` and the figure-side helpers are untouched).
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- The diff script `scripts/diff_event_table_phase1.py` is committed and produces the expected report; PR description carries the report's output.
- Old code paths flagged for removal in this phase are actually removed (specifically: the inline `is_swr` definition in `build_event_table`).
- User-facing documentation listed as tasks is updated, not deferred — module docstrings, `tables.py` module header, and the `CLAUDE.md` note.
