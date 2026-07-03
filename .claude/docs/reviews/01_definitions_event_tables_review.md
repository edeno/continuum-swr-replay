# Event-Definition and Table-Construction Audit

## Findings

1. **Medium: sorted time-index is a silent precondition.** Event timing is taken from the first/last row in each grouped event, not from sorted min/max time, in both the classified table and replay event table ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:284](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L284), [src/continuum_swr_replay/analysis/replay/tables.py:383](../../../src/continuum_swr_replay/analysis/replay/tables.py#L383)). Several helpers also use `np.searchsorted` or transition logic assuming chronological `dot.index` ([src/continuum_swr_replay/analysis/replay/_helpers.py:83](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L83), [src/continuum_swr_replay/analysis/replay/_helpers.py:166](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L166)). If a manually constructed or concatenated frame is not sorted within session, durations, midpoint, SWR overlap, stop assignment, and trial context can silently corrupt. Tests cover unsorted `trials`, but not unsorted `dot`.

2. **Medium: replay table builders are per-session by convention, not by guard.** `build_event_table` reads `animal/day/epoch` from the first row and groups only by `non_local_event` ([src/continuum_swr_replay/analysis/replay/tables.py:337](../../../src/continuum_swr_replay/analysis/replay/tables.py#L337), [src/continuum_swr_replay/analysis/replay/tables.py:379](../../../src/continuum_swr_replay/analysis/replay/tables.py#L379)). That is consistent with the module's per-session contract, but a multi-session `dot` would mix same-numbered labels across sessions and hash them under the first session key. The classified table is session-scoped; the replay table should either enforce one session or internally group by session.

3. **Medium: non-contiguous duplicate labels are collapsed into one event.** `intervals_from_labels` groups by label id and takes min/max time ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:112](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L112)); `build_classified_nonlocal_event_table` and `build_event_table` also group by `non_local_event` id ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:284](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L284), [src/continuum_swr_replay/analysis/replay/tables.py:379](../../../src/continuum_swr_replay/analysis/replay/tables.py#L379)). This is safe only if detector labels are unique contiguous runs within session. A reused label after a gap would become one long event, potentially creating false SWR overlap and wrong stop/content summaries. I did not find a test that asserts label contiguity or fails on duplicate non-contiguous labels.

4. **Low/test gap: full label windows and filtered survivor windows are documented and implemented, but not directly regression-tested when they differ.** `build_event_table` filters rows by label, immobility, and non-local probability ([src/continuum_swr_replay/analysis/replay/tables.py:180](../../../src/continuum_swr_replay/analysis/replay/tables.py#L180), [src/continuum_swr_replay/analysis/replay/tables.py:352](../../../src/continuum_swr_replay/analysis/replay/tables.py#L352)), while `label_event_*` and `is_swr` are merged from the full label window ([src/continuum_swr_replay/analysis/replay/tables.py:593](../../../src/continuum_swr_replay/analysis/replay/tables.py#L593)). Existing synthetic replay-table tests use events whose whole label window survives the filter, so they do not prove that `event_start`/`event_end` can shrink while `label_event_start`/`label_event_end` and SWR class remain full-window.

## Verdict

Approve with reservations for the canonical SWR/non-SWR classification path. The strict positive-duration overlap rule is consistently implemented from `SWR_event` labels, and the fast scoped test suite passes. The main risks are input-contract risks: sorted time, single-session replay-table inputs, and one contiguous interval per event label should be made explicit and tested before publication-scale regeneration.

## Verified Behavior

- Non-local event detection uses `prob > threshold`, not `>=`, and applies `minimum_duration` through `segment_boolean_series` ([src/continuum_swr_replay/analysis/hippocampus/event_detection.py:77](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L77)). Unit tests pin threshold, duration, boundary, empty, and speed behavior ([tests/unit/test_event_detection.py:47](../../../tests/unit/test_event_detection.py#L47), [tests/unit/test_event_detection.py:216](../../../tests/unit/test_event_detection.py#L216)).
- The classified non-local event table is full-label-window by design: no immobility or probability filter is applied before timing/SWR classification ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:237](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L237)).
- The replay `event_table` uses filtered survivor windows for legacy timing and content summaries, then left-merges full-window classification/speed/peak-probability fields by `event_id` with `validate="one_to_one"` and an unmatched-row guard ([src/continuum_swr_replay/analysis/replay/tables.py:219](../../../src/continuum_swr_replay/analysis/replay/tables.py#L219), [src/continuum_swr_replay/analysis/replay/tables.py:635](../../../src/continuum_swr_replay/analysis/replay/tables.py#L635)).
- `is_swr` is strict positive-duration overlap from label-derived `SWR_event` intervals: the predicate is `(swr_end > event_start) & (swr_start < event_end)` in direct overlap code and geometry classification ([src/continuum_swr_replay/analysis/hippocampus/event_detection.py:244](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L244), [src/continuum_swr_replay/analysis/hippocampus/event_detection.py:451](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L451)). The classified table derives SWR intervals from `SWR_event`, not `ripple_times` ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:316](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L316)).
- `ripple_times` appears only as a plotting-consistency guard for selected examples, not as the canonical table classifier ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:368](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L368)).
- Event and stop IDs are stable hashes over session keys plus event/stop index ([src/continuum_swr_replay/analysis/replay/_helpers.py:301](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L301), [src/continuum_swr_replay/analysis/replay/_helpers.py:324](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L324)); classified-table hash collisions are checked with `validate_event_ids` ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:361](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L361)).

## Indexing and Alignment Risks

- Durations are `last_timestamp - first_timestamp`, not inclusive sample count times `dt`; tests intentionally pin that an 11-bin event has 10 intervals and a one-bin event has duration 0 ([tests/analysis/hippocampus/test_classified_event_table.py:225](../../../tests/analysis/hippocampus/test_classified_event_table.py#L225), [tests/analysis/hippocampus/test_classified_event_table.py:235](../../../tests/analysis/hippocampus/test_classified_event_table.py#L235)). Confirm this is the intended physical duration for manuscript duration claims.
- Event-to-stop assignment uses filtered `event_start` and requires it to fall inside an immobility bout ([src/continuum_swr_replay/analysis/replay/tables.py:581](../../../src/continuum_swr_replay/analysis/replay/tables.py#L581)). This is coherent for filtered survivor windows, but should be tested when the full label window begins before the survivor window.
- Trial context now sorts `trials` and uses NaN-strict first-row extraction, preventing future-context leakage ([src/continuum_swr_replay/analysis/replay/_helpers.py:75](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L75), [src/continuum_swr_replay/analysis/replay/_helpers.py:264](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L264)). Pre-trial event tests cover this regression ([tests/unit/test_event_table_pre_trial_no_leak.py:110](../../../tests/unit/test_event_table_pre_trial_no_leak.py#L110)).

## Invariants Checked

- Ran:

```bash
uv run pytest tests/analysis/hippocampus/test_classified_event_table.py tests/unit/test_event_detection.py tests/unit/test_swr_overlap_consistency.py tests/unit/test_replay_tables.py tests/unit/test_event_table_pre_trial_no_leak.py tests/unit/test_trial_sort_invariance.py tests/unit/test_stop_table_bout_start_no_leak.py
```

Result: **116 passed in 7.93 s**.

- Tests cover every SWR geometry, including contained, partial, event-contains-SWR, multiple-SWR gap, boundary touch, single-bin events, and cross-session isolation ([tests/analysis/hippocampus/test_classified_event_table.py:29](../../../tests/analysis/hippocampus/test_classified_event_table.py#L29)).
- Tests verify boundary-touch intervals are non-SWR for both direct overlap and classified table paths ([tests/unit/test_event_detection.py:330](../../../tests/unit/test_event_detection.py#L330), [tests/analysis/hippocampus/test_classified_event_table.py:178](../../../tests/analysis/hippocampus/test_classified_event_table.py#L178)).
- Tests verify table-driven overlap counts match the direct classifier across geometries ([tests/unit/test_swr_overlap_consistency.py:93](../../../tests/unit/test_swr_overlap_consistency.py#L93)).
- Cross-table tests verify event-stop membership, stop `n_events`, event-sequence row counts, posterior row sums, and event-sequence/event-table ID agreement ([tests/unit/test_replay_tables.py:487](../../../tests/unit/test_replay_tables.py#L487)).

## Test Integrity

The tests are generally high value and would fail on the most dangerous historical regressions: inclusive SWR overlap, clipped SWR intervals, cross-session SWR leakage in the classified table, time-column/index drift, future trial-context leakage, and event/sequence ID drift.

Missing tests to add before publication reruns:

- unsorted `dot.index` should either sort within session or raise clearly;
- multi-session input to per-session replay builders should raise;
- same positive `non_local_event` label appearing in two separated runs should raise;
- filtered survivor window shorter than full label window should preserve both `event_*` and `label_event_*` with expected values;
- NaN-only speed/probability inside an event should have explicit semantics;
- an event whose full label window touches or crosses two immobility bouts should be rejected or assigned by documented rule.

## Numerical and Performance Notes

No heavy numerical computation is in this layer. Interval geometry uses a 1 microsecond tolerance only for SWR union coverage in the multi-SWR subclass calculation ([src/continuum_swr_replay/analysis/hippocampus/event_detection.py:42](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L42)). I did not run floating-point fuzz tests around that tolerance.

Overlap classification broadcasts event-by-SWR arrays per session ([src/continuum_swr_replay/analysis/hippocampus/event_detection.py:241](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L241)). This is fine for the test sizes and likely fine per session, but no realistic memory benchmark was run.

## Scientific Conclusion Status

Within the tested contracts, the definitions support the paper's core denominator: non-local events are classified as SWR-coincident only by strict positive-duration overlap with `SWR_event` labels. I did not run data-heavy checks on the full `references/nihpp-2024.09.23.613567v1.pdf`-derived dataset or regenerate claim counts, so this review does not validate the reported prevalence numbers.

## Next Checks

1. Add a shared `validate_dot_event_contract` for monotonic time within session, one session for replay builders, positive labels forming one contiguous run, nonnegative durations, and no missing session keys.
2. Run the validator across every saved `data_over_time.pkl` before any figure/table regeneration; report counts of duplicate labels, unsorted frames, zero-duration events, and events with filtered windows shorter than label windows.
3. Add a one-session fixture where full label windows and filtered survivor windows intentionally differ, and assert downstream columns use the intended boundary.
4. Recompute the claim-table denominators from the canonical classified table and compare to the stated `39,500/65,913` non-SWR count before quoting it.
