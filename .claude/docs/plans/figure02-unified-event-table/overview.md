# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Background

Today, four Figure 2 panels derive their event populations independently:

| Panel | Event source | Boundary | Immobility filter | Overlap rule |
|-------|--------------|----------|-------------------|--------------|
| a (examples) | live `results.acausal_state_probabilities` | sub-sample threshold-crossing via `get_confident_times` | `max_speed < 4 cm/s` | 3-way `classify_swr_overlap` |
| b (bar) | `data_over_time` labels (min/max of bins) | sample-quantized | endpoints (first AND last bin) | 3-way `classify_swr_overlap` |
| c (triggered avg) | `data_over_time` labels via `get_event_types` | sample-quantized | endpoints | binary, `event_strict_overlap_keys` (strict positive duration) |
| d (timing offsets) | `data_over_time` labels | sample-quantized | **none** | binary, strict positive duration |

And `analysis/replay/tables.py:build_event_table` (the replay-content table) uses a *fourth* rule: `is_swr = bool((g["SWR_event"] > 0).any())` over per-event timebins — inclusive-timebin overlap.

That's three or four conventions for the same "is this non-local event SWR-coincident" question, depending on how you count. Even when the boundaries are consistent (b/c/d), an event that sits next to a SWR but doesn't quantize-overlap can be counted differently by `classify_swr_overlap` (used by b) vs `event_strict_overlap_keys` (used by c) — and silently does so.

## Current codebase integration points

- `src/continuum_swr_replay/analysis/replay/tables.py:168-499` — `build_event_table`. The existing canonical per-event table (one row per non-local event with timing + replay-specific context). Filters to *immobile + non-local-prob-above-threshold* timebins before deriving event boundaries, which means its `event_start/event_end` are the surviving-timebin window, not the original label boundary. Phase 1 splits the un-filtered, classification-focused core out of this and recomposes.
- `src/continuum_swr_replay/analysis/replay/tables.py:368-377` — current `is_swr` definition (inclusive-timebin). Phase 1 replaces.
- `src/continuum_swr_replay/analysis/hippocampus/event_detection.py:255-336` — `classify_swr_overlap` (3-way, strict positive duration). Phase 1 reuses verbatim.
- `src/continuum_swr_replay/analysis/hippocampus/event_detection.py:339-509` — `classify_swr_overlap_geometry` (6-way subclass). Phase 1 reuses for the optional `overlap_subclass` column.
- `src/continuum_swr_replay/analysis/hippocampus/event_analysis.py:232-365` — `compute_event_timing_offsets`. Phase 2 deletes (panel d reads `matched_swr_start/end` columns directly).
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:540-609` — `compute_swr_overlap_class_counts`. Phase 2 simplifies to read from the new table.
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:420-537` — `get_event_types`. Phase 2 leaves alone (panel c continues to use it) but adds a parity test.
- `scripts/generate_figure02.py:102-129` — `pick_example_events`. Phase 2 reduces to a table filter.
- `scripts/generate_figure02.py:142-269` — `generate_figure`. Phase 2 adds the example-class assertion and updates logging.
- `scripts/prepare_replay_content_data.py:165-167` — call site for `build_event_table`. Phase 1 re-runs this script to regenerate `event_table.pkl`.

## Scope and dependency policy

### Goals

- One canonical per-non-local-event table with timing, peak non-local probability, per-event speed summaries (max and endpoints), 3-way SWR overlap class (and optional 6-way subclass), and the matched SWR's start/end.
- Figure 2 panels a/b/d read from this table directly; panel c's triggered-average helper remains in place for this phase but is parity-tested against the table so overlap splits cannot silently drift.
- Replay-content tables in `analysis/replay/tables.py` are built on top of the same canonical core (single overlap rule everywhere).
- Figure 2's `pick_example_events`, `compute_swr_overlap_class_counts`, and `compute_event_timing_offsets` either shrink to a table filter or are deleted. `get_event_types` remains because other figures use it.
- The example event drawn in panel a is provably in the same class column it's plotted under — enforced by an assertion in `generate_figure02.py`.

### Non-Goals

- Changing what events count as "non-local" (the existing threshold + minimum duration in `get_confident_times` / `_in_event_mask` are out of scope).
- Adding new immobility methods. The new table exposes both `max_speed < threshold` and `endpoints < threshold` as flags; panels pick which to apply. We do not introduce e.g. mean-speed, fraction-immobile, etc.
- Changing Figure 2's narrative or panel order.
- Rewriting Figure 3 panel a (which reuses `plot_event_example_stack`). The plotting helper continues to accept the same arguments; only the *selector* in front of it changes.
- Touching the figurl pipeline or anything related to interactive visualization.

### Dependency policy

No new third-party dependencies. The new module uses pandas, numpy, and existing helpers in `analysis/hippocampus/event_detection.py`.

## Metrics

- **Legacy bit-identity for replay-content columns.** After phase 1, regenerating `event_table.pkl` reproduces — row-for-row — the existing replay-content population and the columns derived from the legacy filtered window: `event_start`, `event_end`, `event_duration`, `event_midpoint_time`, `current_arm`, `prev_arm`, `next_arm`, `last_rewarded_arm`, `*_patch`, `is_after_*`, `reward_outcome`, `is_stay`, `n_nonlocal_bins`, `nonlocal_weight_sum`, posterior columns (`posterior_*`), `n_junction_bins`, `n_unique_arms`, `time_since_*`, `stop_idx`, and `stop_id`. Phase 1 task 2 keeps the filtered groupby for these aggregates and timing columns, so they cannot drift.
- **Classified full-window timing is additive in replay tables.** The new classified table derives full label-window timing as `event_start`, `event_end`, `event_duration`, and `event_midpoint_time`. When those columns are merged into replay-content `event_table.pkl`, keep the legacy filtered-window timing names unchanged and add the classified timing under `label_event_start`, `label_event_end`, `label_event_duration`, and `label_event_midpoint_time`. The diff script reports the label-vs-legacy timing deltas but does not permit the legacy timing columns to move.
- **Replay population identity.** `event_table.pkl` must have the same `event_id` set and row count before and after phase 1. Events present in the classified table but absent from `_in_event_mask` stay out of replay-content outputs. Any new-vs-old `event_id` set delta is a stop-and-investigate failure, not an expected result.
- **is_swr drift.** Count of events where the new strict-positive-duration `is_swr` disagrees with the old inclusive-timebin `is_swr`. Expected change; sanity bound <1% of events.
- **Figure 2 panel parity.** Regenerate `figure02.{pdf,png}` before and after phase 2; compare panels b/c/d numerically (counts in b; mean curves in c; histogram bins in d) and panel a visually (example events may flip if the per-event-table classification disagrees with the old `pick_example_events`).
- **Test parity.** All existing unit tests under `tests/` pass at phase 1 boundary and phase 2 boundary.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Changing `is_swr` semantics shifts coefficients in replay-content regression models that have already been run for the paper. | Phase 1 documents the change in the function's docstring and the prepare-script log; the regen step diffs old vs new `event_table.pkl` and reports `is_swr` drift before fitting any model. Downstream regression results must be regenerated. |
| `build_event_table`'s current `event_start/event_end` are the immobile-high-prob *survivors* of the timebin filter, not the original event-label boundary. Figure 2 needs the original boundary, but replay-content consumers already use the existing timing and `stop_id` semantics. | The new `analysis/hippocampus/event_table.py:build_classified_nonlocal_event_table` derives full boundaries from `dot.groupby("non_local_event")` over *all* event-label-positive rows (no speed/prob filter). The replay-side `build_event_table` keeps its existing timing columns and adds full-window timing as `label_event_*`, preserving replay population, `stop_id`, and event-arm predictor behavior. |
| Panel c uses `get_event_types`, which calls `event_strict_overlap_keys`. After phase 1, `overlap_class != "non_swr"` from the new table should be equivalent — but the two implementations live separately. | Phase 2 adds a parity test asserting `event_strict_overlap_keys(dot, "non_local_event", "SWR_event")` keys equal the `(animal, day, epoch, non_local_event)` keys of the new table where `overlap_class != "non_swr"`. If they disagree, fail loudly. |
| Hash collisions in `event_id` once we filter the table differently for figures vs regression. | `event_id = hash(animal, day, epoch, non_local_event)` is already stable across runs ([_helpers.py:301-310](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L301)); the new table inherits it. Tests assert uniqueness within session. |

## Rollout Strategy

Phase 1 ships in one PR: new module + refactored `build_event_table` + cache regen. After merge, `event_table.pkl` is regenerated; downstream replay-content models (`run_stage5b_retrospective.py`, `run_stage6_learning.py`) must be rerun before their results are considered current. Phase 1 lists this in its `Review` block.

Phase 2 ships in one PR: Figure 2 panel-a/b/d rewire + panel-c parity test + assertion + helper deletions. It depends on phase 1 (the table must already carry the new columns), but doesn't require regenerating `event_table.pkl` again — phase 2 only changes consumers, not producers.

No feature flag, no deprecation period. The old inclusive `is_swr` is gone after phase 1. The figure-side helpers are gone after phase 2.

**Downstream artifacts.** Figures and analyses that consume `event_table.pkl` — at minimum, Figure 3 (`scripts/generate_figure03.py`) and the replay-content regression outputs from `run_stage5b_retrospective.py` / `run_stage6_learning.py` — will temporarily mismatch their last-published versions until those scripts are rerun against the regenerated table. Phase 1's PR description must list which downstream artifacts are now stale. Refitting is out of scope for both phases of this plan; the user decides when to refresh those artifacts.

## Open Questions

1. **Should `build_classified_nonlocal_event_table` cache to disk, or is it cheap enough to compute on demand?** Current best answer: compute on demand. The replay-content prepare script already drives it once per session and persists the filtered replay table with the classified columns. Figure 2 computes the unfiltered classified table from `data_over_time` on demand because its population differs from replay `_in_event_mask`.
2. **Does panel d's population gain immobility filtering, or stay session-wide?** Current best answer: align with panels b/c (require immobility). The figure caption already implies the same population as b/c. Phase 2 makes this explicit and documents in the panel's docstring.

## Estimated Effort

- **Phase 1:** ~250 net LOC. New module ~200 LOC (mostly column derivation + classification glue); `build_event_table` shrinks by ~100 LOC where boundary/is_swr/peak code moves out; ~150 LOC of tests for the new module.
- **Phase 2:** ~−150 net LOC (deletions outnumber additions). `pick_example_events` shrinks from ~30 LOC to a 5-line filter; `compute_swr_overlap_class_counts` shrinks similarly; `compute_event_timing_offsets` deleted entirely (~80 LOC removed) and replaced by a 10-line column subtraction.
