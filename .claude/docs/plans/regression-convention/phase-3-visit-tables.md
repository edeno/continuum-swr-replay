# Phase 3 — The well visit as the unit: clocks, stop ends and new covariates in the tables

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#rate-design)

Branch: `visit-clocks` from master after the precondition of C3. Code does not depend on Phases 1 and 2,
so it can run beside them. Ends with one rebuild of the canonical tables.

**Inputs to read first:**

- The design, sections 2.7, 4.1, 4.2 (uncertainty), 4.7 (clock overlap), and section 9's rounds on decision
  2 and the outside events.
- The visit labels on master: `src/continuum_swr_replay/analysis/event_tables/visit_context.py`
  (`window_visit_context`, `visit_context_of_samples`; `VISIT_CONTEXT_COLUMNS`) and how
  `stop_table.py` and `event_table.py` call it (C3 lists what they are).
- `src/continuum_swr_replay/analysis/event_tables/stop_table.py:61-262` (`build_stop_table`; the clocks at
  `:188-189`; reward lags at `:160-176`).
- `src/continuum_swr_replay/analysis/event_tables/event_table.py:328` (`_attach_time_since_stop`), `:545`
  (`build_event_table`); `arm_patch_tables.py:57` (`build_event_arm_table`).
- `src/continuum_swr_replay/analysis/behavior/covariates/candidate_arm.py:76-113` (`_history_by_event`),
  `:116` (`attach_candidate_arm_covariates`); `behavior/covariates/history/reward_history.py:32`, `:97`
  (the windowed reward counts and fractions to mirror); `behavior/covariates/history/_common.py:29`
  (`_prepare_port_experiences`: `arrival_time`, `departure_time`).
- `src/continuum_swr_replay/analysis/replay_models/per_bin_design.py:394-1029` (`aggregate_to_timebins`,
  its at-risk coverage `_event_coverage_by_bin` `:911`, excluded-time drop `_drop_excluded_bins` `:981`, the
  outcome-boundary shift columns `:85-91`): the logic the visit design reuses.
- `src/continuum_swr_replay/tables/covariate_registry.py:601` (`stamp_covariate_schema`) and the registry
  rows; `tables/versions.py:25` (`COVARIATE_SCHEMA_VERSION`); `docs/covariate_dictionary.md`.
- `scripts/tables/prepare_replay_content_data.py:490-560` (`--from-cache`, `--output-dir`),
  `scripts/checks/compare_to_golden_baseline.py` (`--expected-added-columns`), `scripts/checks/table_parity.py`,
  `scripts/tables/write_regeneration_manifest.py`, `docs/reproduce.md:59-66` (table build order).
- Scratch: `S/explore_leave/part_a.py`, `part_b.py`, `part_b4.py`, `S/my_checks/three_clock_overlap.py`,
  `leave_definition_check.py`, `S/verify_rate/s1_covariates.py` ([appendix](appendix.md#scratch-map)).

**Contracts referenced:** [C2](shared-contracts.md#c2-data-the-database-git-hygiene),
[C3](shared-contracts.md#c3-precondition-the-visit-labels-are-on-master),
[C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention) rule 1, [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session).

**Designs referenced:** [rate design](designs.md#rate-design), [content design](designs.md#content-design)
(the uncertainty measures).

**Decisions this phase applies** (all settled by the lead on 2026-11-20; [overview.md](overview.md#decisions-for-the-lead)):
L2 (the two uncertainty measures, defined in [designs.md#content-design](designs.md#content-design)), L3
(arrival is the first poke-in; `last_well_arrival_time_s` equals it wherever both exist), L4 (the event set
is the events that start inside a visit; a stop crossing the final poke-out counts up to it), ASK-1 (the
ContFrag tables get the same new columns), ASK-4 (the dense stop time-bin table is removed), and C14 (one
unit, one event set, one set of clocks).

## Tasks

- **Check the precondition** (C3) and record `count_visit_context` for the event, stop and event-arm
  tables in the PR as the "before" counts.
- **Stop table: the visit columns** (in `build_stop_table`, `stop_table.py:61`, using the
  `port_experiences` it already takes; vectorized with `np.searchsorted` over visit intervals):
  - `well_visit_id` (`Int64`) for every row that starts inside a visit, `spans_visit_end` included (the
    labels branch sets it for `in_visit` rows only): the port experience whose
    `[arrival_time, departure_time)` holds the row's first analyzed sample. Same change in the event table
    and the event-arm and event-patch tables (`visit_context.window_visit_context`, its `well_visit_id`
    assignment).
  - `overlapping_visit_id` (`Int64`): the visit the stop overlaps at all (a stop overlaps at most one: the
    animal moves between wells), so a `spans_visit_start` stop (stopped at the well before the first
    poke-in) is placed too; missing when none.
  - `visit_first_poke_in_s`, `visit_last_poke_out_s`: that visit's `arrival_time` and `departure_time`.
  - `in_visit_immobile_s`: seconds of the stop inside that visit, less excluded time.
  - `visit_stretch_index`: 0, 1, ... for the stops of one visit in time order.
  - `stop_end_kind`: `"leaves"` when no later stop of the same visit starts before
    `visit_last_poke_out_s`, `"small_movement"` when one does, `"outside_visit"` when
    `overlapping_visit_id` is missing. With the visit ending at the last poke-out (design 4.1), no position
    or speed threshold is needed.
- **Event table: the visit clocks** (in `build_event_table`, `event_table.py:545`, next to the visit context):
  for events that start inside a visit (`in_visit` and `spans_visit_end`), `time_since_visit_start_s`
  (event start minus first poke-in), `time_until_visit_end_s` (last poke-out minus event start, positive for
  both labels), `time_until_stretch_end_s` (end of the containing stop, clipped at the last poke-out, minus
  event start); missing otherwise. Carry them to the event-arm table as event context columns
  (`arm_patch_tables.py:57`) so the content design reads them per row.
- **Current arm of an event that crosses the final poke-out**: for `spans_visit_end` events, `current_arm`,
  `is_current_arm` and the current patch are the visit's well's arm and patch (check what the builder gives
  today; it reads position at the event; set it from the visit if they differ, and count the changed rows in
  the PR).
- **Event-arm table: the two new uncertainty covariates** (L2, lead: "Do your rec"), computed in
  `_history_by_event` (`candidate_arm.py:76`) beside the reward count and fraction, from outcomes known
  before the event (the same rule as `recent_reward_rate_all_events`, `reward_history.py:97`):
  `reward_outcome_variance_candidate_arm_last10_visits` (the variance, `ddof = 0`, of the arm's 0/1
  outcomes over the last 10 visits to its well) and `log1p_n_visits_candidate_arm` (log of one plus the
  visits to the well earlier in the epoch). Trunk arms get the rule `attach_arm_uncertainty` uses
  (`value_block.py:206`: 0 where the stop is present, NaN where absent). Names are calls; Phase 6a's gate may
  change a definition, in which case it changes here in a follow-up rebuild of the same phase family, not by
  a second column.
- **The same columns in the ContFrag tables** (ASK-1, lead: "Move them"): the ContFrag stop, event and
  event-arm tables (`scripts/tables/prepare_contfrag_replay_content_data.py`, `analysis/contfrag/`) gain
  `well_visit_id`, the visit clocks, the current-arm rule above and the two uncertainty covariates through
  the same functions, so Phases 5a and 6a can fit the SWR- and MUA-sourced events as two more event sources.
  The visit labels themselves come from the labels branch (lead: "Label them now").
- **Register and document**: every new column in `tables/covariate_registry.py` (producer, table, timing)
  and in `docs/covariate_dictionary.md`; bump `COVARIATE_SCHEMA_VERSION` (`tables/versions.py:25`) only if
  the registry's contract requires it for added columns (read `assert_versions_compatible`, `:37`).
- **The visit bin design** (`src/continuum_swr_replay/analysis/replay_models/visit_design.py`, new):
  `visit_timebins(stop_table, onsets, events, *, kind, bin_s=RATE_BIN_S, excluded_intervals=None)` builds
  15 ms bins over every immobility stop of a session (not only the visit's), as in
  [designs.md#rate-design](designs.md#rate-design): each bin carries `visit_context` (`in_visit` when
  inside a visit interval and a stop; otherwise the stop's own label), `well_visit_id` (from the visit
  intervals, so it is set for the in-visit bins of `spans_visit_start` and `spans_visit_end` stops),
  `visit_stretch_index`, `exposure`, `count` (starts of the kind's events that start inside a visit, C3),
  the three clocks `s`,
  `u`, `m` (missing outside a visit), the last known outcome and the ten known lags with the outcome shift
  at `outcome_known_time_s`. At-risk time excludes time inside an ongoing event of the kind (reuse
  `_event_coverage_by_bin`, generalized from stop time to absolute time) and excluded time. The rate model
  then calls `select_visit_context(bins)` (C4). Vectorized over bins; no Python loop over bins.
- **Rebuild the canonical tables once**, into a staging root first:
  1. Freeze the current tables as the baseline (`scripts/checks/freeze_regression_baseline.py`, or copy the
     table files to `Processed-Data/replay_content/_baseline_pre_visit_clocks/`).
  2. Build into a staging root, from the behavior cache, in the order of `docs/reproduce.md:59-66`:
     `prepare_replay_content_data.py --from-cache --stop-only --output-dir <staging>`, then
     `behavior/build_qvals_table.py --write` against the staging root (or check that the qvals table is
     unchanged and copy it), then `prepare_replay_content_data.py --from-cache --output-dir <staging>`, then
     `tables/build_timebin_tables.py` (the event-onset table only, ASK-4), then the ContFrag tables
     (`prepare_contfrag_replay_content_data.py --event-source SWR` and `--event-source MUA`).
  3. `scripts/checks/compare_to_golden_baseline.py --candidate <staging> --baseline <frozen>
     --expected-added-columns <every new column> --expected-drift-columns well_visit_id <the current-arm
     columns, if the current-arm rule changed rows>`: every other column and every row identical; the
     drifted columns change only on `spans_visit_end` rows (check by a diff restricted to those rows).
  4. Message the other session ([C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session)) that
     the tables will change; swap the staging tables into `Processed-Data/replay_content/`; write the
     regeneration manifest (`scripts/tables/write_regeneration_manifest.py`).
- **Stored analysis: model-free curves on the three clocks**
  (`scripts/replay_models/rate/describe_visit_clocks.py`, new; method from `S/explore_leave/part_b4.py`):
  per kind and animal, event-start rate per at-risk second in bins of `s`, `u` and `m`, and the counts of
  stop-end kinds per animal; writes `Processed-Data/replay_content/models_convention/descriptive/visit_clock_curves.parquet`. Phase 5a compares its model curves to these.
- **Docs**: `docs/covariate_dictionary.md` (its "F.1 Two clocks" section, lines 382-423, rewritten for the visit clocks) gains a section on the
  visit unit, its clocks and the stop-end kind; `docs/data-lineage.md` lists the new columns.

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-3](legacy-inventory.md#phase-3)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- Any rate or content fit (Phases 5a, 6a).
- Changing what `visit_context` means or its thresholds (owned by the labels work).
- Figures 2 and 3 (Phase 4).
- Removing `aggregate_to_timebins` and the stop-grain clock columns: they leave with their last reader
  (inventory, "Cohort selectors and stop clocks").

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/event_tables/test_visit_columns.py::test_stop_end_kind_on_synthetic_visits` | three visits, one with three stops (two small movements, one leaves), one with one stop that crosses the last poke-out (`leaves`, `spans_visit_end`, `well_visit_id` set), one with a stop that starts before the first poke-in (`spans_visit_start`, `overlapping_visit_id` set, `well_visit_id` missing), one stop between visits (`outside_visit`) |
| `test_visit_columns.py::test_well_visit_id_for_rows_starting_inside` | events and stops labeled `in_visit` or `spans_visit_end` carry `well_visit_id`; others do not |
| `test_visit_columns.py::test_spans_visit_end_current_arm_is_the_visit_well` | an event crossing the final poke-out has the visit's well's arm as its current arm |
| `test_visit_columns.py::test_uncertainty_definitions` | on a hand outcome sequence, the variance equals `p(1 - p)` over the last 10 visits and the count equals `log1p` of the earlier visits |
| `tests/analysis/contfrag/` (updated) | the ContFrag tables carry the same new columns |
| `test_visit_columns.py::test_in_visit_immobile_seconds` | a stop crossing the poke-out counts only the seconds before it; excluded time is subtracted |
| `test_visit_columns.py::test_event_visit_clocks` | `time_since_visit_start_s`, `time_until_visit_end_s`, `time_until_stretch_end_s` on hand events; missing outside a visit |
| `test_visit_columns.py::test_new_uncertainty_covariates_use_known_outcomes_only` | an outcome known after the event start does not enter either covariate; trunk arms follow the stated rule |
| `tests/analysis/replay_models/test_visit_design.py::test_clocks_and_labels_per_bin` | `s + u` constant within a visit; `m = u` in the last stretch; bins outside visits labeled by their stop and with missing clocks; `select_visit_context` keeps exactly the in-visit bins |
| `test_visit_design.py::test_at_risk_excludes_ongoing_events` | a bin half covered by an ongoing event of the kind has half its exposure; an event of the other kind changes nothing |
| `test_visit_design.py::test_outcome_shift` | lag 1 switches to the current trial's outcome at `outcome_known_time_s`, not before |
| `test_visit_design.py::test_exposure_sums_to_in_visit_immobile_seconds` | over a session, in-visit exposure plus in-event time equals the stops' `in_visit_immobile_s` |
| `test_visit_design.py::test_same_events_as_content` | every event of the default set (`STARTS_IN_WELL_VISIT`) starts in an at-risk bin and is counted once; no other event is counted (rate and content analyze the same events) |
| `test_visit_design.py::test_one_real_epoch` (slow, real data) | on one epoch, bin counts equal the starts of the default-set events of each kind in the rebuilt event table; exposure equals the stop table's `in_visit_immobile_s` sum |
| `tests/repository/test_models_table_mirrors_specs.py`, `test_canonical_tables_read_with_stamp.py` | still pass with the new columns |
| table parity (script, not a test) | `compare_to_golden_baseline.py` with `--expected-added-columns` passes on the full rebuild |

## Fixtures

Synthetic port experiences, stops, events and outcomes in `tests/analysis/event_tables/conftest.py` (or the
test module), covering: a visit with no stop, a stop with zero exposure, a stop wholly in excluded time, an
event that crosses the last poke-out, the first visit of an epoch with no earlier outcome. One real epoch
through the `processed_data_root` fixture for the slow test.

## Verification and runtime

- Before: the counts per label (C3 snippet) and the frozen tables.
- Smoke test: build one epoch's tables into a scratch root with `--from-cache --test` (two sessions per
  animal) and time it; extrapolate to 242 epochs before the full build (the `--from-cache` path is described
  as minutes; confirm).
- After: parity as above; the counts per label unchanged (labels are not touched); the new columns' missing
  shares per label printed in the PR (for example, `stop_end_kind` per animal against the design's section
  2.7 exploration, as a sanity check, not as a reported number).
- The visit bin design for one session: time and memory; extrapolate to 242 (the rate fits build it per
  session on the fly, so it must be well under a second per session).

## Rebuilt

The canonical table set in `Processed-Data/replay_content/` (event, event_arm, event_patch, event_sequence,
stop, and the time-bin tables if touched), one atomic set. Every stored output that recorded the old table
files now reads as stale; its values in `paper/reported/` become pending at the next rebuild of each
summary. Do not rebuild summaries for that alone. The descriptive curves file.

## Exit criteria

Parity passes with only the declared added columns; the slow real-epoch test passes; the other session has
been told before the swap; the PR lists the new columns and their meaning.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
