# Phase 4 — Figures 2 and 3 and the events summary on the common subset

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#c4-the-selection-function-is-the-only-filter)

Branch: `figures-2-3-common-subset` from a master that holds the visit labels (C3) **and** the other
session's Figure 2 branch (`swr-restricted-models` / `swr-trial-folds`). Independent of Phases 1 to 3,
except the fold parity task, which needs Phase 2.

## Start condition, and what to do if it is not met

```bash
git fetch origin
git merge-base --is-ancestor origin/swr-trial-folds master && echo merged
git merge-base --is-ancestor origin/swr-restricted-models master && echo merged
```

If either is not on master, do not start this phase and do not edit any file that branch changes
(`scripts/figures/generate_figure02.py`, `src/continuum_swr_replay/reported/events.py`,
`src/continuum_swr_replay/paths.py`, `src/continuum_swr_replay/analysis/event_evidence/*`). Message that
session (SendMessage if reachable; otherwise ask the lead to relay): the labels and the selection function
are on master and it was to adopt `select_visit_context`. Meanwhile continue with Phases 1, 2, 3, 5a, 6a,
8. Do not move Figure 3 alone: its values come from `reported/events.py`, which the other branch changes,
and a figure on the subset beside a summary not on it would disagree. If the other session has already
adopted the selection function in its branch, this phase shrinks to what it left out; read its diff first.

**Inputs to read first:**

- The design, section 4.1 ("label in the tables, filter at analysis"), section 7 stage 0b, and the
  section 9 rounds five to eight.
- `scripts/figures/generate_figure02.py:1299-1455` (`generate_figure`; the event table read at `:1326`;
  `check_table_against_frames` `:491`, which compares the table's class counts with the decode frames, so
  it must see the unfiltered table), `:438` (`_compute_figure_inputs`), `:564` (`detector_comparison`).
- `scripts/figures/generate_figure03.py:175-210` (`read_inputs`; the event table read at `:178`).
- `src/continuum_swr_replay/reported/events.py:200-244` (`EventsInputs`, `default_inputs`), `:2168`
  (`summarize`; the event table read at `:2172`).
- `src/continuum_swr_replay/analysis/event_properties/event_statistics.py:96`, `:127`
  (`overlap_class_animal_shares`, `overlap_class_statistics`).
- The producers of the stored inputs (table below) and what each writes.
- Scratch: `S/explore_outside_visits/analyze.py`, `analyze2.py`, `tables/q6_examples*.csv` (the two data
  points to check).

**Contracts referenced:** [C3](shared-contracts.md#c3-precondition-the-visit-labels-are-on-master),
[C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures),
[C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session).

**Decision needed first:** L5 (whether the SWR-grain and time-grain panels move to the subset).

## Tasks

- **Inventory every input of Figures 2 and 3 and of the events summary** and classify it. Start from this
  table (producers found by the directory constant in `paths.py`); confirm each by reading the producer:

  | Input (`EventsInputs` field) | Producer | Grain | How it moves to the subset |
  |---|---|---|---|
  | `event_table` | canonical table | event | `select_visit_context` after reading |
  | `event_participation` | `scripts/event_evidence/measure_event_participation.py` | event (`event_id`) | keep the rows of selected `event_id`s at read |
  | `tetrode_holdout`, `likelihood_advantage` | `scripts/event_evidence/score_tetrode_holdout.py`, `score_event_likelihood_advantage.py` (the other session's) | event windows | as the other session's code reads them; filter by `event_id` if the rows carry it, else re-run with the selected events |
  | `spike_evidence_dir`, `heldout_by_swr` | `scripts/event_evidence/...`, `split_state_evidence_by_swr.py` | event or bout | read; filter by `event_id` or re-run |
  | `event_features_run` | `analysis/detector_comparison/feature_run.py` | event and SWR | event rows by `event_id`; SWR rows per L5 |
  | `content_along_arms` | `scripts/event_properties/measure_content_along_arms.py` | event time bin | keep rows of selected events |
  | `event_gamma`, `event_start_rest_theta`, `class_structure` | `scripts/event_properties/measure_event_gamma.py`, `measure_event_start_and_rest_theta.py`, `measure_ripple_class_structure.py` | event | keep rows of selected events |
  | `property_bootstrap` | `scripts/event_properties/bootstrap_event_properties.py` | bootstrap over events | re-run on the selected events |
  | `brain_state_reference` | `scripts/event_properties/measure_brain_state_reference.py` | immobile time | per L5: re-run with an in-visit mask |
  | `nonlocal_by_ripple_power` | `scripts/event_properties/measure_nonlocal_by_ripple_power.py` | immobile time bins (Figure 2j) | per L5: re-run with an in-visit mask |
  | `decoder_quality` | decode quality scripts | decode | unchanged (not an event statistic) |
  | Figure 2's cached render inputs (`_figure02_cache_path`, `:380`) | `generate_figure02.py` | example session | recompute with `--force-recompute` |

- **Apply the selection at every event read**: `generate_figure02.py:1326` (after
  `check_table_against_frames`, which keeps the full table), `generate_figure03.py:178`,
  `reported/events.py:2172`, and in each re-run producer. One call, the default subset; no other filter.
- **In-visit mask for time-grain inputs** (if L5 says yes): a sample-level mask from the visit intervals of
  the port experiences (`visit_context.visit_context_of_samples` on master gives the label per sample) and
  the producers' immobility mask; the producers take it through one new argument and record counts.
- **Counts per label in the summary**: `reported/events.py` adds values from `count_visit_context` (events
  per label, kept and dropped, per kind) and states in each affected value's description that it is over
  events inside a well visit.
- **The fold function**: make the event-evidence `trial_folds` (`analysis/event_evidence/tetrode_holdout.py`
  on master after the merge) call `analysis/regression/folds.py`'s function, or move it there and import it,
  and add a parity test on a synthetic epoch. Needs Phase 2 on master; if Phase 2 is not yet merged, leave
  this task for Phase 5a and say so in the PR.
- **The two data points** (design section 9): epoch `chimi20200212_02_r1` with events whose head position is
  50 to 90 cm off the track while the projection is at the well, and the few events labeled at one well while
  the tracked position is far from it. Print the rows (event ids, positions, labels) from the rebuilt tables,
  plot the epoch's head positions with the track, and report to the lead. Do not change the labels here.
- **Re-run the producers** classified "re-run" above, smoke test on one epoch first (C11), then all
  epochs; GPU only if a producer needs it (the held-out scoring belongs to the other session: coordinate
  before re-running any of its runs).
- **Rebuild and redraw**: commit, `summarize.py --summary events`, `emit_reported_values.py`, then
  `generate_figure02.py --force-recompute` and `generate_figure03.py`; commit the figure files under
  `paper/figures/figure02/` and `figure03/`.
- **Docs**: `docs/figure-pipeline.md` (Figures 2 and 3 are over events inside a well visit) and the captions'
  source (if captions are generated) say the subset and the count per label.

## Deliberately not in this phase

- Changing the labels, their radii or the cohort (the labels work).
- Figure 2's held-out analysis itself (the other session).
- Describing the events outside the visits (sixth round: not now).
- Any regression model.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/reported/test_events_summary_subset.py::test_summary_reads_selected_events` | on a synthetic event table with every label, the summary's class counts equal those of the `in_visit` rows, and the per-label count values equal `count_visit_context` |
| `tests/scripts/figures/test_figure02_subset.py::test_frame_check_sees_full_table` | `check_table_against_frames` passes on the full table and the figure draws from the selected rows |
| `tests/analysis/event_evidence/test_trial_folds_shared.py::test_same_folds` | the event-evidence fold function and the shared one give identical folds on a synthetic epoch (if the fold task is done here) |
| producer smoke tests (real data, slow) | for each re-run producer, one epoch: rows only for selected events; counts per label written |
| `tests/repository/test_documented_values.py`, `test_markers.py` | still pass after the emit |

## Fixtures

A synthetic event table with every `visit_context` value (built in the test); the real tables for the
producer smoke tests.

## Verification and runtime

- Before: copy `paper/reported/events.json` and the current figure PNGs to the phase scratch directory.
- After: diff the summary key by key; every changed value's change has the expected direction (fewer
  events; the No SWR share lower, as the design's diagnostic of section 9 shows, without quoting it); list
  the changed keys in the PR with old and new values read from the two files.
- Spot-check: for three epochs (one with many outside events, one with few, `chimi20200212_02_r1`), the
  number of events dropped by the selection equals the event table's count of non-`in_visit` labels.
- Runtime: each producer on one epoch, then extrapolate; the property bootstrap and the in-visit re-runs of
  the time-grain producers are the long ones.

## Rebuilt

`Processed-Data` outputs of the re-run producers; `paper/reported/events.json`, `paper/reported_values.tex`,
`paper/reported/questions.md`; `paper/figures/figure02/`, `paper/figures/figure03/`.

## Exit criteria

Every input of Figures 2 and 3 and of the events summary reads through the selection (or is listed as not an
event statistic); the summary records counts per label; both figures redrawn; the two data points reported to
the lead.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
