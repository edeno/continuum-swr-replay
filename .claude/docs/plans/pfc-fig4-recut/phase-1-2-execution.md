# Recut phases 1–2, execution plan against the 2026-09-14 tree

Spec authority: `phase-1-canonical-events.md`, `phase-2-matched-control.md`, and
`shared-contracts.md` §"Matched-control interface" (frozen signature and four invariants — do not
weaken) and §"Provenance sidecar schema" (the keys named below). Those files were written on
2026-08-07; every line anchor in them is stale. The PFC path has since gained (branch
`pfc-atwell-cohort`, merged 2026-09-14): `build_session_event_sets` / `build_session_fit_mask` in
`analysis/pfc/glm.py` (one event construction shared by `run_session` and
`scripts/figures/generate_figure04.py`), the well-anchored stop cohort (`analysis/pfc/cohort.py`,
`--stop-cohort`), the decode→position alignment step, and a provenance sidecar per session written
by `scripts/pfc/run_pfc_analysis_bandit.py` via `fit_provenance.write_fit_provenance`. The
immobility inequality is already `<` on both paths. Build on those; do not re-create them.

**What the user wants (binding):** "it is very important for this paper that we are talking about
the same events everywhere in all analyses." Measured on the 71 refit sessions: the PFC GLM's
detected non-local events equal the canonical event table's count in every session, but its
SWR/non-SWR split (overlap with `data["ripple_times"]`) labels a median 44.7 % of events SWR-coupled
where the canonical `is_swr` (strict positive-duration overlap with the speed-gated `SWR_event`
labels) gives 43.0 %.

## Decisions (binding)

1. **Non-local events come from the shipped canonical event table**, not from a run-time
   recomputation: `Processed-Data/replay_content/event_table.parquet`, restricted to the session by
   `(animal, session_date, epoch)`, columns `event_id, event_start_s, event_end_s, is_swr,
   occupancy_state` (renamed to `start_time`/`end_time` for the GLM). Its sha256 goes in the
   sidecar. This is what Stage 5B/6 and the content/rate families read, so Figure 4 becomes
   traceable to the same `event_id`s. (`phase-1` proposed recomputing via
   `build_classified_nonlocal_event_table`; the shipped table is that function's output on the
   canonical cached inputs and is the paper's single source of truth.)
2. **SWR intervals come from the same source the rate ladders' SWR population uses**: the cached
   `Processed-Data/{epoch_identifier_HPC}_80_data_over_time.pkl` `SWR_event` labels, speed-gated
   and turned into intervals by exactly the helpers `analysis/hippocampus/event_table.py` uses
   (`intervals_from_labels` on the speed-gated dot — read `build_classified_nonlocal_event_table`
   ~222–340 and call the same functions). The dot's sha256 goes in the sidecar. `data["ripple_times"]`
   leaves the GLM path entirely and survives only in the diagnostic script.
3. **The old construction survives only in `scripts/qa/diagnose_pfc_event_definition.py`**
   (`get_confident_times` → `build_speed_eligible_event_cohort` → `get_overlap_times` on
   `ripple_times`), which reports old-vs-new per session.
4. **Cohort restriction unchanged**: `events_in_stops` (onset inside a well-anchored stop) applies to
   the canonical events and SWR intervals before the split, as today. Additionally assert that every
   canonical event lies inside the aligned position frame (the 27 trimmed sessions were checked:
   no event in any dropped tail) and count any that do not.
5. **Matched control is drawn inside the fit mask** (`is_immobile & in_cohort_stop`, the
   `build_session_fit_mask` output), not raw immobility: the control must sample the same bins the
   models are fit on. The frozen signature `make_matched_control_times(event_times, is_immobile,
   rng)` is kept; the caller passes the fit mask as `is_immobile`. Seed =
   `int.from_bytes(hashlib.sha256(f"{nwb_file_name}|{epoch_name}".encode()).digest()[:8], "little")`,
   recorded in the sidecar.
6. **The engagement test is nested.** `phase-2`'s literal pair `("non_local_no_SWR",
   "matched_control")` compares two non-nested single-regressor models, which the LR test does not
   license (the same reason the `SWR + non_local_no_SWR vs. non_local` pair was removed on
   2026-05-06). Implement instead: design `matched_control` (parallel to `non_local_no_SWR`),
   design `matched_control + non_local_no_SWR`, and the pairs
   `("matched_control + non_local_no_SWR", "matched_control")` — the engagement test: non-SWR
   replay adds beyond a duration-matched non-event baseline — and `("matched_control", "constant")`
   — does the baseline itself modulate. Name them exactly so; extract the `matched_control`
   modulation array.
7. **No batch rerun in these phases** (Phase 3). Validate on three real sessions to a scratch dir.
   `results_exist` gains `is_swr_definition` and the event-table/dot hashes, so the 71 canonical
   at-well bundles (fit on the old split) will be refit by the next batch — say so in the docs.

## Global Constraints

- `uv run …`; `ruff check`/`ruff format --diff` clean on added lines; no plan/phase/task references
  in shipped code/tests/docs/commit messages; nothing under `Processed-Data/`, `logs/`,
  `.superpowers/`, `paper/figures/**/*.png` committed; commit trailer `Co-Authored-By: Claude Fable 5.1
  <noreply@anthropic.com>` then `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`;
  no `git checkout`/`stash`/`clean`; stage by name; never stage `anna_yamls*`, `.vscode/`,
  `.claude/docs/plans/finish-covariate-rename/`.
- `analysis/pfc/glm.py` must stay importable without Spyglass (keep heavy imports deferred as now);
  import-layering test green.
- No fits against canonical output names; smokes go to a scratch `--output-dir`.
- Tests value-level; `slow` marks over 20 s; real-data tests `slow`.

### Task 1: Canonical events and SWR intervals on the PFC path; diagnostic; sidecar

Files: new `src/continuum_swr_replay/analysis/pfc/canonical_events.py` —
`load_session_canonical_events(nwb_file_name, epoch_name, *, event_table_path=None) ->
tuple[pd.DataFrame, str]` (session rows of the shipped event table with `event_id, start_time,
end_time, is_swr, occupancy_state`, sorted by `start_time`; the table's sha256; raise if the session
has no rows) and `load_session_swr_intervals(nwb_file_name, epoch_name, *, dot_path=None) ->
tuple[pd.DataFrame, str]` (from the `_HPC_80_data_over_time.pkl` cache through the same
speed-gating + `intervals_from_labels` helpers `event_table.py` uses; columns `start_time,
end_time`; the dot's sha256; raise if missing). Derive the session key the way `pfc/cohort.py`
does. `analysis/pfc/glm.py` `build_session_event_sets` (~1133–1300): replace the
`get_confident_times`/`build_speed_eligible_event_cohort`/`get_overlap_times`/`data["ripple_times"]`
block with the two loaders (accept the frames as arguments — `canonical_events`, `swr_intervals` —
so the function stays offline-testable; `run_session` and `generate_figure04.py` load them);
`event_times` keys unchanged (`non_local_no_SWR`, `non_local_SWR`, `non_local`, `SWR`); the split
is `is_swr`; counts gain `n_events_outside_frame` (canonical events not inside the aligned position
span — assert 0 or log loudly and drop, per Decision 4) and keep the existing before/after and
coverage counts. Sidecar (`scripts/pfc/run_pfc_analysis_bandit.py` `extra`): add
`is_swr_definition="canonical_event_table"`, `immobility_inequality="<"`,
`canonical_event_table_path/_sha256`, `swr_interval_source="data_over_time.SWR_event"`,
`data_over_time_path/_sha256`, `n_events_total/n_events_swr/n_events_non_swr` (after the cohort
restriction), `n_matched_control=0`; `results_exist` compares `is_swr_definition` and both hashes
too. New `scripts/qa/diagnose_pfc_event_definition.py` (`--animal --date --epoch`, `--out-dir`
default `.claude/research/pfc_event_diagnostic_outputs/`): builds events both ways (old logic
inlined here, verbatim from the deleted block; new via the loaders), reports total count old vs new,
n reclassified SWR↔non-SWR, event-start shift distribution, boundary/partial-overlap cases; prints a
table and writes a CSV. `run_session`/`build_session_event_sets` docstrings state the canonical
definition. Tests: `tests/analysis/pfc/test_canonical_events.py` (loaders on a synthetic parquet +
dot pickle: session restriction, sha, raise on missing; SWR intervals equal `event_table.py`'s on the
same synthetic dot), `test_glm_pipeline.py` (`build_session_event_sets` on a fixture where the old
ripple split and canonical `is_swr` disagree on one boundary event → the canonical column wins;
`event_times` keys unchanged; an event outside the frame is counted), a grep test that
`data["ripple_times"]`/`get_overlap_times`/`get_confident_times` no longer appear in
`analysis/pfc/` or `scripts/figures/generate_figure04.py`, `test_run_pfc_analysis_bandit.py`
(sidecar keys; `results_exist` refuses a sidecar with the old definition), and one `slow` real-data
test (a session with existing decode results: `run_session`'s SWR/non-SWR counts equal the shipped
event table's at-well counts after the cohort restriction). Smoke: three sessions
(`j1620210710_02_r1`, `chimi20200223_06_r3` — a trimmed one — and one wilbur) to a scratch
`--output-dir` with `/usr/bin/time -v`; run the diagnostic on the same three and report the
reclassification counts.

### Task 2: Matched non-event control

Files: `analysis/pfc/glm.py` — `make_matched_control_times(event_times, is_immobile, rng,
max_attempts=50)` with the frozen contract (pure helpers `_contiguous_true_runs`,
`_bout_containing`, `_within_immobile`, `_overlaps_any` next to the other pure helpers; runs are
contiguous True runs of the mask passed in; a control window must lie within the same run as its
source event's onset, entirely in True bins, overlap no real event (all four `event_times` classes),
preserve duration; unplaceable events dropped and counted); `run_session` calls it with the fit
mask and the Decision-5 seed after the cohort restriction and adds `"matched_control"` to
`event_times`; `_make_design_matrices` gains `matched_control` and `matched_control +
non_local_no_SWR`; `process_neuron` gains the two Decision-6 pairs and the `matched_control`
modulation extraction; `run_session` returns the extra modulation array; the runner saves
`{epoch_identifier}_modulation.matched_control_modulation.npy`, sets `n_matched_control`,
`n_matched_control_unplaced`, `matched_control_seed` in the sidecar, and `results_exist` checks the
new file; `generate_figure04.py` panel a threads the control through (no new panel). Docstrings.
Tests (`tests/analysis/pfc/test_matched_control.py`): the five invariant tests from `phase-2`'s
validation slice (within-mask, no overlap with any real event class, count ≤ and durations equal,
deterministic by seed / different by seed, unplaceable dropped), plus `process_neuron` on the
synthetic fixture with a neuron that tracks non-SWR events: the engagement pair's p-value is small
and the `matched_control` modulation array is finite; and a `slow` real-data save test. Smoke: the
same three sessions to scratch; report `n_matched_control` / unplaced per session and the engagement
p-value distribution across neurons for one session.

### Task 3: Docs

`docs/design/cohort_and_timing_rules.md` (PFC section: events and SWR intervals are the canonical
ones; the matched control), `docs/README.md` ("Where results live": the sidecar now records the
event definition; the 71 canonical bundles predate it and are refit by the next batch),
`scripts/README.md` (the diagnostic), `paper/claims_table.md` row 20 (a dated note that the PFC
GLM now reads the canonical `event_id`/`is_swr` and carries a matched control, numbers pending the
rerun; keep 5 cells and RETRACTED annotations), `docs/regression_pipeline_map.md` regenerated;
`PLAN.md` status of the recut plan updated (phases 1–2 landed, 0/2b/3/4 open). Retired-name,
mirror, map tests green; full `uv run pytest tests/unit -q`.

## Review

Final review (Fable): the old split is gone from every fit path (grep test); events and SWR
intervals are the canonical ones (real-data test); the matched control satisfies the four frozen
invariants with failing-if-violated tests and is drawn inside the fit mask; the engagement test is
nested; sidecars and `results_exist` carry the definition; no batch rerun happened; docs say the
71 canonical bundles predate the definition; no plan references; suite green.
