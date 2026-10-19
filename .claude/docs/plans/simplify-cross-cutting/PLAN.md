# Cross-cutting simplification and the two behaviour changes

Status: complete 2026-10-19 — tasks 1–9 + final review fix wave on branch `simplify-repo-2026-10-19` (e70ab12..2b38381); not merged to master (awaiting the lead). Ledger: ledger.md; final review: final-review.md; code-organization review: ../../reviews/2026-10-19-code-organization-review.md.

Source: the 2026-10-19 `/simplify` whole-repo review (reuse, simplification,
efficiency, altitude agents). The local, behaviour-preserving fixes landed in
e70ab12. This plan carries the cross-cutting refactors and the two behaviour
changes the lead approved on 2026-10-19 ("commit and do the cross cutting
refactors and changes to how things behave"). No spec document; the review
findings quoted in each task are the requirements, and rulings are recorded
in the SDD ledger.

## Global Constraints

1. **Outputs unchanged** except where a task is marked *Behaviour change*.
   Scientific tables, frames and figures produced by the refactored code must
   be identical to before (bit-identical; float summation-order differences
   are NOT acceptable unless the task says so).
2. **Golden reference for table-producing code.** A before-change table set
   for one session per animal (5 sessions) built at e70ab12 lives at
   `/tmp/claude-39494/-cumulus-edeno-continuum-swr-replay/abb6ddbf-eeb3-43b5-9505-88838c9b5af2/scratchpad/golden_e70ab12/rc`
   (build log: `golden_e70ab12/prepare.log`). A task that touches the table builders, `swr_source`, event labels or
   event classification must rebuild the same 5 sessions into a fresh scratch
   directory and compare with
   `continuum_swr_replay.table_parity.compare_table_roots` (no allowlist).
   Rebuild command (run from the repo root):
   `uv run python scripts/decoding/prepare_replay_content_data.py --from-cache --max-per-animal 1 --output-dir <scratch>/rc --contingency-dir Processed-Data/replay_content --behavior-cache-dir Processed-Data/replay_content/_behavior_cache`
   Never write into `Processed-Data/` itself.
3. **Simulated checks first.** For each moved or merged function, compare old
   vs new on simulated inputs including edge cases (empty, single row, NaN,
   ties) before relying on the real-data check.
4. **Tests.** Run the targeted tests for every module touched
   (`uv run pytest -q -p no:cacheprovider <files>`; `-n` is not installed).
   The full suite (~30 min) is run by the controller at the end. Add or
   update tests where a task changes a contract.
5. **Scope.** Touch only what the task needs. Do not touch untracked files
   (`scripts/figures/generate_replay_story_figures.py`,
   `tests/integration/test_replay_story_figures.py`, `paper/figures/replay_story/`,
   `.claude/docs/plans/{finish-covariate-rename,representation-selection}/`,
   `docs/reference_*_2026-09-17.md`, `.vscode/`). Never `git stash`,
   `git reset --hard`, `git checkout -- <file>` or `git clean`.
6. **Style.** Match surrounding code: NumPy docstrings with shapes, type
   hints, `ruff check` and `ruff format` clean on touched files.
7. **Docs.** Update `CLAUDE.md`-referenced docs (`docs/models_table.md`,
   `docs/regression_pipeline_map.md` via `scripts/qa/write_pipeline_map.py`,
   design docs that name a moved function) when a task moves a public name.
8. **Commits.** One or more commits per task on the current branch, ending
   with the session attribution lines given in the dispatch.

## Task 1: One `replace_outputs` with rollback, one run-id check (*Behaviour change*)

`replace_outputs` exists in four runners:
`scripts/decoding/run_carey_detection.py` (~l.406),
`scripts/decoding/run_swr_detection.py` (~l.914),
`scripts/qa/run_inventory_comparison.py` (~l.596),
`scripts/qa/run_event_features.py` (~l.320). The last two remove every table
when a write fails (`except BaseException`); the first two leave a partial
set. The `--run-id` validation
`RUN_ID_PATTERN.fullmatch(...) or value in {".", ".."}` is repeated in
`run_artifact_detection.py` (~l.443), `run_carey_detection.py` (~l.583),
`run_swr_detection.py` (~l.1278), `run_inventory_comparison.py` (~l.835),
`run_event_features.py` (~l.898).

- Put one `replace_outputs(paths, outputs, ...)` with rollback-on-failure
  always on in `src/continuum_swr_replay/table_io.py` (or `decoding_common.py`
  if its dependencies live there — say which and why), keeping every caller's
  current success-path bytes identical. Behaviour change: the carey and swr
  runners now also roll back on a failed write.
- Put one run-id validator (an argparse `type=` callable plus the pattern) in
  `decoding_common.py` and use it at all five sites; error text may be unified.
- Tests: a failing-write test proving rollback for the shared helper; the
  existing runner tests still pass.

## Task 2: `aggregate_to_timebins` derives excluded time itself (*Behaviour change*)

`analysis/replay/per_bin_design.py` `aggregate_to_timebins(..., excluded_intervals=None)`
treats `None` as "no mask", so every caller computes
`excluded_intervals_by_session(stops_g)` and passes it in:
`scripts/rate/fit_per_bin_rate_models.py` (~l.182,193),
`scripts/rate/fit_swr_contrast_rate.py` (~l.415,426),
`scripts/rate/fit_uncertainty_rate.py` (~l.436,448),
`scripts/rate/fit_cross_history_rate.py` (~l.166, via `common`),
`scripts/rate/plot_model_checking.py` (~l.257),
`scripts/qa/compare_fits.py` (~l.772); the same comment is copied into 4
scripts. A caller that forgets the argument silently analyses excluded time.

- Make the default derive the intervals from the stop table it is given
  (the same `excluded_intervals_by_session` call the scripts make today), and
  keep an explicit way to say "no mask" for tests and synthetic data (e.g. an
  empty mapping or a named sentinel — choose, document, and make `None` no
  longer mean "no mask" silently). Check what `excluded_intervals_by_session`
  does for sessions with no artifact run configured; the CLAUDE.md rule says
  "With no artifact run configured, the rule changes nothing".
- Remove the per-caller computation and the copied comment; current callers'
  outputs must be identical (verify one script path on real data, e.g. build
  the timebin design for 1–2 sessions before/after).
- Update tests that relied on `None` meaning no mask.

## Task 3: Model rungs defined in scripts move into `model_specs.py`

`analysis/replay/model_specs.py` is the stated single source of truth for
every model ladder (`docs/models_table.md` mirrors it;
`tests/unit/test_models_table_mirrors_specs.py` enforces the rung sets). Two
scripts define rungs locally: `scripts/rate/fit_uncertainty_rate.py`
(~l.196 `MU_UNCERTAINTY`, parent `M7r_time_histcos_rew`) and
`scripts/rate/fit_cross_history_rate.py` (~l.79-101 `_spec` → `Mself` /
`Mselfcross`).

- Move both into `model_specs.py` as named ladders/specs with identical
  formulas, names, parents and any other fields; import them in the scripts.
- Add them to `docs/models_table.md` and register them in the mirror test.
- Prove the formula strings and spec objects are identical before/after.

## Task 4: The two decode runners share their HPC branches

`scripts/decoding/run_decoding_bandit.py` and
`scripts/decoding/run_decoding_contfrag.py` are ~80% the same (perform
decoding, save results/model, organize results, run decode). Each repeats the
same HPC-only branches (~bandit l.195,311,421,556; contfrag l.193,299,377,499):
waveform features, `with_swr_source(drop_masked_events=False)`,
`hpc_artifact_intervals`, `require_hpc_decode_sources`. They have drifted:
bandit persists `return_outputs="log_likelihood"` and contfrag does not.

- Factor the shared HPC behaviour (and as much of the shared runner flow as
  stays simple) into `src/continuum_swr_replay/decoding_common.py` (or a
  `scripts/decoding/_decode_runner.py` if it is script-only), with a small
  per-runner record for what differs (detector factory, config class,
  data-over-time function, loader, output suffix, `non_local_threshold`,
  figurl, and the log-likelihood output).
- Preserve each runner's current behaviour exactly, including the
  log-likelihood difference (record it explicitly in the per-runner record;
  do not unify it — the lead decides that).
- `DecodingConfig.to_sorted_spikes_params` is called via
  `getattr(config, f"to_{param_type}_params")()`; if you touch that dispatch,
  make it greppable.
- Verification: a GPU decode is too slow for a before/after run. Instead
  prove the calls each runner makes are unchanged — e.g. patch the detector
  class/IO with recorders and compare the recorded arguments (estimate
  params, predict kwargs, save paths) before/after for both runners and each
  brain area, on a tiny synthetic `data` dict.

## Task 5: One implementation of strict SWR overlap

Two implementations of "strict positive-duration overlap":
(A) `analysis/hippocampus/event_detection.py` (~l.142-224)
`event_strict_overlap_keys` / `event_strict_overlap_mask`, with a private
label→bounds `_bounds` duplicating `event_immobility.intervals_from_labels`,
used by `figure_panels.py`, `scripts/qa/analyze_event_spike_evidence.py`,
`scripts/content/analyze_content_occupancy.py`,
`analysis/replay/tables_event.py`, `analysis/contfrag/tables.py`,
`quality_metrics.py`, `data_aggregation.py`;
(B) the canonical builder `analysis/hippocampus/event_table.py`
(~l.328-337) uses `classify_swr_overlap_geometry` on `intervals_from_labels`,
renaming `event_start`/`event_end` to `start_time`/`end_time` for the call.

- Make A use `intervals_from_labels` and the same geometry predicate B uses
  (or B use A), so one function defines strict overlap; zero-duration handling
  (`positive_duration_interval_mask`) must apply identically to both.
- Parity first: before changing anything, write a parity check of A vs B on
  the 5 golden sessions' decode frames and on simulated edge cases (touching
  intervals, zero-duration events, nested, multiple SWRs). If they differ
  anywhere, STOP the merge for that case and report the difference with
  examples (do not silently pick one).
- Golden comparison (Global Constraint 2) must be identical.

## Task 6: The decode frame's time axis is read one way

Production decode frames are time-indexed with no `time` column (comments at
`analysis/hippocampus/event_table.py` ~l.255-261 and
`analysis/pfc/canonical_events.py` ~l.256-258), yet ~12 sites check
"`time` column or index": `event_table.py:~262`,
`event_immobility.py:~286` (`intervals_from_labels` prefers a `time` column),
`swr_source.py:~834`, `memoryless_sequence/event_table.py:~435`,
`pfc/canonical_events.py:~259`, `inventory_comparison/features.py:~176`,
`inventory_comparison/adapters.py:~154`,
`scripts/decoding/run_carey_detection.py:~223`,
`scripts/decoding/run_memoryless_detection.py:~570`,
`scripts/qa/run_inventory_comparison.py:~335`,
`scripts/qa/analyze_event_spike_evidence.py:~230`,
`scripts/figures/generate_inventory_examples.py:~399`.

- Make `intervals_from_labels` read only the index, and make the decode-frame
  reader (`swr_source.with_swr_source` / its frame helper) normalise to, or
  assert, a time index. Delete the caller branches.
- Check every caller's real input (and tests/fixtures) really is
  time-indexed; a fixture that passes a `time` column gets updated, a real
  caller that does is converted at its read point.
- Golden comparison must be identical.

## Task 7: Speed gating and the label contract are applied once, at the read

CLAUDE.md: the analysed-time rule "is applied once, where a decode frame is
read (`swr_source.with_swr_source`), so every table inherits it". In practice
`validate_nonlocal_event_probability_contract` + `filter_event_labels_by_speed`
are re-run downstream at ~12 sites:
`visualization/figure_panels.py:~415-421`,
`analysis/replay/_tables_common.py:~110-122` (`_filter_external_event_labels`),
`analysis/contfrag/tables.py:~167-177` (`_filter_contfrag_event_labels`,
whose docstring says the re-filter "is a no-op"),
`analysis/replay/timebin_tables.py:~110-118, ~192-200`,
`analysis/hippocampus/data_aggregation.py:~833-842, ~1021-1028`,
`analysis/hippocampus/quality_metrics.py:~860-869`,
`analysis/pfc/canonical_events.py:~262`,
`scripts/qa/analyze_event_spike_evidence.py:~188-192, ~362`,
`scripts/content/analyze_content_occupancy.py:~164, ~185-190`,
`scripts/figures/generate_event_gallery.py:~248`. `swr_source._drops` already
speed-gates `SWR_event`; non-local and MUA labels "were speed-qualified when
the decode was made"; `_tables_common._in_event_mask` raises if a non-local
label breaks the speed rule. The derived non-local probability is also
computed under two names (`nonlocal_prob`, `Non-Local_prob`) at ~10 sites.

- First establish facts: for each site, is the re-filter a no-op on
  production frames at the canonical threshold (check on the 5 golden
  sessions)? Which callers pass a non-default speed threshold, and why (a
  sensitivity analysis must keep its own gating)?
- Make `with_swr_source` validate the whole-event immobility / probability
  contract for every label column it hands out (raise on violation, using the
  existing drop-reason machinery if a drop rather than a raise is already the
  rule there), then delete the downstream re-filters that are no-ops at the
  canonical threshold; keep (and comment) any that implement a deliberate
  non-default threshold.
- Optional, only if it stays simple: emit the derived non-local probability
  once, under one name, at the read; otherwise leave the column sites alone
  and say so.
- Update CLAUDE.md's description only if it becomes inaccurate (it should
  become accurate). Golden comparison must be identical.

## Task 8: One epoch-identifier parser

Canonical: `data_loaders/utils.py:parse_nwb_filename`. Re-implementations:
`[:-8]`/`[-8:]` slicing in `analysis/replay/prepare_data.py:~147`,
`scripts/decoding/prepare_contfrag_replay_content_data.py:~135`,
`scripts/decoding/refresh_contfrag_behavioral_covariates.py:~233,250`,
`scripts/qa/run_inventory_comparison.py:~781` (`_epoch_parts`),
`scripts/qa/report_event_features.py:~554` (`_session_of`),
`scripts/decoding/run_memoryless_detection.py:~404`;
`animal_from_epoch_id` in `scripts/qa/analyze_event_spike_evidence.py:~124`
and `scripts/content/analyze_content_occupancy.py:~106`;
`parse_epoch_identifier` in `analysis/hippocampus/state_evidence/terms.py:~24`,
`scripts/content/analyze_content_occupancy.py:~100`,
`scripts/decoding/run_memoryless_detection.py:~242` (the variant-aware
`EpochKey` one); `parse_epoch_id` in `scripts/figures/generate_event_gallery.py:~159`;
`_extract_animal` in `analysis/pfc/load_population.py:~156`. The j16 bug
(digit-stripping animal parsers) has happened before.

- Promote one `EpochKey`/`parse_epoch_identifier` (from
  `run_memoryless_detection`) into `data_loaders/utils.py` next to
  `parse_nwb_filename` (or `session_keys.py` — choose by dependency
  direction) and route every site through it. Keep each site's output type
  (tuple, str, int date) the same at its call boundary.
- Test it on every animal's id form, including j16, `_contfrag` / variant
  suffixes and the PFC areas.
- Leave the untracked `generate_replay_story_figures.py` `.replace({"j": "j16"})`
  alone (lead's work in progress); mention it in the report.

## Task 9: One decode-file path helper; the gallery reads the register (*Behaviour change*)

`"_80_data_over_time.pkl"` is hard-coded ~12 times:
`analysis/replay/prepare_data.py:~155`, `analysis/hippocampus/data_aggregation.py:~770`,
`scripts/qa/run_state_evidence.py:~326,385`, `scripts/qa/diagnose_trial_detection.py:~52`,
`scripts/content/quantify_swr_overlap_geometry.py:~123`,
`scripts/figures/generate_figure03.py:~261`,
`scripts/figures/generate_supp_pipeline_data_flow.py:~101`,
`scripts/figures/generate_event_gallery.py:~156,168,1674`; plus three
per-script path helpers with three ways of computing the threshold suffix:
`run_memoryless_detection.py:~284` (`input_paths`),
`run_inventory_comparison.py:~305` (`decode_path`),
`run_state_evidence.py:~323` (`decode_files`).
`generate_event_gallery.py:~167` globs `*_HPC_80_data_over_time.pkl` instead
of `analysed_decode_files`, so pending/excluded epochs can appear.

- One `decode_paths(epoch_id, root, ...)` (data-over-time, results, detector,
  environment; threshold suffix computed one way from
  `DEFAULT_NON_LOCAL_THRESHOLD`) in `data_loaders/epoch_register.py` next to
  `analysed_decode_files` (or `decoding_common.py` — choose by dependency
  direction, say why). Route every site through it; paths must be identical.
- Behaviour change: the event gallery enumerates epochs through
  `analysed_decode_files` (analysed epochs only). Report how many epochs the
  old glob found vs the register.
