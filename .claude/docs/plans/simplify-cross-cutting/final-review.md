# Final whole-branch review — simplify-cross-cutting

Range: 345c134 (master) .. 44996e7 (`simplify-repo-2026-10-19`), 13 commits, 128 files.
Reviewer: opus, read-only. Date 2026-10-19.

## How the review was split

I did not page the 930 KB package linearly. I read it per commit with `git show <sha> -- <paths>`
and `git diff <a> <b> -- <paths>`, in these passes:

1. PLAN.md (Global Constraints and tasks), the ledger, and the deferred minors.
2. The read point and T5×T6×T7: `swr_source._frame_epoch` / `with_swr_source` / `artifact_event_drops`,
   `event_immobility` (new `validate_whole_event_immobility_contract`, index-only `intervals_from_labels`),
   `event_detection.strict_overlap_matrix` and its five consumers, and every builder whose re-filter was
   deleted (`_tables_common`, `tables_event`, `tables_stop`, `timebin_tables`, `contfrag/tables`,
   `data_aggregation`, `quality_metrics`, `figure_panels`, `event_table`, the gallery). I also grepped for
   frames that reach those builders without going through the read.
3. T4×T9 at the save path: `_decode_runner.py` (whole file), both runner shells, the old contfrag runner
   at 8664f23 (identifier order, detector factory), `decoding_common.results_exist`, and `decode_files.py`.
4. T8×T9: the `parse_epoch_identifier` / `EpochKey` code, every migrated call site, `analysed_epoch_ids` /
   `analysed_decode_files`, and the gallery's enumeration.
5. T1–T3: `replace_outputs`, `parse_run_id`, the `aggregate_to_timebins` default, `NO_EXCLUDED_TIME`, and
   the `model_specs` moves.
6. e70ab12 (/simplify; it had no per-task review): every `src/` hunk with logic in it (`quality_metrics` run
   slices, `spatial_mapping._nearest_bin_index`, `per_bin_design` binj and sort skip, kernel-ratio helpers,
   `pfc/glm` and `heldout` → `merge_intervals`, `_load_saved_decode`, `safe_mode`, `_track_edge_colors`,
   `plot_spike_times`, `simulation`), then the scripts with logic in them (retrospective runner,
   ContFrag publish step, memoryless manifest). Formatting-only hunks were skimmed.
7. Docs and targeted tests. Two runs, no `-n`: 288 passed (layering, lazy imports, swr_source,
   event_detection, utils, decode_paths, replace_outputs, model specs and mirror), then 177 passed with
   2 warnings (pipeline map current, retired names, decoding_save_dir, decode_folder_readers,
   artifact_exposure, replay_tables, figure_panel_event_masks, data_aggregation_event_counts).

## Strengths

- **The analyzed-time rule now matches CLAUDE.md.** `_frame_epoch` (swr_source.py:828-869) is the one place
  that refuses a stale frame: a `time` column, a time index that does not increase, a moving or speed-NaN
  non-local or MUA label, and a non-local bin at or below the threshold. The decode runner passes its own
  threshold, and so do `run_quality_metrics` and `canonical_events`. SWRs keep their `moving` drop. The
  ledger records the evidence: 484/484 frames were no-ops, 38/38 site outputs were identical, and the
  golden 7 tables matched at atol 0.
- **Strict overlap has one definition.** `strict_overlap_matrix` (event_detection.py:141) builds in the
  zero-duration rule (`positive_duration_interval_mask`), so the classifiers, `_match_swr`,
  `get_overlap_times`, `event_strict_overlap_keys` and the gallery cannot drift apart again. The A≡B parity
  check ran before the merge, as the plan asked. Event-level `is_swr` is still strict. The stop-level
  `is_swr_stop` is untouched.
- **The decode runners' log-likelihood drift is recorded, not hidden.** `DecodeRunner.hpc_log_likelihood` and
  a parametrized test pin it. The identifier suffix order is unchanged from the old contfrag runner.
  `algorithm_params` replaces the `getattr` dispatch, so the method names can be grepped.
- **The leaf module `decode_files.py` is placed well.** The lazy-import test pins six readers free of
  `dask.distributed` and `dask_cuda`. The `int(t*100)` truncation is kept and documented.
- **`EpochKey` refuses unknown suffixes** instead of misreading them. j16 is covered. 3,025 ids were checked
  old against new.
- **e70ab12's vectorizations are correct.** I checked each against its old code. `_run_slices`/`find_objects`
  on a 1-D label array is equivalent. The searchsorted nearest bin handles ties, including the left-neighbor
  tie. `binj = arange - repeat(cumsum - nbin)` is equivalent. The sort is skipped only when the
  self-history block already sorted and a left merge on unique keys keeps that order. `merge_intervals` has
  the same algorithm as `_merge_windows`, including merging touching windows.
- The pipeline map and the retired-names doc tests pass. `docs/design/cohort_and_timing_rules.md` was updated
  for T7.

## Issues

### Critical

None.

### Important

**I1. The `speed_threshold` parameters left on the builders no longer gate SWR or MUA labels, so a
non-default value silently gives a mixed cohort.**
T7 deleted `_filter_external_event_labels` / `_filter_contfrag_event_labels` and the classifier's SWR gate.
The `speed_threshold` parameters stayed:

- `contfrag/tables.py:248, 560, 708`
- `tables_event.py:544`
- `tables_stop.py:71`
- `hippocampus/event_table.py:186`

At the default value nothing changes, because the read already gated at 4 cm/s. At a stricter value:

- `build_contfrag_event_table` gates the primary detector's events at the given threshold
  (`_in_contfrag_event_mask`, l.308). It tests overlap against the other detector's labels at 4 cm/s
  (l.296-300).
- `build_classified_nonlocal_event_table` flags the non-local events at the given threshold. It classifies
  them against SWRs gated at 4 cm/s. The docstring at l.200-203 now says this.
- Neither raises.

At a looser value, `canonical_events.load_session_swr_intervals` (l.269) and
`memoryless_sequence.canonical_swr_intervals` (l.466) silently behave as if the value were 4. This is the
deferred T7 minor. `canonical_swr_intervals`' docstring still claims it takes "the same two steps the
canonical table's SWRs take". At a non-default threshold it no longer does.

No production caller passes a non-default value today. `diagnose_pfc_event_definition.py` exposes the
option on its command line. A researcher who sets `speed_threshold=2` for a sensitivity check would
reasonably expect every cohort to be gated at 2. Instead some cohorts are gated at 2 and the rest at 4,
with no error.

Fix, choose one:
- (a) Fail closed. Every builder and kept filter raises `ValueError` unless `speed_threshold ==
  DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC` (or unless it is ≤ the default, for the two kept SWR filters).
- (b) Re-apply `filter_event_labels_by_speed` to SWR and MUA only when `speed_threshold < default`, and
  raise when it is above the default.

(a) is simpler and matches the "one rule" intent. Either way, add one test per builder.

**I2. The read's time-axis guard still lets an integer index through, and `intervals_from_labels` now
silently ignores a `time` column.**
`_frame_epoch` (swr_source.py:854-859) refuses a `time` column and a non-increasing index. It accepts a
`RangeIndex`. A single-epoch frame that went through `reset_index(drop=True)`, or a concat with
`ignore_index=True`, passes the check. Its row numbers are then used as seconds:

- `_drops` classifies the inventory SWRs against a speed series indexed 0..n.
- No inventory SWR lands on the grid, so `SWR_event` comes back empty.
- Excluded time matches nothing.

Nothing raises. This is the deferred T6 minor, promoted, because it is the one silent path left at the
read whose purpose is to refuse malformed frames.

T6 also changed `intervals_from_labels` (event_immobility.py:310-335) from "prefer a `time` column" to "any
column is ignored". It deleted the `set_index("time")` promotion in `build_classified_nonlocal_event_table`
and `canonical_events`. A frame with a `time` column and a non-time index used to be classified correctly.
It is now classified on the wrong axis with no error. All production frames are time-indexed (484
audited), so this is latent.

Several entry points accept frames that did not come through the read. The ledger names
`generate_figure02.py:246` and the exploratory `_fig2b_*` scripts.

Fix: add one small check, for example `require_time_index(frame)`: the index is float and increasing, and
there is no `time` column. Call it from `_frame_epoch` and from `intervals_from_labels`. Both are cheap.
Add a RangeIndex case to `TestTheLabelContract`.

### Minor (new; deferred minors are triaged below)

- **M1.** The decode identifier is still built in two places, and its inverse is a third copy of the suffix
  rule. They are `_decode_runner.run_decode` (l.393-401), `hippocampus/result_loading._load_saved_decode`, and
  `data_loaders.utils.parse_epoch_identifier`. A `build_epoch_identifier(EpochKey-like)` next to
  `decode_paths` would close the T8/T9 loop. This is a follow-up, not needed for the wave.
- **M2.** Two modules import private helpers across modules. `contfrag/result_loading.py` imports
  `hippocampus.result_loading._load_saved_decode`. `replay/learning.py` imports
  `replay.retrospective._session_keys`. Both came from e70ab12. Drop the underscore or move the helpers to a
  shared module.
- **M3.** `artifact_event_drops` / `read_artifact_event_drops` (swr_source.py:949, 998) validate at 0.8 only,
  while `with_swr_source` / `read_dot_with_swr_source` take `non_local_threshold`. This is the deferred T7
  minor; listed here because it is the read's API asymmetry. Add the keyword argument.
- **M4.** `_frame_epoch` uses `is_monotonic_increasing`, which is non-strict. A duplicated timestamp
  passes. The decode runner itself requires strictly increasing time (`_decode_runner.py:300`). Use
  `index.is_unique and is_monotonic_increasing`, or a strict diff, in the same check as I2.
- **M5.** `per_bin_design`'s module docstring still says it is "deliberately downstream of nothing but the
  design constants". It now imports `swr_source`. This is the deferred T2 wording minor; it is correct but
  misleading.

## Deferred-minor triage (33 in the ledger)

**Promote (2):**
- T6 `_frame_epoch` accepts a RangeIndex → I2.
- T7 kept filters treat a threshold above 4 as 4 → folded into I1.

**Fix in the wave (15):**
1. T1: `replace_outputs` silently skips outputs not in `paths`. Assert `set(outputs) <= set(paths)`. A typo'd
   output name currently drops a table without a word.
2. T1: `--run-id` in `run_memoryless_detection.py:1113` and the report and figure scripts is still
   unvalidated. Use `type=parse_run_id`. This completes T1's "every site".
3. T2: `NO_EXCLUDED_TIME` is a shared mutable frame. Document it as do-not-mutate, or make it a function.
4. T2: the inverted AST guard (`test_artifact_exposure.py:859`) does not cover `scripts/qa/compare_fits.py`.
5. T2: the `per_bin_design.py` module docstring wording (= M5).
6. T3: `docs/behavior_model_covariates.md:150` names `fit_uncertainty_rate.RL_COVARIATES`. It is now
   `model_specs.MU_UNCERTAINTY_RL_COVARIATES` (a GC7 doc update).
7. T3: the `docs/models_table.md` Estimators table has no Cross-history row.
8. T3: `ALL_PER_BIN_SPECS` should assert that the ladders' keys are disjoint.
9. T3: `fit_uncertainty_rate.py` has three `model_specs` import blocks, and `model_specs.py:3` says "four
   model families", which is now stale.
10. T4: no test pins each runner's `aggregate` / `load_saved`. A swapped loader on a resume would silently
    reuse the wrong decode's results.
11. T4: `scripts/README.md:5` says scripts "do not import each other". `_decode_runner`, `_report` and several
    `from scripts.…` imports contradict it.
12. T4: the tautological assert in `test_decoding_save_dir.py` (~l.1377 at the time).
13. T7: `artifact_event_drops` has no `non_local_threshold` keyword argument (= M3).
14. T9: `generate_event_gallery.render_epoch` reads `data_over_time` from `data_dir` (l.1685) but `results`
    from `PROCESSED_DATA_DIR` (l.1696). With `--data-dir` pointing elsewhere, one event's frame and posterior
    come from two different decodes. Use one root.
15. T9: `analyze_content_occupancy` (l.157/161) and `analyze_event_spike_evidence` (l.143/349) name the file
    with the default 80 but validate at their `NON_LOCAL_THRESHOLD` constant. Derive both from the constant
    (`threshold_percent=non_local_threshold_percent(NON_LOCAL_THRESHOLD)`).

**Leave (16):**
- T1 `stamped_tables` placement; the two `write_json` (different contracts, correctly not merged); the
  runner-level KeyboardInterrupt test (the shared helper is tested).
- T2 test warnings: 2 seen in my targeted run; check them in the full-suite run.
- T4 contfrag error wording.
- T5 the docstring's consumer list; the `_match_swr` 1×n matrix.
- T6 the ruff reflow comments in the test.
- T7 the `data_aggregation` whole-file reflow (ruff formatting is allowed); `run_state_evidence.py:1036`
  unreachable.
- T8 the redundant `key.*` locals; the fixtures that hand-slice ids; `check_pfc_ridge_lambda0` (outside the
  brief); the untracked `.replace({"j": "j16"})` (the lead's WIP).
- T9 contfrag's `_contfrag` name part (`f"_contfrag{data_over_time_suffix(None)}"` is clear enough).
- T9 `run_quality_metrics --threshold 29/57/58` truncation (pre-existing): flag it for the lead. The new read
  validates at `threshold/100`, and that stays consistent with the `_28_`-style names, so T7 does not make it
  worse.

## Declined to judge

- Whether the contfrag runner should also keep its log-likelihood. The plan reserves this for the lead.
- Whether a moving non-local or MUA label should be refused rather than dropped at the read. The T7 ruling
  (option A) settled this; 0/484 frames are affected.
- The scientific content of the MU and cross-history covariate blocks. T3 is a pure move, proven
  byte-identical.
- The extra artifact-file reads after T2's default: one per `aggregate_to_timebins` call, where the old code
  read once per session. These are small reads, a few per session. Nothing was measured, so I make no
  performance claim.
- Golden-table parity and full-suite results. I relied on the ledger's records (atol 0 after each
  table-affecting task) and did not re-run the 5-session rebuild. The full suite is running elsewhere.
- GPU and `LocalCUDACluster` behavior of `_decode_runner`. The controller verified it (scratchpad
  `task4_gpu/probe.py`); I did not re-run it.
- `analyze_content_occupancy.find_sessions`' `epoch_id.startswith(animal)` prefix match. It is pre-existing,
  and no current animal name is a prefix of another.
- The public keyword arguments T7 removed (`nonlocal_threshold` on `get_event_types`,
  `compute_swr_nl_event_counts`, `build_event_onset_table`, the figure panels). Any old caller fails loudly
  with a TypeError, and no in-repo caller remains. Notebooks are not maintained code.
- The edits to untracked WIP files (replay-story script and test, plan dirs, `.vscode`). They are not on
  the branch.

## Recommendations

1. Fix I1 and I2 first. Both are small, fail-closed guards with one test each, and neither changes a
   production output. Re-run `test_swr_source`, `test_replay_tables`, `test_classified_event_table` and the
   contfrag table tests.
2. Then run the 15-item minor wave. Items 1, 2, 10, 14 and 15 protect against silent mismatches; the rest
   are doc and test hygiene.
3. After the wave, spot-check one golden session through `with_swr_source`. It should be unchanged, since
   both guards are no-ops on time-indexed frames.
4. Later (M1): one `build_epoch_identifier` next to `decode_paths`, so the identifier rule lives in one
   module in both directions.

## Assessment: ready to merge? With fixes

The nine tasks meet their intent across the branch. The read point, the strict-overlap predicate, the shared
runner, the parser and the path module work together without conflicts. The invariants still hold:
analyzed time is applied once, overlap is strict at the event level and inclusive at the stop level, and
junction bins are untouched (the `spatial_mapping` change keeps nearest-over-real-bins). Two Important
gaps should close before merging: the builders' `speed_threshold` that no longer gates every cohort (I1),
and the read's integer-index / `time`-column blind spot (I2). Both can fail silently, and both are cheap to
make fail closed.
