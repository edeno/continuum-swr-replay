# SDD ledger — plan: .claude/docs/plans/simplify-cross-cutting/PLAN.md

Branch simplify-repo-2026-10-19 (main tree, in place). Merge base for final review: 345c134 (master). Plan base: e70ab12.
No spec; review findings in the plan are the requirements — rulings provisional.

## Pre-flight scan

| Pair / task | Shared file or interface | Finding |
|---|---|---|
| T1 × T6 | run_carey_detection.py, run_inventory_comparison.py | different regions (replace_outputs/run-id vs time-axis branch); no conflict |
| T1 × T8, T9 | run_inventory_comparison.py | different regions (_epoch_parts, decode_path); no conflict |
| T1 × T4 | decoding_common.py | T1 adds run-id validator; T4 refactors runner code there — T4 must keep T1's addition |
| T2 × T3 | fit_uncertainty_rate.py, fit_cross_history_rate.py | T2 edits excluded-interval plumbing, T3 moves specs; independent |
| T5 × T6 | event_immobility.intervals_from_labels | T5 makes the strict-overlap rule consume intervals_from_labels; T6 changes it to index-only. T5's new callers (data_aggregation on all-session frames may carry `time` as a column) must be audited by T6 |
| T5 × T7 | figure_panels.py ~415-424, analyze_event_spike_evidence, analyze_content_occupancy, quality_metrics, data_aggregation, contfrag/tables, tables_event | adjacent code (overlap mask follows the speed filter); sequential, T7 rebases on T5 |
| T6 × T7 | swr_source.with_swr_source | both add read-point normalisation/validation; T7 builds on T6 |
| T6 × T8 | run_memoryless_detection.py, analyze_event_spike_evidence.py | different regions |
| T7 × T9 | generate_event_gallery.py, data_aggregation.py | different regions (speed filter vs decode path/glob) |
| T8 × T9 | prepare_data.py, run_memoryless_detection.py, generate_event_gallery.py, run_inventory_comparison.py | T8 parses ids, T9 builds paths from them; T9 consumes T8's parser if useful |
| T1 self | — | rollback behaviour change vs "success-path bytes identical": consistent |
| T2 self | — | default-derive vs explicit no-mask sentinel: consistent |
| T3 self | — | consistent |
| T4 self | — | "preserve behaviour incl. log-likelihood drift": consistent |
| T5 self | — | stop-and-report if A ≠ B: consistent |
| T6 self | — | consistent |
| T7 self | — | keep deliberate non-default thresholds: consistent |
| T8 self | — | consistent |
| T9 self | — | identical paths + gallery behaviour change: consistent |

Ruling: execute tasks 1→9 sequentially in place on the branch, no worktree — lead's standing SDD preference — cost if wrong: none (all local commits).
Ruling: T5 runs before T6, and T6 must audit every intervals_from_labels caller including those T5 adds — order follows the plan; T6 is the time-axis owner — cost if wrong: a caller passing a `time` column breaks loudly in T6's tests.
Ruling: golden reference is 1 session per animal (5 sessions) at e70ab12, not the full 242-epoch cohort — a full regen is hours; 5/5 animals covers per-animal quirks — cost if wrong: a rare-case difference in an unsampled session is missed until the next full regen.
Ruling: implementers and task reviewers on opus; final whole-branch review on fable — lead's standing preference (fable rationed) — cost if wrong: none.
Ruling: commits stay local; push is blocked on breeze (gh missing) — cost if wrong: unpushed work until pushed from zephyr.

## Tasks
Task 1: dispatched (BASE b25b3b9, implementer a9cd15ceb1dcf39e5)
Ruling: final whole-branch review on opus, not fable — lead said 2026-10-19 09:31 'we don't need to use fable for the final whole branch review' — cost if wrong: none.
Task 1: implementer DONE (9a9603c); review dispatched (a1e516b78c023f085)
Task 1: ⚠️ success-path bytes on real outputs — resolved by controller: implementer's simulated old-vs-new sha256 check (scratchpad/task1/bytes_check.py, 23 entries identical) covers all 4 runners; runners don't touch builders so golden not required.
Task 1: minor (deferred): stamped_tables (run_inventory_comparison.py:602) belongs next to swr_source.stamp_swr_source
Task 1: minor (deferred): decoding_common.write_json (strict) vs state_evidence/io.write_json (str fallback) — two write_json in src, do not merge blindly
Task 1: minor (deferred): replace_outputs silently skips outputs not in paths — could assert set(outputs) <= set(paths)
Task 1: minor (deferred): no runner-level KeyboardInterrupt rollback test for carey/swr
Task 1: minor (deferred): --run-id in run_memoryless_detection.py:1113 and report/figure scripts still unvalidated
Task 1: complete (commits b25b3b9..9a9603c, review clean)
Task 2: dispatched (BASE 9a9603c, implementer aa29c3b9b8d8f20f0)
Task 2: implementer DONE (afefd94); review dispatched (a8b6867a0d88cb662). Note: exploratory/_covariate_ranking.py already broken pre-task (reward_lags=), left alone.
Task 2: ⚠️ real-data identity — resolved by controller: ran scratchpad/task2/compare.py before.pkl after.pkl → ALL IDENTICAL 12 (3 sessions × 4 populations, frames + attrs).
Task 2: minor (deferred): NO_EXCLUDED_TIME is a shared mutable module-level DataFrame (per_bin_design.py:503) — document do-not-mutate
Task 2: minor (deferred): inverted AST guard (test_artifact_exposure.py:859) does not cover scripts/qa/compare_fits.py
Task 2: minor (deferred): per_bin_design.py:456-459 module docstring wording
Task 2: minor (deferred): test warnings (2/43) unattributed — check in full-suite run; test_artifact_exposure.py:792 magic 8+1
Task 2: complete (commits 9a9603c..afefd94, review clean)
Task 3: dispatched (BASE afefd94, implementer a0ab839c21e9730f1)
Task 3: implementer DONE (8664f23); review dispatched
Task 3: ⚠️ test pass output unseen — resolved: covered by the controller's end-of-branch full suite.
Task 3: minor (deferred): docs/behavior_model_covariates.md:150 still names fit_uncertainty_rate.RL_COVARIATES (GC7 doc update) → model_specs.MU_UNCERTAINTY_RL_COVARIATES
Task 3: minor (deferred): docs/models_table.md Estimators table lacks a Cross-history row
Task 3: minor (deferred): ALL_PER_BIN_SPECS (test_model_specs.py) should assert disjoint keys
Task 3: minor (deferred): fit_uncertainty_rate.py three model_specs import blocks; model_specs.py:3 docstring "four model families" stale
Task 3: complete (commits afefd94..8664f23, review clean)
Task 4: dispatched (BASE 8664f23, implementer a8c046f4f80770ad2)
Ruling: Task 4 keeps the bandit-only return_outputs="log_likelihood" persistence as an explicit per-runner field, not unified — lead decides whether the drift is intended — cost if wrong: contfrag decodes keep lacking saved log-likelihoods until decided.
Ruling: after the final whole-branch review + fix wave, dispatch an independent opus agent to review the overall code organization of the repo — lead asked 2026-10-19 10:11 — cost if wrong: none.
Task 4: implementer DONE_WITH_CONCERNS (051b056): contfrag non-increasing-time error text unified; runners put repo root on sys.path (dask LocalCluster verified, LocalCUDACluster not); detector class lookup now raises on unknown name (was silent sorted-spikes fallback). Concerns are observations → proceed to review.
Task 4: ⚠️ LocalCUDACluster worker import — resolved by controller: scratchpad/task4_gpu/probe.py, LocalCUDACluster(CUDA_VISIBLE_DEVICES="1"), cwd=/tmp, worker unpickled DecodeRunner → ('scripts.decoding._decode_runner', True, True, '1').
Task 4: minor (deferred): contfrag non-increasing-time error wording now the bandit's (_decode_runner.py:304)
Task 4: minor (deferred): no test pins each runner's aggregate/load_saved field (bandit get_data_over_time/load_results, contfrag get_contfrag_data_over_time/load_contfrag_results)
Task 4: minor (deferred): scripts/README.md:5 "do not import each other" contradicts _-prefixed shared modules
Task 4: minor (deferred): tautological assert test_decoding_save_dir.py:1377
Task 4: complete (commits 8664f23..051b056, review clean)
Task 5: dispatched (BASE 051b056, implementer a038f97e4f7c2eff8)
Task 5: implementer DONE (0cfe2c4); A≡B parity on 5 golden sessions; golden {} 7 tables (qvals is an input, not rebuilt). Open note: A reads index, B reads time column when they disagree — no production frame has a time column → Task 6 territory.
Task 5: ⚠️ no production frame carries a `time` column for A callers — carried into Task 6 (time-axis owner) with the caller list: tables_event.py:140, contfrag/tables.py:311, quality_metrics.py:1017, data_aggregation.py:864/902/1060/1065, figure_panels.py:428, analyze_event_spike_evidence, analyze_content_occupancy.
Task 5: ⚠️ B bounds positional times[0]/[-1] vs A min/max — agree on sorted index; precondition in validate_dot_event_contract (unchanged) — carried into Task 6 (read point should assert a sorted time index).
Task 5: minor (deferred): strict_overlap_matrix docstring lists consumers by name (stale-prone)
Task 5: minor (deferred): _match_swr builds 1×n matrix per event via np.array wrappers (readability)
Task 5: complete (commits 051b056..0cfe2c4, review clean)
Task 6: dispatched (BASE 0cfe2c4, implementer abe9bffb0af298dbf), carrying Task 5's two ⚠️ items
Task 6: implementer DONE (d8778a6); 484 decode pickles audited (time index, no column); golden {} 7 tables atol=0
Task 6: ⚠️ test output not saved — resolved: covered by end-of-branch full suite. ⚠️ classifier callers outside the source read (generate_figure02.py:246, generate_event_gallery.py:246, exploratory) rely on upstream ordering — accepted per brief's "one place".
Task 6: minor (deferred): _frame_epoch accepts a RangeIndex (reset_index'd frame) — could assert index.name == "time" / float dtype (brief said "normalise to, or assert, a time index")
Task 6: minor (deferred): ruff reflow left trailing comments after closing parens in test_classified_event_table.py (~diff l.862-877)
Task 6: complete (commits 0cfe2c4..d8778a6, review clean)
Task 7: dispatched (BASE d8778a6, implementer a20950504fddde2d6); facts-first, stop if any re-filter is not a no-op on production data
Task 7: implementer NEEDS_CONTEXT (uncommitted): 484/484 frames no-op, 38/38 site outputs identical, golden {} ×7; read-point raise refuses inventory-comparison fixtures' moving state-space event (55 errors)
Ruling: Task 7 keeps the immobility raise at the read (option A); inventory/event-features fixtures drop the moving state-space event, moving-event coverage retargeted to a detector whose events can move — brief mandates fail-closed read; 0/484 production frames carry a moving label — cost if wrong: hand-built frames with moving labels can't be fed to the inventory comparison without going around the read.
Task 7: implementer DONE (e9cd5af) after option-A fixture fix; review dispatched
Task 7: review dispatched (ab31a04729f69c21d)
Task 7: review ✅/approved (task-7-review.md). ⚠️ read-time cost — resolved by controller: scratchpad/task7_ctrl/time_validators.py, validators 0.12 s on a 1.3–1.4 M-row frame (read 0.45 s excl. pickle load) — negligible. ⚠️ notebooks not searched for removed kwargs — accepted (notebooks not maintained code).
Task 7: minor (deferred): kept filters (canonical_events.py:~253, memoryless_sequence/event_table.py:~465) silently treat a threshold > 4 cm/s as 4 (pre-existing) — suggest ValueError
Task 7: minor (deferred): artifact_event_drops (swr_source.py:993) lacks non_local_threshold kwarg
Task 7: minor (deferred): data_aggregation.py whole-file ruff reflow
Task 7: minor (deferred): run_state_evidence.py:1036 event_bins_at_running_speed unreachable non-zero through the read (kept per ruling)
Task 7: complete (commits d8778a6..e9cd5af, review clean)
Task 8: dispatched (BASE e9cd5af, implementer a5bce6049c863e13e)
Task 8: implementer DONE_WITH_CONCERNS (736e18b): 3,025 real ids old≡new, golden {} ×7; malformed ids now raise ValueError (no real id affected); review dispatched
Task 8: review ✅/approved (task-8-review.md). ⚠️ parity not re-run by reviewer — resolved by controller: re-ran scratchpad/task8/check_parsers.py compare → ALL IDENTICAL; golden compare.log {} ×7 at atol 0.
Task 8: minor (deferred): run_state_evidence.py:~399-404 redundant key.* locals
Task 8: minor (deferred): test fixtures still hand-slice ids (test_run_memoryless_detection.py:365, test_decode_folder_readers.py:66-69)
Task 8: minor (deferred): check_pfc_ridge_lambda0.py:183 hand-splits {stem}_{epoch} (no area; outside brief)
Task 8: minor (deferred): untracked generate_replay_story_figures.py .replace({"j": "j16"}) is a no-op now (CSV already says j16) — lead's WIP, not touched
Task 8: complete (commits e9cd5af..736e18b, review clean)
Task 9: dispatched (BASE 736e18b, implementer a39e0727b78c7b147)
Task 9: implementer DONE_WITH_CONCERNS (755fcc0, 58ea08f): paths byte-identical 242 epochs × variants × thresholds; golden {} ×7; gallery glob 242 = register 242 in Processed-Data (backup/staging glob had +11/13 SUSTAINED_NOISE); decode_paths placed in decoding_common → dask.distributed now imported by pfc events/result_loading/data_aggregation (~3.9 s); review dispatched
Task 9: review ✅ spec / needs fixes: Important — decode_paths in decoding_common drags dask.distributed+dask_cuda into 5 library readers and adds a fragile hippocampus→decoding_common→swr_source edge; fix = leaf module decode_files.py (task-9-review.md)
Task 9: fix round 1 dispatched (resume a39e0727b78c7b147): move to decode_files.py + lazy-import test; gallery logs missing analysed decodes; drop duplicate _decode_paths wrappers if trivial
Task 9: minor (deferred): contfrag/data_aggregation builds `_contfrag` name part by hand
Task 9: minor (deferred, pre-existing, flag for lead): run_quality_metrics --threshold 29/57/58 looks for _29_/_57_/_58_ but 0.29/0.57/0.58 decodes write _28_/_56_/_57_ (int truncation); memoryless names _contfrag_80_ for a ContFrag variant
Task 9: minor (deferred, pre-existing): generate_event_gallery render_epoch reads dot from data_dir but results from PROCESSED_DATA_DIR
Task 9: fix round 1 DONE (44996e7): decode_files.py leaf module, lazy-import test (6 modules), gallery warns on missing analysed decodes, wrappers deleted; paths 242 byte-identical; scoped re-review dispatched (ac5dc56fa8e9370b0, sonnet — scoped re-review of a small fix diff)
Ruling: scoped re-review of Task 9's fix round on sonnet, not opus — skill tiers scoped re-reviews of small fix diffs cheap-to-mid; findings are mechanical (module move, warning) — cost if wrong: a missed regression in the move, still caught by the opus final whole-branch review.
Task 9: re-review — all 3 findings fixed (task-9-rereview.md)
Task 9: ⚠️ gallery counts / golden / path dump not re-run by first reviewer — resolved: fix round re-ran path parity (paths_fix1_cmp.txt, 242 byte-identical); golden {} ×7 from implementer's run at 755fcc0; fix round changed only prepare_data's import line.
Task 9: minor (deferred): analyze_content_occupancy / analyze_event_spike_evidence call decode_paths with the default threshold rather than their NON_LOCAL_THRESHOLD constant (equal today)
Task 9: complete (commits 736e18b..44996e7, review clean after 1 fix round)
Final review: dispatched (merge base 345c134..44996e7, opus, a6311dfa4b45e57c4); full suite running at 44996e7 (scratchpad/fullsuite)
Final review: With fixes — 0 Critical, 2 Important (I1 builders' speed_threshold no longer gates SWR/MUA; I2 read accepts RangeIndex / intervals_from_labels ignores a time column), 15 minors for the wave, 16 leave (final-review.md)
Ruling: I1 fix = table builders raise ValueError on a non-default speed_threshold; the two kept knob filters (canonical_events, memoryless) raise only on a threshold looser than 4 cm/s (stricter stays honoured — diagnose --immobility-threshold is a real sensitivity knob) — no production caller passes a non-default value — cost if wrong: a future stricter-threshold sensitivity run on the builders must re-add the label filter.
Ruling: I2 fix = one shared time-axis check (float, strictly increasing index, no `time` column, not a RangeIndex) used by _frame_epoch and intervals_from_labels — T6 audit found 0/484 frames with a time column — cost if wrong: hand-built frames with integer indices must be given a float time index.
Ruling: wave item 14 (gallery one root) = both frame and results read from data_dir (default PROCESSED_DATA_DIR) — a frame and posterior from two decodes is never intended — cost if wrong: none for the default path.
Ruling: hold the fix wave until the in-flight full suite at 44996e7 finishes, so its results aren't contaminated by edits and any failure can join the wave — cost: ~15 min wall clock.
Full suite at 44996e7: 7 failed (exactly the 7 untracked-WIP failures: test_replay_story_figures ×5, test_session_keys ×2), 5073 passed, 2 skipped, 78 warnings, 30:41 (scratchpad/fullsuite/pytest_44996e7_full.txt)
Final fix wave: dispatched (BASE 44996e7, fresh opus implementer a226d8986b5e8f10b)
Final fix wave: DONE (049506a..7e54217, 6 commits); golden {} ×7; scoped re-review + full suite next
Final fix wave: scoped re-review dispatched (opus abdd4962e72ce6835 — 120 KB behavioural diff, so not the cheap tier); full suite running at 7e54217; code-organization review dispatched (opus ab1d7319d5f057e0b → .claude/docs/reviews/2026-10-19-code-organization-review.md)
Final fix wave: re-review — all 17 fixed (final-rereview.md); guards misfire on no production path; parse_run_id accepts all 22 existing run-id folders
Ruling: the residual minor (build_event_arm_table, tables_arm_patch.py:67, unguarded speed_threshold) is applied by the controller as a one-line guard + test after the in-flight full suite finishes — mechanical, mirrors the ContFrag event-arm guard — cost if wrong: none (a non-default value already raises upstream in build_event_table).
Code-organization review: DONE — .claude/docs/reviews/2026-10-19-code-organization-review.md, 12 findings (top: eager replay/__init__ ~4.5 s imports, 6 package-level cycles via hippocampus, run infra split 3 ways, scripts-as-libraries 35 imports, inline output paths ×50), 9 questions for the lead; not acted on (lead decides a follow-up plan)
Final: residual minor applied (2b38381); 107 targeted passed. Branch complete.
