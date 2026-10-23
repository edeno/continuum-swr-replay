# SDD ledger — plan: .claude/docs/plans/code-organization/PLAN.md

Branch code-organization-2026-10-19 (main tree, in place), from master 470a393; plan commit 8971a76. Merge base for final review: 470a393.
Spec: .claude/docs/reviews/2026-10-19-code-organization-review.md + lead's answers (PLAN.md "Decisions").

## Pre-flight scan

| Pair / task | Shared file or interface | Finding |
|---|---|---|
| T1 × T4–T16 | test_import_layering KNOWN_VIOLATIONS, JAX_FREE | later tasks only delete allowlist entries / rename JAX_FREE entries when modules move; T5/T6/T9 add packages to the foundation set (new rules, not allowlist growth) — consistent |
| T2 × all | full-suite = default + -m real_data | consistent; controller runs both |
| T3 × T6,T7,T9,T13 | test_doc_paths_resolve | later moves must keep it green (codemod covers *.md) — consistent |
| T4 × T6 | hippocampus/artifacts/masks.py | T4 takes the generic interval functions out; T6 moves the remainder to events/artifact_masks — consistent |
| T4 × T14 | pfc/glm.py | T4 removes get_spikecount_per_time_bin, _overlaps_any, _contiguous_true_runs before T14 splits — consistent |
| T5 × T6 | replay/event_ids, cohort_selection | classified_table imports event_ids → moved to tables/ in T5 first, so events/ stays foundation-only — consistent |
| T5 × T10 | CANONICAL_TABLE_BASENAMES | table_parity → tables/parity (T5) → paths (T10, parity imports back) — consistent |
| T6 × T7 × T10 × T11 | events/swr_source | T7 splits it; T10 moves detection roots out of swr_inventory; T11 routes swr_inventory's epoch-id builder via format_epoch_identifier — order matches |
| T8 × T9 | decoding_common.py | T8 takes io/provenance/cli/batch/logging; T9 takes config + decode masks and deletes the module — consistent |
| T8 × T13, T9 × T13 | runner scripts, _decode_runner | T8 de-libraries the runners, T9 moves _decode_runner into src, T13 moves files — order matches |
| T11 × T15 | session_tables.py:114 _LEAF_ARMS | T11 edits before T15 splits — consistent |
| T12 × T13 | script-to-script imports | T12 removes them before T13 regroups (review's order) — consistent |
| WIP × T12/T14 | model_specs, per_bin_design, pfc.load_population (imported by untracked WIP) | T12 adds to model_specs (module stays); T14 leaves load_population — consistent |
| Q5 × all | memoryless_sequence.config/.detect pickled | no task moves them — consistent |
| T1 self | lazy __init__ vs import-time side effects | Review Focus 1 + golden check — consistent |
| T2 self | opt-in real_data vs loud failure under -m real_data | consistent |
| T3 self | allowlist must end empty or justified | consistent |
| T4 self | stop if a replaced copy disagrees | consistent |
| T5 self | table_io → swr_source back edge: stop if no clean move | consistent |
| T6 self | moved set must be closed; stop otherwise | consistent |
| T7 self | swr_inventory is the lower layer | consistent |
| T8 self | json_ready variants: stop if not identical | consistent |
| T9 self | results readers identical or stop; CPU worker probe | consistent |
| T10 self | resolved-path parity | consistent |
| T11 self | order-preserving animal lists | consistent |
| T12 self | output-subdir pinning | consistent |
| T13 self | PRODUCED_BY new path, no stubs (lead Q1) | consistent |
| T14 self | needs CPU PFC run into scratch; stop if no output-dir option | consistent |
| T15 self | pixel-identical render | consistent |
| T16 self | consistent |

Ruling: execute T1→T16 sequentially in place, opus implementers and task reviewers, opus final review — lead's standing SDD preference (fable not used, per 2026-10-19) — cost if wrong: none.
Ruling: event substrate → analysis/events/ — lead chose (a) 2026-10-19 19:30 after a pros/cons — cost if wrong: large import churn, reversible by git.
Ruling: no re-export shims; all in-repo imports rewritten; only pickled modules (memoryless_sequence.config/.detect, which do not move) would get aliases — controller's call with Q1 "no stubs" spirit — cost if wrong: external notebooks/out-of-repo code using old paths break loudly on import.
Ruling: Q5 answered by controller: grep over 6,396 non-backup pickles found package refs only in 706 memoryless candidate pickles (memoryless_sequence.config/.detect) — cost if wrong: a pickle in a _backup dir referencing a moved module would need an alias to load.
Ruling: start Phase A (T1–T3, no code moves) while the lead reviews the plan; Phase B begins after — autonomy preference + writing-plans review gate — cost if wrong: Phase A rework if the lead changes the guards.

## Tasks
Task 1: dispatched (BASE 8971a76, implementer afcde96f85f1fb9fc)
Ruling: task reviews only at phase boundaries — one opus review of the whole phase diff after T3 (Phase A), after T11 (Phase B), and the final whole-branch review after T16 (Phase C) — lead said 2026-10-19 19:42 'only at the phase boundaries'; per task the controller only checks the commit, the golden log (GOLDEN IDENTICAL) and the saved test output — cost if wrong: a defect in an early task is found later and its fix may touch more code.
Task 1: DONE_WITH_CONCERNS (765db11): 9 leaves 4.1–5.8 s → ≤0.6 s, JAX-free; golden.log GOLDEN IDENTICAL; 889 targeted passed. Note for T5/T6: stale workaround comments at hippocampus/event_table.py:74-80 and session_keys.py:19-24 (T5 removes the session_keys workaround; T6 moves event_table) — fix there.
Task 1: complete (controller checks: commit, golden.log, targeted_tests.log)
Task 2: dispatched (BASE 765db11, implementer a16b348257f754e58)
Task 2: DONE_WITH_CONCERNS (49c04dc): default 5129 passed / 7 known failures / 19 real_data skips (21.7 min); -m real_data 18 passed, 1 failed (test_matched_control real-session: results_exist call stale after silent skips — fixed in 49c04dc, verified offline, not re-run: ~77 min); added autouse conftest fixture restoring ARTIFACT_INTERVALS_RUN_ID for real_data tests (beyond brief — phase review to check)
Task 2: carry → Task 12: 3 synthetic tests need the Spyglass DB at import (test_generate_figure04 collection error, one in test_run_state_evidence, one in test_table_builders_read_the_register)
Ruling: tests/QUICKSTART.md staleness folded into Task 3 (docs) — same doc-home cleanup — cost if wrong: none.
Task 2: complete (controller checks: commit, both suite logs)
Task 3: dispatched (BASE 49c04dc, implementer ae27248a09e29f36d)
Task 3: DONE_WITH_CONCERNS (428f771): doc-path test 3 tests 1.0 s, mutation-checked; pipeline map current; concerns: top-level plans/ skipped as historical, compare_fits default report path → .claude/docs/reviews/, short analysis/...py refs not covered (42, 1 dead), CLAUDE.md names gone tables.py
Task 3: complete
Phase A: review dispatched (8971a76..428f771, opus)
Phase A: review — spec ❌ (fixable I1/I3/I4), quality needs fixes, nothing Critical (phase-A-review.md). M4: compare_fits default report path → .claude/docs/reviews/ is Phase A's one deliberate behaviour change (accepted).
Ruling: M6 — drop figure_utilities from FOUNDATION (plan's list omits it; a plotting helper must stay free to import visualization) — cost if wrong: none.
Ruling: M2 — move top-level plans/ to .claude/docs/plans/2026-04-legacy/ in its own commit — same doc-home rule as the lead's docs/audits answer; a move, not a deletion — cost if wrong: one git mv back.
Ruling: plan Global Constraints amended: grep bare <basename>. references in *.md/docstrings after each move (review I2) — cost if wrong: none.
Carry → T4 brief: add a dask check for intervals (_HEAVY_MODULES lacks dask). Carry → T5/T6/T7: tests/conftest.py:61-100 swr_source import + tests/README swr_source.ARTIFACT_INTERVALS_RUN_ID must follow the moves; regenerate CLAUDE.md tree (T5/6/7/9/13/16). Carry → T12: name the 2 DB-at-import runtime tests (test_run_state_evidence, test_table_builders_read_the_register).
Controller: matched_control real_data rerun started in tmux 'matched_control_rerun' at 428f771 → scratchpad/phaseA/matched_control_rerun.txt
Phase A: fix wave dispatched (BASE 428f771, implementer a4b231e611287c4fa)
Phase A: fix wave DONE (6e97cb4..7a70921, 8 commits): guards 50 passed; no-DB collect 0 errors (5166 collected)
Ruling: no scoped re-review of the Phase A fix wave — lead wants review only at phase boundaries; fixes are small, test-pinned, and the Phase B review re-reads the guards — cost if wrong: a guard defect surfaces at the Phase B review.
Phase A: complete
Task 4: dispatched (BASE 7a709713, implementer afe5edadff87a9f7c)
Task 4: BLOCKED (no commits) — lfp._overlap_any vs masks.overlaps_intervals differ only on NaN/inverted bounds; latent merge_intervals NaN bug ([NaN,4],[NaN,5] → [[4,NaN]]); 4th convention event_eligibility.overlaps_dropped_events (ignores zero-length dropped intervals, counts zero-length events; docstring falsely claims strict); glm._overlaps_any is open '<'
Ruling: canonical interval primitives raise ValueError on NaN/inverted bounds (separate commit after the pure move), conditional on a read-only production scan finding none — fail-closed, fixes the merge bug — cost if wrong: a caller with legitimately NaN rows breaks loudly.
Ruling: overlaps_dropped_events keeps its behaviour (no science change in a refactor); docstring corrected, named in the conventions table; divergence-from-strict counts measured — QUESTION FOR LEAD: should it become strict? — cost if wrong: none now.
Ruling: glm._overlaps_any moves unchanged as an accurately named open test; interval_mask + pad_intervals move (generic); swr_inventories.drop_overlapping_events inline copy replaced if edge-case identical — cost if wrong: none.
Task 4: resumed with rulings
Task 4: DONE_WITH_CONCERNS (122da74..348c671, 4 commits): golden GOLDEN IDENTICAL ×4 (after every commit); PFC binning parity 184 trains identical; 2006 targeted passed; production scan 242 epochs + inventories: 0 NaN/inverted rows → raise added; overlaps_dropped_events differs from strict in 0 cases cohort-wide (answers the lead question: no data impact). Controller verified the WIP files that were briefly swept into a commit are untracked and unmodified (Sep 17 mtimes).
Task 4: minor (for Phase B review): overlaps_closed/half_open names not created (closed keeps overlaps_intervals; no half-open exists); interval-set family + overlaps_open unvalidated; leftover local overlap copies listed in task-4-report (incl. lfp._in_intervals)
Task 4: complete
Task 5: dispatched (BASE 348c671e, implementer ab896052e9718823f)
Phase A ⚠️ resolved: matched_control real_data rerun at 428f771 → 1 passed in 1:30:41 (scratchpad/phaseA/matched_control_rerun.txt)
Task 5: DONE_WITH_CONCERNS (0781c98, 4e3979c): golden self + 5-session GOLDEN IDENTICAL; 1789 targeted passed; backup pickles 0 hits / 3,950. Design: table_io→swr_source back edge inverted — SWR_SOURCE_TABLES in tables/io, publish_canonical(require_swr_source=) from all 9 SWR publish sites (for Phase B review / lead).
Carry → T6: add intervals.py (and tables/) to the CLAUDE.md tree if missing. Pre-existing, not ours: scripts/exploratory/_univariate_coefs.py imports nonexistent analysis.replay.rate_model.
Task 5: complete
Task 6: dispatched (BASE 4e3979ca, implementer ab1fdbb91148a8186)
Task 6: DONE_WITH_CONCERNS (a1d720de): golden GOLDEN IDENTICAL; 43/43 sites identical incl. artifact_event_drops; 3300 targeted passed; guards green. Stop condition: swr_inventories.kay_version_check needs inventory_comparison.schema/overlap. Dropped 7 cohort_selection re-exports from replay/__init__ (no shims).
Ruling: option A — move kay_version_check into analysis/inventory_comparison/ (it compares Kay inventory versions; cuts the last hippocampus↔inventory_comparison cycle) — cost if wrong: one function's import path.
Ruling: run_artifact_detection.py:161 manifest string "masks.PADDING_S" left unchanged — it is recorded provenance, possibly compared by resume logic — cost if wrong: a stale name in new manifests.
Task 6: resumed for option A
Task 6: option A done (54e312d7): kay_version_check → inventory_comparison/kay_version.py; 8/8 outputs identical on 2 epochs; hippocampus rule + no_package_cycles allowlists now EMPTY; "masks.PADDING_S" kept — manifest settings compared on resume (run_artifact_detection.py:473-479).
Carry → T7: inventory_comparison/__init__.py docstring still lists the moved `intervals` submodule.
Task 6: complete
Task 7: dispatched (BASE 54e312d7, implementer af06fda032d20f10e)
Task 7: DONE_WITH_CONCERNS (c5e07903): golden GOLDEN IDENTICAL; site check 105 keys 0 diffs (BASE worktree vs tree); 3288 targeted passed. swr_observed_* → analysed_time (they call excluded_intervals); analysed_time imports 5 private names from swr_inventory (Phase B review: make public?); adapters.swr_inventory function shadows the module → imported `as source_inventory` in 4 files; manifest string "excluded time (swr_source.excluded_intervals)" kept (compared on resume).
Task 7: complete
Task 8: dispatched (BASE c5e07903, implementer a4e4c2a0ea557089d)
Task 8: DONE_WITH_CONCERNS (92718f3c): golden GOLDEN IDENTICAL; json_ready identical on 737 real JSON; 266 manifests byte-identical; resume settings 12/12 match; 2441+154 targeted passed. For Phase B review: (1) runs.io.replace_outputs imports tables.io.write_table so tables.io keeps its own atomic writer (4 → 2 writers) — consider putting replace_outputs in tables/ or passing the writer so runs stays lowest; (2) mkstemp temp names + --no-optional-locks share the move commit (violates "move and behaviour change never share a commit"); (3) marks_sha256 / memoryless_output_paths / MARKS_CACHE_DIR → analysis/hippocampus/detection_inputs.py; (4) json_ready Mapping/ndarray branch never reached by existing outputs.
Task 8: complete
Task 9: dispatched (BASE 92718f3c, implementer a0341f181af1236c6)
Task 9: DONE_WITH_CONCERNS (282444b4): GOLDEN IDENTICAL; PATHS_IDENTICAL 242 epochs incl. both runners' save paths; READERS IDENTICAL (5 sessions × bandit/contfrag, state_bins compared by MultiIndex.equals — No-Spike NaN position breaks Dataset.identical even for same reader); CPU LocalCluster probe OK; 820 targeted passed. load_conditional_posterior → hippocampus/posterior_analysis; run_state_evidence fixture gained the position coordinate real decodes carry.
Ruling: _decode_runner.py stays in scripts/decoding/ as a sanctioned `_` shared module — it imports analysis (with_swr_source, HPC masks) and has CLI, so it cannot join the analysis-free decoding/ package — cost if wrong: the dask worker keeps needing the repo root on sys.path.
Task 9: complete
Task 10: dispatched (BASE 282444b4, implementer a0854871d038bd078)
Task 10: DONE_WITH_CONCERNS (6972f0f4): GOLDEN IDENTICAL (self + rebuild); 297 codemod + 27 hand sites resolve identically; 169 module-level paths in 74 modules match BASE; test_paths 32 passed; 2812+194 targeted. For Phase B review: ~40 sites spell root / MODEL_DIRS[k].name — a model_dir(key, root) helper would read better; ARTIFACT_INTERVALS_ROOT renamed ARTIFACT_INTERVALS_DIR everywhere; fixed a test-redirect hazard in analyze_content_occupancy.
Task 10: complete
Task 11: dispatched (BASE 6972f0f4, implementer a472033fd0dd69039)
Task 11: DONE_WITH_CONCERNS (ccf9e13d, 17bc5147): parity 0 diffs (258 values, 380 signatures, 28,222 inline checks, 9,954 epoch ids); GOLDEN IDENTICAL; 3144 targeted passed. Analysis sampling defaults use DEFAULT_SAMPLING_FREQUENCY_HZ (float) so manifests keep 500.0; canonical_events dropped its parse_nwb_filename check (output identical — Phase B review: is the validation lost?); CLAUDE.md tree updated for COLORS.
Task 11: complete
Phase B: review dispatched (7a709713..HEAD, opus); full suite (default) + -m real_data (tmux 'phaseB_realdata') started
Phase B: review — spec ✅, quality approved with fixes, nothing Critical; all layering allowlists incl. cycles EMPTY; decoding/ imports only parameters+types (phase-B-review.md). I1: 63 sites spell root / MODEL_DIRS[k].name → add model_dir(key, root) before Phase C. M5: restore the NWB-name check in format_epoch_identifier. Rulings by reviewer accepted: replace_outputs stays in runs.io; require_swr_source sound; private names + source_inventory alias kept; Task 8's bundled change accepted (no output change), no history rewrite.
Phase B: fix wave waits for the default suite at 17bc5147 to finish (avoid contaminating it)
Phase B: default suite at 17bc5147 — 7 known failures only, 5262 passed, 19 skipped, 19:55
Phase B: fix wave dispatched (BASE 17bc5147, implementer a2642bd3a8288c234): items 1–7 + M7; skip M1, M8, M11
Ruling: Task 8's mkstemp/--no-optional-locks changes bundled into the move commit 92718f3c are accepted as-is — they changed no output (266 manifests byte-identical), and history is not rewritten — cost if wrong: a bisect lands on a mixed commit.
Note (M13): commit 282444b4's subject says "…and runner live in decoding/" but the runner stayed in scripts/decoding/_decode_runner.py (body says so); not rewritten.
Phase B: fix wave DONE (4c75305..71d7c55, 9 commits): golden ×2 GOLDEN IDENTICAL; 58 rewritten expressions + 235 module-level paths identical; MODEL_DIRS replaced by paths.model_dir(key, root) (no alias); NWB-name check restored in session_data_over_time_path, not format_epoch_identifier (54 synthetic tests use 1-digit fake dates; the wider version is saved in scratch); M7: interval_mask replaces lfp._in_intervals (identical on valid input; old copy wrong on 53 bad-input cases).
Ruling: accept the narrower NWB-name check placement — it restores exactly what Task 11 lost without rewriting 54 synthetic fixtures — cost if wrong: format_epoch_identifier accepts malformed names (loud failure later).
Ruling: no scoped re-review of the Phase B fix wave (lead: phase-boundary reviews only); the final whole-branch review covers it — cost if wrong: found at the final review.
Phase B: complete (real_data suite at 17bc5147 still running in tmux phaseB_realdata)
Task 12: dispatched (BASE 71d7c55b, implementer a54de8d489bdbe7ef)
Phase B: -m real_data suite at 17bc5147 — 19 passed, 0 failed, 1:57:13 (scratchpad/phaseB/pytest_realdata_17bc5147.txt)
Task 12: DONE_WITH_CONCERNS (8c0aeca..8f69070, 7 commits): golden identical; output locations identical (378 names, 864 arg combos, 6,048 partial-ladder names, 2,304 CLI arg lists); 1091 targeted passed; both carried no-DB tests pass without DB; script→script imports 27 → 21, all allowlisted (shrink-only). For final review: contrast_aggregation ~510 → ~1,150 lines; model_specs now imports pandas + content_response (leaf-only helpers; layering forbade cohort_selection); 3 scripts load another script by file path (guard can't see); new public names beyond the brief.
Task 12: complete
Task 13: dispatched (BASE 8f690706, implementer a1f5c87a1f35e720e)
Task 13: DONE_WITH_CONCERNS (46389e0, 3782844): 27 git mv renames; GOLDEN IDENTICAL via scripts/tables/; resume settings 12/12 identical (no script paths in them); CPU LocalCluster probe OK for 6 moved dask runners; test_produced_by_paths_exist added; 2116+426 targeted passed. Implementer's calls: analyze_event_spike_evidence → detection/, refresh_contfrag_behavioral_covariates → tables/, _report.py split (detection half → detection/_report.py). 4 committed CSVs keep old produced_by rows (historical records).
Carry → final fix wave: the 5 PFC audits dropped out of the regression pipeline map (scripts/pfc not a mapped root) — add scripts/pfc as a mapped root.
Task 13: complete
Task 14: dispatched (BASE 3782844e, implementer a9b26761a4c6a9904)
Ruling: lead 2026-10-20 09:52 — if the chimi (s2) PFC runs are not done 80 min after their 08:42 start, kill them and move on; Task 14 then rests on j16 bit-identity (BASE-a = BASE-b = move = decomp) + targeted PFC tests — cost if wrong: a chimi-specific divergence in the split goes unseen.
Task 14: DONE_WITH_CONCERNS (26844eea pure move, 779c08c7 decomposition): both sessions (j16 06_r3, chimi 06_r3) bit-identical across BASE-a, BASE-b, move, decomp (neuron_stats + 5 modulation arrays); chimi runs finished 09:59–10:01 before the 10:02 cut, nothing killed; 478 PFC/guard tests passed. git does not detect the glm.py→session.py rename (<50% similarity). Controller removed the 3 scratch worktrees (base/move/decomp; decomp's edits are committed as 779c08c7).
Task 14: complete
Task 15+16: dispatched together (BASE 779c08c7, implementer ad79a9f79a0eac20b)
Ruling: Tasks 15 and 16 go to one implementer, one after the other — both are small pure moves (session_tables split, types.py rename) on disjoint files — cost if wrong: none.
Task 15+16: implementer stopped on a usage limit (2026-10-20 ~10:50); Task 15 committed (7783e97a, 6de3c467); Task 16 rename uncommitted in tree; resumed 2026-10-22 after reset
Task 15: DONE (7783e97a git mv → package, 6de3c467 split into _common/responses/arm_covariates/patch_covariates/stop_covariates): 92 top-level defs byte-identical; 2-session renders (49 panels + covariate figure) pixel-identical BASE vs BASE vs commit1 vs commit2; 162 passed.
Task 16: DONE (994417dd types.py → schemas.py): 46 codemod hits + 3 hand; stdlib-shadow test fails at BASE, passes now; 2075 passed; GOLDEN IDENTICAL; 0/18,232 pickles reference continuum_swr_replay.types (positive control found the 706 memoryless pickles). Controller fixed CLAUDE.md:86 tree line (path-only edit the plan allows).
FLAG FOR LEAD (pre-existing bug, not fixed): visualization session_tables trial_rail panel raises ValueError on real data — renumbers trials from 0 while real trial numbers start at 1, so they mismatch the port-experience stream.
Note: runs/io.py, tables/io.py, state_evidence/io.py, memoryless_sequence/statistics.py also carry stdlib names (only bite when a tool runs from inside their directory) — not changed.
Task 15: complete
Task 16: complete
Final review: dispatched (470a393..HEAD, opus); full suites (default + -m real_data) started
Final review: With fixes — 0 Critical; Phase C verified by AST (T14 49 defs verbatim, T15 92 verbatim, T16 3 doctest lines); KNOWN_VIOLATIONS empty for all 6 rules; no old paths outside documented frozen strings (final-review.md). Important: I1 scripts/pfc missing from pipeline map ROOTS; I2 no doc states the layering rule, no old→new table for ~40 moved src modules. Minor: M1 3 ruff import-order errors from T16; M8 swr_inventory *_dir() docstrings wrong about patching; M3 confusing near-duplicate names.
Ruling: final wave = I1, I2, M1, M8. I2 lands as a "Code layout" section + moved-modules table in docs/README.md, plus ONE pointer line in CLAUDE.md (not a new paragraph — CLAUDE.md is the lead's instruction file) — cost if wrong: the lead rewords one line.
Ruling: M3 renames (swr_inventory vs swr_inventories, schemas.py vs table schemas) deferred to the lead — renaming is churn with no correctness gain — cost if wrong: names stay confusing until a follow-up.
Ruling: trial_rail bug → its own branch with a failing test first (not in this branch) — reviewer confirmed on real data; pre-existing — cost if wrong: the panel stays broken on real data until then.
Final review: fix wave waits for the default suite to finish
Final: default suite at 6e511d39 — 7 known failures only, 5287 passed, 19 skipped, 23:56
Final fix wave: dispatched (BASE 6e511d39)
Final fix wave: DONE (10d9db09, 600d94a7, c7f97e8b, a3883ef7, c8ed8c23): pipeline map has the 5 PFC audits again (identical to 8f690706); ruff clean as master; docs/README.md Code layout section + old→new table (24 R/D src paths, script-checked); CLAUDE.md +1 pointer line; 179 guard tests passed.
Ruling: no scoped re-review of the final fix wave — docs, import order and map roots only, all test-pinned; lead prefers fewer reviews — cost if wrong: a doc wording issue.
Final: -m real_data suite at 6e511d39 — 19 passed, 0 failed, 2:09:44 (scratchpad/final_org/pytest_realdata_6e511d39.txt)
Final: branch complete, not merged — merge into master awaits the lead's approval

## Phase D (2026-10-23)

Phases A–C and the final fix wave: complete — see .claude/docs/plans/code-organization/ledger.md (archived, through a0cf6ab1).
Phase D: lead 2026-10-23 "Do it in this branch" — goal: an outside scientist can understand the analysis by reading the code.
Ruling: keep runs/io.py, tables/io.py, state_evidence/io.py — io is preloaded at startup so they cannot shadow (final review M10) and package.io is a common pattern — cost if wrong: one more rename later.
Ruling: keep memoryless_sequence/statistics.py — renaming breaks the 706 memoryless candidate pickles (Q5) — cost if wrong: it shadows statistics when run from inside its directory, now pinned by an allowlist.
Ruling: names — events/swr_inventory→swr_source, hippocampus/swr_inventories→swr_detectors, events/detection→nonlocal_detection, schemas→data_types, adapters.swr_inventory()→source_swr_inventory() — each says what the module does — cost if wrong: one more rename.
Ruling: Tasks 17 and 18 go to one implementer, one after the other — disjoint except docs/README.md — cost if wrong: none.
Task 17+18: dispatched (BASE be88764e, implementer a2a88dcb920738c7c, opus)
Lead 2026-10-23 04:15: "You can move the files in this case" — controller runs the Task 17 mv list (Processed-Data, this move only) after Task 17 is committed and the list is checked; implementer still never writes into Processed-Data/.
Task 17+18: DONE_WITH_CONCERNS (41bdfaed T17; c10223b2, fca8c683, 14f69a86, 9ec5f367, a02b2049 renames; 4691953d shadow test; 15dbe9a7 docs): GOLDEN IDENTICAL after T17 and at HEAD; 10,351-pickle byte scan: no old path referenced (control 706); full suite 7 known failures, 5,291 passed, 19 skipped.
Ruling (corrects the earlier keep): rename memoryless_sequence/statistics.py → window_statistics.py — the 706 candidate pickles reference only .config and .detect (implementer's scan); the earlier "pinned by Q5" was wrong — cost if wrong: none found; a pickle outside the scan would fail to load loudly.
Disk move (controller, lead-approved): canonical root + 6 staging roots — qvals_table.parquet + qvals_standardization_reference.json → root; canonical swr_power_* (4 files) → models_per_event_swr_power/. sha256 + mtime identical (scratchpad/phaseD/disk_move.log). replay_content_regen_staging_pre_v4_2333f07 left as a frozen snapshot.
Ruling: relative symlinks left at every old path — covariate-explorer, phase3, phase4-run worktrees share Processed-Data and run code that reads the old paths; a symlink is the same file, so no stale copy — cost if wrong: 18 symlinks to remove later. Remove once those branches are on the merged master.
After move: check_fit_provenance on the canonical root 93/93 fresh with new code.
Task 18b: DONE (2e55b21b statistics → window_statistics; 6549f9b7 docs range): 737 targeted passed; not on the table path.
Task 17: complete
Task 18: complete
Phase D review: dispatched (be88764e..ee0f9ca1, opus, ac2be43c33af07f26)
Phase D review: With fixes — 0 Critical; I1 provenance/manifest/freeze tools see old layout via symlinks; I2 an old-branch write (os.replace) silently breaks a symlink; I3 source_swr_inventory misnames a reader of any detector; M1–M5 docs/tests; M6 swr_power script lives in scripts/rate (noted) (phase-D-review.md)
Ruling: I1 fix — canonical table names resolve at the root only in check_fit_provenance; write_regeneration_manifest and freeze_regression_baseline skip symlinks; tests for each — cost if wrong: none.
Ruling: I2 — keep the symlinks (old worktrees still read them) and add the hazard + a must-print-nothing find check to the PLAN cleanup entry and docs/README "Where results live" — a guard in new code against old-branch writes would be a special case for a transitional state — cost if wrong: an old-branch rebuild leaves a stale root copy until someone runs the check.
Ruling: I3 — rename to swr_detector_inventory; the earlier name was the controller's ruling, not the lead's, and "source" means Kay in this package — cost if wrong: one more rename.
Ruling: M1–M5 fixed in the wave; M6 flagged for the lead (moving scripts/rate/fit_swr_power_model.py changes its pipeline id) — cost if wrong: a per-event model stays under scripts/rate/.
Phase D fix wave: dispatched to the implementer (resume)
Phase D fix wave: DONE_WITH_CONCERNS (42240716 I1, 81d035af I3, 7c9f9f0f I2+M1–M4, 294a9e89 M5): 420 targeted passed, ruff clean; no GOLDEN rerun (nothing on the table path changed behaviour).
Ruling: accept the implementer's I2 check (loop over the 7 symlinked roots, -maxdepth 1 ! -type l) over the controller's whole-tree find — the whole-tree find hits 53 real files in backups/golden baselines by design and could never print nothing — cost if wrong: none; the cleanup delete still matches exactly the 18 symlinks (checked).
Ruling: no scoped re-review of the fix wave — each fix test-pinned, lead prefers fewer reviews — cost if wrong: a doc wording issue.
Phase D: default suite at 294a9e89 — 7 known failures only, 5,296 passed, 19 skipped, 29:23. -m real_data not rerun: Phase D changes are renames (pure, imports exercised at collection, which imports the real_data test modules too) and output paths now pinned by unit tests.
Phase D: complete. Branch complete, not merged — merge awaits the lead.
