# Phase 2a ledger (archived from .superpowers/sdd/phase-2a-execution/progress.md on 2026-09-05)

# SDD ledger — plan: .claude/docs/plans/regression-pipeline-fixes/phase-2a-execution.md

Branch: regression-pipeline-fixes-phase-2a (off master @ 6244433, phase 1 merged)
Spec authority: phase-2a-content-fractions.md, shared-contracts.md, designs.md, overview.md decisions, review 2026-09-03.
Models: implementers + task reviewers = opus; final whole-branch review = fable (crucial junction only).
Isolation: branch in place (user choice). Processed-Data shared; backups before any write.

## Pre-flight scan

| pair / task | produces vs consumes | finding |
| --- | --- | --- |
| T1 -> T3 | T1 persists patch_fraction; T3's --response fraction resolves the patch response to it | T3 must not run before T1's promotion; unit tests use synthetic frames |
| T2 -> T3 | T2 adds bse_cluster/estimator to fitter output; T3 records estimator in frames | consistent |
| T2 -> T4 | T4 z-scoring via attach_standardized; independent of T2 | independent |
| T3 -> T5 | T5's --leaf-only composes with --response and the canonical_output_subdir keying | consistent; T5 must extend canonical_output_subdirs so `_leaf` variants are guarded |
| T4 BASEc change -> T6 | every C/CS rung gains 3 columns; T6 refits all content contrasts | consistent |
| T1 regen | full --from-cache rebuild (~1h41) to promote one column | accepted cost; parity of the four other tables re-verified for free |
| T3 reporting | count-response headline drops lrt_significant/child_better_aic; fraction runs print them as event-scale | quasi-likelihood caveat still applies to fractions — reporting must say "event-scale quasi-AIC" not "exact" (Ruling 1) |
| Phase-1 inheritance | effect-scale backstop in aggregate_contrast now applies to content on refit (fitting.py persists sd_) | intended (final-review out-of-scope note); T6's comparison must report how many content drops moved |
| Phase-1 inheritance | plot_swr_contrast_coefs.py filters content at raw |coef|<10 (display) | T3 reader-gating touches that file; align the display filter with the aggregation rule there (Ruling 2) |

Ruling 1: fraction-response ΔAIC/LRT are quasi-likelihood quantities; docs and printed headlines label them "event-scale (quasi)". Cost if wrong: a label.
Ruling 2: when T3 gates plot_swr_contrast_coefs.py's content half, its raw |coef|<10 display filter is replaced by the aggregation's rule (separated_terms or |coef|*sd >= 10 when sd_ exists). Cost if wrong: a figure shows a few more sessions.
Ruling 3 (scope): no worktree — implementers confined to their file lists; no checkout/stash/clean.

## Task log
Task 1: BASE=31d5e66 dispatched (opus)
Task 1: code committed (173334a); full regen running in tmux patch_table_regen (ETA ~12:45); implementer idle, to be resumed on completion. Monitor armed.
Task 2: BASE=173334a dispatched (opus) — disjoint files from Task 1's remaining work (promotion/audit)
Task 2: implementer DONE (c146652). Ruling 4: bse_cluster_<term> is emitted for every conditional Poisson fit (counts and fractions) — the contract's "fractions only" line is relaxed; shared-contracts.md wording to follow in Task 6's docs. Cost if wrong: an extra column on count fits.
Task 2: review dispatched (opus). Working tree carries Task 1's uncommitted audit script (expected).
Task 2: review clean (approved; 0 Important). minors (deferred): se_source majority label; finiteness not positivity gate on clustered SE; pct_sig_session still model-based; unsorted groups untested; soft/posterior responses labeled count.
Task 2: complete (commits 173334a..c146652, review clean)
Task 3: BASE=c146652 dispatched (opus) — unit-level only until Task 1 promotes the patch table
Task 3: implementer DONE (593d82d); found swr-content _run dropped sd_/separated_terms (backstop inert on content until now); review dispatched (opus)
Task 3: review clean (approved; 0 Important). minors (deferred): fit_model_per_session default response_col still arm_count_hard; analyze_uncertainty_content gates after 3 table loads; human-readable provenance JSON lacks --response; content runner excluded from the AST read_table drift test; git shelled twice per artifact.
Task 3: complete (commits c146652..593d82d, review clean)
Task 4: BASE=593d82d dispatched (opus)
Task 4: implementer DONE (db7fbb1); review dispatched (opus)
Task 4: review — spec ✅; 1 Important (sd_arm computed after the stratum filter), 6 Minor.
Task 4: minors (deferred): product names typed twice without a guard (fixing in round 1); rung-selection test skips non-BASEc rungs; is_swr NA->True cast; standardization.py docstring says fit-time _z is elsewhere; n_nonlocal_bins is post-treatment w.r.t. is_swr (nt twins are the sensitivity); omitted stratum ~39 events/session vs 18 params.
Task 4: fix round 1/5 dispatched (resume implementer)
Task 4: fix round 1 implemented (365ed6f); re-review dispatched (opus)
Task 1: implementer DONE (173334a, 20c8ff2); patch table promoted dff5b2b…; review dispatched (opus) on the two Task-1 commits
Task 4: fix round 1/5 (3 addressed, 0 open; commits db7fbb1..365ed6f)
Task 4: Ruling 5: the runner's per-SD scale reference (_ARM_SD_TERMS) is computed over leaf arms only (candidate_arm < 6), matching the persisted-table leaf support and the leaf-only fitting support of the value rungs; trunk structural zeros must not set the unit. Applied as a pre-refit step in Task 6. Cost if wrong: std_* columns change scale; raw coefficients unaffected.
Task 4: minors (deferred): is_last_rewarded_noncurrent_arm missing from BINARY_TERMS (label); task_reward_probability_candidate_arm dead entry in _ARM_SD_TERMS.
Task 4: complete (commits 593d82d..365ed6f, review clean after 1 fix round)
Task 5: BASE=20c8ff2 dispatched (opus)
Task 1: review clean (approved; 0 Important). minors (deferred): patch fraction sum tolerance shares the 1e-5 posterior constant (three tolerances exist for one estimand); patch row with an event_id absent from event_table degrades to NaN silently; arm_fraction lacks a catalog entry.
Task 1: complete (commits 173334a, 20c8ff2; patch table promoted dff5b2b…, backup ceb3579… in _pre_fix_backup_20260905/)
Task 5: implementer DONE (ac4f8e9). Concern: leaf-only fraction fits leave event mass = 1 − trunk fraction, downweighting trunk-heavy (SWR) events.
Task 5: Ruling 6: in every leaf-only fit with the fraction response, the response is renormalized over the retained candidate set (sums to 1 over six leaves) via one shared helper in content_response.py used by both content runners; events with zero leaf mass are dropped as zero-response and audited; hard_count leaf-only fits are unchanged. Cost if wrong: leaf-only coefficients describe leaf-conditional allocation (which is the estimand anyway).
Task 5: pre-review fix dispatched (resume implementer)
Task 5: ruling implemented (b7cdd89); review dispatched (opus) on 20c8ff2..b7cdd89
Task 5: review — spec ✅ incl. ruling; 1 Important (SWR runner books zero-leaf-mass events as own-predictor-missing), 5 Minor.
Task 5: minors (deferred): leaf filter duplicated across runners; lambda groupby aggregations; leaf-only cohort not a subset of the nine-arm one (document with ΔAIC non-comparability).
Task 5: fix round 1/5 dispatched (resume implementer) — audit bucket; meta from fitted rows; extra-clash test
Task 5: fix round 1 implemented (2a0f42e); re-review dispatched (opus)
Task 5: re-review clean (fix round 1, 2a0f42e) — all 3 findings fixed, no new breakage; suite 1863 passed
Task 5: complete
Task 6: dispatching implementer (opus); BASE = 2a0f42e; pre-step = Ruling 5 (_ARM_SD_TERMS SD reference over leaf arms only)
Note: drafted .claude/docs/plans/regression-pipeline-fixes/phase-2b-execution.md (uncommitted; commit as first step of the phase-2b branch, not on 2a)
Task 6: implementer DONE_WITH_CONCERNS (1197b52 dbdef81 b5399bd b4d7230); CS5_armvalue flips -0.996 (0/5) -> +0.237 (3/5, q=0.12), CS6 collapses; omitted-stratum 87% rewarded verified plausible (stop-level 61% rewarded, event-level 87% = reward gating); review dispatched (opus)
Task 6: review spec ✅ / quality needs fixes — 1 Important (audit std_coef columns span two per-SD units after Ruling 5) + 7 minors; fix round 1 dispatched (resume implementer)
Diagnostic (controller, scratch $SP/cs5_factorial): CS5 swr_x_reward_belief_mean_candidate_arm_preoutcome, session-median coef / n_animals_pos over the shipped at-well leaf-only cohort:
  count NT(old BASEc) -0.98 0/5 | count +TSC -1.03 0/5 | count full(+length products) -0.40 1/5 | fraction NT -0.55 1/5 | fraction +TSC -0.52 0/5 | fraction full -0.11 3/5
  → the event-length control and the fraction response each remove ~half the old effect independently; together it is gone. _TSC alone changes nothing.
Task 6: fix round 1 implemented (f467a6f ae3450c); suite 1888 passed; re-review dispatched (opus)
Controller commit 0bda277: CS5 response x design factorial (scripts/exploratory/_cs5_response_by_design_factorial.py + docs/audits/2026-09-05-cs5-response-by-design-factorial.{md,csv}); to be covered by the final whole-branch review
Task 6: re-review clean (f467a6f ae3450c); 2 new minors carried to the final fix wave: _disjoint_rungs_note hard-codes the cause sentence + header prose hard-codes 0.2493/0.1190; _suffixed_side empty guard can drop an empty stratum silently from the stratum table/power note
Task 6: complete
Final whole-branch review: dispatched (fable), base master 6244433, head 0bda277
Final review (fable): merge-ready NO pending one fix wave. Important: I1 analyze_uncertainty_content.py reads models/Mu.pkl (now in leaf bundle) → dead path; I2 plot_swr_contrast_coefs._global_sd recomputes per-SD over 9 arms (0.2493) vs audited leaf SD (0.1190), 2.1x off; S1 docs must state CS ladder = length-adjusted (direct) SWR effect, length is post-treatment wrt is_swr, report CS and nt twins together. Minors: I3 freeze_regression_baseline dirs; I4 ΔAIC axis label scale; I5 pct_sig_session uses model pval vs cluster SE; S3 quote 12.3% zero-leaf-mass drop + SWR skew in content_model_design.md; compare_fits hard-coded cause/0.2493/0.1190 + _suffixed_side empty stratum; 8 format-debt lines on branch-added code. Suite 1888 passed. Fix wave dispatched (fresh opus implementer).
Ruling 7: S1 is resolved as a documentation/estimand statement (CS = direct effect net of length; nt = total effect), NOT by changing BASEc — deciding which is the science is the user's call; both are shipped. Cost if wrong: a doc rewrite.
Ruling 8: add `nt` twins for CS4_armreward, CS5_armvalue, CS6_armuncert (BASEc_NT base, same key term, same leaf-only rule) so the audited table carries the total (uncontrolled) and direct (length-adjusted) SWR effects side by side; refit ONLY fit_swr_contrast_content.py (3 strata) inside the fix wave — ~1 h compute, ladders untouched. Cost if wrong: one contrast refit + backup churn.
Fix wave: BASE 0bda277
Fix wave: DONE (434e3d3 a039527 db483c7 700b38e 9a7c856 edacbd3 f19ee12 a421740 7201b5b a15e36c); suite 1902 passed; contrast refit with nt twins: CS4 -1.20 0/5 | CS4nt -1.33 0/5; CS5 +0.24 3/5 q=.117 | CS5nt -0.65 1/5 q=5.6e-6; CS6 +0.004 3/5 | CS6nt +0.18 5/5 q=6e-15. Backup _pre_nt_twins_backup_20260905. Scoped re-review dispatched (opus).
Fix wave: re-review clean (a15e36c); suite 1902 passed; 3 minors noted (pct_significant denominator over usable sessions; sys.path sibling import / basicConfig ordering — move naming helpers into src/ if a 4th consumer appears; design-delta table not twin-adjacent). Phase 2a: COMPLETE — archived to .claude/docs/plans/regression-pipeline-fixes/phase-2a-ledger.md, merged to master.
