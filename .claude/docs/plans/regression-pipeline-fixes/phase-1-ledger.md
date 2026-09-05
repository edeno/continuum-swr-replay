# Phase 1 execution ledger (rulings, review outcomes, deferred minors)

Branch: regression-pipeline-fixes-phase-1 (off collapse-covariate-boundary @ 53fad0c)
Plan commits: 2aebc36 (review + plan), 1ef9230 (execution view)
Spec authority: phase-1-rate-runners.md, shared-contracts.md, designs.md, review 2026-09-03.
Models: implementers + task reviewers = opus; fix rounds 4-5 = opus (fresh agent);
final whole-branch review = fable (user: Fable only at crucial junctions).
Isolation: branch in place (user choice); Processed-Data is shared real data — backups before any write.

## Pre-flight scan

| pair / task | produces vs consumes | finding |
| --- | --- | --- |
| T1 -> T2 | T1 adds 4 stop columns + catalog; T2 regenerates the table | consistent; T2 must not run before T1 merges to the working tree |
| T1 -> T3 | T1 defines REWARD_HISTORY_* constants in model_specs; T3 builds formulas from them | consistent |
| T2 -> T4 | T4's runner assertion "freeze drops zero stops" needs the regenerated stop table | consistent only after T2; unit tests use synthetic designs |
| T3 -> T4/T7 | T3 updates fixtures with basis columns; T4/T7 tests rely on them | consistent |
| T3 -> T5 | T5's aggregation test uses coef/bse/separated only | independent |
| T4 xfail removal -> T8 | the real-data cohort test fails on the OLD all_models.pkl until T8 refits | CONFLICT (see Ruling 1) |
| T6 -> T8 | T8 needs the provenance writer in the runners | consistent |
| T5 output-dir keying -> phase 2a | canonical models/ currently holds a Mu/Mv/Mvu subset; phase 2a restores | out of phase scope; noted |
| T2 regen path | regenerate_stop_table.py imports load_data (may need DB); prepare_data --stop-only refuses to clobber a complete table | T2 brief already lists the fallback (full --from-cache into staging, promote stop table only) |
| T3 self-consistency | MU_UNCERTAINTY is constructed positionally in fit_uncertainty_rate.py; replacing reward_lags with reward_basis changes the positional slot | brief tells the implementer to switch to keyword construction |
| T3 self-consistency | fit_per_bin_rate_model's lag-1 contrast guard must move to rcos_0 | in brief |
| T8 | long-running; tmux; backups first | in brief |

Ruling 1: `test_rate_ladder_rungs_share_a_cohort` (xfail removed in T4) reads the live
`all_models.pkl`, which is the pre-fix artifact until T8. T4 marks it `pytest.mark.slow`; unit runs
in T4-T7 use `-m "not slow"`; T8 runs it explicitly and reports. Cost if wrong: a slow-marked test
that nobody runs — mitigated by T8's explicit run. (Check: does pyproject define a `slow` marker?
see pre-flight output below.)

Ruling 2 (scope): no worktree — implementers must not touch files outside their task's file list,
and must not run `git checkout`/`git stash`/`git clean` (the user's untracked files live in this tree).

## Task log
Baseline: 1631 passed, 1 xfailed (2:40) at 1ef9230. slow marker exists in pyproject (Ruling 1 stands).
Task 1: BASE=1ef9230 dispatched (opus)
Task 1: implementer DONE (4fda746); review dispatched (opus)
Task 1: review clean (spec ✅, approved). ⚠️ resolved: 4 warnings in full run vs 20 at baseline — fewer, not new.
Task 1: minor (deferred): COVARIATE_SCHEMA_VERSION not bumped for the column-set change (readers fail loud via columns=; note in T2 audit)
Task 1: minor (deferred): definitions.py window strings hard-code "10" instead of REWARD_HISTORY_N_LAGS
Task 1: minor (deferred): build_stop_table computes the lag matrix twice (N_REWARD_LAGS vs REWARD_HISTORY_N_LAGS both 10)
Task 1: minor (deferred): test (i) restates the implementation formula; discrimination carried by (ii),(iii),(vi)
Task 1: minor (deferred): "support" = nonzero weight in code vs ">0.05" in prose; relies on contiguous known lags — add a docstring note
Task 1: minor (deferred): no tests for empty inputs / non-default n_lags, n_basis
Task 1: complete (commits 1ef9230..4fda746, review clean)
Task 2: BASE=4fda746 dispatched (opus)
Task 2: implementer DONE (5091169); review dispatched (opus)
Task 2: review — spec ✅ except undocumented gate breakage; 2 Important, 4 Minor.
Task 2: Ruling 3: fix the golden-baseline gate by adding `expected_added_columns` to table_parity.diff_table (+ CLI flag) rather than re-freezing the baseline — keeps the pre-fix baseline usable for later phases; src/table_parity.py edit allowed for this round despite the brief's no-src rule. Cost if wrong: a re-freeze is still cheap later.
Task 2: minor (deferred): per-animal 0.4–2.5% band asserted without source (audit script)
Task 2: minor (deferred): incomplete_share prints counts as floats
Task 2: fix round 1/5 dispatched (resume implementer) — findings: gate hard-fails; parity script not reusable; float-cast bug in table_parity; promotion command missing from audit doc
Task 2: fix round 1 implemented (3ab6ea8); re-review dispatched (opus)
Task 2: fix round 1/5 (4 addressed, 0 open; commits 5091169..3ab6ea8)
Task 2: minor (deferred): audit script no longer asserts dtype equality of pre-existing columns (diff_table compares values only)
Task 2: minor (deferred): compare_to_golden_baseline.py logs only the drift message when both flags given; comma-splitting inconsistent between the two flags
Task 2: minor (deferred): no Float64/pd.NA test on the float path of diff_table
Task 2: complete (commits 4fda746..3ab6ea8, review clean after 1 fix round). Stop table promoted b261d993…; backup _pre_fix_backup_20260904/.
Task 3: BASE=3ab6ea8 dispatched (opus)
Task 3: implementer DONE_WITH_CONCERNS (9f3c4ef): stale fits (T8), dead reward_rate toggle, std_factor label for rcos_0, weaker identifiability guard (per brief), normalized kernel reconstruction; review dispatched (opus)
Task 3: review — spec ✅ rung by rung; 2 Important, 6 Minor.
Task 3: Ruling 4: std_unit for reward_history_rcos_* (and is_swr: products) becomes a third label "per_unit" via a PER_UNIT_TERMS frozenset in contrast_aggregation.std_factor (factor 1.0); fixed in this task rather than deferred to T5. Cost if wrong: a label only.
Task 3: minor (deferred): kernel reconstruction assumes complete history (docstring caveat)
Task 3: minor (deferred): median-of-exp vs exp-of-median disagree between plot and summary
Task 3: minor (deferred): paren-scan assertion in test_swr_contrast_specs_place_the_indicator_outside_the_interaction is near-vacuous
Task 3: minor (deferred): aggregate_to_timebins Returns docstring omits reward_history_incomplete
Task 3: minor (deferred): reward_rate toggle live but unreachable
Task 3: minor (deferred): rate_models.md:147 reads as current spec; covariate_dictionary lacks rows for the four columns (T9 owns)
Task 3: fix round 1/5 dispatched (resume implementer) — findings: plot_swr_contrast_coefs.py consumer; per_unit label
Task 3: fix round 1 implemented (9170013); re-review dispatched (opus)
Task 3: fix round 1/5 (2 addressed, 0 open; commits 9f3c4ef..9170013)
Task 3: minor (deferred): PER_UNIT_TERMS covers is_swr: products but not swr_x_ alias (unreachable today)
Task 3: minor (deferred): fit_swr_contrast_content.py:165 dead merge of known_reward_outcome_lag_1; fit_uncertainty_rate.py:118 stale comment
Task 3: complete (commits 3ab6ea8..9170013, review clean after 1 fix round)
Task 4: BASE=9170013 dispatched (opus)
Task 4: implementer DONE (9456cbd); concerns: cross_validate_per_stop_rate_model still dropna (no-op), exploratory _covariate_ranking.py broken since T3; review dispatched (opus)
Task 4: review clean (spec ✅, approved; 0 Important)
Task 4: minor (deferred): zero-drop raise can print an empty offender dict when exposure<=0 drops a stop
Task 4: minor (deferred): cross_validate_per_stop_rate_model keeps its dropna (unreachable after the freeze)
Task 4: minor (deferred): contrast audit books G-union-only drops under n_excluded_session for S rungs
Task 4: minor (deferred): freeze_per_bin_cohort does not require stop_id completeness while CV does
Task 4: minor (deferred): slow marker not deselected by default; test_rate_ladder_rungs_share_a_cohort is red until T8 refit
Task 4: minor (deferred): per-stop freeze spans all outcome columns (spec-mandated)
Task 4: complete (commits 9170013..9456cbd, review clean)
Task 5: BASE=9456cbd dispatched (opus)
Task 5: implementer DONE (3419acf) with a correctness concern: post-outcome gate + union freeze would drop 37,122/105,359 stops (35%) from EVERY G/S rung.
Task 5: Ruling 5: the SWR-rate runner freezes two cohorts per (session, population): the PRIMARY cohort over the union of all non-descriptive rung columns (G0-G4, G6, G8; S0-S6, S8, S10), and for each descriptive rung (G5, G7, S7, S9) a POST-OUTCOME cohort = primary design re-frozen with that rung's post-outcome columns, on which the rung AND a companion null (G0_base / S0_rate) are fit so their ΔAIC shares rows. Rows carry `cohort` in {"primary","post_outcome"}. Descriptive rungs are declared associations, not nested ladder members, so this does not weaken the contract. Cost if wrong: extra runner branching; can be collapsed later.
Task 5: minor (deferred): partial-ladder fits in canonical models/ written under the old rule remain until phase 2a restores the base ladder.
Task 5: pre-review fix dispatched (resume implementer) to implement Ruling 5.
Task 5: Ruling 5 implemented (d8e690b). Ruling 5a (accepted deviations): one shared post-outcome cohort + one null refit per ladder (both post-outcome covariates are NaN on identical rows); n_in stays the uniform post-join denominator so cohort_record's single reconciliation holds. Cost if wrong: none scientifically.
Task 5: review dispatched (opus) on 9456cbd..d8e690b
Task 5: review — spec ✅ for 4 fixes + ruling; 2 Important, 5 Minor.
Task 5: Ruling 6: onset reconciliation counts only onsets whose bin survives the exposure filter (n_expected adds `min(tmax, immobility) - floor(t/bin_s)*bin_s > 1e-4`), and the count of onsets excluded that way is recorded in out.attrs["n_onsets_in_dropped_bins"] so the loss is visible, not silent. Cost if wrong: an onset in the last 0.1 ms of a stop is uncounted (0 on canonical tables).
Task 5: minor (deferred): plot_swr_contrast_coefs.py tolerates a missing cohort column (stale pickle) — make required after T8
Task 5: minor (deferred): POST_OUTCOME_RATE_RUNGS vs DESCRIPTIVE_RATE_RUNGS cross-reference
Task 5: minor (deferred): _fit_one_session 113 lines; duplicated skip-merge loop
Task 5: minor (deferred): output-dir keying uses given --models order, not ladder order
Task 5: minor (deferred): cross_history_lags path has binning fix but no count reconciliation
Task 5: fix round 1/5 dispatched (resume implementer) — findings: canonical-dir guard incomplete; reconciliation raises on sub-1e-4 exposure bins
Task 5: fix round 1 implemented (d710f2d); re-review dispatched (opus). n_in deviation already accepted under Ruling 5a.
Task 5: fix round 1/5 (2 addressed, 0 open; commits d8e690b..d710f2d)
Task 5: minor (deferred): n_onsets_in_dropped_bins is recorded in attrs but no runner logs it (pandas drops attrs through merges) — surface in per-bin runner logs
Task 5: minor (deferred): canonical-dir guard compares basenames only (intentional, documented)
Task 5: complete (commits 9456cbd..d710f2d, review clean after 1 pre-review ruling round + 1 fix round)
Task 6: BASE=d710f2d dispatched (opus)
Task 6: implementer DONE (c8505c1). INCIDENT: `fit_per_stop_rate_models.py --help` (no argparse) ran a full refit (25 s) and overwrote models_per_stop_rate/all_models.pkl (the 2026-09-02 19:03 fit) before Task 8's backup. Old artifact unrecoverable.
Task 6: Ruling 7: Task 8 uses Processed-Data/replay_content/_pre_v5_regression_refit_backup_20260902/models_per_stop_rate/all_models.pkl as the Family-A "before" artifact (one refit older than the lost one; same spec family), labelled as such in the comparison doc; Task 8 also gives the per-stop runner an argparse (--max-per-animal, --no-cv) so unknown flags error instead of refitting. Cost if wrong: the Family-A before/after is against an older baseline.
Task 6: review dispatched (opus)
Task 6: review — spec ✅; 2 Important (hash-at-write window; 4 ungated readers), 4 Minor.
Task 6: Ruling 8: inputs are hashed immediately after loading (writer accepts precomputed records) and the four remaining readers (plot_model_checking, plot_per_bin_rate_summary, plot_swr_contrast_coefs, plot_uncertainty_rate) get the gate in this task rather than phase 3. Cost if wrong: none; scope grows by ~40 lines.
Task 6: minor (deferred): two exception types for "not fresh" (RuntimeError vs FileNotFoundError)
Task 6: minor (deferred): reader tests execute scripts at module scope, creating paper/figures/rate
Task 6: minor (deferred): fitter docstrings do not mention the sidecar
Task 6: fix round 1/5 dispatched (resume implementer)
Task 6: fix round 1 implemented (619af83); re-review dispatched (opus)
Task 6: fix round 1/5 (3 addressed, 0 open; commits c8505c1..619af83)
Task 6: minor (deferred): plot_swr_contrast_coefs.py parses argv at module level (import-hostile; already drew at import)
Task 6: minor (deferred): plot_uncertainty_rate.UNCERTAINTY_RATE constant unused
Task 6: minor (deferred): write_fit_provenance does not validate record keys (fails safe as stale)
Task 6: minor (deferred): _module_constants evals top-level assignments in scanned scripts
Task 6: complete (commits d710f2d..619af83, review clean after 1 fix round)
Task 7: BASE=619af83 dispatched (opus)
Task 7: implementer DONE (bf30574); review dispatched (opus)
Task 7: review clean (approved). minor (deferred): recorder predict(exog=None) would TypeError; __getattr__ recursion under copy/pickle.
Task 7: complete (commits 619af83..bf30574, review clean)
Task 8: BASE=bf30574 dispatched (opus) — long-running refit in tmux
Task 8: implementer DONE (5f53d5b). Refits complete (A 32 s, B 35 min, SWR 19 min, MU 11 min); cohorts identical across rungs; kernel dissociation holds (lag-1-equivalent RR 1.43 vs 0.90, 5/5). Reward-block joint AIC collapsed from -109..-163 to -1.6..-5.0 (the A2 artefact confirmed). swr_contrast_rate_audited.pkl still 0 rows: 55% of fits flagged separated (S models 85-92%, G 45-58%).
Task 8: diagnosis (controller, read-only): huge SEs in flagged fits come from reward_history_incomplete (22% of flagged fits) and hist_rcos_* (6-9% each, doubled under is_swr*); flag is fit-level so one nuisance term discards a session for every term.
Task 8: Ruling 9: separation becomes term-level. fit_status returns separated_terms (names); nuisance terms (reward_history_incomplete, Intercept) are excluded from the judgment; runners persist separated_terms and sd_<term>; aggregate_contrast drops a session for term T only if T is separated (fit-level flag kept as a diagnostic count); per-bin CV fold invalidation likewise ignores nuisance terms. Requires refitting B, SWR-contrast, MU, A (~1 h). Scheduled as Task 8b after the Task 8 review. Cost if wrong: an hour of refits.
Task 8: review dispatched (opus)
Task 8: review — spec ✅ (all data claims re-derived exactly); 1 Important (rate_models.md:88-94 stale numbers), 5 Minor.
Task 8: Ruling 10: the kernel summary in summarize_per_bin_rate_models.py uses per-session reconstruction then the median (matching the figure), not median-of-coefficients; doc text describes that derivation. Cost if wrong: headline 1.43/0.90 becomes 1.38/0.91.
Task 8: minor (deferred): compare_fits.py old-rung step maps are module constants (limited reuse)
Task 8: minor (deferred): orphaned reward_kernel_per_bin_rate.png (untracked)
Task 8: fix round 1/5 dispatched (resume implementer) — findings: rate_models.md stale paragraph; summarizer/figure derivation mismatch; delta_aic pair-loss count; --max-per-animal 0 guard
Task 8: fix round 1 implemented (38eb072); re-review dispatched (opus)
Task 8: fix round 1/5 (4 addressed, 0 open; commits 5f53d5b..38eb072)
Task 8: minor (deferred): per_bin_rate_results.md:99 "~0.05 higher" overstated for non-SWR rows; leftover "lag>=2 suppression" wording vs "from lag 3"
Task 8: minor (deferred): compare_fits.reward_quantities KeyErrors on a pre-basis "new" artifact; n_unpaired=0 for absent rungs
Task 8: minor (deferred): reconstruct_reward_kernels drops NaN rows internally while the plotter re-keys by animal (aligned only via prior dropna)
Task 8: minor (deferred): fit_cross_history_rate.py --write clobbers canonical cross_history_models.pkl regardless of --max-per-animal
Task 8: complete (commits bf30574..38eb072, review clean after 1 fix round). Live rate fits carry provenance; kernel 1.38/1.18/0.91; audited SWR table still empty pending Task 10.
Task 10: BASE=38eb072 dispatched (opus) — term-level separation + second refit
Task 10: implementer DONE (7221bea). Audited SWR rows 0 -> 25 (8 G rungs x 3 pops + S1_rew); CV validity A 60%->92%, B M1_rew 24%->89%; coefficients bit-identical to Task 8. Open: S3-S9 products dropped because the is_swr main effect separates in 31-38% of fits and the brief's "either factor" rule inherits it; S0_rate's is_swr main effect itself exceeds the 20% gate (genuine per-session non-identification).
Task 10: Ruling 11 (to apply after review): an is_swr:<c> product is judged on its own coefficient only (its bse already reflects its identification); S0_rate's missing row is an honest non-identification, reported as such, not hacked. Aggregation must be re-runnable from the persisted per-session fits (--aggregate-only) so rule changes do not require refits. Cost if wrong: products reported where the main effect is unidentified — flagged in the audit by the fit-level count.
Task 10: review dispatched (opus)
Task 10: review clean (approved; 6 Minor). Bit-identity of 61 coef/aic/llf columns and the 31-38% is_swr separation confirmed.
Task 10: minor (deferred): choice_prediction computes predictor_sd but does not pass it (comment needed)
Task 10: minor (deferred): fit_status evaluates bse*scale on non-usable columns (spurious RuntimeWarning)
Task 10: minor (deferred): separated_for_term treats NaN separated_terms as not separated
Task 10: minor (deferred): report-only numbers (1.55% vs 1.6%; S10 at 67%)
Task 10: fix round 1/5 dispatched (resume implementer) — Ruling 11 (own-coefficient product rule, --aggregate-only) + term_diagnostics docstring + Family-A CV validity in audit doc
Task 10: fix round 1 implemented (70de16d): own-coefficient product rule; --aggregate-only; audited rows 25 -> 32 (S1, S3-S9 emit). Correction recorded: S0_rate's is_swr does not separate (0%); S0/S2/S10 have no key term by construction. Ruling 11's S0 clause was based on a wrong premise — no action needed.
Task 10: re-review dispatched (opus)
Task 10: fix round 1/5 (4 addressed, 0 open; commits 7221bea..70de16d). Audited rows 32; per-session pickle untouched; provenance verified.
Task 10: Ruling 12: docs/audits/2026-08-25-covariate-regen-staging-audit.md (the user's pre-existing untracked file) was swept into 70de16d; left in history rather than rewritten — surface to the user at finish (git rm --cached if unwanted). Cost if wrong: one unintended doc in the branch.
Task 10: minor (deferred): drops on S5-S9 rows come from the |coef|>=10 backstop, not the per-term flag; audit prose says "unidentified" — reword in Task 9; consider splitting the two mechanisms in emitted columns
Task 10: minor (deferred): --aggregate-only sidecar omits the qvals/stop tables that set std_factor
Task 10: minor (deferred): audited sidecar git hash is 7221bea-dirty (produced pre-commit)
Task 10: minor (deferred): _write_aggregation silently no-ops on an empty frame
Task 10: complete (commits 38eb072..70de16d, review clean after 1 fix round)
Task 9: BASE=70de16d dispatched (opus)
Task 9: implementer DONE (ebedfaf); concern: compare_fits.py still emits 'dropped as unidentified' and would overwrite the hand-reworded audit caption; review dispatched (opus)
Task 9: review — spec ✅; 2 Important (generator overwrite of hand prose; separation sentence misstates bse*sd>=1), 7 Minor.
Task 9: Ruling 13: compare_fits.py emits the two-mechanism prose and computes the per-mechanism drop decomposition from the per-session pickle, then the audit doc is regenerated so doc == generator (docs-only constraint lifted for that script). Cost if wrong: none.
Task 9: minor (deferred): bump supports/peaks cite the basis matrix not a results doc; dictionary lacks the regen-audit link; changelog voice in rate_models.md; HMM-era names in the MU block (pre-existing); parameter-count conventions at rate_models.md:306
Task 9: fix round 1/5 dispatched (resume implementer)
Task 9: fix round 1 implemented (4818179); re-review dispatched (opus). minor (deferred): aggregate_contrast docstring still says 'SE per SD'
Task 9: fix round 1/5 (4 addressed, 0 open; commits ebedfaf..4818179)
Task 9: minor (deferred): contrast_aggregation.py:163 and test_fit_diagnostics.py:254 keep the inverted "SE per SD" phrasing; audited_term_drop_table docstring says "or a factor of it"; aggregate_contrast hard-codes the literal 10; rate_models.md says "every runner" writes a sidecar (only the four rate fitters do)
Task 9: complete (commits 70de16d..4818179, review clean after 1 fix round)
ALL TASKS COMPLETE. Final whole-branch review dispatched (fable) on 53fad0c..4818179.
FINAL REVIEW (fable): With fixes. 0 Critical, 2 Important: (1) |coef|>=10 backstop in aggregate_contrast selects on the estimate (bse*sd <= 0.6 on every backstop-only drop); (2) plot_per_bin_rate_summary.py panel C reconstructs the kernel from median coefficients (rejected estimator). Ruling 12 (keep the swept audit doc) endorsed. All other rulings agreed.
Ruling 14: the backstop becomes effect-scale — a session is dropped for term T only if T is in separated_terms, or |coef_T| * sd_T >= 10 when sd_T is persisted; the legacy |coef| >= 10 applies only to frames without sd_ columns. Then --aggregate-only + regenerate the audit doc. Cost if wrong: a few more sessions retained in S8/G6 rows.
Final fix wave dispatched (opus): #1, #2, plus one-sentence doc items (S-ladder is_swr:rcos_1 collinearity note; M7 vs M7r invariance; "every runner" wording; R1 CV figure).
Final fix wave implemented (7d75a90, 642d497): effect-scale backstop, panel C, doc items; S8_uncert -0.27 -> -1.05 (n_dropped 32->0), G6/SWR -0.11 -> -0.71; scoped re-review dispatched (opus)
Final fix wave re-review: all addressed, no new Critical/Important. minor (deferred): aggregate_contrast has no zero-SD / NaN-sd guard (nuisance-only today); SEPARATION_COEF_ABS docstring covers one of its two uses; content-ladder aggregation will switch to the effect-scale backstop on its next refit (fitting.py persists sd_); two display filters still use raw |coef|<10 (fit_swr_contrast_content.py:470, plot_swr_contrast_coefs.py:179); G6_uncert/SWR animal median (-0.71) vs session median (+0.25) opposite signs; rate_models.md:291 "44 of 218" is the fit-level count.
PHASE 1 COMPLETE at 642d497. Awaiting the user's merge/PR decision.
