# Phase 4 execution plan (task-structured view of phase-4-docs-claims.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-4-docs-claims.md`, `overview.md`
(decisions: inference = 5-animal sign consistency + animal bootstrap; fractions shipped; every
content claim names its estimand — direct/length-adjusted vs total — and its estimator), and the
review `.claude/docs/reviews/2026-09-03-regression-pipeline-review.md` (finding D6 and the numeric
claims phases 1–3 changed). Where this file and those differ, they win.

**The only sources of numbers** (nothing else may be quoted; every number in a changed doc must
trace to one of these files, and the reviewer will spot-check ten):

- `docs/audits/2026-09-04-rate-refit-comparison.md` (phase 1) and the phase-3 rate audit
  `docs/audits/<date>-rate-refit-arrival-clock.md` (phase 3 supersedes phase 1 wherever both report
  the same rung).
- `docs/audits/2026-09-05-content-refit-comparison.md` (phase 2a, regenerated during the fix wave
  with the `nt` twins), `docs/audits/2026-09-05-cs5-response-by-design-factorial.md`.
- `docs/design/content_estimator_comparison.md` (phase 2b; the DM concentration result, the
  three-estimator CS5 table, the length-adjusted twin).
- The audited pickles themselves, read-only, ONLY to fill a value the audits do not print
  (`Processed-Data/replay_content/models/swr_contrast_content_audited{,_dm,_dm_len}.pkl`,
  `models_per_bin_rate/all_models.pkl`, `models_per_stop_rate/`), quoted with the artifact path.

## Global Constraints

- Docs-only phase: no code change except the two small scripts and the two tests named below.
  No fit, no regeneration.
- `paper/claims_table.md` IS edited in this phase (rows 19a–19g and the 3b framing row); keep every
  existing RETRACTED annotation and add new ones rather than deleting history.
- Every content claim names (i) the response (per-event fraction, shipped), (ii) the estimand
  (direct = `BASEc` rungs with the event-length control; total = `nt` twins), and (iii) the
  estimator where the DM comparison differs. Numbers: animal-balanced median, animals positive
  of 5, session sign-rank FDR q; bootstrap interval where the source has one.
- Retired names never appear outside the allowlist (`docs/design/model_history.md`,
  `docs/design/2026-08-21-covariate-naming-scheme.md`, the registry): `is_last_rewarded_elsewhere`,
  `swr_x_is_last_rewarded_elsewhere`, `is_after_omission`, `swr_x_q_arm`, bare `arm_uncertainty`,
  `recent_switch_rate`, `reward_lag1`, `q_arm`.
- No plan/phase/task references in shipped docs, tests, scripts, commit messages.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` then
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`.
- Do not run `git checkout` / `git stash` / `git clean`; do not sweep in untracked user files.

### Task 1: `docs/models_table.md` rebuilt from `model_specs.py`, with a mirror test

Files: `docs/models_table.md` (whole file; sections at ~9 Conventions, ~81 causal-hygiene, ~110
cohort-audit tables, ~170 per-bin rate, ~200 per-stop rate, ~208 content-arm, ~272 content-patch,
~439 C/CS contrast ladders), new `docs/design/cohort_and_timing_rules.md`, new
`tests/unit/test_models_table_mirrors_specs.py`, `src/continuum_swr_replay/analysis/replay/model_specs.py`
(read-only: `MODEL_PER_BIN_RATE` incl. `M0a_const_arrival`/`M5a_time_arrival`/`M7ra_time_arrival_histcos_rew`,
`MODEL_PER_STOP_RATE`, `MODEL_HIERARCHY` + `Mv/Mu/Mvu`, `MODEL_PATCH`, `SWR_CONTRAST_CONTENT_ARM_LADDER`
incl. the `nt` twins and `CS5b`, `_PATCH_LADDER`, `LEAF_ONLY_CONTRAST_RUNGS`, the per-bin
`is_swr`-contrast ladders).

- Rebuild the catalog so every executable ladder's rung set appears with its R-notation formula
  and key term; add the arrival-clock rungs (with the arrival spline spelled out and the
  window/knots), `Mu`/`Mvu`, the leaf-only chain (`models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu`), the
  `nt` twins, the DM estimator row and `_dm`/`_dm_len` bundles (already partly present — verify and
  complete). Document the `nt`/`b`/`r`/`a` suffixes and the `_hard_count`/`_leaf`/`_dm` bundle
  suffixes in "Conventions". Move the "Covariate causal-hygiene" and "Cohort-audit tables" prose
  to `docs/design/cohort_and_timing_rules.md`, leaving links.
- `test_models_table_mirrors_specs.py`: parse the R-notation code blocks (or a machine-readable
  rung list per family — choose one format and document it at the top of the doc) for rung names
  and assert, per family, set equality with the executable ladder's keys. It must fail today for
  the missing arrival rungs before the doc is fixed (state that in the report).

### Task 2: Rate docs from the refits

Files: `docs/rate_models.md` (Family A ~22, Family B ~83, model checking ~241, caveats ~369),
`docs/per_bin_rate_results.md` (sections 1–6; date its header; state the cohort), the phase-1 and
phase-3 rate audits (sources).

- Replace every number with the refit values; state the cohort (218 sessions in the content
  tables; say what the rate cohort is from the audit); the reward-history kernel is now the
  three basis features (the per-lag rate-ratio list is replaced by the fitted-kernel figure
  produced in phase 1 — cite its path); the arrival-clock variant and the "Stops are immobility
  fragments" section (phase 3 writes the first version; verify it against the phase-3 audit and
  fix anything phase 3 left as "pending"); the A4 decision outcome (dummy rungs retired or kept,
  with the measured CV NaN share); the arrival ladder reads different fractions of each population
  (SWR ≈ 92 % vs non-SWR ≈ 68 % of onsets on the smoke — quote the corpus number from the phase-3
  audit).

### Task 3: Synthesis doc and the claims table

Files: `docs/replay_rate_content_swr_synthesis.md` (Layer 1 ~37, Layer 3 ~137, updates ~178–400,
Confirmed ~425, Hypotheses ~457, Caveats ~500), `paper/claims_table.md` (rows 19a–19g at ~51–57
and the 3b framing row ~50; also grep the whole file for retired names and stale numbers such as
`swr_x_q_arm −2.45`, `−1.03`, `is_last_rewarded_elsewhere −1.66`, `+0.98`, `+0.83`, the lag-1 RR
`1.42 / 3.10 / 1.08`).

- Synthesis: remove every retired name; disambiguate "M7" as `M7_time_hist_rew` /
  `M7r_time_histcos_rew` / `M7ra_time_arrival_histcos_rew`; replace numbers from the refits; add
  the estimator-comparison paragraph with a link to `content_estimator_comparison.md`; add the
  estimand paragraph (direct vs total; event length post-treatment; both shipped).
- Claims table, rows 19a–19g and 3b framing: canonical predictor names
  (`is_last_rewarded_noncurrent_arm`, `reward_belief_mean_candidate_arm_preoutcome`,
  `reward_belief_variance_candidate_arm_preoutcome_z`, `reward_fraction_candidate_arm_last10_visits`,
  …); restate the "immediate reward boost" and "lag 2–10 suppression" claims in terms of the three
  reward-history basis coefficients (feature 0 = immediate component); current rung names; every
  number replaced with the refit value or marked "pending" if the refit did not reproduce it;
  add the outcome-stratum and event-length-controlled results for CS1/CS5; the CS5 row must carry
  the estimand × estimator table (direct: fraction +0.24 3/5 q 0.12 | DM −0.32 0/5 | count −0.40
  1/5; total `nt`: fraction −0.65 4/5 neg q 6e-6 | DM −0.64 0/5 q 8e-10) verbatim from the sources
  and the sentence that the direct effect's sign is estimator-dependent; the CS6 row likewise
  (direct null; total +0.15…+0.18, 4–5/5); cite the DM concentration result for "SWR broadens"
  (`log_phi_is_swr` +1.80 total / +1.50 length-adjusted, 5/5, 100 % of sessions, ≈ 6× / ≈ 4.5×);
  the "SWR avoids last-rewarded" family stays REFUSED as unidentified (CS5b: n = 132, 94
  separated); 19g's `arm_uncertainty +0.83` becomes the shipped `Mu` fraction-response value from
  the content audit (or "pending" if that bundle is not in the audit — it is in the leaf bundle;
  read `models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/Mu.pkl` through the uncertainty reader's output
  if the audit lacks it, and say where the number came from). Keep the RETRACTED annotations; add
  a dated "REVISED 2026-09" note per row saying what changed and why (response, SEs, length
  control, estimator).

### Task 4: Paths, structure, code map, builder floor, task description, retired-name scan

Files: `docs/covariate_dictionary.md` (~13, ~90 paths → `covariates/history/`), `docs/README.md`
(~86 "content M0–M6 + Mv/Mu/Mvu / P0–P3 + Pv"; ~90 remove the missing plan link; ~93 verify the
`_dm` rows), `CLAUDE.md` (~162 → `analysis/replay/spatial_mapping.py` — verify the real path
first; ~215 → `tables_event.py`; ~216-217 delete the `label_event_*` alias sentence if the alias
is gone — check `tables_event.py`), `scripts/README.md` (add the `qa/` scripts:
`compare_fits.py`, `check_fit_provenance.py`, `audit_event_patch_fraction.py`,
`audit_stop_table_reward_history.py`, `freeze_regression_baseline.py`, `compare_to_golden_baseline.py`,
and `content/compare_content_estimators.py` if missing), delete
`.claude/docs/reviews/08_regression_pipeline_code_map.md` and replace with a symbol-anchored
`docs/regression_pipeline_map.md` generated by new `scripts/qa/write_pipeline_map.py` (`--check`
mode diffs the committed map; module → public function names, no line numbers),
`docs/design/builder_test_floor.md` (re-run `scripts/qa/mutation_check.py` on the `history/`
package; refresh numbers and file names; record the runtime), `docs/task_description.md` (~34-64
reward structure: the non-canonical contingency share → the shipped 99.2 % in {0.2, 0.5, 0.8}, or
explain the parse difference with a pointer to `qvals_table`), new
`tests/unit/test_docs_have_no_retired_names.py` (grep `docs/`, `paper/`, `CLAUDE.md`,
`scripts/README.md` for the retired names; allowlist as in Global Constraints; it must fail before
Task 3 lands and pass after — run it first and record the hit count, currently ~15).

## Review

Before the final review, confirm: every number in the changed docs traces to a listed source
(spot-check ten, including both CS5 estimand rows and the concentration numbers); no retired name
outside the allowlist; both new tests and the `--check` mode pass; the mirror test fails on the
pre-fix doc (recorded in the Task 1 report); RETRACTED annotations preserved; no plan/phase
references; `CLAUDE.md` paths verified against the tree.
