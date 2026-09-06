# Phase 13 — Clean production refit, promotion, and reporting cutover

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Run every corrected canonical analysis from regenerated tables, validate all identities/diagnostics,
promote one coherent result generation, and rebuild documentation/figures/claims only from that run.
This is a compute and scientific reconciliation PR; it does not hide failed gates by changing formulas.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) — every manifest, cohort, fold, validity, and summary gate.
- [Phase-7 source audit and refresh design](phase-7-run-artifacts.md) — required source-input gate and
  two-tier reuse/re-decode policy; production cannot begin from presence-only decoder/table files.
- [docs/models_table.md](../../../../docs/models_table.md) — currently stale mirror to regenerate/check.
- [docs/rate_models.md:67-82](../../../../docs/rate_models.md#L67) and [:200-220](../../../../docs/rate_models.md#L200) — invalid comparability/ranking text to replace from the clean run.
- [paper/claims_table.md:53-58](../../../../paper/claims_table.md#L53) — rate, choice, and PFC claims requiring re-adjudication.
- [scripts/rate/fit_per_bin_rate_models.py:271-329](../../../../scripts/rate/fit_per_bin_rate_models.py#L271) — existing smoke/full entry point.
- [scripts/content/fit_replay_content_models.py:274-330](../../../../scripts/content/fit_replay_content_models.py#L274), [scripts/content/fit_swr_contrast_content.py:486-586](../../../../scripts/content/fit_swr_contrast_content.py#L486), and [scripts/rate/fit_swr_contrast_rate.py:340-475](../../../../scripts/rate/fit_swr_contrast_rate.py#L340) — canonical family runners.
- [07 comprehensive audit](../../reviews/07_comprehensive_regression_model_statistical_audit.md) — findings checklist that the final report closes or leaves open.

**Contracts referenced:** all contracts in [shared-contracts.md](shared-contracts.md); none may be relaxed during production execution.

## Tasks

- Verify that all regression and PFC code-bearing phases are merged, the confirmatory/comparison/session
  registries are locked, and the production analysis worktree is clean. Record that single
  code/config `git_sha` in every run. Do not change formulas, schemas, thresholds, or analysis code
  after this gate; a required code change abandons the candidate and starts a new release ID.
- Capture a pre-refit baseline report from legacy artifacts for descriptive before/after comparison:
  artifact dates/schemas, current headline estimates, cohort sizes, fit failures, and known invalid
  comparisons. Include the source audit's out-of-interval/frozen exposure, retired extra columns,
  Q-terminal-tail profile, and per-family artifact/input modification times. Archive exactly one
  read-only, hash-indexed baseline snapshot/report; label it legacy/invalid and do not promote it or
  use it as an acceptance target.
- Run the Phase-7 source audit across every expected session before rebuilding anything. For sessions
  with complete reconstructable source/detector/training/parameter/output provenance matching the
  locked spec, reuse the saved decoder model/results and regenerate derived `data_over_time` into a
  new release-candidate root. Equal time axes alone do not authorize reuse. For all other sessions,
  force a full `--overwrite --output-root <candidate-root>` re-decode; never overwrite the legacy
  inputs. Stage all refreshed bandit and ContFrag bundles, rerun the source audit, and require zero
  samples/exposure outside valid intervals plus exact source/results/derived time and provenance
  hashes before promotion.
- Freeze `expected_sessions.parquet` before compute and require every family-scoped session to end as
  `ok` or match an already-approved entry in `config/regression_session_exclusions.yaml`. A new
  source/QC/cache/fit failure remains a failure row and cannot be waived during this phase.
- Regenerate the bijective session/epoch crosswalk and complete behavior trial table first, then the
  separately keyed Beta-model trial table, and finally all Phase-8 canonical replay and ContFrag
  event/arm/patch/stop tables, qvals table, event-onset table, and stop-timebin table via strict joins
  from the validated source bundles. Run all invariant/alignment audits and one-animal smoke. Require
  exact current schemas—no retired HMM or other stale extra columns. Record
  wall-clock, peak memory, disk use, expected full-run cost, and available resources before launching
  any job over five minutes.
- Smoke each canonical family on one session per animal: stable arm/patch content with hard and soft
  responses, per-stop and per-bin rate, SWR contrasts, uncertainty sensitivities, decision choice,
  Stage-6 descriptive model, cross-history, and SWR power. Validate manifests, expected-cell coverage,
  cohort/fold hashes, rank/fit statuses, robust uncertainty fields, and animal-summary schemas.
- After smoke gates pass, run the full build/refits in durable logged jobs. Do not promote partial family
  runs. Audit production outputs for all five animals, expected sessions/populations/models, exact
  comparison identities, bootstrap success, and missing/invalid cells with reasons.
- Produce the declared Q-terminal-tail report and refit the prespecified primary non-Q models on both
  their maximal cohort and the common Q-observable window. Report the paired effect/retention delta;
  do not impute Q or require numerical invariance.
- Apply the model decision rules without peeking for significance: content primary is the stable M4-style
  arm specification with patch resolution and hard/soft sensitivity; rate prediction uses identical-fold
  held-out score and rate inference uses the prespecified identified specification; choice uses D0/D1
  held-out delta; Stage 6 remains descriptive; sparse/value/uncertainty terms remain sensitivities.
  Select the per-bin paper model only by the locked `PB_PAPER_RANKING` complete-session,
  animal-balanced mean-CV rule and its score/complexity/name tie break; if five-animal complete
  coverage fails, report no winner.
  Load exact comparison membership/parents and the confirmatory multiplicity family from the checked-in
  registries created before this run; do not add, remove, or reclassify a hypothesis after inspecting
  results. A null interaction is reported as no detected difference, not equivalence.
- Generate a machine-readable `model_report.parquet` from specs + manifests. Each row
  states scientific question, estimand, fit purpose/set, response/link/offset/grain, predictors/parent,
  exact N/retention/hash, fold scheme/hash, rank/separation/convergence, robust method, transformed effect
  and interval, all five animal effects, comparison eligibility, and unavailable reason.
- Finalize all corrected required-family runs without changing `CURRENT.json`, assemble them into one
  frozen release-candidate manifest, and prohibit further source/table/model writes to that candidate.
  Run PFC production Phases 3–4 against this candidate's exact
  `canonical_table_generation_sha256`; validate the resulting `pfc_completion.json`. If it fails,
  record PFC as unavailable and claim 20 as pending without blocking unrelated corrected regressions.
  Any candidate source/table change invalidates the PFC completion and returns to the appropriate
  build step.
- After the PFC gate, regenerate every release-owned Markdown report, table, figure, and proposed
  documentation file under `<regression_root>/release_outputs/<release_id>.staging/`, using the
  explicit frozen-candidate resolver. Do not read `CURRENT.json` and do not write live repository
  destinations at this step. PFC Phase 4 supplies its staged Figure-4 files and machine-readable
  claim-20 adjudication; this phase composes the shared claims documents from all row-scoped owners.
  This phase owns `docs/models_table.md`, `docs/rate_models.md`, the covariate docs, and the regression
  evidence/wording for claims 19a–19g plus any registered SWR-power result. It may update only the
  corresponding rows/panels in `paper/claims_table.md` and `paper/figure_claims_plan.md`.
  [`covariate-harmonization` Phase 3](../covariate-harmonization/phase-3-canonical-is-swr-sweep.md)
  owns claim 14's ancillary `is_swr` sensitivity;
  [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md) owns the
  cross-correlogram, continuum verdict/claim 24, and synthesis narrative; and
  [`pfc-fig4-recut` Phase 4](../pfc-fig4-recut/phase-4-inference-figure.md) owns claim 20/Figure 4.
  Mechanically render claim 20 from that phase's adjudication when the PFC component validates, or
  render its prespecified pending/unavailable template when it does not; do not invent PFC wording.
  Leave explicit pending placeholders for the other downstream owners and for the unowned W2b/W3
  reality work. Replace causal/predictive/equivalence language in owned rows with the exact estimand;
  use formal interactions or mark the claim unavailable.
- Append a resolution section to the 07 audit and create
  `.claude/docs/reviews/09_covariate_construction_review_resolution.md`, using the
  [pipeline map](../../../../docs/regression_pipeline_map.md) as its executable-code index.
  Map every finding to code/manifest evidence, a remaining limitation, or an explicitly withdrawn
  claim. Include before/after numerical deltas without requiring the old conclusion to survive. Stage
  both review documents in the same release-output tree.
- Validate every staged output and PFC/component hash, atomically finalize the release-output
  directory, write the completed global release manifest, and only then advance the single
  regression-release pointer. Invoke `publish_release_outputs(release_id)` afterward to copy those
  exact bytes to repository destinations and verify their hashes; no report or figure is recomputed
  after promotion.
- Move legacy fixed-path artifacts into one hash-indexed, read-only
  `<regression_root>/legacy/<timestamp>/` archive only after verifying no consumer reads them; do not
  delete them. `audit_legacy_regression_artifacts.py` must return zero canonical consumers, zero
  per-family “current” pointers, and zero ambiguous current files.

## Deliberately not in this phase

- Fixing a failed estimator by changing its formula, threshold, cohort, or multiplicity family during
  the full run. Return to the owning phase with a reviewed change.
- Treating unchanged numbers as proof of validity or changed numbers as failure.
- Promoting a claim from a sensitivity model because the prespecified primary model is null/invalid.

## Validation slice

| Test | Asserts |
| --- | --- |
| `audit_replay_source_integrity.py` (slow/integration) | every expected session has a valid manifest or explicit exclusion; promoted sessions have zero invalid-time exposure and exact time hashes |
| decoder attestation audit | every reused decoder has exact source/detector/training/parameter/output evidence; all others were rebuilt in the untouched candidate root |
| canonical table invariant suite | exact schemas/keys/cardinalities, timing, trial/decision, suffix-invariant history, patch exposure, and qvals alignment pass |
| per-family smoke matrix (slow) | one session/animal produces complete current-schema run and report rows |
| `audit_promoted_regression_runs.py` | all expected cells/animals/hashes/folds/diagnostics validate; no mixed run |
| frozen-candidate/PFC integration audit | PFC completion names the exact candidate generation; source/table mutation invalidates it before promotion |
| staged-release-output audit | every report/figure/doc is generated from the explicit candidate, hash-declared before promotion, and absent from default current resolution |
| comparison audit | every emitted AIC/BIC/LRT/CV number satisfies the strict comparison contract |
| model report schema test | every claimed effect has estimand, units, robust interval, N, five-animal values, and provenance |
| hard-versus-soft content sensitivity | direction/effect discrepancy is reported; no silent preference switching |
| Q-terminal-tail sensitivity | base models retain tail rows; mapped-window estimates and omission/event profile are reported without imputation |
| figure/table smoke (slow) | all paper-facing outputs load only promoted runs and contain no stale model/column names |
| release-output publication smoke | live repository outputs are byte-identical copies of the immutable release-output tree; no post-promotion recomputation occurs |
| legacy-consumer audit | no canonical code reads fixed legacy artifacts; archived files and their hash index remain recoverable |
| full targeted + repository test suite | passes before `CURRENT.json` promotion and after legacy quarantine |

## Fixtures

Production data are required. Source-validation smoke includes one ordinary aligned session and every
session the baseline audit marks time-mismatched; model smoke uses one session per animal before full
runs. Keep source/run manifests, logs, timing/memory measurements, before/after and Q-tail reports, and
generated small summary tables; large fitted artifacts remain in the configured processed-data run
directories.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against code/results/docs and the appropriate
figure reviewer against regenerated figures. Confirm:
- Every promoted claim traces to one complete manifest and exact comparison/uncertainty contract.
- Every promoted table row traces through a valid session-input manifest; no frozen/out-of-interval
  tail or presence-only cache survives.
- Every expected session remains accounted for as valid, failed, or preapproved/family-scoped excluded;
  no production failure was silently waived.
- All five animal effects are visible; sessions/neurons are not presented as biological replication.
- Invalid/null results are reported honestly and do not trigger post-hoc primary-model changes.
- No stale fixed artifact or old covariate/model name remains in a canonical consumer or owned claim;
  downstream/unowned claims remain explicit placeholders rather than being silently rewritten here.
- The audit resolution table distinguishes fixed issues from irreducible limitations.
- Every task is implemented, no failed gate is bypassed by scope creep, and all named legacy paths are removed or quarantined.
- Full validation and smoke suites pass with slow/integration work marked and logs/manifests retained.
- Production names/docstrings do not mention this plan or phase numbers; every user-facing document listed above is updated.
