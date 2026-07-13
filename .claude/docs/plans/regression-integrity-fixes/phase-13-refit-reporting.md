# Phase 13 — Clean production refit, promotion, and reporting cutover

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Run every corrected canonical analysis from regenerated tables, validate all identities/diagnostics,
promote one coherent result generation, and rebuild documentation/figures/claims only from that run.
This is a compute and scientific reconciliation PR; it does not hide failed gates by changing formulas.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) — every manifest, cohort, fold, validity, and summary gate.
- [docs/models_table.md](../../../../docs/models_table.md) — currently stale mirror to regenerate/check.
- [docs/rate_models.md:67-82](../../../../docs/rate_models.md#L67) and [:200-220](../../../../docs/rate_models.md#L200) — invalid comparability/ranking text to replace from the clean run.
- [paper/claims_table.md:53-58](../../../../paper/claims_table.md#L53) — rate, choice, and PFC claims requiring re-adjudication.
- [scripts/rate/fit_per_bin_rate_models.py:271-329](../../../../scripts/rate/fit_per_bin_rate_models.py#L271) — existing smoke/full entry point.
- [scripts/content/fit_replay_content_models.py:274-330](../../../../scripts/content/fit_replay_content_models.py#L274), [scripts/content/fit_swr_contrast_content.py:486-586](../../../../scripts/content/fit_swr_contrast_content.py#L486), and [scripts/rate/fit_swr_contrast_rate.py:340-475](../../../../scripts/rate/fit_swr_contrast_rate.py#L340) — canonical family runners.
- [07 comprehensive audit](../../reviews/07_comprehensive_regression_model_statistical_audit.md) — findings checklist that the final report closes or leaves open.

**Contracts referenced:** all contracts in [shared-contracts.md](shared-contracts.md); none may be relaxed during production execution.

## Tasks

- Capture a pre-refit baseline report from legacy artifacts for descriptive before/after comparison:
  artifact dates/schemas, current headline estimates, cohort sizes, fit failures, and known invalid
  comparisons. Label it legacy/invalid; do not promote it or use it as an acceptance target.
- Regenerate Phase-8 canonical tables. Run all invariant/alignment audits and one-animal smoke. Record
  wall-clock, peak memory, disk use, expected full-run cost, and available resources before launching
  any job over five minutes.
- Smoke each canonical family on one session per animal: stable arm/patch content with hard and soft
  responses, per-stop and per-bin rate, SWR contrasts, uncertainty sensitivities, decision choice,
  Stage-6 descriptive model, cross-history, and SWR power. Validate manifests, expected-cell coverage,
  cohort/fold hashes, rank/fit statuses, robust uncertainty fields, and animal-summary schemas.
- After smoke gates pass, run the full build/refits in durable logged jobs. Do not promote partial family
  runs. Audit production outputs for all five animals, expected sessions/populations/models, exact
  comparison identities, bootstrap success, and missing/invalid cells with reasons.
- Apply the model decision rules without peeking for significance: content primary is the stable M4-style
  arm specification with patch resolution and hard/soft sensitivity; rate prediction uses identical-fold
  held-out score and rate inference uses the prespecified identified specification; choice uses D0/D1
  held-out delta; Stage 6 remains descriptive; sparse/value/uncertainty terms remain sensitivities.
- Generate a machine-readable `model_report.parquet` and Markdown report from specs + manifests. Each row
  states scientific question, estimand, fit purpose/set, response/link/offset/grain, predictors/parent,
  exact N/retention/hash, fold scheme/hash, rank/separation/convergence, robust method, transformed effect
  and interval, all five animal effects, comparison eligibility, and unavailable reason.
- Atomically promote corrected runs. Quarantine/remove legacy fixed-path artifacts after verifying no
  consumer reads them. `audit_legacy_regression_artifacts.py` must return zero canonical consumers and
  zero ambiguous “current” files.
- Regenerate paper-facing tables/figures only through strict current-run loaders. Update
  `docs/models_table.md`, `docs/rate_models.md`, covariate docs, synthesis, `paper/claims_table.md`, and
  `paper/figure_claims_plan.md`. Replace causal/predictive/equivalence language with the exact estimand;
  use formal interactions/equivalence intervals or mark the claim unavailable.
- Run/validate the dedicated PFC plan. If `pfc_completion.json` fails, omit Figure-4 cross-area promotion
  and mark claim 20 pending; do not block unrelated corrected regression runs.
- Append a resolution section to the 07 audit mapping every finding to code/manifest evidence, remaining
  limitation, or explicitly withdrawn claim. Include before/after numerical deltas without requiring
  the old conclusion to survive.

## Deliberately not in this phase

- Fixing a failed estimator by changing its formula, threshold, cohort, or multiplicity family during
  the full run. Return to the owning phase with a reviewed change.
- Treating unchanged numbers as proof of validity or changed numbers as failure.
- Promoting a claim from a sensitivity model because the prespecified primary model is null/invalid.

## Validation slice

| Test | Asserts |
| --- | --- |
| canonical table invariant suite | exact keys/cardinalities, timing, trial/decision, history, and qvals alignment pass |
| per-family smoke matrix (slow) | one session/animal produces complete current-schema run and report rows |
| `audit_promoted_regression_runs.py` | all expected cells/animals/hashes/folds/diagnostics validate; no mixed run |
| comparison audit | every emitted AIC/BIC/LRT/CV number satisfies the strict comparison contract |
| model report schema test | every claimed effect has estimand, units, robust interval, N, five-animal values, and provenance |
| hard-versus-soft content sensitivity | direction/effect discrepancy is reported; no silent preference switching |
| figure/table smoke (slow) | all paper-facing outputs load only promoted runs and contain no stale model/column names |
| legacy-consumer audit | no canonical code reads fixed legacy artifacts |
| full targeted + repository test suite | passes before `CURRENT.json` promotion and after legacy quarantine |

## Fixtures

Production data are required. Smoke one session per animal before full runs. Keep logs, manifests,
timing/memory measurements, before/after report, and generated small summary tables; large fitted
artifacts remain in the configured processed-data run directories.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against code/results/docs and the appropriate
figure reviewer against regenerated figures. Confirm:
- Every promoted claim traces to one complete manifest and exact comparison/uncertainty contract.
- All five animal effects are visible; sessions/neurons are not presented as biological replication.
- Invalid/null results are reported honestly and do not trigger post-hoc primary-model changes.
- No stale fixed artifact or old covariate/model name remains in a canonical consumer or claim.
- The audit resolution table distinguishes fixed issues from irreducible limitations.
- Every task is implemented, no failed gate is bypassed by scope creep, and all named legacy paths are removed or quarantined.
- Full validation and smoke suites pass with slow/integration work marked and logs/manifests retained.
- Production names/docstrings do not mention this plan or phase numbers; every user-facing document listed above is updated.
