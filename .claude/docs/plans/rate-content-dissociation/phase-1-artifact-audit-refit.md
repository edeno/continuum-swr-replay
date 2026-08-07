# Phase 1 - Artifact audit and content refit

> **SUPERSEDED — DO NOT EXECUTE (2026-07-29).** This phase is retained only as
> historical design context. Canonical corrective work is owned by
> [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md); the only
> surviving downstream rate/content task is
> [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md).

[<- back to PLAN.md](PLAN.md) . [overview](overview.md)

First make the content artifacts trustworthy. The current coefficient readout
already found the main danger: some content artifacts were produced under older
covariate names or before the current event-wise missing-data policy. This phase
does not change scientific models; it verifies and regenerates their outputs.

**Questions:** Which existing content artifacts are current? Which are stale?
Can every headline content coefficient be traced to the current spec and
complete-case contract?

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/model_specs.py:201` - primary arm
  hierarchy, including `reward_rate_arm_k10`.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:867` - SWR-contrast
  arm/patch content ladders and key terms.
- `src/continuum_swr_replay/analysis/replay/fitting.py:61` - conditional
  Poisson fitter and event-wise missing-data policy.
- `scripts/content/fit_replay_content_models.py:493` - common complete-case
  filtering across requested primary content models.
- `scripts/content/fit_swr_contrast_content.py:159` - per-session SWR-contrast
  content fit.
- Existing artifacts under `Processed-Data/replay_content/models*`.

## Tasks

- Add `scripts/qa/audit_replay_content_artifacts.py`.
  - Load primary arm/patch and SWR-contrast arm/patch artifacts.
  - Compare every model's observed `coef_*` columns against current
    `model_specs.py`.
  - Flag deprecated aliases such as `coef_recent_reward_rate`.
  - Flag missing key terms such as `coef_reward_rate_arm_k10`,
    `coef_swr_x_reward_rate_arm_k10`, `coef_swr_x_current`, and
    `coef_swr_x_current_patch`.
  - Emit `Processed-Data/replay_content/model_audits/content_artifact_audit.csv`
    and a markdown summary.
- Add safe staging controls before any full refit.
  - Primary content runner: add `--max-per-animal` or equivalent session-slice
    filtering for smoke runs.
  - Primary content runner: add `--output-dir` or `--output-suffix` so smoke and
    audited refits cannot overwrite canonical `models/` or `models_patch/`
    outputs by accident.
  - SWR-contrast content runner: support an explicit staged output directory or
    suffix for both raw arm/patch artifacts and audited aggregate/provenance
    outputs.
  - Refuse to overwrite canonical artifact paths unless an explicit
    `--promote-canonical`-style flag is passed after audit success.
- Add fit-availability output to both content runners.
  - Write one row per attempted `(model, population, animal, day, epoch)`.
  - Record successful fits and skipped/failed fits with `skip_reason`,
    pre/post complete-case counts, and model size.
  - Use the shared [fit-availability contract](shared-contracts.md#fit-availability-contract).
- Add provenance writing to content runners or a small wrapper that records
  command, script path, git state, input hashes, output hashes, row counts, and
  complete-case policy.
- Smoke-regenerate content artifacts on a small slice:
  - `scripts/content/fit_replay_content_models.py` for arm and patch hierarchies;
  - `scripts/content/fit_swr_contrast_content.py --max-per-animal 1`.
- Run the audit on smoke outputs, then run the full refit only after smoke
  passes.
- Save regenerated outputs first under an audited/staging directory. Promote or
  document canonical paths only after the audit passes.
- Update any docs that currently quote stale content coefficients so they point
  to the audit status rather than stale values.

## Gate

Proceed to Phase 2 only when:

- primary arm/patch and SWR-contrast arm/patch content artifacts pass the audit;
- all headline content terms are present or have explicit, quantified unfit
  counts from the fit-availability table;
- retained arm events have 9 rows and retained patch events have 3 rows;
- each artifact has provenance metadata;
- staged/smoke outputs cannot overwrite canonical artifacts without an explicit
  promotion flag.

If the audit shows stale artifacts and the refit cannot complete, stop here and
write the blocked reason. Do not interpret content coefficients downstream.

## Deliberately not in this phase

- No new scientific covariates.
- No continuous SWR mark.
- No choice prediction.
- No paper figure claims.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_content_artifact_audit_detects_alias` | A fixture artifact with `coef_recent_reward_rate` is flagged stale. |
| `test_content_artifact_audit_current_terms` | A fixture artifact matching current specs passes. |
| `test_complete_event_candidate_sets` | Audit rejects retained arm/patch events with partial candidate sets. |
| `test_fit_availability_records_skips` | A failed or skipped session produces a row with a machine-readable reason. |
| `test_refit_staging_no_canonical_overwrite` | Smoke/staged output flags write outside canonical artifact paths by default. |
| smoke runner | `--max-per-animal 1` writes artifacts and provenance; audit passes on smoke outputs. |

## Fixtures

- Tiny synthetic coefficient tables with current and stale coefficient names.
- Tiny event-arm and event-patch tables with complete and incomplete event
  candidate sets.
- Real-data smoke on one session per animal.

## Review

Confirm that this phase only audits/refits existing content models; no
downstream interpretation uses stale artifacts; provenance is machine-readable;
and the canonical paths are not overwritten until staged outputs pass.
