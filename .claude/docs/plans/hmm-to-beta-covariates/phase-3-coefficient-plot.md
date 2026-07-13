# Phase 3 — Coefficient forest plot: incorporate the beta covariates

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Repoint the "regression covariates next to the content response" figure — the
coefficient forest plot — from HMM terms to the beta terms now in the refit
pickles. The plotting engine is name-agnostic; this is a literal-string swap in
three hardcoded blocks plus the SD list. Depends on Phase 2's refit pickles
(`swr_contrast_rate.pkl`, `swr_contrast_content_arm.pkl`) carrying `coef_<beta_term>`
columns.

**Inputs to read first:**

- [scripts/figures/plot_swr_contrast_coefs.py:38-75](../../../../scripts/figures/plot_swr_contrast_coefs.py) — `_global_sd` reads each covariate's SD from the source tables (`:40-49`); the `SD` literal list (`:52-56`); `std_factor` (`:62-75`) matches `term`/`swr_x_term` against `SD` (unmatched continuous terms silently fall through to factor 1.0 — a correctness trap for a renamed term).
- [scripts/figures/plot_swr_contrast_coefs.py:140-207](../../../../scripts/figures/plot_swr_contrast_coefs.py) — hardcoded `rate_rows` (`:140-153`), `mod_rows` (`:162-173`), `con_rows` (`:174-180`) term tuples + labels, and the three figure builders.
- [shared-contracts.md](shared-contracts.md#contract-1--the-covariate-mapping) — the term names to substitute.

**Contracts referenced:**

- [Contract 1](shared-contracts.md#contract-1--the-covariate-mapping) — term-name substitutions must match the fitted `coef_<term>` columns exactly, else the row renders empty.

## Tasks

- **`SD` list (`:52-56`):** replace `q_arm_hmm`, `hmm_relative_value`, `hmm_rpe`,
  `reward_entropy`, `hmm_value_update` with `q_arm`,
  `switch_advantage_mu_preoutcome_sess`, `rpe`,
  `global_mu_update_abs_postoutcome_sess`, `beta_reward_entropy`. Confirm each
  resolves to a real column in `event_arm_table`/`qvals_table`/`stop_table` (the
  `_global_sd` loop, `:44-48`), so `std_factor` scales rather than falling
  through to 1.0.
- **Rate rows (`:140-153`):** repoint the `("…", "G4_value", …, "hmm_relative_value")`,
  `G5_rpe/hmm_rpe`, `G6_uncert/reward_entropy`, `G7_update/hmm_value_update`
  tuples to the beta terms; update labels ("rel. value (beta ΔV)", "RPE (beta)",
  "belief entropy (beta)", "value-update (beta)").
- **Content-modulation rows (`:162-173`):** `q_arm_hmm (C4) → q_arm (C4)`;
  `rpe_x_current`/`relval_x_current`/`update_x_current` keep their names (Phase 2
  kept the interaction column names, only repointed sources); `entropy_x_current`
  now reflects `arm_uncertainty_z` — relabel accordingly.
- **Contrast rows (`:174-180`):** `swr_x_q_arm_hmm (CS5) → swr_x_q_arm (CS5)`,
  label "swr × value (beta)".
- **Regenerate all three figures** and confirm no row is blank (blank ⇒ a term
  name mismatches its `coef_<term>` column).
- **Update any caption/README** that names the HMM covariates for this figure
  (search the figure's output dir and `paper/figures/` index).

## Deliberately not in this phase

- Model or covariate changes — done in Phase 2.
- Narrative / synthesis-doc edits — Phase 4 (this phase only touches the figure
  script and its immediate caption).

## Validation slice

| Test | Asserts |
| --- | --- |
| no blank rows | each of the three regenerated figures has a non-empty coefficient distribution for every listed row (assert per-row session count > 0). |
| SD resolved | every continuous term in the row lists returns a non-1.0 `std_factor` (i.e. matched an `SD` entry) — guards the silent fall-through. |
| sign check | the `swr × value (beta)` row median is negative (matches Phase 2's `swr_x_q_arm`). |
| render | all three figures write PNG+PDF without error. |

## Fixtures

- Phase-2 refit pickles (`swr_contrast_rate.pkl`, `swr_contrast_content_arm.pkl`,
  `swr_contrast_content_patch.pkl`).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every term string matches a `coef_<term>` column in the refit pickles (no blank rows).
- No continuous term falls through `std_factor` to 1.0.
- Labels no longer say "HMM"; the value row reads as beta value-avoidance.
- Only the figure script + caption changed; no model/doc drift.
