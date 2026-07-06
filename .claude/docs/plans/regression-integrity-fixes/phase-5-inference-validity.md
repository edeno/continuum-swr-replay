# Phase 5 — Inference validity (animal-level primary; cohort/construct sensitivity)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Contracts referenced:** [cohort-audit schema](shared-contracts.md#cohort-audit-schema) — attach to every
sensitivity refit; [covariate timing](shared-contracts.md#covariate-timing) — labels which fits are retrospective.

Promote animal-level (n=5) evidence to the primary inference for the base content and rate ladders
(session-level Wilcoxon becomes descriptive, matching the is_swr contrast runners), and add the
common-cohort and `swr_x_value` cohort/construct sensitivity fits that the audit and the complete-case
discussion call for. **Runs LAST — depends on phases 3 AND 4** (F4): it consumes the phase-3 cohort/
term-diagnostics tables *and* must fit on the **phase-4-cleaned designs** (encoded `reward_rate_arm_k10`,
`reward_lag1`-swapped R1/R2, timing-gated/relabeled value rungs). Paper-facing aggregation and
sensitivity therefore reflect the cleaned covariates, not the pre-cleanup ones. The full hierarchical
mixed model stays a documented non-goal (overview Open Question 2).

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:83-121](../../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py) — the per-animal median + `n_animals_pos/n_animals` + BH-FDR + `|coef|<10` filter to reuse.
- [scripts/content/fit_replay_content_models.py:238-255](../../../../scripts/content/fit_replay_content_models.py) — current session-level Wilcoxon "primary" summary and BH-FDR; `:495-510` union cohort.
- [scripts/rate/fit_per_stop_rate_models.py](../../../../scripts/rate/fit_per_stop_rate_models.py) and [scripts/rate/fit_per_bin_rate_models.py](../../../../scripts/rate/fit_per_bin_rate_models.py) — the rate ladder aggregation points (currently session-level).
- MEMORY: `project_swr_value_content_construct_dependence.md` — beta `q_arm` (0/5) vs HMM `q_arm_hmm` (5/5) sign flip; the sensitivity here must report both.

## Tasks

- **Refactor the per-animal aggregation into a reusable function.** Lift the core of
  `contrast_aggregation.py:83-113` into `aggregate_per_animal(session_coefs) -> {coef_animal_median,
  n_animals_pos, n_animals_used, fdr_p_descriptive}` (return `n_animals_used` — the animals actually
  surviving this rung — so the function contract matches the `n_animals_pos/n_animals_used` reporting
  contract) so both the contrast runners and the base ladders call the same code. Keep the existing
  contrast-runner behavior byte-identical (it already uses this logic).
- **Promote animal-level to primary in the base content ladder.** In `fit_replay_content_models.py`,
  add the per-animal aggregation as the **primary** reported summary (per-rung `coef_animal_median` +
  `n_animals_pos/n_animals_used`), and relabel the existing session Wilcoxon output as `*_descriptive`.
  **Always report the denominator `n_animals_used` (Medium, round 7)** — after common-cohort/time-gating/
  term filtering, a rung may retain fewer than all 5 animals, so `n_animals_pos/5` would misstate the
  sign-consistency base. Use `n_animals_pos/n_animals_used`, and attach a **headline warning** (or drop
  the rung from headline claims) when `n_animals_used < 5`, so a 3/3 is never silently read as 3/5. Do not delete
  the session summary (still useful as a within-animal descriptive), just demote its label/role. Same
  change for the per-stop and per-bin rate ladders' aggregation steps.
- **Common-cohort sensitivity fits — content AND rate, WITHIN timing/claim families (Medium, rounds 6+9).**
  Add a `--common-cohort` mode to the content contrast runner (`fit_swr_contrast_content.py`), the content
  base ladder, **and the rate contrast runner** (`fit_swr_contrast_rate.py`). **Intersect only within a
  comparable family, not across all rungs (round 9):** after phase 4 the rungs no longer share an estimand
  — predictive-safe rungs fit on all stops, retrospective post-outcome rungs (RPE/update) fit on the
  post-current-outcome subset, and descriptive/future-referencing peri-switch rungs are not predictive at
  all. Forcing one all-rung intersection would shrink every rung down to the retrospective subset and put
  non-comparable estimands in one ΔAIC table. So define the families — `{predictive, retrospective,
  descriptive}` — and compute a common cohort **within each family**; only ΔAIC *within* a family is
  meaningful. The rate ladder also compares AIC across rungs after a qvals **inner join** + per-model
  `dropna` (`fit_swr_contrast_rate.py:228,137`), so its within-family ΔAIC is equally cohort-dependent.
  Emit these alongside the default per-rung-cohort fits with a phase-3 cohort record for each. If any
  paper-facing ΔAIC is claimed it must use the within-family common-cohort fits; otherwise label those AIC
  comparisons explicitly as
  cohort-dependent descriptive summaries in `docs/rate_models.md`.
- **`swr_x_value` sensitivity — TWO separate checks, not a 2×2 (F3).** Beta `q_arm`, `q_arm_hmm`, and
  `p_reward_arm` have **identical** missingness (verified: 8,064 rows / 896 events / 310 stops; **zero**
  rows where beta is present but HMM is missing). So beta `q_arm` **cannot** restore the qvals tail —
  the original "plus-tail beta proxy" is empirically impossible. Split into:
  1. **Construct sensitivity (shared qvals-present cohort).** `scripts/content/sensitivity_swr_value.py`
     refits `swr_x_value` on the *same* HMM-present cohort using construct ∈ {beta `q_arm`, HMM
     `q_arm_hmm`}, reporting sign + `n_animals_pos/n_animals_used` for each. This is the honest test of the known
     construct-dependence (beta − vs HMM +) and is fully computable today.
  2. **Missing-tail sensitivity (descriptive only).** Characterize the 896 dropped events *without*
     value covariates: their SWR fraction, replay content (current-arm bias, off-current spread), and
     session-position/reward profile (reuse phase-3 `drop_profile`). Report whether the dropped tail
     looks systematically different from the fitted cohort on the *observable* (non-value) axes. This
     bounds the risk that the value effect is cohort-driven **without** fabricating value for the tail.
  Per the resolved mechanism-B decision (overview *Resolved decisions #2*), the tail is **dropped** —
  value models are defined on "trials with a fitted HMM state"; the upstream fit is **not** extended and
  no proxy is substituted. The descriptive check exists to show the drop does not distort the value
  effect, not to recover it. The runner's output must state the construct sensitivity and the descriptive
  tail profile side by side and explicitly say the tail is dropped, not value-imputed.
- **Value-controlled choice model — REQUIRED if the choice model is paper-facing (round 6).** If the
  choice model's `replay_frac` effect supports any paper claim, a value-controlled sensitivity is
  **not elective**: add a `C_value` choice spec that includes `q_arm_hmm` in the choice predictor set
  (`choice_prediction.py` data prep already pulls per-arm columns; add `q_arm_hmm` to the selected columns
  and the spec in `model_specs.py`), and report whether `replay_frac`'s effect survives controlling for
  value (this also partly addresses the `dist_to_arm`-laundering caveat). Only skip it if the choice model
  is explicitly *not* paper-facing — in which case phase-1's corrected doc (choice uses no value covariate)
  stands and this is recorded as a deliberate non-claim, not an omission.
- **Docs.** Update `docs/models_table.md` and `docs/rate_models.md` to state that population inference
  for all ladders is per-animal sign consistency reported as `n_animals_pos/n_animals_used` (primary),
  with a headline warning when `n_animals_used < 5`, and session Wilcoxon as descriptive. Document the
  within-family `--common-cohort` and `sensitivity_swr_value` outputs.

## Deliberately not in this phase

- The NumPyro mixed conditional-logit with animal random effects — overview Open Question 2, explicit
  non-goal. Per-animal aggregation is the deliverable; the mixed model is a later, separate plan.
- Changing which covariates are used — that is phase 4. This phase changes *how coefficients are
  aggregated and stress-tested*, not the design matrices (except the `C_value` add — required for any paper-facing choice claim).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/unit/test_aggregate_per_animal.py::test_matches_contrast` | refactored `aggregate_per_animal` reproduces `contrast_aggregation`'s existing output on a fixed session-coef frame (byte-identical) |
| `::test_sign_consistency` | 5-animal frame with 4 positive medians → `n_animals_pos=4, n_animals=5` |
| `tests/integration/test_common_cohort.py` (slow) | on one cached session, rungs **within the same timing/claim family** share an identical event set (phase-3 audit `n_modeled` equal within family); rungs in **different** families (predictive vs retrospective) are **not** forced to equal `n_modeled` — the retrospective family's cohort is a strict subset (round 9) |
| `sensitivity_swr_value.py` smoke | prints construct sensitivity (beta vs HMM, sign + `n_animals_pos/n_animals_used`) on the shared cohort, AND the descriptive missing-tail profile — with an explicit "tail not value-imputed" statement (F3) |
| `tests/unit/test_value_missingness.py::test_identical_missingness` | q_arm / q_arm_hmm / p_reward_arm share NaN rows exactly; 0 beta-present-HMM-missing — guards against re-introducing a beta-restores-tail assumption |

## Fixtures

Unit tests use hand-built session-coefficient frames (no fitting). Integration/smoke use a few cached
sessions across ≥2 animals so the per-animal aggregation is exercised on >1 animal.

## Review

Dispatch `code-reviewer`. Confirm:
- The refactored aggregation reproduces the contrast runner's existing numbers exactly (no silent change to the already-correct is_swr ladders).
- Base-ladder "primary" is now per-animal; session Wilcoxon relabeled descriptive, not deleted.
- Common-cohort fits genuinely share one event/stop set across rungs (checked via phase-3 audit).
- The `swr_x_value` sensitivity reports both constructs on the **shared** cohort AND a descriptive missing-tail profile, and does **not** claim beta restores the tail (F3) — no cherry-picking the HMM-positive result.
- Docs updated in the same PR.
