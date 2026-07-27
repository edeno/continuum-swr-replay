# Phase 4 — Docs & memory reconciliation to the beta value sign

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Bring the prose in line with the beta-only ladders and the beta refit numbers.
The load-bearing scientific change: the value–content result flips construct — the
retired HMM said content *prefers* high value (`swr_x_q_arm_hmm ≈ +1.06`); beta
says content *avoids* high value (`swr_x_q_arm ≈ −2.5`). Everything that hedged
"cite HMM, flag construct-dependence" collapses to a single beta result. Uses the
Phase-2/3 refit numbers as the source of truth (do not copy stale figures).

**Inputs to read first:**

- [docs/replay_rate_content_swr_synthesis.md](../../../../docs/replay_rate_content_swr_synthesis.md) — the value/uncertainty sections and the "Update 2026-07-02/07-03" construct-dependence blocks.
- [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md), [docs/behavior_models.md](../../../../docs/behavior_models.md), [docs/models_table.md](../../../../docs/models_table.md), [docs/design/content_model_design.md](../../../../docs/design/content_model_design.md) — covariate definitions and the model-table mirror of `model_specs.py`.
- [shared-contracts.md](shared-contracts.md) — the mapping and the beta-entropy definition to document.

**Contracts referenced:**

- [Contract 1 + notes](shared-contracts.md#contract-1--the-covariate-mapping) — the beta-only value construct and the sign flip.

## Tasks

- **`docs/covariate_dictionary.md` / `docs/behavior_models.md`:** replace the HMM
  covariate entries with the beta definitions; document `beta_reward_entropy`
  (Contract 2) and that the value construct is now `q_arm` only. State that
  `state_entropy`/`reward_entropy` (HMM) are removed.
- **`docs/models_table.md` / `docs/design/content_model_design.md`:** update the
  rung mirror so `C4`/`CS5`/`Mv`/rate `VALUE/RPE/UNCERT/UPDATE` name the beta
  covariates (matches `model_specs.py` post-Phase-2).
- **`docs/replay_rate_content_swr_synthesis.md`:** rewrite the value–content
  narrative to the beta sign (content avoids high value, `swr_x_q_arm ≈ −2.5`,
  from the Phase-2 refit); **delete** the construct-dependence apparatus (the
  "report HMM / flag sign flip / beta robustness column" hedges). Refresh any
  value/RPE/entropy rate numbers from the Phase-2 refit. Reframe the
  exploit/explore note: the `Mvu` degeneracy was a cross-arm-FE artifact — the
  within-arm frame separates beta value and uncertainty (this unblocks the beta
  exploit/explore model; note it as available, not "needs HMM").
- **Memory notes:** update `project_swr_value_content_construct_dependence`
  (construct-dependence is moot — one construct now; keep the historical note
  that HMM was retired) and `project_arm_uncertainty_content` (the "pair with HMM
  value" recommendation is superseded — beta value pairs cleanly within-arm).
  Update `MEMORY.md` pointers.
- **Sweep for stale value claims:** grep docs + `paper/` for `q_arm_hmm`,
  `+1.06`, "prefers high value", "construct-dependence", "HMM"; reconcile or
  remove each. `.claude/docs/reviews/*` are archived audit records — leave as
  historical, do not rewrite.

## Deliberately not in this phase

- Any code, spec, runner, or figure change — Phases 1–3.
- New scientific claims — this phase reconciles existing prose to the refit
  numbers, it does not add findings.

## Validation slice

| Test | Asserts |
| --- | --- |
| grep clean (active docs) | no `q_arm_hmm` / `hmm_relative_value` / `reward_entropy` in `docs/` except explicit "retired HMM" historical notes. |
| number provenance | every value/RPE/entropy number in the synthesis doc traces to a Phase-2 refit pickle/CSV (spot-check 3 against `swr_contrast_content_term_diagnostics.csv`). |
| sign consistency | synthesis states content *avoids* high value; no residual "prefers"/`+1.06` for the beta result. |
| dictionary completeness | `beta_reward_entropy`, `switch_advantage_mu_preoutcome_sess`, `global_mu_update_abs_postoutcome_sess`, `rpe` each have a covariate-dictionary entry. |

## Fixtures

- Phase-2 refit CSVs (`swr_contrast_*_term_diagnostics.csv`) as the numeric source.

## Review

Before opening the PR, dispatch `code-reviewer` (docs review) against the diff. Confirm:
- No active doc still asserts the HMM value sign or the construct-dependence hedge.
- Every migrated number cites a refit artifact, not a stale figure.
- The covariate dictionary documents all four new beta covariates.
- Archived `.claude/docs/reviews/*` left intact as historical record.
