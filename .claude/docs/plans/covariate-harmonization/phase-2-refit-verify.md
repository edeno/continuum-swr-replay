# Phase 2 — Re-fit all families, verify parity, refresh docs

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Re-fit every model family on the current tables using the canonical covariates from
phase 1, and **verify the refactor preserved everything it should**: covariates whose
definition did not change must reproduce their pre-refactor coefficients to tolerance;
the deliberately-changed ones (per-arm reward 5→10, `leaky_q`→`q`) are reported, not
asserted. Then refresh the numbers in the docs/claims.

**Inputs to read first:**
- [overview.md](overview.md#metrics) — the parity tolerance + coverage rules.
- [shared-contracts.md](shared-contracts.md) — to confirm the consumed schema.
- The runners: `fit_per_bin_rate_models.py`, `fit_uncertainty_rate.py`, `fit_replay_content_models.py` (arm `{all,T,F}` + patch), `fit_swr_power_model.py`, `compare_event_source_models.py`, `fit_per_stop_rate_model.py`.
- Current fit outputs (the pre-refactor baseline to compare against): `Processed-Data/replay_content/models*/all_models.pkl`, `uncertainty_rate.pkl`, `swr_power_model.pkl`, `event_source_comparison/non_local_HPC_comparison_fits.pkl`.

## Tasks

- **Baseline capture (before re-fitting).** Copy the current fit pickles to a baseline
  dir (e.g. `…/_pre_harmonization/`). These are the comparison set for parity.
- **Re-fit all families** on the current (2026-06-24) tables with canonical covariates:
  per-bin rate ladder, MU uncertainty, per-stop rate, SWR-power, content arm `{all,T,F}`
  + patch `{all,T,F}`, choice (`compare_event_source_models.py --event-source non_local`).
  Long runs in tmux; smoke one session per runner first (timing) per the batch rule.
- **Parity check script** `scripts/_verify_harmonization_parity.py`: for each family,
  join baseline vs new per-session coefficients on the **unchanged-definition**
  covariates and assert |median Δcoef| < tol (e.g. 1e-6) and unchanged signed-rank
  verdicts; for the **changed** covariates (`reward_rate_arm_k10` vs old
  `recent_reward_rate`; `q_*` vs `leaky_q_value_diff`) print an old→new comparison
  table (no assert). Report per-family coverage (session counts unchanged).
- **Refresh docs/claims with the changed numbers** (this is a task, not deferred):
  - `docs/covariate_dictionary.md` — flip status to implemented; record per-animal α range and the per-arm-reward 5→10 effect on the headline content coefficients.
  - `paper/claims_table.md` — update any content/choice numbers that moved (the per-arm reward window change touches M3+ content and the choice baseline); the M0–M7 ladder coefficients should be unchanged (value block is additive) — confirm and note.
  - `docs/replay_rate_content_swr_synthesis.md` + `docs/per_bin_rate_results.md` — refresh any covariate numbers that changed (MU `q_current` replacing `leaky_q`; coupling `q_current` term).

## Deliberately not in this phase
- No new covariate definitions or spec changes (all in phase 1).
- No rollup-table regen; no re-decode; no reality re-run.
- No new figures unless a refreshed number invalidates an existing one.

## Validation slice

| Test | Asserts |
| --- | --- |
| `_verify_harmonization_parity` (unchanged block) | reward kernel, `p_reward_current`, `rpe`, `value_diff_patch`, `recent_switch_rate`, `is_after_switch`, spatial role, spline, self-history coefficients reproduce baseline to tol; signed-rank verdicts unchanged. |
| `_verify_harmonization_parity` (changed block) | old→new comparison printed for `reward_rate_arm_k10` and `q_*`; finite, sensible; reported. |
| coverage check | each family's session count == baseline (no merge-induced shrinkage; NaN-drops counted). |
| M0–M7 / P0–P4 invariance | the shipped content ladder coefficients are unchanged vs the re-verified fits (value block additive). |
| docs numbers | every coefficient quoted in the updated docs matches the new fit outputs (spot-check the headline rows). |

## Fixtures
Uses the real fit outputs (baseline copies + new). No synthetic fixtures; the parity
check is the verification.

## Review
Dispatch `code-reviewer` against the diff. Confirm: parity holds for all
unchanged-definition covariates (this is the core correctness gate — a failure means
a rename silently changed a definition); changed covariates are reported with old→new;
coverage unchanged; M0–M7/P0–P4 invariance verified; every doc number updated to match
the new fits (no stale coefficients left in `covariate_dictionary.md` / `claims_table.md`
/ the synthesis docs); no plan/phase references in code, test, or commit-message-bound
artifacts.
