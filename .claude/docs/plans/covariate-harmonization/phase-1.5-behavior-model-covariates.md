# Phase 1.5 — Behavior-model (RL) covariate integration

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Status: COMPLETE** on branch `covar-harmon-phase1-definitions` (commits
`d56e265`..`cba4280`; multi-agent code-reviewed, tests green). Not yet merged.

## Why this phase exists (supersedes D3-as-planned)

Phase-1 as written (D3) intended `q_current` / `q_arm` to be a **fitted-α leaky-Q
level** — promote `_bandit_csv_contingency.py`'s `_q_chosen` / `_fit_alpha` into
`covariates.py` and extend it to all arms. During phase-1 that plan was **replaced**
by integrating the prior paper's actual **SpatialBanditTask behavior model** (the
Beta-Bernoulli meta-learning fit, `beta_stable_withleaf`, 2025-11) directly from
Alison's DataJoint schemas. The external model is the canonical value signal the
paper compares against; a local leaky-Q re-implementation would have been a weaker
stand-in. So `q_arm` / `q_current` are now the **beta-model belief** (the per-port
`Q`), not a leaky-Q level, and the whole nominal block moved to the same exactly
aligned source. `leaky_q_value_diff` is still kept as the reported MU comparison
(D3's "kept" decision stands — it is the only leaky-Q construct that remains).

## What was done

1. **DataJoint loader** `analysis/behavior/qvals_table.py` → `qvals_table.pkl`
   (105,344 stops × 79 cols). Sources everything from Alison's read-only schemas:
   `alison_rlmodel.BehaviorModelResults` (values) × `alison_behav.TrialsInfoByEpoch.ByTrial`
   (poke/reward times) via virtual modules (no import of her julia/spyglass code).
   Stop→trial by the exact poke_out interval rule (`assign_stops_to_trials`).
   - **Model covariates:** per-port `Q`→`q_port{i}`/`q_chosen`, `rpe` (= reward −
     q_chosen), Beta `*_mu`/`*_var`, ΔV (`switch_advantage`), value-update
     magnitudes, HMM reward/state entropy — the Comrie et al. decision variables.
   - **Nominal covariates:** `p_reward_current`/`p_reward_port{i}`/`value_diff_patch`/
     `trials_since_block_change` from `p_rew_leaf` — retiring the difflib
     `contingency_table_csv.pkl` (~2.1% at-well misalignment) for **all** consumers.
2. **Vendored DV pipeline** `analysis/behavior/_spatial_bandit_rl.py` (AST-copied
   from Alison's `rl_helpers.py`, 2 documented perf/compat shims — her module can't
   be imported: julia deps).
3. **Consumers migrated** to `qvals_table` for the per-arm value block
   (`attach_contingency_value_block` maps `p_reward_port`/`q_port` → `p_reward_arm`/
   `q_arm`): `prepare_data.py`, `prepare_contfrag_replay_content_data.py`,
   `fit_swr_power_model.py`, `_covariate_ranking.py`, `_content_reward_interactions.py`,
   `fit_uncertainty_rate.py`. `_bandit_csv_contingency.py` **deleted** (not promoted).
4. **Verification layer** (also serves phase-2's "verify" intent):
   - `_verify_covariate_invariants.py` extended with physical bounds (schedule levels,
     `rpe∈[-1,1]`, `value_diff_patch∈[0,0.8]`, entropies).
   - `_audit_covariate_accuracy.py` (new): calibration vs independent DIO reward,
     ΔV-predicts-choice, leakage exposure (0% pre-reward), coverage bias, complete-case
     drop, non-RL spatial/decode checks, gated `reward_rate_arm_k10` reconstruction.
   - `test_qvals_table.py` (new) + strengthened `test_covariates.py`/`test_replay_tables.py`.
5. **Docs:** `docs/behavior_model_covariates.md` (data dictionary) +
   `docs/covariate_dictionary.md` §F (temporal causality) / §G (base-covariate catalog),
   and D2/D3 retagged (`q_arm` = model belief, source `qvals_table`).

## Validation (done)

- Alignment: at-well arm/reward agreement 99.8–100% vs position/DIO; leaf-join 100%;
  entropy coverage 100%.
- Calibration: empirical DIO reward tracks nominal `p_reward` (slope 0.89); ΔV higher
  on switch than stay for all 5 animals.
- Leakage: 0% of modeled events precede reward delivery.
- Refactor equivalence: `build_animal('chimi')` byte-identical to the committed pkl.
- Regenerated `event_arm_table` value block: 98.6% coverage, leaf `p_reward_arm`/
  `q_arm` exact-match qvals ports, trunk 0.0; full invariant verifier PASS (47/47).

## Dependency added

Read-only access to Alison's DataJoint schemas (`alison_rlmodel`, `alison_behav`) on
`lmf-db.cin.ucsf.edu`, and the read-only precomputed fits under `/stelmo/alison/`.
No new Python package dependency. **Do not write to** `/stelmo/alison`, `/home/alison`,
or Alison's schemas.

## Out of scope (still phase-2)

Re-fitting the non-MU families on the regenerated tables and the design-matrix parity
proof — see [phase-2](phase-2-refit-verify.md). Only `uncertainty_rate` was refit here.
