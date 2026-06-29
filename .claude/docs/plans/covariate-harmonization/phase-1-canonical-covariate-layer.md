# Phase 1 — Harmonize definitions, update builders, regenerate tables

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Change the covariate definitions in `covariates.py` (created in phase-0) to canonical
(D1–D5 as revised), update **both** table builders (replay + contfrag) to emit them,
**regenerate** the canonical tables (reusing existing decodes — no re-decode), and
point every model family at the canonical names. Ends with the code coherent, unit
tests green, the regenerated tables passing the rebuild-integrity check, and a
one-session smoke fit per family. (Re-fitting the full dataset + parity = phase 2.)

**Inputs to read first:**
- [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) — the spec.
- [shared-contracts.md](shared-contracts.md) — names, where-they-live, rebuild-integrity contract, patch aggregator.
- `tables.py:1247-1252` (replay covariate batch), `:1277` `nominal_reward_prob`, `:1646` schema list; `analysis/contfrag/tables.py:36,638,657,680,767` (contfrag covariates).
- `model_specs.py:194,205,310` + `MODEL_CHOICE`; `fit_uncertainty_rate.py:86,198`; `fit_swr_power_model.py:41`; `compare_event_source_models.py`+`choice_prediction.py`; consumers `simulation.py`, `learning.py`, `visualization/session_tables.py`.

**Contracts referenced:** [canonical-covariate-names](shared-contracts.md#canonical-covariate-names), [where-the-covariates-live](shared-contracts.md#where-the-covariates-live), [rebuild-integrity](shared-contracts.md#rebuild-integrity-contract-the-safety-rule).

## Tasks

- **Change definitions in `covariates.py`:** `recent_reward_rate_all_events` callers use **k=10** (`reward_rate_arm_k10`); `recent_trial_reward_rate`→exposed as `reward_rate_trial_k10` (rename only); **extend the promoted Q rollout to all-arm** so `q_arm`/`q_current` come from one per-animal-α fit (record α to a **sidecar JSON**). Compute `p_reward_arm` (per-arm nominal value) from the contingency `p_reward_port{0..5}`. **Keep `_leaky_q_value_diff`** (retained comparison — do not remove).
- **Update the replay builder** (`tables.py`): emit `reward_rate_arm_k10`, `q_arm`, and **populate + rename `nominal_reward_prob`→`p_reward_arm`** (`:1277`); update the schema list (`:1646`) and the viz registry (`session_tables.py` `nominal_reward_prob` panel) to `p_reward_arm`. Patch builder: aggregate per [the aggregator rule](shared-contracts.md#where-the-covariates-live) (mean for fractions/probs).
- **Update the contfrag builder** (`analysis/contfrag/tables.py`): apply the same canonical definitions (`recent_reward_rate_per_arm`→k=10, `swr_x_recent_reward_rate`→`swr_x_reward_rate_arm_k10`, `recent_trial_reward_rate`→`reward_rate_trial_k10`, add `p_reward_arm`/`q_arm`).
- **Add + run the input-contract validator before regenerating** (audit 01): a `validate_dot_event_contract` (monotonic time within session, single session per `dot`, positive `non_local_event` labels forming one contiguous run, non-negative durations). Run it across **every** saved `data_over_time.pkl`; report counts of unsorted frames / duplicate-non-contiguous labels / zero-duration events / multi-session inputs; **fail closed** if any violation would corrupt the regen.
- **Fix the `TMAX` edge** (audit 03): `aggregate_to_timebins` uses `int(tmax/bin_s)` → 3.99 s, dropping ~169 onsets. Use `ceil(tmax/bin_s)` and clip the final bin's exposure to `tmax`; add a non-divisible-`tmax` regression test. (Done here since the per-bin design is being touched; phase-2 re-fits the ladder.)
- **Regenerate the canonical tables** (replay + contfrag) via the prep scripts, reusing existing decodes. **Smoke one session first** (`nvidia-smi`/cores check, time it, extrapolate — contfrag gets its own extrapolation) before the full batch; run in **tmux**; document the session name; **quiesce concurrent table readers** and write-to-temp-then-swap. On completion, run the rebuild-integrity check (response/posterior columns reproduce the backup) before proceeding.
- **Spec updates** (`model_specs.py`): rename `recent_reward_rate`→`reward_rate_arm_k10` in `_M3`/M-ladder/patch/`MODEL_CHOICE` and `swr_x_recent_reward_rate`→`swr_x_reward_rate_arm_k10`. Add a **value block** as additive rungs (e.g. `Mv = M5 + (p_reward_arm, q_arm)`, patch analogue) — **M0–M2 and any rung without the per-arm rate stay bit-identical**; M3–M7 change at k=10 by design (flagged for phase-2 re-verify).
- **Runner updates:** `fit_uncertainty_rate.py` — **add** `q_current` to `CURATED_COVARIATES`, **keep** `leaky_q_value_diff`, drop `--q-alpha`. `fit_swr_power_model.py` — add `q_current`. `compare_event_source_models.py`/`choice_prediction.py` — canonical names + `p_reward_arm`/`q_arm`, **and switch the choice baseline `dist_to_arm`→`arm_dist_rank`** (audit 04: `build_choice_dataset` carries only `dist_to_arm`; the ranked-distance baseline is the dictionary's choice-set control). Sweep `simulation.py`/`learning.py` for the renames; update the `test_uncertainty_covariates`/`test_compare_event_source_models` expectations to the canonical covariate + model set.
- **Docstrings/docs:** mark `docs/covariate_dictionary.md` as implemented-in-code; `covariates.py` module docstring points at it.

## Deliberately not in this phase
- **No re-decode** (decodes reused by the rebuild).
- **No full-dataset re-fit / parity verification** — phase 2 (only one-session smoke fits here).
- No spline / self-history / spatial / `arm_dist_rank` changes.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_covariates::test_arm_reward_window` | `reward_rate_arm_k10` = rewarded fraction of last 10 visits to that arm; trunk 0.0; never-visited NaN. |
| `test_covariates::test_q_all_arm` | `q_arm` at current arm == `q_current`; per-animal α recorded to sidecar; one fit. |
| `test_covariates::test_p_reward_arm` | `p_reward_arm` == decoded port prob per arm; == `p_reward_current` at current arm; the old `nominal_reward_prob` name is gone (renamed, no orphan). |
| `test_specs::test_predictors_resolve` | every spec predictor resolves to a present canonical column (no stale `recent_reward_rate`/`nominal_reward_prob`/`q_chosen`). |
| rebuild-integrity (slow) | regenerated replay+contfrag tables' response/posterior columns reproduce the backup; only covariate columns differ. |
| grep gate | no `recent_reward_rate`/`recent_trial_reward_rate`/`nominal_reward_prob`/`q_chosen` left in `src/`+`scripts/` (incl. contfrag/simulation/learning/viz). |
| smoke fit (slow) | one session per family (incl. contfrag) fits; M0–M2 / P-rungs-without-the-rate bit-identical vs a pre-change one-session fit; value block finite. |

## Fixtures
Synthetic trial/visit sequences for the window/Q/value unit tests; one real session for the rebuild-integrity + smoke fits.

## Review
Dispatch `code-reviewer`. Confirm: canonical names/schema match [shared-contracts](shared-contracts.md); **both** builders updated (replay + contfrag); rebuild-integrity holds (response columns reproduced); `leaky_q_value_diff` retained; `nominal_reward_prob` populated+renamed with no orphan (viz + schema updated); M0–M2 invariant, M3–M7 change acknowledged; grep gate clean across all of `src/`+`scripts/`; α in a sidecar not `.attrs`; tests exercise definitions not tautologies; no plan/phase references in code/test names.
