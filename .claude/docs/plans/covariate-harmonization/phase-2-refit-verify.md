# Phase 2 — Re-fit all families, verify, refresh docs

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Re-fit every family (replay **and** contfrag) on the regenerated canonical tables and
**verify the harmonization preserved what it should**: unchanged-definition covariates
reproduce (via design-matrix identity, the strong deterministic guarantee); the
deliberately-changed ones are reported; the content rungs that contain the per-arm
reward rate (which moved 5→10) are re-verified. Then refresh the docs/claims numbers.

**Inputs to read first:**
- [overview.md](overview.md#metrics) — the metrics (design-matrix identity, rebuild integrity, coverage).
- The runners (replay + contfrag): `fit_per_bin_rate_models.py`, `fit_uncertainty_rate.py`, `fit_replay_content_models.py` (arm `{all,T,F}` + patch, **and** `--event-source SWR/MUA` for contfrag), `fit_swr_power_model.py`, `compare_event_source_models.py`, `fit_per_stop_rate_model.py`.
- The pre-change baseline: the backed-up tables + fit pickles from phase 1.

## Tasks

- **Confirm baseline exists** — the **backed-up pre-rebuild tables** *and* the old fit pickles (phase 1). The parity script **re-assembles both design matrices** (old spec applied to the backup table, new spec to the regenerated table) rather than relying on the fit pickles, which persist only coefficients — the old-table backup makes this implementable.
- **Re-fit all families** on the regenerated tables: per-bin rate ladder, MU uncertainty, per-stop rate, SWR-power, content arm `{all,T,F}` + patch `{all,T,F}`, contfrag SWR/MUA, choice (`compare_event_source_models.py`). Smoke one session per runner first (timing); long runs in tmux.
- **Parity / verification script** `scripts/_verify_harmonization_parity.py`:
  - **design-matrix identity** — for the unchanged-definition covariates (reward kernel, `p_reward_current`, `rpe`, `value_diff_patch`, `recent_switch_rate`, `is_after_switch`, `leaky_q_value_diff`, spatial, spline, self-history, and the rename-only `reward_rate_trial_k10`), assert the model design columns are **identical** to baseline; coefficients then match within the fitters' convergence tolerance (don't over-tighten to 1e-6).
  - **rebuild integrity** — response/posterior columns of the regenerated tables reproduce the backup (already checked in phase 1; re-confirm at fit time).
  - **changed covariates** — `reward_rate_arm_k10` (vs old 5-visit) and the new `q_*`: print old→new comparison (no assert).
  - **coverage** — per-family session counts == baseline.
- **Re-verify the content rungs that moved.** M3–M7 (arm) and the patch analogues + the contfrag content fits contain the per-arm reward rate, now k=10 → re-run the content verification (`_verify_content_models.py`) and check whether the re-verified claims (esp. the `recent_reward_rate`→`reward_rate_arm_k10` "not value-selective" result and `swr_x_*`) still hold at k=10; note any that shift, against the autocorrelation-invalidation caveats already in `claims_table.md`.
- **Refresh docs/claims numbers** (a task, not deferred): `docs/covariate_dictionary.md` (status implemented + per-animal α range); `paper/claims_table.md` (moved content/choice numbers; confirm M0–M2 unchanged; **update the stale #19c per-bin lag-1 numbers** 1.42/3.10/1.08 → current and cite **M7r** not dummy-M7 per audit 03); `docs/replay_rate_content_swr_synthesis.md` + `docs/per_bin_rate_results.md` (MU `q_current` term, coupling `q_current`, the k=10 content numbers, the `TMAX`=3.99→4.0 fix's effect on the rate ladder). **Flag (not fix here):** these p-values remain session-level; animal-level/clustered inference is the deferred backlog (overview).

## Deliberately not in this phase
- No new covariate definitions / spec changes (phase 1).
- No re-decode; no reality re-run.
- No new figures unless a refreshed number invalidates an existing one.

## Validation slice

| Test | Asserts |
| --- | --- |
| `_verify_harmonization_parity` (design-matrix) | unchanged-definition covariate columns identical to baseline; coefficients within fitter tolerance; signed-rank verdicts unchanged. |
| `_verify_harmonization_parity` (rebuild integrity) | regenerated tables' response/posterior columns reproduce the backup. |
| `_verify_harmonization_parity` (changed) | `reward_rate_arm_k10` and `q_*` old→new printed; finite/sensible; reported. |
| coverage check | each family's session count == baseline. |
| content re-verification | M0–M2 unchanged; M3–M7 / contfrag content re-checked at k=10; shifts reported against the autocorrelation caveats. |
| docs numbers | every coefficient quoted in the updated docs matches the new fits (spot-check headline rows). |

## Fixtures
Real fit outputs (baseline backups + new). No synthetic fixtures; the parity check is the verification.

## Review
Dispatch `code-reviewer`. Confirm: design-matrix identity holds for all unchanged covariates (the core gate — a failure = a rename silently changed a definition); rebuild integrity confirmed; changed covariates reported old→new; coverage unchanged; **contfrag re-fit and re-verified** (not left at 5-visit); M3–M7 shifts surfaced and reconciled with the content claims; every doc number updated (no stale coefficients in `covariate_dictionary.md`/`claims_table.md`/synthesis docs); no plan/phase references in artifacts.
