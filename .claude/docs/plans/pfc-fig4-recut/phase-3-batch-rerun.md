# Phase 3 — Full batch rerun + population regeneration

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Regenerate all mPFC modulation artifacts with Phase-0-valid GLMs, Phase-1 canonical events,
the Phase-2 matched control, and Phase-2b temporal inference, each carrying a provenance sidecar. This is
primarily a compute phase — the code is done; the work is smoke-test, extrapolate,
then run under `tmux`, and confirm the old artifacts are fully replaced.

**Inputs to read first:**

- [scripts/pfc/run_pfc_analysis_bandit.py:97-200](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L97) — `run_analysis`, the Dask-parallel batch driver; `--animal/--date/--epoch/--overwrite/--n_workers` args; `git_hash` threading (line 105); save block (lines 168-183); `results_exist` gate (lines 86-95).
- [overview.md § Rollout Strategy](overview.md) — the four `*_modulation.npy` files are overwritten in place; matched-control `.npy` + `.provenance.json` are additive.

**Contracts referenced:**

- [Provenance sidecar schema](shared-contracts.md#provenance-sidecar-schema) — every rerun session must emit one with `is_swr_definition == "canonical_event_table"` and `n_matched_control > 0`.

## Tasks

- **Smoke test one session** end-to-end (`--animal wilbur --date 20210329 --epoch 02_r1 --overwrite`). Measure wall-clock and peak memory (`/usr/bin/time -v`, and `nvidia-smi` if the neuron GLMs touch GPU — confirm whether they do; `process_neuron` is statsmodels/CPU, but `load_data`/decoding load may not be). Confirm the session writes: 4 legacy `.npy` + 1 matched-control `.npy` + 1 `.provenance.json`, and that the sidecar counts are non-zero.
- **Extrapolate + decide chunking.** ~71 sessions × 5 animals, imbalanced. From the smoke-test per-session cost, estimate total wall-clock and whether to run per-animal in parallel or serialize. Record the estimate in the PR description. Only launch the full run after this.
- **Add an overwrite-safety check.** Before the batch, confirm `results_exist` ([lines 86-95](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L86-L95)) now includes the matched-control `.npy` and the `.provenance.json`, so a resumed run does not treat a Phase-1/2-incomplete session as done. A session is "done" only if all 6 files exist. Verify by deleting one sidecar and confirming that session re-runs.
- **Run the full batch under `tmux`** (session name `pfc_recut`, documented in the PR). Use `--overwrite` so every session is regenerated against current code; the point of this phase is to purge stale artifacts. Capture logs (the runner already writes per-session logs with the git hash).
- **Post-run provenance audit** (`scripts/qa/audit_pfc_provenance.py`, new, small): glob every `*_modulation.provenance.json`, assert all have `is_swr_definition == "canonical_event_table"`, `immobility_inequality == "<"`, `n_matched_control > 0`, current fit schema, nuisance/null metadata, one common blocked-score hash, acceptable valid-permutation fraction, and a git hash matching the run. Report count of sessions, per-animal breakdown, total events / SWR / non-SWR / matched. Fail loudly on any session missing a sidecar or carrying a stale definition.
- **Confirm no orphaned stale artifacts.** Assert no consumed pickle still contains the removed non-nested LR column `"SWR + non_local_no_SWR vs. non_local"` (the review's stale-artifact signature). The audit script checks this too.

## Deliberately not in this phase

- Any code change to GLM mechanics, event definition, control, or temporal inference (Phases 0/1/2/2b are frozen inputs). If the rerun surfaces a bug, fix it in the owning phase follow-up PR, not here.
- `load_population` per-animal aggregation and the figure (Phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| Smoke session (manual) | 6 files written; sidecar counts non-zero; wall-clock + memory recorded. |
| `results_exist` re-run guard (manual) | Deleting one sidecar causes that session to re-run under `--overwrite`-off resume. |
| `audit_pfc_provenance.py` | All regenerated sessions carry canonical events, current fit/null schemas and hashes; per-animal event counts printed; zero stale LR columns. |
| Loader re-load (slow) | `load_population_data(PROCESSED_DATA_DIR)` loads the regenerated artifacts without shape/alignment errors (existing loader guards pass). |

## Fixtures

Real data only — this is the production rerun. No synthetic fixtures. The smoke
session is the gate; the audit script operates on the real `PROCESSED_DATA_DIR`.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff (audit script + any runner glue) and include in the PR description: smoke-test timing/memory, the extrapolation, and the provenance-audit summary (session count, per-animal event totals). Confirm:
- Every regenerated session has a canonical, current-git sidecar (audit output pasted in PR).
- No stale non-nested LR columns remain in any consumed artifact.
- The `tmux` session name and full-run command are documented for reproducibility.
- No code changes to GLM/event/control logic snuck in under cover of the rerun.
