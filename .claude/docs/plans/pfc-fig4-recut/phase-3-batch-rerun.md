# Phase 3 — Full batch rerun + population regeneration

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Regenerate all mPFC modulation artifacts with Phase-0-valid GLMs, Phase-1 canonical events,
the Phase-2 matched control, and Phase-2b temporal inference, each carrying a
source-bound provenance sidecar. Production execution starts only after
regression-integrity Phase 13 freezes its refreshed canonical-table/session-input
generation. This is primarily a compute phase — the analysis code is done; the work
is upstream-generation validation, runner glue, smoke-test, extrapolation, then a
logged batch under `tmux`.

**Inputs to read first:**

- [scripts/pfc/run_pfc_analysis_bandit.py:97-200](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L97) — `run_analysis`, the Dask-parallel batch driver; `--animal/--date/--epoch/--overwrite/--n_workers` args; `git_hash` threading (line 105); save block (lines 168-183); `results_exist` gate (lines 86-95).
- [overview.md § Rollout Strategy](overview.md) — every old/new output is staged in one new immutable PFC run; legacy artifacts are not overwritten.
- [`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md) and its [session-input/run contracts](../regression-integrity-fixes/shared-contracts.md) — Phase 13 first audits/refreshes source bundles and freezes one staged canonical-table generation. This phase consumes that explicit generation; it never discovers inputs from legacy fixed paths.

**Contracts referenced:**

- [Provenance sidecar schema](shared-contracts.md#provenance-sidecar-schema) — every rerun session must emit one with `is_swr_definition == "canonical_event_table"` and `n_matched_control > 0`.
- [PFC completion manifest](shared-contracts.md#pfc-completion-manifest) — this phase
  supplies all per-session and batch-audit evidence but must not write
  `pfc_completion.json`; Phase 4 is the sole producer.

## Tasks

### Pre-freeze implementation PR

- Extend the runner with explicit `--release-candidate`, `--pfc-root`, and `--pfc-run-id` arguments.
  Validate the candidate's complete source audit and expected PFC session manifests; reject mutable,
  incomplete, hash-mismatched, or legacy fixed-path input. Record the canonical-table-generation,
  session-input, canonical-event-table, and clean candidate git hashes in every sidecar.
- Make every session output, sidecar, and log live beneath
  `<pfc_root>/runs/<pfc_run_id>.staging/`. Refuse an existing finalized run ID. `--overwrite`
  replaces only files in that staging run and never a legacy/finalized artifact.
- Replace `results_exist` with a source-aware resume guard requiring the matched-control artifact,
  sidecar, run/generation/session/event hashes, and current fit/null schemas. A missing or stale
  member reruns that session only.
- Add `scripts/qa/audit_pfc_provenance.py`. It validates expected/completed/excluded sessions, all five
  animals, sidecar/source identities, fit/null schemas, blocked-score hashes, permutation success,
  current fit-status accounting, and absence of the retired non-nested LR column. It writes the
  machine-readable batch audit inside the PFC staging run but never writes `pfc_completion.json`.
- Land the runner, resume guard, audit script, and synthetic tests before the shared code/config
  freeze. The implementation PR contains no real-data batch output.

### Post-freeze production record — no source/config changes

- Verify the checked-out `git_sha` is the clean frozen commit and select the explicit immutable
  Phase-13 release candidate. Any code/config or candidate change abandons this run.
- Smoke one session end-to-end
  (`--animal wilbur --date 20210329 --epoch 02_r1 --overwrite --pfc-run-id <id>`). Measure
  wall-clock and peak memory (`/usr/bin/time -v`, plus `nvidia-smi` only if the measured path uses a
  GPU). Confirm four modulation arrays, one matched-control array, one provenance sidecar, non-zero
  counts, complete fit-status rows, and exact upstream hashes.
- From the smoke cost, record the ~71-session wall-clock/resource estimate and fixed chunking plan,
  then run the full batch under durable logging. First-pass `--overwrite` remains confined to the
  PFC staging run.
- Run the already-frozen provenance audit. Require exact session/exclusion reconciliation, five
  animals, one code/candidate identity, all attempted fit cells accounted for, zero stale LR columns,
  and all source/control/null gates. Retain the audit and logs as production evidence; do not edit
  code, patch hashes, write `pfc_completion.json`, or create any pointer.

## Deliberately not in this phase

- Any code change to GLM mechanics, event definition, control, or temporal inference (Phases 0/1/2/2b are frozen inputs). If the rerun surfaces a bug, fix it in the owning phase follow-up PR, not here.
- `load_population` per-animal aggregation and the figure (Phase 4).
- Final completion-manifest emission. Phase 4 owns the final loader/inference/figure/doc
  validation and is the only phase allowed to install `pfc_completion.json`.

## Validation slice

| Test | Asserts |
| --- | --- |
| upstream-generation gate | incomplete/mutable generation, absent session manifest, legacy path, or hash mismatch fails before any PFC staging artifact is written |
| immutable output root | overwrite/resume touches only the named staging run; legacy and finalized run hashes remain unchanged and no current pointer exists |
| Smoke session (manual) | 6 files written; sidecar counts non-zero; upstream hashes match the frozen generation; wall-clock + memory recorded. |
| `results_exist` re-run guard (manual) | Deleting one sidecar or changing its source hash causes only that session to re-run under `--overwrite`-off resume. |
| `audit_pfc_provenance.py` | All expected sessions reconcile to completed/reviewed-excluded, carry the same run/generation identity and exact session source hashes, current fit/null schemas, five-animal coverage, and zero stale LR columns; no completion file is written. |
| Loader re-load (slow) | `load_population_data(<pfc staging run>)` loads the regenerated artifacts without shape/alignment errors (existing loader guards pass). |

## Fixtures

Use tiny manifest fixtures for upstream hash mismatch, stale resume, staging isolation, and final-run
immutability tests. The production rerun itself uses real data. The smoke session is the compute gate;
the audit script operates on the explicit PFC staging root and Phase-13 staged-generation manifest.

## Review

Before the implementation PR, dispatch `scientific-code-reviewer` against the runner/audit diff and
confirm all synthetic validation rows above pass. After the freeze, review the production record
separately; it contains smoke timing/memory, the extrapolation, commands/log hashes, and the
provenance-audit summary but no source/config diff. Across both checkpoints confirm:
- Every regenerated session has a canonical, current-git sidecar (audit output pasted in PR).
- Every sidecar binds to the same frozen canonical generation and the correct
  per-session source manifest; changed upstream hashes force recomputation.
- No stale non-nested LR columns remain in any consumed artifact.
- No `pfc_completion.json`, PFC-specific current pointer, or mutation of a legacy/finalized artifact
  exists at the end of Phase 3.
- The `tmux` session name and full-run command are documented for reproducibility.
- No post-freeze code/config change or GLM/event/control edit is hidden in the production record.
