# Phase 1c — Smoke, corpus refit under contract 2, audit and summary

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: phases 1a and 1b change every estimate. This phase reruns the three-session smoke, then the 71-session corpus, and regenerates the audit and the population summary so the contract-2 numbers replace the contract-1 numbers everywhere they are quoted.

**Inputs to read first:**

- `.claude/docs/reviews/2026-09-18-pfc-marked-family-smoke.md` — the smoke protocol, the sessions used (j16 20210710 02_r1, senor 20201101 08_r4, wilbur 20210331 10_r5), and the isolation-refit check.
- `.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md` and `-summary.md` — the contract-1 numbers to compare against.
- [scripts/pfc/run_pfc_marked_model_family.py:800-960](../../../../scripts/pfc/run_pfc_marked_model_family.py#L800) — CLI and `main`; the runner is resumable and its skip check will refit everything under contract 2.
- [scripts/pfc/summarize_pfc_marked_family.py:59-120](../../../../scripts/pfc/summarize_pfc_marked_family.py#L59) — `load_bundles`; it must refuse a mixed-contract directory.
- `logs/pfc_marked_family_20260918/sessions_gpu{4,7,9}.txt` — the per-worker session lists of the first batch; reuse the ranked list.

**Contracts referenced:**

- [Bundle directory suffixes](shared-contracts.md#bundle-directory-suffixes) — the rename before the refit.
- [Fit contract version 2](shared-contracts.md#fit-contract-version-2).

## Tasks

- **Commit the launcher.** The one-off `launch_corpus.sh` used on 2026-09-18 (balanced by unit count, one tmux worker per GPU, per-worker `sessions_gpu<N>.txt`, `EXIT` line appended to each worker's stdout) becomes `scripts/pfc/launch_pfc_marked_family_workers.sh` with a docstring header; the ranked session list becomes `scripts/pfc/mpfc_sessions_by_unit_count.txt` (71 lines, `epoch_identifier n_units`). Add both to `scripts/README.md`.

- **Rename the contract-1 tree** to `Processed-Data/pfc_marked_model_family_v1_single_intercept/` and add one line to the two 2026-09-19 audits saying where it went and that contract 2 supersedes it.

- **`load_bundles` contract guard.** In `summarize_pfc_marked_family.py`, read `bundle_contract_version` from each sidecar and raise if the directory holds more than one value; print the version in the report header.

- **Smoke** (long-running task: measure first). Run the three smoke sessions with the default flags on three idle cards (`nvidia-smi` first; the family used 34 GB per worker). Record per-model wall time and peak memory as before. Then the checks that this phase exists for, scripted in `scripts/qa/compare_pfc_marked_family_contracts.py` (new, ~150 LOC) reading a v1 and a v2 bundle directory for the same sessions:
  - with `--no-baseline`, the v2 spatial-role base kernels equal the v1 ones bit for bit for the unchanged adjustments (regression on the refactor, not on the science);
  - with the baseline on, the pre-onset means of `rewarded_context`, `switch_patch_context`, `content_weighted_novelty_fraction` and `previous_arm_fraction` population curves shrink by more than half against v1 on the same three sessions;
  - the interaction marks' median |r| with their constituents is below 0.2 in the sidecar geometry;
  - blow-up units (|Δ| > 1000) in the exploration interaction are zero on the three sessions.
  If the first or the last check fails, stop and fix before the corpus.

- **Corpus.** Launch all twelve models on every idle card (three were free on 2026-09-18; use whatever `nvidia-smi` shows free, never cards 0, 1, 3, 5, 6, 8 while they hold other users' jobs). Expect about 15 GPU-hours. Monitor worker stdout for `EXIT`, `Traceback`, `MemoryError`, `convergence gate`.

- **Audit and summary.** Rerun the corpus health script (the checks in `2026-09-19-pfc-marked-family-corpus.md`: statuses, one contract, ineligibility by model and mark, blow-ups, NaN units, non-converged) as `scripts/qa/audit_pfc_marked_family_corpus.py` (new, ~120 LOC, committed this time), then `summarize_pfc_marked_family.py`. Write `docs/audits/<date>-pfc-marked-family-corpus-v2.md` with the same tables plus a v1-versus-v2 section: for every target present in both, pre-onset mean, post-onset mean, fraction of gaining units, side by side. Update `docs/README.md`, `docs/design/pfc_marked_model_family.md` (the "What the first smoke showed" and "Artifacts" paragraphs), and `scripts/README.md`.

## Deliberately not in this phase

- No marginal tier (phase 2) and no permutation draws (phase 3); the corpus is refit with the joint tier only so its cost is known before the extra tiers are added.
- No interpretation beyond the side-by-side tables; the summary stays descriptive.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_load_bundles_refuses_mixed_contract_versions` | two synthetic sidecars with versions 1 and 2 in one directory → `ValueError` naming both. |
| `test_contract_comparison_script_pairs_sessions_by_identifier` | the comparison script matches v1 and v2 bundles on `epoch_identifier` and model, and reports a missing pair rather than skipping it. |
| `test_corpus_audit_script_counts_statuses_and_blowups` | on three synthetic bundles (complete, ineligible, complete-with-one-blowup) the audit table has the right counts. |
| smoke (`slow`, GPU) | the four checks above, recorded in the audit with numbers. |
| corpus (`slow`, GPU) | 71 × 12 pairs, one contract, zero failed sessions; blow-up units under 10 corpus-wide per model. |

## Fixtures

Synthetic sidecars and small bundles built in `tests/unit/test_summarize_pfc_marked_family.py` (extend its existing helpers). Real data: the three smoke sessions.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the v1 tree is renamed, not left under the production path.
- User-facing documentation listed as tasks is updated, not deferred.
