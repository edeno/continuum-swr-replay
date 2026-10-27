# Phase 3b (C, runs) — Smoke and corpus refit of the follow-up marked family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: phase 3a's models answer questions 3, 5 and 8 only once fit on the corpus. This phase measures the cost on three sessions, checks the controls behave, then fits the 75 canonical mPFC sessions.

**Inputs to read first:**

- `.claude/docs/plans/pfc-marked-family-v2/phase-1c-corpus-refit.md` — smoke protocol, launcher, contract guard, audit script; implement its launcher, rename and audit tasks as written, with the paths below.
- `.claude/docs/reviews/2026-09-18-pfc-marked-family-smoke.md` (ported in phase 0) — the 2026-09-18 smoke: about 34 GB GPU per worker; sessions j16 20210710 02_r1, senor 20201101 08_r4, wilbur 20210331 10_r5.
- `scripts/pfc/run_pfc_marked_model_family.py` after phase 3a.

**Contracts referenced:** [Bundle trees](shared-contracts.md#bundle-trees), [Commits and GPU runs](shared-contracts.md#commits-and-gpu-runs).

## Tasks

- **Rename** `Processed-Data/pfc_marked_model_family/` → `Processed-Data/pfc_marked_model_family_v1_single_intercept/` (v2 1c), one line in each ported 2026-09-19 audit.
- **Smoke**: the three 2026-09-18 sessions, `--model-set followup --tier all --slope-permutations 30`, one free card each, `--output-dir Processed-Data/pfc_response_followups/marked_family_smoke`. Record per-model wall time and peak memory. Checks (in `scripts/qa/audit_pfc_marked_family_corpus.py`, new as v2 1c describes): `--no-baseline` reproduces the contract-1 base kernels bit for bit on one session; with baseline, the `rewarded_context` pre-onset (−200 to −50 ms) population slope mean is inside its null band; blow-up units (|Δ| > 1000) are zero; every target has a marginal bundle and a null array.
- **Extrapolate** the batch cost and record it; then the **corpus**: the 75 canonical sessions (the PFC cohort record `Processed-Data/pfc_cohort_mPFC.json`), partitioned by unit count, tmux `pfc_c_marked_family`, logs `logs/pfc_response_followups/marked_family/`, output `Processed-Data/pfc_response_followups/marked_family/`.
- **Audit and summary**: the audit script over the corpus (statuses, one contract, ineligibility by model × mark × class, blow-ups, non-converged), then the summary → `summary.parquet`. Audit report `.claude/docs/reviews/<date>-pfc-followup-marked-family-corpus.md` with tables of statuses and of which targets lie outside their null band (keys of `summary.parquet`, no typed numbers beyond counts read from it).

## Deliberately not in this phase

- No reported values (phase 4).
- No refit of the v1 recommended models.

## Validation slice

| Test | Asserts |
| --- | --- |
| v2 1c's `test_load_bundles_refuses_mixed_contract_versions`, `test_corpus_audit_script_counts_statuses_and_blowups` | as there |
| smoke (`slow`, GPU, ledger) | the four checks above |
| corpus (`slow`, GPU, ledger) | 75 sessions × follow-up models × tiers, one contract, failures zero or each explained |

## Fixtures

Synthetic sidecars in `tests/unit/test_summarize_pfc_marked_family.py`; real data: the smoke sessions, then the corpus.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the v1 tree is renamed, not left under the production path.
- User-facing documentation listed as tasks is updated, not deferred.
