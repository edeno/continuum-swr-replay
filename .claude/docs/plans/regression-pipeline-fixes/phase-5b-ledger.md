# Phase 5b ledger (archived from .superpowers/sdd/phase-5b-execution/progress.md on 2026-09-06)

# SDD ledger — plan: .claude/docs/plans/regression-pipeline-fixes/phase-5b-execution.md
Branch regression-pipeline-fixes-phase-5b off master 8d48d04 (phases 1–5a merged); plan commit cd06a31. Spec: phase-5b-readability-structure.md; all contracts in shared-contracts.md preserved.

Pre-flight scan:
| pair | produces / consumes | finding |
| T0→T4 | golden baseline pre5b + content smoke → parity/1e-12 | consistent (same gate as 5a) |
| T1→T2 | design_constants.py home for constants; the basis function (`raised_cosine_history_basis`, now in _helpers) — T1 or T2? | Ruling 1: the basis/kernel functions move to `per_bin_design.py` in T2; `design_constants.py` holds constants ONLY (no functions) — a constants module must not import the per-bin design |
| T1 layering test | "covariates/ must not import model_specs" | after T1 the six importers use design_constants; the test must FAIL before T1's moves (record) |
| T2 `_helpers` split → T3 registry trim | independent files | consistent |
| T2 event timing once | deleting `_merge_classified_and_verify_timing` removes the bit-for-bit check between the two derivations | Ruling 2: run the check on the cached session BEFORE deleting and record it passes; keep one `assert_event_label_alignment`; the parity gate (T4) is the ongoing proof |
| T3 CLI relocation | `prepare_data --from-cache` invocation is quoted in three audits and the phase-5a/5b briefs | Ruling 3: historical audits are records — leave; update docs/README.md, scripts/README.md, the runner docstrings, the pipeline map, and CLAUDE.md's Development Setup if it quotes the module form |
| T3 registry trim | `LEGACY_COLUMN_NAMES` frozenset extracted from the ledger | Ruling 4: record the extraction command + count in the report; the deny-list must reject every name the current ledger-derived list rejects (test: compute both at the pre-trim commit and assert set equality) |
| T3 validators → table_contracts | runner science path < 200 lines | if unreachable without behavior change, report the count and stop |
| All | parity gate any diff = STOP | Ruling 5 |
Tasks 0+1: dispatching one implementer (opus); BASE = cd06a31
Tasks 0+1: implementer DONE (8832e33 532ba50 8237b7f); suite 2120 passed + 1 strict xfail (hippocampus rule, tripwire for Task 2); pre-move layering failures: 16 covariates→model_specs edges, 2 data_loaders→analysis, 3 hippocampus→replay._helpers; content smoke byte-identical vs _golden_baseline_pre5b run. Notes: _helpers.py was a 7th importer (now design_constants); RECENCY_SPLINE_* added to model_specs.__all__ (re-export explicit). Review dispatched (opus).
Tasks 0+1: review spec ✅ / quality needs fixes — 1 Important (layering test's relative-import resolver off by one for __init__.py: relative imports there escape all three rules; latent today, matters for Task 2's new packages) + 2 minors; fix round 1 dispatched (resume implementer)
Tasks 0+1: fix round 1 implemented (f518ba3); suite 2121 passed + 1 xfail; re-review dispatched (sonnet)
Tasks 0+1: re-review clean (f518ba3)
Tasks 0+1: complete
Task 2: dispatching implementer (opus); BASE = f518ba3
Task 2: implementer DONE (b9f6efc 5b41308 b52287d 9db4ce3 d2e5d44); suite 2136 passed (xfail → pass, +14); content smoke byte-identical; _helpers.py 403 (kept set exactly as ruled + stay_indicator pair) — Ruling 6: accept 403; 16 session-key sites unified (brief named 8); is_stay needed two functions (stay_indicator_from_patches + arm wrapper) because shuffles.py recomputes from a shifted next_patch column — no semantic divergence at any of the 5 sites; tables_event never had an assert_event_label_alignment call (n_unclassified guard retained). Review dispatched (opus).
Task 2: review spec ✅ / quality Approved; 1 code minor (stay_indicator_from_patches type annotations) carried as Task 3 pre-step; the event-timing equality proven structural (same row set, same formulas). 
Task 2: complete
Task 3: dispatching implementer (opus); BASE = d2e5d44
Task 3: implementer DONE (6716783 fe9426d 0e6c4c7 4ff145b); suite 2130 passed (−6 = registry test rewrite 39→33); content smoke byte-identical; stop-table smoke exact. Ruling 7: covariate_registry.py 908 (kept functions alone 578) — < 400 unreachable, accepted. Ruling 8: content runner science path 427 lines unchanged (validators were outside it) — < 200 unreachable without decomposing fit logic, accepted. Deny-list: one global frozenset shown equivalent to the ten per-grain sets over all 386 materialized columns. Review dispatched (opus).
Task 3: review spec ✅ / quality Approved; controller fixed minors 1, 2, 6 (bf21a5e); minor 5 (narrow the appendix test to the Appendix section; drop the sha-pinned test after merge) carried to the final wave.
Task 3: complete
Task 4: dispatching implementer (opus); BASE = bf21a5e
Task 4: PASS (05a6a39 audit) — 8 tables byte-identical to _golden_baseline_pre5b; full 5-session rebuild via the relocated CLI exact except the cohort-scoped event_arm _z columns (affine to 5e-15); content smoke 0.0; full suite incl. slow 2144 passed / 0 failed; 56/56 --help; layering 16/2/3 → 0/0/0. _helpers.py 405 (Ruling 6 + 2 annotation lines). Carried minors for the fix wave: sha-pinned deny-list test (drop after merge; narrow the appendix companion to the Appendix section); prepare_data.py basicConfig at import (pre-existing).
Task 4: complete
Final whole-branch review: dispatched (fable), base master 8d48d04, head 05a6a39
Final review (fable): merge-ready YES after one fix wave. Important #3: value_block.py:176/249-250 and test_uncertainty_covariates.py:944 still instruct `python -m …qvals_table --write`, which now imports and exits 0 silently → replace with `uv run python scripts/behavior/build_qvals_table.py --write`. Minors: #1 deny-list comment/audit wording ("identical over canonical and materialized columns; stricter cross-grain by design"); #2 audit line for the dropped `choice`/`fit_time` grains in assert_canonical_vocabulary; #4 ruff format five test files newly dirty (test_contfrag_hash_safety, test_harmonization_parity_guard, test_replay_content_runner_schema, test_session_keys, test_switch_flag_retired); #5 replace the sha-pinned deny-list test + the whole-doc companion with one appendix-parsing test (LEGACY_COLUMN_NAMES == non-audit legacy names in the Appendix tables, 219; _LEGACY_RENAMED == canonical-bearing). Optional #6 (event_start_s == first in-event bin fixture test), #7 pre-existing. Close-out follow-ups listed by the reviewer → PLAN.md. Fix wave dispatched (opus).
Fix wave: DONE (111d29a a8f6e2c; fad1c61 is the controller's PLAN.md follow-up list); suite 2129 passed; F6 already covered by an existing test; scoped re-review dispatched (sonnet)
Fix wave: re-review clean (a8f6e2c); controller fixed the one stale test name in the audit. Phase 5b: COMPLETE — archived, merged to master. PLAN COMPLETE.
