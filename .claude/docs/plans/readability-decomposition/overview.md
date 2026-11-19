# Overview — Scope, integration, review policy, verification, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

References verified against the tree at planning time; re-grep the anchor before editing.

**Lying internal names (phase 1):**
- `covariates/history.py:495` `cumulative_reward_all_events` — docstring at `:503` admits it is a rolling k=10 window, "not a running total". Rename to `rolling_reward_count_all_events`. The **column it fills stays** `reward_count_candidate_arm_last10_known_trials` (frozen by the registry).
- `covariates/history.py:1218` `cumulative_reward_per_arm` — the thin per-arm wrapper; rename to match.
- The interaction block to annotate: `covariates/candidate_arm.py:266-298` (~40 `output[...] =` lines building `swr_x_*` / `mua_x_*` products).

**Content-fit `main()` (phase 2):** `scripts/content/fit_replay_content_models.py:970` (`main()`, ~580 lines). Existing helpers to lift the call-sites of: `_validate_canonical_table_contract:223`, `fit_model_per_session:538`, `_publish_staged_directory:808`, `_write_complete_output_bundle:849`, `print_model_results:881`.

**`build_event_table` (phase 3):** `src/continuum_swr_replay/analysis/replay/tables_event.py:100` (`build_event_table`, ends ~635). Nine numbered banner blocks at `:252, :278, :290, :305, :434, :447, :457, :499, :524, :544` — each becomes a named stage.

**`history.py` split (phase 4):** `src/continuum_swr_replay/analysis/replay/covariates/history.py` (1536 lines). Symbol → target module in [shared-contracts.md](shared-contracts.md#history-package). The two reward-rate functions to unify: `cumulative_reward_all_events:495` (vectorized) and `recent_reward_rate_all_events:560` (nested Python loops). The diagnostic learner to isolate: `_q_chosen:1246`, `q_all_arm:1265` (its own comment: "DIAGNOSTIC per-animal fitted-alpha learner; not a fitted covariate"), `_fit_alpha:1310`. `history.py` is imported widely (`grep -rn "covariates.history\|from .history\|covariates import" src scripts tests`); the re-export shim must preserve the public API exactly.

## Scope and dependency policy

### Goals

- A scientist can find the entry point and follow raw → tables → covariates → fits by reading the code.
- No god-functions: `build_event_table` and content-fit `main()` each show their spine in one screen.
- One concern per module in the history layer; internal names that describe what they compute.

### Non-Goals

- **No change to the science** — estimands, model specs, cohorts, covariate definitions, standardization references untouched.
- **No change to persisted column names** or published file paths. Function renames never rename a column (the registry decouples them).
- No new dependencies; no reformatting of untouched code.

### Dependency policy

No dependency changes.

## Between-phase independent review (required)

Each phase's `Review` section (executor-run at the phase boundary) dispatches **two** independent reviewers before the PR merges and before the next phase starts:

1. `code-reviewer` (or equivalent) against the diff — correctness, scope, plan-vs-implementation drift.
2. **An independent readability agent** (fresh general-purpose subagent, no stake in the change) that reads the *changed code* and confirms the phase **improved** legibility for a naive scientist and did not regress it — same lens as the review that motivated this plan (can a scientist trace the data processing here?). Prompt it with the specific files the phase touched and ask for a keep/adjust verdict with file:line evidence.

Do not start the next phase until both reviewers are addressed. This catches decomposition that moved code without making it clearer before the next phase builds on it.

## Metrics

- Parity: **byte-identical** canonical tables (phase 5 regen vs the frozen `_golden_baseline_v5_v4cache` — a tables-only snapshot of committed master + the v4 cache; the `compare_to_golden_baseline` tool, exact `atol=0`).
- The reward-rate unification: the vectorized form equals the current loop form **exactly** on the equivalence-test fixtures (phase 4).
- Function size: `build_event_table` spine and content-fit `main()` each ≤ ~120 lines after extraction.
- `history.py` (1536 lines) → four concern modules, none a catch-all.
- Full unit suite green at every phase boundary.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A "pure code motion" stage extraction silently changes a value | Full suite (builder-parity/from-cache-parity) per phase + the phase-5 regen byte-diff; the between-phase readability review is a second set of eyes. |
| The reward-rate vectorization changes a covariate value | Mandatory loop-vs-vectorized equivalence test on synthetic data (phase 4), asserting exact equality before the loop form is deleted. |
| `history.py` split breaks a wide importer | The `history/__init__.py` re-export shim preserves the public API; a grep-style test asserts every previously-importable name still resolves. |
| A rename leaks into a persisted column name | Phase 1 renames functions only; a test asserts the filled column names are unchanged, and the registry is not touched. |
| Line numbers drift between planning and execution | Every phase lists "Inputs to read first"; the executor re-greps the anchor. |

## Rollout Strategy

Replace-in-place, no flags. Backward compatibility preserved by re-export shims (`history/__init__.py`, and `tables.py` / `covariates/__init__.py` already do this) and by leaving every persisted column name and published path unchanged. Each phase is an independent PR that leaves the suite green; the phase-5 byte-diff is the consolidating gate.

## Open Questions

1. **Canonical value history placement** — `canonical_port_value_history`/`canonical_trial_switch_history` produce real covariates (not diagnostics), so they go in `reward_history.py`/a value module, NOT `history_diagnostics.py`. Resolved in [shared-contracts.md](shared-contracts.md#history-package).

## Estimated Effort

Phase 1 ~40 LOC (renames + comments). Phase 2 ~0 net (code motion into helpers). Phase 3 ~0 net (stage extraction). Phase 4 largest: ~0 net motion across 4 new modules + one vectorization rewrite (~40 LOC changed) + equivalence test. Phase 5 is a verification run, no code.
