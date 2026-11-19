# Phase 4 — Split history.py by concern; unify the reward-rate functions

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [history package layout](shared-contracts.md#history-package)

`history.py` is 1536 lines carrying reward history, visit recency, the Beta value/uncertainty join, and a diagnostic Q-learner — its own docstring calls it "a large module." Split it into a concern-per-module package with a re-export shim, and unify the two reward-rate functions (one looped, one vectorized) into one vectorized form under an equivalence guard.

**Inputs to read first:**

- [history package layout + re-export API](shared-contracts.md#history-package) — the symbol→module map and invariants.
- `covariates/history.py` in full (1536 lines) — trace each function's helper calls before moving, to place shared helpers in `_common` and avoid cycles.
- `covariates/history.py:495-559` (`cumulative_reward_all_events`, vectorized; renamed to `rolling_reward_count_all_events` in phase 1) and `:560-658` (`recent_reward_rate_all_events`, **nested Python loops**) — the pair to unify.
- `covariates/__init__.py` — the `from .history import ...` re-export list the shim must reproduce.
- `grep -rn "covariates.history\|from .history\|from ...covariates import" src scripts tests` — every importer whose path must keep resolving.

## Tasks

- Create the package `covariates/history/` with `_common.py`, `reward_history.py`, `visit_recency.py`, `value_uncertainty.py`, `diagnostics.py` per the [layout](shared-contracts.md#history-package). Move each function/constant verbatim to its module; update intra-package imports; put helpers used by ≥2 modules in `_common`. This is code motion — no logic change except the unification below.
- **Unify the reward-rate functions** (the one logic change, guarded): rewrite `recent_reward_rate_all_events` in vectorized form matching the `rolling_reward_count_all_events` window pattern (cumsum-with-leading-zero + `searchsorted` cutoffs), or extract a shared `_windowed_reward_stat` both call. **Before** deleting the loop version, add `test_recent_reward_rate_vectorized_equals_loop`: keep the current loop implementation as a reference function in the test, generate several synthetic experience/event frames (including the edge cases: no history, fewer than `k` experiences, ties on `outcome_known_time`, all-omission), and assert the vectorized output equals the loop output **exactly** (`np.testing.assert_array_equal`, NaN-aware). Only after it passes may the loop form be removed.
- Write `history/__init__.py`: re-export every previously-public name so the shim mirrors the old module's surface exactly, with a reading-order docstring in the style of `covariates/__init__.py:1-18` (reward history → visit recency → value/uncertainty → diagnostics; note the diagnostic learner produces no cataloged covariate). Delete the old `history.py`.
- **Docs:** update `covariates/__init__.py`'s reading-order thread to point at the new `history/` sub-modules by concern.

## Deliberately not in this phase

- Any change to a covariate's value beyond the equivalence-preserving vectorization — the byte-diff (phase 5) proves it.
- Splitting `context.py` / `candidate_arm.py` — out of scope.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recent_reward_rate_vectorized_equals_loop` | Vectorized == retained loop reference, exactly, across edge-case fixtures. **Gate for deleting the loop.** |
| `test_history_public_api_preserved` | Every name importable from the old `covariates.history` still resolves through the new package (grep-style over the prior surface). |
| existing covariate/builder suites | Green (all importers unbroken). |
| full unit suite | Green. |

## Fixtures

Synthetic experience/event frames for the equivalence test (reuse `tests/unit/_port_experience_fixtures.py`), covering: empty, `< k` experiences, ties on `outcome_known_time`, all-omission, all-reward.

## Review

Before opening the PR, dispatch **both** reviewers (overview "Between-phase independent review"):
- `code-reviewer`: the split is pure motion except the unification; the equivalence test genuinely gates the loop deletion; the shim preserves the exact public surface; no cycle through `_common`.
- **Independent readability agent**: given the new `history/` package + `__init__` reading order, confirm a scientist can now find the reward-history / recency / value / diagnostic concerns as separate readable modules — keep/adjust with evidence.
Confirm: equivalence test passes before the loop is deleted; API-preservation test passes; module names use scientific vocabulary, not plan references.
