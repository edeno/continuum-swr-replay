# Phase 1 — Close the builder value-level test gaps

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Raise the covariate-builder test floor so a later refactor that changes a covariate *value* fails a named test at the line, not an opaque table diff. Target: mutation kill rate ≥95% on `candidate_arm.py` + `history.py`, **zero surviving `side=` mutants**. No production code changes — tests only.

**Inputs to read first:**

- `docs/design/builder_test_floor.md` (from phase 0) — the survivor list this phase must beat.
- `src/continuum_swr_replay/analysis/replay/covariates/history.py:462,547,698,747,1519` — the five `searchsorted side=` sites (reward-history / recency window edges). Read each function whole to learn what value it computes.
- `src/continuum_swr_replay/analysis/replay/covariates/history.py:1322-1340` — `load_contingency` / `recent_reward_rate_all_events` / `cumulative_reward_all_events` cut on `outcome_known_time` with `side="right"`; these are the convention-A semantics (a window includes the current outcome).
- `src/continuum_swr_replay/analysis/replay/covariates/candidate_arm.py:170,173,185,203,260` — the 5 survivors (comparison + `+/−`).
- `tests/unit/test_covariates.py`, `tests/unit/test_uncertainty_covariates.py` — existing structure to extend; follow their fixture style.

## Tasks

- **Triage every survivor first.** For each of the 36, decide real-gap vs equivalent mutant by reading the site. Record the verdict in `docs/design/builder_test_floor.md`. Equivalent mutants (e.g. a `<` vs `<=` on a boundary provably never hit) are documented with the argument, not tested. Do not pad the kill rate with tests against equivalent mutants.
- **Kill the five `side=` survivors with golden-value tests.** For each window builder, construct a small synthetic experience/event frame where the current outcome sits exactly on the cutoff, and assert the covariate value with `side="right"` (current outcome included) differs from what `side="left"` would give — pinning the exact expected number. Example shape (adapt per function): three experiences with `outcome_known_time = [1.0, 2.0, 3.0]`, an event at `event_start_s = 2.0`, assert `recent_reward_rate_all_events` counts the `t=2.0` experience. These tests encode convention A and are the highest-value output of the phase.
- **Kill the comparison-operator survivors** (`history.py:147,149,167,304,336,337,338,339,1448,1524,1532`; `candidate_arm.py:173,185`) with boundary-value tests: inputs that straddle the compared threshold, asserting the covariate flips at the right side of the boundary.
- **Kill the `+/−` survivors** (`history.py:359,444,445,700,702,748,1313,1314,1421,1514,1525`; `candidate_arm.py:170,203,260`) with tests that would give a wrong sign/offset under the swapped operator (e.g. a recency/lag whose value would shift by a known delta).
- **Re-run `scripts/qa/mutation_check.py`** and record the new score in `docs/design/builder_test_floor.md`. Gate: ≥95% killed, zero `side=` survivors, all remaining survivors documented-equivalent.

## Deliberately not in this phase

- Any change to `history.py` / `candidate_arm.py` production code. If a survivor reveals an actual bug (not just a coverage gap), stop and surface it — a covariate-value bug is a science change, out of scope for a readability refactor, and must be decided separately.
- Making builders I/O-pure — that is phase 5. Tests here may use on-disk fixtures via `load_contingency` if needed; phase 5 removes that need.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recent_reward_rate_includes_current_outcome_on_cutoff` | Event on the cutoff counts the current experience (`side="right"`); a `left` flip would drop it. Exact count asserted. |
| `test_cumulative_reward_window_boundary` | Cumulative reward at an event straddling the window edge matches the hand-computed value. |
| `test_visit_recency_lag_offset` | Recency value shifts by the known delta iff the `+/−` is correct. |
| `test_candidate_arm_reward_history_boundary` | The 5 candidate_arm survivors each flip a covariate at the boundary. |
| `mutation_check` re-run (slow) | ≥95% killed on both modules; 0 `side=` survivors. |
| existing suite | Green. |

## Fixtures

Small synthetic experience/event/contingency frames in `conftest.py` (or the test module), one per window builder, with `outcome_known_time` values placed exactly on/around the cutoffs. Reuse the existing covariate-test fixture idioms.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every survivor is either killed by a real test or documented as equivalent — no untriaged survivors, no tests targeting equivalent mutants to inflate the score.
- No production `src/` code changed; if a real bug was found it was surfaced, not silently fixed.
- The `side=` tests genuinely distinguish `right` from `left` (would fail under the flip).
- Tests exercise behavior, not tautologies; setup in fixtures.
- No docstring/test name references this plan.
