# Phase 1 — Honest internal names + interaction annotations

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Rename the internal functions whose names still lie (the registry decouples internal names from the frozen column names, so this is safe), and annotate the interaction block a scientist can't read without cross-referencing. Output-preserving.

**Inputs to read first:**

- `covariates/history.py:495-559` — `cumulative_reward_all_events`; its docstring (`:503`) admits it is a rolling k=10 window, "not a running total". This is the canonical lying name.
- `covariates/history.py:1218` — `cumulative_reward_per_arm`, the thin per-arm wrapper.
- `covariates/candidate_arm.py:266-298` — the ~40-line `swr_x_*` / `mua_x_*` interaction assignment block.
- `grep -rn "cumulative_reward_all_events\|cumulative_reward_per_arm" src scripts tests` — every caller to update.

## Tasks

- Rename `cumulative_reward_all_events` → `rolling_reward_count_all_events` and `cumulative_reward_per_arm` → `rolling_reward_count_per_arm`, updating every caller found by grep (in `tables_*`, `candidate_arm.py`, etc.). Update the docstrings to lead with what the function computes. **Do not** change the column the function fills (`reward_count_candidate_arm_last10_known_trials`) or touch the registry. If any test or external caller imports the old name, add it as a one-line alias in the defining module with a `# deprecated name` comment rather than breaking it — but prefer updating the caller.
- Scan for other internal names the code's own docstrings flag as misnomers (`grep -rn "despite the name\|not a\|really a\|misnomer\|the name is kept" src/continuum_swr_replay`); rename only those whose fix is unambiguous and whose column name is registry-frozen. List any you decline to rename with the reason.
- Annotate `candidate_arm.py:266-298`: one comment line per interaction family (`swr_x_is_current_arm`/`swr_x_is_prev_arm`, `swr_x_reward_*`, `swr_x_arm_uncertainty`, the `mua_x_*` mirror) stating what question it tests (e.g. "# does SWR replay concentrate on the current arm more than non-SWR?"). Comments only — no code change.

## Deliberately not in this phase

- Splitting `history.py` (phase 4) — this phase renames within the current file so the split moves already-correct names.
- Any column rename or registry edit — persisted names are frozen.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_rolling_reward_count_fills_unchanged_column` | The renamed function still produces `reward_count_candidate_arm_last10_known_trials` (byte-equal values on a fixture) — the rename didn't touch output. |
| `test_no_caller_uses_old_reward_count_name` | Grep-style: `cumulative_reward_all_events`/`_per_arm` appear nowhere outside a documented deprecation alias. |
| existing suite | Green (all renamed callers resolve). |

## Fixtures

Reuse the reward-history fixtures in `tests/unit/test_builder_value_floor.py` / `test_recent_trial_reward_rate.py`.

## Review

Before opening the PR, dispatch **both** reviewers (overview "Between-phase independent review"):
- `code-reviewer`: renames are complete, no column/registry change, the value-unchanged test is real.
- **Independent readability agent**: given the renamed functions and the annotated interaction block, confirm the names now say what they compute and the interactions read without cross-referencing — keep/adjust with evidence.
Confirm: validation slice passes; no name references this plan.
