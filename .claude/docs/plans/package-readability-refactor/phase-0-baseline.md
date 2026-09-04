# Phase 0 — Freeze the v5 golden baseline; move the mutation harness into the repo

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Behavior-preserving groundwork: nothing in the package changes. This phase creates the two instruments every later phase depends on — a frozen reference to diff against, and a re-runnable coverage check.

**Inputs to read first:**

- `src/continuum_swr_replay/table_io.py` — `read_table`/`write_table`/`parquet_path`; the parity comparison reuses `read_table`.
- `scripts/decoding/write_regeneration_manifest.py:394` — `_verify_baseline`: the existing keyed candidate-vs-baseline diff. The phase-0 baseline-compare script generalizes this to the full canonical set.
- The session's mutation harness at `/tmp/.../scratchpad/mutate.py` (ephemeral) — its logic is the starting point for the in-repo tool; re-author it cleanly, do not copy the scratch path.

## Tasks

- **Freeze the baseline.** Add `scripts/qa/freeze_regression_baseline.py` that copies the current canonical tables (`event_table`, `event_arm_table`, `event_patch_table`, `event_sequence_table`, `stop_table`, `qvals_table`, `event_onset_table`, `stop_timebin_table`) and the model-output directories into a timestamped read-only reference under `Processed-Data/replay_content/_golden_baseline_v5/` (gitignored data; the script is tracked). Record a `manifest.json` with each file's sha256 and row count. This reuses the existing backups made this session where present.
- **Baseline-compare tool.** Add `scripts/qa/compare_to_golden_baseline.py` that, given a candidate `replay_content` root, diffs every canonical table against the frozen baseline by registered row key (`row_keys(table=...)`), per-column, numerics at `atol=0` and objects by equality — raising on any structural or value drift. Factor the comparison so it shares logic with `write_regeneration_manifest._verify_baseline` rather than forking a third parity engine (see overview Risk: two parity engines).
- **Mutation harness → repo.** Add `scripts/qa/mutation_check.py`: an objective AST mutation runner (stdlib `ast` + `subprocess`) that mutates a target module's comparison operators, `searchsorted side=` literals, and `+`/`−` binops, runs a named fast test subset with `pytest -x`, and reports killed/survived with file:line. Parameterize `--targets` and `--tests`. It must restore each file in a `finally` and fail loudly if the baseline test run is not green. Default targets: `covariates/candidate_arm.py`, `covariates/history.py`; default tests: `test_covariates.py test_uncertainty_covariates.py test_covariate_timing.py`.
- **Record the phase-0 mutation score** (70% / 36 survivors) into `docs/design/builder_test_floor.md` as the starting point phase 1 must beat, listing the survivor lines. This doc is the durable home for the coverage target (not a plan file — see the plan-milestone-naming quality bar).

## Deliberately not in this phase

- Writing any new test or fixing any survivor — that is phase 1. Phase 0 only *measures and records*.
- Any change to package code under `src/` — this phase adds `scripts/qa/**` and `docs/design/**` only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_compare_to_golden_baseline_detects_drift` | Mutating one value in a copied table makes `compare_to_golden_baseline` raise; an untouched copy passes. |
| `test_mutation_check_scores_a_known_module` (slow, `pytest.mark.slow`) | Running `mutation_check` on a tiny fixture module with one tested and one untested branch reports exactly one killed and one survivor. |
| existing suite | Green — no `src/` change, so it must be untouched. |

## Fixtures

A 2-row synthetic canonical `stop_table` (in `conftest.py`) for the baseline-compare drift test. A 6-line fixture module + its one test for the mutation-check self-test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task is implemented as specified; "Deliberately not in this phase" honored (no `src/` edits, no survivor fixes).
- The baseline-compare tool reuses `_verify_baseline` logic rather than forking a third parity engine.
- Validation slice passes; slow tests marked.
- Tests exercise real behavior, not tautologies; shared setup in fixtures.
- No docstring/module/test name references this plan or a phase number.
