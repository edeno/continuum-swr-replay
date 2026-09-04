# Phase 2 — Content-fit main() into named stages

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [stage convention](shared-contracts.md#stage-convention)

The content-fit `main()` is ~580 lines; the actual estimation is a few lines buried in cohort-freezing, contract checks, and staging. Lift the stages out so `main()` reads load → cohort → fit → publish in one screen. Output-preserving code motion, suite-verified.

**Inputs to read first:**

- `scripts/content/fit_replay_content_models.py:970-1550` — the whole `main()`. Map its regions: arg parsing, table load + `_validate_canonical_table_contract` (call site of `:223`), cohort selection/common-cohort freezing, the per-session fit loop (calls `fit_model_per_session:538` at ~`:1323`), model comparison, cohort audit, and the staged publish (`_write_complete_output_bundle:849` / `_publish_staged_directory:808`).
- [stage convention](shared-contracts.md#stage-convention).

## Tasks

- Extract a `run_content_fit(...)` (and named sub-helpers: e.g. `_load_and_validate_content_tables`, `_freeze_common_cohort`, `_fit_all_rungs`, `_audit_and_compare`, `_publish_content_bundle`) so `main()` becomes: parse args → `run_content_fit(args)`. Each helper moves an existing region verbatim; no computation changes. The `--hierarchy`/`--swr-subset`/`--grain`/`--models` branching stays intact.
- Keep the existing named helpers (`_validate_canonical_table_contract`, `fit_model_per_session`, `_write_complete_output_bundle`, `print_model_results`) as-is; this phase lifts their *call sites* out of `main()`, it does not rewrite them.
- Name the removed structure: the old inline `main()` body is replaced by the extracted helpers in this same phase — no parallel path.

## Deliberately not in this phase

- The rate-fit runner (`fit_per_bin_rate_models.py`) — same shape, but out of scope; note it as a follow-on if the pattern transfers cleanly.
- Any change to the fit math, the cohort rule, or the staging semantics.

## Validation slice

| Test | Asserts |
| --- | --- |
| existing `test_replay_content_runner_schema` / `test_replay_content_model_selection` | Pass unchanged — the runner's outputs and selection are identical post-extraction. |
| `test_content_fit_smoke` (if present; else a `--help` + import smoke) | `main()`/`run_content_fit` import and dispatch without error. |
| full unit suite | Green. |

## Fixtures

None new — leans on the existing content-runner tests.

## Review

Before opening the PR, dispatch **both** reviewers (overview "Between-phase independent review"):
- `code-reviewer`: the extraction is pure motion (review as moved hunks), branching preserved, no parallel `main()` left.
- **Independent readability agent**: given the new `main()` + `run_content_fit`, confirm a scientist can now see load → cohort → fit → publish at a glance and find the estimation call quickly — keep/adjust with evidence.
Confirm: validation slice passes; helper/test names use scientific vocabulary, not plan references.
